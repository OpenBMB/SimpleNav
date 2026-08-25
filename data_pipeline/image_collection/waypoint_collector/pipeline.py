from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

from airsim_plugin.camera_views import parse_camera_views
from waypoint_collector.assembly import assemble_videos
from waypoint_collector.envs import (
    inspect_scene_source,
    prepare_isolated_scene_source,
    prepare_scene_source,
    scene_archive_uncompressed_size,
    scene_source_path,
)
from waypoint_collector.indexing import (
    RequestIndexSummary,
    build_request_index,
    scene_sort_key,
)
from waypoint_collector.metadata import write_metadata_artifacts
from waypoint_collector.pilot import run_pilot
from waypoint_collector.publish import build_updated_manifest, publish_artifacts
from waypoint_collector.state import CollectorState
from waypoint_collector.validation import validate_artifacts
from waypoint_collector.worker import (
    WorkerConfig,
    reconcile_episode_outputs,
    run_render_workers,
)


@dataclass(frozen=True)
class CollectorConfig:
    package_dir: Path
    env_archive_root: Path
    env_cache_root: Path
    gpus: tuple
    workers: int
    skipped_scenes: tuple
    views: tuple
    camera_seed: int
    channel_order: str
    run_id: str
    state_root: Path
    base_control_port: int
    frame_attempts: int
    failed_episode_retry_rounds: int
    estimated_output_gib: float
    space_safety_factor: float
    resume: bool
    image_width: int = 224
    image_height: int = 224
    worker_env_cache_roots: tuple = ()
    deep_video_validation: bool = False
    assembly_workers: int = 1

    @property
    def request_path(self):
        return self.package_dir / "render/render_requests.jsonl"

    @property
    def manifest_path(self):
        return self.package_dir / "manifest.json"

    @property
    def staging_root(self):
        return self.package_dir / ".render_staging" / self.run_id

    @property
    def episode_video_root(self):
        return self.staging_root / "rendered_episodes"

    @property
    def final_root(self):
        return self.staging_root / "final"

    @property
    def state_path(self):
        return self.state_root / self.run_id / "state.sqlite3"


class CollectorPipeline:
    PHASES = (
        "preflight", "prepare-envs", "pilot", "render", "assemble",
        "validate", "publish",
    )

    def __init__(self, config):
        self.config = config
        self.repository_root = Path(__file__).parents[1].resolve()
        self._render_outputs_reconciled = False

    @classmethod
    def from_args(cls, args):
        package_dir = Path(args.package_dir).expanduser().resolve()
        env_cache_root = Path(args.env_cache_root).expanduser().resolve()
        state_root = (
            Path(args.state_root).expanduser().resolve()
            if args.state_root
            else env_cache_root / ".waypoint_collector_state"
        )
        gpus = tuple(int(value.strip()) for value in args.gpus.split(",") if value.strip())
        worker_env_cache_roots = tuple(
            Path(value.strip()).expanduser().resolve()
            for value in args.worker_env_cache_roots.split(",")
            if value.strip()
        )
        skipped = tuple(value.strip() for value in args.skip_scene.split(",") if value.strip())
        views = parse_camera_views(args.views)
        config = CollectorConfig(
            package_dir=package_dir,
            env_archive_root=Path(args.env_archive_root).expanduser().resolve(),
            env_cache_root=env_cache_root,
            gpus=gpus,
            workers=int(args.workers),
            skipped_scenes=skipped,
            views=views,
            camera_seed=int(args.camera_seed),
            channel_order=args.channel_order,
            image_width=int(args.image_width),
            image_height=int(args.image_height),
            run_id=str(args.run_id),
            state_root=state_root,
            base_control_port=int(args.base_control_port),
            frame_attempts=int(args.frame_attempts),
            failed_episode_retry_rounds=int(args.failed_episode_retry_rounds),
            estimated_output_gib=float(args.estimated_output_gib),
            space_safety_factor=float(args.space_safety_factor),
            resume=bool(args.resume),
            worker_env_cache_roots=worker_env_cache_roots,
            deep_video_validation=bool(args.deep_video_validation),
            assembly_workers=int(args.assembly_workers),
        )
        if not config.gpus:
            raise ValueError("at least one GPU is required")
        if config.workers < 1 or config.workers > len(config.gpus):
            raise ValueError("workers must be between 1 and the number of GPUs")
        if (
            config.worker_env_cache_roots
            and len(config.worker_env_cache_roots) != config.workers
        ):
            raise ValueError(
                "worker environment roots must be empty or contain one root per worker"
            )
        if config.channel_order not in ("rgb", "bgr"):
            raise ValueError("unsupported channel order")
        if config.image_width <= 0 or config.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if config.image_width % 2 or config.image_height % 2:
            raise ValueError("image dimensions must be even for YUV420P video")
        if config.frame_attempts < 1:
            raise ValueError("frame attempts must be positive")
        if config.failed_episode_retry_rounds < 0:
            raise ValueError("failed episode retry rounds must be non-negative")
        if config.assembly_workers < 1:
            raise ValueError("assembly_workers must be positive")
        return cls(config)

    def _state(self):
        return CollectorState(self.config.state_path)

    def _phase_complete(self, phase):
        state = self._state()
        try:
            return state.get_metadata("phase:{}".format(phase)) == "complete"
        finally:
            state.close()

    def _mark_phase_complete(self, phase):
        state = self._state()
        try:
            state.set_metadata("phase:{}".format(phase), "complete")
        finally:
            state.close()

    def _write_report(self, name, payload):
        report_root = self.config.staging_root / "reports"
        report_root.mkdir(parents=True, exist_ok=True)
        path = report_root / "{}.json".format(name)
        temporary = path.with_name(".{}.partial".format(path.name))
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(path))
        return path

    def _summary_from_state(self, state):
        keys = (
            "total_requests", "available_requests", "missing_requests",
            "total_episodes", "available_episodes", "missing_episodes",
        )
        values = {key: state.get_metadata(key) for key in keys}
        if any(value is None for value in values.values()):
            return None
        scenes = state.get_metadata("scene_ids", "")
        return RequestIndexSummary(
            **{key: int(value) for key, value in values.items()},
            scene_ids=tuple(value for value in scenes.split(",") if value),
        )

    def preflight(self):
        config = self.config
        if not config.package_dir.is_dir():
            raise FileNotFoundError(config.package_dir)
        if not config.request_path.is_file():
            raise FileNotFoundError(config.request_path)
        if not config.manifest_path.is_file():
            raise FileNotFoundError(config.manifest_path)
        package_manifest = json.loads(
            config.manifest_path.read_text(encoding="utf-8")
        )
        declared_width = int(
            package_manifest.get("render_image_width", 224)
        )
        declared_height = int(
            package_manifest.get("render_image_height", 224)
        )
        if (
            declared_width != config.image_width
            or declared_height != config.image_height
        ):
            raise ValueError(
                "collector image dimensions {}x{} do not match package "
                "render dimensions {}x{}".format(
                    config.image_width, config.image_height,
                    declared_width, declared_height,
                )
            )
        if not config.env_archive_root.is_dir():
            raise FileNotFoundError(config.env_archive_root)
        occupied_ports = []
        for worker_index in range(config.workers):
            for port in (
                config.base_control_port + worker_index * 100,
                config.base_control_port + worker_index * 100 + 1,
            ):
                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    occupied_ports.append(port)
                finally:
                    probe.close()
        if occupied_ports:
            raise RuntimeError(
                "collector ports are already occupied: {}; choose another --base-control-port".format(
                    ", ".join(str(port) for port in occupied_ports)
                )
            )
        try:
            graphics_probe = subprocess.run(
                ["vulkaninfo", "--summary"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=20,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(
                "Vulkan graphics runtime is unavailable; AirSim RGB rendering cannot start"
            ) from error
        device_names = [
            line.split("=", 1)[1].strip()
            for line in graphics_probe.stdout.splitlines()
            if "deviceName" in line and "=" in line
        ]
        hardware_devices = [
            name for name in device_names
            if "llvmpipe" not in name.lower() and "cpu" not in name.lower()
        ]
        if graphics_probe.returncode != 0 or not hardware_devices:
            detail = next(
                (
                    line.strip() for line in graphics_probe.stdout.splitlines()
                    if "libGLX_nvidia" in line or "Failed loading" in line
                ),
                "detected devices: {}".format(device_names or "none"),
            )
            raise RuntimeError(
                "no hardware Vulkan renderer is available ({}). Install or mount the GPU graphics/Vulkan driver before collection".format(
                    detail
                )
            )
        config.staging_root.mkdir(parents=True, exist_ok=True)
        state = self._state()
        try:
            existing_jobs = sum(1 for _ in state.iter_jobs())
            summary = self._summary_from_state(state)
            stat = config.request_path.stat()
            stored_width = state.get_metadata("image_width")
            stored_height = state.get_metadata("image_height")
            dimensions_match = (
                (
                    stored_width is None and stored_height is None
                    and config.image_width == 224 and config.image_height == 224
                )
                or (
                    stored_width == str(config.image_width)
                    and stored_height == str(config.image_height)
                )
            )
            signature_matches = (
                state.get_metadata("request_path") == str(config.request_path) and
                state.get_metadata("request_size") == str(stat.st_size) and
                state.get_metadata("request_mtime_ns") == str(stat.st_mtime_ns) and
                dimensions_match
            )
            if existing_jobs:
                if not config.resume:
                    raise RuntimeError(
                        "collector state already exists; use --resume or a new --run-id"
                    )
                if not signature_matches or summary is None:
                    raise RuntimeError("render request file changed since the saved run")
            else:
                summary = build_request_index(
                    config.request_path, state,
                    skipped_scenes=config.skipped_scenes,
                    shard_target_requests=50000,
                    image_width=config.image_width,
                    image_height=config.image_height,
                )

            archive_report = {}
            blocking = []
            required_cache_bytes = 0
            for scene_id in summary.scene_ids:
                source_path = scene_source_path(config.env_archive_root, scene_id)
                inspection = inspect_scene_source(config.env_archive_root, scene_id)
                archive_report[scene_id] = {
                    "source_path": str(source_path),
                    "source_kind": "directory" if source_path.is_dir() else "zip",
                    "complete": inspection.complete,
                    "missing_members": list(inspection.missing_members),
                    "skipped": scene_id in config.skipped_scenes,
                }
                if scene_id not in config.skipped_scenes and not inspection.complete:
                    blocking.append(scene_id)
                if (
                    scene_id not in config.skipped_scenes
                    and inspection.complete
                    and source_path.is_file()
                    and not (
                        config.env_cache_root / "env_{}".format(scene_id) /
                        ".archive_manifest.json"
                    ).is_file()
                ):
                    required_cache_bytes += scene_archive_uncompressed_size(
                        source_path
                    )
            if blocking:
                raise RuntimeError(
                    "required scene archive(s) are incomplete: {}".format(
                        ", ".join(blocking)
                    )
                )
            config.env_cache_root.mkdir(parents=True, exist_ok=True)
            cache_disk = shutil.disk_usage(config.env_cache_root)
            cache_required_with_margin = int(required_cache_bytes * 1.2)
            if cache_disk.free < cache_required_with_margin:
                raise RuntimeError(
                    "insufficient scene cache space: {} bytes free, {} required".format(
                        cache_disk.free, cache_required_with_margin
                    )
                )
            disk = shutil.disk_usage(config.package_dir)
            completed_requests = sum(
                job.request_count for job in state.iter_jobs()
                if state.job_status(job.episode_id) == "complete"
            )
            remaining_fraction = (
                max(0, summary.available_requests - completed_requests) /
                max(1, summary.available_requests)
            )
            configured_final_estimate = int(
                config.estimated_output_gib * (1024 ** 3)
            )
            frame_based_final_estimate = int(
                summary.available_requests * len(config.views) * 16 * 1024
                * config.image_width * config.image_height / (224 * 224)
            )
            estimated_final_output = max(
                configured_final_estimate, frame_based_final_estimate
            )
            estimated_remaining = int(
                estimated_final_output * remaining_fraction
            )
            required_free = int(estimated_remaining * config.space_safety_factor)
            if disk.free < required_free:
                raise RuntimeError(
                    "insufficient free space: {} bytes free, {} required".format(
                        disk.free, required_free
                    )
                )
            report = {
                "summary": asdict(summary),
                "archives": archive_report,
                "disk_free_bytes": disk.free,
                "estimated_final_output_bytes": estimated_final_output,
                "scene_cache_free_bytes": cache_disk.free,
                "scene_cache_required_bytes": cache_required_with_margin,
                "estimated_remaining_bytes": estimated_remaining,
                "required_free_bytes": required_free,
                "state_path": str(config.state_path),
                "staging_root": str(config.staging_root),
                "graphics_devices": hardware_devices,
                "image_width": config.image_width,
                "image_height": config.image_height,
            }
            self._write_report("preflight", report)
            return report
        finally:
            state.close()

    def prepare_envs(self):
        state = self._state()
        try:
            scene_ids = sorted(
                {job.scene_id for job in state.iter_jobs()
                 if job.scene_id not in self.config.skipped_scenes},
                key=scene_sort_key,
            )
        finally:
            state.close()
        prepared = []
        for scene_id in scene_ids:
            result = prepare_scene_source(
                self.config.env_archive_root,
                self.config.env_cache_root,
                scene_id,
            )
            prepared.append({
                "scene_id": scene_id,
                "scene_root": str(result.scene_root),
                "reused": result.reused,
                "sha256": result.sha256,
            })
        worker_scenes = {}
        for worker_root in self.config.worker_env_cache_roots:
            worker_prepared = []
            for scene_id in scene_ids:
                result = prepare_isolated_scene_source(
                    self.config.env_cache_root,
                    worker_root,
                    scene_id,
                )
                worker_prepared.append({
                    "scene_id": scene_id,
                    "scene_root": str(result.scene_root),
                    "reused": result.reused,
                    "sha256": result.sha256,
                })
            worker_scenes[str(worker_root)] = worker_prepared
        self._write_report(
            "prepare-envs",
            {"scenes": prepared, "worker_scenes": worker_scenes},
        )
        return prepared

    def pilot(self):
        state = self._state()
        try:
            result = run_pilot(
                state=state,
                request_path=self.config.request_path,
                repository_root=self.repository_root,
                env_cache_root=self.config.env_cache_root,
                output_root=self.config.staging_root / "pilot",
                views=self.config.views,
                gpu=self.config.gpus[0],
                base_control_port=self.config.base_control_port,
                channel_order=self.config.channel_order,
                camera_seed=self.config.camera_seed,
                skipped_scenes=self.config.skipped_scenes,
                image_width=self.config.image_width,
                image_height=self.config.image_height,
            )
        finally:
            state.close()
        self._write_report("pilot", result)
        return result

    def _resolved_channel_order(self):
        return self.config.channel_order

    def _reconcile_render_outputs(self):
        if self._render_outputs_reconciled or not self.config.state_path.is_file():
            return
        state = self._state()
        try:
            if not any(True for _ in state.iter_jobs()):
                return
            reconcile_episode_outputs(
                state, self.config.episode_video_root, self.config.views,
                image_width=self.config.image_width,
                image_height=self.config.image_height,
            )
            self._render_outputs_reconciled = True
        finally:
            state.close()

    def render(self):
        self._reconcile_render_outputs()
        state = self._state()
        try:
            if self.config.resume:
                state.reset_interrupted_jobs()
                state.reset_failed_jobs()
        finally:
            state.close()
        channel_order = self._resolved_channel_order()
        configs = []
        for worker_index in range(self.config.workers):
            configs.append(WorkerConfig(
                worker_index=worker_index,
                gpu=self.config.gpus[worker_index],
                control_port=self.config.base_control_port + worker_index * 100,
                repository_root=str(self.repository_root),
                env_cache_root=str(
                    self.config.worker_env_cache_roots[worker_index]
                    if self.config.worker_env_cache_roots
                    else self.config.env_cache_root
                ),
                request_path=str(self.config.request_path),
                state_path=str(self.config.state_path),
                episode_video_root=str(self.config.episode_video_root),
                log_path=str(
                    self.config.staging_root / "logs" /
                    "worker-{}.log".format(worker_index)
                ),
                views=self.config.views,
                skipped_scenes=self.config.skipped_scenes,
                camera_seed=self.config.camera_seed,
                channel_order=channel_order,
                frame_attempts=self.config.frame_attempts,
                failed_episode_retry_rounds=(
                    self.config.failed_episode_retry_rounds
                ),
                image_width=self.config.image_width,
                image_height=self.config.image_height,
            ))
        run_render_workers(configs)
        state = self._state()
        try:
            counts = state.status_counts()
        finally:
            state.close()
        if counts.get("pending", 0) or counts.get("running", 0):
            raise RuntimeError(
                "render worker pool returned with unfinished jobs: {}".format(counts)
            )
        if counts.get("failed", 0):
            raise RuntimeError(
                "render did not complete all available episodes: {}".format(counts)
            )
        report = {
            "status_counts": counts,
            "channel_order": channel_order,
            "failed_episode_retry_rounds": (
                self.config.failed_episode_retry_rounds
            ),
        }
        self._write_report("render", report)
        return report

    def assemble(self):
        state = self._state()
        try:
            assembly = assemble_videos(
                state, self.config.episode_video_root, self.config.final_root,
                self.config.views, self.config.skipped_scenes,
                assembly_workers=self.config.assembly_workers,
            )
            metadata = write_metadata_artifacts(
                self.config.request_path, state, self.config.final_root / "meta",
                self.config.views, camera_seed=self.config.camera_seed,
                skipped_scenes=self.config.skipped_scenes,
                image_width=self.config.image_width,
                image_height=self.config.image_height,
            )
        finally:
            state.close()
        report = {"assembly": asdict(assembly), "metadata": asdict(metadata)}
        self._write_report("assemble", report)
        return report

    def validate(self):
        state = self._state()
        try:
            result = validate_artifacts(
                self.config.final_root, state, self.config.views,
                self.config.skipped_scenes,
                deep_video_probe=self.config.deep_video_validation,
                image_width=self.config.image_width,
                image_height=self.config.image_height,
            )
        finally:
            state.close()
        report = asdict(result)
        self._write_report("validate", report)
        return report

    def publish(self):
        if not self._phase_complete("validate"):
            raise RuntimeError("validation must complete before publish")
        state = self._state()
        try:
            summary = self._summary_from_state(state)
            if summary is None:
                raise RuntimeError("preflight summary is missing")
        finally:
            state.close()
        original_manifest = json.loads(
            self.config.manifest_path.read_text(encoding="utf-8")
        )
        manifest = build_updated_manifest(
            original_manifest,
            total_episodes=summary.total_episodes,
            available_episodes=summary.available_episodes,
            missing_episodes=summary.missing_episodes,
            total_requests=summary.total_requests,
            available_requests=summary.available_requests,
            missing_requests=summary.missing_requests,
            skipped_scenes=self.config.skipped_scenes,
            views=self.config.views,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
        )
        publish_artifacts(self.config.final_root, self.config.package_dir, manifest)
        self._write_report("publish", {"manifest": manifest})
        return manifest

    def execute(self, command):
        if command == "run":
            if self.config.resume:
                self._reconcile_render_outputs()
            results = {}
            for phase in self.PHASES:
                if self.config.resume and self._phase_complete(phase):
                    results[phase] = "already_complete"
                    continue
                method = getattr(self, phase.replace("-", "_"))
                results[phase] = method()
                self._mark_phase_complete(phase)
            print(json.dumps(results, indent=2, sort_keys=True, default=str))
            return 0
        method = getattr(self, command.replace("-", "_"))
        result = method()
        self._mark_phase_complete(command)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
