from dataclasses import dataclass
import json
import multiprocessing
from pathlib import Path
import signal
import time

from msgpackrpc.error import TimeoutError as RpcTimeoutError, TransportError
from PIL import Image

from waypoint_collector.airsim_session import AirSimServerRuntime
from waypoint_collector.cameras import episode_camera_records
from waypoint_collector.renderer import (
    EpisodeRenderError,
    InvalidFrameRenderError,
    SessionRenderError,
    render_episode,
)
from waypoint_collector.requests import iter_render_requests, validate_episode_sequence
from waypoint_collector.state import CollectorState
from waypoint_collector.video import (
    EpisodeVideoSink,
    episode_marker_path,
    remove_episode_artifacts,
    remove_episode_transient_artifacts,
    validate_episode_commit,
    validate_episode_videos,
    legacy_episode_files_present,
    write_episode_commit_marker,
)


@dataclass(frozen=True)
class WorkerConfig:
    worker_index: int
    gpu: int
    control_port: int
    repository_root: str
    env_cache_root: str
    request_path: str
    state_path: str
    episode_video_root: str
    log_path: str
    views: tuple
    skipped_scenes: tuple
    camera_seed: int
    channel_order: str
    image_width: int = 224
    image_height: int = 224
    frame_attempts: int = 10
    failed_episode_retry_rounds: int = 3


def _episode_videos_complete(root, episode_id, views, expected_frames,
                             image_width=224, image_height=224):
    return validate_episode_commit(
        root, episode_id, views, expected_frames,
        image_width=image_width, image_height=image_height,
    )


def _remove_incomplete_episode_videos(root, episode_id, views):
    remove_episode_artifacts(root, episode_id, views)


def write_invalid_frame_debug(root, episode_id, frames, error_message):
    debug_root = Path(root) / "debug_invalid_frames" / str(episode_id)
    debug_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for view, frame in sorted(frames.items()):
        path = debug_root / "{}.png".format(view)
        Image.fromarray(frame).save(path)
        paths.append(path)
    (debug_root / "error.txt").write_text(
        "{}\n".format(error_message), encoding="utf-8"
    )
    return tuple(paths)


def _episode_ids_with_artifacts(root, views):
    episode_ids = set()

    commits_root = root / "commits"
    if commits_root.is_dir():
        for path in commits_root.iterdir():
            name = path.name
            if name.startswith(".") and name.endswith(".json.partial"):
                episode_ids.add(name[1:-len(".json.partial")])
            elif name.endswith(".json"):
                episode_ids.add(name[:-len(".json")])

    attempts_root = root / "attempts"
    if attempts_root.is_dir():
        episode_ids.update(path.name for path in attempts_root.iterdir())

    for view in views:
        episode_root = root / "episodes" / view
        if not episode_root.is_dir():
            continue
        for path in episode_root.iterdir():
            name = path.name
            if name.startswith(".") and name.endswith(".partial.mp4"):
                episode_ids.add(name[1:-len(".partial.mp4")])
            elif name.endswith(".mp4"):
                episode_ids.add(name[:-len(".mp4")])

    return episode_ids


def reconcile_episode_outputs(state, root, views, image_width=224,
                              image_height=224):
    root = Path(root)
    views = tuple(views)
    if not root.exists():
        return
    artifact_episode_ids = _episode_ids_with_artifacts(root, views)
    for job in state.iter_jobs():
        status = state.job_status(job.episode_id)
        if status != "complete" and job.episode_id not in artifact_episode_ids:
            continue
        if status == "missing_scene":
            remove_episode_artifacts(root, job.episode_id, views)
            continue
        if validate_episode_commit(
            root, job.episode_id, views, job.request_count,
            probe_videos=False,
            image_width=image_width, image_height=image_height,
        ):
            remove_episode_transient_artifacts(root, job.episode_id, views)
            if status != "complete":
                state.mark_complete(job.episode_id)
            continue
        marker = episode_marker_path(root, job.episode_id)
        if (
            status == "complete"
            and not marker.exists()
            and legacy_episode_files_present(root, job.episode_id, views)
        ):
            write_episode_commit_marker(
                root,
                job.episode_id,
                views,
                {view: job.request_count for view in views},
            )
            remove_episode_transient_artifacts(root, job.episode_id, views)
            continue
        remove_episode_artifacts(root, job.episode_id, views)
        state.reset_job_pending(job.episode_id)


def render_worker(config, runtime_factory=AirSimServerRuntime,
                  sink_factory=EpisodeVideoSink):
    state = CollectorState(config.state_path)

    def create_runtime():
        return runtime_factory(
            repository_root=config.repository_root,
            env_root=config.env_cache_root,
            gpu=config.gpu,
            control_port=config.control_port,
            log_path=config.log_path,
            image_width=config.image_width,
            image_height=config.image_height,
        )

    runtime = create_runtime()
    session = None
    current_scene = None
    worker_id = "worker-{}".format(config.worker_index)
    try:
        while True:
            if current_scene is None:
                job = state.claim_next_job(
                    worker_id, skipped_scenes=config.skipped_scenes
                )
            else:
                job = state.claim_pending_in_scene(worker_id, current_scene)
                if job is None and state.scene_primary_work_remaining(current_scene):
                    time.sleep(1.0)
                    continue
                if job is None:
                    job = state.claim_failed_for_worker(
                        worker_id,
                        current_scene,
                        config.failed_episode_retry_rounds,
                    )
                if job is None:
                    job = state.claim_next_job(
                        worker_id, skipped_scenes=config.skipped_scenes
                    )
            if job is None:
                return 0
            if _episode_videos_complete(
                config.episode_video_root, job.episode_id, config.views,
                job.request_count,
                image_width=config.image_width,
                image_height=config.image_height,
            ):
                state.mark_complete(job.episode_id)
                continue
            _remove_incomplete_episode_videos(
                config.episode_video_root, job.episode_id, config.views
            )
            requests = tuple(iter_render_requests(
                config.request_path,
                byte_start=job.byte_start,
                byte_end=job.byte_end,
                start_index=job.start_index,
                expected_width=config.image_width,
                expected_height=config.image_height,
            ))
            validate_episode_sequence(requests)
            if len(requests) != job.request_count:
                raise RuntimeError(
                    "episode {} request range contains {} rows, expected {}".format(
                        job.episode_id, len(requests), job.request_count
                    )
                )
            camera_records = episode_camera_records(
                job.episode_id,
                seed=config.camera_seed,
                views=config.views,
                zero_position_delta_views=job.zero_position_camera_fallback_views,
            )
            if session is None or current_scene != job.scene_id:
                if session is not None:
                    session.close()
                    session = None
                while session is None:
                    try:
                        session = runtime.open_scene(
                            job.scene_id, channel_order=config.channel_order
                        )
                    except (RpcTimeoutError, TransportError, ConnectionError):
                        runtime.close(suppress_unavailable_scene_rpc=True)
                        runtime = create_runtime()
                        time.sleep(1.0)
                current_scene = job.scene_id
            sink = sink_factory(
                config.episode_video_root, job.episode_id, config.views,
                image_width=config.image_width,
                image_height=config.image_height,
            )
            try:
                rendered = render_episode(
                    session=session,
                    requests=requests,
                    camera_records=camera_records,
                    views=config.views,
                    sink=sink,
                    frame_attempts=config.frame_attempts,
                    retry_delay_seconds=1.0,
                    image_width=config.image_width,
                    image_height=config.image_height,
                )
            except (SessionRenderError, InvalidFrameRenderError) as error:
                sink.abort()
                if isinstance(error, InvalidFrameRenderError):
                    try:
                        write_invalid_frame_debug(
                            config.episode_video_root,
                            job.episode_id,
                            error.frames,
                            str(error),
                        )
                    except OSError as debug_error:
                        print(
                            "failed to save invalid-frame debug for {}: {}".format(
                                job.episode_id, debug_error
                            )
                        )
                    if error.invalid_view is not None:
                        state.enable_zero_position_camera_fallback_view(
                            job.episode_id, error.invalid_view
                        )
                state.mark_failed(job.episode_id, str(error))
                session.close()
                session = None
                runtime.close(suppress_unavailable_scene_rpc=True)
                runtime = None
                runtime = create_runtime()
                continue
            except EpisodeRenderError as error:
                sink.abort()
                state.mark_failed(job.episode_id, str(error))
                continue
            except Exception:
                sink.abort()
                raise
            if rendered != job.request_count:
                sink.abort()
                raise RuntimeError("rendered frame count mismatch")
            sink.commit()
            state.mark_complete(job.episode_id)
    finally:
        if session is not None:
            session.close()
        if runtime is not None:
            runtime.close()
        state.close()


def _worker_entry(config):
    def _terminate(_signal_number, _frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _terminate)
    raise SystemExit(render_worker(config))


def _render_progress_payload(state_path, failed_episode_retry_rounds):
    state = CollectorState(state_path)
    try:
        row = state.connection.execute(
            """SELECT
                   COUNT(*) AS episodes_total,
                   SUM(CASE WHEN status='complete' THEN 1 ELSE 0 END) AS episodes_complete,
                   SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS episodes_pending,
                   SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS episodes_running,
                   SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS episodes_failed,
                   SUM(CASE WHEN status='missing_scene' THEN 1 ELSE 0 END) AS episodes_missing,
                   COALESCE(SUM(request_count), 0) AS requests_total,
                   COALESCE(SUM(CASE WHEN status='complete' THEN request_count ELSE 0 END), 0) AS requests_complete,
                   SUM(CASE WHEN status='failed' AND attempts<=? THEN 1 ELSE 0 END) AS retry_waiting,
                   SUM(CASE WHEN status='failed' AND attempts=? THEN 1 ELSE 0 END) AS final_retry_next,
                   SUM(CASE WHEN status='failed' AND attempts>? THEN 1 ELSE 0 END) AS retry_exhausted
               FROM jobs""",
            (
                int(failed_episode_retry_rounds),
                int(failed_episode_retry_rounds),
                int(failed_episode_retry_rounds),
            ),
        ).fetchone()
        payload = {key: int(row[key] or 0) for key in row.keys()}
    finally:
        state.close()
    payload["episodes_remaining"] = (
        payload["episodes_total"] - payload["episodes_complete"]
        - payload["episodes_missing"]
    )
    payload["requests_remaining"] = (
        payload["requests_total"] - payload["requests_complete"]
    )
    payload["episode_percent"] = round(
        100.0 * payload["episodes_complete"] /
        max(1, payload["episodes_total"] - payload["episodes_missing"]), 2
    )
    payload["request_percent"] = round(
        100.0 * payload["requests_complete"] /
        max(1, payload["requests_total"]), 2
    )
    return payload


def _print_render_progress(config):
    payload = _render_progress_payload(
        config.state_path, config.failed_episode_retry_rounds
    )
    print(
        "[render-progress] {}".format(
            json.dumps(payload, sort_keys=True)
        ),
        flush=True,
    )


def run_render_workers(configs):
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_worker_entry, args=(config,)) for config in configs]
    for process in processes:
        process.start()
    _print_render_progress(configs[0])
    last_progress_at = time.monotonic()
    remaining = set(processes)
    while remaining:
        failed = []
        for process in tuple(remaining):
            process.join(timeout=0)
            if process.exitcode is None:
                continue
            remaining.remove(process)
            if process.exitcode != 0:
                failed.append((process.pid, process.exitcode))
        if failed:
            for process in remaining:
                process.terminate()
            for process in remaining:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)
            _print_render_progress(configs[0])
            raise RuntimeError("render worker failure(s): {}".format(failed))
        if remaining:
            if time.monotonic() - last_progress_at >= 60.0:
                _print_render_progress(configs[0])
                last_progress_at = time.monotonic()
            time.sleep(1)
    _print_render_progress(configs[0])
