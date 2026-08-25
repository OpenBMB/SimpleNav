from __future__ import annotations

import json
import math
import pickle
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tool.navvla.adapters.base import NavVLASourceAdapter, register_adapter
from tool.navvla.context_index import ContextIndexConfig
from tool.navvla.lerobot_v3_writer import write_navvla_lerobot_dataset
from tool.navvla.schema import NavVLACameraSpec, NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec
from tool.navvla.statistics import body_frame_action_from_pose


OPENSCENE_VERSION_DIR = "openscene-v1.1"
SOURCE_SPLIT_TO_TARGET = {
    "trainval": "vln_train",
    "test": "vln_val_seen",
    "mini": "vln_train",
}
CAMERA_CHANNELS: tuple[tuple[str, str, str, float], ...] = (
    ("cam_f0", "CAM_F0", "front", 0.0),
    ("cam_b0", "CAM_B0", "rear", math.pi),
    ("cam_l0", "CAM_L0", "front_left", math.pi / 4.0),
    ("cam_l1", "CAM_L1", "left", math.pi / 2.0),
    ("cam_l2", "CAM_L2", "rear_left", 3.0 * math.pi / 4.0),
    ("cam_r0", "CAM_R0", "front_right", -math.pi / 4.0),
    ("cam_r1", "CAM_R1", "right", -math.pi / 2.0),
    ("cam_r2", "CAM_R2", "rear_right", -3.0 * math.pi / 4.0),
)
OPENSCENE_CAMERAS: tuple[NavVLACameraSpec, ...] = tuple(
    NavVLACameraSpec(name=name, video_key=name, viewpoint_type=viewpoint, azimuth_rad=azimuth)
    for name, _channel, viewpoint, azimuth in CAMERA_CHANNELS
)
CAMERA_NAME_TO_CHANNEL = {name: channel for name, channel, _viewpoint, _azimuth in CAMERA_CHANNELS}
CAMERA_CHANNEL_TO_NAME = {channel: name for name, channel, _viewpoint, _azimuth in CAMERA_CHANNELS}
OPENSCENE_CONTEXT_INDEX_CONFIG = ContextIndexConfig(
    budget_num_cameras=len(OPENSCENE_CAMERAS),
    history_camera_names=tuple(camera.name for camera in OPENSCENE_CAMERAS),
)
INSTRUCTION_TEXT = "Drive through the recorded urban scene and imitate the expert ego trajectory."
PLATFORM_TEXT = (
    "Platform: ground vehicle. Task: open-loop urban driving imitation from recorded OpenScene data. "
    "Action: local ego-trajectory waypoints (dx, dy, dz, dyaw)."
)
INSTRUCTION_SOURCE = "conservative_template"
TASK_SUBTYPE = "openscene_open_loop_imitation"
TRAJECTORY_SEMANTICS = "open_loop_ego_trajectory_imitation"
STATE_MODE = "openscene_global_ego_pose_xyz_yaw"


class OpenSceneAdapter(NavVLASourceAdapter):
    name = "openscene"

    def __init__(
        self,
        *,
        media_cache_root: str | Path | None = None,
        reuse_media_cache: bool = False,
        fail_on_missing_media: bool = False,
        fps: float = 2.0,
        action_horizon: int = 8,
    ) -> None:
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.reuse_media_cache = bool(reuse_media_cache)
        self.fail_on_missing_media = bool(fail_on_missing_media)
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.filter_report: dict[str, Any] = {}
        self.summary: dict[str, Any] = {}
        self.load_workers: int | None = None

    def configure(
        self,
        *,
        media_cache_root: str | Path | None = None,
        reuse_media_cache: bool = False,
        fail_on_missing_media: bool = False,
        fps: float = 2.0,
        action_horizon: int = 8,
        load_workers: int | None = None,
        **kwargs: Any,
    ) -> "OpenSceneAdapter":
        super().configure(**kwargs)
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.reuse_media_cache = bool(reuse_media_cache)
        self.fail_on_missing_media = bool(fail_on_missing_media)
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.load_workers = load_workers
        return self

    def load_episodes(
        self,
        source_root: str | Path,
        *,
        split: str = "trainval",
        max_episodes: int | None = None,
        load_workers: int | None = None,
    ) -> list[NavVLAEpisode]:
        source_root = Path(source_root)
        source_split = normalize_source_split(split)
        target_split = target_split_name(source_split)
        version_root = resolve_version_root(source_root)
        scene_names, scene_to_shard = load_scene_shard_map(version_root, source_split=source_split)
        if max_episodes is not None:
            scene_names = scene_names[: int(max_episodes)]
        if not scene_names:
            raise FileNotFoundError(f"no OpenScene scenes found for split={source_split!r} under {version_root}")

        source_scenes = load_metadata_scenes(version_root, source_split=source_split, scene_names=scene_names)
        media_cache_root = resolve_media_cache_root(source_root, media_cache_root=self.media_cache_root)
        episodes, filtered_episodes, original_frame_count, kept_frame_count = load_openscene_episode_batches(
            source_scenes,
            version_root=version_root,
            media_cache_root=media_cache_root,
            source_split=source_split,
            target_split=target_split,
            scene_to_shard=scene_to_shard,
            fps=self.fps,
            action_horizon=self.action_horizon,
            reuse_media_cache=self.reuse_media_cache,
            fail_on_missing_media=self.fail_on_missing_media,
            load_workers=load_workers,
        )
        episodes = renumber_episode_task_indices(episodes)
        self.filter_report = build_filter_report(
            source_split=source_split,
            target_split=target_split,
            original_episode_count=len(source_scenes),
            kept_episode_count=len(episodes),
            original_frame_count=original_frame_count,
            kept_frame_count=kept_frame_count,
            filtered_episodes=filtered_episodes,
            reuse_media_cache=self.reuse_media_cache,
            fail_on_missing_media=self.fail_on_missing_media,
        )
        if not episodes:
            raise FileNotFoundError(f"no media-complete OpenScene episodes found for split={source_split!r} under {version_root}")
        self.summary = {
            "source_root": str(source_root),
            "version_root": str(version_root),
            "source_split": source_split,
            "target_split": target_split,
            "source_episodes": len(episodes),
            "source_frames": sum(len(episode.frames) for episode in episodes),
            "instruction_source": INSTRUCTION_SOURCE,
            "task_text_policy": "fixed conservative imitation template; raw driving_command kept in source_metadata",
            "trajectory_semantics": TRAJECTORY_SEMANTICS,
            "action_mode": "anchor_relative_body_frame_xyz_yaw",
            "state_mode": STATE_MODE,
            "action_horizon": self.action_horizon,
            "action_dz_policy": "fixed_zero_ground_vehicle",
            "camera_names": [camera.name for camera in OPENSCENE_CAMERAS],
            "camera_channels": {camera.name: CAMERA_NAME_TO_CHANNEL[camera.name] for camera in OPENSCENE_CAMERAS},
            "media_cache_root": str(media_cache_root),
            "filtered_episode_count": self.filter_report.get("filtered_episode_count", 0),
        }
        return episodes

    def convert(
        self,
        *,
        source_root: str | Path,
        output_root: str | Path,
        dataset_name: str,
        max_episodes: int | None,
        fps: float,
        action_horizon: int,
        overwrite: bool,
        repair_existing: bool = False,
        split: str = "trainval",
        control_frequency_hz: float | None = None,
        context_policy_version: str = "bats-v1",
        cache_policy_version: str = "smoke-coarse-v1",
        cache_workers: int | None = None,
        write_visual_token_cache: bool = True,
        visual_token_profile: Any | None = None,
        visual_token_encoder: Any | None = None,
        visual_token_encoder_factory: Any | None = None,
        episodes_per_file: int = 20,
        files_per_chunk: int = 50,
        load_workers: int | None = None,
    ) -> dict[str, Any]:
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        source_split = normalize_source_split(split)
        target_split = target_split_name(source_split)
        episodes = self.load_episodes(
            source_root,
            split=source_split,
            max_episodes=max_episodes,
            load_workers=self.load_workers if load_workers is None else load_workers,
        )
        spec = NavVLADatasetSpec(
            dataset_name=dataset_name,
            fps=float(fps),
            control_frequency_hz=float(control_frequency_hz) if control_frequency_hz is not None else float(fps),
            action_horizon=int(action_horizon),
            action_dim=4,
            state_dim=4,
            context_policy_version=context_policy_version,
            cache_policy_version=cache_policy_version,
            split=target_split,
            episodes_per_file=episodes_per_file,
            files_per_chunk=files_per_chunk,
            state_mode=STATE_MODE,
        )
        summary = write_navvla_lerobot_dataset(
            episodes,
            output_root=Path(output_root),
            spec=spec,
            overwrite=overwrite,
            repair_existing=repair_existing,
            cache_workers=cache_workers,
            write_visual_token_cache=write_visual_token_cache,
            visual_token_profile=visual_token_profile,
            visual_token_encoder=visual_token_encoder,
            visual_token_encoder_factory=visual_token_encoder_factory,
            context_index_config=OPENSCENE_CONTEXT_INDEX_CONFIG,
        )
        summary["adapter_summary"] = self.summary
        report_path = update_conversion_report(summary.get("dataset_root"), adapter_summary=self.summary)
        if report_path is not None:
            summary["conversion_report"] = str(report_path)
        filter_report_path = write_filter_report(summary["dataset_root"], self.filter_report)
        summary["openscene_filter_report"] = str(filter_report_path)
        summary["openscene_filtered_episodes"] = self.filter_report.get("filtered_episode_count", 0)
        return summary


def normalize_source_split(split: str) -> str:
    value = str(split).strip().lower()
    aliases = {
        "trainval": "trainval",
        "train": "trainval",
        "vln_train": "trainval",
        "test": "test",
        "val": "test",
        "val_seen": "test",
        "vln_val_seen": "test",
        "mini": "mini",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(f"unsupported OpenScene split for conversion: {split}") from exc


def target_split_name(split: str) -> str:
    return SOURCE_SPLIT_TO_TARGET[normalize_source_split(split)]


def default_dataset_name(split: str) -> str:
    return target_split_name(split)


def resolve_version_root(source_root: str | Path) -> Path:
    source_root = Path(source_root)
    if source_root.name == OPENSCENE_VERSION_DIR:
        version_root = source_root
    else:
        version_root = source_root / OPENSCENE_VERSION_DIR
    if not version_root.is_dir():
        raise FileNotFoundError(f"OpenScene version root not found: {version_root}")
    return version_root


def resolve_media_cache_root(source_root: str | Path, *, media_cache_root: str | Path | None) -> Path:
    if media_cache_root is not None:
        return Path(media_cache_root)
    root = Path(source_root)
    if root.name == OPENSCENE_VERSION_DIR:
        root = root.parent
    return root / ".navvla_media_cache" / "openscene"


def load_scene_shard_map(version_root: Path, *, source_split: str) -> tuple[list[str], dict[str, str]]:
    suffix = "0-199" if source_split == "trainval" else "0-31"
    path = version_root / f"openscene_sensor_{source_split}_{suffix}.json"
    if not path.is_file():
        raise FileNotFoundError(f"OpenScene sensor shard map not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    scene_names: list[str] = []
    scene_to_shard: dict[str, str] = {}
    for shard_key, scenes in payload.items():
        if not isinstance(scenes, list):
            raise ValueError(f"OpenScene shard map entry must be a list: {shard_key}")
        for scene in scenes:
            scene_name = str(scene)
            scene_names.append(scene_name)
            scene_to_shard[scene_name] = str(shard_key)
    return scene_names, scene_to_shard


def load_metadata_scenes(
    version_root: Path,
    *,
    source_split: str,
    scene_names: Sequence[str],
) -> list[tuple[int, str, list[dict[str, Any]]]]:
    archive_path = version_root / f"openscene_metadata_{source_split}.tgz"
    if not archive_path.is_file():
        raise FileNotFoundError(f"OpenScene metadata archive not found: {archive_path}")
    wanted = {str(name) for name in scene_names}
    found: dict[str, list[dict[str, Any]]] = {}
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            if not (member.isfile() and member.name.endswith(".pkl")):
                continue
            scene_name = Path(member.name).stem
            if scene_name not in wanted:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise FileNotFoundError(f"unable to read OpenScene metadata member: {member.name}")
            frames = pickle.load(handle)
            if not isinstance(frames, list):
                raise ValueError(f"OpenScene metadata member must contain a frame list: {member.name}")
            found[scene_name] = frames
            if len(found) == len(wanted):
                break
    missing = [name for name in scene_names if name not in found]
    if missing:
        raise FileNotFoundError(f"OpenScene metadata missing for {len(missing)} scenes; first missing: {missing[0]}")
    return [(index, scene_name, found[scene_name]) for index, scene_name in enumerate(scene_names)]


def load_openscene_episode_batches(
    source_scenes: list[tuple[int, str, list[dict[str, Any]]]],
    *,
    version_root: Path,
    media_cache_root: Path,
    source_split: str,
    target_split: str,
    scene_to_shard: dict[str, str],
    fps: float,
    action_horizon: int,
    reuse_media_cache: bool,
    fail_on_missing_media: bool,
    load_workers: int | None,
) -> tuple[list[NavVLAEpisode], list[dict[str, Any]], int, int]:
    jobs = [
        (
            source_index,
            scene_name,
            frames,
            str(version_root),
            str(media_cache_root),
            source_split,
            target_split,
            scene_to_shard[scene_name],
            float(fps),
            int(action_horizon),
            bool(reuse_media_cache),
            bool(fail_on_missing_media),
        )
        for source_index, scene_name, frames in source_scenes
    ]
    workers = resolve_load_workers(load_workers)
    if workers == 1 or len(jobs) == 1:
        results = [_load_openscene_episode_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            results = list(executor.map(_load_openscene_episode_job, jobs))

    episodes: list[NavVLAEpisode] = []
    filtered: list[dict[str, Any]] = []
    original_frame_count = 0
    kept_frame_count = 0
    for _source_index, episode, filtered_entry, original_frames, kept_frames in results:
        original_frame_count += int(original_frames)
        kept_frame_count += int(kept_frames)
        if episode is not None:
            episodes.append(episode)
        if filtered_entry is not None:
            filtered.append(filtered_entry)
    return episodes, filtered, original_frame_count, kept_frame_count


def resolve_load_workers(load_workers: int | None) -> int:
    if load_workers is None:
        return 1
    if load_workers < 1:
        raise ValueError(f"load_workers must be >= 1, got {load_workers}")
    return int(load_workers)


def _load_openscene_episode_job(job: tuple[Any, ...]) -> tuple[int, NavVLAEpisode | None, dict[str, Any] | None, int, int]:
    (
        source_index,
        scene_name,
        frames,
        version_root_str,
        media_cache_root_str,
        source_split,
        target_split,
        shard_key,
        fps,
        action_horizon,
        reuse_media_cache,
        fail_on_missing_media,
    ) = job
    try:
        episode = build_episode(
            source_index=source_index,
            scene_name=scene_name,
            frames=frames,
            version_root=Path(version_root_str),
            media_cache_root=Path(media_cache_root_str),
            source_split=source_split,
            target_split=target_split,
            shard_key=shard_key,
            fps=float(fps),
            action_horizon=int(action_horizon),
            reuse_media_cache=bool(reuse_media_cache),
        )
        return int(source_index), episode, None, len(frames), len(episode.frames)
    except OpenSceneMediaError as exc:
        if fail_on_missing_media:
            raise
        return int(source_index), None, filtered_episode_entry(source_index, scene_name, frames, exc.issues), len(frames), 0


def build_episode(
    *,
    source_index: int,
    scene_name: str,
    frames: list[dict[str, Any]],
    version_root: Path,
    media_cache_root: Path,
    source_split: str,
    target_split: str,
    shard_key: str,
    fps: float,
    action_horizon: int,
    reuse_media_cache: bool,
) -> NavVLAEpisode:
    if not frames:
        raise ValueError(f"OpenScene scene has no frames: {scene_name}")
    poses = [pose_from_frame(frame) for frame in frames]
    media_by_frame = materialize_episode_media(
        scene_name=scene_name,
        frames=frames,
        version_root=version_root,
        media_cache_root=media_cache_root,
        source_split=source_split,
        target_split=target_split,
        shard_key=shard_key,
        reuse_media_cache=reuse_media_cache,
    )
    first_frame = frames[0]
    first_timestamp_us = int(first_frame.get("timestamp") or 0)
    scene_id = str(first_frame.get("scene_name") or scene_name)
    trajectory_id = str(first_frame.get("scene_token") or scene_name)
    episode_frames: list[NavVLAFrame] = []
    for frame_index, frame in enumerate(frames):
        timestamp_us = int(frame.get("timestamp") or first_timestamp_us)
        timestamp = (timestamp_us - first_timestamp_us) / 1.0e6 if first_timestamp_us else frame_index / float(fps)
        episode_frames.append(
            NavVLAFrame(
                frame_index=frame_index,
                timestamp=float(timestamp),
                media_paths=media_by_frame[frame_index],
                state=[clean_float(value) for value in poses[frame_index]],
                action=action_chunk_for_frame(poses, frame_idx=frame_index, horizon=action_horizon),
                action_available=True,
                source_frame_index=int(frame.get("frame_idx", frame_index)),
                source_metadata=source_metadata_for_frame(
                    frame,
                    source_split=source_split,
                    target_split=target_split,
                    scene_name=scene_name,
                    shard_key=shard_key,
                    camera_archive=str(camera_archive_for_shard(version_root, source_split=source_split, shard_key=shard_key)),
                ),
            )
        )
    task = NavVLATaskSpec(
        task_index=int(source_index),
        instruction=INSTRUCTION_TEXT,
        task_type="driving",
        task_subtype=TASK_SUBTYPE,
        platform_text=PLATFORM_TEXT,
        dataset_source="openscene",
        scene_id=scene_id,
    )
    return NavVLAEpisode(
        episode_id=str(scene_name),
        trajectory_id=trajectory_id,
        task=task,
        frames=episode_frames,
        cameras=camera_specs_from_frame(first_frame),
        split=target_split,
    )


def materialize_episode_media(
    *,
    scene_name: str,
    frames: list[dict[str, Any]],
    version_root: Path,
    media_cache_root: Path,
    source_split: str,
    target_split: str,
    shard_key: str,
    reuse_media_cache: bool,
) -> list[dict[str, Path]]:
    archive_path = camera_archive_for_shard(version_root, source_split=source_split, shard_key=shard_key)
    issues: list[dict[str, Any]] = []
    media_by_frame: list[dict[str, Path]] = [{} for _ in frames]
    pending_members: dict[str, dict[str, Any]] = {}
    if not archive_path.is_file() and not reuse_media_cache:
        raise OpenSceneMediaError(
            [
                {
                    "frame_index": 0,
                    "camera_name": "",
                    "reason": "missing_camera_archive",
                    "camera_archive": str(archive_path),
                    "message": f"OpenScene camera archive not found: {archive_path}",
                }
            ]
        )

    for frame_index, frame in enumerate(frames):
        cams = frame.get("cams")
        if not isinstance(cams, dict):
            issues.append(media_issue(frame_index, "", "missing_cams", "OpenScene frame has no cams mapping"))
            continue
        for camera_name, channel, _viewpoint, _azimuth in CAMERA_CHANNELS:
            camera_info = cams.get(channel)
            if not isinstance(camera_info, dict):
                issues.append(media_issue(frame_index, camera_name, "missing_camera_metadata", f"missing camera metadata for {channel}"))
                continue
            data_path = str(camera_info.get("data_path") or "")
            if not data_path:
                issues.append(media_issue(frame_index, camera_name, "empty_camera_data_path", f"empty data_path for {channel}"))
                continue
            cache_path = media_cache_root / target_split / scene_name / camera_name / f"{frame_index:06d}.jpg"
            if cache_path.exists() and cache_path.stat().st_size > 0:
                media_by_frame[frame_index][camera_name] = cache_path
                continue
            if reuse_media_cache:
                issues.append(
                    media_issue(
                        frame_index,
                        camera_name,
                        "missing_cached_image",
                        f"missing cached OpenScene image: {cache_path}",
                        cache_path=str(cache_path),
                        source_data_path=data_path,
                    )
                )
                continue
            member_name = f"{OPENSCENE_VERSION_DIR}/sensor_blobs/{source_split}/{data_path}"
            pending_members[normalize_archive_member_name(member_name)] = {
                "frame_index": frame_index,
                "camera_name": camera_name,
                "cache_path": cache_path,
                "member_name": member_name,
                "source_data_path": data_path,
            }

    if pending_members and not reuse_media_cache:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                key = normalize_archive_member_name(member.name)
                pending = pending_members.pop(key, None)
                if pending is None:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    issues.append(
                        media_issue(
                            pending["frame_index"],
                            pending["camera_name"],
                            "unreadable_camera_member",
                            f"unable to read OpenScene camera member: {member.name}",
                            camera_archive=str(archive_path),
                            source_data_path=pending["source_data_path"],
                        )
                    )
                    continue
                cache_path = pending["cache_path"]
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(handle.read())
                media_by_frame[pending["frame_index"]][pending["camera_name"]] = cache_path
                if not pending_members:
                    break
        for pending in pending_members.values():
            issues.append(
                media_issue(
                    pending["frame_index"],
                    pending["camera_name"],
                    "missing_camera_member",
                    f"OpenScene camera member not found: {pending['member_name']}",
                    camera_archive=str(archive_path),
                    source_data_path=pending["source_data_path"],
                )
            )
    if issues:
        raise OpenSceneMediaError(issues)
    return media_by_frame


def normalize_archive_member_name(member_name: str) -> str:
    prefix = f"{OPENSCENE_VERSION_DIR}/"
    return member_name[len(prefix) :] if member_name.startswith(prefix) else member_name


def camera_archive_for_shard(version_root: Path, *, source_split: str, shard_key: str) -> Path:
    shard_index = str(shard_key).rsplit("_", 1)[-1]
    return version_root / f"openscene_sensor_{source_split}_camera" / f"openscene_sensor_{source_split}_camera_{shard_index}.tgz"


def pose_from_frame(frame: dict[str, Any]) -> list[float]:
    translation = frame.get("ego2global_translation")
    rotation = frame.get("ego2global_rotation")
    if translation is None or rotation is None:
        raise ValueError("OpenScene frame requires ego2global_translation and ego2global_rotation")
    x, y, z = (clean_float(value) for value in translation[:3])
    yaw = clean_float(quaternion_yaw(rotation))
    return [x, y, z, yaw]


def action_chunk_for_frame(poses: Sequence[Sequence[float]], *, frame_idx: int, horizon: int) -> list[list[float]]:
    current = poses[frame_idx]
    chunk: list[list[float]] = []
    for future_idx in range(frame_idx + 1, min(len(poses), frame_idx + 1 + int(horizon))):
        action = body_frame_action_from_pose(current, poses[future_idx]).astype(float)
        action[2] = 0.0
        chunk.append([clean_float(value) for value in action.tolist()])
    return chunk


def camera_specs_from_frame(frame: dict[str, Any]) -> list[NavVLACameraSpec]:
    cams = frame.get("cams")
    if not isinstance(cams, dict):
        raise ValueError("OpenScene frame requires cams mapping")
    specs: list[NavVLACameraSpec] = []
    for base in OPENSCENE_CAMERAS:
        channel = CAMERA_NAME_TO_CHANNEL[base.name]
        camera_info = cams.get(channel, {})
        intrinsics = matrix_or_none(camera_info.get("cam_intrinsic") if isinstance(camera_info, dict) else None)
        extrinsics_body = extrinsics_from_camera_info(camera_info) if isinstance(camera_info, dict) else None
        specs.append(
            NavVLACameraSpec(
                name=base.name,
                video_key=base.video_key,
                viewpoint_type=base.viewpoint_type,
                azimuth_rad=base.azimuth_rad,
                intrinsics=intrinsics,
                extrinsics_body=extrinsics_body,
                calibration_status="openscene-camera" if intrinsics is not None or extrinsics_body is not None else "unknown",
            )
        )
    return specs


def extrinsics_from_camera_info(camera_info: dict[str, Any]) -> list[list[float]] | None:
    rotation = camera_info.get("sensor2lidar_rotation")
    translation = camera_info.get("sensor2lidar_translation")
    if rotation is None or translation is None:
        return None
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trans = np.asarray(translation, dtype=np.float64).reshape(3)
    return [
        [clean_float(rot[0, 0]), clean_float(rot[0, 1]), clean_float(rot[0, 2]), clean_float(trans[0])],
        [clean_float(rot[1, 0]), clean_float(rot[1, 1]), clean_float(rot[1, 2]), clean_float(trans[1])],
        [clean_float(rot[2, 0]), clean_float(rot[2, 1]), clean_float(rot[2, 2]), clean_float(trans[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrix_or_none(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        return None
    return [[clean_float(item) for item in row] for row in matrix.tolist()]


def source_metadata_for_frame(
    frame: dict[str, Any],
    *,
    source_split: str,
    target_split: str,
    scene_name: str,
    shard_key: str,
    camera_archive: str,
) -> dict[str, Any]:
    camera_sources = {}
    cams = frame.get("cams") if isinstance(frame.get("cams"), dict) else {}
    for camera_name, channel, _viewpoint, _azimuth in CAMERA_CHANNELS:
        camera_info = cams.get(channel, {}) if isinstance(cams, dict) else {}
        camera_sources[camera_name] = str(camera_info.get("data_path") or "")
    return {
        "source_dataset": "openscene",
        "source_split": source_split,
        "target_split": target_split,
        "instruction_source": INSTRUCTION_SOURCE,
        "task_text_policy": "fixed conservative imitation template; raw driving_command kept in source_metadata",
        "trajectory_semantics": TRAJECTORY_SEMANTICS,
        "scene_name": str(frame.get("scene_name") or scene_name),
        "scene_token": str(frame.get("scene_token") or ""),
        "log_name": str(frame.get("log_name") or scene_name),
        "log_token": str(frame.get("log_token") or ""),
        "map_location": none_if_empty(frame.get("map_location")),
        "vehicle_name": none_if_empty(frame.get("vehicle_name")),
        "openscene_token": str(frame.get("token") or ""),
        "openscene_frame_idx": int(frame.get("frame_idx", 0)),
        "openscene_timestamp_us": int(frame.get("timestamp") or 0),
        "driving_command": to_builtin(frame.get("driving_command")),
        "ego_dynamic_state": to_builtin(frame.get("ego_dynamic_state")),
        "traffic_lights": to_builtin(frame.get("traffic_lights")),
        "roadblock_ids": to_builtin(frame.get("roadblock_ids")),
        "annotation_summary": annotation_summary(frame.get("anns")),
        "camera_channels": {camera_name: CAMERA_NAME_TO_CHANNEL[camera_name] for camera_name in CAMERA_NAME_TO_CHANNEL},
        "camera_sources": camera_sources,
        "camera_shard_key": shard_key,
        "camera_archive": camera_archive,
        "action_dz_policy": "fixed_zero_ground_vehicle",
        "unused_modalities_downloaded": False,
        "lidar_path": str(frame.get("lidar_path") or ""),
        "occ_gt_final_path": str(frame.get("occ_gt_final_path") or ""),
        "flow_gt_final_path": str(frame.get("flow_gt_final_path") or ""),
    }


def annotation_summary(anns: Any) -> dict[str, Any]:
    if not isinstance(anns, dict):
        return {"available": False}
    names = anns.get("gt_names")
    names_list = [str(name) for name in np.asarray(names).reshape(-1).tolist()] if names is not None else []
    boxes = anns.get("gt_boxes")
    box_count = int(np.asarray(boxes).shape[0]) if boxes is not None and np.asarray(boxes).ndim >= 1 else len(names_list)
    return {
        "available": True,
        "gt_box_count": box_count,
        "gt_name_counts": dict(sorted(Counter(names_list).items())),
        "has_gt_velocity_3d": anns.get("gt_velocity_3d") is not None,
        "instance_count": len(anns.get("instance_tokens") or []),
        "track_count": len(anns.get("track_tokens") or []),
    }


def filtered_episode_entry(
    source_index: int,
    scene_name: str,
    frames: list[dict[str, Any]],
    media_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    for issue in media_issues:
        reason = str(issue.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    first = frames[0] if frames else {}
    return {
        "episode_index": int(source_index),
        "episode_id": str(scene_name),
        "trajectory_id": str(first.get("scene_token") or scene_name),
        "scene_id": str(first.get("scene_name") or scene_name),
        "episode_frame_count": len(frames),
        "missing_media_count": len(media_issues),
        "missing_media_reason_counts": reason_counts,
        "missing_media_examples": media_issues[:20],
    }


def build_filter_report(
    *,
    source_split: str,
    target_split: str,
    original_episode_count: int,
    kept_episode_count: int,
    original_frame_count: int,
    kept_frame_count: int,
    filtered_episodes: list[dict[str, Any]],
    reuse_media_cache: bool,
    fail_on_missing_media: bool,
) -> dict[str, Any]:
    filtered_frame_count = sum(int(entry["episode_frame_count"]) for entry in filtered_episodes)
    return {
        "dataset": "openscene",
        "source_split": source_split,
        "target_split": target_split,
        "filter_policy": "skip_episode_on_missing_or_invalid_media",
        "filter_granularity": "episode",
        "reuse_media_cache": bool(reuse_media_cache),
        "fail_on_missing_media": bool(fail_on_missing_media),
        "original_episode_count": int(original_episode_count),
        "kept_episode_count": int(kept_episode_count),
        "filtered_episode_count": int(len(filtered_episodes)),
        "original_frame_count": int(original_frame_count),
        "kept_frame_count": int(kept_frame_count),
        "filtered_frame_count": int(filtered_frame_count),
        "filtered_episodes": filtered_episodes,
    }


def write_filter_report(dataset_root: str | Path, report: dict[str, Any]) -> Path:
    path = Path(dataset_root) / "meta" / "openscene_filtered_episodes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def update_conversion_report(dataset_root_value: Any, *, adapter_summary: dict[str, Any]) -> Path | None:
    if not dataset_root_value:
        return None
    report_path = Path(dataset_root_value) / "conversion_report.json"
    if not report_path.exists():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "instruction_source": INSTRUCTION_SOURCE,
            "task_text_policy": adapter_summary.get("task_text_policy"),
            "trajectory_semantics": TRAJECTORY_SEMANTICS,
            "dataset_source": "openscene",
            "source_split": adapter_summary.get("source_split"),
            "target_split": adapter_summary.get("target_split"),
            "camera_names": adapter_summary.get("camera_names"),
            "camera_channels": adapter_summary.get("camera_channels"),
            "state_mode": STATE_MODE,
            "action_dz_policy": "fixed_zero_ground_vehicle",
        }
    )
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def renumber_episode_task_indices(episodes: list[NavVLAEpisode]) -> list[NavVLAEpisode]:
    out: list[NavVLAEpisode] = []
    for task_index, episode in enumerate(episodes):
        out.append(replace(episode, task=replace(episode.task, task_index=task_index)))
    return out


def media_issue(frame_index: int, camera_name: str, reason: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "frame_index": int(frame_index),
        "camera_name": camera_name,
        "reason": reason,
        "message": message,
        **extra,
    }


class OpenSceneMediaError(FileNotFoundError):
    def __init__(self, issues: list[dict[str, Any]]) -> None:
        self.issues = issues
        first = issues[0] if issues else {}
        super().__init__(str(first.get("message") or "OpenScene media issue"))


def quaternion_yaw(rotation: Iterable[float]) -> float:
    w, x, y, z = (float(value) for value in rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clean_float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"OpenScene numeric value must be finite, got {value}")
    return 0.0 if abs(out) < 1e-7 else out


def none_if_empty(value: Any) -> Any:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def to_builtin(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


register_adapter(OpenSceneAdapter())
