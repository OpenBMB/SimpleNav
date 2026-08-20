from __future__ import annotations

import json
import math
from bisect import bisect_left
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tool.navvla.adapters._nuscenes_splits import create_splits_scenes
from tool.navvla.adapters.base import NavVLASourceAdapter, register_adapter
from tool.navvla.context_index import ContextIndexConfig
from tool.navvla.lerobot_v3_writer import write_navvla_lerobot_dataset
from tool.navvla.schema import NavVLACameraSpec, NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec
from tool.navvla.statistics import body_frame_action_from_pose


CAMERA_CHANNELS: tuple[tuple[str, str, str, float], ...] = (
    ("front", "CAM_FRONT", "front", 0.0),
    ("front_left", "CAM_FRONT_LEFT", "front_left", math.pi / 4.0),
    ("front_right", "CAM_FRONT_RIGHT", "front_right", -math.pi / 4.0),
    ("back", "CAM_BACK", "back", math.pi),
    ("back_left", "CAM_BACK_LEFT", "back_left", 3.0 * math.pi / 4.0),
    ("back_right", "CAM_BACK_RIGHT", "back_right", -3.0 * math.pi / 4.0),
)
NUSCENES_CAMERAS: tuple[NavVLACameraSpec, ...] = tuple(
    NavVLACameraSpec(name=name, video_key=name, viewpoint_type=viewpoint, azimuth_rad=azimuth)
    for name, _channel, viewpoint, azimuth in CAMERA_CHANNELS
)
CAMERA_NAME_TO_CHANNEL = {name: channel for name, channel, _viewpoint, _azimuth in CAMERA_CHANNELS}
CAMERA_CHANNEL_TO_NAME = {channel: name for name, channel, _viewpoint, _azimuth in CAMERA_CHANNELS}
NUSCENES_CONTEXT_INDEX_CONFIG = ContextIndexConfig(
    budget_num_cameras=len(NUSCENES_CAMERAS),
    history_camera_names=tuple(camera.name for camera in NUSCENES_CAMERAS),
)

PLATFORM_TEXT = (
    "Platform: ground vehicle. Task: open-loop urban driving from nuScenes scene description. "
    "Action: local ego-trajectory waypoints (dx, dy, dz, dyaw)."
)
INSTRUCTION_SOURCE = "frame.generated_from_scene_description_ego_state_and_diffusiondrive_command"
TASK_TEXT_POLICY = "frame template with scene.description, DiffusionDrive-style command, ego dynamics, and 3 past waypoints"
TASK_SUBTYPE = "nuscenes_frame_instruction"
TRAJECTORY_SEMANTICS = "open_loop_ego_trajectory_imitation"
COMMAND_SOURCE = "diffusiondrive_final_lateral_offset_6_keyframes"
DYNAMICS_SOURCE = "can_bus_pose_with_state_fallback"
COMMAND_HORIZON_KEYFRAMES = 6
COMMAND_LATERAL_THRESHOLD_M = 2.0
PAST_TRAJECTORY_WAYPOINTS = 3


class NuScenesAdapter(NavVLASourceAdapter):
    name = "nuscenes"

    def __init__(self, *, dataset_version: str = "v1.0-trainval", action_horizon: int = 8) -> None:
        self.dataset_version = str(dataset_version)
        self.action_horizon = int(action_horizon)
        self.summary: dict[str, Any] = {}

    def configure(
        self,
        *,
        dataset_version: str = "v1.0-trainval",
        action_horizon: int = 8,
        fps: float = 2.0,
        **kwargs: Any,
    ) -> "NuScenesAdapter":
        del fps
        super().configure(**kwargs)
        self.dataset_version = str(dataset_version)
        self.action_horizon = int(action_horizon)
        return self

    def load_episodes(
        self,
        source_root: str | Path,
        *,
        split: str = "train",
        max_episodes: int | None = None,
    ) -> list[NavVLAEpisode]:
        source_root = Path(source_root)
        source_split = source_split_name(split)
        target_split = target_split_name(source_split)
        meta = self._load_metadata(source_root)
        scene_names = selected_scene_names(source_split, available_scene_names=set(meta["scene_by_name"]))
        if max_episodes is not None:
            scene_names = scene_names[: int(max_episodes)]
        if not scene_names:
            raise FileNotFoundError(f"no nuScenes scenes found for split={source_split!r} under {source_root}")

        episodes: list[NavVLAEpisode] = []
        next_task_index = 0
        for scene_name in scene_names:
            scene = meta["scene_by_name"][scene_name]
            samples = list(walk_scene_samples(scene, meta["sample_by_token"]))
            if not samples:
                continue
            poses, pose_metadata = poses_for_samples(samples, meta)
            cameras = camera_specs_for_episode(samples[0], meta)
            frames = frames_for_scene(
                source_root=source_root,
                scene=scene,
                samples=samples,
                poses=poses,
                pose_metadata=pose_metadata,
                meta=meta,
                source_split=source_split,
                target_split=target_split,
                action_horizon=self.action_horizon,
                task_index_start=next_task_index,
            )
            next_task_index += len(frames)
            task = frames[0].task
            if task is None:
                raise ValueError(f"nuScenes scene {scene_name} produced frames without frame-level tasks")
            episodes.append(
                NavVLAEpisode(
                    episode_id=str(scene["name"]),
                    trajectory_id=str(scene["token"]),
                    task=task,
                    frames=frames,
                    cameras=cameras,
                    split=target_split,
                )
            )

        if not episodes:
            raise FileNotFoundError(f"no convertible nuScenes scenes found for split={source_split!r} under {source_root}")
        self.summary = {
            "source_root": str(source_root),
            "dataset_version": self.dataset_version,
            "source_split": source_split,
            "target_split": target_split,
            "source_episodes": len(episodes),
            "source_frames": sum(len(episode.frames) for episode in episodes),
            "instruction_source": INSTRUCTION_SOURCE,
            "task_text_policy": TASK_TEXT_POLICY,
            "instruction_granularity": "frame",
            "command_source": COMMAND_SOURCE,
            "command_horizon_keyframes": COMMAND_HORIZON_KEYFRAMES,
            "command_lateral_threshold_m": COMMAND_LATERAL_THRESHOLD_M,
            "dynamics_source": DYNAMICS_SOURCE,
            "trajectory_semantics": TRAJECTORY_SEMANTICS,
            "action_mode": "anchor_relative_body_frame_xyz_yaw",
            "action_horizon": self.action_horizon,
            "camera_names": [camera.name for camera in NUSCENES_CAMERAS],
            "camera_channels": {camera.name: CAMERA_NAME_TO_CHANNEL[camera.name] for camera in NUSCENES_CAMERAS},
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
        control_frequency_hz: float | None = None,
        repair_existing: bool = False,
        split: str = "train",
        context_policy_version: str = "bats-v1",
        cache_policy_version: str = "smoke-coarse-v1",
        cache_workers: int | None = None,
        write_visual_token_cache: bool = True,
        visual_token_profile: Any | None = None,
        visual_token_encoder: Any | None = None,
        visual_token_encoder_factory: Any | None = None,
        episodes_per_file: int = 20,
        files_per_chunk: int = 50,
    ) -> dict[str, Any]:
        self.action_horizon = int(action_horizon)
        source_split = source_split_name(split)
        target_split = target_split_name(source_split)
        episodes = self.load_episodes(source_root, split=source_split, max_episodes=max_episodes)
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
            state_mode="nuscenes_global_ego_pose_xyz_yaw",
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
            context_index_config=NUSCENES_CONTEXT_INDEX_CONFIG,
        )
        summary["adapter_summary"] = self.summary
        report_path = update_conversion_report(summary.get("dataset_root"), adapter_summary=self.summary)
        if report_path is not None:
            summary["conversion_report"] = str(report_path)
        return summary

    def _meta_root(self, source_root: Path) -> Path:
        return Path(source_root) / self.dataset_version

    def _load_table(self, source_root: Path, name: str) -> list[dict[str, Any]]:
        path = self._meta_root(source_root) / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing nuScenes metadata table: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_metadata(self, source_root: Path) -> dict[str, Any]:
        scenes = self._load_table(source_root, "scene")
        samples = self._load_table(source_root, "sample")
        sample_data = self._load_table(source_root, "sample_data")
        ego_poses = self._load_table(source_root, "ego_pose")
        calibrated_sensors = self._load_table(source_root, "calibrated_sensor")
        sensors = self._load_table(source_root, "sensor")
        logs = self._load_table(source_root, "log")
        return {
            "scene_by_name": {str(row["name"]): row for row in scenes},
            "sample_by_token": {str(row["token"]): row for row in samples},
            "sample_data_by_sample": index_keyframe_sample_data(sample_data, calibrated_sensors, sensors),
            "ego_pose_by_token": {str(row["token"]): row for row in ego_poses},
            "calibrated_sensor_by_token": {str(row["token"]): row for row in calibrated_sensors},
            "sensor_by_token": {str(row["token"]): row for row in sensors},
            "log_by_token": {str(row["token"]): row for row in logs},
        }


def source_split_name(split: str) -> str:
    normalized = split.strip().lower()
    aliases = {
        "train": "train",
        "vln_train": "train",
        "val": "val",
        "val_seen": "val",
        "vln_val_seen": "val",
        "mini_train": "mini_train",
        "mini_val": "mini_val",
        "all": "all",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported nuScenes split for conversion: {split}") from exc


def target_split_name(split: str) -> str:
    source_split = source_split_name(split)
    if source_split in {"train", "mini_train"}:
        return "vln_train"
    if source_split in {"val", "mini_val"}:
        return "vln_val_seen"
    if source_split == "all":
        return "all"
    raise ValueError(f"unsupported nuScenes split for conversion: {split}")


def default_dataset_name(split: str) -> str:
    target_split = target_split_name(split)
    if target_split not in {"vln_train", "vln_val_seen"}:
        raise ValueError(f"nuScenes conversion does not define a default dataset name for split={split!r}")
    return target_split


def selected_scene_names(split: str, *, available_scene_names: set[str]) -> list[str]:
    source_split = source_split_name(split)
    if source_split == "all":
        return sorted(available_scene_names)
    splits = create_splits_scenes()
    if source_split not in splits:
        raise ValueError(f"nuScenes split {source_split!r} is not in official split helper")
    return sorted(name for name in splits[source_split] if name in available_scene_names)


def scene_description_text(scene: dict[str, Any]) -> str:
    description = str(scene.get("description") or "").strip()
    if description:
        return description
    return f"nuScenes recorded driving scene {scene['name']}"


def walk_scene_samples(scene: dict[str, Any], sample_by_token: dict[str, dict[str, Any]]) -> Iterable[dict[str, Any]]:
    token = str(scene["first_sample_token"])
    last_token = str(scene["last_sample_token"])
    seen: set[str] = set()
    while token:
        if token in seen:
            raise ValueError(f"cycle detected in nuScenes sample chain for scene {scene['name']}: {token}")
        seen.add(token)
        sample = sample_by_token[token]
        yield sample
        if token == last_token:
            break
        token = str(sample.get("next") or "")


def poses_for_samples(samples: Sequence[dict[str, Any]], meta: dict[str, Any]) -> tuple[list[list[float]], list[dict[str, Any]]]:
    poses: list[list[float]] = []
    pose_metadata: list[dict[str, Any]] = []
    for sample in samples:
        sample_token = str(sample["token"])
        sd_by_channel = meta["sample_data_by_sample"].get(sample_token, {})
        lidar = sd_by_channel.get("LIDAR_TOP")
        if lidar is None:
            raise KeyError(f"nuScenes sample {sample_token} has no LIDAR_TOP keyframe sample_data")
        ego_pose_token = str(lidar["ego_pose_token"])
        ego_pose = meta["ego_pose_by_token"][ego_pose_token]
        x, y, z = (clean_float(value) for value in ego_pose["translation"])
        yaw = clean_float(quaternion_yaw(ego_pose["rotation"]))
        poses.append([x, y, z, yaw])
        pose_metadata.append(
            {
                "ego_pose_token": ego_pose_token,
                "lidar_top_filename": str(lidar.get("filename") or ""),
                "lidar_top_sample_data_token": str(lidar.get("token") or ""),
            }
        )
    return poses, pose_metadata


def frames_for_scene(
    *,
    source_root: Path,
    scene: dict[str, Any],
    samples: Sequence[dict[str, Any]],
    poses: Sequence[Sequence[float]],
    pose_metadata: Sequence[dict[str, Any]],
    meta: dict[str, Any],
    source_split: str,
    target_split: str,
    action_horizon: int,
    task_index_start: int,
) -> list[NavVLAFrame]:
    description = scene_description_text(scene)
    log = meta["log_by_token"].get(str(scene.get("log_token") or ""), {})
    first_timestamp_us = int(samples[0].get("timestamp") or 0)
    dynamics = dynamics_for_scene(source_root=source_root, scene_name=str(scene["name"]), samples=samples, poses=poses)
    frames: list[NavVLAFrame] = []
    for frame_index, sample in enumerate(samples):
        media_paths, camera_sources, camera_calibrated_tokens = camera_media_paths(sample, source_root=source_root, meta=meta)
        timestamp_us = int(sample.get("timestamp") or first_timestamp_us)
        command_onehot, command_text = diffusiondrive_command_for_sample_frame(samples, meta, frame_idx=frame_index)
        past_waypoints = past_waypoints_for_frame(poses, frame_idx=frame_index)
        dynamics_row = dynamics[frame_index]
        instruction = frame_instruction_text(
            scene_description=description,
            command_text=command_text,
            velocity=float(dynamics_row["velocity"]),
            acceleration=float(dynamics_row["acceleration"]),
            yaw_rate=float(dynamics_row["yaw_rate"]),
            past_waypoints=past_waypoints,
        )
        task_metadata = {
            "scene_name": str(scene["name"]),
            "scene_token": str(scene["token"]),
            "sample_token": str(sample["token"]),
            "nuscenes_sample_token": str(sample["token"]),
            "frame_index": int(frame_index),
            "source_frame_index": int(frame_index),
            "command_onehot": command_onehot,
            "command_text": command_text,
            "command_source": COMMAND_SOURCE,
            "command_horizon_keyframes": COMMAND_HORIZON_KEYFRAMES,
            "command_lateral_threshold_m": COMMAND_LATERAL_THRESHOLD_M,
            "instruction_granularity": "frame",
            "dynamics_source": str(dynamics_row["dynamics_source"]),
        }
        frame_task = NavVLATaskSpec(
            task_index=int(task_index_start + frame_index),
            instruction=instruction,
            task_type="driving",
            task_subtype=TASK_SUBTYPE,
            platform_text=PLATFORM_TEXT,
            dataset_source="nuscenes",
            scene_id=str(scene["name"]),
            metadata=task_metadata,
        )
        source_metadata = {
            "source_dataset": "nuscenes",
            "source_split": source_split,
            "target_split": target_split,
            "instruction_source": INSTRUCTION_SOURCE,
            "task_text_policy": TASK_TEXT_POLICY,
            "instruction_granularity": "frame",
            "trajectory_semantics": TRAJECTORY_SEMANTICS,
            "scene_name": str(scene["name"]),
            "scene_token": str(scene["token"]),
            "scene_description": description,
            "nuscenes_sample_token": str(sample["token"]),
            "nuscenes_timestamp_us": timestamp_us,
            "timestamp_s": (timestamp_us - first_timestamp_us) / 1.0e6,
            "location": log.get("location"),
            "log_token": scene.get("log_token"),
            "camera_channels": {name: CAMERA_NAME_TO_CHANNEL[name] for name in CAMERA_NAME_TO_CHANNEL},
            "camera_sources": camera_sources,
            "camera_calibrated_sensor_tokens": camera_calibrated_tokens,
            "action_dz_policy": "fixed_zero_ground_vehicle",
            "command_onehot": command_onehot,
            "command_text": command_text,
            "command_source": COMMAND_SOURCE,
            "dynamics_source": str(dynamics_row["dynamics_source"]),
            "instruction_velocity_mps": clean_float(dynamics_row["velocity"]),
            "instruction_acceleration_mps2": clean_float(dynamics_row["acceleration"]),
            "instruction_yaw_rate_radps": clean_float(dynamics_row["yaw_rate"]),
            "past_trajectory_waypoints": [[clean_float(value) for value in waypoint] for waypoint in past_waypoints],
            **pose_metadata[frame_index],
        }
        frames.append(
            NavVLAFrame(
                frame_index=int(frame_index),
                timestamp=float((timestamp_us - first_timestamp_us) / 1.0e6),
                media_paths=media_paths,
                state=[clean_float(value) for value in poses[frame_index]],
                action=action_chunk_for_frame(poses, frame_idx=frame_index, horizon=action_horizon),
                action_available=True,
                source_frame_index=int(frame_index),
                source_metadata=source_metadata,
                task=frame_task,
            )
        )
    return frames


def action_chunk_for_frame(poses: Sequence[Sequence[float]], *, frame_idx: int, horizon: int) -> list[list[float]]:
    current = poses[frame_idx]
    chunk: list[list[float]] = []
    for future_idx in range(frame_idx + 1, min(len(poses), frame_idx + 1 + int(horizon))):
        action = body_frame_action_from_pose(current, poses[future_idx]).astype(float)
        action[2] = 0.0
        chunk.append([clean_float(value) for value in action.tolist()])
    return chunk


def diffusiondrive_command_for_frame(
    poses: Sequence[Sequence[float]],
    *,
    frame_idx: int,
    horizon: int = COMMAND_HORIZON_KEYFRAMES,
    lateral_threshold_m: float = COMMAND_LATERAL_THRESHOLD_M,
) -> tuple[list[int], str]:
    if not poses:
        raise ValueError("poses must be non-empty")
    if frame_idx < 0 or frame_idx >= len(poses):
        raise IndexError(f"frame_idx out of range: {frame_idx}")
    future_idx = min(int(frame_idx) + int(horizon), len(poses) - 1)
    relative = body_frame_action_from_pose(poses[frame_idx], poses[future_idx]).astype(float)
    dy_body = float(relative[1])
    if dy_body <= -float(lateral_threshold_m):
        return [1, 0, 0], "Turn Right"
    if dy_body >= float(lateral_threshold_m):
        return [0, 1, 0], "Turn Left"
    return [0, 0, 1], "Go Straight"


def diffusiondrive_command_for_sample_frame(
    samples: Sequence[dict[str, Any]],
    meta: dict[str, Any],
    *,
    frame_idx: int,
    horizon: int = COMMAND_HORIZON_KEYFRAMES,
    lateral_threshold_m: float = COMMAND_LATERAL_THRESHOLD_M,
) -> tuple[list[int], str]:
    if not samples:
        raise ValueError("samples must be non-empty")
    if frame_idx < 0 or frame_idx >= len(samples):
        raise IndexError(f"frame_idx out of range: {frame_idx}")
    future_idx = min(int(frame_idx) + int(horizon), len(samples) - 1)
    offset = diffusiondrive_lidar_frame_future_offset(samples[frame_idx], samples[future_idx], meta)
    return diffusiondrive_command_from_lidar_x(float(offset[0]), lateral_threshold_m=lateral_threshold_m)


def diffusiondrive_command_from_lidar_x(lidar_x: float, *, lateral_threshold_m: float = COMMAND_LATERAL_THRESHOLD_M) -> tuple[list[int], str]:
    if float(lidar_x) >= float(lateral_threshold_m):
        return [1, 0, 0], "Turn Right"
    if float(lidar_x) <= -float(lateral_threshold_m):
        return [0, 1, 0], "Turn Left"
    return [0, 0, 1], "Go Straight"


def diffusiondrive_lidar_frame_future_offset(
    current_sample: dict[str, Any],
    future_sample: dict[str, Any],
    meta: dict[str, Any],
) -> np.ndarray:
    current_lidar = lidar_top_sample_data(current_sample, meta)
    future_lidar = lidar_top_sample_data(future_sample, meta)
    current_ego = meta["ego_pose_by_token"][str(current_lidar["ego_pose_token"])]
    future_ego = meta["ego_pose_by_token"][str(future_lidar["ego_pose_token"])]
    current_calibrated = meta["calibrated_sensor_by_token"][str(current_lidar["calibrated_sensor_token"])]
    future_calibrated = meta["calibrated_sensor_by_token"][str(future_lidar["calibrated_sensor_token"])]

    current_ego_translation = np.asarray(current_ego["translation"], dtype=np.float64)
    current_ego_rotation = np.asarray(quaternion_to_rotation_matrix(current_ego["rotation"]), dtype=np.float64)
    current_lidar_translation = np.asarray(current_calibrated["translation"], dtype=np.float64)
    current_lidar_rotation = np.asarray(quaternion_to_rotation_matrix(current_calibrated["rotation"]), dtype=np.float64)

    future_ego_translation = np.asarray(future_ego["translation"], dtype=np.float64)
    future_ego_rotation = np.asarray(quaternion_to_rotation_matrix(future_ego["rotation"]), dtype=np.float64)
    future_lidar_translation = np.asarray(future_calibrated["translation"], dtype=np.float64)

    future_lidar_global = future_ego_translation + future_ego_rotation @ future_lidar_translation
    future_in_current_ego = current_ego_rotation.T @ (future_lidar_global - current_ego_translation)
    return current_lidar_rotation.T @ (future_in_current_ego - current_lidar_translation)


def lidar_top_sample_data(sample: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    sample_token = str(sample["token"])
    sd_by_channel = meta["sample_data_by_sample"].get(sample_token, {})
    lidar = sd_by_channel.get("LIDAR_TOP")
    if lidar is None:
        raise KeyError(f"nuScenes sample {sample_token} has no LIDAR_TOP keyframe sample_data")
    return lidar


def past_waypoints_for_frame(
    poses: Sequence[Sequence[float]],
    *,
    frame_idx: int,
    history: int = PAST_TRAJECTORY_WAYPOINTS,
) -> list[list[float]]:
    if frame_idx < 0 or frame_idx >= len(poses):
        raise IndexError(f"frame_idx out of range: {frame_idx}")
    current = poses[frame_idx]
    start = max(0, int(frame_idx) - int(history))
    waypoints: list[list[float]] = []
    for past_idx in range(start, int(frame_idx)):
        relative = body_frame_action_from_pose(current, poses[past_idx]).astype(float)
        waypoints.append([clean_float(relative[0]), clean_float(relative[1]), clean_float(relative[3])])
    return waypoints


def frame_instruction_text(
    *,
    scene_description: str,
    command_text: str,
    velocity: float,
    acceleration: float,
    yaw_rate: float,
    past_waypoints: Sequence[Sequence[float]],
) -> str:
    waypoints_text = format_waypoints(past_waypoints)
    return (
        f"{scene_description}. The current high-level command is '{command_text}'. "
        f"Current forward velocity is {float(velocity):.3f} m/s, acceleration is {float(acceleration):.3f} m/s^2, "
        f"and yaw rate is {float(yaw_rate):.3f} rad/s. "
        "The vehicle's recent past trajectory relative to its current position, represented by "
        f"{len(past_waypoints)} waypoint(s) (x, y, yaw), is: {waypoints_text}."
    )


def format_waypoints(past_waypoints: Sequence[Sequence[float]]) -> str:
    if not past_waypoints:
        return "[]"
    return "[" + ", ".join(f"({float(x):.3f}, {float(y):.3f}, {float(yaw):.3f})" for x, y, yaw in past_waypoints) + "]"


def dynamics_for_scene(
    *,
    source_root: Path,
    scene_name: str,
    samples: Sequence[dict[str, Any]],
    poses: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    messages = load_can_bus_pose_messages(source_root, scene_name)
    if messages:
        message_times = [int(row["utime"]) for row in messages]
        rows: list[dict[str, Any]] = []
        for sample in samples:
            message = messages[locate_nearest_timestamp(message_times, int(sample.get("timestamp") or 0))]
            rows.append(
                {
                    "velocity": clean_float((message.get("vel") or [0.0])[0]),
                    "acceleration": clean_float((message.get("accel") or [0.0])[0]),
                    "yaw_rate": clean_float((message.get("rotation_rate") or [0.0, 0.0, 0.0])[2]),
                    "dynamics_source": "can_bus_pose",
                }
            )
        return rows
    return state_fallback_dynamics(samples=samples, poses=poses)


def load_can_bus_pose_messages(source_root: Path, scene_name: str) -> list[dict[str, Any]]:
    path = Path(source_root) / "can_bus" / f"{scene_name}_pose.json"
    if not path.is_file():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return []
    return sorted((row for row in rows if isinstance(row, dict) and "utime" in row), key=lambda row: int(row["utime"]))


def locate_nearest_timestamp(sorted_timestamps: Sequence[int], timestamp: int) -> int:
    if not sorted_timestamps:
        raise ValueError("sorted_timestamps must be non-empty")
    pos = bisect_left(sorted_timestamps, int(timestamp))
    if pos <= 0:
        return 0
    if pos >= len(sorted_timestamps):
        return len(sorted_timestamps) - 1
    before = int(sorted_timestamps[pos - 1])
    after = int(sorted_timestamps[pos])
    return pos - 1 if abs(int(timestamp) - before) <= abs(after - int(timestamp)) else pos


def state_fallback_dynamics(*, samples: Sequence[dict[str, Any]], poses: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
    if len(samples) != len(poses):
        raise ValueError(f"samples and poses length mismatch: {len(samples)} != {len(poses)}")
    if not poses:
        return []
    timestamps_s = [float(int(sample.get("timestamp") or 0)) / 1.0e6 for sample in samples]
    velocities = [0.0] * len(poses)
    yaw_rates = [0.0] * len(poses)
    for idx in range(len(poses)):
        if idx > 0:
            prev_idx, next_idx = idx - 1, idx
        elif len(poses) > 1:
            prev_idx, next_idx = idx, idx + 1
        else:
            prev_idx = next_idx = idx
        dt = max(timestamps_s[next_idx] - timestamps_s[prev_idx], 1.0e-6)
        relative = body_frame_action_from_pose(poses[prev_idx], poses[next_idx]).astype(float)
        velocities[idx] = clean_float(float(relative[0]) / dt)
        yaw_rates[idx] = clean_float(float(relative[3]) / dt)
    accelerations = [0.0] * len(poses)
    for idx in range(len(poses)):
        if idx > 0:
            prev_idx, next_idx = idx - 1, idx
        elif len(poses) > 1:
            prev_idx, next_idx = idx, idx + 1
        else:
            prev_idx = next_idx = idx
        dt = max(timestamps_s[next_idx] - timestamps_s[prev_idx], 1.0e-6)
        accelerations[idx] = clean_float((velocities[next_idx] - velocities[prev_idx]) / dt)
    return [
        {
            "velocity": clean_float(velocities[idx]),
            "acceleration": clean_float(accelerations[idx]),
            "yaw_rate": clean_float(yaw_rates[idx]),
            "dynamics_source": "state_fallback",
        }
        for idx in range(len(poses))
    ]


def camera_specs_for_episode(sample: dict[str, Any], meta: dict[str, Any]) -> list[NavVLACameraSpec]:
    sd_by_channel = meta["sample_data_by_sample"].get(str(sample["token"]), {})
    specs: list[NavVLACameraSpec] = []
    for base in NUSCENES_CAMERAS:
        channel = CAMERA_NAME_TO_CHANNEL[base.name]
        sd = sd_by_channel.get(channel)
        intrinsics: list[list[float]] | None = None
        extrinsics_body: list[list[float]] | None = None
        calibration_status = "unknown"
        if sd is not None:
            calibrated = meta["calibrated_sensor_by_token"].get(str(sd.get("calibrated_sensor_token") or ""))
            if calibrated is not None:
                if calibrated.get("camera_intrinsic"):
                    intrinsics = [[clean_float(value) for value in row] for row in calibrated["camera_intrinsic"]]
                extrinsics_body = calibrated_sensor_to_body_matrix(calibrated)
                calibration_status = "nuscenes-calibrated_sensor"
        specs.append(
            NavVLACameraSpec(
                name=base.name,
                video_key=base.video_key,
                viewpoint_type=base.viewpoint_type,
                azimuth_rad=base.azimuth_rad,
                intrinsics=intrinsics,
                extrinsics_body=extrinsics_body,
                calibration_status=calibration_status,
            )
        )
    return specs


def camera_media_paths(
    sample: dict[str, Any],
    *,
    source_root: Path,
    meta: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, str], dict[str, str]]:
    sd_by_channel = meta["sample_data_by_sample"].get(str(sample["token"]), {})
    media_paths: dict[str, Path] = {}
    camera_sources: dict[str, str] = {}
    calibrated_tokens: dict[str, str] = {}
    for camera in NUSCENES_CAMERAS:
        channel = CAMERA_NAME_TO_CHANNEL[camera.name]
        sd = sd_by_channel.get(channel)
        if sd is None:
            raise KeyError(f"nuScenes sample {sample['token']} is missing keyframe camera {channel}")
        filename = str(sd.get("filename") or "")
        if not filename:
            raise ValueError(f"nuScenes sample_data {sd.get('token')} for {channel} has empty filename")
        path = Path(filename)
        resolved = path if path.is_absolute() else source_root / path
        if not resolved.is_file():
            raise FileNotFoundError(f"nuScenes camera image not found for sample {sample['token']} {channel}: {resolved}")
        media_paths[camera.video_key] = resolved
        camera_sources[camera.name] = filename
        calibrated_tokens[camera.name] = str(sd.get("calibrated_sensor_token") or "")
    return media_paths, camera_sources, calibrated_tokens


def index_keyframe_sample_data(
    sample_data: list[dict[str, Any]],
    calibrated_sensors: list[dict[str, Any]],
    sensors: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    channel_by_sensor_token = {str(sensor["token"]): str(sensor["channel"]) for sensor in sensors}
    channel_by_calibrated_token = {
        str(row["token"]): channel_by_sensor_token.get(str(row.get("sensor_token") or ""))
        for row in calibrated_sensors
    }
    keep_channels = set(CAMERA_CHANNEL_TO_NAME) | {"LIDAR_TOP"}
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for row in sample_data:
        if not bool(row.get("is_key_frame")):
            continue
        channel = channel_by_calibrated_token.get(str(row.get("calibrated_sensor_token") or "")) or channel_from_filename(
            str(row.get("filename") or "")
        )
        if channel not in keep_channels:
            continue
        index.setdefault(str(row["sample_token"]), {})[str(channel)] = row
    return index


def channel_from_filename(filename: str) -> str | None:
    parts = Path(filename).parts
    if len(parts) >= 2 and parts[0] in {"samples", "sweeps"}:
        return parts[1]
    return None


def quaternion_yaw(rotation: Iterable[float]) -> float:
    w, x, y, z = (float(value) for value in rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_to_rotation_matrix(rotation: Iterable[float]) -> list[list[float]]:
    w, x, y, z = (float(value) for value in rotation)
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]


def calibrated_sensor_to_body_matrix(calibrated: dict[str, Any]) -> list[list[float]]:
    rotation = quaternion_to_rotation_matrix(calibrated["rotation"])
    translation = [clean_float(value) for value in calibrated["translation"]]
    return [
        [clean_float(rotation[0][0]), clean_float(rotation[0][1]), clean_float(rotation[0][2]), translation[0]],
        [clean_float(rotation[1][0]), clean_float(rotation[1][1]), clean_float(rotation[1][2]), translation[1]],
        [clean_float(rotation[2][0]), clean_float(rotation[2][1]), clean_float(rotation[2][2]), translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def update_conversion_report(dataset_root_value: Any, *, adapter_summary: dict[str, Any]) -> Path | None:
    if not dataset_root_value:
        return None
    dataset_root = Path(dataset_root_value)
    report_path = dataset_root / "conversion_report.json"
    if not report_path.exists():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "instruction_source": INSTRUCTION_SOURCE,
            "task_text_policy": adapter_summary.get("task_text_policy"),
            "instruction_granularity": "frame",
            "command_source": COMMAND_SOURCE,
            "command_horizon_keyframes": COMMAND_HORIZON_KEYFRAMES,
            "command_lateral_threshold_m": COMMAND_LATERAL_THRESHOLD_M,
            "dynamics_source": DYNAMICS_SOURCE,
            "trajectory_semantics": TRAJECTORY_SEMANTICS,
            "dataset_source": "nuscenes",
            "source_split": adapter_summary.get("source_split"),
            "target_split": adapter_summary.get("target_split"),
            "camera_names": adapter_summary.get("camera_names"),
            "camera_channels": adapter_summary.get("camera_channels"),
        }
    )
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def clean_float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"nuScenes numeric value must be finite, got {value}")
    return 0.0 if abs(out) < 1e-7 else out


register_adapter(NuScenesAdapter())
