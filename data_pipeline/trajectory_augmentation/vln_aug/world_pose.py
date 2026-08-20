"""Resolve simulator world poses without confusing episode-local training state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from vln_aug.lerobot_io import EpisodeMetadata


DIRECT_ABSOLUTE_STATE_MODES = {
    "source_world_absolute_pose_xyz_yaw",
    "indooruav_world_pose_xy_zdown_yaw_minus_pi_over_2",
    "nuscenes_global_ego_pose_xyz_yaw",
}


def _wrap_angle(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def transform_world_poses_for_alignment(
    world_poses: np.ndarray, transform: str
) -> np.ndarray:
    poses = np.asarray(world_poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 4:
        raise ValueError(f"world poses must have shape [N, 4], got {poses.shape}")
    if transform == "identity":
        return poses.copy()
    if transform == "reflect-y-yaw":
        reflected = poses.copy()
        reflected[:, 1] *= -1.0
        reflected[:, 3] *= -1.0
        return reflected
    if transform == "reflect-y-z-yaw":
        reflected = poses.copy()
        reflected[:, 1] *= -1.0
        reflected[:, 2] *= -1.0
        reflected[:, 3] *= -1.0
        return reflected
    raise ValueError(f"unsupported world-pose alignment transform: {transform}")


def validate_episode_local_alignment(
    local_poses: np.ndarray,
    world_poses: np.ndarray,
    *,
    position_tolerance_m: float = 1e-3,
    yaw_tolerance_rad: float = 1e-3,
) -> dict[str, float]:
    """Verify that local poses are the first-world-pose body-aligned transform."""

    local = np.asarray(local_poses, dtype=float)
    world = np.asarray(world_poses, dtype=float)
    if local.ndim != 2 or local.shape[1] != 4 or local.shape != world.shape:
        raise ValueError(
            f"local/world pose arrays must share shape [N, 4], got {local.shape} and {world.shape}"
        )
    if len(local) == 0 or not np.all(np.isfinite(local)) or not np.all(np.isfinite(world)):
        raise ValueError("local/world pose arrays must be non-empty and finite")

    yaw0 = float(world[0, 3])
    cosine = float(np.cos(yaw0))
    sine = float(np.sin(yaw0))
    delta_xy = world[:, :2] - world[0, :2]
    recovered = np.empty_like(world)
    recovered[:, 0] = cosine * delta_xy[:, 0] + sine * delta_xy[:, 1]
    recovered[:, 1] = -sine * delta_xy[:, 0] + cosine * delta_xy[:, 1]
    recovered[:, 2] = world[:, 2] - world[0, 2]
    recovered[:, 3] = _wrap_angle(world[:, 3] - yaw0)

    position_errors = np.linalg.norm(recovered[:, :3] - local[:, :3], axis=1)
    yaw_errors = np.abs(_wrap_angle(recovered[:, 3] - local[:, 3]))
    max_position_error = float(np.max(position_errors))
    max_yaw_error = float(np.max(yaw_errors))
    if max_position_error > position_tolerance_m or max_yaw_error > yaw_tolerance_rad:
        raise ValueError(
            "coordinate contract mismatch between episode-local observation.state and "
            f"source world path: max_position_error_m={max_position_error:.6g}, "
            f"max_yaw_error_rad={max_yaw_error:.6g}"
        )
    return {
        "max_position_error_m": max_position_error,
        "max_yaw_error_rad": max_yaw_error,
    }


class AerialVLNOriginalPoseIndex:
    """Index canonical AerialVLN world-space reference paths by episode identity."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        episodes = payload.get("episodes") if isinstance(payload, dict) else None
        if not isinstance(episodes, list):
            raise ValueError("original AerialVLN JSON must contain an episodes list")
        self._episodes = {}
        for episode in episodes:
            episode_id = str(episode.get("episode_id", ""))
            if not episode_id:
                raise ValueError("original AerialVLN episode is missing episode_id")
            if episode_id in self._episodes:
                raise ValueError(f"duplicate original AerialVLN episode_id: {episode_id}")
            self._episodes[episode_id] = episode

    def poses_for_episode(self, metadata: EpisodeMetadata) -> np.ndarray:
        episode = self._episodes.get(str(metadata.episode_id))
        if episode is None:
            raise ValueError(
                f"source episode {metadata.episode_id} is missing from {self.path}"
            )
        source_trajectory_id = str(episode.get("trajectory_id", ""))
        if metadata.trajectory_id and source_trajectory_id != str(metadata.trajectory_id):
            raise ValueError(
                f"source trajectory mismatch for episode {metadata.episode_id}: "
                f"expected {metadata.trajectory_id}, found {source_trajectory_id}"
            )
        if str(episode.get("scene_id")) != str(metadata.scene_id):
            raise ValueError(
                f"source scene mismatch for episode {metadata.episode_id}: "
                f"expected {metadata.scene_id}, found {episode.get('scene_id')}"
            )
        path = np.asarray(episode.get("reference_path"), dtype=float)
        if path.ndim != 2 or path.shape[1] != 6:
            raise ValueError(
                f"source episode {metadata.episode_id} reference_path must have shape [N, 6]"
            )
        if len(path) != int(metadata.length):
            raise ValueError(
                f"source episode {metadata.episode_id} reference_path length {len(path)} "
                f"does not match LeRobot episode length {metadata.length}"
            )
        if not np.all(np.isfinite(path)):
            raise ValueError(
                f"source episode {metadata.episode_id} reference_path contains non-finite values"
            )
        return path[:, [0, 1, 2, 5]].copy()


class OpenFlyAnnotationPoseIndex:
    """Index raw AirSim world poses from OpenFly Annotation JSON."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, list):
            raise ValueError("OpenFly Annotation JSON must contain a list")
        self._trajectories = {}
        for trajectory in payload:
            trajectory_id = str(trajectory.get("image_path", ""))
            if not trajectory_id:
                raise ValueError("OpenFly Annotation trajectory is missing image_path")
            if trajectory_id in self._trajectories:
                raise ValueError(f"duplicate OpenFly image_path: {trajectory_id}")
            self._trajectories[trajectory_id] = trajectory

    def poses_for_episode(self, metadata: EpisodeMetadata) -> np.ndarray:
        trajectory_id = str(metadata.trajectory_id)
        trajectory = self._trajectories.get(trajectory_id)
        if trajectory is None:
            raise ValueError(
                f"source trajectory {trajectory_id} is missing from {self.path}"
            )
        source_scene = trajectory_id.split("/", 1)[0]
        if source_scene != str(metadata.scene_id):
            raise ValueError(
                f"source scene mismatch for trajectory {trajectory_id}: "
                f"expected {metadata.scene_id}, found {source_scene}"
            )
        position_rows = trajectory.get("pos")
        yaw = np.asarray(trajectory.get("yaw"), dtype=float)
        if not isinstance(position_rows, list):
            raise ValueError(
                f"OpenFly trajectory {trajectory_id} pos must be a list"
            )
        if yaw.ndim != 1 or len(yaw) != len(position_rows):
            raise ValueError(
                f"OpenFly trajectory {trajectory_id} yaw length must match pos"
            )
        positions = []
        for index, row in enumerate(position_rows):
            position = np.asarray(row, dtype=float)
            if position.shape not in {(3,), (4,)}:
                raise ValueError(
                    f"OpenFly trajectory {trajectory_id} pos[{index}] must contain x,y,z or x,y,z,yaw"
                )
            if position.shape == (4,):
                yaw_error = abs(
                    float(_wrap_angle(np.asarray([position[3] - yaw[index]]))[0])
                )
                if yaw_error > 1e-6:
                    raise ValueError(
                        f"OpenFly trajectory {trajectory_id} embedded yaw differs from yaw[{index}]"
                    )
            positions.append(position[:3])
        positions = np.asarray(positions, dtype=float)
        if len(positions) != int(metadata.length):
            raise ValueError(
                f"OpenFly trajectory {trajectory_id} length {len(positions)} "
                f"does not match LeRobot episode length {metadata.length}"
            )
        poses = np.column_stack((positions, yaw))
        if not np.all(np.isfinite(poses)):
            raise ValueError(f"OpenFly trajectory {trajectory_id} contains non-finite poses")
        return poses


class FrameMetadataWorldPoseStream:
    """Stream world poses from frame JSONL using monotonic global frame indices."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._stream = None
        self._last_requested_index = -1
        self._current_record = None

    def __enter__(self) -> "FrameMetadataWorldPoseStream":
        self._stream = self.path.open("r", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream = None

    def read_indices(self, indices: Iterable[int]) -> np.ndarray:
        if self._stream is None:
            raise RuntimeError("FrameMetadataWorldPoseStream must be used as a context manager")
        requested = [int(value) for value in indices]
        if not requested:
            return np.empty((0, 4), dtype=float)
        if any(right <= left for left, right in zip(requested, requested[1:])):
            raise ValueError("world-pose frame indices must be strictly increasing")
        if requested[0] <= self._last_requested_index:
            raise ValueError("world-pose frame indices must be strictly increasing across episodes")

        poses = []
        for target_index in requested:
            record = self._advance_to(target_index)
            source_pose = record.get("source_metadata", {}).get("source_pose")
            if source_pose is None:
                raise ValueError(
                    f"frame metadata index {target_index} has no source_metadata.source_pose"
                )
            pose = np.asarray(source_pose, dtype=float)
            if pose.shape != (6,) or not np.all(np.isfinite(pose)):
                raise ValueError(
                    f"frame metadata index {target_index} source_pose must be finite [x,y,z,roll,pitch,yaw]"
                )
            poses.append([pose[0], pose[1], pose[2], pose[5]])
            self._last_requested_index = target_index
        return np.asarray(poses, dtype=float)

    def _advance_to(self, target_index: int) -> dict:
        while self._current_record is None or int(self._current_record["index"]) < target_index:
            line = self._stream.readline()
            if not line:
                raise ValueError(
                    f"frame metadata ended before requested global index {target_index}"
                )
            record = json.loads(line)
            if "index" not in record:
                raise ValueError("frame metadata record is missing global index")
            self._current_record = record
        actual_index = int(self._current_record["index"])
        if actual_index != target_index:
            raise ValueError(
                f"frame metadata global index mismatch: requested {target_index}, found {actual_index}"
            )
        return self._current_record


def observation_state_is_explicit_world_pose(info: dict) -> bool:
    navvla = info.get("navvla", {})
    if navvla.get("state_mode") in DIRECT_ABSOLUTE_STATE_MODES:
        return True
    return (
        navvla.get("stored_observation_state") == "absolute_pose_ned_xyz_yaw"
        and int(navvla.get("state_dim", 0)) == 4
    )


def choose_world_pose_source(
    info: dict,
    *,
    requested: str,
    original_trajectory_json: Path,
    frame_metadata_path: Path,
) -> str:
    if requested not in {
        "auto",
        "original-json",
        "openfly-annotation",
        "frame-metadata",
        "observation-state",
    }:
        raise ValueError(f"unsupported world pose source: {requested}")
    state_is_world = observation_state_is_explicit_world_pose(info)
    if requested == "original-json":
        if not original_trajectory_json.is_file():
            raise ValueError(
                f"original AerialVLN trajectory JSON is missing: {original_trajectory_json}"
            )
        return "original-json"
    if requested == "openfly-annotation":
        if not original_trajectory_json.is_file():
            raise ValueError(
                f"original OpenFly Annotation JSON is missing: {original_trajectory_json}"
            )
        return "openfly-annotation"
    if requested == "observation-state":
        if not state_is_world:
            mode = info.get("navvla", {}).get("state_mode", "unknown")
            raise ValueError(
                "refusing to treat observation.state as world pose because its state_mode "
                f"is {mode}"
            )
        return "observation-state"
    if requested == "frame-metadata":
        if not frame_metadata_path.is_file():
            raise ValueError(f"frame world-pose metadata is missing: {frame_metadata_path}")
        return "frame-metadata"
    if state_is_world:
        return "observation-state"
    if original_trajectory_json.is_file():
        return "original-json"
    if frame_metadata_path.is_file():
        return "frame-metadata"
    mode = info.get("navvla", {}).get("state_mode", "unknown")
    raise ValueError(
        "no canonical world pose source is available: observation.state is "
        f"{mode}; neither {original_trajectory_json} nor {frame_metadata_path} is available"
    )
