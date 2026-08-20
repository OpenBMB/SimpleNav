import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vln_aug.actions import build_observation_actions, observation_indices


@dataclass(frozen=True)
class CameraSpec:
    key: str
    height: int
    width: int
    metadata: dict | None = None


@dataclass(frozen=True)
class IntermediatePaths:
    control_path: Path
    observation_path: Path
    render_request_path: Path


def write_intermediate_episode(
    output_dir: Path,
    dataset_key: str,
    source_episode_index: int,
    source_episode_id: str,
    scene_id: str,
    control_poses: np.ndarray,
    cameras: list[CameraSpec],
    horizon: int = 8,
    terminal_action_available: bool = True,
    coordinate_metadata: dict | None = None,
) -> IntermediatePaths:
    controls = np.asarray(control_poses, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != 4:
        raise ValueError("control_poses must have shape [N, 4]")
    if len(controls) < 1:
        raise ValueError("control trajectory must contain at least one pose")
    if not cameras:
        raise ValueError("at least one camera is required")
    if not scene_id:
        raise ValueError("scene_id is required for later rendering")
    if not coordinate_metadata:
        raise ValueError("coordinate metadata is required for later rendering")
    if any(not camera.key or camera.height <= 0 or camera.width <= 0 for camera in cameras):
        raise ValueError("camera key and dimensions are required")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    control_path = root / "control_trajectory_1hz.parquet"
    observation_path = root / "observation_plan_every_5_waypoints.parquet"
    render_request_path = root / "render_requests_every_5_waypoints.jsonl"

    stored_controls = controls.astype(np.float32)
    render_indices = set(observation_indices(len(controls), 5).astype(int).tolist())
    control_table = pa.table(
        {
            "control_index": pa.array(np.arange(len(controls), dtype=np.int64)),
            "timestamp": pa.array(np.arange(len(controls), dtype=np.float64)),
            "pose": pa.array(stored_controls.tolist(), type=pa.list_(pa.float32(), 4)),
            "is_render_time": pa.array(
                [index in render_indices for index in range(len(controls))]
            ),
            "is_terminal": pa.array((np.arange(len(controls)) == len(controls) - 1).tolist()),
        }
    )
    pq.write_table(control_table, control_path)

    control_indices, actions = build_observation_actions(
        stored_controls, render_stride=5, horizon=horizon
    )
    row_count = len(control_indices)
    observation_table = pa.table(
        {
            "episode_index": pa.array(np.zeros(row_count, dtype=np.int64)),
            "frame_index": pa.array(np.arange(row_count, dtype=np.int64)),
            "timestamp": pa.array(control_indices.astype(np.float64)),
            "observation.state": pa.array(
                stored_controls[control_indices].tolist(), type=pa.list_(pa.float32(), 4)
            ),
            "action": pa.array(
                actions.tolist(), type=pa.list_(pa.list_(pa.float32(), 4), horizon)
            ),
            "action.padding_mask": pa.array(
                np.zeros((row_count, horizon), dtype=bool).tolist(), type=pa.list_(pa.bool_(), horizon)
            ),
            "next.done": pa.array((np.arange(row_count) == row_count - 1).tolist()),
            "sample.action_available": pa.array(np.ones(row_count, dtype=bool).tolist()),
            "source_episode_index": pa.array(
                np.repeat(source_episode_index, row_count).astype(np.int64)
            ),
            "control_index": pa.array(control_indices),
        }
    )
    pq.write_table(observation_table, observation_path)

    with render_request_path.open("w", encoding="utf-8") as stream:
        for frame_index, control_index in enumerate(control_indices):
            pose = stored_controls[control_index].astype(float).tolist()
            for camera in cameras:
                request_id = (
                    f"{dataset_key}/source_ep_{source_episode_index:06d}/"
                    f"frame_{frame_index:06d}/{camera.key}"
                )
                payload = {
                    "request_id": request_id,
                    "dataset_key": dataset_key,
                    "source_episode_index": source_episode_index,
                    "source_episode_id": source_episode_id,
                    "scene_id": scene_id,
                    "frame_index": frame_index,
                    "control_index": int(control_index),
                    "timestamp": float(control_index),
                    "body_pose_xyz_yaw": pose,
                    "camera_key": camera.key,
                    "expected_height": camera.height,
                    "expected_width": camera.width,
                    "expected_channels": 3,
                    "expected_image_relpath": (
                        f"rendered_images/frame_{frame_index:06d}/{camera.key}.png"
                    ),
                    "coordinate_metadata": coordinate_metadata or {},
                    "camera_metadata": camera.metadata or {},
                }
                payload.update(camera.metadata or {})
                payload.update(coordinate_metadata or {})
                stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    return IntermediatePaths(control_path, observation_path, render_request_path)
