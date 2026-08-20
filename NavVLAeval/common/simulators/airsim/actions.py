from __future__ import annotations

import math
from typing import Any

import numpy as np

from NavVLAeval.common.types import Pose4D
from starVLA.dataloader.airsim_utils import wrap_to_pi_array


AIRSIM_ACTION_SEMANTICS = {
    "anchor_relative_frd_xyz_yaw",
    "anchor_relative_body_frame_xyz_yaw",
    "body_frame_xyz_yaw",
}


def rotation_matrix_from_yaw(yaw: float) -> np.ndarray:
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    return np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def wrap_to_pi(value: Any) -> np.ndarray:
    return (np.asarray(value, dtype=np.float32) + math.pi) % (2 * math.pi) - math.pi


def rollout_body_frame_action(pose: Pose4D, raw_action: np.ndarray) -> Pose4D:
    action = np.asarray(raw_action, dtype=np.float32)
    if action.shape[-1] != 4:
        raise ValueError(f"raw_action must have last dim 4, got {action.shape}")
    dx_body, dy_body, dz, dyaw = [float(value) for value in action.reshape(-1, 4)[-1]]
    cos_yaw = math.cos(float(pose.yaw))
    sin_yaw = math.sin(float(pose.yaw))
    dx_world = cos_yaw * dx_body - sin_yaw * dy_body
    dy_world = sin_yaw * dx_body + cos_yaw * dy_body
    next_yaw = float(wrap_to_pi_array(np.asarray(float(pose.yaw) + dyaw, dtype=np.float32)))
    return Pose4D(x=pose.x + dx_world, y=pose.y + dy_world, z=pose.z + dz, yaw=next_yaw)


def airsim_actions_to_world_waypoints(
    *,
    current_pose: Pose4D | np.ndarray,
    raw_actions: np.ndarray,
    action_semantics: str | None,
) -> np.ndarray:
    semantics = str(action_semantics or "anchor_relative_frd_xyz_yaw")
    if semantics in {"anchor_relative_frd_xyz_yaw", "anchor_relative_body_frame_xyz_yaw"}:
        return anchor_relative_body_frame_actions_to_world_waypoints(
            current_pose=current_pose,
            raw_actions=raw_actions,
        )
    if semantics == "body_frame_xyz_yaw":
        return body_frame_actions_to_world_waypoints(
            current_pose=current_pose,
            raw_actions=raw_actions,
        )
    allowed = ", ".join(sorted(AIRSIM_ACTION_SEMANTICS))
    raise ValueError(f"Unsupported AirSim action semantics {action_semantics!r}; expected one of: {allowed}")


def body_frame_actions_to_world_waypoints(
    *,
    current_pose: Pose4D | np.ndarray,
    raw_actions: np.ndarray,
) -> np.ndarray:
    pose = _pose4d_from_pose_like(current_pose)
    actions = _as_action_chunk(raw_actions)
    position_world = pose.as_array()[:3].astype(np.float32)
    yaw_world = float(pose.yaw)
    waypoints = []
    for action_body_frame in actions:
        position_world = position_world + rotation_matrix_from_yaw(yaw_world) @ action_body_frame[:3]
        yaw_world = float(wrap_to_pi(yaw_world + float(action_body_frame[3])))
        waypoints.append(np.concatenate([position_world.astype(np.float32), np.asarray([yaw_world], dtype=np.float32)]))
    return np.asarray(waypoints, dtype=np.float32).reshape(-1, 4)


def anchor_relative_body_frame_actions_to_world_waypoints(
    *,
    current_pose: Pose4D | np.ndarray,
    raw_actions: np.ndarray,
) -> np.ndarray:
    pose = _pose4d_from_pose_like(current_pose)
    actions = _as_action_chunk(raw_actions)
    anchor_position_world = pose.as_array()[:3].astype(np.float32)
    anchor_yaw_world = float(pose.yaw)
    anchor_rotation_world = rotation_matrix_from_yaw(anchor_yaw_world)
    waypoints = []
    for action_body_frame in actions:
        position_world = anchor_position_world + anchor_rotation_world @ action_body_frame[:3]
        yaw_world = float(wrap_to_pi(anchor_yaw_world + float(action_body_frame[3])))
        waypoints.append(np.concatenate([position_world.astype(np.float32), np.asarray([yaw_world], dtype=np.float32)]))
    return np.asarray(waypoints, dtype=np.float32).reshape(-1, 4)


def _pose4d_from_pose_like(current_pose: Pose4D | np.ndarray) -> Pose4D:
    if isinstance(current_pose, Pose4D):
        return current_pose
    pose_array = np.asarray(current_pose, dtype=np.float32).reshape(4)
    return Pose4D(float(pose_array[0]), float(pose_array[1]), float(pose_array[2]), float(pose_array[3]))


def _as_action_chunk(raw_actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(raw_actions, dtype=np.float32)
    if actions.size == 0:
        raise ValueError("raw action chunk must not be empty")
    if actions.shape[-1] != 4:
        raise ValueError(f"raw action chunk must have last dimension 4, got {actions.shape}")
    return actions.reshape(-1, 4)
