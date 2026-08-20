from __future__ import annotations

from math import cos, degrees, radians, sin
from typing import Sequence

import numpy as np

from NavVLAeval.common.types import Pose4D

CM_PER_M = 100.0


def nav_pose_from_unreal_cm(pose_cm: Sequence[float]) -> Pose4D:
    if len(pose_cm) < 5:
        raise ValueError("Unreal pose must contain at least [x, y, z, roll, yaw]")
    return Pose4D(
        float(pose_cm[0]) / CM_PER_M,
        float(pose_cm[1]) / CM_PER_M,
        -float(pose_cm[2]) / CM_PER_M,
        float(pose_cm[4]),
    )


def unreal_pose_from_nav(pose: Pose4D) -> tuple[float, float, float, float, float, float]:
    return (
        float(pose.x) * CM_PER_M,
        float(pose.y) * CM_PER_M,
        -float(pose.z) * CM_PER_M,
        0.0,
        _normalize_degrees(float(pose.yaw)),
        0.0,
    )


def starvla_waypoints_to_nav(current_pose: Pose4D, raw_actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(raw_actions, dtype=np.float32).reshape(-1, raw_actions.shape[-1])
    if actions.shape[1] < 4:
        raise ValueError(f"StarVLA action must have at least 4 columns, got {actions.shape}")
    theta = radians(float(current_pose.yaw))
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    waypoints = []
    for action in actions:
        dx_body, dy_body, dz_down, dyaw_rad = (float(action[0]), float(action[1]), float(action[2]), float(action[3]))
        dx_world = dx_body * cos_theta - dy_body * sin_theta
        dy_world = dx_body * sin_theta + dy_body * cos_theta
        dyaw_deg = degrees(dyaw_rad)
        waypoints.append(
            [
                float(current_pose.x) + dx_world,
                float(current_pose.y) + dy_world,
                float(current_pose.z) + dz_down,
                _normalize_degrees(float(current_pose.yaw) + dyaw_deg),
            ]
        )
    return np.asarray(waypoints, dtype=np.float32)


def starvla_waypoints_to_unreal_cm(current_pose: Pose4D, raw_actions: np.ndarray) -> np.ndarray:
    nav_waypoints = starvla_waypoints_to_nav(current_pose, raw_actions)
    return nav_waypoints_to_unreal_cm(nav_waypoints)


def nav_waypoints_to_unreal_cm(nav_waypoints: np.ndarray) -> np.ndarray:
    unreal_waypoints = []
    for x_m, y_m, z_down_m, yaw_deg in np.asarray(nav_waypoints, dtype=np.float32).reshape(-1, 4):
        unreal_waypoints.append([x_m * CM_PER_M, y_m * CM_PER_M, -z_down_m * CM_PER_M, yaw_deg])
    return np.asarray(unreal_waypoints, dtype=np.float32)


def preprocessed_uavflow_pose_cm_to_nav_m(pose: Sequence[float]) -> list[float]:
    if len(pose) < 6:
        raise ValueError("UAV-Flow preprocessed pose must contain [x, y, z, roll, yaw, pitch]")
    return [
        float(pose[0]) / CM_PER_M,
        float(pose[1]) / CM_PER_M,
        -float(pose[2]) / CM_PER_M,
        float(pose[3]),
        float(pose[4]),
        float(pose[5]),
    ]


def _normalize_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0
