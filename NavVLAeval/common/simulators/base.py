from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

from NavVLAeval.common.types import Pose4D


class WaypointExecutionMode(str, Enum):
    TELEPORT_FINAL = "teleport_final"
    TELEPORT_EACH_WAYPOINT = "teleport_each_waypoint"
    PATH = "path"


@dataclass(frozen=True)
class WaypointExecutionConfig:
    mode: WaypointExecutionMode = WaypointExecutionMode.TELEPORT_FINAL
    execute_waypoints_per_step: int | None = None


@dataclass(frozen=True)
class WaypointExecutionPlan:
    original_waypoints: np.ndarray
    executed_waypoints: np.ndarray
    selected_waypoint_indices: list[int]
    mode: WaypointExecutionMode

    @property
    def original_waypoint_count(self) -> int:
        return int(self.original_waypoints.shape[0])

    @property
    def executed_waypoint_count(self) -> int:
        return int(self.executed_waypoints.shape[0])

    def diagnostics(self) -> dict[str, Any]:
        return {
            "action_execution_mode": self.mode.value,
            "original_waypoint_count": self.original_waypoint_count,
            "executed_waypoint_count": self.executed_waypoint_count,
            "selected_waypoint_indices": list(self.selected_waypoint_indices),
            "world_waypoints": self.original_waypoints.tolist(),
            "executed_world_waypoints": self.executed_waypoints.tolist(),
        }


@dataclass(frozen=True)
class WaypointExecutionResult:
    next_pose: Pose4D
    original_waypoint_count: int
    executed_waypoint_count: int
    selected_waypoint_indices: list[int] = field(default_factory=list)
    completed_waypoint_count: int = 0
    attempted_waypoint_count: int = 0
    collision: bool = False
    collision_reason: str | None = None
    action_observations: list[dict[str, Any]] = field(default_factory=list)
    actual_waypoint_poses: list[Pose4D] = field(default_factory=list)
    pose_mismatches: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_diagnostics(self) -> dict[str, Any]:
        diagnostics = dict(self.diagnostics)
        diagnostics.update(
            {
                "original_waypoint_count": self.original_waypoint_count,
                "executed_waypoint_count": self.executed_waypoint_count,
                "selected_waypoint_indices": list(self.selected_waypoint_indices),
                "completed_waypoint_count": self.completed_waypoint_count,
                "attempted_waypoint_count": self.attempted_waypoint_count,
                "collision": self.collision,
                "collision_reason": self.collision_reason,
                "actual_waypoint_poses": [pose.as_array().tolist() for pose in self.actual_waypoint_poses],
                "pose_mismatch_count": len(self.pose_mismatches),
                "pose_mismatches": list(self.pose_mismatches),
            }
        )
        return diagnostics


@runtime_checkable
class PoseControlBackend(Protocol):
    def reset_pose(self, pose: Pose4D) -> None:
        ...


@runtime_checkable
class ObjectPlacementBackend(Protocol):
    def set_object(self, object_info: dict[str, Any]) -> bool:
        ...


@runtime_checkable
class WaypointProjectorBackend(Protocol):
    def project_action_to_world(self, current_pose: Pose4D, raw_actions: np.ndarray) -> np.ndarray:
        ...


def parse_action_execution_mode(value: str | WaypointExecutionMode | None) -> WaypointExecutionMode:
    if value is None:
        return WaypointExecutionMode.TELEPORT_FINAL
    if isinstance(value, WaypointExecutionMode):
        return value
    try:
        return WaypointExecutionMode(str(value))
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in WaypointExecutionMode)
        raise ValueError(f"Unsupported action_execution_mode {value!r}; expected one of: {allowed}") from exc


def validate_execute_waypoints_per_step(value: int | None) -> int | None:
    if value is None:
        return None
    count = int(value)
    if count <= 0:
        raise ValueError("execute_waypoints_per_step must be a positive integer or null")
    return count


def build_waypoint_execution_config(
    *,
    mode: str | WaypointExecutionMode | None = None,
    execute_waypoints_per_step: int | None = None,
) -> WaypointExecutionConfig:
    return WaypointExecutionConfig(
        mode=parse_action_execution_mode(mode),
        execute_waypoints_per_step=validate_execute_waypoints_per_step(execute_waypoints_per_step),
    )


def build_waypoint_execution_plan(
    *,
    world_waypoints: np.ndarray,
    config: WaypointExecutionConfig,
) -> WaypointExecutionPlan:
    original_waypoints = np.asarray(world_waypoints, dtype=np.float32).reshape(-1, 4)
    executed_waypoints, selected_indices = select_waypoints_for_step(
        original_waypoints,
        execute_waypoints_per_step=config.execute_waypoints_per_step,
    )
    return WaypointExecutionPlan(
        original_waypoints=original_waypoints,
        executed_waypoints=executed_waypoints,
        selected_waypoint_indices=selected_indices,
        mode=config.mode,
    )


def select_waypoints_for_step(
    waypoints: np.ndarray,
    *,
    execute_waypoints_per_step: int | None,
) -> tuple[np.ndarray, list[int]]:
    waypoints = np.asarray(waypoints, dtype=np.float32).reshape(-1, 4)
    if waypoints.shape[0] == 0:
        raise ValueError("raw action chunk must contain at least one waypoint")
    count = validate_execute_waypoints_per_step(execute_waypoints_per_step)
    if count is None or count >= waypoints.shape[0]:
        selected = waypoints
    else:
        selected = waypoints[:count]
    return selected.astype(np.float32, copy=False), list(range(int(selected.shape[0])))


def pose_from_waypoint(waypoint: np.ndarray) -> Pose4D:
    values = np.asarray(waypoint, dtype=np.float32).reshape(4)
    return Pose4D(float(values[0]), float(values[1]), float(values[2]), float(values[3]))
