from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from NavVLAeval.common.runtime_defaults import BaseBenchmarkRuntime
from NavVLAeval.common.simulators.base import ObjectPlacementBackend, PoseControlBackend
from NavVLAeval.common.simulators.unrealzoo.coordinates import nav_pose_from_unreal_cm
from NavVLAeval.common.types import EpisodeHistory, EvalEpisode, Pose4D, StepState, TerminationStatus


class UAVFlowBenchmarkSpec:
    def __init__(
        self,
        *,
        stop_policy: str = "small_delta",
        small_delta_pos: float = 0.03,
        small_delta_yaw: float = 1.0,
        small_delta_steps: int = 10,
        **runtime_kwargs: Any,
    ) -> None:
        if runtime_kwargs:
            unknown = ", ".join(sorted(str(key) for key in runtime_kwargs))
            raise ValueError(f"Unsupported UAV-Flow benchmark kwargs: {unknown}")
        self.stop_policy = str(stop_policy)
        if self.stop_policy not in {"small_delta"}:
            raise ValueError(f"Unsupported UAV-Flow stop_policy: {self.stop_policy}")
        self.small_delta_pos = float(small_delta_pos)
        self.small_delta_yaw = float(small_delta_yaw)
        self.small_delta_steps = int(small_delta_steps)

    def validate_episode(self, episode: EvalEpisode, *, env: Any, dataset: Any) -> None:
        del env, dataset
        payload = episode.payload
        for key in ("initial_pos_cm", "end_pos_cm", "reference_path_preprocessed_m"):
            if key not in payload:
                raise ValueError(f"UAV-Flow episode {episode.episode_uid} is missing payload[{key!r}]")
        self.create_runtime(None).initial_pose(episode)

    def create_runtime(self, cfg: Any) -> "UAVFlowBenchmarkRuntime":
        del cfg
        return UAVFlowBenchmarkRuntime(
            stop_policy=self.stop_policy,
            small_delta_pos=self.small_delta_pos,
            small_delta_yaw=self.small_delta_yaw,
            small_delta_steps=self.small_delta_steps,
        )


class UAVFlowBenchmarkRuntime(BaseBenchmarkRuntime):
    def __init__(
        self,
        *,
        stop_policy: str,
        small_delta_pos: float,
        small_delta_yaw: float,
        small_delta_steps: int,
    ) -> None:
        self.stop_policy = stop_policy
        if self.stop_policy not in {"small_delta"}:
            raise ValueError(f"Unsupported UAV-Flow stop_policy: {self.stop_policy}")
        self.small_delta_pos = float(small_delta_pos)
        self.small_delta_yaw = float(small_delta_yaw)
        self.small_delta_steps = int(small_delta_steps)
        self._recent_deltas: dict[str, deque[tuple[float, float]]] = {}

    def initial_pose(self, episode: EvalEpisode) -> Pose4D:
        return nav_pose_from_unreal_cm(episode.payload["initial_pos_cm"])

    def goal_position(self, episode: EvalEpisode) -> np.ndarray:
        return nav_pose_from_unreal_cm(episode.payload["end_pos_cm"]).as_array()[:3]

    def prepare_environment(self, episode: EvalEpisode, env, initial_pose: Pose4D) -> None:
        self._recent_deltas[episode.episode_uid] = deque(maxlen=self.small_delta_steps)
        object_info = episode.payload.get("object")
        if object_info:
            if not isinstance(env, ObjectPlacementBackend):
                raise TypeError("UAV-Flow environment backend must implement set_object(object_info)")
            if not env.set_object(object_info):
                raise RuntimeError(f"UAV-Flow failed to place object for {episode.episode_uid}")
        if not isinstance(env, PoseControlBackend):
            raise TypeError("UAV-Flow environment backend must implement reset_pose(pose)")
        env.reset_pose(initial_pose)

    def instruction_for_step(self, episode: EvalEpisode, history: EpisodeHistory | None, step: int) -> str:
        del history, step
        return episode.instruction

    def prepare_observation_for_model(
        self,
        *,
        episode: EvalEpisode,
        history: EpisodeHistory,
        step: int,
        observation: dict[str, Any],
        instruction: str,
    ) -> dict[str, Any]:
        del history, step
        prepared = dict(observation)
        uavflow_pose = dict(prepared.get("uavflow_pose") or {})
        uavflow_pose.setdefault("instruction", instruction)
        uavflow_pose.setdefault("navvla_eval", {})
        uavflow_pose["navvla_eval"].update({"episode_id": episode.source_episode_id, "episode_uid": episode.episode_uid})
        prepared["uavflow_pose"] = uavflow_pose
        prepared.setdefault("target_position", self.goal_position(episode))
        return prepared

    def distance_to_goal(self, pose: Pose4D, episode: EvalEpisode) -> float:
        return float(np.linalg.norm(pose.as_array()[:3] - self.goal_position(episode)))

    def gt_path_length(self, episode: EvalEpisode) -> float:
        path = np.asarray([item[:3] for item in episode.payload["reference_path_preprocessed_m"]], dtype=np.float32)
        if len(path) < 2:
            return 0.0
        return float(sum(np.linalg.norm(curr - prev) for prev, curr in zip(path[:-1], path[1:])))

    def is_success(self, pose: Pose4D, episode: EvalEpisode) -> bool:
        del pose, episode
        return False

    def update_termination(self, state: StepState) -> TerminationStatus:
        recent = self._recent_deltas.setdefault(state.episode.episode_uid, deque(maxlen=self.small_delta_steps))
        for pose_delta, yaw_delta in _waypoint_deltas(state):
            recent.append((pose_delta, yaw_delta))
        small_delta_done = len(recent) >= self.small_delta_steps and all(
            pos < self.small_delta_pos and yaw < self.small_delta_yaw for pos, yaw in recent
        )
        done = bool(small_delta_done)
        reason = "running"
        if small_delta_done:
            reason = "small_delta_stop"
        return TerminationStatus(
            done=done,
            success=0,
            oracle_success=0,
            reason=reason,
            failure=None,
            failure_type=None,
            diagnostics={"small_delta_stop": small_delta_done},
        )

    def log_step_artifacts(self, state: StepState, artifacts: Any) -> dict[str, Any]:
        del artifacts
        payload = state.episode.payload
        task_subtype = payload.get("task_subtype") or payload.get("subtype") or payload.get("label")
        return {
            "small_delta_stop": bool(state.diagnostics.get("small_delta_stop", False)),
            "task_subtype": str(task_subtype) if task_subtype is not None else None,
        }

def _waypoint_deltas(state: StepState) -> list[tuple[float, float]]:
    waypoints = np.asarray(state.world_waypoints, dtype=np.float32).reshape(-1, state.world_waypoints.shape[-1])
    if waypoints.size == 0:
        return []
    previous = np.asarray(state.pose_before.as_array(), dtype=np.float32)
    deltas: list[tuple[float, float]] = []
    for waypoint in waypoints:
        current = np.asarray(waypoint[:4], dtype=np.float32)
        pos_delta = float(np.linalg.norm(current[:3] - previous[:3]))
        yaw_delta = abs(_normalize_degrees_delta(float(current[3]) - float(previous[3])))
        deltas.append((pos_delta, yaw_delta))
        previous = current
    return deltas


def _normalize_degrees_delta(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0
