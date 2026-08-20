from __future__ import annotations

from typing import Any

import numpy as np

from NavVLAeval.common.runtime_defaults import BaseBenchmarkRuntime
from NavVLAeval.common.types import (
    EpisodeHistory,
    EvalEpisode,
    Pose4D,
    StepState,
    TerminationStatus,
)


class OpenFlyBenchmarkSpec:
    def __init__(
        self,
        *,
        success_radius: float = 2.0,
        stop_action_threshold: float = 1e-3,
        termination_mode: str = "success_or_action",
        stop_action_measure: str = "chunk_max_abs",
        stop_action_confirmations: int = 1,
    ):
        self.success_radius = float(success_radius)
        self.stop_action_threshold = float(stop_action_threshold)
        self.termination_mode = _validate_termination_mode(termination_mode)
        self.stop_action_measure = _validate_stop_action_measure(stop_action_measure)
        self.stop_action_confirmations = _validate_stop_action_confirmations(stop_action_confirmations)

    def validate_episode(self, episode: EvalEpisode, *, env: Any, dataset: Any) -> None:
        del env, dataset
        payload = episode.payload
        if not str(payload.get("env_name") or "").strip():
            raise ValueError(f"OpenFly episode {episode.episode_uid} is missing payload['env_name']")
        _positions(payload, episode)
        _yaw(payload, episode)

    def create_runtime(self, cfg: Any) -> "OpenFlyBenchmarkRuntime":
        del cfg
        return OpenFlyBenchmarkRuntime(
            success_radius=self.success_radius,
            stop_action_threshold=self.stop_action_threshold,
            termination_mode=self.termination_mode,
            stop_action_measure=self.stop_action_measure,
            stop_action_confirmations=self.stop_action_confirmations,
        )


class OpenFlyBenchmarkRuntime(BaseBenchmarkRuntime):
    def __init__(
        self,
        *,
        success_radius: float,
        stop_action_threshold: float,
        termination_mode: str,
        stop_action_measure: str,
        stop_action_confirmations: int,
    ):
        self.success_radius = float(success_radius)
        self.stop_action_threshold = float(stop_action_threshold)
        self.termination_mode = _validate_termination_mode(termination_mode)
        self.stop_action_measure = _validate_stop_action_measure(stop_action_measure)
        self.stop_action_confirmations = _validate_stop_action_confirmations(stop_action_confirmations)
        self._low_action_streak = 0

    def stop_at_first_success_waypoint(self) -> bool:
        return self.termination_mode == "success_or_action"

    def log_step_artifacts(self, state: StepState, artifacts: Any) -> dict[str, Any]:
        del artifacts
        return {
            "stop_action_values": {
                measure: _stop_action_value(state.raw_action_chunk, measure)
                for measure in sorted(_STOP_ACTION_MEASURES)
            }
        }

    def initial_pose(self, episode: EvalEpisode) -> Pose4D:
        positions = _positions(episode.payload, episode)
        yaw = _yaw(episode.payload, episode)
        return Pose4D(float(positions[0][0]), float(positions[0][1]), float(positions[0][2]), float(yaw[0]))

    def prepare_environment(self, episode: EvalEpisode, env, initial_pose: Pose4D) -> None:
        del episode, env, initial_pose
        self._low_action_streak = 0
        return None

    def instruction_for_step(self, episode: EvalEpisode, history: EpisodeHistory | None, step: int) -> str:
        del history, step
        return episode.instruction

    def goal_position(self, episode: EvalEpisode) -> np.ndarray:
        positions = _positions(episode.payload, episode)
        return np.asarray(positions[-1][:3], dtype=np.float32)

    def distance_to_goal(self, pose: Pose4D, episode: EvalEpisode) -> float:
        return float(np.linalg.norm(pose.as_array()[:3] - self.goal_position(episode)))

    def gt_path_length(self, episode: EvalEpisode) -> float:
        positions = np.asarray(_positions(episode.payload, episode), dtype=np.float32)
        if len(positions) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(positions[:, :3], axis=0), axis=1).sum())

    def is_success(self, pose: Pose4D, episode: EvalEpisode) -> bool:
        return self.distance_to_goal(pose, episode) < self.success_radius

    def update_termination(self, state: StepState) -> TerminationStatus:
        stop_action_value = _stop_action_value(state.raw_action_chunk, self.stop_action_measure)
        if stop_action_value < self.stop_action_threshold:
            self._low_action_streak += 1
        else:
            self._low_action_streak = 0
        action_stop = self._low_action_streak >= self.stop_action_confirmations
        success = int(self.is_success(state.pose_after, state.episode))
        done = bool(action_stop or (success and self.termination_mode == "success_or_action"))
        return TerminationStatus(
            done=done,
            success=success,
            oracle_success=success,
            reason="success" if done and success else ("stop" if done else "running"),
            failure=None,
            failure_type=None,
            diagnostics={
                "action_stop": action_stop,
                "stop_action_measure": self.stop_action_measure,
                "stop_action_value": stop_action_value,
                "stop_action_streak": self._low_action_streak,
                "stop_action_confirmations": self.stop_action_confirmations,
            },
        )


_TERMINATION_MODES = {"success_or_action", "action_or_max_steps"}
_STOP_ACTION_MEASURES = {
    "chunk_max_abs",
    "chunk_max_xyz_norm",
    "final_xyz_norm",
    "tail4_max_segment_xyz_norm",
}


def _validate_termination_mode(value: str) -> str:
    mode = str(value).strip()
    if mode not in _TERMINATION_MODES:
        raise ValueError(f"termination_mode must be one of {sorted(_TERMINATION_MODES)}, got {value!r}")
    return mode


def _validate_stop_action_measure(value: str) -> str:
    measure = str(value).strip()
    if measure not in _STOP_ACTION_MEASURES:
        raise ValueError(f"stop_action_measure must be one of {sorted(_STOP_ACTION_MEASURES)}, got {value!r}")
    return measure


def _validate_stop_action_confirmations(value: int) -> int:
    confirmations = int(value)
    if confirmations <= 0:
        raise ValueError(f"stop_action_confirmations must be positive, got {value!r}")
    return confirmations


def _stop_action_value(raw_action_chunk: np.ndarray, measure: str) -> float:
    action = np.asarray(raw_action_chunk, dtype=np.float32)
    if action.ndim != 2 or action.shape[0] == 0 or action.shape[1] < 3:
        raise ValueError(f"raw_action_chunk must have non-empty [horizon, dim>=3] shape, got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("raw_action_chunk contains non-finite values")
    if measure == "chunk_max_abs":
        return float(np.max(np.abs(action)))
    xyz_norm = np.linalg.norm(action[:, :3], axis=1)
    if measure == "chunk_max_xyz_norm":
        return float(np.max(xyz_norm))
    if measure == "final_xyz_norm":
        return float(xyz_norm[-1])
    if measure == "tail4_max_segment_xyz_norm":
        if action.shape[0] < 5:
            raise ValueError(
                "tail4_max_segment_xyz_norm requires at least 5 action waypoints, "
                f"got horizon {action.shape[0]}"
            )
        tail_segment_norm = np.linalg.norm(np.diff(action[-5:, :3], axis=0), axis=1)
        return float(np.max(tail_segment_norm))
    raise AssertionError(f"unhandled stop_action_measure: {measure}")


def _positions(payload: dict[str, Any], episode: EvalEpisode) -> list[Any]:
    positions = payload.get("pos") or payload.get("positions")
    if not positions:
        raise ValueError(f"OpenFly episode {episode.episode_uid} is missing positions")
    return positions


def _yaw(payload: dict[str, Any], episode: EvalEpisode) -> list[Any]:
    yaw = payload.get("yaw")
    if not yaw:
        raise ValueError(f"OpenFly episode {episode.episode_uid} is missing yaw")
    return yaw
