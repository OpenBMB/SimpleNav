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


class AerialVLNBenchmarkSpec:
    def __init__(
        self,
        *,
        success_radius: float = 20.0,
        termination_mode: str = "success_or_max_steps",
        stop_action_threshold: float = -1.0,
        stop_action_measure: str = "final_segment_xyz_norm",
        stop_action_confirmations: int = 1,
        ndtw_success_distance: float = 1.0,
    ):
        self.success_radius = float(success_radius)
        self.termination_mode = _validate_termination_mode(termination_mode)
        self.stop_action_threshold = float(stop_action_threshold)
        self.stop_action_measure = _validate_stop_action_measure(stop_action_measure)
        self.stop_action_confirmations = _validate_stop_action_confirmations(stop_action_confirmations)
        self.ndtw_success_distance = float(ndtw_success_distance)
        if self.ndtw_success_distance <= 0:
            raise ValueError(f"ndtw_success_distance must be positive, got {ndtw_success_distance!r}")

    def validate_episode(self, episode: EvalEpisode, *, env: Any, dataset: Any) -> None:
        del env, dataset
        payload = episode.payload
        for key in ("env_name", "start_pose", "goal_position", "reference_path_m"):
            if key not in payload or payload[key] in (None, ""):
                raise ValueError(f"AerialVLN episode {episode.episode_uid} is missing payload[{key!r}]")
        runtime = self.create_runtime(None)
        runtime.initial_pose(episode)
        runtime.goal_position(episode)
        runtime.gt_path_length(episode)

    def create_runtime(self, cfg: Any) -> "AerialVLNBenchmarkRuntime":
        del cfg
        return AerialVLNBenchmarkRuntime(
            success_radius=self.success_radius,
            termination_mode=self.termination_mode,
            stop_action_threshold=self.stop_action_threshold,
            stop_action_measure=self.stop_action_measure,
            stop_action_confirmations=self.stop_action_confirmations,
        )


class AerialVLNBenchmarkRuntime(BaseBenchmarkRuntime):
    def __init__(
        self,
        *,
        success_radius: float,
        termination_mode: str,
        stop_action_threshold: float,
        stop_action_measure: str,
        stop_action_confirmations: int,
    ):
        self.success_radius = float(success_radius)
        self.termination_mode = _validate_termination_mode(termination_mode)
        self.stop_action_threshold = float(stop_action_threshold)
        self.stop_action_measure = _validate_stop_action_measure(stop_action_measure)
        self.stop_action_confirmations = _validate_stop_action_confirmations(stop_action_confirmations)
        self._low_action_streak = 0

    def stop_at_first_success_waypoint(self) -> bool:
        return self.termination_mode == "success_or_max_steps"

    def initial_pose(self, episode: EvalEpisode) -> Pose4D:
        pose = _pose_values(episode.payload.get("start_pose"), episode=episode, label="start_pose")
        return Pose4D(float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3]))

    def prepare_environment(self, episode: EvalEpisode, env, initial_pose: Pose4D) -> None:
        del episode
        self._low_action_streak = 0
        env.reset_pose(initial_pose)

    def log_step_artifacts(self, state: StepState, artifacts: Any) -> dict[str, Any]:
        del artifacts
        return {
            "stop_action_values": {
                self.stop_action_measure: _stop_action_value(state.raw_action_chunk, self.stop_action_measure)
            }
        }

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
        del history
        prepared = dict(observation)
        metadata = dict(prepared.get("navvla_eval") or {})
        # The runtime dataset assigns frame_index in control-tick units.  Keep
        # that dense index so the online history can include every waypoint
        # observation; ``step`` is only the coarser model-inference index.
        frame_index = metadata.get("frame_index", step)
        metadata.update(
            {
                "episode_id": episode.source_episode_id,
                "episode_uid": episode.episode_uid,
                "scene_id": episode.scene_id,
                "frame_index": int(frame_index),
                "model_step": int(step),
            }
        )
        prepared["navvla_eval"] = metadata
        prepared["instruction"] = instruction
        prepared["goal_position"] = self.goal_position(episode)
        return prepared

    def goal_position(self, episode: EvalEpisode) -> np.ndarray:
        goal = episode.payload.get("goal_position")
        if goal is None or len(goal) < 3:
            raise ValueError(f"AerialVLN episode {episode.episode_uid} is missing goal_position")
        return np.asarray(goal[:3], dtype=np.float32)

    def distance_to_goal(self, pose: Pose4D, episode: EvalEpisode) -> float:
        return float(np.linalg.norm(pose.as_array()[:3] - self.goal_position(episode)))

    def gt_path_length(self, episode: EvalEpisode) -> float:
        path = np.asarray(_reference_path(episode), dtype=np.float32)
        if path.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1).sum())

    def is_success(self, pose: Pose4D, episode: EvalEpisode) -> bool:
        return self.distance_to_goal(pose, episode) <= self.success_radius

    def update_termination(self, state: StepState) -> TerminationStatus:
        success = int(self.is_success(state.pose_after, state.episode))
        action_stop = False
        stop_action_value = None
        if self.termination_mode == "action_or_max_steps":
            stop_action_value = _stop_action_value(state.raw_action_chunk, self.stop_action_measure)
            if stop_action_value < self.stop_action_threshold:
                self._low_action_streak += 1
            else:
                self._low_action_streak = 0
            action_stop = self._low_action_streak >= self.stop_action_confirmations
        done = bool(action_stop or (success and self.termination_mode == "success_or_max_steps"))
        diagnostics = {"distance": state.distance_after}
        if stop_action_value is not None:
            diagnostics.update(
                {
                    "action_stop": action_stop,
                    "stop_action_measure": self.stop_action_measure,
                    "stop_action_value": stop_action_value,
                    "stop_action_threshold": self.stop_action_threshold,
                    "stop_action_streak": self._low_action_streak,
                    "stop_action_confirmations": self.stop_action_confirmations,
                }
            )
        return TerminationStatus(
            done=done,
            success=success,
            oracle_success=success,
            reason="success" if done and success else ("stop" if done else "running"),
            failure=None,
            failure_type=None,
            diagnostics=diagnostics,
        )


_TERMINATION_MODES = {"success_or_max_steps", "action_or_max_steps", "max_steps"}
_STOP_ACTION_MEASURES = {"final_segment_xyz_norm"}


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
    if action.ndim != 2 or action.shape[0] < 2 or action.shape[1] < 3:
        raise ValueError(
            "final_segment_xyz_norm requires raw_action_chunk shape [horizon>=2, dim>=3], "
            f"got {action.shape}"
        )
    if not np.isfinite(action).all():
        raise ValueError("raw_action_chunk contains non-finite values")
    if measure == "final_segment_xyz_norm":
        return float(np.linalg.norm(action[-1, :3] - action[-2, :3]))
    raise AssertionError(f"unhandled stop_action_measure: {measure}")


def _pose_values(value: Any, *, episode: EvalEpisode, label: str) -> list[float]:
    if value is None or len(value) < 4:
        raise ValueError(f"AerialVLN episode {episode.episode_uid} is missing {label}")
    return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]


def _reference_path(episode: EvalEpisode) -> list[Any]:
    path = episode.payload.get("reference_path_m") or episode.payload.get("reference_path")
    if not isinstance(path, list) or not path:
        raise ValueError(f"AerialVLN episode {episode.episode_uid} is missing reference_path_m")
    return path
