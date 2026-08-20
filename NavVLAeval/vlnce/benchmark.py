from __future__ import annotations

from math import isfinite
from typing import Any
from NavVLAeval.common.runtime_defaults import BaseBenchmarkRuntime
from NavVLAeval.common.types import EpisodeHistory, EvalEpisode, Pose4D, StepState, TerminationStatus


class VLNCEBenchmarkSpec:
    def __init__(
        self,
        *,
        task_name: str = "r2r",
        split: str = "val_unseen",
        roles: list[str] | tuple[str, ...] = (),
        languages: list[str] | tuple[str, ...] = (),
        success_distance: float = 3.0,
        stop_on_success_radius: bool = True,
        **runtime_kwargs: Any,
    ) -> None:
        if runtime_kwargs:
            unknown = ", ".join(sorted(str(key) for key in runtime_kwargs))
            raise ValueError(f"Unsupported VLN-CE benchmark kwargs: {unknown}")
        self.task_name = str(task_name).lower()
        self.split = str(split)
        self.roles = tuple(str(role) for role in roles)
        self.languages = tuple(str(language) for language in languages)
        self.success_distance = float(success_distance)
        self.stop_on_success_radius = bool(stop_on_success_radius)

    def validate_episode(self, episode: EvalEpisode, *, env: Any, dataset: Any) -> None:
        del env, dataset
        if not episode.source_episode_id:
            raise ValueError(f"VLN-CE episode {episode.episode_uid} is missing source_episode_id")
        if not episode.instruction:
            raise ValueError(f"VLN-CE episode {episode.episode_uid} is missing instruction")
        self.create_runtime(None).initial_pose(episode)

    def create_runtime(self, cfg: Any) -> "VLNCERuntime":
        del cfg
        return VLNCERuntime(
            task_name=self.task_name,
            success_distance=self.success_distance,
            stop_on_success_radius=self.stop_on_success_radius,
        )


class VLNCERuntime(BaseBenchmarkRuntime):
    def __init__(self, *, task_name: str, success_distance: float = 3.0, stop_on_success_radius: bool = True) -> None:
        self.task_name = str(task_name).lower()
        self.success_distance = float(success_distance)
        self.stop_on_success_radius = bool(stop_on_success_radius)
        self._last_metrics: dict[str, dict[str, Any]] = {}
        self._oracle_success: dict[str, int] = {}

    def initial_pose(self, episode: EvalEpisode) -> Pose4D:
        reference_path = episode.payload.get("reference_path") or []
        if reference_path:
            first = reference_path[0]
            return Pose4D(float(first[0]), float(first[1]), float(first[2]), 0.0)
        start_position = episode.payload.get("start_position")
        if start_position is not None:
            return Pose4D(float(start_position[0]), float(start_position[1]), float(start_position[2]), 0.0)
        return Pose4D(0.0, 0.0, 0.0, 0.0)

    def prepare_environment(self, episode: EvalEpisode, env, initial_pose: Pose4D) -> None:
        del env, initial_pose
        self._last_metrics[episode.episode_uid] = {}
        self._oracle_success[episode.episode_uid] = 0

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
        if instruction:
            prepared["instruction"] = instruction
        prepared["vlnce_episode"] = {
            "episode_id": episode.source_episode_id,
            "episode_uid": episode.episode_uid,
            "scene_id": episode.scene_id,
            "task_name": self.task_name,
            "simulator": "habitat",
        }
        return prepared

    def needs_post_action_observation(self) -> bool:
        return False

    def distance_to_goal(self, pose: Pose4D, episode: EvalEpisode) -> float:
        del pose
        metrics = self._last_metrics.get(episode.episode_uid) or {}
        distance = _success_radius_distance(metrics)
        if distance is not None:
            return float(distance)
        return 0.0

    def gt_path_length(self, episode: EvalEpisode) -> float:
        return float(episode.payload.get("gt_path_length", 0.0))

    def is_success(self, pose: Pose4D, episode: EvalEpisode) -> bool:
        del pose
        metrics = self._last_metrics.get(episode.episode_uid) or {}
        distance = _success_radius_distance(metrics)
        if distance is not None and distance <= self.success_distance:
            return True
        return bool(metrics.get("success", False))

    def update_termination(self, state: StepState) -> TerminationStatus:
        metrics = dict(state.diagnostics.get("metrics") or {})
        distance_to_goal = _success_radius_distance(metrics)
        within_success_radius = distance_to_goal is not None and distance_to_goal <= self.success_distance
        reached_success_radius = self.stop_on_success_radius and within_success_radius
        habitat_done = bool(state.post_observation.get("done", False))
        done = bool(habitat_done or reached_success_radius)
        success = int(metrics.get("success", 0))
        oracle_success = max(
            int(metrics.get("oracle_success", metrics.get("success", 0))),
            int(self._oracle_success.get(state.episode.episode_uid, 0)),
        )
        success_source = "habitat_success" if success else "none"
        if within_success_radius:
            success = 1
            oracle_success = 1
            success_source = "success_radius"
            metrics["success"] = 1.0
            spl = _success_radius_spl(metrics=metrics, path_length=state.path_length, episode=state.episode)
            if spl is not None:
                metrics["spl"] = spl
        metrics["oracle_success"] = float(oracle_success)
        self._oracle_success[state.episode.episode_uid] = int(oracle_success)
        self._last_metrics[state.episode.episode_uid] = metrics
        return TerminationStatus(
            done=done,
            success=success,
            oracle_success=oracle_success,
            reason="success_radius" if reached_success_radius else ("habitat_done" if habitat_done else "running"),
            failure=None,
            failure_type=None,
            diagnostics={
                "metrics": metrics,
                "success_distance": self.success_distance,
                "stop_on_success_radius": self.stop_on_success_radius,
                "within_success_radius": within_success_radius,
                "reached_success_radius": reached_success_radius,
                "distance_to_goal": distance_to_goal,
                "success_source": success_source,
            },
        )

    def log_step_artifacts(self, state: StepState, artifacts: Any) -> dict[str, Any]:
        del artifacts
        return {
            "habitat_metrics": dict(state.diagnostics.get("metrics") or {}),
        }


VLNCER2RBenchmarkSpec = VLNCEBenchmarkSpec
VLNCERxRBenchmarkSpec = VLNCEBenchmarkSpec


def _optional_metric_float(metrics: dict[str, Any], key: str) -> float | None:
    if key not in metrics:
        return None
    try:
        value = float(metrics[key])
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def _success_radius_distance(metrics: dict[str, Any]) -> float | None:
    euclidean_distance = _optional_metric_float(metrics, "euclidean_distance_to_goal")
    if euclidean_distance is not None:
        return euclidean_distance
    return _optional_metric_float(metrics, "distance_to_goal")


def _success_radius_spl(*, metrics: dict[str, Any], path_length: float, episode: EvalEpisode) -> float | None:
    gt_path_length = _optional_payload_float(episode.payload, "gt_path_length")
    if gt_path_length is None or gt_path_length <= 0.0:
        return None
    metric_path_length = _optional_metric_float(metrics, "path_length")
    executed_path_length = float(path_length) if metric_path_length is None else metric_path_length
    return float(gt_path_length / max(executed_path_length, gt_path_length, 1e-6))


def _optional_payload_float(payload: dict[str, Any], key: str) -> float | None:
    if key not in payload:
        return None
    try:
        value = float(payload[key])
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) else None
