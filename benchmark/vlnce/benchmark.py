from __future__ import annotations

from typing import Any

from benchmark.common.runtime_defaults import BaseBenchmarkRuntime
from benchmark.common.types import EpisodeHistory, EvalEpisode, Pose4D, StepState, TerminationStatus


class VLNCEBenchmark(BaseBenchmarkRuntime):
    def __init__(
        self,
        *,
        task_name: str = "r2r",
        split: str = "val_unseen",
        roles: list[str] | tuple[str, ...] = (),
        languages: list[str] | tuple[str, ...] = (),
        success_distance: float = 3.0,
        stop_on_success_radius: bool = False,
        distance_metric: str = "euclidean",
        **runtime_kwargs: Any,
    ) -> None:
        if runtime_kwargs:
            unknown = ", ".join(sorted(str(key) for key in runtime_kwargs))
            raise ValueError(f"Unsupported VLN-CE benchmark kwargs: {unknown}")
        self.distance_metric = distance_metric
        if distance_metric not in {"euclidean", "geodesic"}:
            raise ValueError("distance_metric must be euclidean or geodesic")
        self.task_name = str(task_name).lower()
        self.split = str(split)
        self.roles = tuple(str(role) for role in roles)
        self.languages = tuple(str(language) for language in languages)
        self.success_distance = float(success_distance)
        self.stop_on_success_radius = bool(stop_on_success_radius)
        self._last_metrics = {}

    def validate_episode(self, episode: EvalEpisode) -> None:
        if not episode.source_episode_id:
            raise ValueError(f"VLN-CE episode {episode.episode_uid} is missing source_episode_id")
        if not episode.instruction:
            raise ValueError(f"VLN-CE episode {episode.episode_uid} is missing instruction")
        self.initial_pose(episode)

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

    def instruction_for_step(self, episode: EvalEpisode, history: EpisodeHistory | None, step: int) -> str:
        del history, step
        return episode.instruction

    def gt_path_length(self, episode: EvalEpisode) -> float:
        return float(episode.payload["gt_path_length"])

    def log_step_artifacts(self, state: StepState, artifacts: Any) -> dict[str, Any]:
        del artifacts
        return {
            "habitat_metrics": dict(state.diagnostics.get("metrics") or {}),
        }

    def observe(self, episode, observation):
        self._last_metrics[episode.episode_uid] = dict(observation["metrics"])

    def distance_to_goal(self, pose, episode):
        key = "euclidean_distance_to_goal" if self.distance_metric == "euclidean" else "distance_to_goal"
        return float(self._last_metrics[episode.episode_uid][key])

    def is_success(self, pose, episode):
        return self.distance_to_goal(pose, episode) <= self.success_distance

    def update_termination(self, state):
        self.observe(state.episode, state.post_observation)
        reached = self.is_success(state.pose_after, state.episode)
        metrics = self._last_metrics[state.episode.episode_uid]
        radius_stop = self.stop_on_success_radius and reached
        native_stop_success = bool(metrics.get("success", False))
        done = bool(state.post_observation["done"] or radius_stop)
        # Simulator time limits alone do not constitute a stop prediction.
        success = int(radius_stop or (native_stop_success and reached))
        return TerminationStatus(
            done,
            success,
            int(reached),
            "success_radius" if radius_stop else ("habitat_done" if done else "running"),
            None,
            None,
        )
