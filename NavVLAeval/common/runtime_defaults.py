from __future__ import annotations

from typing import Any

from NavVLAeval.common.types import EnvironmentStepResult, EpisodeHistory, EvalEpisode, StepState


class BaseBenchmarkRuntime:
    def stop_at_first_success_waypoint(self) -> bool:
        return True

    def prepare_observation_for_model(
        self,
        *,
        episode: EvalEpisode,
        history: EpisodeHistory,
        step: int,
        observation: dict[str, Any],
        instruction: str,
    ) -> dict[str, Any]:
        del episode, history, step, instruction
        return observation

    def needs_post_action_observation(self) -> bool:
        return False

    def log_step_artifacts(self, state: StepState, artifacts: Any) -> None:
        del state, artifacts
        return None

    def offline_transition(self, state: StepState) -> EnvironmentStepResult:
        del state
        raise RuntimeError("offline_transition is unsupported for this runtime")


class BaseRuntimeDatasetAdapter:
    def history_observations_for_update(
        self,
        *,
        pre_observation: dict[str, Any],
        post_observation: dict[str, Any],
        step_result: EnvironmentStepResult,
        action_observations: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        del pre_observation, step_result, action_observations
        return [post_observation]
