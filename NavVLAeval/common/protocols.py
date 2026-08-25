from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np

from NavVLAeval.common.types import (
    EnvironmentStepResult,
    EvalEpisode,
    EpisodeHistory,
    Pose4D,
    StepState,
    TerminationStatus,
)


class EvalInputAdapter(Protocol):
    def load_episodes(self, cfg: Any, *, max_samples: int | None) -> list[EvalEpisode]:
        ...

    def fingerprint(self, cfg: Any) -> str:
        ...


class BenchmarkSpec(Protocol):
    def validate_episode(self, episode: EvalEpisode, *, env: Any, dataset: Any) -> None:
        ...

    def create_runtime(self, cfg: Any) -> "BenchmarkRuntime":
        ...


class BenchmarkRuntime(Protocol):
    def initial_pose(self, episode: EvalEpisode) -> Pose4D:
        ...

    def prepare_environment(self, episode: EvalEpisode, env: "EnvironmentBackend", initial_pose: Pose4D) -> None:
        ...

    def instruction_for_step(self, episode: EvalEpisode, history: EpisodeHistory, step: int) -> str:
        ...

    def prepare_observation_for_model(
        self,
        *,
        episode: EvalEpisode,
        history: EpisodeHistory,
        step: int,
        observation: dict[str, Any],
        instruction: str,
    ) -> dict[str, Any]:
        ...

    def needs_post_action_observation(self) -> bool:
        ...

    def distance_to_goal(self, pose: Pose4D, episode: EvalEpisode) -> float:
        ...

    def gt_path_length(self, episode: EvalEpisode) -> float:
        ...

    def is_success(self, pose: Pose4D, episode: EvalEpisode) -> bool:
        ...

    def update_termination(self, state: StepState) -> TerminationStatus:
        ...

    def log_step_artifacts(self, state: StepState, artifacts: "EpisodeArtifactWriterProtocol") -> None:
        ...

    def offline_transition(self, state: StepState) -> EnvironmentStepResult:
        ...


class EnvironmentBackend(Protocol):
    type: str

    def start_episode(self, episode: EvalEpisode, initial_pose: Pose4D) -> dict[str, Any]:
        ...

    def get_observation(self) -> dict[str, Any]:
        ...

    def apply_action(self, current_pose: Pose4D, raw_actions: np.ndarray) -> EnvironmentStepResult:
        ...

    def close_episode(self) -> None:
        ...

    def close(self) -> None:
        ...


class EpisodeArtifactWriterProtocol(Protocol):
    episode_dir: Any
    diagnostics: dict[str, Any]

    def write_eval_info(self, payload: dict[str, Any]) -> None:
        ...

    def write_step_json(self, relative_path: str | Any, payload: Mapping[str, Any]) -> None:
        ...

    def write_image(self, relative_path: str | Any, image: Any) -> bool:
        ...

    def add_warning(self, message: str) -> None:
        ...
