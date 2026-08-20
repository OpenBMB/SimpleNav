from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from NavVLAeval.common.runner.backend_plan import (
    OfflineWorkerBackendPlan,
    WorkerBackendPlan,
)


def AirSimWorkerBackendPlan(*, type: str = "airsim", airsim_port: int, settings_root: str | Path) -> WorkerBackendPlan:
    if type != "airsim":
        raise ValueError(f"AirSimWorkerBackendPlan type must be 'airsim', got {type!r}")
    return WorkerBackendPlan(type="airsim", kwargs={"airsim_port": int(airsim_port), "settings_root": settings_root})


@dataclass(frozen=True)
class Pose4D:
    x: float
    y: float
    z: float
    yaw: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.z, self.yaw], dtype=np.float32)


@dataclass(frozen=True)
class EvalEpisode:
    episode_uid: str
    source_episode_id: str
    scene_id: str
    instruction: str
    source: str
    input_namespace: str
    input_root: str
    payload: dict[str, Any]


@dataclass
class EpisodeHistory:
    images: list[Any] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    poses: list[Pose4D] = field(default_factory=list)
    raw_actions: list[np.ndarray] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    long_memory_tokens: np.ndarray | None = None
    long_memory_tvi: np.ndarray | None = None
    long_memory_blocks: list[dict[str, Any]] = field(default_factory=list)
    long_memory_frame_indices: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class ActionPrediction:
    normalized_actions: np.ndarray
    raw_actions: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentStepResult:
    next_pose: Pose4D
    observation: dict[str, Any]
    data_done: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)
    action_observations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class WorkerPlan:
    worker_index: int
    physical_gpu_id: int
    episodes: list[EvalEpisode]
    run_root: Path
    worker_log_path: Path
    backend: WorkerBackendPlan
    episode_attempts: dict[str, str]


@dataclass(frozen=True)
class RunPlan:
    schema_version: int
    benchmark: str
    run_name: str
    config_sha256: str
    input_fingerprint: str
    total_episode_uids: list[str]
    skipped_episode_uids: list[str]
    pending_episode_uids: list[str]
    worker_plan_paths: list[str]


@dataclass(frozen=True)
class EpisodeResult:
    episode_uid: str
    source_episode_id: str
    scene_id: str
    instruction: str
    success: int
    oracle_success: int
    final_distance: float
    path_length: float
    gt_path_length: float
    steps: int
    failure: str | None
    failure_type: str | None
    termination_reason: str
    failure_traceback: str | None = None
    nDTW: float | None = None


@dataclass(frozen=True)
class StepState:
    episode: EvalEpisode
    step_index: int
    artifact_step_index: int
    instruction: str
    history: EpisodeHistory
    pre_observation: dict[str, Any]
    post_observation: dict[str, Any]
    pose_before: Pose4D
    pose_after: Pose4D
    prediction: ActionPrediction
    raw_action_chunk: np.ndarray
    world_waypoints: np.ndarray
    executed_action_count: int
    distance_before: float
    distance_after: float
    path_length: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    action_observations: list[dict[str, Any]] = field(default_factory=list)
    executed_world_waypoints: np.ndarray | None = None
    termination: TerminationStatus | None = None


@dataclass(frozen=True)
class TerminationStatus:
    done: bool
    success: int
    oracle_success: int
    reason: str
    failure: str | None
    failure_type: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
