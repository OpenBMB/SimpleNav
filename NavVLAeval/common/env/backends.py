from __future__ import annotations

from typing import Any

import numpy as np

from NavVLAeval.common.runner.backend_plan import WorkerBackendPlan
from NavVLAeval.common.config import EnvConfig, load_class
from NavVLAeval.common.types import EnvironmentStepResult, EvalEpisode, Pose4D


class OfflineReplayBackend:
    type = "offline"

    def __init__(self) -> None:
        self._episode: EvalEpisode | None = None
        self._frames: list[dict[str, Any]] = []
        self._cursor = 0
        self._pose = Pose4D(0.0, 0.0, 0.0, 0.0)

    def start_episode(self, episode: EvalEpisode, initial_pose: Pose4D) -> dict[str, Any]:
        frames = episode.payload.get("offline_frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"offline replay episode {episode.episode_uid} is missing offline_frames")
        self._episode = episode
        self._frames = [frame if isinstance(frame, dict) else {"pose": frame} for frame in frames]
        self._cursor = 0
        self._pose = _pose_from_frame(self._frames[0], fallback=initial_pose, episode_uid=episode.episode_uid)
        return {"frame_count": len(self._frames)}

    def get_observation(self) -> dict[str, Any]:
        self._require_started()
        return _observation_from_frame(self._frames[self._cursor], self._pose)

    def apply_action(self, current_pose: Pose4D, raw_actions: np.ndarray) -> EnvironmentStepResult:
        del current_pose, raw_actions
        self._require_started()
        self._cursor = min(self._cursor + 1, len(self._frames) - 1)
        self._pose = _pose_from_frame(self._frames[self._cursor], fallback=self._pose, episode_uid=self._episode_uid())
        return EnvironmentStepResult(
            next_pose=self._pose,
            observation=self.get_observation(),
            data_done=self._cursor >= len(self._frames) - 1,
            diagnostics={"offline_frame_index": self._cursor},
        )

    def close_episode(self) -> None:
        self._episode = None
        self._frames = []
        self._cursor = 0

    def close(self) -> None:
        self.close_episode()

    def _require_started(self) -> None:
        if self._episode is None or not self._frames:
            raise RuntimeError("offline replay backend has no active episode")

    def _episode_uid(self) -> str:
        return self._episode.episode_uid if self._episode is not None else "<unknown>"


def create_environment_backend(
    *,
    cfg: EnvConfig,
    worker_backend: WorkerBackendPlan,
    physical_gpu_id: int,
    start_process: bool = True,
):
    if cfg.type != worker_backend.type:
        raise ValueError(f"worker backend type {worker_backend.type!r} does not match env.type {cfg.type!r}")
    if worker_backend.type == "offline" and not getattr(cfg, "backend_class_path", None):
        return OfflineReplayBackend()
    if not getattr(cfg, "backend_class_path", None):
        if worker_backend.type in {"airsim", "unrealzoo"}:
            raise ValueError(f"env.backend_class_path is required for {worker_backend.type} env")
        raise ValueError(f"env.backend_class_path is required for worker backend type: {worker_backend.type!r}")
    backend_cls = load_class(cfg.backend_class_path)
    return backend_cls(
        cfg=cfg,
        worker_backend=worker_backend,
        physical_gpu_id=physical_gpu_id,
        start_process=start_process,
    )
def _pose_from_frame(frame: dict[str, Any], *, fallback: Pose4D, episode_uid: str) -> Pose4D:
    pose = frame.get("pose")
    if pose is None:
        pose = frame.get("state")
    if isinstance(pose, dict):
        position = pose.get("position") or pose.get("xyz")
        yaw = float(pose.get("yaw", fallback.yaw))
        if position is None:
            raise ValueError(f"offline frame for {episode_uid} has pose dict without position")
        return Pose4D(float(position[0]), float(position[1]), float(position[2]), yaw)
    if pose is None:
        return fallback
    if hasattr(pose, "tolist"):
        pose = pose.tolist()
    if len(pose) < 4:
        raise ValueError(f"offline frame for {episode_uid} has pose shorter than 4")
    return Pose4D(float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3]))


def _observation_from_frame(frame: dict[str, Any], pose: Pose4D) -> dict[str, Any]:
    observation = frame.get("observation")
    if isinstance(observation, dict):
        payload = dict(observation)
    else:
        payload = {}
    if "state" not in payload:
        payload["state"] = pose.as_array()
    elif not isinstance(payload["state"], np.ndarray):
        payload["state"] = np.asarray(payload["state"], dtype=np.float32)
    return payload
