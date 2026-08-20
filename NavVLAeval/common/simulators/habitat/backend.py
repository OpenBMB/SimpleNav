from __future__ import annotations

from math import pi
from typing import Any

import numpy as np

from NavVLAeval.common.config import EnvConfig, load_class
from NavVLAeval.common.runner.backend_plan import WorkerBackendPlan
from NavVLAeval.common.simulators.habitat.action_adapter import BodyFrameContinuousActionAdapter
from NavVLAeval.common.simulators.habitat.vlnce031_runtime import VLNCE031HabitatRuntime, runtime_kwargs_from_cfg
from NavVLAeval.common.types import EnvironmentStepResult, EvalEpisode, Pose4D


class VLNCE031HabitatBackendPlanner:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)

    def plan_worker_backend(
        self,
        *,
        cfg: EnvConfig,
        store,
        worker_index: int,
        physical_gpu_id: int,
    ) -> WorkerBackendPlan:
        del cfg, store, worker_index, physical_gpu_id
        return WorkerBackendPlan(type="habitat", kwargs={})


class VLNCE031HabitatBackend:
    type = "habitat"

    def __init__(
        self,
        *,
        cfg: EnvConfig,
        worker_backend: WorkerBackendPlan,
        physical_gpu_id: int,
        start_process: bool = True,
        runtime_factory: Any | None = None,
        action_adapter: Any | None = None,
    ) -> None:
        del start_process
        self.cfg = cfg
        self.worker_backend = worker_backend
        self.physical_gpu_id = int(physical_gpu_id)
        self.capture_action_observations = bool(cfg.kwargs.get("capture_action_observations", True))
        self.action_adapter = action_adapter or _build_action_adapter(cfg)
        factory = runtime_factory or VLNCE031HabitatRuntime
        self.runtime = factory(**runtime_kwargs_from_cfg(cfg.kwargs, physical_gpu_id=self.physical_gpu_id))
        self._latest_payload: dict[str, Any] | None = None
        self._episode: EvalEpisode | None = None

    def start_episode(self, episode: EvalEpisode, initial_pose: Pose4D) -> dict[str, Any]:
        del initial_pose
        self._episode = episode
        self._latest_payload = self.runtime.reset(
            {"episode_id": episode.source_episode_id, "item": episode.payload}
        )
        return {
            "simulator": "habitat",
            "episode_id": episode.source_episode_id,
        }

    def get_observation(self) -> dict[str, Any]:
        if self._latest_payload is None:
            raise RuntimeError("Habitat backend has no active episode")
        return _observation_from_payload(self._latest_payload)

    def apply_action(self, current_pose: Pose4D, raw_actions: np.ndarray) -> EnvironmentStepResult:
        raw_action_array = np.asarray(raw_actions, dtype=np.float32)
        action_chunk = raw_action_array.reshape(-1, raw_action_array.shape[-1])
        if action_chunk.shape[0] <= 0:
            raise ValueError("Habitat action chunk must contain at least one action")

        anchor_pose = self._latest_observation_pose() or current_pose
        current_action_pose = anchor_pose
        action_observations: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        actual_waypoint_poses: list[Pose4D] = []
        payload: dict[str, Any] | None = None
        data_done = False

        for action_index in range(action_chunk.shape[0]):
            decision = self.action_adapter.to_server_action(
                action_chunk,
                stop_prob=None,
                action_index=action_index,
                anchor_pose=anchor_pose,
                current_pose=current_action_pose,
            )
            server_payload = dict(decision["server_payload"])
            decisions.append(decision)
            payload = self.runtime.step(server_payload)
            self._latest_payload = payload
            observation = _observation_from_payload(payload)
            if isinstance(observation.get("pose"), Pose4D):
                current_action_pose = observation["pose"]
                actual_waypoint_poses.append(current_action_pose)
            if self.capture_action_observations:
                action_observations.append(observation)
            data_done = bool(payload.get("done", False))
            if data_done or decision.get("action_label") == "STOP":
                break

        assert payload is not None
        observation = _observation_from_payload(payload)
        selected_waypoint_indices = [
            int(decision["log"]["action_index"])
            for decision in decisions
            if decision.get("log", {}).get("action_index") is not None
        ]
        executed_action_count = len(decisions)
        return EnvironmentStepResult(
            next_pose=observation.get("pose", Pose4D(0.0, 0.0, 0.0, 0.0)),
            observation=observation,
            data_done=data_done,
            diagnostics={
                "habitat_action": dict(decisions[-1].get("log") or {}),
                "habitat_actions": [dict(decision.get("log") or {}) for decision in decisions],
                "metrics": dict(payload.get("metrics") or {}),
                "habitat_metrics": dict(payload.get("metrics") or {}),
                "original_waypoint_count": int(action_chunk.shape[0]),
                "executed_model_waypoint_count": int(executed_action_count),
                "executed_native_action_count": int(executed_action_count),
                "executed_waypoint_count": int(executed_action_count),
                "skipped_waypoint_count": 0,
                "selected_waypoint_indices": selected_waypoint_indices,
                "actual_waypoint_poses": [pose.as_array().tolist() for pose in actual_waypoint_poses],
                "waypoint_control": _waypoint_control_diagnostics(
                    anchor_pose=anchor_pose,
                    raw_actions=action_chunk,
                    actual_waypoint_poses=actual_waypoint_poses,
                ),
                "capture_action_observations": self.capture_action_observations,
            },
            action_observations=action_observations,
        )

    def close_episode(self) -> None:
        self._latest_payload = None
        self._episode = None

    def close(self) -> None:
        self.runtime.close()
        self.close_episode()

    def _latest_observation_pose(self) -> Pose4D | None:
        if self._latest_payload is None:
            return None
        observation = _observation_from_payload(self._latest_payload)
        pose = observation.get("pose")
        return pose if isinstance(pose, Pose4D) else None

    def project_action_to_world(self, current_pose: Pose4D, raw_actions: np.ndarray) -> np.ndarray:
        """Project anchor-relative NavVLA waypoints to the VLN-CE world frame."""
        action_chunk = np.asarray(raw_actions, dtype=np.float32).reshape(-1, 4)
        return np.asarray(
            [_anchor_relative_waypoint_to_world_pose(current_pose, waypoint).as_array() for waypoint in action_chunk],
            dtype=np.float32,
        )


def _build_action_adapter(cfg: EnvConfig) -> BodyFrameContinuousActionAdapter:
    adapter_class_path = cfg.kwargs.get("action_adapter_class_path")
    adapter_kwargs = dict(cfg.kwargs.get("action_adapter_kwargs") or {})
    if adapter_class_path:
        adapter_cls = load_class(str(adapter_class_path))
        return adapter_cls(**adapter_kwargs)
    return BodyFrameContinuousActionAdapter(**adapter_kwargs)


def _observation_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    observation = dict(payload.get("observation") or {})
    converted: dict[str, Any] = {}
    images = _images_from_observation_payload(observation)
    if images:
        converted["images"] = images
        converted["image"] = images["front"] if "front" in images else next(iter(images.values()))
    if observation.get("depth") is not None:
        converted["depth"] = np.asarray(observation["depth"], dtype=np.float32)
    pose = payload.get("pose")
    if pose is not None:
        converted["pose"] = _pose_from_payload(pose)
        converted["state"] = converted["pose"].as_array()
    converted["instruction"] = str(payload.get("instruction") or "")
    converted["metrics"] = dict(payload.get("metrics") or {})
    converted["done"] = bool(payload.get("done", False))
    converted["habitat_payload"] = payload
    return converted


def _images_from_observation_payload(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    camera_keys = {
        "front": ("rgb", "rgb_front", "front_rgb"),
        "left": ("rgb_left", "left_rgb"),
        "right": ("rgb_right", "right_rgb"),
        "rear": ("rgb_rear", "rear_rgb", "back_rgb", "rgb_back"),
    }
    images: dict[str, np.ndarray] = {}
    for camera_name, keys in camera_keys.items():
        for key in keys:
            if observation.get(key) is not None:
                images[camera_name] = np.asarray(observation[key], dtype=np.uint8)
                break
    return images


def _pose_from_payload(pose: Any) -> Pose4D:
    values = np.asarray(pose, dtype=np.float64).reshape(-1)
    if values.size < 4:
        raise ValueError(f"Habitat pose must contain at least 4 values, got {pose!r}")
    habitat_x = float(values[0])
    habitat_y = float(values[1])
    habitat_z = float(values[2])
    habitat_yaw = float(values[3])
    return Pose4D(
        -habitat_z,
        habitat_x,
        -habitat_y,
        _wrap_to_pi(-habitat_yaw - (pi / 2.0)),
    )


def _anchor_relative_waypoint_to_world_pose(anchor_pose: Pose4D, waypoint: np.ndarray) -> Pose4D:
    forward, right, vertical, yaw = [float(value) for value in np.asarray(waypoint, dtype=np.float64).reshape(4)]
    anchor_yaw = float(anchor_pose.yaw)
    return Pose4D(
        float(anchor_pose.x - np.sin(anchor_yaw) * forward - np.cos(anchor_yaw) * right),
        float(anchor_pose.y + np.cos(anchor_yaw) * forward - np.sin(anchor_yaw) * right),
        float(anchor_pose.z + vertical),
        _wrap_to_pi(anchor_yaw + yaw),
    )


def _waypoint_control_diagnostics(
    *,
    anchor_pose: Pose4D,
    raw_actions: np.ndarray,
    actual_waypoint_poses: list[Pose4D],
) -> list[dict[str, Any]]:
    """Record requested and actual waypoint poses for control-error auditing."""
    expected = [
        _anchor_relative_waypoint_to_world_pose(anchor_pose, waypoint)
        for waypoint in np.asarray(raw_actions, dtype=np.float32).reshape(-1, 4)[: len(actual_waypoint_poses)]
    ]
    diagnostics: list[dict[str, Any]] = []
    for index, (target, actual) in enumerate(zip(expected, actual_waypoint_poses, strict=True)):
        position_error = float(np.linalg.norm(actual.as_array()[:3] - target.as_array()[:3]))
        yaw_error = _wrap_to_pi(float(actual.yaw) - float(target.yaw))
        diagnostics.append(
            {
                "action_index": index,
                "target_pose": target.as_array().tolist(),
                "actual_pose": actual.as_array().tolist(),
                "position_error_m": position_error,
                "yaw_error_rad": yaw_error,
            }
        )
    return diagnostics


def _wrap_to_pi(value: float) -> float:
    return float((float(value) + pi) % (2.0 * pi) - pi)
