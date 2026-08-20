from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from NavVLAeval.common.runtime_defaults import BaseBenchmarkRuntime
from NavVLAeval.common.simulators.base import ObjectPlacementBackend, PoseControlBackend
from NavVLAeval.common.types import (
    EnvironmentStepResult,
    EpisodeHistory,
    EvalEpisode,
    Pose4D,
    StepState,
    TerminationStatus,
)

TRAVELUAV_RGB_FOLDERS = ["frontcamera", "leftcamera", "rightcamera", "rearcamera", "downcamera"]
TRAVELUAV_DEPTH_CLOSE_VALUE = 1.0
TRAVELUAV_DEPTH_CLOSE_FRACTION = 0.1
TRAVELUAV_DEPTH_TINY_DIFF = 3.0
TRAVELUAV_DEPTH_STUCK_DISTANCE = 0.1


class TravelUAVBenchmarkSpec:
    def __init__(
        self,
        *,
        stop_policy: str = "none",
        success_radius: float = 20.0,
        dataset_root: str | Path | None = None,
        always_help: bool = False,
        use_gt: bool = False,
        depth_collision_policy: str = "stop",
        ignore_movement_collision: bool = False,
        groundingdino_config: str | Path | None = None,
        groundingdino_model_path: str | Path | None = None,
        dino_device: str | int = "cuda",
        **runtime_kwargs,
    ):
        if runtime_kwargs:
            unknown = ", ".join(sorted(str(key) for key in runtime_kwargs))
            raise ValueError(f"Unsupported TravelUAV benchmark kwargs: {unknown}")
        self.stop_policy = str(stop_policy)
        self.success_radius = float(success_radius)
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.always_help = bool(always_help)
        self.use_gt = bool(use_gt)
        if self.stop_policy not in {"none", "dino"}:
            raise ValueError(f"Unsupported TravelUAV stop_policy: {self.stop_policy}")
        self.depth_collision_policy = str(depth_collision_policy)
        if self.depth_collision_policy not in {"stop", "log_only"}:
            raise ValueError(f"Unsupported TravelUAV depth_collision_policy: {self.depth_collision_policy}")
        self.ignore_movement_collision = bool(ignore_movement_collision)
        self.groundingdino_config = Path(groundingdino_config) if groundingdino_config else None
        self.groundingdino_model_path = Path(groundingdino_model_path) if groundingdino_model_path else None
        self.dino_device = dino_device

    def validate_episode(self, episode: EvalEpisode, *, env: Any, dataset: Any) -> None:
        del env, dataset
        payload = episode.payload
        for key in ("env_name", "trajectory"):
            if key not in payload or payload[key] in (None, ""):
                raise ValueError(f"TravelUAV episode {episode.episode_uid} is missing payload[{key!r}]")
        runtime = self.create_runtime(None)
        runtime.initial_pose(episode)
        runtime.goal_position(episode)

    def create_runtime(self, cfg: Any) -> "TravelUAVBenchmarkRuntime":
        del cfg
        return TravelUAVBenchmarkRuntime(
            stop_policy=self.stop_policy,
            success_radius=self.success_radius,
            always_help=self.always_help,
            use_gt=self.use_gt,
            depth_collision_policy=self.depth_collision_policy,
            ignore_movement_collision=self.ignore_movement_collision,
            groundingdino_config=self.groundingdino_config,
            groundingdino_model_path=self.groundingdino_model_path,
            dino_device=self.dino_device,
        )


class TravelUAVBenchmarkRuntime(BaseBenchmarkRuntime):
    def __init__(
        self,
        *,
        stop_policy: str,
        success_radius: float,
        always_help: bool,
        use_gt: bool,
        depth_collision_policy: str,
        ignore_movement_collision: bool,
        groundingdino_config: str | Path | None,
        groundingdino_model_path: str | Path | None,
        dino_device: str | int,
    ):
        self.stop_policy = stop_policy
        self.success_radius = float(success_radius)
        self.always_help = bool(always_help)
        self.use_gt = bool(use_gt)
        if self.stop_policy not in {"none", "dino"}:
            raise ValueError(f"Unsupported TravelUAV stop_policy: {self.stop_policy}")
        self.depth_collision_policy = str(depth_collision_policy)
        self.ignore_movement_collision = bool(ignore_movement_collision)
        self.groundingdino_config = Path(groundingdino_config) if groundingdino_config else None
        self.groundingdino_model_path = Path(groundingdino_model_path) if groundingdino_model_path else None
        self.dino_device = dino_device
        self._dino_monitor = None
        self._object_description_cache: dict[Path, dict[str, str]] = {}
        self._distance_history: dict[str, list[float]] = {}
        self._termination_state: dict[str, dict[str, Any]] = {}

    def initial_pose(self, episode: EvalEpisode) -> Pose4D:
        payload = episode.payload
        if payload.get("raw_start_pose") is not None:
            return _pose_from_raw_step(payload["raw_start_pose"])
        for key in ("trajectory_raw_detailed", "trajectory_raw"):
            trajectory = payload.get(key)
            if isinstance(trajectory, list) and trajectory:
                return _pose_from_raw_step(trajectory[0])
        trajectory = payload.get("trajectory")
        if isinstance(trajectory, list) and trajectory:
            return _pose_from_trajectory_step(trajectory[0])
        pose = payload.get("start_pose") or payload.get("pose")
        if pose is None or len(pose) < 4:
            raise ValueError(f"TravelUAV episode {episode.episode_uid} is missing start pose")
        return Pose4D(float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3]))

    def prepare_environment(self, episode: EvalEpisode, env, initial_pose: Pose4D) -> None:
        self._termination_state[episode.episode_uid] = {
            "success": False,
            "oracle_success": False,
            "early_end": False,
        }
        self._distance_history[episode.episode_uid] = []
        object_info = _object_info_for_episode(episode)
        if object_info:
            if not isinstance(env, ObjectPlacementBackend):
                raise TypeError("TravelUAV environment backend must implement set_object(object_info)")
            if not env.set_object(object_info):
                raise RuntimeError(f"TravelUAV failed to place object for {episode.episode_uid}")
        if not isinstance(env, PoseControlBackend):
            raise TypeError("TravelUAV environment backend must implement reset_pose(pose)")
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
        traveluav_episode = prepared.get("traveluav_episode")
        if not isinstance(traveluav_episode, dict):
            raise ValueError(f"TravelUAV observation for {episode.episode_uid} is missing traveluav_episode")
        traveluav_episode = dict(traveluav_episode)
        metadata = dict(traveluav_episode.get("navvla_eval") or {})
        metadata["episode_id"] = episode.source_episode_id
        metadata["episode_uid"] = episode.episode_uid
        traveluav_episode["navvla_eval"] = metadata
        traveluav_episode["instruction"] = instruction
        for key in (
            "object_description",
            "object_desc",
            "object_name",
            "target",
            "target_position",
            "object_position",
        ):
            value = _payload_value(episode.payload, key)
            if value is not None:
                traveluav_episode[key] = value
        prepared["traveluav_episode"] = traveluav_episode
        prepared["stage"] = self.stage_for_observation(episode=episode, observation=prepared)
        return prepared

    def stage_for_observation(self, *, episode: EvalEpisode, observation: dict[str, Any]) -> str:
        traveluav_episode = observation.get("traveluav_episode")
        if not isinstance(traveluav_episode, dict):
            raise ValueError(f"TravelUAV observation for {episode.episode_uid} is missing traveluav_episode")
        sensors = traveluav_episode.get("sensors")
        if not isinstance(sensors, dict):
            raise ValueError(f"TravelUAV observation for {episode.episode_uid} is missing sensors")
        state = sensors.get("state")
        imu = sensors.get("imu")
        if not isinstance(state, dict) or not isinstance(imu, dict):
            raise ValueError(f"TravelUAV observation for {episode.episode_uid} is missing state/imu sensors")
        position = state.get("position")
        rotation = imu.get("rotation")
        if position is None or rotation is None:
            raise ValueError(f"TravelUAV observation for {episode.episode_uid} is missing position/rotation")
        trajectory = episode.payload.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError(f"TravelUAV episode {episode.episode_uid} is missing trajectory for stage computation")
        return _traveluav_body_frame_stage_from_trajectory(
            current_position=np.asarray(position, dtype=np.float32).reshape(3),
            trajectory=[_trajectory_position(step) for step in trajectory],
            rotation=np.asarray(rotation, dtype=np.float32).reshape(3, 3),
        )

    def goal_position(self, episode: EvalEpisode) -> np.ndarray:
        position = _goal_position_from_payload(episode.payload)
        if position is None:
            raise ValueError(f"TravelUAV episode {episode.episode_uid} is missing goal/target position")
        return np.asarray(position, dtype=np.float32).reshape(-1)[:3]

    def distance_to_goal(self, pose: Pose4D, episode: EvalEpisode) -> float:
        return float(np.linalg.norm(pose.as_array()[:3] - self.goal_position(episode)))

    def gt_path_length(self, episode: EvalEpisode) -> float:
        trajectory = episode.payload.get("trajectory")
        if isinstance(trajectory, list) and len(trajectory) >= 2:
            positions = [_trajectory_position(step) for step in trajectory]
            return float(sum(np.linalg.norm(curr - prev) for prev, curr in zip(positions[:-1], positions[1:])))
        return float(np.linalg.norm(self.goal_position(episode) - self.initial_pose(episode).as_array()[:3]))

    def is_success(self, pose: Pose4D, episode: EvalEpisode) -> bool:
        return self.distance_to_goal(pose, episode) < self.success_radius

    def update_termination(self, state: StepState) -> TerminationStatus:
        distance = self.distance_to_goal(state.pose_after, state.episode)
        distance_history = self._distance_history.setdefault(state.episode.episode_uid, [])
        distance_history.append(distance)
        termination_state = self._termination_state.setdefault(
            state.episode.episode_uid,
            {"success": False, "oracle_success": False, "early_end": False},
        )
        movement_collision, movement_reason = _movement_collision(state.post_observation)
        depth_collision = _depth_collision_payload(state.pre_observation, state.post_observation)
        collision_diagnostics: dict[str, Any] = {}
        if movement_collision:
            collision_diagnostics["movement_collision"] = True
            if movement_reason:
                collision_diagnostics["movement_collision_reason"] = movement_reason
            if not self.ignore_movement_collision and self.depth_collision_policy == "stop":
                return TerminationStatus(
                    done=True,
                    success=0,
                    oracle_success=0,
                    reason=f"collision:movement:{movement_reason}" if movement_reason else "collision:movement",
                    failure=None,
                    failure_type=None,
                    diagnostics={"distance": distance, **collision_diagnostics},
                )
        if self.ignore_movement_collision and depth_collision["collision"] and self.depth_collision_policy == "stop":
            return TerminationStatus(
                done=True,
                success=0,
                oracle_success=0,
                reason=f"collision:depth:{depth_collision['reason']}",
                failure=None,
                failure_type=None,
                diagnostics={"distance": distance, "depth_collision": depth_collision, **collision_diagnostics},
            )
        if _target_distance_increasing_for_10frames(distance_history):
            return TerminationStatus(
                done=True,
                success=0,
                oracle_success=0,
                reason="distance_increasing_10frames",
                failure=None,
                failure_type=None,
                diagnostics={"distance": distance, **collision_diagnostics},
            )
        predicted_done = self._predicted_done(state)
        oracle_waypoints = state.executed_world_waypoints if state.executed_world_waypoints is not None else state.world_waypoints
        waypoint_oracle = _action_waypoints_enter_goal(oracle_waypoints, state.episode, success_radius=self.success_radius)
        current_success = bool(self.is_success(state.pose_after, state.episode))
        termination_state["oracle_success"] = bool(termination_state["oracle_success"] or waypoint_oracle or current_success)
        if predicted_done:
            if distance <= self.success_radius and not termination_state["early_end"]:
                termination_state["success"] = True
            elif distance > self.success_radius:
                termination_state["early_end"] = True
        done = bool(
            termination_state["success"]
            or (termination_state["oracle_success"] and termination_state["early_end"])
        )
        reason = "running"
        if termination_state["success"]:
            reason = "success"
        elif termination_state["oracle_success"] and termination_state["early_end"]:
            reason = "oracle_success_after_early_end"
        elif predicted_done:
            reason = "early_end"
        return TerminationStatus(
            done=done,
            success=int(termination_state["success"]),
            oracle_success=int(termination_state["oracle_success"]),
            reason=reason,
            failure=None,
            failure_type=None,
            diagnostics={"distance": distance, "predicted_done": predicted_done, **collision_diagnostics},
        )

    def needs_post_action_observation(self) -> bool:
        return False

    def log_step_artifacts(self, state: StepState, artifacts) -> dict[str, Any]:
        del artifacts
        return _traveluav_step_artifacts(state)

    def offline_transition(self, state: StepState) -> EnvironmentStepResult:
        raise RuntimeError("offline_transition is unsupported for this runtime")

    def _predicted_done(self, state: StepState) -> bool:
        if self.stop_policy == "none":
            return False
        if self.stop_policy == "dino":
            return self._dino_predicted_done(state)
        raise ValueError(f"Unsupported TravelUAV stop_policy: {self.stop_policy}")

    def _dino_predicted_done(self, state: StepState) -> bool:
        monitor = self._get_dino_monitor()
        episode_payload = _traveluav_monitor_episode(state)
        object_query = _dino_object_query(state.episode, cache=self._object_description_cache)
        return bool(monitor.get_dino_results(episode_payload, object_query))

    def _get_dino_monitor(self):
        if self._dino_monitor is not None:
            return self._dino_monitor
        if self.groundingdino_config is None:
            raise FileNotFoundError("groundingdino_config is required when TravelUAV DINO stop is enabled")
        if self.groundingdino_model_path is None:
            raise FileNotFoundError("groundingdino_model_path is required when TravelUAV DINO stop is enabled")

        from NavVLAeval.traveluav.dino_monitor import TravelUAVDinoMonitor

        self._dino_monitor = TravelUAVDinoMonitor(
            groundingdino_config=self.groundingdino_config,
            groundingdino_model_path=self.groundingdino_model_path,
            device=self.dino_device,
        )
        return self._dino_monitor


def _trajectory_position(step: Any) -> np.ndarray:
    if isinstance(step, dict):
        for key in ("position", "pose", "location", "xyz"):
            if key in step:
                return np.asarray(step[key], dtype=np.float32).reshape(-1)[:3]
    return np.asarray(step, dtype=np.float32).reshape(-1)[:3]


def _payload_value(payload: dict[str, Any], key: str) -> Any:
    for container in (payload, payload.get("benchmark_metadata"), payload.get("source_metadata")):
        if isinstance(container, dict) and key in container and container[key] is not None:
            return container[key]
    return None


def _payload_text(payload: dict[str, Any], key: str) -> str:
    value = _payload_value(payload, key)
    if isinstance(value, str):
        return value.strip()
    return ""


def _goal_position_from_payload(payload: dict[str, Any]) -> Any:
    position = _first_payload_position(
        _payload_value(payload, "object_position"),
        _payload_value(payload, "target_position"),
        _payload_value(payload, "target"),
        payload.get("goal_position"),
    )
    return position.tolist() if position is not None else None


def _object_info_for_episode(episode: EvalEpisode) -> dict[str, Any] | None:
    payload = episode.payload
    position = _first_payload_position(
        _payload_value(payload, "object_position"),
        _payload_value(payload, "target_position"),
        _payload_value(payload, "target"),
    )
    object_info = _payload_value(payload, "object") or _payload_value(payload, "object_info")
    if isinstance(object_info, dict):
        prepared = dict(object_info)
        if position is not None and "pose" not in prepared:
            prepared["pose"] = position.tolist() + [0.0, 0.0, 0.0, 1.0]
        return prepared
    asset_name = (_payload_text(payload, "object_name") or _payload_text(payload, "target_object")).strip()
    if not asset_name or position is None:
        return None
    return {
        "asset_name": asset_name,
        "pose": position.tolist() + [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    }


def _payload_position(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if all(axis in value for axis in ("x", "y", "z")):
            return np.asarray([value["x"], value["y"], value["z"]], dtype=np.float32).reshape(3)
        for key in ("position", "pose", "location", "xyz"):
            if key in value:
                return _payload_position(value[key])
        return None
    try:
        return np.asarray(value, dtype=np.float32).reshape(-1)[:3]
    except (TypeError, ValueError):
        return None


def _first_payload_position(*values: Any) -> np.ndarray | None:
    for value in values:
        position = _payload_position(value)
        if position is not None:
            return position
    return None


def _traveluav_body_frame_stage_from_trajectory(
    *,
    current_position: Any,
    trajectory: list[np.ndarray],
    rotation: Any,
) -> str:
    points = [np.asarray(point, dtype=np.float32).reshape(-1)[:3] for point in trajectory]
    if not points:
        return "cruise"
    target_position = _traveluav_forward_shortest_position(current_position, points)
    return _traveluav_body_frame_stage(
        current_position=current_position,
        target_position=target_position,
        rotation=rotation,
        final_position=points[-1],
    )


def _traveluav_forward_shortest_position(current_position: Any, trajectory: list[np.ndarray]) -> np.ndarray:
    points = [np.asarray(point, dtype=np.float32).reshape(-1)[:3] for point in trajectory]
    if not points:
        return np.asarray(current_position, dtype=np.float32).reshape(-1)[:3]
    current = np.asarray(current_position, dtype=np.float32).reshape(-1)[:3]
    xy = current[:2]
    distances = [float(np.linalg.norm(point[:2] - xy)) for point in points]
    true_index = int(np.argmin(distances))
    for point in points[min(true_index, len(points) - 1) :]:
        distance = float(np.linalg.norm(point[:2] - xy))
        if distance > 6.0 or np.array_equal(point, points[-1]):
            return point
    return points[-1]


def _traveluav_body_frame_stage(
    *,
    current_position: Any,
    target_position: Any,
    rotation: Any,
    final_position: Any | None = None,
    takeoff_delta_z: float = -3.0,
    landing_delta_z: float = 7.0,
    landing_distance_m: float = 10.0,
    turn_threshold_deg: float = 20.0,
) -> str:
    current = np.asarray(current_position, dtype=np.float32).reshape(-1)[:3]
    target = np.asarray(target_position, dtype=np.float32).reshape(-1)[:3]
    rotation_array = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    target_body = rotation_array.T @ (target - current)

    if float(target_body[2]) < takeoff_delta_z:
        return "take off"

    if final_position is not None:
        final = np.asarray(final_position, dtype=np.float32).reshape(-1)[:3]
        if float(np.linalg.norm(current[:2] - final[:2])) < landing_distance_m:
            return "landing"

    if float(target_body[2]) > landing_delta_z:
        return "landing"

    forward = float(target_body[0])
    lateral = float(target_body[1])
    horizontal_norm = float(np.linalg.norm(target_body[:2]))
    if horizontal_norm <= 1e-6:
        return "cruise"

    if forward > 0.0:
        lateral_angle = abs(math.degrees(math.atan2(lateral, forward)))
        if lateral_angle <= turn_threshold_deg:
            return "cruise"

    if abs(lateral) <= 1e-6:
        return "cruise"
    return "right" if lateral > 0.0 else "left"


def _pose_from_raw_step(step: Any) -> Pose4D:
    if isinstance(step, dict):
        values = step.get("position") or step.get("pose") or step.get("location")
        yaw = step.get("yaw", 0.0)
        if isinstance(values, dict):
            return Pose4D(float(values["x"]), float(values["y"]), float(values["z"]), float(yaw))
        return Pose4D(float(values[0]), float(values[1]), float(values[2]), float(yaw))
    values = np.asarray(step, dtype=np.float32).reshape(-1)
    yaw = float(values[3]) if len(values) > 3 else 0.0
    return Pose4D(float(values[0]), float(values[1]), float(values[2]), yaw)


def _pose_from_trajectory_step(step: Any) -> Pose4D:
    position = _trajectory_position(step)
    yaw = 0.0
    if isinstance(step, dict) and "yaw" in step:
        yaw = float(step["yaw"])
    elif not isinstance(step, dict):
        values = np.asarray(step, dtype=np.float32).reshape(-1)
        if len(values) > 3:
            yaw = float(values[3])
    return Pose4D(float(position[0]), float(position[1]), float(position[2]), yaw)


def _movement_collision(observation: dict[str, Any]) -> tuple[bool, str | None]:
    episode = observation.get("traveluav_episode") if isinstance(observation, dict) else None
    state = episode.get("sensors", {}).get("state", {}) if isinstance(episode, dict) else {}
    movement = state.get("movement", {}) if isinstance(state, dict) else {}
    if isinstance(movement, dict) and movement.get("collision"):
        return True, str(movement.get("collision_reason") or movement.get("reason") or "")
    return False, None


def _target_distance_increasing_for_10frames(distances: list[float]) -> bool:
    if len(distances) < 10:
        return False
    recent = distances[-10:]
    return all(curr > prev for prev, curr in zip(recent[:-1], recent[1:]))


def _action_waypoints_enter_goal(waypoints: Any, episode: EvalEpisode, *, success_radius: float) -> bool:
    if waypoints is None:
        return False
    array = np.asarray(waypoints, dtype=np.float32)
    if array.size == 0:
        return False
    array = array.reshape(-1, array.shape[-1])
    if array.shape[-1] < 3:
        return False
    goal = np.asarray(_goal_position_from_payload(episode.payload), dtype=np.float32).reshape(1, 3)
    return bool(np.any(np.linalg.norm(array[:, :3] - goal, axis=1) < float(success_radius)))


def _traveluav_step_artifacts(state: StepState) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "depth_collision": _depth_collision_payload(state.pre_observation, state.post_observation),
    }
    dino_payload = _dino_payload(state)
    if dino_payload:
        payload["dino"] = dino_payload
    return payload


def _dino_payload(state: StepState) -> dict[str, Any]:
    diagnostics = dict(getattr(state, "diagnostics", {}) or {})
    dino = diagnostics.get("dino") or diagnostics.get("dino_result")
    return dino if isinstance(dino, dict) else {}


def _depth_collision_payload(pre_observation: dict[str, Any], post_observation: dict[str, Any]) -> dict[str, Any]:
    pre_episode = _traveluav_episode_payload(pre_observation)
    post_episode = _traveluav_episode_payload(post_observation)
    pre_depths = pre_episode.get("depth") if isinstance(pre_episode, dict) else None
    post_depths = post_episode.get("depth") if isinstance(post_episode, dict) else None
    per_camera: list[dict[str, Any]] = []
    tiny_diff = False
    close_cameras: list[str] = []
    if isinstance(post_depths, (list, tuple)):
        pre_depth_list = pre_depths if isinstance(pre_depths, (list, tuple)) else []
        diffs: list[float] = []
        for index, post_depth in enumerate(post_depths):
            camera_name = TRAVELUAV_RGB_FOLDERS[index] if index < len(TRAVELUAV_RGB_FOLDERS) else f"camera_{index}"
            post_image = np.asarray(post_depth, dtype=np.float32)
            if post_image.size == 0:
                continue
            close_ratio = float(np.mean(post_image <= TRAVELUAV_DEPTH_CLOSE_VALUE))
            close = bool(close_ratio > TRAVELUAV_DEPTH_CLOSE_FRACTION)
            diff_mean = None
            if index < len(pre_depth_list):
                pre_image = np.asarray(pre_depth_list[index], dtype=np.float32)
                if pre_image.shape == post_image.shape:
                    diff_mean = float(np.mean(np.abs(pre_image - post_image)))
                    diffs.append(diff_mean)
            if close:
                close_cameras.append(camera_name)
            per_camera.append(
                {
                    "camera": camera_name,
                    "close": close,
                    "close_pixel_ratio": close_ratio,
                    "mean_abs_depth_diff": diff_mean,
                }
            )
        if diffs and np.all(np.asarray(diffs, dtype=np.float32) < TRAVELUAV_DEPTH_TINY_DIFF):
            tiny_diff = True
    translation_distance = _translation_distance(pre_episode, post_episode)
    stuck_distance = translation_distance is not None and translation_distance < TRAVELUAV_DEPTH_STUCK_DISTANCE
    reason = None
    if tiny_diff:
        reason = "tiny diff"
    elif close_cameras:
        reason = "close"
    elif stuck_distance:
        reason = "distance"
    return {
        "collision": reason is not None,
        "reason": reason,
        "close_cameras": close_cameras,
        "per_camera": per_camera,
        "translation_distance": translation_distance,
        "thresholds": {
            "close_value": TRAVELUAV_DEPTH_CLOSE_VALUE,
            "close_fraction": TRAVELUAV_DEPTH_CLOSE_FRACTION,
            "tiny_diff": TRAVELUAV_DEPTH_TINY_DIFF,
            "stuck_distance": TRAVELUAV_DEPTH_STUCK_DISTANCE,
        },
    }


def _traveluav_episode_payload(observation: dict[str, Any]) -> dict[str, Any]:
    episode = observation.get("traveluav_episode") if isinstance(observation, dict) else None
    return episode if isinstance(episode, dict) else {}


def _translation_distance(pre_episode: dict[str, Any], post_episode: dict[str, Any]) -> float | None:
    pre_position = _episode_position(pre_episode)
    post_position = _episode_position(post_episode)
    if pre_position is None or post_position is None:
        return None
    return float(np.linalg.norm(pre_position - post_position))


def _episode_position(episode: dict[str, Any]) -> np.ndarray | None:
    state = episode.get("sensors", {}).get("state", {}) if isinstance(episode, dict) else {}
    position = state.get("position") if isinstance(state, dict) else None
    if position is None:
        return None
    return np.asarray(position, dtype=np.float32).reshape(-1)[:3]


def _traveluav_monitor_episode(state: StepState) -> list[dict[str, Any]]:
    episode_payloads = []
    for observation in state.history.observations:
        payload = observation.get("traveluav_episode") if isinstance(observation, dict) else None
        if isinstance(payload, dict):
            episode_payloads.append(payload)
    current = state.post_observation.get("traveluav_episode") if isinstance(state.post_observation, dict) else None
    if not isinstance(current, dict):
        raise ValueError("TravelUAV DINO stop requires post_observation['traveluav_episode']")
    episode_payloads.append(current)
    return episode_payloads


def _dino_object_query(episode: EvalEpisode, *, cache: dict[Path, dict[str, str]]) -> str:
    object_desc = _payload_text(episode.payload, "object_desc")
    if object_desc:
        return object_desc
    raise ValueError(f"TravelUAV episode {episode.episode_uid} is missing object_desc metadata for DINO")


def _episode_object_name(episode: EvalEpisode) -> str:
    object_name = _payload_text(episode.payload, "object_name")
    if object_name:
        return object_name
    object_info = _payload_value(episode.payload, "object")
    if isinstance(object_info, dict):
        return str(object_info.get("asset_name") or object_info.get("object_name") or "").strip()
    return ""


def _canonical_dino_object_name(object_name: str) -> str:
    if object_name.startswith("AASM_"):
        return "SM_" + object_name.removeprefix("AASM_")
    return object_name


def _direct_object_description_query(episode: EvalEpisode) -> str:
    value = _payload_value(episode.payload, "object_description")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
    return ""


def _object_description_mapping(path: Path, *, cache: dict[Path, dict[str, str]]) -> dict[str, str]:
    if path in cache:
        return cache[path]
    if not path.exists():
        raise FileNotFoundError(f"TravelUAV object_description.json does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"TravelUAV object_description.json must contain a list: {path}")
    mapping: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        object_name = str(item.get("object_name") or "").strip()
        object_desc = str(item.get("object_desc") or "").strip()
        if object_name and object_desc:
            mapping[object_name] = object_desc
    cache[path] = mapping
    return mapping
