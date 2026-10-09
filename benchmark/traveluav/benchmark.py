"""TravelUAV task rules over normalized episodes and AirSim observations."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np

from benchmark.common.runtime_defaults import BaseBenchmarkRuntime
from benchmark.common.types import Pose4D, TerminationStatus

TRAVELUAV_RGB_FOLDERS = ["frontcamera", "leftcamera", "rightcamera", "rearcamera", "downcamera"]
TRAVELUAV_DEPTH_CLOSE_VALUE = 1.0
TRAVELUAV_DEPTH_CLOSE_FRACTION = 0.1
TRAVELUAV_DEPTH_TINY_DIFF = 3.0
TRAVELUAV_DEPTH_STUCK_DISTANCE = 0.1


class TravelUAVBenchmark(BaseBenchmarkRuntime):
    def __init__(
        self,
        *,
        stop_policy="none",
        success_radius=20.0,
        use_gt=False,
        depth_collision_policy="stop",
        ignore_movement_collision=False,
        groundingdino_config=None,
        groundingdino_model_path=None,
        dino_device="cuda",
    ):
        if stop_policy not in {"none", "dino"}:
            raise ValueError(f"Unsupported TravelUAV stop_policy: {stop_policy}")
        if depth_collision_policy not in {"stop", "log_only"}:
            raise ValueError(f"Unsupported TravelUAV depth_collision_policy: {depth_collision_policy}")
        self.stop_policy = stop_policy
        self.success_radius = float(success_radius)
        self.use_gt = use_gt
        self.depth_collision_policy = depth_collision_policy
        self.ignore_movement_collision = ignore_movement_collision
        self.groundingdino_config = groundingdino_config
        self.groundingdino_model_path = groundingdino_model_path
        self.dino_device = dino_device
        self._dino_monitor = None
        self._distances = deque(maxlen=10)

    def validate_episode(self, episode):
        if self.stop_policy == "dino" and not episode.payload["object_desc"]:
            raise ValueError(f"TravelUAV DINO requires object_desc for {episode.episode_uid}")

    def initial_pose(self, episode):
        return Pose4D(*episode.payload["start_pose"])

    def prepare_environment(self, episode, env, initial_pose):
        self._distances.clear()
        object_info = episode.payload["object"]
        if object_info is not None and not env.set_object(object_info):
            raise RuntimeError(f"TravelUAV failed to place object for {episode.episode_uid}")
        env.reset_pose(initial_pose)

    def instruction_for_step(self, episode, history, step):
        return episode.instruction

    def prepare_observation_for_model(self, *, episode, history, step, observation, instruction):
        if self.use_gt:
            sensors = observation["traveluav_episode"]["sensors"]
            position = np.asarray(sensors["state"]["position"])
            points = np.asarray(episode.payload["trajectory"])
            distances = np.linalg.norm(points[:, :2] - position[:2], axis=1)
            nearest = int(distances.argmin())
            ahead = np.flatnonzero((distances[nearest:] > 6.0) | np.all(points[nearest:] == points[-1], axis=1))
            target = points[nearest + ahead[0]]
            stage = _traveluav_body_frame_stage(
                current_position=position,
                target_position=target,
                rotation=sensors["imu"]["rotation"],
                final_position=points[-1],
            )
            description = episode.payload["object_description"]
            instruction = f"Stage: {stage}\n\nInstruction: Fly {stage} and find the target. {description}".rstrip()
        return {"instruction": instruction}

    def goal_position(self, episode):
        return np.asarray(episode.payload["goal_position"])

    def distance_to_goal(self, pose, episode):
        return float(np.linalg.norm(pose.as_array()[:3] - self.goal_position(episode)))

    def gt_path_length(self, episode):
        return episode.payload["gt_path_length"]

    def is_success(self, pose, episode):
        return self.distance_to_goal(pose, episode) < self.success_radius

    def update_termination(self, state):
        distance = state.distance_after
        self._distances.append(distance)
        movement = state.post_observation["traveluav_episode"]["sensors"]["state"]["movement"]
        depth = _depth_collision_payload(state.pre_observation, state.post_observation)
        diagnostics = {
            "distance": distance,
            "movement_collision": movement["collision"],
            "movement_collision_reason": movement["collision_reason"],
            "depth_collision": depth,
        }
        reason = "running"
        if self.depth_collision_policy == "stop":
            if self.ignore_movement_collision and depth["collision"]:
                reason = f"collision:depth:{depth['reason']}"
            elif not self.ignore_movement_collision and movement["collision"]:
                suffix = f":{movement['collision_reason']}" if movement["collision_reason"] else ""
                reason = f"collision:movement{suffix}"
        if reason == "running" and len(self._distances) == 10:
            if np.all(np.diff(self._distances) > 0):
                reason = "distance_increasing_10frames"
        if reason == "running" and self.stop_policy == "dino":
            stopped = self._get_dino_monitor().get_dino_results(
                state.post_observation["traveluav_episode"], state.episode.payload["object_desc"]
            )
            diagnostics["dino"] = {"stop": bool(stopped)}
            if stopped:
                reason = "dino_stop"
        reached = distance < self.success_radius
        # Worker has already checked every actual pose for OSR; do not rescore predicted waypoints.
        return TerminationStatus(
            reason != "running",
            int(reason == "dino_stop" and reached),
            int(reached),
            reason,
            None,
            None,
            diagnostics,
        )

    def log_step_artifacts(self, state, artifacts):
        return state.termination.diagnostics

    def _get_dino_monitor(self):
        if self._dino_monitor is None:
            from benchmark.traveluav.dino_monitor import TravelUAVDinoMonitor

            self._dino_monitor = TravelUAVDinoMonitor(
                groundingdino_config=Path(self.groundingdino_config),
                groundingdino_model_path=Path(self.groundingdino_model_path),
                device=self.dino_device,
            )
        return self._dino_monitor


def _traveluav_body_frame_stage(
    *,
    current_position: np.ndarray,
    target_position: np.ndarray,
    rotation: np.ndarray,
    final_position: np.ndarray | None = None,
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


def _depth_collision_payload(pre_observation, post_observation):
    pre = pre_observation["traveluav_episode"]
    post = post_observation["traveluav_episode"]
    per_camera, close_cameras, diffs = [], [], []
    for camera, before, after in zip(TRAVELUAV_RGB_FOLDERS, pre["depth"], post["depth"], strict=True):
        before, after = np.asarray(before, dtype=np.float32), np.asarray(after, dtype=np.float32)
        if before.shape != after.shape or not after.size:
            raise ValueError(f"TravelUAV depth shape changed or is empty for {camera}")
        close_ratio = float(np.mean(after <= TRAVELUAV_DEPTH_CLOSE_VALUE))
        close = close_ratio > TRAVELUAV_DEPTH_CLOSE_FRACTION
        diff = float(np.mean(np.abs(before - after)))
        diffs.append(diff)
        if close:
            close_cameras.append(camera)
        per_camera.append(
            {"camera": camera, "close": close, "close_pixel_ratio": close_ratio, "mean_abs_depth_diff": diff}
        )
    translation = float(np.linalg.norm(post_observation["pose"].as_array()[:3] - pre_observation["pose"].as_array()[:3]))
    reason = None
    if np.all(np.asarray(diffs) < TRAVELUAV_DEPTH_TINY_DIFF):
        reason = "tiny diff"
    elif close_cameras:
        reason = "close"
    elif translation < TRAVELUAV_DEPTH_STUCK_DISTANCE:
        reason = "distance"
    return {
        "collision": reason is not None,
        "reason": reason,
        "close_cameras": close_cameras,
        "per_camera": per_camera,
        "translation_distance": translation,
        "thresholds": {
            "close_value": TRAVELUAV_DEPTH_CLOSE_VALUE,
            "close_fraction": TRAVELUAV_DEPTH_CLOSE_FRACTION,
            "tiny_diff": TRAVELUAV_DEPTH_TINY_DIFF,
            "stuck_distance": TRAVELUAV_DEPTH_STUCK_DISTANCE,
        },
    }
