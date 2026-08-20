from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from NavVLAeval.common.runner.backend_plan import WorkerBackendPlan
from NavVLAeval.common.config import EnvConfig
from NavVLAeval.common.simulators.airsim.actions import airsim_actions_to_world_waypoints
from NavVLAeval.common.simulators.airsim.observation import AirSimObservationBuilder
from NavVLAeval.common.simulators.airsim.process import (
    AirSimLaunchConfig,
    build_airsim_launch_command,
    build_airsim_launch_env,
    copytree_with_hardlinks,
    kill_pid,
    kill_process_group,
    pid_for_listening_port,
    resolve_airsim_start_script,
    resolve_binary_settings_path,
)
from NavVLAeval.common.simulators.airsim.rpc import patch_msgpackrpc_transport
from NavVLAeval.common.simulators.airsim.settings import write_airsim_settings
from NavVLAeval.common.simulators.base import (
    WaypointExecutionConfig,
    WaypointExecutionMode,
    WaypointExecutionResult,
    build_waypoint_execution_config,
    build_waypoint_execution_plan,
    pose_from_waypoint,
)
from NavVLAeval.common.types import EnvironmentStepResult, EvalEpisode, Pose4D


AIRSIM_RPC_TIMEOUT_SEC = 40
TELEPORT_POSE_TOLERANCE_M = 0.05

LEGACY_AIRSIM_ENV_KWARGS = {
    "camera_profile": "settings_profile/sensor_profile",
    "openfly_render_sync_frames": "teleport_render_sync_frames",
    "openfly_render_warmup_sec": "render_warmup_sec",
}


class AirSimBackendPlanner:
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
        base_airsim_port = self.kwargs.get("base_airsim_port", cfg.kwargs.get("base_airsim_port"))
        if base_airsim_port is None:
            raise ValueError("env.kwargs.base_airsim_port is required for AirSim backend planning")
        port = int(base_airsim_port) + worker_index
        return WorkerBackendPlan(
            type="airsim",
            kwargs={
                "airsim_port": port,
                "settings_root": store.settings_root(
                    worker_index=worker_index,
                    gpu_id=physical_gpu_id,
                    port=port,
                ),
            },
        )


class AirSimEnvironmentBackend:
    type = "airsim"

    def __init__(
        self,
        *,
        cfg: EnvConfig,
        worker_backend: WorkerBackendPlan,
        physical_gpu_id: int,
        start_process: bool = True,
    ) -> None:
        self.cfg = cfg
        self.kwargs = dict(cfg.kwargs)
        self._reject_legacy_kwargs()
        self.worker_backend = worker_backend
        self.env_root = Path(self._required_kwarg("env_root"))
        self.render_lib_root = Path(self._required_kwarg("render_lib_root"))
        self.settings_root = Path(self._required_worker_kwarg("settings_root"))
        self.airsim_port = int(self._required_worker_kwarg("airsim_port"))
        self.camera_name = str(self.kwargs.get("camera_name") or "front")
        self.physical_gpu_id = int(physical_gpu_id)
        self.ue_args = list(self.kwargs.get("ue_args") or ())
        self.startup_timeout = float(self.kwargs.get("startup_timeout") or 120.0)
        self.episode_startup_settle_sec = float(self.kwargs.get("episode_startup_settle_sec") or 0.0)
        if self.episode_startup_settle_sec < 0:
            raise ValueError("env.kwargs.episode_startup_settle_sec must be non-negative")
        self.render_warmup_sec = float(self.kwargs.get("render_warmup_sec") or 0.0)
        render_sync_frames = self.kwargs.get("teleport_render_sync_frames")
        if render_sync_frames is None:
            render_sync_frames = 3
        self.teleport_render_sync_frames = int(render_sync_frames)
        if self.teleport_render_sync_frames < 0:
            raise ValueError("env.kwargs.teleport_render_sync_frames must be non-negative")
        self.ignore_collision = bool(self.kwargs.get("ignore_collision", False))
        self.reset_ignore_collision = bool(self.kwargs.get("reset_ignore_collision", False))
        self.capture_action_observations = bool(self.kwargs.get("capture_action_observations", True))
        self.start_process = bool(start_process)
        self.settings_profile = str(self.kwargs.get("settings_profile") or "openfly")
        self.env_layout = str(self.kwargs.get("layout") or self.settings_profile or "openfly")
        self.recording_folder = Path(self.kwargs["recording_folder"]) if self.kwargs.get("recording_folder") is not None else None
        self.recording_camera_name = self.kwargs.get("recording_camera_name")
        self.recording_interval = self.kwargs.get("recording_interval")
        self.camera_resolution_overrides = dict(self.kwargs.get("camera_resolution_overrides") or {})
        self.external_camera_resolution_overrides = dict(self.kwargs.get("external_camera_resolution_overrides") or {})
        self.clock_speed = self.kwargs.get("clock_speed")
        self.view_mode = self.kwargs.get("view_mode")
        self.action_waypoint_semantics = str(
            self.kwargs.get("action_waypoint_semantics")
            or "anchor_relative_frd_xyz_yaw"
        )
        self.airsim_z_sign = float(self.kwargs.get("airsim_z_sign", 1.0))
        if self.airsim_z_sign not in {-1.0, 1.0}:
            raise ValueError("env.kwargs.airsim_z_sign must be either -1 or 1")
        self.action_execution_config = self._build_action_execution_config()
        self.observation_builder = AirSimObservationBuilder(
            profile=str(self.kwargs.get("sensor_profile") or self.settings_profile),
            camera_name=self.camera_name,
        )
        self.process: subprocess.Popen | None = None
        self.client: Any | None = None
        self.airsim: Any | None = None
        self.settings_path: Path | None = None
        self.current_env_name: str | None = None
        self.object_count = 0

    def start_episode(self, episode: EvalEpisode, initial_pose: Pose4D) -> dict[str, Any]:
        env_name = str(episode.payload.get("env_name") or "").strip()
        if not env_name:
            raise ValueError(f"AirSim episode {episode.episode_uid} is missing payload['env_name']")
        needs_render_warmup = self.current_env_name != env_name or self.client is None
        self.start(env_name)
        if needs_render_warmup and self.episode_startup_settle_sec > 0:
            time.sleep(self.episode_startup_settle_sec)
        if self.settings_profile == "openfly":
            if self.client is None:
                raise RuntimeError("AirSim client is not connected")
            self.client.simPause(True)
        self.reset_pose(initial_pose)
        if needs_render_warmup and self.render_warmup_sec > 0:
            self._wait_for_valid_render()
        return {"env_name": env_name}

    def start(self, env_name: str) -> None:
        if self.current_env_name == env_name and self.client is not None:
            return
        self.close()
        self.current_env_name = env_name
        command = self.prepare_launch(env_name)
        print(
            json.dumps(
                {
                    "type": "airsim_launch",
                    "env_name": env_name,
                    "physical_gpu_id": self.physical_gpu_id,
                    "airsim_port": self.airsim_port,
                    "settings_path": str(self.settings_path),
                    "command": command,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if self.start_process:
            self.process = subprocess.Popen(
                command,
                env=build_airsim_launch_env(self.render_lib_root, physical_gpu_id=self.physical_gpu_id),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,
            )
        self.connect()

    def prepare_launch(self, env_name: str) -> list[str]:
        start_script = resolve_airsim_start_script(self.env_root, env_name, layout=self.env_layout)
        if not start_script.exists():
            raise FileNotFoundError(f"missing AirSim start script: {start_script}")
        if self.env_layout == "openfly":
            env_dir = self.env_root / env_name
            worker_env_dir = self.settings_root / "env_copies" / env_name
            copytree_with_hardlinks(env_dir, worker_env_dir)
            start_script = worker_env_dir / "LinuxNoEditor" / "start.sh"
            settings_path = resolve_binary_settings_path(worker_env_dir)
        else:
            settings_path = self.settings_root / "settings" / f"{env_name}_{self.airsim_port}.json"
        write_airsim_settings(
            settings_path,
            api_server_port=self.airsim_port,
            profile=self.settings_profile,
            recording_folder=self.recording_folder,
            recording_camera_name=self.recording_camera_name,
            recording_interval=self.recording_interval,
            camera_resolution_overrides=self.camera_resolution_overrides,
            external_camera_resolution_overrides=self.external_camera_resolution_overrides,
            clock_speed=self.clock_speed,
            view_mode=self.view_mode,
        )
        self.settings_path = settings_path
        return build_airsim_launch_command(
            AirSimLaunchConfig(
                start_script=start_script,
                physical_gpu_id=self.physical_gpu_id,
                settings_path=settings_path,
                ue_args=self.ue_args,
                settings_argument_style="space" if self.env_layout == "aerialvln" else "equals",
            )
        )

    def connect(self) -> None:
        patch_msgpackrpc_transport()
        import airsim

        self.airsim = airsim
        deadline = time.monotonic() + self.startup_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            client = airsim.MultirotorClient(port=self.airsim_port, timeout_value=AIRSIM_RPC_TIMEOUT_SEC)
            try:
                client.confirmConnection()
                if self.settings_profile == "aerialvln":
                    client.enableApiControl(True, vehicle_name="Drone_1")
                    client.armDisarm(True, vehicle_name="Drone_1")
                else:
                    client.enableApiControl(True)
                    client.armDisarm(True)
                self.client = client
                return
            except Exception as exc:
                last_error = exc
                time.sleep(2.0)
        raise RuntimeError(f"AirSim did not accept RPC connections within {self.startup_timeout:.1f}s") from last_error

    def get_observation(self) -> dict[str, Any]:
        client, airsim = self._require_client()
        return self.observation_builder.build(client=client, airsim=airsim)

    def _wait_for_valid_render(self) -> None:
        deadline = time.monotonic() + self.render_warmup_sec
        while True:
            image = np.asarray(self.get_observation().get("image"))
            if image.size > 0 and np.any(image):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"AirSim renderer remained blank for {self.render_warmup_sec:.1f}s"
                )
            time.sleep(min(1.0, remaining))

    def apply_action(self, current_pose: Pose4D, raw_actions: np.ndarray) -> EnvironmentStepResult:
        plan = self._build_waypoint_plan(current_pose=current_pose, raw_actions=raw_actions)
        result = self._execute_waypoint_plan(plan)
        self.observation_builder.last_movement_result = result.as_diagnostics()
        observation = self.get_observation()
        diagnostics = plan.diagnostics()
        diagnostics.update(result.as_diagnostics())
        return EnvironmentStepResult(
            next_pose=result.next_pose,
            observation=observation,
            data_done=False,
            diagnostics=diagnostics,
            action_observations=result.action_observations,
        )

    def project_action_to_world(self, current_pose: Pose4D, raw_actions: np.ndarray) -> np.ndarray:
        return self._project_action_to_world_waypoints(current_pose=current_pose, raw_actions=raw_actions)

    def reset_pose(self, pose: Pose4D) -> None:
        self.observation_builder.last_movement_result = {"collision": False, "collision_reason": None}
        # Some simulator worlds start at a default spawn that intersects map
        # geometry.  The explicit reset policy applies only to this controlled
        # placement; waypoint execution still follows ``ignore_collision``.
        self._set_pose(
            pose,
            synchronize_render=True,
            ignore_collision=self.reset_ignore_collision,
        )

    def set_pose(self, pose: Pose4D) -> None:
        self._set_pose(
            pose,
            synchronize_render=True,
            ignore_collision=self.ignore_collision,
        )

    def _set_pose(
        self,
        pose: Pose4D,
        *,
        synchronize_render: bool,
        ignore_collision: bool = False,
    ) -> None:
        client, airsim = self._require_client()
        airsim_pose = self._pose_to_airsim_coordinates(pose)
        target_pose = airsim.Pose(
            airsim.Vector3r(float(airsim_pose.x), float(airsim_pose.y), float(airsim_pose.z)),
            airsim.to_quaternion(0, 0, float(airsim_pose.yaw)),
        )
        if self.settings_profile == "traveluav":
            client.simPause(True)
            for _ in range(self.teleport_render_sync_frames if synchronize_render else 0):
                client.simSetKinematics(state=target_pose, ignore_collision=ignore_collision)
                client.simContinueForFrames(1)
                client.simPause(True)
            target_state = airsim.KinematicsState()
            target_state.position = airsim.Vector3r(float(airsim_pose.x), float(airsim_pose.y), float(airsim_pose.z))
            target_state.orientation = airsim.to_quaternion(0, 0, float(airsim_pose.yaw))
            target_state.linear_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
            target_state.angular_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
            target_state.linear_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
            target_state.angular_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
            client.simSetKinematics(state=target_state, ignore_collision=ignore_collision)
            client.simPause(True)
            return
        if synchronize_render and self.teleport_render_sync_frames > 0:
            client.simPause(True)
        if self.settings_profile == "aerialvln":
            client.simSetVehiclePose(target_pose, ignore_collision, vehicle_name="Drone_1")
        else:
            client.simSetVehiclePose(target_pose, ignore_collision)
        if synchronize_render:
            self._synchronize_renderer_after_pose_update()

    def _synchronize_renderer_after_pose_update(self) -> None:
        if self.teleport_render_sync_frames <= 0:
            return
        client, _airsim = self._require_client()
        client.simContinueForFrames(self.teleport_render_sync_frames)
        client.simPause(True)

    def set_object(self, object_info: dict[str, Any]) -> bool:
        client, airsim = self._require_client()
        asset_name = str(object_info["asset_name"])
        scale_values = object_info.get("scale", [1.0, 1.0, 1.0])
        pose_values = object_info.get("pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        pose = _object_pose_from_values(airsim, pose_values)
        scale = airsim.Vector3r(float(scale_values[0]), float(scale_values[1]), float(scale_values[2]))
        if self.object_count > 0:
            client.simDestroyObject(f"my_object_{self.object_count - 1}")
        success = client.simSpawnObject(
            f"my_object_{self.object_count}",
            asset_name,
            pose,
            scale,
            physics_enabled=False,
            is_blueprint=False,
        )
        self.object_count += 1
        self.observation_builder.target_position = np.asarray(pose_values[:3], dtype=np.float32)
        client.simContinueForFrames(1)
        client.simPause(True)
        return bool(success)

    def close_episode(self) -> None:
        return None

    def close(self) -> None:
        if self.process:
            kill_process_group(self.process.pid)
        if self.start_process:
            kill_pid(pid_for_listening_port(self.airsim_port))
        self.process = None
        self.client = None
        self.airsim = None

    def _build_waypoint_plan(self, *, current_pose: Pose4D, raw_actions: np.ndarray):
        world_waypoints = self._project_action_to_world_waypoints(
            current_pose=current_pose,
            raw_actions=raw_actions,
        )
        return build_waypoint_execution_plan(
            world_waypoints=world_waypoints,
            config=self.action_execution_config,
        )

    def _project_action_to_world_waypoints(self, *, current_pose: Pose4D, raw_actions: np.ndarray) -> np.ndarray:
        return airsim_actions_to_world_waypoints(
            current_pose=current_pose,
            raw_actions=raw_actions,
            action_semantics=self.action_waypoint_semantics,
        )

    def _pose_to_airsim_coordinates(self, pose: Pose4D) -> Pose4D:
        return Pose4D(pose.x, pose.y, pose.z * self.airsim_z_sign, pose.yaw)

    def _pose_from_airsim_coordinates(self, pose: Pose4D) -> Pose4D:
        return Pose4D(pose.x, pose.y, pose.z * self.airsim_z_sign, pose.yaw)

    def _execute_waypoint_plan(self, plan) -> WaypointExecutionResult:
        waypoints = plan.executed_waypoints
        if waypoints.shape[0] == 0:
            raise ValueError("waypoint execution plan has no selected waypoints")
        if plan.mode == WaypointExecutionMode.TELEPORT_FINAL:
            return self._teleport_to_final_waypoint(plan)
        if plan.mode == WaypointExecutionMode.TELEPORT_EACH_WAYPOINT:
            return self._teleport_each_waypoint(plan)
        if plan.mode == WaypointExecutionMode.PATH:
            return self._move_path_by_waypoints(plan)
        raise ValueError(f"Unsupported AirSim waypoint execution mode: {plan.mode}")

    def _teleport_to_final_waypoint(self, plan) -> WaypointExecutionResult:
        final_pose = pose_from_waypoint(plan.executed_waypoints[-1])
        self.set_pose(final_pose)
        actual_pose = self._actual_vehicle_pose()
        reached_target = _positions_match(actual_pose, final_pose)
        return WaypointExecutionResult(
            next_pose=actual_pose,
            original_waypoint_count=plan.original_waypoint_count,
            executed_waypoint_count=plan.executed_waypoint_count,
            selected_waypoint_indices=plan.selected_waypoint_indices,
            completed_waypoint_count=plan.executed_waypoint_count if reached_target else 0,
            attempted_waypoint_count=plan.executed_waypoint_count,
            collision=not reached_target,
            collision_reason=None if reached_target else "pose_update_blocked",
            diagnostics={"execution_mode": plan.mode.value, "actual_pose": actual_pose.as_array().tolist()},
        )

    def _teleport_each_waypoint(self, plan) -> WaypointExecutionResult:
        action_observations = []
        actual_waypoint_poses = []
        pose_mismatches = []
        current_pose = self._actual_vehicle_pose()
        completed = 0
        for waypoint_index, waypoint in enumerate(plan.executed_waypoints):
            target_pose = pose_from_waypoint(waypoint)
            should_capture = self.capture_action_observations
            self._set_pose(
                target_pose,
                synchronize_render=True,
                ignore_collision=self.ignore_collision,
            )
            current_pose = self._actual_vehicle_pose()
            actual_waypoint_poses.append(current_pose)
            if not _positions_match(current_pose, target_pose):
                pose_mismatches.append(
                    {
                        "waypoint_index": int(waypoint_index),
                        "target_pose": target_pose.as_array().tolist(),
                        "actual_pose": current_pose.as_array().tolist(),
                        "position_error_m": float(
                            np.linalg.norm(current_pose.as_array()[:3] - target_pose.as_array()[:3])
                        ),
                        "z_error_m": float(current_pose.z - target_pose.z),
                    }
                )
            else:
                completed += 1
            if should_capture:
                action_observations.append(self.get_observation())
        collision = bool(pose_mismatches) and not self.ignore_collision
        return WaypointExecutionResult(
            next_pose=current_pose,
            original_waypoint_count=plan.original_waypoint_count,
            executed_waypoint_count=plan.executed_waypoint_count,
            selected_waypoint_indices=plan.selected_waypoint_indices,
            completed_waypoint_count=completed,
            attempted_waypoint_count=len(actual_waypoint_poses),
            collision=collision,
            collision_reason="pose_update_blocked" if collision else None,
            action_observations=action_observations,
            actual_waypoint_poses=actual_waypoint_poses,
            pose_mismatches=pose_mismatches,
            diagnostics={"execution_mode": plan.mode.value, "actual_pose": current_pose.as_array().tolist()},
        )

    def _actual_vehicle_pose(self) -> Pose4D:
        client, _airsim = self._require_client()
        if self.settings_profile == "aerialvln":
            airsim_pose = client.simGetVehiclePose(vehicle_name="Drone_1")
        else:
            airsim_pose = client.simGetVehiclePose()
        return self._pose_from_airsim_coordinates(_pose4d_from_airsim_pose(airsim_pose))

    def _move_path_by_waypoints(self, plan) -> WaypointExecutionResult:
        client, airsim = self._require_client()
        collision = False
        collision_reason = None
        completed = 0
        action_observations = []
        client.enableApiControl(True)
        client.armDisarm(True)
        client.simPause(False)
        for waypoint_index, waypoint in enumerate(plan.executed_waypoints):
            yaw_deg = float(np.degrees(float(waypoint[3])))
            airsim_waypoint = self._pose_to_airsim_coordinates(pose_from_waypoint(waypoint))
            client.moveToPositionAsync(
                float(airsim_waypoint.x),
                float(airsim_waypoint.y),
                float(airsim_waypoint.z),
                velocity=1,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg),
                lookahead=3,
                adaptive_lookahead=1,
            )
            if not _wait_until_waypoint_reached(
                client,
                np.asarray([airsim_waypoint.x, airsim_waypoint.y, airsim_waypoint.z], dtype=np.float32),
            ):
                collision = True
                collision_reason = "stuck max len"
                break
            completed = waypoint_index + 1
            action_observations.append(self.get_observation())
        client.simPause(True)
        next_pose = self._pose_from_airsim_coordinates(_actual_pose_from_multirotor_state(client))
        return WaypointExecutionResult(
            next_pose=next_pose,
            original_waypoint_count=plan.original_waypoint_count,
            executed_waypoint_count=plan.executed_waypoint_count,
            selected_waypoint_indices=plan.selected_waypoint_indices,
            completed_waypoint_count=completed,
            collision=collision,
            collision_reason=collision_reason,
            action_observations=action_observations,
            diagnostics={"execution_mode": plan.mode.value},
        )

    def _build_action_execution_config(self) -> WaypointExecutionConfig:
        return build_waypoint_execution_config(
            mode=self.kwargs.get("action_execution_mode"),
        )

    def _reject_legacy_kwargs(self) -> None:
        for legacy_key, canonical_key in LEGACY_AIRSIM_ENV_KWARGS.items():
            if legacy_key in self.kwargs:
                raise ValueError(
                    f"env.kwargs.{legacy_key} is not supported; use env.kwargs.{canonical_key}"
                )

    def _required_kwarg(self, key: str) -> Any:
        value = self.kwargs.get(key)
        if value is None:
            raise ValueError(f"env.kwargs.{key} is required for AirSim backend")
        return value

    def _required_worker_kwarg(self, key: str) -> Any:
        value = self.worker_backend.kwargs.get(key)
        if value is None:
            raise ValueError(f"AirSim worker backend missing kwargs[{key!r}]")
        return value

    def _require_client(self) -> tuple[Any, Any]:
        if self.client is None or self.airsim is None:
            raise RuntimeError("AirSim client is not connected")
        return self.client, self.airsim


def _object_pose_from_values(airsim: Any, pose_values: Any):
    values = list(pose_values)
    position = airsim.Vector3r(float(values[0]), float(values[1]), float(values[2]))
    if len(values) >= 7:
        orientation = airsim.Quaternionr(float(values[3]), float(values[4]), float(values[5]), float(values[6]))
    else:
        orientation = airsim.to_quaternion(0, 0, 0)
    return airsim.Pose(position, orientation)


def _wait_until_waypoint_reached(
    client: Any,
    target_xyz: np.ndarray,
    *,
    timeout_sec: float = 5.0,
    distance_tolerance: float = 0.5,
    stuck_window: int = 200,
    stuck_distance: float = 0.1,
) -> bool:
    target = np.asarray(target_xyz, dtype=np.float32).reshape(3)
    position_queue: list[np.ndarray] = []
    previous_distance = float("inf")
    start_time = time.perf_counter()
    while True:
        if time.perf_counter() - start_time > timeout_sec:
            return False
        state = client.getMultirotorState(vehicle_name="")
        position = np.asarray(list(state.kinematics_estimated.position), dtype=np.float32).reshape(3)
        position_queue.append(position)
        if len(position_queue) > stuck_window:
            historical_position = position_queue.pop(0)
            if float(np.linalg.norm(position - historical_position)) < stuck_distance:
                return False
        distance = float(np.linalg.norm(position - target))
        if distance <= distance_tolerance or distance > previous_distance:
            return True
        previous_distance = distance
        time.sleep(0.005)


def _actual_pose_from_multirotor_state(client: Any) -> Pose4D:
    state = client.getMultirotorState(vehicle_name="")
    position = np.asarray(list(state.kinematics_estimated.position), dtype=np.float32).reshape(3)
    orientation = state.kinematics_estimated.orientation
    yaw = _yaw_from_airsim_quaternion(orientation)
    return Pose4D(float(position[0]), float(position[1]), float(position[2]), yaw)


def _pose4d_from_airsim_pose(airsim_pose: Any) -> Pose4D:
    position = airsim_pose.position
    return Pose4D(
        float(getattr(position, "x_val", 0.0)),
        float(getattr(position, "y_val", 0.0)),
        float(getattr(position, "z_val", 0.0)),
        _yaw_from_airsim_quaternion(airsim_pose.orientation),
    )


def _positions_match(actual_pose: Pose4D, target_pose: Pose4D) -> bool:
    return bool(
        np.linalg.norm(actual_pose.as_array()[:3] - target_pose.as_array()[:3])
        <= TELEPORT_POSE_TOLERANCE_M
    )


def _yaw_from_airsim_quaternion(orientation: Any) -> float:
    x = float(getattr(orientation, "x_val", 0.0))
    y = float(getattr(orientation, "y_val", 0.0))
    z = float(getattr(orientation, "z_val", 0.0))
    w = float(getattr(orientation, "w_val", 1.0))
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
