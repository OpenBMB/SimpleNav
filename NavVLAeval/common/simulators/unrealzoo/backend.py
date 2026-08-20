from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import time
from types import MethodType
from typing import Any

import numpy as np

from NavVLAeval.common.runner.backend_plan import WorkerBackendPlan
from NavVLAeval.common.config import EnvConfig
from NavVLAeval.common.simulators.base import select_waypoints_for_step
from NavVLAeval.common.simulators.unrealzoo.coordinates import (
    nav_waypoints_to_unreal_cm,
    nav_pose_from_unreal_cm,
    starvla_waypoints_to_nav,
    starvla_waypoints_to_unreal_cm,
    unreal_pose_from_nav,
)
from NavVLAeval.common.types import EnvironmentStepResult, EvalEpisode, Pose4D


class UnrealZooBackendPlanner:
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
        del store
        if worker_index != 0:
            raise ValueError("UnrealZoo backend currently supports single-card single-worker evaluation only")
        merged = dict(cfg.kwargs)
        merged.update(self.kwargs)
        env_id = _required_str(merged, "env_id")
        unreal_env_root = _required_str(merged, "unreal_env_root")
        return WorkerBackendPlan(
            type="unrealzoo",
            kwargs={
                "env_id": env_id,
                "unreal_env_root": unreal_env_root,
                "physical_gpu_id": int(physical_gpu_id),
                "worker_index": int(worker_index),
            },
        )


class UnrealZooEnvironmentBackend:
    type = "unrealzoo"

    def __init__(
        self,
        *,
        cfg: EnvConfig,
        worker_backend: WorkerBackendPlan,
        physical_gpu_id: int,
        start_process: bool = True,
    ) -> None:
        self.cfg = cfg
        self.worker_backend = worker_backend
        self.physical_gpu_id = int(physical_gpu_id)
        self.start_process = bool(start_process)
        self.kwargs = dict(cfg.kwargs)
        self.env_id = str(worker_backend.kwargs.get("env_id") or self.kwargs.get("env_id") or "")
        unreal_env_root = worker_backend.kwargs.get("unreal_env_root") or self.kwargs.get("unreal_env_root")
        unrealzoo_gym_root = self.kwargs.get("unrealzoo_gym_root")
        self.unreal_env_root = Path(str(unreal_env_root)) if unreal_env_root else None
        self.unrealzoo_gym_root = Path(str(unrealzoo_gym_root)) if unrealzoo_gym_root else None
        self.render_lib_root = Path(str(self.kwargs["render_lib_root"])) if self.kwargs.get("render_lib_root") is not None else None
        self.resolution = tuple(int(v) for v in self.kwargs.get("resolution", (256, 256)))
        self.display = self.kwargs.get("display")
        self.offscreen = bool(self.kwargs.get("offscreen", True))
        self.sleep_time = int(self.kwargs.get("sleep_time", 30))
        self.post_connect_warmup_sec = float(self.kwargs.get("post_connect_warmup_sec", 0.0))
        self.connect_retries = int(self.kwargs.get("connect_retries", 6))
        self.connect_retry_interval = float(self.kwargs.get("connect_retry_interval", 5.0))
        self.configure_ue_after_connect = bool(self.kwargs.get("configure_ue_after_connect", False))
        self.agent_categories = [str(v) for v in self.kwargs.get("agent_categories", ["player"])]
        self.population_min = int(self.kwargs.get("population_min", 2))
        self.population_max = int(self.kwargs.get("population_max", self.population_min))
        self.random_target = bool(self.kwargs.get("random_target", False))
        self.random_tracker = bool(self.kwargs.get("random_tracker", False))
        self.camera_id = int(self.kwargs.get("camera_id", 0))
        self.viewmode = str(self.kwargs.get("viewmode", "lit"))
        self.player_name: str | None = None
        self.env = None
        self._pose = Pose4D(0.0, 0.0, 0.0, 0.0)
        self._trajectory: list[dict[str, Any]] = []
        if not self.env_id:
            raise ValueError("env.kwargs.env_id is required for UnrealZoo backend")
        if self.unreal_env_root is None:
            raise ValueError("env.kwargs.unreal_env_root is required for UnrealZoo backend")
        if self.unrealzoo_gym_root is None:
            raise ValueError("env.kwargs.unrealzoo_gym_root is required for UnrealZoo backend")

    def start_episode(self, episode: EvalEpisode, initial_pose: Pose4D) -> dict[str, Any]:
        self._ensure_env()
        self._pose = initial_pose
        self._trajectory = []
        self.reset_pose(initial_pose)
        if self.post_connect_warmup_sec > 0.0:
            time.sleep(self.post_connect_warmup_sec)
        return {"env_id": self.env_id, "scene_id": episode.scene_id}

    def get_observation(self) -> dict[str, Any]:
        env = self._require_env()
        image = env.unwrapped.unrealcv.get_image(self.camera_id, self.viewmode)
        state = self._pose.as_array()
        return {
            "image": np.asarray(image),
            "state": state,
            "sim_pose_cm": unreal_pose_from_nav(self._pose),
            "uavflow_pose": self._uavflow_pose_payload(image=np.asarray(image)),
        }

    def apply_action(self, current_pose: Pose4D, raw_actions: np.ndarray) -> EnvironmentStepResult:
        env = self._require_env()
        original_nav_waypoints = starvla_waypoints_to_nav(current_pose, raw_actions)
        nav_waypoints, selected_indices = select_waypoints_for_step(
            original_nav_waypoints,
            execute_waypoints_per_step=None,
        )
        unreal_waypoints = nav_waypoints_to_unreal_cm(nav_waypoints)
        action_observations: list[dict[str, Any]] = []
        for waypoint, nav_waypoint in zip(unreal_waypoints, nav_waypoints):
            x_cm, y_cm, z_cm, yaw_deg = [float(v) for v in waypoint]
            env.unwrapped.unrealcv.set_obj_location(self._player(), [x_cm, y_cm, z_cm])
            env.unwrapped.unrealcv.set_rotation(self._player(), yaw_deg - 180.0)
            self._pose = Pose4D(float(nav_waypoint[0]), float(nav_waypoint[1]), float(nav_waypoint[2]), float(nav_waypoint[3]))
            self._set_camera()
            action_observations.append(self.get_observation())
            self._trajectory.append(
                {
                    "state_m": [[x_cm / 100.0, y_cm / 100.0, -z_cm / 100.0], [0.0, yaw_deg, 0.0]],
                    "state_cm": [[x_cm, y_cm, z_cm], [0.0, yaw_deg, 0.0]],
                }
            )
        last = nav_waypoints[-1]
        self._pose = Pose4D(float(last[0]), float(last[1]), float(last[2]), float(last[3]))
        diagnostics = {
            "original_waypoint_count": int(original_nav_waypoints.shape[0]),
            "executed_waypoint_count": int(nav_waypoints.shape[0]),
            "selected_waypoint_indices": selected_indices,
            "world_waypoints": original_nav_waypoints.tolist(),
            "executed_world_waypoints": nav_waypoints.tolist(),
            "unreal_waypoints_cm": unreal_waypoints.tolist(),
        }
        return EnvironmentStepResult(
            next_pose=self._pose,
            observation=action_observations[-1] if action_observations else self.get_observation(),
            data_done=False,
            diagnostics=diagnostics,
            action_observations=action_observations,
        )

    def project_action_to_world(self, current_pose: Pose4D, raw_actions: np.ndarray) -> np.ndarray:
        return starvla_waypoints_to_nav(current_pose, raw_actions)

    def reset_pose(self, pose: Pose4D) -> None:
        env = self._require_env()
        x_cm, y_cm, z_cm, _roll, yaw_deg, _pitch = unreal_pose_from_nav(pose)
        env.unwrapped.unrealcv.set_obj_location(self._player(), [x_cm, y_cm, z_cm])
        env.unwrapped.unrealcv.set_rotation(self._player(), yaw_deg - 180.0)
        self._pose = pose
        self._set_camera()

    def set_object(self, object_info: dict[str, Any]) -> bool:
        env = self._require_env()
        obj_id = object_info.get("obj_id")
        use_obj = object_info.get("use_obj")
        obj_pos = object_info.get("obj_pos")
        obj_rot = object_info.get("obj_rot", [0, 0, 0])
        if obj_id is None or use_obj is None or obj_pos is None:
            return False
        if hasattr(env.unwrapped.unrealcv, "new_obj_fromPath"):
            env.unwrapped.unrealcv.new_obj_fromPath(str(use_obj), str(obj_id), obj_pos, obj_rot)
            return True
        return False

    def close_episode(self) -> None:
        return None

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def trajectory_log(self) -> list[dict[str, Any]]:
        return list(self._trajectory)

    def _ensure_env(self) -> None:
        if self.env is not None:
            return
        if not self.start_process:
            return
        self._prepare_imports()
        self._prepare_render_environment()
        gym = importlib.import_module("gym")
        importlib.import_module("gym_unrealcv")
        config_ue = importlib.import_module("gym_unrealcv.envs.wrappers.configUE")
        os.environ["UnrealEnv"] = str(self.unreal_env_root)
        env = gym.make(self.env_id)
        env = config_ue.ConfigUEWrapper(
            env,
            resolution=self.resolution,
            display=self.display,
            offscreen=self.offscreen,
            gpu_id=self.physical_gpu_id,
            sleep_time=self.sleep_time,
        )
        augmentation = importlib.import_module("gym_unrealcv.envs.wrappers.augmentation")
        env = augmentation.RandomPopulationWrapper(
            env,
            self.population_min,
            self.population_max,
            random_target=self.random_target,
            random_tracker=self.random_tracker,
        )
        self._patch_unrealzoo_launch(env.unwrapped)
        env.unwrapped.agents_category = list(self.agent_categories)
        env.reset()
        self.env = env
        env.unwrapped.unrealcv.set_viewport(self._player())
        env.unwrapped.unrealcv.set_phy(self._player(), 0)

    def _prepare_render_environment(self) -> None:
        if self.render_lib_root is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.physical_gpu_id)
            return
        required = [
            self.render_lib_root / "lib",
            self.render_lib_root / "etc" / "nvidia_icd.json",
            self.render_lib_root / "etc" / "10_nvidia.json",
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"missing UnrealZoo render dependency: {path}")
        current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        lib_path = str(self.render_lib_root / "lib")
        if current_ld_library_path:
            entries = current_ld_library_path.split(":")
            if lib_path not in entries:
                os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld_library_path}"
        else:
            os.environ["LD_LIBRARY_PATH"] = lib_path
        os.environ["VK_DRIVER_FILES"] = str(self.render_lib_root / "etc" / "nvidia_icd.json")
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(self.render_lib_root / "etc" / "10_nvidia.json")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.physical_gpu_id)

    def _patch_unrealzoo_launch(self, unwrapped_env) -> None:
        def launch_ue_env(env_self):
            env_ip, env_port = env_self.ue_binary.start(
                docker=env_self.docker,
                resolution=env_self.resolution,
                display=env_self.display,
                opengl=env_self.use_opengl,
                offscreen=env_self.offscreen_rendering,
                nullrhi=env_self.nullrhi,
                gpu_id=env_self.gpu_id,
                sleep_time=env_self.sleep_time,
            )
            last_error: Exception | None = None
            character_api = importlib.import_module("gym_unrealcv.envs.agent.character").Character_API
            for attempt in range(max(1, self.connect_retries)):
                try:
                    env_self.unrealcv = character_api(
                        port=env_port,
                        ip=env_ip,
                        resolution=env_self.resolution,
                        comm_mode=env_self.comm_mode,
                    )
                    if self.configure_ue_after_connect:
                        env_self.unrealcv.set_map(env_self.env_name)
                        env_self.unrealcv.config_ue(quality=env_self.render_quality, Lumen=env_self.use_lumen)
                    return True
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 >= max(1, self.connect_retries):
                        break
                    time.sleep(self.connect_retry_interval)
            raise RuntimeError(f"failed to connect UnrealCV server after launch: {last_error}") from last_error

        unwrapped_env.launch_ue_env = MethodType(launch_ue_env, unwrapped_env)

    def _prepare_imports(self) -> None:
        root = str(self.unrealzoo_gym_root)
        if root not in sys.path:
            sys.path.insert(0, root)

    def _require_env(self):
        self._ensure_env()
        if self.env is None:
            raise RuntimeError("UnrealZoo environment is not started")
        return self.env

    def _player(self) -> str:
        env = self._require_env()
        if self.player_name is None:
            if not env.unwrapped.player_list:
                raise RuntimeError("UnrealZoo environment has no player")
            self.player_name = str(env.unwrapped.player_list[0])
        return self.player_name

    def _set_camera(self) -> None:
        env = self._require_env()
        if hasattr(env.unwrapped.unrealcv, "set_cam"):
            env.unwrapped.unrealcv.set_cam(self._player())

    def _uavflow_pose_payload(self, *, image: np.ndarray) -> dict[str, Any]:
        return {
            "rgb": [image],
            "sensors": {
                "state": {"position": [self._pose.x, self._pose.y, self._pose.z]},
                "imu": {"rotation": np.eye(3, dtype=np.float32)},
            },
        }


def nav_pose_from_sim_pose_cm(pose_cm) -> Pose4D:
    return nav_pose_from_unreal_cm(pose_cm)


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"env.kwargs.{key} is required for UnrealZoo backend")
    return value
