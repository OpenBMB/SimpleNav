from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VLNCE_RUNTIME_ROOT = REPO_ROOT / "local" / "simulators" / "VLN-CE"
DEFAULT_EVT_BENCH_ROOT = DEFAULT_VLNCE_RUNTIME_ROOT / "Evt-bench"
DEFAULT_HABITAT_SIM_SITE_PACKAGES = (
    DEFAULT_VLNCE_RUNTIME_ROOT
    / "build_py310_habitat_sim_031"
    / "lib"
    / "python3.10"
    / "site-packages"
)
class VLNCE031HabitatRuntime:
    def __init__(
        self,
        *,
        data_root: str | Path,
        split: str,
        gpu_id: int,
        task_name: str = "r2r",
        seed: int = 0,
        evt_bench_root: str | Path | None = None,
        habitat_lab_root: str | Path | None = None,
        habitat_sim_site_packages: str | Path | None = None,
        benchmark_config_path: str | Path | None = None,
        success_distance: float = 3.0,
        image_size: int = 224,
        camera_height: float = 1.25,
        continuous_control_mode: str = "filtered_pose_delta",
        max_episode_steps: int = 500,
        roles: list[str] | tuple[str, ...] | None = None,
        languages: list[str] | tuple[str, ...] | None = None,
        content_scenes: list[str] | tuple[str, ...] | None = None,
        load_on_init: bool = True,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = str(split)
        self.gpu_id = int(gpu_id)
        self.task_name = str(task_name or "r2r").lower()
        self.seed = int(seed)
        self.evt_bench_root = Path(evt_bench_root or DEFAULT_EVT_BENCH_ROOT).expanduser().resolve()
        self.habitat_lab_root = Path(habitat_lab_root or self.evt_bench_root / "habitat-lab").expanduser().resolve()
        self.habitat_sim_site_packages = (
            Path(habitat_sim_site_packages).expanduser().resolve()
            if habitat_sim_site_packages
            else DEFAULT_HABITAT_SIM_SITE_PACKAGES
        )
        self.benchmark_config_path = Path(
            benchmark_config_path
            or self.habitat_lab_root / "habitat" / "config" / "benchmark" / "nav" / "vln_r2r.yaml"
        ).expanduser().resolve()
        self.success_distance = float(success_distance)
        self.image_size = int(image_size)
        self.camera_height = float(camera_height)
        self.continuous_control_mode = str(continuous_control_mode)
        self.max_episode_steps = int(max_episode_steps)
        self.roles = tuple(str(role) for role in (roles or ("guide",)))
        self.languages = tuple(str(language) for language in (languages or ()))
        self.content_scenes = tuple(str(scene) for scene in (content_scenes or ()))
        self.env: Any | None = None
        self.dataset: Any | None = None
        self._last_observation: dict[str, Any] = {}
        if load_on_init:
            self._load()

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.env is None:
            self._load()
        assert self.env is not None
        episode_id = str(payload.get("episode_id") or dict(payload.get("item") or {}).get("episode_id"))
        if not episode_id:
            raise KeyError("VLNCE031 reset payload must contain episode_id")
        self._seek_episode(episode_id)
        observation = self.env.reset()
        actual_episode_id = str(getattr(self.env.current_episode, "episode_id", ""))
        if actual_episode_id != episode_id:
            raise RuntimeError(
                "VLN-CE Habitat reset episode mismatch: "
                f"requested_episode_id={episode_id!r}, actual_episode_id={actual_episode_id!r}"
            )
        self._last_observation = dict(observation)
        return self._response(observation)

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("VLNCE031 runtime has no active Habitat Env")
        action = dict(payload)
        if "action_args" not in action:
            action["action_args"] = {}
        observation = self.env.step(action)
        self._last_observation = dict(observation)
        return self._response(observation)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
        self.dataset = None
        self._last_observation = {}

    def _load(self) -> None:
        apply_runtime_python_paths(
            {
                "data_root": str(self.data_root),
                "evt_bench_root": str(self.evt_bench_root),
                "habitat_lab_root": str(self.habitat_lab_root),
                "habitat_sim_site_packages": str(self.habitat_sim_site_packages),
            }
        )
        apply_gym_spaces_compat()
        importlib.import_module("habitat.datasets.vln.r2r_vln_dataset")
        importlib.import_module("habitat.tasks.vln.vln")
        actions_module = importlib.import_module("NavVLAeval.common.simulators.habitat.vlnce031_actions")
        measures_module = importlib.import_module("NavVLAeval.common.simulators.habitat.vlnce031_measures")
        datasets_module = importlib.import_module("NavVLAeval.common.simulators.habitat.vlnce031_datasets")

        get_config = importlib.import_module("habitat.config.default").get_config
        read_write = importlib.import_module("habitat.config").read_write
        make_dataset = importlib.import_module("habitat.datasets").make_dataset
        habitat = importlib.import_module("habitat")

        cfg = get_config(str(self.benchmark_config_path))
        with read_write(cfg):
            cfg.habitat.seed = self.seed
            cfg.habitat.environment.max_episode_steps = self.max_episode_steps
            cfg.habitat.environment.iterator_options.shuffle = False
            cfg.habitat.environment.iterator_options.max_scene_repeat_steps = -1
            datasets_module.configure_dataset_config(
                cfg,
                data_root=self.data_root,
                task_name=self.task_name,
                split=self.split,
                roles=self.roles,
                languages=self.languages,
                content_scenes=self.content_scenes,
            )
            cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = habitat_gpu_device_id(self.gpu_id)
            cfg.habitat.simulator.agents.main_agent.sim_sensors = structured_camera_sensor_configs(
                image_size=self.image_size,
                height=self.camera_height,
            )
            cfg.habitat.task.lab_sensors = {}
            cfg.habitat.task.actions = actions_module.structured_action_configs(control_mode=self.continuous_control_mode)
            cfg.habitat.task.measurements = measures_module.structured_measurement_configs(
                split=self.split,
                gt_path=datasets_module.vlnce_split_gt_path(
                    self.data_root,
                    task_name=self.task_name,
                    split=self.split,
                    roles=self.roles,
                ),
                success_distance=self.success_distance,
            )

        self.dataset = make_dataset(id_dataset=cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        self.env = habitat.Env(config=cfg, dataset=self.dataset)

    def _seek_episode(self, episode_id: str) -> None:
        assert self.env is not None
        iterator = getattr(self.env, "_episode_iterator", None)
        if hasattr(iterator, "set_next_episode_by_id"):
            for candidate in _episode_id_candidates(episode_id):
                try:
                    iterator.set_next_episode_by_id(candidate)
                    self.env.episode_iterator = iterator
                    return
                except ValueError:
                    continue
        episodes = list(getattr(self.env, "episodes", []) or [])
        for index, episode in enumerate(episodes):
            if str(getattr(episode, "episode_id", "")) == str(episode_id):
                self.env.episodes = episodes[index:] + episodes[:index]
                return
        raise KeyError(f"VLN-CE {self.task_name} episode_id={episode_id!r} is not present in Habitat dataset")

    def _response(self, observation: dict[str, Any]) -> dict[str, Any]:
        assert self.env is not None
        metrics = _jsonable(dict(self.env.get_metrics()))
        episode = self.env.current_episode
        euclidean_distance = _euclidean_distance_to_episode_goal(self.env, episode)
        if euclidean_distance is not None:
            if "distance_to_goal" in metrics:
                metrics["geodesic_distance_to_goal"] = metrics["distance_to_goal"]
            metrics["distance_to_goal"] = euclidean_distance
            metrics["euclidean_distance_to_goal"] = euclidean_distance
        return {
            "observation": dict(observation),
            "pose": _agent_pose(self.env),
            "instruction": str(getattr(getattr(episode, "instruction", None), "instruction_text", "") or ""),
            "episode_id": str(getattr(episode, "episode_id", "")),
            "scene_id": str(getattr(episode, "scene_id", "")),
            "metrics": metrics,
            "done": bool(self.env.episode_over),
        }


def runtime_kwargs_from_cfg(kwargs: dict[str, Any], *, physical_gpu_id: int) -> dict[str, Any]:
    return {
        "data_root": Path(str(kwargs.get("data_root") or kwargs.get("vlnce_data_root") or ".")).expanduser().resolve(),
        "split": str(kwargs.get("split") or "val_seen"),
        "gpu_id": int(physical_gpu_id),
        "task_name": str(kwargs.get("task_name") or kwargs.get("benchmark_name") or "r2r").lower(),
        "seed": int(kwargs.get("seed", 0)),
        "evt_bench_root": _optional_path(kwargs.get("evt_bench_root")),
        "habitat_lab_root": _optional_path(kwargs.get("habitat_lab_root")),
        "habitat_sim_site_packages": _optional_path(kwargs.get("habitat_sim_site_packages")),
        "benchmark_config_path": _optional_path(kwargs.get("benchmark_config_path") or kwargs.get("habitat_config_path")),
        "success_distance": float(kwargs.get("success_distance", 3.0)),
        "image_size": int(kwargs.get("image_size", 224)),
        "camera_height": float(kwargs.get("camera_height", 1.25)),
        "continuous_control_mode": str(kwargs.get("continuous_control_mode") or "filtered_pose_delta"),
        "max_episode_steps": int(kwargs.get("max_episode_steps", 500)),
        "roles": tuple(str(role) for role in (kwargs.get("roles") or ("guide",))),
        "languages": tuple(str(language) for language in (kwargs.get("languages") or ())),
        "content_scenes": kwargs.get("content_scenes") or kwargs.get("scene_ids"),
    }


def runtime_python_paths(kwargs: dict[str, Any]) -> list[Path]:
    data_root = Path(str(kwargs.get("data_root") or kwargs.get("vlnce_data_root") or ".")).expanduser().resolve()
    evt_bench_root = Path(str(kwargs.get("evt_bench_root") or DEFAULT_EVT_BENCH_ROOT)).expanduser().resolve()
    habitat_lab_root = Path(str(kwargs.get("habitat_lab_root") or evt_bench_root / "habitat-lab")).expanduser().resolve()
    paths = [habitat_lab_root, evt_bench_root]
    habitat_sim_site_packages = _habitat_sim_site_packages_path(kwargs, data_root=data_root, evt_bench_root=evt_bench_root)
    if habitat_sim_site_packages is not None:
        paths.append(habitat_sim_site_packages)
    if data_root != evt_bench_root:
        paths.append(data_root)
    return paths


def apply_runtime_python_paths(kwargs: dict[str, Any]) -> None:
    for path in reversed(runtime_python_paths(kwargs)):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def apply_gym_spaces_compat() -> None:
    try:
        import gym
        from gym import spaces
        from typing import Any as TypingAny
    except Exception:
        return
    if not hasattr(spaces, "Space") and hasattr(gym, "Space"):
        spaces.Space = gym.Space
    if not hasattr(spaces, "space") and hasattr(spaces, "Space"):
        spaces.space = spaces.Space
    if not hasattr(gym, "spaces"):
        gym.spaces = spaces
    try:
        gym_core = importlib.import_module("gym.core")
    except Exception:
        return
    for alias in ("ActType", "ObsType", "RenderFrame"):
        if not hasattr(gym_core, alias):
            setattr(gym_core, alias, TypingAny)


def structured_camera_sensor_configs(*, image_size: int = 224, height: float = 1.25, hfov: int = 90) -> dict[str, Any]:
    from habitat.config.default_structured_configs import HabitatSimRGBSensorConfig

    @dataclass
    class VLNCE031RGBSensorConfig(HabitatSimRGBSensorConfig):
        uuid: str = "rgb"

    base = {
        "height": int(image_size),
        "width": int(image_size),
        "hfov": int(hfov),
        "position": [0.0, float(height), 0.0],
    }
    orientations = {
        "rgb": [0.0, 0.0, 0.0],
        "rgb_left": [0.0, float(np.pi / 2.0), 0.0],
        "rgb_right": [0.0, float(-np.pi / 2.0), 0.0],
        "rgb_rear": [0.0, float(np.pi), 0.0],
    }
    return {
        name: VLNCE031RGBSensorConfig(uuid=name, orientation=orientation, **base)
        for name, orientation in orientations.items()
    }


def habitat_gpu_device_id(requested_gpu_id: int) -> int:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices and "," not in visible_devices:
        return 0
    return int(requested_gpu_id)


def _habitat_sim_site_packages_path(kwargs: dict[str, Any], *, data_root: Path, evt_bench_root: Path) -> Path | None:
    raw = kwargs.get("habitat_sim_site_packages")
    if raw:
        return Path(str(raw)).expanduser().resolve()
    candidates = [
        data_root / "build_py310_habitat_sim_031" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        evt_bench_root.parent / "build_py310_habitat_sim_031" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        DEFAULT_HABITAT_SIM_SITE_PACKAGES,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _agent_pose(env: Any) -> list[float]:
    state = env.sim.get_agent_state()
    position = np.asarray(state.position, dtype=np.float64).reshape(-1)
    return [float(position[0]), float(position[1]), float(position[2]), _heading_from_quaternion(state.rotation)]


def _euclidean_distance_to_episode_goal(env: Any, episode: Any) -> float | None:
    try:
        agent_position = env.sim.get_agent_state().position
    except Exception:
        return None
    return euclidean_distance_to_goal(agent_position, _episode_goal_positions(episode))


def euclidean_distance_to_goal(agent_position: Any, goal_positions: Any) -> float | None:
    try:
        position = np.asarray(agent_position, dtype=np.float64).reshape(-1)[:3]
    except Exception:
        return None
    if position.shape[0] != 3:
        return None
    distances: list[float] = []
    for goal_position in goal_positions or ():
        try:
            target = np.asarray(goal_position, dtype=np.float64).reshape(-1)[:3]
        except Exception:
            continue
        if target.shape[0] != 3:
            continue
        distance = float(np.linalg.norm(target - position))
        if np.isfinite(distance):
            distances.append(distance)
    return min(distances) if distances else None


def _episode_goal_positions(episode: Any) -> list[Any]:
    goals = getattr(episode, "goals", None)
    if goals is None and isinstance(episode, dict):
        goals = episode.get("goals")
    positions: list[Any] = []
    for goal in goals or ():
        if isinstance(goal, dict):
            position = goal.get("position")
        else:
            position = getattr(goal, "position", None)
        if position is not None:
            positions.append(position)
    goal_position = getattr(episode, "goal_position", None)
    if goal_position is None and isinstance(episode, dict):
        goal_position = episode.get("goal_position")
    if goal_position is not None:
        positions.append(goal_position)
    return positions


def _heading_from_quaternion(rotation: Any) -> float:
    try:
        from habitat.tasks.utils import cartesian_to_polar
        from habitat.utils.geometry_utils import quaternion_rotate_vector

        heading_vector = quaternion_rotate_vector(rotation.inverse(), np.asarray([0.0, 0.0, -1.0]))
        return float(cartesian_to_polar(-heading_vector[2], heading_vector[0])[1])
    except Exception:
        return 0.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value)).expanduser().resolve()


def _episode_id_candidates(episode_id: str) -> list[Any]:
    text = str(episode_id)
    candidates: list[Any] = [text]
    if text.isdigit():
        candidates.append(int(text))
    return candidates
