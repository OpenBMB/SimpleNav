from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf


ALLOWED_EVAL_COMBINATIONS = {
    ("traveluav", "eval_json", "airsim"),
    ("traveluav", "navvla_lerobot_v3", "airsim"),
    ("openfly", "starvla_episode_json", "airsim"),
    ("aerialvln", "aerialvln_json", "airsim"),
    ("fixture_offline_replay", "fixture_episode_json", "offline"),
    ("fixture_offline_transition", "fixture_episode_json", "offline"),
}

AIRSIM_ONLY_ENV_FIELDS = {
    "env_root",
    "render_lib_root",
    "base_airsim_port",
    "layout",
    "settings_root",
    "settings_profile",
    "sensor_profile",
    "camera_name",
    "ue_args",
    "startup_timeout",
    "episode_startup_settle_sec",
    "render_warmup_sec",
    "reset_ignore_collision",
    "ignore_collision",
    "capture_action_observations",
    "action_execution_mode",
    "action_waypoint_semantics",
    "airsim_z_sign",
    "execute_waypoints_per_step",
    "teleport_render_sync_frames",
    "recording_folder",
    "recording_camera_name",
    "recording_interval",
    "camera_resolution_overrides",
    "external_camera_resolution_overrides",
    "clock_speed",
}

LEGACY_AIRSIM_ENV_FIELDS = {
    "camera_profile": "settings_profile/sensor_profile",
    "openfly_render_sync_frames": "teleport_render_sync_frames",
    "openfly_render_warmup_sec": "render_warmup_sec",
}

UNREALZOO_PATH_ENV_FIELDS = {
    "unreal_env_root",
    "binary_path",
    "unrealzoo_gym_root",
    "scene_config_path",
    "render_lib_root",
}

HABITAT_PATH_ENV_FIELDS = {
    "data_root",
    "vlnce_data_root",
    "evt_bench_root",
    "habitat_lab_root",
    "habitat_sim_site_packages",
    "benchmark_config_path",
    "habitat_config_path",
    "nvidia_egl_root",
    "scenes_dir",
}

BENCHMARK_KWARG_PATH_FIELDS = {
    "groundingdino_config",
    "groundingdino_model_path",
    "dataset_root",
}

MODEL_PATH_FIELDS = {
    "repo_root",
}

MODEL_CONFIG_OVERRIDE_PATH_FIELDS = {
    ("framework", "qwenvl", "base_vlm"),
    ("framework", "navvla", "visual_cache_encoder_ckpt"),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    class_path: str
    max_steps: int
    kwargs: dict[str, Any] = field(default_factory=dict)
    max_samples: int | None = None


@dataclass(frozen=True)
class InputRootConfig:
    namespace: str
    path: Path


@dataclass(frozen=True)
class InputConfig:
    type: str
    adapter_class_path: str
    namespace: str | None = None
    path: Path | None = None
    roots: tuple[InputRootConfig, ...] = ()
    data_root: Path | None = None
    split: str | None = None
    max_samples: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    checkpoint: Path
    unnorm_key: str | None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeDatasetConfig:
    action_type: str | None = None
    action_horizon: int | None = None
    runtime_adapter: str | None = None
    dataset_py: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvConfig:
    type: str
    backend_class_path: str | None = None
    planner_class_path: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    transition_mode: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParallelConfig:
    gpu_ids: tuple[int, ...]
    worker_timeout_sec: float = 0.0
    partition_mode: str = "contiguous_episode_chunks"


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    run_name: str
    save_step_artifacts: bool = True
    save_images: bool = True
    image_cameras: tuple[str, ...] | None = None
    action_observation_image_policy: str = "step"
    metrics: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalConfig:
    benchmark: BenchmarkConfig
    input: InputConfig
    model: ModelConfig
    dataset: RuntimeDatasetConfig
    env: EnvConfig
    parallel: ParallelConfig
    output: OutputConfig
    raw: dict[str, Any] = field(default_factory=dict)


def load_class(class_path: str) -> type[Any]:
    if ":" not in str(class_path):
        raise ValueError(f"class_path must use module:ClassName format, got {class_path!r}")
    module_name, class_name = str(class_path).split(":", 1)
    if not module_name or not class_name:
        raise ValueError(f"class_path must use module:ClassName format, got {class_path!r}")
    module = importlib.import_module(module_name)
    try:
        value = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"class_path target does not exist: {class_path}") from exc
    if not isinstance(value, type):
        raise TypeError(f"class_path target is not a class: {class_path}")
    return value


def load_eval_config(
    config_path: str | Path,
    overrides: Iterable[str] | None = None,
) -> EvalConfig:
    config_path = Path(config_path)
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    typed = _build_typed_config(resolved, base_dir=config_path.parent)
    _validate_eval_config(typed)
    return typed


def _as_path(value: Any, *, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _build_typed_config(data: dict[str, Any], *, base_dir: Path) -> EvalConfig:
    for section in ("benchmark", "input", "model", "dataset", "env", "parallel", "output"):
        if section not in data or not isinstance(data[section], dict):
            raise ValueError(f"{section} section is required")

    benchmark_raw = dict(data["benchmark"])
    input_raw = dict(data["input"])
    model_raw = dict(data["model"])
    dataset_raw = dict(data["dataset"])
    env_raw = dict(data["env"])
    parallel_raw = dict(data["parallel"])
    output_raw = dict(data["output"])

    benchmark_kwargs = dict(benchmark_raw.get("kwargs") or {})
    for key in BENCHMARK_KWARG_PATH_FIELDS:
        if key in benchmark_kwargs and benchmark_kwargs[key] is not None:
            benchmark_kwargs[key] = str(_as_path(benchmark_kwargs[key], base_dir=base_dir))

    for key in MODEL_PATH_FIELDS:
        if key in model_raw and model_raw[key] is not None:
            model_raw[key] = str(_as_path(model_raw[key], base_dir=base_dir))
    config_overrides = model_raw.get("config_overrides")
    if isinstance(config_overrides, dict):
        for path_keys in MODEL_CONFIG_OVERRIDE_PATH_FIELDS:
            parent: Any = config_overrides
            for key in path_keys[:-1]:
                if not isinstance(parent, dict):
                    parent = None
                    break
                parent = parent.get(key)
            leaf = path_keys[-1]
            if isinstance(parent, dict) and parent.get(leaf) is not None:
                parent[leaf] = str(_as_path(parent[leaf], base_dir=base_dir))

    roots = []
    for root in input_raw.get("roots") or []:
        if not isinstance(root, dict):
            raise ValueError("input.roots entries must be mappings")
        namespace = str(root.get("namespace") or "").strip()
        if not namespace:
            raise ValueError("input.roots[].namespace is required")
        if root.get("path") is None:
            raise ValueError("input.roots[].path is required")
        roots.append(InputRootConfig(namespace=namespace, path=_as_path(root["path"], base_dir=base_dir)))

    input_path = _as_path(input_raw["path"], base_dir=base_dir) if input_raw.get("path") is not None else None
    data_root = _as_path(input_raw["data_root"], base_dir=base_dir) if input_raw.get("data_root") is not None else None

    env_type = str(env_raw.get("type") or "").strip()
    env_kwargs = _env_kwargs(env_raw, base_dir=base_dir)

    return EvalConfig(
        benchmark=BenchmarkConfig(
            name=str(benchmark_raw.get("name") or "").strip(),
            class_path=str(benchmark_raw.get("class_path") or "").strip(),
            max_steps=int(benchmark_raw.get("max_steps", 0)),
            kwargs=benchmark_kwargs,
            max_samples=_optional_int(benchmark_raw.get("max_samples")),
        ),
        input=InputConfig(
            type=str(input_raw.get("type") or "").strip(),
            adapter_class_path=str(input_raw.get("adapter_class_path") or "").strip(),
            namespace=_optional_str(input_raw.get("namespace")),
            path=input_path,
            roots=tuple(roots),
            data_root=data_root,
            split=_optional_str(input_raw.get("split")),
            max_samples=_optional_int(input_raw.get("max_samples")),
            raw=input_raw,
        ),
        model=ModelConfig(
            checkpoint=_as_path(model_raw["checkpoint"], base_dir=base_dir),
            unnorm_key=_optional_str(model_raw.get("unnorm_key")),
            kwargs={key: value for key, value in model_raw.items() if key not in {"checkpoint", "unnorm_key"}},
        ),
        dataset=RuntimeDatasetConfig(
            action_type=_optional_str(dataset_raw.get("action_type")),
            action_horizon=_optional_int(dataset_raw.get("action_horizon")),
            runtime_adapter=_optional_str(dataset_raw.get("runtime_adapter")),
            dataset_py=_optional_str(dataset_raw.get("dataset_py")),
            raw=dataset_raw,
        ),
        env=EnvConfig(
            type=env_type,
            backend_class_path=_optional_str(env_raw.get("backend_class_path")),
            planner_class_path=_optional_str(env_raw.get("planner_class_path")),
            kwargs=env_kwargs,
            transition_mode=_optional_str(env_raw.get("transition_mode")),
            raw=env_raw,
        ),
        parallel=ParallelConfig(
            gpu_ids=tuple(int(gpu_id) for gpu_id in (parallel_raw.get("gpu_ids") or ())),
            worker_timeout_sec=float(parallel_raw.get("worker_timeout_sec", 0.0)),
            partition_mode=str(parallel_raw.get("partition_mode", "contiguous_episode_chunks")),
        ),
        output=OutputConfig(
            root=_as_path(output_raw["root"], base_dir=base_dir),
            run_name=str(output_raw.get("run_name") or "").strip(),
            save_step_artifacts=bool(output_raw.get("save_step_artifacts", True)),
            save_images=bool(output_raw.get("save_images", True)),
            image_cameras=_optional_str_tuple(output_raw.get("image_cameras")),
            action_observation_image_policy=str(output_raw.get("action_observation_image_policy", "step")),
            metrics=_optional_str_tuple(output_raw.get("metrics")) or (),
            raw=output_raw,
        ),
        raw=data,
    )


def _env_kwargs(env_raw: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    kwargs = _mapping_field(env_raw.get("kwargs"), "env.kwargs")
    for legacy_key, canonical_key in LEGACY_AIRSIM_ENV_FIELDS.items():
        if legacy_key in kwargs:
            raise ValueError(
                f"env.kwargs.{legacy_key} is not supported; use env.kwargs.{canonical_key}"
            )
    for key in AIRSIM_ONLY_ENV_FIELDS:
        if key in kwargs and kwargs.get(key) is not None:
            raw_value = kwargs[key]
        else:
            continue
        if key in {
            "env_root",
            "render_lib_root",
            "recording_folder",
            "settings_root",
        }:
            kwargs[key] = _as_path(raw_value, base_dir=base_dir)
        elif key == "base_airsim_port":
            kwargs[key] = _optional_int(raw_value)
        elif key in {
            "execute_waypoints_per_step",
            "teleport_render_sync_frames",
        }:
            kwargs[key] = _optional_int(raw_value)
        elif key in {
            "startup_timeout",
            "episode_startup_settle_sec",
            "render_warmup_sec",
            "recording_interval",
            "clock_speed",
            "airsim_z_sign",
        }:
            kwargs[key] = _optional_float(raw_value)
        elif key in {"capture_action_observations", "reset_ignore_collision", "ignore_collision"}:
            kwargs[key] = _optional_bool(raw_value)
        elif key == "ue_args":
            kwargs[key] = tuple(str(item) for item in (raw_value or ()))
        elif key in {"camera_resolution_overrides", "external_camera_resolution_overrides"}:
            kwargs[key] = _resolution_overrides(raw_value)
        else:
            kwargs[key] = _optional_str(raw_value)
    for key in UNREALZOO_PATH_ENV_FIELDS:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = _as_path(kwargs[key], base_dir=base_dir)
    for key in HABITAT_PATH_ENV_FIELDS:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = _as_path(kwargs[key], base_dir=base_dir)
    return kwargs


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_str_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [value]
    else:
        items = list(value)
    parsed = tuple(str(item).strip() for item in items if str(item).strip())
    return parsed or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"expected boolean value, got {value!r}")


def _mapping_field(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _resolution_overrides(value: Any) -> dict[str, tuple[int, int]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("camera resolution overrides must be a mapping")
    parsed: dict[str, tuple[int, int]] = {}
    for key, resolution in value.items():
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise ValueError(f"camera resolution override for {key!r} must be [width, height]")
        parsed[str(key)] = (int(resolution[0]), int(resolution[1]))
    return parsed


def _validate_eval_config(cfg: EvalConfig) -> None:
    if not cfg.benchmark.name:
        raise ValueError("benchmark.name is required")
    if not cfg.benchmark.class_path:
        raise ValueError("benchmark.class_path is required")
    if cfg.benchmark.max_steps <= 0:
        raise ValueError("benchmark.max_steps must be positive")
    if not cfg.input.type:
        raise ValueError("input.type is required")
    if not cfg.input.adapter_class_path:
        raise ValueError("input.adapter_class_path is required")
    _validate_input_namespaces(cfg.input)

    combo = (cfg.benchmark.name, cfg.input.type, cfg.env.type)
    if cfg.env.type in {"airsim", "offline"} and combo not in ALLOWED_EVAL_COMBINATIONS:
        raise ValueError(f"Unsupported benchmark/input/env combination: {combo}")

    if not cfg.parallel.gpu_ids:
        raise ValueError("parallel.gpu_ids must not be empty")
    if len(set(cfg.parallel.gpu_ids)) != len(cfg.parallel.gpu_ids):
        raise ValueError(f"parallel.gpu_ids contains duplicate ids: {list(cfg.parallel.gpu_ids)}")
    if cfg.parallel.partition_mode != "contiguous_episode_chunks":
        raise ValueError("parallel.partition_mode must be contiguous_episode_chunks")
    if not cfg.output.run_name:
        raise ValueError("output.run_name is required")
    _validate_output_action_observation_image_policy(cfg.output.action_observation_image_policy)
    _validate_output_metrics(cfg.output.metrics)

    _validate_env(cfg.env)
    _validate_paths(cfg)
    _validate_stats(cfg)
    load_class(cfg.benchmark.class_path)
    load_class(cfg.input.adapter_class_path)


def _validate_input_namespaces(input_cfg: InputConfig) -> None:
    if input_cfg.roots:
        for root in input_cfg.roots:
            if not root.namespace:
                raise ValueError("input.roots[].namespace is required")
        return
    if not input_cfg.namespace:
        raise ValueError("input.namespace is required")

def _validate_output_action_observation_image_policy(policy: str) -> None:
    allowed = {"step", "action", "both", "none"}
    if str(policy) not in allowed:
        raise ValueError(f"output.action_observation_image_policy must be one of {sorted(allowed)}, got {policy!r}")


def _validate_output_metrics(metrics: tuple[str, ...]) -> None:
    allowed = {"SR", "OSR", "NE", "SPL", "standard_SPL", "nDTW", "path_length", "gt_path_length", "steps_taken"}
    unknown = sorted(set(metrics) - allowed)
    if unknown:
        raise ValueError(f"Unsupported output.metrics entries: {unknown}")


def _validate_env(env: EnvConfig) -> None:
    if env.type == "airsim":
        if env.kwargs.get("env_root") is None:
            raise ValueError("env.kwargs.env_root is required for AirSim env")
        if env.kwargs.get("render_lib_root") is None:
            raise ValueError("env.kwargs.render_lib_root is required for AirSim env")
        if env.kwargs.get("base_airsim_port") is None:
            raise ValueError("env.kwargs.base_airsim_port is required for AirSim env")
        if env.kwargs.get("settings_root") is not None or env.raw.get("settings_root") is not None:
            raise ValueError("env.settings_root is not allowed; planning derives per-run settings paths")
        if not env.backend_class_path:
            raise ValueError("env.backend_class_path is required for AirSim env")
        if not env.planner_class_path:
            raise ValueError("env.planner_class_path is required for AirSim env")
        execute_waypoints_per_step = env.kwargs.get("execute_waypoints_per_step")
        if execute_waypoints_per_step is not None and int(execute_waypoints_per_step) <= 0:
            raise ValueError("env.kwargs.execute_waypoints_per_step must be a positive integer or null")
        render_warmup_sec = env.kwargs.get("render_warmup_sec")
        if render_warmup_sec is not None and float(render_warmup_sec) < 0:
            raise ValueError("env.kwargs.render_warmup_sec must be non-negative")
        episode_startup_settle_sec = env.kwargs.get("episode_startup_settle_sec")
        if episode_startup_settle_sec is not None and float(episode_startup_settle_sec) < 0:
            raise ValueError("env.kwargs.episode_startup_settle_sec must be non-negative")
        action_execution_mode = env.kwargs.get("action_execution_mode")
        if action_execution_mode is not None and action_execution_mode not in {"teleport_final", "teleport_each_waypoint", "path"}:
            raise ValueError(f"Unsupported env.kwargs.action_execution_mode: {action_execution_mode!r}")
        airsim_z_sign = env.kwargs.get("airsim_z_sign")
        if airsim_z_sign is not None and float(airsim_z_sign) not in {-1.0, 1.0}:
            raise ValueError("env.kwargs.airsim_z_sign must be either -1 or 1")
        return

    if env.type != "offline":
        if not env.backend_class_path:
            raise ValueError("custom env.type requires env.backend_class_path")
        return

    forbidden = sorted(field for field in AIRSIM_ONLY_ENV_FIELDS if env.raw.get(field) is not None or field in env.kwargs)
    if forbidden:
        raise ValueError(f"offline env cannot define simulator-specific AirSim fields: {forbidden}")
    if env.transition_mode not in {None, "replay_logged_trajectory", "benchmark_defined_transition"}:
        raise ValueError(f"Unsupported offline transition_mode: {env.transition_mode}")


def _validate_paths(cfg: EvalConfig) -> None:
    required_paths: list[tuple[str, Path | None]] = [
        ("model.checkpoint", cfg.model.checkpoint),
    ]
    if cfg.input.path is not None:
        required_paths.append(("input.path", cfg.input.path))
    if cfg.input.data_root is not None:
        required_paths.append(("input.data_root", cfg.input.data_root))
        if cfg.input.split:
            split_path = cfg.input.data_root / "splits" / f"{cfg.input.split}.txt"
            annotation_path = cfg.input.data_root / "Annotation" / f"{cfg.input.split}.json"
            required_paths.append(("input.split", split_path if split_path.exists() else annotation_path))
    for index, root in enumerate(cfg.input.roots):
        required_paths.append((f"input.roots[{index}].path", root.path))
    if cfg.env.type == "airsim":
        required_paths.extend(
            [
                ("env.kwargs.env_root", cfg.env.kwargs.get("env_root")),
                ("env.kwargs.render_lib_root", cfg.env.kwargs.get("render_lib_root")),
            ]
        )
    if cfg.env.type == "unrealzoo":
        required_paths.append(("env.kwargs.unreal_env_root", cfg.env.kwargs.get("unreal_env_root")))
        if cfg.env.kwargs.get("binary_path") is not None:
            required_paths.append(("env.kwargs.binary_path", cfg.env.kwargs.get("binary_path")))
        if cfg.env.kwargs.get("render_lib_root") is not None:
            required_paths.extend(
                [
                    ("env.kwargs.render_lib_root", cfg.env.kwargs.get("render_lib_root")),
                    ("env.kwargs.render_lib_root.lib", cfg.env.kwargs.get("render_lib_root") / "lib"),
                    ("env.kwargs.render_lib_root.etc.nvidia_icd", cfg.env.kwargs.get("render_lib_root") / "etc" / "nvidia_icd.json"),
                    ("env.kwargs.render_lib_root.etc.10_nvidia", cfg.env.kwargs.get("render_lib_root") / "etc" / "10_nvidia.json"),
                ]
            )
    for label, path in required_paths:
        if path is None or not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")


def _validate_stats(cfg: EvalConfig) -> None:
    stats = _load_dataset_statistics(cfg.model.checkpoint)
    if not stats:
        raise FileNotFoundError(f"dataset_statistics.json does not exist near checkpoint: {cfg.model.checkpoint}")
    keys = list(stats.keys())
    if len(keys) > 1 and not cfg.model.unnorm_key:
        raise ValueError(f"model.unnorm_key is required when dataset_statistics has multiple keys: {keys}")
    if cfg.model.unnorm_key and cfg.model.unnorm_key not in stats:
        raise ValueError(f"model.unnorm_key {cfg.model.unnorm_key!r} not found in dataset_statistics keys: {keys}")


def _load_dataset_statistics(checkpoint: Path) -> dict[str, Any]:
    candidates = []
    if checkpoint.is_file():
        candidates.extend([checkpoint.parent / "dataset_statistics.json", checkpoint.parent.parent / "dataset_statistics.json"])
    else:
        candidates.extend([checkpoint / "dataset_statistics.json", checkpoint.parent / "dataset_statistics.json"])
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"dataset_statistics.json must contain an object: {candidate}")
            return data
    return {}
