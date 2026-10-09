"""Evaluation configuration: tasks, simulator execution and remote policy access."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf


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
    uri: str = "ws://127.0.0.1:10093"
    statistics_key: str | None = None
    request_timeout_sec: float = 120.0
    seed: int = 42


@dataclass(frozen=True)
class EnvConfig:
    type: str
    backend_class_path: str | None = None
    planner_class_path: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParallelConfig:
    gpu_ids: tuple[int, ...]
    worker_timeout_sec: float = 0.0


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
    env: EnvConfig
    parallel: ParallelConfig
    output: OutputConfig
    observation: dict[str, Any] = field(default_factory=dict)
    driver_environment: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def load_class(class_path: str):
    module, name = class_path.split(":")
    return getattr(importlib.import_module(module), name)


def _as_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def load_driver_environment(path: str | Path | None) -> dict[str, str]:
    """Validate explicit driver paths once; an empty entry inherits the environment."""
    if path is None:
        return {}
    path = Path(path).resolve()
    values = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    allowed = {"LD_LIBRARY_PATH", "__EGL_VENDOR_LIBRARY_FILENAMES", "VK_DRIVER_FILES"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown driver environment entries: {sorted(unknown)}")
    result = {}
    for key, entries in values.items():
        if not isinstance(entries, list):
            raise TypeError(f"{key} must be a list of paths")
        paths = [_as_path(entry, path.parent) for entry in entries]
        for item in paths:
            valid = item.is_dir() if key == "LD_LIBRARY_PATH" else item.is_file()
            if not valid:
                raise FileNotFoundError(f"{key}: {item}")
        if paths:
            result[key] = os.pathsep.join(map(str, paths))
    return result


def apply_driver_environment(values: dict[str, str], env: dict[str, str]) -> None:
    for key, value in values.items():
        if key == "LD_LIBRARY_PATH":
            value = os.pathsep.join(
                dict.fromkeys((value + os.pathsep + env.get(key, "")).strip(os.pathsep).split(os.pathsep))
            )
        env[key] = value


def load_eval_config(config_path: str | Path, overrides: Iterable[str] | None = None) -> EvalConfig:
    path = Path(config_path).resolve()
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    data = OmegaConf.to_container(config, resolve=True)
    return _build_typed_config(data, base_dir=path.parent)


def _build_typed_config(data: dict[str, Any], *, base_dir: Path) -> EvalConfig:
    benchmark, inputs, model, env, parallel, output = (
        dict(data[key]) for key in ("benchmark", "input", "model", "env", "parallel", "output")
    )
    if "dataset" in data or "checkpoint" in model:
        raise ValueError("Evaluation uses model.uri; model/input configuration belongs to the saved model assets")
    for key in ("groundingdino_config", "groundingdino_model_path", "dataset_root"):
        if benchmark.get("kwargs", {}).get(key):
            benchmark["kwargs"][key] = str(_as_path(benchmark["kwargs"][key], base_dir))
    for key in ("path", "data_root"):
        if inputs.get(key):
            inputs[key] = str(_as_path(inputs[key], base_dir))
    roots = [
        InputRootConfig(str(root["namespace"]), _as_path(root["path"], base_dir)) for root in inputs.get("roots", [])
    ]
    inputs["roots"] = [{"namespace": root.namespace, "path": str(root.path)} for root in roots]
    env_kwargs = dict(env.get("kwargs", {}))
    for key in (
        "env_root",
        "recording_folder",
        "unreal_env_root",
        "binary_path",
        "unrealzoo_gym_root",
        "scene_config_path",
        "data_root",
        "vlnce_data_root",
        "benchmark_config_path",
        "habitat_config_path",
        "scenes_dir",
    ):
        if env_kwargs.get(key):
            env_kwargs[key] = str(_as_path(env_kwargs[key], base_dir))
    env["kwargs"] = env_kwargs
    if env["type"] not in {"unrealcv", "airsim", "habitat", "offline"}:
        raise ValueError(f"Unknown simulator: {env['type']}")
    if env["type"] != "offline" and not env.get("backend_class_path"):
        raise ValueError("env.backend_class_path is required")
    gpu_ids = tuple(map(int, parallel["gpu_ids"]))
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("parallel.gpu_ids must be nonempty and unique")
    if int(benchmark["max_steps"]) <= 0:
        raise ValueError("benchmark.max_steps must be positive")
    if float(data.get("observation", {}).get("fps", 1)) <= 0 or float(model.get("request_timeout_sec", 120)) <= 0:
        raise ValueError("observation.fps and model.request_timeout_sec must be positive")
    if env_kwargs.get("execute_waypoints_per_step") is not None and int(env_kwargs["execute_waypoints_per_step"]) <= 0:
        raise ValueError("execute_waypoints_per_step must be positive")
    if output.get("action_observation_image_policy", "step") not in {"step", "action", "both", "none"}:
        raise ValueError("Invalid output.action_observation_image_policy")
    from benchmark.common.log.metrics import normalize_metric_keys

    metrics = normalize_metric_keys(output.get("metrics"))
    output["root"] = str(_as_path(output["root"], base_dir))
    driver_path = data.get("driver_paths_file")
    if driver_path:
        driver_path = str(_as_path(driver_path, base_dir))
    drivers = dict(data["driver_environment"]) if "driver_environment" in data else load_driver_environment(driver_path)
    resolved = {
        **data,
        "benchmark": benchmark,
        "input": inputs,
        "env": env,
        "output": output,
        "driver_paths_file": driver_path,
        "driver_environment": drivers,
    }
    return EvalConfig(
        benchmark=BenchmarkConfig(**benchmark),
        input=InputConfig(
            type=inputs["type"],
            adapter_class_path=inputs["adapter_class_path"],
            namespace=inputs.get("namespace"),
            path=Path(inputs["path"]) if inputs.get("path") else None,
            data_root=Path(inputs["data_root"]) if inputs.get("data_root") else None,
            split=inputs.get("split"),
            max_samples=inputs.get("max_samples"),
            roots=tuple(roots),
            raw=inputs,
        ),
        model=ModelConfig(**model),
        env=EnvConfig(
            type=env["type"],
            backend_class_path=env.get("backend_class_path"),
            planner_class_path=env.get("planner_class_path"),
            kwargs=env_kwargs,
            raw=env,
        ),
        parallel=ParallelConfig(gpu_ids, float(parallel.get("worker_timeout_sec", 0))),
        output=OutputConfig(
            root=Path(output["root"]),
            run_name=output["run_name"],
            save_step_artifacts=bool(output.get("save_step_artifacts", True)),
            save_images=bool(output.get("save_images", True)),
            image_cameras=tuple(output["image_cameras"]) if output.get("image_cameras") else None,
            action_observation_image_policy=output.get("action_observation_image_policy", "step"),
            metrics=metrics,
            raw=output,
        ),
        observation=dict(data.get("observation", {})),
        driver_environment=drivers,
        raw=resolved,
    )
