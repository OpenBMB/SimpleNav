"""Portable dataset profiles for the trajectory augmentation pipeline."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vln_aug.trajectory import TrajectoryConfig


SUPPORTED_PROFILE_VERSION = 1
SUPPORTED_POSE_MODES = {
    "adapter",
    "auto",
    "frame-metadata",
    "observation-state",
}
SUPPORTED_TRANSFORMS = {"identity", "reflect-y-yaw", "reflect-y-z-yaw"}


@dataclass(frozen=True)
class DatasetProfile:
    path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class ExportPlan:
    profile_path: Path
    dataset_root: Path
    dataset_key: str
    source_split: Path
    output_dir: Path
    world_pose_mode: str
    world_pose_adapter: Any | None
    export_kwargs: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "profile": str(self.profile_path),
            "dataset_root": str(self.dataset_root),
            "dataset_key": self.dataset_key,
            "source_split": str(self.source_split),
            "output_dir": str(self.output_dir),
            "world_pose_mode": self.world_pose_mode,
            "world_pose_adapter": (
                type(self.world_pose_adapter).__module__
                + ":"
                + type(self.world_pose_adapter).__qualname__
                if self.world_pose_adapter is not None
                else None
            ),
            "world_pose_source_kind": self.export_kwargs.get(
                "world_pose_source_kind"
            ),
            "coordinate_alignment_transform": self.export_kwargs.get(
                "coordinate_alignment_transform"
            ),
            "render_coordinate_transform": self.export_kwargs.get(
                "render_coordinate_transform"
            ),
            "image_stride_choices": list(
                self.export_kwargs["image_stride_choices"]
            ),
            "image_stride_policy": self.export_kwargs["image_stride_policy"],
            "render_image_width": self.export_kwargs["render_image_width"],
            "render_image_height": self.export_kwargs["render_image_height"],
            "require_enhanced_sibling": self.export_kwargs[
                "require_enhanced_sibling"
            ],
        }


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"profile field {key!r} must be an object")
    return value


def _require_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"profile field {key!r} must be a non-empty string")
    return value.strip()


def _validate_transform(world_pose: dict[str, Any], key: str) -> str:
    value = str(world_pose.get(key, "identity"))
    if value not in SUPPORTED_TRANSFORMS:
        raise ValueError(
            f"world_pose.{key} must be one of {sorted(SUPPORTED_TRANSFORMS)}"
        )
    return value


def load_dataset_profile(path: str | Path) -> DatasetProfile:
    profile_path = Path(path).resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset profile must contain a JSON object")
    if int(payload.get("schema_version", -1)) != SUPPORTED_PROFILE_VERSION:
        raise ValueError(
            f"profile schema_version must be {SUPPORTED_PROFILE_VERSION}"
        )
    _require_text(payload, "dataset_key")
    paths = _require_mapping(payload, "paths")
    _require_text(paths, "train_split")
    _require_text(paths, "output_dir")
    world_pose = _require_mapping(payload, "world_pose")
    mode = _require_text(world_pose, "mode")
    if mode not in SUPPORTED_POSE_MODES:
        raise ValueError(
            f"world_pose.mode must be one of {sorted(SUPPORTED_POSE_MODES)}"
        )
    if mode == "adapter":
        _require_text(world_pose, "class")
        _require_text(world_pose, "path")
        _require_text(world_pose, "source_kind")
        options = world_pose.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("world_pose.options must be an object")
    _validate_transform(world_pose, "alignment_transform")
    _validate_transform(world_pose, "render_transform")
    return DatasetProfile(path=profile_path, payload=payload)


def _resolve_inside(dataset_root: Path, relative: str, *, field: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"{field} must be relative to dataset root")
    resolved = (dataset_root / candidate).resolve()
    if resolved != dataset_root and dataset_root not in resolved.parents:
        raise ValueError(f"{field} must stay inside dataset root")
    return resolved


def _load_adapter(spec: str, path: Path, options: dict[str, Any]) -> Any:
    if ":" not in spec:
        raise ValueError("world_pose.class must use module:Class syntax")
    module_name, attribute_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    adapter_type = module
    for component in attribute_name.split("."):
        adapter_type = getattr(adapter_type, component)
    adapter = adapter_type(path, **options)
    if not callable(getattr(adapter, "poses_for_episode", None)):
        raise TypeError("world pose adapter must define poses_for_episode(metadata)")
    return adapter


def _string_set(value: Any, *, field: str) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, list) or not all(isinstance(item, (str, int)) for item in value):
        raise ValueError(f"{field} must be a list of strings or integers")
    return {str(item) for item in value}


def _integer_set(value: Any, *, field: str) -> set[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"{field} must be a list of integers")
    return {int(item) for item in value}


def build_export_plan(
    profile: DatasetProfile,
    *,
    dataset_root: str | Path,
) -> ExportPlan:
    root = Path(dataset_root).resolve()
    payload = profile.payload
    paths = payload["paths"]
    source_split = _resolve_inside(root, paths["train_split"], field="paths.train_split")
    output_dir = _resolve_inside(root, paths["output_dir"], field="paths.output_dir")
    if output_dir == source_split or source_split in output_dir.parents:
        raise ValueError("paths.output_dir must be outside the source split")

    world_pose = payload["world_pose"]
    pose_mode = world_pose["mode"]
    adapter = None
    world_pose_metadata_path = None
    original_trajectory_json = None
    world_pose_source = pose_mode
    source_kind = None
    alignment_transform = _validate_transform(world_pose, "alignment_transform")
    render_transform = _validate_transform(world_pose, "render_transform")
    render_pose_source = world_pose.get("render_pose_source")
    if pose_mode == "adapter":
        adapter_path = _resolve_inside(
            root, world_pose["path"], field="world_pose.path"
        )
        adapter = _load_adapter(
            world_pose["class"], adapter_path, world_pose.get("options", {})
        )
        world_pose_source = "auto"
        source_kind = world_pose["source_kind"]
    elif pose_mode == "frame-metadata" and world_pose.get("path"):
        world_pose_metadata_path = _resolve_inside(
            root, world_pose["path"], field="world_pose.path"
        )

    sampling = payload.get("sampling", {})
    if not isinstance(sampling, dict):
        raise ValueError("sampling must be an object")
    stride_choices = sampling.get("image_stride_choices", [1, 3, 5])
    if not isinstance(stride_choices, list) or not stride_choices:
        raise ValueError("sampling.image_stride_choices must be a non-empty list")

    selection = payload.get("selection", {})
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object")
    eligible_package = None
    if selection.get("eligible_package"):
        eligible_package = _resolve_inside(
            root,
            selection["eligible_package"],
            field="selection.eligible_package",
        )

    trajectory = payload.get("trajectory", {})
    if not isinstance(trajectory, dict):
        raise ValueError("trajectory must be an object")
    config_fields = {
        "control_frequency_hz",
        "speed_mean_mps",
        "speed_std_mps",
        "speed_min_mps",
        "speed_max_mps",
        "smoothing_strength",
        "dense_samples_per_meter",
        "max_deviation_m",
        "turn_slowdown_enabled",
        "turn_speed_min_factor",
        "turn_curvature_start_rad_per_m",
        "turn_curvature_full_rad_per_m",
        "turn_smoothing_multiplier",
    }
    unknown_trajectory_fields = set(trajectory) - config_fields
    if unknown_trajectory_fields:
        raise ValueError(
            "unsupported trajectory fields: "
            + ", ".join(sorted(unknown_trajectory_fields))
        )
    trajectory_config = TrajectoryConfig(**trajectory)

    render = payload.get("render", {})
    if not isinstance(render, dict):
        raise ValueError("render must be an object")
    render_image_width = int(render.get("image_width", 224))
    render_image_height = int(render.get("image_height", 224))
    if render_image_width <= 0 or render_image_height <= 0:
        raise ValueError("render image dimensions must be positive")
    if render_image_width % 2 or render_image_height % 2:
        raise ValueError("render image dimensions must be even for YUV420P video")

    export_kwargs = {
        "require_enhanced_sibling": bool(
            payload.get("require_enhanced_sibling", True)
        ),
        "config": trajectory_config,
        "image_stride_choices": tuple(int(value) for value in stride_choices),
        "image_stride_policy": sampling.get(
            "image_stride_policy", "fixed-per-episode"
        ),
        "image_interval_seed": int(sampling.get("image_interval_seed", 0)),
        "render_image_width": render_image_width,
        "render_image_height": render_image_height,
        "retain_fraction": selection.get("retain_fraction"),
        "excluded_scene_ids": _string_set(
            selection.get("exclude_scene_ids"), field="selection.exclude_scene_ids"
        ),
        "include_scene_ids": _string_set(
            selection.get("include_scene_ids"), field="selection.include_scene_ids"
        ),
        "selection_seed": int(selection.get("seed", 0)),
        "balanced_image_strides": bool(
            selection.get("balanced_image_strides", False)
        ),
        "eligible_episode_indices": None,
        "sample_per_stride": int(selection.get("sample_per_stride", 0)),
        "sample_episode_indices": _integer_set(
            payload.get("sample_episode_indices"), field="sample_episode_indices"
        )
        or set(),
        "include_episode_indices": _integer_set(
            selection.get("include_episode_indices"),
            field="selection.include_episode_indices",
        ),
        "world_pose_source": world_pose_source,
        "world_pose_metadata_path": world_pose_metadata_path,
        "original_trajectory_json": original_trajectory_json,
        "world_pose_adapter": adapter,
        "world_pose_source_kind": source_kind,
        "coordinate_alignment_transform": alignment_transform,
        "render_coordinate_transform": render_transform,
        "render_pose_source": render_pose_source,
    }
    if eligible_package is not None:
        from vln_aug.lightweight_subset import read_eligible_episode_indices

        export_kwargs["eligible_episode_indices"] = read_eligible_episode_indices(
            eligible_package
        )

    return ExportPlan(
        profile_path=profile.path,
        dataset_root=root,
        dataset_key=payload["dataset_key"],
        source_split=source_split,
        output_dir=output_dir,
        world_pose_mode=pose_mode,
        world_pose_adapter=adapter,
        export_kwargs=export_kwargs,
    )


def validate_export_plan(plan: ExportPlan, *, require_new_output: bool = True) -> dict[str, Any]:
    errors = []
    info_path = plan.source_split / "meta" / "info.json"
    episode_root = plan.source_split / "meta" / "episodes"
    if not info_path.is_file():
        errors.append(f"missing LeRobot metadata: {info_path}")
    if not episode_root.is_dir():
        errors.append(f"missing LeRobot episode metadata directory: {episode_root}")
    if require_new_output and (plan.output_dir.exists() or plan.output_dir.is_symlink()):
        errors.append(f"output already exists: {plan.output_dir}")
    return {**plan.summary(), "valid": not errors, "errors": errors}
