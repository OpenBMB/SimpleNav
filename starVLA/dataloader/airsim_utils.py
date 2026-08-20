from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image

EPS = 1e-6
MODE_MEAN_STD = "mean_std"
MODE_ANGLE_PI = "angle_pi"
MODE_Q01_Q99 = "q01_q99"
MODE_SCALE = "scale"


def wrap_to_pi_array(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def build_ego_relative_action_chunk(current_state: np.ndarray, future_states: np.ndarray) -> np.ndarray:
    current_state = np.asarray(current_state, dtype=np.float32)
    future_states = np.asarray(future_states, dtype=np.float32)
    if current_state.shape != (4,):
        raise ValueError(f"Expected current_state shape (4,), got {current_state.shape}")
    if future_states.ndim != 2 or future_states.shape[-1] != 4:
        raise ValueError(f"Expected future_states shape (H, 4), got {future_states.shape}")

    x_t, y_t, z_t, yaw_t = current_state
    dx_world = future_states[:, 0] - x_t
    dy_world = future_states[:, 1] - y_t
    dz = future_states[:, 2] - z_t
    dyaw = wrap_to_pi_array(future_states[:, 3] - yaw_t)

    cos_yaw = math.cos(float(yaw_t))
    sin_yaw = math.sin(float(yaw_t))
    dx_body = cos_yaw * dx_world + sin_yaw * dy_world
    dy_body = -sin_yaw * dx_world + cos_yaw * dy_world

    return np.stack([dx_body, dy_body, dz, dyaw], axis=-1).astype(np.float32)


def build_previous_step_action_chunk(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[-1] != 4:
        raise ValueError(f"Expected states shape (N, 4), got {states.shape}")
    if states.shape[0] < 2:
        raise ValueError(f"Expected at least 2 states, got {states.shape[0]}")

    chunks = [
        build_ego_relative_action_chunk(states[index - 1], states[index : index + 1])[0]
        for index in range(1, states.shape[0])
    ]
    return np.stack(chunks, axis=0).astype(np.float32)


def action_normalization_modes(action_type: str) -> list[str]:
    if action_type in {"ego_relative_xyz_yaw", "body_frame_xyz_yaw"}:
        return [MODE_Q01_Q99, MODE_Q01_Q99, MODE_Q01_Q99, MODE_ANGLE_PI]
    return [MODE_MEAN_STD, MODE_MEAN_STD, MODE_MEAN_STD, MODE_ANGLE_PI]


def repeated_action_modes(count: int, action_type: str = "body_frame_xyz_yaw") -> list[str]:
    return [mode for _ in range(count) for mode in action_normalization_modes(action_type)]


def build_stats(values: np.ndarray, modes: list[str], *, dim: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != dim:
        raise ValueError(f"Expected last dim {dim}, got {values.shape}")
    if len(modes) != dim:
        raise ValueError(f"Expected {dim} normalization modes, got {len(modes)}")
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "normalization_modes": modes,
        "mask": [True] * dim,
        "binary_mask": [False] * dim,
    }


def _get_modes(stats: dict[str, Any], dims: int) -> list[str]:
    modes = stats.get("normalization_modes")
    if modes is None:
        return [MODE_Q01_Q99] * dims
    modes = list(modes)
    if len(modes) != dims:
        raise ValueError(f"Expected {dims} normalization modes, got {len(modes)}")
    return modes


def normalize_array(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    normalized = values.copy()
    modes = _get_modes(stats, values.shape[-1])

    mean = np.asarray(stats.get("mean", np.zeros(values.shape[-1], dtype=np.float32)), dtype=np.float32)
    std = np.asarray(stats.get("std", np.ones(values.shape[-1], dtype=np.float32)), dtype=np.float32)
    scale = np.asarray(stats.get("scale", np.ones(values.shape[-1], dtype=np.float32)), dtype=np.float32)
    q01 = np.asarray(stats.get("q01", np.zeros(values.shape[-1], dtype=np.float32)), dtype=np.float32)
    q99 = np.asarray(stats.get("q99", np.ones(values.shape[-1], dtype=np.float32)), dtype=np.float32)

    for index, mode in enumerate(modes):
        if mode == MODE_MEAN_STD:
            normalized[..., index] = (normalized[..., index] - mean[..., index]) / np.maximum(std[..., index], EPS)
        elif mode == MODE_SCALE:
            normalized[..., index] = normalized[..., index] / np.maximum(scale[..., index], EPS)
        elif mode == MODE_ANGLE_PI:
            normalized[..., index] = wrap_to_pi_array(normalized[..., index]) / np.pi
        elif mode == MODE_Q01_Q99:
            axis_scale = np.maximum(q99[..., index] - q01[..., index], EPS)
            normalized[..., index] = 2 * ((normalized[..., index] - q01[..., index]) / axis_scale) - 1
        else:
            raise ValueError(f"Unsupported normalization mode: {mode}")

    return normalized.astype(np.float32)


def unnormalize_array(normalized_values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    values = np.asarray(normalized_values, dtype=np.float32).copy()
    modes = _get_modes(stats, values.shape[-1])

    mean = np.asarray(stats.get("mean", np.zeros(values.shape[-1], dtype=np.float32)), dtype=np.float32)
    std = np.asarray(stats.get("std", np.ones(values.shape[-1], dtype=np.float32)), dtype=np.float32)
    scale = np.asarray(stats.get("scale", np.ones(values.shape[-1], dtype=np.float32)), dtype=np.float32)
    q01 = np.asarray(stats.get("q01", np.zeros(values.shape[-1], dtype=np.float32)), dtype=np.float32)
    q99 = np.asarray(stats.get("q99", np.ones(values.shape[-1], dtype=np.float32)), dtype=np.float32)

    for index, mode in enumerate(modes):
        if mode == MODE_MEAN_STD:
            values[..., index] = values[..., index] * np.maximum(std[..., index], EPS) + mean[..., index]
        elif mode == MODE_SCALE:
            values[..., index] = values[..., index] * np.maximum(scale[..., index], EPS)
        elif mode == MODE_ANGLE_PI:
            values[..., index] = wrap_to_pi_array(values[..., index] * np.pi)
        elif mode == MODE_Q01_Q99:
            axis_scale = np.maximum(q99[..., index] - q01[..., index], EPS)
            values[..., index] = 0.5 * (values[..., index] + 1) * axis_scale + q01[..., index]
        else:
            raise ValueError(f"Unsupported normalization mode: {mode}")

    return values.astype(np.float32)


def config_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return value not in {"False", "false", "0", 0, False}


def resolve_obs_image_size(data_cfg: Any) -> tuple[int, int] | None:
    target = data_cfg.get("obs_image_size", None)
    if target is None:
        target = data_cfg.get("image_size", None)
    if target is None:
        return None
    if len(target) != 2:
        raise ValueError(f"obs_image_size must have two elements, got {target}")
    return int(target[0]), int(target[1])


def resize_image_tree(images: Any, target_size: tuple[int, int] | None) -> Any:
    if target_size is None:
        return images
    if isinstance(images, Image.Image):
        return images.resize(target_size)
    if isinstance(images, list):
        return [resize_image_tree(image, target_size) for image in images]
    if isinstance(images, tuple):
        return tuple(resize_image_tree(image, target_size) for image in images)
    return images
