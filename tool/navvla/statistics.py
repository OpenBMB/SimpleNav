from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

EPS = 1e-6
ACTION_DIM = 4
MODE_Q01_Q99 = "q01_q99"
MODE_ANGLE_PI = "angle_pi"
ACTION_NORMALIZATION_MODES = [MODE_Q01_Q99, MODE_Q01_Q99, MODE_Q01_Q99, MODE_Q01_Q99]


def wrap_to_pi(values: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def body_frame_action_from_pose(current_pose: Sequence[float], next_pose: Sequence[float]) -> np.ndarray:
    current = np.asarray(current_pose, dtype=np.float32)
    nxt = np.asarray(next_pose, dtype=np.float32)
    if current.shape[0] < 4 or nxt.shape[0] < 4:
        raise ValueError(f"pose must contain at least [x, y, z, yaw], got {current.shape} and {nxt.shape}")
    yaw = float(current[3])
    dx_world = float(nxt[0] - current[0])
    dy_world = float(nxt[1] - current[1])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.asarray(
        [
            cos_yaw * dx_world + sin_yaw * dy_world,
            -sin_yaw * dx_world + cos_yaw * dy_world,
            float(nxt[2] - current[2]),
            float(wrap_to_pi(float(nxt[3] - current[3]))),
        ],
        dtype=np.float32,
    )


def normalize_values(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    normalized = values.copy()
    modes = list(stats.get("normalization_modes", ACTION_NORMALIZATION_MODES))
    if len(modes) != values.shape[-1]:
        raise ValueError(f"Expected {values.shape[-1]} normalization modes, got {len(modes)}")
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    for index, mode in enumerate(modes):
        if mode == MODE_Q01_Q99:
            scale = np.maximum(q99[..., index] - q01[..., index], EPS)
            normalized[..., index] = 2.0 * ((normalized[..., index] - q01[..., index]) / scale) - 1.0
        elif mode == MODE_ANGLE_PI:
            normalized[..., index] = np.asarray(wrap_to_pi(normalized[..., index]), dtype=np.float32) / np.pi
        else:
            raise ValueError(f"Unsupported normalization mode: {mode}")
    return normalized.astype(np.float32)


def unnormalize_values(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    modes = list(stats.get("normalization_modes", ACTION_NORMALIZATION_MODES))
    if len(modes) != values.shape[-1]:
        raise ValueError(f"Expected {values.shape[-1]} normalization modes, got {len(modes)}")
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    for index, mode in enumerate(modes):
        if mode == MODE_Q01_Q99:
            scale = np.maximum(q99[..., index] - q01[..., index], EPS)
            values[..., index] = 0.5 * (values[..., index] + 1.0) * scale + q01[..., index]
        elif mode == MODE_ANGLE_PI:
            values[..., index] = np.asarray(wrap_to_pi(values[..., index] * np.pi), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported normalization mode: {mode}")
    return values.astype(np.float32)


def build_action_statistics(action_steps: np.ndarray) -> dict[str, Any]:
    actions = np.asarray(action_steps, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected action steps shape [N, 4], got {actions.shape}")
    if actions.shape[0] == 0:
        raise ValueError("cannot compute action statistics from zero valid action steps")
    return {
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "q01": np.quantile(actions, 0.01, axis=0).tolist(),
        "q99": np.quantile(actions, 0.99, axis=0).tolist(),
        "normalization_modes": ACTION_NORMALIZATION_MODES,
        "mask": [True] * ACTION_DIM,
        "binary_mask": [False] * ACTION_DIM,
    }


def build_repeated_state_statistics(action_stats: dict[str, Any], history_steps: int) -> dict[str, Any]:
    if history_steps < 0:
        raise ValueError(f"history_steps must be non-negative, got {history_steps}")
    state_stats: dict[str, Any] = {}
    for key in ("mean", "std", "min", "max", "q01", "q99"):
        single = np.asarray(action_stats[key], dtype=np.float32).reshape(ACTION_DIM)
        state_stats[key] = np.tile(single, history_steps).tolist()
    state_stats["normalization_modes"] = ACTION_NORMALIZATION_MODES * history_steps
    state_stats["mask"] = [True] * (ACTION_DIM * history_steps)
    state_stats["binary_mask"] = [False] * (ACTION_DIM * history_steps)
    return state_stats


def build_dataset_statistics(
    *,
    dataset_key: str,
    action_steps: np.ndarray,
    num_trajectories: int,
    num_transitions: int,
) -> dict[str, Any]:
    action_stats = build_action_statistics(action_steps)
    return {
        dataset_key: {
            "action": action_stats,
            "num_trajectories": int(num_trajectories),
            "num_transitions": int(num_transitions),
            "state_mode": "variable_bats_history_relative_body_frame_actions",
            "action_mode": "anchor_relative_body_frame_xyz_yaw",
            "action_anchor": "current_frame_pose",
            "action_padding_mask_policy": "all_false_zero_tail_unmasked",
        }
    }


def write_dataset_statistics(path: str | Path, stats: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def read_dataset_statistics(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flatten_valid_action_steps_from_rows(rows: pd.DataFrame) -> np.ndarray:
    steps: list[np.ndarray] = []
    for _idx, row in rows.iterrows():
        action = _action_step_array(row["action"])
        mask = np.asarray(row["action.padding_mask"], dtype=bool).reshape(-1)
        if action.shape[0] != mask.shape[0]:
            raise ValueError(f"action and padding mask length mismatch: {action.shape[0]} vs {mask.shape[0]}")
        if np.any(~mask):
            steps.append(action[~mask])
    if not steps:
        return np.zeros((0, ACTION_DIM), dtype=np.float32)
    return np.concatenate(steps, axis=0).astype(np.float32)


def flatten_valid_action_steps_from_episodes(episodes: Iterable[Any]) -> np.ndarray:
    steps: list[np.ndarray] = []
    for episode in episodes:
        for frame in episode.frames:
            if not frame.action_available or frame.action is None:
                continue
            action = _action_step_array(frame.action)
            if action.size:
                steps.append(action)
    if not steps:
        return np.zeros((0, ACTION_DIM), dtype=np.float32)
    return np.concatenate(steps, axis=0).astype(np.float32)


def _action_step_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        return np.zeros((0, ACTION_DIM), dtype=np.float32)
    if array.dtype == object and array.ndim == 1:
        rows = [np.asarray(item, dtype=np.float32).reshape(ACTION_DIM) for item in array]
        return np.stack(rows, axis=0).astype(np.float32)
    return array.astype(np.float32).reshape(-1, ACTION_DIM)
