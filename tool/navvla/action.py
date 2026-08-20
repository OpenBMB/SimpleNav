from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ActionChunk:
    values: np.ndarray
    padding_mask: np.ndarray
    action_available: bool


def resolve_timestamp(source_timestamp: float | int | None, *, frame_index: int, fps: float) -> float:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if source_timestamp is not None:
        return float(source_timestamp)
    return float(frame_index) / float(fps)


def pad_action_chunk(
    action: Sequence[Sequence[float]] | np.ndarray | None,
    *,
    horizon: int = 8,
    action_dim: int = 4,
    action_available: bool = True,
) -> ActionChunk:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")

    values = np.zeros((horizon, action_dim), dtype=np.float32)
    padding_mask = np.zeros((horizon,), dtype=bool)

    if action is None:
        return ActionChunk(values=values, padding_mask=padding_mask, action_available=bool(action_available))

    action_array = np.asarray(action, dtype=np.float32)
    if action_array.size == 0:
        return ActionChunk(values=values, padding_mask=padding_mask, action_available=bool(action_available))
    if action_array.ndim != 2 or action_array.shape[1] != action_dim:
        raise ValueError(f"action must have shape [N, action_dim={action_dim}], got {action_array.shape}")
    if action_array.shape[0] > horizon:
        action_array = action_array[:horizon]

    count = int(action_array.shape[0])
    if count:
        values[:count] = action_array
    return ActionChunk(values=values, padding_mask=padding_mask, action_available=bool(action_available))
