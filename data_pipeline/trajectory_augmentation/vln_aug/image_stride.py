"""Stable dataset-level image collection stride assignment."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def normalize_image_stride_choices(choices: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in choices)
    if not normalized:
        raise ValueError("image stride choices must not be empty")
    if any(value <= 0 for value in normalized):
        raise ValueError("image stride choices must be positive")
    if len(set(normalized)) != len(normalized):
        raise ValueError("image stride choices must be unique")
    return normalized


def assign_image_stride(
    dataset_key: str,
    episode_id: str,
    choices: Sequence[int] = (1, 3, 5),
) -> int:
    normalized = normalize_image_stride_choices(choices)
    payload = f"{dataset_key}:{episode_id}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return normalized[bucket % len(normalized)]


def stable_image_interval_seed(
    dataset_key: str,
    episode_id: str,
    base_seed: int = 0,
) -> int:
    """Return a reproducible integer seed for one episode's interval schedule."""
    payload = f"{dataset_key}:{episode_id}:{int(base_seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
