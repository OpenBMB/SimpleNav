"""Deterministic lightweight dataset selection and balanced stride planning."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from vln_aug.image_stride import normalize_image_stride_choices
from vln_aug.lerobot_io import EpisodeMetadata


@dataclass(frozen=True)
class LightweightSubsetPlan:
    source_episode_count: int
    target_episode_count: int
    excluded_scene_episode_count: int
    selected_episode_indices: tuple[int, ...]
    stride_by_episode_index: dict[int, int]
    stride_episode_counts: dict[int, int]
    excluded_scene_ids: tuple[str, ...]
    seed: int
    retain_fraction: float


def plan_lightweight_subset(
    episodes: Iterable[EpisodeMetadata],
    *,
    retain_fraction: float,
    excluded_scene_ids: set[str] | Sequence[str] = (),
    seed: int,
    stride_choices: Sequence[int],
) -> LightweightSubsetPlan:
    fraction = float(retain_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("retain_fraction must be in (0, 1]")
    choices = normalize_image_stride_choices(stride_choices)
    excluded = {str(value) for value in excluded_scene_ids}
    records = sorted(episodes, key=lambda item: item.episode_index)
    indices = [item.episode_index for item in records]
    if len(indices) != len(set(indices)):
        raise ValueError("episode indices must be unique")
    target = int(len(records) * fraction)
    if target <= 0:
        raise ValueError("retain_fraction selects zero episodes")
    candidates = [item for item in records if str(item.scene_id) not in excluded]
    if len(candidates) < target:
        raise ValueError(
            "too few eligible episodes after scene exclusion: "
            f"need {target}, found {len(candidates)}"
        )

    rng = random.Random(int(seed))
    sampled = rng.sample(candidates, target)
    stride_by_episode_index = {
        item.episode_index: choices[position % len(choices)]
        for position, item in enumerate(sampled)
    }
    selected_indices = tuple(sorted(stride_by_episode_index))
    stride_counts = Counter(stride_by_episode_index.values())
    return LightweightSubsetPlan(
        source_episode_count=len(records),
        target_episode_count=target,
        excluded_scene_episode_count=len(records) - len(candidates),
        selected_episode_indices=selected_indices,
        stride_by_episode_index=stride_by_episode_index,
        stride_episode_counts={
            stride: int(stride_counts.get(stride, 0)) for stride in choices
        },
        excluded_scene_ids=tuple(sorted(excluded)),
        seed=int(seed),
        retain_fraction=fraction,
    )


def read_eligible_episode_indices(package_dir: Path) -> set[int]:
    metadata_path = (
        Path(package_dir) / "trajectories" / "augmentation_metadata.jsonl"
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"eligible package metadata does not exist: {metadata_path}"
        )
    indices = set()
    with metadata_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            index = int(payload["source_episode_index"])
            if index in indices:
                raise ValueError(
                    f"duplicate source_episode_index {index} at line {line_number}"
                )
            indices.add(index)
    if not indices:
        raise ValueError("eligible package contains no episode metadata")
    return indices
