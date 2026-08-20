"""Deterministic representative-episode selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .lerobot_io import EpisodeMetadata


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[EpisodeMetadata, ...]
    reason: str


def select_representative_episodes(
    episodes: Iterable[EpisodeMetadata],
) -> SelectionResult:
    """Select scene-diverse low/high-length representatives when possible."""

    records = sorted(episodes, key=_record_key)
    if not records:
        return SelectionResult((), "fallback_no_episodes_available")
    if len(records) == 1:
        return SelectionResult(
            (records[0],), "fallback_only_one_episode_available"
        )

    candidates = _quantile_window(records)
    pair = _widest_pair(candidates, require_scene_diversity=True, require_length_diversity=True)
    if pair is not None:
        return SelectionResult(
            _ordered_pair(pair),
            "selected_scene_diverse_lower_upper_length_quantiles",
        )

    pair = _widest_pair(candidates, require_scene_diversity=True)
    if pair is not None:
        return SelectionResult(_ordered_pair(pair), "fallback_no_length_diversity")

    pair = (candidates[0], candidates[-1])
    return SelectionResult(_ordered_pair(pair), "fallback_no_scene_diversity")


def _widest_pair(
    records: list[EpisodeMetadata],
    *,
    require_scene_diversity: bool,
    require_length_diversity: bool = False,
) -> tuple[EpisodeMetadata, EpisodeMetadata] | None:
    """Find the deterministic maximum-span eligible pair in linear time."""

    earliest: EpisodeMetadata | None = None
    earliest_other_scene: EpisodeMetadata | None = None
    best: tuple[EpisodeMetadata, EpisodeMetadata] | None = None
    best_key: tuple[int, int, int, str, str] | None = None
    for high in records:
        low = earliest
        if require_scene_diversity and low is not None and low.scene_id == high.scene_id:
            low = earliest_other_scene
        if low is not None and not (
            require_length_diversity and low.length == high.length
        ):
            candidate_key = (
                -(high.length - low.length),
                low.episode_index,
                high.episode_index,
                low.episode_id,
                high.episode_id,
            )
            if best_key is None or candidate_key < best_key:
                best = low, high
                best_key = candidate_key
        if earliest is None:
            earliest = high
        elif earliest_other_scene is None and high.scene_id != earliest.scene_id:
            earliest_other_scene = high
    return best


def _quantile_window(records: list[EpisodeMetadata]) -> list[EpisodeMetadata]:
    if len(records) < 20:
        return records
    low = int(len(records) * 0.10)
    high = max(low + 1, int(len(records) * 0.90) - 1)
    return records[low : high + 1]


def _ordered_pair(
    pair: tuple[EpisodeMetadata, EpisodeMetadata],
) -> tuple[EpisodeMetadata, EpisodeMetadata]:
    return tuple(sorted(pair, key=_record_key))


def _record_key(item: EpisodeMetadata) -> tuple[int, int, str]:
    return item.length, item.episode_index, item.episode_id
