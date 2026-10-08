from __future__ import annotations

from typing import Any

import numpy as np

from NavVLAeval.common.config import InputConfig, load_class
from NavVLAeval.common.types import EvalEpisode


def load_eval_episodes(cfg: InputConfig, *, max_samples: int | None = None) -> list[EvalEpisode]:
    adapter = load_class(cfg.adapter_class_path)()
    episodes = adapter.load_episodes(cfg, max_samples=max_samples)
    if not isinstance(episodes, list):
        raise TypeError("input adapter load_episodes() must return list[EvalEpisode]")
    for episode in episodes:
        payload = episode.payload
        points = np.asarray(payload["reference_points"], dtype=float)
        if points.size and (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()):
            raise ValueError(f"Invalid reference_points for {episode.episode_uid}")
    _validate_episodes(episodes)
    return episodes


def _validate_episodes(episodes: list[Any]) -> None:
    seen: set[str] = set()
    for index, episode in enumerate(episodes):
        if not isinstance(episode, EvalEpisode):
            raise TypeError(f"input adapter returned non-EvalEpisode at index {index}: {type(episode).__name__}")
        for field_name in (
            "episode_uid",
            "source_episode_id",
            "scene_id",
            "instruction",
            "source",
            "input_namespace",
            "input_root",
        ):
            value = getattr(episode, field_name)
            if not str(value).strip():
                raise ValueError(f"EvalEpisode {field_name} must be non-empty for {episode.episode_uid!r}")
        if episode.episode_uid in seen:
            raise ValueError(f"duplicate episode_uid: {episode.episode_uid}")
        seen.add(episode.episode_uid)
