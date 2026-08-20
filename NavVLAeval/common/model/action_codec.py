from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from NavVLAeval.common.types import ActionPrediction
from starVLA.dataloader.airsim_utils import unnormalize_array


def _stats_candidates(checkpoint: str | Path) -> list[Path]:
    path = Path(str(checkpoint))
    if path.is_file():
        return [path.parent / "dataset_statistics.json", path.parent.parent / "dataset_statistics.json"]
    return [path / "dataset_statistics.json", path.parent / "dataset_statistics.json"]


def _load_dataset_statistics(checkpoint: str | Path) -> dict[str, Any]:
    for candidate in _stats_candidates(checkpoint):
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
    searched = ", ".join(str(path) for path in _stats_candidates(checkpoint))
    raise FileNotFoundError(f"dataset_statistics.json not found; searched: {searched}")


class ActionCodec:
    def __init__(self, action_stats: dict[str, Any], *, framework_name: str | None = None):
        self.action_stats = action_stats
        self.framework_name = framework_name

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        unnorm_key: str | None,
        framework_name: str | None = None,
        model_outputs_raw: bool = False,
    ) -> "ActionCodec":
        if model_outputs_raw:
            raise ValueError("raw model outputs are not supported in AirSim eval; expected normalized_actions")
        stats = _load_dataset_statistics(checkpoint)
        if unnorm_key is None:
            if len(stats) != 1:
                raise ValueError(f"model.unnorm_key is required; available keys: {list(stats.keys())}")
            unnorm_key = next(iter(stats))
        if unnorm_key not in stats:
            raise KeyError(f"unnorm_key {unnorm_key!r} not found in dataset statistics: {list(stats.keys())}")
        return cls(stats[unnorm_key]["action"], framework_name=framework_name)

    def decode(self, model_response: dict[str, Any]) -> ActionPrediction:
        if "normalized_actions" not in model_response:
            raise KeyError("model response must contain normalized_actions")
        normalized_actions = np.asarray(model_response["normalized_actions"], dtype=np.float32)
        raw_actions = unnormalize_array(normalized_actions, self.action_stats)
        metadata = dict(model_response.get("metadata") or {})
        metadata["framework_name"] = self.framework_name
        return ActionPrediction(
            normalized_actions=normalized_actions,
            raw_actions=raw_actions,
            metadata=metadata,
        )
