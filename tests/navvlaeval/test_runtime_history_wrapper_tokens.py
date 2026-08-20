from __future__ import annotations

import numpy as np

import NavVLAeval.common.data.runtime_history as runtime_history
from NavVLAeval.common.data.runtime_dataset import get_runtime_dataset_adapter
from NavVLAeval.common.data.runtime_history import OnlineHistoryConfig, build_online_navvla_history_sample
from NavVLAeval.common.types import EpisodeHistory
from starVLA.model.modules.bats import BATSSelectionResult


def test_runtime_adapter_propagates_visual_wrapper_tokens() -> None:
    adapter = get_runtime_dataset_adapter(
        {
            "runtime_adapter": "aerialvln",
            "current_wrapper_tokens": 2,
            "history_wrapper_tokens": 0,
        }
    )

    assert adapter.history_config.current_wrapper_tokens == 2
    assert adapter.history_config.history_wrapper_tokens == 0


def test_online_bats_uses_configured_visual_wrapper_tokens(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_select_bats_history(**kwargs):
        captured.update(kwargs)
        return BATSSelectionResult([], [], 0.0, 191)

    monkeypatch.setattr(runtime_history, "select_bats_history", fake_select_bats_history)
    build_online_navvla_history_sample(
        observation={
            "image": np.ones((4, 4, 3), dtype=np.uint8),
            "navvla_eval": {"frame_index": 1, "timestamp": 1.0},
        },
        history=EpisodeHistory(
            observations=[
                {
                    "navvla_online_visual_tokens": {
                        "front": np.ones((4, 2), dtype=np.float16),
                    },
                    "navvla_eval": {"frame_index": 0, "timestamp": 0.0},
                }
            ]
        ),
        instruction="go",
        state=None,
        config=OnlineHistoryConfig(
            history_policy="bats",
            current_wrapper_tokens=2,
            history_wrapper_tokens=0,
        ),
    )

    assert captured["current_wrapper_tokens"] == 2
    assert captured["history_wrapper_tokens"] == 0
