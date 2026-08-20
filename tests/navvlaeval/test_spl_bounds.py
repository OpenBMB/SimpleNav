from __future__ import annotations

from NavVLAeval.common.log.metrics import _all_episode_metrics


def test_spl_is_bounded_when_success_path_is_shorter_than_reference() -> None:
    metrics = _all_episode_metrics(
        {
            "success": 1,
            "oracle_success": 1,
            "final_distance": 0.0,
            "path_length": 0.01,
            "gt_path_length": 10.0,
            "steps": 1,
        }
    )

    assert metrics["SPL"] == 1.0
