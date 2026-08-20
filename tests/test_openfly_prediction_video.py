from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tool.openfly_prediction_video import (
    find_stop_decision,
    longest_true_run,
    openfly_positions_xyz,
    smooth_and_resample_poses,
    tail4_max_segment_xyz_norm,
)


def _payload(segment_length: float) -> dict:
    xyz = np.zeros((8, 4), dtype=float)
    xyz[:, 0] = np.arange(8) * segment_length
    return {"action_waypoints": xyz.tolist()}


def test_tail4_measure_uses_adjacent_segments() -> None:
    assert np.isclose(tail4_max_segment_xyz_norm(_payload(0.25)["action_waypoints"]), 0.25)


def test_stop_decision_requires_three_consecutive_low_chunks(tmp_path: Path) -> None:
    values = [0.2, 0.4, 0.29, 0.30, 0.1]
    records = []
    for index, value in enumerate(values):
        path = tmp_path / f"{index * 8:06d}.json"
        path.write_text(json.dumps(_payload(value)), encoding="utf-8")
        records.append((path, _payload(value)))

    decision = find_stop_decision(records, threshold=0.31, confirmations=3)

    assert decision.stop_file_stem == 32
    assert decision.kept_chunk_count == 5
    assert np.allclose(decision.confirmation_values, [0.29, 0.30, 0.1])


def test_dense_fit_triples_count_and_preserves_endpoints() -> None:
    poses = np.asarray(
        [
            [0.0, 0.0, 0.0, 3.0],
            [1.0, 0.1, 0.0, 3.1],
            [2.0, 0.0, 0.1, -3.1],
            [3.0, 0.2, 0.0, -3.0],
            [4.0, 0.0, 0.0, -2.9],
        ],
        dtype=np.float32,
    )

    dense = smooth_and_resample_poses(poses, density_factor=3.0, smoothing_window=5)

    assert dense.shape == (15, 4)
    assert np.allclose(dense[0], poses[0])
    assert np.allclose(dense[-1], poses[-1])
    assert np.isfinite(dense).all()


def test_longest_true_run_selects_contiguous_quality_window() -> None:
    assert longest_true_run([False, True, True, False, True]) == (1, 3)


def test_openfly_positions_xyz_accepts_mixed_xyz_and_xyz_yaw() -> None:
    positions = openfly_positions_xyz([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0]])

    assert positions.shape == (2, 3)
    assert np.allclose(positions, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
