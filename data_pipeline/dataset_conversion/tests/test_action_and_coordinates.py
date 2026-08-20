from __future__ import annotations

import numpy as np
import pytest

from navvla_conversion.action import pad_action_chunk
from navvla_conversion.adapters.enhanced_vln import render_poses_to_training_local
from navvla_conversion.statistics import body_frame_action_from_pose


def test_pad_action_chunk_zero_fills_without_marking_padding() -> None:
    chunk = pad_action_chunk([[1.0, 2.0, 3.0, 4.0]], horizon=3, action_dim=4)
    np.testing.assert_allclose(
        chunk.values,
        [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    )
    assert chunk.padding_mask.tolist() == [False, False, False]


def test_body_frame_action_respects_anchor_yaw() -> None:
    action = body_frame_action_from_pose([0.0, 0.0, 1.0, np.pi / 2], [0.0, 2.0, 0.0, np.pi])
    np.testing.assert_allclose(action, [2.0, 0.0, -1.0, np.pi / 2], atol=1e-6)


def test_enhanced_openfly_pose_is_first_body_aligned_frd() -> None:
    poses = [[10.0, 20.0, -4.0, np.pi / 2], [10.0, 22.0, -3.0, np.pi / 2]]
    local = render_poses_to_training_local(poses, dataset_key="OpenFly_lerobot")
    np.testing.assert_allclose(local[0], [0.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(local[1], [2.0, 0.0, 1.0, 0.0], atol=1e-6)


def test_enhanced_pose_rejects_unknown_scene_family() -> None:
    with pytest.raises(ValueError, match="unsupported enhanced dataset_key"):
        render_poses_to_training_local([[0, 0, 0, 0], [1, 0, 0, 0]], dataset_key="unknown")
