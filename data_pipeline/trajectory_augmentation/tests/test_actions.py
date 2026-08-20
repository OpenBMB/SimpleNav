import unittest

import numpy as np

from vln_aug.actions import (
    build_action_chunk,
    build_observation_actions,
    random_observation_indices,
)
from vln_aug.image_stride import stable_image_interval_seed


class ActionTests(unittest.TestCase):
    def test_converts_future_world_waypoints_to_anchor_body_frame(self):
        poses = np.array(
            [
                [0.0, 0.0, 0.0, np.pi / 2],
                [0.0, 1.0, 0.0, np.pi / 2],
                [0.0, 2.0, 0.0, np.pi / 2],
            ]
        )

        chunk = build_action_chunk(poses, anchor_index=0, horizon=2)

        np.testing.assert_allclose(chunk[:, 0], [1.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(chunk[:, 1:], 0.0, atol=1e-6)

    def test_tail_repeats_last_valid_absolute_waypoint(self):
        poses = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
            ]
        )

        chunk = build_action_chunk(poses, anchor_index=1, horizon=8)

        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_allclose(chunk, expected)

    def test_terminal_row_repeats_terminal_relative_waypoint(self):
        poses = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        chunk = build_action_chunk(poses, anchor_index=1, horizon=8)

        np.testing.assert_allclose(chunk, np.zeros((8, 4)))

    def test_observation_rows_are_sampled_every_five_control_steps(self):
        poses = np.zeros((11, 4), dtype=float)
        poses[:, 0] = np.arange(11)

        indices, actions = build_observation_actions(poses, render_stride=5, horizon=8)

        np.testing.assert_array_equal(indices, [0, 5, 10])
        self.assertEqual(actions.shape, (3, 8, 4))

    def test_observation_rows_include_non_aligned_real_terminal(self):
        poses = np.zeros((8, 4), dtype=float)
        poses[:, 0] = np.arange(8)

        indices, actions = build_observation_actions(poses, render_stride=5, horizon=8)

        np.testing.assert_array_equal(indices, [0, 5, 7])
        self.assertEqual(actions.shape, (3, 8, 4))
        np.testing.assert_allclose(actions[-1], np.zeros((8, 4)))

    def test_random_observation_indices_are_reproducible_and_include_terminal(self):
        first = random_observation_indices(80, (5, 6, 7, 8), seed=20260718)
        second = random_observation_indices(80, (5, 6, 7, 8), seed=20260718)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(int(first[0]), 0)
        self.assertEqual(int(first[-1]), 79)
        gaps = np.diff(first)
        self.assertTrue(np.all(gaps > 0))
        self.assertTrue(all(int(gap) in (5, 6, 7, 8) for gap in gaps[:-1]))
        self.assertLessEqual(int(gaps[-1]), 8)

    def test_random_observation_indices_change_with_seed(self):
        first = random_observation_indices(200, (5, 6, 7, 8), seed=1)
        second = random_observation_indices(200, (5, 6, 7, 8), seed=2)

        self.assertFalse(np.array_equal(first, second))

    def test_random_observation_indices_handle_short_trajectory(self):
        indices = random_observation_indices(4, (5, 6, 7, 8), seed=123)

        np.testing.assert_array_equal(indices, [0, 3])

    def test_image_interval_seed_is_stable_per_dataset_episode_and_base_seed(self):
        first = stable_image_interval_seed("dataset", "episode-1", 20260718)
        second = stable_image_interval_seed("dataset", "episode-1", 20260718)

        self.assertEqual(first, second)
        self.assertNotEqual(
            first, stable_image_interval_seed("dataset", "episode-2", 20260718)
        )


if __name__ == "__main__":
    unittest.main()
