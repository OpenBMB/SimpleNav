import unittest

import numpy as np

from vln_aug.trajectory import (
    TrajectoryConfig,
    _turn_intensity_from_xyz,
    smooth_and_retime,
    stable_trajectory_seed,
)


class TrajectoryTests(unittest.TestCase):
    def test_turn_intensity_handles_repeated_arc_at_spatial_cusp(self):
        xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        arc = np.array([0.0, 1.0, 1.0, 2.0])

        intensity = _turn_intensity_from_xyz(xyz, arc, TrajectoryConfig())

        self.assertEqual(intensity.shape, (4,))
        self.assertTrue(np.all(np.isfinite(intensity)))
        self.assertTrue(np.all((0.0 <= intensity) & (intensity <= 1.0)))

    def test_stable_seed_depends_on_dataset_and_episode(self):
        self.assertEqual(
            stable_trajectory_seed("OpenFly_lerobot", 372),
            stable_trajectory_seed("OpenFly_lerobot", 372),
        )
        self.assertNotEqual(
            stable_trajectory_seed("OpenFly_lerobot", 372),
            stable_trajectory_seed("AerialVLN_lerobot", 372),
        )

    def test_preserves_endpoints_and_builds_aligned_one_hz_controls(self):
        poses = np.array(
            [
                [0.0, 0.0, 2.0, 3.05],
                [2.0, 0.25, 2.0, 3.12],
                [4.0, -0.20, 2.0, -3.10],
                [6.0, 0.15, 2.0, -3.00],
                [10.0, 0.0, 2.0, -2.90],
            ]
        )
        cfg = TrajectoryConfig(
            speed_mean_mps=1.0,
            speed_std_mps=0.0,
            speed_min_mps=0.9,
            speed_max_mps=1.1,
            smoothing_strength=0.2,
        )

        result = smooth_and_retime(poses, cfg, seed=7)

        np.testing.assert_allclose(result.control_poses[0], poses[0], atol=1e-6)
        np.testing.assert_allclose(result.control_poses[result.movement_steps, :3], poses[-1, :3], atol=1e-6)
        self.assertEqual(result.total_steps, result.movement_steps)
        np.testing.assert_allclose(np.diff(result.control_times), 1.0)
        self.assertGreaterEqual(result.cruise_speed_mps, 0.9)
        self.assertLessEqual(result.cruise_speed_mps, 1.1)
        self.assertLessEqual(result.movement_speed_mps, result.cruise_speed_mps)
        self.assertGreaterEqual(
            result.minimum_local_speed_mps,
            cfg.turn_speed_min_factor * result.cruise_speed_mps - 1e-6,
        )
        self.assertTrue(np.all(np.isfinite(result.control_poses)))
        self.assertLess(np.max(np.abs(np.diff(result.control_poses[:, 3]))), np.pi)

    def test_real_terminal_is_preserved_without_synthetic_hover(self):
        poses = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [6.8, 0.0, 0.0, 0.0],
            ]
        )
        cfg = TrajectoryConfig(speed_mean_mps=1.0, speed_std_mps=0.0)

        result = smooth_and_retime(poses, cfg, seed=0)

        self.assertEqual(result.movement_steps, 7)
        self.assertEqual(result.total_steps, 7)
        self.assertEqual(len(result.control_poses), 8)
        np.testing.assert_allclose(result.control_poses[-1], poses[-1])

    def test_rejects_degenerate_position_trajectory(self):
        poses = np.array(
            [
                [1.0, 2.0, 3.0, 0.0],
                [1.0, 2.0, 3.0, 0.5],
            ]
        )
        with self.assertRaises(ValueError):
            smooth_and_retime(poses, TrajectoryConfig(), seed=0)

    def test_rejects_path_too_short_for_minimum_one_second_speed(self):
        poses = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0, 0.0],
            ]
        )
        with self.assertRaisesRegex(ValueError, "shorter than minimum one-second travel"):
            smooth_and_retime(poses, TrajectoryConfig(speed_min_mps=0.9), seed=0)

    def test_smoothed_curve_stays_inside_adaptive_source_corridor(self):
        poses = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.15, 0.0, 0.05],
                [2.0, -0.10, 0.0, -0.03],
                [3.0, 0.12, 0.0, 0.04],
                [4.0, 0.0, 0.0, 0.0],
            ]
        )
        result = smooth_and_retime(
            poses,
            TrajectoryConfig(smoothing_strength=10.0, max_deviation_m=0.08),
            seed=0,
        )
        self.assertLessEqual(result.max_deviation_m, 0.08 + 1e-6)

    def test_right_angle_corner_is_locally_smoothed_inside_corridor(self):
        poses = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 0.0, np.pi / 2],
            ]
        )

        result = smooth_and_retime(
            poses,
            TrajectoryConfig(
                smoothing_strength=0.1,
                max_deviation_m=0.3,
                speed_mean_mps=1.0,
                speed_std_mps=0.0,
            ),
            seed=0,
        )

        self.assertGreater(result.max_deviation_m, 0.01)
        self.assertLessEqual(result.max_deviation_m, 0.3 + 1e-6)
        dense = result.smoothed_dense_poses[:, :2]
        near_corner = dense[np.argmin(np.linalg.norm(dense - [5.0, 0.0], axis=1))]
        self.assertLess(near_corner[0], 5.0)
        self.assertGreater(near_corner[1], 0.0)

    def test_reported_speed_matches_continuous_smoothed_path_length(self):
        poses = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.1, 0.2, 0.0, 0.1],
                [2.0, 0.0, 0.0, 0.2],
            ]
        )
        result = smooth_and_retime(poses, TrajectoryConfig(), seed=3)
        actual_length = np.linalg.norm(
            np.diff(result.smoothed_dense_poses[:, :3], axis=0), axis=1
        ).sum()
        self.assertAlmostEqual(result.path_length_m, actual_length, places=6)
        self.assertAlmostEqual(
            result.movement_speed_mps, actual_length / result.movement_steps, places=6
        )

    def test_turn_aware_retiming_adds_points_and_slows_at_corners(self):
        poses = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [8.0, 0.0, 0.0, 0.0],
                [8.0, 8.0, 0.0, np.pi / 2],
                [16.0, 8.0, 0.0, 0.0],
            ]
        )
        result = smooth_and_retime(
            poses,
            TrajectoryConfig(
                speed_mean_mps=1.0,
                speed_std_mps=0.0,
                max_deviation_m=0.3,
                turn_speed_min_factor=0.5,
            ),
            seed=0,
        )

        step_distance = np.linalg.norm(np.diff(result.control_poses[:, :3], axis=0), axis=1)
        turn_mask = result.control_turn_intensity >= 0.5
        straight_mask = result.control_turn_intensity <= 0.05

        self.assertGreater(np.count_nonzero(turn_mask), 2)
        self.assertGreater(np.count_nonzero(straight_mask), 4)
        self.assertLess(
            float(np.median(step_distance[turn_mask])),
            0.8 * float(np.median(step_distance[straight_mask])),
        )
        self.assertLess(result.minimum_local_speed_mps, result.cruise_speed_mps)
        np.testing.assert_allclose(result.control_poses[0], poses[0], atol=1e-6)
        np.testing.assert_allclose(result.control_poses[-1, :3], poses[-1, :3], atol=1e-6)
        np.testing.assert_allclose(np.diff(result.control_times), 1.0)
        self.assertEqual(result.total_steps, result.movement_steps)

    def test_turn_slowdown_can_be_disabled_for_uniform_arc_spacing(self):
        poses = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 0.0, np.pi / 2],
            ]
        )
        result = smooth_and_retime(
            poses,
            TrajectoryConfig(
                speed_mean_mps=1.0,
                speed_std_mps=0.0,
                turn_slowdown_enabled=False,
            ),
            seed=0,
        )
        self.assertAlmostEqual(result.minimum_local_speed_mps, result.cruise_speed_mps)
        np.testing.assert_allclose(result.control_turn_intensity, 0.0)


if __name__ == "__main__":
    unittest.main()
