import tempfile
import unittest
from pathlib import Path

import numpy as np

from vln_aug.trajectory import RetimedTrajectory
from vln_aug.visualize import (
    build_sampling_audit_figure,
    build_trajectory_comparison_figure,
    compute_sampling_audit,
    compute_deviation_profile,
    compute_trajectory_metrics,
    display_indices,
    plot_trajectory_comparison,
    plot_sampling_audit,
)


class VisualizationTests(unittest.TestCase):
    def _result(self):
        source = np.array([[0.0, 0.0, 0.0, 0.0], [5.0, 0.5, 1.0, 0.1]])
        dense = np.array([[0.0, 0.0, 0.0, 0.0], [2.5, 0.1, 0.5, 0.05], [5.0, 0.5, 1.0, 0.1]])
        controls = np.zeros((6, 4), dtype=float)
        controls[:, 0] = np.arange(6)
        controls[:, 1] = np.linspace(0.0, 0.5, 6)
        controls[:, 2] = np.linspace(0.0, 1.0, 6)
        controls[:, 3] = np.linspace(0.0, 0.1, 6)
        return RetimedTrajectory(
            source_poses=source,
            smoothed_dense_poses=dense,
            control_poses=controls,
            control_times=np.arange(6, dtype=float),
            movement_steps=5,
            total_steps=5,
            path_length_m=5.1,
            movement_speed_mps=1.02,
            max_deviation_m=0.05,
        )

    def test_computes_core_metrics(self):
        metrics = compute_trajectory_metrics(self._result())
        self.assertEqual(metrics["control_frequency_hz"], 1.0)
        self.assertEqual(metrics["render_frame_count"], 2)
        self.assertAlmostEqual(metrics["movement_speed_mps"], 1.02)

    def test_sampling_audit_reports_every_five_plus_terminal(self):
        controls = np.zeros((13, 4), dtype=float)
        controls[:, 0] = np.arange(13) * 0.98
        result = self._result()
        result = RetimedTrajectory(
            source_poses=result.source_poses,
            smoothed_dense_poses=controls,
            control_poses=controls,
            control_times=np.arange(13, dtype=float),
            movement_steps=12,
            total_steps=12,
            path_length_m=11.76,
            movement_speed_mps=0.98,
            max_deviation_m=0.0,
        )

        audit = compute_sampling_audit(result)

        self.assertEqual(audit["image_waypoint_indices"], [0, 5, 10, 12])
        self.assertEqual(audit["image_waypoint_index_gaps"], [5, 5, 2])
        self.assertTrue(audit["regular_image_gaps_are_five"])
        self.assertTrue(audit["real_terminal_included"])
        self.assertAlmostEqual(audit["target_arc_step_m"], 0.98)

    def test_sampling_audit_accepts_episode_specific_stride(self):
        controls = np.zeros((11, 4), dtype=float)
        controls[:, 0] = np.arange(11)
        base = self._result()
        result = RetimedTrajectory(
            source_poses=base.source_poses,
            smoothed_dense_poses=controls,
            control_poses=controls,
            control_times=np.arange(11, dtype=float),
            movement_steps=10,
            total_steps=10,
            path_length_m=10.0,
            movement_speed_mps=1.0,
            max_deviation_m=0.0,
        )

        audit = compute_sampling_audit(result, image_stride=3)
        metrics = compute_trajectory_metrics(result, image_stride=3)

        self.assertEqual(audit["image_waypoint_indices"], [0, 3, 6, 9, 10])
        self.assertTrue(audit["regular_image_gaps_match_stride"])
        self.assertEqual(audit["image_stride_waypoints"], 3)
        self.assertEqual(metrics["image_stride_waypoints"], 3)
        self.assertEqual(metrics["render_frame_count"], 5)

    def test_writes_before_after_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "comparison.png"
            plot_trajectory_comparison(self._result(), output, title="episode")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)

    def test_writes_sampling_audit_png_with_index_and_local_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sampling_audit.png"
            plot_sampling_audit(self._result(), output, title="episode")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)

        figure, axes, audit = build_sampling_audit_figure(self._result(), title="episode")
        try:
            self.assertEqual(set(axes), {"index", "speed", "tightest", "terminal"})
            self.assertEqual(audit["image_waypoint_indices"], [0, 5])
            self.assertIn("Image waypoint index gaps", axes["index"].get_title())
            self.assertIn("step distance", axes["speed"].get_title())
            self.assertIn("projection", axes["tightest"].get_title())
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_deviation_profile_has_progress_and_detects_offset(self):
        source = np.array(
            [[0.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]], dtype=float
        )
        smoothed = np.array(
            [[0.0, 0.0, 0.0, 0.0], [2.5, 0.2, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]],
            dtype=float,
        )
        profile = compute_deviation_profile(source, smoothed)
        np.testing.assert_allclose(profile.progress_percent, [0.0, 50.0, 100.0])
        self.assertGreater(profile.distance_m[1], 0.19)
        self.assertEqual(profile.max_index, 1)

    def test_deviation_is_to_source_polyline_not_only_source_vertices(self):
        source = np.array(
            [[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]], dtype=float
        )
        smoothed = np.array(
            [[0.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]],
            dtype=float,
        )
        profile = compute_deviation_profile(source, smoothed)
        np.testing.assert_allclose(profile.distance_m, 0.0, atol=1e-10)
        np.testing.assert_allclose(profile.nearest_source_xyz[1], [5.0, 0.0, 0.0])

    def test_numerical_noise_on_collinear_large_coordinates_is_zeroed(self):
        source = np.array(
            [[775.0, -1700.0, 20.0, 0.0], [785.0, -1580.0, 20.0, 0.0]],
            dtype=float,
        )
        fractions = np.linspace(0.0, 1.0, 121)
        smoothed_xyz = source[0, :3] + fractions[:, None] * (
            source[1, :3] - source[0, :3]
        )
        smoothed = np.column_stack((smoothed_xyz, np.zeros(len(fractions))))
        profile = compute_deviation_profile(source, smoothed)
        np.testing.assert_array_equal(profile.distance_m, 0.0)

    def test_display_indices_are_thinned_and_keep_both_endpoints(self):
        indices = display_indices(1000, max_points=80)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 999)
        self.assertLessEqual(len(indices), 80)
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_four_panel_figure_exposes_distinct_views(self):
        figure, axes, profile = build_trajectory_comparison_figure(
            self._result(), title="episode", image_stride=3
        )
        try:
            self.assertEqual(
                set(axes), {"source", "enhanced", "overlay", "deviation"}
            )
            self.assertIn("Original trajectory", axes["source"].get_title())
            self.assertIn("Enhanced sampling", axes["enhanced"].get_title())
            self.assertIn("True-coordinate XY overlay", axes["overlay"].get_title())
            self.assertIn("3D distance", axes["deviation"].get_ylabel())
            self.assertIn("Deviation", axes["deviation"].get_title())
            legend_labels = [
                text.get_text()
                for axis in (axes["enhanced"], axes["overlay"])
                for text in axis.get_legend().get_texts()
            ]
            self.assertTrue(any("every 3 waypoints" in label for label in legend_labels))
            self.assertFalse(any("every 5 waypoints" in label for label in legend_labels))
            self.assertEqual(profile.max_index, int(np.argmax(profile.distance_m)))
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
