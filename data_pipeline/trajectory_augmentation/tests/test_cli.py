import unittest
from pathlib import Path

from vln_aug.cli import build_parser


class CliTests(unittest.TestCase):
    def test_parser_exposes_inspect_select_preview_enhance_validate(self):
        parser = build_parser()
        help_text = parser.format_help()
        for command in ("inspect", "select", "preview", "enhance", "validate"):
            self.assertIn(command, help_text)

    def test_parser_exposes_frame_trajectory_export(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "export-trajectories",
                "--train-split",
                "/tmp/Dataset/vln_train",
                "--dataset-key",
                "Dataset",
                "--sample-episode-indices",
                "1,9",
                "--include-episode-indices",
                "1,9",
                "--image-stride-choices",
                "5,6,7,8",
                "--image-stride-policy",
                "deterministic-random-per-interval",
                "--image-interval-seed",
                "20260718",
                "--image-width",
                "448",
                "--image-height",
                "448",
                "--retain-fraction",
                "0.5",
                "--exclude-scene-ids",
                "1",
                "--include-scene-ids",
                "env_airsim_16,env_airsim_18",
                "--selection-seed",
                "20260716",
                "--balanced-image-strides",
                "--eligible-package",
                "/tmp/Dataset/vln_train_enhanced_old_v1",
                "--sample-per-stride",
                "1",
                "--world-pose-source",
                "original-json",
                "--original-trajectory-json",
                "/tmp/Dataset/aerialvln_json/train.json",
                "--world-pose-metadata",
                "/tmp/Dataset/vln_train/meta/navvla_frame_metadata.jsonl",
                "--turn-speed-min-factor",
                "0.6",
                "--turn-curvature-start",
                "0.1",
                "--turn-curvature-full",
                "0.5",
                "--turn-smoothing-multiplier",
                "2.5",
                "--allow-preview-output",
            ]
        )

        self.assertEqual(args.command, "export-trajectories")
        self.assertEqual(args.dataset_key, "Dataset")
        self.assertEqual(args.sample_episode_indices, "1,9")
        self.assertEqual(args.include_episode_indices, "1,9")
        self.assertEqual(args.image_stride_choices, "5,6,7,8")
        self.assertEqual(
            args.image_stride_policy, "deterministic-random-per-interval"
        )
        self.assertEqual(args.image_interval_seed, 20260718)
        self.assertEqual(args.image_width, 448)
        self.assertEqual(args.image_height, 448)
        self.assertAlmostEqual(args.retain_fraction, 0.5)
        self.assertEqual(args.exclude_scene_ids, "1")
        self.assertEqual(args.include_scene_ids, "env_airsim_16,env_airsim_18")
        self.assertEqual(args.selection_seed, 20260716)
        self.assertTrue(args.balanced_image_strides)
        self.assertEqual(
            args.eligible_package,
            Path("/tmp/Dataset/vln_train_enhanced_old_v1"),
        )
        self.assertEqual(args.sample_per_stride, 1)
        self.assertEqual(args.world_pose_source, "original-json")
        self.assertEqual(
            args.original_trajectory_json,
            Path("/tmp/Dataset/aerialvln_json/train.json"),
        )
        self.assertEqual(
            args.world_pose_metadata,
            Path("/tmp/Dataset/vln_train/meta/navvla_frame_metadata.jsonl"),
        )
        self.assertAlmostEqual(args.turn_speed_min_factor, 0.6)
        self.assertAlmostEqual(args.turn_curvature_start, 0.1)
        self.assertAlmostEqual(args.turn_curvature_full, 0.5)
        self.assertAlmostEqual(args.turn_smoothing_multiplier, 2.5)
        self.assertTrue(args.allow_preview_output)

    def test_parser_exposes_package_validation(self):
        parser = build_parser()
        args = parser.parse_args(
            ["validate-trajectory-package", "--package-dir", "/tmp/package"]
        )
        self.assertEqual(args.command, "validate-trajectory-package")

    def test_parser_exposes_profile_validation_and_export(self):
        parser = build_parser()
        validate_args = parser.parse_args(
            [
                "validate-profile",
                "--profile",
                "/tmp/profile.json",
                "--dataset-root",
                "/tmp/Dataset",
            ]
        )
        export_args = parser.parse_args(
            [
                "export-profile",
                "--profile",
                "/tmp/profile.json",
                "--dataset-root",
                "/tmp/Dataset",
                "--dry-run",
            ]
        )

        self.assertEqual(validate_args.command, "validate-profile")
        self.assertEqual(validate_args.profile, Path("/tmp/profile.json"))
        self.assertEqual(export_args.command, "export-profile")
        self.assertTrue(export_args.dry_run)

    def test_parser_accepts_openfly_annotation_world_pose_source(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "export-trajectories",
                "--train-split",
                "/tmp/OpenFly/vln_train",
                "--dataset-key",
                "OpenFly_lerobot",
                "--world-pose-source",
                "openfly-annotation",
            ]
        )

        self.assertEqual(args.world_pose_source, "openfly-annotation")

    def test_default_dataset_root_is_explicitly_required(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["preview"])


if __name__ == "__main__":
    unittest.main()
