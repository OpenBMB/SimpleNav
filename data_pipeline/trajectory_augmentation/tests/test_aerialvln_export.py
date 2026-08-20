import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vln_aug.aerialvln_export import (
    build_aerialvln_episode,
    build_export_record,
    export_train_split,
    validate_trajectory_package,
    write_trajectory_package,
)
from vln_aug.lerobot_io import EpisodeMetadata
from vln_aug.image_stride import assign_image_stride, stable_image_interval_seed
from vln_aug.actions import random_observation_indices
from vln_aug.trajectory import RetimedTrajectory


def sample_trajectory() -> RetimedTrajectory:
    controls = np.array(
        [
            [1.0, 2.0, 3.0, np.pi / 2],
            [2.0, 2.0, 3.0, np.pi / 2],
            [3.0, 2.0, 3.0, np.pi / 2],
            [4.0, 2.0, 3.0, np.pi / 2],
            [5.0, 2.0, 3.0, np.pi / 2],
            [6.0, 2.0, 3.0, np.pi / 2],
            [7.0, 2.0, 3.0, np.pi / 2],
            [8.0, 2.0, 3.0, np.pi / 2],
        ],
        dtype=float,
    )
    return RetimedTrajectory(
        source_poses=controls[[0, 3, 7]],
        smoothed_dense_poses=controls,
        control_poses=controls,
        control_times=np.arange(len(controls), dtype=float),
        movement_steps=7,
        total_steps=7,
        path_length_m=7.0,
        movement_speed_mps=1.0,
        max_deviation_m=0.0,
        cruise_speed_mps=1.0,
        minimum_local_speed_mps=0.6,
    )


class AerialVLNExportTests(unittest.TestCase):
    def test_builds_original_episode_shell_from_enhanced_frame_trajectory(self):
        metadata = EpisodeMetadata(
            episode_index=42,
            episode_id="source-episode",
            scene_id="scene-5",
            length=3,
            data_chunk_index=0,
            data_file_index=0,
        )

        episode = build_aerialvln_episode(
            dataset_key="OpenFly_lerobot",
            metadata=metadata,
            trajectory_id="source-trajectory",
            task_index=9,
            instruction_text="fly to the building",
            trajectory=sample_trajectory(),
        )

        self.assertEqual(episode["episode_id"], "source-episode__enhanced_v1")
        self.assertEqual(episode["trajectory_id"], "source-trajectory__enhanced_v1")
        self.assertEqual(episode["scene_id"], "scene-5")
        self.assertEqual(episode["instruction"]["instruction_text"], "fly to the building")
        self.assertEqual(episode["actions"], [])
        self.assertEqual(
            set(episode),
            {
                "episode_id",
                "trajectory_id",
                "scene_id",
                "start_position",
                "start_rotation",
                "goals",
                "reference_path",
                "actions",
                "instruction",
            },
        )
        np.testing.assert_allclose(episode["start_position"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            episode["start_rotation"],
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
            atol=1e-7,
        )
        np.testing.assert_allclose(episode["goals"][0]["position"], [8.0, 2.0, 3.0])
        self.assertEqual(len(episode["reference_path"]), 8)
        np.testing.assert_allclose(
            episode["reference_path"][-1],
            [8.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2],
        )

    def test_export_record_contains_direct_renderer_metadata(self):
        metadata = EpisodeMetadata(
            episode_index=42,
            episode_id="source-episode",
            scene_id="scene-5",
            length=3,
            data_chunk_index=0,
            data_file_index=0,
            trajectory_id="source-trajectory",
            task_index=9,
            tasks=("fly to the building",),
        )

        record = build_export_record(
            dataset_key="OpenFly_lerobot",
            metadata=metadata,
            trajectory=sample_trajectory(),
        )

        self.assertEqual(record["metadata"]["source_episode_id"], "source-episode")
        self.assertEqual(record["metadata"]["source_trajectory_id"], "source-trajectory")
        stride = assign_image_stride(
            "OpenFly_lerobot", "source-episode", (1, 3, 5)
        )
        self.assertEqual(record["metadata"]["collection_stride_waypoints"], stride)
        expected = list(range(0, 8, stride))
        if expected[-1] != 7:
            expected.append(7)
        self.assertEqual(record["metadata"]["collection_waypoint_indices"], expected)
        self.assertAlmostEqual(record["metadata"]["movement_speed_mps"], 1.0)
        self.assertAlmostEqual(record["metadata"]["cruise_speed_mps"], 1.0)
        self.assertAlmostEqual(record["metadata"]["minimum_local_speed_mps"], 0.6)
        self.assertAlmostEqual(record["metadata"]["target_arc_step_m"], 1.0)
        self.assertEqual(record["metadata"]["control_frequency_hz"], 1.0)

    def test_export_record_accepts_exact_balanced_stride_override(self):
        metadata = EpisodeMetadata(
            episode_index=42,
            episode_id="source-episode",
            scene_id="scene-5",
            length=3,
            data_chunk_index=0,
            data_file_index=0,
        )

        record = build_export_record(
            dataset_key="AerialVLN_lerobot",
            metadata=metadata,
            trajectory=sample_trajectory(),
            image_stride=6,
            image_stride_choices=(5, 6, 7, 8),
        )

        self.assertEqual(record["metadata"]["collection_stride_waypoints"], 6)
        self.assertEqual(record["metadata"]["collection_waypoint_indices"], [0, 6, 7])

    def test_export_record_supports_random_interval_schedule_within_episode(self):
        metadata = EpisodeMetadata(
            episode_index=42,
            episode_id="source-episode",
            scene_id="scene-5",
            length=3,
            data_chunk_index=0,
            data_file_index=0,
        )

        record = build_export_record(
            dataset_key="AerialVLN_lerobot",
            metadata=metadata,
            trajectory=sample_trajectory(),
            image_stride_choices=(5, 6, 7, 8),
            image_stride_policy="deterministic-random-per-interval",
            image_interval_seed=20260718,
        )

        episode_seed = stable_image_interval_seed(
            "AerialVLN_lerobot", "source-episode", 20260718
        )
        expected = random_observation_indices(
            8, (5, 6, 7, 8), seed=episode_seed
        ).tolist()
        exported = record["metadata"]
        self.assertEqual(
            exported["collection_stride_policy"],
            "deterministic_random_per_interval",
        )
        self.assertEqual(exported["collection_stride_choices_waypoints"], [5, 6, 7, 8])
        self.assertEqual(exported["collection_stride_seed"], episode_seed)
        self.assertEqual(exported["collection_waypoint_indices"], expected)
        self.assertEqual(exported["collection_waypoint_gaps"], np.diff(expected).tolist())
        self.assertNotIn("collection_stride_waypoints", exported)

    def test_writes_valid_trajectory_only_package_without_lerobot_info(self):
        episode = build_aerialvln_episode(
            dataset_key="dataset",
            metadata=EpisodeMetadata(0, "ep", "scene", 3, 0, 0),
            trajectory_id="traj",
            task_index=0,
            instruction_text="go",
            trajectory=sample_trajectory(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "vln_train_enhanced"
            report = write_trajectory_package(
                output,
                dataset_key="dataset",
                source_split=Path(tmp) / "vln_train",
                records=[
                    {
                        "episode": episode,
                        "metadata": {
                            "episode_id": episode["episode_id"],
                            "trajectory_id": episode["trajectory_id"],
                            "source_episode_id": "ep",
                            "source_trajectory_id": "traj",
                            "source_episode_index": 0,
                            "scene_id": "scene",
                            "source_pose_kind": "frame_observation_state",
                            "trajectory_mode": "absolute_pose_sequence",
                            "control_frequency_hz": 1.0,
                            "movement_speed_mps": 1.0,
                            "target_arc_step_m": 1.0,
                            "collection_stride_waypoints": 5,
                            "collection_waypoint_indices": [0, 5, 7],
                            "collection_includes_real_terminal": True,
                            "action_horizon": 8,
                            "action_tail_policy": "repeat_last_absolute_waypoint",
                            "enhanced_waypoint_count": 8,
                        },
                    }
                ],
                failures=[],
                cameras=[
                    {
                        "camera_key": "front_image",
                        "camera_name": "front",
                        "height": 224,
                        "width": 224,
                        "channels": 3,
                        "camera_metadata": {"viewpoint_type": "front"},
                    }
                ],
                coordinate_metadata={"state_mode": "absolute_xyz_yaw"},
            )

            self.assertTrue((output / "trajectories" / "train.json").is_file())
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "validation.json").is_file())
            self.assertTrue((output / "render" / "render_requests.jsonl").is_file())
            self.assertFalse((output / "meta" / "info.json").exists())
            payload = json.loads((output / "trajectories" / "train.json").read_text())
            self.assertEqual(payload["episodes"], [episode])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["trajectory_only"])
            self.assertEqual(manifest["episode_count"], 1)
            self.assertEqual(report["valid_episode_count"], 1)
            metadata_lines = (output / "trajectories" / "augmentation_metadata.jsonl").read_text().splitlines()
            self.assertEqual(len(metadata_lines), 1)
            metadata = json.loads(metadata_lines[0])
            self.assertEqual(metadata["collection_waypoint_indices"], [0, 5, 7])
            self.assertEqual(metadata["source_pose_kind"], "frame_observation_state")
            requests = [
                json.loads(line)
                for line in (output / "render" / "render_requests.jsonl").read_text().splitlines()
            ]
            self.assertEqual([item["waypoint_index"] for item in requests], [0, 5, 7])
            self.assertEqual([item["image_index"] for item in requests], [0, 1, 2])
            self.assertEqual(requests[0]["position_xyz"], [1.0, 2.0, 3.0])
            self.assertEqual(requests[0]["collection_stride_waypoints"], 5)
            np.testing.assert_allclose(
                requests[0]["orientation_quaternion_wxyz"],
                [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
            )
            self.assertEqual(requests[0]["pose_xyz_rpy"], [1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2])
            self.assertEqual(requests[0]["expected_image_relpath"], "rendered_images/ep__enhanced_v1/front_image/frame_000000.png")
            package_report = validate_trajectory_package(output)
            self.assertTrue(package_report["valid"])
            self.assertEqual(package_report["render_request_count"], 3)

            request_path = output / "render" / "render_requests.jsonl"
            changed = [json.loads(line) for line in request_path.read_text().splitlines()]
            changed[0]["expected_width"] = 646
            request_path.write_text(
                "".join(json.dumps(item) + "\n" for item in changed)
            )
            incompatible = validate_trajectory_package(output)
            self.assertFalse(incompatible["valid"])
            self.assertTrue(
                any(
                    "224x224x3" in item.get("error", "")
                    for item in incompatible["errors"]
                )
            )

    def test_writes_and_validates_random_interval_render_requests(self):
        metadata = EpisodeMetadata(0, "ep-random", "scene", 3, 0, 0)
        record = build_export_record(
            dataset_key="dataset",
            metadata=metadata,
            trajectory=sample_trajectory(),
            image_stride_choices=(5, 6, 7, 8),
            image_stride_policy="deterministic-random-per-interval",
            image_interval_seed=20260718,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "candidate"
            write_trajectory_package(
                output,
                dataset_key="dataset",
                source_split=Path(tmp) / "vln_train",
                records=[record],
                failures=[],
                cameras=[
                    {
                        "camera_key": "front_image",
                        "camera_name": "front",
                        "height": 224,
                        "width": 224,
                        "channels": 3,
                    }
                ],
                coordinate_metadata={"state_mode": "absolute_xyz_yaw"},
            )

            requests = [
                json.loads(line)
                for line in (output / "render" / "render_requests.jsonl")
                .read_text()
                .splitlines()
            ]
            indices = record["metadata"]["collection_waypoint_indices"]
            expected_gaps = [0] + np.diff(indices).astype(int).tolist()
            self.assertEqual([item["waypoint_index"] for item in requests], indices)
            self.assertEqual(
                [item["collection_gap_waypoints"] for item in requests], expected_gaps
            )
            self.assertTrue(
                all(
                    item["collection_stride_policy"]
                    == "deterministic_random_per_interval"
                    for item in requests
                )
            )
            self.assertTrue(validate_trajectory_package(output)["valid"])

    def test_exports_tiny_train_split_and_records_rejected_short_episode(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Dataset"
            train = root / "vln_train"
            (train / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (train / "data" / "chunk-000").mkdir(parents=True)
            (train / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "data_path": "data/chunk-{chunk_index:03d}/part-{file_index:03d}.parquet",
                        "navvla": {"state_mode": "source_world_absolute_pose_xyz_yaw"},
                        "features": {
                            "observation.images.front_image": {"shape": [8, 12, 3]}
                        },
                    }
                )
            )
            episode_rows = [
                {
                    "episode_index": 0,
                    "episode_id": "valid",
                    "trajectory_id": "traj-valid",
                    "task_index": 0,
                    "scene_id": "scene-a",
                    "tasks": ["go forward"],
                    "length": 3,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                },
                {
                    "episode_index": 1,
                    "episode_id": "short",
                    "trajectory_id": "traj-short",
                    "task_index": 1,
                    "scene_id": "scene-b",
                    "tasks": ["move slightly"],
                    "length": 2,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                },
            ]
            pq.write_table(
                pa.Table.from_pylist(episode_rows),
                train / "meta" / "episodes" / "chunk-000" / "part-000.parquet",
            )
            data_rows = [
                {"episode_index": 0, "frame_index": 0, "observation.state": [0.0, 0.0, 0.0, 0.0]},
                {"episode_index": 0, "frame_index": 1, "observation.state": [1.0, 0.0, 0.0, 0.0]},
                {"episode_index": 0, "frame_index": 2, "observation.state": [2.0, 0.0, 0.0, 0.0]},
                {"episode_index": 1, "frame_index": 0, "observation.state": [0.0, 0.0, 0.0, 0.0]},
                {"episode_index": 1, "frame_index": 1, "observation.state": [0.2, 0.0, 0.0, 0.0]},
            ]
            pq.write_table(
                pa.Table.from_pylist(data_rows),
                train / "data" / "chunk-000" / "part-000.parquet",
            )
            output = root / "vln_train_enhanced"

            summary = export_train_split(
                source_split=train,
                output_dir=output,
                dataset_key="Dataset",
                sample_episode_indices={0},
                image_stride_choices=(5, 6, 7, 8),
                image_stride_policy="deterministic-random-per-interval",
                image_interval_seed=20260718,
            )

            self.assertEqual(summary["valid_episode_count"], 1)
            self.assertEqual(summary["failure_count"], 1)
            payload = json.loads((output / "trajectories" / "train.json").read_text())
            self.assertEqual(
                [item["episode_id"] for item in payload["episodes"]],
                ["valid__enhanced_v1"],
            )
            self.assertEqual(
                payload["episodes"][0]["trajectory_id"],
                "traj-valid__enhanced_v1",
            )
            self.assertEqual(
                payload["episodes"][0]["instruction"]["instruction_text"],
                "go forward",
            )
            self.assertTrue(
                (output / "validation" / "samples" / "episode_000000_comparison.png").is_file()
            )
            self.assertTrue(
                (output / "validation" / "samples" / "episode_000000_sampling_audit.png").is_file()
            )
            failures = (output / "validation" / "failures.jsonl").read_text().splitlines()
            self.assertEqual(len(failures), 1)
            self.assertIn("shorter than minimum", failures[0])
            requests = [
                json.loads(line)
                for line in (output / "render" / "render_requests.jsonl").read_text().splitlines()
            ]
            self.assertTrue(requests)
            self.assertEqual(requests[0]["expected_height"], 224)
            self.assertEqual(requests[0]["expected_width"], 224)
            self.assertEqual(requests[0]["expected_channels"], 3)
            self.assertTrue(
                all(
                    item["collection_stride_policy"]
                    == "deterministic_random_per_interval"
                    for item in requests
                )
            )
            self.assertEqual(
                requests[0]["camera_metadata"]["source_feature_shape"],
                [8, 12, 3],
            )

    def test_relative_training_state_uses_original_json_world_pose_for_rendering(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "AerialVLN"
            train = root / "vln_train"
            (train / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (train / "data" / "chunk-000").mkdir(parents=True)
            (train / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "data_path": "data/chunk-{chunk_index:03d}/part-{file_index:03d}.parquet",
                        "navvla": {
                            "state_mode": "episode_relative_first_body_aligned_pose_xyz_yaw",
                            "stored_episode_world_origin": "discarded",
                        },
                        "features": {
                            "observation.images.front_image": {"shape": [8, 12, 3]}
                        },
                    }
                )
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "episode_index": 0,
                            "episode_id": "world-episode",
                            "trajectory_id": "world-trajectory",
                            "task_index": 0,
                            "scene_id": "5",
                            "tasks": ["go north"],
                            "length": 3,
                            "data/chunk_index": 0,
                            "data/file_index": 0,
                        }
                    ]
                ),
                train / "meta" / "episodes" / "chunk-000" / "part-000.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "episode_index": 0,
                            "frame_index": frame,
                            "index": frame,
                            "observation.state": [float(frame), 0.0, 0.0, 0.0],
                        }
                        for frame in range(3)
                    ]
                ),
                train / "data" / "chunk-000" / "part-000.parquet",
            )
            world_poses = [
                [10.0, 20.0, -3.0, 0.0, 0.0, np.pi / 2],
                [10.0, 21.0, -3.0, 0.0, 0.0, np.pi / 2],
                [10.0, 22.0, -3.0, 0.0, 0.0, np.pi / 2],
            ]
            (root / "aerialvln_json").mkdir()
            (root / "aerialvln_json" / "train.json").write_text(
                json.dumps(
                    {
                        "episodes": [
                        {
                            "episode_id": "world-episode",
                            "trajectory_id": "world-trajectory",
                            "scene_id": 5,
                            "reference_path": world_poses,
                        }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            output = root / "vln_train_enhanced"
            summary = export_train_split(
                source_split=train,
                output_dir=output,
                dataset_key="AerialVLN",
            )

            self.assertEqual(summary["valid_episode_count"], 1)
            episode = json.loads(
                (output / "trajectories" / "episodes.jsonl").read_text().splitlines()[0]
            )
            np.testing.assert_allclose(episode["start_position"], [10.0, 20.0, -3.0])
            np.testing.assert_allclose(
                episode["reference_path"][-1],
                [10.0, 22.0, -3.0, 0.0, 0.0, np.pi / 2],
            )
            metadata = json.loads(
                (output / "trajectories" / "augmentation_metadata.jsonl")
                .read_text()
                .splitlines()[0]
            )
            self.assertEqual(
                metadata["source_pose_kind"],
                "original_aerialvln_reference_path_world_pose",
            )
            self.assertEqual(
                metadata["training_state_mode"],
                "episode_relative_first_body_aligned_pose_xyz_yaw",
            )
            self.assertLess(metadata["coordinate_alignment_max_position_error_m"], 1e-6)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                manifest["source_pose_kind"],
                "original_aerialvln_reference_path_world_pose",
            )
            request = json.loads(
                (output / "render" / "render_requests.jsonl").read_text().splitlines()[0]
            )
            np.testing.assert_allclose(request["position_xyz"], [10.0, 20.0, -3.0])

    def test_openfly_annotation_exports_all_included_scenes_with_balanced_strides(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "OpenFly"
            train = root / "vln_train"
            (train / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (train / "data" / "chunk-000").mkdir(parents=True)
            (train / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "data_path": "data/chunk-{chunk_index:03d}/part-{file_index:03d}.parquet",
                        "navvla": {
                            "state_mode": "episode_relative_first_body_aligned_pose_xyz_yaw",
                            "coordinate_fix": "reflected stored absolute state y/yaw and action dy/dyaw",
                        },
                        "features": {
                            "observation.images.front_image": {"shape": [8, 12, 3]}
                        },
                    }
                )
            )
            episode_rows = []
            data_rows = []
            annotations = []
            for index in range(8):
                scene = "env_airsim_18" if index < 6 else "env_ue_bigcity"
                trajectory_id = f"{scene}/astar_data/short/traj-{index}"
                episode_rows.append(
                    {
                        "episode_index": index,
                        "episode_id": f"{index:06d}",
                        "trajectory_id": trajectory_id,
                        "task_index": index,
                        "scene_id": scene,
                        "tasks": [f"task-{index}"],
                        "length": 3,
                        "data/chunk_index": 0,
                        "data/file_index": 0,
                    }
                )
                for frame in range(3):
                    data_rows.append(
                        {
                            "episode_index": index,
                            "frame_index": frame,
                            "index": index * 3 + frame,
                            "observation.state": [float(frame), 0.0, 0.0, 0.0],
                        }
                    )
                annotations.append(
                    {
                        "image_path": trajectory_id,
                        "pos": [
                            [100.0 + frame, 20.0 + index, -3.0]
                            for frame in range(3)
                        ],
                        "yaw": [0.0, 0.0, 0.0],
                    }
                )
            pq.write_table(
                pa.Table.from_pylist(episode_rows),
                train / "meta" / "episodes" / "chunk-000" / "part-000.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(data_rows),
                train / "data" / "chunk-000" / "part-000.parquet",
            )
            annotation_path = root / "Annotation" / "train.json"
            annotation_path.parent.mkdir()
            annotation_path.write_text(json.dumps(annotations), encoding="utf-8")

            output = root / "vln_train_enhanced"
            summary = export_train_split(
                source_split=train,
                output_dir=output,
                dataset_key="OpenFly_lerobot",
                retain_fraction=1.0,
                include_scene_ids={"env_airsim_18"},
                selection_seed=20260717,
                balanced_image_strides=True,
                image_stride_choices=(5, 6, 7, 8),
                world_pose_source="openfly-annotation",
                original_trajectory_json=annotation_path,
            )

            self.assertEqual(summary["valid_episode_count"], 6)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                manifest["selection_stride_episode_counts"],
                {"5": 2, "6": 2, "7": 1, "8": 1},
            )
            self.assertEqual(
                manifest["selection_included_scene_ids"], ["env_airsim_18"]
            )
            metadata = [
                json.loads(line)
                for line in (output / "trajectories" / "augmentation_metadata.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertTrue(
                all(
                    item["source_pose_kind"]
                    == "original_openfly_annotation_world_pose"
                    for item in metadata
                )
            )
            self.assertTrue(
                all(
                    item["coordinate_alignment_transform"] == "reflect-y-yaw"
                    for item in metadata
                )
            )
            request = json.loads(
                (output / "render" / "render_requests.jsonl").read_text().splitlines()[0]
            )
            self.assertGreater(request["position_xyz"][0], 90.0)
            self.assertLess(request["position_xyz"][1], -19.0)
            self.assertGreater(request["position_xyz"][2], 2.0)

    def test_full_export_does_not_publish_when_plot_generation_fails(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Dataset"
            train = root / "vln_train"
            (train / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (train / "data" / "chunk-000").mkdir(parents=True)
            (train / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "data_path": "data/chunk-{chunk_index:03d}/part-{file_index:03d}.parquet",
                        "navvla": {"state_mode": "source_world_absolute_pose_xyz_yaw"},
                        "features": {
                            "observation.images.front_image": {"shape": [8, 12, 3]}
                        },
                    }
                )
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [{
                        "episode_index": 0,
                        "episode_id": "valid",
                        "trajectory_id": "traj",
                        "task_index": 0,
                        "scene_id": "scene",
                        "tasks": ["go"],
                        "length": 2,
                        "data/chunk_index": 0,
                        "data/file_index": 0,
                    }]
                ),
                train / "meta" / "episodes" / "chunk-000" / "part-000.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"episode_index": 0, "frame_index": 0, "observation.state": [0.0, 0.0, 0.0, 0.0]},
                        {"episode_index": 0, "frame_index": 1, "observation.state": [2.0, 0.0, 0.0, 0.0]},
                    ]
                ),
                train / "data" / "chunk-000" / "part-000.parquet",
            )
            output = root / "vln_train_enhanced"

            with patch(
                "vln_aug.aerialvln_export.plot_trajectory_comparison",
                side_effect=RuntimeError("plot failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "plot failed"):
                    export_train_split(
                        source_split=train,
                        output_dir=output,
                        dataset_key="Dataset",
                        sample_episode_indices={0},
                    )

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".vln_train_enhanced.export-*")), [])

    def test_selected_preview_export_accepts_tool_output_and_only_exports_requested_episode(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            train = base / "Dataset" / "vln_train"
            (train / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (train / "data" / "chunk-000").mkdir(parents=True)
            (train / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "data_path": "data/chunk-{chunk_index:03d}/part-{file_index:03d}.parquet",
                        "navvla": {"state_mode": "source_world_absolute_pose_xyz_yaw"},
                        "features": {"observation.images.front_image": {"shape": [8, 12, 3]}},
                    }
                )
            )
            episodes = []
            rows = []
            for index in (0, 1):
                episodes.append(
                    {
                        "episode_index": index,
                        "episode_id": f"ep-{index}",
                        "trajectory_id": f"traj-{index}",
                        "task_index": index,
                        "scene_id": f"scene-{index}",
                        "tasks": [f"task-{index}"],
                        "length": 3,
                        "data/chunk_index": 0,
                        "data/file_index": 0,
                    }
                )
                rows.extend(
                    {
                        "episode_index": index,
                        "frame_index": frame,
                        "observation.state": [float(frame), float(index), 0.0, 0.0],
                    }
                    for frame in range(3)
                )
            pq.write_table(
                pa.Table.from_pylist(episodes),
                train / "meta" / "episodes" / "chunk-000" / "part-000.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(rows),
                train / "data" / "chunk-000" / "part-000.parquet",
            )
            output = base / "preview" / "Dataset"

            summary = export_train_split(
                source_split=train,
                output_dir=output,
                dataset_key="Dataset",
                sample_episode_indices={1},
                include_episode_indices={1},
                require_enhanced_sibling=False,
            )

            self.assertEqual(summary["episode_count"], 1)
            episode = json.loads(
                (output / "trajectories" / "episodes.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(episode["episode_id"], "ep-1__enhanced_v1")


if __name__ == "__main__":
    unittest.main()
