import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vln_aug.aerialvln_export import export_train_split
from vln_aug.dataset_profile import build_export_plan, load_dataset_profile


class FixturePoseIndex:
    def __init__(self, path, *, pose_order):
        self.path = Path(path)
        self.pose_order = pose_order
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.episodes = payload if isinstance(payload, dict) else {}

    def poses_for_episode(self, metadata):
        return np.asarray(self.episodes[str(metadata.episode_id)], dtype=float)


class DatasetProfileTests(unittest.TestCase):
    def _write_profile(self, root: Path, payload: dict) -> Path:
        path = root / "profile.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_builds_portable_export_plan_with_custom_pose_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vln_train" / "meta").mkdir(parents=True)
            (root / "annotations").mkdir()
            (root / "annotations" / "train.json").write_text("[]", encoding="utf-8")
            profile_path = self._write_profile(
                root,
                {
                    "schema_version": 1,
                    "dataset_key": "ExampleVLN",
                    "paths": {
                        "train_split": "vln_train",
                        "output_dir": "vln_train_enhanced",
                    },
                    "world_pose": {
                        "mode": "adapter",
                        "class": "tests.test_dataset_profile:FixturePoseIndex",
                        "path": "annotations/train.json",
                        "options": {"pose_order": "xyz-yaw"},
                        "source_kind": "example_absolute_xyz_yaw",
                        "alignment_transform": "identity",
                        "render_transform": "reflect-y-z-yaw",
                    },
                    "sampling": {
                        "image_stride_choices": [5],
                        "image_stride_policy": "fixed-per-episode",
                    },
                    "render": {
                        "image_width": 448,
                        "image_height": 448,
                    },
                },
            )

            profile = load_dataset_profile(profile_path)
            plan = build_export_plan(profile, dataset_root=root)

        self.assertEqual(plan.dataset_key, "ExampleVLN")
        self.assertEqual(plan.source_split, root / "vln_train")
        self.assertEqual(plan.output_dir, root / "vln_train_enhanced")
        self.assertEqual(type(plan.world_pose_adapter).__name__, "FixturePoseIndex")
        self.assertEqual(plan.world_pose_adapter.path, root / "annotations" / "train.json")
        self.assertEqual(plan.world_pose_adapter.pose_order, "xyz-yaw")
        self.assertEqual(plan.export_kwargs["image_stride_choices"], (5,))
        self.assertEqual(plan.export_kwargs["render_image_width"], 448)
        self.assertEqual(plan.export_kwargs["render_image_height"], 448)
        self.assertEqual(plan.export_kwargs["world_pose_source_kind"], "example_absolute_xyz_yaw")
        self.assertEqual(plan.export_kwargs["render_coordinate_transform"], "reflect-y-z-yaw")
        self.assertEqual(plan.summary()["world_pose_mode"], "adapter")

    def test_rejects_profile_paths_that_escape_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_path = self._write_profile(
                root,
                {
                    "schema_version": 1,
                    "dataset_key": "UnsafeVLN",
                    "paths": {
                        "train_split": "../outside/vln_train",
                        "output_dir": "vln_train_enhanced",
                    },
                    "world_pose": {"mode": "observation-state"},
                },
            )

            profile = load_dataset_profile(profile_path)
            with self.assertRaisesRegex(ValueError, "must stay inside dataset root"):
                build_export_plan(profile, dataset_root=root)

    def test_rejects_output_inside_read_only_source_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_path = self._write_profile(
                root,
                {
                    "schema_version": 1,
                    "dataset_key": "UnsafeOutputVLN",
                    "paths": {
                        "train_split": "vln_train",
                        "output_dir": "vln_train/candidate",
                    },
                    "world_pose": {"mode": "observation-state"},
                    "require_enhanced_sibling": False,
                },
            )

            profile = load_dataset_profile(profile_path)
            with self.assertRaisesRegex(ValueError, "outside the source split"):
                build_export_plan(profile, dataset_root=root)

    def test_rejects_adapter_without_required_coordinate_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_path = self._write_profile(
                root,
                {
                    "schema_version": 1,
                    "dataset_key": "IncompleteVLN",
                    "paths": {
                        "train_split": "vln_train",
                        "output_dir": "vln_train_enhanced",
                    },
                    "world_pose": {
                        "mode": "adapter",
                        "class": "tests.test_dataset_profile:FixturePoseIndex",
                        "path": "annotations/train.json",
                    },
                },
            )

            with self.assertRaisesRegex(ValueError, "source_kind"):
                load_dataset_profile(profile_path)

    def test_profile_drives_custom_adapter_transform_through_full_export(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "vln_train"
            (train / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (train / "data" / "chunk-000").mkdir(parents=True)
            (root / "annotations").mkdir()
            (train / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "data_path": "data/chunk-{chunk_index:03d}/part-{file_index:03d}.parquet",
                        "navvla": {
                            "state_mode": "episode_relative_first_body_aligned_pose_xyz_yaw"
                        },
                        "features": {
                            "observation.images.front_image": {"shape": [8, 12, 3]}
                        },
                    }
                ),
                encoding="utf-8",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "episode_index": 0,
                            "episode_id": "ep",
                            "trajectory_id": "traj",
                            "task_index": 0,
                            "scene_id": "scene",
                            "tasks": ["go"],
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
                            "observation.state": [float(frame), 0.0, 0.0, 0.0],
                        }
                        for frame in range(3)
                    ]
                ),
                train / "data" / "chunk-000" / "part-000.parquet",
            )
            (root / "annotations" / "train.json").write_text(
                json.dumps(
                    {
                        "ep": [
                            [10.0, 20.0, -3.0, 0.0],
                            [11.0, 20.0, -3.0, 0.0],
                            [12.0, 20.0, -3.0, 0.0],
                        ]
                    }
                ),
                encoding="utf-8",
            )
            profile_path = self._write_profile(
                root,
                {
                    "schema_version": 1,
                    "dataset_key": "ExampleVLN",
                    "paths": {
                        "train_split": "vln_train",
                        "output_dir": "vln_train_enhanced",
                    },
                    "world_pose": {
                        "mode": "adapter",
                        "class": "tests.test_dataset_profile:FixturePoseIndex",
                        "path": "annotations/train.json",
                        "options": {"pose_order": "xyz-yaw"},
                        "source_kind": "example_absolute_xyz_yaw",
                        "alignment_transform": "identity",
                        "render_transform": "reflect-y-z-yaw",
                    },
                    "sampling": {"image_stride_choices": [5]},
                    "render": {"image_width": 448, "image_height": 448},
                },
            )

            plan = build_export_plan(
                load_dataset_profile(profile_path), dataset_root=root
            )
            summary = export_train_split(
                source_split=plan.source_split,
                output_dir=plan.output_dir,
                dataset_key=plan.dataset_key,
                **plan.export_kwargs,
            )

            request = json.loads(
                (plan.output_dir / "render" / "render_requests.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            metadata = json.loads(
                (plan.output_dir / "trajectories" / "augmentation_metadata.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )

        self.assertEqual(summary["valid_episode_count"], 1)
        self.assertEqual(request["expected_width"], 448)
        self.assertEqual(request["expected_height"], 448)
        np.testing.assert_allclose(request["position_xyz"], [10.0, -20.0, 3.0])
        self.assertEqual(metadata["source_pose_kind"], "example_absolute_xyz_yaw")
        self.assertEqual(metadata["coordinate_alignment_transform"], "identity")


if __name__ == "__main__":
    unittest.main()
