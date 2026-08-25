import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vln_aug.lerobot_io import EpisodeMetadata
from vln_aug.world_pose import (
    AerialVLNOriginalPoseIndex,
    FrameMetadataWorldPoseStream,
    OpenFlyAnnotationPoseIndex,
    transform_world_poses_for_alignment,
    validate_episode_local_alignment,
)


class FrameMetadataWorldPoseStreamTests(unittest.TestCase):
    def test_resolves_selected_global_indices_to_world_xyz_yaw(self):
        records = [
            {
                "index": 0,
                "source_metadata": {
                    "source_pose": [10.0, 20.0, 30.0, 0.1, 0.2, 0.3]
                },
            },
            {
                "index": 1,
                "source_metadata": {
                    "source_pose": [11.0, 21.0, 31.0, 0.4, 0.5, 0.6]
                },
            },
            {
                "index": 2,
                "source_metadata": {
                    "source_pose": [12.0, 22.0, 32.0, 0.7, 0.8, 0.9]
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "navvla_frame_metadata.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            with FrameMetadataWorldPoseStream(path) as poses:
                actual = poses.read_indices([0, 2])

        np.testing.assert_allclose(
            actual,
            [[10.0, 20.0, 30.0, 0.3], [12.0, 22.0, 32.0, 0.9]],
        )

    def test_rejects_non_monotonic_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "navvla_frame_metadata.jsonl"
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "index": index,
                            "source_metadata": {
                                "source_pose": [float(index), 0, 0, 0, 0, 0]
                            },
                        }
                    )
                    + "\n"
                    for index in range(3)
                ),
                encoding="utf-8",
            )

            with FrameMetadataWorldPoseStream(path) as poses:
                poses.read_indices([1])
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    poses.read_indices([0])


class AerialVLNOriginalPoseIndexTests(unittest.TestCase):
    def test_reads_world_reference_path_by_source_episode_identity(self):
        episode = {
            "episode_id": "ep-a",
            "trajectory_id": "traj-a",
            "scene_id": 5,
            "reference_path": [
                [10.0, 20.0, -3.0, 0.0, 0.0, 1.2],
                [11.0, 21.0, -4.0, 0.0, 0.0, 1.3],
            ],
        }
        metadata = EpisodeMetadata(
            episode_index=7,
            episode_id="ep-a",
            scene_id="5",
            length=2,
            data_chunk_index=0,
            data_file_index=0,
            trajectory_id="traj-a",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json"
            path.write_text(json.dumps({"episodes": [episode]}), encoding="utf-8")

            poses = AerialVLNOriginalPoseIndex(path).poses_for_episode(metadata)

        np.testing.assert_allclose(
            poses,
            [[10.0, 20.0, -3.0, 1.2], [11.0, 21.0, -4.0, 1.3]],
        )

    def test_rejects_reference_path_length_mismatch(self):
        metadata = EpisodeMetadata(0, "ep", "5", 2, 0, 0, trajectory_id="traj")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json"
            path.write_text(
                json.dumps(
                    {
                        "episodes": [
                            {
                                "episode_id": "ep",
                                "trajectory_id": "traj",
                                "scene_id": 5,
                                "reference_path": [[0, 0, 0, 0, 0, 0]],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "length"):
                AerialVLNOriginalPoseIndex(path).poses_for_episode(metadata)


class EpisodeCoordinateContractTests(unittest.TestCase):
    def test_accepts_first_frame_body_aligned_local_state(self):
        yaw0 = np.pi / 2
        local = np.array(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, -2.0, 0.2], [2.0, 1.0, -2.0, 0.4]]
        )
        rotation = np.array(
            [[np.cos(yaw0), -np.sin(yaw0)], [np.sin(yaw0), np.cos(yaw0)]]
        )
        world = np.empty_like(local)
        world[:, :2] = local[:, :2] @ rotation.T + np.array([10.0, 20.0])
        world[:, 2] = local[:, 2] - 3.0
        world[:, 3] = local[:, 3] + yaw0

        metrics = validate_episode_local_alignment(local, world)

        self.assertLess(metrics["max_position_error_m"], 1e-10)
        self.assertLess(metrics["max_yaw_error_rad"], 1e-10)

    def test_rejects_world_path_that_does_not_match_local_geometry(self):
        local = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        wrong_world = np.array([[10.0, 20.0, 0.0, 0.0], [10.0, 22.0, 0.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "coordinate contract mismatch"):
            validate_episode_local_alignment(local, wrong_world)

    def test_reflects_openfly_world_y_and_yaw_only_for_local_alignment(self):
        world = np.array(
            [[10.0, 20.0, -3.0, 0.5], [11.0, 18.0, -4.0, -0.25]]
        )

        aligned = transform_world_poses_for_alignment(world, "reflect-y-yaw")

        np.testing.assert_allclose(
            aligned,
            [[10.0, -20.0, -3.0, -0.5], [11.0, -18.0, -4.0, 0.25]],
        )
        np.testing.assert_allclose(world[0], [10.0, 20.0, -3.0, 0.5])

    def test_reflects_openfly_render_coordinates_in_y_z_and_yaw(self):
        world = np.array(
            [[10.0, 20.0, -3.0, 0.5], [11.0, 18.0, 4.0, -0.25]]
        )

        rendered = transform_world_poses_for_alignment(
            world, "reflect-y-z-yaw"
        )

        np.testing.assert_allclose(
            rendered,
            [[10.0, -20.0, 3.0, -0.5], [11.0, -18.0, -4.0, 0.25]],
        )


class OpenFlyAnnotationPoseIndexTests(unittest.TestCase):
    def test_reads_raw_airsim_world_pos_and_yaw_by_trajectory_id(self):
        annotation = [
            {
                "image_path": "env_airsim_18/astar_data/low_short/traj-a",
                "pos": [[1900.0, 953.0, -36.0], [1900.0, 944.0, -36.0]],
                "yaw": [-np.pi / 2, -np.pi / 2],
            }
        ]
        metadata = EpisodeMetadata(
            episode_index=0,
            episode_id="000000",
            scene_id="env_airsim_18",
            length=2,
            data_chunk_index=0,
            data_file_index=0,
            trajectory_id="env_airsim_18/astar_data/low_short/traj-a",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json"
            path.write_text(json.dumps(annotation), encoding="utf-8")

            poses = OpenFlyAnnotationPoseIndex(path).poses_for_episode(metadata)

        np.testing.assert_allclose(
            poses,
            [[1900.0, 953.0, -36.0, -np.pi / 2], [1900.0, 944.0, -36.0, -np.pi / 2]],
        )

    def test_accepts_mixed_three_and_four_value_positions_in_updown_trajectory(self):
        annotation = [
            {
                "image_path": "env_airsim_16/astar_data/low_average_updown/traj-a",
                "pos": [
                    [-922.0, -37.0, 2.0, np.pi / 3],
                    [-919.0, -32.0, 5.0],
                ],
                "yaw": [np.pi / 3, np.pi / 3],
            }
        ]
        metadata = EpisodeMetadata(
            0,
            "ep",
            "env_airsim_16",
            2,
            0,
            0,
            trajectory_id="env_airsim_16/astar_data/low_average_updown/traj-a",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json"
            path.write_text(json.dumps(annotation), encoding="utf-8")

            poses = OpenFlyAnnotationPoseIndex(path).poses_for_episode(metadata)

        np.testing.assert_allclose(
            poses,
            [[-922.0, -37.0, 2.0, np.pi / 3], [-919.0, -32.0, 5.0, np.pi / 3]],
        )

    def test_rejects_conflicting_embedded_yaw_in_four_value_position(self):
        annotation = [
            {
                "image_path": "env_airsim_16/astar_data/low_average_updown/traj-a",
                "pos": [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                "yaw": [0.0, 0.0],
            }
        ]
        metadata = EpisodeMetadata(
            0,
            "ep",
            "env_airsim_16",
            2,
            0,
            0,
            trajectory_id="env_airsim_16/astar_data/low_average_updown/traj-a",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json"
            path.write_text(json.dumps(annotation), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "embedded yaw"):
                OpenFlyAnnotationPoseIndex(path).poses_for_episode(metadata)

    def test_rejects_annotation_scene_or_length_mismatch(self):
        annotation = [
            {
                "image_path": "env_airsim_18/astar_data/low_short/traj-a",
                "pos": [[0.0, 0.0, 0.0]],
                "yaw": [0.0],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.json"
            path.write_text(json.dumps(annotation), encoding="utf-8")
            index = OpenFlyAnnotationPoseIndex(path)

            with self.assertRaisesRegex(ValueError, "scene mismatch"):
                index.poses_for_episode(
                    EpisodeMetadata(
                        0,
                        "ep",
                        "env_airsim_16",
                        1,
                        0,
                        0,
                        trajectory_id="env_airsim_18/astar_data/low_short/traj-a",
                    )
                )
            with self.assertRaisesRegex(ValueError, "length"):
                index.poses_for_episode(
                    EpisodeMetadata(
                        0,
                        "ep",
                        "env_airsim_18",
                        2,
                        0,
                        0,
                        trajectory_id="env_airsim_18/astar_data/low_short/traj-a",
                    )
                )

    def test_rejects_missing_source_pose_instead_of_using_local_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "navvla_frame_metadata.jsonl"
            path.write_text(
                json.dumps({"index": 0, "source_metadata": {}}) + "\n",
                encoding="utf-8",
            )

            with FrameMetadataWorldPoseStream(path) as poses:
                with self.assertRaisesRegex(ValueError, "source_pose"):
                    poses.read_indices([0])


if __name__ == "__main__":
    unittest.main()
