import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vln_aug.intermediate import CameraSpec, write_intermediate_episode


class IntermediateOutputTests(unittest.TestCase):
    def test_writes_one_hz_controls_point_two_hz_rows_actions_and_render_requests(self):
        controls = np.zeros((11, 4), dtype=float)
        controls[:, 0] = np.arange(11, dtype=float)
        cameras = [
            CameraSpec(
                key="front",
                height=8,
                width=12,
                metadata={"viewpoint_type": "front", "azimuth_rad": 0.0},
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            result = write_intermediate_episode(
                output_dir=root,
                dataset_key="dataset-a",
                source_episode_index=42,
                source_episode_id="episode-42",
                scene_id="scene-a",
                control_poses=controls,
                cameras=cameras,
                horizon=8,
                terminal_action_available=True,
                coordinate_metadata={"state_mode": "source_world_absolute_pose_xyz_yaw"},
            )

            control = pq.read_table(result.control_path)
            observation = pq.read_table(result.observation_path)
            render_flags = control.column("is_render_time").to_pylist()
            self.assertEqual(
                [index for index, enabled in enumerate(render_flags) if enabled],
                [0, 5, 10],
            )
            self.assertEqual(control.num_rows, 11)
            self.assertEqual(observation.num_rows, 3)
            self.assertEqual(observation.column("timestamp").to_pylist(), [0.0, 5.0, 10.0])
            self.assertEqual(observation.column("frame_index").to_pylist(), [0, 1, 2])
            self.assertEqual(observation.column("next.done").to_pylist(), [False, False, True])
            self.assertEqual(
                observation.column("sample.action_available").to_pylist(),
                [True, True, True],
            )
            actions = observation.column("action").to_pylist()
            self.assertEqual(len(actions[0]), 8)
            self.assertEqual(len(actions[0][0]), 4)
            np.testing.assert_allclose(actions[1][-1], [5.0, 0.0, 0.0, 0.0])
            np.testing.assert_allclose(actions[2], np.zeros((8, 4)))
            self.assertTrue(pa.types.is_fixed_size_list(control.schema.field("pose").type))
            self.assertEqual(control.schema.field("pose").type.list_size, 4)
            action_type = observation.schema.field("action").type
            self.assertTrue(pa.types.is_fixed_size_list(action_type))
            self.assertEqual(action_type.list_size, 8)
            self.assertTrue(pa.types.is_fixed_size_list(action_type.value_type))
            self.assertEqual(action_type.value_type.list_size, 4)
            self.assertTrue(pa.types.is_float32(action_type.value_type.value_type))

            lines = result.render_request_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertIn('"scene_id": "scene-a"', lines[0])
            self.assertIn('"camera_key": "front"', lines[0])
            self.assertIn('"expected_height": 8', lines[0])
            self.assertIn('"expected_width": 12', lines[0])
            self.assertIn('"viewpoint_type": "front"', lines[0])
            self.assertIn('"state_mode": "source_world_absolute_pose_xyz_yaw"', lines[0])

    def test_writes_non_aligned_real_terminal_as_final_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            controls = np.zeros((8, 4), dtype=float)
            controls[:, 0] = np.arange(8)
            result = write_intermediate_episode(
                output_dir=Path(tmp),
                dataset_key="dataset-a",
                source_episode_index=1,
                source_episode_id="one",
                scene_id="scene-a",
                control_poses=controls,
                cameras=[CameraSpec(key="front", height=8, width=8)],
                coordinate_metadata={"state_mode": "absolute"},
            )
            observation = pq.read_table(result.observation_path)
            control = pq.read_table(result.control_path)
            self.assertEqual(observation.column("control_index").to_pylist(), [0, 5, 7])
            self.assertEqual(observation.column("timestamp").to_pylist(), [0.0, 5.0, 7.0])
            self.assertEqual(
                [
                    index
                    for index, enabled in enumerate(control.column("is_render_time").to_pylist())
                    if enabled
                ],
                [0, 5, 7],
            )

    def test_rejects_render_plan_without_scene_camera_or_coordinate_metadata(self):
        controls = np.zeros((6, 4), dtype=float)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_intermediate_episode(
                    output_dir=Path(tmp),
                    dataset_key="dataset-a",
                    source_episode_index=1,
                    source_episode_id="one",
                    scene_id="",
                    control_poses=controls,
                    cameras=[CameraSpec(key="front", height=8, width=8)],
                    coordinate_metadata={},
                )

    def test_actions_reconstruct_from_the_quantized_stored_control_trajectory(self):
        controls = np.zeros((11, 4), dtype=float)
        controls[:, 0] = 1000.123456 + np.arange(11) * 0.987654
        controls[:, 1] = -500.765432 + np.arange(11) * 0.123456
        controls[:, 3] = np.linspace(2.9, 3.4, 11)
        with tempfile.TemporaryDirectory() as tmp:
            result = write_intermediate_episode(
                output_dir=Path(tmp),
                dataset_key="dataset-a",
                source_episode_index=1,
                source_episode_id="one",
                scene_id="scene",
                control_poses=controls,
                cameras=[CameraSpec(key="front", height=8, width=8)],
                coordinate_metadata={"state_mode": "absolute"},
            )
            stored_control = np.asarray(
                pq.read_table(result.control_path).column("pose").to_pylist(), dtype=float
            )
            stored_action = np.asarray(
                pq.read_table(result.observation_path).column("action").to_pylist(), dtype=float
            )
            from vln_aug.actions import build_observation_actions

            _, reconstructed = build_observation_actions(stored_control, render_stride=5, horizon=8)
            np.testing.assert_allclose(stored_action, reconstructed, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
