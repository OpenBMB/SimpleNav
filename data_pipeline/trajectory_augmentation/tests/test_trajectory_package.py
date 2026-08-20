import tempfile
import unittest
from pathlib import Path

from vln_aug.trajectory_package import iter_episodes, iter_render_requests


class TrajectoryPackageReaderTests(unittest.TestCase):
    def test_streams_episodes_and_render_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "trajectories").mkdir()
            (root / "render").mkdir()
            (root / "trajectories" / "episodes.jsonl").write_text(
                '{"episode_id":"ep","reference_path":[[0,0,0,0,0,0],[1,0,0,0,0,0]]}\n'
            )
            (root / "render" / "render_requests.jsonl").write_text(
                '{"request_id":"r","scene_id":"s","waypoint_index":0,'
                '"position_xyz":[0,0,0],"orientation_quaternion_wxyz":[1,0,0,0],'
                '"camera_key":"front","expected_image_relpath":"images/a.png"}\n'
            )

            self.assertEqual([item["episode_id"] for item in iter_episodes(root)], ["ep"])
            self.assertEqual(
                [item["request_id"] for item in iter_render_requests(root)], ["r"]
            )


if __name__ == "__main__":
    unittest.main()
