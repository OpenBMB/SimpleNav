import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from waypoint_collector.cameras import episode_camera_records, episode_camera_rows
from waypoint_collector.cli import build_parser
from waypoint_collector.layout import part_location
from waypoint_collector.pipeline import CollectorPipeline
from waypoint_collector.requests import RenderRequest, iter_render_requests
from waypoint_collector.indexing import build_request_index
from waypoint_collector.state import CollectorState
from waypoint_collector import worker


def request_payload(**overrides):
    payload = {
        "schema_version": "1.0",
        "request_id": "episode-7/front_image/frame_000000",
        "dataset_key": "AerialVLN_lerobot",
        "episode_id": "episode-7",
        "trajectory_id": "trajectory-7",
        "source_episode_id": "source-7",
        "source_episode_index": 1001,
        "scene_id": "5",
        "image_index": 0,
        "waypoint_index": 0,
        "timestamp": 0.0,
        "position_xyz": [1.0, 2.0, -3.0],
        "orientation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "expected_height": 224,
        "expected_width": 224,
        "expected_channels": 3,
    }
    payload.update(overrides)
    return payload


class RenderRequestTests(unittest.TestCase):
    def test_streams_requests_with_global_index_and_byte_ranges(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "requests.jsonl"
            payloads = [
                request_payload(),
                request_payload(
                    request_id="episode-7/front_image/frame_000001",
                    image_index=1,
                    waypoint_index=5,
                    timestamp=5.0,
                ),
            ]
            path.write_text(
                "".join(json.dumps(item) + "\n" for item in payloads),
                encoding="utf-8",
            )

            requests = list(iter_render_requests(path))

            self.assertEqual([item.index for item in requests], [0, 1])
            self.assertEqual(requests[0].byte_start, 0)
            self.assertEqual(requests[0].byte_end, requests[1].byte_start)
            self.assertEqual(requests[1].waypoint_index, 5)

    def test_rejects_non_unit_quaternion_and_wrong_dimensions(self):
        with self.assertRaisesRegex(ValueError, "unit quaternion"):
            RenderRequest.from_payload(
                request_payload(orientation_quaternion_wxyz=[2.0, 0.0, 0.0, 0.0]),
                index=0,
            )

    def test_accepts_448_request_when_collector_dimensions_match(self):
        request = RenderRequest.from_payload(
            request_payload(expected_width=448, expected_height=448),
            index=0,
            expected_width=448,
            expected_height=448,
        )

        self.assertEqual(request.expected_width, 448)
        self.assertEqual(request.expected_height, 448)
        with self.assertRaisesRegex(ValueError, "224x224x3"):
            RenderRequest.from_payload(
                request_payload(expected_width=256),
                index=0,
            )

    def test_indexes_named_openfly_scenes_in_natural_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request_path = root / "requests.jsonl"
            payloads = [
                request_payload(
                    episode_id="episode-18", request_id="episode-18/front_image/frame_000000",
                    scene_id="env_airsim_18", source_episode_index=0,
                ),
                request_payload(
                    episode_id="episode-gz", request_id="episode-gz/front_image/frame_000000",
                    scene_id="env_airsim_gz", source_episode_index=1,
                ),
            ]
            request_path.write_text(
                "".join(json.dumps(item) + "\n" for item in payloads),
                encoding="utf-8",
            )
            state = CollectorState(root / "state.sqlite3")
            try:
                summary = build_request_index(request_path, state)
            finally:
                state.close()

            self.assertEqual(summary.scene_ids, ("env_airsim_18", "env_airsim_gz"))

class SceneAssignmentTests(unittest.TestCase):
    def test_new_worker_prefers_a_scene_without_running_worker(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = CollectorState(Path(temporary_directory) / "state.sqlite3")
            try:
                state.add_episode("scene-a-0", "scene-a", 0, 0, 1, 1)
                state.add_episode("scene-a-1", "scene-a", 1, 1, 2, 1)
                state.add_episode("scene-b-0", "scene-b", 2, 2, 3, 1)
                first = state.claim_next_job("worker-0")
                second = state.claim_next_job("worker-1")
            finally:
                state.close()

        self.assertEqual(first.scene_id, "scene-a")
        self.assertEqual(second.scene_id, "scene-b")


class CameraPoseNoiseTests(unittest.TestCase):
    def test_legacy_zero_position_fallback_does_not_change_camera_pose(self):
        normal = episode_camera_records(
            "episode", seed=1, views=("front", "back")
        )
        legacy_fallback = episode_camera_records(
            "episode", seed=1, views=("front", "back"),
            zero_position_delta_views=("back",),
        )
        self.assertEqual(legacy_fallback, normal)


class LayoutTests(unittest.TestCase):
    def test_maps_source_episode_to_lerobot_chunk_and_part(self):
        self.assertEqual(part_location(0), (0, 0))
        self.assertEqual(part_location(19), (0, 0))
        self.assertEqual(part_location(20), (0, 1))
        self.assertEqual(part_location(999), (0, 49))
        self.assertEqual(part_location(1000), (1, 0))
        self.assertEqual(part_location(16376), (16, 18))


class CameraRowsTests(unittest.TestCase):
    def test_camera_rows_are_episode_deterministic_and_view_independent(self):
        first = episode_camera_rows(
            episode_id="episode-7", scene_id="5", seed=1, views="all"
        )
        repeated = episode_camera_rows(
            episode_id="episode-7", scene_id="5", seed=1, views="all"
        )
        other = episode_camera_rows(
            episode_id="episode-8", scene_id="5", seed=1, views="all"
        )

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other)
        self.assertEqual(len({row["fov_degrees"] for row in first}), 4)
        self.assertTrue(all(90.0 <= row["fov_degrees"] <= 120.0 for row in first))
        self.assertEqual(
            {row["view"]: row["base_pose"] for row in first},
            {
                "front": {"x": 0.25, "y": 0.0, "z": 0.0, "yaw": 0, "pitch": 0.0, "roll": 0.0},
                "back": {"x": -0.25, "y": 0.0, "z": 0.0, "yaw": 180, "pitch": 0.0, "roll": 0.0},
                "left": {"x": 0.0, "y": -0.25, "z": 0.0, "yaw": -90, "pitch": 0.0, "roll": 0.0},
                "right": {"x": 0.0, "y": 0.25, "z": 0.0, "yaw": 90, "pitch": 0.0, "roll": 0.0},
            },
        )


class CollectorCliTests(unittest.TestCase):
    def test_accepts_one_isolated_environment_root_per_render_worker(self):
        parser = build_parser()
        args = parser.parse_args([
            "render",
            "--package-dir", "/tmp/package",
            "--env-archive-root", "/tmp/archive",
            "--env-cache-root", "/tmp/shared-cache",
            "--gpus", "4,5,6,7",
            "--workers", "4",
            "--worker-env-cache-roots",
            "/tmp/cache-4,/tmp/cache-5,/tmp/cache-6,/tmp/cache-7",
        ])

        config = CollectorPipeline.from_args(args).config

        self.assertEqual(
            tuple(str(path) for path in config.worker_env_cache_roots),
            ("/tmp/cache-4", "/tmp/cache-5", "/tmp/cache-6", "/tmp/cache-7"),
        )

    def test_channel_order_is_explicit_rgb_or_bgr_without_auto_detection(self):
        parser = build_parser()
        subparsers = next(action for action in parser._actions if action.dest == "command")
        render_parser = subparsers.choices["render"]
        channel_action = next(
            action for action in render_parser._actions if action.dest == "channel_order"
        )
        self.assertEqual(channel_action.default, "rgb")
        self.assertEqual(tuple(channel_action.choices), ("rgb", "bgr"))

    def test_retry_defaults_use_ten_frame_attempts_without_scene_restarts(self):
        parser = build_parser()
        subparsers = next(action for action in parser._actions if action.dest == "command")
        render_parser = subparsers.choices["render"]
        actions = {action.dest: action for action in render_parser._actions}
        self.assertEqual(actions["frame_attempts"].default, 10)
        self.assertEqual(actions["failed_episode_retry_rounds"].default, 3)
        self.assertNotIn("scene_restarts", actions)

    def test_accepts_configurable_448_image_dimensions(self):
        parser = build_parser()
        args = parser.parse_args([
            "render",
            "--package-dir", "/tmp/package",
            "--env-archive-root", "/tmp/archive",
            "--env-cache-root", "/tmp/cache",
            "--gpus", "0",
            "--workers", "1",
            "--image-width", "448",
            "--image-height", "448",
        ])

        config = CollectorPipeline.from_args(args).config

        self.assertEqual(config.image_width, 448)
        self.assertEqual(config.image_height, 448)


class RenderProgressTests(unittest.TestCase):
    def test_reports_completed_remaining_and_retry_queues(self):
        self.assertTrue(hasattr(worker, "_render_progress_payload"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_path = Path(temporary_directory) / "state.sqlite3"
            state = CollectorState(state_path)
            request_counts = (10, 20, 30, 40, 50, 60)
            for index, request_count in enumerate(request_counts):
                state.add_episode(
                    "episode-{}".format(index), "5", index,
                    index, index + 1, request_count,
                )
            state.connection.executemany(
                "UPDATE jobs SET status=?, attempts=? WHERE episode_id=?",
                (
                    ("complete", 1, "episode-0"),
                    ("pending", 0, "episode-1"),
                    ("running", 1, "episode-2"),
                    ("failed", 3, "episode-3"),
                    ("failed", 10, "episode-4"),
                    ("failed", 11, "episode-5"),
                ),
            )
            state.connection.commit()
            state.close()

            payload = worker._render_progress_payload(
                str(state_path), failed_episode_retry_rounds=10
            )

            self.assertEqual(payload["episodes_total"], 6)
            self.assertEqual(payload["episodes_complete"], 1)
            self.assertEqual(payload["episodes_remaining"], 5)
            self.assertEqual(payload["requests_total"], 210)
            self.assertEqual(payload["requests_complete"], 10)
            self.assertEqual(payload["requests_remaining"], 200)
            self.assertEqual(payload["retry_waiting"], 2)
            self.assertEqual(payload["final_retry_next"], 1)
            self.assertEqual(payload["retry_exhausted"], 1)


class DefaultCollectionEntryTests(unittest.TestCase):
    def test_waypoint_entry_runs_from_an_external_working_directory(self):
        repository = Path(__file__).parents[1]
        entry = repository / "scripts" / "collect_waypoints.sh"
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = subprocess.run(
                [str(entry), "--help"],
                cwd=temporary_directory,
                env={"PATH": os.environ["PATH"], "VLN_COLLECT_PYTHON": sys.executable},
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: python -m waypoint_collector", result.stdout)


if __name__ == "__main__":
    unittest.main()
