import json
from pathlib import Path
import tempfile
import unittest

from waypoint_collector.publish import build_updated_manifest, publish_artifacts


class PublishTests(unittest.TestCase):
    def test_builds_partial_manifest_with_exact_counts(self):
        manifest = build_updated_manifest(
            {"dataset_key": "AerialVLN_lerobot", "trajectory_only": True},
            total_episodes=16368,
            available_episodes=14275,
            missing_episodes=2093,
            total_requests=2019918,
            available_requests=1474558,
            missing_requests=545360,
            skipped_scenes=("1",),
            views=("front", "back", "left", "right"),
            image_width=448,
            image_height=448,
        )

        self.assertFalse(manifest["trajectory_only"])
        self.assertFalse(manifest["complete_lerobot_split"])
        self.assertEqual(manifest["image_status"], "partial")
        self.assertEqual(manifest["missing_scene_ids"], ["1"])
        self.assertEqual(manifest["available_episode_count"], 14275)
        self.assertEqual(manifest["available_render_request_count"], 1474558)
        self.assertEqual(manifest["image_width"], 448)
        self.assertEqual(manifest["image_height"], 448)

    def test_publishes_videos_and_metadata_then_replaces_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package = root / "package"
            staging = package / ".render_staging/run-1/final"
            (staging / "videos/front_image/chunk-000").mkdir(parents=True)
            (staging / "videos/front_image/chunk-000/part-000.mp4").write_bytes(b"video")
            (staging / "meta").mkdir()
            (staging / "meta/navvla_video_index.parquet").write_bytes(b"index")
            package.mkdir(exist_ok=True)
            (package / "manifest.json").write_text('{"image_status":"not_collected"}\n')
            manifest = {"image_status": "partial"}

            publish_artifacts(staging, package, manifest)

            self.assertTrue((package / "videos/front_image/chunk-000/part-000.mp4").is_file())
            self.assertTrue((package / "meta/navvla_video_index.parquet").is_file())
            self.assertEqual(json.loads((package / "manifest.json").read_text()), manifest)

    def test_resumes_publish_after_video_tree_was_already_moved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package = root / "package"
            staging = package / ".render_staging/run-1/final"
            (package / "videos/front_image").mkdir(parents=True)
            (package / "videos/front_image/part.mp4").write_bytes(b"video")
            (staging / "meta").mkdir(parents=True)
            (staging / "meta/navvla_video_index.parquet").write_bytes(b"index")
            package.mkdir(exist_ok=True)
            journal = {
                "staging_final_root": str(staging.resolve()),
                "manifest": {"image_status": "partial"},
            }
            (package / ".waypoint_publish.json").write_text(json.dumps(journal))

            publish_artifacts(staging, package, journal["manifest"])

            self.assertTrue((package / "meta/navvla_video_index.parquet").is_file())
            self.assertEqual(json.loads((package / "manifest.json").read_text()), journal["manifest"])
            self.assertFalse((package / ".waypoint_publish.json").exists())


if __name__ == "__main__":
    unittest.main()
