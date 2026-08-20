import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from waypoint_collector.assembly import assemble_videos
from waypoint_collector.indexing import build_request_index
from waypoint_collector.metadata import write_metadata_artifacts
from waypoint_collector.state import CollectorState
from waypoint_collector.validation import validate_artifacts
from waypoint_collector import video
from waypoint_collector.video import EpisodeVideoSink


def payload(episode, source_index, scene, image_index, waypoint_index,
            image_width=224, image_height=224):
    return {
        "request_id": "{}/front_image/frame_{:06d}".format(episode, image_index),
        "episode_id": episode,
        "trajectory_id": "trajectory-{}".format(source_index),
        "source_episode_index": source_index,
        "scene_id": str(scene),
        "image_index": image_index,
        "waypoint_index": waypoint_index,
        "timestamp": float(waypoint_index),
        "position_xyz": [1.0, 2.0, -3.0],
        "orientation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "expected_height": image_height,
        "expected_width": image_width,
        "expected_channels": 3,
    }


def frame(value, image_width=224, image_height=224):
    result = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    result[:, :, 0] = value
    result[0, 0, 1] = value + 1
    return result


class Request:
    def __init__(self, image_index):
        self.image_index = image_index


class AssemblyValidationTests(unittest.TestCase):
    def test_native_448_frames_encode_assemble_and_validate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request_path = root / "requests.jsonl"
            rows = [
                payload("a", 0, 5, 0, 0, 448, 448),
                payload("a", 0, 5, 1, 5, 448, 448),
            ]
            request_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            state = CollectorState(root / "state.sqlite3")
            try:
                build_request_index(
                    request_path, state, image_width=448, image_height=448
                )
                state.mark_complete("a")
                views = ("front", "back", "left", "right")
                sink = EpisodeVideoSink(
                    root / "render", "a", views,
                    image_width=448, image_height=448,
                )
                for image_index in range(2):
                    sink.append(
                        Request(image_index),
                        {
                            view: frame(
                                10 + offset + image_index, 448, 448
                            )
                            for offset, view in enumerate(views)
                        },
                    )
                sink.commit()
                assemble_videos(
                    state, root / "render", root / "final", views, ()
                )
                write_metadata_artifacts(
                    request_path, state, root / "final/meta", views,
                    skipped_scenes=(), image_width=448, image_height=448,
                )
                validation = validate_artifacts(
                    root / "final", state, views, skipped_scenes=(),
                    deep_video_probe=True,
                    image_width=448, image_height=448,
                )
            finally:
                state.close()

            cameras = json.loads(
                (root / "final/meta/navvla_cameras.json").read_text()
            )

        self.assertEqual(validation.video_file_count, 4)
        self.assertTrue(
            all(camera["width"] == 448 for camera in cameras["cameras"])
        )
        self.assertTrue(
            all(camera["height"] == 448 for camera in cameras["cameras"])
        )

    def test_assembly_runs_independent_parts_in_parallel(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = CollectorState(root / "state.sqlite3")
            views = ("front",)
            for episode_id, source_index in (("first", 0), ("second", 20)):
                state.add_episode(
                    episode_id, "5", source_index, 0, 10, 1
                )
                state.mark_complete(episode_id)
                path = (
                    root / "render" / "episodes" / "front" /
                    "{}.mp4".format(episode_id)
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"episode")
                video.write_episode_commit_marker(
                    root / "render", episode_id, views, {"front": 1}
                )

            barrier = threading.Barrier(2)
            active = 0
            peak_active = 0
            lock = threading.Lock()

            def copy_part(_inputs, output_path):
                nonlocal active, peak_active
                with lock:
                    active += 1
                    peak_active = max(peak_active, active)
                barrier.wait(timeout=2)
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_bytes(b"part")
                with lock:
                    active -= 1

            with patch(
                "waypoint_collector.assembly.assemble_part_video",
                side_effect=copy_part,
            ):
                summary = assemble_videos(
                    state, root / "render", root / "final", views, (),
                    assembly_workers=2,
                )

            self.assertEqual(summary.video_file_count, 2)
            self.assertEqual(peak_active, 2)
            state.close()

    def test_assembles_available_episodes_and_validates_missing_placeholders(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request_path = root / "requests.jsonl"
            rows = [
                payload("a", 0, 5, 0, 0),
                payload("a", 0, 5, 1, 5),
                payload("b", 1, 1, 0, 0),
            ]
            request_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            state = CollectorState(root / "state.sqlite3")
            build_request_index(request_path, state, skipped_scenes=("1",))
            state.mark_complete("a")
            views = ("front", "back", "left", "right")
            sink = EpisodeVideoSink(root / "render", "a", views)
            self.assertEqual(sink.attempt_dir.parent.parent, root / "render" / "attempts")
            self.assertFalse(
                any(
                    path.parent == root / "render" / "episodes" / view
                    for view, path in sink.temporary_paths.items()
                )
            )
            for image_index in range(2):
                sink.append(
                    Request(image_index),
                    {view: frame(10 + offset + image_index) for offset, view in enumerate(views)},
                )
            sink.commit()

            marker_path = video.episode_marker_path(root / "render", "a")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(marker["schema_version"], 1)
            self.assertEqual(marker["attempt_id"], sink.attempt_id)
            self.assertEqual(marker["episode_id"], "a")
            self.assertEqual(set(marker["videos"]), set(views))
            for view in views:
                self.assertEqual(
                    marker["videos"][view],
                    {
                        "path": "episodes/{}/a.mp4".format(view),
                        "frame_count": 2,
                    },
                )
            self.assertTrue(
                video.validate_episode_commit(
                    root / "render", "a", views, expected_frames=2
                )
            )
            self.assertFalse(sink.attempt_dir.exists())

            assembly = assemble_videos(
                state=state,
                episode_video_root=root / "render",
                final_root=root / "final",
                views=views,
                skipped_scenes=("1",),
            )
            metadata = write_metadata_artifacts(
                request_path, state, root / "final/meta", views,
                skipped_scenes=("1",), batch_size=2,
            )
            validation = validate_artifacts(
                final_root=root / "final",
                state=state,
                views=views,
                skipped_scenes=("1",),
            )

            self.assertEqual(assembly.video_file_count, 4)
            self.assertEqual(assembly.available_frame_count, 8)
            self.assertEqual(metadata.total_rows, 12)
            self.assertEqual(validation.video_file_count, 4)
            self.assertEqual(validation.available_index_rows, 8)
            self.assertEqual(validation.missing_index_rows, 4)

            with patch("waypoint_collector.validation.probe_video") as probe:
                fast_validation = validate_artifacts(
                    final_root=root / "final",
                    state=state,
                    views=views,
                    skipped_scenes=("1",),
                    deep_video_probe=False,
                )
            probe.assert_not_called()
            self.assertEqual(fast_validation.video_file_count, 4)

    def test_commit_failure_before_marker_removes_staged_and_promoted_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sink = EpisodeVideoSink(root, "episode", ("front", "back"))
            sink.append(
                Request(0),
                {"front": frame(10), "back": frame(11)},
            )

            with patch(
                "waypoint_collector.video.write_episode_commit_marker",
                side_effect=OSError("marker write failed"),
            ):
                with self.assertRaisesRegex(OSError, "marker write failed"):
                    sink.commit()

            self.assertFalse(sink.attempt_dir.exists())
            self.assertFalse(video.episode_marker_path(root, "episode").exists())
            for path in sink.final_paths.values():
                self.assertFalse(path.exists())

    def test_commit_validation_rejects_malformed_mismatched_and_traversal_markers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            marker_path = root / "commits" / "episode.json"
            marker_path.parent.mkdir(parents=True)

            invalid_markers = (
                "not-json",
                json.dumps({"schema_version": 1}),
                json.dumps({
                    "schema_version": 1,
                    "attempt_id": "d6ce4ea1-3b28-42b7-8a70-f2924c4a61fd",
                    "episode_id": "other",
                    "videos": {
                        "front": {
                            "path": "episodes/front/episode.mp4",
                            "frame_count": 1,
                        }
                    },
                }),
                json.dumps({
                    "schema_version": 1,
                    "attempt_id": "d6ce4ea1-3b28-42b7-8a70-f2924c4a61fd",
                    "episode_id": "episode",
                    "videos": {
                        "front": {
                            "path": "../outside.mp4",
                            "frame_count": 1,
                        }
                    },
                }),
            )
            for contents in invalid_markers:
                with self.subTest(contents=contents):
                    marker_path.write_text(contents, encoding="utf-8")
                    self.assertFalse(
                        video.validate_episode_commit(
                            root, "episode", ("front",), expected_frames=1
                        )
                    )

    def test_assembly_blocks_if_any_non_skipped_episode_is_not_complete(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = CollectorState(root / "state.sqlite3")
            state.add_episode("a", "5", 0, 0, 10, 2)

            with self.assertRaisesRegex(RuntimeError, "not complete"):
                assemble_videos(
                    state, root / "render", root / "final",
                    ("front", "back", "left", "right"), ("1",),
                )

    def test_assembly_rejects_complete_episode_without_valid_commit_marker(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = CollectorState(root / "state.sqlite3")
            state.add_episode("episode", "5", 0, 0, 10, 1)
            state.mark_complete("episode")
            episode_path = root / "render/episodes/front/episode.mp4"
            episode_path.parent.mkdir(parents=True)
            episode_path.write_bytes(b"validated-on-commit")

            marker_path = video.episode_marker_path(root / "render", "episode")
            cases = (None, "not-json")
            for marker_contents in cases:
                with self.subTest(marker_contents=marker_contents):
                    marker_path.unlink(missing_ok=True)
                    if marker_contents is not None:
                        marker_path.parent.mkdir(parents=True, exist_ok=True)
                        marker_path.write_text(marker_contents, encoding="utf-8")
                    with patch(
                        "waypoint_collector.video.probe_video",
                        return_value={
                            "codec_name": "h264",
                            "profile": "High",
                            "pix_fmt": "yuv420p",
                            "width": 224,
                            "height": 224,
                            "avg_frame_rate": "1/1",
                            "frame_count": 1,
                        },
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "invalid episode commit"
                        ):
                            assemble_videos(
                                state, root / "render", root / "final",
                                ("front",), (),
                            )
            state.close()

    def test_assembly_accepts_valid_marker_backed_episode(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = CollectorState(root / "state.sqlite3")
            state.add_episode("episode", "5", 0, 0, 10, 1)
            state.mark_complete("episode")
            episode_path = root / "render/episodes/front/episode.mp4"
            episode_path.parent.mkdir(parents=True)
            episode_path.write_bytes(b"validated-on-commit")
            video.write_episode_commit_marker(
                root / "render", "episode", ("front",), {"front": 1},
                attempt_id="d6ce4ea1-3b28-42b7-8a70-f2924c4a61fd",
            )

            with patch(
                "waypoint_collector.video.probe_video",
                return_value={
                    "codec_name": "h264",
                    "profile": "High",
                    "pix_fmt": "yuv420p",
                    "width": 224,
                    "height": 224,
                    "avg_frame_rate": "1/1",
                    "frame_count": 1,
                },
            ), patch("waypoint_collector.assembly.assemble_part_video"):
                result = assemble_videos(
                    state, root / "render", root / "final", ("front",), (),
                )

            self.assertEqual(result.video_file_count, 1)
            self.assertEqual(result.available_frame_count, 1)
            state.close()

    def test_assembly_uses_marker_only_validation_before_concat(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = CollectorState(root / "state.sqlite3")
            state.add_episode("episode", "5", 0, 0, 10, 1)
            state.mark_complete("episode")
            episode_path = root / "render/episodes/front/episode.mp4"
            episode_path.parent.mkdir(parents=True)
            episode_path.write_bytes(b"validated-on-commit")
            video.write_episode_commit_marker(
                root / "render", "episode", ("front",), {"front": 1},
                attempt_id="d6ce4ea1-3b28-42b7-8a70-f2924c4a61fd",
            )

            with patch(
                "waypoint_collector.assembly.validate_episode_commit",
                return_value=True,
            ) as validate_commit, patch("waypoint_collector.assembly.assemble_part_video"):
                assemble_videos(
                    state, root / "render", root / "final", ("front",), (),
                )

            self.assertEqual(validate_commit.call_args.kwargs["probe_videos"], False)
            state.close()


if __name__ == "__main__":
    unittest.main()
