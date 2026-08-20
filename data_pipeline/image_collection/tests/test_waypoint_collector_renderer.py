import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from msgpackrpc import error as rpc_error
import numpy as np

from waypoint_collector import airsim_session, renderer, video, worker
from waypoint_collector.renderer import EpisodeRenderError, render_episode
from waypoint_collector.indexing import build_request_index
from waypoint_collector.pipeline import CollectorConfig, CollectorPipeline
from waypoint_collector.requests import RenderRequest
from waypoint_collector.state import CollectorState
from waypoint_collector.worker import (
    WorkerConfig,
    _episode_videos_complete,
    render_worker,
)


def request(image_index=0):
    return RenderRequest.from_payload(
        {
            "request_id": "episode-7/front_image/frame_{:06d}".format(image_index),
            "episode_id": "episode-7",
            "trajectory_id": "trajectory-7",
            "source_episode_index": 7,
            "scene_id": "5",
            "image_index": image_index,
            "waypoint_index": image_index * 5,
            "timestamp": image_index * 5.0,
            "position_xyz": [float(image_index), 2.0, -3.0],
            "orientation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "expected_height": 224,
            "expected_width": 224,
            "expected_channels": 3,
        },
        index=image_index,
    )


def valid_frame(value):
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[:, :, 0] = value
    frame[0, 0, 1] = value + 1
    return frame


class FakeSession:
    def __init__(self, fail_captures=0):
        self.fail_captures = fail_captures
        self.camera_calls = []
        self.pose_calls = []
        self.capture_calls = []
        self.closed = False
        self.close_count = 0

    def apply_camera_records(self, records):
        self.camera_calls.append(tuple(records))

    def set_vehicle_pose(self, item):
        self.pose_calls.append(item)

    def verify_vehicle_pose(self, item):
        return 0.0, 0.0

    def capture_rgb(self, views):
        self.capture_calls.append(tuple(views))
        if self.fail_captures:
            self.fail_captures -= 1
            raise RuntimeError("temporary image failure")
        return {view: valid_frame(index + 10) for index, view in enumerate(views)}

    def close(self, **_kwargs):
        self.closed = True
        self.close_count += 1


class CameraSetupLossSession(FakeSession):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def apply_camera_records(self, records):
        super().apply_camera_records(records)
        raise self.error


class WaypointLossSession(FakeSession):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def capture_rgb(self, views):
        self.capture_calls.append(tuple(views))
        raise self.error


class ConstantFrameSession(FakeSession):
    def capture_rgb(self, views):
        self.capture_calls.append(tuple(views))
        return {
            view: np.zeros((224, 224, 3), dtype=np.uint8)
            for view in views
        }


class ReadbackSession(FakeSession):
    def __init__(self, position_error, rotation_error):
        super().__init__()
        self.position_error = position_error
        self.rotation_error = rotation_error

    def verify_vehicle_pose(self, item):
        return self.position_error, self.rotation_error


class RecordingSink:
    def __init__(self):
        self.frames = []
        self.committed = False
        self.aborted = False

    def append(self, item, frames):
        self.frames.append((item.image_index, tuple(sorted(frames))))

    def commit(self):
        self.committed = True

    def abort(self):
        self.aborted = True


class FailingAppendSink(RecordingSink):
    def append(self, item, frames):
        raise OSError("video writer failed")


class EpisodeSession(FakeSession):
    def __init__(self, failures_remaining=1):
        super().__init__()
        self.failures_remaining = failures_remaining

    def set_vehicle_pose(self, item):
        super().set_vehicle_pose(item)
        self.current_episode = item.episode_id

    def capture_rgb(self, views):
        if self.current_episode == "episode-fail" and self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("recoverable failure")
        return super().capture_rgb(views)


class FakeRuntime:
    def __init__(self, session=None, **_kwargs):
        self.session = session or EpisodeSession()
        self.open_count = 0
        self.close_count = 0

    def open_scene(self, _scene_id, channel_order="rgb"):
        self.open_count += 1
        return self.session

    def close(self, **_kwargs):
        self.close_count += 1


class TrackingRuntime(FakeRuntime):
    def __init__(self, session=None, open_error=None, close_error=None, **kwargs):
        super().__init__(session=session, **kwargs)
        self.open_error = open_error
        self.close_error = close_error
        self.opened_scenes = []

    def open_scene(self, scene_id, channel_order="rgb"):
        self.open_count += 1
        self.opened_scenes.append(str(scene_id))
        if self.open_error is not None:
            raise self.open_error
        return self.session

    def close(self, **_kwargs):
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


class RuntimeSequenceFactory:
    def __init__(self, runtimes):
        self.runtimes = list(runtimes)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.runtimes:
            raise AssertionError("unexpected runtime construction")
        return self.runtimes.pop(0)


class WorkerSink(RecordingSink):
    completed_episodes = []
    aborted_episodes = []

    def __init__(self, _root, episode_id, _views, **_kwargs):
        super().__init__()
        self.episode_id = episode_id

    def commit(self):
        super().commit()
        self.completed_episodes.append(self.episode_id)

    def abort(self):
        super().abort()
        self.aborted_episodes.append(self.episode_id)


class FailingAbortWorkerSink(WorkerSink):
    def abort(self):
        super().abort()
        raise OSError("sink abort failed")


class CloseFailingSession(WaypointLossSession):
    def close(self):
        self.close_count += 1
        raise RuntimeError("session cleanup failed")


class RendererTests(unittest.TestCase):
    def test_accepts_constant_rgb_frame_as_valid_scene_content(self):
        self.assertTrue(renderer.validate_rgb_frames(
            {"front": np.zeros((224, 224, 3), dtype=np.uint8)},
            ("front",),
        ))

    def test_accepts_configured_448_rgb_frame(self):
        self.assertTrue(renderer.validate_rgb_frames(
            {"front": np.zeros((448, 448, 3), dtype=np.uint8)},
            ("front",),
            image_width=448,
            image_height=448,
        ))

    @staticmethod
    def _valid_probe(frame_count=1):
        return {
            "codec_name": "h264",
            "profile": "High",
            "pix_fmt": "yuv420p",
            "width": 224,
            "height": 224,
            "avg_frame_rate": "1/1",
            "frame_count": frame_count,
        }

    @staticmethod
    def _touch_episode_videos(root, episode_id, views, contents=b"legacy"):
        paths = {}
        for view_name in views:
            path = root / "episodes" / view_name / "{}.mp4".format(episode_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
            paths[view_name] = path
        return paths

    @staticmethod
    def _write_commit_marker(root, episode_id, views, frame_count=1):
        marker_path = root / "commits" / "{}.json".format(episode_id)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(
            json.dumps({
                "schema_version": 1,
                "attempt_id": "d6ce4ea1-3b28-42b7-8a70-f2924c4a61fd",
                "episode_id": episode_id,
                "videos": {
                    view_name: {
                        "path": "episodes/{}/{}.mp4".format(
                            view_name, episode_id
                        ),
                        "frame_count": frame_count,
                    }
                    for view_name in views
                },
            }) + "\n",
            encoding="utf-8",
        )
        return marker_path

    @staticmethod
    def _session_loss(message="AirSim disconnected"):
        transport_error = rpc_error.TransportError(message)
        unavailable = airsim_session.AirSimSessionUnavailableError(message)
        unavailable.__cause__ = transport_error
        return unavailable, transport_error

    @staticmethod
    def _write_worker_jobs(root, episode_ids):
        request_path = root / "requests.jsonl"
        rows = []
        for index, episode_id in enumerate(episode_ids):
            payload = {
                "request_id": "{}/front_image/frame_000000".format(episode_id),
                "episode_id": episode_id,
                "trajectory_id": "trajectory-{}".format(index),
                "source_episode_index": index,
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
            rows.append(json.dumps(payload) + "\n")
        request_path.write_text("".join(rows), encoding="utf-8")
        state_path = root / "state.sqlite3"
        state = CollectorState(state_path)
        build_request_index(request_path, state)
        state.close()
        return request_path, state_path

    @staticmethod
    def _worker_config(root, request_path, state_path, frame_attempts=1):
        return WorkerConfig(
            worker_index=0,
            gpu=0,
            control_port=31000,
            repository_root=str(root),
            env_cache_root=str(root),
            request_path=str(request_path),
            state_path=str(state_path),
            episode_video_root=str(root / "videos"),
            log_path=str(root / "worker.log"),
            views=("front",),
            skipped_scenes=(),
            camera_seed=1,
            channel_order="rgb",
            frame_attempts=frame_attempts,
            failed_episode_retry_rounds=3,
        )

    def test_completed_episode_video_check_uses_all_view_frame_counts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for view in ("front", "back"):
                path = root / "episodes" / view / "episode.mp4"
                path.parent.mkdir(parents=True)
                path.touch()
            self._write_commit_marker(
                root, "episode", ("front", "back"), frame_count=7
            )
            with patch(
                "waypoint_collector.video.probe_video",
                return_value=self._valid_probe(frame_count=7),
            ):
                self.assertTrue(
                    _episode_videos_complete(
                        root, "episode", ("front", "back"), expected_frames=7
                    )
                )

    def test_reconciliation_finalizes_marker_backed_noncomplete_job(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = CollectorState(root / "state.sqlite3")
            state.add_episode("episode", "5", 0, 0, 10, 1)
            self._touch_episode_videos(root / "videos", "episode", ("front",))
            self._write_commit_marker(
                root / "videos", "episode", ("front",), frame_count=1
            )

            with patch(
                "waypoint_collector.video.probe_video",
                return_value=self._valid_probe(),
            ) as probe_video:
                worker.reconcile_episode_outputs(
                    state, root / "videos", ("front",)
                )

            self.assertEqual(state.job_status("episode"), "complete")
            probe_video.assert_not_called()
            state.close()

    def test_reconciliation_skips_new_run_without_rendered_episode_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = CollectorState(root / "state.sqlite3")
            state.add_episode("episode", "5", 0, 0, 10, 1)

            with patch.object(state, "reset_job_pending") as reset_pending:
                worker.reconcile_episode_outputs(
                    state, root / "rendered_episodes", ("front",)
                )

            self.assertEqual(state.job_status("episode"), "pending")
            reset_pending.assert_not_called()
            state.close()

    def test_reconciliation_skips_pending_episode_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = CollectorState(root / "state.sqlite3")
            state.add_episode("episode", "5", 0, 0, 10, 1)
            video_root = root / "rendered_episodes"
            video_root.mkdir()

            with (
                patch.object(state, "reset_job_pending") as reset_pending,
                patch(
                    "waypoint_collector.worker.validate_episode_commit"
                ) as validate_commit,
            ):
                worker.reconcile_episode_outputs(
                    state, video_root, ("front",)
                )

            self.assertEqual(state.job_status("episode"), "pending")
            reset_pending.assert_not_called()
            validate_commit.assert_not_called()
            state.close()

    def test_reconciliation_backfills_legacy_complete_marker_without_rewriting_videos(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = CollectorState(root / "state.sqlite3")
            state.add_episode("episode", "5", 0, 0, 10, 1)
            state.mark_complete("episode")
            paths = self._touch_episode_videos(
                root / "videos", "episode", ("front", "back")
            )
            before = {view_name: path.read_bytes() for view_name, path in paths.items()}

            with patch(
                "waypoint_collector.worker.validate_episode_videos"
            ) as validate_videos:
                worker.reconcile_episode_outputs(
                    state, root / "videos", ("front", "back")
                )

            validate_videos.assert_not_called()

            self.assertEqual(state.job_status("episode"), "complete")
            self.assertTrue(
                video.episode_marker_path(root / "videos", "episode").is_file()
            )
            self.assertEqual(
                {view_name: path.read_bytes() for view_name, path in paths.items()},
                before,
            )
            state.close()

    def test_reconciliation_resets_invalid_complete_job_and_removes_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = CollectorState(root / "state.sqlite3")
            state.add_episode("episode", "5", 0, 0, 10, 1)
            state.mark_complete("episode")
            paths = self._touch_episode_videos(
                root / "videos", "episode", ("front", "back")
            )
            self._write_commit_marker(
                root / "videos", "episode", ("front", "back"), frame_count=1
            )
            paths["back"].unlink()

            worker.reconcile_episode_outputs(
                state, root / "videos", ("front", "back")
            )

            self.assertEqual(state.job_status("episode"), "pending")
            self.assertFalse(
                video.episode_marker_path(root / "videos", "episode").exists()
            )
            self.assertFalse(paths["front"].exists())
            state.close()

    def test_reconciliation_removes_abandoned_attempt_and_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = CollectorState(root / "state.sqlite3")
            state.add_episode("episode", "5", 0, 0, 10, 1)
            attempt = (
                root / "videos" / "attempts" / "episode" /
                "d6ce4ea1-3b28-42b7-8a70-f2924c4a61fd"
            )
            attempt.mkdir(parents=True)
            (attempt / "front.mp4").touch()
            partial = (
                root / "videos" / "episodes" / "front" /
                ".episode.partial.mp4"
            )
            partial.parent.mkdir(parents=True)
            partial.touch()

            worker.reconcile_episode_outputs(
                state, root / "videos", ("front",)
            )

            self.assertEqual(state.job_status("episode"), "pending")
            self.assertFalse(attempt.exists())
            self.assertFalse(partial.exists())
            state.close()

    def test_worker_skips_only_valid_marker_backed_complete_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request_path, state_path = self._write_worker_jobs(root, ("episode",))
            self._touch_episode_videos(root / "videos", "episode", ("front",))
            self._write_commit_marker(
                root / "videos", "episode", ("front",), frame_count=1
            )
            runtime = TrackingRuntime()

            with patch(
                "waypoint_collector.video.probe_video",
                return_value=self._valid_probe(),
            ):
                result = render_worker(
                    self._worker_config(root, request_path, state_path),
                    runtime_factory=lambda **_kwargs: runtime,
                    sink_factory=lambda *_args: self.fail("sink must not be created"),
                )

            state = CollectorState(state_path)
            self.assertEqual(result, 0)
            self.assertEqual(state.job_status("episode"), "complete")
            self.assertEqual(runtime.open_count, 0)
            state.close()

    def test_retries_an_individual_frame_three_total_attempts(self):
        session = FakeSession(fail_captures=2)
        sink = RecordingSink()

        render_episode(
            session, [request()], ("record",), ("front", "back", "left", "right"),
            sink, frame_attempts=3,
        )

        self.assertEqual(len(session.pose_calls), 3)
        self.assertEqual(len(session.capture_calls), 3)
        self.assertEqual(len(sink.frames), 1)

    def test_camera_setup_session_loss_is_classified_immediately(self):
        unavailable, transport_error = self._session_loss()
        session = CameraSetupLossSession(unavailable)

        with self.assertRaises(renderer.SessionRenderError) as caught:
            render_episode(
                session, [request()], ("record",), ("front",), RecordingSink(),
                frame_attempts=3,
            )

        self.assertIs(caught.exception.__cause__, unavailable)
        self.assertIs(caught.exception.__cause__.__cause__, transport_error)
        self.assertEqual(len(session.camera_calls), 1)
        self.assertEqual(len(session.pose_calls), 0)

    def test_waypoint_session_loss_retries_then_preserves_original_cause(self):
        unavailable, transport_error = self._session_loss()
        session = WaypointLossSession(unavailable)

        with self.assertRaises(renderer.SessionRenderError) as caught:
            render_episode(
                session, [request()], ("record",), ("front",), RecordingSink(),
                frame_attempts=3,
            )

        self.assertEqual(len(session.pose_calls), 3)
        self.assertEqual(len(session.capture_calls), 3)
        self.assertIs(caught.exception.__cause__, unavailable)
        self.assertIs(caught.exception.__cause__.__cause__, transport_error)

    def test_constant_frames_are_written_without_error(self):
        sink = RecordingSink()
        rendered = render_episode(
            ConstantFrameSession(), [request()], ("record",), ("front",),
            sink, frame_attempts=2,
        )

        self.assertEqual(rendered, 1)
        self.assertEqual(sink.frames, [(0, ("front",))])

    def test_worker_persists_invalid_frame_debug_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = worker.write_invalid_frame_debug(
                root,
                "episode-constant",
                {"front": np.zeros((224, 224, 3), dtype=np.uint8)},
                "front frame is empty or constant",
            )

            self.assertEqual(
                paths,
                (root / "debug_invalid_frames" / "episode-constant" / "front.png",),
            )
            self.assertTrue(paths[0].is_file())
            self.assertEqual(
                (root / "debug_invalid_frames" / "episode-constant" / "error.txt").read_text(
                    encoding="utf-8"
                ),
                "front frame is empty or constant\n",
            )

    def test_non_finite_pose_readback_errors_are_rejected(self):
        cases = (
            (float("nan"), 0.0),
            (float("inf"), 0.0),
            (0.0, float("nan")),
            (0.0, float("-inf")),
        )
        for position_error, rotation_error in cases:
            with self.subTest(
                position_error=position_error,
                rotation_error=rotation_error,
            ):
                session = ReadbackSession(position_error, rotation_error)
                with self.assertRaisesRegex(
                    EpisodeRenderError, "non-finite vehicle pose readback error"
                ) as caught:
                    render_episode(
                        session, [request()], ("record",), ("front",),
                        RecordingSink(), frame_attempts=2,
                    )

                self.assertNotIsInstance(
                    caught.exception, renderer.SessionRenderError
                )
                self.assertEqual(len(session.pose_calls), 2)
                self.assertEqual(len(session.capture_calls), 0)

    def test_sink_append_errors_are_not_classified_as_render_errors(self):
        with self.assertRaisesRegex(OSError, "video writer failed"):
            render_episode(
                FakeSession(), [request()], ("record",), ("front",),
                FailingAppendSink(), frame_attempts=1,
            )

    def test_worker_replaces_runtime_and_retries_in_same_scene(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request_path, state_path = self._write_worker_jobs(
                root, ("episode-fail", "episode-ok")
            )
            unavailable, _ = self._session_loss()
            first_session = WaypointLossSession(unavailable)
            replacement_session = EpisodeSession(failures_remaining=0)
            first_runtime = TrackingRuntime(session=first_session)
            replacement_runtime = TrackingRuntime(session=replacement_session)
            runtime_factory = RuntimeSequenceFactory(
                (first_runtime, replacement_runtime)
            )
            WorkerSink.completed_episodes = []
            WorkerSink.aborted_episodes = []

            result = render_worker(
                self._worker_config(root, request_path, state_path),
                runtime_factory=runtime_factory,
                sink_factory=WorkerSink,
            )

            state = CollectorState(state_path)
            self.assertEqual(result, 0)
            self.assertEqual(state.status_counts(), {"complete": 2})
            state.close()
            self.assertEqual(
                WorkerSink.completed_episodes, ["episode-ok", "episode-fail"]
            )
            self.assertEqual(WorkerSink.aborted_episodes, ["episode-fail"])
            self.assertEqual(first_runtime.opened_scenes, ["5"])
            self.assertEqual(replacement_runtime.opened_scenes, ["5"])
            self.assertEqual(len(runtime_factory.calls), 2)
            self.assertEqual(first_session.close_count, 1)
            self.assertEqual(first_runtime.close_count, 1)
            self.assertEqual(replacement_session.close_count, 1)
            self.assertEqual(replacement_runtime.close_count, 1)

    def test_worker_commits_constant_frames_without_rebuilding_scene(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request_path, state_path = self._write_worker_jobs(
                root, ("episode-constant",)
            )
            first_session = ConstantFrameSession()
            first_runtime = TrackingRuntime(session=first_session)
            runtime_factory = RuntimeSequenceFactory((first_runtime,))
            WorkerSink.completed_episodes = []
            WorkerSink.aborted_episodes = []

            result = render_worker(
                self._worker_config(root, request_path, state_path),
                runtime_factory=runtime_factory,
                sink_factory=WorkerSink,
            )

            state = CollectorState(state_path)
            self.assertEqual(result, 0)
            self.assertEqual(state.status_counts(), {"complete": 1})
            state.close()
            self.assertEqual(WorkerSink.completed_episodes, ["episode-constant"])
            self.assertEqual(WorkerSink.aborted_episodes, [])
            self.assertEqual(first_runtime.opened_scenes, ["5"])
            self.assertEqual(first_session.close_count, 1)
            self.assertEqual(first_runtime.close_count, 1)

    def test_worker_reopens_same_scene_when_final_main_pass_job_crashes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request_path, state_path = self._write_worker_jobs(
                root, ("episode-final",)
            )
            unavailable, _ = self._session_loss()
            first_runtime = TrackingRuntime(
                session=WaypointLossSession(unavailable)
            )
            replacement_runtime = TrackingRuntime(
                session=EpisodeSession(failures_remaining=0)
            )
            runtime_factory = RuntimeSequenceFactory(
                (first_runtime, replacement_runtime)
            )
            WorkerSink.completed_episodes = []
            WorkerSink.aborted_episodes = []

            result = render_worker(
                self._worker_config(root, request_path, state_path),
                runtime_factory=runtime_factory,
                sink_factory=WorkerSink,
            )

            state = CollectorState(state_path)
            self.assertEqual(result, 0)
            self.assertEqual(state.status_counts(), {"complete": 1})
            state.close()
            self.assertEqual(WorkerSink.completed_episodes, ["episode-final"])
            self.assertEqual(WorkerSink.aborted_episodes, ["episode-final"])
            self.assertEqual(first_runtime.opened_scenes, ["5"])
            self.assertEqual(replacement_runtime.opened_scenes, ["5"])
            self.assertEqual(len(runtime_factory.calls), 2)

    def test_worker_treats_sink_abort_failure_as_fatal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request_path, state_path = self._write_worker_jobs(
                root, ("episode-fail",)
            )
            unavailable, _ = self._session_loss()
            runtime = TrackingRuntime(session=WaypointLossSession(unavailable))
            runtime_factory = RuntimeSequenceFactory((runtime,))
            FailingAbortWorkerSink.completed_episodes = []
            FailingAbortWorkerSink.aborted_episodes = []

            with self.assertRaisesRegex(OSError, "sink abort failed"):
                render_worker(
                    self._worker_config(root, request_path, state_path),
                    runtime_factory=runtime_factory,
                    sink_factory=FailingAbortWorkerSink,
                )

            state = CollectorState(state_path)
            self.assertEqual(state.status_counts(), {"running": 1})
            state.close()
            self.assertEqual(len(runtime_factory.calls), 1)

    def test_worker_treats_failed_state_update_as_fatal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request_path, state_path = self._write_worker_jobs(
                root, ("episode-fail",)
            )
            unavailable, _ = self._session_loss()
            runtime = TrackingRuntime(session=WaypointLossSession(unavailable))
            runtime_factory = RuntimeSequenceFactory((runtime,))
            WorkerSink.completed_episodes = []
            WorkerSink.aborted_episodes = []

            with patch.object(
                CollectorState,
                "mark_failed",
                side_effect=sqlite3.OperationalError("state update failed"),
            ):
                with self.assertRaisesRegex(
                    sqlite3.OperationalError, "state update failed"
                ):
                    render_worker(
                        self._worker_config(root, request_path, state_path),
                        runtime_factory=runtime_factory,
                        sink_factory=WorkerSink,
                    )

            state = CollectorState(state_path)
            self.assertEqual(state.status_counts(), {"running": 1})
            state.close()
            self.assertEqual(len(runtime_factory.calls), 1)

    def test_worker_treats_session_and_runtime_cleanup_failures_as_fatal(self):
        for cleanup_target in ("session", "runtime"):
            with self.subTest(cleanup_target=cleanup_target):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    request_path, state_path = self._write_worker_jobs(
                        root, ("episode-fail",)
                    )
                    unavailable, _ = self._session_loss()
                    if cleanup_target == "session":
                        session = CloseFailingSession(unavailable)
                        runtime = TrackingRuntime(session=session)
                    else:
                        session = WaypointLossSession(unavailable)
                        runtime = TrackingRuntime(
                            session=session,
                            close_error=RuntimeError("runtime cleanup failed"),
                        )
                    runtime_factory = RuntimeSequenceFactory((runtime,))
                    WorkerSink.completed_episodes = []
                    WorkerSink.aborted_episodes = []

                    with self.assertRaisesRegex(
                        RuntimeError, "{} cleanup failed".format(cleanup_target)
                    ):
                        render_worker(
                            self._worker_config(root, request_path, state_path),
                            runtime_factory=runtime_factory,
                            sink_factory=WorkerSink,
                        )

                    self.assertEqual(len(runtime_factory.calls), 1)
                    if cleanup_target == "session":
                        self.assertGreaterEqual(session.close_count, 1)
                    else:
                        self.assertGreaterEqual(runtime.close_count, 1)

    def test_worker_treats_replacement_scene_open_failure_as_fatal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request_path, state_path = self._write_worker_jobs(
                root, ("episode-fail",)
            )
            unavailable, _ = self._session_loss()
            first_runtime = TrackingRuntime(
                session=WaypointLossSession(unavailable)
            )
            replacement_runtime = TrackingRuntime(
                session=EpisodeSession(failures_remaining=0),
                open_error=RuntimeError("replacement scene open failed"),
            )
            runtime_factory = RuntimeSequenceFactory(
                (first_runtime, replacement_runtime)
            )
            WorkerSink.completed_episodes = []
            WorkerSink.aborted_episodes = []

            with self.assertRaisesRegex(
                RuntimeError, "replacement scene open failed"
            ):
                render_worker(
                    self._worker_config(root, request_path, state_path),
                    runtime_factory=runtime_factory,
                    sink_factory=WorkerSink,
                )

            state = CollectorState(state_path)
            self.assertEqual(state.status_counts(), {"running": 1})
            state.close()
            self.assertEqual(len(runtime_factory.calls), 2)
            self.assertEqual(replacement_runtime.opened_scenes, ["5"])

    def test_worker_restarts_runtime_until_scene_open_timeout_recovers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request_path, state_path = self._write_worker_jobs(
                root, ("episode",)
            )
            first_runtime = TrackingRuntime(
                open_error=rpc_error.TimeoutError("first timeout")
            )
            second_runtime = TrackingRuntime(
                open_error=rpc_error.TimeoutError("second timeout")
            )
            recovered_runtime = TrackingRuntime(
                session=EpisodeSession(failures_remaining=0)
            )
            runtime_factory = RuntimeSequenceFactory(
                (first_runtime, second_runtime, recovered_runtime)
            )
            WorkerSink.completed_episodes = []
            WorkerSink.aborted_episodes = []

            with patch("waypoint_collector.worker.time.sleep"):
                result = render_worker(
                    self._worker_config(root, request_path, state_path),
                    runtime_factory=runtime_factory,
                    sink_factory=WorkerSink,
                )

            state = CollectorState(state_path)
            self.assertEqual(result, 0)
            self.assertEqual(state.status_counts(), {"complete": 1})
            state.close()
            self.assertEqual(WorkerSink.completed_episodes, ["episode"])
            self.assertEqual(len(runtime_factory.calls), 3)
            self.assertEqual(first_runtime.close_count, 1)
            self.assertEqual(second_runtime.close_count, 1)
            self.assertEqual(recovered_runtime.close_count, 1)

    def test_worker_retries_failed_episode_after_same_scene_primary_work(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request_path = root / "requests.jsonl"
            rows = []
            for index, episode_id in enumerate(("episode-fail", "episode-ok")):
                payload = {
                    "request_id": "{}/front_image/frame_000000".format(episode_id),
                    "episode_id": episode_id,
                    "trajectory_id": "trajectory-{}".format(index),
                    "source_episode_index": index,
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
                rows.append(json.dumps(payload) + "\n")
            request_path.write_text("".join(rows), encoding="utf-8")
            state_path = root / "state.sqlite3"
            state = CollectorState(state_path)
            build_request_index(request_path, state)
            state.close()
            WorkerSink.completed_episodes = []
            WorkerSink.aborted_episodes = []
            runtime = FakeRuntime(session=EpisodeSession(failures_remaining=1))

            result = render_worker(
                WorkerConfig(
                    worker_index=0,
                    gpu=0,
                    control_port=31000,
                    repository_root=str(root),
                    env_cache_root=str(root),
                    request_path=str(request_path),
                    state_path=str(state_path),
                    episode_video_root=str(root / "videos"),
                    log_path=str(root / "worker.log"),
                    views=("front",),
                    skipped_scenes=(),
                    camera_seed=1,
                    channel_order="rgb",
                    frame_attempts=1,
                    failed_episode_retry_rounds=3,
                ),
                runtime_factory=lambda **_kwargs: runtime,
                sink_factory=WorkerSink,
            )

            state = CollectorState(state_path)
            self.assertEqual(result, 0)
            self.assertEqual(state.status_counts(), {"complete": 2})
            self.assertEqual(
                WorkerSink.completed_episodes, ["episode-ok", "episode-fail"]
            )
            self.assertEqual(WorkerSink.aborted_episodes, ["episode-fail"])
            self.assertEqual(runtime.open_count, 1)
            state.close()

    def test_pipeline_does_not_relaunch_workers_for_failed_episodes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = CollectorConfig(
                package_dir=root,
                env_archive_root=root,
                env_cache_root=root,
                gpus=(0,),
                workers=1,
                skipped_scenes=(),
                views=("front",),
                camera_seed=1,
                channel_order="rgb",
                run_id="test",
                state_root=root / "state",
                base_control_port=31000,
                frame_attempts=10,
                failed_episode_retry_rounds=3,
                estimated_output_gib=1.0,
                space_safety_factor=1.0,
                resume=False,
            )
            pipeline = CollectorPipeline(config)
            state = CollectorState(config.state_path)
            state.add_episode("episode-a", "5", 0, 0, 1, 1)
            state.close()
            calls = []

            def run_workers(_configs):
                calls.append(True)
                current = CollectorState(config.state_path)
                current.mark_failed("episode-a", "persistent")
                current.close()

            with patch("waypoint_collector.pipeline.run_render_workers", side_effect=run_workers):
                with self.assertRaisesRegex(RuntimeError, "did not complete"):
                    pipeline.render()

            self.assertEqual(len(calls), 1)

    def test_prepare_envs_populates_every_isolated_worker_scene_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "sources"
            scene = source_root / "env_airsim_18" / "LinuxNoEditor"
            for relative_path, contents in (
                ("start.sh", "#!/bin/sh\n"),
                ("city/Binaries/Linux/AirVLN-Linux-Shipping", "binary"),
                ("city/Binaries/Linux/settings.json", "{}\n"),
                ("city/Content/Paks/AirVLN-LinuxNoEditor.pak", "assets"),
            ):
                path = scene / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            worker_roots = (root / "worker-4", root / "worker-5")
            config = CollectorConfig(
                package_dir=root / "package",
                env_archive_root=source_root,
                env_cache_root=root / "cache",
                gpus=(4, 5),
                workers=2,
                skipped_scenes=(),
                views=("front",),
                camera_seed=1,
                channel_order="rgb",
                run_id="test",
                state_root=root / "state",
                base_control_port=32000,
                frame_attempts=8,
                failed_episode_retry_rounds=10,
                estimated_output_gib=1.0,
                space_safety_factor=1.0,
                resume=True,
                worker_env_cache_roots=worker_roots,
            )
            pipeline = CollectorPipeline(config)
            state = CollectorState(config.state_path)
            state.add_episode("episode-a", "env_airsim_18", 0, 0, 1, 1)
            state.close()

            pipeline.prepare_envs()

            source_settings = (
                scene / "city/Binaries/Linux/settings.json"
            )
            for worker_root in worker_roots:
                worker_settings = (
                    worker_root / "env_airsim_18/LinuxNoEditor/city/"
                    "Binaries/Linux/settings.json"
                )
                self.assertTrue(worker_settings.is_file())
                self.assertNotEqual(
                    worker_settings.stat().st_ino, source_settings.stat().st_ino
                )

    def test_resume_run_reconciles_before_skipping_completed_render_phase(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = CollectorConfig(
                package_dir=root,
                env_archive_root=root,
                env_cache_root=root,
                gpus=(0,),
                workers=1,
                skipped_scenes=(),
                views=("front",),
                camera_seed=1,
                channel_order="rgb",
                run_id="test",
                state_root=root / "state",
                base_control_port=31000,
                frame_attempts=10,
                failed_episode_retry_rounds=3,
                estimated_output_gib=1.0,
                space_safety_factor=1.0,
                resume=True,
            )
            pipeline = CollectorPipeline(config)
            state = CollectorState(config.state_path)
            state.add_episode("episode", "5", 0, 0, 10, 1)
            for phase in (
                "preflight", "prepare-envs", "pilot", "render",
                "validate", "publish",
            ):
                state.set_metadata("phase:{}".format(phase), "complete")
            state.close()
            self._touch_episode_videos(
                config.episode_video_root, "episode", ("front",)
            )
            self._write_commit_marker(
                config.episode_video_root, "episode", ("front",), frame_count=1
            )
            observed_statuses = []

            def inspect_assembly_state():
                current = CollectorState(config.state_path)
                observed_statuses.append(current.job_status("episode"))
                current.close()
                return {"assembled": True}

            with patch(
                "waypoint_collector.video.probe_video",
                return_value=self._valid_probe(),
            ), patch.object(
                pipeline, "assemble", side_effect=inspect_assembly_state
            ):
                self.assertEqual(pipeline.execute("run"), 0)

            self.assertEqual(observed_statuses, ["complete"])

if __name__ == "__main__":
    unittest.main()
