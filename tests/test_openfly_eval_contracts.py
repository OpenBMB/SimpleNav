from __future__ import annotations

import json
import math
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

import NavVLAeval.common.simulators.airsim.backend as airsim_backend_module
from NavVLAeval.common.config import EnvConfig, InputConfig
from NavVLAeval.common.data.runtime_dataset import OPENFLY_PLATFORM_TEXT, OpenFlyRuntimeDatasetAdapter
from NavVLAeval.common.runner.backend_plan import WorkerBackendPlan
from NavVLAeval.common.simulators.airsim.backend import AirSimEnvironmentBackend
from NavVLAeval.common.simulators.airsim.settings import build_airsim_settings
from NavVLAeval.common.types import ActionPrediction, EpisodeHistory, EvalEpisode, Pose4D
from NavVLAeval.openfly.benchmark import OpenFlyBenchmarkSpec
from NavVLAeval.openfly.inputs import OpenFlyStarVLAInputAdapter


TRAINING_PLATFORM_TEXT = (
    "The platform is UAV for urban uav navigation. The control frequency is 1 Hz. "
    "Please predict the next 8 local 3D waypoints (dx, dy, dz, dyaw) to execute the following task:"
)


def test_openfly_platform_text_matches_training_metadata() -> None:
    example = OpenFlyRuntimeDatasetAdapter().build_example(
        observation={"image": np.zeros((2, 2, 3), dtype=np.uint8)},
        history=EpisodeHistory(),
        instruction="go forward",
    )

    assert OPENFLY_PLATFORM_TEXT == TRAINING_PLATFORM_TEXT
    assert example["platform_text"] == TRAINING_PLATFORM_TEXT
    assert " ".join([example["platform_text"], example["lang"]]) == f"{TRAINING_PLATFORM_TEXT} go forward"


def test_openfly_airsim_cameras_are_448_square() -> None:
    cameras = build_airsim_settings(api_server_port=41451, profile="openfly")["Vehicles"]["Drone_1"]["Cameras"]

    for camera_name in ("front_custom", "0"):
        capture_settings = cameras[camera_name]["CaptureSettings"]
        assert [(item["ImageType"], item["Width"], item["Height"]) for item in capture_settings] == [
            (0, 448, 448),
            (2, 448, 448),
        ]


def test_openfly_z_up_contract_is_reflected_only_at_airsim_boundary(tmp_path: Path) -> None:
    backend = AirSimEnvironmentBackend(
        cfg=EnvConfig(
            type="airsim",
            kwargs={
                "env_root": tmp_path / "env",
                "render_lib_root": tmp_path / "render",
                "settings_profile": "openfly",
                "airsim_z_sign": -1.0,
            },
        ),
        worker_backend=WorkerBackendPlan(
            type="airsim",
            kwargs={"airsim_port": 41451, "settings_root": tmp_path / "settings"},
        ),
        physical_gpu_id=0,
        start_process=False,
    )
    logical_pose = Pose4D(x=10.0, y=20.0, z=30.0, yaw=0.5)
    waypoints = backend.project_action_to_world(
        logical_pose,
        np.asarray([[1.0, 2.0, 3.0, 0.25]], dtype=np.float32),
    )

    class FakeVector3r:
        def __init__(self, x: float, y: float, z: float):
            self.x_val = x
            self.y_val = y
            self.z_val = z

    class FakePose:
        def __init__(self, position, orientation):
            self.position = position
            self.orientation = orientation

    class FakeClient:
        def simSetVehiclePose(self, target_pose, ignore_collision):
            self.target_pose = target_pose
            self.ignore_collision = ignore_collision

        def simContinueForFrames(self, frames):
            self.render_sync_frames = frames

        def simPause(self, paused):
            self.paused = paused

    backend.airsim = types.SimpleNamespace(
        Vector3r=FakeVector3r,
        Pose=FakePose,
        to_quaternion=lambda _pitch, _roll, yaw: yaw,
    )
    backend.client = FakeClient()
    backend.set_pose(logical_pose)

    np.testing.assert_allclose(waypoints[0, 2], 33.0, atol=1e-6)
    np.testing.assert_allclose(waypoints[0, 3], 0.75, atol=1e-6)
    assert math.isfinite(float(waypoints[0, 0]))
    assert math.isfinite(float(waypoints[0, 1]))
    assert backend.client.target_pose.position.z_val == -30.0
    assert backend.client.render_sync_frames == 3
    assert backend.client.paused is True
    assert backend._pose_from_airsim_coordinates(Pose4D(10.0, 20.0, -30.0, 0.5)) == logical_pose


def test_openfly_new_environment_pauses_and_polls_until_renderer_returns_nonblack_frame(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = AirSimEnvironmentBackend(
        cfg=EnvConfig(
            type="airsim",
            kwargs={
                "env_root": tmp_path / "env",
                "render_lib_root": tmp_path / "render",
                "settings_profile": "openfly",
                "render_warmup_sec": 25.0,
            },
        ),
        worker_backend=WorkerBackendPlan(
            type="airsim",
            kwargs={"airsim_port": 41451, "settings_root": tmp_path / "settings"},
        ),
        physical_gpu_id=0,
        start_process=False,
    )
    events = []

    class FakeClient:
        def simPause(self, paused):
            events.append(("pause", paused))

    backend.client = FakeClient()
    backend.current_env_name = None
    monkeypatch.setattr(backend, "start", lambda env_name: events.append(("start", env_name)))
    monkeypatch.setattr(backend, "reset_pose", lambda pose: events.append(("reset", pose)))
    frames = iter(
        [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
        ]
    )
    monkeypatch.setattr(
        backend,
        "get_observation",
        lambda: (events.append(("capture", None)), {"image": next(frames)})[1],
    )
    monkeypatch.setattr(airsim_backend_module.time, "sleep", lambda seconds: events.append(("sleep", seconds)))
    episode = SimpleNamespace(episode_uid="seen:000001", payload={"env_name": "env_airsim_23"})
    pose = Pose4D(x=1.0, y=2.0, z=3.0, yaw=0.4)

    backend.start_episode(episode, pose)

    assert events == [
        ("start", "env_airsim_23"),
        ("pause", True),
        ("reset", pose),
        ("capture", None),
        ("sleep", 1.0),
        ("capture", None),
    ]


def test_openfly_scene_filter_selects_only_requested_environments(tmp_path: Path) -> None:
    data_root = tmp_path / "openfly"
    annotation_dir = data_root / "Annotation"
    annotation_dir.mkdir(parents=True)
    records = []
    for index, scene_id in enumerate(("env_airsim_18", "env_airsim_23", "env_airsim_16")):
        records.append(
            {
                "episode_id": f"ep_{index}",
                "image_path": f"{scene_id}/trajectory_{index}",
                "gpt_instruction": "go",
                "pos": [[0, 0, 0], [1, 0, 0]],
                "yaw": [0, 0],
            }
        )
    (annotation_dir / "seen.json").write_text(json.dumps(records), encoding="utf-8")
    cfg = InputConfig(
        type="starvla_episode_json",
        adapter_class_path="NavVLAeval.openfly.inputs:OpenFlyStarVLAInputAdapter",
        namespace="seen",
        data_root=data_root,
        split="seen",
        raw={"scene_ids": ["env_airsim_23", "env_airsim_16"]},
    )

    episodes = OpenFlyStarVLAInputAdapter().load_episodes(cfg, max_samples=None)

    assert [episode.scene_id for episode in episodes] == ["env_airsim_23", "env_airsim_16"]


def test_openfly_cached_history_does_not_store_image_fingerprints() -> None:
    adapter = OpenFlyRuntimeDatasetAdapter()
    history = EpisodeHistory()
    action = np.zeros((1, 4), dtype=np.float32)
    observation = {
        "image": np.ones((4, 4, 3), dtype=np.uint8),
        "navvla_eval": {"frame_index": 0, "timestamp": 0.0},
        "navvla_online_visual_tokens": {"front": np.ones((4, 16), dtype=np.float16)},
    }

    adapter.update_history(
        history=history,
        observation=observation,
        prediction=ActionPrediction(normalized_actions=action, raw_actions=action),
        instruction="go",
    )

    assert "navvla_online_dhash" not in history.observations[0]


def test_openfly_portable_eval_config_matches_runtime_contract() -> None:
    eval_cfg = OmegaConf.load("NavVLAeval/openfly/config_portable.yaml")

    assert eval_cfg.benchmark.max_steps == 80
    assert eval_cfg.dataset.bats_token_budget == 1024
    assert "dhash_threshold" not in eval_cfg.dataset
    assert eval_cfg.env.kwargs.render_warmup_sec == 25
    assert eval_cfg.env.kwargs.action_execution_mode == "teleport_each_waypoint"
    assert eval_cfg.dataset.action_horizon == 8
    assert eval_cfg.dataset.action_dim == 4
    assert eval_cfg.dataset.state_dim == 0
    assert eval_cfg.benchmark.kwargs.stop_action_threshold == 0.31
    assert list(eval_cfg.parallel.gpu_ids) == [0]


def test_openfly_termination_ignores_small_actions_until_success_radius() -> None:
    runtime = OpenFlyBenchmarkSpec(success_radius=20.0, stop_action_threshold=-1).create_runtime(None)
    episode = EvalEpisode(
        episode_uid="seen:test",
        source_episode_id="test",
        scene_id="env_airsim_23",
        instruction="go",
        source="fixture",
        input_namespace="seen",
        input_root="fixture",
        payload={
            "env_name": "env_airsim_23",
            "pos": [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            "yaw": [0.0, 0.0],
        },
    )
    state = SimpleNamespace(
        episode=episode,
        pose_after=Pose4D(70.0, 0.0, 0.0, 0.0),
        raw_action_chunk=np.zeros((8, 4), dtype=np.float32),
    )

    termination = runtime.update_termination(state)

    assert termination.done is False
    assert termination.success == 0
    assert termination.oracle_success == 0
    assert termination.reason == "running"
    assert termination.diagnostics["action_stop"] is False


def test_openfly_termination_stops_immediately_inside_success_radius() -> None:
    runtime = OpenFlyBenchmarkSpec(success_radius=20.0, stop_action_threshold=-1).create_runtime(None)
    episode = EvalEpisode(
        episode_uid="seen:test",
        source_episode_id="test",
        scene_id="env_airsim_23",
        instruction="go",
        source="fixture",
        input_namespace="seen",
        input_root="fixture",
        payload={
            "env_name": "env_airsim_23",
            "pos": [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            "yaw": [0.0, 0.0],
        },
    )
    state = SimpleNamespace(
        episode=episode,
        pose_after=Pose4D(85.0, 0.0, 0.0, 0.0),
        raw_action_chunk=np.ones((8, 4), dtype=np.float32),
    )

    termination = runtime.update_termination(state)

    assert termination.done is True
    assert termination.success == 1
    assert termination.oracle_success == 1
    assert termination.reason == "success"


def test_openfly_action_or_max_steps_does_not_stop_inside_success_radius() -> None:
    runtime = OpenFlyBenchmarkSpec(
        success_radius=20.0,
        stop_action_threshold=0.1,
        termination_mode="action_or_max_steps",
    ).create_runtime(None)
    episode = EvalEpisode(
        episode_uid="seen:test",
        source_episode_id="test",
        scene_id="env_airsim_23",
        instruction="go",
        source="fixture",
        input_namespace="seen",
        input_root="fixture",
        payload={
            "env_name": "env_airsim_23",
            "pos": [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            "yaw": [0.0, 0.0],
        },
    )
    state = SimpleNamespace(
        episode=episode,
        pose_after=Pose4D(85.0, 0.0, 0.0, 0.0),
        raw_action_chunk=np.ones((8, 4), dtype=np.float32),
    )

    termination = runtime.update_termination(state)

    assert runtime.stop_at_first_success_waypoint() is False
    assert termination.done is False
    assert termination.success == 1
    assert termination.oracle_success == 1
    assert termination.reason == "running"


def test_openfly_action_stop_measures() -> None:
    episode = EvalEpisode(
        episode_uid="seen:test",
        source_episode_id="test",
        scene_id="env_airsim_23",
        instruction="go",
        source="fixture",
        input_namespace="seen",
        input_root="fixture",
        payload={
            "env_name": "env_airsim_23",
            "pos": [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            "yaw": [0.0, 0.0],
        },
    )
    action = np.asarray(
        [
            [0.1, 0.0, 0.0, 2.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.125, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0],
            [0.375, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    state = SimpleNamespace(
        episode=episode,
        pose_after=Pose4D(0.0, 0.0, 0.0, 0.0),
        raw_action_chunk=action,
    )
    expected_values = {
        "chunk_max_abs": 2.0,
        "chunk_max_xyz_norm": 0.5,
        "final_xyz_norm": 0.5,
        "tail4_max_segment_xyz_norm": 0.125,
    }

    for measure, expected_value in expected_values.items():
        runtime = OpenFlyBenchmarkSpec(
            stop_action_threshold=0.6,
            termination_mode="action_or_max_steps",
            stop_action_measure=measure,
        ).create_runtime(None)
        termination = runtime.update_termination(state)
        assert termination.diagnostics["stop_action_value"] == expected_value
        assert termination.done is (measure != "chunk_max_abs")
        assert termination.reason == ("stop" if measure != "chunk_max_abs" else "running")

    diagnostics = runtime.log_step_artifacts(state, None)
    assert diagnostics["stop_action_values"] == expected_values


def test_openfly_rejects_unknown_termination_configuration() -> None:
    with np.testing.assert_raises_regex(ValueError, "termination_mode"):
        OpenFlyBenchmarkSpec(termination_mode="unknown")
    with np.testing.assert_raises_regex(ValueError, "stop_action_measure"):
        OpenFlyBenchmarkSpec(stop_action_measure="unknown")
    with np.testing.assert_raises_regex(ValueError, "stop_action_confirmations"):
        OpenFlyBenchmarkSpec(stop_action_confirmations=0)


def test_openfly_action_stop_requires_configured_confirmations() -> None:
    runtime = OpenFlyBenchmarkSpec(
        stop_action_threshold=0.1,
        termination_mode="action_or_max_steps",
        stop_action_measure="final_xyz_norm",
        stop_action_confirmations=3,
    ).create_runtime(None)
    episode = EvalEpisode(
        episode_uid="seen:test",
        source_episode_id="test",
        scene_id="env_airsim_23",
        instruction="go",
        source="fixture",
        input_namespace="seen",
        input_root="fixture",
        payload={
            "env_name": "env_airsim_23",
            "pos": [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            "yaw": [0.0, 0.0],
        },
    )
    state = SimpleNamespace(
        episode=episode,
        pose_after=Pose4D(0.0, 0.0, 0.0, 0.0),
        raw_action_chunk=np.zeros((8, 4), dtype=np.float32),
    )

    first = runtime.update_termination(state)
    second = runtime.update_termination(state)
    third = runtime.update_termination(state)

    assert first.done is False
    assert second.done is False
    assert third.done is True
    assert third.diagnostics["stop_action_streak"] == 3
    runtime.prepare_environment(episode, None, Pose4D(0.0, 0.0, 0.0, 0.0))
    assert runtime.update_termination(state).diagnostics["stop_action_streak"] == 1
