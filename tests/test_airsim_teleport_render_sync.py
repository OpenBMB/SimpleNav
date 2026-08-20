from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

import NavVLAeval.common.simulators.airsim.backend as airsim_backend_module
from NavVLAeval.common.config import EnvConfig
from NavVLAeval.common.runner.backend_plan import WorkerBackendPlan
from NavVLAeval.common.runner.worker import _truncate_action_prediction
from NavVLAeval.common.simulators.airsim.backend import AirSimEnvironmentBackend
from NavVLAeval.common.types import ActionPrediction, EvalEpisode, Pose4D


class _Vector3r:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x_val = x
        self.y_val = y
        self.z_val = z


class _Pose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation


class _KinematicsState:
    pass


def _backend(tmp_path: Path, *, profile: str, **kwargs) -> AirSimEnvironmentBackend:
    env_kwargs = {
        "env_root": tmp_path / "env",
        "render_lib_root": tmp_path / "render",
        "layout": profile,
        "settings_profile": profile,
        "sensor_profile": profile,
        "camera_name": "front_0" if profile == "aerialvln" else "front",
        "action_execution_mode": "teleport_final",
        "teleport_render_sync_frames": 5,
        **kwargs,
    }
    return AirSimEnvironmentBackend(
        cfg=EnvConfig(type="airsim", kwargs=env_kwargs),
        worker_backend=WorkerBackendPlan(
            type="airsim",
            kwargs={"airsim_port": 41451, "settings_root": tmp_path / "settings"},
        ),
        physical_gpu_id=0,
        start_process=False,
    )


def _airsim_module():
    return types.SimpleNamespace(
        Vector3r=_Vector3r,
        Pose=_Pose,
        KinematicsState=_KinematicsState,
        to_quaternion=lambda _pitch, _roll, yaw: yaw,
    )


def test_aerialvln_direct_pose_update_uses_generic_render_barrier(tmp_path: Path) -> None:
    backend = _backend(tmp_path, profile="aerialvln")
    events = []

    class Client:
        def simPause(self, paused):
            events.append(("pause", paused))

        def simSetVehiclePose(self, pose, ignore_collision, vehicle_name=None):
            events.append(("set", pose, ignore_collision, vehicle_name))

        def simContinueForFrames(self, frames):
            events.append(("continue", frames))

    backend.airsim = _airsim_module()
    backend.client = Client()

    backend.set_pose(Pose4D(1.0, 2.0, 3.0, 0.4))

    assert [event[0] for event in events] == ["pause", "set", "continue", "pause"]
    assert events[1][3] == "Drone_1"
    assert events[2] == ("continue", 5)


def test_reset_ignore_collision_is_an_explicit_airsim_policy(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path, profile="aerialvln", reset_ignore_collision=True)
    events = []

    class Client:
        def simPause(self, _paused):
            return None

        def simSetVehiclePose(self, _pose, ignore_collision, vehicle_name=None):
            events.append((ignore_collision, vehicle_name))

        def simContinueForFrames(self, _frames):
            return None

    backend.airsim = _airsim_module()
    backend.client = Client()

    backend.reset_pose(Pose4D(1.0, 2.0, 3.0, 0.4))
    backend.set_pose(Pose4D(2.0, 3.0, 4.0, 0.5))

    assert events == [(True, "Drone_1"), (False, "Drone_1")]


def test_aerialvln_waypoint_uses_configured_ignore_collision(tmp_path: Path) -> None:
    backend = _backend(
        tmp_path,
        profile="aerialvln",
        ignore_collision=True,
    )
    events = []

    class Client:
        def simPause(self, _paused):
            return None

        def simSetVehiclePose(self, _pose, ignore_collision, vehicle_name=None):
            events.append((ignore_collision, vehicle_name))

        def simContinueForFrames(self, _frames):
            return None

    backend.airsim = _airsim_module()
    backend.client = Client()

    backend.set_pose(Pose4D(2.0, 3.0, 4.0, 0.5))

    assert events == [(True, "Drone_1")]


def test_teleport_each_waypoint_records_pose_mismatch_but_attempts_full_chunk(tmp_path: Path) -> None:
    backend = _backend(
        tmp_path,
        profile="aerialvln",
        action_execution_mode="teleport_each_waypoint",
        capture_action_observations=True,
        ignore_collision=False,
    )
    pose_updates = []

    class Client:
        def simPause(self, _paused):
            return None

        def simSetVehiclePose(self, _pose, ignore_collision, vehicle_name=None):
            assert vehicle_name == "Drone_1"
            assert ignore_collision is False
            pose_updates.append(_pose)

        def simContinueForFrames(self, _frames):
            return None

        def simGetVehiclePose(self, vehicle_name=None):
            assert vehicle_name == "Drone_1"
            orientation = types.SimpleNamespace(x_val=0.0, y_val=0.0, z_val=0.0, w_val=1.0)
            return _Pose(_Vector3r(0.25, 0.0, 0.0), orientation)

    backend.airsim = _airsim_module()
    backend.client = Client()
    backend.get_observation = lambda: {"image": np.zeros((2, 2, 3), dtype=np.uint8)}
    plan = backend._build_waypoint_plan(
        current_pose=Pose4D(0.0, 0.0, 0.0, 0.0),
        raw_actions=np.asarray([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )

    result = backend._teleport_each_waypoint(plan)

    assert len(pose_updates) == 2
    assert result.collision is True
    assert result.completed_waypoint_count == 0
    assert result.attempted_waypoint_count == 2
    assert len(result.action_observations) == 2
    assert len(result.pose_mismatches) == 2
    assert [pose.as_array().tolist() for pose in result.actual_waypoint_poses] == [
        [0.25, 0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0, 0.0],
    ]
    assert result.next_pose == Pose4D(0.25, 0.0, 0.0, 0.0)


def test_teleport_final_records_full_chunk_as_attempted(tmp_path: Path) -> None:
    backend = _backend(tmp_path, profile="traveluav")

    class Client:
        def __init__(self) -> None:
            self.position = _Vector3r()

        def simPause(self, _paused):
            return None

        def simSetKinematics(self, state, ignore_collision):
            assert ignore_collision is False
            self.position = state.position

        def simContinueForFrames(self, _frames):
            return None

        def simGetVehiclePose(self):
            orientation = types.SimpleNamespace(x_val=0.0, y_val=0.0, z_val=0.0, w_val=1.0)
            return _Pose(self.position, orientation)

    backend.airsim = _airsim_module()
    backend.client = Client()
    plan = backend._build_waypoint_plan(
        current_pose=Pose4D(0.0, 0.0, 0.0, 0.0),
        raw_actions=np.asarray([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )

    result = backend._teleport_to_final_waypoint(plan)

    assert result.collision is False
    assert result.completed_waypoint_count == plan.executed_waypoint_count
    assert result.attempted_waypoint_count == plan.executed_waypoint_count


def test_aerialvln_connects_as_named_multirotor_and_enables_control(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = _backend(tmp_path, profile="aerialvln")
    events = []

    class MultirotorClient:
        def __init__(self, *, port, timeout_value):
            events.append(("multirotor_client", port, timeout_value))

        def confirmConnection(self):
            events.append(("confirm",))

        def enableApiControl(self, enabled, vehicle_name=""):
            events.append(("api_control", enabled, vehicle_name))

        def armDisarm(self, armed, vehicle_name=""):
            events.append(("arm", armed, vehicle_name))

    class VehicleClient:
        def __init__(self, **_kwargs):
            raise AssertionError("AerialVLN must use MultirotorClient")

    fake_airsim = types.SimpleNamespace(
        MultirotorClient=MultirotorClient,
        VehicleClient=VehicleClient,
    )
    monkeypatch.setitem(sys.modules, "airsim", fake_airsim)
    monkeypatch.setattr(airsim_backend_module, "patch_msgpackrpc_transport", lambda: None)

    backend.connect()

    assert events == [
        ("multirotor_client", 41451, 40),
        ("confirm",),
        ("api_control", True, "Drone_1"),
        ("arm", True, "Drone_1"),
    ]
    assert backend.client is not None


def test_airsim_waits_for_configured_world_settle_before_first_pose_reset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = _backend(tmp_path, profile="aerialvln", episode_startup_settle_sec=3)
    events = []

    def start(env_name):
        events.append(("start", env_name))
        backend.current_env_name = env_name
        backend.client = object()

    backend.start = start
    backend.reset_pose = lambda pose: events.append(("reset", pose))
    monkeypatch.setattr(airsim_backend_module.time, "sleep", lambda seconds: events.append(("sleep", seconds)))
    episode = EvalEpisode(
        episode_uid="aerialvln:test",
        source_episode_id="test",
        scene_id="10",
        instruction="fly forward",
        source="aerialvln_json",
        input_namespace="aerialvln",
        input_root="/tmp/aerialvln.json",
        payload={"env_name": "env_10"},
    )
    initial_pose = Pose4D(1.0, 2.0, -3.0, 0.4)

    backend.start_episode(episode, initial_pose)
    backend.start_episode(episode, initial_pose)

    assert events == [
        ("start", "env_10"),
        ("sleep", 3.0),
        ("reset", initial_pose),
        ("start", "env_10"),
        ("reset", initial_pose),
    ]


@pytest.mark.parametrize(
    ("legacy_key", "canonical_key"),
    [
        ("camera_profile", "settings_profile/sensor_profile"),
        ("openfly_render_sync_frames", "teleport_render_sync_frames"),
        ("openfly_render_warmup_sec", "render_warmup_sec"),
    ],
)
def test_legacy_airsim_config_aliases_are_rejected(
    tmp_path: Path,
    legacy_key: str,
    canonical_key: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{legacy_key}.*{canonical_key}"):
        _backend(tmp_path, profile="openfly", **{legacy_key: 2})


def test_negative_generic_render_sync_frames_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="teleport_render_sync_frames must be non-negative"):
        _backend(tmp_path, profile="aerialvln", teleport_render_sync_frames=-1)


def test_worker_owns_execute_waypoints_per_step_before_airsim_backend(tmp_path: Path) -> None:
    raw_actions = np.arange(32, dtype=np.float32).reshape(8, 4)
    prediction = ActionPrediction(
        normalized_actions=raw_actions / 32.0,
        raw_actions=raw_actions,
    )

    truncated = _truncate_action_prediction(
        prediction,
        execute_waypoints_per_step=3,
    )
    backend = _backend(
        tmp_path,
        profile="openfly",
        execute_waypoints_per_step=3,
    )

    assert truncated.raw_actions.shape == (3, 4)
    assert truncated.metadata["original_action_horizon"] == 8
    assert backend.action_execution_config.execute_waypoints_per_step is None


def test_traveluav_uses_generic_render_sync_frame_count(tmp_path: Path) -> None:
    backend = _backend(tmp_path, profile="traveluav", teleport_render_sync_frames=2)
    events = []

    class Client:
        def simPause(self, paused):
            events.append(("pause", paused))

        def simSetKinematics(self, state, ignore_collision):
            events.append(("set_kinematics", state, ignore_collision))

        def simContinueForFrames(self, frames):
            events.append(("continue", frames))

    backend.airsim = _airsim_module()
    backend.client = Client()

    backend.set_pose(Pose4D(1.0, 2.0, 3.0, 0.4))

    assert [event for event in events if event[0] == "continue"] == [
        ("continue", 1),
        ("continue", 1),
    ]


@pytest.mark.parametrize(
    "config_path",
    [
        "NavVLAeval/openfly/config_portable.yaml",
        "NavVLAeval/aerialvln/config_portable.yaml",
        "NavVLAeval/traveluav/config_portable.yaml",
    ],
)
def test_airsim_benchmark_configs_use_one_canonical_runtime_schema(config_path: str) -> None:
    cfg = OmegaConf.load(Path(__file__).parents[1] / config_path)
    kwargs = cfg.env.kwargs
    simulator_internal_keys = {
        "recording_folder",
        "recording_camera_name",
        "recording_interval",
        "camera_resolution_overrides",
        "external_camera_resolution_overrides",
        "clock_speed",
        "view_mode",
        "nvidia_egl_root",
        "nvidia_egl_lib_dir",
        "nvidia_egl_vendor_json",
    }

    assert simulator_internal_keys.isdisjoint(kwargs)
    assert kwargs.teleport_render_sync_frames == 3
    assert kwargs.episode_startup_settle_sec >= 0
    assert kwargs.render_warmup_sec >= 0
    assert isinstance(kwargs.capture_action_observations, bool)
    assert isinstance(kwargs.reset_ignore_collision, bool)
    assert cfg.model.inference_seed == 42
    assert kwargs.get("ignore_collision", False) in {True, False}
    assert cfg.dataset.bats_sampling_mode == "priority_capped"
    assert cfg.dataset.bats_seed == 42
    assert kwargs.execute_waypoints_per_step == 8
    assert kwargs.airsim_z_sign in {-1, 1}
    assert kwargs.layout
    assert kwargs.settings_profile
    assert kwargs.sensor_profile
    assert kwargs.camera_name
    assert "camera_profile" not in kwargs
    assert "openfly_render_sync_frames" not in kwargs
    assert "openfly_render_warmup_sec" not in kwargs


@pytest.mark.parametrize(
    "config_path",
    [
        "NavVLAeval/traveluav/config_portable.yaml",
        "NavVLAeval/traveluav/config_template.yaml",
    ],
)
def test_traveluav_eval_configs_match_training_source_frame_stride(config_path: str) -> None:
    cfg = OmegaConf.load(Path(__file__).parents[1] / config_path)

    assert cfg.dataset.history_candidate_source_stride == 5


@pytest.mark.parametrize("benchmark", ["openfly", "aerialvln", "traveluav"])
def test_airsim_run_scripts_use_repo_relative_default_config_and_overrides(benchmark: str) -> None:
    script = (
        Path(__file__).parents[1] / "NavVLAeval" / benchmark / "run_eval.sh"
    ).read_text(encoding="utf-8")

    assert "config=${config:-" not in script
    assert 'repo_root="$(cd -- "$script_dir/../.." && pwd)"' in script
    if benchmark == "aerialvln":
        assert 'config_path="$script_dir/config_portable.yaml"' in script
        assert '--config "$config_path"' in script
    else:
        assert '--config "$script_dir/config_portable.yaml"' in script
    assert '"$@"' in script
