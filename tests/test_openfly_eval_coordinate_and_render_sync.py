from __future__ import annotations

import json
import math
import types
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from NavVLAeval.common.config import EnvConfig, InputConfig
from NavVLAeval.common.runner.backend_plan import WorkerBackendPlan
from NavVLAeval.common.simulators.airsim.backend import AirSimEnvironmentBackend
from NavVLAeval.common.types import Pose4D
from NavVLAeval.openfly.inputs import OpenFlyStarVLAInputAdapter


def _backend(tmp_path: Path, **kwargs) -> AirSimEnvironmentBackend:
    env_kwargs = {
        "env_root": tmp_path / "env",
        "render_lib_root": tmp_path / "render",
        "settings_profile": "openfly",
        "sensor_profile": "openfly",
        "camera_name": "front_custom",
        "action_execution_mode": "teleport_each_waypoint",
        "action_waypoint_semantics": "anchor_relative_body_frame_xyz_yaw",
        "airsim_z_sign": -1.0,
        "teleport_render_sync_frames": 3,
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


class _FakeVector3r:
    def __init__(self, x: float, y: float, z: float):
        self.x_val = x
        self.y_val = y
        self.z_val = z


class _FakePose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation


def _install_fake_airsim(backend: AirSimEnvironmentBackend, events: list[tuple]) -> None:
    class FakeClient:
        def __init__(self):
            self.pose = _FakePose(
                _FakeVector3r(0.0, 0.0, 0.0),
                types.SimpleNamespace(x_val=0.0, y_val=0.0, z_val=0.0, w_val=1.0),
            )

        def simSetVehiclePose(self, pose, ignore_collision):
            self.pose = pose
            events.append(("set", pose, ignore_collision))

        def simGetVehiclePose(self):
            return self.pose

        def simContinueForFrames(self, frames):
            events.append(("continue", frames))

        def simPause(self, paused):
            events.append(("pause", paused))

    backend.airsim = types.SimpleNamespace(
        Vector3r=_FakeVector3r,
        Pose=_FakePose,
        to_quaternion=lambda _pitch, _roll, yaw: yaw,
    )
    backend.client = FakeClient()


def test_openfly_annotation_input_is_reflected_into_training_canonical_coordinates(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "Annotation"
    annotation_dir.mkdir(parents=True)
    (annotation_dir / "seen.json").write_text(
        json.dumps(
            [
                {
                    "episode_id": "turn",
                    "image_path": "env_airsim_23/trajectory",
                    "gpt_instruction": "turn left",
                    "pos": [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
                    "yaw": [0.5, -0.25],
                }
            ]
        ),
        encoding="utf-8",
    )
    cfg = InputConfig(
        type="starvla_episode_json",
        adapter_class_path="NavVLAeval.openfly.inputs:OpenFlyStarVLAInputAdapter",
        namespace="seen",
        data_root=tmp_path,
        split="seen",
    )

    episode = OpenFlyStarVLAInputAdapter().load_episodes(cfg, max_samples=None)[0]

    assert episode.payload["pos"] == [[10.0, -20.0, 30.0], [11.0, -21.0, 31.0]]
    np.testing.assert_allclose(episode.payload["yaw"], [-0.5, 0.25], atol=1e-7)


def test_openfly_annotation_input_reflects_z_only_when_configured(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "Annotation"
    annotation_dir.mkdir(parents=True)
    (annotation_dir / "seen.json").write_text(
        json.dumps(
            [
                {
                    "episode_id": "turn",
                    "image_path": "env_airsim_23/trajectory",
                    "gpt_instruction": "turn left",
                    "pos": [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
                    "yaw": [0.5, -0.25],
                }
            ]
        ),
        encoding="utf-8",
    )
    cfg = InputConfig(
        type="starvla_episode_json",
        adapter_class_path="NavVLAeval.openfly.inputs:OpenFlyStarVLAInputAdapter",
        namespace="seen",
        data_root=tmp_path,
        split="seen",
        raw={"source_z_sign": -1},
    )

    episode = OpenFlyStarVLAInputAdapter().load_episodes(cfg, max_samples=None)[0]

    assert episode.payload["pos"] == [[10.0, -20.0, -30.0], [11.0, -21.0, -31.0]]
    np.testing.assert_allclose(episode.payload["yaw"], [-0.5, 0.25], atol=1e-7)


def test_openfly_episode_json_input_uses_the_same_coordinate_reflection(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode.json"
    episode_path.write_text(
        json.dumps(
            {
                "frames": [
                    {"state": [10.0, 20.0, 30.0, 0.5]},
                    {"state": [11.0, 21.0, 31.0, -0.25]},
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = OpenFlyStarVLAInputAdapter()._load_payload(episode_path)

    assert payload["pos"] == [[10.0, -20.0, 30.0], [11.0, -21.0, 31.0]]
    np.testing.assert_allclose(payload["yaw"], [-0.5, 0.25], atol=1e-7)


def test_openfly_set_pose_advances_renderer_before_returning(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    events: list[tuple] = []
    _install_fake_airsim(backend, events)

    backend.set_pose(Pose4D(10.0, -20.0, 30.0, -0.5))

    assert [event[0] for event in events] == ["pause", "set", "continue", "pause"]
    assert events[2] == ("continue", 3)
    target_pose = events[1][1]
    assert target_pose.position.x_val == 10.0
    assert target_pose.position.y_val == -20.0
    assert target_pose.position.z_val == -30.0
    assert math.isclose(target_pose.orientation, -0.5)


def test_openfly_post_step_executes_all_waypoints_but_captures_only_final_observation(tmp_path: Path) -> None:
    backend = _backend(tmp_path, capture_action_observations=False)
    events: list[tuple] = []
    _install_fake_airsim(backend, events)
    observation_calls = []
    backend.get_observation = lambda: (
        observation_calls.append(len(observation_calls)),
        {"image": np.full((2, 2, 3), len(observation_calls), dtype=np.uint8)},
    )[1]
    raw_actions = np.asarray(
        [[float(index), 0.0, 0.0, 0.0] for index in range(1, 9)],
        dtype=np.float32,
    )

    result = backend.apply_action(Pose4D(0.0, 0.0, 0.0, 0.0), raw_actions)

    assert len([event for event in events if event[0] == "set"]) == 8
    assert [event for event in events if event[0] == "continue"] == [("continue", 3)] * 8
    assert len(observation_calls) == 1
    assert result.action_observations == []
    assert result.observation["image"][0, 0, 0] == 1
    assert result.next_pose == Pose4D(8.0, 0.0, 0.0, 0.0)


def test_openfly_portable_config_uses_action_observation_history() -> None:
    config_path = Path(__file__).parents[1] / "NavVLAeval" / "openfly" / "config_portable.yaml"

    cfg = OmegaConf.load(config_path)

    assert cfg.dataset.history_update_mode == "action_observations"
    assert cfg.env.kwargs.capture_action_observations is True
    assert cfg.env.kwargs.teleport_render_sync_frames == 3
    assert "openfly_render_sync_frames" not in cfg.env.kwargs
