from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from NavVLAeval.common.config import EnvConfig
from NavVLAeval.common.runner.backend_plan import WorkerBackendPlan
from NavVLAeval.common.simulators.unrealzoo.backend import UnrealZooEnvironmentBackend
from NavVLAeval.common.simulators.unrealzoo.coordinates import (
    nav_pose_from_unreal_cm,
    starvla_waypoints_to_nav,
    starvla_waypoints_to_unreal_cm,
)
from NavVLAeval.common.types import Pose4D


def test_uavflow_body_axes_project_to_x_forward_y_right_z_down_and_yaw_right_positive() -> None:
    current = Pose4D(10.0, 20.0, 3.0, 0.0)
    actions = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, math.pi / 2.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        starvla_waypoints_to_nav(current, actions),
        [
            [11.0, 20.0, 3.0, 0.0],
            [10.0, 21.0, 3.0, 0.0],
            [10.0, 20.0, 4.0, 0.0],
            [10.0, 20.0, 3.0, 90.0],
        ],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        starvla_waypoints_to_unreal_cm(current, actions),
        [
            [1100.0, 2000.0, -300.0, 0.0],
            [1000.0, 2100.0, -300.0, 0.0],
            [1000.0, 2000.0, -400.0, 0.0],
            [1000.0, 2000.0, -300.0, 90.0],
        ],
        atol=1e-5,
    )


def test_uavflow_body_axes_rotate_with_current_right_positive_yaw() -> None:
    current = Pose4D(10.0, 20.0, 3.0, 90.0)
    actions = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    np.testing.assert_allclose(
        starvla_waypoints_to_nav(current, actions),
        [[10.0, 21.0, 3.0, 90.0], [9.0, 20.0, 3.0, 90.0]],
        atol=1e-6,
    )


def test_uavflow_unreal_pose_conversion_preserves_yaw_sign_and_converts_z_up_to_z_down() -> None:
    pose = nav_pose_from_unreal_cm([100.0, 200.0, -300.0, 0.0, 45.0, 0.0])

    np.testing.assert_allclose(pose.as_array(), [1.0, 2.0, 3.0, 45.0], atol=1e-6)


class _FakeUnrealCV:
    def __init__(self) -> None:
        self.locations: list[tuple[str, list[float]]] = []
        self.rotations: list[tuple[str, float]] = []

    def set_obj_location(self, player: str, location: list[float]) -> None:
        self.locations.append((player, list(location)))

    def set_rotation(self, player: str, yaw: float) -> None:
        self.rotations.append((player, float(yaw)))

    def set_cam(self, player: str) -> None:
        del player

    def get_image(self, camera_id: int, viewmode: str) -> np.ndarray:
        del camera_id, viewmode
        return np.zeros((4, 4, 3), dtype=np.uint8)


def test_uavflow_unrealzoo_backend_sends_aligned_location_and_rotation_commands() -> None:
    cfg = EnvConfig(
        type="unrealzoo",
        kwargs={
            "env_id": "test-env",
            "unreal_env_root": "/tmp/test-unrealzoo",
            "unrealzoo_gym_root": "/tmp/unrealzoo-gym",
        },
    )
    worker_backend = WorkerBackendPlan(
        type="unrealzoo",
        kwargs={"env_id": "test-env", "unreal_env_root": "/tmp/test-unrealzoo"},
    )
    backend = UnrealZooEnvironmentBackend(
        cfg=cfg,
        worker_backend=worker_backend,
        physical_gpu_id=0,
        start_process=False,
    )
    unrealcv = _FakeUnrealCV()
    backend.env = SimpleNamespace(unwrapped=SimpleNamespace(unrealcv=unrealcv, player_list=["player-0"]))

    result = backend.apply_action(
        Pose4D(10.0, 20.0, 3.0, 0.0),
        np.asarray([[0.0, 1.0, 1.0, math.pi / 2.0]], dtype=np.float32),
    )

    assert unrealcv.locations == [("player-0", [1000.0, 2100.0, -400.0])]
    assert unrealcv.rotations == [("player-0", -90.0)]
    np.testing.assert_allclose(result.next_pose.as_array(), [10.0, 21.0, 4.0, 90.0], atol=1e-6)
    np.testing.assert_allclose(result.diagnostics["unreal_waypoints_cm"], [[1000.0, 2100.0, -400.0, 90.0]])
