from __future__ import annotations

import sys

import numpy as np
import pytest


_HABITAT_IMPORT_PATHS = [
    "NavVLAeval/vlnce/3dparty/habitat-lab",
    "NavVLAeval/vlnce/3dparty/build_py310_habitat_sim_031/lib/python3.10/site-packages",
]
sys.path[:0] = _HABITAT_IMPORT_PATHS
pytest.importorskip("habitat_sim", reason="Habitat tests require the optional VLN-CE simulator environment")

# Match VLNCE031HabitatRuntime._load(): Habitat-Lab 0.3.1 references
# gym.spaces.Space, while the pinned Gym 0.10.9 exposes it as gym.Space.
from NavVLAeval.common.simulators.habitat.vlnce031_runtime import apply_gym_spaces_compat

apply_gym_spaces_compat()


def test_vlnce_lateral_left_maps_to_habitat_local_negative_x() -> None:
    sys.path[:0] = [
        "NavVLAeval/vlnce/3dparty/habitat-lab",
        "NavVLAeval/vlnce/3dparty/build_py310_habitat_sim_031/lib/python3.10/site-packages",
    ]
    from NavVLAeval.common.simulators.habitat.vlnce031_actions import _body_delta_to_habitat_local

    assert np.allclose(_body_delta_to_habitat_local(0.25, 0.5), np.asarray([-0.25, 0.0, -0.5]))


def test_collision_stop_pose_delta_stops_at_first_filtered_substep() -> None:
    sys.path[:0] = [
        "NavVLAeval/vlnce/3dparty/habitat-lab",
        "NavVLAeval/vlnce/3dparty/build_py310_habitat_sim_031/lib/python3.10/site-packages",
    ]
    from NavVLAeval.common.simulators.habitat.vlnce031_actions import _collision_stop_pose_delta_position

    class _Sim:
        @staticmethod
        def step_filter(start, target):
            del start
            # An obstacle starts at x=0.12, and the filter returns its boundary.
            return np.minimum(target, np.asarray([0.12, 0.0, 0.0]))

        @staticmethod
        def is_navigable(position):
            return bool(np.all(np.isfinite(position)))

    reached = _collision_stop_pose_delta_position(
        _Sim(), np.zeros(3), np.asarray([1.0, 0.0, 0.0]), max_substep_m=0.05
    )

    assert np.allclose(reached, np.asarray([0.12, 0.0, 0.0]))


def test_collision_stop_pose_delta_does_not_slide_along_obstacle() -> None:
    from NavVLAeval.common.simulators.habitat.vlnce031_actions import _collision_stop_pose_delta_position

    class _Sim:
        @staticmethod
        def step_filter(start, target):
            start = np.asarray(start, dtype=np.float64)
            desired = np.asarray(target, dtype=np.float64) - start
            # Simulate Habitat's tangential wall slide: same length, wrong direction.
            return start + np.asarray([0.0, desired[0], 0.0])

        @staticmethod
        def is_navigable(position):
            return bool(np.all(np.isfinite(position)))

    start = np.zeros(3)
    reached = _collision_stop_pose_delta_position(
        _Sim(), start, np.asarray([1.0, 0.0, 0.0]), max_substep_m=0.05
    )

    assert np.allclose(reached, start)


def test_collision_slide_pose_delta_retains_tangential_filter_motion() -> None:
    from NavVLAeval.common.simulators.habitat.vlnce031_actions import _collision_slide_pose_delta_position

    class _Sim:
        calls = 0

        @classmethod
        def step_filter(cls, start, target):
            cls.calls += 1
            start = np.asarray(start, dtype=np.float64)
            # Each requested forward increment hits a wall and slides right.
            return start + np.asarray([0.0, 0.05, 0.0])

        @staticmethod
        def is_navigable(position):
            return bool(np.all(np.isfinite(position)))

    reached = _collision_slide_pose_delta_position(
        _Sim(), np.zeros(3), np.asarray([0.10, 0.0, 0.0]), max_substep_m=0.05
    )

    assert _Sim.calls == 2
    assert np.allclose(reached, np.asarray([0.0, 0.10, 0.0]))
