from __future__ import annotations

import numpy as np

from NavVLAeval.common.types import ActionPrediction, EpisodeHistory, EvalEpisode, Pose4D, StepState
from NavVLAeval.traveluav.benchmark import TravelUAVBenchmarkSpec


def _observation(
    *,
    position: list[float],
    depth_value: float,
    movement_collision: bool = False,
) -> dict:
    return {
        "traveluav_episode": {
            "depth": [np.full((4, 4), depth_value, dtype=np.float32)],
            "sensors": {
                "state": {
                    "position": position,
                    "movement": {
                        "collision": movement_collision,
                        "collision_reason": "pose_mismatch" if movement_collision else "",
                    },
                }
            },
        }
    }


def _step_state(*, pre_observation: dict, post_observation: dict) -> StepState:
    episode = EvalEpisode(
        episode_uid="traveluav_seen:episode-1",
        source_episode_id="episode-1",
        scene_id="env-1",
        instruction="Fly to the target.",
        source="traveluav",
        input_namespace="traveluav_seen",
        input_root="/tmp/traveluav",
        payload={"goal_position": [100.0, 0.0, 0.0]},
    )
    pose_before = Pose4D(0.0, 0.0, 0.0, 0.0)
    pose_after = Pose4D(1.0, 0.0, 0.0, 0.0)
    return StepState(
        episode=episode,
        step_index=0,
        artifact_step_index=0,
        instruction=episode.instruction,
        history=EpisodeHistory(),
        pre_observation=pre_observation,
        post_observation=post_observation,
        pose_before=pose_before,
        pose_after=pose_after,
        prediction=ActionPrediction(
            normalized_actions=np.zeros((1, 4), dtype=np.float32),
            raw_actions=np.zeros((1, 4), dtype=np.float32),
        ),
        raw_action_chunk=np.zeros((1, 4), dtype=np.float32),
        world_waypoints=np.zeros((1, 4), dtype=np.float32),
        executed_action_count=1,
        distance_before=100.0,
        distance_after=99.0,
        path_length=1.0,
    )


def test_depth_only_policy_logs_movement_collision_without_stopping() -> None:
    runtime = TravelUAVBenchmarkSpec(
        stop_policy="none",
        depth_collision_policy="stop",
        ignore_movement_collision=True,
    ).create_runtime(None)
    state = _step_state(
        pre_observation=_observation(position=[0.0, 0.0, 0.0], depth_value=20.0),
        post_observation=_observation(
            position=[1.0, 0.0, 0.0],
            depth_value=10.0,
            movement_collision=True,
        ),
    )

    termination = runtime.update_termination(state)

    assert termination.done is False
    assert termination.reason == "running"
    assert termination.diagnostics["movement_collision"] is True
    assert termination.diagnostics["movement_collision_reason"] == "pose_mismatch"


def test_default_policy_keeps_movement_collision_stopping() -> None:
    runtime = TravelUAVBenchmarkSpec(
        stop_policy="none",
        depth_collision_policy="stop",
    ).create_runtime(None)
    state = _step_state(
        pre_observation=_observation(position=[0.0, 0.0, 0.0], depth_value=20.0),
        post_observation=_observation(
            position=[1.0, 0.0, 0.0],
            depth_value=10.0,
            movement_collision=True,
        ),
    )

    termination = runtime.update_termination(state)

    assert termination.done is True
    assert termination.reason == "collision:movement:pose_mismatch"


def test_depth_only_policy_stops_on_depth_collision() -> None:
    runtime = TravelUAVBenchmarkSpec(
        stop_policy="none",
        depth_collision_policy="stop",
        ignore_movement_collision=True,
    ).create_runtime(None)
    state = _step_state(
        pre_observation=_observation(position=[0.0, 0.0, 0.0], depth_value=10.0),
        post_observation=_observation(
            position=[1.0, 0.0, 0.0],
            depth_value=10.0,
            movement_collision=True,
        ),
    )

    termination = runtime.update_termination(state)

    assert termination.done is True
    assert termination.reason == "collision:depth:tiny diff"
    assert termination.diagnostics["movement_collision"] is True
    assert termination.diagnostics["depth_collision"]["collision"] is True
    assert termination.diagnostics["depth_collision"]["reason"] == "tiny diff"


def test_portable_config_enables_depth_only_collision_stopping() -> None:
    import yaml

    with open("NavVLAeval/traveluav/config_portable.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    assert config["benchmark"]["kwargs"]["depth_collision_policy"] == "stop"
    assert config["benchmark"]["kwargs"]["ignore_movement_collision"] is True
    assert config["env"]["kwargs"]["ignore_collision"] is False
