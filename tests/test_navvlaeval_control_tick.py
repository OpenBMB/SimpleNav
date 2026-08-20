from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from NavVLAeval.common.log.artifacts import ArtifactStore
from NavVLAeval.common.runner.backend_plan import WorkerBackendPlan
from NavVLAeval.common.runner.worker import (
    _effective_executed_action_count,
    _executed_world_waypoints,
    _run_episode,
    _waypoint_count_through_first_success,
)
from NavVLAeval.common.types import (
    ActionPrediction,
    EnvironmentStepResult,
    EpisodeHistory,
    EvalEpisode,
    Pose4D,
    StepState,
    TerminationStatus,
    WorkerPlan,
)


class _Runtime:
    def initial_pose(self, episode: EvalEpisode) -> Pose4D:
        del episode
        return Pose4D(0.0, 0.0, 0.0, 0.0)

    def prepare_environment(self, episode: EvalEpisode, env, initial_pose: Pose4D) -> None:
        del episode, env, initial_pose

    def instruction_for_step(self, episode: EvalEpisode, history: EpisodeHistory, step: int) -> str:
        del history, step
        return episode.instruction

    def distance_to_goal(self, pose: Pose4D, episode: EvalEpisode) -> float:
        del pose, episode
        return 1.0

    def gt_path_length(self, episode: EvalEpisode) -> float:
        del episode
        return 1.0

    def is_success(self, pose: Pose4D, episode: EvalEpisode) -> bool:
        del pose, episode
        return False

    def update_termination(self, state: StepState) -> TerminationStatus:
        return TerminationStatus(
            done=state.step_index >= 1,
            success=0,
            oracle_success=0,
            reason="done" if state.step_index >= 1 else "continue",
            failure=None,
            failure_type=None,
        )


class _RuntimeDataset:
    def __init__(self) -> None:
        self.prepared_control_ticks: list[int] = []
        self.example_control_ticks: list[int] = []

    def prepare_observation_for_step(self, *, observation: dict, step: int) -> dict:
        self.prepared_control_ticks.append(int(step))
        prepared = dict(observation)
        prepared["control_tick"] = int(step)
        return prepared

    def build_example(self, *, observation: dict, history: EpisodeHistory, instruction: str) -> dict:
        del history, instruction
        self.example_control_ticks.append(int(observation["control_tick"]))
        return {"observation": observation}

    def history_observations_for_update(
        self,
        *,
        pre_observation: dict,
        post_observation: dict,
        step_result: EnvironmentStepResult,
        action_observations: list[dict] | None = None,
    ) -> list[dict]:
        del post_observation, step_result, action_observations
        assert "navvla_online_visual_tokens" in pre_observation
        return [pre_observation]

    def update_history(
        self,
        *,
        history: EpisodeHistory,
        observation: dict,
        prediction: ActionPrediction,
        instruction: str,
    ) -> EpisodeHistory:
        del prediction
        history.observations.append(observation)
        history.instructions.append(instruction)
        return history


class _Model:
    def predict_action(self, example: dict) -> dict:
        del example
        actions = np.zeros((3, 4), dtype=np.float32)
        return {
            "normalized_actions": actions,
            "metadata": {
                "online_current_visual_tokens": [
                    {"camera_name": "front", "tokens": np.zeros((4, 8), dtype=np.float32)}
                ]
            },
        }


class _Codec:
    def decode(self, response: dict) -> ActionPrediction:
        actions = np.asarray(response["normalized_actions"], dtype=np.float32)
        return ActionPrediction(
            normalized_actions=actions,
            raw_actions=actions,
            metadata=dict(response.get("metadata") or {}),
        )


class _Environment:
    type = "offline"

    def start_episode(self, episode: EvalEpisode, initial_pose: Pose4D) -> None:
        del episode, initial_pose

    def get_observation(self) -> dict:
        return {"image": np.zeros((2, 2, 3), dtype=np.uint8)}

    def apply_action(self, current_pose: Pose4D, raw_actions: np.ndarray) -> EnvironmentStepResult:
        del raw_actions
        return EnvironmentStepResult(
            next_pose=current_pose,
            observation=self.get_observation(),
            data_done=False,
            diagnostics={"executed_waypoint_count": 3},
            action_observations=[self.get_observation() for _ in range(3)],
        )

    def close_episode(self) -> None:
        return None


def test_effective_control_ticks_prefer_completed_waypoints() -> None:
    raw_actions = np.zeros((3, 4), dtype=np.float32)
    step_result = EnvironmentStepResult(
        next_pose=Pose4D(0.0, 0.0, 0.0, 0.0),
        observation={},
        data_done=False,
        diagnostics={"executed_waypoint_count": 3, "completed_waypoint_count": 2},
    )

    assert _effective_executed_action_count(raw_actions, step_result) == 2


def test_effective_control_ticks_prefer_attempted_waypoints_over_pose_matches() -> None:
    raw_actions = np.zeros((3, 4), dtype=np.float32)
    step_result = EnvironmentStepResult(
        next_pose=Pose4D(0.0, 0.0, 0.0, 0.0),
        observation={},
        data_done=False,
        diagnostics={
            "executed_waypoint_count": 3,
            "completed_waypoint_count": 0,
            "attempted_waypoint_count": 3,
        },
    )

    assert _effective_executed_action_count(raw_actions, step_result) == 3


def test_oracle_waypoints_use_actual_rpc_poses_not_planned_waypoints() -> None:
    planned = np.asarray([[10.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    step_result = EnvironmentStepResult(
        next_pose=Pose4D(0.25, 0.0, 0.0, 0.0),
        observation={},
        data_done=False,
        diagnostics={
            "executed_world_waypoints": planned.tolist(),
            "completed_waypoint_count": 0,
            "actual_waypoint_poses": [[0.25, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]],
        },
    )

    actual = _executed_world_waypoints(planned, step_result, executed_action_count=2)

    np.testing.assert_allclose(actual, [[0.25, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]])


def test_waypoint_execution_stops_at_the_first_success_waypoint() -> None:
    class Runtime:
        def is_success(self, pose: Pose4D, episode: EvalEpisode) -> bool:
            del episode
            return pose.x >= 2.0

    episode = EvalEpisode(
        episode_uid="episode-1",
        source_episode_id="episode-1",
        scene_id="scene",
        instruction="go",
        source="test",
        input_namespace="test",
        input_root="/tmp",
        payload={},
    )
    waypoints = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    assert _waypoint_count_through_first_success(Runtime(), episode=episode, waypoints=waypoints) == 2


def test_run_episode_advances_tvi_by_executed_control_ticks(tmp_path: Path) -> None:
    episode = EvalEpisode(
        episode_uid="episode-1",
        source_episode_id="episode-1",
        scene_id="scene",
        instruction="go",
        source="test",
        input_namespace="test",
        input_root=str(tmp_path),
        payload={},
    )
    worker = WorkerPlan(
        worker_index=0,
        physical_gpu_id=0,
        episodes=[episode],
        run_root=tmp_path,
        worker_log_path=tmp_path / "worker.log",
        backend=WorkerBackendPlan(type="offline"),
        episode_attempts={episode.episode_uid: "attempt-1"},
    )
    cfg = SimpleNamespace(
        benchmark=SimpleNamespace(name="test", max_steps=2, kwargs={}),
        env=SimpleNamespace(transition_mode=None, kwargs={}),
        output=SimpleNamespace(
            run_name="test",
            save_step_artifacts=False,
            save_images=False,
            image_cameras=None,
            action_observation_image_policy="step",
            metrics=(),
        ),
    )
    dataset = _RuntimeDataset()

    result = _run_episode(
        cfg=cfg,
        worker=worker,
        store=ArtifactStore(tmp_path),
        episode=episode,
        runtime=_Runtime(),
        model=_Model(),
        env_backend=_Environment(),
        action_codec=_Codec(),
        runtime_dataset=dataset,
        run_identity={"config_sha256": "cfg", "input_fingerprint": "input"},
    )

    assert result.failure is None
    assert dataset.prepared_control_ticks == [0, 1, 2, 3, 3, 3, 4, 5, 6, 6]
    assert dataset.example_control_ticks == [0, 3]
