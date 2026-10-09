"""Simulator rollout. Model state and preprocessing live behind the policy client."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from NavVLAeval.common.config import load_eval_config
from NavVLAeval.common.env.backends import create_environment_backend
from NavVLAeval.common.log.artifacts import ArtifactStore, EpisodeArtifactWriter, write_json_atomic
from NavVLAeval.common.log.metrics import MetricEvaluator, episode_metric_payload, ndtw_score
from NavVLAeval.common.runner.backend_plan import WorkerBackendPlan
from NavVLAeval.common.runtime_components import build_benchmark_runtime, build_model
from NavVLAeval.common.types import (
    ActionPrediction,
    EpisodeHistory,
    EpisodeResult,
    EvalEpisode,
    Pose4D,
    StepState,
    TerminationStatus,
    WorkerPlan,
)


def observation_frame(observation, *, frame_id, timestamp_s, pose):
    """Only observed inputs cross the model boundary; goals and metrics stay here."""
    frame = {
        "frame_id": int(frame_id),
        "timestamp_s": float(observation.get("timestamp_s", timestamp_s)),
        "images": observation["images"],
        "body_pose": observation.get("body_pose", pose.as_array()),
    }
    for key in ("camera_poses", "body_rotation"):
        if key in observation:
            frame[key] = observation[key]
    return frame


def run_worker_plan(*, cfg, worker, runtime, model, env_backend):
    evaluator = MetricEvaluator(metric_keys=cfg.output.metrics)
    for episode in worker.episodes:
        evaluator.add(
            _run_episode(
                cfg=cfg,
                worker=worker,
                store=ArtifactStore(worker.run_root),
                episode=episode,
                runtime=runtime,
                model=model,
                env_backend=env_backend,
            )
        )
    return evaluator.summary()


def _run_episode(*, cfg, worker, store, episode, runtime, model, env_backend):
    history = EpisodeHistory()
    path_length, steps, tick = 0.0, 0, 0
    oracle = False
    failure, failure_traceback, failure_type = None, None, None
    final_distance, gt_path_length = None, None
    termination = TerminationStatus(False, 0, 0, "running", None, None)
    writer = EpisodeArtifactWriter(store, episode)
    session_id = episode.episode_uid
    stage = "simulator_reset"
    try:
        initial_pose = runtime.initial_pose(episode)
        env_backend.start_episode(episode, initial_pose)
        runtime.prepare_environment(episode, env_backend, initial_pose)
        observation = env_backend.get_observation()
        current_pose = observation["pose"]
        history.poses.append(current_pose)
        runtime.observe(episode, observation)
        final_distance = runtime.distance_to_goal(current_pose, episode)
        gt_path_length = runtime.gt_path_length(episode)
        oracle = runtime.is_success(current_pose, episode)
        fps = float(cfg.observation.get("fps", 1.0))
        frames = [observation_frame(observation, frame_id=0, timestamp_s=0, pose=current_pose)]
        stage = "model_reset"
        model.reset(
            session_id=session_id,
            episode_id=episode.source_episode_id,
            statistics_key=cfg.model.statistics_key,
            seed=cfg.model.seed + worker.episode_ordinals[episode.episode_uid],
        )
        for step in range(cfg.benchmark.max_steps):
            steps = step + 1
            instruction = runtime.instruction_for_step(episode, history, step)
            prepared = runtime.prepare_observation_for_model(
                episode=episode, history=history, step=step, observation=observation, instruction=instruction
            )
            instruction = prepared.get("instruction", instruction)
            stage = "model_prediction"
            response = model.predict(frames=frames, instruction=instruction)
            if response.get("stop", False):
                termination = runtime.finalize(
                    episode, current_pose, reason="model_stop", previous=termination, oracle_success=oracle
                )
                break
            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] < 4 or not np.isfinite(actions).all():
                raise ValueError(f"Policy actions must be finite [horizon, dim>=4], got {actions.shape}")
            if response["action_spec"] != "anchor_relative_body_frame_xyz_yaw":
                raise ValueError(f"Unsupported action specification: {response['action_spec']}")
            stop_reason = None
            if cfg.raw.get("stop_rule") == "mean_adjacent_waypoint_translation":
                motion = (
                    float(np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1).mean())
                    if len(actions) > 1
                    else float(np.linalg.norm(actions[0, :3]))
                )
                if motion <= float(cfg.raw["stop_threshold"]):
                    stop_reason = stop_reason or "waypoint_motion_stop"
            if stop_reason:
                termination = runtime.finalize(
                    episode, current_pose, reason=stop_reason, previous=termination, oracle_success=oracle
                )
                break
            keep = cfg.env.kwargs.get("execute_waypoints_per_step")
            if keep is not None:
                actions = actions[: int(keep)]
            world_waypoints = env_backend.project_action_to_world(current_pose, actions)
            if cfg.raw.get("stop_at_first_success_waypoint", False):
                for index, row in enumerate(world_waypoints):
                    if runtime.is_success(Pose4D(*map(float, row[:4])), episode):
                        actions = actions[: index + 1]
                        break
            prediction = ActionPrediction(raw_actions=actions, metadata=response.get("metadata", {}))
            stage = "simulator_step"
            before = current_pose
            distance_before = final_distance
            step_result = env_backend.apply_action(before, actions)
            current_pose = step_result.next_pose
            actual_rows = step_result.diagnostics.get("actual_waypoint_poses", [current_pose.as_array()])
            actual = [Pose4D(*map(float, row[:4])) for row in actual_rows]
            if not actual or not np.array_equal(actual[-1].as_array(), current_pose.as_array()):
                actual.append(current_pose)
            for pose in actual:
                path_length += float(np.linalg.norm(pose.as_array()[:3] - history.poses[-1].as_array()[:3]))
                history.poses.append(pose)
                oracle = bool(oracle or runtime.is_success(pose, episode))
            post = step_result.observation
            observations = list(step_result.action_observations)
            if not observations or not np.array_equal(observations[-1]["pose"].as_array(), current_pose.as_array()):
                observations.append(post)
            frames = []
            for obs in observations:
                tick += 1
                runtime.observe(episode, obs)
                pose = obs["pose"]
                oracle = bool(oracle or runtime.is_success(pose, episode))
                frames.append(observation_frame(obs, frame_id=tick, timestamp_s=tick / fps, pose=pose))
            runtime.observe(episode, post)
            final_distance = float(runtime.distance_to_goal(current_pose, episode))
            stage = "task_scoring"
            state = StepState(
                episode=episode,
                step_index=step,
                artifact_step_index=tick - len(frames),
                instruction=instruction,
                history=history,
                pre_observation=observation,
                post_observation=post,
                pose_before=before,
                pose_after=current_pose,
                prediction=prediction,
                raw_action_chunk=actions,
                world_waypoints=world_waypoints,
                executed_world_waypoints=np.asarray([p.as_array() for p in actual]),
                executed_action_count=len(actual),
                distance_before=distance_before,
                distance_after=final_distance,
                path_length=path_length,
                diagnostics=step_result.diagnostics,
                action_observations=observations,
            )
            termination = runtime.update_termination(state)
            oracle = bool(oracle or termination.oracle_success)
            termination = replace(termination, oracle_success=int(oracle))
            state = replace(state, termination=termination)
            if cfg.output.save_step_artifacts:
                writer.write_common_step_artifacts(
                    state=state,
                    benchmark_specific=runtime.log_step_artifacts(state, writer) or {},
                    save_images=cfg.output.save_images,
                    image_cameras=cfg.output.image_cameras,
                    action_observation_image_policy=cfg.output.action_observation_image_policy,
                )
            observation = post
            if termination.done:
                break
            if step_result.data_done:
                termination = runtime.finalize(
                    episode, current_pose, reason="data_done", previous=termination, oracle_success=oracle
                )
                break
        else:
            termination = runtime.finalize(
                episode, current_pose, reason="max_steps", previous=termination, oracle_success=oracle
            )
    except Exception as exc:
        failure, failure_type, failure_traceback = f"{type(exc).__name__}: {exc}", stage, traceback.format_exc()
        termination = TerminationStatus(True, 0, int(oracle), "failure", failure, failure_type)
    finally:
        for label, cleanup in (
            ("model_close", lambda: model.close_session(session_id)),
            ("simulator_close", env_backend.close_episode),
        ):
            try:
                cleanup()
            except Exception as exc:
                if failure is None:
                    failure, failure_type, failure_traceback = (
                        f"{type(exc).__name__}: {exc}",
                        label,
                        traceback.format_exc(),
                    )
                    termination = TerminationStatus(True, 0, int(oracle), "failure", failure, failure_type)
                else:
                    logging.exception("Episode cleanup failed after %s", failure)
    reference = episode.payload.get("reference_points", [])
    ndtw = ndtw_score(
        [pose.as_array()[:3] for pose in history.poses],
        reference,
        success_distance=float(
            cfg.benchmark.kwargs.get(
                "ndtw_success_distance",
                cfg.benchmark.kwargs.get("success_distance", cfg.benchmark.kwargs.get("success_radius", 1.0)),
            )
        ),
    )
    result = EpisodeResult(
        episode_uid=episode.episode_uid,
        source_episode_id=episode.source_episode_id,
        scene_id=episode.scene_id,
        instruction=episode.instruction,
        success=int(termination.success),
        oracle_success=int(oracle),
        final_distance=final_distance,
        path_length=path_length,
        gt_path_length=gt_path_length,
        steps=steps,
        failure=failure or termination.failure,
        failure_type=failure_type or termination.failure_type,
        termination_reason=termination.reason,
        failure_traceback=failure_traceback,
        nDTW=ndtw,
    )
    writer.write_step_json("trajectory.json", {"poses": [pose.as_array().tolist() for pose in history.poses]})
    writer.write_eval_info(
        {
            **asdict(result),
            "metrics": episode_metric_payload(result, metric_keys=cfg.output.metrics),
            "schema_version": 2,
            "benchmark": cfg.benchmark.name,
            "run_name": cfg.output.run_name,
            "input_namespace": episode.input_namespace,
            "input_root": episode.input_root,
            "source": episode.source,
            "status": "failed" if result.failure else "completed",
            "attempt_id": worker.episode_attempts[episode.episode_uid],
            "worker_index": worker.worker_index,
            "physical_gpu_id": worker.physical_gpu_id,
            "backend": worker.backend.to_jsonable(),
            "paths": {},
        }
    )
    return result


def load_worker_plan(path):
    payload = json.loads(Path(path).read_text())
    payload["episodes"] = [EvalEpisode(**episode) for episode in payload["episodes"]]
    payload["backend"] = WorkerBackendPlan.from_jsonable(payload["backend"])
    payload["run_root"] = Path(payload["run_root"])
    payload["worker_log_path"] = Path(payload["worker_log_path"])
    return WorkerPlan(**payload)


def main():
    def stop_worker(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_worker)
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-plan", required=True)
    worker = load_worker_plan(parser.parse_args().worker_plan)
    cfg = load_eval_config(worker.run_root / "config.yaml")
    runtime = build_benchmark_runtime(cfg)
    model = build_model(cfg, worker.worker_index)
    env = None
    try:
        env = create_environment_backend(
            cfg=cfg.env, worker_backend=worker.backend, physical_gpu_id=worker.physical_gpu_id
        )
        summary = run_worker_plan(cfg=cfg, worker=worker, runtime=runtime, model=model, env_backend=env)
        write_json_atomic(worker.run_root / "worker_logs" / f"worker_{worker.worker_index}_summary.json", summary)
    finally:
        try:
            model.close()
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
