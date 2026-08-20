from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from starVLA.model.modules.qwen35_vision import BFLOAT16_BITS_STORAGE_ENCODING
from NavVLAeval.common.runner.backend_plan import WorkerBackendPlan
from NavVLAeval.common.log.artifacts import ArtifactStore, EpisodeArtifactWriter, write_json_atomic
from NavVLAeval.common.config import load_eval_config
from NavVLAeval.common.env.backends import create_environment_backend
from NavVLAeval.common.log.metrics import MetricEvaluator, episode_metric_payload, ndtw_score
from NavVLAeval.common.runner.planning import config_identity_sha256
from NavVLAeval.common.runtime_components import (
    build_action_codec,
    build_benchmark_runtime,
    build_model,
    build_runtime_dataset,
)
from NavVLAeval.common.types import (
    ActionPrediction,
    EnvironmentStepResult,
    EpisodeHistory,
    EpisodeResult,
    EvalEpisode,
    Pose4D,
    StepState,
    TerminationStatus,
    WorkerPlan,
)


def run_worker_plan(
    *,
    cfg,
    worker: WorkerPlan,
    runtime,
    model,
    env_backend,
    action_codec,
    runtime_dataset,
    run_identity: dict[str, str],
) -> dict[str, Any]:
    evaluator = MetricEvaluator(metric_keys=cfg.output.metrics)
    store = ArtifactStore(worker.run_root)
    for episode in worker.episodes:
        result = _run_episode_with_simulator_retry(
            cfg=cfg,
            worker=worker,
            store=store,
            episode=episode,
            runtime=runtime,
            model=model,
            env_backend=env_backend,
            action_codec=action_codec,
            runtime_dataset=runtime_dataset,
            run_identity=run_identity,
        )
        evaluator.add(result)
    return evaluator.summary()


def _run_episode_with_simulator_retry(
    *,
    cfg,
    worker: WorkerPlan,
    store: ArtifactStore,
    episode: EvalEpisode,
    runtime,
    model,
    env_backend,
    action_codec,
    runtime_dataset,
    run_identity: dict[str, str],
) -> EpisodeResult:
    max_attempts = 2 if worker.backend.type == "airsim" else 1
    for attempt_index in range(max_attempts):
        result = _run_episode(
            cfg=cfg,
            worker=worker,
            store=store,
            episode=episode,
            runtime=runtime,
            model=model,
            env_backend=env_backend,
            action_codec=action_codec,
            runtime_dataset=runtime_dataset,
            run_identity=run_identity,
        )
        if not _is_msgpackrpc_timeout(result):
            return result
        env_backend.close()
        if attempt_index + 1 < max_attempts:
            print(
                json.dumps(
                    {
                        "type": "airsim_episode_retry",
                        "episode_uid": episode.episode_uid,
                        "attempt": attempt_index + 2,
                        "reason": result.failure,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return result


def _is_msgpackrpc_timeout(result: EpisodeResult) -> bool:
    return "msgpackrpc.error.TimeoutError" in str(result.failure_traceback or "")


def _run_episode(
    *,
    cfg,
    worker: WorkerPlan,
    store: ArtifactStore,
    episode: EvalEpisode,
    runtime,
    model,
    env_backend,
    action_codec,
    runtime_dataset,
    run_identity: dict[str, str],
) -> EpisodeResult:
    writer = EpisodeArtifactWriter(store, episode)
    history = EpisodeHistory()
    initial_pose = runtime.initial_pose(episode)
    current_pose = initial_pose
    path_length = 0.0
    steps = 0
    executed_waypoint_index = 0
    try:
        env_backend.start_episode(episode, initial_pose)
        runtime.prepare_environment(episode, env_backend, initial_pose)
        reset_observation = env_backend.get_observation()
        reset_pose = reset_observation.get("pose")
        if isinstance(reset_pose, Pose4D):
            # Native simulators own their coordinate transform.  In particular,
            # VLN-CE JSON stores Habitat (x, y, z), while the evaluator uses
            # NavVLA Pose4D.  Use the simulator's reset pose before projecting
            # the first anchor-relative action chunk.
            current_pose = reset_pose
        history.poses.append(current_pose)
        termination = None
        episode_oracle_success = bool(runtime.is_success(initial_pose, episode))
        for step in range(int(cfg.benchmark.max_steps)):
            steps = step + 1
            pre_observation_frame_index = executed_waypoint_index
            pre_observation = env_backend.get_observation()
            pre_observation = runtime_dataset.prepare_observation_for_step(
                observation=pre_observation,
                step=pre_observation_frame_index,
            )
            instruction = runtime.instruction_for_step(episode, history, step)
            pre_observation = _prepare_observation_for_model(
                runtime,
                episode=episode,
                history=history,
                step=step,
                observation=pre_observation,
                instruction=instruction,
            )
            distance_before = float(runtime.distance_to_goal(current_pose, episode))
            example = runtime_dataset.build_example(
                observation=pre_observation,
                history=history,
                instruction=instruction,
            )
            full_prediction = _predict_and_decode(model=model, action_codec=action_codec, example=example)
            model_input_diagnostics = _model_input_diagnostics(
                example=example,
                prediction=full_prediction,
                runtime_dataset=runtime_dataset,
                history=history,
            )
            stop_threshold = _waypoint_motion_stop_threshold(cfg)
            if stop_threshold is not None:
                motion_score = _mean_adjacent_waypoint_translation(full_prediction.raw_actions)
                if motion_score <= stop_threshold:
                    success = int(runtime.is_success(current_pose, episode))
                    episode_oracle_success = bool(episode_oracle_success or bool(success))
                    termination = TerminationStatus(
                        done=True,
                        success=success,
                        oracle_success=int(episode_oracle_success),
                        reason="waypoint_motion_stop",
                        failure=None,
                        failure_type=None,
                        diagnostics={
                            "stop_rule": "mean_adjacent_waypoint_translation",
                            "stop_threshold": stop_threshold,
                            "motion_score": motion_score,
                        },
                    )
                    break
            pre_observation = _attach_online_visual_token_cache(pre_observation, full_prediction)
            full_world_waypoints = _world_waypoints(env_backend, current_pose, full_prediction.raw_actions)
            prediction = _truncate_action_prediction(
                full_prediction,
                execute_waypoints_per_step=_execute_waypoints_per_step(cfg),
            )
            success_waypoint_count = None
            stop_at_first_success = getattr(runtime, "stop_at_first_success_waypoint", None)
            if stop_at_first_success is None or bool(stop_at_first_success()):
                success_waypoint_count = _waypoint_count_through_first_success(
                    runtime,
                    episode=episode,
                    waypoints=_world_waypoints(env_backend, current_pose, prediction.raw_actions),
                )
            if success_waypoint_count is not None:
                prediction = _truncate_action_prediction(
                    prediction,
                    execute_waypoints_per_step=success_waypoint_count,
                )
                prediction.metadata["stop_at_first_success_waypoint"] = True
            raw_action_chunk = np.asarray(prediction.raw_actions, dtype=np.float32).reshape(-1, prediction.raw_actions.shape[-1])
            executed_action_count = int(raw_action_chunk.shape[0])
            planned_executed_world_waypoints = _selected_world_waypoints(
                full_world_waypoints,
                executed_action_count=executed_action_count,
            )
            if getattr(cfg.env, "transition_mode", None) == "benchmark_defined_transition":
                pre_state = StepState(
                    episode=episode,
                    step_index=step,
                    artifact_step_index=executed_waypoint_index,
                    instruction=instruction,
                    history=history,
                    pre_observation=pre_observation,
                    post_observation=pre_observation,
                    pose_before=current_pose,
                    pose_after=current_pose,
                    prediction=prediction,
                    raw_action_chunk=raw_action_chunk,
                    world_waypoints=full_world_waypoints,
                    executed_world_waypoints=planned_executed_world_waypoints,
                    executed_action_count=executed_action_count,
                    distance_before=distance_before,
                    distance_after=distance_before,
                    path_length=path_length,
                    diagnostics={},
                    action_observations=[],
                )
                step_result = runtime.offline_transition(pre_state)
            else:
                step_result = env_backend.apply_action(current_pose, prediction.raw_actions)
            executed_action_count = _effective_executed_action_count(raw_action_chunk, step_result)
            executed_world_waypoints = _executed_world_waypoints(
                full_world_waypoints,
                step_result,
                executed_action_count=executed_action_count,
            )
            action_observations = _prepare_action_observations_for_step(
                runtime_dataset,
                action_observations=step_result.action_observations,
                first_frame_index=pre_observation_frame_index + 1,
            )
            next_pose = step_result.next_pose
            path_length += float(np.linalg.norm(next_pose.as_array()[:3] - current_pose.as_array()[:3]))
            distance_after = float(runtime.distance_to_goal(next_pose, episode))
            if _needs_post_action_observation(runtime):
                post_observation = env_backend.get_observation()
            else:
                post_observation = step_result.observation
            post_observation = runtime_dataset.prepare_observation_for_step(
                observation=post_observation,
                step=pre_observation_frame_index + executed_action_count,
            )
            update_observations = _dataset_history_observations_for_update(
                runtime_dataset,
                pre_observation=pre_observation,
                post_observation=post_observation,
                step_result=step_result,
                action_observations=action_observations,
            )
            state = StepState(
                episode=episode,
                step_index=step,
                artifact_step_index=executed_waypoint_index,
                instruction=instruction,
                history=history,
                pre_observation=pre_observation,
                post_observation=post_observation,
                pose_before=current_pose,
                pose_after=next_pose,
                prediction=prediction,
                raw_action_chunk=raw_action_chunk,
                world_waypoints=full_world_waypoints,
                executed_world_waypoints=executed_world_waypoints,
                executed_action_count=executed_action_count,
                distance_before=distance_before,
                distance_after=distance_after,
                path_length=path_length,
                diagnostics={
                    **dict(step_result.diagnostics),
                    "captured_action_observation_count": len(action_observations),
                    "history_update": _history_update_diagnostics(update_observations),
                    "model_input": model_input_diagnostics,
                },
                action_observations=list(action_observations),
            )
            episode_oracle_success = bool(
                episode_oracle_success
                or _waypoints_enter_success(runtime, episode=episode, waypoints=executed_world_waypoints)
            )
            termination = runtime.update_termination(state)
            episode_oracle_success = bool(episode_oracle_success or bool(termination.oracle_success))
            if int(termination.oracle_success) != int(episode_oracle_success):
                diagnostics = dict(termination.diagnostics)
                diagnostics["waypoint_oracle_success"] = True
                termination = replace(
                    termination,
                    oracle_success=int(episode_oracle_success),
                    diagnostics=diagnostics,
                )
            state = replace(state, termination=termination)
            if cfg.output.save_step_artifacts:
                _log_step_artifacts(cfg, runtime, state, writer)
            update_observations = _attach_history_visual_token_cache(
                model=model,
                runtime_dataset=runtime_dataset,
                observations=update_observations,
                instruction=instruction,
            )
            for update_observation in update_observations:
                history = runtime_dataset.update_history(
                    history=history,
                    observation=update_observation,
                    prediction=prediction,
                    instruction=instruction,
                )
            history.poses.append(next_pose)
            current_pose = next_pose
            executed_waypoint_index += executed_action_count
            if termination.done or step_result.data_done:
                break
        else:
            success = int(runtime.is_success(current_pose, episode))
            episode_oracle_success = bool(episode_oracle_success or bool(success))
            termination = TerminationStatus(
                done=True,
                success=success,
                oracle_success=int(episode_oracle_success),
                reason="max_steps",
                failure=None,
                failure_type=None,
                diagnostics={},
            )
        if termination is None:
            success = int(runtime.is_success(current_pose, episode))
            episode_oracle_success = bool(episode_oracle_success or bool(success))
            termination = TerminationStatus(
                done=True,
                success=success,
                oracle_success=int(episode_oracle_success),
                reason="max_steps",
                failure=None,
                failure_type=None,
                diagnostics={},
            )
        result = EpisodeResult(
            episode_uid=episode.episode_uid,
            source_episode_id=episode.source_episode_id,
            scene_id=episode.scene_id,
            instruction=episode.instruction,
            success=int(termination.success),
            oracle_success=int(termination.oracle_success),
            final_distance=float(runtime.distance_to_goal(current_pose, episode)),
            path_length=float(path_length),
            gt_path_length=float(runtime.gt_path_length(episode)),
            steps=steps,
            failure=termination.failure,
            failure_type=termination.failure_type,
            termination_reason=termination.reason,
            failure_traceback=None,
            nDTW=_episode_ndtw(cfg, episode, history),
        )
    except Exception as exc:
        failure_traceback = traceback.format_exc()
        result = EpisodeResult(
            episode_uid=episode.episode_uid,
            source_episode_id=episode.source_episode_id,
            scene_id=episode.scene_id,
            instruction=episode.instruction,
            success=0,
            oracle_success=0,
            final_distance=0.0,
            path_length=path_length,
            gt_path_length=0.0,
            steps=steps,
            failure=f"{type(exc).__name__}: {exc}",
            failure_type=_classify_failure(exc),
            termination_reason="failure",
            failure_traceback=failure_traceback,
        )
    finally:
        env_backend.close_episode()
    writer.write_eval_info(_eval_info_payload(cfg, worker, episode, result, run_identity))
    return result


def _predict_and_decode(*, model, action_codec, example: dict[str, Any]) -> ActionPrediction:
    response = model.predict_action(example)
    try:
        return action_codec.decode(response)
    except Exception as exc:
        raise ActionDecodeError(str(exc)) from exc


def _waypoint_motion_stop_threshold(cfg) -> float | None:
    rule = str((getattr(cfg, "raw", {}) or {}).get("stop_rule") or "").strip()
    if not rule:
        return None
    if rule != "mean_adjacent_waypoint_translation":
        raise ValueError(f"unsupported stop_rule: {rule!r}")
    value = (getattr(cfg, "raw", {}) or {}).get("stop_threshold")
    if value is None:
        raise ValueError("stop_threshold is required when stop_rule is enabled")
    threshold = float(value)
    if threshold <= 0:
        raise ValueError(f"stop_threshold must be positive, got {threshold}")
    return threshold


def _mean_adjacent_waypoint_translation(actions: Any) -> float:
    waypoints = np.asarray(actions, dtype=np.float32)
    if waypoints.ndim != 2 or waypoints.shape[0] < 2 or waypoints.shape[1] < 3:
        raise ValueError(f"stop rule requires [horizon>=2, action_dim>=3] actions, got {waypoints.shape}")
    return float(np.linalg.norm(np.diff(waypoints[:, :3], axis=0), axis=1).mean())


def _attach_online_visual_token_cache(observation: dict[str, Any], prediction: ActionPrediction) -> dict[str, Any]:
    records = prediction.metadata.get("online_current_visual_tokens")
    if not records:
        return observation
    cache: dict[str, Any] = {}
    for record in records:
        if not isinstance(record, dict) or "camera_name" not in record or "tokens" not in record:
            continue
        tokens = np.asarray(record["tokens"])
        if tokens.ndim != 2 or tokens.shape[0] <= 0:
            continue
        if str(record.get("storage_encoding", "")) == BFLOAT16_BITS_STORAGE_ENCODING and tokens.dtype != np.uint16:
            raise TypeError(
                "model bfloat16_bits visual cache must use numpy uint16 tokens, "
                f"got {tokens.dtype}"
            )
        structured = any(
            key in record
            for key in ("grid_thw", "cache_stage", "visual_token_profile", "encoder_ckpt", "storage_encoding")
        )
        cache[str(record["camera_name"])] = (
            {
                "tokens": tokens,
                **({"grid_thw": np.asarray(record["grid_thw"], dtype=np.int64).reshape(3)} if record.get("grid_thw") is not None else {}),
                **({"cache_stage": str(record["cache_stage"])} if record.get("cache_stage") else {}),
                **({"visual_token_profile": str(record["visual_token_profile"])} if record.get("visual_token_profile") else {}),
                **({"encoder_ckpt": str(record["encoder_ckpt"])} if record.get("encoder_ckpt") else {}),
                **(
                    {"storage_encoding": str(record["storage_encoding"])}
                    if record.get("storage_encoding")
                    else {}
                ),
            }
            if structured
            else tokens
        )
    if not cache:
        return observation
    prepared = dict(observation)
    existing = prepared.get("navvla_online_visual_tokens")
    if isinstance(existing, dict):
        merged = dict(existing)
        merged.update(cache)
        cache = merged
    prepared["navvla_online_visual_tokens"] = cache
    return prepared


def _model_input_diagnostics(
    *,
    example: dict[str, Any],
    prediction: ActionPrediction,
    runtime_dataset,
    history: EpisodeHistory,
) -> dict[str, Any]:
    """Return compact evidence for the online history actually sent to the model.

    The payload is intentionally shape/index-only: it makes history/cache and
    execution investigations reproducible without serializing visual embeddings.
    """
    metadata = example.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    history_steps = metadata.get("history_steps")
    history_steps = history_steps if isinstance(history_steps, list) else []
    history_frame_indices = [
        int(item.get("frame_index", item.get("index")))
        for item in history_steps
        if isinstance(item, dict) and item.get("frame_index", item.get("index")) is not None
    ]
    cached_history = example.get("history_cached_embeds")
    cached_history_shape = None if cached_history is None else list(np.asarray(cached_history).shape)
    image_shapes = {
        str(camera_name): list(np.asarray(image).shape)
        for camera_name, image in dict(example.get("images") or {}).items()
        if image is not None
    }
    current_cache_shapes = []
    for record in list(prediction.metadata.get("online_current_visual_tokens") or []):
        if not isinstance(record, dict) or "tokens" not in record:
            continue
        current_cache_shapes.append(
            {
                "camera_name": str(record.get("camera_name", "")),
                "shape": list(np.asarray(record["tokens"]).shape),
            }
        )
    history_config = getattr(runtime_dataset, "history_config", None)
    stored_history_frame_indices = []
    for item in history.observations:
        if not isinstance(item, dict):
            continue
        item_metadata = item.get("navvla_eval")
        if isinstance(item_metadata, dict) and item_metadata.get("frame_index") is not None:
            stored_history_frame_indices.append(int(item_metadata["frame_index"]))
    return {
        "history_policy": metadata.get("history_policy"),
        "history_frames": _frame_index_summary(history_frame_indices),
        "stored_history_frames": _frame_index_summary(stored_history_frame_indices),
        "history_cached_embed_shape": cached_history_shape,
        "current_image_shapes": image_shapes,
        "current_frame_cached_as_history": current_cache_shapes,
        "configured_current_visual_tokens": getattr(history_config, "current_visual_tokens", None),
        "configured_history_visual_tokens": getattr(history_config, "history_visual_tokens", None),
    }


def _frame_index_summary(frame_indices: list[int], *, edge_count: int = 8) -> dict[str, Any]:
    values = [int(value) for value in frame_indices]
    if not values:
        return {"count": 0, "head": [], "tail": [], "contiguous": True}
    return {
        "count": len(values),
        "head": values[:edge_count],
        "tail": values[-edge_count:],
        "min": min(values),
        "max": max(values),
        "contiguous": all(right == left + 1 for left, right in zip(values, values[1:])),
    }


def _attach_history_visual_token_cache(
    *,
    model,
    runtime_dataset,
    observations: list[dict[str, Any]],
    instruction: str,
) -> list[dict[str, Any]]:
    encoder = getattr(model, "encode_history_images", None)
    image_provider = getattr(runtime_dataset, "history_visual_images", None)
    if encoder is None or image_provider is None or not observations:
        return observations

    requests: list[tuple[int, str, Any]] = []
    for observation_index, observation in enumerate(observations):
        existing_cache = observation.get("navvla_online_visual_tokens")
        cached_cameras = set(existing_cache) if isinstance(existing_cache, dict) else set()
        camera_images = image_provider(observation=observation, instruction=instruction)
        if not isinstance(camera_images, dict):
            raise TypeError("history_visual_images must return a camera-name dictionary")
        for camera_name, image in camera_images.items():
            if image is not None and str(camera_name) not in cached_cameras:
                requests.append((observation_index, str(camera_name), image))
    if not requests:
        return observations

    encoded = encoder([image for _observation_index, _camera_name, image in requests])
    if encoded is None:
        return observations
    encoded = list(encoded)
    if len(encoded) != len(requests):
        raise ValueError(
            f"history image encoder returned {len(encoded)} token blocks for {len(requests)} images"
        )

    expected_tokens = None
    history_config = getattr(runtime_dataset, "history_config", None)
    if history_config is not None:
        expected_tokens = int(history_config.history_visual_tokens)
    prepared = list(observations)
    for (observation_index, camera_name, _image), encoded_value in zip(requests, encoded, strict=True):
        payload = dict(encoded_value) if isinstance(encoded_value, dict) else {"tokens": encoded_value}
        token_array = np.asarray(payload.get("tokens"))
        if token_array.ndim != 2 or token_array.shape[0] <= 0:
            raise ValueError(
                f"history image encoder returned invalid tokens for {camera_name}: shape={token_array.shape}"
            )
        if (
            str(payload.get("storage_encoding", "")) == BFLOAT16_BITS_STORAGE_ENCODING
            and token_array.dtype != np.uint16
        ):
            raise TypeError(
                "history encoder bfloat16_bits cache must use numpy uint16 tokens, "
                f"got {token_array.dtype}"
            )
        if not payload.get("cache_stage") and expected_tokens is not None and int(token_array.shape[0]) != expected_tokens:
            raise ValueError(
                f"history image encoder returned {token_array.shape[0]} tokens for {camera_name}; "
                f"expected {expected_tokens}"
            )
        observation = dict(prepared[observation_index])
        cache = dict(observation.get("navvla_online_visual_tokens") or {})
        if payload.get("cache_stage"):
            grid = np.asarray(payload.get("grid_thw"), dtype=np.int64).reshape(3)
            stage = str(payload.get("cache_stage"))
            if stage == "vit_postmerge_pool4":
                valid = bool((grid > 0).all()) and int(grid[1]) % 2 == 0 and int(grid[2]) % 2 == 0
                valid = valid and int(token_array.shape[0]) == 4
            elif stage == "vit_postmerge":
                valid = int(grid.prod()) // 4 == int(token_array.shape[0])
            else:
                valid = int(grid.prod()) == int(token_array.shape[0])
            if not valid:
                raise ValueError(f"history cache grid {grid.tolist()} does not match token shape {token_array.shape}")
            cache[camera_name] = {
                "tokens": token_array,
                "grid_thw": grid,
                "cache_stage": str(payload["cache_stage"]),
                **({"visual_token_profile": str(payload["visual_token_profile"])} if payload.get("visual_token_profile") else {}),
                **({"encoder_ckpt": str(payload["encoder_ckpt"])} if payload.get("encoder_ckpt") else {}),
                **(
                    {"storage_encoding": str(payload["storage_encoding"])}
                    if payload.get("storage_encoding")
                    else {}
                ),
            }
        else:
            cache[camera_name] = token_array
        observation["navvla_online_visual_tokens"] = cache
        prepared[observation_index] = observation
    return prepared


def _prepare_observation_for_model(runtime, **kwargs) -> dict[str, Any]:
    hook = getattr(runtime, "prepare_observation_for_model", None)
    if hook is None:
        return kwargs["observation"]
    return hook(**kwargs)


def _needs_post_action_observation(runtime) -> bool:
    hook = getattr(runtime, "needs_post_action_observation", None)
    if hook is None:
        return False
    return bool(hook())


def _log_step_artifacts(cfg, runtime, state: StepState, writer: EpisodeArtifactWriter) -> None:
    hook = getattr(runtime, "log_step_artifacts", None)
    benchmark_specific = hook(state, writer) if hook is not None else None
    if benchmark_specific is None:
        benchmark_specific = {}
    writer.write_common_step_artifacts(
        state=state,
        benchmark_specific=benchmark_specific,
        save_images=cfg.output.save_images,
        image_cameras=cfg.output.image_cameras,
        action_observation_image_policy=cfg.output.action_observation_image_policy,
    )


def _prepare_action_observations_for_step(
    runtime_dataset,
    *,
    action_observations: list[dict[str, Any]],
    first_frame_index: int,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for offset, observation in enumerate(action_observations):
        prepared.append(
            runtime_dataset.prepare_observation_for_step(
                observation=observation,
                step=int(first_frame_index) + int(offset),
            )
        )
    return prepared


def _effective_executed_action_count(raw_action_chunk: np.ndarray, step_result: EnvironmentStepResult) -> int:
    diagnostics = dict(step_result.diagnostics or {})
    for key in ("attempted_waypoint_count", "completed_waypoint_count", "executed_waypoint_count"):
        if diagnostics.get(key) is not None:
            count = int(diagnostics[key])
            if count >= 0:
                return count
    if step_result.action_observations:
        return len(step_result.action_observations)
    return int(np.asarray(raw_action_chunk).reshape(-1, np.asarray(raw_action_chunk).shape[-1]).shape[0])


def _history_update_diagnostics(observations: list[dict[str, Any]]) -> dict[str, Any]:
    frame_indices = []
    for observation in observations:
        metadata = observation.get("navvla_eval")
        if isinstance(metadata, dict) and metadata.get("frame_index") is not None:
            frame_indices.append(int(metadata["frame_index"]))
    return {
        "observation_count": len(observations),
        "frame_indices": frame_indices,
    }


def _executed_world_waypoints(
    full_world_waypoints: np.ndarray,
    step_result: EnvironmentStepResult,
    *,
    executed_action_count: int,
) -> np.ndarray:
    diagnostics = dict(step_result.diagnostics or {})
    if diagnostics.get("actual_waypoint_poses") is not None:
        return _waypoint_array(diagnostics["actual_waypoint_poses"])
    if diagnostics.get("executed_world_waypoints") is not None:
        waypoints = _waypoint_array(diagnostics["executed_world_waypoints"])
        completed = diagnostics.get("completed_waypoint_count")
        if completed is not None:
            completed_count = max(0, int(completed))
            waypoints = waypoints[: min(completed_count, waypoints.shape[0])]
        return _waypoints_for_oracle_from_execution_mode(waypoints, diagnostics)
    return _selected_world_waypoints(
        full_world_waypoints,
        executed_action_count=executed_action_count,
    )


def _selected_world_waypoints(full_world_waypoints: np.ndarray, *, executed_action_count: int) -> np.ndarray:
    waypoints = _waypoint_array(full_world_waypoints)
    keep = max(0, min(int(executed_action_count), int(waypoints.shape[0])))
    return waypoints[:keep]


def _waypoints_for_oracle_from_execution_mode(waypoints: np.ndarray, diagnostics: dict[str, Any]) -> np.ndarray:
    mode = str(diagnostics.get("action_execution_mode") or diagnostics.get("execution_mode") or "")
    if mode == "teleport_final" and waypoints.shape[0] > 0:
        return waypoints[-1:]
    return waypoints


def _waypoint_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return array.reshape(-1, array.shape[-1])


def _waypoints_enter_success(runtime, *, episode: EvalEpisode, waypoints: np.ndarray | None) -> bool:
    if waypoints is None:
        return False
    waypoint_array = _waypoint_array(waypoints)
    if waypoint_array.shape[0] == 0 or waypoint_array.shape[-1] < 3:
        return False
    for waypoint in waypoint_array:
        yaw = float(waypoint[3]) if waypoint.shape[0] > 3 else 0.0
        pose = Pose4D(float(waypoint[0]), float(waypoint[1]), float(waypoint[2]), yaw)
        if runtime.is_success(pose, episode):
            return True
    return False


def _waypoint_count_through_first_success(
    runtime,
    *,
    episode: EvalEpisode,
    waypoints: np.ndarray | None,
) -> int | None:
    waypoint_array = _waypoint_array(waypoints)
    if waypoint_array.shape[0] == 0 or waypoint_array.shape[-1] < 3:
        return None
    for index, waypoint in enumerate(waypoint_array):
        yaw = float(waypoint[3]) if waypoint.shape[0] > 3 else 0.0
        pose = Pose4D(float(waypoint[0]), float(waypoint[1]), float(waypoint[2]), yaw)
        if runtime.is_success(pose, episode):
            return index + 1
    return None


def _dataset_history_observations_for_update(
    runtime_dataset,
    *,
    pre_observation: dict[str, Any],
    post_observation: dict[str, Any],
    step_result: EnvironmentStepResult,
    action_observations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    hook = getattr(runtime_dataset, "history_observations_for_update", None)
    if hook is None:
        waypoint_observations = list(action_observations if action_observations is not None else step_result.action_observations)
        return waypoint_observations if waypoint_observations else [post_observation]
    return hook(
        pre_observation=pre_observation,
        post_observation=post_observation,
        step_result=step_result,
        action_observations=action_observations,
    )


def _execute_waypoints_per_step(cfg) -> int | None:
    env_kwargs = getattr(cfg.env, "kwargs", {}) or {}
    value = env_kwargs.get("execute_waypoints_per_step")
    if value is None:
        return None
    count = int(value)
    if count <= 0:
        raise ValueError(f"execute_waypoints_per_step must be positive, got {value}")
    return count


def _truncate_action_prediction(prediction: ActionPrediction, *, execute_waypoints_per_step: int | None) -> ActionPrediction:
    if execute_waypoints_per_step is None:
        return prediction
    raw_actions = np.asarray(prediction.raw_actions)
    normalized_actions = np.asarray(prediction.normalized_actions)
    if raw_actions.ndim == 0 or normalized_actions.ndim == 0:
        raise ValueError("action prediction arrays must include an action-horizon dimension")
    action_count = int(raw_actions.shape[0])
    if action_count <= 0:
        raise ValueError("action prediction raw_actions must contain at least one action")
    keep = min(int(execute_waypoints_per_step), action_count)
    truncated_raw = raw_actions[:keep].copy()
    truncated_normalized = normalized_actions[:keep].copy()
    metadata = dict(prediction.metadata)
    metadata["execute_waypoints_per_step"] = keep
    metadata["original_action_horizon"] = action_count
    return ActionPrediction(
        normalized_actions=truncated_normalized,
        raw_actions=truncated_raw,
        metadata=metadata,
    )


def _eval_info_payload(
    cfg,
    worker: WorkerPlan,
    episode: EvalEpisode,
    result: EpisodeResult,
    run_identity: dict[str, str],
) -> dict[str, Any]:
    return {
        "episode_uid": episode.episode_uid,
        "metrics": episode_metric_payload(result, metric_keys=cfg.output.metrics),
        "schema_version": 1,
        "benchmark": cfg.benchmark.name,
        "run_name": cfg.output.run_name,
        "config_sha256": run_identity["config_sha256"],
        "input_fingerprint": run_identity["input_fingerprint"],
        "source_episode_id": episode.source_episode_id,
        "input_namespace": episode.input_namespace,
        "input_root": episode.input_root,
        "scene_id": episode.scene_id,
        "backend": _backend_payload(worker, episode),
        "source": episode.source,
        "instruction": episode.instruction,
        "status": "failed" if result.failure is not None else "completed",
        "attempt_id": worker.episode_attempts[episode.episode_uid],
        "worker_index": worker.worker_index,
        "physical_gpu_id": worker.physical_gpu_id,
        "steps": result.steps,
        "failure": result.failure,
        "failure_type": result.failure_type,
        "failure_traceback": result.failure_traceback,
        "termination_reason": result.termination_reason,
        "paths": {},
    }


def _backend_payload(worker: WorkerPlan, episode: EvalEpisode) -> dict[str, Any]:
    payload = {"type": worker.backend.type}
    if worker.backend.type == "airsim":
        airsim_port = worker.backend.kwargs.get("airsim_port")
        if airsim_port is None:
            raise ValueError("AirSim worker backend is missing kwargs['airsim_port']")
        payload["airsim_port"] = int(airsim_port)
        env_name = str(episode.payload.get("env_name") or "").strip()
        if not env_name:
            raise ValueError(f"AirSim episode {episode.episode_uid} is missing payload['env_name']")
        payload["env_name"] = env_name
    return payload


def _spl(result: EpisodeResult) -> float:
    if not result.success:
        return 0.0
    return result.gt_path_length / max(result.path_length, result.gt_path_length, 1e-6)


def _standard_spl(result: EpisodeResult) -> float:
    if not result.success:
        return 0.0
    return result.gt_path_length / max(result.path_length, result.gt_path_length, 1e-6)


def _episode_ndtw(cfg, episode: EvalEpisode, history: EpisodeHistory) -> float | None:
    if "nDTW" not in set(cfg.output.metrics):
        return None
    reference = episode.payload.get("reference_path_preprocessed_m") or episode.payload.get("reference_path_m") or episode.payload.get("trajectory")
    if not isinstance(reference, list):
        return None
    predicted = [pose.as_array()[:3].tolist() for pose in history.poses]
    reference_points = []
    for item in reference:
        if isinstance(item, dict):
            position = item.get("position") or item.get("xyz") or item.get("pose") or item.get("state")
        else:
            position = item
        if position is not None:
            reference_points.append(np.asarray(position, dtype=np.float32).reshape(-1)[:3].tolist())
    return ndtw_score(predicted, reference_points, success_distance=float(getattr(cfg.benchmark, "kwargs", {}).get("ndtw_success_distance", 1.0)))


def _classify_failure(exc: Exception) -> str:
    if isinstance(exc, ActionDecodeError):
        return "action_decode"
    text = str(exc).lower()
    if "model" in text:
        return "model_inference"
    if "observation" in text:
        return "observation_decode"
    return "benchmark_runtime"


class ActionDecodeError(RuntimeError):
    pass


def _world_waypoints(env_backend: Any, current_pose: Pose4D, raw_actions: np.ndarray) -> np.ndarray:
    if hasattr(env_backend, "project_action_to_world"):
        return np.asarray(env_backend.project_action_to_world(current_pose, raw_actions), dtype=np.float32).reshape(-1, 4)
    return np.asarray(raw_actions, dtype=np.float32).reshape(-1, raw_actions.shape[-1])


def load_worker_plan(path: str | Path) -> WorkerPlan:
    worker_path = Path(path)
    payload = json.loads(worker_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"worker plan must be a JSON object: {worker_path}")
    backend_payload = _required_mapping(payload, "backend", worker_path)
    try:
        backend = WorkerBackendPlan.from_jsonable(backend_payload)
        if backend.type not in {"offline", "airsim", "unrealzoo", "habitat"}:
            raise ValueError(f"Unsupported worker backend type: {backend.type!r}")
    except ValueError as exc:
        raise ValueError(f"invalid worker backend plan in {worker_path}: {exc}") from exc
    episodes_payload = payload.get("episodes")
    if not isinstance(episodes_payload, list):
        raise ValueError(f"worker plan episodes must be a list: {worker_path}")
    episodes = [_episode_from_payload(item, worker_path=worker_path) for item in episodes_payload]
    attempts = _required_mapping(payload, "episode_attempts", worker_path)
    return WorkerPlan(
        worker_index=int(payload["worker_index"]),
        physical_gpu_id=int(payload["physical_gpu_id"]),
        episodes=episodes,
        run_root=Path(payload["run_root"]),
        worker_log_path=Path(payload["worker_log_path"]),
        backend=backend,
        episode_attempts={str(key): str(value) for key, value in attempts.items()},
    )


def _episode_from_payload(payload: Any, *, worker_path: Path) -> EvalEpisode:
    if not isinstance(payload, dict):
        raise ValueError(f"worker plan episode must be an object: {worker_path}")
    required = {
        "episode_uid",
        "source_episode_id",
        "scene_id",
        "instruction",
        "source",
        "input_namespace",
        "input_root",
        "payload",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"worker plan episode is missing required fields {missing}: {worker_path}")
    if not isinstance(payload["payload"], dict):
        raise ValueError(f"worker plan episode payload must be an object: {worker_path}")
    return EvalEpisode(
        episode_uid=str(payload["episode_uid"]),
        source_episode_id=str(payload["source_episode_id"]),
        scene_id=str(payload["scene_id"]),
        instruction=str(payload["instruction"]),
        source=str(payload["source"]),
        input_namespace=str(payload["input_namespace"]),
        input_root=str(payload["input_root"]),
        payload=dict(payload["payload"]),
    )


def _required_mapping(payload: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"worker plan {key} must be an object: {path}")
    return value


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one NavVLAeval worker plan.")
    parser.add_argument("--worker-plan", required=True)
    return parser


def _build_model_and_environment(cfg: Any, worker: WorkerPlan) -> tuple[Any, Any]:
    if cfg.env.type == "habitat":
        env_backend = create_environment_backend(
            cfg=cfg.env,
            worker_backend=worker.backend,
            physical_gpu_id=worker.physical_gpu_id,
        )
        try:
            return build_model(cfg), env_backend
        except Exception:
            env_backend.close()
            raise
    model = build_model(cfg)
    env_backend = create_environment_backend(
        cfg=cfg.env,
        worker_backend=worker.backend,
        physical_gpu_id=worker.physical_gpu_id,
    )
    return model, env_backend


def main() -> None:
    args = build_argparser().parse_args()
    worker = load_worker_plan(args.worker_plan)
    config_path = worker.run_root / "config.yaml"
    cfg = load_eval_config(config_path)
    run_plan_path = ArtifactStore(worker.run_root).run_plan_path
    run_plan_payload = json.loads(run_plan_path.read_text(encoding="utf-8"))
    config_sha256 = config_identity_sha256(cfg)
    if str(run_plan_payload.get("config_sha256")) != config_sha256:
        raise SystemExit(f"worker config identity does not match run_plan: {config_path}")
    runtime = build_benchmark_runtime(cfg)
    model, env_backend = _build_model_and_environment(cfg, worker)
    try:
        action_codec = build_action_codec(cfg)
        runtime_dataset = build_runtime_dataset(cfg)
        summary = run_worker_plan(
            cfg=cfg,
            worker=worker,
            runtime=runtime,
            model=model,
            env_backend=env_backend,
            action_codec=action_codec,
            runtime_dataset=runtime_dataset,
            run_identity={
                "config_sha256": str(run_plan_payload["config_sha256"]),
                "input_fingerprint": str(run_plan_payload["input_fingerprint"]),
            },
        )
        write_json_atomic(worker.run_root / "worker_logs" / f"worker_{worker.worker_index}_summary.json", summary)
    finally:
        env_backend.close()


if __name__ == "__main__":
    main()
