"""Export enhanced frame-based trajectories in an AerialVLN episode shell."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np

from vln_aug.actions import observation_indices, random_observation_indices
from vln_aug.image_stride import (
    assign_image_stride,
    normalize_image_stride_choices,
    stable_image_interval_seed,
)
from vln_aug.lerobot_io import EpisodeMetadata, iter_episode_tables
from vln_aug.lerobot_io import read_episode_metadata
from vln_aug.lightweight_subset import plan_lightweight_subset
from vln_aug.trajectory import (
    RetimedTrajectory,
    TrajectoryConfig,
    smooth_and_retime,
    stable_trajectory_seed,
)
from vln_aug.visualize import (
    compute_trajectory_metrics,
    plot_sampling_audit,
    plot_trajectory_comparison,
)
from vln_aug.world_pose import (
    AerialVLNOriginalPoseIndex,
    FrameMetadataWorldPoseStream,
    OpenFlyAnnotationPoseIndex,
    choose_world_pose_source,
    transform_world_poses_for_alignment,
    validate_episode_local_alignment,
)


AERIALVLN_RENDER_HEIGHT = 224
AERIALVLN_RENDER_WIDTH = 224
AERIALVLN_RENDER_CHANNELS = 3


def _validate_render_dimensions(width: int, height: int) -> tuple[int, int]:
    resolved_width = int(width)
    resolved_height = int(height)
    if resolved_width <= 0 or resolved_height <= 0:
        raise ValueError("render image dimensions must be positive")
    if resolved_width % 2 or resolved_height % 2:
        raise ValueError("render image dimensions must be even for YUV420P video")
    return resolved_width, resolved_height


def _yaw_quaternion_wxyz(yaw: float) -> list[float]:
    half = float(yaw) / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _enhanced_id(source_id: str) -> str:
    return f"{source_id}__enhanced_v1"


def _safe_path_component(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_." else "_" for character in str(value))
    return safe or "episode"


def build_aerialvln_episode(
    *,
    dataset_key: str,
    metadata: EpisodeMetadata,
    trajectory_id: str,
    task_index: int,
    instruction_text: str,
    trajectory: RetimedTrajectory,
) -> dict:
    controls = np.asarray(trajectory.control_poses, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != 4 or len(controls) < 2:
        raise ValueError("enhanced control trajectory must have shape [N, 4], N >= 2")
    if not np.all(np.isfinite(controls)):
        raise ValueError("enhanced control trajectory contains non-finite values")
    source_episode_id = str(metadata.episode_id)
    source_trajectory_id = str(trajectory_id)
    return {
        "episode_id": _enhanced_id(source_episode_id),
        "trajectory_id": _enhanced_id(source_trajectory_id),
        "scene_id": str(metadata.scene_id),
        "start_position": controls[0, :3].tolist(),
        "start_rotation": _yaw_quaternion_wxyz(float(controls[0, 3])),
        "goals": [{"position": controls[-1, :3].tolist()}],
        "reference_path": [
            [float(x), float(y), float(z), 0.0, 0.0, float(yaw)]
            for x, y, z, yaw in controls
        ],
        "actions": [],
        "instruction": {"instruction_text": str(instruction_text)},
    }


def build_export_record(
    *,
    dataset_key: str,
    metadata: EpisodeMetadata,
    trajectory: RetimedTrajectory,
    image_stride_choices: tuple[int, ...] = (1, 3, 5),
    image_stride: int | None = None,
    image_stride_policy: str = "fixed-per-episode",
    image_interval_seed: int = 0,
    source_pose_kind: str = "frame_observation_state",
    training_state_mode: str | None = None,
    coordinate_alignment: dict[str, float] | None = None,
    coordinate_alignment_transform: str = "identity",
) -> dict:
    trajectory_id, task_index, instruction_text = _source_episode_fields(metadata)
    episode = build_aerialvln_episode(
        dataset_key=dataset_key,
        metadata=metadata,
        trajectory_id=trajectory_id,
        task_index=task_index,
        instruction_text=instruction_text,
        trajectory=trajectory,
    )
    choices = normalize_image_stride_choices(image_stride_choices)
    if image_stride_policy == "fixed-per-episode":
        stride = (
            int(image_stride)
            if image_stride is not None
            else assign_image_stride(dataset_key, str(metadata.episode_id), choices)
        )
        if stride not in choices:
            raise ValueError("image_stride must be one of image_stride_choices")
        indices = observation_indices(len(trajectory.control_poses), stride)
        collection_metadata = {
            "collection_stride_policy": "fixed_per_episode",
            "collection_stride_waypoints": int(stride),
            "collection_waypoint_indices": indices.astype(int).tolist(),
            "collection_waypoint_gaps": np.diff(indices).astype(int).tolist(),
        }
    elif image_stride_policy == "deterministic-random-per-interval":
        if image_stride is not None:
            raise ValueError("image_stride override is incompatible with random intervals")
        episode_seed = stable_image_interval_seed(
            dataset_key, str(metadata.episode_id), image_interval_seed
        )
        indices = random_observation_indices(
            len(trajectory.control_poses), choices, seed=episode_seed
        )
        collection_metadata = {
            "collection_stride_policy": "deterministic_random_per_interval",
            "collection_stride_choices_waypoints": list(choices),
            "collection_stride_seed": int(episode_seed),
            "collection_waypoint_indices": indices.astype(int).tolist(),
            "collection_waypoint_gaps": np.diff(indices).astype(int).tolist(),
        }
    else:
        raise ValueError(f"unsupported image_stride_policy: {image_stride_policy}")
    alignment = coordinate_alignment or {}
    return {
        "episode": episode,
        "metadata": {
            "episode_id": episode["episode_id"],
            "trajectory_id": episode["trajectory_id"],
            "source_episode_id": str(metadata.episode_id),
            "source_trajectory_id": str(trajectory_id),
            "source_episode_index": int(metadata.episode_index),
            "source_task_index": int(task_index),
            "scene_id": str(metadata.scene_id),
            "source_pose_kind": str(source_pose_kind),
            "training_state_mode": training_state_mode,
            "coordinate_alignment_max_position_error_m": alignment.get(
                "max_position_error_m"
            ),
            "coordinate_alignment_max_yaw_error_rad": alignment.get(
                "max_yaw_error_rad"
            ),
            "coordinate_alignment_transform": str(coordinate_alignment_transform),
            "trajectory_mode": "absolute_pose_sequence",
            "control_frequency_hz": 1.0,
            "movement_speed_mps": float(trajectory.movement_speed_mps),
            "cruise_speed_mps": float(trajectory.cruise_speed_mps),
            "minimum_local_speed_mps": float(
                trajectory.minimum_local_speed_mps
            ),
            "target_arc_step_m": float(
                trajectory.cruise_speed_mps or trajectory.movement_speed_mps
            ),
            **collection_metadata,
            "collection_includes_real_terminal": True,
            "action_horizon": 8,
            "action_tail_policy": "repeat_last_absolute_waypoint",
            "source_frame_count": int(len(trajectory.source_poses)),
            "enhanced_waypoint_count": int(len(trajectory.control_poses)),
            "path_length_m": float(trajectory.path_length_m),
            "max_deviation_m": float(trajectory.max_deviation_m),
        },
    }


def validate_exported_episode(episode: dict) -> list[str]:
    errors = []
    required = (
        "episode_id",
        "trajectory_id",
        "scene_id",
        "start_position",
        "start_rotation",
        "goals",
        "reference_path",
        "actions",
        "instruction",
    )
    for key in required:
        if key not in episode:
            errors.append(f"missing field: {key}")
    if errors:
        return errors
    path = np.asarray(episode["reference_path"], dtype=float)
    if path.ndim != 2 or path.shape[1] != 6 or len(path) < 2:
        errors.append("reference_path must have shape [N, 6], N >= 2")
        return errors
    if not np.all(np.isfinite(path)):
        errors.append("reference_path contains non-finite values")
    if not np.allclose(path[0, :3], episode["start_position"], atol=1e-6):
        errors.append("start_position does not match reference_path start")
    if not episode["goals"] or not np.allclose(
        path[-1, :3], episode["goals"][0]["position"], atol=1e-6
    ):
        errors.append("goal position does not match reference_path terminal")
    indices = np.arange(0, len(path), 5, dtype=int)
    expected = np.arange(0, len(path), 5, dtype=int)
    if expected[-1] != len(path) - 1:
        expected = np.r_[expected, len(path) - 1]
    if episode["actions"] != []:
        errors.append("native discrete actions must remain empty for absolute-pose trajectories")
    return errors


def write_trajectory_package(
    output_dir: Path,
    *,
    dataset_key: str,
    source_split: Path,
    records: Iterable[dict],
    failures: Iterable[dict],
    cameras: list[dict],
    coordinate_metadata: dict,
    extra_manifest: dict | None = None,
) -> dict:
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite trajectory package: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    episode_count = 0
    render_request_count = 0
    image_stride_episode_counts: dict[int, int] = {}
    image_sampling_policy_episode_counts: dict[str, int] = {}
    image_interval_gap_counts: dict[int, int] = {}
    invalid = []
    failure_count = 0
    render_shapes = {
        (
            int(camera["width"]),
            int(camera["height"]),
            int(camera["channels"]),
        )
        for camera in cameras
    }
    if len(render_shapes) != 1:
        raise ValueError("all render cameras must use one common image shape")
    render_width, render_height, render_channels = next(iter(render_shapes))
    _validate_render_dimensions(render_width, render_height)
    if render_channels != AERIALVLN_RENDER_CHANNELS:
        raise ValueError("render cameras must use three RGB channels")
    try:
        trajectories = staging / "trajectories"
        trajectories.mkdir()
        render_dir = staging / "render"
        render_dir.mkdir()
        validation_dir = staging / "validation"
        validation_dir.mkdir()
        train_json = trajectories / "train.json"
        episodes_jsonl = trajectories / "episodes.jsonl"
        metadata_jsonl = trajectories / "augmentation_metadata.jsonl"
        render_jsonl = render_dir / "render_requests.jsonl"
        with (
            train_json.open("w", encoding="utf-8") as stream,
            episodes_jsonl.open("w", encoding="utf-8") as episode_stream,
            metadata_jsonl.open("w", encoding="utf-8") as metadata_stream,
            render_jsonl.open("w", encoding="utf-8") as render_stream,
        ):
            stream.write('{"episodes":[')
            first = True
            for record in records:
                episode = record["episode"]
                metadata = record["metadata"]
                errors = validate_exported_episode(episode)
                if errors:
                    invalid.append(
                        {"episode_id": episode.get("episode_id"), "errors": errors}
                    )
                    continue
                if not first:
                    stream.write(",")
                json.dump(episode, stream, ensure_ascii=False, separators=(",", ":"))
                compact_episode = json.dumps(episode, ensure_ascii=False, separators=(",", ":"))
                episode_stream.write(compact_episode + "\n")
                metadata = dict(metadata)
                policy = str(
                    metadata.get("collection_stride_policy", "fixed_per_episode")
                )
                image_sampling_policy_episode_counts[policy] = (
                    image_sampling_policy_episode_counts.get(policy, 0) + 1
                )
                stride = metadata.get("collection_stride_waypoints")
                if stride is not None:
                    stride = int(stride)
                    image_stride_episode_counts[stride] = (
                        image_stride_episode_counts.get(stride, 0) + 1
                    )
                collection_indices = [
                    int(value) for value in metadata["collection_waypoint_indices"]
                ]
                collection_gaps = np.diff(collection_indices).astype(int).tolist()
                metadata["collection_waypoint_gaps"] = collection_gaps
                for gap in collection_gaps:
                    image_interval_gap_counts[gap] = (
                        image_interval_gap_counts.get(gap, 0) + 1
                    )
                metadata["camera_count"] = len(cameras)
                metadata["render_request_count"] = (
                    len(collection_indices) * len(cameras)
                )
                metadata_stream.write(
                    json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                safe_episode_id = _safe_path_component(episode["episode_id"])
                for image_index, waypoint_index in enumerate(collection_indices):
                    pose = episode["reference_path"][waypoint_index]
                    yaw = float(pose[5])
                    collection_gap = (
                        0
                        if image_index == 0
                        else waypoint_index - collection_indices[image_index - 1]
                    )
                    for camera in cameras:
                        camera_key = str(camera["camera_key"])
                        request = {
                            "schema_version": "1.0",
                            "request_id": (
                                f"{episode['episode_id']}/{camera_key}/frame_{image_index:06d}"
                            ),
                            "dataset_key": str(dataset_key),
                            "episode_id": episode["episode_id"],
                            "trajectory_id": episode["trajectory_id"],
                            "source_episode_id": metadata["source_episode_id"],
                            "source_episode_index": metadata["source_episode_index"],
                            "scene_id": episode["scene_id"],
                            "image_index": int(image_index),
                            "waypoint_index": int(waypoint_index),
                            "collection_stride_policy": policy,
                            "collection_gap_waypoints": int(collection_gap),
                            "timestamp": float(waypoint_index),
                            "position_xyz": [float(value) for value in pose[:3]],
                            "orientation_quaternion_wxyz": _yaw_quaternion_wxyz(yaw),
                            "pose_xyz_rpy": [float(value) for value in pose],
                            "camera_key": camera_key,
                            "camera_name": str(camera["camera_name"]),
                            "expected_height": int(camera["height"]),
                            "expected_width": int(camera["width"]),
                            "expected_channels": int(camera["channels"]),
                            "expected_image_relpath": (
                                f"rendered_images/{safe_episode_id}/{camera_key}/"
                                f"frame_{image_index:06d}.png"
                            ),
                            "coordinate_metadata": coordinate_metadata,
                            "camera_metadata": camera.get("camera_metadata", {}),
                        }
                        if stride is not None:
                            request["collection_stride_waypoints"] = stride
                        if "collection_stride_choices_waypoints" in metadata:
                            request["collection_stride_choices_waypoints"] = metadata[
                                "collection_stride_choices_waypoints"
                            ]
                        render_stream.write(
                            json.dumps(request, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
                        render_request_count += 1
                first = False
                episode_count += 1
            stream.write("]}\n")

        with (validation_dir / "failures.jsonl").open("w", encoding="utf-8") as stream:
            for failure in failures:
                stream.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")
                failure_count += 1

        manifest = {
            "schema_version": "1.0",
            "trajectory_only": True,
            "complete_lerobot_split": False,
            "dataset_key": str(dataset_key),
            "source_split": str(Path(source_split).resolve()),
            "episode_count": episode_count,
            "failure_count": failure_count,
            "trajectory_file": "trajectories/train.json",
            "streaming_trajectory_file": "trajectories/episodes.jsonl",
            "augmentation_metadata_file": "trajectories/augmentation_metadata.jsonl",
            "render_request_file": "render/render_requests.jsonl",
            "trajectory_format": "aerialvln_episode_shell_absolute_pose_sequence",
            "source_pose_kind": coordinate_metadata.get(
                "render_pose_source", "frame_observation_state"
            ),
            "image_status": "not_collected",
            "render_request_count": render_request_count,
            "render_image_width": render_width,
            "render_image_height": render_height,
            "render_image_channels": render_channels,
            "image_stride_episode_counts": {
                str(key): value
                for key, value in sorted(image_stride_episode_counts.items())
            },
            "image_sampling_policy_episode_counts": dict(
                sorted(image_sampling_policy_episode_counts.items())
            ),
            "image_interval_gap_counts": {
                str(key): value for key, value in sorted(image_interval_gap_counts.items())
            },
        }
        manifest.update(extra_manifest or {})
        validation = {
            "valid": not invalid and episode_count > 0,
            "valid_episode_count": episode_count,
            "invalid_episode_count": len(invalid),
            "failure_count": failure_count,
            "render_request_count": render_request_count,
            "invalid": invalid,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (validation_dir / "summary.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            "# Enhanced VLN Trajectory Package\n\n"
            "- `trajectories/train.json`: AerialVLN-compatible complete trajectories.\n"
            "- `trajectories/episodes.jsonl`: streaming form, one episode per line.\n"
            "- `render/render_requests.jsonl`: direct absolute-pose image collection requests.\n"
            "- `validation/`: conversion validation and sampled plots.\n"
            "\nThis is a trajectory-only package until rendered images are collected.\n",
            encoding="utf-8",
        )
        if not validation["valid"]:
            raise ValueError("trajectory package validation failed")
        os.replace(staging, output)
        return validation
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_trajectory_package(package_dir: Path) -> dict:
    root = Path(package_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_render_shape = (
        int(manifest.get("render_image_width", AERIALVLN_RENDER_WIDTH)),
        int(manifest.get("render_image_height", AERIALVLN_RENDER_HEIGHT)),
        int(manifest.get("render_image_channels", AERIALVLN_RENDER_CHANNELS)),
    )
    episodes_path = root / "trajectories" / "episodes.jsonl"
    metadata_path = root / "trajectories" / "augmentation_metadata.jsonl"
    render_path = root / "render" / "render_requests.jsonl"
    errors = []
    episode_count = 0
    request_count = 0
    seen_requests = set()
    seen_paths = set()
    with (
        episodes_path.open("r", encoding="utf-8") as episode_stream,
        metadata_path.open("r", encoding="utf-8") as metadata_stream,
        render_path.open("r", encoding="utf-8") as render_stream,
    ):
        pending_request = None
        render_line_number = 0
        for episode_line_number, (episode_line, metadata_line) in enumerate(
            zip(episode_stream, metadata_stream), 1
        ):
            episode = json.loads(episode_line)
            metadata = json.loads(metadata_line)
            episode_count += 1
            episode_errors = validate_exported_episode(episode)
            if episode_errors:
                errors.append(
                    {
                        "file": str(episodes_path),
                        "line": episode_line_number,
                        "errors": episode_errors,
                    }
                )
            if metadata.get("episode_id") != episode.get("episode_id"):
                errors.append(
                    {
                        "file": str(metadata_path),
                        "line": episode_line_number,
                        "error": "metadata episode_id does not match episode stream",
                    }
                )
            collection_indices = [
                int(value) for value in metadata.get("collection_waypoint_indices", [])
            ]
            collection_policy = str(
                metadata.get("collection_stride_policy", "fixed_per_episode")
            )
            collection_gaps = np.diff(collection_indices).astype(int).tolist()
            terminal_index = len(episode.get("reference_path", [])) - 1
            if (
                not collection_indices
                or collection_indices[0] != 0
                or collection_indices[-1] != terminal_index
                or any(gap <= 0 for gap in collection_gaps)
            ):
                errors.append(
                    {
                        "file": str(metadata_path),
                        "line": episode_line_number,
                        "error": "collection indices must increase from zero through the real terminal",
                    }
                )
            if metadata.get("collection_waypoint_gaps", collection_gaps) != collection_gaps:
                errors.append(
                    {
                        "file": str(metadata_path),
                        "line": episode_line_number,
                        "error": "collection waypoint gaps do not match collection indices",
                    }
                )
            if collection_policy == "deterministic_random_per_interval":
                choices = normalize_image_stride_choices(
                    metadata.get("collection_stride_choices_waypoints", ())
                )
                if any(gap not in choices for gap in collection_gaps[:-1]) or (
                    collection_gaps and collection_gaps[-1] > max(choices)
                ):
                    errors.append(
                        {
                            "file": str(metadata_path),
                            "line": episode_line_number,
                            "error": "random interval schedule contains a gap outside its choices",
                        }
                    )
            expected_requests = int(metadata["render_request_count"])
            for _ in range(expected_requests):
                line = pending_request or render_stream.readline()
                pending_request = None
                render_line_number += 1
                if not line:
                    errors.append(
                        {
                            "file": str(render_path),
                            "line": render_line_number,
                            "error": "missing render request",
                        }
                    )
                    break
                request = json.loads(line)
                request_count += 1
                render_shape = (
                    int(request.get("expected_width", -1)),
                    int(request.get("expected_height", -1)),
                    int(request.get("expected_channels", -1)),
                )
                if render_shape != expected_render_shape:
                    errors.append(
                        {
                            "file": str(render_path),
                            "line": render_line_number,
                            "error": (
                                "render request shape must match package manifest "
                                f"{expected_render_shape[0]}x{expected_render_shape[1]}x"
                                f"{expected_render_shape[2]} output"
                            ),
                        }
                    )
                if request.get("episode_id") != episode.get("episode_id"):
                    errors.append(
                        {
                            "file": str(render_path),
                            "line": render_line_number,
                            "error": "render request episode order mismatch",
                        }
                    )
                    continue
                waypoint_index = int(request["waypoint_index"])
                image_index = int(request.get("image_index", -1))
                if request.get("collection_stride_policy", "fixed_per_episode") != collection_policy:
                    errors.append(
                        {
                            "file": str(render_path),
                            "line": render_line_number,
                            "error": "render request sampling policy differs from episode metadata",
                        }
                    )
                if collection_policy == "fixed_per_episode" and int(
                    request.get("collection_stride_waypoints", -1)
                ) != int(metadata["collection_stride_waypoints"]):
                    errors.append(
                        {
                            "file": str(render_path),
                            "line": render_line_number,
                            "error": "render request stride differs from episode metadata",
                        }
                    )
                if not 0 <= image_index < len(collection_indices) or waypoint_index != collection_indices[image_index]:
                    errors.append(
                        {
                            "file": str(render_path),
                            "line": render_line_number,
                            "error": "waypoint does not match the indexed collection schedule",
                        }
                    )
                    continue
                expected_gap = 0 if image_index == 0 else collection_gaps[image_index - 1]
                if int(request.get("collection_gap_waypoints", expected_gap)) != expected_gap:
                    errors.append(
                        {
                            "file": str(render_path),
                            "line": render_line_number,
                            "error": "render request gap differs from collection schedule",
                        }
                    )
                pose = episode["reference_path"][waypoint_index]
                if not np.allclose(request["pose_xyz_rpy"], pose, atol=1e-7):
                    errors.append(
                        {
                            "file": str(render_path),
                            "line": render_line_number,
                            "error": "render pose differs from reference_path",
                        }
                    )
                if (
                    request["request_id"] in seen_requests
                    or request["expected_image_relpath"] in seen_paths
                ):
                    errors.append(
                        {
                            "file": str(render_path),
                            "line": render_line_number,
                            "error": "duplicate request or image path",
                        }
                    )
                seen_requests.add(request["request_id"])
                seen_paths.add(request["expected_image_relpath"])
        extra_episode = episode_stream.readline()
        extra_metadata = metadata_stream.readline()
        extra_render = render_stream.readline()
        if extra_episode or extra_metadata or extra_render:
            errors.append({"error": "package streams have extra unmatched records"})
    return {
        "valid": episode_count > 0 and not errors,
        "episode_count": episode_count,
        "render_request_count": request_count,
        "errors": errors,
    }


def _source_episode_fields(metadata: EpisodeMetadata) -> tuple[str, int, str]:
    trajectory_id = metadata.trajectory_id or str(metadata.episode_index)
    instruction_text = metadata.tasks[0] if metadata.tasks else ""
    return trajectory_id, metadata.task_index, instruction_text


def _export_environment_metadata(
    source_split: Path,
    *,
    render_image_width: int = AERIALVLN_RENDER_WIDTH,
    render_image_height: int = AERIALVLN_RENDER_HEIGHT,
) -> tuple[list[dict], dict]:
    render_image_width, render_image_height = _validate_render_dimensions(
        render_image_width, render_image_height
    )
    info = json.loads((source_split / "meta" / "info.json").read_text(encoding="utf-8"))
    camera_config_path = source_split / "meta" / "navvla_cameras.json"
    camera_config = (
        json.loads(camera_config_path.read_text(encoding="utf-8"))
        if camera_config_path.is_file()
        else {}
    )
    cameras = []
    for feature_name, feature in info.get("features", {}).items():
        if not feature_name.startswith("observation.images."):
            continue
        camera_key = feature_name.removeprefix("observation.images.")
        shape = feature.get("shape", [])
        if len(shape) != 3:
            raise ValueError(f"camera feature has invalid shape: {feature_name}={shape}")
        camera_metadata = dict(next(
            (
                value
                for value in camera_config.values()
                if value.get("video_key") == camera_key
            ),
            {},
        ))
        camera_metadata["source_feature_shape"] = [int(value) for value in shape]
        cameras.append(
            {
                "camera_key": camera_key,
                "camera_name": str(camera_metadata.get("name", camera_key)),
                "height": render_image_height,
                "width": render_image_width,
                "channels": AERIALVLN_RENDER_CHANNELS,
                "camera_metadata": camera_metadata,
            }
        )
    if not cameras:
        raise ValueError("source split has no image camera feature")
    navvla = info.get("navvla", {})
    coordinate_metadata = {
        "state_mode": navvla.get("state_mode"),
        "state_order": navvla.get("state_order", ["x", "y", "z", "yaw"]),
        "coordinate_convention": navvla.get("coordinate_convention"),
        "coordinate_frame": navvla.get("coordinate_frame"),
        "coordinate_frame_id": navvla.get("coordinate_frame_id"),
    }
    return cameras, coordinate_metadata


def export_train_split(
    *,
    source_split: Path,
    output_dir: Path,
    dataset_key: str,
    sample_episode_indices: set[int] | None = None,
    include_episode_indices: set[int] | None = None,
    require_enhanced_sibling: bool = True,
    config: TrajectoryConfig | None = None,
    image_stride_choices: tuple[int, ...] = (1, 3, 5),
    image_stride_policy: str = "fixed-per-episode",
    image_interval_seed: int = 0,
    render_image_width: int = AERIALVLN_RENDER_WIDTH,
    render_image_height: int = AERIALVLN_RENDER_HEIGHT,
    retain_fraction: float | None = None,
    excluded_scene_ids: set[str] | None = None,
    include_scene_ids: set[str] | None = None,
    selection_seed: int = 0,
    balanced_image_strides: bool = False,
    eligible_episode_indices: set[int] | None = None,
    sample_per_stride: int = 0,
    world_pose_source: str = "auto",
    world_pose_metadata_path: Path | None = None,
    original_trajectory_json: Path | None = None,
    world_pose_adapter=None,
    world_pose_source_kind: str | None = None,
    coordinate_alignment_transform: str | None = None,
    render_coordinate_transform: str | None = None,
    render_pose_source: str | None = None,
) -> dict:
    """Enhance all valid frame trajectories and publish a trajectory-only package."""

    source = Path(source_split).resolve()
    output = Path(output_dir).resolve()
    if require_enhanced_sibling and (
        output.parent != source.parent or output.name != f"{source.name}_enhanced"
    ):
        raise ValueError("output must be the required enhanced sibling of the source split")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite trajectory package: {output}")
    export_staging = output.parent / f".{output.name}.export-{uuid.uuid4().hex}"
    trajectory_config = config or TrajectoryConfig()
    stride_choices = normalize_image_stride_choices(image_stride_choices)
    samples = set(sample_episode_indices or ())
    included = None if include_episode_indices is None else set(include_episode_indices)
    selection_plan = None
    if sample_per_stride < 0:
        raise ValueError("sample_per_stride must be non-negative")
    if retain_fraction is not None:
        if included is not None:
            raise ValueError(
                "retain_fraction cannot be combined with include_episode_indices"
            )
        if not balanced_image_strides:
            raise ValueError(
                "retain_fraction requires balanced_image_strides for exact groups"
            )
        source_metadata = read_episode_metadata(source)
        if eligible_episode_indices is not None:
            eligible = set(int(value) for value in eligible_episode_indices)
            source_metadata = [
                item for item in source_metadata if item.episode_index in eligible
            ]
            if len(source_metadata) != len(eligible):
                found = {item.episode_index for item in source_metadata}
                missing = sorted(eligible - found)
                raise ValueError(
                    f"eligible package references missing source episodes: {missing[:10]}"
                )
        included_scenes = {str(value) for value in (include_scene_ids or set())}
        if included_scenes:
            source_metadata = [
                item
                for item in source_metadata
                if str(item.scene_id) in included_scenes
            ]
            if not source_metadata:
                raise ValueError("include_scene_ids selects no source episodes")
        selection_plan = plan_lightweight_subset(
            source_metadata,
            retain_fraction=retain_fraction,
            excluded_scene_ids=excluded_scene_ids or set(),
            seed=selection_seed,
            stride_choices=stride_choices,
        )
        included = set(selection_plan.selected_episode_indices)
        if sample_per_stride:
            for stride in stride_choices:
                candidates = sorted(
                    index
                    for index, assigned in selection_plan.stride_by_episode_index.items()
                    if assigned == stride
                )
                samples.update(candidates[:sample_per_stride])
    failures = []
    metrics = []
    plot_payloads = []

    cameras, coordinate_metadata = _export_environment_metadata(
        source,
        render_image_width=render_image_width,
        render_image_height=render_image_height,
    )
    info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    frame_metadata_path = (
        Path(world_pose_metadata_path).resolve()
        if world_pose_metadata_path is not None
        else source / "meta" / "navvla_frame_metadata.jsonl"
    )
    original_json_path = (
        Path(original_trajectory_json).resolve()
        if original_trajectory_json is not None
        else source.parent / "aerialvln_json" / "train.json"
    )
    if world_pose_adapter is not None:
        if not world_pose_source_kind:
            raise ValueError("world_pose_source_kind is required for a custom adapter")
        if not callable(getattr(world_pose_adapter, "poses_for_episode", None)):
            raise TypeError("world pose adapter must define poses_for_episode(metadata)")
        resolved_pose_source = "adapter"
    else:
        resolved_pose_source = choose_world_pose_source(
            info,
            requested=world_pose_source,
            original_trajectory_json=original_json_path,
            frame_metadata_path=frame_metadata_path,
        )
    default_render_pose_source = (
        "original_aerialvln_reference_path_world_pose"
        if resolved_pose_source == "original-json"
        else (
            "original_openfly_annotation_world_pose"
            if resolved_pose_source == "openfly-annotation"
            else (
                "frame_source_metadata_world_pose"
                if resolved_pose_source == "frame-metadata"
                else (
                    world_pose_source_kind
                    if resolved_pose_source == "adapter"
                    else "frame_observation_state"
                )
            )
        )
    )
    alignment_transform = coordinate_alignment_transform or (
        "reflect-y-yaw"
        if resolved_pose_source == "openfly-annotation"
        else "identity"
    )
    render_transform = render_coordinate_transform or (
        "reflect-y-z-yaw"
        if resolved_pose_source == "openfly-annotation"
        else "identity"
    )
    coordinate_metadata.update(
        {
            "render_pose_mode": "source_world_absolute_pose_xyz_yaw",
            "render_pose_source": render_pose_source or default_render_pose_source,
            "training_state_mode": info.get("navvla", {}).get("state_mode"),
            "render_coordinate_transform": render_transform,
        }
    )

    original_pose_index = world_pose_adapter or (
        AerialVLNOriginalPoseIndex(original_json_path)
        if resolved_pose_source == "original-json"
        else (
            OpenFlyAnnotationPoseIndex(original_json_path)
            if resolved_pose_source == "openfly-annotation"
            else None
        )
    )
    pose_reader_context = (
        FrameMetadataWorldPoseStream(frame_metadata_path)
        if resolved_pose_source == "frame-metadata"
        else nullcontext(None)
    )

    def record_stream(pose_reader):
        for metadata, table in iter_episode_tables(source, episode_indices=included):
            try:
                if "observation.state" not in table.column_names:
                    raise ValueError("observation.state is missing")
                training_poses = np.asarray(
                    table.column("observation.state").to_pylist(), dtype=float
                )
                if training_poses.ndim != 2 or training_poses.shape[1] != 4:
                    raise ValueError(
                        f"expected frame observation.state shape [N, 4], got {training_poses.shape}"
                    )
                if original_pose_index is not None:
                    source_poses = original_pose_index.poses_for_episode(metadata)
                    source_pose_kind = (
                        world_pose_source_kind
                        if resolved_pose_source == "adapter"
                        else (
                            "original_openfly_annotation_world_pose"
                            if resolved_pose_source == "openfly-annotation"
                            else "original_aerialvln_reference_path_world_pose"
                        )
                    )
                elif pose_reader is None:
                    source_poses = training_poses
                    source_pose_kind = "frame_observation_state"
                else:
                    if "index" not in table.column_names:
                        raise ValueError(
                            "global frame index is required to join frame world-pose metadata"
                        )
                    source_poses = pose_reader.read_indices(
                        table.column("index").to_pylist()
                    )
                    source_pose_kind = "frame_source_metadata_world_pose"
                aligned_poses = transform_world_poses_for_alignment(
                    source_poses, alignment_transform
                )
                render_poses = transform_world_poses_for_alignment(
                    source_poses, render_transform
                )
                coordinate_alignment = (
                    validate_episode_local_alignment(
                        training_poses,
                        aligned_poses,
                    )
                    if source_pose_kind != "frame_observation_state"
                    else {
                        "max_position_error_m": 0.0,
                        "max_yaw_error_rad": 0.0,
                    }
                )
                trajectory = smooth_and_retime(
                    render_poses,
                    trajectory_config,
                    seed=stable_trajectory_seed(dataset_key, metadata.episode_index),
                )
                record = build_export_record(
                    dataset_key=dataset_key,
                    metadata=metadata,
                    trajectory=trajectory,
                    image_stride_choices=stride_choices,
                    image_stride=(
                        selection_plan.stride_by_episode_index[metadata.episode_index]
                        if selection_plan is not None
                        and image_stride_policy == "fixed-per-episode"
                        else None
                    ),
                    image_stride_policy=image_stride_policy,
                    image_interval_seed=image_interval_seed,
                    source_pose_kind=source_pose_kind,
                    training_state_mode=info.get("navvla", {}).get("state_mode"),
                    coordinate_alignment=coordinate_alignment,
                    coordinate_alignment_transform=alignment_transform,
                )
                collection_indices = record["metadata"]["collection_waypoint_indices"]
                stride = int(record["metadata"].get("collection_stride_waypoints", stride_choices[0]))
                episode_metrics = compute_trajectory_metrics(
                    trajectory,
                    image_stride=stride,
                    image_indices=collection_indices,
                )
                episode_metrics.update(
                    {
                        "episode_index": int(metadata.episode_index),
                        "episode_id": str(metadata.episode_id),
                        "scene_id": str(metadata.scene_id),
                    }
                )
                metrics.append(episode_metrics)
                if metadata.episode_index in samples:
                    plot_payloads.append(
                        (metadata, trajectory, stride, collection_indices)
                    )
                yield record
            except Exception as error:
                failures.append(
                    {
                        "episode_index": int(metadata.episode_index),
                        "episode_id": str(metadata.episode_id),
                        "scene_id": str(metadata.scene_id),
                        "reason": str(error),
                    }
                )

    try:
        with pose_reader_context as pose_reader:
            validation = write_trajectory_package(
                export_staging,
                dataset_key=dataset_key,
                source_split=source,
                records=record_stream(pose_reader),
                failures=failures,
                cameras=cameras,
                coordinate_metadata=coordinate_metadata,
                extra_manifest={
                "requested_sample_episode_indices": sorted(samples),
                "image_stride_policy": image_stride_policy,
                "image_interval_seed": int(image_interval_seed),
                "image_stride_choices": list(stride_choices),
                **(
                    {
                        "selection_policy": "deterministic_random_retention_after_scene_filter",
                        "selection_seed": selection_plan.seed,
                        "selection_retain_fraction": selection_plan.retain_fraction,
                        "selection_source_episode_count": selection_plan.source_episode_count,
                        "selection_target_episode_count": selection_plan.target_episode_count,
                        "selection_excluded_scene_ids": list(selection_plan.excluded_scene_ids),
                        "selection_excluded_scene_episode_count": selection_plan.excluded_scene_episode_count,
                        "selection_included_scene_ids": sorted(
                            str(value) for value in (include_scene_ids or set())
                        ),
                        "selection_stride_episode_counts": {
                            str(key): value
                            for key, value in selection_plan.stride_episode_counts.items()
                        },
                    }
                    if selection_plan is not None
                    else {}
                ),
                },
            )
        if (
            selection_plan is not None
            and validation["valid_episode_count"]
            != selection_plan.target_episode_count
        ):
            raise ValueError(
                "lightweight export did not preserve the requested episode count: "
                f"expected {selection_plan.target_episode_count}, "
                f"got {validation['valid_episode_count']}"
            )
        plot_dir = export_staging / "validation" / "samples"
        plot_dir.mkdir(parents=True)
        for metadata, trajectory, stride, collection_indices in plot_payloads:
            plot_choices = (
                stride_choices
                if image_stride_policy == "deterministic-random-per-interval"
                else None
            )
            plot_trajectory_comparison(
                trajectory,
                plot_dir / f"episode_{metadata.episode_index:06d}_comparison.png",
                title=(
                    f"{dataset_key} episode {metadata.episode_index} "
                    f"scene={metadata.scene_id}"
                ),
                image_stride=stride,
                image_indices=collection_indices,
                image_stride_choices=plot_choices,
            )
            plot_sampling_audit(
                trajectory,
                plot_dir / f"episode_{metadata.episode_index:06d}_sampling_audit.png",
                title=(
                    f"{dataset_key} episode {metadata.episode_index} "
                    f"scene={metadata.scene_id}"
                ),
                image_stride=stride,
                image_indices=collection_indices,
                image_stride_choices=plot_choices,
            )
        (export_staging / "trajectory_metrics.jsonl").write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                for item in metrics
            ),
            encoding="utf-8",
        )
        manifest_path = export_staging / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "metrics_file": "trajectory_metrics.jsonl",
                "validation_plot_dir": "validation/samples",
                "valid_episode_count": validation["valid_episode_count"],
                "failure_count": len(failures),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validation["failure_count"] = len(failures)
        package_validation = validate_trajectory_package(export_staging)
        if not package_validation["valid"]:
            raise ValueError("final trajectory package contract validation failed")
        validation.update(package_validation)
        validation_path = export_staging / "validation.json"
        validation_path.write_text(
            json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (export_staging / "validation" / "summary.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(export_staging, output)
        return validation
    except Exception:
        shutil.rmtree(export_staging, ignore_errors=True)
        raise
