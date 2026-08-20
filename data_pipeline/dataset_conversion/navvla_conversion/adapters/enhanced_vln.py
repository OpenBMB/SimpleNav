from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import uuid
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from navvla_conversion.adapters.base import NavVLASourceAdapter, register_adapter
from navvla_conversion.derived_artifacts import finalize_derived_artifacts
from navvla_conversion.statistics import body_frame_action_from_pose

VIEWS = ("front", "back", "left", "right")
VIDEO_KEYS = tuple(f"{view}_image" for view in VIEWS)
CAMERA_POSE_NAMES = ("x_world", "y_world", "z_world", "yaw_body", "roll_body", "pitch_body", "fov")
SUPPORTED_DATASETS = {
    "AerialVLN_lerobot": {
        "dataset_source": "aerialvln",
        "source_pose_kind": "original_aerialvln_reference_path_world_pose",
    },
    "OpenFly_lerobot": {
        "dataset_source": "openfly",
        "source_pose_kind": "original_openfly_annotation_world_pose",
    },
}
PLATFORM_TEXT = (
    "The platform is UAV for urban uav navigation. The control frequency is 1 Hz. "
    "Please predict the next 8 local 3D waypoints (dx, dy, dz, dyaw) to execute the following task:"
)
SPLIT = "vln_train"
FPS = 1.0
CONTROL_FREQUENCY_HZ = 1.0
ACTION_HORIZON = 8
EPISODES_PER_FILE = 20
FILES_PER_CHUNK = 50


def resolve_load_workers(load_workers: int | None) -> int:
    if load_workers is None:
        return 1
    workers = int(load_workers)
    if workers < 1:
        raise ValueError(f"load_workers must be >= 1, got {load_workers}")
    return workers


def _wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def rebase_report_paths(value: Any, *, staging: str | Path, target: str | Path) -> Any:
    staging_text = str(Path(staging))
    target_text = str(Path(target))
    if isinstance(value, dict):
        return {key: rebase_report_paths(item, staging=staging_text, target=target_text) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase_report_paths(item, staging=staging_text, target=target_text) for item in value]
    if isinstance(value, tuple):
        return tuple(rebase_report_paths(item, staging=staging_text, target=target_text) for item in value)
    if isinstance(value, str) and (value == staging_text or value.startswith(staging_text + os.sep)):
        return target_text + value[len(staging_text) :]
    return value


def render_poses_to_training_local(render_poses: Sequence[Sequence[float]], *, dataset_key: str) -> np.ndarray:
    poses = np.asarray(render_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 4 or len(poses) < 2:
        raise ValueError(f"enhanced render poses must have shape [N, 4], N >= 2, got {poses.shape}")
    if not np.all(np.isfinite(poses)):
        raise ValueError("enhanced render poses contain non-finite values")
    if dataset_key not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported enhanced dataset_key: {dataset_key}")

    absolute = poses.copy()

    yaw0 = float(absolute[0, 3])
    cosine = math.cos(yaw0)
    sine = math.sin(yaw0)
    delta_xy = absolute[:, :2] - absolute[0, :2]
    local = np.empty_like(absolute)
    local[:, 0] = cosine * delta_xy[:, 0] + sine * delta_xy[:, 1]
    local[:, 1] = -sine * delta_xy[:, 0] + cosine * delta_xy[:, 1]
    local[:, 2] = absolute[:, 2] - absolute[0, 2]
    local[:, 3] = _wrap_angle(absolute[:, 3] - yaw0)
    local[np.abs(local) < 1.0e-7] = 0.0
    return local.astype(np.float32)


def camera_pose7(*, vehicle_pose: Sequence[float], camera_parameters: dict[str, Any]) -> list[float]:
    vehicle = np.asarray(vehicle_pose, dtype=np.float64)
    if vehicle.shape != (4,) or not np.all(np.isfinite(vehicle)):
        raise ValueError(f"vehicle pose must contain finite [x, y, z, yaw], got {vehicle_pose!r}")
    final = camera_parameters.get("final_pose")
    if not isinstance(final, dict):
        raise ValueError("camera parameters must contain final_pose")
    try:
        mount = np.asarray([final["x"], final["y"], final["z"]], dtype=np.float64)
        yaw = math.radians(float(final["yaw"]))
        roll = math.radians(float(final["roll"]))
        pitch = math.radians(float(final["pitch"]))
        fov = math.radians(float(camera_parameters["fov_degrees"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid camera parameters: {camera_parameters!r}") from exc
    values = np.asarray([*mount, yaw, roll, pitch, fov], dtype=np.float64)
    if not np.all(np.isfinite(values)) or fov <= 0.0:
        raise ValueError(f"camera parameters must be finite with positive FOV: {camera_parameters!r}")

    cosine = math.cos(float(vehicle[3]))
    sine = math.sin(float(vehicle[3]))
    camera_x = float(vehicle[0]) + cosine * float(mount[0]) - sine * float(mount[1])
    camera_y = float(vehicle[1]) + sine * float(mount[0]) + cosine * float(mount[1])
    camera_z = float(vehicle[2]) + float(mount[2])
    result = [camera_x, camera_y, camera_z, yaw, roll, pitch, fov]
    return [0.0 if abs(value) < 1.0e-7 else float(value) for value in result]


def _action_chunk(local_poses: np.ndarray, *, waypoint_index: int) -> list[list[float]]:
    if waypoint_index < 0 or waypoint_index >= len(local_poses):
        raise IndexError(f"waypoint_index={waypoint_index} outside trajectory length={len(local_poses)}")
    current = local_poses[waypoint_index]
    result = []
    terminal = len(local_poses) - 1
    for offset in range(1, ACTION_HORIZON + 1):
        future = local_poses[min(waypoint_index + offset, terminal)]
        action = body_frame_action_from_pose(current, future).astype(np.float64)
        action[np.abs(action) < 1.0e-7] = 0.0
        result.append(action.tolist())
    return result


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield payload


def _group_jsonl(path: Path, key: str) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    for value, rows in groupby(_iter_jsonl(path), key=lambda row: str(row.get(key))):
        yield value, list(rows)


def _required_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_manifest(source: Path) -> dict[str, Any]:
    manifest = json.loads(_required_file(source, "manifest.json").read_text(encoding="utf-8"))
    dataset_key = str(manifest.get("dataset_key") or "")
    if dataset_key not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported enhanced manifest dataset_key={dataset_key!r}")
    expected_pose_kind = SUPPORTED_DATASETS[dataset_key]["source_pose_kind"]
    if manifest.get("source_pose_kind") != expected_pose_kind:
        raise ValueError(
            f"enhanced manifest source_pose_kind={manifest.get('source_pose_kind')!r}, expected {expected_pose_kind!r}"
        )
    if manifest.get("trajectory_format") != "aerialvln_episode_shell_absolute_pose_sequence":
        raise ValueError(f"unsupported enhanced trajectory_format={manifest.get('trajectory_format')!r}")
    if manifest.get("image_status") != "collected" or int(manifest.get("missing_render_request_count", 0)) != 0:
        raise ValueError("enhanced package must have all rendered images collected")
    if tuple(manifest.get("views") or ()) != VIEWS:
        raise ValueError(f"enhanced package views must be {list(VIEWS)}, got {manifest.get('views')!r}")
    return manifest


def _reference_poses(episode: dict[str, Any]) -> np.ndarray:
    reference = np.asarray(episode.get("reference_path"), dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] < 6:
        raise ValueError(
            f"enhanced episode {episode.get('episode_id')} reference_path must have shape [N, >=6], "
            f"got {reference.shape}"
        )
    poses = reference[:, [0, 1, 2, 5]]
    if len(poses) < 2 or not np.all(np.isfinite(poses)):
        raise ValueError(f"enhanced episode {episode.get('episode_id')} contains invalid reference poses")
    return poses


def _instruction(episode: dict[str, Any]) -> str:
    instruction = episode.get("instruction")
    value = instruction.get("instruction_text") if isinstance(instruction, dict) else instruction
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"enhanced episode {episode.get('episode_id')} has no instruction")
    return text


def _task_subtype(dataset_key: str, trajectory_id: str) -> str:
    if dataset_key == "AerialVLN_lerobot":
        return "aerialvln"
    parts = trajectory_id.split("/")
    if len(parts) >= 3 and parts[1] == "astar_data":
        return parts[2]
    return "openfly"


def _data_schema() -> pa.Schema:
    fields = [
        pa.field("episode_index", pa.int64()),
        pa.field("frame_index", pa.int64()),
        pa.field("timestamp", pa.float64()),
        pa.field("task_index", pa.int64()),
        pa.field("observation.state", pa.list_(pa.float32(), list_size=4)),
        pa.field("action", pa.list_(pa.list_(pa.float64()))),
        pa.field("action.padding_mask", pa.list_(pa.bool_())),
        pa.field("next.done", pa.bool_()),
        pa.field("sample.action_available", pa.bool_()),
        pa.field("context.index_key", pa.string()),
        pa.field("source_frame_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("sample.state_available", pa.bool_()),
    ]
    fields.extend(pa.field(f"observation.camera_pose.{view}", pa.list_(pa.float32(), list_size=7)) for view in VIEWS)
    return pa.schema(fields, metadata={b"navvla.stored_pose_schema_version": b"episode-relative-camera-pose-v3-fov"})


def _write_data_shard(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=_data_schema()), path, compression="zstd")


def _process_episode_job(job: dict[str, Any]) -> dict[str, Any]:
    dataset_key = str(job["dataset_key"])
    dataset_name = str(job["dataset_name"])
    source = str(job["source"])
    source_pose_kind = job["source_pose_kind"]
    episode_index = int(job["episode_index"])
    first_data_index = int(job["first_data_index"])
    episode = job["episode"]
    frame_rows = job["frame_rows"]
    camera_rows = job["camera_rows"]

    episode_id = str(episode.get("episode_id") or "")
    trajectory_id = str(episode.get("trajectory_id") or "")
    scene_id = str(episode.get("scene_id") or "")
    if not episode_id or not trajectory_id or not scene_id:
        raise ValueError(f"enhanced episode is missing identity fields: {episode!r}")
    if str(job["frame_episode_id"]) != episode_id or str(job["camera_episode_id"]) != episode_id:
        raise ValueError(
            f"enhanced episode order mismatch: episode={episode_id} frames={job['frame_episode_id']} "
            f"cameras={job['camera_episode_id']}"
        )
    if len(camera_rows) != len(VIEWS):
        raise ValueError(f"enhanced episode {episode_id} must have four camera parameter rows")
    cameras_by_view = {str(row.get("view")): row for row in camera_rows}
    if tuple(cameras_by_view) != VIEWS:
        raise ValueError(f"enhanced episode {episode_id} camera order/views mismatch: {list(cameras_by_view)}")

    render_poses = _reference_poses(episode)
    local_poses = render_poses_to_training_local(render_poses, dataset_key=dataset_key)
    instruction = _instruction(episode)
    data_rows: list[dict[str, Any]] = []
    frame_metadata_rows: list[dict[str, Any]] = []
    for frame_position, metadata in enumerate(frame_rows):
        image_index = int(metadata.get("image_index", -1))
        waypoint_index = int(metadata.get("waypoint_index", -1))
        source_index = int(metadata.get("index", -1))
        timestamp = float(metadata.get("timestamp", math.nan))
        global_index = first_data_index + frame_position
        if image_index != frame_position:
            raise ValueError(
                f"enhanced episode {episode_id} image_index={image_index} is not contiguous at position={frame_position}"
            )
        if source_index != global_index:
            raise ValueError(
                f"enhanced global index must be contiguous: episode={episode_id} got={source_index} "
                f"expected={global_index}"
            )
        if waypoint_index < 0 or waypoint_index >= len(local_poses):
            raise ValueError(
                f"enhanced episode {episode_id} waypoint_index={waypoint_index} outside trajectory "
                f"length={len(local_poses)}"
            )
        expected_timestamp = float(waypoint_index) / CONTROL_FREQUENCY_HZ
        if not math.isfinite(timestamp) or abs(timestamp - expected_timestamp) > 1.0e-6:
            raise ValueError(
                f"enhanced episode {episode_id} timestamp={timestamp} does not match 1 Hz "
                f"waypoint_index={waypoint_index}"
            )
        row = {
            "episode_index": episode_index,
            "frame_index": frame_position,
            "timestamp": timestamp,
            "task_index": episode_index,
            "observation.state": local_poses[waypoint_index].tolist(),
            "action": _action_chunk(local_poses, waypoint_index=waypoint_index),
            "action.padding_mask": [False] * ACTION_HORIZON,
            "next.done": frame_position == len(frame_rows) - 1,
            "sample.action_available": True,
            "context.index_key": f"{dataset_name}/{SPLIT}/{episode_id}/f{frame_position:06d}/bats-v1",
            "source_frame_index": waypoint_index,
            "index": global_index,
            "sample.state_available": True,
        }
        for view in VIEWS:
            row[f"observation.camera_pose.{view}"] = camera_pose7(
                vehicle_pose=local_poses[waypoint_index],
                camera_parameters=cameras_by_view[view],
            )
        data_rows.append(row)
        frame_metadata_rows.append(
            {
                "index": global_index,
                "source_frame_index": waypoint_index,
                "source_metadata": {
                    "source_dataset": dataset_key,
                    "source_package": source,
                    "source_episode_id": str(metadata.get("episode_id")),
                    "source_episode_index": int(metadata.get("source_episode_index", episode_index)),
                    "scene_id": scene_id,
                    "trajectory_id": trajectory_id,
                    "image_index": image_index,
                    "waypoint_index": waypoint_index,
                    "request_id": str(metadata.get("request_id") or ""),
                    "render_pose_source": source_pose_kind,
                    "render_to_training_transform": "identity",
                },
            }
        )

    dataset_source = str(SUPPORTED_DATASETS[dataset_key]["dataset_source"])
    task_subtype = _task_subtype(dataset_key, trajectory_id)
    shard = _shard_for_episode(episode_index)
    return {
        "episode_index": episode_index,
        "worker_pid": os.getpid(),
        "shard": shard,
        "data_rows": data_rows,
        "frame_metadata_rows": frame_metadata_rows,
        "multiview_rows": frame_rows,
        "camera_rows": camera_rows,
        "task_row": {
            "task_index": episode_index,
            "task": instruction,
            "task_type": "navigation",
            "task_subtype": task_subtype,
            "platform_text": PLATFORM_TEXT,
            "dataset_source": dataset_source,
            "answer": None,
        },
        "task_sidecar_row": {
            "task_index": episode_index,
            "task_type": "navigation",
            "task_subtype": task_subtype,
            "platform_text": PLATFORM_TEXT,
            "dataset_source": dataset_source,
            "answer": None,
            "source_enhanced_dataset_key": dataset_key,
        },
        "episode_row": {
            "episode_index": episode_index,
            "episode_id": episode_id,
            "trajectory_id": trajectory_id,
            "task_index": episode_index,
            "split": SPLIT,
            "scene_id": scene_id,
            "tasks": [instruction],
            "length": len(frame_rows),
            "data/chunk_index": shard[0],
            "data/file_index": shard[1],
            "first_data_index": first_data_index,
        },
    }


def _iter_processed_episode_jobs(jobs: Iterator[dict[str, Any]], *, workers: int) -> Iterator[dict[str, Any]]:
    if workers == 1:
        for job in jobs:
            yield _process_episode_job(job)
        return

    executor = ProcessPoolExecutor(max_workers=workers)
    pending = deque()
    try:
        for job in jobs:
            pending.append(executor.submit(_process_episode_job, job))
            if len(pending) >= workers * 4:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _shard_for_episode(episode_index: int) -> tuple[int, int]:
    linear_file_index = episode_index // EPISODES_PER_FILE
    return linear_file_index // FILES_PER_CHUNK, linear_file_index % FILES_PER_CHUNK


def _write_video_index_subset(
    source: Path,
    output: Path,
    *,
    total_frames: int,
    full_dataset: bool,
) -> dict[tuple[str, int, int], int | None]:
    source_path = _required_file(source, "meta/navvla_video_index.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    referenced: dict[tuple[str, int, int], int | None] = {}
    if full_dataset:
        shutil.copy2(source_path, output)
        for path in sorted((source / "videos").glob("*_image/chunk-*/*.mp4")):
            relative = path.relative_to(source / "videos")
            video_key = relative.parts[0]
            chunk_index = int(relative.parts[1].split("-")[1])
            file_index = int(relative.stem.split("-")[1])
            referenced[(video_key, chunk_index, file_index)] = None
        return referenced

    parquet = pq.ParquetFile(source_path)
    writer: pq.ParquetWriter | None = None
    try:
        for batch in parquet.iter_batches(batch_size=131072):
            table = pa.Table.from_batches([batch])
            filtered = table.filter(pc.less(table["index"], pa.scalar(total_frames, type=pa.int64())))
            if len(filtered):
                if writer is None:
                    writer = pq.ParquetWriter(output, filtered.schema, compression="zstd")
                writer.write_table(filtered)
                for row in filtered.select(["video_key", "chunk_index", "file_index", "video_frame_index"]).to_pylist():
                    key = (str(row["video_key"]), int(row["chunk_index"]), int(row["file_index"]))
                    selected_count = int(row["video_frame_index"]) + 1
                    referenced[key] = max(int(referenced.get(key) or 0), selected_count)
            if len(filtered) < len(table):
                break
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("selected enhanced episodes produced an empty video index")
    return referenced


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"video does not open: {path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if count <= 0:
        raise ValueError(f"video has no decodable frames: {path}")
    return count


def copy_video_prefix(
    source: str | Path,
    target: str | Path,
    *,
    selected_frame_count: int,
    source_frame_count: int,
    fps: float = FPS,
) -> None:
    source_path = Path(source)
    target_path = Path(target)
    selected = int(selected_frame_count)
    available = int(source_frame_count)
    if selected <= 0 or selected > available:
        raise ValueError(f"selected video prefix must be within 1..{available}, got {selected}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if selected == available:
        shutil.copy2(source_path, target_path)
        return
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-frames:v",
        str(selected),
        "-an",
        "-r",
        str(float(fps)),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(target_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to trim partial enhanced video shards") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        raise RuntimeError(f"failed to trim {source_path} to {selected} frames: {message}") from exc
    actual = _video_frame_count(target_path)
    if actual != selected:
        raise ValueError(f"trimmed video frame count mismatch for {target_path}: {actual} != {selected}")


def _copy_video_job(job: tuple[str, str, str, int | None]) -> None:
    source_root, output_root, relative_text, selected_count = job
    relative = Path(relative_text)
    source_path = _required_file(Path(source_root), relative_text)
    output_path = Path(output_root) / relative
    if selected_count is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)
    else:
        copy_video_prefix(
            source_path,
            output_path,
            selected_frame_count=selected_count,
            source_frame_count=_video_frame_count(source_path),
            fps=FPS,
        )


def _copy_videos(
    source: Path,
    output: Path,
    referenced: dict[tuple[str, int, int], int | None],
    *,
    workers: int = 1,
) -> int:
    jobs = []
    for (video_key, chunk_index, file_index), selected_count in sorted(referenced.items()):
        relative = Path("videos") / video_key / f"chunk-{chunk_index:03d}" / f"part-{file_index:03d}.mp4"
        jobs.append((str(source), str(output), str(relative), selected_count))
    resolved_workers = min(resolve_load_workers(workers), 4, len(jobs)) if jobs else 1
    if resolved_workers == 1:
        for job in jobs:
            _copy_video_job(job)
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            list(executor.map(_copy_video_job, jobs, chunksize=1))
    return len(jobs)


def _episode_video_offsets(video_index_path: Path, first_indices: set[int]) -> dict[tuple[int, str], float]:
    offsets: dict[tuple[int, str], float] = {}
    parquet = pq.ParquetFile(video_index_path)
    for batch in parquet.iter_batches(
        batch_size=131072,
        columns=["index", "video_key", "available", "video_frame_index"],
    ):
        for row in pa.Table.from_batches([batch]).to_pylist():
            index = int(row["index"])
            if index in first_indices and bool(row["available"]):
                offsets[(index, str(row["video_key"]))] = float(row["video_frame_index"]) / FPS
        if len(offsets) == len(first_indices) * len(VIDEO_KEYS):
            break
    missing = [(index, key) for index in sorted(first_indices) for key in VIDEO_KEYS if (index, key) not in offsets]
    if missing:
        raise ValueError(f"video index is missing episode starts: {missing[:10]}")
    return offsets


def _write_info(root: Path, *, dataset_name: str, total_episodes: int, total_frames: int, total_videos: int) -> None:
    features: dict[str, Any] = {}
    video_path = {}
    for view, video_key in zip(VIEWS, VIDEO_KEYS, strict=True):
        features[f"observation.images.{video_key}"] = {
            "dtype": "video",
            "shape": [224, 224, 3],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": FPS, "video.height": 224, "video.width": 224, "video.channels": 3},
        }
        video_path[video_key] = f"videos/{video_key}/chunk-{{chunk_index:03d}}/part-{{file_index:03d}}.mp4"
        features[f"observation.camera_pose.{view}"] = {
            "dtype": "float32",
            "shape": [7],
            "names": list(CAMERA_POSE_NAMES),
        }
    features.update(
        {
            "observation.state": {"dtype": "float32", "shape": [4], "names": ["x", "y", "z", "yaw"]},
            "action": {"dtype": "float32", "shape": [32], "names": ["action"]},
            "action.padding_mask": {"dtype": "bool", "shape": [8], "names": ["horizon"]},
            "timestamp": {"dtype": "float64", "shape": [1], "names": ["timestamp"]},
            "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
            "episode_index": {"dtype": "int64", "shape": [1], "names": ["episode_index"]},
            "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
            "source_frame_index": {"dtype": "int64", "shape": [1], "names": ["source_frame_index"]},
            "index": {"dtype": "int64", "shape": [1], "names": ["index"]},
            "next.done": {"dtype": "bool", "shape": [1], "names": ["done"]},
            "sample.action_available": {"dtype": "bool", "shape": [1], "names": ["action_available"]},
            "sample.state_available": {"dtype": "bool", "shape": [1], "names": ["state_available"]},
            "context.index_key": {"dtype": "string", "shape": [1], "names": ["context_index_key"]},
        }
    )
    info = {
        "codebase_version": "v3.0",
        "dataset_name": dataset_name,
        "robot_type": "navvla_navigation",
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(total_episodes),
        "total_videos": int(total_videos),
        "chunks_size": FILES_PER_CHUNK,
        "fps": FPS,
        "splits": {SPLIT: f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/part-{file_index:03d}.parquet",
        "video_path": video_path,
        "features": features,
        "navvla": {
            "schema_version": "0.1",
            "action_horizon": ACTION_HORIZON,
            "action_dim": 4,
            "control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "action_horizon_seconds": 8.0,
            "episodes_per_file": EPISODES_PER_FILE,
            "files_per_chunk": FILES_PER_CHUNK,
            "state_dim": 4,
            "state_mode": "episode_relative_first_body_aligned_pose_xyz_yaw",
            "state_order": ["x", "y", "z", "yaw"],
            "coordinate_convention": "x_forward_y_right_z_down_yaw_right_positive",
            "action_mode": "anchor_relative_body_frame_xyz_yaw",
            "action_anchor": "current_frame_pose",
            "timestamp_policy": "enhanced_image_waypoint_index_over_1hz",
            "tail_action_policy": "repeat_last_valid_waypoint_to_horizon; empty_terminal_chunk_stays_zero",
            "action_padding_mask_policy": "all_false_repeat_last_tail_unmasked",
            "stored_pose_schema_version": "episode-relative-camera-pose-v3-fov",
            "stored_pose_available": True,
            "stored_observation_state_mode": "episode_relative_first_body_aligned_pose_xyz_yaw",
            "stored_coordinate_frame": "first_frame_body_aligned_x_forward_y_right_z_down_yaw_right_positive",
            "stored_episode_world_origin": "discarded",
            "camera_pose_dim": 7,
            "camera_pose_position_frame": "episode_relative_first_body_aligned",
            "camera_pose_rotation_frame": "body_mount",
            "camera_pose_angle_unit": "radian",
            "camera_pose_fov_unit": "radian",
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def _write_modality(root: Path) -> None:
    payload = {
        "video": {video_key: {"original_key": f"observation.images.{video_key}"} for video_key in VIDEO_KEYS},
        "state": {
            name: {
                "start": index,
                "end": index + 1,
                "absolute": True,
                "dtype": "float32",
                "original_key": "observation.state",
            }
            for index, name in enumerate(("x", "y", "z", "yaw"))
        },
        "action": {
            name: {"start": index, "end": index + 1, "absolute": False, "dtype": "float32", "original_key": "action"}
            for index, name in enumerate(("dx", "dy", "dz", "dyaw"))
        },
        "annotation": {"language.language_instruction": {"original_key": "task_index"}},
        "camera_pose": {
            view: {
                "original_key": f"observation.camera_pose.{view}",
                "position_frame": "episode_relative_first_body_aligned",
                "rotation_frame": "body_mount",
                "angle_unit": "radian",
                "fov_unit": "radian",
                "dtype": "float32",
            }
            for view in VIEWS
        },
    }
    (root / "meta" / "modality.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_schema_ext(root: Path) -> None:
    payload = {
        "schema_version": "0.1",
        "context_policy_version": "bats-v1",
        "cache_policy_version": "smoke-coarse-v1",
        "frame_metadata": "meta/navvla_frame_metadata.jsonl",
        "video_index": "meta/navvla_video_index.parquet",
        "context_index_manifest": "meta/navvla_context_index_manifest.json",
        "context_index": "meta/context_index/budget_<budget>",
        "context_meta": "meta/context_index/budget_<budget>/context_meta.parquet",
        "context_arrays": "meta/context_index/budget_<budget>/context_arrays",
        "context_debug": f"cache/context_index_debug/budget_<budget>/{SPLIT}.parquet",
    }
    (root / "meta" / "navvla_schema_ext.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_cameras(root: Path, source_camera_payload: dict[str, Any]) -> None:
    source_rows = source_camera_payload.get("cameras")
    if not isinstance(source_rows, list):
        raise ValueError("enhanced navvla_cameras.json must contain a cameras list")
    cameras: dict[str, Any] = {}
    for row in source_rows:
        view = str(row.get("view") or "")
        if view not in VIEWS:
            raise ValueError(f"unsupported enhanced camera view: {view!r}")
        base = row.get("base_pose") or {}
        cameras[view] = {
            "name": view,
            "video_key": str(row.get("video_key") or f"{view}_image"),
            "viewpoint_type": view,
            "azimuth_rad": math.radians(float(base.get("yaw", 0.0))),
            "intrinsics": None,
            "extrinsics_body": None,
            "calibration_status": "episode_randomized",
            "yaw_body_rad": math.radians(float(base.get("yaw", 0.0))),
            "roll_body_rad": math.radians(float(base.get("roll", 0.0))),
            "pitch_body_rad": math.radians(float(base.get("pitch", 0.0))),
            "body_attitude_for_translation": "yaw_only",
            "episode_parameters_file": "meta/navvla_episode_camera_parameters.jsonl",
        }
    if tuple(cameras) != VIEWS:
        raise ValueError(f"enhanced camera order must be {list(VIEWS)}, got {list(cameras)}")
    (root / "meta" / "navvla_cameras.json").write_text(json.dumps(cameras, indent=2) + "\n", encoding="utf-8")


def convert_enhanced_vln_package(
    source_root: str | Path,
    *,
    output_root: str | Path,
    dataset_name: str,
    max_episodes: int | None = None,
    build_derived_artifacts: bool = True,
    load_workers: int | None = None,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    output_parent = Path(output_root).resolve()
    target = output_parent / dataset_name
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(target)
    if target == source or target.is_relative_to(source):
        raise ValueError(f"enhanced output must be outside the source package: source={source} target={target}")
    if max_episodes is not None and int(max_episodes) <= 0:
        raise ValueError(f"max_episodes must be positive, got {max_episodes}")

    manifest = _validate_manifest(source)
    dataset_key = str(manifest["dataset_key"])
    source_camera_payload = json.loads(_required_file(source, "meta/navvla_cameras.json").read_text(encoding="utf-8"))
    episodes_path = _required_file(source, "trajectories/episodes.jsonl")
    frames_path = _required_file(source, "meta/navvla_multiview_frame_metadata.jsonl")
    camera_parameters_path = _required_file(source, "meta/navvla_episode_camera_parameters.jsonl")

    output_parent.mkdir(parents=True, exist_ok=True)
    staging = output_parent / f".{dataset_name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    (staging / "data").mkdir()
    (staging / "meta" / "episodes").mkdir(parents=True)

    episode_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    task_sidecar_rows: list[dict[str, Any]] = []
    total_frames = 0
    selected_episodes = 0
    pending_data_rows: list[dict[str, Any]] = []
    pending_shard: tuple[int, int] | None = None
    workers = resolve_load_workers(load_workers)
    episode_worker_pids: set[int] = set()
    base_conversion_complete = False

    try:

        def episode_jobs() -> Iterator[dict[str, Any]]:
            frame_groups = _group_jsonl(frames_path, "episode_id")
            camera_groups = _group_jsonl(camera_parameters_path, "episode_id")
            first_data_index = 0
            for episode_index, episode in enumerate(_iter_jsonl(episodes_path)):
                if max_episodes is not None and episode_index >= int(max_episodes):
                    break
                try:
                    frame_episode_id, frame_rows = next(frame_groups)
                    camera_episode_id, camera_rows = next(camera_groups)
                except StopIteration as exc:
                    raise ValueError("enhanced metadata ended before trajectories/episodes.jsonl") from exc
                yield {
                    "dataset_key": dataset_key,
                    "dataset_name": dataset_name,
                    "source": str(source),
                    "source_pose_kind": manifest.get("source_pose_kind"),
                    "episode_index": episode_index,
                    "first_data_index": first_data_index,
                    "episode": episode,
                    "frame_episode_id": frame_episode_id,
                    "frame_rows": frame_rows,
                    "camera_episode_id": camera_episode_id,
                    "camera_rows": camera_rows,
                }
                first_data_index += len(frame_rows)

        with (
            (staging / "meta" / "navvla_frame_metadata.jsonl").open("w", encoding="utf-8") as frame_metadata_out,
            (staging / "meta" / "navvla_multiview_frame_metadata.jsonl").open("w", encoding="utf-8") as multiview_out,
            (staging / "meta" / "navvla_episode_camera_parameters.jsonl").open(
                "w", encoding="utf-8"
            ) as camera_parameters_out,
        ):
            for processed in _iter_processed_episode_jobs(episode_jobs(), workers=workers):
                shard = tuple(processed["shard"])
                if pending_shard is None:
                    pending_shard = shard
                elif shard != pending_shard:
                    _write_data_shard(
                        staging / "data" / f"chunk-{pending_shard[0]:03d}" / f"part-{pending_shard[1]:03d}.parquet",
                        pending_data_rows,
                    )
                    pending_data_rows = []
                    pending_shard = shard
                pending_data_rows.extend(processed["data_rows"])
                for record in processed["frame_metadata_rows"]:
                    frame_metadata_out.write(json.dumps(record, separators=(",", ":")) + "\n")
                for metadata in processed["multiview_rows"]:
                    multiview_out.write(json.dumps(metadata, separators=(",", ":")) + "\n")
                for camera_row in processed["camera_rows"]:
                    camera_parameters_out.write(json.dumps(camera_row, separators=(",", ":")) + "\n")
                task_rows.append(processed["task_row"])
                task_sidecar_rows.append(processed["task_sidecar_row"])
                episode_rows.append(processed["episode_row"])
                total_frames += len(processed["data_rows"])
                selected_episodes += 1
                episode_worker_pids.add(int(processed["worker_pid"]))

        if pending_shard is not None:
            _write_data_shard(
                staging / "data" / f"chunk-{pending_shard[0]:03d}" / f"part-{pending_shard[1]:03d}.parquet",
                pending_data_rows,
            )
        if selected_episodes == 0 or total_frames == 0:
            raise ValueError("enhanced selection produced no episodes or frames")

        full_dataset = max_episodes is None and selected_episodes == int(
            manifest.get("episode_count", selected_episodes)
        )
        video_index_path = staging / "meta" / "navvla_video_index.parquet"
        referenced = _write_video_index_subset(
            source,
            video_index_path,
            total_frames=total_frames,
            full_dataset=full_dataset,
        )
        video_copy_workers = min(workers, 4, len(referenced)) if referenced else 1
        copied_videos = _copy_videos(source, staging, referenced, workers=workers)
        offsets = _episode_video_offsets(video_index_path, {int(row["first_data_index"]) for row in episode_rows})
        for row in episode_rows:
            first_index = int(row.pop("first_data_index"))
            for video_key in VIDEO_KEYS:
                row[f"videos/{video_key}/from_timestamp"] = offsets[(first_index, video_key)]

        for chunk_file in sorted({_shard_for_episode(int(row["episode_index"])) for row in episode_rows}):
            chunk_index, file_index = chunk_file
            rows = [
                row
                for row in episode_rows
                if int(row["data/chunk_index"]) == chunk_index and int(row["data/file_index"]) == file_index
            ]
            path = staging / "meta" / "episodes" / f"chunk-{chunk_index:03d}" / f"part-{file_index:03d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(path, index=False)
        pd.DataFrame(task_rows).to_parquet(staging / "meta" / "tasks.parquet", index=False)
        (staging / "meta" / "navvla_tasks.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in task_sidecar_rows),
            encoding="utf-8",
        )

        _write_cameras(staging, source_camera_payload)
        _write_info(
            staging,
            dataset_name=dataset_name,
            total_episodes=selected_episodes,
            total_frames=total_frames,
            total_videos=copied_videos,
        )
        _write_modality(staging)
        _write_schema_ext(staging)
        (staging / "meta" / "source_enhanced_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        report = {
            "source_root": str(source),
            "dataset_root": str(target),
            "dataset_key": dataset_key,
            "total_episodes": selected_episodes,
            "total_frames": total_frames,
            "copied_videos": copied_videos,
            "fps": FPS,
            "control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "image_interval_seconds": 5.0,
            "camera_pose_dim": 7,
            "visual_token_cache": "not_generated",
            "load_workers": workers,
            "episode_worker_pids": sorted(episode_worker_pids),
            "video_copy_workers": video_copy_workers,
        }
        (staging / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        base_conversion_complete = True

        if build_derived_artifacts:
            return finalize_enhanced_vln_staging(staging)

        os.replace(staging, target)
        return report
    except BaseException:
        if not base_conversion_complete:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def finalize_enhanced_vln_staging(
    staging_root: str | Path,
) -> dict[str, Any]:
    staging = Path(staging_root).resolve()
    if not staging.is_dir():
        raise FileNotFoundError(staging)
    report_path = staging / "conversion_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    target = Path(str(report.get("dataset_root") or "")).resolve()
    if target == staging:
        raise ValueError(f"staging report must point to a distinct final target: {target}")
    if target.exists():
        raise FileExistsError(target)
    required = [
        staging / "meta" / "info.json",
        staging / "meta" / "tasks.parquet",
        staging / "meta" / "navvla_video_index.parquet",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"enhanced staging is incomplete: {missing}")
    report["derived_artifacts"] = finalize_derived_artifacts(
        staging,
        apply=True,
        token_budgets=(1024,),
        budget_num_cameras=len(VIEWS),
        history_camera_names=VIEWS,
    )
    report = rebase_report_paths(report, staging=staging, target=target)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(staging, target)
    return report


class EnhancedVLNAdapter(NavVLASourceAdapter):
    name = "enhanced_vln"

    def __init__(self, *, fps: float = FPS, action_horizon: int = ACTION_HORIZON) -> None:
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.load_workers = 1

    def configure(
        self,
        *,
        fps: float = FPS,
        action_horizon: int = ACTION_HORIZON,
        load_workers: int | None = None,
        **kwargs: Any,
    ) -> "EnhancedVLNAdapter":
        super().configure(**kwargs)
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.load_workers = resolve_load_workers(load_workers)
        if self.fps != FPS:
            raise ValueError(f"enhanced_vln requires video fps={FPS}, got {self.fps}")
        if self.action_horizon != ACTION_HORIZON:
            raise ValueError(f"enhanced_vln requires action_horizon={ACTION_HORIZON}, got {self.action_horizon}")
        return self

    def load_episodes(
        self,
        source_root: str | Path,
        *,
        split: str = "train",
        max_episodes: int | None = None,
    ) -> list[Any]:
        raise NotImplementedError("enhanced_vln uses a streaming prepacked-video converter; call convert()")

    def convert(
        self,
        *,
        source_root: str | Path,
        output_root: str | Path,
        dataset_name: str,
        max_episodes: int | None,
        fps: float,
        action_horizon: int,
        overwrite: bool,
        control_frequency_hz: float | None = None,
        repair_existing: bool = False,
        split: str = "train",
        write_workers: int | None = None,
        write_visual_token_cache: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del write_workers, kwargs, write_visual_token_cache
        if overwrite or repair_existing:
            raise ValueError("enhanced_vln never overwrites or repairs an existing target; choose a new sibling output")
        if float(fps) != FPS:
            raise ValueError(f"enhanced_vln requires fps={FPS}, got {fps}")
        if int(action_horizon) != ACTION_HORIZON:
            raise ValueError(f"enhanced_vln requires action_horizon={ACTION_HORIZON}, got {action_horizon}")
        if control_frequency_hz is not None and float(control_frequency_hz) != CONTROL_FREQUENCY_HZ:
            raise ValueError(
                f"enhanced_vln requires control_frequency_hz={CONTROL_FREQUENCY_HZ}, got {control_frequency_hz}"
            )
        if split not in {"train", SPLIT}:
            raise ValueError(f"enhanced_vln only supports the train split, got {split!r}")
        return convert_enhanced_vln_package(
            source_root,
            output_root=output_root,
            dataset_name=dataset_name,
            max_episodes=max_episodes,
            build_derived_artifacts=True,
            load_workers=self.load_workers,
        )


register_adapter(EnhancedVLNAdapter())
