from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from navvla_conversion.action import pad_action_chunk, resolve_timestamp
from navvla_conversion.context_index import ContextIndexConfig, build_context_indexes
from navvla_conversion.schema import NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec
from navvla_conversion.workers import resolve_workers as resolve_write_workers
from navvla_conversion.statistics import (
    build_dataset_statistics,
    flatten_valid_action_steps_from_rows,
    write_dataset_statistics,
)
DATA_PATH_PATTERN = "data/chunk-{chunk_index:03d}/part-{file_index:03d}.parquet"


class StreamingVideoWriter:
    def __init__(self, path: Path, *, fps: float) -> None:
        self.path = path
        self.fps = float(fps)
        self._shape: tuple[int, int, int] | None = None
        self._frame_count = 0
        self._process: subprocess.Popen | None = None

    def write(self, frame: np.ndarray) -> int:
        if self._shape is None:
            self._shape = tuple(frame.shape)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._process = _open_ffmpeg_process(self.path, frame.shape, fps=self.fps)
        elif self._shape != tuple(frame.shape):
            raise ValueError(f"{self.path} shape mismatch: expected {self._shape}, got {frame.shape}")
        if self._process is None or self._process.stdin is None:
            raise RuntimeError(f"video writer is not open: {self.path}")
        video_frame_index = self._frame_count
        self._process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self._frame_count += 1
        return video_frame_index

    def close(self) -> None:
        if self._process is None:
            return
        process = self._process
        self._process = None
        if process.stdin is not None:
            process.stdin.close()
            process.stdin = None
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace") if stderr else stdout.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg H.264 encoding failed for {self.path}: {message}")


class BufferedVideoWriter:
    def __init__(self, path: Path, *, fps: float) -> None:
        self.path = path
        self.fps = float(fps)
        self._shape: tuple[int, int, int] | None = None
        self._frame_paths: list[str] = []

    def write(self, image_path: str | Path, shape: tuple[int, int, int]) -> int:
        if self._shape is None:
            self._shape = shape
        elif self._shape != shape:
            raise ValueError(f"{self.path} shape mismatch: expected {self._shape}, got {shape}")
        video_frame_index = len(self._frame_paths)
        self._frame_paths.append(str(image_path))
        return video_frame_index

    def job(self) -> dict[str, Any] | None:
        if self._shape is None or not self._frame_paths:
            return None
        return {
            "path": str(self.path),
            "fps": self.fps,
            "shape": self._shape,
            "frame_paths": list(self._frame_paths),
        }


def write_navvla_lerobot_dataset(
    episodes: list[NavVLAEpisode],
    *,
    output_root: Path,
    spec: NavVLADatasetSpec,
    overwrite: bool = False,
    repair_existing: bool = False,
    write_workers: int | None = None,
    write_visual_token_cache: bool = False,
    visual_token_profile: Any | None = None,
    visual_token_encoder: Any | None = None,
    visual_token_encoder_factory: Any | None = None,
    context_index_config: ContextIndexConfig | None = None,
) -> dict[str, Any]:
    del write_visual_token_cache, visual_token_profile, visual_token_encoder, visual_token_encoder_factory
    if not episodes:
        raise ValueError("episodes must be non-empty")
    if overwrite and repair_existing:
        raise ValueError("overwrite and repair_existing are mutually exclusive")

    dataset_root = output_root / spec.dataset_name
    existing_root = dataset_root
    repair_plan: dict[str, Any] | None = None
    if dataset_root.exists():
        if overwrite:
            shutil.rmtree(dataset_root)
        elif repair_existing:
            repair_plan = _plan_existing_repair(dataset_root, episodes, spec=spec)
            staging_root = output_root / f".{spec.dataset_name}.repair_tmp"
            if staging_root.exists():
                shutil.rmtree(staging_root)
            dataset_root = staging_root
        else:
            raise FileExistsError(f"{dataset_root} exists; pass overwrite=True")

    media_keys = _media_keys(episodes)
    data_rows_by_file: dict[tuple[int, int], list[dict[str, Any]]] = {}
    episode_rows_by_file: dict[tuple[int, int], list[dict[str, Any]]] = {}
    tasks_rows = []
    video_index_rows = []
    frame_metadata_rows = []
    video_writers: dict[tuple[str, int, int], StreamingVideoWriter] = {}
    media_shapes: dict[str, tuple[int, int, int]] = {}
    tail_padding_frames = 0
    padded_steps = 0
    global_index = 0
    streaming_video_count = 0
    buffered_video_jobs: list[dict[str, Any]] = []

    resolved_write_workers = resolve_write_workers(write_workers)
    use_parallel_video = True
    shard_groups = _group_episodes_by_write_shard(episodes, spec=spec)
    if repair_plan is not None:
        shard_groups = [group for group in shard_groups if group[0] in repair_plan["rewrite_shards"]]
    write_jobs = [
        _build_write_shard_job(
            chunk_index=shard_key[0],
            file_index=shard_key[1],
            global_index_start=global_index_start,
            episode_items=episode_items,
            media_keys=media_keys,
            dataset_root=dataset_root,
            spec=spec,
            use_parallel_video=use_parallel_video,
        )
        for shard_key, global_index_start, episode_items in shard_groups
    ]
    try:
        if not write_jobs:
            shard_results = []
        elif resolved_write_workers == 1 or len(write_jobs) == 1:
            shard_results = [_process_write_shard_job(job) for job in write_jobs]
        else:
            max_workers = min(resolved_write_workers, len(write_jobs))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                shard_results = list(executor.map(_process_write_shard_job, write_jobs, chunksize=1))

        for shard_result in shard_results:
            shard_key = tuple(shard_result["shard_key"])
            data_rows_by_file[shard_key] = shard_result["data_rows"]
            episode_rows_by_file[shard_key] = shard_result["episode_rows"]
            tasks_rows.extend(shard_result["tasks_rows"])
            video_index_rows.extend(shard_result["video_index_rows"])
            frame_metadata_rows.extend(shard_result["frame_metadata_rows"])
            tail_padding_frames += shard_result["tail_padding_frames"]
            padded_steps += shard_result["padded_steps"]
            for media_key, shape in shard_result["media_shapes"].items():
                shape_tuple = tuple(shape)
                if media_key not in media_shapes:
                    media_shapes[media_key] = shape_tuple
                elif media_shapes[media_key] != shape_tuple:
                    raise ValueError(f"{media_key} shape mismatch: expected {media_shapes[media_key]}, got {shape_tuple}")
            for job in shard_result["buffered_video_jobs"]:
                buffered_video_jobs.append(job)
            streaming_video_count += shard_result["streaming_video_count"]
            global_index = max(global_index, shard_result["global_index_end"])
    finally:
        for writer in video_writers.values():
            writer.close()

    if repair_plan is not None:
        _copy_reusable_repair_shards(repair_plan, dataset_root)
        reusable_video_rows = _read_reusable_video_index_rows(repair_plan)
        video_index_rows = reusable_video_rows + video_index_rows
    task_specs = _task_specs_from_episodes(episodes)
    tasks_rows = [_task_row(task) for task in task_specs]
    frame_metadata_rows = _frame_metadata_rows_from_episodes(episodes)
    if repair_plan is not None:
        media_shapes = _media_shapes_from_episodes(episodes)
        tail_padding_frames, padded_steps = _tail_padding_from_episodes(episodes, spec=spec)
    global_index = sum(len(episode.frames) for episode in episodes)

    for (chunk_index, file_index), rows in sorted(data_rows_by_file.items()):
        path = dataset_root / DATA_PATH_PATTERN.format(chunk_index=chunk_index, file_index=file_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
    for (chunk_index, file_index), rows in sorted(episode_rows_by_file.items()):
        path = dataset_root / "meta" / "episodes" / f"chunk-{chunk_index:03d}" / f"part-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
    pd.DataFrame(tasks_rows).to_parquet(dataset_root / "meta" / "tasks.parquet", index=False)
    pd.DataFrame(video_index_rows).to_parquet(dataset_root / "meta" / "navvla_video_index.parquet", index=False)

    _write_info_json(
        dataset_root,
        spec=spec,
        media_shapes=media_shapes,
        total_frames=global_index,
        total_episodes=len(episodes),
        total_tasks=len(tasks_rows),
        total_videos=len(buffered_video_jobs),
    )
    _write_modality_json(dataset_root, spec=spec, media_keys=media_keys)
    _write_navvla_tasks(dataset_root, task_specs)
    _write_navvla_cameras(dataset_root, episodes)
    _write_frame_metadata(dataset_root, frame_metadata_rows)
    _write_schema_ext(dataset_root, spec=spec)
    statistics = build_dataset_statistics(
        dataset_key=f"{spec.dataset_name}_{spec.split}",
        action_steps=_flatten_written_action_steps(dataset_root),
        num_trajectories=len(episodes),
        num_transitions=global_index,
    )
    write_dataset_statistics(dataset_root / "dataset_statistics.json", statistics)

    _write_buffered_videos_parallel_jobs(buffered_video_jobs, workers=resolved_write_workers)

    build_context_indexes(
        episodes,
        spec=spec,
        output_root=dataset_root,
        config=context_index_config or ContextIndexConfig(),
        cache_manifest=None,
    )
    _write_report(
        dataset_root,
        total_frames=global_index,
        total_episodes=len(episodes),
        tail_padding_frames=tail_padding_frames,
        padded_steps=padded_steps,
    )

    final_root = dataset_root
    if repair_plan is not None:
        if existing_root.exists():
            shutil.rmtree(existing_root)
        dataset_root.rename(existing_root)
        final_root = existing_root
        repair_plan["summary"]["existing_root"] = str(existing_root)

    summary = {
        "dataset_root": str(final_root),
        "total_frames": global_index,
        "total_episodes": len(episodes),
        "visual_token_cache": {
            "generated_by_writer": False,
            "reason": "not generated by the standalone dataset converter",
        },
    }
    if repair_plan is not None:
        summary["repair_existing"] = repair_plan["summary"]
    return summary


def _media_keys(episodes: list[NavVLAEpisode]) -> list[str]:
    keys: list[str] = []
    for episode in episodes:
        for camera in episode.cameras:
            if camera.video_key not in keys:
                keys.append(camera.video_key)
    return keys


def _plan_existing_repair(dataset_root: Path, episodes: list[NavVLAEpisode], *, spec: NavVLADatasetSpec) -> dict[str, Any]:
    shard_groups = _group_episodes_by_write_shard(episodes, spec=spec)
    reusable_shards: set[tuple[int, int]] = set()
    rewrite_shards: set[tuple[int, int]] = set()
    complete_episodes = 0
    repaired_episodes = 0
    for shard_key, _global_index_start, episode_items in shard_groups:
        if _existing_shard_is_complete(dataset_root, shard_key, episode_items):
            reusable_shards.add(shard_key)
            complete_episodes += len(episode_items)
        else:
            rewrite_shards.add(shard_key)
            repaired_episodes += len(episode_items)
    return {
        "dataset_root": dataset_root,
        "reusable_shards": reusable_shards,
        "rewrite_shards": rewrite_shards,
        "summary": {
            "enabled": True,
            "existing_root": str(dataset_root),
            "complete_episodes": complete_episodes,
            "repaired_episodes": repaired_episodes,
            "reused_shards": len(reusable_shards),
            "rewritten_shards": len(rewrite_shards),
        },
    }


def _existing_shard_is_complete(
    dataset_root: Path,
    shard_key: tuple[int, int],
    episode_items: list[tuple[int, NavVLAEpisode]],
) -> bool:
    chunk_index, file_index = shard_key
    data_path = dataset_root / DATA_PATH_PATTERN.format(chunk_index=chunk_index, file_index=file_index)
    episode_path = dataset_root / "meta" / "episodes" / f"chunk-{chunk_index:03d}" / f"part-{file_index:03d}.parquet"
    if not data_path.exists() or not episode_path.exists():
        return False
    try:
        data = pd.read_parquet(data_path)
        episode_rows = pd.read_parquet(episode_path)
    except Exception:
        return False
    required_data_columns = {"episode_index", "frame_index", "index", "context.index_key"}
    required_episode_columns = {"episode_index", "episode_id", "length", "data/chunk_index", "data/file_index"}
    if not required_data_columns.issubset(data.columns) or not required_episode_columns.issubset(episode_rows.columns):
        return False
    for episode_index, episode in episode_items:
        rows = data[data["episode_index"].astype(int) == int(episode_index)]
        meta = episode_rows[episode_rows["episode_index"].astype(int) == int(episode_index)]
        if len(rows) != len(episode.frames) or len(meta) != 1:
            return False
        meta_row = meta.iloc[0]
        if str(meta_row["episode_id"]) != episode.episode_id:
            return False
        if int(meta_row["length"]) != len(episode.frames):
            return False
        if int(meta_row["data/chunk_index"]) != chunk_index or int(meta_row["data/file_index"]) != file_index:
            return False
        if rows["frame_index"].astype(int).tolist() != [int(frame.frame_index) for frame in episode.frames]:
            return False
        expected_indexes = list(range(int(rows["index"].min()), int(rows["index"].min()) + len(episode.frames)))
        if rows["index"].astype(int).tolist() != expected_indexes:
            return False
    return True


def _copy_reusable_repair_shards(repair_plan: dict[str, Any], dataset_root: Path) -> None:
    existing_root = repair_plan["dataset_root"]
    for chunk_index, file_index in sorted(repair_plan["reusable_shards"]):
        relative_paths = [
            Path(DATA_PATH_PATTERN.format(chunk_index=chunk_index, file_index=file_index)),
            Path("meta") / "episodes" / f"chunk-{chunk_index:03d}" / f"part-{file_index:03d}.parquet",
        ]
        for video_path in (existing_root / "videos").glob(f"*/chunk-{chunk_index:03d}/part-{file_index:03d}.mp4"):
            relative_paths.append(video_path.relative_to(existing_root))
        for relative_path in relative_paths:
            source = existing_root / relative_path
            if not source.exists():
                continue
            target = dataset_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _read_reusable_video_index_rows(repair_plan: dict[str, Any]) -> list[dict[str, Any]]:
    existing_root = repair_plan["dataset_root"]
    path = existing_root / "meta" / "navvla_video_index.parquet"
    if not path.exists():
        return []
    try:
        rows = pd.read_parquet(path)
    except Exception:
        return []
    shard_keys = repair_plan["reusable_shards"]
    if not shard_keys:
        return []
    keep = rows.apply(lambda row: (int(row["chunk_index"]), int(row["file_index"])) in shard_keys, axis=1)
    return rows[keep].to_dict("records")


def _task_row(task: NavVLATaskSpec) -> dict[str, Any]:
    return {
        "task_index": int(task.task_index),
        "task": task.instruction,
        "task_type": task.task_type,
        "task_subtype": task.task_subtype,
        "platform_text": task.platform_text,
        "dataset_source": task.dataset_source,
        "answer": task.answer,
    }


def _task_specs_from_episodes(episodes: list[NavVLAEpisode]) -> list[NavVLATaskSpec]:
    tasks: list[NavVLATaskSpec] = []
    seen = set()
    for episode in episodes:
        for task in _ordered_tasks_for_episode(episode):
            if task.task_index in seen:
                continue
            seen.add(task.task_index)
            tasks.append(task)
    return tasks


def _ordered_tasks_for_episode(episode: NavVLAEpisode) -> list[NavVLATaskSpec]:
    frame_tasks = [frame.task for frame in episode.frames if frame.task is not None]
    return frame_tasks if frame_tasks else [episode.task]


def _task_for_frame(episode: NavVLAEpisode, frame: NavVLAFrame) -> NavVLATaskSpec:
    return frame.task if frame.task is not None else episode.task


def _frame_metadata_rows_from_episodes(episodes: list[NavVLAEpisode]) -> list[dict[str, Any]]:
    rows = []
    global_index = 0
    for episode in episodes:
        for frame in episode.frames:
            rows.append(
                {
                    "index": global_index,
                    "source_frame_index": frame.source_frame_index,
                    "source_metadata": frame.source_metadata,
                }
            )
            global_index += 1
    return rows


def _media_shapes_from_episodes(episodes: list[NavVLAEpisode]) -> dict[str, tuple[int, int, int]]:
    media_shapes: dict[str, tuple[int, int, int]] = {}
    for episode in episodes:
        for frame in episode.frames:
            for media_key, image_path in frame.media_paths.items():
                shape = _read_rgb_image_shape(Path(image_path))
                if media_key not in media_shapes:
                    media_shapes[media_key] = shape
                elif media_shapes[media_key] != shape:
                    raise ValueError(f"{media_key} shape mismatch: expected {media_shapes[media_key]}, got {shape}")
    return media_shapes


def _tail_padding_from_episodes(episodes: list[NavVLAEpisode], *, spec: NavVLADatasetSpec) -> tuple[int, int]:
    tail_padding_frames = 0
    padded_steps = 0
    for episode in episodes:
        for frame in episode.frames:
            action_chunk = pad_action_chunk(
                frame.action,
                horizon=spec.action_horizon,
                action_dim=spec.action_dim,
                action_available=frame.action_available,
            )
            if action_chunk.padding_mask.any():
                tail_padding_frames += 1
                padded_steps += int(action_chunk.padding_mask.sum())
    return tail_padding_frames, padded_steps


def _episode_shard(episode_index: int, *, spec: NavVLADatasetSpec) -> tuple[int, int]:
    linear_file_index = episode_index // spec.episodes_per_file
    return linear_file_index // spec.files_per_chunk, linear_file_index % spec.files_per_chunk


def _read_rgb_image_shape(image_path: Path) -> tuple[int, int, int]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
    return (height, width, 3)


def _group_episodes_by_write_shard(
    episodes: list[NavVLAEpisode],
    *,
    spec: NavVLADatasetSpec,
) -> list[tuple[tuple[int, int], int, list[tuple[int, NavVLAEpisode]]]]:
    groups: dict[tuple[int, int], list[tuple[int, NavVLAEpisode]]] = {}
    order: list[tuple[int, int]] = []
    for episode_index, episode in enumerate(episodes):
        shard_key = _episode_shard(episode_index, spec=spec)
        if shard_key not in groups:
            groups[shard_key] = []
            order.append(shard_key)
        groups[shard_key].append((episode_index, episode))

    grouped: list[tuple[tuple[int, int], int, list[tuple[int, NavVLAEpisode]]]] = []
    global_index_start = 0
    for shard_key in order:
        episode_items = groups[shard_key]
        grouped.append((shard_key, global_index_start, episode_items))
        global_index_start += sum(len(episode.frames) for _, episode in episode_items)
    return grouped


def _build_write_shard_job(
    *,
    chunk_index: int,
    file_index: int,
    global_index_start: int,
    episode_items: list[tuple[int, NavVLAEpisode]],
    media_keys: list[str],
    dataset_root: Path,
    spec: NavVLADatasetSpec,
    use_parallel_video: bool,
) -> tuple[Any, ...]:
    return (
        chunk_index,
        file_index,
        global_index_start,
        [episode_index for episode_index, _episode in episode_items],
        [episode for _episode_index, episode in episode_items],
        list(media_keys),
        str(dataset_root),
        spec.dataset_name,
        spec.split,
        spec.context_policy_version,
        float(spec.fps),
        int(spec.action_horizon),
        int(spec.action_dim),
        int(spec.state_dim),
        bool(use_parallel_video),
    )


def _process_write_shard_job(job: tuple[Any, ...]) -> dict[str, Any]:
    (
        chunk_index,
        file_index,
        global_index_start,
        episode_indices,
        episodes,
        media_keys,
        dataset_root_str,
        dataset_name,
        split,
        context_policy_version,
        fps,
        action_horizon,
        action_dim,
        state_dim,
        use_parallel_video,
    ) = job
    dataset_root = Path(dataset_root_str)
    shard_key = (chunk_index, file_index)
    data_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    tasks_rows: list[dict[str, Any]] = []
    video_index_rows: list[dict[str, Any]] = []
    frame_metadata_rows: list[dict[str, Any]] = []
    media_shapes: dict[str, tuple[int, int, int]] = {}
    buffered_video_writers: dict[tuple[str, int, int], BufferedVideoWriter] = {}
    streaming_video_writers: dict[tuple[str, int, int], StreamingVideoWriter] = {}
    tail_padding_frames = 0
    padded_steps = 0
    global_index = global_index_start

    try:
        for episode_index, episode in zip(episode_indices, episodes):
            tasks_rows.append(_task_row(episode.task))
            episode_start_video_indices: dict[str, int] = {}
            for frame_pos, frame in enumerate(episode.frames):
                timestamp = resolve_timestamp(frame.timestamp, frame_index=frame.frame_index, fps=fps)
                action_chunk = pad_action_chunk(
                    frame.action,
                    horizon=action_horizon,
                    action_dim=action_dim,
                    action_available=frame.action_available,
                )
                if action_chunk.padding_mask.any():
                    tail_padding_frames += 1
                    padded_steps += int(action_chunk.padding_mask.sum())

                for media_key in media_keys:
                    if media_key not in frame.media_paths:
                        video_index_rows.append(
                            {
                                "index": global_index,
                                "video_key": media_key,
                                "available": False,
                                "video_frame_index": -1,
                                "chunk_index": chunk_index,
                                "file_index": file_index,
                            }
                        )
                        continue
                    image_path = Path(frame.media_paths[media_key])
                    shape = _read_rgb_image_shape(image_path)
                    if media_key not in media_shapes:
                        media_shapes[media_key] = shape
                    elif media_shapes[media_key] != shape:
                        raise ValueError(f"{media_key} shape mismatch: expected {media_shapes[media_key]}, got {shape}")
                    if use_parallel_video:
                        writer = _buffered_video_writer_for(
                            buffered_video_writers,
                            dataset_root=dataset_root,
                            media_key=media_key,
                            chunk_index=chunk_index,
                            file_index=file_index,
                            fps=fps,
                        )
                        video_frame_index = writer.write(image_path, shape)
                    else:
                        with Image.open(image_path) as image:
                            image_array = np.asarray(image.convert("RGB"))
                        writer = _video_writer_for(
                            streaming_video_writers,
                            dataset_root=dataset_root,
                            media_key=media_key,
                            chunk_index=chunk_index,
                            file_index=file_index,
                            fps=fps,
                        )
                        video_frame_index = writer.write(image_array)
                    episode_start_video_indices.setdefault(media_key, video_frame_index)
                    video_index_rows.append(
                        {
                            "index": global_index,
                            "video_key": media_key,
                            "available": True,
                            "video_frame_index": video_frame_index,
                            "chunk_index": chunk_index,
                            "file_index": file_index,
                        }
                    )

                context_key = (
                    f"{dataset_name}/{split}/{episode.episode_id}/"
                    f"f{frame.frame_index:06d}/{context_policy_version}"
                )
                state_values = np.asarray(frame.state, dtype=np.float32).tolist()
                if len(state_values) != state_dim:
                    raise ValueError(
                        f"frame {frame.frame_index} observation.state must match state_dim={state_dim}, "
                        f"got {len(state_values)}"
                    )
                data_rows.append(
                    {
                        "episode_index": episode_index,
                        "frame_index": int(frame.frame_index),
                        "timestamp": timestamp,
                        "task_index": int(_task_for_frame(episode, frame).task_index),
                        "observation.state": state_values,
                        "action": action_chunk.values.tolist(),
                        "action.padding_mask": action_chunk.padding_mask.tolist(),
                        "next.done": frame_pos == len(episode.frames) - 1,
                        "sample.action_available": bool(action_chunk.action_available),
                        "context.index_key": context_key,
                        "source_frame_index": -1 if frame.source_frame_index is None else int(frame.source_frame_index),
                        "index": global_index,
                    }
                )
                frame_metadata_rows.append(
                    {
                        "index": global_index,
                        "source_frame_index": frame.source_frame_index,
                        "source_metadata": frame.source_metadata,
                    }
                )
                global_index += 1

            episode_tasks = _ordered_tasks_for_episode(episode)
            episode_row = {
                "episode_index": episode_index,
                "episode_id": episode.episode_id,
                "trajectory_id": episode.trajectory_id,
                "task_index": int(episode_tasks[0].task_index),
                "split": episode.split,
                "scene_id": episode.task.scene_id,
                "tasks": [task.instruction for task in episode_tasks],
                "length": len(episode.frames),
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
            }
            for media_key in media_keys:
                episode_row[f"videos/{media_key}/from_timestamp"] = episode_start_video_indices.get(media_key, 0) / fps
            episode_rows.append(episode_row)
    finally:
        for writer in streaming_video_writers.values():
            writer.close()

    buffered_video_jobs: list[dict[str, Any]] = []
    for (media_key, writer_chunk_index, writer_file_index), writer in buffered_video_writers.items():
        job = writer.job()
        if job is None:
            continue
        buffered_video_jobs.append(
            {
                **job,
                "media_key": media_key,
                "chunk_index": writer_chunk_index,
                "file_index": writer_file_index,
            }
        )

    return {
        "shard_key": shard_key,
        "data_rows": data_rows,
        "episode_rows": episode_rows,
        "tasks_rows": tasks_rows,
        "video_index_rows": video_index_rows,
        "frame_metadata_rows": frame_metadata_rows,
        "tail_padding_frames": tail_padding_frames,
        "padded_steps": padded_steps,
        "media_shapes": media_shapes,
        "buffered_video_jobs": buffered_video_jobs,
        "streaming_video_count": len(streaming_video_writers),
        "global_index_end": global_index,
    }


def _video_writer_for(
    video_writers: dict[tuple[str, int, int], StreamingVideoWriter],
    *,
    dataset_root: Path,
    media_key: str,
    chunk_index: int,
    file_index: int,
    fps: float,
) -> StreamingVideoWriter:
    key = (media_key, chunk_index, file_index)
    if key not in video_writers:
        video_writers[key] = StreamingVideoWriter(
            dataset_root / "videos" / media_key / f"chunk-{chunk_index:03d}" / f"part-{file_index:03d}.mp4",
            fps=fps,
        )
    return video_writers[key]


def _buffered_video_writer_for(
    video_writers: dict[tuple[str, int, int], BufferedVideoWriter],
    *,
    dataset_root: Path,
    media_key: str,
    chunk_index: int,
    file_index: int,
    fps: float,
) -> BufferedVideoWriter:
    key = (media_key, chunk_index, file_index)
    if key not in video_writers:
        video_writers[key] = BufferedVideoWriter(
            dataset_root / "videos" / media_key / f"chunk-{chunk_index:03d}" / f"part-{file_index:03d}.mp4",
            fps=fps,
        )
    return video_writers[key]


def _write_buffered_videos_parallel(video_writers: dict[tuple[str, int, int], BufferedVideoWriter], *, workers: int) -> None:
    jobs = [job for writer in video_writers.values() if (job := writer.job()) is not None]
    _write_buffered_videos_parallel_jobs(jobs, workers=workers)


def _write_buffered_videos_parallel_jobs(jobs: list[dict[str, Any]], *, workers: int) -> None:
    encode_jobs = [
        {
            "path": job["path"],
            "fps": job["fps"],
            "shape": job["shape"],
            "frame_paths": job["frame_paths"],
        }
        for job in jobs
    ]
    if not encode_jobs:
        return
    max_workers = min(workers, len(encode_jobs))
    if max_workers == 1:
        for job in encode_jobs:
            _write_video_from_paths(job)
        return
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_write_video_from_paths, job) for job in encode_jobs]
        for future in futures:
            future.result()


def _flatten_written_action_steps(dataset_root: Path) -> np.ndarray:
    steps: list[np.ndarray] = []
    for path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        rows = pd.read_parquet(path, columns=["action", "action.padding_mask"])
        action_steps = flatten_valid_action_steps_from_rows(rows)
        if len(action_steps):
            steps.append(action_steps)
    if not steps:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(steps, axis=0).astype(np.float32)


def _write_video_from_paths(job: dict[str, Any]) -> None:
    path = Path(job["path"])
    shape = tuple(job["shape"])
    path.parent.mkdir(parents=True, exist_ok=True)
    process = _open_ffmpeg_process(path, shape, fps=float(job["fps"]))
    try:
        if process.stdin is None:
            raise RuntimeError("ffmpeg stdin pipe was not created")
        try:
            for frame_path in job["frame_paths"]:
                image = Image.open(frame_path).convert("RGB")
                image_array = np.asarray(image)
                if tuple(image_array.shape) != shape:
                    raise ValueError(f"{path} shape mismatch while encoding: expected {shape}, got {tuple(image_array.shape)}")
                process.stdin.write(np.ascontiguousarray(image_array, dtype=np.uint8).tobytes())
            process.stdin.close()
            process.stdin = None
            stdout, stderr = process.communicate()
        except Exception:
            process.kill()
            process.communicate()
            raise
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for NavVLA H.264 video encoding") from exc
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace") if stderr else stdout.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg H.264 encoding failed for {path}: {message}")


def _write_video(path: Path, frames: list[np.ndarray], *, fps: float) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    process = _open_ffmpeg_process(path, frames[0].shape, fps=fps)
    try:
        if process.stdin is None:
            raise RuntimeError("ffmpeg stdin pipe was not created")
        try:
            for frame in frames:
                process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            process.stdin.close()
            process.stdin = None
            stdout, stderr = process.communicate()
        except Exception:
            process.kill()
            process.communicate()
            raise
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for NavVLA H.264 video encoding") from exc
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace") if stderr else stdout.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg H.264 encoding failed for {path}: {message}")


def _open_ffmpeg_process(path: Path, shape: tuple[int, ...], *, fps: float) -> subprocess.Popen:
    height, width, channels = shape
    if channels != 3:
        raise ValueError(f"H.264 video writer expects RGB frames with 3 channels, got shape {shape}")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for NavVLA H.264 video encoding") from exc


def _write_info_json(
    dataset_root: Path,
    *,
    spec: NavVLADatasetSpec,
    media_shapes: dict[str, tuple[int, int, int]],
    total_frames: int,
    total_episodes: int,
    total_tasks: int,
    total_videos: int,
) -> None:
    features: dict[str, Any] = {}
    video_path = {}
    for media_key, shape in media_shapes.items():
        height, width, channels = shape
        feature_key = f"observation.images.{media_key}"
        features[feature_key] = {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": spec.fps, "video.height": height, "video.width": width, "video.channels": channels},
        }
        video_path[media_key] = f"videos/{media_key}/chunk-{{chunk_index:03d}}/part-{{file_index:03d}}.mp4"

    features.update(
        {
            "observation.state": {"dtype": "float32", "shape": [spec.state_dim], "names": _state_names(spec.state_dim)},
            "action": {"dtype": "float32", "shape": [spec.action_horizon * spec.action_dim], "names": ["action"]},
            "action.padding_mask": {"dtype": "bool", "shape": [spec.action_horizon], "names": ["horizon"]},
            "timestamp": {"dtype": "float64", "shape": [1], "names": ["timestamp"]},
            "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
            "episode_index": {"dtype": "int64", "shape": [1], "names": ["episode_index"]},
            "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
            "source_frame_index": {"dtype": "int64", "shape": [1], "names": ["source_frame_index"]},
            "index": {"dtype": "int64", "shape": [1], "names": ["index"]},
            "next.done": {"dtype": "bool", "shape": [1], "names": ["done"]},
            "sample.action_available": {"dtype": "bool", "shape": [1], "names": ["action_available"]},
            "context.index_key": {"dtype": "string", "shape": [1], "names": ["context_index_key"]},
        }
    )
    control_frequency_hz = spec.control_frequency_hz or spec.fps
    navvla_metadata = {
        "schema_version": "0.1",
        "action_horizon": spec.action_horizon,
        "action_dim": spec.action_dim,
        "control_frequency_hz": control_frequency_hz,
        "action_horizon_seconds": spec.action_horizon / control_frequency_hz,
        "episodes_per_file": spec.episodes_per_file,
        "files_per_chunk": spec.files_per_chunk,
        "state_dim": spec.state_dim,
        "state_mode": spec.state_mode,
        "state_order": ["x", "y", "z", "yaw"],
        "action_mode": "anchor_relative_body_frame_xyz_yaw",
        "action_anchor": "current_frame_pose",
        "timestamp_policy": "episode_relative_timestamp_from_frame0_source_timestamp_else_frame_index_over_fps",
        "tail_action_policy": "zero_pad_to_horizon",
        "action_padding_mask_policy": "all_false_zero_tail_unmasked",
        "action_normalization": _action_normalization(),
    }
    info = {
        "codebase_version": "v3.0",
        "dataset_name": spec.dataset_name,
        "robot_type": "navvla_navigation",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_videos,
        "chunks_size": spec.files_per_chunk,
        "fps": spec.fps,
        "splits": {spec.split: f"0:{total_episodes}"},
        "data_path": DATA_PATH_PATTERN,
        "video_path": video_path,
        "features": features,
        "navvla": navvla_metadata,
    }
    (dataset_root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def _write_modality_json(dataset_root: Path, *, spec: NavVLADatasetSpec, media_keys: list[str]) -> None:
    modality = {
        "video": {key: {"original_key": f"observation.images.{key}"} for key in media_keys},
        "state": {
            name: {"start": idx, "end": idx + 1, "absolute": True, "dtype": "float32", "original_key": "observation.state"}
            for idx, name in enumerate(_state_names(spec.state_dim))
        },
        "action": {
            "dx": {"start": 0, "end": 1, "absolute": False, "dtype": "float32", "original_key": "action"},
            "dy": {"start": 1, "end": 2, "absolute": False, "dtype": "float32", "original_key": "action"},
            "dz": {"start": 2, "end": 3, "absolute": False, "dtype": "float32", "original_key": "action"},
            "dyaw": {"start": 3, "end": 4, "absolute": False, "dtype": "float32", "original_key": "action"},
        },
        "annotation": {"language.language_instruction": {"original_key": "task_index"}},
    }
    (dataset_root / "meta" / "modality.json").write_text(json.dumps(modality, indent=2), encoding="utf-8")


def _write_navvla_tasks(dataset_root: Path, tasks: list[NavVLATaskSpec]) -> None:
    lines = []
    seen = set()
    for task in tasks:
        if task.task_index in seen:
            continue
        seen.add(task.task_index)
        payload = {
            "task_index": task.task_index,
            "task_type": task.task_type,
            "task_subtype": task.task_subtype,
            "platform_text": task.platform_text,
            "dataset_source": task.dataset_source,
            "answer": task.answer,
        }
        payload.update(task.metadata)
        lines.append(json.dumps(payload, ensure_ascii=False))
    (dataset_root / "meta" / "navvla_tasks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_navvla_cameras(dataset_root: Path, episodes: list[NavVLAEpisode]) -> None:
    cameras = {}
    for episode in episodes:
        for camera in episode.cameras:
            cameras[camera.name] = {
                "name": camera.name,
                "video_key": camera.video_key,
                "viewpoint_type": camera.viewpoint_type,
                "azimuth_rad": camera.azimuth_rad,
                "intrinsics": camera.intrinsics,
                "extrinsics_body": camera.extrinsics_body,
                "calibration_status": camera.calibration_status,
            }
    (dataset_root / "meta" / "navvla_cameras.json").write_text(json.dumps(cameras, indent=2), encoding="utf-8")


def _write_frame_metadata(dataset_root: Path, rows: list[dict[str, Any]]) -> None:
    path = dataset_root / "meta" / "navvla_frame_metadata.jsonl"
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_schema_ext(dataset_root: Path, *, spec: NavVLADatasetSpec) -> None:
    payload = {
        "schema_version": "0.1",
        "context_policy_version": spec.context_policy_version,
        "cache_policy_version": spec.cache_policy_version,
        "history_fields": ["context.index_key"],
        "frame_metadata": "meta/navvla_frame_metadata.jsonl",
        "video_index": "meta/navvla_video_index.parquet",
        "context_index_manifest": "meta/navvla_context_index_manifest.json",
        "context_index": "meta/context_index/budget_<budget>",
        "context_meta": "meta/context_index/budget_<budget>/context_meta.parquet",
        "context_arrays": "meta/context_index/budget_<budget>/context_arrays",
        "context_debug": f"cache/context_index_debug/budget_<budget>/{spec.split}.parquet",
    }
    (dataset_root / "meta" / "navvla_schema_ext.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_report(
    dataset_root: Path,
    *,
    total_frames: int,
    total_episodes: int,
    tail_padding_frames: int,
    padded_steps: int,
) -> None:
    payload = {
        "total_frames": total_frames,
        "total_episodes": total_episodes,
        "tail_padding": {"frames_with_padding": tail_padding_frames, "padded_steps": padded_steps},
        "visual_token_cache": {
            "status": "not_generated",
            "reason": "standalone dataset conversion does not run a model encoder",
        },
        "rejected_rows": [],
    }
    (dataset_root / "conversion_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _state_names(state_dim: int) -> list[str]:
    base = ["x", "y", "z", "yaw", "vx", "vy", "vz", "yaw_rate", "last_dx", "last_dy", "last_dz", "last_dyaw"]
    if state_dim <= len(base):
        return base[:state_dim]
    return base + [f"state_{idx}" for idx in range(len(base), state_dim)]


def _action_normalization() -> dict[str, dict[str, Any]]:
    return {
        "dx": {"mean": 0.0, "std": 1.0, "clip": [-3.0, 3.0], "unit": "meter"},
        "dy": {"mean": 0.0, "std": 1.0, "clip": [-3.0, 3.0], "unit": "meter"},
        "dz": {"mean": 0.0, "std": 1.0, "clip": [-3.0, 3.0], "unit": "meter"},
        "dyaw": {"mean": 0.0, "std": 1.0, "clip": [-3.14, 3.14], "unit": "radian"},
    }
