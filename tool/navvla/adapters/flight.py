from __future__ import annotations

import json
import math
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tool.navvla.adapters.base import NavVLASourceAdapter, register_adapter
from tool.navvla.context_index import ContextIndexConfig
from tool.navvla.workers import resolve_workers as resolve_load_workers
from tool.navvla.lerobot_v3_writer import write_navvla_lerobot_dataset
from tool.navvla.schema import NavVLACameraSpec, NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec


SOURCE_SPLIT_TO_TARGET = {
    "train": "vln_train",
    "vln_train": "vln_train",
    "eval": "vln_val_seen",
    "val_seen": "vln_val_seen",
    "vln_val_seen": "vln_val_seen",
}
EVAL_SPLIT_ALIASES = {"eval", "val_seen", "vln_val_seen"}
FLIGHT_CAMERA = NavVLACameraSpec(
    name="front",
    video_key="front_image",
    viewpoint_type="front",
    azimuth_rad=0.0,
    calibration_status="unknown",
)
FLIGHT_CONTEXT_INDEX_CONFIG = ContextIndexConfig(budget_num_cameras=1, history_camera_names=("front",))
PLATFORM_TEXT = "Platform: UAV. Task: instruction-conditioned navigation. Action: local 3D waypoints (dx, dy, dz, dyaw)."
STATE_PLACEHOLDER = [0.0, 0.0, 0.0, 0.0]
STATE_MODE = "unavailable_zero_placeholder"
MISSION_PATTERN = re.compile(r'The overall mission instruction is:\s*"(.*?)"', re.DOTALL)


class FlightEpisodeError(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class FlightAdapter(NavVLASourceAdapter):
    name = "flight"

    def __init__(
        self,
        *,
        media_cache_root: str | Path | None = None,
        fail_on_missing_media: bool = False,
        fps: float = 1.0,
        action_horizon: int = 8,
    ) -> None:
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.fail_on_missing_media = bool(fail_on_missing_media)
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.filter_report: dict[str, Any] = {}
        self.load_workers: int | None = None

    def configure(
        self,
        *,
        media_cache_root: str | Path | None = None,
        fail_on_missing_media: bool = False,
        fps: float = 1.0,
        action_horizon: int = 8,
        load_workers: int | None = None,
        **kwargs: Any,
    ) -> "FlightAdapter":
        super().configure(**kwargs)
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.fail_on_missing_media = bool(fail_on_missing_media)
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.load_workers = load_workers
        return self

    def load_episodes(
        self,
        source_root: str | Path,
        *,
        split: str = "train",
        max_episodes: int | None = None,
        load_workers: int | None = None,
    ) -> list[NavVLAEpisode]:
        source_root = Path(source_root)
        source_split = normalize_flight_source_split(split)
        target_split = normalize_source_split(split)
        media_cache_root = resolve_media_cache_root(source_root, media_cache_root=self.media_cache_root)
        resolved_workers = resolve_flight_load_workers(load_workers)

        episodes: list[NavVLAEpisode] = []
        filtered: list[dict[str, Any]] = []
        original_episode_count = 0
        kept_frame_count = 0
        if resolved_workers > 1 and max_episodes is None:
            episodes, filtered, original_episode_count, kept_frame_count = load_flight_episode_jobs_parallel(
                source_root=source_root,
                media_cache_root=media_cache_root,
                source_split=source_split,
                target_split=target_split,
                action_horizon=self.action_horizon,
                fail_on_missing_media=self.fail_on_missing_media,
                load_workers=resolved_workers,
            )
        else:
            seen_bases: set[tuple[str, str]] = set()
            for source_family, annotation_dir in flight_annotation_dirs(source_root, source_split):
                for annotation_path in sorted(annotation_dir.glob("*.json")):
                    base_id = annotation_base_id(annotation_path)
                    dedupe_key = (source_family, base_id)
                    if dedupe_key in seen_bases:
                        filtered.append(
                            filter_entry(
                                source_family=source_family,
                                annotation_path=annotation_path,
                                video_path=source_root / f"{source_family}_videos" / f"{base_id}.mp4",
                                reason="duplicate_annotation",
                                message=f"duplicate FLIGHT annotation for {source_family}/{base_id}",
                            )
                        )
                        continue
                    original_episode_count += 1
                    video_path = source_root / f"{source_family}_videos" / f"{base_id}.mp4"
                    try:
                        episode = build_episode(
                            annotation_path=annotation_path,
                            video_path=video_path,
                            media_cache_root=media_cache_root,
                            source_family=source_family,
                            source_split=source_split,
                            base_id=base_id,
                            task_index=len(episodes),
                            target_split=target_split,
                            action_horizon=self.action_horizon,
                            fail_on_missing_media=self.fail_on_missing_media,
                        )
                    except FlightEpisodeError as exc:
                        filtered.append(
                            filter_entry(
                                source_family=source_family,
                                annotation_path=annotation_path,
                                video_path=video_path,
                                reason=exc.reason,
                                message=str(exc),
                            )
                        )
                        continue
                    seen_bases.add(dedupe_key)
                    episodes.append(episode)
                    kept_frame_count += len(episode.frames)
                    if max_episodes is not None and len(episodes) >= max_episodes:
                        break
                if max_episodes is not None and len(episodes) >= max_episodes:
                    break

        self.filter_report = build_filter_report(
            original_episode_count=original_episode_count,
            kept_episode_count=len(episodes),
            kept_frame_count=kept_frame_count,
            filtered_episodes=filtered,
            source_split=source_split,
            target_split=target_split,
        )
        if not episodes:
            raise FileNotFoundError(f"no FLIGHT episodes found under {source_root}")
        return renumber_episode_task_indices(episodes)

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
        context_policy_version: str = "bats-v1",
        cache_policy_version: str = "smoke-coarse-v1",
        cache_workers: int | None = None,
        write_visual_token_cache: bool = True,
        visual_token_profile: Any | None = None,
        visual_token_encoder: Any | None = None,
        visual_token_encoder_factory: Any | None = None,
        episodes_per_file: int = 20,
        files_per_chunk: int = 50,
        load_workers: int | None = None,
    ) -> dict[str, Any]:
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        target_split = normalize_source_split(split)
        episodes = self.load_episodes(
            source_root,
            split=split,
            max_episodes=max_episodes,
            load_workers=self.load_workers if load_workers is None else load_workers,
        )
        spec = NavVLADatasetSpec(
            dataset_name=dataset_name,
            fps=fps,
            control_frequency_hz=float(control_frequency_hz) if control_frequency_hz is not None else 2.0,
            action_horizon=action_horizon,
            action_dim=4,
            state_dim=4,
            context_policy_version=context_policy_version,
            cache_policy_version=cache_policy_version,
            split=target_split,
            episodes_per_file=episodes_per_file,
            files_per_chunk=files_per_chunk,
            state_mode=STATE_MODE,
        )
        summary = write_navvla_lerobot_dataset(
            episodes,
            output_root=Path(output_root),
            spec=spec,
            overwrite=overwrite,
            repair_existing=repair_existing,
            cache_workers=cache_workers,
            write_visual_token_cache=write_visual_token_cache,
            visual_token_profile=visual_token_profile,
            visual_token_encoder=visual_token_encoder,
            visual_token_encoder_factory=visual_token_encoder_factory,
            context_index_config=FLIGHT_CONTEXT_INDEX_CONFIG,
        )
        report_path = write_filter_report(summary["dataset_root"], self.filter_report)
        summary["flight_filter_report"] = str(report_path)
        summary["flight_filtered_episodes"] = self.filter_report.get("filtered_episode_count", 0)
        return summary


def normalize_source_split(split: str) -> str:
    value = str(split).strip()
    try:
        return SOURCE_SPLIT_TO_TARGET[value]
    except KeyError as exc:
        raise ValueError(f"unsupported FLIGHT split: {split}; supported splits: train, eval, vln_val_seen") from exc


def normalize_flight_source_split(split: str) -> str:
    value = str(split).strip()
    if value in EVAL_SPLIT_ALIASES:
        return "eval"
    normalize_source_split(value)
    return "train"


def flight_annotation_dirs(source_root: Path, source_split: str) -> tuple[tuple[str, Path], ...]:
    if source_split == "eval":
        return (("fgvln", source_root / "fgvln_eval"), ("lhflow", source_root / "lhflow_eval"))
    return (("fgvln", source_root / "fgvln_annotations"), ("lhflow", source_root / "lhflow_annotations"))


def annotation_base_id(path: Path) -> str:
    name = path.name
    if name.endswith(".json"):
        name = name[:-5]
    if name.endswith(".1"):
        name = name[:-2]
    if name.endswith("_streaming"):
        name = name[: -len("_streaming")]
    return name


def resolve_media_cache_root(source_root: Path, *, media_cache_root: str | Path | None) -> Path:
    if media_cache_root is not None:
        return Path(media_cache_root)
    return source_root / ".navvla_media_cache" / "flight"


def resolve_flight_load_workers(load_workers: int | None) -> int:
    if load_workers is None:
        return 1
    return resolve_load_workers(load_workers)


def renumber_episode_task_indices(episodes: list[NavVLAEpisode]) -> list[NavVLAEpisode]:
    return [replace(episode, task=replace(episode.task, task_index=index), episode_id=f"{index:05d}") for index, episode in enumerate(episodes)]


def load_flight_episode_jobs_parallel(
    *,
    source_root: Path,
    media_cache_root: Path,
    source_split: str,
    target_split: str,
    action_horizon: int,
    fail_on_missing_media: bool,
    load_workers: int,
) -> tuple[list[NavVLAEpisode], list[dict[str, Any]], int, int]:
    jobs, duplicate_filtered, original_episode_count = collect_flight_episode_jobs(
        source_root=source_root,
        media_cache_root=media_cache_root,
        source_split=source_split,
        target_split=target_split,
        action_horizon=action_horizon,
        fail_on_missing_media=fail_on_missing_media,
    )
    if not jobs:
        return [], duplicate_filtered, original_episode_count, 0

    max_workers = min(int(load_workers), len(jobs))
    results: dict[int, tuple[NavVLAEpisode | None, dict[str, Any] | None]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for order_index, episode, filtered_entry in executor.map(_load_flight_episode_job, jobs):
            results[int(order_index)] = (episode, filtered_entry)

    episodes: list[NavVLAEpisode] = []
    filtered: list[dict[str, Any]] = list(duplicate_filtered)
    kept_frame_count = 0
    for order_index in range(len(jobs)):
        episode, filtered_entry = results[order_index]
        if filtered_entry is not None:
            filtered.append(filtered_entry)
            continue
        if episode is None:
            continue
        episodes.append(episode)
        kept_frame_count += len(episode.frames)
    return renumber_episode_task_indices(episodes), filtered, original_episode_count, kept_frame_count


def collect_flight_episode_jobs(
    *,
    source_root: Path,
    media_cache_root: Path,
    source_split: str,
    target_split: str,
    action_horizon: int,
    fail_on_missing_media: bool,
) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]], int]:
    jobs: list[tuple[Any, ...]] = []
    filtered: list[dict[str, Any]] = []
    original_episode_count = 0
    seen_bases: set[tuple[str, str]] = set()
    for source_family, annotation_dir in flight_annotation_dirs(source_root, source_split):
        for annotation_path in sorted(annotation_dir.glob("*.json")):
            base_id = annotation_base_id(annotation_path)
            dedupe_key = (source_family, base_id)
            video_path = source_root / f"{source_family}_videos" / f"{base_id}.mp4"
            if dedupe_key in seen_bases:
                filtered.append(
                    filter_entry(
                        source_family=source_family,
                        annotation_path=annotation_path,
                        video_path=video_path,
                        reason="duplicate_annotation",
                        message=f"duplicate FLIGHT annotation for {source_family}/{base_id}",
                    )
                )
                continue
            seen_bases.add(dedupe_key)
            original_episode_count += 1
            jobs.append(
                (
                    len(jobs),
                    str(annotation_path),
                    str(video_path),
                    str(media_cache_root),
                    source_family,
                    source_split,
                    base_id,
                    target_split,
                    int(action_horizon),
                    bool(fail_on_missing_media),
                )
            )
    return jobs, filtered, original_episode_count


def _load_flight_episode_job(job: tuple[Any, ...]) -> tuple[int, NavVLAEpisode | None, dict[str, Any] | None]:
    (
        order_index,
        annotation_path,
        video_path,
        media_cache_root,
        source_family,
        source_split,
        base_id,
        target_split,
        action_horizon,
        fail_on_missing_media,
    ) = job
    annotation = Path(annotation_path)
    video = Path(video_path)
    try:
        episode = build_episode(
            annotation_path=annotation,
            video_path=video,
            media_cache_root=Path(media_cache_root),
            source_family=str(source_family),
            source_split=str(source_split),
            base_id=str(base_id),
            task_index=int(order_index),
            target_split=str(target_split),
            action_horizon=int(action_horizon),
            fail_on_missing_media=bool(fail_on_missing_media),
        )
    except FlightEpisodeError as exc:
        return (
            int(order_index),
            None,
            filter_entry(
                source_family=str(source_family),
                annotation_path=annotation,
                video_path=video,
                reason=exc.reason,
                message=str(exc),
            ),
        )
    return int(order_index), episode, None


def build_episode(
    *,
    annotation_path: Path,
    video_path: Path,
    media_cache_root: Path,
    source_family: str,
    source_split: str,
    base_id: str,
    task_index: int,
    target_split: str,
    action_horizon: int,
    fail_on_missing_media: bool,
) -> NavVLAEpisode:
    rows = read_annotation(annotation_path)
    if not video_path.exists():
        raise FlightEpisodeError("missing_video", f"FLIGHT video not found: {video_path}")
    video_info = probe_video(video_path)
    instruction = unique_instruction(rows)
    chunk_map = build_chunk_map(
        rows,
        source_family=source_family,
        source_split=source_split,
        annotation_path=annotation_path,
        video_path=video_path,
    )
    if not chunk_map:
        raise FlightEpisodeError("empty_action", f"no FLIGHT action chunks found in {annotation_path}")

    frames: list[NavVLAFrame] = []
    max_media_timestamp = max_extractable_timestamp(video_info)
    reader = VideoFrameReader(video_path)
    try:
        for timestamp in sorted(chunk_map):
            if timestamp > max_media_timestamp:
                continue
            action = action_for_anchor(chunk_map, timestamp, horizon=action_horizon)
            if not action:
                continue
            try:
                image_path = materialize_video_frame(
                    video_path,
                    media_cache_root=media_cache_root,
                    source_family=source_family,
                    base_id=base_id,
                    timestamp=timestamp,
                    reader=reader,
                )
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                if fail_on_missing_media:
                    raise
                raise FlightEpisodeError("decode_error", f"failed to extract FLIGHT video frame {video_path}@{timestamp}: {exc}") from exc
            source_frame_index = source_video_frame_index(timestamp, video_info)
            source_chunk = chunk_map[timestamp]
            frames.append(
                NavVLAFrame(
                    frame_index=len(frames),
                    timestamp=float(timestamp),
                    media_paths={"front_image": image_path},
                    state=list(STATE_PLACEHOLDER),
                    action=action,
                    action_available=bool(action),
                    source_frame_index=source_frame_index,
                    source_metadata={
                        "source_dataset": "flight",
                        "source_family": source_family,
                        "source_split": source_split,
                        "target_split": target_split,
                        "source_json": str(annotation_path),
                        "source_video": str(video_path),
                        "source_video_name": video_path.name,
                        "trajectory_id": f"flight_{source_family}_{base_id}",
                        "source_timestamp": float(timestamp),
                        "video_window": source_chunk["video_window"],
                        "text_stream": source_chunk["text_stream"],
                        "source_action_shape": source_chunk["source_action_shape"],
                        "source_state": source_chunk.get("source_state"),
                        "state_available": False,
                        "state_mode": STATE_MODE,
                        "action_units": "xy_zdown_m_yaw_rad_from_source_cm_degree",
                        "converted_coordinate_frame": "x_forward_y_right_z_down_yaw_right_positive",
                        "action_anchor": "source_local_waypoint_anchor_timestamp",
                    },
                )
            )
    finally:
        reader.close()
    if not frames:
        raise FlightEpisodeError("empty_frames", f"no FLIGHT frames built for {annotation_path}")

    task = NavVLATaskSpec(
        task_index=task_index,
        instruction=instruction,
        task_type="navigation",
        task_subtype=f"flight_{source_family}",
        platform_text=PLATFORM_TEXT,
        dataset_source=f"flight_{source_family}",
        scene_id=f"flight_{source_family}",
    )
    return NavVLAEpisode(
        episode_id=f"{task_index:05d}",
        task=task,
        frames=frames,
        cameras=[FLIGHT_CAMERA],
        split=target_split,
        trajectory_id=f"flight_{source_family}_{base_id}",
    )


def read_annotation(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FlightEpisodeError("json_parse_error", f"invalid FLIGHT JSON {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise FlightEpisodeError("schema_error", f"FLIGHT annotation must be a list: {path}")
    return payload


def unique_instruction(rows: list[dict[str, Any]]) -> str:
    instructions = set()
    for row in rows:
        if row.get("role") != "user":
            continue
        for text in user_texts(row):
            match = MISSION_PATTERN.search(text)
            if match:
                instructions.add(clean_text(match.group(1)))
    instructions.discard("")
    if len(instructions) != 1:
        raise FlightEpisodeError("instruction_error", f"expected exactly one FLIGHT overall mission instruction, got {len(instructions)}")
    return next(iter(instructions))


def user_texts(row: dict[str, Any]) -> list[str]:
    content = row.get("content")
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    return [str(item.get("text") or "") for item in content if isinstance(item, dict) and item.get("type") == "text"]


def user_video_window(row: dict[str, Any]) -> dict[str, Any]:
    content = row.get("content")
    if not isinstance(content, list):
        return {}
    for item in content:
        if isinstance(item, dict) and item.get("type") == "video":
            return {
                "video": item.get("video"),
                "video_start": item.get("video_start"),
                "video_end": item.get("video_end"),
            }
    return {}


def assistant_text_stream(row: dict[str, Any]) -> list[list[Any]]:
    content = row.get("content")
    if not isinstance(content, list):
        return []
    for item in content:
        if isinstance(item, dict) and "text_stream" in item:
            value = item.get("text_stream")
            return value if isinstance(value, list) else []
    return []


def build_chunk_map(
    rows: list[dict[str, Any]],
    *,
    source_family: str,
    source_split: str,
    annotation_path: Path,
    video_path: Path,
) -> dict[float, dict[str, Any]]:
    chunk_map: dict[float, dict[str, Any]] = {}
    user_row: dict[str, Any] | None = None
    for row in rows:
        role = row.get("role")
        if role == "user":
            user_row = row
            continue
        if role != "assistant" or user_row is None:
            continue
        video_window = user_video_window(user_row)
        text_stream = assistant_text_stream(row)
        action = row.get("action")
        if not isinstance(action, list):
            continue
        source_action_shape = nested_shape(action)
        if is_lhflow_eval_action(source_family=source_family, source_split=source_split, action=action):
            video_start = video_window.get("video_start")
            try:
                window_start = float(video_start)
            except (TypeError, ValueError) as exc:
                raise FlightEpisodeError("video_window_error", f"invalid FLIGHT eval video_start in {annotation_path}") from exc
            for chunk_index, raw_chunk in enumerate(action):
                timestamp = clean_timestamp(window_start + float(chunk_index))
                segment_index, segment = text_stream_segment_at(text_stream, timestamp, annotation_path)
                converted = convert_source_chunk(raw_chunk)
                insert_chunk_entry(
                    chunk_map,
                    {
                        "timestamp": timestamp,
                        "chunk": converted,
                        "source_family": source_family,
                        "source_json": str(annotation_path),
                        "source_video": str(video_path),
                        "video_window": video_window,
                        "text_stream": segment,
                        "segment_index": segment_index,
                        "subchunk_index": chunk_index,
                        "source_action_shape": source_action_shape,
                        "source_state": row.get("state"),
                        "action_chunking": "lhflow_eval_video_window_1s_chunks",
                    },
                    annotation_path=annotation_path,
                )
            continue
        if len(text_stream) != len(action):
            raise FlightEpisodeError(
                "action_shape_error",
                f"FLIGHT text_stream/action outer length mismatch in {annotation_path}: {len(text_stream)} vs {len(action)}",
            )
        for segment_index, (segment, segment_chunks) in enumerate(zip(text_stream, action)):
            if not isinstance(segment, list) or len(segment) < 3:
                raise FlightEpisodeError("text_stream_error", f"invalid FLIGHT text_stream entry in {annotation_path}")
            if not isinstance(segment_chunks, list):
                raise FlightEpisodeError("action_shape_error", f"invalid FLIGHT action segment in {annotation_path}")
            segment_start = float(segment[0])
            for chunk_index, raw_chunk in enumerate(segment_chunks):
                timestamp = clean_timestamp(segment_start + float(chunk_index))
                converted = convert_source_chunk(raw_chunk)
                entry = {
                    "timestamp": timestamp,
                    "chunk": converted,
                    "source_family": source_family,
                    "source_json": str(annotation_path),
                    "source_video": str(video_path),
                    "video_window": video_window,
                    "text_stream": segment,
                    "segment_index": segment_index,
                    "subchunk_index": chunk_index,
                    "source_action_shape": source_action_shape,
                    "source_state": row.get("state"),
                }
                insert_chunk_entry(chunk_map, entry, annotation_path=annotation_path)
    return chunk_map


def is_lhflow_eval_action(*, source_family: str, source_split: str, action: list[Any]) -> bool:
    return source_family == "lhflow" and source_split == "eval" and nested_shape(action) == [8, 4, 4]


def text_stream_segment_at(
    text_stream: list[list[Any]],
    timestamp: float,
    annotation_path: Path,
) -> tuple[int, list[Any]]:
    fallback: tuple[int, list[Any]] | None = None
    for segment_index, segment in enumerate(text_stream):
        if not isinstance(segment, list) or len(segment) < 3:
            raise FlightEpisodeError("text_stream_error", f"invalid FLIGHT text_stream entry in {annotation_path}")
        if fallback is None:
            fallback = (segment_index, segment)
        try:
            segment_start = float(segment[0])
            segment_end = float(segment[1])
        except (TypeError, ValueError) as exc:
            raise FlightEpisodeError("text_stream_error", f"invalid FLIGHT text_stream time in {annotation_path}") from exc
        if segment_start <= float(timestamp) < segment_end:
            return segment_index, segment
    if fallback is not None:
        return fallback
    raise FlightEpisodeError("text_stream_error", f"empty FLIGHT text_stream for eval action in {annotation_path}")


def insert_chunk_entry(chunk_map: dict[float, dict[str, Any]], entry: dict[str, Any], *, annotation_path: Path) -> None:
    timestamp = float(entry["timestamp"])
    if timestamp in chunk_map:
        if not np.allclose(np.asarray(chunk_map[timestamp]["chunk"]), np.asarray(entry["chunk"]), atol=1e-6):
            raise FlightEpisodeError(
                "action_conflict",
                f"conflicting FLIGHT action chunks at timestamp={timestamp} in {annotation_path}",
            )
        return
    chunk_map[timestamp] = entry


def convert_source_chunk(raw_chunk: Any) -> list[list[float]]:
    if not isinstance(raw_chunk, list) or len(raw_chunk) != 4:
        raise FlightEpisodeError("action_shape_error", f"FLIGHT action chunk must have 4 waypoints, got {nested_shape(raw_chunk)}")
    converted = []
    for raw_waypoint in raw_chunk:
        if not isinstance(raw_waypoint, list) or len(raw_waypoint) != 4:
            raise FlightEpisodeError("action_shape_error", f"FLIGHT waypoint must be [x,y,z,yaw], got {raw_waypoint!r}")
        converted.append(
            [
                clean_float(float(raw_waypoint[0]) * 0.01),
                clean_float(float(raw_waypoint[1]) * 0.01),
                clean_float(-float(raw_waypoint[2]) * 0.01),
                clean_float(math.radians(float(raw_waypoint[3]))),
            ]
        )
    return converted


def action_for_anchor(chunk_map: dict[float, dict[str, Any]], timestamp: float, *, horizon: int) -> list[list[float]]:
    action: list[list[float]] = []
    for offset in range(3):
        chunk_entry = chunk_map.get(clean_timestamp(timestamp + float(offset)))
        if chunk_entry is None:
            continue
        chunk = list(chunk_entry["chunk"])
        action.extend(chunk if offset == 0 else chunk[2:])
        if len(action) >= horizon:
            break
    return action[:horizon]


def materialize_video_frame(
    video_path: Path,
    *,
    media_cache_root: Path,
    source_family: str,
    base_id: str,
    timestamp: float,
    reader: "VideoFrameReader | None" = None,
) -> Path:
    frame_index = int(round(float(timestamp) * 1000.0))
    cache_path = media_cache_root / source_family / base_id / f"{frame_index:010d}.png"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        if is_valid_cached_image(cache_path):
            return cache_path
        cache_path.unlink(missing_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    array = reader.read(timestamp) if reader is not None else read_video_frame(video_path, timestamp=timestamp)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    try:
        Image.fromarray(array, mode="RGB").save(tmp_path, format="PNG")
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return cache_path


def is_valid_cached_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False
    return True


class VideoFrameReader:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self._cv2 = None
        self._cap = None
        try:
            import cv2
        except ImportError:
            return
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return
        self._cv2 = cv2
        self._cap = cap

    def read(self, timestamp: float) -> np.ndarray:
        if self._cap is None or self._cv2 is None:
            return read_video_frame_ffmpeg(self.video_path, timestamp=timestamp)
        self._cap.set(self._cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise ValueError(f"cannot decode video frame: {self.video_path}@{timestamp}")
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def read_video_frame(video_path: Path, *, timestamp: float) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return read_video_frame_ffmpeg(video_path, timestamp=timestamp)

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"cannot decode video frame: {video_path}@{timestamp}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def read_video_frame_ffmpeg(video_path: Path, *, timestamp: float) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, float(timestamp)):.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    if not result.stdout:
        raise ValueError(f"ffmpeg produced no frame for {video_path}@{timestamp}")
    from io import BytesIO

    with Image.open(BytesIO(result.stdout)) as image:
        return np.asarray(image.convert("RGB"))


def probe_video(video_path: Path) -> dict[str, float]:
    try:
        import cv2
    except ImportError:
        return probe_video_ffprobe(video_path)

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise FlightEpisodeError("decode_error", f"cannot open FLIGHT video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        cap.release()
    if fps <= 0:
        return probe_video_ffprobe(video_path)
    return {"fps": fps, "frame_count": frame_count}


def probe_video_ffprobe(video_path: Path) -> dict[str, float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise FlightEpisodeError("decode_error", f"failed to probe FLIGHT video {video_path}: {exc}") from exc
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    fps = parse_frame_rate(str(stream.get("avg_frame_rate") or "0/0"))
    if fps <= 0:
        raise FlightEpisodeError("decode_error", f"invalid FLIGHT video fps for {video_path}")
    frame_count = float(stream.get("nb_frames") or 0.0)
    return {"fps": fps, "frame_count": frame_count}


def max_extractable_timestamp(video_info: dict[str, float]) -> float:
    fps = float(video_info.get("fps") or 0.0)
    frame_count = float(video_info.get("frame_count") or 0.0)
    if fps <= 0 or frame_count <= 0:
        return float("inf")
    return max(0.0, (frame_count - 1.0) / fps)


def source_video_frame_index(timestamp: float, video_info: dict[str, float]) -> int:
    fps = float(video_info.get("fps") or 0.0)
    frame_count = int(video_info.get("frame_count") or 0)
    index = int(round(float(timestamp) * fps)) if fps > 0 else int(round(float(timestamp)))
    if frame_count > 0:
        index = min(index, frame_count - 1)
    return max(0, index)


def parse_frame_rate(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(value)


def nested_shape(value: Any) -> list[int]:
    shape = []
    while isinstance(value, list):
        shape.append(len(value))
        value = value[0] if value else None
    return shape


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def clean_timestamp(value: float) -> float:
    return round(float(value), 6)


def clean_float(value: float) -> float:
    value = float(value)
    return 0.0 if abs(value) < 1e-7 else value


def filter_entry(
    *,
    source_family: str,
    annotation_path: Path,
    video_path: Path,
    reason: str,
    message: str,
) -> dict[str, Any]:
    return {
        "source_family": source_family,
        "annotation_path": str(annotation_path),
        "video_path": str(video_path),
        "episode_base_id": annotation_base_id(annotation_path),
        "reason": reason,
        "message": message,
    }


def build_filter_report(
    *,
    original_episode_count: int,
    kept_episode_count: int,
    kept_frame_count: int,
    filtered_episodes: list[dict[str, Any]],
    source_split: str,
    target_split: str,
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    for entry in filtered_episodes:
        reason = str(entry.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "dataset": "flight",
        "source_split": source_split,
        "target_split": target_split,
        "filter_policy": "skip_episode_on_parse_media_instruction_action_error",
        "filter_granularity": "episode",
        "original_episode_count": int(original_episode_count),
        "kept_episode_count": int(kept_episode_count),
        "filtered_episode_count": int(len(filtered_episodes)),
        "kept_frame_count": int(kept_frame_count),
        "filtered_reason_counts": reason_counts,
        "filtered_episodes": filtered_episodes,
    }


def write_filter_report(dataset_root: str | Path, report: dict[str, Any]) -> Path:
    path = Path(dataset_root) / "meta" / "flight_filtered_episodes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


register_adapter(FlightAdapter())
