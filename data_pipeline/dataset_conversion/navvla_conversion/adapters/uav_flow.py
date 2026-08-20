from __future__ import annotations

import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from navvla_conversion.adapters.base import NavVLASourceAdapter, register_adapter
from navvla_conversion.workers import resolve_workers as resolve_load_workers
from navvla_conversion.schema import NavVLACameraSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec
from navvla_conversion.statistics import body_frame_action_from_pose


UAV_FLOW_CAMERA = NavVLACameraSpec(
    name="front",
    video_key="front_image",
    viewpoint_type="front",
    azimuth_rad=0.0,
    calibration_status="unknown",
)
PLATFORM_TEXT = "Platform: UAV. Task: instruction-conditioned navigation. Action: local 3D waypoints (dx, dy, dz, dyaw)."


class UAVFlowAdapter(NavVLASourceAdapter):
    name = "uav_flow"

    def __init__(
        self,
        *,
        media_cache_root: str | Path | None = None,
        variant: str = "real",
        fps: float = 5.0,
        action_horizon: int = 8,
        instruction_field: str = "instruction",
        reuse_media_cache: bool = False,
        load_workers: int | None = None,
    ) -> None:
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.variant = normalize_variant(variant)
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.instruction_field = instruction_field
        self.reuse_media_cache = bool(reuse_media_cache)
        self.load_workers = load_workers
        self.summary: dict[str, Any] = {"rejected_episodes": 0, "rejections": []}

    def configure(
        self,
        *,
        media_cache_root: str | Path | None = None,
        variant: str = "real",
        fps: float = 5.0,
        action_horizon: int = 8,
        instruction_field: str = "instruction",
        reuse_media_cache: bool = False,
        load_workers: int | None = None,
        **kwargs: Any,
    ) -> "UAVFlowAdapter":
        super().configure(**kwargs)
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.variant = normalize_variant(variant)
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.instruction_field = str(instruction_field)
        self.reuse_media_cache = bool(reuse_media_cache)
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
        root = Path(source_root)
        parquet_paths = sorted(root.glob(f"{split}-*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"no UAV-Flow parquet shards found under {root} for split={split}")
        media_cache_root = resolve_media_cache_root(root, media_cache_root=self.media_cache_root, variant=self.variant)
        self.summary = {"rejected_episodes": 0, "rejections": []}

        resolved_workers = resolve_load_workers(self.load_workers if load_workers is None else load_workers)
        if resolved_workers == 1 or len(parquet_paths) == 1:
            episodes: list[NavVLAEpisode] = []
            for parquet_path in parquet_paths:
                for group in iter_episode_groups(parquet_path, include_image=not self.reuse_media_cache):
                    if max_episodes is not None and len(episodes) >= max_episodes:
                        return episodes
                    try:
                        episode = build_episode(
                            group,
                            source_root=root,
                            parquet_path=parquet_path,
                            media_cache_root=media_cache_root,
                            task_index=len(episodes),
                            split=split,
                            variant=self.variant,
                            fps=self.fps,
                            action_horizon=self.action_horizon,
                            instruction_field=self.instruction_field,
                            reuse_media_cache=self.reuse_media_cache,
                        )
                    except EmptyInstructionError:
                        self.summary["rejected_episodes"] += 1
                        self.summary["rejections"].append(
                            {
                                "source_id": str(group["source_id"]),
                                "source_parquet": str(parquet_path),
                                "reason": "empty_instruction",
                            }
                        )
                        continue
                    episodes.append(episode)
            if not episodes:
                raise FileNotFoundError(f"no UAV-Flow episodes found under {root}")
            return episodes

        max_workers = min(resolved_workers, len(parquet_paths))
        jobs = [
            (
                shard_index,
                str(parquet_path),
                str(root),
                str(media_cache_root),
                split,
                self.variant,
                self.fps,
                self.action_horizon,
                self.instruction_field,
                self.reuse_media_cache,
            )
            for shard_index, parquet_path in enumerate(parquet_paths)
        ]
        shard_results: dict[int, tuple[list[NavVLAEpisode], list[dict[str, Any]]]] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for shard_index, shard_episodes, rejections in executor.map(_load_shard_job, jobs):
                shard_results[shard_index] = (shard_episodes, rejections)

        episodes = []
        for shard_index in range(len(parquet_paths)):
            shard_episodes, rejections = shard_results[shard_index]
            self.summary["rejected_episodes"] += len(rejections)
            self.summary["rejections"].extend(rejections)
            for episode in shard_episodes:
                if max_episodes is not None and len(episodes) >= max_episodes:
                    break
                episodes.append(replace(episode, episode_id=f"{len(episodes):05d}"))
            if max_episodes is not None and len(episodes) >= max_episodes:
                break

        if not episodes:
            raise FileNotFoundError(f"no UAV-Flow episodes found under {root}")
        return [
            replace(episode, task=replace(episode.task, task_index=task_index))
            for task_index, episode in enumerate(episodes)
        ]


class EmptyInstructionError(ValueError):
    pass


def _load_parquet_episodes(
    parquet_path: Path,
    *,
    source_root: Path,
    media_cache_root: Path,
    split: str,
    variant: str,
    fps: float,
    action_horizon: int,
    instruction_field: str,
    reuse_media_cache: bool,
) -> tuple[list[NavVLAEpisode], list[dict[str, Any]]]:
    episodes: list[NavVLAEpisode] = []
    rejections: list[dict[str, Any]] = []
    for group in iter_episode_groups(parquet_path, include_image=not reuse_media_cache):
        try:
            episode = build_episode(
                group,
                source_root=source_root,
                parquet_path=parquet_path,
                media_cache_root=media_cache_root,
                task_index=len(episodes),
                split=split,
                variant=variant,
                fps=fps,
                action_horizon=action_horizon,
                instruction_field=instruction_field,
                reuse_media_cache=reuse_media_cache,
            )
        except EmptyInstructionError:
            rejections.append(
                {
                    "source_id": str(group["source_id"]),
                    "source_parquet": str(parquet_path),
                    "reason": "empty_instruction",
                }
            )
            continue
        episodes.append(episode)
    return episodes, rejections


def _load_shard_job(
    job: tuple[int, str, str, str, str, str, float, int, str, bool],
) -> tuple[int, list[NavVLAEpisode], list[dict[str, Any]]]:
    (
        shard_index,
        parquet_path_str,
        source_root_str,
        media_cache_root_str,
        split,
        variant,
        fps,
        action_horizon,
        instruction_field,
        reuse_media_cache,
    ) = job
    episodes, rejections = _load_parquet_episodes(
        Path(parquet_path_str),
        source_root=Path(source_root_str),
        media_cache_root=Path(media_cache_root_str),
        split=split,
        variant=variant,
        fps=fps,
        action_horizon=action_horizon,
        instruction_field=instruction_field,
        reuse_media_cache=reuse_media_cache,
    )
    return shard_index, episodes, rejections


def normalize_variant(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"real", "uav-flow-real"}:
        return "real"
    if normalized in {"sim", "simulation", "uav-flow-sim"}:
        return "sim"
    raise ValueError(f"unsupported UAV-Flow variant: {value}")


def resolve_uav_flow_source_root(root: str | Path, variant: str) -> Path:
    root = Path(root)
    variant = normalize_variant(variant)
    if root.name == ("UAV-Flow-Real" if variant == "real" else "UAV-Flow-Sim"):
        return root
    child = root / ("UAV-Flow-Real" if variant == "real" else "UAV-Flow-Sim")
    return child


def resolve_media_cache_root(source_root: Path, *, media_cache_root: str | Path | None, variant: str) -> Path:
    normalized_variant = normalize_variant(variant)
    if media_cache_root is not None:
        cache_root = Path(media_cache_root)
        variant_root = cache_root / normalized_variant
        if variant_root.is_dir():
            return variant_root
        return cache_root
    family_root = source_root.parent if source_root.name.startswith("UAV-Flow-") else source_root
    return family_root / ".navvla_media_cache" / normalized_variant


def iter_episode_groups(parquet_path: Path, *, include_image: bool = True):
    columns = ["id", "frame_idx", "log"]
    if include_image:
        columns.insert(2, "image")
    table = pq.read_table(parquet_path, columns=columns)
    ids = table.column("id").to_pylist()
    frame_indices = table.column("frame_idx").to_pylist()
    images = table.column("image").to_pylist() if include_image else [None] * len(ids)
    logs = table.column("log").to_pylist()
    groups: dict[str, list[dict[str, Any]]] = {}
    for row_index, (source_id, frame_idx, image, log_text) in enumerate(zip(ids, frame_indices, images, logs)):
        source_id = str(source_id)
        groups.setdefault(source_id, []).append({"row_index": row_index, "frame_idx": int(frame_idx), "image": image, "log": log_text})
    for source_id in sorted(groups):
        yield {"source_id": source_id, "rows": sorted(groups[source_id], key=lambda row: (row["frame_idx"], row["row_index"]))}


def build_episode(
    group: dict[str, Any],
    *,
    source_root: Path,
    parquet_path: Path,
    media_cache_root: Path,
    task_index: int,
    split: str,
    variant: str,
    fps: float,
    action_horizon: int,
    instruction_field: str,
    reuse_media_cache: bool,
) -> NavVLAEpisode:
    source_id = str(group["source_id"])
    rows = sorted(group["rows"], key=lambda row: row["frame_idx"])
    if not rows:
        raise ValueError(f"empty UAV-Flow episode in {parquet_path}: {source_id}")
    log = json.loads(rows[0]["log"])
    raw_logs = list(log.get("raw_logs") or [])
    preprocessed_logs = list(log.get("preprocessed_logs") or [])
    if len(raw_logs) != len(rows) or len(preprocessed_logs) != len(rows):
        raise ValueError(
            f"log/frame length mismatch for {source_id}: rows={len(rows)} raw={len(raw_logs)} preprocessed={len(preprocessed_logs)}"
        )
    instruction = choose_instruction(log, instruction_field=instruction_field)
    instruction_unified = choose_instruction(log, instruction_field="instruction_unified")
    task = NavVLATaskSpec(
        task_index=task_index,
        instruction=instruction,
        task_type="navigation",
        task_subtype=uav_flow_task_subtype(instruction_unified),
        platform_text=PLATFORM_TEXT,
        dataset_source=f"uav_flow_{variant}",
        scene_id=f"uav_flow_{variant}",
    )

    poses = [pose4_from_raw_state(raw_state, variant=variant) for raw_state in raw_logs]
    raw_timestamps = [timestamp_for_raw_state(raw_state, frame_index=idx, fps=fps)[0] for idx, raw_state in enumerate(raw_logs)]
    timestamp_anchor = raw_timestamps[0] if raw_timestamps else 0.0
    frames = []
    for frame_position, row in enumerate(rows):
        frame_idx = int(row["frame_idx"])
        if frame_idx != frame_position:
            raise ValueError(f"non-contiguous frame_idx for {source_id}: expected {frame_position}, got {frame_idx}")
        image_path = materialize_image(
            row.get("image"),
            media_cache_root=media_cache_root,
            source_id=source_id,
            frame_idx=frame_idx,
            reuse_media_cache=reuse_media_cache,
        )
        raw_timestamp, _timestamp_source = timestamp_for_raw_state(raw_logs[frame_idx], frame_index=frame_idx, fps=fps)
        timestamp = float(raw_timestamp) - float(timestamp_anchor)
        action = action_chunk_for_frame(poses, frame_idx=frame_idx, horizon=action_horizon)
        source_metadata = {"instruction_unified": instruction_unified}
        frames.append(
            NavVLAFrame(
                frame_index=frame_idx,
                timestamp=timestamp,
                media_paths={"front_image": image_path},
                state=pose4_from_raw_state(raw_logs[frame_idx], variant=variant),
                action=action,
                action_available=bool(action),
                source_frame_index=frame_idx,
                source_metadata=source_metadata,
            )
        )

    return NavVLAEpisode(
        episode_id=f"{int(task_index):05d}",
        trajectory_id=source_id,
        task=task,
        frames=frames,
        cameras=[UAV_FLOW_CAMERA],
        split=split,
    )


def pose4_from_raw_state(raw_state: list[float] | tuple[float, ...], *, variant: str = "sim") -> list[float]:
    if len(raw_state) < 5:
        raise ValueError(f"raw state must contain at least [x,y,z,*,yaw], got length {len(raw_state)}")
    position_scale = uav_flow_position_scale(variant)
    return [
        float(raw_state[0]) * position_scale,
        float(raw_state[1]) * position_scale,
        -float(raw_state[2]) * position_scale,
        math.radians(float(raw_state[4])),
    ]


def uav_flow_position_scale(variant: str) -> float:
    normalized_variant = normalize_variant(variant)
    if normalized_variant == "real":
        return 1.0
    return 0.01


def action_chunk_for_frame(poses: list[list[float]], *, frame_idx: int, horizon: int) -> list[list[float]]:
    current = poses[frame_idx]
    chunk = []
    for future_idx in range(frame_idx + 1, min(len(poses), frame_idx + 1 + horizon)):
        action = [_clean_float(value) for value in body_frame_action_from_pose(current, poses[future_idx]).astype(float).tolist()]
        chunk.append(action)
    return chunk


def _clean_float(value: float) -> float:
    value = float(value)
    return 0.0 if abs(value) < 1e-7 else value


def timestamp_for_raw_state(raw_state: list[Any], *, frame_index: int, fps: float) -> tuple[float, str]:
    if len(raw_state) > 6 and raw_state[6] is not None:
        return float(raw_state[6]), "raw_logs_timestamp"
    if fps <= 0:
        raise ValueError(f"fps must be positive for timestamp fallback, got {fps}")
    return float(frame_index) / float(fps), "frame_index_over_fps"


def uav_flow_task_subtype(instruction_unified: str) -> str:
    text = instruction_unified.strip()
    if not text:
        return "Unknown"
    lowered = text.lower()
    ordered_rules = [
        ("Ascend", ("ascend", "climb", "rise")),
        ("Descend", ("descend", "drop", "lower")),
        ("Approach", ("approach",)),
        ("Retreat", ("retreat", "back away", "move away")),
        ("Surround", ("surround", "circle", "orbit")),
        ("Rotate", ("rotate", "spin")),
        ("Turn", ("turn",)),
        ("Shift", ("shift", "translate", "strafe")),
        ("Pass", ("pass", "fly through", "go through", "through the")),
        ("Land", ("land", "landing")),
        ("Move", ("move", "navigate", "proceed", "toward", "go ", "fly ")),
    ]
    for subtype, patterns in ordered_rules:
        if any(pattern in lowered for pattern in patterns):
            return subtype
    return text.split(maxsplit=1)[0].strip(" ,.;:") or "Unknown"


def choose_instruction(log: dict[str, Any], *, instruction_field: str) -> str:
    if instruction_field not in {"instruction", "instruction_unified"}:
        raise ValueError(f"unsupported instruction field: {instruction_field}")
    value = str(log.get(instruction_field) or "").strip()
    if value:
        return value
    fallback_key = "instruction" if instruction_field == "instruction_unified" else "instruction_unified"
    fallback = str(log.get(fallback_key) or "").strip()
    if fallback:
        return fallback
    raise EmptyInstructionError("both instruction and instruction_unified are empty")


def text_anomaly(log: dict[str, Any]) -> dict[str, bool]:
    instruction = str(log.get("instruction") or "")
    instruction_unified = str(log.get("instruction_unified") or "")
    return {
        "empty_instruction": not bool(instruction.strip()),
        "empty_instruction_unified": not bool(instruction_unified.strip()),
        "contains_xx": "xx" in instruction.lower() or "xx" in instruction_unified.lower(),
    }


def materialize_image(image_value: Any, *, media_cache_root: Path, source_id: str, frame_idx: int, reuse_media_cache: bool = False) -> Path:
    episode_dir = media_cache_root / sanitize_episode_id(source_id)
    path = episode_dir / f"{frame_idx:06d}.png"
    if reuse_media_cache:
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing cached image for {source_id} frame {frame_idx}: {path}")
        return path
    if not isinstance(image_value, dict):
        raise ValueError(f"expected parquet image struct for {source_id} frame {frame_idx}")
    image_bytes = image_value.get("bytes")
    if not image_bytes:
        raise ValueError(f"missing image bytes for {source_id} frame {frame_idx}")
    episode_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != len(image_bytes):
        path.write_bytes(image_bytes)
    return path


def sanitize_episode_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not sanitized:
        raise ValueError("empty source id after sanitization")
    return sanitized


register_adapter(UAVFlowAdapter())
