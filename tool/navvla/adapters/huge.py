from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tool.navvla.adapters.base import NavVLASourceAdapter, register_adapter
from tool.navvla.workers import resolve_workers as resolve_load_workers
from tool.navvla.lerobot_v3_writer import write_navvla_lerobot_dataset
from tool.navvla.schema import NavVLACameraSpec, NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec
from tool.navvla.statistics import body_frame_action_from_pose


HUGE_CAMERA = NavVLACameraSpec(
    name="front",
    video_key="front_image",
    viewpoint_type="front",
    azimuth_rad=0.0,
    calibration_status="unknown",
)
PLATFORM_TEXT = "Platform: UAV. Task: instruction-conditioned navigation. Action: local 3D waypoints (dx, dy, dz, dyaw)."


class HUGEAdapter(NavVLASourceAdapter):
    name = "huge"

    def __init__(
        self,
        *,
        media_cache_root: str | Path | None = None,
        fps: float = 5.0,
        action_horizon: int = 8,
        reuse_media_cache: bool = False,
    ) -> None:
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.reuse_media_cache = bool(reuse_media_cache)
        self.summary: dict[str, Any] = {}
        self.load_workers: int | None = None

    def configure(
        self,
        *,
        media_cache_root: str | Path | None = None,
        reuse_media_cache: bool = False,
        fps: float = 5.0,
        action_horizon: int = 8,
        load_workers: int | None = None,
        **kwargs: Any,
    ) -> "HUGEAdapter":
        super().configure(**kwargs)
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.reuse_media_cache = bool(reuse_media_cache)
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
        source_split = source_split_name(split)
        target_split = target_split_name(source_split)
        split_root = source_root / source_split
        if not split_root.is_dir():
            raise FileNotFoundError(f"HUGE split root not found: {split_root}")

        tasks = load_huge_tasks(split_root / "meta" / "tasks.jsonl")
        episode_metadata = load_huge_episode_metadata(split_root / "meta" / "episodes.jsonl")
        parquet_paths = sorted((split_root / "data").glob("chunk-*/*.parquet"))
        if max_episodes is not None:
            parquet_paths = parquet_paths[: int(max_episodes)]
        if not parquet_paths:
            raise FileNotFoundError(f"no HUGE parquet episodes found under {split_root / 'data'}")

        media_cache_root = resolve_media_cache_root(source_root, media_cache_root=self.media_cache_root)
        jobs = []
        for parquet_path in parquet_paths:
            source_episode_index = episode_index_from_path(parquet_path)
            metadata = episode_metadata.get(source_episode_index)
            if metadata is None:
                raise ValueError(f"HUGE episode metadata missing for episode_index={source_episode_index}")
            jobs.append(
                (
                    str(source_root),
                    str(parquet_path),
                    str(media_cache_root),
                    source_split,
                    target_split,
                    self.action_horizon,
                    self.reuse_media_cache,
                    tasks,
                    metadata,
                )
            )

        workers = resolve_load_workers(load_workers)
        if workers == 1 or len(jobs) == 1:
            episodes = [_load_huge_episode_job(job) for job in jobs]
        else:
            max_workers = min(workers, len(jobs))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                episodes = list(executor.map(_load_huge_episode_job, jobs, chunksize=8))

        self.summary = {
            "source_root": str(source_root),
            "source_split": source_split,
            "target_split": target_split,
            "source_episodes": len(episodes),
            "source_frames": sum(len(episode.frames) for episode in episodes),
            "media_cache_root": str(media_cache_root),
            "reuse_media_cache": self.reuse_media_cache,
        }
        return episodes

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
        load_workers: int | None = None,
        write_visual_token_cache: bool = True,
        visual_token_profile: Any | None = None,
        visual_token_encoder: Any | None = None,
        visual_token_encoder_factory: Any | None = None,
        episodes_per_file: int = 20,
        files_per_chunk: int = 50,
    ) -> dict[str, Any]:
        episodes = self.load_episodes(
            source_root,
            split=split,
            max_episodes=max_episodes,
            load_workers=self.load_workers if load_workers is None else load_workers,
        )
        target_split = target_split_name(split)
        spec = NavVLADatasetSpec(
            dataset_name=dataset_name,
            fps=fps,
            action_horizon=action_horizon,
            action_dim=4,
            state_dim=4,
            control_frequency_hz=control_frequency_hz if control_frequency_hz is not None else fps,
            context_policy_version=context_policy_version,
            cache_policy_version=cache_policy_version,
            split=target_split,
            episodes_per_file=episodes_per_file,
            files_per_chunk=files_per_chunk,
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
        )
        summary["adapter_summary"] = self.summary
        return summary


def source_split_name(split: str) -> str:
    normalized = split.strip().lower()
    aliases = {
        "train": "train",
        "vln_train": "train",
        "test_seen": "test_seen",
        "val_seen": "test_seen",
        "vln_val_seen": "test_seen",
        "test_unseen": "test_unseen",
        "val_unseen": "test_unseen",
        "vln_val_unseen": "test_unseen",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported HUGE split: {split}") from exc


def target_split_name(split: str) -> str:
    source_split = source_split_name(split)
    return {
        "train": "vln_train",
        "test_seen": "vln_val_seen",
        "test_unseen": "vln_val_unseen",
    }[source_split]


def resolve_media_cache_root(source_root: Path, *, media_cache_root: str | Path | None) -> Path:
    if media_cache_root is not None:
        return Path(media_cache_root)
    return source_root / ".navvla_media_cache" / "huge"


def load_huge_tasks(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"HUGE tasks file not found: {path}")
    tasks: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_index = int(row["task_index"])
            task_text = str(row.get("task") or "").strip()
            if not task_text:
                raise ValueError(f"HUGE task_index={task_index} has empty task text")
            tasks[task_index] = row
    if not tasks:
        raise ValueError(f"HUGE tasks file is empty: {path}")
    return tasks


def load_huge_episode_metadata(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"HUGE episodes metadata file not found: {path}")
    episodes: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            episodes[int(row["episode_index"])] = row
    if not episodes:
        raise ValueError(f"HUGE episodes metadata file is empty: {path}")
    return episodes


def _load_huge_episode_job(job: tuple[Any, ...]) -> NavVLAEpisode:
    (
        source_root_str,
        parquet_path_str,
        media_cache_root_str,
        source_split,
        target_split,
        action_horizon,
        reuse_media_cache,
        tasks,
        metadata,
    ) = job
    return build_huge_episode(
        source_root=Path(source_root_str),
        parquet_path=Path(parquet_path_str),
        media_cache_root=Path(media_cache_root_str),
        source_split=source_split,
        target_split=target_split,
        action_horizon=int(action_horizon),
        reuse_media_cache=bool(reuse_media_cache),
        tasks=tasks,
        episode_metadata=metadata,
    )


def build_huge_episode(
    *,
    source_root: Path,
    parquet_path: Path,
    media_cache_root: Path,
    source_split: str,
    target_split: str,
    action_horizon: int,
    reuse_media_cache: bool,
    tasks: dict[int, dict[str, Any]],
    episode_metadata: dict[str, Any],
) -> NavVLAEpisode:
    columns = ["state", "actions", "env_id", "timestamp", "frame_index", "episode_index", "task_index"]
    if not reuse_media_cache:
        columns.insert(0, "image")
    table = pq.read_table(parquet_path, columns=columns)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"empty HUGE episode parquet: {parquet_path}")

    source_episode_index = int(rows[0]["episode_index"])
    expected_episode_index = episode_index_from_path(parquet_path)
    if source_episode_index != expected_episode_index:
        raise ValueError(
            f"HUGE episode_index mismatch for {parquet_path}: row={source_episode_index} filename={expected_episode_index}"
        )
    if any(int(row["episode_index"]) != source_episode_index for row in rows):
        raise ValueError(f"HUGE episode has mixed episode_index values: {parquet_path}")

    frame_indices = [int(row["frame_index"]) for row in rows]
    if frame_indices != list(range(len(rows))):
        raise ValueError(f"HUGE episode has non-contiguous frame_index values: {parquet_path}")

    task_indices = {int(row["task_index"]) for row in rows}
    if len(task_indices) != 1:
        raise ValueError(f"HUGE episode has mixed task_index values: {parquet_path}")
    task_index = task_indices.pop()
    task_row = tasks.get(task_index)
    if task_row is None:
        raise ValueError(f"HUGE episode references missing task_index={task_index}: {parquet_path}")

    env_ids = {str(row["env_id"]) for row in rows}
    if len(env_ids) != 1:
        raise ValueError(f"HUGE episode has mixed env_id values: {parquet_path}")
    scene_id = env_ids.pop()

    metadata = episode_metadata
    if int(metadata.get("episode_index", -1)) != source_episode_index:
        raise ValueError(f"HUGE episode metadata mismatch for episode_index={source_episode_index}")
    if int(metadata.get("length", -1)) != len(rows):
        raise ValueError(
            f"HUGE episode length mismatch for {parquet_path}: rows={len(rows)} metadata={metadata.get('length')}"
        )
    if str(metadata.get("env_id")) != scene_id:
        raise ValueError(f"HUGE episode env_id mismatch for {parquet_path}: rows={scene_id} metadata={metadata.get('env_id')}")

    instruction = str((metadata.get("tasks") or [""])[0]).strip()
    task_text = str(task_row.get("task") or "").strip()
    if not instruction:
        raise ValueError(f"HUGE episode has empty instruction: {parquet_path}")
    if instruction != task_text:
        raise ValueError(f"HUGE task text mismatch for {parquet_path}: task_index={task_index}")

    states = [state4(row["state"], parquet_path=parquet_path, frame_index=int(row["frame_index"])) for row in rows]
    source_actions = [action4(row["actions"], parquet_path=parquet_path, frame_index=int(row["frame_index"])) for row in rows]
    episode_id = f"{source_episode_index:06d}"
    task = NavVLATaskSpec(
        task_index=task_index,
        instruction=instruction,
        task_type="navigation",
        task_subtype=str(task_row.get("task_id") or metadata.get("task_id") or "huge"),
        platform_text=PLATFORM_TEXT,
        dataset_source="huge",
        scene_id=scene_id,
    )

    frames: list[NavVLAFrame] = []
    for row_index, row in enumerate(rows):
        frame_index = int(row["frame_index"])
        image_path = materialize_huge_image(
            None if reuse_media_cache else row.get("image"),
            media_cache_root=media_cache_root,
            target_split=target_split,
            episode_id=episode_id,
            frame_index=frame_index,
            reuse_media_cache=reuse_media_cache,
        )
        source_metadata = {
            "source_dataset": "huge",
            "source_split": source_split,
            "source_episode_index": source_episode_index,
            "source_parquet": str(parquet_path.relative_to(source_root)),
            "source_image_path": None if reuse_media_cache else image_source_path(row.get("image")),
            "source_world_delta_action": source_actions[row_index],
            "source_env_id": scene_id,
            "source_task_id": str(task_row.get("task_id") or metadata.get("task_id") or ""),
        }
        frames.append(
            NavVLAFrame(
                frame_index=frame_index,
                timestamp=float(row["timestamp"]),
                media_paths={"front_image": image_path},
                state=states[row_index],
                action=action_chunk_for_frame(states, frame_idx=row_index, horizon=action_horizon),
                action_available=True,
                source_frame_index=frame_index,
                source_metadata=source_metadata,
            )
        )

    return NavVLAEpisode(
        episode_id=episode_id,
        trajectory_id=f"{source_split}/episode_{source_episode_index:06d}",
        task=task,
        frames=frames,
        cameras=[HUGE_CAMERA],
        split=target_split,
    )


def episode_index_from_path(path: Path) -> int:
    stem = path.stem
    prefix = "episode_"
    if not stem.startswith(prefix):
        raise ValueError(f"HUGE parquet filename must match episode_XXXXXX.parquet: {path}")
    return int(stem[len(prefix) :])


def state4(value: Any, *, parquet_path: Path, frame_index: int) -> list[float]:
    values = list(value or [])
    if len(values) != 4:
        raise ValueError(f"HUGE state must have length 4 at {parquet_path} frame {frame_index}, got {len(values)}")
    return [_clean_float(item) for item in values]


def action4(value: Any, *, parquet_path: Path, frame_index: int) -> list[float]:
    values = list(value or [])
    if len(values) != 4:
        raise ValueError(f"HUGE source action must have length 4 at {parquet_path} frame {frame_index}, got {len(values)}")
    return [_clean_float(item) for item in values]


def action_chunk_for_frame(poses: list[list[float]], *, frame_idx: int, horizon: int) -> list[list[float]]:
    current = poses[frame_idx]
    chunk = []
    for future_idx in range(frame_idx + 1, min(len(poses), frame_idx + 1 + horizon)):
        action = body_frame_action_from_pose(current, poses[future_idx]).astype(float).tolist()
        chunk.append([_clean_float(value) for value in action])
    return chunk


def materialize_huge_image(
    image_value: Any,
    *,
    media_cache_root: Path,
    target_split: str,
    episode_id: str,
    frame_index: int,
    reuse_media_cache: bool,
) -> Path:
    path = media_cache_root / target_split / episode_id / f"{frame_index:06d}.png"
    if reuse_media_cache:
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing cached HUGE image for episode {episode_id} frame {frame_index}: {path}")
        return path
    if not isinstance(image_value, dict):
        raise ValueError(f"expected HUGE image struct for episode {episode_id} frame {frame_index}")
    image_bytes = image_value.get("bytes")
    if not image_bytes:
        raise ValueError(f"missing HUGE image bytes for episode {episode_id} frame {frame_index}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != len(image_bytes):
        path.write_bytes(image_bytes)
    return path


def image_source_path(image_value: Any) -> str | None:
    if not isinstance(image_value, dict):
        return None
    value = image_value.get("path")
    return None if value is None else str(value)


def _clean_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"HUGE numeric value must be finite, got {value}")
    return 0.0 if abs(value) < 1e-7 else value


register_adapter(HUGEAdapter())
