from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from navvla_conversion.context_index import DEFAULT_CONTEXT_TOKEN_BUDGET, resolve_context_index_paths


MEDIA_DECODE_MODES = {"none", "sampled", "all"}
DATA_REQUIRED_COLUMNS = {
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
    "observation.state",
    "action",
    "action.padding_mask",
    "next.done",
    "sample.action_available",
    "context.index_key",
    "source_frame_index",
    "index",
}
CONTEXT_REQUIRED_COLUMNS = {
    "index",
    "bats_k",
    "history_offset",
    "history_count",
    "long_memory_offset",
    "long_memory_count",
}


def validate_navvla_lerobot_dataset(
    dataset_root: str | Path,
    *,
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    check_media_decode: str = "none",
    media_decode_sample: int = 3,
    data_rows_per_shard: int = 3,
    sample_seed: int = 42,
) -> dict[str, Any]:
    """Validate standalone NavVLA LeRobot v3 artifacts without model imports.

    The validator checks the complete metadata and shard inventories, all row counts,
    episode/task/index relationships, compact context storage, and video references.
    Row payloads and video decoding are sampled deterministically unless full media
    decoding is requested.
    """

    root = Path(dataset_root).resolve()
    if check_media_decode not in MEDIA_DECODE_MODES:
        raise ValueError(f"check_media_decode must be one of {sorted(MEDIA_DECODE_MODES)}")
    if data_rows_per_shard < 1:
        raise ValueError("data_rows_per_shard must be positive")

    info = _read_json(root / "meta" / "info.json")
    _read_json(root / "meta" / "modality.json")
    cameras = _read_json(root / "meta" / "navvla_cameras.json")
    schema_ext = _read_json(root / "meta" / "navvla_schema_ext.json")
    statistics = _read_json(root / "dataset_statistics.json")
    if not statistics:
        raise ValueError("dataset_statistics.json contains no dataset entries")

    data_report = _validate_data_shards(
        root / "data",
        rows_per_shard=int(data_rows_per_shard),
        sample_seed=int(sample_seed),
        info=info,
    )
    total_rows = int(data_report["rows"])
    episodes = _read_parquet_shards(root / "meta" / "episodes")
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    _validate_relations(info=info, data_report=data_report, episodes=episodes, tasks=tasks)

    frame_metadata_path = root / "meta" / "navvla_frame_metadata.jsonl"
    frame_metadata_rows = _count_jsonl(frame_metadata_path) if frame_metadata_path.exists() else 0
    if frame_metadata_path.exists() and frame_metadata_rows != total_rows:
        raise ValueError(
            f"frame metadata row count mismatch: {frame_metadata_rows} != {total_rows}"
        )

    context_report = _validate_context(root, token_budget=int(token_budget), expected_rows=total_rows)
    video_report = _validate_videos(
        root,
        info=info,
        mode=check_media_decode,
        sample_count=int(media_decode_sample),
        sample_seed=int(sample_seed),
    )
    state_mode = str((info.get("navvla") or {}).get("state_mode") or "unknown")
    return {
        "dataset_root": str(root),
        "valid": True,
        "state_contract": {
            "state_mode": state_mode,
            "state_dim": int((info.get("navvla") or {}).get("state_dim", 0)),
            "world_pose_assumed": False,
        },
        "counts": {
            "frames": total_rows,
            "episodes": int(len(episodes)),
            "tasks": int(len(tasks)),
            "frame_metadata": int(frame_metadata_rows),
        },
        "artifacts": {
            "data": data_report["artifacts"],
            "context": context_report,
            "videos": video_report,
            "schema_ext": schema_ext,
            "cameras": len(cameras),
        },
    }


def _validate_data_shards(
    data_root: Path,
    *,
    rows_per_shard: int,
    sample_seed: int,
    info: dict[str, Any],
) -> dict[str, Any]:
    paths = sorted(data_root.glob("chunk-*/part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards found under {data_root}")
    expected_schema = None
    expected_index = 0
    episode_counts: dict[int, int] = {}
    task_indices: set[int] = set()
    sampled_rows: list[dict[str, Any]] = []
    sampled_indices: list[int] = []
    rng = random.Random(sample_seed)

    join_columns = ["index", "episode_index", "frame_index", "task_index", "context.index_key"]
    for path in paths:
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        if rows <= 0:
            raise ValueError(f"zero-row parquet shard: {path}")
        schema = parquet.schema_arrow
        if expected_schema is None:
            expected_schema = schema
        elif schema != expected_schema:
            raise ValueError(f"parquet schema mismatch: {path}")
        missing = DATA_REQUIRED_COLUMNS - set(schema.names)
        if missing:
            raise ValueError(f"data shard is missing {sorted(missing)}: {path}")

        joins = pd.read_parquet(path, columns=join_columns)
        indices = joins["index"].astype(int).to_numpy()
        wanted = np.arange(expected_index, expected_index + rows, dtype=np.int64)
        if indices.shape != wanted.shape or not np.array_equal(indices, wanted):
            raise ValueError(f"global data indexes are not contiguous at {path}")
        expected_index += rows
        if joins["context.index_key"].isna().any():
            raise ValueError(f"null context.index_key found in {path}")
        for episode_index, count in joins["episode_index"].value_counts().items():
            key = int(episode_index)
            episode_counts[key] = episode_counts.get(key, 0) + int(count)
        task_indices.update(int(value) for value in joins["task_index"].unique())

        local = _sample_indices(rows, rows_per_shard, rng=rng)
        if local:
            payload = pd.read_parquet(path).iloc[local]
            sampled_rows.extend(payload.to_dict("records"))
            sampled_indices.extend(int(indices[index]) for index in local)

    _validate_sampled_rows(sampled_rows, info=info)
    return {
        "rows": expected_index,
        "episode_counts": episode_counts,
        "task_indices": task_indices,
        "artifacts": {
            "parquet_shards": {"scope": "full", "checked": len(paths), "total": len(paths)},
            "rows": {
                "scope": "sampled",
                "checked": len(sampled_rows),
                "total": expected_index,
                "sample_indices": sampled_indices,
                "seed": sample_seed,
            },
        },
    }


def _validate_sampled_rows(rows: list[dict[str, Any]], *, info: dict[str, Any]) -> None:
    navvla = info.get("navvla") or {}
    state_dim = int(navvla.get("state_dim", 4))
    action_horizon = int(navvla.get("action_horizon", 8))
    action_dim = int(navvla.get("action_dim", 4))
    if state_dim != 4:
        raise ValueError(f"NavVLA state_dim must be 4, got {state_dim}")
    for row in rows:
        state = np.asarray(row["observation.state"], dtype=float).reshape(-1)
        if state.shape != (state_dim,) or not np.all(np.isfinite(state)):
            raise ValueError(f"invalid observation.state at index={row.get('index')}")
        action = _flatten_numeric(row["action"])
        mask = np.asarray(row["action.padding_mask"], dtype=bool).reshape(-1)
        if action.shape != (action_horizon * action_dim,) or not np.all(np.isfinite(action)):
            raise ValueError(f"invalid action at index={row.get('index')}")
        if mask.shape != (action_horizon,):
            raise ValueError(f"invalid action.padding_mask at index={row.get('index')}")
        if not np.isfinite(float(row["timestamp"])):
            raise ValueError(f"non-finite timestamp at index={row.get('index')}")


def _flatten_numeric(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        return np.concatenate(
            [np.asarray(item, dtype=float).reshape(-1) for item in array],
            axis=0,
        )
    return array.astype(float, copy=False).reshape(-1)


def _validate_relations(
    *,
    info: dict[str, Any],
    data_report: dict[str, Any],
    episodes: pd.DataFrame,
    tasks: pd.DataFrame,
) -> None:
    required_episode_columns = {
        "episode_index",
        "episode_id",
        "scene_id",
        "length",
        "data/chunk_index",
        "data/file_index",
    }
    missing = required_episode_columns - set(episodes.columns)
    if missing:
        raise ValueError(f"episode metadata is missing {sorted(missing)}")
    total_rows = int(data_report["rows"])
    if int(info["total_frames"]) != total_rows:
        raise ValueError("info.total_frames does not match data rows")
    if int(info["total_episodes"]) != len(episodes):
        raise ValueError("info.total_episodes does not match episode rows")
    if int(episodes["length"].sum()) != total_rows:
        raise ValueError("sum of episode lengths does not match data rows")
    if episodes["episode_index"].duplicated().any():
        raise ValueError("duplicate episode_index in episode metadata")
    invalid_scene = episodes["scene_id"].isna() | episodes["scene_id"].astype(str).str.strip().eq("")
    if invalid_scene.any():
        raise ValueError("episode metadata contains an invalid scene_id")
    metadata_lengths = {
        int(row.episode_index): int(row.length)
        for row in episodes[["episode_index", "length"]].itertuples(index=False)
    }
    if metadata_lengths != data_report["episode_counts"]:
        raise ValueError("episode metadata lengths do not match data rows")
    task_indices = set(int(value) for value in tasks["task_index"].tolist())
    if not data_report["task_indices"].issubset(task_indices):
        raise ValueError("some data task_index values do not resolve")


def _validate_context(root: Path, *, token_budget: int, expected_rows: int) -> dict[str, Any]:
    paths = resolve_context_index_paths(root, token_budget=token_budget)
    parquet = pq.ParquetFile(paths.meta_path)
    missing = CONTEXT_REQUIRED_COLUMNS - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"context metadata is missing {sorted(missing)}")
    rows = int(parquet.metadata.num_rows)
    if rows != expected_rows:
        raise ValueError(f"context row count mismatch: {rows} != {expected_rows}")
    arrays = {}
    for name in (
        "history_frame_index",
        "history_camera_mask",
        "long_memory_frame_index",
        "long_memory_camera_mask",
    ):
        path = paths.arrays_path / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r")
        if array.ndim != 1:
            raise ValueError(f"context array must be one-dimensional: {path}")
        arrays[name] = int(array.shape[0])
    if arrays["history_frame_index"] != arrays["history_camera_mask"]:
        raise ValueError("history context arrays have different lengths")
    if arrays["long_memory_frame_index"] != arrays["long_memory_camera_mask"]:
        raise ValueError("long-memory context arrays have different lengths")
    if rows:
        tail = parquet.read_row_group(parquet.num_row_groups - 1).to_pandas().iloc[-1]
        if int(tail["history_offset"]) + int(tail["history_count"]) > arrays["history_frame_index"]:
            raise ValueError("history context offsets exceed the array length")
        if int(tail["long_memory_offset"]) + int(tail["long_memory_count"]) > arrays["long_memory_frame_index"]:
            raise ValueError("long-memory context offsets exceed the array length")
    if not paths.debug_path.is_file():
        raise FileNotFoundError(paths.debug_path)
    return {
        "scope": "full",
        "token_budget": token_budget,
        "rows": rows,
        "arrays": arrays,
        "meta_path": str(paths.meta_path),
    }


def _validate_videos(
    root: Path,
    *,
    info: dict[str, Any],
    mode: str,
    sample_count: int,
    sample_seed: int,
) -> dict[str, Any]:
    index_path = root / "meta" / "navvla_video_index.parquet"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    parquet = pq.ParquetFile(index_path)
    required = {"index", "video_key", "available", "video_frame_index", "chunk_index", "file_index"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"video index is missing {sorted(missing)}")
    rows = int(parquet.metadata.num_rows)
    video_files = sorted((root / "videos").glob("*/chunk-*/part-*.mp4"))
    decoded = 0
    candidates = video_files
    if mode == "sampled" and len(candidates) > sample_count:
        candidates = random.Random(sample_seed).sample(candidates, max(1, sample_count))
    if mode == "none":
        candidates = []
    for path in candidates:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"failed to open video: {path}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise ValueError(f"video contains no frames: {path}")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"failed to decode video: {path}")
            decoded += 1
        finally:
            capture.release()
    declared_keys = set((info.get("video_path") or {}).keys())
    actual_keys = {path.parents[1].name for path in video_files}
    if actual_keys - declared_keys:
        raise ValueError(f"video directories are not declared in info.json: {sorted(actual_keys - declared_keys)}")
    return {
        "index_rows": rows,
        "files": len(video_files),
        "decode_mode": mode,
        "decoded_files": decoded,
    }


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet_shards(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("chunk-*/part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards found under {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _sample_indices(length: int, count: int, *, rng: random.Random) -> list[int]:
    if count <= 1:
        return [0]
    if length <= count:
        return list(range(length))
    values = {0, length - 1}
    while len(values) < min(length, count):
        values.add(rng.randrange(length))
    return sorted(values)
