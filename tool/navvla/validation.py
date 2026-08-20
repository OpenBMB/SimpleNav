from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from starVLA.model.modules.qwen35_vision import BFLOAT16_BITS_STORAGE_ENCODING
from tool.navvla.context_index import (
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    iter_context_refs,
    load_runtime_context_index,
    resolve_context_index_paths,
)
from tool.navvla.visual_token_cache import MMAP_NPY_VISUAL_TOKEN_FORMAT, NPZ_VISUAL_TOKEN_FORMAT, profile_cache_root

CACHE_REQUIRED_VISUAL_TOKEN_MODES = {"offline_cache", "cached_history_online_current"}
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


def validate_navvla_lerobot_dataset(
    dataset_root: str | Path,
    *,
    visual_token_mode: str = "online_images",
    smoke_load: int = 0,
    smoke_load_all: bool = False,
    required_cameras: list[str] | None = None,
    image_resize: tuple[int, int] | None = None,
    check_media_decode: str = "none",
    media_decode_sample: int = 3,
    visual_token_profile: str = "qwen3_vl_4b_pooled_history",
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    cache_sample_size: int = 10,
    data_rows_per_shard: int = 3,
    sample_seed: int = 42,
) -> dict[str, Any]:
    root = Path(dataset_root)
    visual_token_mode = str(visual_token_mode)
    if visual_token_mode not in {"online_images", "offline_cache", "cached_history_online_current"}:
        raise ValueError(f"unsupported visual_token_mode={visual_token_mode!r}")
    if check_media_decode not in MEDIA_DECODE_MODES:
        raise ValueError(f"check_media_decode must be one of {sorted(MEDIA_DECODE_MODES)}, got {check_media_decode!r}")

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    json.loads((root / "meta" / "modality.json").read_text(encoding="utf-8"))
    cameras = json.loads((root / "meta" / "navvla_cameras.json").read_text(encoding="utf-8"))
    schema_ext = json.loads((root / "meta" / "navvla_schema_ext.json").read_text(encoding="utf-8"))
    statistics = json.loads((root / "dataset_statistics.json").read_text(encoding="utf-8"))
    if not statistics:
        raise ValueError("dataset_statistics.json contains no dataset entries")

    frame_metadata_path = root / "meta" / "navvla_frame_metadata.jsonl"
    conversion_report_path = root / "conversion_report.json"
    conversion_report = json.loads(conversion_report_path.read_text(encoding="utf-8")) if conversion_report_path.exists() else None
    frame_metadata_count, frame_metadata_sample = _read_frame_metadata_inventory(frame_metadata_path)
    data, sampled_data_rows, data_artifacts = _read_data_inventory(
        root / "data",
        rows_per_shard=int(data_rows_per_shard),
        sample_seed=int(sample_seed),
    )
    _reject_legacy_missing_fields_outputs(
        info=info,
        schema_ext=schema_ext,
        conversion_report=conversion_report,
        frame_metadata_sample=frame_metadata_sample,
    )
    state_contract_report = _validate_state_contract(info, sampled_data_rows)
    episodes = _read_parquet_shards(root / "meta" / "episodes")
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    navvla_tasks_by_index = {int(row["task_index"]): row for row in tasks.to_dict("records")}
    legacy_tasks_path = root / "meta" / "navvla_tasks.jsonl"
    if legacy_tasks_path.exists():
        for line in legacy_tasks_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_index = int(row["task_index"])
            navvla_tasks_by_index[task_index] = {**navvla_tasks_by_index.get(task_index, {}), **row}
    navvla_tasks = list(navvla_tasks_by_index.values())
    context_paths = resolve_context_index_paths(root, token_budget=int(token_budget))
    runtime_context = load_runtime_context_index(context_paths)
    context = pd.read_parquet(context_paths.meta_path)
    context_debug = pd.read_parquet(context_paths.debug_path, columns=["index"]) if context_paths.debug_path.exists() else None
    video_index_path = root / "meta" / "navvla_video_index.parquet"
    video_index = pd.read_parquet(video_index_path) if video_index_path.exists() else None

    if int(info["total_frames"]) != len(data):
        raise ValueError("total_frames does not match data rows")
    if int(info["total_episodes"]) != len(episodes):
        raise ValueError("total_episodes does not match episode rows")
    if int(episodes["length"].sum()) != len(data):
        raise ValueError("sum of episode lengths does not match data rows")
    if frame_metadata_path.exists():
        if frame_metadata_count != len(data):
            raise ValueError("frame metadata row count does not match data rows")

    task_indices = set(tasks["task_index"].astype(int).tolist())
    if not set(data["task_index"].astype(int).tolist()).issubset(task_indices):
        raise ValueError("some task_index values do not resolve")
    navvla_task_indices = {int(row["task_index"]) for row in navvla_tasks}
    if not set(data["task_index"].astype(int).tolist()).issubset(navvla_task_indices):
        raise ValueError("some task_index values do not resolve in navvla_tasks")
    for row in navvla_tasks:
        missing = {"task_index", "task_type", "task_subtype", "platform_text", "dataset_source", "answer"} - set(row)
        if missing:
            raise ValueError(f"navvla_tasks row is missing keys: {sorted(missing)}")

    forbidden_context_columns = {
        "dataset.source",
        "split.scene_id",
        "anchor_timestamp",
        "keep_probability",
        "token_count_before",
        "tvi_time",
        "tvi_phi",
        "context.index_key",
        "current_tvi_time",
        "history_steps",
        "history_blocks",
        "history_token_refs",
        "history_mask",
    }
    context_columns = _parquet_columns(context_paths.meta_path)
    if forbidden_context_columns.intersection(context_columns):
        raise ValueError(f"context index has debug or redundant columns: {sorted(forbidden_context_columns.intersection(context_columns))}")
    required_context_columns = {
        "index",
        "bats_k",
        "history_offset",
        "history_count",
        "long_memory_offset",
        "long_memory_count",
    }
    missing_context_columns = required_context_columns - context_columns
    if missing_context_columns:
        raise ValueError(f"context index is missing compact fields: {sorted(missing_context_columns)}")
    if not set(data["index"].astype(int).tolist()).issubset(set(context["index"].astype(int).tolist())):
        raise ValueError("some data indexes do not resolve in context index")
    if context_debug is not None and not set(context_debug["index"].astype(int).tolist()).issubset(set(context["index"].astype(int).tolist())):
        raise ValueError("some context debug indexes do not resolve in main context index")
    if context_debug is not None and "pose_change_protection_available" in context_debug.columns:
        raise ValueError("context debug shard should not contain pose-change protection fields")

    context_alignment = _validate_runtime_context_alignment(runtime_context)
    context_storage_report = {
        "rows": int(len(context)),
        "checked_columns": [],
        "synchronized": True,
        "storage": "meta/context_index",
    }

    token_ref_count, token_refs = _collect_history_token_refs(root, token_budget=int(token_budget))
    token_cache_report = _validate_visual_token_cache(
        root,
        token_refs=token_refs,
        token_ref_count=token_ref_count,
        visual_token_mode=visual_token_mode,
        visual_token_profile=visual_token_profile,
        token_budget=int(token_budget),
        cache_sample_size=int(cache_sample_size),
        sample_seed=int(sample_seed),
    )

    video_report, media_decode_report = _validate_videos_and_media(
        root,
        info=info,
        video_index=video_index,
        check_media_decode=check_media_decode,
        media_decode_sample=media_decode_sample,
    )
    smoke_report = _smoke_load_dataset(
        root,
        enabled=smoke_load > 0 or smoke_load_all,
        smoke_load=smoke_load,
        smoke_load_all=smoke_load_all,
        visual_token_mode=visual_token_mode,
        required_cameras=required_cameras,
        image_resize=image_resize,
        visual_token_profile=visual_token_profile,
        token_budget=int(token_budget),
    )
    artifacts: dict[str, Any] = dict(data_artifacts)
    if token_cache_report.get("required_for_visual_token_mode"):
        index_records = int(token_cache_report["index_records"])
        sample_indices = list(token_cache_report.get("sample_indices", []))
        artifacts["visual_cache_index"] = {"scope": "full", "checked": index_records, "total": index_records}
        artifacts["visual_cache_tensors"] = {
            "scope": "sampled",
            "checked": len(sample_indices),
            "total": index_records,
            "sample_indices": sample_indices,
            "seed": int(sample_seed),
        }
    return {
        "dataset_root": str(root),
        "visual_token_mode": visual_token_mode,
        "token_budget": int(token_budget),
        "context_index": {
            "meta_path": str(context_paths.meta_path),
            "debug_path": str(context_paths.debug_path),
        },
        "total_frames": len(data),
        "total_episodes": len(episodes),
        "total_tasks": len(tasks),
        "camera_count": len(cameras),
        "statistics_keys": sorted(statistics.keys()),
        "state_contract": state_contract_report,
        "context_alignment": context_alignment,
        "context_storage": context_storage_report,
        "token_cache": token_cache_report,
        "video_index": video_report,
        "media_decode": media_decode_report,
        "smoke_load": smoke_report,
        "artifacts": artifacts,
    }


def _reject_legacy_smoke_cache(root: Path) -> None:
    legacy_manifest = root / "cache" / "visual_tokens" / "manifest.jsonl"
    legacy_tokens = root / "cache" / "visual_tokens" / "tokens"
    if legacy_manifest.exists():
        for line in legacy_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("visual_head") == "smoke_token":
                raise ValueError("smoke visual token cache is not valid for cached_history_online_current")
        raise ValueError("legacy visual token manifest.jsonl is not valid for cached visual token mode")
    if legacy_tokens.exists() and any(legacy_tokens.glob("*.npy")):
        raise ValueError("smoke visual token .npy cache is not valid for cached_history_online_current")


def _visual_token_values_are_finite(values: np.ndarray, *, storage_encoding: str) -> bool:
    array = np.asarray(values)
    if storage_encoding == BFLOAT16_BITS_STORAGE_ENCODING:
        if array.dtype != np.dtype(np.uint16):
            raise TypeError(
                "bfloat16_bits visual cache must use numpy uint16 values, "
                f"got {array.dtype}"
            )
        exponent_bits = np.bitwise_and(array, np.uint16(0x7F80))
        return not bool(np.any(exponent_bits == np.uint16(0x7F80)))
    return bool(np.isfinite(array).all())


def _validate_visual_token_cache(
    root: Path,
    *,
    token_refs: set[str],
    token_ref_count: int,
    visual_token_mode: str,
    visual_token_profile: str,
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    cache_sample_size: int = 10,
    sample_seed: int = 42,
) -> dict[str, Any]:
    required = visual_token_mode in CACHE_REQUIRED_VISUAL_TOKEN_MODES
    report: dict[str, Any] = {
        "token_budget": int(token_budget),
        "history_token_ref_count": int(token_ref_count),
        "unique_history_token_ref_count": len(token_refs),
        "required_for_visual_token_mode": required,
        "validated": False,
    }
    if visual_token_mode in CACHE_REQUIRED_VISUAL_TOKEN_MODES:
        _reject_legacy_smoke_cache(root)
        if not token_refs:
            raise ValueError(f"visual_token_mode={visual_token_mode!r} requires history_token_refs, but none were found")
        profile_root = profile_cache_root(root, visual_token_profile)
        manifest_path = profile_root / "manifest.json"
        index_path = profile_root / "index.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing visual token profile manifest: {manifest_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"missing visual token profile index: {index_path}")
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_payload.get("visual_head") == "smoke_token":
            raise ValueError("smoke visual token cache is not valid for cached visual token mode")
        file_format = str(manifest_payload.get("file_format", NPZ_VISUAL_TOKEN_FORMAT))
        expected_dtype = np.dtype(manifest_payload.get("dtype", "float16"))
        storage_encoding = str(manifest_payload.get("storage_encoding", ""))
        if storage_encoding and storage_encoding != BFLOAT16_BITS_STORAGE_ENCODING:
            raise ValueError(f"unsupported visual token storage_encoding: {storage_encoding!r}")
        if storage_encoding == BFLOAT16_BITS_STORAGE_ENCODING and expected_dtype != np.dtype(np.uint16):
            raise ValueError("bfloat16_bits visual cache manifest must use dtype=uint16")
        if file_format not in {NPZ_VISUAL_TOKEN_FORMAT, MMAP_NPY_VISUAL_TOKEN_FORMAT}:
            raise ValueError(f"unsupported visual token profile file_format: {file_format}")
        array_keys = set(manifest_payload.get("array_keys", []))
        if "image_embeds" not in array_keys:
            raise ValueError("visual token profile manifest must declare image_embeds")
        if file_format == NPZ_VISUAL_TOKEN_FORMAT and not {"image_embeds", "deepstack_embeds"}.issubset(array_keys):
            raise ValueError("visual token profile manifest must declare image_embeds and deepstack_embeds")
        index_file = pq.ParquetFile(index_path)
        index_records = int(index_file.metadata.num_rows)
        index_columns = set(index_file.schema_arrow.names)
        required_columns = {"ref", "episode_id", "trajectory_id", "frame_index", "source_frame_index", "data_index", "camera_name", "video_key"}
        if file_format == MMAP_NPY_VISUAL_TOKEN_FORMAT:
            required_columns |= {"shard_path", "row_index", "token_count", "hidden_dim"}
        else:
            required_columns |= {"path"}
        missing_columns = required_columns - index_columns
        if missing_columns:
            raise ValueError(f"visual token profile index is missing columns: {sorted(missing_columns)}")
        sample_indices = _deterministic_sample_indices(index_records, cache_sample_size, seed=sample_seed)
        sampled_rows = _sample_parquet_rows(index_file, sample_indices)
        sampled = 0
        for row in sampled_rows.itertuples(index=False):
            for field in ("ref", "episode_id", "trajectory_id", "camera_name", "video_key"):
                if not str(getattr(row, field)).strip():
                    raise ValueError(f"visual token profile index contains empty {field}: {index_path}")
            for field in ("frame_index", "source_frame_index", "data_index"):
                value = getattr(row, field)
                if pd.isna(value) or int(value) < 0:
                    raise ValueError(f"visual token profile index contains invalid {field}: {index_path}")
            if file_format == MMAP_NPY_VISUAL_TOKEN_FORMAT:
                if not str(row.shard_path).strip():
                    raise ValueError(f"visual token profile index contains empty shard_path: {index_path}")
                token_path = root / str(row.shard_path)
                if token_path.suffix != ".npy":
                    raise ValueError(f"visual token profile must point to mmap .npy shards: {token_path}")
                if not token_path.exists():
                    raise FileNotFoundError(f"missing visual token shard: {token_path}")
                shard = np.load(token_path, mmap_mode="r", allow_pickle=False)
                if shard.dtype != expected_dtype:
                    raise TypeError(
                        f"visual token shard dtype {shard.dtype} does not match manifest dtype "
                        f"{expected_dtype}: {token_path}"
                    )
                row_index = int(row.row_index)
                if row_index < 0 or row_index >= int(shard.shape[0]):
                    raise IndexError(f"visual token shard row_index is out of range: {token_path}:{row_index}")
                if int(row.token_count) <= 0 or int(row.hidden_dim) <= 0:
                    raise ValueError(f"visual token profile index contains non-positive tensor dimensions: {index_path}")
                if int(shard.shape[1]) != int(row.token_count) or int(shard.shape[2]) != int(row.hidden_dim):
                    raise ValueError(f"visual token shard shape does not match index row: {token_path}")
                values = np.asarray(shard[row_index])
            else:
                if not str(row.path).strip():
                    raise ValueError(f"visual token profile index contains empty path: {index_path}")
                token_path = root / str(row.path)
                if token_path.suffix != ".npz":
                    raise ValueError(f"visual token profile must point to npz files: {token_path}")
                if not token_path.exists():
                    raise FileNotFoundError(f"missing visual token file: {token_path}")
                with np.load(token_path, allow_pickle=False) as payload:
                    if "image_embeds" not in payload:
                        raise ValueError(f"missing image_embeds in {token_path}")
                    if "deepstack_embeds" not in payload:
                        raise ValueError(f"missing deepstack_embeds in {token_path}")
                    values = np.asarray(payload["image_embeds"])
                    deepstack_values = np.asarray(payload["deepstack_embeds"])
                    if values.dtype != expected_dtype or deepstack_values.dtype != expected_dtype:
                        raise TypeError(
                            f"visual token file dtype does not match manifest dtype {expected_dtype}: {token_path}"
                        )
                    if deepstack_values.size == 0 or not _visual_token_values_are_finite(
                        deepstack_values,
                        storage_encoding=storage_encoding,
                    ):
                        raise ValueError(f"deepstack visual token tensor is empty or non-finite: {token_path}")
            if values.dtype != expected_dtype:
                raise TypeError(
                    f"visual token tensor dtype {values.dtype} does not match manifest dtype "
                    f"{expected_dtype}: {token_path}"
                )
            if values.size == 0 or not _visual_token_values_are_finite(
                values,
                storage_encoding=storage_encoding,
            ):
                raise ValueError(f"visual token tensor is empty or non-finite: {token_path}")
            sampled += 1
        report.update(
            {
                "profile": visual_token_profile,
                "manifest_present": True,
                "index_records": index_records,
                "sampled_files": sampled,
                "sample_indices": sample_indices,
                "sample_seed": int(sample_seed),
                "validated": True,
            }
        )
        return report

    if token_refs:
        report["skipped_reason"] = "visual_token_mode does not require cached token files"
    return report


def _deterministic_sample_indices(total: int, count: int, *, seed: int) -> list[int]:
    total = max(0, int(total))
    count = max(0, min(int(count), total))
    if count == 0:
        return []
    anchors = [0, total // 2, total - 1]
    selected = {index for index in anchors if 0 <= index < total}
    rng = random.Random(int(seed))
    candidates = [index for index in range(total) if index not in selected]
    rng.shuffle(candidates)
    selected.update(candidates[: max(0, count - len(selected))])
    return sorted(selected)[:count]


def _sample_parquet_rows(parquet: pq.ParquetFile, indices: list[int]) -> pd.DataFrame:
    if not indices:
        return pd.DataFrame(columns=parquet.schema_arrow.names)
    row_group_starts: list[int] = []
    offset = 0
    for group_index in range(parquet.num_row_groups):
        row_group_starts.append(offset)
        offset += int(parquet.metadata.row_group(group_index).num_rows)
    requested_by_group: dict[int, list[int]] = {}
    for index in indices:
        group_index = max(position for position, start in enumerate(row_group_starts) if start <= index)
        requested_by_group.setdefault(group_index, []).append(int(index - row_group_starts[group_index]))
    sampled = []
    for group_index, local_indices in sorted(requested_by_group.items()):
        frame = parquet.read_row_group(group_index).to_pandas()
        sampled.append(frame.iloc[local_indices])
    return pd.concat(sampled, ignore_index=True)


def _parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _collect_history_token_refs(root: Path, *, token_budget: int) -> tuple[int, set[str]]:
    refs = list(iter_context_refs(root, token_budget=int(token_budget)))
    return int(len(refs)), set(refs)


def _validate_runtime_context_alignment(runtime_context) -> dict[str, Any]:
    meta = runtime_context.meta
    arrays = runtime_context.arrays
    report: dict[str, Any] = {"rows": int(len(meta)), "camera_names": list(runtime_context.camera_names)}
    camera_count = len(runtime_context.camera_names)
    valid_camera_bits = (1 << camera_count) - 1 if camera_count else 0
    for prefix in ("history", "long_memory"):
        offsets = meta[f"{prefix}_offset"].to_numpy(dtype=np.int64, copy=False)
        counts = meta[f"{prefix}_count"].to_numpy(dtype=np.int64, copy=False)
        frame_indices = arrays[f"{prefix}_frame_index"]
        camera_masks = arrays[f"{prefix}_camera_mask"]
        expected_offsets = np.zeros(len(counts), dtype=np.int64)
        if len(counts) > 1:
            np.cumsum(counts[:-1], out=expected_offsets[1:])
        if np.any(offsets < 0) or np.any(counts < 0) or np.any(offsets != expected_offsets):
            raise ValueError(f"{prefix} compact context offsets/counts are invalid")
        expected_total = int(offsets[-1] + counts[-1]) if len(counts) else 0
        if len(frame_indices) != len(camera_masks) or expected_total != len(frame_indices):
            raise ValueError(f"{prefix} compact context arrays do not match metadata")
        if np.any(frame_indices < 0):
            raise ValueError(f"{prefix} compact context contains negative frame indices")
        invalid_bits = np.uint64(((1 << 64) - 1) ^ valid_camera_bits)
        invalid_masks = (camera_masks == 0) | ((camera_masks & invalid_bits) != 0)
        if np.any(invalid_masks):
            raise ValueError(f"{prefix} compact context contains invalid camera masks")
        report[f"{prefix}_frames"] = int(len(frame_indices))
        report[f"max_{prefix}_frames"] = int(counts.max()) if len(counts) else 0
    return report


def _validate_videos_and_media(
    root: Path,
    *,
    info: dict[str, Any],
    video_index: pd.DataFrame | None,
    check_media_decode: str,
    media_decode_sample: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checked_videos = 0
    decoded_frames = 0
    for video_key, pattern in info["video_path"].items():
        if video_index is not None and {"chunk_index", "file_index"}.issubset(video_index.columns):
            locations = (
                video_index[video_index["video_key"] == video_key][["video_key", "chunk_index", "file_index"]]
                .drop_duplicates()
                .sort_values(["chunk_index", "file_index"])
                .to_dict("records")
            )
        else:
            locations = [{"video_key": video_key, "chunk_index": 0, "file_index": 0}]
        for location in locations:
            chunk_index = int(location["chunk_index"])
            file_index = int(location["file_index"])
            path = root / pattern.format(chunk_index=chunk_index, file_index=file_index)
            cap = cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened():
                    raise ValueError(f"video does not open: {path}")
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count <= 0:
                    raise ValueError(f"video has no frames: {path}")
                checked_videos += 1
                available_rows = None
                if video_index is not None:
                    available_rows = video_index[(video_index["video_key"] == video_key) & (video_index["available"])]
                    if {"chunk_index", "file_index"}.issubset(video_index.columns):
                        available_rows = available_rows[
                            (available_rows["chunk_index"].astype(int) == chunk_index)
                            & (available_rows["file_index"].astype(int) == file_index)
                        ]
                    if len(available_rows) != frame_count:
                        raise ValueError(
                            f"video_index count mismatch for {video_key} chunk={chunk_index} file={file_index}: "
                            f"{len(available_rows)} rows vs {frame_count} video frames"
                        )
                if check_media_decode != "none":
                    frame_indices = _decode_frame_indices(
                        frame_count=frame_count,
                        mode=check_media_decode,
                        sample_count=media_decode_sample,
                        available_rows=available_rows,
                    )
                    for frame_index in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            raise ValueError(f"failed to decode {path} frame {frame_index}")
                        decoded_frames += 1
            finally:
                cap.release()
    return (
        {"checked_videos": checked_videos, "video_index_present": video_index is not None},
        {"mode": check_media_decode, "decoded_frames": decoded_frames},
    )


def _smoke_load_dataset(
    root: Path,
    *,
    enabled: bool,
    smoke_load: int,
    smoke_load_all: bool,
    visual_token_mode: str,
    required_cameras: list[str] | None,
    image_resize: tuple[int, int] | None,
    visual_token_profile: str = "qwen3_vl_4b_pooled_history",
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}

    profile_manifest_path = profile_cache_root(root, visual_token_profile) / "manifest.json"
    profile_file_format = None
    if profile_manifest_path.exists():
        profile_file_format = str(
            json.loads(profile_manifest_path.read_text(encoding="utf-8")).get(
                "file_format", NPZ_VISUAL_TOKEN_FORMAT
            )
        )
    if (
        visual_token_mode == "cached_history_online_current"
        and profile_file_format == MMAP_NPY_VISUAL_TOKEN_FORMAT
    ):
        return _smoke_load_cpm_dataset(
            root,
            smoke_load=smoke_load,
            smoke_load_all=smoke_load_all,
            visual_token_mode=visual_token_mode,
            required_cameras=required_cameras,
            image_resize=image_resize,
            visual_token_profile=visual_token_profile,
            token_budget=token_budget,
        )

    from starVLA.dataloader.navvla_lerobot_datasets import NavVLALeRobotDataset

    dataset = NavVLALeRobotDataset(
        root,
        visual_token_mode=visual_token_mode,
        visual_token_profile=visual_token_profile,
        token_budget=token_budget,
        required_cameras=required_cameras,
        image_resize=image_resize,
    )
    if smoke_load_all:
        indices = list(range(len(dataset)))
    else:
        indices = _representative_indices(len(dataset), max(1, int(smoke_load)))
    samples = []
    for index in indices:
        sample = dataset[index]
        samples.append(
            {
                "index": int(index),
                "history_blocks": len(sample["metadata"].get("history_blocks", [])),
                "history_token_refs": len(sample["metadata"].get("history_token_refs", [])),
                "history_mask": int(sample["history_mask"].shape[0]),
                "action_shape": list(sample["action"].shape),
            }
        )
    return {"enabled": True, "rows": len(dataset), "loaded_samples": len(indices), "examples": samples[:10]}


def _smoke_load_cpm_dataset(
    root: Path,
    *,
    smoke_load: int,
    smoke_load_all: bool,
    visual_token_mode: str,
    required_cameras: list[str] | None,
    image_resize: tuple[int, int] | None,
    visual_token_profile: str,
    token_budget: int,
) -> dict[str, Any]:
    from starVLA.dataloader.cpm_lerobot.collate import NavVLACPMCollator
    from starVLA.dataloader.cpm_lerobot.dataset import NavVLACPMDataset

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    splits = info.get("splits")
    split = str(next(iter(splits), root.name)) if isinstance(splits, dict) else str(root.name)
    dataset = NavVLACPMDataset(
        root,
        split=split,
        visual_token_mode=visual_token_mode,
        visual_token_profile=visual_token_profile,
        token_budget=token_budget,
        required_cameras=required_cameras,
        image_resize=image_resize,
        require_long_memory_tokens=False,
    )
    if smoke_load_all:
        indices = list(range(len(dataset)))
    else:
        indices = _representative_indices(len(dataset), max(1, int(smoke_load)))
    collator = NavVLACPMCollator()
    examples = []
    history_cached_shape = None
    for start in range(0, len(indices), 16):
        batch_indices = indices[start : start + 16]
        samples = [dataset[index] for index in batch_indices]
        batch = collator(samples)
        if history_cached_shape is None:
            history_cached_shape = list(batch["history_cached_embeds"].shape)
        if len(examples) < 10:
            examples.extend(
                {
                    "index": int(index),
                    "history_blocks": len(sample["metadata"].get("history_blocks", [])),
                    "history_token_refs": len(sample["metadata"].get("history_token_refs", [])),
                    "history_mask": int(sample["history_mask"].shape[0]),
                    "action_shape": list(sample["action"].shape),
                }
                for index, sample in zip(batch_indices, samples, strict=True)
            )
    return {
        "enabled": True,
        "reader": "NavVLACPMDataset",
        "collator": "NavVLACPMCollator",
        "rows": len(dataset),
        "loaded_samples": len(indices),
        "history_cached_shape": history_cached_shape,
        "examples": examples[:10],
    }


def _read_parquet_shards(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("chunk-*/part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards found under {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _read_data_inventory(
    root: Path,
    *,
    rows_per_shard: int,
    sample_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    paths = sorted(root.glob("chunk-*/part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards found under {root}")
    join_columns = ["index", "task_index", "context.index_key"]
    expected_schema = None
    join_tables = []
    sampled_rows: list[dict[str, Any]] = []
    global_sample_indices: list[int] = []
    total_rows = 0
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
        missing = sorted(DATA_REQUIRED_COLUMNS - set(schema.names))
        if missing:
            raise ValueError(f"data parquet shard is missing required columns {missing}: {path}")
        join_tables.append(pd.read_parquet(path, columns=join_columns))
        local_indices = _deterministic_sample_indices(rows, rows_per_shard, seed=sample_seed)
        if local_indices:
            sampled = _sample_parquet_rows(parquet, local_indices)
            sampled_rows.extend(sampled.to_dict("records"))
            global_sample_indices.extend(total_rows + index for index in local_indices)
        total_rows += rows
    data = pd.concat(join_tables, ignore_index=True)
    forbidden = {"dataset.source", "split.scene_id"}.intersection(expected_schema.names if expected_schema is not None else [])
    if forbidden:
        raise ValueError(f"data parquet has redundant columns: {sorted(forbidden)}")
    artifacts = {
        "data_parquet": {"scope": "full", "checked": len(paths), "total": len(paths)},
        "data_rows": {
            "scope": "sampled",
            "checked": len(sampled_rows),
            "total": total_rows,
            "sample_indices": global_sample_indices,
            "seed": int(sample_seed),
        },
    }
    return data, sampled_rows, artifacts


def _read_frame_metadata_inventory(path: Path, *, sample_size: int = 16) -> tuple[int, list[dict[str, Any]]]:
    if not path.exists():
        return 0, []
    count = 0
    sample: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = None
            if '"missing_fields"' in line:
                payload = json.loads(line)
                if "missing_fields" in payload:
                    raise ValueError("meta/navvla_frame_metadata.jsonl must not contain missing_fields")
            count += 1
            if len(sample) < int(sample_size):
                if payload is None:
                    payload = json.loads(line)
                sample.append(payload)
    return count, sample


def _validate_state_contract(info: dict[str, Any], data_rows: list[dict[str, Any]]) -> dict[str, Any]:
    navvla = info.get("navvla", {})
    state_dim = int(navvla.get("state_dim", -1))
    if state_dim != 4:
        raise ValueError(f"NavVLA state contract requires state_dim=4, got {state_dim}")
    features = info.get("features", {})
    state_feature = features.get("observation.state")
    if not isinstance(state_feature, dict):
        raise ValueError("meta/info.json features must include observation.state")
    shape = list(state_feature.get("shape") or [])
    if shape != [4]:
        raise ValueError(f"observation.state feature shape must be [4], got {shape}")
    action_horizon_raw = navvla.get("action_horizon")
    action_dim_raw = navvla.get("action_dim")
    action_horizon = int(action_horizon_raw) if action_horizon_raw is not None else None
    action_dim = int(action_dim_raw) if action_dim_raw is not None else None
    if (action_horizon is None) != (action_dim is None):
        raise ValueError("NavVLA action contract requires action_horizon and action_dim together")
    if action_horizon is not None and (action_horizon <= 0 or action_dim is None or action_dim <= 0):
        raise ValueError(f"NavVLA action contract requires positive action_horizon/action_dim, got {action_horizon}/{action_dim}")
    bad_rows = []
    for row_index, row in enumerate(data_rows):
        state = row.get("observation.state")
        if hasattr(state, "tolist"):
            state = state.tolist()
        if state is None or len(state) != 4:
            bad_rows.append({"row": row_index, "field": "observation.state", "length": None if state is None else len(state)})
            continue
        if not np.isfinite(np.asarray(state, dtype=np.float64)).all():
            bad_rows.append({"row": row_index, "field": "observation.state", "reason": "non-finite"})
            continue
        if action_horizon is not None and action_dim is not None:
            context_key = str(row.get("context.index_key") or "").strip()
            if not context_key:
                bad_rows.append({"row": row_index, "field": "context.index_key", "reason": "empty"})
                continue
            action = row.get("action")
            if hasattr(action, "tolist"):
                action = action.tolist()
            try:
                action_values = np.asarray(action, dtype=np.float64)
            except (TypeError, ValueError):
                action_values = np.asarray([], dtype=np.float64)
            if action_values.size != action_horizon * action_dim or not np.isfinite(action_values).all():
                bad_rows.append(
                    {
                        "row": row_index,
                        "field": "action",
                        "size": int(action_values.size),
                        "expected_size": action_horizon * action_dim,
                    }
                )
                continue
            padding_mask = _as_list(row.get("action.padding_mask"))
            if len(padding_mask) != action_horizon:
                bad_rows.append(
                    {"row": row_index, "field": "action.padding_mask", "length": len(padding_mask), "expected": action_horizon}
                )
                continue
            for field in ("timestamp", "task_index", "episode_index", "frame_index", "source_frame_index", "index"):
                value = row.get(field)
                if value is None or pd.isna(value) or not np.isfinite(float(value)):
                    bad_rows.append({"row": row_index, "field": field, "reason": "missing or non-finite"})
                    break
        if len(bad_rows) >= 5:
            break
    if bad_rows:
        first_field = bad_rows[0]["field"]
        if first_field == "observation.state" and "length" in bad_rows[0]:
            raise ValueError(f"sampled data observation.state must have length 4; examples={bad_rows}")
        raise ValueError(f"sampled data {first_field} is invalid; examples={bad_rows}")
    return {
        "state_dim": state_dim,
        "feature_shape": shape,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "checked_rows": len(data_rows),
    }
def _reject_legacy_missing_fields_outputs(
    *,
    info: dict[str, Any],
    schema_ext: dict[str, Any] | None,
    conversion_report: dict[str, Any] | None,
    frame_metadata_sample: list[dict[str, Any]],
) -> None:
    if "missing_field_policy" in info.get("navvla", {}):
        raise ValueError("meta/info.json navvla block must not contain missing_field_policy")
    if schema_ext is not None and "missing_field_policy" in schema_ext:
        raise ValueError("meta/navvla_schema_ext.json must not contain missing_field_policy")
    if conversion_report is not None and "missing_fields" in conversion_report:
        raise ValueError("conversion_report.json must not contain missing_fields")
    for row in frame_metadata_sample:
        if "missing_fields" in row:
            raise ValueError("meta/navvla_frame_metadata.jsonl must not contain missing_fields")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _nested_equal(left: Any, right: Any) -> bool:
    return _normalize_nested(left) == _normalize_nested(right)


def _normalize_nested(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _normalize_nested(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_nested(item) for item in value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _short_repr(value: Any, *, max_len: int = 240) -> str:
    text = repr(_normalize_nested(value))
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _decode_frame_indices(
    *,
    frame_count: int,
    mode: str,
    sample_count: int,
    available_rows: pd.DataFrame | None,
) -> list[int]:
    if mode == "all":
        if available_rows is not None and "video_frame_index" in available_rows.columns:
            return sorted(set(int(value) for value in available_rows["video_frame_index"].tolist()))
        return list(range(int(frame_count)))
    sample_count = max(1, int(sample_count))
    if available_rows is not None and "video_frame_index" in available_rows.columns and not available_rows.empty:
        values = sorted(set(int(value) for value in available_rows["video_frame_index"].tolist()))
        return _sample_sorted_values(values, sample_count)
    return _sample_sorted_values(list(range(int(frame_count))), sample_count)


def _sample_sorted_values(values: list[int], sample_count: int) -> list[int]:
    if len(values) <= sample_count:
        return values
    if sample_count == 1:
        return [values[0]]
    positions = [round(index * (len(values) - 1) / (sample_count - 1)) for index in range(sample_count)]
    return [values[position] for position in sorted(set(positions))]


def _representative_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    if length <= count:
        return list(range(length))
    if count == 1:
        return [0]
    positions = [round(index * (length - 1) / (count - 1)) for index in range(count)]
    return sorted(set(int(position) for position in positions))
