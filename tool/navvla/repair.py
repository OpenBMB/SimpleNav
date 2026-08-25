from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from tool.navvla.context_index import (
    DEFAULT_CONTEXT_TOKEN_BUDGETS,
    CONTEXT_INDEX_VERSION,
    ContextIndexConfig,
    build_context_indexes_streaming,
    iter_parquet_batches,
    load_runtime_context_index,
    normalize_context_token_budgets,
)
from tool.navvla.schema import NavVLACameraSpec, NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec
from tool.navvla.statistics import (
    build_dataset_statistics,
    flatten_valid_action_steps_from_rows,
    write_dataset_statistics,
)
from tool.navvla.validation import validate_navvla_lerobot_dataset
from tool.navvla.visual_token_cache import MMAP_NPY_VISUAL_TOKEN_FORMAT, VisualTokenProfile


def repair_navvla_dataset(
    dataset_root: str | Path,
    *,
    apply: bool = False,
    token_budgets: tuple[int, ...] | list[int] | None = None,
    budget_num_cameras: int | None = None,
    history_camera_names: tuple[str, ...] | list[str] | None = None,
    history_visual_tokens: int = 4,
    current_visual_tokens: int = 64,
    tvi_tokens: int = 1,
    context_epsilon: float = 0.1,
    context_seed: int = 42,
    include_long_memory: bool = True,
) -> dict[str, Any]:
    root = Path(dataset_root)
    data_paths = sorted((root / "data").glob("chunk-*/part-*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"no data parquet shards found under {root / 'data'}")
    video_index_path = root / "meta" / "navvla_video_index.parquet"
    if not video_index_path.exists():
        raise FileNotFoundError(f"missing video index; automatic reconstruction is not yet safe: {video_index_path}")
    budgets = normalize_context_token_budgets(tuple(token_budgets or DEFAULT_CONTEXT_TOKEN_BUDGETS))
    actions: list[dict[str, Any]] = []
    missing_budgets = [budget for budget in budgets if not _context_budget_complete(root, budget)]
    if missing_budgets:
        actions.append(
            {"type": "rebuild_context", "token_budgets": list(budgets), "missing_token_budgets": missing_budgets}
        )
    statistics_path = root / "dataset_statistics.json"
    if not statistics_path.exists() or statistics_path.stat().st_size <= 0:
        actions.append({"type": "rebuild_statistics", "path": str(statistics_path)})
    cache_index_repairs = _recoverable_mmap_cache_indexes(root)
    for profile_name in cache_index_repairs:
        actions.append({"type": "rebuild_mmap_cache_index", "profile": profile_name})

    report: dict[str, Any] = {"dataset_root": str(root), "applied": bool(apply), "actions": actions}
    if not apply:
        return report

    applied: list[dict[str, Any]] = []
    if missing_budgets:
        applied.append(
            {
                "type": "rebuild_context",
                "result": _apply_context_repair(
                    root,
                    token_budgets=tuple(budgets),
                    budget_num_cameras=budget_num_cameras,
                    history_camera_names=history_camera_names,
                    history_visual_tokens=history_visual_tokens,
                    current_visual_tokens=current_visual_tokens,
                    tvi_tokens=tvi_tokens,
                    context_epsilon=context_epsilon,
                    context_seed=context_seed,
                    include_long_memory=include_long_memory,
                ),
            }
        )
    if any(action["type"] == "rebuild_statistics" for action in actions):
        applied.append({"type": "rebuild_statistics", "result": _rebuild_statistics(root, data_paths)})
    for profile_name in cache_index_repairs:
        applied.append(
            {
                "type": "rebuild_mmap_cache_index",
                "profile": profile_name,
                "result": _rebuild_mmap_cache_index(root, profile_name),
            }
        )
    report["applied_actions"] = applied
    report["validation"] = validate_navvla_lerobot_dataset(
        root,
        smoke_load=0,
        token_budget=_default_context_budget(root),
    )
    return report


def _context_budget_complete(root: Path, budget: int) -> bool:
    manifest_path = root / "meta" / "navvla_context_index_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if int(manifest.get("version", 0)) != CONTEXT_INDEX_VERSION:
        return False
    entry = (manifest.get("entries") or {}).get(str(int(budget)))
    if not isinstance(entry, dict) or str(entry.get("selection_policy")) != "bats":
        return False
    return all((root / str(entry[key])).exists() for key in ("meta_path", "arrays_path", "debug_path"))


def _default_context_budget(root: Path) -> int:
    manifest = json.loads((root / "meta" / "navvla_context_index_manifest.json").read_text(encoding="utf-8"))
    if "default_token_budget" in manifest:
        return int(manifest["default_token_budget"])
    available = [int(value) for value in manifest.get("available_token_budgets", [])]
    if not available:
        raise ValueError("context index manifest contains no available token budgets")
    return available[0]


def _apply_context_repair(
    root: Path,
    *,
    token_budgets: tuple[int, ...],
    budget_num_cameras: int | None,
    history_camera_names: tuple[str, ...] | list[str] | None,
    history_visual_tokens: int,
    current_visual_tokens: int,
    tvi_tokens: int,
    context_epsilon: float,
    context_seed: int,
    include_long_memory: bool,
) -> dict[str, Any]:
    cameras = _read_existing_cameras(root)
    episodes = _read_existing_episodes(root, cameras=cameras)
    if not episodes:
        raise ValueError(f"no episodes found in existing dataset root: {root}")
    split = episodes[0].split
    spec = _read_existing_dataset_spec(root, split=split)
    config = ContextIndexConfig(
        epsilon=float(context_epsilon),
        bats_token_budget=int(token_budgets[0]),
        current_visual_tokens=int(current_visual_tokens),
        history_visual_tokens=int(history_visual_tokens),
        tvi_tokens=int(tvi_tokens),
        seed=int(context_seed),
        budget_num_cameras=budget_num_cameras,
        history_camera_names=tuple(history_camera_names) if history_camera_names else None,
        include_long_memory=bool(include_long_memory),
    )
    results = build_context_indexes_streaming(
        episodes,
        spec=spec,
        output_root=root,
        config=config,
        cache_manifest=None,
        token_budgets=token_budgets,
        progress_description=f"context {root.name}",
    )
    per_budget = {str(budget): _context_result_summary(result) for budget, result in sorted(results.items())}
    return {
        "mode": "context_index_only",
        "selection_policy": "bats",
        "split": split,
        "token_budgets": [int(value) for value in token_budgets],
        "default_token_budget": int(token_budgets[0]),
        "per_budget": per_budget,
    }


def _rebuild_statistics(root: Path, data_paths: list[Path]) -> dict[str, Any]:
    data = pd.concat(
        [pd.read_parquet(path, columns=["action", "action.padding_mask"]) for path in data_paths],
        ignore_index=True,
    )
    episode_paths = sorted((root / "meta" / "episodes").glob("chunk-*/part-*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"no episode parquet shards found under {root / 'meta' / 'episodes'}")
    episodes = pd.concat([pd.read_parquet(path) for path in episode_paths], ignore_index=True)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    split = str(episodes.iloc[0]["split"])
    dataset_key = f"{info['dataset_name']}_{split}"
    statistics = build_dataset_statistics(
        dataset_key=dataset_key,
        action_steps=flatten_valid_action_steps_from_rows(data),
        num_trajectories=len(episodes),
        num_transitions=len(data),
    )
    write_dataset_statistics(root / "dataset_statistics.json", statistics)
    return {"dataset_key": dataset_key, "rows": len(data), "episodes": len(episodes)}


def _recoverable_mmap_cache_indexes(root: Path) -> list[str]:
    profiles_root = root / "cache" / "visual_tokens"
    if not profiles_root.exists():
        return []
    recoverable = []
    for manifest_path in sorted(profiles_root.glob("*/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(payload.get("file_format")) != MMAP_NPY_VISUAL_TOKEN_FORMAT:
            continue
        profile_root = manifest_path.parent
        if (profile_root / "index.parquet").exists():
            continue
        if list((profile_root / "checkpoint_indexes").glob("*.parquet")) or list(profile_root.glob("index.rank*.parquet")):
            recoverable.append(profile_root.name)
    return recoverable


def _rebuild_mmap_cache_index(root: Path, profile_name: str) -> dict[str, Any]:
    from tool.navvla.cli.generate_visual_cache import rebuild_mmap_profile_index

    manifest_path = root / "cache" / "visual_tokens" / profile_name / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile_fields = {field: payload[field] for field in VisualTokenProfile.__dataclass_fields__ if field in payload}
    return rebuild_mmap_profile_index(root, VisualTokenProfile(**profile_fields))


def _context_result_summary(result) -> dict[str, Any]:
    runtime = load_runtime_context_index(result)
    rows = len(runtime.meta)
    bats_target_frames_max = 0.0
    for batch in iter_parquet_batches(result.debug_path, columns=["bats_target_frames"]):
        if len(batch):
            bats_target_frames_max = max(bats_target_frames_max, float(batch["bats_target_frames"].max()))
    unique_refs = set()
    for row in runtime.meta.to_dict("records"):
        materialized = runtime.materialize_meta_row(row)
        for prefix in ("history", "long_memory"):
            for frame in materialized[f"{prefix}_frames"]:
                for camera_index, camera_name in enumerate(runtime.camera_names):
                    if int(frame["camera_mask"]) & (1 << camera_index):
                        unique_refs.add((int(frame["frame_index"]), camera_name))
    return {
        "rows": rows,
        "history_steps_max": int(runtime.meta["history_count"].max()) if rows else 0,
        "long_memory_steps_max": int(runtime.meta["long_memory_count"].max()) if rows else 0,
        "unique_history_token_refs": len(unique_refs),
        "bats_target_frames_max": float(bats_target_frames_max),
        "meta_path": str(result.meta_path),
        "debug_path": str(result.debug_path),
    }


def _read_existing_dataset_spec(root: Path, *, split: str) -> NavVLADatasetSpec:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    schema_ext = json.loads((root / "meta" / "navvla_schema_ext.json").read_text(encoding="utf-8"))
    navvla = info.get("navvla", {})
    return NavVLADatasetSpec(
        dataset_name=str(info["dataset_name"]),
        fps=float(info["fps"]),
        control_frequency_hz=float(navvla.get("control_frequency_hz", info["fps"])),
        action_horizon=int(navvla.get("action_horizon", 8)),
        action_dim=int(navvla.get("action_dim", 4)),
        state_dim=int(navvla.get("state_dim", 4)),
        context_policy_version="bats-v1",
        cache_policy_version=str(
            schema_ext.get("cache_policy_version", navvla.get("cache_policy_version", "profile-cache-v1"))
        ),
        split=split,
        episodes_per_file=int(navvla.get("episodes_per_file", 20)),
        files_per_chunk=int(navvla.get("files_per_chunk", info.get("chunks_size", 50))),
    )


def _read_existing_cameras(root: Path) -> list[NavVLACameraSpec]:
    payload = json.loads((root / "meta" / "navvla_cameras.json").read_text(encoding="utf-8"))
    cameras = [
        NavVLACameraSpec(
            name=str(row.get("name") or name),
            video_key=str(row["video_key"]),
            viewpoint_type=str(row.get("viewpoint_type") or name),
            azimuth_rad=float(row.get("azimuth_rad", 0.0)),
            intrinsics=row.get("intrinsics"),
            extrinsics_body=row.get("extrinsics_body"),
            calibration_status=str(row.get("calibration_status", "unknown")),
        )
        for name, row in payload.items()
    ]
    if not cameras:
        raise ValueError(f"no cameras found in {root / 'meta' / 'navvla_cameras.json'}")
    return cameras


def _read_existing_episodes(root: Path, *, cameras: list[NavVLACameraSpec]) -> list[NavVLAEpisode]:
    data = _read_parquet_shards(root / "data").sort_values("index").reset_index(drop=True)
    episodes_meta = _read_parquet_shards(root / "meta" / "episodes").sort_values("episode_index").reset_index(drop=True)
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_rows = {int(row["task_index"]): row for row in tasks.to_dict("records")}
    task_text = {
        task_index: str(row.get("task") or row.get("instruction") or row.get("platform_text") or "Navigation task")
        for task_index, row in task_rows.items()
    }
    navvla_tasks = dict(task_rows)
    legacy_tasks_path = root / "meta" / "navvla_tasks.jsonl"
    if legacy_tasks_path.exists():
        for line in legacy_tasks_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            task_index = int(payload["task_index"])
            navvla_tasks[task_index] = {**navvla_tasks.get(task_index, {}), **payload}
    grouped = {
        int(episode_index): rows.sort_values("frame_index").to_dict("records")
        for episode_index, rows in data.groupby("episode_index", sort=False)
    }
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    media_keys = {camera.video_key for camera in cameras}
    episodes: list[NavVLAEpisode] = []
    for episode_row in episodes_meta.to_dict("records"):
        episode_index = int(episode_row["episode_index"])
        task_index = int(episode_row["task_index"])
        task_payload = navvla_tasks.get(task_index, {})
        task = NavVLATaskSpec(
            task_index=task_index,
            instruction=task_text.get(task_index, "Navigation task"),
            task_type=str(task_payload.get("task_type", "navigation")),
            task_subtype=str(task_payload.get("task_subtype", "navigation")),
            platform_text=str(task_payload.get("platform_text", "Platform: navigation agent.")),
            dataset_source=str(task_payload.get("dataset_source", "unknown")),
            scene_id=str(episode_row["scene_id"]),
            answer=task_payload.get("answer"),
        )
        frames = []
        for row in grouped.get(episode_index, []):
            media_paths = {key: Path(str(row.get(key))) for key in media_keys if row.get(key)}
            if not media_paths:
                media_paths = {camera.video_key: root / "videos" for camera in cameras}
            frames.append(
                NavVLAFrame(
                    frame_index=int(row["frame_index"]),
                    timestamp=float(row["timestamp"]),
                    media_paths=media_paths,
                    state=list(row["observation.state"]),
                    action=list(row["action"]),
                    source_frame_index=int(row["source_frame_index"]) if row.get("source_frame_index") is not None else None,
                    data_index=int(row["index"]),
                )
            )
        episodes.append(
            NavVLAEpisode(
                episode_id=str(episode_row["episode_id"]),
                task=task,
                frames=frames,
                cameras=list(cameras),
                split=str(episode_row.get("split") or info.get("split") or "vln_train"),
                trajectory_id=str(episode_row.get("trajectory_id") or episode_row["episode_id"]),
            )
        )
    return episodes


def _read_parquet_shards(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("chunk-*/part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards found under {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
