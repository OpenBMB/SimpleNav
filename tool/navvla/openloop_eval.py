from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd

from tool.navvla.repair import repair_navvla_dataset
from tool.navvla.validation import validate_navvla_lerobot_dataset
from tool.navvla.visual_token_cache import (
    DEFAULT_QWEN35_POOLED_HISTORY_VISUAL_TOKEN_PROFILE,
    default_qwen35_pooled_history_visual_token_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA_ROOT = REPO_ROOT / "local/data"
DEFAULT_OPENLOOP_EVAL_ROOT = LOCAL_DATA_ROOT / "navvla_openloop_eval_v1"
DEFAULT_QWEN35_ENCODER = REPO_ROOT / "local/models/Qwen3.5-4B"
OPENLOOP_EVAL_VERSION = "navvla-openloop-eval-v1"
SPLITS = ("vln_val_seen", "vln_val_unseen")


@dataclass(frozen=True)
class OpenLoopDatasetSpec:
    name: str
    train_root: Path
    eval_parent_root: Path
    dataset_statistics_key: str
    checkpoint_statistics_key: str
    required_cameras: tuple[str, ...]

    def split_root(self, split: str) -> Path:
        return self.eval_parent_root / str(split)


DEFAULT_OPENLOOP_DATASETS = (
    OpenLoopDatasetSpec(
        name="openfly",
        train_root=LOCAL_DATA_ROOT / "OpenFly_lerobot/vln_train",
        eval_parent_root=LOCAL_DATA_ROOT / "OpenFly_lerobot",
        dataset_statistics_key="vln_train_vln_train",
        checkpoint_statistics_key="openfly",
        required_cameras=("front",),
    ),
    OpenLoopDatasetSpec(
        name="aerialvln",
        train_root=LOCAL_DATA_ROOT / "AerialVLN_lerobot/vln_train",
        eval_parent_root=LOCAL_DATA_ROOT / "AerialVLN_lerobot",
        dataset_statistics_key="vln_train_vln_train",
        checkpoint_statistics_key="aerialvln",
        required_cameras=("front",),
    ),
    OpenLoopDatasetSpec(
        name="traveluav",
        train_root=LOCAL_DATA_ROOT / "TravelUAV_lerobot/vln_train",
        eval_parent_root=LOCAL_DATA_ROOT / "TravelUAV_lerobot",
        dataset_statistics_key="vln_train_train",
        checkpoint_statistics_key="traveluav",
        required_cameras=("front", "left", "right", "rear"),
    ),
    OpenLoopDatasetSpec(
        name="r2r",
        train_root=LOCAL_DATA_ROOT / "VLNCE_lerobot/navvla_lerobot_full_vlnce_r2r/vln_train",
        eval_parent_root=LOCAL_DATA_ROOT / "VLNCE_lerobot/navvla_lerobot_full_vlnce_r2r",
        dataset_statistics_key="vln_train_train",
        checkpoint_statistics_key="r2r",
        required_cameras=("front", "left", "right", "rear"),
    ),
    OpenLoopDatasetSpec(
        name="rxr",
        train_root=LOCAL_DATA_ROOT / "VLNCE_lerobot/navvla_lerobot_vlnce_rxr/vln_train",
        eval_parent_root=LOCAL_DATA_ROOT / "VLNCE_lerobot/navvla_lerobot_vlnce_rxr",
        dataset_statistics_key="vln_train_train",
        checkpoint_statistics_key="rxr",
        required_cameras=("front", "left", "right", "rear"),
    ),
)


def build_openloop_eval_suite(
    *,
    output_root: str | Path = DEFAULT_OPENLOOP_EVAL_ROOT,
    dataset_specs: Iterable[OpenLoopDatasetSpec] = DEFAULT_OPENLOOP_DATASETS,
    seed: int = 42,
    episodes_per_split: int = 100,
    targets_per_split: int = 400,
    token_budget: int = 512,
    overwrite: bool = False,
    generate_visual_cache: bool = True,
    encoder_ckpt: str | Path = DEFAULT_QWEN35_ENCODER,
    cache_batch_size: int = 8,
    cache_prefetch_batches: int = 2,
    validate: bool = True,
) -> dict[str, Any]:
    output_root = Path(output_root)
    specs = tuple(dataset_specs)
    if not specs:
        raise ValueError("dataset_specs must not be empty")
    if episodes_per_split <= 0 or targets_per_split <= 0:
        raise ValueError("episodes_per_split and targets_per_split must be positive")

    output_root.mkdir(parents=True, exist_ok=True)
    targets_root = output_root / "targets"
    targets_root.mkdir(parents=True, exist_ok=True)
    split_reports: list[dict[str, Any]] = []
    started_at = time.time()

    for spec in specs:
        _validate_dataset_spec(spec)
        for split in SPLITS:
            source_root = spec.split_root(split)
            target_root = output_root / spec.name / split
            split_report = build_openloop_eval_split(
                source_root=source_root,
                train_root=spec.train_root,
                output_root=target_root,
                dataset_name=spec.name,
                split=split,
                dataset_statistics_key=spec.dataset_statistics_key,
                checkpoint_statistics_key=spec.checkpoint_statistics_key,
                required_cameras=spec.required_cameras,
                seed=seed,
                episodes_per_split=episodes_per_split,
                targets_per_split=targets_per_split,
                token_budget=token_budget,
                overwrite=overwrite,
            )
            target_path = targets_root / f"{spec.name}_{split}.jsonl"
            _write_jsonl(target_path, split_report.pop("targets"))
            split_report["targets_path"] = str(target_path)
            split_reports.append(split_report)

    cache_reports: dict[str, Any] = {}
    if generate_visual_cache:
        cache_reports = generate_openloop_eval_visual_caches(
            split_roots=[Path(report["output_root"]) for report in split_reports],
            encoder_ckpt=encoder_ckpt,
            batch_size=cache_batch_size,
            prefetch_batches=cache_prefetch_batches,
        )
        for report in split_reports:
            root = str(report["output_root"])
            report["visual_cache"] = cache_reports[root]
            _merge_conversion_report(Path(root), {"visual_cache": cache_reports[root]})

    validation_reports: dict[str, Any] = {}
    if validate:
        for spec in specs:
            for split in SPLITS:
                root = output_root / spec.name / split
                validation_reports[str(root)] = validate_navvla_lerobot_dataset(
                    root,
                    visual_token_mode=(
                        "cached_history_online_current" if generate_visual_cache else "online_images"
                    ),
                    visual_token_profile=DEFAULT_QWEN35_POOLED_HISTORY_VISUAL_TOKEN_PROFILE,
                    token_budget=token_budget,
                    required_cameras=list(spec.required_cameras),
                    image_resize=(256, 256),
                    smoke_load=1,
                    check_media_decode="sampled",
                )
        for report in split_reports:
            root = str(report["output_root"])
            report["validation"] = validation_reports[root]
            _merge_conversion_report(Path(root), {"validation": validation_reports[root]})

    selection_report = _selection_report(
        split_reports,
        seed=seed,
        episodes_per_split=episodes_per_split,
        targets_per_split=targets_per_split,
    )
    registry = {
        "version": OPENLOOP_EVAL_VERSION,
        "created_at_unix": time.time(),
        "build_seconds": time.time() - started_at,
        "output_root": str(output_root),
        "seed": int(seed),
        "token_budget": int(token_budget),
        "visual_token_profile": DEFAULT_QWEN35_POOLED_HISTORY_VISUAL_TOKEN_PROFILE,
        "encoder_ckpt": str(encoder_ckpt),
        "datasets": [
            {
                **_jsonable(asdict(spec)),
                "eval_root_dir": str(output_root / spec.name),
                "splits": {
                    split: str(output_root / spec.name / split) for split in SPLITS
                },
            }
            for spec in specs
        ],
    }
    _write_json(output_root / "selection_report.json", selection_report)
    _write_json(output_root / "registry.json", registry)
    return {"registry": registry, "selection_report": selection_report}


def build_openloop_eval_split(
    *,
    source_root: str | Path,
    train_root: str | Path,
    output_root: str | Path,
    dataset_name: str,
    split: str,
    dataset_statistics_key: str,
    checkpoint_statistics_key: str,
    required_cameras: Iterable[str],
    seed: int = 42,
    episodes_per_split: int = 100,
    targets_per_split: int = 400,
    token_budget: int = 512,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_root = Path(source_root)
    train_root = Path(train_root)
    output_root = Path(output_root)
    required_cameras = tuple(str(value) for value in required_cameras)
    started_at = time.time()
    if not source_root.exists():
        raise FileNotFoundError(f"source split root does not exist: {source_root}")
    statistics_path = train_root / "dataset_statistics.json"
    if not statistics_path.exists():
        raise FileNotFoundError(f"training statistics do not exist: {statistics_path}")

    inventory = _load_source_inventory(source_root)
    unknown_cameras = set(required_cameras) - set(inventory["cameras"])
    if unknown_cameras:
        raise KeyError(f"required cameras missing from {source_root}: {sorted(unknown_cameras)}")
    candidates, valid_frames_by_episode = _eligible_episodes(
        inventory,
        required_cameras=required_cameras,
        check_video_files=False,
    )
    selected = select_scene_stratified_episodes(
        candidates,
        count=episodes_per_split,
        seed=seed,
        dataset_name=dataset_name,
        split=split,
    )
    required_video_keys = {
        str(inventory["cameras"][camera_name]["video_key"])
        for camera_name in required_cameras
    }
    video_artifact_cache: dict[Path, int | None] = {}
    bad_selected = [
        row
        for row in selected
        if not _episode_media_bounds_are_decodable(
            inventory,
            row,
            required_video_keys=required_video_keys,
            video_artifact_cache=video_artifact_cache,
        )
    ]
    if bad_selected:
        raise ValueError(
            "selected episodes reference missing or undecodable video shards: "
            f"{[row['episode_id'] for row in bad_selected[:10]]}"
        )
    target_source_indexes = select_target_source_indexes(
        selected,
        valid_frames_by_episode=valid_frames_by_episode,
        count=targets_per_split,
    )
    selection_payload = {
        "dataset": dataset_name,
        "split": split,
        "seed": int(seed),
        "source_episode_indices": [int(row["episode_index"]) for row in selected],
        "source_target_indexes": [int(value) for value in target_source_indexes],
    }
    selection_hash = hashlib.sha256(
        json.dumps(selection_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    staging_root = output_root.with_name(f".{output_root.name}.building")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass overwrite=True to replace it")
        shutil.rmtree(output_root)
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True)
    try:
        write_report = _write_selected_split(
            inventory,
            selected=selected,
            target_source_indexes=target_source_indexes,
            output_root=staging_root,
            dataset_name=dataset_name,
            split=split,
            selection_hash=selection_hash,
            required_cameras=required_cameras,
        )
        shutil.copy2(statistics_path, staging_root / "dataset_statistics.json")
        statistics = json.loads((staging_root / "dataset_statistics.json").read_text(encoding="utf-8"))
        if dataset_statistics_key not in statistics:
            raise KeyError(
                f"training statistics {statistics_path} do not contain {dataset_statistics_key!r}; "
                f"keys={sorted(statistics)}"
            )
        context_report = repair_navvla_dataset(
            staging_root,
            apply=True,
            token_budgets=(int(token_budget),),
            budget_num_cameras=len(required_cameras),
            history_camera_names=required_cameras,
            history_visual_tokens=4,
            current_visual_tokens=64,
            tvi_tokens=1,
            context_seed=seed,
            include_long_memory=False,
        )
        _merge_conversion_report(
            staging_root,
            {
                "version": OPENLOOP_EVAL_VERSION,
                "dataset": dataset_name,
                "split": split,
                "source_root": str(source_root),
                "train_root": str(train_root),
                "dataset_statistics_key": dataset_statistics_key,
                "checkpoint_statistics_key": checkpoint_statistics_key,
                "required_cameras": list(required_cameras),
                "selection_hash": selection_hash,
                "context_build": context_report,
                "visual_cache": {"status": "pending"},
            },
        )
        staging_root.rename(output_root)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    source_to_new = write_report["source_to_new_data_index"]
    target_rows = []
    data_by_source_index = inventory["data"].set_index("index", drop=False)
    episode_by_index = {
        int(row["episode_index"]): row for row in selected
    }
    new_episode_by_source = write_report["source_to_new_episode_index"]
    for source_index in target_source_indexes:
        data_row = data_by_source_index.loc[int(source_index)]
        source_episode_index = int(data_row["episode_index"])
        episode_row = episode_by_index[source_episode_index]
        target_rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "dataset_source": write_report["dataset_source"],
                "checkpoint_statistics_key": checkpoint_statistics_key,
                "episode_index": int(new_episode_by_source[source_episode_index]),
                "source_episode_index": source_episode_index,
                "episode_id": str(episode_row["episode_id"]),
                "frame_index": int(data_row["frame_index"]),
                "index": int(source_to_new[int(source_index)]),
                "source_index": int(source_index),
                "scene_id": str(episode_row["scene_id"]),
            }
        )
    if len(target_rows) != targets_per_split:
        raise AssertionError(f"target manifest has {len(target_rows)} rows, expected {targets_per_split}")

    return {
        "dataset": dataset_name,
        "split": split,
        "source_root": str(source_root),
        "train_root": str(train_root),
        "output_root": str(output_root),
        "dataset_statistics_key": dataset_statistics_key,
        "checkpoint_statistics_key": checkpoint_statistics_key,
        "required_cameras": list(required_cameras),
        "seed": int(seed),
        "selection_hash": selection_hash,
        "available_scenes": int(len({str(row["scene_id"]) for row in candidates})),
        "candidate_episodes": int(len(candidates)),
        "selected_episodes": int(len(selected)),
        "selected_targets": int(len(target_rows)),
        "selected_rows": int(write_report["total_frames"]),
        "scene_counts": _value_counts(selected, "scene_id"),
        "build_seconds": time.time() - started_at,
        "targets": target_rows,
    }


def select_scene_stratified_episodes(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    dataset_name: str,
    split: str,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_scene.setdefault(str(row["scene_id"]), []).append(row)
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} eligible episodes are available, need {count}")
    if len(by_scene) > count:
        raise ValueError(f"{len(by_scene)} scenes cannot each receive one of only {count} episode slots")

    quotas = {scene: 1 for scene in by_scene}
    remaining = count - len(by_scene)
    capacities = {scene: len(rows) - 1 for scene, rows in by_scene.items()}
    if remaining:
        weight_total = sum(len(rows) for rows in by_scene.values())
        exact = {scene: remaining * len(rows) / weight_total for scene, rows in by_scene.items()}
        for scene in sorted(by_scene):
            allocation = min(capacities[scene], int(math.floor(exact[scene])))
            quotas[scene] += allocation
            capacities[scene] -= allocation
        missing = count - sum(quotas.values())
        remainder_order = sorted(
            by_scene,
            key=lambda scene: (
                -(exact[scene] - math.floor(exact[scene])),
                _stable_digest(seed, dataset_name, split, scene),
            ),
        )
        while missing:
            progressed = False
            for scene in remainder_order:
                if capacities[scene] <= 0:
                    continue
                quotas[scene] += 1
                capacities[scene] -= 1
                missing -= 1
                progressed = True
                if missing == 0:
                    break
            if not progressed:
                raise ValueError("scene capacities cannot satisfy the requested episode count")

    selected: list[dict[str, Any]] = []
    for scene in sorted(by_scene):
        rows = sorted(
            by_scene[scene],
            key=lambda row: (str(row["episode_id"]), int(row["episode_index"])),
        )
        rng = random.Random(f"{int(seed)}:{dataset_name}:{split}:{scene}")
        rng.shuffle(rows)
        selected.extend(rows[: quotas[scene]])
    selected.sort(key=lambda row: int(row["episode_index"]))
    if len(selected) != count:
        raise AssertionError(f"selected {len(selected)} episodes, expected {count}")
    return selected


def select_target_source_indexes(
    selected_episodes: list[dict[str, Any]],
    *,
    valid_frames_by_episode: dict[int, list[int]],
    count: int,
) -> list[int]:
    selected: dict[int, list[int]] = {}
    for episode_row in selected_episodes:
        episode_index = int(episode_row["episode_index"])
        eligible = valid_frames_by_episode[episode_index]
        initial_count = min(4, len(eligible))
        selected[episode_index] = [eligible[position] for position in _even_positions(len(eligible), initial_count)]

    missing = count - sum(len(values) for values in selected.values())
    episode_order = [int(row["episode_index"]) for row in selected_episodes]
    while missing > 0:
        progressed = False
        for episode_index in episode_order:
            chosen = set(selected[episode_index])
            extra = next(
                (value for value in valid_frames_by_episode[episode_index] if value not in chosen),
                None,
            )
            if extra is None:
                continue
            selected[episode_index].append(extra)
            missing -= 1
            progressed = True
            if missing == 0:
                break
        if not progressed:
            break
    if missing != 0:
        available = sum(len(valid_frames_by_episode[int(row["episode_index"])]) for row in selected_episodes)
        raise ValueError(f"selected episodes contain only {available} valid target frames, need {count}")

    targets = [
        source_index
        for episode_index in episode_order
        for source_index in sorted(selected[episode_index])
    ]
    if len(targets) != count or len(set(targets)) != count:
        raise AssertionError("target selection must contain exactly the requested number of unique rows")
    return targets


def generate_openloop_eval_visual_caches(
    *,
    split_roots: Iterable[str | Path],
    encoder_ckpt: str | Path = DEFAULT_QWEN35_ENCODER,
    batch_size: int = 8,
    prefetch_batches: int = 2,
) -> dict[str, Any]:
    from tool.navvla.cli.generate_visual_cache import load_visual_encoder
    from tool.navvla.profile_visual_cache import generate_profile_cache_parallel

    profile = default_qwen35_pooled_history_visual_token_profile(encoder_ckpt=str(encoder_ckpt))
    encoder = load_visual_encoder(encoder_ckpt=str(encoder_ckpt), profile=profile)
    reports: dict[str, Any] = {}
    for root_value in split_roots:
        root = Path(root_value)
        reports[str(root)] = generate_profile_cache_parallel(
            root,
            profile=profile,
            encoder=encoder,
            skip_existing=True,
            batch_size=int(batch_size),
            prefetch_batches=int(prefetch_batches),
            input_resize=(256, 256),
        )
    return reports


def _load_source_inventory(source_root: Path) -> dict[str, Any]:
    data_paths = sorted((source_root / "data").glob("chunk-*/part-*.parquet"))
    episode_paths = sorted((source_root / "meta" / "episodes").glob("chunk-*/part-*.parquet"))
    if not data_paths or not episode_paths:
        raise FileNotFoundError(f"missing data or episode shards under {source_root}")
    selection_columns = [
        "index",
        "episode_index",
        "frame_index",
        "action",
        "action.padding_mask",
        "sample.action_available",
    ]
    data = pd.concat(
        [pd.read_parquet(path, columns=selection_columns) for path in data_paths],
        ignore_index=True,
    )
    data = data.sort_values("index", kind="stable").reset_index(drop=True)
    episodes = pd.concat([pd.read_parquet(path) for path in episode_paths], ignore_index=True)
    episodes = episodes.sort_values("episode_index", kind="stable").reset_index(drop=True)
    tasks = pd.read_parquet(source_root / "meta" / "tasks.parquet")
    video_index = pd.read_parquet(source_root / "meta" / "navvla_video_index.parquet")
    info = json.loads((source_root / "meta" / "info.json").read_text(encoding="utf-8"))
    cameras = json.loads((source_root / "meta" / "navvla_cameras.json").read_text(encoding="utf-8"))
    if data["index"].astype(int).tolist() != list(range(len(data))):
        raise ValueError(f"source data index is not contiguous from zero: {source_root}")
    return {
        "root": source_root,
        "data": data,
        "episodes": episodes,
        "tasks": tasks,
        "video_index": video_index,
        "info": info,
        "cameras": cameras,
    }


def _eligible_episodes(
    inventory: dict[str, Any],
    *,
    required_cameras: tuple[str, ...],
    check_video_files: bool = True,
) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    data = inventory["data"]
    episodes = inventory["episodes"]
    video_index = inventory["video_index"]
    cameras = inventory["cameras"]
    video_keys = {str(cameras[name]["video_key"]) for name in required_cameras}
    required_video_rows = video_index[video_index["video_key"].astype(str).isin(video_keys)]
    available_counts = (
        required_video_rows[required_video_rows["available"].astype(bool)]
        .groupby("index", sort=False)["video_key"]
        .nunique()
    )
    media_valid_indexes = set(available_counts[available_counts == len(video_keys)].index.astype(int))
    valid_frames_by_episode: dict[int, list[int]] = {}
    for episode_index, rows in data.groupby("episode_index", sort=False):
        episode_source_indexes = rows["index"].astype(int).tolist()
        if not set(episode_source_indexes).issubset(media_valid_indexes):
            continue
        valid_indexes = []
        for payload in rows.to_dict("records"):
            source_index = int(payload["index"])
            if source_index not in media_valid_indexes:
                continue
            if not bool(payload.get("sample.action_available", False)):
                continue
            padding = np.asarray(payload["action.padding_mask"], dtype=bool).reshape(-1)
            action = np.asarray(
                [np.asarray(step, dtype=np.float64).reshape(-1) for step in payload["action"]],
                dtype=np.float64,
            )
            if padding.size and np.any(~padding) and action.size and np.isfinite(action).all():
                valid_indexes.append(source_index)
        if valid_indexes:
            valid_frames_by_episode[int(episode_index)] = valid_indexes

    video_artifact_cache: dict[Path, int | None] = {}
    candidates = [
        row
        for row in episodes.to_dict("records")
        if int(row["episode_index"]) in valid_frames_by_episode
        and (
            not check_video_files
            or _episode_media_bounds_are_decodable(
                inventory,
                row,
                required_video_keys=video_keys,
                video_artifact_cache=video_artifact_cache,
            )
        )
    ]
    valid_frames_by_episode = {
        int(row["episode_index"]): valid_frames_by_episode[int(row["episode_index"])] for row in candidates
    }
    return candidates, valid_frames_by_episode


def _episode_media_bounds_are_decodable(
    inventory: dict[str, Any],
    episode_row: dict[str, Any],
    *,
    required_video_keys: set[str],
    video_artifact_cache: dict[Path, int | None],
) -> bool:
    source_root = inventory["root"]
    info = inventory["info"]
    chunk_index = int(episode_row["data/chunk_index"])
    file_index = int(episode_row["data/file_index"])
    for video_key in required_video_keys:
        pattern = info["video_path"].get(video_key)
        if pattern is None:
            return False
        path = source_root / pattern.format(chunk_index=chunk_index, file_index=file_index)
        if path not in video_artifact_cache:
            video_artifact_cache[path] = _decodable_video_frame_count(path)
        frame_count = video_artifact_cache[path]
        if frame_count is None:
            return False
        start = float(episode_row.get(f"videos/{video_key}/from_timestamp", 0.0))
        first_frame = int(round(start * float(info["fps"])))
        if frame_count < first_frame + int(episode_row["length"]):
            return False
    return True


def _decodable_video_frame_count(path: Path) -> int | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return None
        return frame_count
    finally:
        capture.release()


def _write_selected_split(
    inventory: dict[str, Any],
    *,
    selected: list[dict[str, Any]],
    target_source_indexes: list[int],
    output_root: Path,
    dataset_name: str,
    split: str,
    selection_hash: str,
    required_cameras: tuple[str, ...],
) -> dict[str, Any]:
    source_root = inventory["root"]
    selected_source_shards = sorted(
        {
            (int(row["data/chunk_index"]), int(row["data/file_index"]))
            for row in selected
        }
    )
    source_data = pd.concat(
        [
            pd.read_parquet(
                source_root
                / "data"
                / f"chunk-{chunk_index:03d}"
                / f"part-{file_index:03d}.parquet"
            )
            for chunk_index, file_index in selected_source_shards
        ],
        ignore_index=True,
    )
    source_tasks = inventory["tasks"]
    source_video_index = inventory["video_index"]
    info = inventory["info"]
    cameras = {
        camera_name: inventory["cameras"][camera_name] for camera_name in required_cameras
    }
    fps = float(info["fps"])
    # One episode per output shard keeps every cropped video independent and avoids
    # reusing any source or destination video frame offsets.
    episodes_per_file = 1
    files_per_chunk = int(info.get("navvla", {}).get("files_per_chunk", info.get("chunks_size", 50)))
    selected_source_episode_indexes = {int(row["episode_index"]) for row in selected}
    selected_data = source_data[
        source_data["episode_index"].astype(int).isin(selected_source_episode_indexes)
    ].copy()
    selected_data = selected_data.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)
    used_source_tasks = sorted(set(selected_data["task_index"].astype(int).tolist()))
    task_remap = {source_index: new_index for new_index, source_index in enumerate(used_source_tasks)}
    episode_remap = {
        int(row["episode_index"]): new_index for new_index, row in enumerate(selected)
    }
    data_index_remap = {
        int(row["index"]): new_index for new_index, row in selected_data.iterrows()
    }

    output_root.joinpath("meta", "episodes").mkdir(parents=True, exist_ok=True)
    data_rows_by_shard: dict[tuple[int, int], list[dict[str, Any]]] = {}
    episode_rows_by_shard: dict[tuple[int, int], list[dict[str, Any]]] = {}
    video_rows: list[dict[str, Any]] = []
    video_crop_jobs: list[dict[str, Any]] = []
    frame_metadata_rows: list[dict[str, Any]] = []
    source_video_by_key = {
        (int(row["index"]), str(row["video_key"])): row
        for row in source_video_index.to_dict("records")
    }
    source_data_by_episode = {
        int(episode_index): rows.sort_values("frame_index", kind="stable")
        for episode_index, rows in selected_data.groupby("episode_index", sort=False)
    }
    dataset_source = str(
        source_tasks.iloc[0].get("dataset_source", dataset_name) if len(source_tasks) else dataset_name
    )
    video_file_count = 0
    for new_episode_index, episode_row in enumerate(selected):
        source_episode_index = int(episode_row["episode_index"])
        chunk_index, file_index = _episode_shard(
            new_episode_index,
            episodes_per_file=episodes_per_file,
            files_per_chunk=files_per_chunk,
        )
        shard = (chunk_index, file_index)
        rows = source_data_by_episode[source_episode_index]
        episode_video_starts = {str(camera["video_key"]): 0 for camera in cameras.values()}
        for local_position, source_row in enumerate(rows.to_dict("records")):
            source_data_index = int(source_row["index"])
            new_data_index = data_index_remap[source_data_index]
            output_row = _jsonable(dict(source_row))
            output_row["episode_index"] = new_episode_index
            output_row["task_index"] = task_remap[int(source_row["task_index"])]
            output_row["observation.state"] = np.asarray(
                source_row["observation.state"], dtype=np.float32
            ).reshape(-1)[:4].tolist()
            output_row["next.done"] = local_position == len(rows) - 1
            output_row["context.index_key"] = (
                f"{dataset_name}/{split}/{episode_row['episode_id']}/"
                f"f{int(source_row['frame_index']):06d}/bats-v1"
            )
            output_row["source_frame_index"] = int(
                source_row.get("source_frame_index", source_row["frame_index"])
            )
            output_row["index"] = new_data_index
            data_rows_by_shard.setdefault(shard, []).append(output_row)
            frame_metadata_rows.append(
                {
                    "index": new_data_index,
                    "source_frame_index": output_row["source_frame_index"],
                    "source_metadata": {
                        "source_root": str(source_root),
                        "source_index": source_data_index,
                        "source_episode_index": source_episode_index,
                    },
                }
            )
            for camera_name, camera in cameras.items():
                video_key = str(camera["video_key"])
                video_row = source_video_by_key[(source_data_index, video_key)]
                if not bool(video_row["available"]):
                    raise ValueError(
                        f"selected frame has unavailable media: index={source_data_index} camera={camera_name}"
                    )
                video_rows.append(
                    {
                        "index": new_data_index,
                        "video_key": video_key,
                        "available": True,
                        "video_frame_index": local_position,
                        "chunk_index": chunk_index,
                        "file_index": file_index,
                    }
                )

        for camera_name, camera in cameras.items():
            video_key = str(camera["video_key"])
            source_video_rows = [
                source_video_by_key[(int(source_index), video_key)]
                for source_index in rows["index"].astype(int).tolist()
            ]
            source_shards = {
                (int(row["chunk_index"]), int(row["file_index"])) for row in source_video_rows
            }
            source_frame_indices = [int(row["video_frame_index"]) for row in source_video_rows]
            if len(source_shards) != 1 or source_frame_indices != list(
                range(source_frame_indices[0], source_frame_indices[0] + len(rows))
            ):
                raise ValueError(
                    f"episode {episode_row['episode_id']} camera {camera_name} is not one contiguous video slice"
                )
            source_chunk_index, source_file_index = next(iter(source_shards))
            source_video_path = source_root / info["video_path"][video_key].format(
                chunk_index=source_chunk_index,
                file_index=source_file_index,
            )
            output_video_path = (
                output_root
                / "videos"
                / video_key
                / f"chunk-{chunk_index:03d}"
                / f"part-{file_index:03d}.mp4"
            )
            video_crop_jobs.append(
                {
                    "source": str(source_video_path),
                    "output": str(output_video_path),
                    "start_frame": source_frame_indices[0],
                    "frame_count": len(rows),
                    "fps": fps,
                }
            )
        video_file_count += len(cameras)

        used_episode_tasks = list(dict.fromkeys(rows["task_index"].astype(int).tolist()))
        task_text_by_index = source_tasks.set_index("task_index")["task"].to_dict()
        output_episode = _jsonable(dict(episode_row))
        output_episode.update(
            {
                "episode_index": new_episode_index,
                "task_index": task_remap[used_episode_tasks[0]],
                "split": split,
                "tasks": [str(task_text_by_index[index]) for index in used_episode_tasks],
                "length": int(len(rows)),
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
            }
        )
        for camera in cameras.values():
            video_key = str(camera["video_key"])
            output_episode[f"videos/{video_key}/from_timestamp"] = episode_video_starts[video_key] / fps
        episode_rows_by_shard.setdefault(shard, []).append(output_episode)
    _write_video_crop_jobs(video_crop_jobs)

    for (chunk_index, file_index), rows in sorted(data_rows_by_shard.items()):
        path = output_root / "data" / f"chunk-{chunk_index:03d}" / f"part-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
    for (chunk_index, file_index), rows in sorted(episode_rows_by_shard.items()):
        path = output_root / "meta" / "episodes" / f"chunk-{chunk_index:03d}" / f"part-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)

    task_rows = []
    source_task_by_index = source_tasks.set_index("task_index", drop=False)
    for source_task_index in used_source_tasks:
        row = _jsonable(dict(source_task_by_index.loc[source_task_index]))
        row["task_index"] = task_remap[source_task_index]
        task_rows.append(row)
    tasks = pd.DataFrame(task_rows)
    tasks["task_index"] = tasks["task_index"].astype("int64")
    tasks.to_parquet(output_root / "meta" / "tasks.parquet", index=False)
    pd.DataFrame(video_rows).to_parquet(output_root / "meta" / "navvla_video_index.parquet", index=False)
    _write_jsonl(output_root / "meta" / "navvla_frame_metadata.jsonl", frame_metadata_rows)
    _write_navvla_tasks(output_root / "meta" / "navvla_tasks.jsonl", task_rows)
    _write_json(output_root / "meta" / "navvla_cameras.json", cameras)
    _write_subset_info(
        output_root,
        source_info=info,
        cameras=cameras,
        split=split,
        total_frames=len(selected_data),
        total_episodes=len(selected),
        total_tasks=len(task_rows),
        total_videos=video_file_count,
    )
    _write_subset_modality(output_root, source_root=source_root, cameras=cameras)
    _write_json(
        output_root / "meta" / "navvla_schema_ext.json",
        {
            "schema_version": "0.1",
            "context_policy_version": "bats-v1",
            "cache_policy_version": "profile-cache-v1",
            "history_fields": ["context.index_key"],
            "frame_metadata": "meta/navvla_frame_metadata.jsonl",
            "video_index": "meta/navvla_video_index.parquet",
            "context_index_manifest": "meta/navvla_context_index_manifest.json",
            "context_index": "meta/context_index/budget_<budget>",
            "context_meta": "meta/context_index/budget_<budget>/context_meta.parquet",
            "context_arrays": "meta/context_index/budget_<budget>/context_arrays",
            "context_debug": f"cache/context_index_debug/budget_<budget>/{split}.parquet",
        },
    )
    _write_json(
        output_root / "conversion_report.json",
        {
            "version": OPENLOOP_EVAL_VERSION,
            "source_root": str(source_root),
            "selection_hash": selection_hash,
            "selected_source_episode_indices": [int(row["episode_index"]) for row in selected],
            "selected_source_target_indexes": [int(value) for value in target_source_indexes],
            "episode_index_remap": {str(key): value for key, value in episode_remap.items()},
            "task_index_remap": {str(key): value for key, value in task_remap.items()},
            "total_frames": len(selected_data),
            "total_episodes": len(selected),
            "video_build": {
                "independent_reencoded_videos": True,
                "files": video_file_count,
                "index_rows": len(video_rows),
                "workers": _video_workers(len(video_crop_jobs)),
            },
            "rejected_rows": [],
        },
    )
    return {
        "total_frames": len(selected_data),
        "dataset_source": dataset_source,
        "source_to_new_data_index": data_index_remap,
        "source_to_new_episode_index": episode_remap,
    }


def _write_video_crop_jobs(jobs: list[dict[str, Any]]) -> None:
    if not jobs:
        return
    workers = _video_workers(len(jobs))
    if workers == 1:
        for job in jobs:
            _write_video_crop_job(job)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for completed, _result in enumerate(executor.map(_write_video_crop_job, jobs, chunksize=1), start=1):
            if completed == len(jobs) or completed % 20 == 0:
                print(f"video crops: {completed}/{len(jobs)}", flush=True)


def _video_workers(job_count: int) -> int:
    configured = max(1, int(os.environ.get("NAVVLA_OPENLOOP_VIDEO_WORKERS", "8")))
    return min(configured, max(1, int(job_count)))


def _write_video_crop_job(job: dict[str, Any]) -> str:
    source = Path(job["source"])
    output = Path(job["output"])
    start_frame = int(job["start_frame"])
    frame_count = int(job["frame_count"])
    fps = float(job["fps"])
    output.parent.mkdir(parents=True, exist_ok=True)
    end_frame = start_frame + frame_count
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        str(output),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg video crop failed for {source} -> {output}: "
            f"{completed.stderr.decode('utf-8', errors='replace')}"
        )
    capture = cv2.VideoCapture(str(output))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cropped video does not open: {output}")
        actual_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if actual_frames != frame_count:
        raise ValueError(
            f"cropped video frame count mismatch for {output}: expected {frame_count}, got {actual_frames}"
        )
    return str(output)


def _write_subset_info(
    output_root: Path,
    *,
    source_info: dict[str, Any],
    cameras: dict[str, dict[str, Any]],
    split: str,
    total_frames: int,
    total_episodes: int,
    total_tasks: int,
    total_videos: int,
) -> None:
    info = json.loads(json.dumps(source_info))
    info.update(
        {
            "total_frames": int(total_frames),
            "total_episodes": int(total_episodes),
            "total_tasks": int(total_tasks),
            "total_videos": int(total_videos),
            "splits": {split: f"0:{int(total_episodes)}"},
        }
    )
    navvla = info.setdefault("navvla", {})
    navvla["episodes_per_file"] = 1
    navvla["state_dim"] = 4
    navvla["state_mode"] = "episode_relative_first_body_aligned_pose_xyz_yaw"
    navvla.pop("missing_field_policy", None)
    state_feature = info.setdefault("features", {}).setdefault("observation.state", {})
    state_feature.update(
        {
            "dtype": "float32",
            "shape": [4],
            "names": ["x", "y", "z", "yaw"],
        }
    )
    info["features"].setdefault(
        "source_frame_index",
        {"dtype": "int64", "shape": [1], "names": ["source_frame_index"]},
    )
    selected_video_keys = {str(camera["video_key"]) for camera in cameras.values()}
    info["video_path"] = {
        key: value for key, value in info.get("video_path", {}).items() if key in selected_video_keys
    }
    info["features"] = {
        key: value
        for key, value in info["features"].items()
        if not key.startswith("observation.images.")
        or key.removeprefix("observation.images.") in selected_video_keys
    }
    _write_json(output_root / "meta" / "info.json", info)


def _write_subset_modality(
    output_root: Path,
    *,
    source_root: Path,
    cameras: dict[str, dict[str, Any]],
) -> None:
    source_path = source_root / "meta" / "modality.json"
    modality = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    modality["state"] = {
        name: {
            "start": index,
            "end": index + 1,
            "absolute": True,
            "dtype": "float32",
            "original_key": "observation.state",
        }
        for index, name in enumerate(("x", "y", "z", "yaw"))
    }
    selected_video_keys = {str(camera["video_key"]) for camera in cameras.values()}
    modality["video"] = {
        key: value for key, value in modality.get("video", {}).items() if key in selected_video_keys
    }
    _write_json(output_root / "meta" / "modality.json", modality)


def _write_navvla_tasks(path: Path, task_rows: list[dict[str, Any]]) -> None:
    rows = []
    required = ("task_index", "task_type", "task_subtype", "platform_text", "dataset_source", "answer")
    for task in task_rows:
        row = {key: task.get(key) for key in required}
        rows.append(row)
    _write_jsonl(path, rows)


def _selection_report(
    split_reports: list[dict[str, Any]],
    *,
    seed: int,
    episodes_per_split: int,
    targets_per_split: int,
) -> dict[str, Any]:
    total_episodes = sum(int(report["selected_episodes"]) for report in split_reports)
    total_targets = sum(int(report["selected_targets"]) for report in split_reports)
    return {
        "version": OPENLOOP_EVAL_VERSION,
        "seed": int(seed),
        "episodes_per_split": int(episodes_per_split),
        "targets_per_split": int(targets_per_split),
        "total_splits": len(split_reports),
        "total_episodes": total_episodes,
        "total_targets": total_targets,
        "splits": split_reports,
    }


def _validate_dataset_spec(spec: OpenLoopDatasetSpec) -> None:
    if not spec.name:
        raise ValueError("dataset name must not be empty")
    if not spec.required_cameras:
        raise ValueError(f"{spec.name} required_cameras must not be empty")


def _episode_shard(
    episode_index: int,
    *,
    episodes_per_file: int,
    files_per_chunk: int,
) -> tuple[int, int]:
    linear_file = int(episode_index) // int(episodes_per_file)
    return linear_file // int(files_per_chunk), linear_file % int(files_per_chunk)


def _even_positions(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    if count == 1:
        return [0]
    if count >= length:
        return list(range(length))
    return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})


def _stable_digest(*values: Any) -> str:
    return hashlib.sha256(":".join(str(value) for value in values).encode("utf-8")).hexdigest()


def _value_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _merge_conversion_report(root: Path, updates: dict[str, Any]) -> None:
    path = root / "conversion_report.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(_jsonable(updates))
    _write_json(path, payload)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_jsonable(row), ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
