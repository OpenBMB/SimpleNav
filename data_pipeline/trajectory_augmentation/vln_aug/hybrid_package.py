"""Build a valid package that keeps completed legacy renders and replaces the rest."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

from .aerialvln_export import validate_trajectory_package


def select_episode_payloads(
    legacy: Iterable[dict], fixed: Iterable[dict], completed_episode_ids: set[str]
) -> list[dict]:
    """Select legacy data for completed episodes and fixed-stride data otherwise."""
    legacy = list(legacy)
    fixed = list(fixed)
    legacy_ids = [str(item["episode_id"]) for item in legacy]
    fixed_ids = [str(item["episode_id"]) for item in fixed]
    if legacy_ids != fixed_ids:
        raise ValueError("legacy and fixed packages do not have the same episode order")
    if len(set(legacy_ids)) != len(legacy_ids):
        raise ValueError("episode ids must be unique")
    return [
        old if str(old["episode_id"]) in completed_episode_ids else new
        for old, new in zip(legacy, fixed)
    ]


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield line, json.loads(line)


def _iter_request_groups(path: Path):
    current_id = None
    group: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            payload = json.loads(line)
            episode_id = str(payload["episode_id"])
            if current_id is None:
                current_id = episode_id
            elif episode_id != current_id:
                yield current_id, group
                current_id, group = episode_id, []
            group.append(line)
    if current_id is not None:
        yield current_id, group


def _paired_rows(legacy_path: Path, fixed_path: Path, label: str):
    old_rows = _iter_jsonl(legacy_path)
    new_rows = _iter_jsonl(fixed_path)
    while True:
        try:
            old_line, old = next(old_rows)
        except StopIteration:
            try:
                next(new_rows)
            except StopIteration:
                return
            raise ValueError(f"fixed {label} has extra rows")
        try:
            new_line, new = next(new_rows)
        except StopIteration as error:
            raise ValueError(f"legacy {label} has extra rows") from error
        old_id = str(old["episode_id"])
        new_id = str(new["episode_id"])
        if old_id != new_id:
            raise ValueError(f"{label} episode order mismatch: {old_id} != {new_id}")
        yield old_id, old_line, old, new_line, new


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_hybrid_package(
    legacy_package: str | Path,
    fixed_package: str | Path,
    output_package: str | Path,
    completed_episode_ids: set[str],
) -> dict:
    """Create a mixed package whose old completed videos match its metadata exactly."""
    legacy = Path(legacy_package)
    fixed = Path(fixed_package)
    output = Path(output_package)
    if output.exists():
        raise FileExistsError(output)
    for root in (legacy, fixed):
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError(root / "manifest.json")

    shutil.copytree(
        legacy,
        output,
        ignore=shutil.ignore_patterns(".render_staging", "videos", "meta", ".waypoint_publish.json"),
    )
    try:
        output_trajectories = output / "trajectories"
        output_render = output / "render"
        episode_count = 0
        metadata_rows = 0
        policy_counts: Counter[str] = Counter()
        stride_counts: Counter[int] = Counter()
        gap_counts: Counter[int] = Counter()

        episodes_out = output_trajectories / "episodes.jsonl"
        metadata_out = output_trajectories / "augmentation_metadata.jsonl"
        train_out = output_trajectories / "train.json"
        with (
            episodes_out.open("w", encoding="utf-8") as episode_handle,
            metadata_out.open("w", encoding="utf-8") as metadata_handle,
            train_out.open("w", encoding="utf-8") as train_handle,
        ):
            train_handle.write('{"episodes":[')
            first = True
            episode_rows = _paired_rows(
                legacy / "trajectories" / "episodes.jsonl",
                fixed / "trajectories" / "episodes.jsonl",
                "episodes",
            )
            metadata_rows_iter = _paired_rows(
                legacy / "trajectories" / "augmentation_metadata.jsonl",
                fixed / "trajectories" / "augmentation_metadata.jsonl",
                "augmentation metadata",
            )
            for episode_row, metadata_row in zip(episode_rows, metadata_rows_iter):
                episode_id, old_episode_line, _, new_episode_line, _ = episode_row
                metadata_id, old_meta_line, old_meta, new_meta_line, new_meta = metadata_row
                if episode_id != metadata_id:
                    raise ValueError(f"episode/metadata mismatch for {episode_id}")
                use_legacy = episode_id in completed_episode_ids
                selected_episode_line = old_episode_line if use_legacy else new_episode_line
                selected_meta_line = old_meta_line if use_legacy else new_meta_line
                selected_meta = old_meta if use_legacy else new_meta
                episode_handle.write(selected_episode_line)
                if not first:
                    train_handle.write(",")
                train_handle.write(selected_episode_line.strip())
                first = False
                metadata_handle.write(selected_meta_line)
                policy_counts[str(selected_meta.get("collection_stride_policy", "fixed_per_episode"))] += 1
                stride = selected_meta.get("collection_stride_waypoints")
                if stride is not None:
                    stride_counts[int(stride)] += 1
                indices = [int(value) for value in selected_meta["collection_waypoint_indices"]]
                for left, right in zip(indices, indices[1:]):
                    gap_counts[right - left] += 1
                episode_count += 1
                metadata_rows += 1
            try:
                next(episode_rows)
                raise ValueError("extra episode rows")
            except StopIteration:
                pass
            try:
                next(metadata_rows_iter)
                raise ValueError("extra metadata rows")
            except StopIteration:
                pass
            train_handle.write("]}\n")
        if episode_count != metadata_rows:
            raise ValueError("episode and metadata row counts differ")

        metrics_name = "trajectory_metrics.jsonl"
        old_metrics = legacy / metrics_name
        new_metrics = fixed / metrics_name
        if old_metrics.is_file() != new_metrics.is_file():
            raise ValueError("trajectory metrics presence differs")
        if old_metrics.is_file():
            with (output / metrics_name).open("w", encoding="utf-8") as handle:
                for episode_id, old_line, _, new_line, _ in _paired_rows(
                    old_metrics, new_metrics, "trajectory metrics"
                ):
                    handle.write(old_line if episode_id in completed_episode_ids else new_line)

        request_count = 0
        with (output_render / "render_requests.jsonl").open("w", encoding="utf-8") as handle:
            old_groups = _iter_request_groups(legacy / "render" / "render_requests.jsonl")
            new_groups = _iter_request_groups(fixed / "render" / "render_requests.jsonl")
            while True:
                try:
                    old_id, old_group = next(old_groups)
                except StopIteration:
                    try:
                        next(new_groups)
                    except StopIteration:
                        break
                    raise ValueError("fixed render requests have extra episode groups")
                try:
                    new_id, new_group = next(new_groups)
                except StopIteration as error:
                    raise ValueError("legacy render requests have extra episode groups") from error
                if old_id != new_id:
                    raise ValueError(f"render request episode order mismatch: {old_id} != {new_id}")
                selected = old_group if old_id in completed_episode_ids else new_group
                handle.writelines(selected)
                request_count += len(selected)

        manifest = json.loads((legacy / "manifest.json").read_text(encoding="utf-8"))
        manifest.update({
            "episode_count": episode_count,
            "render_request_count": request_count,
            "image_stride_episode_counts": {str(key): value for key, value in sorted(stride_counts.items())},
            "image_sampling_policy_episode_counts": dict(sorted(policy_counts.items())),
            "image_interval_gap_counts": {str(key): value for key, value in sorted(gap_counts.items())},
            "hybrid_collection": {
                "legacy_completed_episode_count": len(completed_episode_ids),
                "legacy_stride_choices_waypoints": [3, 4, 5, 6],
                "fixed_stride_remaining_waypoints": 5,
                "fixed_stride_remaining_episode_count": episode_count - len(completed_episode_ids),
            },
        })
        _atomic_json(output / "manifest.json", manifest)
        validation = validate_trajectory_package(output)
        if not validation.get("valid"):
            raise ValueError("hybrid trajectory package validation failed")
        _atomic_json(output / "validation.json", validation)
        _atomic_json(output / "validation" / "summary.json", validation)
        return {
            "episode_count": episode_count,
            "render_request_count": request_count,
            "completed_episode_count": len(completed_episode_ids),
            "validation": validation,
        }
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
