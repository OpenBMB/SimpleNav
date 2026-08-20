from __future__ import annotations

"""CosFly-only split and validation-anchor preparation."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TRACE_TYPES = ("ORI", "aug_001")
DEFAULT_SEEN_PARENT_QUOTAS = {
    "Town01": 1,
    "Town02": 2,
    "Town03": 1,
    "Town04": 1,
    "Town05": 3,
    "Town06": 1,
    "Town07": 3,
}


def build_cosfly_split_manifest(
    source_root: str | Path,
    *,
    output_path: str | Path | None = None,
    seed: int = 42,
    seen_parent_quotas: dict[str, int] | None = None,
    seen_anchor_count_per_trace: int = 60,
    unseen_anchor_count_per_trace: int = 520,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    quotas = dict(DEFAULT_SEEN_PARENT_QUOTAS if seen_parent_quotas is None else seen_parent_quotas)
    complete, excluded = inventory_cosfly_parent_pairs(source_root)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        groups[str(row["scenario_group_key"])].append(row)

    seen_group_keys: set[str] = set()
    for town_family, quota in quotas.items():
        family_groups = [
            rows for rows in groups.values() if rows[0]["town_family"] == town_family
        ]
        if not family_groups:
            raise ValueError(f"CosFly has no complete scenario groups for required family {town_family}")
        seen_group_keys.update(
            rows[0]["scenario_group_key"]
            for rows in _select_groups_for_parent_quota(
                family_groups,
                parent_quota=int(quota),
                seed=seed,
                town_family=town_family,
            )
        )

    parents = []
    for row in complete:
        family = str(row["town_family"])
        group_key = str(row["scenario_group_key"])
        if family == "Town10":
            split = "unseen"
        elif group_key in seen_group_keys:
            split = "seen"
        else:
            split = "train"
        parents.append({**row, "split": split})
    parents.sort(key=lambda row: str(row["parent_id"]))

    seen_parents = [row for row in parents if row["split"] == "seen"]
    unseen_parents = [row for row in parents if row["split"] == "unseen"]
    anchors = []
    anchors.extend(
        _build_paired_anchors(
            seen_parents,
            split="seen",
            count_per_trace=int(seen_anchor_count_per_trace),
            seed=seed,
        )
    )
    anchors.extend(
        _build_paired_anchors(
            unseen_parents,
            split="unseen",
            count_per_trace=int(unseen_anchor_count_per_trace),
            seed=seed,
        )
    )
    anchors.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["parent_id"]),
            int(row["frame_index"]),
            str(row["trace_type"]),
        )
    )

    validation_counts = {
        trace_type: sum(row["trace_type"] == trace_type for row in anchors)
        for trace_type in TRACE_TYPES
    }
    validation_counts.update(
        {
            "seen": sum(row["split"] == "seen" for row in anchors),
            "unseen": sum(row["split"] == "unseen" for row in anchors),
            "total": len(anchors),
        }
    )
    parent_counts = {
        split: sum(row["split"] == split for row in parents)
        for split in ("train", "seen", "unseen")
    }
    manifest = {
        "schema": "cosfly_navvla_split_manifest_v1",
        "source_root": str(source_root),
        "seed": int(seed),
        "trace_types": list(TRACE_TYPES),
        "split_policy": {
            "group_by": ["town_family", "source.path_index", "source.scenario_id", "source.scenario_name"],
            "seen_town_families": sorted(quotas),
            "unseen_town_family": "Town10",
            "seen_parent_quotas": quotas,
            "anchor_eligible_frame_range": "[4, num_frames-9] inclusive",
        },
        "parents": parents,
        "excluded_parents": excluded,
        "validation_anchors": anchors,
        "summary": {
            "complete_parent_pairs": len(parents),
            "excluded_parent_pairs": len(excluded),
            "parent_counts": parent_counts,
            "episode_counts": {key: value * len(TRACE_TYPES) for key, value in parent_counts.items()},
            "validation_anchor_counts": validation_counts,
        },
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def inventory_cosfly_parent_pairs(
    source_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_root = Path(source_root).resolve()
    parent_roots = sorted(source_root.glob("Town*/trajectory_*"), key=lambda path: path.as_posix())
    if not parent_roots:
        raise FileNotFoundError(f"no CosFly parent trajectories found under {source_root}")

    complete = []
    excluded = []
    for parent_root in parent_roots:
        parent_id = parent_root.relative_to(source_root).as_posix()
        try:
            payloads = {}
            for trace_type in TRACE_TYPES:
                trajectory_path = parent_root / trace_type / "trajectory.json"
                if not trajectory_path.is_file():
                    raise ValueError(f"missing {trace_type}/trajectory.json")
                payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
                points = payload.get("points")
                if payload.get("schema") != "drone_nav_traj_v7" or not isinstance(points, list) or not points:
                    raise ValueError(f"invalid {trace_type} v7 trajectory payload")
                missing_frames = []
                for point in points:
                    frame_index = int(point.get("index", -1))
                    image_path = (
                        parent_root
                        / trace_type
                        / "frames_playback"
                        / f"frame_{frame_index:05d}"
                        / "rgb.png"
                    )
                    if not image_path.is_file() or image_path.stat().st_size <= 0:
                        missing_frames.append(frame_index)
                if missing_frames:
                    raise ValueError(
                        f"{trace_type} missing {len(missing_frames)} RGB frames; first={missing_frames[:5]}"
                    )
                payloads[trace_type] = payload

            frame_counts = {trace_type: len(payloads[trace_type]["points"]) for trace_type in TRACE_TYPES}
            if len(set(frame_counts.values())) != 1:
                raise ValueError(f"ORI/aug frame count mismatch: {frame_counts}")
            source_rows = {trace_type: payloads[trace_type].get("source") or {} for trace_type in TRACE_TYPES}
            source_keys = {trace_type: _source_scenario_tuple(source_rows[trace_type]) for trace_type in TRACE_TYPES}
            if source_keys["ORI"] != source_keys["aug_001"]:
                raise ValueError(f"ORI/aug source scenario mismatch: {source_keys}")
            town = parent_root.parent.name
            town_family = normalize_town_family(town)
            scenario_group_key = _scenario_group_key(town_family, source_keys["ORI"])
            complete.append(
                {
                    "parent_id": parent_id,
                    "town": town,
                    "town_family": town_family,
                    "scenario_group_key": scenario_group_key,
                    "source": {
                        "path_index": source_keys["ORI"][0],
                        "scenario_id": source_keys["ORI"][1],
                        "scenario_name": source_keys["ORI"][2],
                    },
                    "num_frames": frame_counts["ORI"],
                }
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            excluded.append({"parent_id": parent_id, "reason": str(exc)})
    return complete, excluded


def normalize_town_family(town: str) -> str:
    if town in {"Town10HD", "Town10HD_Opt"}:
        return "Town10"
    return town.removesuffix("_Opt")


def _source_scenario_tuple(source: dict[str, Any]) -> tuple[Any, Any, str]:
    return source.get("path_index"), source.get("scenario_id"), str(source.get("scenario_name", ""))


def _scenario_group_key(town_family: str, source_key: tuple[Any, Any, str]) -> str:
    return json.dumps([town_family, *source_key], separators=(",", ":"), ensure_ascii=True)


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _select_groups_for_parent_quota(
    groups: list[list[dict[str, Any]]],
    *,
    parent_quota: int,
    seed: int,
    town_family: str,
) -> list[list[dict[str, Any]]]:
    ordered = sorted(
        groups,
        key=lambda rows: _stable_rank(seed, str(rows[0]["scenario_group_key"])),
    )
    possibilities: dict[int, list[int]] = {0: []}
    for index, rows in enumerate(ordered):
        group_size = len(rows)
        updated = dict(possibilities)
        for total, selected in possibilities.items():
            next_total = total + group_size
            if next_total <= parent_quota and next_total not in updated:
                updated[next_total] = [*selected, index]
        possibilities = updated
    if parent_quota not in possibilities:
        sizes = [len(rows) for rows in ordered]
        raise ValueError(
            f"cannot select exactly {parent_quota} seen parents for {town_family} without splitting "
            f"scenario groups; group sizes={sizes}"
        )
    return [ordered[index] for index in possibilities[parent_quota]]


def _build_paired_anchors(
    parents: list[dict[str, Any]],
    *,
    split: str,
    count_per_trace: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count_per_trace < 0:
        raise ValueError(f"anchor count must be non-negative, got {count_per_trace}")
    if count_per_trace and not parents:
        raise ValueError(f"cannot allocate {count_per_trace} anchors without {split} parents")
    ordered = sorted(parents, key=lambda row: _stable_rank(seed, str(row["parent_id"])))
    base, extras = divmod(count_per_trace, len(ordered)) if ordered else (0, 0)
    anchors = []
    for parent_index, parent in enumerate(ordered):
        count = base + int(parent_index < extras)
        frame_indices = _evenly_spaced_anchor_indices(int(parent["num_frames"]), count=count)
        parent_id = str(parent["parent_id"])
        town, trajectory_name = parent_id.split("/", 1)
        for frame_index in frame_indices:
            for trace_type in TRACE_TYPES:
                anchors.append(
                    {
                        "split": split,
                        "parent_id": parent_id,
                        "trace_type": trace_type,
                        "episode_id": f"{town}__{trajectory_name}__{trace_type}",
                        "frame_index": int(frame_index),
                    }
                )
    return anchors


def _evenly_spaced_anchor_indices(num_frames: int, *, count: int) -> list[int]:
    if count == 0:
        return []
    low = 4
    high = int(num_frames) - 9
    available = high - low + 1
    if available < count:
        raise ValueError(
            f"trajectory with {num_frames} frames has only {max(available, 0)} eligible anchors, needs {count}"
        )
    if count == 1:
        return [(low + high) // 2]
    indices = [round(low + index * (high - low) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise AssertionError(f"anchor selection produced duplicate indices: {indices}")
    return indices
