from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.interpolate import PchipInterpolator

from tool.navvla.statistics import body_frame_action_from_pose, build_action_statistics, wrap_to_pi


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = {
    "aerialvln": REPO_ROOT / "local/data/AerialVLN_lerobot/vln_train",
    "openfly": REPO_ROOT / "local/data/OpenFly_lerobot/vln_train",
}
TARGET_SPACING_METERS = 1.0
MAX_YAW_STEP_RADIANS = math.radians(15.0)
CONTROL_FREQUENCY_HZ = 1.0
ACTION_HORIZON = 8
TIMESTAMP_POLICY = "original_frame_state_dense_waypoint_index_over_1hz"


def build_dense_waypoints(
    states: np.ndarray,
    *,
    target_spacing: float = TARGET_SPACING_METERS,
    max_yaw_step: float = MAX_YAW_STEP_RADIANS,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] < 4:
        raise ValueError(f"states must have shape [N, >=4], got {states.shape}")
    if states.shape[0] == 0:
        raise ValueError("cannot interpolate an empty episode")
    if target_spacing <= 0 or max_yaw_step <= 0:
        raise ValueError("target_spacing and max_yaw_step must be positive")
    states = states[:, :4]
    if not np.isfinite(states).all():
        raise ValueError("states contain non-finite values")
    if states.shape[0] == 1:
        return states.copy(), np.asarray([0], dtype=np.int64)

    position_distance = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1)
    yaw_delta = np.asarray(wrap_to_pi(np.diff(states[:, 3])), dtype=np.float64)
    position_steps = np.floor(position_distance / float(target_spacing) + 0.5).astype(np.int64)
    yaw_steps = np.floor(np.abs(yaw_delta) / float(max_yaw_step) + 0.5).astype(np.int64)
    segment_steps = np.maximum(1, np.maximum(position_steps, yaw_steps))

    return _sample_pchip(states, yaw_delta=yaw_delta, segment_steps=segment_steps)


def _sample_pchip(
    states: np.ndarray,
    *,
    yaw_delta: np.ndarray,
    segment_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame_waypoints = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(segment_steps, dtype=np.int64)]
    )
    knot_time = frame_waypoints.astype(np.float64)
    unwrapped_yaw = np.empty(states.shape[0], dtype=np.float64)
    unwrapped_yaw[0] = float(states[0, 3])
    unwrapped_yaw[1:] = unwrapped_yaw[0] + np.cumsum(yaw_delta)
    position_spline = PchipInterpolator(knot_time, states[:, :3], axis=0)
    dense_parts: list[np.ndarray] = []
    for segment_index, count_value in enumerate(segment_steps):
        count = int(count_value)
        samples = max(65, count * 32 + 1)
        parameter = np.linspace(knot_time[segment_index], knot_time[segment_index + 1], samples)
        sampled_position = np.asarray(position_spline(parameter), dtype=np.float64)
        cumulative = np.concatenate(
            [np.asarray([0.0]), np.cumsum(np.linalg.norm(np.diff(sampled_position, axis=0), axis=1))]
        )
        if float(cumulative[-1]) <= 1e-12:
            alpha = np.linspace(0.0, 1.0, count + 1)
            position = np.repeat(states[segment_index : segment_index + 1, :3], count + 1, axis=0)
        else:
            targets = np.linspace(0.0, float(cumulative[-1]), count + 1)
            alpha = targets / float(cumulative[-1])
            position = np.column_stack(
                [np.interp(targets, cumulative, sampled_position[:, axis]) for axis in range(3)]
            )
        yaw = unwrapped_yaw[segment_index] + alpha * yaw_delta[segment_index]
        part = np.column_stack([position, np.asarray(wrap_to_pi(yaw), dtype=np.float64)])
        if segment_index:
            part = part[1:]
        dense_parts.append(part)
    dense = np.concatenate(dense_parts, axis=0)
    dense[frame_waypoints, :3] = states[:, :3]
    dense[frame_waypoints, 3] = states[:, 3]
    return dense, frame_waypoints


def build_episode_updates(
    states: np.ndarray,
    *,
    horizon: int = ACTION_HORIZON,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    states = np.asarray(states, dtype=np.float64)[:, :4]
    dense, frame_waypoints = build_dense_waypoints(states)
    timestamps = frame_waypoints.astype(np.float64) / CONTROL_FREQUENCY_HZ
    actions = np.zeros((states.shape[0], horizon, 4), dtype=np.float32)
    for frame_position, waypoint_index in enumerate(frame_waypoints):
        future = dense[int(waypoint_index) + 1 : int(waypoint_index) + 1 + horizon]
        for action_index, future_pose in enumerate(future):
            actions[frame_position, action_index] = body_frame_action_from_pose(states[frame_position], future_pose)

    position_steps = np.linalg.norm(np.diff(dense[:, :3], axis=0), axis=1)
    yaw_steps = np.abs(np.asarray(wrap_to_pi(np.diff(dense[:, 3])), dtype=np.float64))
    return timestamps, actions, {
        "frames": int(states.shape[0]),
        "dense_waypoints": int(dense.shape[0]),
        "max_position_step_m": float(position_steps.max()) if position_steps.size else 0.0,
        "max_yaw_step_deg": math.degrees(float(yaw_steps.max())) if yaw_steps.size else 0.0,
    }


def interpolate_dataset_root(
    root: str | Path,
    *,
    dataset_kind: str,
    apply: bool = False,
    max_files: int | None = None,
) -> dict[str, Any]:
    root = Path(root)
    dataset_kind = str(dataset_kind).strip().lower()
    if dataset_kind not in DEFAULT_ROOTS:
        raise ValueError(f"dataset_kind must be one of {sorted(DEFAULT_ROOTS)}, got {dataset_kind!r}")
    if root.name != "vln_train":
        raise ValueError(f"this temporary script only accepts a vln_train root, got {root}")
    if apply and max_files is not None:
        raise ValueError("--max-files is dry-run only; partial in-place dataset updates are forbidden")

    info_path = root / "meta" / "info.json"
    statistics_path = root / "dataset_statistics.json"
    if not info_path.exists() or not statistics_path.exists():
        raise FileNotFoundError(f"not a complete NavVLA LeRobot root: {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    horizon = int(info.get("navvla", {}).get("action_horizon", ACTION_HORIZON))
    if horizon != ACTION_HORIZON:
        raise ValueError(f"expected action horizon {ACTION_HORIZON}, got {horizon}")

    data_paths = sorted((root / "data").glob("chunk-*/part-*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"no data parquet files found under {root / 'data'}")
    selected_paths = data_paths if max_files is None else data_paths[: int(max_files)]
    total_frames = int(info["total_frames"])
    index_capacity = _global_index_capacity(data_paths) if apply else total_frames
    new_timestamp_by_index = np.full(index_capacity, np.nan, dtype=np.float64) if apply else None
    action_memmap_path: Path | None = None
    action_steps: np.memmap | None = None
    if apply:
        handle = tempfile.NamedTemporaryFile(prefix=".interpolated_action_steps_", suffix=".bin", dir=root, delete=False)
        action_memmap_path = Path(handle.name)
        handle.close()
        action_steps = np.memmap(action_memmap_path, mode="w+", dtype=np.float32, shape=(total_frames * horizon, 4))

    report: dict[str, Any] = {
        "root": str(root),
        "dataset_kind": dataset_kind,
        "mode": "apply" if apply else "dry_run",
        "files": 0,
        "episodes": 0,
        "rows": 0,
        "dense_waypoints": 0,
        "max_position_step_m": 0.0,
        "max_yaw_step_deg": 0.0,
    }
    try:
        for data_path in selected_paths:
            table = pq.read_table(data_path)
            _validate_data_table(table, data_path=data_path)
            updated_timestamp = np.asarray(table.column("timestamp").combine_chunks().to_numpy(), dtype=np.float64).copy()
            updated_action = np.zeros((table.num_rows, horizon, 4), dtype=np.float32)
            episode_values = np.asarray(table.column("episode_index").combine_chunks().to_numpy(), dtype=np.int64)
            frame_values = np.asarray(table.column("frame_index").combine_chunks().to_numpy(), dtype=np.int64)
            index_values = np.asarray(table.column("index").combine_chunks().to_numpy(), dtype=np.int64)
            masks = table.column("action.padding_mask").combine_chunks().to_pylist()
            if any(any(bool(value) for value in row) for row in masks):
                raise ValueError(f"{data_path} contains True action.padding_mask entries; refusing to change existing tail semantics")

            starts = np.concatenate(([0], np.flatnonzero(np.diff(episode_values) != 0) + 1, [table.num_rows]))
            for group_index in range(len(starts) - 1):
                start = int(starts[group_index])
                stop = int(starts[group_index + 1])
                episode_index = int(episode_values[start])
                frames = frame_values[start:stop]
                if not np.array_equal(frames, np.arange(stop - start, dtype=np.int64)):
                    raise ValueError(f"episode {episode_index} in {data_path} does not contain contiguous frame_index from zero")
                indices = index_values[start:stop]
                if not np.array_equal(indices, int(indices[0]) + np.arange(stop - start, dtype=np.int64)):
                    raise ValueError(f"episode {episode_index} in {data_path} does not contain contiguous global index values")
                states = np.asarray(table.column("observation.state").slice(start, stop - start).to_pylist(), dtype=np.float64)
                timestamps, actions, episode_report = build_episode_updates(states, horizon=horizon)
                updated_timestamp[start:stop] = timestamps
                updated_action[start:stop] = actions
                report["episodes"] += 1
                report["dense_waypoints"] += int(episode_report["dense_waypoints"])
                report["max_position_step_m"] = max(
                    float(report["max_position_step_m"]), float(episode_report["max_position_step_m"])
                )
                report["max_yaw_step_deg"] = max(
                    float(report["max_yaw_step_deg"]), float(episode_report["max_yaw_step_deg"])
                )

            report["files"] += 1
            report["rows"] += int(table.num_rows)
            if not apply:
                continue
            assert new_timestamp_by_index is not None and action_steps is not None
            if np.any(index_values < 0) or np.any(index_values >= index_capacity):
                raise ValueError(f"global index out of range in {data_path}")
            new_timestamp_by_index[index_values] = updated_timestamp
            action_offset = (int(report["rows"]) - int(table.num_rows)) * horizon
            action_steps[action_offset : action_offset + int(table.num_rows) * horizon] = updated_action.reshape(-1, 4)
            _write_updated_data_table(
                data_path,
                table=table,
                timestamp=updated_timestamp,
                action=updated_action,
            )

        if not apply:
            return report
        assert new_timestamp_by_index is not None and action_steps is not None
        if report["rows"] != total_frames or int(np.isfinite(new_timestamp_by_index).sum()) != total_frames:
            raise ValueError(
                f"full apply did not cover every row: processed={report['rows']} expected={total_frames}"
            )
        action_steps.flush()
        _update_statistics(statistics_path, action_steps=action_steps)
        _update_info(info_path)
        _update_platform_text(root)
        return report
    finally:
        if action_steps is not None:
            del action_steps
        if action_memmap_path is not None:
            action_memmap_path.unlink(missing_ok=True)


def _validate_data_table(table: pa.Table, *, data_path: Path) -> None:
    required = {
        "episode_index",
        "frame_index",
        "timestamp",
        "observation.state",
        "action",
        "action.padding_mask",
        "index",
    }
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"{data_path} is missing required columns: {sorted(missing)}")


def _global_index_capacity(data_paths: Sequence[Path]) -> int:
    maximum = -1
    for path in data_paths:
        table = pq.read_table(path, columns=["index"])
        values = np.asarray(table.column("index").combine_chunks().to_numpy(), dtype=np.int64)
        if values.size:
            maximum = max(maximum, int(values.max()))
    if maximum < 0:
        raise ValueError("dataset has no global index values")
    return maximum + 1


def _write_updated_data_table(
    path: Path,
    *,
    table: pa.Table,
    timestamp: np.ndarray,
    action: np.ndarray,
) -> None:
    timestamp_index = table.schema.get_field_index("timestamp")
    action_index = table.schema.get_field_index("action")
    updated = table.set_column(
        timestamp_index,
        table.schema.field(timestamp_index),
        pa.array(timestamp.tolist(), type=table.schema.field(timestamp_index).type),
    )
    updated = updated.set_column(
        action_index,
        table.schema.field(action_index),
        pa.array(action.tolist(), type=table.schema.field(action_index).type),
    )
    _atomic_write_parquet(path, updated)


def _update_statistics(path: Path, *, action_steps: np.ndarray) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload) != 1:
        raise ValueError(f"expected exactly one dataset statistics entry in {path}")
    dataset_key = next(iter(payload))
    payload[dataset_key]["action"] = build_action_statistics(action_steps)
    _atomic_write_text(path, json.dumps(payload, indent=2))


def _update_info(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    navvla = payload.setdefault("navvla", {})
    navvla["control_frequency_hz"] = CONTROL_FREQUENCY_HZ
    navvla["action_horizon_seconds"] = float(navvla.get("action_horizon", ACTION_HORIZON)) / CONTROL_FREQUENCY_HZ
    navvla["timestamp_policy"] = TIMESTAMP_POLICY
    _atomic_write_text(path, json.dumps(payload, indent=2))


def _update_platform_text(root: Path) -> None:
    pattern = re.compile(r"The control frequency is\s+[0-9.]+\s+Hz\.")
    replacement = "The control frequency is 1 Hz."
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        table = pq.read_table(tasks_path)
        if "platform_text" in table.column_names:
            values = [pattern.sub(replacement, str(value)) for value in table.column("platform_text").to_pylist()]
            if values != table.column("platform_text").to_pylist():
                column_index = table.schema.get_field_index("platform_text")
                table = table.set_column(
                    column_index,
                    table.schema.field(column_index),
                    pa.array(values, type=table.schema.field(column_index).type),
                )
                _atomic_write_parquet(tasks_path, table)

    jsonl_path = root / "meta" / "navvla_tasks.jsonl"
    if jsonl_path.exists():
        lines = []
        changed = False
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            old = str(row.get("platform_text") or "")
            new = pattern.sub(replacement, old)
            if new != old:
                row["platform_text"] = new
                changed = True
            lines.append(json.dumps(row, ensure_ascii=False))
        if changed:
            _atomic_write_text(jsonl_path, "\n".join(lines) + "\n")


def _atomic_write_parquet(path: Path, table: pa.Table) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temp_path, compression="snappy")
    os.replace(temp_path, path)


def _atomic_write_text(path: Path, value: str) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(value, encoding="utf-8")
    os.replace(temp_path, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporarily interpolate only AerialVLN/OpenFly vln_train actions and timestamps in place."
    )
    parser.add_argument("--dataset", choices=["aerialvln", "openfly", "both"], default="both")
    parser.add_argument("--aerialvln-root", type=Path, default=DEFAULT_ROOTS["aerialvln"])
    parser.add_argument("--openfly-root", type=Path, default=DEFAULT_ROOTS["openfly"])
    parser.add_argument("--max-files", type=int, default=None, help="Dry-run only: inspect only the first N parquet files.")
    parser.add_argument("--apply", action="store_true", help="Atomically replace the required columns/metadata in place.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected = ["aerialvln", "openfly"] if args.dataset == "both" else [args.dataset]
    roots = {"aerialvln": args.aerialvln_root, "openfly": args.openfly_root}
    reports = []
    for dataset_kind in selected:
        reports.append(
            interpolate_dataset_root(
                roots[dataset_kind],
                dataset_kind=dataset_kind,
                apply=bool(args.apply),
                max_files=args.max_files,
            )
        )
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
