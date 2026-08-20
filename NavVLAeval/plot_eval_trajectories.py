from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from NavVLAeval.common.log.metrics import summarize_result_metrics, summarize_results_by_scene
from NavVLAeval.openfly.inputs import _source_pose_to_canonical


@dataclass(frozen=True)
class EpisodeComparison:
    eval_info_path: Path
    eval_info: dict[str, Any]
    reference_poses: np.ndarray
    executed_poses: np.ndarray | None
    step_json_paths: tuple[Path, ...]

    @property
    def episode_id(self) -> str:
        return str(self.eval_info["source_episode_id"])


class SourceEpisodeIndex:
    def __init__(self) -> None:
        self._records_by_path: dict[Path, dict[str, dict[str, Any]]] = {}

    def record(self, source_path: Path, episode_id: str) -> dict[str, Any]:
        resolved = source_path.resolve()
        if resolved not in self._records_by_path:
            self._records_by_path[resolved] = _load_source_records(resolved)
        try:
            return self._records_by_path[resolved][str(episode_id)]
        except KeyError as exc:
            raise KeyError(f"source episode {episode_id!r} was not found in {resolved}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot NavVLA reference and executed trajectories from per-step evaluation JSON files."
    )
    parser.add_argument("--run-root", type=Path, required=True, help="Evaluation run directory containing config.yaml and logs/.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-root>/trajectory_comparison.",
    )
    parser.add_argument("--episode-ids", nargs="*", default=None, help="Specific source episode IDs to plot.")
    parser.add_argument("--sample-count", type=int, default=8, help="Number of episodes to select when IDs are omitted.")
    parser.add_argument(
        "--selection",
        choices=("balanced", "success", "unsuccessful", "random"),
        default="balanced",
        help="Deterministic metric-based sampling strategy used when episode IDs are omitted.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution in dots per inch.")
    parser.add_argument(
        "--missing-steps",
        choices=("error", "reference-only"),
        default="error",
        help="How to handle runs without per-step data/*.json artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.resolve()
    config = _load_run_config(run_root)
    eval_info_paths = _eval_info_paths(run_root)
    performance = summarize_evaluation_performance(
        eval_info_paths,
        run_plan_path=run_root / "run_plan.json",
        metric_keys=config.get("output", {}).get("metrics"),
    )
    selected_paths = select_eval_infos(
        eval_info_paths,
        episode_ids=args.episode_ids,
        sample_count=args.sample_count,
        selection=args.selection,
        seed=args.seed,
    )
    output_dir = (args.output_dir or (run_root / "trajectory_comparison")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    performance_path = output_dir / "performance_summary.json"
    performance_path.write_text(json.dumps(performance, indent=2, ensure_ascii=False), encoding="utf-8")

    source_index = SourceEpisodeIndex()
    comparisons: list[EpisodeComparison] = []
    outputs: list[dict[str, Any]] = []
    for eval_info_path in selected_paths:
        comparison = build_episode_comparison(
            eval_info_path,
            config=config,
            source_index=source_index,
            allow_reference_only=args.missing_steps == "reference-only",
        )
        stem = _safe_filename(f"{comparison.eval_info.get('scene_id', 'scene')}__{comparison.episode_id}")
        png_path = output_dir / f"{stem}.png"
        json_path = output_dir / f"{stem}.json"
        plot_episode_comparison(comparison, png_path, dpi=args.dpi)
        payload = comparison_payload(comparison, png_path=png_path)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        comparisons.append(comparison)
        outputs.append(
            {
                "episode_id": comparison.episode_id,
                "scene_id": comparison.eval_info.get("scene_id"),
                "success": _episode_success(comparison.eval_info),
                "trajectory_mode": "reference_and_executed" if comparison.executed_poses is not None else "reference_only",
                "png": str(png_path),
                "json": str(json_path),
            }
        )

    combined_plot_paths = {
        view: output_dir / f"combined_{view}.png"
        for view in ("xy", "xz", "yz", "3d")
    }
    for view, path in combined_plot_paths.items():
        plot_combined_comparisons(comparisons, path, view=view, dpi=args.dpi)
    combined_plot_path = output_dir / "combined_trajectories.png"
    shutil.copyfile(combined_plot_paths["xy"], combined_plot_path)
    manifest = {
        "schema_version": 1,
        "run_root": str(run_root),
        "benchmark": str(config.get("benchmark", {}).get("name") or ""),
        "selection": args.selection,
        "seed": args.seed,
        "requested_sample_count": args.sample_count,
        "missing_steps": args.missing_steps,
        "performance_summary": performance,
        "performance_summary_path": str(performance_path),
        "combined_plot": str(combined_plot_path),
        "combined_plots": {view: str(path) for view, path in combined_plot_paths.items()},
        "outputs": outputs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "performance_summary": performance,
                "performance_summary_path": str(performance_path),
                "combined_plot": str(combined_plot_path),
                "combined_plots": {view: str(path) for view, path in combined_plot_paths.items()},
                "episodes": outputs,
            },
            indent=2,
        )
    )
    return 0


def summarize_evaluation_performance(
    eval_info_paths: Iterable[Path],
    *,
    run_plan_path: Path | None = None,
    metric_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    records = [_load_json_object(Path(path)) for path in sorted(Path(path) for path in eval_info_paths)]
    if not records:
        raise FileNotFoundError("no eval_info.json files were found under the run logs directory")

    selected_metric_keys = tuple(metric_keys) if metric_keys is not None else None
    summary = summarize_result_metrics(records, metric_keys=selected_metric_keys)
    planned_episodes = _planned_episode_count(run_plan_path)
    written_episodes = len(records)
    progress_percent = (
        100.0 * written_episodes / planned_episodes
        if planned_episodes is not None and planned_episodes > 0
        else None
    )
    failure_breakdown = Counter(
        str(record.get("failure_type") or "unknown")
        for record in records
        if record.get("failure") is not None
    )
    status_breakdown = Counter(str(record.get("status") or "unknown") for record in records)
    termination_breakdown = Counter(str(record.get("termination_reason") or "unknown") for record in records)
    metrics_including_failures_as_zero = {
        key: float(
            sum(float((record.get("metrics") or {}).get(key, 0.0) or 0.0) for record in records)
            / written_episodes
        )
        for key in summary["metrics"]
    }
    return {
        "planned_episodes": planned_episodes,
        "written_episodes": written_episodes,
        "progress_percent": progress_percent,
        "completed_episodes": summary["completed_episodes"],
        "failed_episodes": summary["failed_episodes"],
        "metric_episodes": summary["metric_episodes"],
        "metrics": summary["metrics"],
        "scene_metrics": summarize_results_by_scene(records, metric_keys=selected_metric_keys),
        "status_breakdown": dict(sorted(status_breakdown.items())),
        "termination_breakdown": dict(sorted(termination_breakdown.items())),
        "failure_breakdown": dict(sorted(failure_breakdown.items())),
        "metrics_including_failures_as_zero": metrics_including_failures_as_zero,
    }


def select_eval_infos(
    eval_info_paths: Iterable[Path],
    *,
    episode_ids: list[str] | None,
    sample_count: int,
    selection: str,
    seed: int,
) -> list[Path]:
    paths = sorted(Path(path) for path in eval_info_paths)
    if not paths:
        raise FileNotFoundError("no eval_info.json files were found under the run logs directory")
    records = [(path, _load_json_object(path)) for path in paths]
    if episode_ids:
        requested = [str(value) for value in episode_ids]
        by_id: dict[str, Path] = {}
        for path, record in records:
            by_id[str(record.get("source_episode_id") or "")] = path
            by_id[str(record.get("episode_uid") or "")] = path
        missing = [episode_id for episode_id in requested if episode_id not in by_id]
        if missing:
            raise KeyError(f"requested episode IDs were not found: {missing}")
        return list(dict.fromkeys(by_id[episode_id] for episode_id in requested))
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    records = [item for item in records if _episode_is_trajectory_candidate(item[1])]
    if not records:
        raise ValueError("no completed episodes are available for automatic trajectory selection")

    rng = random.Random(int(seed))
    successes = [item for item in records if _episode_success(item[1])]
    unsuccessful = [item for item in records if not _episode_success(item[1])]
    rng.shuffle(successes)
    rng.shuffle(unsuccessful)
    all_records = list(records)
    rng.shuffle(all_records)

    if selection == "success":
        selected = successes[:sample_count]
    elif selection == "unsuccessful":
        selected = unsuccessful[:sample_count]
    elif selection == "random":
        selected = all_records[:sample_count]
    elif selection == "balanced":
        success_count = min(len(successes), (sample_count + 1) // 2)
        unsuccessful_count = min(len(unsuccessful), sample_count - success_count)
        selected = successes[:success_count] + unsuccessful[:unsuccessful_count]
        if len(selected) < sample_count:
            selected_paths = {path for path, _record in selected}
            selected.extend(item for item in all_records if item[0] not in selected_paths)
            selected = selected[:sample_count]
        rng.shuffle(selected)
    else:
        raise ValueError(f"unsupported selection mode: {selection}")
    if not selected:
        raise ValueError(f"selection mode {selection!r} did not match any episodes")
    return [path for path, _record in selected]


def build_episode_comparison(
    eval_info_path: Path,
    *,
    config: dict[str, Any],
    source_index: SourceEpisodeIndex,
    allow_reference_only: bool,
) -> EpisodeComparison:
    eval_info_path = Path(eval_info_path)
    eval_info = _load_json_object(eval_info_path)
    benchmark = str(eval_info.get("benchmark") or config.get("benchmark", {}).get("name") or "").lower()
    if benchmark not in {"openfly", "aerialvln"}:
        raise ValueError(f"unsupported benchmark {benchmark!r}; expected 'openfly' or 'aerialvln'")
    episode_id = str(eval_info.get("source_episode_id") or "").strip()
    if not episode_id:
        raise ValueError(f"eval_info is missing source_episode_id: {eval_info_path}")
    source_path = _source_path(eval_info, config=config, benchmark=benchmark, episode_id=episode_id)
    source_record = source_index.record(source_path, episode_id)
    if benchmark == "openfly":
        source_z_sign = float(config.get("input", {}).get("source_z_sign", 1.0))
        reference = _openfly_reference_poses(source_record, source_z_sign=source_z_sign)
    else:
        reference = _aerialvln_reference_poses(source_record)

    data_dir = eval_info_path.parent / str((eval_info.get("paths") or {}).get("data") or "data")
    step_paths = tuple(sorted(data_dir.glob("*.json"), key=lambda path: int(path.stem))) if data_dir.is_dir() else ()
    if not step_paths:
        if not allow_reference_only:
            failure = eval_info.get("failure")
            if failure is not None:
                missing_reason = (
                    "the evaluation failed before per-step trajectory artifacts were written "
                    f"(failure_type={eval_info.get('failure_type') or 'unknown'}, failure={failure})"
                )
            elif config.get("output", {}).get("save_step_artifacts") is False:
                missing_reason = "the run was evaluated with output.save_step_artifacts=false"
            else:
                missing_reason = "no per-step trajectory artifacts were written for this episode"
            raise FileNotFoundError(
                f"episode {episode_id} has no per-step JSON files under {data_dir}; "
                f"{missing_reason}. "
                "Use --missing-steps reference-only for a GT-only diagnostic plot."
            )
        executed = None
    else:
        executed = _executed_poses(step_paths, initial_pose=reference[0])
    return EpisodeComparison(
        eval_info_path=eval_info_path,
        eval_info=eval_info,
        reference_poses=reference,
        executed_poses=executed,
        step_json_paths=step_paths,
    )


def comparison_payload(comparison: EpisodeComparison, *, png_path: Path) -> dict[str, Any]:
    reference = comparison.reference_poses
    executed = comparison.executed_poses
    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": comparison.eval_info.get("benchmark"),
        "run_name": comparison.eval_info.get("run_name"),
        "episode_uid": comparison.eval_info.get("episode_uid"),
        "source_episode_id": comparison.episode_id,
        "scene_id": comparison.eval_info.get("scene_id"),
        "instruction": comparison.eval_info.get("instruction"),
        "status": comparison.eval_info.get("status"),
        "termination_reason": comparison.eval_info.get("termination_reason"),
        "metrics": comparison.eval_info.get("metrics") or {},
        "trajectory_mode": "reference_and_executed" if executed is not None else "reference_only",
        "reference_path_length_m": _path_length(reference),
        "executed_path_length_m": _path_length(executed) if executed is not None else None,
        "reference_poses": reference.tolist(),
        "executed_poses": executed.tolist() if executed is not None else None,
        "step_json_files": [str(path) for path in comparison.step_json_paths],
        "plot_path": str(png_path),
    }
    if executed is not None:
        payload["final_position_error_m"] = float(np.linalg.norm(executed[-1, :3] - reference[-1, :3]))
    else:
        payload["diagnostic"] = "Executed trajectory unavailable because this episode has no per-step JSON artifacts."
    return payload


def plot_episode_comparison(comparison: EpisodeComparison, output_path: Path, *, dpi: int = 160) -> None:
    reference = comparison.reference_poses
    executed = comparison.executed_poses
    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0])]
    axis_specs = ((0, 1, "X (m)", "Y (m)", "Top view (XY)"), (0, 2, "X (m)", "Z (m)", "Side view (XZ)"), (1, 2, "Y (m)", "Z (m)", "Side view (YZ)"))
    for axis, (x_index, y_index, x_label, y_label, title) in zip(axes, axis_specs):
        _plot_2d(axis, reference, executed, x_index=x_index, y_index=y_index)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.set_aspect("equal", adjustable="datalim")

    axis_3d = fig.add_subplot(grid[1, 1], projection="3d")
    _plot_3d(axis_3d, reference, executed)
    axis_3d.set_xlabel("X (m)")
    axis_3d.set_ylabel("Y (m)")
    axis_3d.set_zlabel("Z (m)")
    axis_3d.set_title("3D trajectory")
    axis_3d.legend(loc="best", fontsize=8)

    metrics = comparison.eval_info.get("metrics") or {}
    metric_text = "  ".join(f"{key}={float(metrics[key]):.3f}" for key in ("SR", "OSR", "NE", "SPL") if metrics.get(key) is not None)
    mode_text = "reference + executed" if executed is not None else "REFERENCE ONLY: no per-step JSON artifacts"
    fig.suptitle(
        f"{comparison.eval_info.get('benchmark', '')} | scene={comparison.eval_info.get('scene_id', '')} | "
        f"episode={comparison.episode_id}\n{metric_text}  termination={comparison.eval_info.get('termination_reason')}  |  {mode_text}",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_combined_comparisons(
    comparisons: list[EpisodeComparison],
    output_path: Path,
    *,
    view: str = "xy",
    dpi: int = 160,
) -> None:
    if not comparisons:
        raise ValueError("cannot plot a combined trajectory figure without episode comparisons")
    view_specs = {
        "xy": (0, 1, "X (m)", "Y (m)"),
        "xz": (0, 2, "X (m)", "Z (m)"),
        "yz": (1, 2, "Y (m)", "Z (m)"),
    }
    if view not in {*view_specs, "3d"}:
        raise ValueError(f"unsupported combined trajectory view: {view!r}")
    columns = min(3, math.ceil(math.sqrt(len(comparisons))))
    rows = math.ceil(len(comparisons) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, 4.6 * rows),
        constrained_layout=True,
        squeeze=False,
        subplot_kw={"projection": "3d"} if view == "3d" else None,
    )
    flat_axes = list(axes.flat)
    for axis, comparison in zip(flat_axes, comparisons):
        if view == "3d":
            _plot_3d(axis, comparison.reference_poses, comparison.executed_poses, show_legend=False)
            axis.set_xlabel("X (m)")
            axis.set_ylabel("Y (m)")
            axis.set_zlabel("Z (m)")
        else:
            x_index, y_index, x_label, y_label = view_specs[view]
            _plot_2d(
                axis,
                comparison.reference_poses,
                comparison.executed_poses,
                x_index=x_index,
                y_index=y_index,
                show_legend=False,
            )
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.grid(True, alpha=0.25)
            axis.set_aspect("equal", adjustable="datalim")
        metrics = comparison.eval_info.get("metrics") or {}
        metric_parts = [
            f"{key}={float(metrics[key]):.3f}"
            for key in ("SR", "NE")
            if metrics.get(key) is not None
        ]
        mode = "reference + executed" if comparison.executed_poses is not None else "reference only"
        axis.set_title(
            f"scene={comparison.eval_info.get('scene_id', '')} | episode={comparison.episode_id}\n"
            f"{'  '.join(metric_parts)}  {mode}",
            fontsize=10,
        )
    for axis in flat_axes[len(comparisons) :]:
        axis.remove()

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
    first = comparisons[0]
    fig.suptitle(
        f"{first.eval_info.get('benchmark', '')} trajectory comparison ({view.upper()}) | "
        f"run={first.eval_info.get('run_name', '')} | episodes={len(comparisons)}",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_2d(
    axis: Any,
    reference: np.ndarray,
    executed: np.ndarray | None,
    *,
    x_index: int,
    y_index: int,
    show_legend: bool = True,
) -> None:
    axis.plot(reference[:, x_index], reference[:, y_index], color="#1f77b4", linewidth=2.0, label="Reference")
    if executed is not None:
        axis.plot(executed[:, x_index], executed[:, y_index], color="#d62728", linewidth=1.5, label="Executed")
    axis.scatter(reference[0, x_index], reference[0, y_index], color="#2ca02c", marker="o", s=45, zorder=4, label="Start")
    axis.scatter(reference[-1, x_index], reference[-1, y_index], color="#ffbf00", edgecolor="#7f6000", marker="*", s=100, zorder=4, label="Goal")
    if show_legend:
        axis.legend(loc="best", fontsize=8)


def _plot_3d(
    axis: Any,
    reference: np.ndarray,
    executed: np.ndarray | None,
    *,
    show_legend: bool = True,
) -> None:
    axis.plot(
        reference[:, 0],
        reference[:, 1],
        reference[:, 2],
        color="#1f77b4",
        linewidth=2.0,
        label="Reference",
    )
    if executed is not None:
        axis.plot(
            executed[:, 0],
            executed[:, 1],
            executed[:, 2],
            color="#d62728",
            linewidth=1.5,
            label="Executed",
        )
    axis.scatter(*reference[0, :3], color="#2ca02c", marker="o", s=55, label="Start")
    axis.scatter(
        *reference[-1, :3],
        color="#ffbf00",
        edgecolor="#7f6000",
        marker="*",
        s=110,
        label="Goal",
    )
    if show_legend:
        axis.legend(loc="best", fontsize=8)


def _executed_poses(step_paths: Iterable[Path], *, initial_pose: np.ndarray) -> np.ndarray:
    poses = [np.asarray(initial_pose, dtype=np.float64).reshape(-1)[:4].tolist()]
    for path in step_paths:
        payload = _load_json_object(path)
        diagnostics = payload.get("diagnostics") or {}
        actual_waypoints = diagnostics.get("actual_waypoint_poses")
        if isinstance(actual_waypoints, list) and actual_waypoints:
            for pose in actual_waypoints:
                poses.append(_pose4(pose, label=f"{path}: diagnostics.actual_waypoint_poses"))
            continue
        state = payload.get("state")
        if state is not None:
            poses.append(_pose4(state, label=f"{path}: state"))
    if len(poses) == 1:
        raise ValueError("per-step JSON files did not contain actual_waypoint_poses or state")
    return np.asarray(poses, dtype=np.float64)


def _openfly_reference_poses(record: dict[str, Any], *, source_z_sign: float) -> np.ndarray:
    frames = record.get("frames")
    if isinstance(frames, list) and frames:
        raw_poses = [frame.get("state") for frame in frames]
    else:
        positions = record.get("pos") or record.get("positions")
        yaws = record.get("yaw")
        if not isinstance(positions, list) or not positions or not isinstance(yaws, list) or not yaws:
            raise ValueError("OpenFly source episode is missing frames or pos/yaw")
        raw_poses = [[*position[:3], yaw] for position, yaw in zip(positions, yaws)]
    converted = []
    for raw_pose in raw_poses:
        # The legacy converter has no source_z_sign keyword; its one-argument result is z-sign neutral.
        position, yaw = _source_pose_to_canonical(raw_pose)
        position = [*position]
        position[2] = float(position[2]) * source_z_sign
        converted.append([*position, yaw])
    return np.asarray(converted, dtype=np.float64)


def _aerialvln_reference_poses(record: dict[str, Any]) -> np.ndarray:
    raw_path = record.get("reference_path") or record.get("trajectory")
    if not isinstance(raw_path, list) or not raw_path:
        raise ValueError("AerialVLN source episode is missing reference_path")
    poses = []
    for point in raw_path:
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            raise ValueError("AerialVLN reference path point must contain xyz")
        yaw = float(point[5]) if len(point) >= 6 else (float(point[3]) if len(point) >= 4 else 0.0)
        poses.append([float(point[0]), float(point[1]), float(point[2]), yaw])
    return np.asarray(poses, dtype=np.float64)


def _source_path(eval_info: dict[str, Any], *, config: dict[str, Any], benchmark: str, episode_id: str) -> Path:
    input_root = Path(str(eval_info.get("input_root") or ""))
    if input_root.is_file():
        return input_root
    if benchmark == "openfly":
        episode_path = input_root / "episodes" / f"{episode_id}.json"
        if episode_path.is_file():
            return episode_path
        split = str(config.get("input", {}).get("split") or eval_info.get("input_namespace") or "")
        annotation_path = input_root / "Annotation" / f"{split}.json"
        if annotation_path.is_file():
            return annotation_path
    raise FileNotFoundError(f"could not resolve source trajectory input for episode {episode_id}: {input_root}")


def _load_source_records(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("episodes") if isinstance(payload, dict) and isinstance(payload.get("episodes"), list) else payload
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        raise ValueError(f"source trajectory file must contain a list or episodes list: {path}")
    index: dict[str, dict[str, Any]] = {}
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        source_id = str(record.get("episode_id") or record.get("id") or record.get("trajectory_id") or f"{record_index:06d}")
        index[source_id] = record
    if len(records) == 1 and isinstance(records[0], dict):
        index.setdefault(path.stem, records[0])
    return index


def _eval_info_paths(run_root: Path) -> list[Path]:
    logs_root = run_root / "logs"
    return [path for path in sorted(logs_root.glob("**/eval_info.json")) if "attempts" not in path.relative_to(logs_root).parts]


def _planned_episode_count(run_plan_path: Path | None) -> int | None:
    if run_plan_path is None or not Path(run_plan_path).is_file():
        return None
    payload = _load_json_object(Path(run_plan_path))
    episode_uids = payload.get("total_episode_uids") or payload.get("episode_uids")
    if isinstance(episode_uids, list):
        return len(episode_uids)
    if isinstance(episode_uids, int):
        return episode_uids
    return None


def _load_run_config(run_root: Path) -> dict[str, Any]:
    config_path = run_root / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"evaluation config does not exist: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation config must contain a mapping: {config_path}")
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _pose4(value: Any, *, label: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 3:
        raise ValueError(f"{label} must contain at least xyz")
    yaw = float(array[3]) if array.size >= 4 else 0.0
    return [float(array[0]), float(array[1]), float(array[2]), yaw]


def _episode_success(eval_info: dict[str, Any]) -> bool:
    return float((eval_info.get("metrics") or {}).get("SR", 0.0) or 0.0) >= 0.5


def _episode_is_trajectory_candidate(eval_info: dict[str, Any]) -> bool:
    status = str(eval_info.get("status") or "").strip().lower()
    if status and status != "completed":
        return False
    return eval_info.get("failure") is None


def _path_length(poses: np.ndarray | None) -> float | None:
    if poses is None or len(poses) < 2:
        return 0.0 if poses is not None else None
    return float(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum())


def _safe_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return safe or "episode"


if __name__ == "__main__":
    raise SystemExit(main())
