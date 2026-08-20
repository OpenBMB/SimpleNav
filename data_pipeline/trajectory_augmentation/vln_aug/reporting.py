import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from vln_aug.lerobot_io import discover_train_splits, extract_episode_rows, read_episode_metadata
from vln_aug.intermediate import CameraSpec, write_intermediate_episode
from vln_aug.selection import select_representative_episodes
from vln_aug.safety import assert_reports_outside_sources, build_file_stat_manifest
from vln_aug.trajectory import TrajectoryConfig, smooth_and_retime, stable_trajectory_seed
from vln_aug.visualize import (
    compute_trajectory_metrics,
    plot_sampling_audit,
    plot_trajectory_comparison,
)


DIRECT_ABSOLUTE_STATE_MODES = {
    "source_world_absolute_pose_xyz_yaw",
    "indooruav_world_pose_xy_zdown_yaw_minus_pi_over_2",
    "nuscenes_global_ego_pose_xyz_yaw",
}


def absolute_pose_support(info: dict) -> tuple[bool, str]:
    navvla = info.get("navvla", {})
    state_mode = navvla.get("state_mode", "unknown")
    if state_mode in DIRECT_ABSOLUTE_STATE_MODES:
        return True, "explicit absolute state mode"
    if (
        navvla.get("stored_observation_state") == "absolute_pose_ned_xyz_yaw"
        and int(navvla.get("state_dim", 0)) == 4
    ):
        return True, "explicit stored absolute pose declaration"
    if state_mode == "unavailable_zero_placeholder":
        return False, "absolute pose unavailable in observation.state"
    if state_mode == "variable_bats_history_relative_body_frame_actions":
        return False, "state schema is not a canonical 4D absolute pose"
    return False, f"absolute pose semantics are not explicitly proven for state_mode={state_mode}"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _dataset_key(dataset_root: Path, train_split: Path) -> str:
    relative = train_split.parent.relative_to(dataset_root)
    return "__".join(relative.parts)


def _stable_seed(dataset_key: str, episode_index: int) -> int:
    return stable_trajectory_seed(dataset_key, episode_index)


def _read_info(train_split: Path) -> dict:
    return json.loads((train_split / "meta" / "info.json").read_text(encoding="utf-8"))


def _trajectory_from_table(table) -> np.ndarray:
    if "observation.state" not in table.column_names:
        raise ValueError("observation.state is missing")
    values = table.column("observation.state").to_pylist()
    state = np.asarray(values, dtype=float)
    if state.ndim != 2 or state.shape[1] != 4:
        raise ValueError(f"expected 4D absolute pose, got {state.shape}")
    return state


def build_isolated_preview_command(
    python_executable: str,
    dataset_root: Path,
    train_split: Path,
    reports_dir: Path,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "vln_aug.cli",
        "preview-one",
        "--dataset-root",
        str(dataset_root),
        "--train-split",
        str(train_split),
        "--reports-dir",
        str(reports_dir),
    ]


def run_isolated_preview_report(dataset_root: Path, reports_dir: Path) -> dict:
    root = Path(dataset_root).resolve()
    reports = Path(reports_dir).resolve()
    train_splits = discover_train_splits(root)
    assert_reports_outside_sources(reports, train_splits)
    reports.mkdir(parents=True, exist_ok=True)
    summary = {"dataset_root": str(root), "splits": []}
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/vln-trajectory-augmentation-matplotlib")
    for train_split in train_splits:
        command = build_isolated_preview_command(sys.executable, root, train_split, reports)
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            summary["splits"].append(
                {
                    "dataset_key": _dataset_key(root, train_split),
                    "train_split": str(train_split),
                    "preview_count": 0,
                    "publish_status": "preview_subprocess_failed",
                    "error": completed.stderr[-4000:],
                }
            )
            continue
        summary["splits"].append(json.loads(completed.stdout))
    _write_json(reports / "summary.json", summary)
    return summary


def summarize_existing_reports(dataset_root: Path, reports_dir: Path) -> dict:
    root = Path(dataset_root).resolve()
    reports = Path(reports_dir).resolve()
    summary = {"dataset_root": str(root), "splits": []}
    for train_split in discover_train_splits(root):
        key = _dataset_key(root, train_split)
        split_report = reports / key
        metrics_path = split_report / "metrics.json"
        selection_path = split_report / "selection.json"
        if not metrics_path.is_file() or not selection_path.is_file():
            summary["splits"].append(
                {
                    "dataset_key": key,
                    "train_split": str(train_split),
                    "selected_count": 0,
                    "preview_count": 0,
                    "publish_status": "missing_preview_report",
                }
            )
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        summary["splits"].append(
            {
                "dataset_key": key,
                "train_split": str(train_split),
                "report_dir": str(split_report),
                "selected_count": len(selection.get("selected", [])),
                "preview_count": sum(
                    item.get("status") == "preview_generated" for item in metrics.get("episodes", [])
                ),
                "publish_status": "blocked_no_real_renderer",
            }
        )
    _write_json(reports / "summary.json", summary)
    return summary


def run_one_split_preview(dataset_root: Path, train_split: Path, reports_dir: Path) -> dict:
    root = Path(dataset_root).resolve()
    reports = Path(reports_dir).resolve()
    train_split = Path(train_split).resolve()
    train_splits = discover_train_splits(root)
    if train_split not in train_splits:
        raise ValueError(f"not an active train split under dataset root: {train_split}")
    assert_reports_outside_sources(reports, train_splits)
    reports.mkdir(parents=True, exist_ok=True)
    source_manifest_before = build_file_stat_manifest(train_split)
    key = _dataset_key(root, train_split)
    summary = None
    if True:
        split_report = reports / key
        info = _read_info(train_split)
        state_mode = info.get("navvla", {}).get("state_mode", "unknown")
        action_mode = info.get("navvla", {}).get("action_mode", "unknown")
        camera_keys = [
            feature.removeprefix("observation.images.")
            for feature in info.get("features", {})
            if feature.startswith("observation.images.")
        ]
        camera_specs = []
        camera_config_path = train_split / "meta" / "navvla_cameras.json"
        camera_config = (
            json.loads(camera_config_path.read_text(encoding="utf-8"))
            if camera_config_path.is_file()
            else {}
        )
        for feature_name, feature in info.get("features", {}).items():
            if not feature_name.startswith("observation.images."):
                continue
            shape = feature.get("shape", [])
            if len(shape) < 2:
                continue
            camera_specs.append(
                CameraSpec(
                    key=feature_name.removeprefix("observation.images."),
                    height=int(shape[0]),
                    width=int(shape[1]),
                    metadata=next(
                        (
                            camera_value
                            for camera_value in camera_config.values()
                            if camera_value.get("video_key")
                            == feature_name.removeprefix("observation.images.")
                        ),
                        {},
                    ),
                )
            )
        capability = {
            "train_split": str(train_split),
            "state_mode": state_mode,
            "action_mode": action_mode,
            "camera_keys": camera_keys,
            "renderer_status": "unavailable",
            "renderer_reason": "no configured real simulator/scene backend found locally",
            "publishable": False,
        }
        state_supported, state_reason = absolute_pose_support(info)
        if not state_supported:
            capability["trajectory_status"] = "unsupported"
            capability["trajectory_reason"] = state_reason
        else:
            capability["trajectory_status"] = "candidate"
            capability["trajectory_reason"] = state_reason

        metadata = read_episode_metadata(train_split)
        selection = select_representative_episodes(metadata)
        selection_payload = {
            "reason": selection.reason,
            "selected": [asdict(item) for item in selection.selected],
        }
        _write_json(split_report / "capability.json", capability)
        _write_json(split_report / "selection.json", selection_payload)

        episode_results = []
        for episode in selection.selected:
            item = {"episode": asdict(episode), "status": "not_processed"}
            try:
                if capability["trajectory_status"] != "candidate":
                    raise ValueError(capability["trajectory_reason"])
                table = extract_episode_rows(train_split, episode)
                poses = _trajectory_from_table(table)
                trajectory = smooth_and_retime(
                    poses,
                    TrajectoryConfig(),
                    seed=_stable_seed(key, episode.episode_index),
                )
                metrics = compute_trajectory_metrics(trajectory)
                plot_path = split_report / f"episode_{episode.episode_index:06d}_comparison.png"
                plot_trajectory_comparison(
                    trajectory,
                    plot_path,
                    title=f"{key} episode {episode.episode_index} scene={episode.scene_id}",
                )
                sampling_audit_path = (
                    split_report / f"episode_{episode.episode_index:06d}_sampling_audit.png"
                )
                sampling_audit = plot_sampling_audit(
                    trajectory,
                    sampling_audit_path,
                    title=f"{key} episode {episode.episode_index} scene={episode.scene_id}",
                )
                sampling_audit_json = (
                    split_report / f"episode_{episode.episode_index:06d}_sampling_audit.json"
                )
                _write_json(sampling_audit_json, sampling_audit)
                controls_path = split_report / f"episode_{episode.episode_index:06d}_control_1hz.npy"
                np.save(controls_path, trajectory.control_poses)
                intermediate_dir = split_report / f"episode_{episode.episode_index:06d}_intermediate"
                intermediate = write_intermediate_episode(
                    output_dir=intermediate_dir,
                    dataset_key=key,
                    source_episode_index=episode.episode_index,
                    source_episode_id=episode.episode_id,
                    scene_id=episode.scene_id,
                    control_poses=trajectory.control_poses,
                    cameras=camera_specs,
                    horizon=8,
                    terminal_action_available=True,
                    coordinate_metadata={
                        "state_mode": state_mode,
                        "action_mode": action_mode,
                        "coordinate_convention": info.get("navvla", {}).get("coordinate_convention"),
                        "coordinate_frame": info.get("navvla", {}).get("coordinate_frame"),
                        "coordinate_frame_id": info.get("navvla", {}).get("coordinate_frame_id"),
                    },
                )
                item.update(
                    {
                        "status": "preview_generated",
                        "metrics": metrics,
                        "plot": str(plot_path),
                        "sampling_audit_plot": str(sampling_audit_path),
                        "sampling_audit_json": str(sampling_audit_json),
                        "control_poses": str(controls_path),
                        "control_parquet": str(intermediate.control_path),
                        "observation_plan": str(intermediate.observation_path),
                        "render_requests": str(intermediate.render_request_path),
                    }
                )
            except Exception as error:
                item.update(
                    {
                        "status": "excluded",
                        "reason": str(error),
                        "error_type": type(error).__name__,
                    }
                )
            episode_results.append(item)

        _write_json(split_report / "metrics.json", {"episodes": episode_results})
        summary = {
            "dataset_key": key,
            "train_split": str(train_split),
            "report_dir": str(split_report),
            "selected_count": len(selection.selected),
            "preview_count": sum(item["status"] == "preview_generated" for item in episode_results),
            "publish_status": "blocked_no_real_renderer",
        }
    source_manifest_after = build_file_stat_manifest(train_split)
    if source_manifest_after != source_manifest_before:
        raise RuntimeError(f"source split changed during preview: {train_split}")
    return summary
