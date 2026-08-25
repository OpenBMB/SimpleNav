import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from vln_aug.actions import build_observation_actions
from vln_aug.render_requests import validate_rendered_images


def validate_intermediate_episode(episode_dir: Path, require_images: bool = False) -> dict:
    root = Path(episode_dir)
    control = pq.read_table(root / "control_trajectory_1hz.parquet")
    observation = pq.read_table(root / "observation_plan_0p2hz.parquet")
    control_times = np.asarray(control.column("timestamp").to_pylist(), dtype=float)
    observation_times = np.asarray(observation.column("timestamp").to_pylist(), dtype=float)
    actions = np.asarray(observation.column("action").to_pylist(), dtype=float)
    states = np.asarray(observation.column("observation.state").to_pylist(), dtype=float)
    control_poses = np.asarray(control.column("pose").to_pylist(), dtype=float)
    done = observation.column("next.done").to_pylist()
    errors = []
    if len(control_times) < 1 or not np.allclose(np.diff(control_times), 1.0):
        errors.append("control timestamps are not 1 Hz")
    if len(observation_times) < 1 or not np.allclose(np.diff(observation_times), 5.0):
        errors.append("observation timestamps are not 0.2 Hz")
    if actions.shape != (len(observation_times), 8, 4):
        errors.append(f"unexpected action shape {actions.shape}")
    if control_poses.shape[1:] == (4,):
        try:
            expected_indices, expected_actions = build_observation_actions(
                control_poses, render_stride=5, horizon=8
            )
            if not np.array_equal(expected_indices.astype(float), observation_times):
                errors.append("observation timestamps do not match control render indices")
            if actions.shape == expected_actions.shape and not np.allclose(
                actions, expected_actions, atol=1e-5
            ):
                errors.append("stored actions do not reconstruct from the 1 Hz control trajectory")
        except Exception as error:
            errors.append(f"action reconstruction failed: {error}")
    if done != [False] * max(0, len(done) - 1) + [True]:
        errors.append("terminal done flags are invalid")
    if len(states) and not np.allclose(actions[-1], 0.0):
        errors.append("terminal row does not repeat the zero-relative terminal waypoint")
    manifest_lines = [line for line in (root / "render_requests_0p2hz.jsonl").read_text().splitlines() if line]
    cameras_per_frame = 0
    if observation.num_rows:
        cameras_per_frame = len(manifest_lines) // observation.num_rows
    if cameras_per_frame < 1 or cameras_per_frame * observation.num_rows != len(manifest_lines):
        errors.append("render request count is inconsistent with observation rows")
    image_report = None
    if require_images:
        image_report = validate_rendered_images(root / "render_requests_0p2hz.jsonl", root)
        if not image_report["complete"]:
            errors.append("rendered images are incomplete or invalid")
    return {
        "valid": not errors,
        "errors": errors,
        "control_rows": control.num_rows,
        "observation_rows": observation.num_rows,
        "render_request_count": len(manifest_lines),
        "cameras_per_frame": cameras_per_frame,
        "image_report": image_report,
    }


def validate_report_tree(reports_dir: Path, require_images: bool = False) -> dict:
    root = Path(reports_dir)
    episode_reports = []
    for episode_dir in sorted(root.glob("*/episode_*_intermediate")):
        result = validate_intermediate_episode(episode_dir, require_images=require_images)
        result["episode_dir"] = str(episode_dir)
        episode_reports.append(result)
    return {
        "valid": bool(episode_reports) and all(item["valid"] for item in episode_reports),
        "episode_count": len(episode_reports),
        "episodes": episode_reports,
    }
