#!/usr/bin/env python3
"""Build a dense HD OpenFly rollout replay without modifying evaluation artifacts.

The pipeline copies one successful evaluation episode, applies the OpenFly
three-confirmation stop rule, smooths and resamples the executed trajectory,
captures one AirSim image per dense pose, and renders a 1080p video with a
top-down route map.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from math import cos, pi, sin, tan
from pathlib import Path
import shutil
import subprocess
import textwrap
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


DEFAULT_EVAL_ROOT = Path(
    "local/eval_results/openfly/"
    "qwen35_openfly_tb1024_ph32_full_bats_stride5_actionobs_8wp_sync3_"
    "learned_token_nostop_n80_20260816"
)
DEFAULT_SOURCE_EPISODE = DEFAULT_EVAL_ROOT / "logs/env_airsim_26/seen/000784"
DEFAULT_CONFIG = Path("NavVLAeval/openfly/config_portable.yaml")
DEFAULT_ANNOTATION = Path("local/data/OpenFly/openfly_env/Annotation/seen.json")
DEFAULT_OUTPUT = Path("eval_visualizations/openfly_000784_hd_v1")


@dataclass(frozen=True)
class StopDecision:
    stop_file_stem: int
    stop_chunk_ordinal: int
    stop_action_value: float
    confirmation_values: tuple[float, ...]
    kept_chunk_count: int


@dataclass(frozen=True)
class Candidate:
    source_episode_id: str
    scene_id: str
    episode_dir: str
    spl: float
    final_distance: float
    instruction_length: int
    executed_path_length: float
    turn_variation: float
    stop_file_stem: int | None
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    rank = subparsers.add_parser("rank", help="rank successful, rule-terminating OpenFly candidates")
    rank.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    rank.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "candidate_ranking.json")
    rank.add_argument("--limit", type=int, default=20)

    prepare = subparsers.add_parser("prepare", help="copy, truncate, smooth, and densify one episode")
    _add_episode_args(prepare)
    prepare.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    prepare.add_argument("--density-factor", type=float, default=3.0)
    prepare.add_argument("--smoothing-window", type=int, default=11)

    capture = subparsers.add_parser("capture", help="capture HD RGB at every dense pose")
    _add_capture_args(capture)

    render = subparsers.add_parser("render", help="render the captured frames and trajectory map")
    _add_render_args(render)

    all_parser = subparsers.add_parser("all", help="run prepare, capture, and render")
    _add_episode_args(all_parser)
    all_parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    all_parser.add_argument("--density-factor", type=float, default=3.0)
    all_parser.add_argument("--smoothing-window", type=int, default=11)
    _add_capture_args(all_parser, include_output=False)
    _add_render_args(all_parser, include_output=False)
    return parser.parse_args()


def _add_episode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-episode-dir", type=Path, default=DEFAULT_SOURCE_EPISODE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stop-threshold", type=float, default=0.31)
    parser.add_argument("--stop-confirmations", type=int, default=3)


def _add_capture_args(parser: argparse.ArgumentParser, *, include_output: bool = True) -> None:
    if include_output:
        parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--airsim-port", type=int, default=41590)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=96)
    parser.add_argument("--pilot-count", type=int, default=0)


def _add_render_args(parser: argparse.ArgumentParser, *, include_output: bool = True) -> None:
    if include_output:
        parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--canvas-width", type=int, default=1920)
    parser.add_argument("--canvas-height", type=int, default=1080)
    parser.add_argument("--hold-final-sec", type=float, default=2.0)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def openfly_positions_xyz(values: Any) -> np.ndarray:
    positions: list[np.ndarray] = []
    for index, value in enumerate(values):
        position = np.asarray(value, dtype=np.float64).reshape(-1)
        if len(position) not in {3, 4}:
            raise ValueError(
                f"OpenFly position {index} must have 3 or 4 values, got {len(position)}"
            )
        if not np.isfinite(position[:3]).all():
            raise ValueError(f"OpenFly position {index} contains non-finite XYZ values")
        positions.append(position[:3])
    if not positions:
        raise ValueError("OpenFly position sequence is empty")
    return np.asarray(positions, dtype=np.float64)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(episode_dir: Path) -> dict[str, Any]:
    files = sorted(path for path in episode_dir.rglob("*") if path.is_file())
    return {
        "root": str(episode_dir.resolve()),
        "file_count": len(files),
        "files": [
            {
                "path": path.relative_to(episode_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in files
        ],
    }


def tail4_max_segment_xyz_norm(action_waypoints: Any) -> float:
    action = np.asarray(action_waypoints, dtype=np.float64)
    if action.ndim != 2 or action.shape[0] < 5 or action.shape[1] < 3:
        raise ValueError(f"expected action waypoints [horizon>=5, dim>=3], got {action.shape}")
    return float(np.linalg.norm(np.diff(action[-5:, :3], axis=0), axis=1).max())


def find_stop_decision(
    data_records: Iterable[tuple[Path, dict[str, Any]]],
    *,
    threshold: float,
    confirmations: int,
) -> StopDecision:
    if confirmations <= 0:
        raise ValueError("confirmations must be positive")
    streak: list[float] = []
    for ordinal, (path, payload) in enumerate(data_records):
        value = tail4_max_segment_xyz_norm(payload["action_waypoints"])
        if value < float(threshold):
            streak.append(value)
        else:
            streak.clear()
        if len(streak) >= confirmations:
            kept = ordinal + 1
            return StopDecision(
                stop_file_stem=int(path.stem),
                stop_chunk_ordinal=ordinal,
                stop_action_value=value,
                confirmation_values=tuple(streak[-confirmations:]),
                kept_chunk_count=kept,
            )
    raise RuntimeError(
        f"episode never met tail4_max_segment_xyz_norm < {threshold} "
        f"for {confirmations} consecutive replans"
    )


def load_data_records(episode_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted((episode_dir / "data").glob("*.json"), key=lambda path: int(path.stem))
    if not paths:
        raise FileNotFoundError(f"no step JSON files under {episode_dir / 'data'}")
    return [(path, _json(path)) for path in paths]


def openfly_initial_pose(annotation_path: Path, source_episode_id: str) -> np.ndarray:
    rows = _json(annotation_path)
    index = int(source_episode_id)
    if not isinstance(rows, list) or not (0 <= index < len(rows)):
        raise ValueError(f"OpenFly annotation has no episode index {source_episode_id}")
    row = rows[index]
    position = openfly_positions_xyz(row["pos"])[0]
    yaw = float(row["yaw"][0])
    # OpenFly's validated render transform is reflect-y-z-yaw.
    return np.asarray([position[0], -position[1], -position[2], wrap_to_pi(-yaw)], dtype=np.float64)


def executed_poses(records: Iterable[tuple[Path, dict[str, Any]]], initial_pose: np.ndarray) -> np.ndarray:
    poses = [np.asarray(initial_pose, dtype=np.float64).reshape(4)]
    for _path, payload in records:
        diagnostics = payload.get("diagnostics") or {}
        actual = np.asarray(diagnostics.get("actual_waypoint_poses") or [], dtype=np.float64).reshape(-1, 4)
        executed_count = min(int(payload.get("executed_action_count", len(actual))), len(actual))
        poses.extend(actual[:executed_count])
    result = np.asarray(poses, dtype=np.float64).reshape(-1, 4)
    if len(result) < 2 or not np.isfinite(result).all():
        raise ValueError("executed trajectory is empty or non-finite")
    return result


def smooth_and_resample_poses(
    poses: np.ndarray,
    *,
    density_factor: float,
    smoothing_window: int,
) -> np.ndarray:
    source = np.asarray(poses, dtype=np.float64).reshape(-1, 4)
    if len(source) < 2:
        raise ValueError("at least two poses are required")
    if density_factor < 1.0:
        raise ValueError("density_factor must be at least 1")

    local_xyz = source[:, :3] - source[0, :3]
    filtered_xyz = local_xyz.copy()
    max_window = len(source) if len(source) % 2 else len(source) - 1
    window = min(max_window, max(3, int(smoothing_window) | 1))
    if window >= 5:
        filtered_xyz = savgol_filter(local_xyz, window_length=window, polyorder=2, axis=0, mode="interp")
        filtered_xyz[0] = local_xyz[0]
        filtered_xyz[-1] = local_xyz[-1]

    segment = np.linalg.norm(np.diff(filtered_xyz, axis=0), axis=1)
    parameter = np.concatenate([[0.0], np.cumsum(segment)])
    keep = np.concatenate([[True], np.diff(parameter) > 1e-7])
    parameter = parameter[keep]
    filtered_xyz = filtered_xyz[keep]
    yaw = np.unwrap(source[:, 3])[keep]
    if len(parameter) < 2:
        raise ValueError("trajectory has no translational extent")

    dense_count = max(len(source), int(round(len(source) * float(density_factor))))
    dense_parameter = np.linspace(parameter[0], parameter[-1], dense_count, dtype=np.float64)
    dense_xyz = np.column_stack(
        [PchipInterpolator(parameter, filtered_xyz[:, axis])(dense_parameter) for axis in range(3)]
    )
    dense_yaw = PchipInterpolator(parameter, yaw)(dense_parameter)
    dense = np.column_stack([dense_xyz + source[0, :3], np.vectorize(wrap_to_pi)(dense_yaw)])
    dense[0] = source[0]
    dense[-1] = source[-1]
    return dense.astype(np.float32)


def prepare_episode(
    *,
    source_episode_dir: Path,
    output_dir: Path,
    annotation_path: Path,
    threshold: float,
    confirmations: int,
    density_factor: float,
    smoothing_window: int,
) -> dict[str, Any]:
    source_episode_dir = source_episode_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == source_episode_dir or source_episode_dir in output_dir.parents:
        raise ValueError("output_dir must not be inside the source evaluation episode")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    records = load_data_records(source_episode_dir)
    decision = find_stop_decision(records, threshold=threshold, confirmations=confirmations)
    kept_records = records[: decision.kept_chunk_count]
    info = _json(source_episode_dir / "eval_info.json")
    manifest_before = source_manifest(source_episode_dir)

    source_copy = output_dir / "source_episode_copy"
    source_copy.mkdir(parents=True)
    shutil.copy2(source_episode_dir / "eval_info.json", source_copy / "eval_info.json")
    (source_copy / "data").mkdir()
    for path, _payload in kept_records:
        shutil.copy2(path, source_copy / "data" / path.name)

    initial = openfly_initial_pose(annotation_path, str(info["source_episode_id"]))
    original_poses = executed_poses(kept_records, initial)
    dense_poses = smooth_and_resample_poses(
        original_poses,
        density_factor=density_factor,
        smoothing_window=smoothing_window,
    )
    np.save(output_dir / "original_executed_poses.npy", original_poses.astype(np.float32))
    np.save(output_dir / "dense_fitted_poses.npy", dense_poses)

    annotation_rows = _json(annotation_path)
    annotation_row = annotation_rows[int(info["source_episode_id"])]
    reference = openfly_positions_xyz(annotation_row["pos"])
    reference_yaw = np.asarray(annotation_row["yaw"], dtype=np.float64)
    reference_render = np.column_stack(
        [reference[:, 0], -reference[:, 1], -reference[:, 2], np.vectorize(wrap_to_pi)(-reference_yaw)]
    ).astype(np.float32)
    np.save(output_dir / "reference_poses.npy", reference_render)

    copied_info = dict(info)
    copied_info["visualization_termination"] = {
        "measure": "tail4_max_segment_xyz_norm",
        "threshold": float(threshold),
        "confirmations": int(confirmations),
        **asdict(decision),
    }
    copied_info["visualization_trajectory"] = {
        "original_pose_count": int(len(original_poses)),
        "dense_pose_count": int(len(dense_poses)),
        "density_ratio": float(len(dense_poses) / len(original_poses)),
        "smoothing": "Savitzky-Golay xyz followed by arc-length PCHIP resampling",
    }
    _write_json(source_copy / "eval_info.json", copied_info)
    _write_json(output_dir / "source_manifest_before.json", manifest_before)
    _write_json(
        output_dir / "prepare_summary.json",
        {
            "source_episode_dir": str(source_episode_dir),
            "output_dir": str(output_dir),
            "scene_id": info["scene_id"],
            "source_episode_id": info["source_episode_id"],
            "stop_decision": asdict(decision),
            "original_pose_count": len(original_poses),
            "dense_pose_count": len(dense_poses),
            "density_ratio": len(dense_poses) / len(original_poses),
            "source_file_count": manifest_before["file_count"],
        },
    )
    return _json(output_dir / "prepare_summary.json")


def rank_candidates(eval_root: Path, *, limit: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for info_path in eval_root.glob("logs/env_*/seen/*/eval_info.json"):
        info = _json(info_path)
        metrics = info.get("metrics") or {}
        if float(metrics.get("SR", 0.0)) != 1.0:
            continue
        records = load_data_records(info_path.parent)
        try:
            decision = find_stop_decision(records, threshold=0.31, confirmations=3)
        except RuntimeError:
            continue
        poses = executed_poses(records[: decision.kept_chunk_count], np.asarray(records[0][1]["diagnostics"]["actual_waypoint_poses"][0]))
        path_length = float(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum())
        turn_variation = float(np.abs(np.diff(np.unwrap(poses[:, 3]))).sum())
        instruction_length = len(str(info.get("instruction") or "").split())
        spl = float(metrics.get("SPL", 0.0))
        score = 2.0 * spl + min(instruction_length / 100.0, 2.0) + min(path_length / 100.0, 2.0) + min(turn_variation / pi, 2.0)
        candidates.append(
            Candidate(
                source_episode_id=str(info["source_episode_id"]),
                scene_id=str(info["scene_id"]),
                episode_dir=str(info_path.parent.resolve()),
                spl=spl,
                final_distance=float(metrics.get("NE", 0.0)),
                instruction_length=instruction_length,
                executed_path_length=path_length,
                turn_variation=turn_variation,
                stop_file_stem=decision.stop_file_stem,
                score=score,
            )
        )
    return sorted(candidates, key=lambda item: (-item.score, -item.spl, item.final_distance))[:limit]


def capture_frames(
    *,
    output_dir: Path,
    config_path: Path,
    gpu_id: int,
    airsim_port: int,
    image_size: int,
    jpeg_quality: int,
    pilot_count: int,
) -> dict[str, Any]:
    """Capture through the evaluation backend and record actual readback poses."""
    from NavVLAeval.common.config import load_eval_config
    from NavVLAeval.common.runner.backend_plan import WorkerBackendPlan
    from NavVLAeval.common.simulators.airsim.backend import AirSimEnvironmentBackend
    from NavVLAeval.common.types import EvalEpisode, Pose4D

    output_dir = output_dir.resolve()
    info = _json(output_dir / "source_episode_copy/eval_info.json")
    poses = np.load(output_dir / "dense_fitted_poses.npy")
    frame_indices = np.arange(len(poses), dtype=int)
    frame_root = output_dir / ("pilot_frames" if pilot_count > 0 else "frames")
    if pilot_count > 0:
        frame_indices = np.unique(np.linspace(0, len(poses) - 1, pilot_count, dtype=int))
    if frame_root.exists():
        raise FileExistsError(f"refusing to overwrite captured frames: {frame_root}")
    frame_root.mkdir(parents=True)

    cfg = load_eval_config(config_path.resolve())
    env_cfg = replace_env_config_image_size(cfg.env, image_size)
    backend = AirSimEnvironmentBackend(
        cfg=env_cfg,
        worker_backend=WorkerBackendPlan(
            type="airsim",
            kwargs={"airsim_port": int(airsim_port), "settings_root": str(output_dir / "airsim_runtime_eval")},
        ),
        physical_gpu_id=int(gpu_id),
    )
    episode = EvalEpisode(
        episode_uid=str(info["episode_uid"]), source_episode_id=str(info["source_episode_id"]),
        scene_id=str(info["scene_id"]), instruction=str(info.get("instruction") or ""),
        source=str(info.get("source") or "openfly_annotation_json"),
        input_namespace=str(info.get("input_namespace") or "seen"), input_root=str(info.get("input_root") or ""),
        payload={"env_name": str(info["scene_id"])},
    )
    records: list[dict[str, Any]] = []
    try:
        first = Pose4D(*map(float, poses[int(frame_indices[0])]))
        backend.start_episode(episode, first)
        for ordinal, pose_index in enumerate(frame_indices):
            target = poses[int(pose_index)]
            if ordinal:
                backend.set_pose(Pose4D(*map(float, target)))
            observation = backend.get_observation()
            image = np.asarray(observation["image"], dtype=np.uint8)
            actual = backend._actual_vehicle_pose().as_array()
            path = frame_root / f"{int(pose_index):06d}.jpg"
            Image.fromarray(image[:, :, :3], mode="RGB").save(
                path, format="JPEG", quality=max(1, min(100, int(jpeg_quality))), subsampling=0
            )
            records.append({
                "pose_index": int(pose_index), "image": path.relative_to(output_dir).as_posix(),
                "target_pose": target.tolist(), "actual_pose": actual.tolist(),
                "position_error_m": float(np.linalg.norm(actual[:3] - target[:3])),
            })
            if (ordinal + 1) % 100 == 0 or ordinal + 1 == len(frame_indices):
                print(f"captured {ordinal + 1}/{len(frame_indices)}", flush=True)
    finally:
        backend.close()
    summary = {
        "frame_count": len(records), "image_size": int(image_size), "pilot": bool(pilot_count > 0),
        "max_position_error_m": max((row["position_error_m"] for row in records), default=0.0),
        "frames": records,
    }
    _write_json(output_dir / ("pilot_capture_manifest.json" if pilot_count > 0 else "capture_manifest.json"), summary)
    if pilot_count <= 0:
        _write_json(output_dir / "frame_selection.json", select_longest_valid_frame_run(output_dir, records))
    return summary


def replace_env_config_image_size(env_cfg: Any, image_size: int) -> Any:
    from dataclasses import replace

    return replace(
        env_cfg,
        kwargs={
            **env_cfg.kwargs,
            "camera_resolution_overrides": {
                "front_custom": (int(image_size), int(image_size)),
                "0": (int(image_size), int(image_size)),
            },
            "capture_action_observations": False,
            "ignore_collision": True,
            "reset_ignore_collision": True,
            "teleport_render_sync_frames": 3,
        },
    )


def select_longest_valid_frame_run(output_dir: Path, records: list[dict[str, Any]]) -> dict[str, int]:
    valid: list[bool] = []
    for row in records:
        with Image.open(output_dir / row["image"]) as image:
            pixels = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32)
        luminance = pixels.mean(axis=2)
        valid.append(
            float(pixels.std()) > 18.0
            and float(np.mean(luminance < 8.0)) < 0.35
            and float(np.mean(luminance > 247.0)) < 0.75
            and float(row["position_error_m"]) < 0.75
        )
    best_start, best_end = longest_true_run(valid)
    return {
        "best_start": best_start,
        "best_end_exclusive": best_end,
        "kept_frames": best_end - best_start,
        "total_frames": len(records),
    }


def longest_true_run(values: Iterable[bool]) -> tuple[int, int]:
    best = (0, 0)
    start: int | None = None
    materialized = list(values)
    for index, value in enumerate(materialized + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    return best


def render_video(
    *,
    output_dir: Path,
    video_path: Path | None,
    fps: float,
    canvas_size: tuple[int, int],
    hold_final_sec: float,
) -> Path:
    output_dir = output_dir.resolve()
    video_path = (video_path or (output_dir / "openfly_000784_prediction_hd_map.mp4")).resolve()
    manifest = _json(output_dir / "capture_manifest.json")
    selection_path = output_dir / "frame_selection.json"
    if selection_path.is_file():
        selection = _json(selection_path)
        start = int(selection["best_start"])
        end = int(selection["best_end_exclusive"])
        manifest = {**manifest, "frames": manifest["frames"][start:end]}
    info = _json(output_dir / "source_episode_copy/eval_info.json")
    dense = np.load(output_dir / "dense_fitted_poses.npy")
    reference = np.load(output_dir / "reference_poses.npy")
    actual_path = np.asarray([row["actual_pose"] for row in manifest["frames"]], dtype=np.float32)
    fonts = _fonts()
    width, height = canvas_size
    video_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
        "-r", str(float(fps)), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(video_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None or process.stderr is None:
        raise RuntimeError("failed to open ffmpeg pipes")
    last_frame: np.ndarray | None = None
    try:
        for ordinal, row in enumerate(manifest["frames"]):
            pose_index = int(row["pose_index"])
            source = Image.open(output_dir / row["image"]).convert("RGB")
            canvas = compose_video_frame(
                source=source,
                pose=actual_path[ordinal],
                trail=actual_path[: ordinal + 1],
                full_path=actual_path,
                planned_path=dense,
                planned_index=pose_index,
                reference_path=reference,
                info=info,
                progress=(ordinal + 1) / len(manifest["frames"]),
                frame_label=f"Frame {ordinal + 1:,} / {len(manifest['frames']):,}",
                canvas_size=canvas_size,
                fonts=fonts,
            )
            last_frame = np.asarray(canvas, dtype=np.uint8)
            process.stdin.write(last_frame.tobytes())
        if last_frame is not None:
            for _ in range(max(1, int(round(fps * hold_final_sec)))):
                process.stdin.write(last_frame.tobytes())
    finally:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg exited with status {returncode}: {stderr.strip()}")
    return video_path


def compose_video_frame(
    *,
    source: Image.Image,
    pose: np.ndarray,
    trail: np.ndarray,
    full_path: np.ndarray,
    planned_path: np.ndarray,
    planned_index: int,
    reference_path: np.ndarray,
    info: dict[str, Any],
    progress: float,
    frame_label: str,
    canvas_size: tuple[int, int],
    fonts: dict[str, ImageFont.ImageFont],
) -> Image.Image:
    width, height = canvas_size
    panel_width = max(500, int(width * 0.34))
    view_width = width - panel_width
    view = _cover_resize(source, (view_width, height))
    _draw_projected_future(view, pose, planned_path, start_index=planned_index)
    canvas = Image.new("RGB", (width, height), (9, 15, 23))
    canvas.paste(view, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    panel_x = view_width
    draw.rectangle((panel_x, 0, width, height), fill=(9, 15, 23, 255))

    x = panel_x + 32
    draw.text((x, 28), "OpenFly Navigation", font=fonts["title"], fill=(241, 246, 251, 255))
    metrics = info.get("metrics") or {}
    draw.rounded_rectangle((x, 86, width - 32, 137), radius=12, fill=(20, 31, 43, 255))
    draw.text((x + 18, 99), f"SUCCESS  ·  SPL {float(metrics.get('SPL', 0.0)):.3f}", font=fonts["label"], fill=(47, 224, 174, 255))

    draw.text((x, 160), "LANGUAGE INSTRUCTION", font=fonts["small"], fill=(133, 151, 171, 255))
    instruction = str(info.get("instruction") or "")
    instruction_lines = textwrap.wrap(instruction, width=52)
    y = 190
    for line in instruction_lines[:7]:
        draw.text((x, y), line, font=fonts["body"], fill=(224, 232, 239, 255))
        y += 25
    if len(instruction_lines) > 7:
        draw.text((x, y), "…", font=fonts["body"], fill=(224, 232, 239, 255))
        y += 25

    map_top = max(390, y + 18)
    map_box = (x, map_top, width - 32, height - 112)
    _draw_trajectory_map(canvas, map_box, reference_path, full_path, trail, pose, fonts)

    bar_left, bar_right = x, width - 32
    bar_top = height - 72
    draw.rounded_rectangle((bar_left, bar_top, bar_right, bar_top + 10), radius=5, fill=(42, 55, 70, 255))
    draw.rounded_rectangle(
        (bar_left, bar_top, bar_left + int((bar_right - bar_left) * progress), bar_top + 10),
        radius=5,
        fill=(26, 224, 184, 255),
    )
    draw.text((bar_left, height - 51), frame_label, font=fonts["small"], fill=(161, 176, 191, 255))
    draw.rounded_rectangle((24, 24, 360, 72), radius=12, fill=(3, 8, 13, 190))
    draw.line((45, 48, 104, 48), fill=(21, 226, 190, 255), width=7)
    draw.text((118, 36), "Upcoming fitted trajectory", font=fonts["small"], fill=(244, 248, 252, 255))
    return canvas


def _cover_resize(image: Image.Image, target: tuple[int, int]) -> Image.Image:
    width, height = target
    scale = max(width / image.width, height / image.height)
    resized = image.resize((int(round(image.width * scale)), int(round(image.height * scale))), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _draw_projected_future(image: Image.Image, pose: np.ndarray, path: np.ndarray, *, start_index: int) -> None:
    future = np.asarray(path[start_index : start_index + 90], dtype=np.float64)
    if len(future) < 2:
        return
    yaw = float(pose[3])
    delta = future[:, :3] - np.asarray(pose[:3], dtype=np.float64)
    forward = cos(yaw) * delta[:, 0] + sin(yaw) * delta[:, 1]
    right = -sin(yaw) * delta[:, 0] + cos(yaw) * delta[:, 1]
    down = delta[:, 2]
    focal = image.width / (2.0 * tan(pi / 4.0))
    valid = forward > 0.2
    u = image.width * 0.5 + focal * right / np.maximum(forward, 1e-4)
    v = image.height * 0.58 + focal * down / np.maximum(forward, 1e-4)
    points = [(float(px), float(py)) for px, py, keep in zip(u, v, valid) if keep and -200 < px < image.width + 200 and -200 < py < image.height + 200]
    if len(points) < 2:
        return
    draw = ImageDraw.Draw(image, "RGBA")
    draw.line(points, fill=(0, 0, 0, 190), width=15, joint="curve")
    draw.line(points, fill=(21, 226, 190, 240), width=8, joint="curve")


def _draw_trajectory_map(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    reference: np.ndarray,
    full_path: np.ndarray,
    trail: np.ndarray,
    pose: np.ndarray,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    map_image = Image.new("RGB", (width, height), (237, 241, 240))
    draw = ImageDraw.Draw(map_image, "RGBA")
    for gx in range(0, width, 55):
        draw.line((gx, 0, gx, height), fill=(192, 201, 201, 80), width=1)
    for gy in range(0, height, 55):
        draw.line((0, gy, width, gy), fill=(192, 201, 201, 80), width=1)

    all_xy = np.vstack([reference[:, :2], full_path[:, :2]])
    minimum = all_xy.min(axis=0)
    maximum = all_xy.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    margin = 32
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])

    def points(array: np.ndarray) -> list[tuple[float, float]]:
        xy = np.asarray(array, dtype=np.float64)[:, :2]
        px = margin + (xy[:, 0] - minimum[0]) * scale
        py = height - margin - (xy[:, 1] - minimum[1]) * scale
        return list(zip(px.tolist(), py.tolist()))

    reference_points = points(reference)
    full_points = points(full_path)
    trail_points = points(trail)
    if len(reference_points) >= 2:
        _draw_dashed(draw, reference_points, fill=(98, 109, 119, 180), width=4, dash=10, gap=8)
    if len(full_points) >= 2:
        draw.line(full_points, fill=(56, 119, 219, 85), width=4, joint="curve")
    if len(trail_points) >= 2:
        draw.line(trail_points, fill=(0, 0, 0, 165), width=11, joint="curve")
        draw.line(trail_points, fill=(20, 220, 164, 255), width=6, joint="curve")
    goal = reference_points[-1]
    draw.ellipse((goal[0] - 9, goal[1] - 9, goal[0] + 9, goal[1] + 9), fill=(255, 191, 61, 255), outline=(91, 68, 15, 255), width=2)
    current = trail_points[-1]
    _draw_robot(draw, current, float(pose[3]))
    draw.rounded_rectangle((12, 12, 226, 48), radius=9, fill=(255, 255, 255, 225))
    draw.text((25, 20), "Top-down trajectory map", font=fonts["small"], fill=(39, 50, 60, 255))

    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1), radius=16, fill=255)
    canvas.paste(map_image, (left, top), mask)
    ImageDraw.Draw(canvas, "RGBA").rounded_rectangle(box, radius=16, outline=(70, 86, 101, 255), width=2)


def _draw_dashed(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], *, fill: tuple[int, ...], width: int, dash: float, gap: float) -> None:
    for start, end in zip(points[:-1], points[1:]):
        delta = np.asarray(end) - np.asarray(start)
        length = float(np.linalg.norm(delta))
        if length <= 1e-6:
            continue
        cursor = 0.0
        while cursor < length:
            finish = min(length, cursor + dash)
            p0 = np.asarray(start) + delta * (cursor / length)
            p1 = np.asarray(start) + delta * (finish / length)
            draw.line((tuple(p0), tuple(p1)), fill=fill, width=width)
            cursor += dash + gap


def _draw_robot(draw: ImageDraw.ImageDraw, center: tuple[float, float], yaw: float) -> None:
    cx, cy = center
    direction = np.asarray([cos(yaw), -sin(yaw)], dtype=np.float64)
    side = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    tip = np.asarray([cx, cy]) + direction * 14
    back = np.asarray([cx, cy]) - direction * 9
    polygon = [tuple(tip), tuple(back + side * 8), tuple(back - side * 8)]
    draw.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), fill=(255, 255, 255, 230), outline=(45, 58, 70, 220), width=2)
    draw.polygon(polygon, fill=(37, 111, 225, 255))


def _fonts() -> dict[str, ImageFont.ImageFont]:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font_path.is_file():
        default = ImageFont.load_default()
        return {key: default for key in ("title", "label", "body", "small")}
    return {
        "title": ImageFont.truetype(str(font_path), 38),
        "label": ImageFont.truetype(str(font_path), 21),
        "body": ImageFont.truetype(str(font_path), 19),
        "small": ImageFont.truetype(str(font_path), 17),
    }


def wrap_to_pi(value: float) -> float:
    return float((float(value) + pi) % (2.0 * pi) - pi)


def verify_source_unchanged(output_dir: Path) -> bool:
    before = _json(output_dir / "source_manifest_before.json")
    after = source_manifest(Path(before["root"]))
    return before == after


def main() -> None:
    args = parse_args()
    if args.command == "rank":
        candidates = rank_candidates(args.eval_root.resolve(), limit=args.limit)
        _write_json(args.output.resolve(), [asdict(candidate) for candidate in candidates])
        print(json.dumps([asdict(candidate) for candidate in candidates], indent=2))
        return
    if args.command in {"prepare", "all"}:
        summary = prepare_episode(
            source_episode_dir=args.source_episode_dir,
            output_dir=args.output_dir,
            annotation_path=args.annotation,
            threshold=args.stop_threshold,
            confirmations=args.stop_confirmations,
            density_factor=args.density_factor,
            smoothing_window=args.smoothing_window,
        )
        print(json.dumps(summary, indent=2))
    if args.command in {"capture", "all"}:
        summary = capture_frames(
            output_dir=args.output_dir,
            config_path=args.config,
            gpu_id=args.gpu_id,
            airsim_port=args.airsim_port,
            image_size=args.image_size,
            jpeg_quality=args.jpeg_quality,
            pilot_count=args.pilot_count,
        )
        print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=2))
    if args.command in {"render", "all"}:
        video = render_video(
            output_dir=args.output_dir,
            video_path=args.video,
            fps=args.fps,
            canvas_size=(args.canvas_width, args.canvas_height),
            hold_final_sec=args.hold_final_sec,
        )
        print(f"saved video: {video}")
    if args.command in {"prepare", "capture", "render", "all"}:
        print(f"source unchanged: {verify_source_unchanged(args.output_dir.resolve())}")


if __name__ == "__main__":
    main()
