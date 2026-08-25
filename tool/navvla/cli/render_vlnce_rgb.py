from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


R2R_FAMILY = "r2r"
RXR_FAMILY = "rxr"


def resolve_dataset_files(
    vlnce_root: Union[str, Path], *, family: str, split: str, role: Optional[str] = None
) -> Tuple[Path, Path]:
    root = Path(vlnce_root)
    family = normalize_family(family)
    if family == R2R_FAMILY:
        base = root / "data" / "datasets" / "R2R_VLNCE_v1-3_preprocessed"
        dataset_name = f"{split}.json.gz"
        if split == "joint_train_envdrop":
            dataset_name = "joint_train_envdrop.gz"
        return base / split / dataset_name, base / split / f"{split}_gt.json.gz"

    if role is None:
        raise ValueError("role is required for RxR rendered splits")
    base = root / "data" / "datasets" / "RxR_VLNCE_v0"
    return base / split / f"{split}_{role}.json.gz", base / split / f"{split}_{role}_gt.json.gz"


def target_split_name(family: str, split: str, role: Optional[str]) -> str:
    family = normalize_family(family)
    if family == R2R_FAMILY:
        return f"r2r_{split}"
    if role is None:
        raise ValueError("role is required for RxR rendered splits")
    return f"rxr_{split}_{role}"


def normalize_family(family: str) -> str:
    value = family.strip().lower()
    if value not in {R2R_FAMILY, RXR_FAMILY}:
        raise ValueError(f"unsupported VLN-CE family: {family}")
    return value


def default_task_config(vlnce_root: Union[str, Path], *, family: str) -> Path:
    root = Path(vlnce_root)
    if normalize_family(family) == R2R_FAMILY:
        return root / "habitat_extensions" / "config" / "vlnce_task.yaml"
    return root / "habitat_extensions" / "config" / "rxr_vlnce_english_task.yaml"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = render_vlnce_rgb(args)
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render VLN-CE GT trajectory RGB frames with the official Habitat/VLN-CE simulator."
    )
    parser.add_argument("--vlnce-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--family", choices=[R2R_FAMILY, RXR_FAMILY], required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--role", choices=["guide", "follower"], default=None)
    parser.add_argument(
        "--languages",
        nargs="*",
        default=["*"],
        help="RxR language filters. Use '*' for all languages.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--start-episode-index", type=int, default=0)
    parser.add_argument("--end-episode-index", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=None, help="Optional RGB width override. Defaults to official config.")
    parser.add_argument("--height", type=int, default=None, help="Optional RGB height override. Defaults to official config.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing PNGs while rebuilding the manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Delete the staged split root before rendering.")
    return parser


def render_vlnce_rgb(args: argparse.Namespace) -> Dict[str, Any]:
    family = normalize_family(args.family)
    if family == RXR_FAMILY and args.role is None:
        raise ValueError("--role is required for RxR")
    vlnce_root = Path(args.vlnce_root).resolve()
    output_root = Path(args.output_root).resolve()
    dataset_path, gt_path = resolve_dataset_files(vlnce_root, family=family, split=args.split, role=args.role)
    if not dataset_path.exists():
        raise FileNotFoundError(f"VLN-CE dataset split not found: {dataset_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"VLN-CE GT split not found: {gt_path}")

    split_name = target_split_name(family, args.split, args.role)
    stage_root = output_root / split_name
    if args.overwrite and stage_root.exists():
        shutil.rmtree(stage_root)
    if stage_root.exists() and not args.resume and (stage_root / "manifest.jsonl").exists():
        raise FileExistsError(f"{stage_root} already has manifest.jsonl; pass --overwrite or --resume")
    stage_root.mkdir(parents=True, exist_ok=True)

    gt = load_json_gz(gt_path)
    task_config = build_task_config(
        vlnce_root,
        family=family,
        split=args.split,
        role=args.role,
        languages=args.languages,
        dataset_path=dataset_path,
        gpu_id=args.gpu_id,
        width=args.width,
        height=args.height,
    )

    manifest_tmp = stage_root / "manifest.jsonl.tmp"
    manifest_path = stage_root / "manifest.jsonl"
    rendered_frames = 0
    reused_frames = 0
    rendered_episodes = 0
    skipped_no_gt = 0

    env = None
    try:
        from habitat import Env

        env = Env(config=task_config)
        total_env_episodes = len(env.episodes)
        end_episode_index = args.end_episode_index if args.end_episode_index is not None else total_env_episodes
        max_episode_count = args.max_episodes if args.max_episodes is not None else total_env_episodes
        with manifest_tmp.open("w", encoding="utf-8") as manifest_handle:
            for env_episode_index in range(total_env_episodes):
                observations = env.reset()
                del observations
                episode = env.current_episode
                if env_episode_index < args.start_episode_index or env_episode_index >= end_episode_index:
                    continue
                if rendered_episodes >= max_episode_count:
                    break
                episode_id = str(episode.episode_id)
                gt_record = gt.get(episode_id)
                if not gt_record or not gt_record.get("locations"):
                    skipped_no_gt += 1
                    continue
                episode_counts = render_episode(
                    env,
                    episode,
                    gt_record,
                    manifest_handle=manifest_handle,
                    stage_root=stage_root,
                    family=family,
                    split=args.split,
                    role=args.role,
                    resume=args.resume,
                    max_frames=args.max_frames_per_episode,
                )
                rendered_frames += episode_counts["rendered_frames"]
                reused_frames += episode_counts["reused_frames"]
                rendered_episodes += 1
    finally:
        if env is not None:
            env.close()

    manifest_tmp.replace(manifest_path)
    summary = {
        "stage_root": str(stage_root),
        "manifest": str(manifest_path),
        "dataset_path": str(dataset_path),
        "gt_path": str(gt_path),
        "family": family,
        "split": args.split,
        "role": args.role,
        "languages": args.languages,
        "rendered_episodes": rendered_episodes,
        "rendered_frames": rendered_frames,
        "reused_frames": reused_frames,
        "skipped_no_gt": skipped_no_gt,
        "rgb_sensor": {
            "width": int(task_config.SIMULATOR.RGB_SENSOR.WIDTH),
            "height": int(task_config.SIMULATOR.RGB_SENSOR.HEIGHT),
            "hfov": float(task_config.SIMULATOR.RGB_SENSOR.HFOV),
        },
        "depth_enabled": "DEPTH_SENSOR" in list(task_config.SIMULATOR.AGENT_0.SENSORS),
    }
    (stage_root / "render_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_task_config(
    vlnce_root: Path,
    *,
    family: str,
    split: str,
    role: Optional[str],
    languages: List[str],
    dataset_path: Path,
    gpu_id: int,
    width: Optional[int],
    height: Optional[int],
):
    sys.path.insert(0, str(vlnce_root))
    from habitat_extensions.config.default import get_extended_config

    config = get_extended_config(str(default_task_config(vlnce_root, family=family)))
    config.defrost()
    config.DATASET.SPLIT = split
    config.DATASET.DATA_PATH = str(dataset_path)
    config.DATASET.SCENES_DIR = str(vlnce_root / "data" / "scene_datasets")
    if family == RXR_FAMILY:
        config.DATASET.ROLES = [role]
        config.DATASET.LANGUAGES = languages or ["*"]
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = int(gpu_id)
    if width is not None:
        config.SIMULATOR.RGB_SENSOR.WIDTH = int(width)
    if height is not None:
        config.SIMULATOR.RGB_SENSOR.HEIGHT = int(height)
    config.TASK.SENSORS = []
    config.TASK.MEASUREMENTS = []
    config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.freeze()
    return config


def render_episode(
    env,
    episode,
    gt_record: Dict[str, Any],
    *,
    manifest_handle,
    stage_root: Path,
    family: str,
    split: str,
    role: Optional[str],
    resume: bool,
    max_frames: Optional[int],
) -> Dict[str, int]:
    episode_id = str(episode.episode_id)
    trajectory_id = str(getattr(episode, "trajectory_id", episode_id))
    instruction = instruction_payload(episode)
    language = instruction.get("language")
    scene_id = str(getattr(episode, "scene_id", ""))
    locations = list(gt_record.get("locations") or [])
    actions = list(gt_record.get("actions") or [])
    if max_frames is not None:
        locations = locations[:max_frames]
    if not locations:
        return {"rendered_frames": 0, "reused_frames": 0}

    rotations = rotations_for_locations(locations, start_rotation=list(getattr(episode, "start_rotation", [])))
    rendered_frames = 0
    reused_frames = 0
    image_dir = stage_root / "rgb" / episode_id
    image_dir.mkdir(parents=True, exist_ok=True)
    for frame_index, location in enumerate(locations):
        rotation_xyzw, habitat_yaw, rotation_source = rotations[frame_index]
        rgb_path = image_dir / f"{frame_index:06d}.png"
        if resume and rgb_path.exists() and rgb_path.stat().st_size > 0:
            reused_frames += 1
        else:
            observations = env.sim.get_observations_at(
                position=[float(location[0]), float(location[1]), float(location[2])],
                rotation=[float(value) for value in rotation_xyzw],
                keep_agent_at_new_pose=True,
            )
            if observations is None or "rgb" not in observations:
                raise RuntimeError(f"failed to render RGB for episode {episode_id} frame {frame_index}")
            write_rgb_png(rgb_path, observations["rgb"])
            rendered_frames += 1

        row = {
            "dataset_family": family,
            "split": split,
            "role": role,
            "language": language,
            "episode_id": episode_id,
            "trajectory_id": trajectory_id,
            "scene_id": scene_id,
            "instruction_text": instruction["instruction_text"],
            "instruction": instruction,
            "frame_index": frame_index,
            "rgb_path": str(rgb_path),
            "position": [float(location[0]), float(location[1]), float(location[2])],
            "source_location": [float(location[0]), float(location[1]), float(location[2])],
            "agent_position": [float(location[0]), float(location[1]), float(location[2])],
            "yaw": navvla_yaw_from_habitat_yaw(habitat_yaw),
            "habitat_yaw": habitat_yaw,
            "rotation_source": rotation_source,
            "rotation_xyzw": [float(value) for value in rotation_xyzw],
            "native_action": actions[frame_index] if frame_index < len(actions) else None,
        }
        manifest_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest_handle.flush()
    return {"rendered_frames": rendered_frames, "reused_frames": reused_frames}


def write_rgb_png(path: Path, rgb_observation: Any) -> None:
    array = np.asarray(rgb_observation)
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise ValueError(f"expected RGB/RGBA observation shape [H,W,3|4], got {array.shape}")
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def rotations_for_locations(
    locations: List[List[float]],
    *,
    start_rotation: List[float],
) -> List[Tuple[List[float], float, str]]:
    rotations = []  # type: List[Tuple[List[float], float, str]]
    previous_habitat_yaw = 0.0
    previous_rotation = [0.0, 0.0, 0.0, 1.0]
    if len(start_rotation) == 4:
        previous_rotation = [float(value) for value in start_rotation]
        previous_habitat_yaw = habitat_yaw_from_xyzw(previous_rotation)
    for frame_index, location in enumerate(locations):
        if frame_index == 0:
            rotations.append((previous_rotation, previous_habitat_yaw, "episode_start_rotation"))
            continue
        previous = locations[frame_index - 1]
        dx = float(location[0] - previous[0])
        dz = float(location[2] - previous[2])
        if abs(dx) < 1e-7 and abs(dz) < 1e-7:
            rotations.append((previous_rotation, previous_habitat_yaw, "reused_previous_rotation_zero_delta"))
            continue
        previous_habitat_yaw = math.atan2(-dx, -dz)
        previous_rotation = xyzw_from_habitat_yaw(previous_habitat_yaw)
        rotations.append((previous_rotation, previous_habitat_yaw, "derived_from_previous_to_current_location"))
    return rotations


def xyzw_from_habitat_yaw(yaw: float) -> List[float]:
    return [0.0, math.sin(float(yaw) / 2.0), 0.0, math.cos(float(yaw) / 2.0)]


def habitat_yaw_from_xyzw(rotation_xyzw: List[float]) -> float:
    if len(rotation_xyzw) != 4:
        raise ValueError(f"expected xyzw quaternion with 4 values, got {rotation_xyzw}")
    y = float(rotation_xyzw[1])
    w = float(rotation_xyzw[3])
    return wrap_to_pi(2.0 * math.atan2(y, w))


def navvla_yaw_from_habitat_yaw(habitat_yaw: float) -> float:
    return wrap_to_pi(-float(habitat_yaw) - math.pi / 2.0)


def wrap_to_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def instruction_payload(episode) -> Dict[str, Any]:
    instruction = getattr(episode, "instruction", None)
    if instruction is None:
        raise ValueError(f"episode {getattr(episode, 'episode_id', '<unknown>')} has no instruction")
    payload = {}  # type: Dict[str, Any]
    for key in ("instruction_id", "instruction_text", "language", "annotator_id", "edit_distance"):
        if hasattr(instruction, key):
            value = getattr(instruction, key)
            if value is not None:
                payload[key] = value
    if "instruction_text" not in payload or not str(payload["instruction_text"]).strip():
        raise ValueError(f"episode {getattr(episode, 'episode_id', '<unknown>')} has empty instruction_text")
    payload["instruction_text"] = str(payload["instruction_text"]).strip()
    return payload


def load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
