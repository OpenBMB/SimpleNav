from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from PIL import Image

from tool.navvla.adapters.base import NavVLASourceAdapter, register_adapter
from tool.navvla.context_index import ContextIndexConfig
from tool.navvla import lerobot_v3_writer as _lerobot_writer
from tool.navvla.lerobot_v3_writer import write_navvla_lerobot_dataset
from tool.navvla.schema import NavVLACameraSpec, NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec
from tool.navvla.statistics import body_frame_action_from_pose, wrap_to_pi


TRACE_TYPES = ("ORI", "aug_001")
TARGET_SPLITS = {
    "train": "vln_train",
    "vln_train": "vln_train",
    "seen": "vln_val_seen",
    "val_seen": "vln_val_seen",
    "vln_val_seen": "vln_val_seen",
    "unseen": "vln_val_unseen",
    "val_unseen": "vln_val_unseen",
    "vln_val_unseen": "vln_val_unseen",
}
MANIFEST_SPLITS = {
    "vln_train": "train",
    "vln_val_seen": "seen",
    "vln_val_unseen": "unseen",
}
STANDARD_STATE_MODE = "source_world_absolute_pose_xyz_yaw"
COSFLY_IMAGE_SIZE = (384, 256)
INSTRUCTION = "Track the target pedestrian."
PLATFORM_TEXT = (
    "The platform is UAV for urban uav tracking. The control frequency is 2 Hz. "
    "Please predict the next 8 local 3D waypoints (dx, dy, dz, dyaw) to execute the following task:"
)
COSFLY_CONTEXT_INDEX_CONFIG = ContextIndexConfig(
    budget_num_cameras=1,
    history_camera_names=("front",),
)


class CosFlyAdapter(NavVLASourceAdapter):
    name = "cosfly"

    def __init__(
        self,
        *,
        fps: float = 2.0,
        action_horizon: int = 8,
        media_cache_root: str | Path | None = None,
        split_manifest_path: str | Path | None = None,
        load_workers: int | None = None,
    ) -> None:
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.split_manifest_path = Path(split_manifest_path) if split_manifest_path is not None else None
        self.load_workers = int(load_workers) if load_workers is not None else 1

    def configure(
        self,
        *,
        fps: float = 2.0,
        action_horizon: int = 8,
        media_cache_root: str | Path | None = None,
        split_manifest_path: str | Path | None = None,
        load_workers: int | None = None,
        **kwargs: Any,
    ) -> "CosFlyAdapter":
        super().configure(**kwargs)
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.split_manifest_path = Path(split_manifest_path) if split_manifest_path is not None else None
        self.load_workers = int(load_workers) if load_workers is not None else 1
        return self

    def load_episodes(
        self,
        source_root: str | Path,
        *,
        split: str = "train",
        max_episodes: int | None = None,
        media_cache_root: str | Path | None = None,
    ) -> list[NavVLAEpisode]:
        source_root = Path(source_root).resolve()
        target_split = normalize_target_split(split)
        if max_episodes is not None and max_episodes % len(TRACE_TYPES) != 0:
            raise ValueError(
                f"CosFly max_episodes must be even so ORI/aug pairs stay together, got {max_episodes}"
            )
        cache_root_value = media_cache_root if media_cache_root is not None else self.media_cache_root
        if cache_root_value is None:
            raise ValueError("CosFly requires a media cache root for 384x256 resized frames")
        cache_root = Path(cache_root_value).resolve()
        if cache_root == source_root or source_root in cache_root.parents:
            raise ValueError(
                f"CosFly media cache must be outside the source root: source={source_root} cache={cache_root}"
            )
        parent_roots = sorted(source_root.glob("Town*/trajectory_*"), key=lambda path: path.as_posix())
        if not parent_roots:
            raise FileNotFoundError(f"no CosFly parent trajectories found under {source_root}")
        if self.split_manifest_path is not None:
            selected_parent_ids = load_split_parent_ids(self.split_manifest_path, target_split=target_split)
            parent_roots = [
                parent_root
                for parent_root in parent_roots
                if parent_root.relative_to(source_root).as_posix() in selected_parent_ids
            ]
            missing_parent_ids = selected_parent_ids.difference(
                parent_root.relative_to(source_root).as_posix() for parent_root in parent_roots
            )
            if missing_parent_ids:
                raise FileNotFoundError(
                    f"CosFly split manifest references missing parents under {source_root}: "
                    f"{sorted(missing_parent_ids)[:5]}"
                )

        if max_episodes is not None:
            parent_roots = parent_roots[: max_episodes // len(TRACE_TYPES)]
        jobs = [
            (
                parent_root,
                parent_index * len(TRACE_TYPES),
                target_split,
                self.action_horizon,
                cache_root,
            )
            for parent_index, parent_root in enumerate(parent_roots)
        ]
        if self.load_workers <= 1 or len(jobs) <= 1:
            parent_episodes = [build_parent_episodes(*job) for job in jobs]
        else:
            with ThreadPoolExecutor(max_workers=min(self.load_workers, len(jobs))) as executor:
                parent_episodes = list(executor.map(lambda job: build_parent_episodes(*job), jobs))
        episodes = [episode for pair in parent_episodes for episode in pair]
        if not episodes:
            raise FileNotFoundError(f"no CosFly episodes found under {source_root}")
        return episodes

    def convert(
        self,
        *,
        source_root: str | Path,
        output_root: str | Path,
        dataset_name: str,
        max_episodes: int | None,
        fps: float,
        action_horizon: int,
        overwrite: bool,
        control_frequency_hz: float | None = None,
        repair_existing: bool = False,
        split: str = "train",
        context_policy_version: str = "bats-v1",
        cache_policy_version: str = "smoke-coarse-v1",
        cache_workers: int | None = None,
        write_visual_token_cache: bool = True,
        visual_token_profile: Any | None = None,
        visual_token_encoder: Any | None = None,
        visual_token_encoder_factory: Any | None = None,
        episodes_per_file: int = 20,
        files_per_chunk: int = 50,
    ) -> dict[str, Any]:
        source_path = Path(source_root).resolve()
        output_path = Path(output_root).resolve()
        if output_path == source_path or source_path in output_path.parents:
            raise ValueError(f"CosFly output root must be outside the source root: source={source_path} output={output_path}")
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        target_split = normalize_target_split(split)
        media_cache_root = self.media_cache_root or (output_path / "_media_cache_384x256")
        episodes = self.load_episodes(
            source_path,
            split=target_split,
            max_episodes=max_episodes,
            media_cache_root=media_cache_root,
        )
        spec = NavVLADatasetSpec(
            dataset_name=dataset_name,
            fps=float(fps),
            control_frequency_hz=(
                float(control_frequency_hz) if control_frequency_hz is not None else float(fps)
            ),
            action_horizon=int(action_horizon),
            action_dim=4,
            state_dim=4,
            context_policy_version=context_policy_version,
            cache_policy_version=cache_policy_version,
            split=target_split,
            episodes_per_file=episodes_per_file,
            files_per_chunk=files_per_chunk,
            state_mode=STANDARD_STATE_MODE,
        )
        summary = write_cosfly_lerobot_dataset(
            episodes,
            output_root=output_path,
            spec=spec,
            overwrite=overwrite,
            repair_existing=repair_existing,
            cache_workers=cache_workers,
            write_visual_token_cache=write_visual_token_cache,
            visual_token_profile=visual_token_profile,
            visual_token_encoder=visual_token_encoder,
            visual_token_encoder_factory=visual_token_encoder_factory,
            context_index_config=COSFLY_CONTEXT_INDEX_CONFIG,
        )
        summary["cosfly_metadata_finalization"] = finalize_cosfly_metadata(
            summary["dataset_root"],
            target_split=target_split,
        )
        return summary


def write_cosfly_lerobot_dataset(*args: Any, **kwargs: Any) -> dict[str, Any]:
    generic_build_context_indexes = _lerobot_writer.build_context_indexes

    def build_2048_context_indexes(*context_args: Any, **context_kwargs: Any) -> Any:
        context_kwargs["token_budgets"] = (2048,)
        return generic_build_context_indexes(*context_args, **context_kwargs)

    with patch.object(_lerobot_writer, "build_context_indexes", build_2048_context_indexes):
        return write_navvla_lerobot_dataset(*args, **kwargs)


def normalize_target_split(split: str) -> str:
    value = str(split).strip()
    try:
        return TARGET_SPLITS[value]
    except KeyError as exc:
        raise ValueError(f"unsupported CosFly split {split!r}; expected train/seen/unseen") from exc


def canonical_dataset_name(split: str) -> str:
    return normalize_target_split(split)


def logical_source_split(split: str) -> str:
    return MANIFEST_SPLITS[normalize_target_split(split)]


def finalize_cosfly_metadata(dataset_root: str | Path, *, target_split: str) -> dict[str, Any]:
    root = Path(dataset_root)
    canonical_name = canonical_dataset_name(target_split)
    source_split = logical_source_split(target_split)
    info_path = root / "meta" / "info.json"
    statistics_path = root / "dataset_statistics.json"
    frame_metadata_path = root / "meta" / "navvla_frame_metadata.jsonl"
    for path in (info_path, statistics_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing CosFly metadata artifact: {path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["dataset_name"] = canonical_name
    navvla = info.get("navvla")
    if not isinstance(navvla, dict):
        raise ValueError(f"CosFly info.json is missing navvla metadata: {info_path}")
    navvla["state_mode"] = STANDARD_STATE_MODE
    _write_json_atomic(info_path, info)

    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    if not isinstance(statistics, dict) or len(statistics) != 1:
        raise ValueError(f"CosFly dataset_statistics.json must contain exactly one dataset key: {statistics_path}")
    statistics_block = next(iter(statistics.values()))
    statistics_key = f"{canonical_name}_{canonical_name}"
    _write_json_atomic(statistics_path, {statistics_key: statistics_block})

    frame_metadata_rows = 0
    if frame_metadata_path.is_file():
        temporary_path = frame_metadata_path.with_suffix(".jsonl.tmp")
        with frame_metadata_path.open("r", encoding="utf-8") as source_handle, temporary_path.open(
            "w", encoding="utf-8"
        ) as output_handle:
            for line in source_handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                source_metadata = row.get("source_metadata")
                if isinstance(source_metadata, dict):
                    source_metadata["source_split"] = source_split
                    source_metadata["target_split"] = canonical_name
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                frame_metadata_rows += 1
        temporary_path.replace(frame_metadata_path)

    return {
        "dataset_root": str(root),
        "dataset_name": canonical_name,
        "source_split": source_split,
        "target_split": canonical_name,
        "state_mode": STANDARD_STATE_MODE,
        "statistics_key": statistics_key,
        "frame_metadata_rows": frame_metadata_rows,
    }


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def load_split_parent_ids(path: str | Path, *, target_split: str) -> set[str]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "cosfly_navvla_split_manifest_v1":
        raise ValueError(f"unsupported CosFly split manifest schema: {manifest_path}")
    parents = payload.get("parents")
    if not isinstance(parents, list):
        raise ValueError(f"CosFly split manifest parents must be a list: {manifest_path}")
    manifest_split = MANIFEST_SPLITS[target_split]
    selected = {
        str(row["parent_id"])
        for row in parents
        if isinstance(row, dict) and str(row.get("split")) == manifest_split
    }
    if not selected:
        raise ValueError(f"CosFly split manifest has no parents for {manifest_split}: {manifest_path}")
    return selected


def load_trace_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "drone_nav_traj_v7" or payload.get("schema_version") != "7.0.0":
        raise ValueError(
            f"unsupported CosFly trajectory schema in {path}: "
            f"{payload.get('schema')} {payload.get('schema_version')}"
        )
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError(f"CosFly trajectory must contain non-empty points: {path}")
    return payload


def build_parent_episodes(
    parent_root: Path,
    task_index_start: int,
    target_split: str,
    action_horizon: int,
    media_cache_root: Path,
) -> list[NavVLAEpisode]:
    episodes = []
    for trace_offset, trace_type in enumerate(TRACE_TYPES):
        trajectory_path = parent_root / trace_type / "trajectory.json"
        if not trajectory_path.is_file():
            raise FileNotFoundError(
                f"CosFly parent trajectory is missing {trace_type} trace JSON: {trajectory_path}"
            )
        episodes.append(
            build_episode(
                parent_root=parent_root,
                trajectory_path=trajectory_path,
                payload=load_trace_payload(trajectory_path),
                trace_type=trace_type,
                task_index=task_index_start + trace_offset,
                target_split=target_split,
                action_horizon=action_horizon,
                media_cache_root=media_cache_root,
            )
        )
    return episodes


def build_letterbox_transform(
    camera_payload: dict[str, Any],
    *,
    target_size: tuple[int, int],
) -> dict[str, Any]:
    try:
        source_width = int(camera_payload["width"])
        source_height = int(camera_payload["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"CosFly camera must contain positive width/height, got {camera_payload!r}") from exc
    target_width, target_height = (int(target_size[0]), int(target_size[1]))
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError(
            f"CosFly image dimensions must be positive: source={source_width}x{source_height} "
            f"target={target_width}x{target_height}"
        )
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    return {
        "source_width": source_width,
        "source_height": source_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "target_width": target_width,
        "target_height": target_height,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "scale_x": resized_width / source_width,
        "scale_y": resized_height / source_height,
    }


def image_transform_metadata(transform: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": "letterbox",
        "source_size": [int(transform["source_width"]), int(transform["source_height"])],
        "resized_content_size": [int(transform["resized_width"]), int(transform["resized_height"])],
        "target_size": [int(transform["target_width"]), int(transform["target_height"])],
        "offset_xy": [int(transform["offset_x"]), int(transform["offset_y"])],
        "scale_xy": [float(transform["scale_x"]), float(transform["scale_y"])],
    }


def transform_intrinsics(intrinsics: Any, transform: dict[str, Any]) -> list[list[float]] | None:
    if intrinsics is None:
        return None
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"CosFly camera intrinsic must be a finite 3x3 matrix, got {intrinsics!r}")
    image_transform = np.asarray(
        [
            [float(transform["scale_x"]), 0.0, float(transform["offset_x"])],
            [0.0, float(transform["scale_y"]), float(transform["offset_y"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return [[_clean_float(value) for value in row] for row in (image_transform @ matrix).tolist()]


def transform_image_uv(value: Any, transform: dict[str, Any]) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"CosFly target image_uv must contain x/y values, got {value!r}")
    return [
        _clean_float(float(value[0]) * float(transform["scale_x"]) + float(transform["offset_x"])),
        _clean_float(float(value[1]) * float(transform["scale_y"]) + float(transform["offset_y"])),
    ]


def transform_bbox_2d(value: Any, transform: dict[str, Any]) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not all(key in value for key in ("xmin", "ymin", "xmax", "ymax")):
        raise ValueError(f"CosFly target bbox_2d must contain xmin/ymin/xmax/ymax, got {value!r}")
    return {
        "xmin": _clean_float(float(value["xmin"]) * float(transform["scale_x"]) + float(transform["offset_x"])),
        "ymin": _clean_float(float(value["ymin"]) * float(transform["scale_y"]) + float(transform["offset_y"])),
        "xmax": _clean_float(float(value["xmax"]) * float(transform["scale_x"]) + float(transform["offset_x"])),
        "ymax": _clean_float(float(value["ymax"]) * float(transform["scale_y"]) + float(transform["offset_y"])),
    }


def materialize_cosfly_image(
    source_path: Path,
    *,
    output_path: Path,
    transform: dict[str, Any],
) -> Path:
    target_size = (int(transform["target_width"]), int(transform["target_height"]))
    if output_path.is_file() and output_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
        try:
            with Image.open(output_path) as cached:
                if cached.mode == "RGB" and cached.size == target_size:
                    return output_path
        except Exception:
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
        source_size = (int(transform["source_width"]), int(transform["source_height"]))
        if rgb.size != source_size:
            raise ValueError(
                f"CosFly RGB size does not match camera metadata: {source_path} "
                f"image={rgb.size} camera={source_size}"
            )
        resized = rgb.resize(
            (int(transform["resized_width"]), int(transform["resized_height"])),
            Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", target_size, color=(0, 0, 0))
        canvas.paste(resized, (int(transform["offset_x"]), int(transform["offset_y"])))
        temporary_path = output_path.with_suffix(".tmp.png")
        canvas.save(temporary_path, format="PNG")
        temporary_path.replace(output_path)
    return output_path


def build_episode(
    *,
    parent_root: Path,
    trajectory_path: Path,
    payload: dict[str, Any],
    trace_type: str,
    task_index: int,
    target_split: str,
    action_horizon: int,
    media_cache_root: Path,
) -> NavVLAEpisode:
    if trace_type not in TRACE_TYPES:
        raise ValueError(f"unsupported CosFly trace type: {trace_type}")
    if payload.get("trace_dir") != trace_type:
        raise ValueError(
            f"CosFly trace_dir mismatch in {trajectory_path}: expected {trace_type}, got {payload.get('trace_dir')}"
        )
    points = list(payload["points"])
    validate_points(points, trajectory_path=trajectory_path)
    poses = [pose4_from_drone_pose(point["drone_pose"]) for point in points]

    town = parent_root.parent.name
    parent_name = parent_root.name
    parent_id = f"{town}/{parent_name}"
    episode_id = f"{town}__{parent_name}__{trace_type}"
    task = NavVLATaskSpec(
        task_index=task_index,
        instruction=INSTRUCTION,
        task_type="tracking",
        task_subtype="urban_uav_tracking",
        platform_text=PLATFORM_TEXT,
        dataset_source="cosfly",
        scene_id=town,
        metadata={
            "source_has_language_instruction": False,
            "instruction_source": "converter_generated",
            "parent_trajectory_id": parent_id,
            "source_trajectory_json": str(trajectory_path),
        },
    )
    camera_payload = payload.get("camera") or {}
    image_transform = build_letterbox_transform(camera_payload, target_size=COSFLY_IMAGE_SIZE)
    transformed_intrinsics = transform_intrinsics(camera_payload.get("intrinsic"), image_transform)
    frames = []
    timestamp0 = float(points[0]["timing"]["trace_timestamp"])
    for frame_index, (point, pose) in enumerate(zip(points, poses, strict=True)):
        source_frame_index = int(point["index"])
        source_image_path = (
            trajectory_path.parent / "frames_playback" / f"frame_{source_frame_index:05d}" / "rgb.png"
        )
        validate_rgb(source_image_path)
        image_path = materialize_cosfly_image(
            source_image_path,
            output_path=(
                media_cache_root
                / target_split
                / episode_id
                / f"{source_frame_index:06d}.png"
            ),
            transform=image_transform,
        )
        action = action_chunk_for_frame(poses, frame_idx=frame_index, horizon=action_horizon)
        target = point.get("target") or {}
        raw_pose = point["drone_pose"]
        source_target_image_uv = target.get("image_uv")
        source_target_bbox_2d = target.get("bbox_2d")
        frames.append(
            NavVLAFrame(
                frame_index=frame_index,
                timestamp=float(point["timing"]["trace_timestamp"]) - timestamp0,
                media_paths={"front_image": image_path},
                state=pose,
                action=action,
                action_available=bool(action),
                source_frame_index=source_frame_index,
                source_metadata={
                    "source_dataset": "cosfly",
                    "source_split": logical_source_split(target_split),
                    "target_split": target_split,
                    "scene_id": town,
                    "episode_id": episode_id,
                    "trajectory_id": episode_id,
                    "parent_trajectory_id": parent_id,
                    "source_trajectory_json": str(trajectory_path),
                    "source_image_path": str(source_image_path),
                    "is_perturbed": bool(point.get("is_perturbed", False)),
                    "perturbation": point.get("perturbation"),
                    "raw_pitch_deg": float(raw_pose.get("pitch", 0.0)),
                    "raw_roll_deg": float(raw_pose.get("roll", 0.0)),
                    "raw_yaw_deg": float(raw_pose["yaw"]),
                    "coordinate_transform": "[x,y,z_up,yaw_deg] -> [x,y,-z,yaw_rad_wrapped]",
                    "target_visible": bool(target.get("visible", False)),
                    "target_in_view": bool(target.get("in_view", False)),
                    "target_depth_m": target.get("depth"),
                    "target_world_location": target.get("world_location"),
                    "image_resize": image_transform_metadata(image_transform),
                    "source_target_image_uv": source_target_image_uv,
                    "target_image_uv": transform_image_uv(source_target_image_uv, image_transform),
                    "source_target_bbox_2d": source_target_bbox_2d,
                    "target_bbox_2d": transform_bbox_2d(source_target_bbox_2d, image_transform),
                },
            )
        )

    camera = NavVLACameraSpec(
        name="front",
        video_key="front_image",
        viewpoint_type="front_tracking",
        azimuth_rad=0.0,
        intrinsics=transformed_intrinsics,
        calibration_status="source_intrinsics_letterboxed_dynamic_extrinsics",
    )
    return NavVLAEpisode(
        episode_id=episode_id,
        trajectory_id=episode_id,
        task=task,
        frames=frames,
        cameras=[camera],
        split=target_split,
    )


def validate_points(points: list[dict[str, Any]], *, trajectory_path: Path) -> None:
    indices = [int(point.get("index", -1)) for point in points]
    if indices != list(range(len(points))):
        raise ValueError(f"CosFly frame indices must be contiguous from zero: {trajectory_path}")
    timestamps = []
    for point in points:
        timing = point.get("timing") or {}
        if "trace_timestamp" not in timing:
            raise ValueError(f"CosFly point is missing timing.trace_timestamp: {trajectory_path}")
        timestamps.append(float(timing["trace_timestamp"]))
        pose4_from_drone_pose(point.get("drone_pose"))
    if any(next_value <= current for current, next_value in zip(timestamps, timestamps[1:])):
        raise ValueError(f"CosFly trace timestamps must be strictly increasing: {trajectory_path}")

    for point, next_point in zip(points, points[1:]):
        waypoint = (point.get("nav_waypoint") or {}).get("t1_world")
        if not isinstance(waypoint, dict):
            raise ValueError(f"CosFly non-terminal point is missing nav_waypoint.t1_world: {trajectory_path}")
        next_pose = next_point["drone_pose"]
        if not np.allclose(
            [float(waypoint[axis]) for axis in ("x", "y", "z")],
            [float(next_pose[axis]) for axis in ("x", "y", "z")],
            atol=1e-5,
        ):
            raise ValueError(f"CosFly waypoint/next-pose mismatch: {trajectory_path} frame {point['index']}")


def validate_rgb(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing or empty CosFly RGB frame: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"undecodable CosFly RGB frame: {path}: {exc}") from exc


def pose4_from_drone_pose(raw_pose: Any) -> list[float]:
    if not isinstance(raw_pose, dict):
        raise ValueError(f"CosFly drone_pose must be an object, got {raw_pose!r}")
    try:
        x = float(raw_pose["x"])
        y = float(raw_pose["y"])
        z_up = float(raw_pose["z"])
        yaw_deg = float(raw_pose["yaw"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"CosFly drone_pose must contain finite x/y/z/yaw values, got {raw_pose!r}") from exc
    values = np.asarray([x, y, z_up, yaw_deg], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"CosFly drone_pose must contain finite x/y/z/yaw values, got {raw_pose!r}")
    yaw_rad = float(wrap_to_pi(math.radians(yaw_deg)))
    return [_clean_float(x), _clean_float(y), _clean_float(-z_up), _clean_float(yaw_rad)]


def action_chunk_for_frame(poses: list[list[float]], *, frame_idx: int, horizon: int) -> list[list[float]]:
    current = poses[frame_idx]
    chunk = []
    for future_idx in range(frame_idx + 1, min(len(poses), frame_idx + 1 + horizon)):
        action = body_frame_action_from_pose(current, poses[future_idx]).astype(float).tolist()
        chunk.append([_clean_float(value) for value in action])
    return chunk


def _clean_float(value: float) -> float:
    value = float(value)
    return 0.0 if abs(value) < 1e-7 else value


register_adapter(CosFlyAdapter())
