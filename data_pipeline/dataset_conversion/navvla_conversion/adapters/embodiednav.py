from __future__ import annotations

import math
import pickle
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from navvla_conversion.adapters.base import NavVLASourceAdapter, register_adapter
from navvla_conversion.context_index import ContextIndexConfig
from navvla_conversion.workers import resolve_workers as resolve_load_workers
from navvla_conversion.lerobot_v3_writer import write_navvla_lerobot_dataset
from navvla_conversion.schema import NavVLACameraSpec, NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec
from navvla_conversion.statistics import body_frame_action_from_pose


FRONT_CAMERA = NavVLACameraSpec(
    name="front",
    video_key="front_image",
    viewpoint_type="front",
    azimuth_rad=0.0,
    calibration_status="unknown",
)
PLATFORM_TEXT = "Platform: UAV. Task: instruction-conditioned navigation. Action: local 3D waypoints (dx, dy, dz, dyaw)."
EMBODIEDNAV_CONTEXT_INDEX_CONFIG = ContextIndexConfig(budget_num_cameras=1, history_camera_names=("front",))
YAW_POLICY = "trajectory_tangent_yaw"


class EmbodiedNavAdapter(NavVLASourceAdapter):
    name = "embodiednav"

    def __init__(
        self,
        *,
        media_cache_root: str | Path | None = None,
        reuse_media_cache: bool = False,
        fps: float = 1.0,
        action_horizon: int = 8,
    ) -> None:
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.reuse_media_cache = bool(reuse_media_cache)
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.summary: dict[str, Any] = {}
        self.load_workers: int | None = None

    def configure(
        self,
        *,
        media_cache_root: str | Path | None = None,
        reuse_media_cache: bool = False,
        fps: float = 1.0,
        action_horizon: int = 8,
        load_workers: int | None = None,
        **kwargs: Any,
    ) -> "EmbodiedNavAdapter":
        super().configure(**kwargs)
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.reuse_media_cache = bool(reuse_media_cache)
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.load_workers = load_workers
        return self

    def load_episodes(
        self,
        source_root: str | Path,
        *,
        split: str = "train",
        max_episodes: int | None = None,
        load_workers: int | None = None,
    ) -> list[NavVLAEpisode]:
        root = Path(source_root)
        samples = load_navi_data(root)
        if max_episodes is not None:
            samples = samples[:max_episodes]
        if not samples:
            raise FileNotFoundError(f"no EmbodiedNav samples found in {root / 'navi_data.pkl'}")

        target_split = target_split_name(split)
        media_cache_root = resolve_media_cache_root(root, media_cache_root=self.media_cache_root)
        zip_index = {} if self.reuse_media_cache else build_zip_index(root)
        jobs = [
            (
                sample_index,
                sample,
                str(root),
                str(media_cache_root),
                target_split,
                float(self.fps),
                int(self.action_horizon),
                bool(self.reuse_media_cache),
                {folder: str(path) for folder, path in zip_index.items()},
            )
            for sample_index, sample in enumerate(samples)
        ]

        resolved_workers = resolve_embodiednav_load_workers(load_workers)
        if resolved_workers == 1 or len(jobs) == 1:
            episodes = [build_episode_from_job(job) for job in jobs]
        else:
            max_workers = min(resolved_workers, len(jobs))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                episodes = list(executor.map(build_episode_from_job, jobs, chunksize=16))

        episodes = [
            replace(episode, episode_id=f"{episode_index:05d}", task=replace(episode.task, task_index=episode_index))
            for episode_index, episode in enumerate(episodes)
        ]
        self.summary = {
            "source_root": str(root),
            "media_cache_root": str(media_cache_root),
            "loaded_episodes": len(episodes),
            "loaded_frames": sum(len(episode.frames) for episode in episodes),
            "reuse_media_cache": self.reuse_media_cache,
            "yaw_policy": YAW_POLICY,
        }
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
        repair_existing: bool = False,
        split: str = "train",
        control_frequency_hz: float | None = None,
        context_policy_version: str = "bats-v1",
        cache_policy_version: str = "smoke-coarse-v1",
        write_workers: int | None = None,
        write_visual_token_cache: bool = False,
        visual_token_profile: Any | None = None,
        visual_token_encoder: Any | None = None,
        visual_token_encoder_factory: Any | None = None,
        episodes_per_file: int = 20,
        files_per_chunk: int = 50,
        load_workers: int | None = None,
    ) -> dict[str, Any]:
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        target_split = target_split_name(split)
        episodes = self.load_episodes(
            source_root,
            split=split,
            max_episodes=max_episodes,
            load_workers=self.load_workers if load_workers is None else load_workers,
        )
        spec = NavVLADatasetSpec(
            dataset_name=dataset_name,
            fps=fps,
            control_frequency_hz=float(control_frequency_hz) if control_frequency_hz is not None else float(fps),
            action_horizon=action_horizon,
            action_dim=4,
            state_dim=4,
            context_policy_version=context_policy_version,
            cache_policy_version=cache_policy_version,
            split=target_split,
            episodes_per_file=episodes_per_file,
            files_per_chunk=files_per_chunk,
        )
        summary = write_navvla_lerobot_dataset(
            episodes,
            output_root=Path(output_root),
            spec=spec,
            overwrite=overwrite,
            repair_existing=repair_existing,
            write_workers=write_workers,
            write_visual_token_cache=write_visual_token_cache,
            visual_token_profile=visual_token_profile,
            visual_token_encoder=visual_token_encoder,
            visual_token_encoder_factory=visual_token_encoder_factory,
            context_index_config=EMBODIEDNAV_CONTEXT_INDEX_CONFIG,
        )
        summary["embodiednav_summary"] = dict(self.summary)
        return summary


def load_navi_data(source_root: str | Path) -> list[dict[str, Any]]:
    path = Path(source_root) / "navi_data.pkl"
    if not path.exists():
        raise FileNotFoundError(f"EmbodiedNav navi_data.pkl not found: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list of sample dictionaries")
    return payload


def target_split_name(split: str) -> str:
    value = split.strip()
    if value == "train":
        return "vln_train"
    if value == "test":
        return "vln_test"
    if value.startswith("vln_"):
        return value
    raise ValueError(f"unsupported EmbodiedNav split: {split}")


def resolve_media_cache_root(source_root: Path, *, media_cache_root: str | Path | None) -> Path:
    if media_cache_root is not None:
        return Path(media_cache_root)
    return source_root / ".navvla_media_cache"


def resolve_embodiednav_load_workers(load_workers: int | None) -> int:
    if load_workers is None:
        return 1
    return resolve_load_workers(load_workers)


def build_zip_index(source_root: str | Path) -> dict[str, Path]:
    image_root = Path(source_root) / "images"
    zip_paths = sorted(image_root.glob("merged_upload_images_part*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"no EmbodiedNav image zip archives found under {image_root}")
    index: dict[str, Path] = {}
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if name.endswith("/") or "/" not in name:
                    continue
                folder = name.split("/", 1)[0]
                if folder.isdigit() and folder not in index:
                    index[folder] = zip_path
    if not index:
        raise FileNotFoundError(f"no sample folders found in EmbodiedNav image archives under {image_root}")
    return index


def build_episode_from_job(job: tuple[Any, ...]) -> NavVLAEpisode:
    (
        sample_index,
        sample,
        source_root_str,
        media_cache_root_str,
        target_split,
        fps,
        action_horizon,
        reuse_media_cache,
        zip_index_raw,
    ) = job
    zip_index = {str(folder): Path(path) for folder, path in dict(zip_index_raw).items()}
    return build_episode(
        sample,
        sample_index=int(sample_index),
        source_root=Path(source_root_str),
        media_cache_root=Path(media_cache_root_str),
        target_split=str(target_split),
        fps=float(fps),
        action_horizon=int(action_horizon),
        reuse_media_cache=bool(reuse_media_cache),
        zip_index=zip_index,
    )


def build_episode(
    sample: dict[str, Any],
    *,
    sample_index: int,
    source_root: Path,
    media_cache_root: Path,
    target_split: str,
    fps: float,
    action_horizon: int,
    reuse_media_cache: bool,
    zip_index: dict[str, Path],
) -> NavVLAEpisode:
    folder = str(required_value(sample, "folder"))
    instruction = str(required_value(sample, "task_desc")).strip()
    if not instruction:
        raise ValueError(f"EmbodiedNav sample {folder} has empty task_desc")
    trajectory = trajectory_points(sample)
    start_rot = float_list(required_value(sample, "start_rot"))
    if len(start_rot) < 3:
        raise ValueError(f"EmbodiedNav sample {folder} start_rot must contain [roll,pitch,yaw]")
    start_yaw = float(start_rot[2])
    yaws, yaw_fallback = trajectory_tangent_yaws(trajectory, start_yaw=start_yaw)
    poses = [
        [float(point[0]), float(point[1]), float(point[2]), float(yaw)]
        for point, yaw in zip(trajectory, yaws)
    ]

    task = NavVLATaskSpec(
        task_index=sample_index,
        instruction=instruction,
        task_type="navigation",
        task_subtype="embodiednav",
        platform_text=PLATFORM_TEXT,
        dataset_source="embodiednav",
        scene_id="embodiednav",
    )
    frames: list[NavVLAFrame] = []
    for frame_index, pose in enumerate(poses):
        media_path, archive_path, image_member = materialize_image(
            folder=folder,
            frame_index=frame_index,
            media_cache_root=media_cache_root,
            reuse_media_cache=reuse_media_cache,
            zip_index=zip_index,
        )
        frames.append(
            NavVLAFrame(
                frame_index=frame_index,
                timestamp=float(frame_index) / float(fps),
                media_paths={"front_image": media_path},
                state=pose,
                action=action_chunk_for_frame(poses, frame_idx=frame_index, horizon=action_horizon),
                action_available=frame_index < len(poses) - 1,
                source_frame_index=frame_index,
                source_metadata={
                    "source_dataset": "embodiednav",
                    "folder": folder,
                    "sample_index": sample_index,
                    "task_desc": instruction,
                    "start_pos": float_list(required_value(sample, "start_pos")),
                    "start_rot": start_rot,
                    "start_ang": float(required_value(sample, "start_ang")),
                    "target_pos": float_list(required_value(sample, "target_pos")),
                    "gt_traj_len": float(required_value(sample, "gt_traj_len")),
                    "source_pose": [float(value) for value in trajectory[frame_index].tolist()],
                    "yaw": float(pose[3]),
                    "yaw_policy": YAW_POLICY,
                    "yaw_fallback": yaw_fallback,
                    "image_archive": str(archive_path) if archive_path is not None else None,
                    "image_member": image_member,
                },
            )
        )

    return NavVLAEpisode(
        episode_id=f"{sample_index:05d}",
        trajectory_id=folder,
        task=task,
        frames=frames,
        cameras=[FRONT_CAMERA],
        split=target_split,
    )


def required_value(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"EmbodiedNav sample is missing required field: {key}")
    return payload[key]


def trajectory_points(sample: dict[str, Any]) -> np.ndarray:
    trajectory = np.asarray(required_value(sample, "gt_traj"), dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or trajectory.shape[0] == 0:
        raise ValueError(f"EmbodiedNav gt_traj must have shape [N,3], got {trajectory.shape}")
    return trajectory


def float_list(value: Any) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=np.float64).reshape(-1).tolist()]


def trajectory_tangent_yaws(trajectory: np.ndarray, *, start_yaw: float, eps: float = 1e-6) -> tuple[list[float], str]:
    if trajectory.shape[0] == 1:
        return [float(start_yaw)], "start_rot_yaw"
    yaws: list[float] = []
    last_valid: float | None = None
    used_start_fallback = False
    horizontal_steps = 0
    for index in range(trajectory.shape[0] - 1):
        delta = trajectory[index + 1] - trajectory[index]
        if math.hypot(float(delta[0]), float(delta[1])) > eps:
            last_valid = math.atan2(float(delta[1]), float(delta[0]))
            horizontal_steps += 1
        elif last_valid is None:
            last_valid = float(start_yaw)
            used_start_fallback = True
        yaws.append(float(last_valid))
    if horizontal_steps == 0:
        yaws.append(float(start_yaw))
        return yaws, "start_rot_yaw"
    yaws.append(float(last_valid if last_valid is not None else start_yaw))
    return yaws, "leading_start_rot_yaw" if used_start_fallback else "none"


def action_chunk_for_frame(poses: list[list[float]], *, frame_idx: int, horizon: int) -> list[list[float]]:
    current = poses[frame_idx]
    chunk = []
    for future_idx in range(frame_idx + 1, min(len(poses), frame_idx + 1 + horizon)):
        action = body_frame_action_from_pose(current, poses[future_idx]).astype(float).tolist()
        chunk.append([clean_float(value) for value in action])
    return chunk


def clean_float(value: float) -> float:
    value = float(value)
    return 0.0 if abs(value) < 1e-7 else value


def materialize_image(
    *,
    folder: str,
    frame_index: int,
    media_cache_root: Path,
    reuse_media_cache: bool,
    zip_index: dict[str, Path],
) -> tuple[Path, Path | None, str]:
    member = image_member(folder=folder, frame_index=frame_index)
    path = media_cache_root / "embodiednav" / sanitize_folder(folder) / f"{frame_index:06d}.png"
    if reuse_media_cache:
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing cached EmbodiedNav image for folder {folder} frame {frame_index}: {path}")
        return path, zip_index.get(folder), member
    if folder not in zip_index:
        raise FileNotFoundError(f"no EmbodiedNav image archive found for folder {folder}")
    archive_path = zip_index[folder]
    with zipfile.ZipFile(archive_path) as archive:
        try:
            image_bytes = archive.read(member)
        except KeyError as exc:
            raise FileNotFoundError(f"EmbodiedNav image member not found: {archive_path}:{member}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != len(image_bytes):
        path.write_bytes(image_bytes)
    return path, archive_path, member


def image_member(*, folder: str, frame_index: int) -> str:
    filename = "initial.png" if frame_index == 0 else f"{frame_index - 1}.png"
    return f"{folder}/{filename}"


def sanitize_folder(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not sanitized:
        raise ValueError("empty EmbodiedNav folder after sanitization")
    return sanitized


register_adapter(EmbodiedNavAdapter())
