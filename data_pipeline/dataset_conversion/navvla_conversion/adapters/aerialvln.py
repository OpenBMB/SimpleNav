from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from navvla_conversion.adapters.base import NavVLASourceAdapter, register_adapter
from navvla_conversion.context_index import ContextIndexConfig
from navvla_conversion.lerobot_v3_writer import write_navvla_lerobot_dataset
from navvla_conversion.schema import NavVLACameraSpec, NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec
from navvla_conversion.statistics import body_frame_action_from_pose


SOURCE_SPLIT_TO_TARGET = {
    "train": "vln_train",
    "val_seen": "vln_val_seen",
    "val_unseen": "vln_val_unseen",
}
FRONT_CAMERA = NavVLACameraSpec(name="front", video_key="front_image", viewpoint_type="front", azimuth_rad=0.0)
PLATFORM_TEXT = "Platform: UAV. Task: instruction-conditioned navigation. Action: local 3D waypoints (dx, dy, dz, dyaw)."
AERIALVLN_CONTEXT_INDEX_CONFIG = ContextIndexConfig(budget_num_cameras=1, history_camera_names=("front",))


class AerialVLNAdapter(NavVLASourceAdapter):
    name = "aerialvln"

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

    def configure(
        self,
        *,
        media_cache_root: str | Path | None = None,
        reuse_media_cache: bool = False,
        fps: float = 1.0,
        action_horizon: int = 8,
        **kwargs: Any,
    ) -> "AerialVLNAdapter":
        super().configure(**kwargs)
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.reuse_media_cache = bool(reuse_media_cache)
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        return self

    def load_episodes(
        self,
        source_root: str | Path,
        *,
        split: str = "train",
        max_episodes: int | None = None,
    ) -> list[NavVLAEpisode]:
        source_root = Path(source_root)
        source_split = normalize_source_split(split)
        if source_split == "test":
            raise ValueError("AerialVLN test split lacks reference_path/actions and cannot be converted to NavVLA actions")
        split_path = source_root / "aerialvln" / f"{source_split}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"AerialVLN split file not found: {split_path}")

        payload = json.loads(split_path.read_text(encoding="utf-8"))
        source_episodes = payload.get("episodes")
        if not isinstance(source_episodes, list):
            raise ValueError(f"{split_path} must contain a top-level episodes list")
        if max_episodes is not None:
            source_episodes = source_episodes[:max_episodes]
        if not source_episodes:
            raise FileNotFoundError(f"no AerialVLN episodes found in {split_path}")

        media_cache_root = resolve_media_cache_root(source_root, media_cache_root=self.media_cache_root)
        target_split = target_split_name(source_split)
        episodes = []
        for task_index, source_episode in enumerate(source_episodes):
            episodes.append(
                build_episode(
                    source_episode,
                    source_root=source_root,
                    media_cache_root=media_cache_root,
                    source_split=source_split,
                    target_split=target_split,
                    task_index=task_index,
                    fps=self.fps,
                    action_horizon=self.action_horizon,
                    reuse_media_cache=self.reuse_media_cache,
                )
            )
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
    ) -> dict[str, Any]:
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        source_split = normalize_source_split(split)
        target_split = target_split_name(source_split)
        episodes = self.load_episodes(source_root, split=source_split, max_episodes=max_episodes)
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
        return write_navvla_lerobot_dataset(
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
            context_index_config=AERIALVLN_CONTEXT_INDEX_CONFIG,
        )


def normalize_source_split(split: str) -> str:
    value = split.strip()
    reverse = {target: source for source, target in SOURCE_SPLIT_TO_TARGET.items()}
    if value in reverse:
        return reverse[value]
    if value in SOURCE_SPLIT_TO_TARGET or value == "test":
        return value
    raise ValueError(f"unsupported AerialVLN split: {split}")


def target_split_name(split: str) -> str:
    source_split = normalize_source_split(split)
    if source_split == "test":
        raise ValueError("AerialVLN test split lacks reference_path/actions and cannot be converted to NavVLA actions")
    return SOURCE_SPLIT_TO_TARGET[source_split]


def resolve_media_cache_root(source_root: Path, *, media_cache_root: str | Path | None) -> Path:
    if media_cache_root is not None:
        return Path(media_cache_root)
    return source_root / ".navvla_media_cache" / "aerialvln"


def build_episode(
    source_episode: dict[str, Any],
    *,
    source_root: Path,
    media_cache_root: Path,
    source_split: str,
    target_split: str,
    task_index: int,
    fps: float,
    action_horizon: int,
    reuse_media_cache: bool,
) -> NavVLAEpisode:
    episode_id = str(required_value(source_episode, "episode_id"))
    trajectory_id = str(required_value(source_episode, "trajectory_id"))
    scene_id = str(required_value(source_episode, "scene_id"))
    instruction = instruction_text(source_episode)
    reference_path = reference_path_poses(source_episode)
    native_actions = list(required_value(source_episode, "actions"))
    if len(native_actions) != len(reference_path):
        raise ValueError(
            f"AerialVLN episode {episode_id} has mismatched reference_path/actions lengths: "
            f"{len(reference_path)} vs {len(native_actions)}"
        )

    task = NavVLATaskSpec(
        task_index=task_index,
        instruction=instruction,
        task_type="navigation",
        task_subtype="aerialvln",
        platform_text=PLATFORM_TEXT,
        dataset_source="aerialvln",
        scene_id=scene_id,
    )
    frames = []
    for frame_index, pose in enumerate(reference_path):
        media_path = materialize_rgb_frame(
            source_root=source_root,
            media_cache_root=media_cache_root,
            source_split=source_split,
            target_split=target_split,
            scene_id=scene_id,
            trajectory_id=trajectory_id,
            frame_index=frame_index,
            reuse_media_cache=reuse_media_cache,
        )
        action = action_chunk_for_frame(reference_path, frame_idx=frame_index, horizon=action_horizon)
        frames.append(
            NavVLAFrame(
                frame_index=frame_index,
                timestamp=float(frame_index) / float(fps),
                media_paths={"front_image": media_path},
                state=pose,
                action=action,
                action_available=bool(action),
                source_frame_index=frame_index,
                source_metadata={
                    "source_dataset": "aerialvln",
                    "source_split": source_split,
                    "target_split": target_split,
                    "scene_id": scene_id,
                    "trajectory_id": trajectory_id,
                    "episode_id": episode_id,
                    "native_action": native_actions[frame_index],
                    "lmdb_path": str(resolve_lmdb_rgb_path(source_root, split=source_split, scene_id=scene_id)),
                    "lmdb_key": lmdb_rgb_key(trajectory_id, frame_index),
                    "source_pose": source_episode["reference_path"][frame_index],
                    "roll": float(source_episode["reference_path"][frame_index][3]),
                    "pitch": float(source_episode["reference_path"][frame_index][4]),
                    "coordinate_frame": "AirSim/Unreal NED-style scene frame; z down is positive",
                    "goals": source_episode.get("goals"),
                },
            )
        )

    return NavVLAEpisode(
        episode_id=episode_id,
        task=task,
        frames=frames,
        cameras=[FRONT_CAMERA],
        split=target_split,
        trajectory_id=trajectory_id,
    )


def required_value(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"AerialVLN episode is missing required field: {key}")
    return payload[key]


def instruction_text(source_episode: dict[str, Any]) -> str:
    instruction = required_value(source_episode, "instruction")
    if isinstance(instruction, dict):
        text = str(instruction.get("instruction_text") or "").strip()
    else:
        text = str(instruction or "").strip()
    if not text:
        raise ValueError("AerialVLN episode has empty instruction.instruction_text")
    return text


def reference_path_poses(source_episode: dict[str, Any]) -> list[list[float]]:
    raw_path = required_value(source_episode, "reference_path")
    if not isinstance(raw_path, list) or not raw_path:
        raise ValueError("AerialVLN episode reference_path must be a non-empty list")
    poses = []
    for index, raw_pose in enumerate(raw_path):
        if not isinstance(raw_pose, list) or len(raw_pose) < 6:
            raise ValueError(f"AerialVLN reference_path[{index}] must contain [x,y,z,roll,pitch,yaw]")
        poses.append([float(raw_pose[0]), float(raw_pose[1]), float(raw_pose[2]), float(raw_pose[5])])
    return poses


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


def resolve_lmdb_rgb_path(source_root: str | Path, *, split: str, scene_id: int | str) -> Path:
    root = Path(source_root)
    source_split = normalize_source_split(split)
    scene = str(scene_id)
    if source_split == "train":
        scene_root = "aerialvln_TF" if scene == "3" else f"aerialvln_TF_s{scene}"
        return root / "collect" / scene_root / "train_rgb"
    if source_split == "val_seen":
        return root / "collect" / f"aerialvln_TFvs_s{scene}" / "val_seen_rgb"
    if source_split == "val_unseen":
        return root / "collect" / f"aerialvln_TFvu_s{scene}" / "val_unseen_rgb"
    raise ValueError(f"AerialVLN split has no RGB LMDB: {split}")


def lmdb_rgb_key(trajectory_id: str, frame_index: int) -> str:
    return f"{trajectory_id}_{int(frame_index)}_rgb"


def materialize_rgb_frame(
    *,
    source_root: Path,
    media_cache_root: Path,
    source_split: str,
    target_split: str,
    scene_id: str,
    trajectory_id: str,
    frame_index: int,
    reuse_media_cache: bool,
) -> Path:
    cache_path = media_cache_root / target_split / trajectory_id / f"{frame_index:06d}.png"
    if reuse_media_cache:
        if not cache_path.exists() or cache_path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"missing cached AerialVLN RGB image for {trajectory_id} frame {frame_index}: {cache_path}"
            )
        return cache_path
    raw = read_lmdb_rgb_payload(
        resolve_lmdb_rgb_path(source_root, split=source_split, scene_id=scene_id),
        key=lmdb_rgb_key(trajectory_id, frame_index),
    )
    array = decode_lmdb_rgb_array(raw)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(cache_path)
    return cache_path


def read_lmdb_rgb_payload(lmdb_path: Path, *, key: str) -> bytes:
    lmdb_module, _msgpack_numpy = ensure_lmdb_dependencies()
    if not lmdb_path.exists():
        raise FileNotFoundError(f"AerialVLN RGB LMDB not found: {lmdb_path}")
    env = lmdb_module.open(str(lmdb_path), readonly=True, lock=False, readahead=False, max_readers=1)
    try:
        with env.begin(write=False) as txn:
            value = txn.get(key.encode("utf-8"))
    finally:
        env.close()
    if value is None:
        raise KeyError(f"AerialVLN RGB LMDB key not found: {lmdb_path}:{key}")
    return value


def ensure_lmdb_dependencies() -> tuple[Any, Any]:
    try:
        import lmdb
        import msgpack_numpy
    except ImportError as exc:
        raise ImportError(
            "AerialVLN RGB decoding requires lmdb>=1.4.1 and msgpack-numpy==0.4.8. "
            "Install the project dependencies before converting AerialVLN."
        ) from exc
    return lmdb, msgpack_numpy


def decode_lmdb_rgb_array(payload: bytes) -> np.ndarray:
    try:
        import msgpack
    except ImportError as exc:
        raise ImportError("AerialVLN RGB decoding requires msgpack>=1.1.") from exc
    try:
        _lmdb, msgpack_numpy = ensure_lmdb_dependencies()
        array = msgpack_numpy.unpackb(payload, raw=False)
        return validate_rgb_array(array)
    except (ImportError, TypeError, ValueError):
        pass

    unpacker = msgpack.Unpacker(raw=True, strict_map_key=False)
    unpacker.feed(payload)
    header = unpacker.unpack()
    if not isinstance(header, dict):
        raise ValueError("AerialVLN RGB LMDB payload is not a msgpack-numpy dict")
    shape = header.get(b"shape")
    dtype = header.get(b"type")
    data = header.get(b"data")
    if shape is None or dtype is None or data is None:
        raise ValueError("AerialVLN RGB LMDB payload is missing shape/type/data")
    if isinstance(dtype, bytes):
        dtype = dtype.decode("ascii")
    expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    remaining = payload[unpacker.tell() :]
    raw_data = bytes(data) + remaining
    if len(raw_data) < expected_bytes:
        raise ValueError(
            f"AerialVLN RGB LMDB payload is truncated: expected {expected_bytes} data bytes, got {len(raw_data)}"
        )
    array = np.frombuffer(raw_data[:expected_bytes], dtype=np.dtype(dtype)).reshape(tuple(int(dim) for dim in shape))
    return validate_rgb_array(array)


def validate_rgb_array(array: Any) -> np.ndarray:
    rgb = np.asarray(array)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"AerialVLN RGB array must have shape [H,W,3], got {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return np.ascontiguousarray(rgb)


register_adapter(AerialVLNAdapter())
