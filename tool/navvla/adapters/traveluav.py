from __future__ import annotations

import json
import math
from pathlib import Path, PurePath
from typing import Any

from tool.navvla.adapters.base import NavVLASourceAdapter, register_adapter
from tool.navvla.context_index import ContextIndexConfig
from tool.navvla.schema import NavVLACameraSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec
from tool.navvla.statistics import body_frame_action_from_pose


TRAVELUAV_CAMERAS = [
    ("front", "front_image", "front", 0.0),
    ("left", "left_image", "left", math.pi / 2.0),
    ("right", "right_image", "right", -math.pi / 2.0),
    ("rear", "rear_image", "rear", math.pi),
    ("down", "down_image", "down", 0.0),
]

TRAVELUAV_HISTORY_CAMERA_NAMES = ("front", "left", "right", "rear")

TRAVELUAV_CONTEXT_INDEX_CONFIG = ContextIndexConfig(
    budget_num_cameras=len(TRAVELUAV_HISTORY_CAMERA_NAMES),
    history_camera_names=TRAVELUAV_HISTORY_CAMERA_NAMES,
)


class TravelUAVAdapter(NavVLASourceAdapter):
    name = "traveluav"

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {"rejected_episodes": 0, "rejections": []}

    def configure(self, *, fps: float = 0.2, action_horizon: int = 8, **kwargs: Any) -> "TravelUAVAdapter":
        super().configure(**kwargs)
        return self

    def load_episodes(
        self,
        source_root: str | Path,
        *,
        split: str = "train",
        max_episodes: int | None = None,
    ) -> list[NavVLAEpisode]:
        root = Path(source_root)
        episode_paths = episode_paths_for_split(root, split=split)
        if max_episodes is not None:
            episode_paths = episode_paths[:max_episodes]
        if not episode_paths:
            raise FileNotFoundError(f"no TravelUAV episodes found under {root}")

        episodes: list[NavVLAEpisode] = []
        self.summary = {"rejected_episodes": 0, "rejections": []}
        for path in episode_paths:
            try:
                episode = load_episode(path, source_root=root, split=split, task_index=len(episodes))
            except (FileNotFoundError, ValueError, KeyError, IndexError) as exc:
                self.summary["rejected_episodes"] += 1
                self.summary["rejections"].append(
                    {
                        "episode_id": path.name,
                        "scene_id": path.parent.name,
                        "source_path": str(path),
                        "reason": str(exc),
                    }
                )
                continue
            episodes.append(episode)
        if not episodes:
            raise FileNotFoundError(f"no valid TravelUAV episodes found under {root}")
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
        context_index_config: ContextIndexConfig | None = None,
    ) -> dict[str, Any]:
        summary = super().convert(
            source_root=source_root,
            output_root=output_root,
            dataset_name=dataset_name,
            max_episodes=max_episodes,
            fps=fps,
            action_horizon=action_horizon,
            overwrite=overwrite,
            control_frequency_hz=control_frequency_hz,
            repair_existing=repair_existing,
            split=split,
            context_policy_version=context_policy_version,
            cache_policy_version=cache_policy_version,
            cache_workers=cache_workers,
            write_visual_token_cache=write_visual_token_cache,
            visual_token_profile=visual_token_profile,
            visual_token_encoder=visual_token_encoder,
            visual_token_encoder_factory=visual_token_encoder_factory,
            episodes_per_file=episodes_per_file,
            files_per_chunk=files_per_chunk,
            context_index_config=context_index_config or TRAVELUAV_CONTEXT_INDEX_CONFIG,
        )
        summary["adapter_summary"] = self.summary
        return summary


def episode_paths_for_split(source_root: Path, *, split: str) -> list[Path]:
    if is_raw_episode_dir(source_root):
        return [source_root]
    raw_split_paths = raw_dataset_episode_paths_for_split(source_root, split=split)
    if raw_split_paths:
        return raw_split_paths
    episodes_dir = source_root / "episodes"
    split_file = source_root / "splits" / f"{split}.txt"
    if split_file.exists():
        paths = []
        for line in split_file.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = episodes_dir / (value if value.endswith(".json") else f"{value}.json")
            paths.append(path)
        return paths
    if episodes_dir.exists():
        return sorted(episodes_dir.glob("*.json"))
    if source_root.is_file() and source_root.suffix == ".json":
        return [source_root]
    return sorted(source_root.glob("*.json"))


def raw_dataset_episode_paths_for_split(source_root: Path, *, split: str) -> list[Path]:
    split_name = raw_dataset_split_filename(split)
    split_candidates = [
        source_root / "data" / "uav_dataset" / split_name,
        source_root / "uav_dataset" / split_name,
    ]
    split_file = next((candidate for candidate in split_candidates if candidate.exists()), None)
    if split_file is None:
        return []
    records = json.loads(split_file.read_text(encoding="utf-8"))
    paths: list[Path] = []
    seen: set[tuple[str, str]] = set()
    for item in records:
        if not isinstance(item, dict) or "json" not in item:
            continue
        parts = PurePath(str(item["json"]).replace("\\", "/")).parts
        if len(parts) < 2:
            continue
        scene_id, episode_id = str(parts[0]), str(parts[1])
        key = (scene_id, episode_id)
        if key in seen:
            continue
        seen.add(key)
        paths.append(source_root / "dataset" / scene_id / episode_id)
    return paths



def raw_dataset_split_filename(split: str) -> str:
    mapping = {
        "train": "trainset.json",
        "vln_train": "trainset.json",
        "val_seen": "seen_valset.json",
        "seen": "seen_valset.json",
        "vln_val_seen": "seen_valset.json",
        "val_unseen": "unseen_valset.json",
        "unseen": "unseen_valset.json",
        "vln_val_unseen": "unseen_valset.json",
    }
    return mapping.get(str(split), f"{split}.json")



def is_raw_episode_dir(path: Path) -> bool:
    return path.is_dir() and (path / "merged_data.json").exists()


def load_episode(path: Path, *, source_root: Path, split: str, task_index: int) -> NavVLAEpisode:
    if is_raw_episode_dir(path):
        return load_raw_episode_dir(path, split=split, task_index=task_index)
    payload = json.loads(path.read_text(encoding="utf-8"))
    inferred = infer_scene_episode_ids(payload, source_root=source_root)
    explicit_episode_id = payload.get("episode_id")
    explicit_path_ids = infer_scene_episode_ids_from_path(explicit_episode_id, source_root=source_root) if explicit_episode_id else None
    episode_id = normalize_explicit_episode_id(explicit_episode_id) if explicit_episode_id else inferred["episode_id"] or path.stem
    trajectory_id = payload.get("trajectory_id")
    if trajectory_id is not None:
        trajectory_id = normalize_explicit_episode_id(trajectory_id)
    else:
        trajectory_id = episode_id
    instruction = str(payload.get("instruction") or payload.get("task") or payload.get("language") or "navigation task")
    scene_id = str(payload.get("scene_id") or inferred["scene_id"] or (explicit_path_ids[0] if explicit_path_ids else None) or infer_scene_id(payload, source_root=source_root))
    cameras = [
        NavVLACameraSpec(name=name, video_key=video_key, viewpoint_type=viewpoint_type, azimuth_rad=azimuth_rad)
        for name, video_key, viewpoint_type, azimuth_rad in TRAVELUAV_CAMERAS
    ]
    task = NavVLATaskSpec(
        task_index=task_index,
        instruction=instruction,
        task_type=str(payload.get("task_type") or "navigation"),
        task_subtype=str(payload.get("task_subtype") or "uav"),
        platform_text=str(payload.get("platform_text") or default_platform_text()),
        dataset_source="traveluav",
        scene_id=scene_id,
        answer=payload.get("answer"),
    )
    episode_source_metadata = traveluav_source_metadata(payload)
    frames = [load_frame(frame, source_root=source_root, episode_source_metadata=episode_source_metadata) for frame in payload.get("frames", [])]
    return NavVLAEpisode(
        episode_id=episode_id,
        task=task,
        frames=frames,
        cameras=cameras,
        split=split,
        trajectory_id=trajectory_id,
    )


def load_raw_episode_dir(path: Path, *, split: str, task_index: int) -> NavVLAEpisode:
    payload = json.loads((path / "merged_data.json").read_text(encoding="utf-8"))
    object_descriptions = json.loads((path / "object_description.json").read_text(encoding="utf-8")) if (path / "object_description.json").exists() else []
    raw_source_metadata = raw_episode_source_metadata(path, object_descriptions=object_descriptions)
    instruction = instruction_from_raw_payload(payload, object_descriptions=object_descriptions)
    scene_id = path.parent.name
    episode_id = path.name
    cameras = [
        NavVLACameraSpec(name=name, video_key=video_key, viewpoint_type=viewpoint_type, azimuth_rad=azimuth_rad)
        for name, video_key, viewpoint_type, azimuth_rad in TRAVELUAV_CAMERAS
    ]
    task = NavVLATaskSpec(
        task_index=task_index,
        instruction=instruction,
        task_type="navigation",
        task_subtype="uav",
        platform_text=default_platform_text(),
        dataset_source="traveluav",
        scene_id=scene_id,
    )
    sparse_indices = [int(value) for value in payload.get("index") or []]
    raw_trajectory = payload.get("trajectory_raw_detailed") or payload.get("trajectory") or []
    trajectory = [pose4_from_traveluav_state(row) for row in raw_trajectory]
    if not sparse_indices:
        sparse_indices = list(range(len(trajectory)))
    if not trajectory:
        raise ValueError(f"raw TravelUAV episode has no trajectory: {path}")
    frames = []
    timestamps = raw_episode_relative_timestamps(path, sparse_indices=sparse_indices)
    for frame_position, source_index in enumerate(sparse_indices):
        if source_index < 0 or source_index >= len(trajectory):
            raise IndexError(f"TravelUAV source index out of range for {path}: {source_index}")
        media_paths = raw_episode_media_paths(path, source_index=source_index)
        action = action_chunk_for_dense_future_steps(trajectory, source_index=source_index, horizon=8)
        frames.append(
            NavVLAFrame(
                frame_index=frame_position,
                timestamp=timestamps[source_index],
                media_paths=media_paths,
                state=trajectory[source_index],
                action=action,
                action_available=bool(action),
                source_frame_index=source_index,
                source_metadata={
                    **raw_source_metadata,
                    "source_state": _source_state_for_metadata(raw_trajectory[source_index]),
                    "source_index": source_index,
                    "source_episode_dir": str(path),
                    "raw_episode_format": "merged_data_json_directory",
                },
            )
        )
    return NavVLAEpisode(
        episode_id=episode_id,
        task=task,
        frames=frames,
        cameras=cameras,
        split=split,
        trajectory_id=episode_id,
    )


def raw_episode_source_metadata(path: Path, *, object_descriptions: list[Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    mark_path = path / "mark.json"
    mark = json.loads(mark_path.read_text(encoding="utf-8")) if mark_path.exists() else {}
    if isinstance(mark, dict):
        object_name = str(mark.get("object_name") or "").strip()
        if object_name:
            metadata["object_name"] = object_name
            object_desc = global_object_description(path).get(canonical_object_description_name(object_name))
            if object_desc:
                metadata["object_desc"] = object_desc
        target = mark.get("target")
        if target is not None:
            metadata["target"] = target
    description = first_object_description_text(object_descriptions)
    if description:
        metadata["object_description"] = description
    return metadata


def first_object_description_text(object_descriptions: list[Any]) -> str | None:
    for item in object_descriptions:
        if isinstance(item, str):
            text = item.strip()
            if text:
                return text
        elif isinstance(item, dict):
            text = str(item.get("object_desc") or item.get("description") or "").strip()
            if text:
                return text
    return None


def global_object_description(path: Path) -> dict[str, str]:
    candidates = [
        traveluav_dataset_root(path) / "data" / "meta" / "object_description.json",
        traveluav_dataset_root(path) / "meta" / "object_description.json",
    ]
    meta_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if meta_path is None:
        return {}
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    mapping: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("object_name") or "").strip()
        desc = str(item.get("object_desc") or "").strip()
        if name and desc:
            mapping[name] = desc
    return mapping


def traveluav_dataset_root(path: Path) -> Path:
    parts = path.resolve().parts
    if "dataset" in parts:
        dataset_index = len(parts) - 1 - list(reversed(parts)).index("dataset")
        return Path(*parts[:dataset_index]) if dataset_index > 0 else Path(parts[0])
    return path.parent.parent


def canonical_object_description_name(object_name: str) -> str:
    if object_name.startswith("AASM_"):
        return "SM_" + object_name.removeprefix("AASM_")
    return object_name


def instruction_from_raw_payload(payload: dict[str, Any], *, object_descriptions: list[Any]) -> str:
    conversations = payload.get("conversations") or []
    for turn in conversations:
        if turn.get("from") == "human" and turn.get("value"):
            return str(turn["value"]).replace("<image>\n", "").strip()
    if object_descriptions:
        return f"Please control the drone and find the target. Target description: {object_descriptions[0]}"
    return "Please control the drone and find the target."


def raw_episode_relative_timestamps(path: Path, *, sparse_indices: list[int]) -> dict[int, float]:
    absolute_seconds: dict[int, float] = {}
    for source_index in sparse_indices:
        timestamp = raw_episode_log_timestamp_seconds(path, source_index=source_index)
        if timestamp is not None:
            absolute_seconds[source_index] = timestamp
    if len(absolute_seconds) == len(sparse_indices) and sparse_indices:
        first = absolute_seconds[sparse_indices[0]]
        return {source_index: _clean_float(absolute_seconds[source_index] - first) for source_index in sparse_indices}
    return {source_index: float(source_index) for source_index in sparse_indices}


def raw_episode_log_timestamp_seconds(path: Path, *, source_index: int) -> float | None:
    log_path = path / "log" / f"{source_index:06d}.json"
    if not log_path.exists():
        return None
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    timestamp = payload.get("timestamp")
    if timestamp is None:
        return None
    return float(timestamp) / 1_000_000_000.0


def raw_episode_media_paths(path: Path, *, source_index: int) -> dict[str, Path]:
    dirname_by_key = {
        "front_image": "frontcamera",
        "left_image": "leftcamera",
        "right_image": "rightcamera",
        "rear_image": "rearcamera",
        "down_image": "downcamera",
    }
    media_paths = {}
    for video_key, dirname in dirname_by_key.items():
        image_path = path / dirname / f"{source_index:06d}.png"
        if image_path.exists():
            media_paths[video_key] = image_path
    if not media_paths:
        raise FileNotFoundError(f"no RGB camera images found for TravelUAV frame {source_index} under {path}")
    return media_paths


def pose4_from_traveluav_state(raw_state: Any) -> list[float]:
    if isinstance(raw_state, dict):
        position = raw_state.get("position") or []
        orientation = raw_state.get("orientation") or []
        if len(position) < 3 or len(orientation) < 4:
            raise ValueError("TravelUAV raw pose dict must contain position[3] and orientation quaternion[4]")
        yaw = math.atan2(
            2.0 * (float(orientation[3]) * float(orientation[2]) + float(orientation[0]) * float(orientation[1])),
            1.0 - 2.0 * (float(orientation[1]) ** 2 + float(orientation[2]) ** 2),
        )
        return [float(position[0]), float(position[1]), float(position[2]), yaw]
    if len(raw_state) < 4:
        raise ValueError(f"TravelUAV raw state must contain at least [x,y,z,yaw], got length {len(raw_state)}")
    yaw_index = 5 if len(raw_state) > 5 else 3
    return [float(raw_state[0]), float(raw_state[1]), float(raw_state[2]), float(raw_state[yaw_index])]


def _source_state_for_metadata(raw_state: Any) -> list[float]:
    if isinstance(raw_state, dict):
        return pose4_from_traveluav_state(raw_state)
    pose4_from_traveluav_state(raw_state)
    return [float(value) for value in raw_state]


def action_chunk_for_dense_future_steps(poses: list[list[float]], *, source_index: int, horizon: int) -> list[list[float]]:
    current = poses[source_index]
    chunk = []
    for future_source_index in range(source_index + 1, min(len(poses), source_index + 1 + horizon)):
        action = [_clean_float(value) for value in body_frame_action_from_pose(current, poses[future_source_index]).astype(float).tolist()]
        chunk.append(action)
    return chunk


def _clean_float(value: float) -> float:
    value = float(value)
    return 0.0 if abs(value) < 1e-7 else value


def normalize_explicit_episode_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("TravelUAV episode_id must be non-empty")
    if "/" in text or "\\" in text:
        return PurePath(text.replace("\\", "/")).name
    return text


def infer_scene_episode_ids_from_path(value: Any, *, source_root: Path | None = None) -> tuple[str, str] | None:
    if value is None:
        return None
    return _infer_scene_episode_from_path(Path(str(value).replace("\\", "/")), source_root=source_root)


def infer_scene_episode_ids(payload: dict[str, Any], *, source_root: Path) -> dict[str, str | None]:
    for frame in payload.get("frames", []):
        for value in _scene_candidate_paths(frame):
            inferred = _infer_scene_episode_from_path(Path(value), source_root=source_root)
            if inferred is not None:
                return {"scene_id": inferred[0], "episode_id": inferred[1]}
    return {"scene_id": None, "episode_id": None}


def traveluav_source_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("object", "object_info", "objects", "target_object", "target"):
        if key in payload and payload[key] is not None:
            metadata[key] = payload[key]
    return metadata


def load_frame(frame: dict[str, Any], *, source_root: Path, episode_source_metadata: dict[str, Any] | None = None) -> NavVLAFrame:
    frame_index = int(frame.get("frame_idx", frame.get("frame_index", 0)))
    media_paths = rgb_media_paths(frame, source_root=source_root)
    state = state_4d(frame)
    source_metadata = dict(episode_source_metadata or {})
    for key in ("object", "object_info", "objects", "target_object", "target"):
        if key in frame and frame[key] is not None:
            source_metadata[key] = frame[key]
    source_timestamp = frame.get("timestamp", frame.get("time"))
    timestamp = float(source_timestamp) if source_timestamp is not None else None
    action = frame.get("action")
    action_available = action is not None
    return NavVLAFrame(
        frame_index=frame_index,
        timestamp=timestamp,
        media_paths=media_paths,
        state=state,
        action=action,
        action_available=action_available,
        source_frame_index=frame.get("source_frame_idx", frame.get("source_frame_index")),
        source_metadata=source_metadata,
    )


def rgb_media_paths(frame: dict[str, Any], *, source_root: Path) -> dict[str, Path]:
    source_media = frame.get("media_paths") or {}
    image_relpaths = frame.get("image_relpaths") or []
    by_position = {
        "front_image": image_relpaths[0] if len(image_relpaths) > 0 else None,
        "left_image": image_relpaths[1] if len(image_relpaths) > 1 else None,
        "right_image": image_relpaths[2] if len(image_relpaths) > 2 else None,
        "rear_image": image_relpaths[3] if len(image_relpaths) > 3 else None,
        "down_image": image_relpaths[4] if len(image_relpaths) > 4 else None,
    }
    media_paths: dict[str, Path] = {}
    for _name, video_key, _viewpoint, _azimuth in TRAVELUAV_CAMERAS:
        value = source_media.get(video_key) or by_position.get(video_key)
        if value is None:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = source_root / path
        media_paths[video_key] = path
    if not media_paths:
        image_relpath = frame.get("image_relpath")
        if image_relpath:
            path = Path(image_relpath)
            media_paths["front_image"] = path if path.is_absolute() else source_root / path
    return media_paths


def state_4d(frame: dict[str, Any]) -> list[float]:
    raw_state = list(frame.get("state") or [])
    if len(raw_state) < 4:
        raise ValueError(f"TravelUAV state must contain at least [x, y, z, yaw] or [x, y, z, roll, pitch, yaw], got {len(raw_state)}")
    x = float(raw_state[0])
    y = float(raw_state[1])
    z = float(raw_state[2])
    yaw = float(raw_state[5] if len(raw_state) > 5 else raw_state[3])
    return [x, y, z, yaw]


def infer_scene_id(payload: dict[str, Any], *, source_root: Path) -> str:
    inferred = infer_scene_episode_ids(payload, source_root=source_root)
    if inferred["scene_id"]:
        return inferred["scene_id"]
    return source_root.name


def _infer_scene_episode_from_path(path: Path, *, source_root: Path | None = None) -> tuple[str, str] | None:
    normalized = Path(str(path).replace("\\", "/"))
    if source_root is not None:
        try:
            relative = normalized.relative_to(source_root)
            if len(relative.parts) >= 2:
                return relative.parts[0], relative.parts[1]
        except ValueError:
            pass
    inferred = _scene_episode_from_dataset_segment(normalized)
    if inferred is not None:
        return inferred
    parts = normalized.parts
    if len(parts) >= 3 and not normalized.is_absolute():
        return parts[0], parts[1]
    return None


def _scene_episode_from_dataset_segment(path: Path) -> tuple[str, str] | None:
    parts = path.parts
    for idx in range(len(parts) - 3, -1, -1):
        if parts[idx] == "dataset" and idx + 2 < len(parts):
            return parts[idx + 1], parts[idx + 2]
    return None


def _scene_from_dataset_segment(path: Path) -> str | None:
    parts = path.parts
    for idx in range(len(parts) - 2, -1, -1):
        part = parts[idx]
        if part == "dataset" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _scene_candidate_paths(frame: dict[str, Any]) -> list[str]:
    values: list[str] = []
    media_paths = frame.get("media_paths") or {}
    for key in ["front_image", "left_image", "right_image", "rear_image", "down_image"]:
        value = media_paths.get(key)
        if value:
            values.append(str(value))
    image_relpath = frame.get("image_relpath")
    if image_relpath:
        values.append(str(image_relpath))
    for value in frame.get("image_relpaths") or []:
        values.append(str(value))
    return values


def default_platform_text() -> str:
    return "Platform: UAV. Task: urban navigation. Action: local 3D waypoints (dx, dy, dz, dyaw)."


register_adapter(TravelUAVAdapter())
