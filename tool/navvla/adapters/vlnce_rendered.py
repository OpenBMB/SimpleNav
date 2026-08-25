from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tool.navvla.adapters.base import NavVLASourceAdapter, register_adapter
from tool.navvla.context_index import ContextIndexConfig
from tool.navvla.lerobot_v3_writer import write_navvla_lerobot_dataset
from tool.navvla.schema import NavVLACameraSpec, NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec
from tool.navvla.statistics import body_frame_action_from_pose


FRONT_CAMERA = NavVLACameraSpec(name="front", video_key="front_image", viewpoint_type="front", azimuth_rad=0.0)
PLATFORM_TEXT = (
    "Platform: Habitat MP3D agent. Task: instruction-conditioned VLN navigation. "
    "Action: local 3D waypoints (dx, dy, dz, dyaw)."
)
VLNCE_CONTEXT_INDEX_CONFIG = ContextIndexConfig(budget_num_cameras=1, history_camera_names=("front",))


class VLNCERenderedAdapter(NavVLASourceAdapter):
    name = "vlnce_rendered"

    def __init__(
        self,
        *,
        fps: float = 1.0,
        action_horizon: int = 8,
    ) -> None:
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)

    def configure(
        self,
        *,
        fps: float = 1.0,
        action_horizon: int = 8,
        **kwargs: Any,
    ) -> "VLNCERenderedAdapter":
        super().configure(**kwargs)
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
        return load_vlnce_rendered_episodes(
            source_root,
            split=split,
            max_episodes=max_episodes,
            fps=self.fps,
            action_horizon=self.action_horizon,
        )

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
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        episodes = self.load_episodes(source_root, split=split, max_episodes=max_episodes)
        target_split = episodes[0].split if episodes else split
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
            cache_workers=cache_workers,
            write_visual_token_cache=write_visual_token_cache,
            visual_token_profile=visual_token_profile,
            visual_token_encoder=visual_token_encoder,
            visual_token_encoder_factory=visual_token_encoder_factory,
            context_index_config=VLNCE_CONTEXT_INDEX_CONFIG,
        )


def load_vlnce_rendered_episodes(
    source_root: str | Path,
    *,
    split: str | None = None,
    max_episodes: int | None = None,
    fps: float = 1.0,
    action_horizon: int = 8,
) -> list[NavVLAEpisode]:
    manifest_path = resolve_manifest_path(source_root)
    rows = list(iter_manifest_rows(manifest_path))
    if not rows:
        raise FileNotFoundError(f"no VLN-CE rendered rows found in {manifest_path}")

    grouped: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        row_target_split = row_target_split_name(row)
        if split and split not in {"*", "all"} and split not in {str(row.get("split")), row_target_split}:
            continue
        key = (row_target_split, str(required_value(row, "episode_id")))
        grouped.setdefault(key, []).append(row)

    episodes: list[NavVLAEpisode] = []
    for task_index, ((target_split, raw_episode_id), episode_rows) in enumerate(grouped.items()):
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
        episodes.append(
            build_episode(
                episode_rows,
                manifest_path=manifest_path,
                target_split=target_split,
                raw_episode_id=raw_episode_id,
                task_index=task_index,
                fps=fps,
                action_horizon=action_horizon,
            )
        )

    if not episodes:
        raise FileNotFoundError(f"no VLN-CE rendered episodes matched split={split!r} in {manifest_path}")
    return episodes


def resolve_manifest_path(source_root: str | Path) -> Path:
    path = Path(source_root)
    if path.is_file():
        return path
    manifest_path = path / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"VLN-CE rendered manifest not found: {manifest_path}")
    return manifest_path


def iter_manifest_rows(manifest_path: Path) -> Iterable[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {manifest_path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{manifest_path}:{line_number} must contain a JSON object")
            yield row


def build_episode(
    episode_rows: list[dict[str, Any]],
    *,
    manifest_path: Path,
    target_split: str,
    raw_episode_id: str,
    task_index: int,
    fps: float,
    action_horizon: int,
) -> NavVLAEpisode:
    rows = sorted(episode_rows, key=lambda row: int(required_value(row, "frame_index")))
    _validate_frame_indices(rows, raw_episode_id=raw_episode_id)
    first = rows[0]
    family = dataset_family(first)
    role = optional_text(first.get("role"))
    instruction = instruction_text(first)
    scene_id = str(required_value(first, "scene_id"))
    trajectory_id = str(first.get("trajectory_id") or raw_episode_id)
    task = NavVLATaskSpec(
        task_index=task_index,
        instruction=instruction,
        task_type="navigation",
        task_subtype=task_subtype(family, role),
        platform_text=PLATFORM_TEXT,
        dataset_source=f"vlnce_{family}",
        scene_id=scene_id,
    )

    poses = [navvla_pose_from_row(row) for row in rows]
    frames = []
    for row, pose in zip(rows, poses):
        frame_index = int(required_value(row, "frame_index"))
        rgb_path = resolve_rgb_path(row, manifest_path=manifest_path)
        action = action_chunk_for_frame(poses, frame_idx=frame_index, horizon=action_horizon)
        frames.append(
            NavVLAFrame(
                frame_index=frame_index,
                timestamp=timestamp_for_row(row, frame_index=frame_index, fps=fps),
                media_paths={"front_image": rgb_path},
                state=pose,
                action=action,
                action_available=bool(action),
                source_frame_index=frame_index,
                source_metadata=source_metadata(row, manifest_path=manifest_path, navvla_state=pose),
            )
        )

    return NavVLAEpisode(
        episode_id=f"{target_split}_{raw_episode_id}",
        task=task,
        frames=frames,
        cameras=[FRONT_CAMERA],
        split=target_split,
        trajectory_id=trajectory_id,
    )


def required_value(row: dict[str, Any], key: str) -> Any:
    if key not in row:
        raise ValueError(f"VLN-CE rendered manifest row is missing required field: {key}")
    return row[key]


def optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def dataset_family(row: dict[str, Any]) -> str:
    family = str(required_value(row, "dataset_family")).strip().lower()
    if family not in {"r2r", "rxr"}:
        raise ValueError(f"unsupported VLN-CE dataset_family: {family}")
    return family


def row_target_split_name(row: dict[str, Any]) -> str:
    family = dataset_family(row)
    source_split = str(required_value(row, "split")).strip()
    role = optional_text(row.get("role"))
    if family == "r2r":
        return f"r2r_{source_split}"
    if not role:
        raise ValueError("RxR rendered rows must include role")
    return f"rxr_{source_split}_{role}"


def task_subtype(family: str, role: str | None) -> str:
    if family == "rxr" and role:
        return f"rxr_{role}"
    return family


def instruction_text(row: dict[str, Any]) -> str:
    value = row.get("instruction_text")
    if isinstance(value, str) and value.strip():
        return value.strip()
    instruction = row.get("instruction")
    if isinstance(instruction, dict):
        for key in ("instruction_text", "text", "instruction"):
            value = instruction.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()
    raise ValueError("VLN-CE rendered episode has empty instruction_text")


def resolve_rgb_path(row: dict[str, Any], *, manifest_path: Path) -> Path:
    raw_path = Path(str(required_value(row, "rgb_path")))
    path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing rendered RGB for episode {row.get('episode_id')} frame {row.get('frame_index')}: {path}")
    return path


def habitat_position_from_row(row: dict[str, Any]) -> list[float]:
    raw = row.get("position", row.get("agent_position", row.get("source_location")))
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError("VLN-CE rendered row must include position/agent_position/source_location with 3 values")
    return [float(raw[0]), float(raw[1]), float(raw[2])]


def navvla_pose_from_row(row: dict[str, Any]) -> list[float]:
    x, habitat_y, z = habitat_position_from_row(row)
    yaw = float(required_value(row, "yaw"))
    return [_clean_float(x), _clean_float(z), _clean_float(habitat_y), _clean_float(yaw)]


def timestamp_for_row(row: dict[str, Any], *, frame_index: int, fps: float) -> float:
    timestamp = row.get("timestamp")
    if timestamp is not None:
        return float(timestamp)
    return float(frame_index) / float(fps)


def action_chunk_for_frame(poses: list[list[float]], *, frame_idx: int, horizon: int) -> list[list[float]]:
    current = poses[frame_idx]
    chunk = []
    for future_idx in range(frame_idx + 1, min(len(poses), frame_idx + 1 + horizon)):
        action = body_frame_action_from_pose(current, poses[future_idx]).astype(float).tolist()
        chunk.append([_clean_float(value) for value in action])
    return chunk


def source_metadata(row: dict[str, Any], *, manifest_path: Path, navvla_state: list[float]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_dataset": f"vlnce_{dataset_family(row)}",
        "source_split": str(required_value(row, "split")),
        "target_split": row_target_split_name(row),
        "scene_id": str(required_value(row, "scene_id")),
        "episode_id": str(required_value(row, "episode_id")),
        "trajectory_id": str(row.get("trajectory_id") or row.get("episode_id")),
        "frame_index": int(required_value(row, "frame_index")),
        "rgb_path": str(resolve_rgb_path(row, manifest_path=manifest_path)),
        "manifest_path": str(manifest_path),
        "habitat_position": habitat_position_from_row(row),
        "navvla_state": list(navvla_state),
        "coordinate_transform": "NavVLA state is [habitat_x, habitat_z, habitat_y, yaw]",
        "rotation_source": row.get("rotation_source"),
    }
    for key in ("role", "language", "native_action", "yaw", "rotation_xyzw", "rotation_wxyz", "source_location"):
        if key in row:
            metadata[key] = row[key]
    return metadata


def _validate_frame_indices(rows: list[dict[str, Any]], *, raw_episode_id: str) -> None:
    indices = [int(required_value(row, "frame_index")) for row in rows]
    expected = list(range(len(indices)))
    if indices != expected:
        raise ValueError(f"VLN-CE rendered episode {raw_episode_id} frame_index values must be contiguous from 0: {indices[:10]}")


def _clean_float(value: float) -> float:
    value = float(value)
    return 0.0 if abs(value) < 1e-7 else value


register_adapter(VLNCERenderedAdapter())
