from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from navvla_conversion.adapters.base import NavVLASourceAdapter, register_adapter
from navvla_conversion.context_index import ContextIndexConfig
from navvla_conversion.lerobot_v3_writer import write_navvla_lerobot_dataset
from navvla_conversion.schema import NavVLACameraSpec, NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec
from navvla_conversion.statistics import body_frame_action_from_pose, wrap_to_pi


SOURCE_SPLIT_TO_TARGET = {
    "train": "vln_train",
    "vln_train": "vln_train",
    "test_seen": "vln_val_seen",
    "val_seen": "vln_val_seen",
    "vln_val_seen": "vln_val_seen",
    "test_unseen": "vln_val_unseen",
    "val_unseen": "vln_val_unseen",
    "vln_val_unseen": "vln_val_unseen",
}
TARGET_SPLIT_TO_SOURCE = {
    "vln_train": "train",
    "vln_val_seen": "test_seen",
    "vln_val_unseen": "test_unseen",
}
INDOORUAV_CAMERA = NavVLACameraSpec(
    name="front",
    video_key="front_image",
    viewpoint_type="front",
    azimuth_rad=0.0,
    calibration_status="unknown",
)
INDOORUAV_CONTEXT_INDEX_CONFIG = ContextIndexConfig(budget_num_cameras=1, history_camera_names=("front",))
PLATFORM_TEXT = "Platform: UAV. Task: instruction-conditioned indoor navigation. Action: local 3D waypoints (dx, dy, dz, dyaw)."
STATE_MODE = "indooruav_world_pose_xy_zdown_yaw_minus_pi_over_2"
YAW_OFFSET_RAD = -math.pi / 2.0
INSTRUCTION_RE = re.compile(r'"instruction"\s*:\s*"(.*?)"', re.DOTALL)


class IndoorUAVAdapter(NavVLASourceAdapter):
    name = "indooruav"

    def __init__(
        self,
        *,
        extracted_root: str | Path | None = None,
        fps: float = 10.0,
        action_horizon: int = 8,
    ) -> None:
        self.extracted_root = Path(extracted_root) if extracted_root is not None else None
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.summary: dict[str, Any] = {}

    def configure(
        self,
        *,
        extracted_root: str | Path | None = None,
        fps: float = 10.0,
        action_horizon: int = 8,
        **kwargs: Any,
    ) -> "IndoorUAVAdapter":
        super().configure(**kwargs)
        self.extracted_root = Path(extracted_root) if extracted_root is not None else None
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        return self

    def load_episodes(
        self,
        source_root: str | Path,
        *,
        split: str = "train",
        max_episodes: int | None = None,
        load_workers: int | None = None,
    ) -> list[NavVLAEpisode]:
        del load_workers
        source_root = Path(source_root)
        extracted_root = self._require_extracted_root()
        source_split = normalize_source_split(split)
        target_split = target_split_name(split)
        rows = read_split_rows(source_root, source_split)
        if max_episodes is not None:
            rows = rows[: int(max_episodes)]
        if not rows:
            raise FileNotFoundError(f"no IndoorUAV rows found for split={source_split} under {source_root}")

        source_groups = {trajectory_parts(row["traj_path"])[0] for row in rows}
        group_roots = find_group_roots(extracted_root, source_groups)
        episodes = []
        for task_index, row in enumerate(rows):
            source_traj_path = normalize_traj_path(row["traj_path"])
            trajectory_root = resolve_trajectory_root(source_traj_path, group_roots=group_roots)
            episode = build_episode(
                trajectory_root=trajectory_root,
                source_traj_path=source_traj_path,
                source_split=source_split,
                target_split=target_split,
                difficulty=str(row.get("difficulty") or ""),
                task_index=task_index,
                fps=self.fps,
                action_horizon=self.action_horizon,
            )
            episodes.append(episode)

        self.summary = {
            "dataset": "indooruav",
            "source_split": source_split,
            "target_split": target_split,
            "episode_count": len(episodes),
            "frame_count": sum(len(episode.frames) for episode in episodes),
            "extracted_root": str(extracted_root),
        }
        return renumber_episode_task_indices(episodes)

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
        episodes = self.load_episodes(source_root, split=split, max_episodes=max_episodes, load_workers=load_workers)
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
            state_mode=STATE_MODE,
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
            context_index_config=INDOORUAV_CONTEXT_INDEX_CONFIG,
        )
        summary["adapter_summary"] = self.summary
        return summary

    def _require_extracted_root(self) -> Path:
        if self.extracted_root is None:
            raise ValueError("IndoorUAVAdapter requires extracted_root pointing to extracted IndoorUAV trajectory folders")
        if not self.extracted_root.exists():
            raise FileNotFoundError(f"IndoorUAV extracted root not found: {self.extracted_root}")
        return self.extracted_root


def normalize_source_split(split: str) -> str:
    value = split.strip()
    if value in TARGET_SPLIT_TO_SOURCE:
        return TARGET_SPLIT_TO_SOURCE[value]
    if value in SOURCE_SPLIT_TO_TARGET:
        return value
    raise ValueError(f"unsupported IndoorUAV split: {split}")


def target_split_name(split: str) -> str:
    source_split = normalize_source_split(split)
    return SOURCE_SPLIT_TO_TARGET[source_split]


def read_split_rows(source_root: Path, source_split: str) -> list[dict[str, str]]:
    csv_path = source_root / f"{source_split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"IndoorUAV split CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for index, row in enumerate(rows):
        if not str(row.get("traj_path") or "").strip():
            raise ValueError(f"{csv_path} row {index} is missing traj_path")
    return rows


def normalize_traj_path(value: str) -> str:
    normalized = str(value).strip().lstrip("/").rstrip("/")
    parts = normalized.split("/")
    if len(parts) != 3:
        raise ValueError(f"IndoorUAV traj_path must be scene_group/scene_id/traj_id, got {value!r}")
    return normalized


def trajectory_parts(value: str) -> tuple[str, str, str]:
    normalized = normalize_traj_path(value)
    group, scene_id, trajectory_id = normalized.split("/")
    return group, scene_id, trajectory_id


def find_group_roots(extracted_root: Path, groups: set[str]) -> dict[str, list[Path]]:
    roots: dict[str, list[Path]] = {group: [] for group in groups}
    for group in groups:
        direct = extracted_root / group
        if direct.is_dir():
            roots[group].append(direct)

    for dirpath, dirnames, _filenames in os.walk(extracted_root):
        dirnames[:] = [name for name in dirnames if name not in {"screenshots", "saved_transformations", "__MACOSX"}]
        found = []
        for dirname in list(dirnames):
            if dirname in groups:
                candidate = Path(dirpath) / dirname
                if candidate not in roots[dirname]:
                    roots[dirname].append(candidate)
                found.append(dirname)
        if found:
            dirnames[:] = [name for name in dirnames if name not in found]
    return roots


def resolve_trajectory_root(source_traj_path: str, *, group_roots: dict[str, list[Path]]) -> Path:
    group, scene_id, trajectory_id = trajectory_parts(source_traj_path)
    for group_root in group_roots.get(group, []):
        candidate = group_root / scene_id / trajectory_id
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"IndoorUAV trajectory not found in extracted root: {source_traj_path}")


def build_episode(
    *,
    trajectory_root: Path,
    source_traj_path: str,
    source_split: str,
    target_split: str,
    difficulty: str,
    task_index: int,
    fps: float,
    action_horizon: int,
) -> NavVLAEpisode:
    group, scene_id, trajectory_id = trajectory_parts(source_traj_path)
    posture = read_json_file(trajectory_root / "posture.json")
    raw_actions = read_json_file(trajectory_root / "real_action.json")
    instruction = instruction_text(trajectory_root)
    if not isinstance(posture, list) or not posture:
        raise ValueError(f"IndoorUAV posture.json must contain a non-empty list: {trajectory_root}")
    action_rows = raw_actions.get("frame") if isinstance(raw_actions, dict) else None
    if not isinstance(action_rows, list) or not action_rows:
        raise ValueError(f"IndoorUAV real_action.json must contain a non-empty frame list: {trajectory_root}")
    if len(posture) != len(action_rows):
        raise ValueError(f"IndoorUAV pose/action length mismatch for {source_traj_path}: {len(posture)} vs {len(action_rows)}")

    poses = [navvla_pose_from_source_pose(raw_pose) for raw_pose in posture]
    source_frame_ids = [int(row.get("frame")) for row in action_rows]
    expected_frame_ids = list(range(1, len(action_rows) + 1))
    if source_frame_ids != expected_frame_ids:
        raise ValueError(f"IndoorUAV source frame ids must be contiguous 1-based for {source_traj_path}")
    image_frame_ids = image_frame_ids_for_trajectory(
        trajectory_root / "screenshots",
        expected_frame_ids=expected_frame_ids,
        source_traj_path=source_traj_path,
    )

    task = NavVLATaskSpec(
        task_index=task_index,
        instruction=instruction,
        task_type="navigation",
        task_subtype="indooruav",
        platform_text=PLATFORM_TEXT,
        dataset_source="indooruav",
        scene_id=f"{group}/{scene_id}",
    )
    frames = []
    for frame_index, source_frame_id in enumerate(image_frame_ids):
        source_pos = source_frame_id - 1
        pose = poses[source_pos]
        raw_pose = posture[source_pos]
        action_row = action_rows[source_pos]
        image_path = trajectory_root / "screenshots" / f"{source_frame_id}.png"
        if not image_path.exists() or image_path.stat().st_size <= 0:
            raise FileNotFoundError(f"IndoorUAV RGB frame not found or empty: {image_path}")
        action = action_chunk_for_frame(poses, frame_idx=source_pos, horizon=action_horizon)
        frames.append(
            NavVLAFrame(
                frame_index=frame_index,
                timestamp=float(frame_index) / float(fps),
                media_paths={"front_image": image_path},
                state=pose,
                action=action,
                action_available=bool(action),
                source_frame_index=source_frame_id,
                source_metadata={
                    "source_dataset": "indooruav",
                    "source_split": source_split,
                    "target_split": target_split,
                    "scene_group": group,
                    "scene_id": scene_id,
                    "trajectory_id": trajectory_id,
                    "difficulty": difficulty,
                    "source_traj_path": source_traj_path,
                    "source_frame_id": source_frame_id,
                    "native_action": dict(action_row.get("actions") or {}),
                    "raw_yaw_deg": float(raw_pose[3]),
                    "yaw_offset_rad": YAW_OFFSET_RAD,
                    "source_image_path": str(image_path),
                    "converted_coordinate_frame": "x_forward_y_right_z_down_yaw_right_positive",
                    "source_image_frame_count": len(image_frame_ids),
                    "source_pose_action_frame_count": len(action_rows),
                },
            )
        )

    return NavVLAEpisode(
        episode_id=f"{int(task_index):05d}",
        task=task,
        frames=frames,
        cameras=[INDOORUAV_CAMERA],
        split=target_split,
        trajectory_id=source_traj_path,
    )


def image_frame_ids_for_trajectory(screenshot_root: Path, *, expected_frame_ids: list[int], source_traj_path: str) -> list[int]:
    if not screenshot_root.is_dir():
        raise FileNotFoundError(f"IndoorUAV screenshots directory not found: {screenshot_root}")
    image_frame_ids = []
    for path in screenshot_root.glob("*.png"):
        try:
            image_frame_ids.append(int(path.stem))
        except ValueError as exc:
            raise ValueError(f"IndoorUAV screenshot filename must be numeric for {source_traj_path}: {path.name}") from exc
    image_frame_ids = sorted(image_frame_ids)
    if not image_frame_ids:
        raise FileNotFoundError(f"IndoorUAV trajectory has no RGB screenshots: {source_traj_path}")
    expected_set = set(expected_frame_ids)
    image_set = set(image_frame_ids)
    extra = sorted(image_set - expected_set)
    if extra:
        raise ValueError(f"IndoorUAV screenshots exceed pose/action frames for {source_traj_path}: {extra[:10]}")
    expected_prefix = list(range(1, image_frame_ids[-1] + 1))
    if image_frame_ids != expected_prefix:
        missing_inside = sorted(set(expected_prefix) - image_set)
        raise FileNotFoundError(f"IndoorUAV non-tail missing RGB frames for {source_traj_path}: {missing_inside[:10]}")
    return image_frame_ids


def read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def instruction_text(trajectory_root: Path) -> str:
    candidates = [trajectory_root / "instruction.json", trajectory_root / "instruction_pro.json"]
    recovered = []
    for path in candidates:
        if not path.exists():
            continue
        text = recover_instruction_from_bytes(path.read_bytes())
        if text:
            return text
        recovered.append(str(path))
    raise ValueError(f"IndoorUAV trajectory has no recoverable instruction: {trajectory_root}; checked {recovered}")


def recover_instruction_from_bytes(payload: bytes) -> str:
    for errors in ("strict", "replace"):
        try:
            text = payload.decode("utf-8", errors=errors)
            value = json.loads(text)
            instruction = str(value.get("instruction") or "").strip() if isinstance(value, dict) else ""
            if instruction:
                return normalize_text(instruction)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    text = payload.decode("utf-8", errors="replace")
    match = INSTRUCTION_RE.search(text)
    if not match:
        return ""
    return normalize_text(match.group(1))


def normalize_text(value: str) -> str:
    text = str(value).replace("\ufffd", " ")
    text = re.sub(r"\\+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def navvla_pose_from_source_pose(raw_pose: Any) -> list[float]:
    if not isinstance(raw_pose, list) or len(raw_pose) < 4:
        raise ValueError(f"IndoorUAV posture entry must be [x,y,z,yaw_deg], got {raw_pose!r}")
    yaw = float(wrap_to_pi(math.radians(float(raw_pose[3])) + YAW_OFFSET_RAD))
    return [_clean_float(float(raw_pose[0])), _clean_float(float(raw_pose[1])), _clean_float(-float(raw_pose[2])), _clean_float(yaw)]


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


def renumber_episode_task_indices(episodes: list[NavVLAEpisode]) -> list[NavVLAEpisode]:
    return [
        replace(episode, episode_id=f"{episode_index:05d}", task=replace(episode.task, task_index=episode_index))
        for episode_index, episode in enumerate(episodes)
    ]


register_adapter(IndoorUAVAdapter())
