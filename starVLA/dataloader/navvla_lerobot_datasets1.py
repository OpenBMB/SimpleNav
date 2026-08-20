from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

from tool.navvla.statistics import (
    body_frame_action_from_pose,
    build_dataset_statistics,
    build_repeated_state_statistics,
    flatten_valid_action_steps_from_rows,
    normalize_values,
    read_dataset_statistics,
    write_dataset_statistics,
)
from tool.navvla.visual_token_cache import TokenCacheManifest, read_token_manifest, validate_token_refs


ACTION_TASK_TYPES = {"navigation", "driving", "tracking"}

DEFAULT_IMAGE_SIZE = (256, 256)

NAVVLA_PLATFORM_ACTION_TEXT = (
    "body-frame cumulative waypoints "
    "(x: forward, y: left, z: up, yaw: heading change)"
)

# platform_type: UAV / autonomous vehicle / indoor mobile robot / tracking robot
_PLATFORM_TYPE_BY_TASK_SUBTYPE = {
    "uav": "UAV",
    "ugv": "autonomous vehicle",
    "wheeled": "indoor mobile robot",
    "indoor": "indoor mobile robot",
    "human_following": "tracking robot",
}

_PLATFORM_TYPE_BY_TASK_TYPE = {
    "tracking": "tracking robot",
    "driving": "autonomous vehicle",
}

_PLATFORM_TYPE_BY_DATASET_SOURCE = {
    "traveluav": "UAV",
    "opentrackvla": "tracking robot",
}


def build_platform_text_from_meta(
    task: Mapping[str, Any],
    info: Mapping[str, Any],
) -> str:
    """Build natural-language platform_text from NavVLA task + info metadata.

    Model instruction order: platform_text + blank line + lang + optional [STATE] + [ACTION].
    """
    navvla_info = info.get("navvla", {})
    control_frequency_hz = navvla_info.get("control_frequency_hz", info.get("fps"))
    if control_frequency_hz is None:
        raise KeyError("info must contain navvla.control_frequency_hz or fps for platform_text")

    dataset_source = str(task.get("dataset_source") or "").strip().lower()
    task_label = _resolve_task_label(
        task_type=task.get("task_type"),
        task_subtype=task.get("task_subtype"),
    )
    platform = _resolve_platform_label(
        dataset_source=dataset_source,
        task_type=task.get("task_type"),
        task_subtype=task.get("task_subtype"),
    )
    fps = _format_control_frequency_hz(float(control_frequency_hz))
    action_horizon = int(navvla_info.get("action_horizon", 8))

    return (
        f"The platform is {platform} for {task_label}.\n"
        f"The control frequency is {fps} Hz.\n"
        f"Please predict the next {action_horizon} {NAVVLA_PLATFORM_ACTION_TEXT} "
        f"to execute the following task:"
    )


def build_navvla_instruction(
    lang: str,
    *,
    platform_text: str = "",
    state_text: str = "",
    use_platform_text: bool = False,
    use_state_text: bool = False,
) -> str:
    """Join platform_text, lang, optional state, and [ACTION] with blank lines."""
    parts: list[str] = []
    if use_platform_text:
        text = str(platform_text).strip()
        if text:
            parts.append(text)
    lang_text = str(lang).strip()
    if lang_text:
        parts.append(lang_text)
    if use_state_text and str(state_text).strip():
        parts.append(f"[STATE] {str(state_text).strip()}")
    parts.append("[ACTION]")
    return "\n\n".join(parts).strip()


def _resolve_platform_label(
    *,
    dataset_source: str,
    task_type: Any,
    task_subtype: Any,
) -> str:
    if dataset_source in _PLATFORM_TYPE_BY_DATASET_SOURCE:
        return _PLATFORM_TYPE_BY_DATASET_SOURCE[dataset_source]

    task_subtype_key = _format_meta_label(task_subtype).lower()
    if task_subtype_key in _PLATFORM_TYPE_BY_TASK_SUBTYPE:
        return _PLATFORM_TYPE_BY_TASK_SUBTYPE[task_subtype_key]

    task_type_key = _format_meta_label(task_type).lower()
    if task_type_key in _PLATFORM_TYPE_BY_TASK_TYPE:
        return _PLATFORM_TYPE_BY_TASK_TYPE[task_type_key]

    if dataset_source:
        return _format_meta_label(dataset_source)
    return "navigation agent"


def _resolve_task_label(
    *,
    task_type: Any,
    task_subtype: Any,
) -> str:
    parts = [
        _format_meta_label(task_type),
        _format_meta_label(task_subtype),
    ]
    label = " ".join(part for part in parts if part)
    return label or "navigation"


def _format_meta_label(value: Any) -> str:
    return str(value or "").strip().replace("_", " ")


def _format_control_frequency_hz(control_frequency_hz: float) -> str:
    fps = float(control_frequency_hz)
    if fps.is_integer():
        return str(int(fps))
    return f"{fps:g}"


class EpisodeGroupedSampler(Sampler[int]):
    def __init__(
        self,
        episode_sample_indices: dict[int, list[int]],
        *,
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        self.episode_sample_indices = {
            int(episode_index): list(indices)
            for episode_index, indices in episode_sample_indices.items()
            if indices
        }
        if not self.episode_sample_indices:
            raise ValueError("episode_sample_indices must contain at least one non-empty episode")
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.episode_sample_indices.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        episode_order = sorted(self.episode_sample_indices)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(episode_order), generator=generator).tolist()
            episode_order = [episode_order[index] for index in order]
        for episode_index in episode_order:
            yield from self.episode_sample_indices[episode_index]


class NavVLACollatedBatch(dict):
    """Dict batch that also supports legacy list-of-samples iteration."""

    def __iter__(self):
        return iter(self.to_samples())

    def to_samples(self) -> list[dict[str, Any]]:
        batch_size = len(self.get("lang", []))
        samples: list[dict[str, Any]] = []
        for batch_index in range(batch_size):
            images = {
                camera: camera_batch[batch_index]
                for camera, camera_batch in self.get("images", {}).items()
                if batch_index < len(camera_batch) and camera_batch[batch_index] is not None
            }
            sample = {
                "images": images,
                "current_tvi": self["current_tvi"][batch_index],
                "history_tokens": self["history_tokens"][batch_index],
                "history_tvi": self["history_tvi"][batch_index],
                "history_mask": self["history_mask"][batch_index],
                "lang": self["lang"][batch_index],
                "platform_text": self.get("platform_text", [""] * batch_size)[batch_index],
                "state": self["state"][batch_index],
                "action": self["action"][batch_index],
                "action_padding_mask": self["action_padding_mask"][batch_index],
                "stop_target": float(self["stop_target"][batch_index]),
                "distance_to_goal": float(self["distance_to_goal"][batch_index]),
                "stop_soft_target": float(self["stop_soft_target"][batch_index]),
                "qa_target": self.get("qa_target", [None] * batch_size)[batch_index],
                "metadata": self["metadata"][batch_index],
            }
            if "history_cached_embeds" in self:
                sample["history_cached_embeds"] = self["history_cached_embeds"][batch_index]
            if "history_cached_mask" in self:
                sample["history_cached_mask"] = self["history_cached_mask"][batch_index]
            if "history_images" in self:
                sample["history_images"] = {
                    camera: camera_batch[batch_index] if batch_index < len(camera_batch) else []
                    for camera, camera_batch in self["history_images"].items()
                }
            samples.append(sample)
        return samples


class NavVLALeRobotDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str = "train",
        video_backend: str = "opencv",
        required_cameras: list[str] | None = None,
        image_resize: tuple[int, int] | None = DEFAULT_IMAGE_SIZE,
        visual_token_mode: str = "offline_cache",
        max_history_frames: int | None = None,
        stop_distance_positive_m: float = 3.0,
        stop_distance_negative_m: float = 10.0,
        use_platform_text: bool = False,
        use_state_text: bool = False,
    ) -> None:
        if video_backend != "opencv":
            raise ValueError(f"unsupported video_backend={video_backend!r}; only 'opencv' is implemented")
        if visual_token_mode not in {"offline_cache", "online_images"}:
            raise ValueError(f"unsupported visual_token_mode={visual_token_mode!r}")
        if max_history_frames is not None and int(max_history_frames) < 0:
            raise ValueError(f"max_history_frames must be non-negative, got {max_history_frames}")
        if float(stop_distance_positive_m) < 0:
            raise ValueError(f"stop_distance_positive_m must be non-negative, got {stop_distance_positive_m}")
        if float(stop_distance_negative_m) <= float(stop_distance_positive_m):
            raise ValueError(
                "stop_distance_negative_m must be greater than stop_distance_positive_m, "
                f"got {stop_distance_negative_m} <= {stop_distance_positive_m}"
            )
        self.root = Path(dataset_root)
        self.split = split
        self.required_cameras = required_cameras
        self.image_resize = tuple(image_resize) if image_resize is not None else DEFAULT_IMAGE_SIZE
        self.visual_token_mode = visual_token_mode
        self.max_history_frames = None if max_history_frames is None else int(max_history_frames)
        self.stop_distance_positive_m = float(stop_distance_positive_m)
        self.stop_distance_negative_m = float(stop_distance_negative_m)
        self.use_platform_text = bool(use_platform_text)
        self.use_state_text = bool(use_state_text)
        self.info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8"))
        self.modality = json.loads((self.root / "meta" / "modality.json").read_text(encoding="utf-8"))
        self.cameras = json.loads((self.root / "meta" / "navvla_cameras.json").read_text(encoding="utf-8"))
        self.data = _read_parquet_shards(self.root / "data")
        self.data = self.data[self.data["episode_index"].notna()].reset_index(drop=True)
        self.episode_sample_indices = _episode_sample_indices(self.data)
        self.goal_position_by_episode = _goal_position_by_episode(self.data)
        self.row_by_episode_frame = _row_by_episode_frame(self.data)
        self.row_by_index = {
            int(row["index"]): row
            for _idx, row in self.data.iterrows()
        }
        self.dataset_key = f"{self.info.get('dataset_name', self.root.name)}_{split}"
        self.dataset_statistics = self._load_or_compute_dataset_statistics()
        self.context = _read_parquet_batched(
            self.root / "meta" / "navvla_context_index.parquet"
        ).set_index("context.index_key", drop=False)
        self.context_tvi_time_by_index = self._context_tvi_time_by_index()
        self.episodes = _read_parquet_shards(self.root / "meta" / "episodes").set_index("episode_index", drop=False)
        video_index_path = self.root / "meta" / "navvla_video_index.parquet"
        self.video_index = (
            pd.read_parquet(video_index_path).set_index(["index", "video_key"], drop=False)
            if video_index_path.exists()
            else None
        )
        self.tasks = self._load_tasks()
        self.token_manifest = self._load_token_manifest()
        self.task_texts = self._load_task_texts()
        if required_cameras is not None:
            available = [camera_name for camera_name in required_cameras if camera_name in self.cameras]
            missing = [camera_name for camera_name in required_cameras if camera_name not in self.cameras]
            if not available:
                raise KeyError(
                    f"none of required_cameras={required_cameras!r} are present in dataset metadata; "
                    f"available cameras: {sorted(self.cameras)}"
                )
            self.required_cameras = available
            self.missing_required_cameras = missing
        else:
            self.missing_required_cameras = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data.iloc[index]
        task = self.tasks[int(row["task_index"])]
        action_available = bool(row["sample.action_available"])
        context_key = str(row["context.index_key"])
        if task["task_type"] in ACTION_TASK_TYPES and action_available and not context_key:
            raise KeyError(f"missing context.index_key for action sample at dataset row {index}")
        if context_key not in self.context.index:
            raise KeyError(f"context.index_key does not resolve: {context_key}")
        context_row = self.context.loc[context_key]

        context_view = self._truncate_context_for_training(context_row)

        images = self._read_images(row)
        current_tvi_time = self._current_tvi_time(row, context_row)
        current_tvi = self._current_tvi(current_tvi_time, images)
        history_token_refs = context_view["history_token_refs"]
        history_tokens, history_images = self._load_history_inputs(row, context_view, history_token_refs)
        history_tvi = self._history_tvi(context_view)
        history_mask = np.asarray(context_view["history_mask"], dtype=bool)
        qa_target = task.get("answer")
        compute_flow_loss = action_available and task["task_type"] in ACTION_TASK_TYPES

        raw_state = _pose4_from_observation_state(row["observation.state"])
        distance_to_goal = self._distance_to_goal(row, raw_state)
        stop_target = self._soft_stop_target(distance_to_goal)
        sample = {
            "images": images,
            "current_tvi": current_tvi,
            "history_tokens": history_tokens,
            "history_tvi": history_tvi,
            "history_mask": history_mask,
            "lang": self.task_texts[int(row["task_index"])],
            "platform_text": self._resolve_platform_text(task),
            "state": self._history_relative_state(row, context_view, raw_state) if self.use_state_text else self._empty_state(),
            "action": self._normalized_action(row),
            "action_padding_mask": np.asarray(_as_list(row["action.padding_mask"]), dtype=bool),
            "stop_target": stop_target,
            "distance_to_goal": distance_to_goal,
            "stop_soft_target": stop_target,
            "qa_target": qa_target,
            "metadata": {
                "task_type": task["task_type"],
                "task_subtype": task["task_subtype"],
                "dataset_source": task["dataset_source"],
                "episode_index": int(row["episode_index"]),
                "frame_index": int(row["frame_index"]),
                "timestamp": float(row["timestamp"]),
                "control_frequency_hz": float(self.info["navvla"]["control_frequency_hz"]),
                "camera": self.cameras,
                "context_index_key": context_key,
                "scene_id": str(self.episodes.loc[int(row["episode_index"])]["scene_id"]) if "scene_id" in self.episodes.columns else None,
                "history_steps": context_view["history_steps"],
                "history_blocks": context_view["history_blocks"],
                "history_token_refs": history_token_refs,
                "token_count": len(history_token_refs),
                "visual_token_mode": self.visual_token_mode,
                "compute_flow_loss": bool(compute_flow_loss),
                "compute_qa_loss": qa_target is not None,
                "raw_state": raw_state,
                "history_state_padding_mask": self._history_state_padding_mask(context_view),
            },
        }
        if self.visual_token_mode == "online_images":
            sample["history_images"] = history_images
        return sample

    def _resolve_platform_text(self, task: dict[str, Any]) -> str:
        if not self.use_platform_text:
            return ""
        return build_platform_text_from_meta(task, self.info)

    def _distance_to_goal(self, row: pd.Series, state: np.ndarray) -> float:
        goal_position = self.goal_position_by_episode[int(row["episode_index"])]
        return float(np.linalg.norm(state[:3] - goal_position))

    def _soft_stop_target(self, distance_to_goal: float) -> float:
        if distance_to_goal <= self.stop_distance_positive_m:
            return 1.0
        if distance_to_goal >= self.stop_distance_negative_m:
            return 0.0
        span = self.stop_distance_negative_m - self.stop_distance_positive_m
        return float((self.stop_distance_negative_m - distance_to_goal) / span)

    def _load_tasks(self) -> dict[int, dict[str, Any]]:
        tasks: dict[int, dict[str, Any]] = {}
        for line in (self.root / "meta" / "navvla_tasks.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            tasks[int(payload["task_index"])] = payload
        if not tasks:
            raise ValueError("meta/navvla_tasks.jsonl contains no tasks")
        return tasks

    def _load_token_manifest(self) -> TokenCacheManifest:
        if self.visual_token_mode == "offline_cache":
            return read_token_manifest(self.root)
        try:
            return read_token_manifest(self.root)
        except FileNotFoundError:
            return TokenCacheManifest(self.root / "cache" / "visual_tokens" / "manifest.jsonl", {})

    def _load_task_texts(self) -> dict[int, str]:
        table = pd.read_parquet(self.root / "meta" / "tasks.parquet").reset_index()
        if "task_index" not in table.columns:
            raise ValueError("meta/tasks.parquet is missing task_index")
        if "task" not in table.columns:
            raise ValueError("meta/tasks.parquet is missing task text index")
        return {int(row["task_index"]): str(row["task"]) for _idx, row in table.iterrows()}

    def save_dataset_statistics(self, save_path: str | Path) -> None:
        write_dataset_statistics(save_path, self.dataset_statistics)

    def _load_or_compute_dataset_statistics(self) -> dict[str, Any]:
        cache_path = self.root / "dataset_statistics.json"
        if cache_path.exists():
            stats = read_dataset_statistics(cache_path)
        else:
            stats = build_dataset_statistics(
                dataset_key=self.dataset_key,
                action_steps=flatten_valid_action_steps_from_rows(self.data),
                history_steps=self._configured_state_history_steps(),
                num_trajectories=int(self.info["total_episodes"]),
                num_transitions=len(self.data),
            )
            write_dataset_statistics(cache_path, stats)
        if self.dataset_key not in stats and len(stats) == 1:
            self.dataset_key = next(iter(stats))
        if self.dataset_key not in stats:
            raise KeyError(f"dataset_statistics.json does not contain {self.dataset_key!r}; keys={sorted(stats)}")
        action_stats = stats[self.dataset_key]["action"]
        stats[self.dataset_key]["state"] = build_repeated_state_statistics(
            action_stats,
            self._configured_state_history_steps(),
        )
        stats[self.dataset_key]["state_history_steps"] = self._configured_state_history_steps()
        stats[self.dataset_key]["state_mode"] = "history_relative_body_frame_actions"
        stats[self.dataset_key].setdefault("action_mode", "anchor_relative_body_frame_xyz_yaw")
        stats[self.dataset_key].setdefault("action_anchor", "current_frame_pose")
        return stats

    def _action_stats(self) -> dict[str, Any]:
        return self.dataset_statistics[self.dataset_key]["action"]

    def _context_tvi_time_by_index(self) -> dict[int, float]:
        if "current_tvi_time" in self.context.columns:
            return {
                int(row["index"]): float(row["current_tvi_time"])
                for _idx, row in self.context.iterrows()
            }
        return {
            int(row_index): float(row["frame_index"])
            for row_index, row in self.row_by_index.items()
        }

    def _current_tvi_time(self, row: pd.Series, context_row: pd.Series) -> float:
        if "current_tvi_time" in context_row.index:
            return float(context_row["current_tvi_time"])
        return float(row["frame_index"])

    def _configured_state_history_steps(self) -> int:
        if self.max_history_frames is not None:
            return int(self.max_history_frames)
        return int(self.info["navvla"].get("state_history_steps", 4))

    def _empty_state(self) -> np.ndarray:
        return np.zeros((0,), dtype=np.float32)

    def _normalized_action(self, row: pd.Series) -> np.ndarray:
        action = _float_array(row["action"]).reshape(
            self.info["navvla"]["action_horizon"],
            self.info["navvla"]["action_dim"],
        )
        mask = np.asarray(_as_list(row["action.padding_mask"]), dtype=bool)
        normalized = normalize_values(action, self._action_stats())
        normalized[mask] = 0.0
        return normalized.astype(np.float32)

    def _history_relative_state(
        self,
        row: pd.Series,
        context_view: pd.Series | dict[str, Any],
        raw_state: np.ndarray,
    ) -> np.ndarray:
        history_steps = _as_list(context_view["history_steps"])
        state_steps = self._configured_state_history_steps()
        raw_chunks = np.zeros((state_steps, self.info["navvla"]["action_dim"]), dtype=np.float32)
        if state_steps == 0:
            return raw_chunks.reshape(-1)

        history_poses = [
            self._pose_for_history_step(int(row["episode_index"]), step)
            for step in history_steps[-state_steps:]
        ]
        ordered_poses = history_poses + [raw_state]
        if len(ordered_poses) >= 2:
            chunks = np.stack(
                [
                    body_frame_action_from_pose(ordered_poses[index - 1], ordered_poses[index])
                    for index in range(1, len(ordered_poses))
                ],
                axis=0,
            )
            raw_chunks[-chunks.shape[0] :] = chunks
        normalized = normalize_values(
            raw_chunks.reshape(-1),
            self.dataset_statistics[self.dataset_key]["state"],
        ).reshape(raw_chunks.shape)
        padding_mask = self._history_state_padding_mask(context_view)
        normalized[padding_mask] = 0.0
        return normalized.reshape(-1).astype(np.float32)

    def _history_state_padding_mask(self, context_view: pd.Series | dict[str, Any]) -> np.ndarray:
        valid = min(len(_as_list(context_view["history_steps"])), self._configured_state_history_steps())
        mask = np.ones((self._configured_state_history_steps(),), dtype=bool)
        if valid:
            mask[-valid:] = False
        return mask

    def _pose_for_history_step(self, episode_index: int, step: dict[str, Any]) -> np.ndarray:
        if "frame_index" not in step:
            raise KeyError("history_steps entries must include frame_index for state lookup")
        key = (int(episode_index), int(step["frame_index"]))
        if key not in self.row_by_episode_frame:
            raise KeyError(f"history frame does not resolve for state lookup: episode={key[0]} frame={key[1]}")
        return _pose4_from_observation_state(self.row_by_episode_frame[key]["observation.state"])

    def _read_images(self, row: pd.Series) -> dict[str, Image.Image]:
        images: dict[str, Image.Image] = {}
        requested = self.required_cameras or list(self.cameras)
        for camera_name in requested:
            camera = self.cameras[camera_name]
            video_key = camera["video_key"]
            pattern = self.info["video_path"].get(video_key)
            if pattern is None:
                continue
            chunk_index = 0
            file_index = 0
            video_frame_index = int(row["index"])
            if self.video_index is not None:
                key = (int(row["index"]), video_key)
                if key not in self.video_index.index:
                    continue
                video_row = self.video_index.loc[key]
                if not bool(video_row["available"]):
                    continue
                video_frame_index = int(video_row["video_frame_index"])
                chunk_index = int(video_row["chunk_index"]) if "chunk_index" in video_row.index else 0
                file_index = int(video_row["file_index"]) if "file_index" in video_row.index else 0
            image = _read_video_frame(self.root / pattern.format(chunk_index=chunk_index, file_index=file_index), video_frame_index)
            images[camera_name] = self._prepare_image(image)
        return images

    def _current_tvi(self, current_tvi_time: float, images: dict[str, Image.Image]) -> np.ndarray:
        values = []
        for camera_name in images:
            values.append([float(current_tvi_time), float(self.cameras[camera_name]["azimuth_rad"])])
        return np.asarray(values, dtype=np.float32).reshape(-1, 2)

    def _history_tvi(self, context_row: pd.Series | dict[str, Any]) -> np.ndarray:
        history_steps = _as_list(context_row["history_steps"])
        history_blocks = _as_list(context_row["history_blocks"])
        legacy_tvi_time = _as_list(context_row["tvi_time"]) if "tvi_time" in context_row else []
        values = []
        for block_index, block in enumerate(history_blocks):
            step = history_steps[int(block["step_index"])]
            if block_index < len(legacy_tvi_time):
                tvi_time = float(legacy_tvi_time[block_index])
            else:
                if "index" not in step:
                    raise KeyError("history_steps entries must include index for TVI lookup")
                history_index = int(step["index"])
                if history_index not in self.context_tvi_time_by_index:
                    raise KeyError(f"history frame index does not resolve to TVI time: {history_index}")
                tvi_time = float(self.context_tvi_time_by_index[history_index])
            camera_name = str(block["camera_name"])
            values.append(
                [
                    tvi_time,
                    float(self.cameras[camera_name]["azimuth_rad"]),
                ]
            )
        return np.asarray(values, dtype=np.float32).reshape(-1, 2)

    def _load_history_tokens(self, refs: list[str]) -> np.ndarray:
        tokens = []
        for ref in refs:
            path = self.token_manifest.ref_to_path[ref]
            if not path.exists():
                raise FileNotFoundError(f"missing visual token cache file for {ref}: {path}")
            tokens.append(np.load(path).astype(np.float32))
        if not tokens:
            return np.zeros((0, 1, 3), dtype=np.float32)
        return np.stack(tokens, axis=0)

    def _load_history_inputs(
        self,
        anchor_row: pd.Series,
        context_row: pd.Series | dict[str, Any],
        refs: list[str],
    ) -> tuple[np.ndarray, dict[str, list[Image.Image]]]:
        if self.visual_token_mode == "offline_cache":
            validate_token_refs(refs, self.token_manifest)
            return self._load_history_tokens(refs), {}
        return np.zeros((0, 1, 3), dtype=np.float32), self._load_history_images(anchor_row, context_row)

    def _truncate_context_for_training(self, context_row: pd.Series) -> dict[str, Any]:
        history_steps = _as_list(context_row["history_steps"])
        history_blocks = _as_list(context_row["history_blocks"])
        history_token_refs = _as_list(context_row["history_token_refs"])
        history_mask = _as_list(context_row["history_mask"])

        if self.max_history_frames is None or len(history_steps) <= self.max_history_frames:
            first_kept_old_step = 0
            kept_steps = history_steps
        else:
            kept_steps = history_steps[-self.max_history_frames :]
            first_kept_old_step = len(history_steps) - len(kept_steps)

        allowed_cameras = set(self.required_cameras) if self.required_cameras is not None else None
        legacy_tvi_time = _as_list(context_row["tvi_time"]) if "tvi_time" in context_row.index else []
        kept_blocks: list[dict[str, Any]] = []
        kept_refs: list[str] = []
        kept_mask: list[Any] = []
        kept_tvi_time: list[Any] = []
        for block_index, block in enumerate(history_blocks):
            old_step_index = int(block["step_index"])
            if old_step_index < first_kept_old_step:
                continue
            camera_name = str(block["camera_name"])
            if allowed_cameras is not None and camera_name not in allowed_cameras:
                continue
            kept_block = dict(block)
            kept_block["step_index"] = old_step_index - first_kept_old_step
            kept_blocks.append(kept_block)
            if block_index < len(history_token_refs):
                kept_refs.append(history_token_refs[block_index])
            kept_mask.append(history_mask[block_index])
            if block_index < len(legacy_tvi_time):
                kept_tvi_time.append(legacy_tvi_time[block_index])

        return {
            "history_steps": kept_steps,
            "history_blocks": kept_blocks,
            "history_token_refs": kept_refs,
            "history_mask": kept_mask,
            "tvi_time": kept_tvi_time,
        }

    def _load_history_images(self, anchor_row: pd.Series, context_row: pd.Series | dict[str, Any]) -> dict[str, list[Image.Image]]:
        history_steps = _as_list(context_row["history_steps"])
        history_blocks = _as_list(context_row["history_blocks"])
        images: dict[str, list[Image.Image]] = {}
        for block in history_blocks:
            camera_name = str(block["camera_name"])
            if self.required_cameras is not None and camera_name not in self.required_cameras:
                continue
            step = history_steps[int(block["step_index"])]
            row = self._row_for_context_frame(
                episode_index=int(anchor_row["episode_index"]),
                frame_index=int(step["frame_index"]),
            )
            camera = self.cameras[camera_name]
            image = self._read_video_image(row, camera["video_key"])
            if image is None:
                continue
            images.setdefault(camera_name, []).append(image)
        return images

    def _row_for_context_frame(self, *, episode_index: int, frame_index: int) -> pd.Series:
        rows = self.data[
            (self.data["episode_index"].astype(int) == int(episode_index))
            & (self.data["frame_index"].astype(int) == int(frame_index))
        ]
        if rows.empty:
            raise KeyError(f"history frame does not resolve in data parquet: episode={episode_index}, frame={frame_index}")
        return rows.iloc[0]

    def _read_video_image(self, row: pd.Series, video_key: str) -> Image.Image | None:
        pattern = self.info["video_path"].get(video_key)
        if pattern is None:
            return None
        chunk_index = 0
        file_index = 0
        video_frame_index = int(row["index"])
        if self.video_index is not None:
            key = (int(row["index"]), video_key)
            if key not in self.video_index.index:
                return None
            video_row = self.video_index.loc[key]
            if not bool(video_row["available"]):
                return None
            video_frame_index = int(video_row["video_frame_index"])
            chunk_index = int(video_row["chunk_index"]) if "chunk_index" in video_row.index else 0
            file_index = int(video_row["file_index"]) if "file_index" in video_row.index else 0
        image = _read_video_frame(self.root / pattern.format(chunk_index=chunk_index, file_index=file_index), video_frame_index)
        return self._prepare_image(image)

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        return image.resize(self.image_resize)


def collate_navvla_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    all_cameras = sorted({camera for sample in batch for camera in sample["images"]})
    images = {camera: [sample["images"].get(camera) for sample in batch] for camera in all_cameras}
    image_masks = {
        camera: np.asarray([sample["images"].get(camera) is not None for sample in batch], dtype=bool)
        for camera in all_cameras
    }

    max_history = max((_history_length(sample) for sample in batch), default=0)
    token_shape = _first_token_shape(batch)
    history_tokens = np.zeros((len(batch), max_history, *token_shape), dtype=np.float32)
    history_tvi = np.zeros((len(batch), max_history, 2), dtype=np.float32)
    history_mask = np.zeros((len(batch), max_history), dtype=bool)
    for batch_index, sample in enumerate(batch):
        length = _history_length(sample)
        if length == 0:
            continue
        token_length = sample["history_tokens"].shape[0]
        if token_length:
            history_tokens[batch_index, :token_length] = sample["history_tokens"]
        tvi_length = sample["history_tvi"].shape[0]
        if tvi_length:
            history_tvi[batch_index, :tvi_length] = sample["history_tvi"]
        mask_length = sample["history_mask"].shape[0]
        if mask_length:
            history_mask[batch_index, :mask_length] = sample["history_mask"]

    collated = NavVLACollatedBatch({
        "images": images,
        "image_masks": image_masks,
        "current_tvi": [sample["current_tvi"] for sample in batch],
        "history_tokens": history_tokens,
        "history_tvi": history_tvi,
        "history_mask": history_mask,
        "lang": [sample["lang"] for sample in batch],
        "platform_text": [sample["platform_text"] for sample in batch],
        "state": np.stack([sample["state"] for sample in batch], axis=0),
        "action": np.stack([sample["action"] for sample in batch], axis=0),
        "action_padding_mask": np.stack([sample["action_padding_mask"] for sample in batch], axis=0),
        "stop_target": np.asarray([sample["stop_target"] for sample in batch], dtype=np.float32),
        "distance_to_goal": np.asarray([sample["distance_to_goal"] for sample in batch], dtype=np.float32),
        "stop_soft_target": np.asarray([sample["stop_soft_target"] for sample in batch], dtype=np.float32),
        "qa_target": [sample["qa_target"] for sample in batch],
        "metadata": np.asarray([sample["metadata"] for sample in batch], dtype=object),
    })
    if any("history_images" in sample for sample in batch):
        history_cameras = sorted({camera for sample in batch for camera in sample.get("history_images", {})})
        collated["history_images"] = {
            camera: [sample.get("history_images", {}).get(camera, []) for sample in batch]
            for camera in history_cameras
        }
    return collated


def _read_video_frame(path: Path, frame_index: int) -> Image.Image:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"video does not open: {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise IndexError(f"failed to read frame {frame_index} from {path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        cap.release()


def _read_parquet_shards(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("chunk-*/part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards found under {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _read_parquet_batched(path: Path, *, batch_size: int = 50_000) -> pd.DataFrame:
    """Read a single parquet file in row batches to avoid PyArrow's ~2GB per-array limit."""
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    if parquet_file.metadata.num_rows == 0:
        return pd.DataFrame()
    frames = [batch.to_pandas() for batch in parquet_file.iter_batches(batch_size=batch_size)]
    return pd.concat(frames, ignore_index=True)


def _goal_position_by_episode(data: pd.DataFrame) -> dict[int, np.ndarray]:
    goals: dict[int, np.ndarray] = {}
    for episode_index, rows in data.groupby("episode_index", sort=False):
        final_state = _pose4_from_observation_state(rows.iloc[-1]["observation.state"])
        goals[int(episode_index)] = final_state[:3]
    return goals


def _row_by_episode_frame(data: pd.DataFrame) -> dict[tuple[int, int], pd.Series]:
    rows = {}
    for _idx, row in data.iterrows():
        rows[(int(row["episode_index"]), int(row["frame_index"]))] = row
    return rows


def _episode_sample_indices(data: pd.DataFrame) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    ordered = data.reset_index(names="_dataset_position").sort_values(["episode_index", "frame_index", "index"], kind="stable")
    for episode_index, rows in ordered.groupby("episode_index", sort=True):
        groups[int(episode_index)] = rows["_dataset_position"].astype(int).tolist()
    return groups


def _pose4_from_observation_state(value: Any) -> np.ndarray:
    state = _float_array(value).reshape(-1)
    if state.shape[0] == 6:
        return np.asarray([state[0], state[1], state[2], state[5]], dtype=np.float32)
    if state.shape[0] >= 4:
        return state[:4].astype(np.float32)
    raise ValueError(f"observation.state must contain at least 4 values, got shape {state.shape}")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return list(value)


def _float_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return np.asarray([_float_array(item) for item in value], dtype=np.float32)
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray([_float_array(item) if isinstance(item, (list, np.ndarray)) else item for item in value], dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _first_token_shape(batch: list[dict[str, Any]]) -> tuple[int, ...]:
    for sample in batch:
        if sample["history_tokens"].shape[0] > 0:
            return tuple(sample["history_tokens"].shape[1:])
    return (1, 3)


def _history_length(sample: dict[str, Any]) -> int:
    return max(
        int(sample["history_tokens"].shape[0]),
        int(sample["history_tvi"].shape[0]),
        int(sample["history_mask"].shape[0]),
    )
