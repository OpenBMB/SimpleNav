from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import pyarrow.parquet as pq

from tool.navvla.statistics import (
    body_frame_action_from_pose,
    build_dataset_statistics,
    build_repeated_state_statistics,
    flatten_valid_action_steps_from_rows,
    normalize_values,
    read_dataset_statistics,
    write_dataset_statistics,
)
from starVLA.dataloader.airsim_utils import config_bool
from tool.navvla.context_index import DEFAULT_CONTEXT_TOKEN_BUDGET, load_runtime_context_index, resolve_context_index_paths
from tool.navvla.visual_token_cache import profile_cache_root


ACTION_TASK_TYPES = {"navigation", "driving", "tracking"}


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
                "action": self["action"][batch_index],
                "action_padding_mask": self["action_padding_mask"][batch_index],
                "distance_to_goal": float(self["distance_to_goal"][batch_index]),
                "qa_target": self.get("qa_target", [None] * batch_size)[batch_index],
                "metadata": self["metadata"][batch_index],
            }
            if "state" in self and bool(self["state_present"][batch_index]):
                sample["state"] = self["state"][batch_index]
            if "history_images" in self:
                sample["history_images"] = {
                    camera: camera_batch[batch_index] if batch_index < len(camera_batch) else []
                    for camera, camera_batch in self["history_images"].items()
                }
            if "history_cached_embeds" in self:
                sample["history_cached_embeds"] = self["history_cached_embeds"][batch_index]
            if "history_cached_deepstack_embeds" in self:
                sample["history_cached_deepstack_embeds"] = self["history_cached_deepstack_embeds"][batch_index]
            if "history_cached_mask" in self:
                sample["history_cached_mask"] = self["history_cached_mask"][batch_index]
            for key in (
                "long_memory_source_tokens",
                "long_memory_source_mask",
                "long_memory_source_tvi",
                "long_memory_tokens",
                "long_memory_tvi",
            ):
                if key in self:
                    sample[key] = self[key][batch_index]
            samples.append(sample)
        return samples


class NavVLALeRobotDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path | None = None,
        *,
        root: str | Path | None = None,
        split: str = "train",
        video_backend: str = "opencv",
        required_cameras: list[str] | None = None,
        image_resize: tuple[int, int] | None = None,
        visual_token_mode: str = "offline_cache",
        visual_token_profile: str = "qwen3_vl_4b_pooled_history",
        token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
        include_state: bool = False,
        include_source_metadata: bool = False,
    ) -> None:
        if video_backend != "opencv":
            raise ValueError(f"unsupported video_backend={video_backend!r}; only 'opencv' is implemented")
        if visual_token_mode not in {"offline_cache", "online_images", "cached_history_online_current"}:
            raise ValueError(f"unsupported visual_token_mode={visual_token_mode!r}")
        if dataset_root is None:
            if root is None:
                raise TypeError("NavVLALeRobotDataset requires dataset_root or root")
            dataset_root = root
        elif root is not None:
            raise TypeError("pass only one of dataset_root or root")
        self.root = Path(dataset_root)
        self.split = split
        self.required_cameras = required_cameras
        self.image_resize = image_resize
        self.visual_token_mode = visual_token_mode
        self.visual_token_profile = visual_token_profile
        self.token_budget = int(token_budget)
        self.include_state = config_bool(include_state, False)
        self.include_source_metadata = config_bool(include_source_metadata, False)
        self._visual_token_dtype = np.dtype(np.float16)
        self.info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8"))
        self.modality = json.loads((self.root / "meta" / "modality.json").read_text(encoding="utf-8"))
        self.cameras = json.loads((self.root / "meta" / "navvla_cameras.json").read_text(encoding="utf-8"))
        self.data = _read_parquet_shards(self.root / "data", columns=self._data_columns())
        self.data = self.data[self.data["episode_index"].notna()].reset_index(drop=True)
        self.episode_sample_indices = _episode_sample_indices(self.data)
        self.goal_position_by_episode = _goal_position_by_episode(self.data)
        self._state_lookup_by_episode: dict[int, dict[str, np.ndarray]] | None = None
        self.dataset_key = f"{self.info.get('dataset_name', self.root.name)}_{split}"
        self.dataset_statistics = self._load_or_compute_dataset_statistics()
        self.context_index_paths = resolve_context_index_paths(self.root, token_budget=self.token_budget)
        self.runtime_context = load_runtime_context_index(self.context_index_paths)
        self.context = self.runtime_context.meta_by_data_index
        self.episodes = _read_parquet_shards(self.root / "meta" / "episodes").set_index("episode_index", drop=False)
        video_index_path = self.root / "meta" / "navvla_video_index.parquet"
        self.video_index = (
            pd.read_parquet(video_index_path).set_index(["index", "video_key"], drop=False)
            if video_index_path.exists()
            else None
        )
        self.tasks = self._load_tasks()
        self.visual_token_ref_to_path = self._load_visual_token_profile_index()
        self.task_texts = self._load_task_texts()
        if required_cameras:
            missing = set(required_cameras) - set(self.cameras)
            if missing:
                raise KeyError(f"required cameras missing from metadata: {sorted(missing)}")

    def _data_columns(self) -> list[str]:
        columns = [
            "index",
            "episode_index",
            "frame_index",
            "timestamp",
            "task_index",
            "context.index_key",
            "sample.action_available",
            "observation.state",
            "action",
            "action.padding_mask",
            "source_frame_index",
        ]
        if self.include_source_metadata:
            columns.append("source_metadata")
        return columns

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data.iloc[index]
        task = self.tasks[int(row["task_index"])]
        action_available = bool(row["sample.action_available"])
        data_index = int(row["index"])
        if data_index not in self.context.index:
            raise KeyError(f"data index does not resolve in BATS context: {data_index}")
        compact_context = self.runtime_context.materialize_meta_row(self.context.loc[data_index])
        context_row = self._expand_bats_context(compact_context, row=row)
        context_key = str(context_row["context.index_key"])

        context_view = self._truncate_context_for_training(context_row)

        images = self._read_images(row)
        current_tvi_time = float(row["timestamp"])
        current_tvi = self._current_tvi(current_tvi_time, images)
        history_token_refs = context_view["history_token_refs"]
        history_tokens, history_images, history_cached = self._load_history_inputs(row, context_view, history_token_refs)
        history_tvi = self._history_tvi(context_view)
        history_mask = np.asarray(context_view["history_mask"], dtype=bool)
        qa_target = task.get("answer")
        compute_flow_loss = action_available and task["task_type"] in ACTION_TASK_TYPES

        raw_state = _pose4_from_observation_state(row["observation.state"])
        distance_to_goal = self._distance_to_goal(row, raw_state)
        sample = {
            "images": images,
            "current_tvi": current_tvi,
            "history_tokens": history_tokens,
            "history_tvi": history_tvi,
            "history_mask": history_mask,
            "lang": self.task_texts[int(row["task_index"])],
            "platform_text": task["platform_text"],
            "action": self._normalized_action(row),
            "action_padding_mask": np.asarray(_as_list(row["action.padding_mask"]), dtype=bool),
            "distance_to_goal": distance_to_goal,
            "qa_target": qa_target,
            "metadata": self._build_sample_metadata(
                row=row,
                task=task,
                context_key=context_key,
                context_view=context_view,
                context_row=context_row,
                history_token_refs=history_token_refs,
                compute_flow_loss=bool(compute_flow_loss),
                qa_target=qa_target,
                raw_state=raw_state,
            ),
        }
        if self.include_state:
            sample["state"] = self._history_relative_state(row, context_view, raw_state)
        if self.visual_token_mode == "online_images":
            sample["history_images"] = history_images
        if history_cached is not None:
            sample.update(history_cached)
        return sample

    def _build_sample_metadata(
        self,
        *,
        row: pd.Series,
        task: dict[str, Any],
        context_key: str,
        context_view: pd.Series | dict[str, Any],
        context_row: pd.Series | dict[str, Any],
        history_token_refs: list[str],
        compute_flow_loss: bool,
        qa_target: Any,
        raw_state: np.ndarray,
    ) -> dict[str, Any]:
        episode_index = int(row["episode_index"])
        episode_row = self.episodes.loc[episode_index]
        metadata = {
            "task_type": task["task_type"],
            "task_subtype": task["task_subtype"],
            "dataset_source": task["dataset_source"],
            "episode_index": episode_index,
            "episode_id": str(episode_row["episode_id"]) if "episode_id" in episode_row.index else None,
            "trajectory_id": str(episode_row["trajectory_id"]) if "trajectory_id" in episode_row.index else None,
            "frame_index": int(row["frame_index"]),
            "source_frame_index": _optional_int(row.get("source_frame_index")),
            "timestamp": float(row["timestamp"]),
            "control_frequency_hz": float(self.info["navvla"]["control_frequency_hz"]),
            "camera": self.cameras,
            "context_index_key": context_key,
            "scene_id": str(episode_row["scene_id"]) if "scene_id" in episode_row.index else None,
            "history_steps": context_view["history_steps"],
            "history_blocks": context_view["history_blocks"],
            "history_token_refs": history_token_refs,
            "bats_k": _optional_float(context_row, "bats_k"),
            "token_count": len(history_token_refs),
            "visual_token_mode": self.visual_token_mode,
            "token_budget": self.token_budget,
            "context_index_path": str(self.context_index_paths.meta_path),
            "compute_flow_loss": bool(compute_flow_loss),
            "compute_qa_loss": qa_target is not None,
            "raw_state": raw_state,
            "history_state_padding_mask": self._history_state_padding_mask(context_view),
        }
        if self.include_source_metadata:
            metadata["source_metadata"] = _optional_source_metadata(row.get("source_metadata"))
        return metadata

    def _distance_to_goal(self, row: pd.Series, state: np.ndarray) -> float:
        goal_position = self.goal_position_by_episode[int(row["episode_index"])]
        return float(np.linalg.norm(state[:3] - goal_position))

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

    def _load_visual_token_profile_index(self) -> dict[str, str]:
        if self.visual_token_mode not in {"offline_cache", "cached_history_online_current"}:
            return {}
        profile_root = profile_cache_root(self.root, self.visual_token_profile)
        manifest_path = profile_root / "manifest.json"
        index_path = profile_root / "index.parquet"
        legacy_manifest = self.root / "cache" / "visual_tokens" / "manifest.jsonl"
        legacy_tokens = self.root / "cache" / "visual_tokens" / "tokens"
        if legacy_manifest.exists():
            raise ValueError("legacy smoke visual token manifest.jsonl is not supported; use a profile cache")
        if legacy_tokens.exists() and any(legacy_tokens.glob("*.npy")):
            raise ValueError("legacy smoke visual token .npy cache is not supported; use a profile cache")
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing visual token profile manifest: {manifest_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"missing visual token profile index: {index_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("file_format") != "npz":
            raise ValueError(f"visual token profile {self.visual_token_profile} must use npz files")
        if manifest.get("visual_head") == "smoke_token":
            raise ValueError("smoke visual token cache is not supported")
        self._visual_token_dtype = np.dtype(manifest.get("dtype", "float16"))
        array_keys = set(manifest.get("array_keys", []))
        if not {"image_embeds", "deepstack_embeds"}.issubset(array_keys):
            raise ValueError(f"visual token profile {self.visual_token_profile} must declare image_embeds and deepstack_embeds")
        index = pd.read_parquet(index_path, columns=["ref", "path"])
        return {str(row.ref): str(row.path) for row in index.itertuples(index=False)}

    def _load_task_texts(self) -> dict[int, str]:
        table = pd.read_parquet(self.root / "meta" / "tasks.parquet").reset_index()
        if "task_index" not in table.columns:
            raise ValueError("meta/tasks.parquet is missing task_index")
        if "task" not in table.columns:
            metadata_task_texts = self._load_task_texts_from_frame_metadata(table["task_index"])
            if metadata_task_texts:
                return metadata_task_texts
            raise ValueError("meta/tasks.parquet is missing task text index")
        return {int(row["task_index"]): str(row["task"]) for _idx, row in table.iterrows()}

    def _load_task_texts_from_frame_metadata(self, task_indices: pd.Series) -> dict[int, str]:
        frame_metadata_path = self.root / "meta" / "navvla_frame_metadata.jsonl"
        if not frame_metadata_path.exists():
            return {}
        needed_task_indices = {int(task_index) for task_index in task_indices.to_numpy(dtype=np.int64)}
        if not needed_task_indices:
            return {}
        frame_to_task = {
            int(frame_index): int(task_index)
            for frame_index, task_index in zip(self.data["index"].to_numpy(), self.data["task_index"].to_numpy())
        }
        task_texts: dict[int, str] = {}
        with frame_metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                task_index = frame_to_task.get(int(payload.get("index", -1)))
                if task_index is None or task_index not in needed_task_indices or task_index in task_texts:
                    continue
                source_metadata = payload.get("source_metadata") or {}
                instruction = source_metadata.get("instruction") or {}
                text = source_metadata.get("instruction_text") or instruction.get("instruction_text")
                if text:
                    task_texts[task_index] = str(text)
                    if len(task_texts) == len(needed_task_indices):
                        break
        return task_texts

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
                num_trajectories=int(self.info["total_episodes"]),
                num_transitions=len(self.data),
            )
            write_dataset_statistics(cache_path, stats)
        if self.dataset_key not in stats and len(stats) == 1:
            self.dataset_key = next(iter(stats))
        if self.dataset_key not in stats:
            raise KeyError(f"dataset_statistics.json does not contain {self.dataset_key!r}; keys={sorted(stats)}")
        stats[self.dataset_key].setdefault("action_mode", "anchor_relative_body_frame_xyz_yaw")
        stats[self.dataset_key].setdefault("action_anchor", "current_frame_pose")
        stats[self.dataset_key]["state_mode"] = "variable_bats_history_relative_body_frame_actions"
        stats[self.dataset_key].pop("state", None)
        stats[self.dataset_key].pop("state_history_steps", None)
        return stats

    def _action_stats(self) -> dict[str, Any]:
        return self.dataset_statistics[self.dataset_key]["action"]

    def _state_history_count_for_context(self, context_view: pd.Series | dict[str, Any]) -> int:
        return len(_as_list(context_view["history_steps"]))

    def _state_stats_for_context(self, context_view: pd.Series | dict[str, Any]) -> dict[str, Any]:
        return build_repeated_state_statistics(self._action_stats(), self._state_history_count_for_context(context_view))

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
        state_steps = self._state_history_count_for_context(context_view)
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
            self._state_stats_for_context(context_view),
        ).reshape(raw_chunks.shape)
        padding_mask = self._history_state_padding_mask(context_view)
        normalized[padding_mask] = 0.0
        return normalized.reshape(-1).astype(np.float32)

    def _history_state_padding_mask(self, context_view: pd.Series | dict[str, Any]) -> np.ndarray:
        return np.zeros((self._state_history_count_for_context(context_view),), dtype=bool)

    def _pose_for_history_step(self, episode_index: int, step: dict[str, Any]) -> np.ndarray:
        if "frame_index" in step:
            return self._pose_for_context_frame_index(episode_index=int(episode_index), frame_index=int(step["frame_index"]))
        if "timestamp" not in step:
            raise KeyError("history_steps entries must include timestamp for state lookup")
        return self._pose_for_context_timestamp(episode_index=int(episode_index), timestamp=float(step["timestamp"]))

    def _state_lookup(self) -> dict[int, dict[str, np.ndarray]]:
        lookup = self._state_lookup_by_episode
        if lookup is not None:
            return lookup
        ordered = self.data[["episode_index", "frame_index", "timestamp", "observation.state"]].sort_values(
            ["episode_index", "timestamp", "frame_index"],
            kind="stable",
        )
        lookup = {}
        for episode_index, rows in ordered.groupby("episode_index", sort=False):
            states = np.stack(
                [_pose4_from_observation_state(value) for value in rows["observation.state"]],
                axis=0,
            ).astype(np.float32)
            lookup[int(episode_index)] = {
                "frame_indices": rows["frame_index"].astype(int).to_numpy(dtype=np.int64),
                "timestamps": rows["timestamp"].astype(float).to_numpy(dtype=np.float64),
                "states": states,
            }
        self._state_lookup_by_episode = lookup
        return lookup

    def _pose_for_context_timestamp(self, *, episode_index: int, timestamp: float) -> np.ndarray:
        episode = self._state_lookup().get(int(episode_index))
        if episode is None:
            raise KeyError(f"history timestamp does not resolve in data parquet: episode={episode_index}, timestamp={timestamp}")
        timestamps = episode["timestamps"]
        position = int(np.searchsorted(timestamps, float(timestamp), side="left"))
        candidates = []
        if position < timestamps.shape[0]:
            candidates.append(position)
        if position > 0:
            candidates.append(position - 1)
        for candidate in candidates:
            if abs(float(timestamps[candidate]) - float(timestamp)) <= 1e-6:
                return np.asarray(episode["states"][candidate], dtype=np.float32)
        raise KeyError(f"history timestamp does not resolve in data parquet: episode={episode_index}, timestamp={timestamp}")

    def _pose_for_context_frame_index(self, *, episode_index: int, frame_index: int) -> np.ndarray:
        episode = self._state_lookup().get(int(episode_index))
        if episode is None:
            raise KeyError(f"history frame does not resolve for state lookup: episode={episode_index} frame={frame_index}")
        frame_indices = episode["frame_indices"]
        matches = np.flatnonzero(frame_indices == int(frame_index))
        if matches.size == 0:
            raise KeyError(f"history frame does not resolve for state lookup: episode={episode_index} frame={frame_index}")
        return np.asarray(episode["states"][int(matches[0])], dtype=np.float32)

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
            if self.image_resize is not None:
                image = image.resize(self.image_resize)
            images[camera_name] = image
        return images

    def _current_tvi(self, current_tvi_time: float, images: dict[str, Image.Image]) -> np.ndarray:
        values = []
        for camera_name in images:
            values.append([float(current_tvi_time), float(self.cameras[camera_name]["azimuth_rad"])])
        return np.asarray(values, dtype=np.float32).reshape(-1, 2)

    def _history_tvi(self, context_row: pd.Series | dict[str, Any]) -> np.ndarray:
        history_steps = _as_list(context_row["history_steps"])
        history_blocks = _as_list(context_row["history_blocks"])
        values = []
        for block in history_blocks:
            step = history_steps[int(block["step_index"])]
            if "timestamp" not in step:
                raise KeyError("history_steps entries must include timestamp for TVI lookup")
            camera_name = str(block["camera_name"])
            values.append(
                [
                    float(step["timestamp"]),
                    float(self.cameras[camera_name]["azimuth_rad"]),
                ]
            )
        return np.asarray(values, dtype=np.float32).reshape(-1, 2)

    def _load_history_tokens(self, refs: list[str]) -> np.ndarray:
        cached = self._load_history_cached_embeddings(refs)
        return cached["history_cached_embeds"]

    def _load_history_cached_embeddings(self, refs: list[str]) -> dict[str, np.ndarray]:
        image_embeds: list[np.ndarray] = []
        deepstack_embeds: list[np.ndarray] = []
        for ref in refs:
            try:
                relpath = self.visual_token_ref_to_path[ref]
            except KeyError as exc:
                raise KeyError(f"visual token ref {ref!r} is missing from profile {self.visual_token_profile}") from exc
            path = self.root / relpath
            if path.suffix != ".npz":
                raise ValueError(f"cached visual token file must be .npz for {ref}: {path}")
            if not path.exists():
                raise FileNotFoundError(f"missing visual token cache file for {ref}: {path}")
            with np.load(path, allow_pickle=False) as payload:
                if "image_embeds" not in payload or "deepstack_embeds" not in payload:
                    raise ValueError(f"visual token cache file missing required arrays: {path}")
                image_embeds.append(np.asarray(payload["image_embeds"]))
                deepstack_embeds.append(np.asarray(payload["deepstack_embeds"]))
        if not image_embeds:
            return {
                "history_cached_embeds": np.zeros((0, 4, 0), dtype=self._visual_token_dtype),
                "history_cached_deepstack_embeds": np.zeros((0, 0, 4, 0), dtype=self._visual_token_dtype),
                "history_cached_mask": np.zeros((0,), dtype=bool),
            }
        stacked_images = np.stack(image_embeds, axis=0)
        if deepstack_embeds:
            if len(deepstack_embeds) != len(image_embeds):
                raise ValueError("all cached history refs must either include deepstack_embeds or omit them")
            stacked_deepstack = np.stack(deepstack_embeds, axis=1)
        else:
            stacked_deepstack = np.zeros(
                (0, stacked_images.shape[0], stacked_images.shape[1], stacked_images.shape[2]),
                dtype=stacked_images.dtype,
            )
        return {
            "history_cached_embeds": stacked_images,
            "history_cached_deepstack_embeds": stacked_deepstack,
            "history_cached_mask": np.ones((stacked_images.shape[0],), dtype=bool),
        }

    def _load_history_inputs(
        self,
        anchor_row: pd.Series,
        context_row: pd.Series | dict[str, Any],
        refs: list[str],
    ) -> tuple[np.ndarray, dict[str, list[Image.Image]], dict[str, np.ndarray] | None]:
        if self.visual_token_mode == "offline_cache":
            return self._load_history_tokens(refs), {}, None
        if self.visual_token_mode == "cached_history_online_current":
            return np.zeros((0, 1, 3), dtype=np.float32), {}, self._load_history_cached_embeddings(refs)
        return np.zeros((0, 1, 3), dtype=np.float32), self._load_history_images(anchor_row, context_row), None

    def _expand_bats_context(self, context_row: dict[str, Any], *, row: pd.Series) -> dict[str, Any]:
        episode_index = int(row["episode_index"])
        episode_row = self.episodes.loc[episode_index]
        episode_id = str(episode_row["episode_id"])
        episode_data = self.data[self.data["episode_index"].astype(int) == episode_index].sort_values(
            ["frame_index", "timestamp"],
            kind="stable",
        )
        frame_indices = episode_data["frame_index"].astype(int).to_numpy(dtype=np.int64)
        timestamps = episode_data["timestamp"].astype(float).to_numpy(dtype=np.float64)
        camera_names = [str(value) for value in context_row.get("camera_names", [])]

        def expand(prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[bool]]:
            steps: list[dict[str, Any]] = []
            blocks: list[dict[str, Any]] = []
            refs: list[str] = []
            masks: list[bool] = []
            for selected in _as_list(context_row[f"{prefix}_frames"]):
                frame_index = int(selected["frame_index"])
                position = int(np.searchsorted(frame_indices, frame_index, side="left"))
                if position >= len(frame_indices) or int(frame_indices[position]) != frame_index:
                    raise KeyError(
                        f"BATS context frame does not resolve: episode={episode_index} frame={frame_index}"
                    )
                step_index = len(steps)
                steps.append({"frame_index": frame_index, "timestamp": float(timestamps[position])})
                camera_mask = int(selected["camera_mask"])
                for camera_index, camera_name in enumerate(camera_names):
                    if not camera_mask & (1 << camera_index):
                        continue
                    blocks.append(
                        {
                            "step_index": step_index,
                            "camera_name": camera_name,
                            "frame_index": frame_index,
                        }
                    )
                    refs.append(f"{episode_id}/{frame_index:06d}/{camera_name}")
                    masks.append(True)
            return steps, blocks, refs, masks

        history_steps, history_blocks, history_refs, history_mask = expand("history")
        long_steps, long_blocks, long_refs, long_mask = expand("long_memory")
        dataset_name = str(self.info.get("dataset_name", self.root.name))
        policy = str(context_row.get("context_policy_version", "bats-v1"))
        return {
            **context_row,
            "context.index_key": (
                f"{dataset_name}/{self.split}/{episode_id}/f{int(row['frame_index']):06d}/{policy}"
            ),
            "history_steps": history_steps,
            "history_blocks": history_blocks,
            "history_token_refs": history_refs,
            "history_mask": history_mask,
            "long_memory_steps": long_steps,
            "long_memory_blocks": long_blocks,
            "long_memory_token_refs": long_refs,
            "long_memory_mask": long_mask,
        }

    def _truncate_context_for_training(self, context_row: pd.Series | dict[str, Any]) -> dict[str, Any]:
        context_key = str(context_row.get("context.index_key", "<unknown>"))
        history_steps = _as_list(context_row["history_steps"])
        history_blocks = _as_list(context_row["history_blocks"])
        history_token_refs = _as_list(context_row["history_token_refs"])
        history_mask = _as_list(context_row["history_mask"])
        per_block_lengths = {
            "history_blocks": len(history_blocks),
            "history_token_refs": len(history_token_refs),
            "history_mask": len(history_mask),
        }
        if "history_tvi" in context_row:
            per_block_lengths["history_tvi"] = len(_as_list(context_row["history_tvi"]))
        expected_blocks = per_block_lengths["history_blocks"]
        mismatched = {
            name: length
            for name, length in per_block_lengths.items()
            if length != expected_blocks
        }
        if mismatched:
            raise ValueError(
                f"context {context_key} has inconsistent per-block history field lengths: "
                f"{per_block_lengths}; expected all per-block fields to match history_blocks={expected_blocks}"
            )
        for block_index, block in enumerate(history_blocks):
            if "step_index" not in block:
                raise KeyError(f"context {context_key} history_blocks[{block_index}] is missing step_index")
            step_index = int(block["step_index"])
            if step_index < 0 or step_index >= len(history_steps):
                raise IndexError(
                    f"context {context_key} history_blocks[{block_index}].step_index={step_index} "
                    f"does not resolve in history_steps length {len(history_steps)}"
                )

        allowed_cameras = set(self.required_cameras) if self.required_cameras is not None else None
        kept_blocks: list[dict[str, Any]] = []
        kept_refs: list[str] = []
        kept_mask: list[Any] = []
        for block_index, block in enumerate(history_blocks):
            camera_name = str(block["camera_name"])
            if allowed_cameras is not None and camera_name not in allowed_cameras:
                continue
            kept_blocks.append(dict(block))
            kept_refs.append(history_token_refs[block_index])
            kept_mask.append(history_mask[block_index])

        return {
            "history_steps": history_steps,
            "history_blocks": kept_blocks,
            "history_token_refs": kept_refs,
            "history_mask": kept_mask,
        }

    def _load_history_images(self, anchor_row: pd.Series, context_row: pd.Series | dict[str, Any]) -> dict[str, list[Image.Image]]:
        history_blocks = _as_list(context_row["history_blocks"])
        history_refs = _as_list(context_row["history_token_refs"])
        images: dict[str, list[Image.Image]] = {}
        for block_index, block in enumerate(history_blocks):
            camera_name = str(block["camera_name"])
            if self.required_cameras is not None and camera_name not in self.required_cameras:
                continue
            row = self._row_for_context_frame(
                episode_index=int(anchor_row["episode_index"]),
                frame_index=_frame_index_from_history_ref(str(history_refs[block_index])),
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

    def _row_for_context_timestamp(self, *, episode_index: int, timestamp: float) -> pd.Series:
        episode_mask = self.data["episode_index"].astype(int) == int(episode_index)
        timestamp_values = self.data["timestamp"].astype(float)
        rows = self.data[episode_mask & np.isclose(timestamp_values, float(timestamp), rtol=0.0, atol=1e-6)]
        if rows.empty:
            raise KeyError(f"history timestamp does not resolve in data parquet: episode={episode_index}, timestamp={timestamp}")
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
        if self.image_resize is not None:
            image = image.resize(self.image_resize)
        return image


def collate_navvla_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    all_cameras = sorted({camera for sample in batch for camera in sample["images"]})
    images = {camera: [sample["images"].get(camera) for sample in batch] for camera in all_cameras}
    image_masks = {
        camera: np.asarray([sample["images"].get(camera) is not None for sample in batch], dtype=bool)
        for camera in all_cameras
    }

    max_history = max((_history_length(sample) for sample in batch), default=0)
    token_shape = _first_token_shape(batch)
    history_tokens = np.zeros((len(batch), max_history, *token_shape), dtype=_first_token_dtype(batch))
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
        "action": np.stack([sample["action"] for sample in batch], axis=0),
        "action_padding_mask": np.stack([sample["action_padding_mask"] for sample in batch], axis=0),
        "distance_to_goal": np.asarray([sample["distance_to_goal"] for sample in batch], dtype=np.float32),
        "qa_target": [sample["qa_target"] for sample in batch],
        "metadata": np.asarray([sample["metadata"] for sample in batch], dtype=object),
    })
    if any("state" in sample for sample in batch):
        max_state_dim = max((int(sample["state"].shape[0]) for sample in batch if "state" in sample), default=0)
        state = np.zeros((len(batch), max_state_dim), dtype=np.float32)
        state_padding_mask = np.ones((len(batch), max_state_dim), dtype=bool)
        state_present = np.zeros((len(batch),), dtype=bool)
        for batch_index, sample in enumerate(batch):
            if "state" not in sample:
                continue
            state_present[batch_index] = True
            state_dim = int(sample["state"].shape[0])
            if state_dim:
                state[batch_index, -state_dim:] = sample["state"]
                state_padding_mask[batch_index, -state_dim:] = False
        collated["state"] = state
        collated["state_padding_mask"] = state_padding_mask
        collated["state_present"] = state_present
    if any("history_cached_embeds" in sample for sample in batch):
        max_cached_history = max((int(sample.get("history_cached_embeds", np.zeros((0,))).shape[0]) for sample in batch), default=0)
        embed_shape = _first_cached_embed_shape(batch)
        deepstack_shape = _first_cached_deepstack_shape(batch)
        history_cached_embeds = np.zeros(
            (len(batch), max_cached_history, *embed_shape),
            dtype=_first_array_dtype(batch, "history_cached_embeds", default=np.float16),
        )
        history_cached_deepstack_embeds = np.zeros(
            (len(batch), deepstack_shape[0], max_cached_history, *deepstack_shape[1:]),
            dtype=_first_array_dtype(batch, "history_cached_deepstack_embeds", default=np.float16),
        )
        history_cached_mask = np.zeros((len(batch), max_cached_history), dtype=bool)
        for batch_index, sample in enumerate(batch):
            cached = sample.get("history_cached_embeds")
            if cached is not None and cached.shape[0] > 0:
                history_cached_embeds[batch_index, : cached.shape[0]] = cached
            cached_deepstack = sample.get("history_cached_deepstack_embeds")
            if cached_deepstack is not None and cached_deepstack.shape[0] > 0 and cached_deepstack.shape[1] > 0:
                history_cached_deepstack_embeds[batch_index, : cached_deepstack.shape[0], : cached_deepstack.shape[1]] = cached_deepstack
            cached_mask = sample.get("history_cached_mask")
            if cached_mask is not None and cached_mask.shape[0] > 0:
                history_cached_mask[batch_index, : cached_mask.shape[0]] = cached_mask
        collated["history_cached_embeds"] = history_cached_embeds
        collated["history_cached_deepstack_embeds"] = history_cached_deepstack_embeds
        collated["history_cached_mask"] = history_cached_mask
    if any("history_images" in sample for sample in batch):
        history_cameras = sorted({camera for sample in batch for camera in sample.get("history_images", {})})
        collated["history_images"] = {
            camera: [sample.get("history_images", {}).get(camera, []) for sample in batch]
            for camera in history_cameras
        }
    return collated


def _read_video_frame(path: Path, frame_index: int) -> Image.Image:
    attempts = max(1, int(os.environ.get("NAVVLA_VIDEO_OPEN_RETRIES", "3")))
    retry_sleep = max(0.0, float(os.environ.get("NAVVLA_VIDEO_RETRY_SLEEP", "0.2")))
    last_error: Exception | None = None
    for attempt in range(attempts):
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                last_error = FileNotFoundError(f"video does not open: {path}")
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    last_error = IndexError(f"failed to read frame {frame_index} from {path}")
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(rgb)
        finally:
            cap.release()
        if attempt + 1 < attempts and retry_sleep:
            time.sleep(retry_sleep)
    assert last_error is not None
    raise last_error


def _read_parquet_shards(root: Path, *, columns: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    paths = sorted(root.glob("chunk-*/part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards found under {root}")
    frames = []
    for path in paths:
        read_columns = None
        if columns is not None:
            available = set(pq.read_schema(path).names)
            read_columns = [column for column in columns if column in available]
        frames.append(pd.read_parquet(path, columns=read_columns))
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


def _frame_index_from_history_ref(ref: str) -> int:
    parts = str(ref).split("/")
    if len(parts) == 3:
        frame_text = parts[1]
    elif len(parts) == 5:
        frame_text = parts[3]
    else:
        raise ValueError(f"history token ref must contain frame_index, got {ref!r}")
    if not frame_text.isdigit():
        raise ValueError(f"history token ref frame_index must be zero-padded digits, got {ref!r}")
    return int(frame_text)


def _episode_sample_indices(data: pd.DataFrame) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    ordered = data.reset_index(names="_dataset_position").sort_values(["episode_index", "frame_index", "index"], kind="stable")
    for episode_index, rows in ordered.groupby("episode_index", sort=True):
        groups[int(episode_index)] = rows["_dataset_position"].astype(int).tolist()
    return groups


def _pose4_from_observation_state(value: Any) -> np.ndarray:
    state = _float_array(value).reshape(-1)
    if state.shape[0] < 4:
        raise ValueError(f"NavVLA observation.state requires at least 4 values, got shape {state.shape}")
    return state[:4].astype(np.float32)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return list(value)


def _optional_source_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, float) and pd.isna(value):
        return {}
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return int(value)


def _optional_float(row: pd.Series | dict[str, Any], key: str) -> float | None:
    if key not in row:
        return None
    value = row[key]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


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


def _first_token_dtype(batch: list[dict[str, Any]]) -> np.dtype:
    for sample in batch:
        value = sample["history_tokens"]
        if value.shape[0] > 0:
            return np.dtype(value.dtype)
    return np.dtype(np.float32)


def _first_array_dtype(batch: list[dict[str, Any]], key: str, *, default: Any) -> np.dtype:
    for sample in batch:
        value = sample.get(key)
        if value is not None:
            return np.dtype(value.dtype)
    return np.dtype(default)


def _first_cached_embed_shape(batch: list[dict[str, Any]]) -> tuple[int, ...]:
    for sample in batch:
        value = sample.get("history_cached_embeds")
        if value is not None and value.shape[0] > 0:
            return tuple(value.shape[1:])
    return (4, 0)


def _first_cached_deepstack_shape(batch: list[dict[str, Any]]) -> tuple[int, ...]:
    for sample in batch:
        value = sample.get("history_cached_deepstack_embeds")
        if value is not None and value.shape[0] > 0:
            return (int(value.shape[0]), *tuple(value.shape[2:]))
    embed_shape = _first_cached_embed_shape(batch)
    return (0, *embed_shape)


def _history_length(sample: dict[str, Any]) -> int:
    return max(
        int(sample["history_tokens"].shape[0]),
        int(sample["history_tvi"].shape[0]),
        int(sample["history_mask"].shape[0]),
    )
