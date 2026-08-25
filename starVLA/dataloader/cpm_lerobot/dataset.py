from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

from starVLA.model.modules.bats import BATSSelectionResult, select_bats_history
from starVLA.model.modules.tvi import (
    LEARNED_TOKEN_TVI_MODE,
    TIME_YAW_TVI_MODE,
    get_tvi_input_dim,
    uses_camera_pose_tvi,
)
from tool.navvla.compute_bats_k import (
    history_frame_capacity,
    resolve_visual_wrapper_tokens,
    visual_block_token_cost,
)
from tool.navvla.context_index import DEFAULT_CONTEXT_TOKEN_BUDGET, resolve_context_index_paths
from tool.navvla.statistics import (
    body_frame_action_from_pose,
    build_repeated_state_statistics,
    normalize_values,
    read_dataset_statistics,
    write_dataset_statistics,
)
from tool.navvla.visual_token_cache import DEFAULT_MINICPM_V46_VISUAL_TOKEN_PROFILE

from .context import CompactRuntimeContextIndex
from .parquet import LazyParquetRows
from .sampler import EpisodeRange
from .utils import as_bool, as_list, float_array, optional_float, optional_int, pose4, read_parquet_shards
from .video import LazyVideoIndex, VideoReaderCache


ACTION_TASK_TYPES = {"navigation", "driving", "tracking"}
EPISODE_CACHE_SIZE = 32
CPM_TASK_COLUMNS = (
    "task_index",
    "task",
    "task_type",
    "task_subtype",
    "platform_text",
    "dataset_source",
    "answer",
)
CPM_NONEMPTY_TASK_COLUMNS = ("task", "task_type", "platform_text", "dataset_source")


class NavVLACPMDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path | None = None,
        *,
        root: str | Path | None = None,
        split: str = "train",
        dataset_statistics_key: str | None = None,
        checkpoint_statistics_key: str | None = None,
        video_backend: str = "opencv",
        required_cameras: list[str] | None = None,
        image_resize: tuple[int, int] | None = None,
        visual_token_mode: str = "cached_history_online_current",
        visual_token_profile: str = DEFAULT_MINICPM_V46_VISUAL_TOKEN_PROFILE,
        history_sampling_mode: str = "bats",
        max_online_history_frames: int | None = None,
        token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
        current_visual_tokens: int = 64,
        history_visual_tokens: int = 4,
        tvi_tokens: int = 1,
        current_wrapper_tokens: int | None = None,
        history_wrapper_tokens: int | None = None,
        bats_seed: int = 42,
        bats_epsilon: float = 0.1,
        bats_k: float = 4.0,
        use_dynamic_bats_k: bool = True,
        budget_num_cameras: int | None = None,
        include_state: bool = False,
        require_long_memory_tokens: bool = False,
        allow_missing_long_memory: bool = True,
        action_extra_dim_mode: str = "none",
        action_path_progress_gamma: float = 2.0,
        tvi_mode: str = TIME_YAW_TVI_MODE,
    ) -> None:
        self.tvi_mode = str(tvi_mode)
        self.tvi_dim = get_tvi_input_dim(self.tvi_mode)
        if video_backend != "opencv":
            raise ValueError(f"unsupported video_backend={video_backend!r}; only 'opencv' is implemented")
        mode = str(action_extra_dim_mode).strip().lower()
        if mode not in {"none", "path_progress"}:
            raise ValueError(f"unsupported action_extra_dim_mode={mode!r}")
        gamma = float(action_path_progress_gamma)
        if gamma <= 0:
            raise ValueError(f"action_path_progress_gamma must be positive, got {gamma}")

        if dataset_root is None:
            dataset_root = root
        elif root is not None and Path(dataset_root) != Path(root):
            raise ValueError(f"dataset_root={dataset_root!r} and root={root!r} refer to different paths")
        if dataset_root is None:
            raise ValueError("NavVLACPMDataset requires dataset_root or root")
        self.root = Path(dataset_root)
        self.split = str(split)
        self.required_cameras = None if required_cameras is None else [str(name) for name in required_cameras]
        self.image_resize = image_resize
        self.visual_token_mode = str(visual_token_mode).strip().lower()
        self.visual_token_profile = str(visual_token_profile).strip()
        if self.visual_token_mode not in {"cached_history_online_current", "online_images"}:
            raise ValueError(f"unsupported visual_token_mode={visual_token_mode!r}")
        if not self.visual_token_profile:
            raise ValueError("visual_token_profile must not be empty")
        self.history_sampling_mode = str(history_sampling_mode).strip().lower()
        if self.history_sampling_mode == "continuous":
            self.history_sampling_mode = "continuous_recent"
        if self.history_sampling_mode not in {"bats", "continuous_recent", "continuous_uniform"}:
            raise ValueError(f"unsupported history_sampling_mode={history_sampling_mode!r}")
        self.max_online_history_frames = (
            None if max_online_history_frames is None else int(max_online_history_frames)
        )
        if self.max_online_history_frames is not None and self.max_online_history_frames < 0:
            raise ValueError(
                "max_online_history_frames must be non-negative or None, "
                f"got {max_online_history_frames}"
            )
        self.token_budget = int(token_budget)
        self.current_visual_tokens = int(current_visual_tokens)
        self.history_visual_tokens = int(history_visual_tokens)
        self.tvi_tokens = int(tvi_tokens)
        self.current_wrapper_tokens, self.history_wrapper_tokens = resolve_visual_wrapper_tokens(
            visual_token_mode=self.visual_token_mode,
            visual_token_profile=self.visual_token_profile,
            current_wrapper_tokens=current_wrapper_tokens,
            history_wrapper_tokens=history_wrapper_tokens,
        )
        if min(self.token_budget, self.current_visual_tokens, self.history_visual_tokens) <= 0:
            raise ValueError("token_budget and visual token costs must be positive")
        if self.tvi_tokens < 0:
            raise ValueError(f"tvi_tokens must be non-negative, got {tvi_tokens}")
        self.bats_seed = int(bats_seed)
        self.bats_epsilon = float(bats_epsilon)
        if not 0.0 <= self.bats_epsilon < 1.0:
            raise ValueError(f"bats_epsilon must be in [0, 1), got {bats_epsilon}")
        self.bats_k = float(bats_k)
        if not np.isfinite(self.bats_k) or self.bats_k < 0.0:
            raise ValueError(f"bats_k must be finite and non-negative, got {bats_k}")
        self.use_dynamic_bats_k = as_bool(use_dynamic_bats_k)
        self.budget_num_cameras = None if budget_num_cameras is None else int(budget_num_cameras)
        if self.budget_num_cameras is not None and self.budget_num_cameras <= 0:
            raise ValueError(f"budget_num_cameras must be positive or None, got {budget_num_cameras}")
        self.include_state = as_bool(include_state)
        self.require_long_memory_tokens = as_bool(require_long_memory_tokens)
        self.allow_missing_long_memory = as_bool(allow_missing_long_memory)
        self.use_context_index = self.history_sampling_mode == "bats" and self.require_long_memory_tokens
        if self.history_sampling_mode != "bats" and self.require_long_memory_tokens:
            raise ValueError("only BATS history supports require_long_memory_tokens=True")
        self.action_extra_dim_mode = mode
        self.action_path_progress_gamma = gamma

        self.info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8"))
        self.cameras = json.loads((self.root / "meta" / "navvla_cameras.json").read_text(encoding="utf-8"))
        if self.required_cameras:
            missing = set(self.required_cameras) - set(self.cameras)
            if missing:
                raise KeyError(f"required cameras missing from metadata: {sorted(missing)}")

        self.data = LazyParquetRows(
            self.root / "data",
            columns=self._data_columns(),
            optional_columns=("source_frame_index",),
        )
        task_path = self.root / "meta" / "tasks.parquet"
        task_schema = pq.read_schema(task_path)
        missing_task_columns = set(CPM_TASK_COLUMNS) - set(task_schema.names)
        if missing_task_columns:
            raise ValueError(
                "meta/tasks.parquet is missing required CPM task columns: "
                f"{sorted(missing_task_columns)}"
            )
        if str(task_schema.field("task_index").type) != "int64":
            raise ValueError(
                "meta/tasks.parquet task_index must have dtype int64, "
                f"got {task_schema.field('task_index').type}"
            )
        episode_columns = ["episode_index", "episode_id", "trajectory_id", "scene_id", "length"]
        episodes = read_parquet_shards(self.root / "meta" / "episodes", columns=episode_columns)
        self.episodes = episodes.sort_values("episode_index", kind="stable").set_index("episode_index", drop=False)
        self._data_episode_ranges = _episode_ranges_from_metadata(self.episodes, data_length=len(self.data))
        self._episode_range_by_index = {value.episode_index: value for value in self._data_episode_ranges}
        self._episode_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

        self.source_dataset_key = f"{self.info.get('dataset_name', self.root.name)}_{self.split}"
        self.dataset_statistics_key = None if dataset_statistics_key is None else str(dataset_statistics_key)
        self.dataset_key = self.dataset_statistics_key or self.source_dataset_key
        self.checkpoint_statistics_key = (
            self.dataset_key if checkpoint_statistics_key is None else str(checkpoint_statistics_key)
        )
        if not self.checkpoint_statistics_key:
            raise ValueError("checkpoint_statistics_key must not be empty")
        self.dataset_statistics = self._load_dataset_statistics()
        self.context_index_paths = (
            resolve_context_index_paths(self.root, token_budget=self.token_budget) if self.use_context_index else None
        )
        self.runtime_context = (
            CompactRuntimeContextIndex(self.context_index_paths) if self.context_index_paths is not None else None
        )
        if self.runtime_context is not None and len(self.runtime_context) != len(self.data):
            raise ValueError(f"context rows={len(self.runtime_context)} do not match data rows={len(self.data)}")
        self._sample_indices = np.arange(len(self.data), dtype=np.int64)
        if self.require_long_memory_tokens and not self.allow_missing_long_memory:
            assert self.runtime_context is not None
            self._sample_indices = np.asarray(
                [
                    data_index
                    for data_index in range(len(self.data))
                    if int(self.runtime_context.meta[data_index]["long_memory_count"]) > 0
                ],
                dtype=np.int64,
            )
            if self._sample_indices.size == 0:
                raise ValueError("require_long_memory_tokens=True filtered out every sample")
        self.episode_ranges = _filtered_episode_ranges(self._data_episode_ranges, self._sample_indices)
        self.episode_sample_indices = {
            episode.episode_index: list(range(episode.start, episode.start + episode.length))
            for episode in self.episode_ranges
        }
        self._history_frame_counts: np.ndarray | None = None
        self._history_frame_count_sentinel: int | None = None
        self._history_frame_indices: dict[int, np.ndarray] = {}
        self._data_episode_starts = np.asarray(
            [episode.start for episode in self._data_episode_ranges],
            dtype=np.int64,
        )
        video_index_path = self.root / "meta" / "navvla_video_index.parquet"
        self.video_index = (
            LazyVideoIndex(
                video_index_path,
                video_keys=[str(camera["video_key"]) for camera in self.cameras.values()],
                data_length=len(self.data),
            )
            if video_index_path.exists()
            else None
        )
        self.task_rows = LazyParquetRows(
            task_path,
            columns=CPM_TASK_COLUMNS,
        )
        task_inventory = pq.read_table(
            task_path,
            columns=["task_index", *CPM_NONEMPTY_TASK_COLUMNS],
        ).to_pylist()
        task_indices = [int(row["task_index"]) for row in task_inventory]
        if len(task_indices) != len(set(task_indices)):
            raise ValueError("meta/tasks.parquet contains duplicate task_index values")
        for row in task_inventory:
            task_index = int(row["task_index"])
            for column in CPM_NONEMPTY_TASK_COLUMNS:
                value = row[column]
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"meta/tasks.parquet task_index={task_index} has empty or non-string {column}"
                    )
        self._task_row_by_index = {
            task_index: row_position for row_position, task_index in enumerate(task_indices)
        }
        self.video_readers = VideoReaderCache()

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
        if uses_camera_pose_tvi(self.tvi_mode):
            columns.extend(
                f"observation.camera_pose.{camera_name}" for camera_name in self.required_cameras or list(self.cameras)
            )
        return columns

    def __len__(self) -> int:
        return int(self._sample_indices.shape[0])

    def encode_sample_index(self, dataset_index: int, sample_index: int) -> int:
        if int(dataset_index) != 0:
            raise IndexError(f"single dataset only accepts dataset_index=0, got {dataset_index}")
        return int(sample_index)

    def history_frame_capacity_for_dataset(self, dataset_index: int) -> int:
        if int(dataset_index) != 0:
            raise IndexError(f"single dataset only accepts dataset_index=0, got {dataset_index}")
        if self.history_sampling_mode == "bats" and not self.use_dynamic_bats_k:
            raise ValueError(
                "length-bucketed BATS sampling requires use_dynamic_bats_k=True; "
                "fixed-k BATS does not have a deterministic clipped history length"
            )
        return self._max_history_steps(camera_count=self._budget_camera_count())

    def prepare_history_frame_counts(self) -> None:
        if self._history_frame_counts is not None:
            return

        max_history_steps = self.history_frame_capacity_for_dataset(0)
        dtype = _history_count_dtype(max_history_steps)
        self._history_frame_count_sentinel = int(np.iinfo(dtype).max)
        self._history_frame_counts = np.full(
            len(self),
            self._history_frame_count_sentinel,
            dtype=dtype,
        )

    def history_frame_counts(self, indices: list[int] | np.ndarray) -> np.ndarray:
        sample_indices = _normalize_sample_indices(indices, length=len(self))
        self.prepare_history_frame_counts()
        assert self._history_frame_counts is not None
        assert self._history_frame_count_sentinel is not None
        if not len(sample_indices):
            return np.asarray([], dtype=self._history_frame_counts.dtype)

        unresolved = sample_indices[
            self._history_frame_counts[sample_indices] == self._history_frame_count_sentinel
        ]
        if len(unresolved):
            self._compute_history_frame_counts(np.unique(unresolved))
        return self._history_frame_counts[sample_indices].copy()

    def history_frame_count(self, index: int) -> int:
        return int(self.history_frame_counts([int(index)])[0])

    def _compute_history_frame_counts(self, sample_indices: np.ndarray) -> None:
        assert self._history_frame_counts is not None
        data_indices = self._sample_indices[sample_indices]
        episode_positions = np.searchsorted(
            self._data_episode_starts,
            data_indices,
            side="right",
        ) - 1
        if np.any(episode_positions < 0):
            raise IndexError("sample data index does not resolve to an episode")

        max_history_steps = self.history_frame_capacity_for_dataset(0)
        if self.history_sampling_mode in {"continuous_recent", "continuous_uniform"}:
            starts = self._data_episode_starts[episode_positions]
            counts = np.minimum(data_indices - starts, max_history_steps)
            self._history_frame_counts[sample_indices] = counts
            return

        for episode_position in np.unique(episode_positions):
            episode = self._data_episode_ranges[int(episode_position)]
            in_episode = episode_positions == episode_position
            selected_samples = sample_indices[in_episode]
            selected_data_indices = data_indices[in_episode]
            if np.any(selected_data_indices >= int(episode.start) + int(episode.length)):
                raise IndexError(
                    f"sample data index does not resolve inside episode {episode.episode_index}"
                )
            frame_indices = self._frame_indices_for_episode(episode)
            episode_id = str(self.episodes.loc[episode.episode_index]["episode_id"])
            for sample_position, data_index in zip(
                selected_samples.tolist(),
                selected_data_indices.tolist(),
                strict=True,
            ):
                selection = self._select_bats_history(
                    episode_id=episode_id,
                    frame_indices=frame_indices,
                    anchor_position=int(data_index) - int(episode.start),
                )
                self._history_frame_counts[int(sample_position)] = len(selection.selected)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index = int(index)
        if sample_index < 0:
            sample_index += len(self)
        if sample_index < 0 or sample_index >= len(self):
            raise IndexError(f"sample index {index} outside length {len(self)}")
        data_index = int(self._sample_indices[sample_index])
        row = self.data[data_index]
        logical_index = int(row["index"])
        task = self._task(int(row["task_index"]))
        action_available = bool(row["sample.action_available"])
        if self.history_sampling_mode == "bats":
            context_row = self._online_bats_context(row, row_position=data_index)
            context_view = context_row
            if self.runtime_context is not None:
                long_memory_row = self.runtime_context.materialize_by_row_position(
                    data_index,
                    expected_index=logical_index,
                )
                long_memory_view = self._expand_bats_context(
                    long_memory_row,
                    row=row,
                    episode_index=int(row["episode_index"]),
                )
                context_view = {
                    **context_view,
                    "long_memory_steps": long_memory_view["long_memory_steps"],
                    "long_memory_blocks": long_memory_view["long_memory_blocks"],
                    "long_memory_token_refs": long_memory_view["long_memory_token_refs"],
                    "long_memory_mask": long_memory_view["long_memory_mask"],
                }
            context_key = str(context_view["context.index_key"])
            context_view = self._truncate_context(context_view)
        else:
            context_row = self._online_continuous_context(row, row_position=data_index)
            context_key = str(context_row["context.index_key"])
            context_view = context_row
        if self._history_frame_counts is not None:
            assert self._history_frame_count_sentinel is not None
            cached_history_count = int(self._history_frame_counts[sample_index])
            actual_history_count = len(context_view["history_steps"])
            if cached_history_count == self._history_frame_count_sentinel:
                self._history_frame_counts[sample_index] = actual_history_count
            elif cached_history_count != actual_history_count:
                raise ValueError(
                    "cached history frame count differs from materialized context: "
                    f"sample={sample_index} data_index={data_index} "
                    f"cached={cached_history_count} actual={actual_history_count}"
                )
        images = self._read_images(row, row_position=data_index)
        raw_state = pose4(row["observation.state"])
        history_refs = [str(ref) for ref in context_view["history_token_refs"]]
        action_padding_mask = np.asarray(as_list(row["action.padding_mask"]), dtype=bool)
        qa_target = task.get("answer")
        compute_flow_loss = bool(action_available and task["task_type"] in ACTION_TASK_TYPES)
        sample: dict[str, Any] = {
            "images": images,
            "current_tvi": self._current_tvi(row, float(row["timestamp"]), images),
            "history_tokens": np.zeros((0, 1, 3), dtype=np.float32),
            "history_tvi": self._context_tvi(
                context_view,
                prefix="history",
                episode_index=int(row["episode_index"]),
            ),
            "history_mask": np.asarray(context_view["history_mask"], dtype=bool),
            "lang": str(task["task"]),
            "platform_text": str(task.get("platform_text", "")),
            "action": self._normalized_action(row),
            "action_padding_mask": action_padding_mask,
            "distance_to_goal": self._distance_to_goal(row, raw_state),
            "qa_target": qa_target,
            "metadata": self._sample_metadata(
                row=row,
                task=task,
                context_key=context_key,
                context_row=context_row,
                context_view=context_view,
                history_refs=history_refs,
                raw_state=raw_state,
                compute_flow_loss=compute_flow_loss,
                qa_target=qa_target,
            ),
        }
        if self.include_state:
            sample["state"] = self._history_relative_state(row, context_view, raw_state)
        if self.require_long_memory_tokens:
            sample["long_memory_source_tvi"] = self._context_tvi(
                context_view,
                prefix="long_memory",
                episode_index=int(row["episode_index"]),
            )
        if self.visual_token_mode == "online_images":
            sample["history_images"] = self._read_history_images(
                context_view,
                episode_index=int(row["episode_index"]),
            )
        return sample

    def _max_history_steps(self, *, camera_count: int) -> int:
        return history_frame_capacity(
            token_budget=self._history_selection_token_budget(camera_count=camera_count),
            num_cameras=camera_count,
            current_visual_tokens=self.current_visual_tokens,
            history_visual_tokens=self.history_visual_tokens,
            tvi_tokens=self.tvi_tokens,
            current_wrapper_tokens=self.current_wrapper_tokens,
            history_wrapper_tokens=self.history_wrapper_tokens,
        )

    def _history_selection_token_budget(self, *, camera_count: int) -> int:
        if self.visual_token_mode != "online_images" or self.max_online_history_frames is None:
            return self.token_budget
        current_cost = int(camera_count) * visual_block_token_cost(
            visual_tokens=self.current_visual_tokens,
            tvi_tokens=self.tvi_tokens,
            wrapper_tokens=self.current_wrapper_tokens,
        )
        history_cost = int(camera_count) * visual_block_token_cost(
            visual_tokens=self.history_visual_tokens,
            tvi_tokens=self.tvi_tokens,
            wrapper_tokens=self.history_wrapper_tokens,
        )
        return min(
            self.token_budget,
            current_cost + self.max_online_history_frames * history_cost,
        )

    def _budget_camera_count(self) -> int:
        camera_count = len(self.required_cameras or self.cameras)
        if self.budget_num_cameras is not None and self.budget_num_cameras != camera_count:
            raise ValueError(
                "budget_num_cameras must match the number of selected cameras: "
                f"configured={self.budget_num_cameras}, selected={camera_count}"
            )
        return camera_count

    def _online_bats_context(self, row: dict[str, Any], *, row_position: int) -> dict[str, Any]:
        episode_index = int(row["episode_index"])
        episode = self.episodes.loc[episode_index]
        episode_id = str(episode["episode_id"])
        episode_payload = self._episode_payload(episode_index)
        frame_indices = episode_payload["frame_indices"]
        timestamps = episode_payload["timestamps"]
        anchor_frame_index = int(row["frame_index"])
        anchor_position = int(np.searchsorted(frame_indices, anchor_frame_index, side="left"))
        if anchor_position >= len(frame_indices) or int(frame_indices[anchor_position]) != anchor_frame_index:
            raise KeyError(
                f"current BATS frame does not resolve: episode={episode_index} frame={anchor_frame_index}"
            )
        episode_range = self._episode_range_by_index[episode_index]
        expected_row_position = int(episode_range.start) + anchor_position
        if expected_row_position != int(row_position):
            raise ValueError(
                f"episode frame order differs from data row order: episode={episode_index} "
                f"frame={anchor_frame_index} expected_row={expected_row_position} actual_row={row_position}"
            )
        selection = self._select_bats_history(
            episode_id=episode_id,
            frame_indices=frame_indices,
            anchor_position=anchor_position,
        )

        camera_names = list(self.required_cameras or self.cameras.keys())
        history_steps: list[dict[str, float | int]] = []
        history_blocks: list[dict[str, Any]] = []
        history_refs: list[str] = []
        history_mask: list[bool] = []
        for selected in selection.selected:
            frame_index = int(selected["frame_index"])
            frame_position = int(np.searchsorted(frame_indices, frame_index, side="left"))
            step_index = len(history_steps)
            history_steps.append(
                {"frame_index": frame_index, "timestamp": float(timestamps[frame_position])}
            )
            for camera_name in camera_names:
                history_blocks.append(
                    {
                        "step_index": step_index,
                        "camera_name": camera_name,
                        "frame_index": frame_index,
                    }
                )
                history_refs.append(f"{episode_id}/{frame_index:06d}/{camera_name}")
                history_mask.append(True)

        dataset_name = str(self.info.get("dataset_name", self.root.name))
        return {
            "context.index_key": (
                f"{dataset_name}/{self.split}/{episode_id}/f{anchor_frame_index:06d}/online-bats-v1"
            ),
            "index": int(row["index"]),
            "current_tvi_time": float(row["timestamp"]),
            "bats_k": float(selection.effective_k),
            "history_steps": history_steps,
            "history_blocks": history_blocks,
            "history_token_refs": history_refs,
            "history_mask": history_mask,
            "long_memory_steps": [],
            "long_memory_blocks": [],
            "long_memory_token_refs": [],
            "long_memory_mask": [],
        }

    def _select_bats_history(
        self,
        *,
        episode_id: str,
        frame_indices: np.ndarray,
        anchor_position: int,
    ) -> BATSSelectionResult:
        anchor_position = int(anchor_position)
        if anchor_position < 0 or anchor_position >= len(frame_indices):
            raise IndexError(
                f"BATS anchor position {anchor_position} outside episode length {len(frame_indices)}"
            )
        anchor_frame_index = int(frame_indices[anchor_position])
        candidates = [
            (int(frame_index), {"frame_index": int(frame_index)})
            for frame_index in frame_indices[:anchor_position].tolist()
        ]
        camera_count = self._budget_camera_count()
        return select_bats_history(
            candidates=candidates,
            anchor_frame_index=anchor_frame_index,
            episode_id=str(episode_id),
            dataset_name=str(self.info.get("dataset_name", self.root.name)),
            seed=self.bats_seed,
            epsilon=self.bats_epsilon,
            k=self.bats_k,
            use_dynamic_bats_k=self.use_dynamic_bats_k,
            token_budget=self._history_selection_token_budget(camera_count=camera_count),
            budget_num_cameras=camera_count,
            current_visual_tokens=self.current_visual_tokens,
            history_visual_tokens=self.history_visual_tokens,
            tvi_tokens=self.tvi_tokens,
            current_wrapper_tokens=self.current_wrapper_tokens,
            history_wrapper_tokens=self.history_wrapper_tokens,
            sampling_mode="priority_capped",
        )

    def _frame_indices_for_episode(self, episode: EpisodeRange) -> np.ndarray:
        cached = self._history_frame_indices.get(int(episode.episode_index))
        if cached is not None:
            return cached
        rows = self.data.read_range(
            int(episode.start),
            int(episode.length),
            columns=["episode_index", "frame_index"],
        )
        if any(int(row["episode_index"]) != int(episode.episode_index) for row in rows):
            raise ValueError(
                f"episode {episode.episode_index} data rows are not contiguous at start={episode.start}"
            )
        frame_indices = np.asarray([int(row["frame_index"]) for row in rows], dtype=np.int64)
        if len(frame_indices) > 1 and np.any(frame_indices[1:] <= frame_indices[:-1]):
            raise ValueError(
                f"episode {episode.episode_index} frame order differs from data row order"
            )
        self._history_frame_indices[int(episode.episode_index)] = frame_indices
        return frame_indices

    def _online_continuous_context(self, row: dict[str, Any], *, row_position: int) -> dict[str, Any]:
        episode_index = int(row["episode_index"])
        episode_range = self._episode_range_by_index[episode_index]
        local_position = int(row_position) - int(episode_range.start)
        if local_position < 0 or local_position >= int(episode_range.length):
            raise IndexError(
                f"row position {row_position} is outside episode={episode_index} "
                f"range [{episode_range.start}, {episode_range.start + episode_range.length})"
            )
        camera_names = list(self.required_cameras or self.cameras.keys())
        max_history_steps = self._max_history_steps(camera_count=self._budget_camera_count())
        available_positions = np.arange(int(episode_range.start), int(row_position), dtype=np.int64)
        if max_history_steps <= 0 or not len(available_positions):
            selected_positions = np.asarray([], dtype=np.int64)
        elif self.history_sampling_mode == "continuous_uniform" and len(available_positions) > max_history_steps:
            selected_positions = np.linspace(
                0,
                len(available_positions) - 1,
                num=max_history_steps,
                dtype=np.int64,
            )
            selected_positions = available_positions[np.unique(selected_positions)]
        else:
            selected_positions = available_positions[-max_history_steps:]
        episode = self.episodes.loc[episode_index]
        episode_id = str(episode["episode_id"])
        history_steps: list[dict[str, float]] = []
        history_blocks: list[dict[str, Any]] = []
        history_refs: list[str] = []
        history_mask: list[bool] = []
        for history_row_position in selected_positions.tolist():
            history_row = self.data[history_row_position]
            step_index = len(history_steps)
            history_steps.append({"timestamp": float(history_row["timestamp"])})
            for camera_name in camera_names:
                history_blocks.append(
                    {
                        "step_index": step_index,
                        "camera_name": camera_name,
                        "frame_index": int(history_row["frame_index"]),
                    }
                )
                history_refs.append(f"{episode_id}/{int(history_row['frame_index']):06d}/{camera_name}")
                history_mask.append(True)
        dataset_name = str(self.info.get("dataset_name", self.root.name))
        return {
            "context.index_key": (
                f"{dataset_name}/{self.split}/{episode_id}/"
                f"f{int(row['frame_index']):06d}/online-{self.history_sampling_mode}-v1"
            ),
            "index": int(row["index"]),
            "current_tvi_time": float(row["timestamp"]),
            "bats_k": None,
            "history_steps": history_steps,
            "history_blocks": history_blocks,
            "history_token_refs": history_refs,
            "history_mask": history_mask,
            "long_memory_steps": [],
            "long_memory_blocks": [],
            "long_memory_token_refs": [],
            "long_memory_mask": [],
        }

    def _expand_bats_context(
        self,
        context_row: dict[str, Any],
        *,
        row: dict[str, Any],
        episode_index: int,
    ) -> dict[str, Any]:
        episode = self.episodes.loc[int(episode_index)]
        episode_id = str(episode["episode_id"])
        episode_payload = self._episode_payload(int(episode_index))
        frame_indices = episode_payload["frame_indices"]
        timestamps = episode_payload["timestamps"]
        camera_names = [str(value) for value in context_row.get("camera_names", [])]

        def expand(prefix: str) -> tuple[list[dict[str, float | int]], list[dict[str, Any]], list[str], list[bool]]:
            steps: list[dict[str, float | int]] = []
            blocks: list[dict[str, Any]] = []
            refs: list[str] = []
            masks: list[bool] = []
            for selected in as_list(context_row[f"{prefix}_frames"]):
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
            "current_tvi_time": float(row["timestamp"]),
            "history_steps": history_steps,
            "history_blocks": history_blocks,
            "history_token_refs": history_refs,
            "history_mask": history_mask,
            "long_memory_steps": long_steps,
            "long_memory_blocks": long_blocks,
            "long_memory_token_refs": long_refs,
            "long_memory_mask": long_mask,
        }

    def _truncate_context(self, context_row: dict[str, Any]) -> dict[str, Any]:
        context_key = str(context_row.get("context.index_key", "<unknown>"))
        output: dict[str, Any] = {}
        for prefix, refs_key in (("history", "history_token_refs"), ("long_memory", "long_memory_token_refs")):
            fields = {
                "steps": f"{prefix}_steps",
                "blocks": f"{prefix}_blocks",
                "refs": refs_key,
                "mask": f"{prefix}_mask",
            }
            missing = [field for field in fields.values() if field not in context_row]
            if missing:
                if prefix == "long_memory" and self.allow_missing_long_memory:
                    output.update({fields["steps"]: [], fields["blocks"]: [], fields["refs"]: [], fields["mask"]: []})
                    continue
                raise KeyError(f"context {context_key} is missing {prefix} fields: {missing}")
            steps = as_list(context_row[fields["steps"]])
            blocks = as_list(context_row[fields["blocks"]])
            refs = as_list(context_row[fields["refs"]])
            masks = as_list(context_row[fields["mask"]])
            lengths = {fields["blocks"]: len(blocks), fields["refs"]: len(refs), fields["mask"]: len(masks)}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"context {context_key} has inconsistent {prefix} field lengths: {lengths}")
            kept_blocks: list[dict[str, Any]] = []
            kept_refs: list[str] = []
            kept_masks: list[bool] = []
            allowed = None if self.required_cameras is None else set(self.required_cameras)
            for block_index, block_value in enumerate(blocks):
                block = dict(block_value)
                if "step_index" not in block:
                    raise KeyError(f"context {context_key} {fields['blocks']}[{block_index}] is missing step_index")
                step_index = int(block["step_index"])
                if step_index < 0 or step_index >= len(steps):
                    raise IndexError(
                        f"context {context_key} {fields['blocks']}[{block_index}].step_index={step_index} "
                        f"does not resolve in {fields['steps']} length {len(steps)}"
                    )
                camera_name = str(block["camera_name"])
                if allowed is not None and camera_name not in allowed:
                    continue
                kept_blocks.append(block)
                kept_refs.append(str(refs[block_index]))
                kept_masks.append(bool(masks[block_index]))
            output.update(
                {
                    fields["steps"]: steps,
                    fields["blocks"]: kept_blocks,
                    fields["refs"]: kept_refs,
                    fields["mask"]: kept_masks,
                }
            )
        return self._truncate_history_to_online_budget(output)

    def _truncate_history_to_online_budget(self, context: dict[str, Any]) -> dict[str, Any]:
        max_history_steps = self._max_history_steps(camera_count=self._budget_camera_count())
        history_steps = as_list(context["history_steps"])
        if len(history_steps) <= max_history_steps:
            return context
        first_step = len(history_steps) - max_history_steps
        kept_blocks: list[dict[str, Any]] = []
        kept_refs: list[str] = []
        kept_masks: list[bool] = []
        for block, ref, mask in zip(
            as_list(context["history_blocks"]),
            as_list(context["history_token_refs"]),
            as_list(context["history_mask"]),
            strict=True,
        ):
            step_index = int(block["step_index"])
            if step_index < first_step:
                continue
            kept = dict(block)
            kept["step_index"] = step_index - first_step
            kept_blocks.append(kept)
            kept_refs.append(str(ref))
            kept_masks.append(bool(mask))
        return {
            **context,
            "history_steps": history_steps[first_step:],
            "history_blocks": kept_blocks,
            "history_token_refs": kept_refs,
            "history_mask": kept_masks,
        }

    def _sample_metadata(
        self,
        *,
        row: dict[str, Any],
        task: dict[str, Any],
        context_key: str,
        context_row: dict[str, Any],
        context_view: dict[str, Any],
        history_refs: list[str],
        raw_state: np.ndarray,
        compute_flow_loss: bool,
        qa_target: Any,
    ) -> dict[str, Any]:
        episode_index = int(row["episode_index"])
        episode = self.episodes.loc[episode_index]
        return {
            "task_type": task["task_type"],
            "task_subtype": task.get("task_subtype"),
            "dataset_source": task["dataset_source"],
            "dataset_root": str(self.root),
            "dataset_statistics_key": str(self.checkpoint_statistics_key),
            "checkpoint_statistics_key": str(self.checkpoint_statistics_key),
            "normalization_dataset_statistics_key": str(self.dataset_key),
            "source_dataset_statistics_key": str(self.source_dataset_key),
            "episode_index": episode_index,
            "episode_id": str(episode["episode_id"]) if "episode_id" in episode.index else None,
            "trajectory_id": str(episode["trajectory_id"]) if "trajectory_id" in episode.index else None,
            "scene_id": str(episode["scene_id"]) if "scene_id" in episode.index else None,
            "frame_index": int(row["frame_index"]),
            "source_frame_index": optional_int(row.get("source_frame_index")),
            "timestamp": float(row["timestamp"]),
            "control_frequency_hz": float(self.info["navvla"]["control_frequency_hz"]),
            "camera": self.cameras,
            "required_cameras": list(self.required_cameras or self.cameras.keys()),
            "context_index_key": context_key,
            "history_steps": context_view["history_steps"],
            "history_blocks": context_view["history_blocks"],
            "history_token_refs": history_refs,
            "long_memory_steps": context_view["long_memory_steps"],
            "long_memory_blocks": context_view["long_memory_blocks"],
            "long_memory_token_refs": context_view["long_memory_token_refs"],
            "long_memory_mask": context_view["long_memory_mask"],
            "bats_k": optional_float(context_row, "bats_k"),
            "bats_seed": self.bats_seed,
            "bats_epsilon": self.bats_epsilon,
            "bats_use_dynamic_k": self.use_dynamic_bats_k,
            "budget_num_cameras": self._budget_camera_count(),
            "token_count": len(history_refs),
            "visual_token_mode": self.visual_token_mode,
            "visual_token_profile": self.visual_token_profile,
            "history_sampling_mode": self.history_sampling_mode,
            "max_online_history_frames": self.max_online_history_frames,
            "history_visual_tokens": self.history_visual_tokens,
            "current_visual_tokens": self.current_visual_tokens,
            "tvi_tokens": self.tvi_tokens,
            "token_budget": self.token_budget,
            "context_index_path": (
                None if self.context_index_paths is None else str(self.context_index_paths.meta_path)
            ),
            "compute_flow_loss": compute_flow_loss,
            "compute_qa_loss": qa_target is not None,
            "raw_state": raw_state,
            "action_statistics": self._action_stats(),
            "history_state_padding_mask": np.zeros((len(context_view["history_steps"]),), dtype=bool),
            "action_extra_dim_mode": self.action_extra_dim_mode,
            "action_path_progress_gamma": float(self.action_path_progress_gamma),
        }

    def _read_images(self, row: dict[str, Any], *, row_position: int) -> dict[str, Image.Image]:
        images: dict[str, Image.Image] = {}
        for camera_name in self.required_cameras or list(self.cameras):
            image = self._read_camera_image(row, row_position=row_position, camera_name=camera_name)
            if image is not None:
                images[camera_name] = image
        return images

    def _read_history_images(
        self,
        context: dict[str, Any],
        *,
        episode_index: int,
    ) -> dict[str, list[Image.Image | None]]:
        step_count = len(as_list(context["history_steps"]))
        images: dict[str, list[Image.Image | None]] = {
            camera_name: [None] * step_count for camera_name in self.required_cameras or list(self.cameras)
        }
        episode_payload = self._episode_payload(episode_index)
        frame_indices = episode_payload["frame_indices"]
        episode_range = self._episode_range_by_index[int(episode_index)]
        for block, mask in zip(as_list(context["history_blocks"]), as_list(context["history_mask"]), strict=True):
            if not bool(mask):
                continue
            camera_name = str(block["camera_name"])
            if camera_name not in images:
                continue
            frame_index = int(block["frame_index"])
            frame_position = int(np.searchsorted(frame_indices, frame_index, side="left"))
            if frame_position >= len(frame_indices) or int(frame_indices[frame_position]) != frame_index:
                raise KeyError(f"history frame does not resolve: episode={episode_index} frame={frame_index}")
            row_position = int(episode_range.start) + frame_position
            row = self.data[row_position]
            images[camera_name][int(block["step_index"])] = self._read_camera_image(
                row,
                row_position=row_position,
                camera_name=camera_name,
            )
        return images

    def _read_camera_image(
        self,
        row: dict[str, Any],
        *,
        row_position: int,
        camera_name: str,
    ) -> Image.Image | None:
        video_key = self.cameras[camera_name]["video_key"]
        pattern = self.info["video_path"].get(video_key)
        if pattern is None:
            return None
        chunk_index = 0
        file_index = 0
        video_frame_index = int(row_position)
        if self.video_index is not None:
            video_row = self.video_index.get_by_row_position(
                row_position,
                video_key,
                expected_index=int(row["index"]),
            )
            if not bool(video_row["available"]):
                return None
            video_frame_index = int(video_row["video_frame_index"])
            chunk_index = int(video_row.get("chunk_index", 0))
            file_index = int(video_row.get("file_index", 0))
        image = self.video_readers.read(
            self.root / pattern.format(chunk_index=chunk_index, file_index=file_index),
            video_frame_index,
        )
        return image.resize(self.image_resize) if self.image_resize is not None else image

    def _current_tvi(
        self,
        row: dict[str, Any],
        timestamp: float,
        images: dict[str, Image.Image],
    ) -> np.ndarray:
        if self.tvi_mode == LEARNED_TOKEN_TVI_MODE:
            values = np.zeros((len(images), self.tvi_dim), dtype=np.float32)
        elif self.tvi_mode == TIME_YAW_TVI_MODE:
            values = [[float(timestamp), float(self.cameras[camera_name]["azimuth_rad"])] for camera_name in images]
        else:
            values = [
                [
                    float(timestamp),
                    *self._camera_pose(
                        row[f"observation.camera_pose.{camera_name}"],
                        camera_name=camera_name,
                        row_context=(
                            f"current row index={row.get('index', '<unknown>')} "
                            f"episode={row.get('episode_index', '<unknown>')} "
                            f"frame={row.get('frame_index', '<unknown>')}"
                        ),
                    ).tolist(),
                ]
                for camera_name in images
            ]
        return np.asarray(values, dtype=np.float32).reshape(-1, self.tvi_dim)

    def _context_tvi(
        self,
        context: dict[str, Any],
        *,
        prefix: str,
        episode_index: int,
    ) -> np.ndarray:
        blocks = as_list(context[f"{prefix}_blocks"])
        if self.tvi_mode == LEARNED_TOKEN_TVI_MODE:
            return np.zeros((len(blocks), self.tvi_dim), dtype=np.float32)
        steps = as_list(context[f"{prefix}_steps"])
        episode_payload = None
        if uses_camera_pose_tvi(self.tvi_mode) and blocks:
            for block in blocks:
                if "frame_index" not in block:
                    raise KeyError(f"{prefix}_blocks entries must include frame_index for camera-pose TVI lookup")
            episode_payload = self._episode_payload(episode_index)
        values = []
        for block in blocks:
            step = steps[int(block["step_index"])]
            if "timestamp" not in step:
                raise KeyError(f"{prefix}_steps entries must include timestamp for TVI lookup")
            camera_name = str(block["camera_name"])
            step_timestamp = float(step["timestamp"])
            if self.tvi_mode == TIME_YAW_TVI_MODE:
                values.append([step_timestamp, float(self.cameras[camera_name]["azimuth_rad"])])
                continue
            assert episode_payload is not None
            frame_index = int(block["frame_index"])
            frame_indices = episode_payload["frame_indices"]
            position = int(np.searchsorted(frame_indices, frame_index, side="left"))
            if position >= len(frame_indices) or int(frame_indices[position]) != frame_index:
                raise KeyError(
                    f"{prefix} frame does not resolve in current episode: episode={episode_index} frame={frame_index}"
                )
            cached_timestamp = float(episode_payload["timestamps"][position])
            if (
                not np.isfinite(step_timestamp)
                or not np.isfinite(cached_timestamp)
                or np.float32(cached_timestamp) != np.float32(step_timestamp)
            ):
                raise ValueError(
                    f"{prefix} timestamp mismatch for episode={episode_index} frame={frame_index}: "
                    f"context={step_timestamp} data={cached_timestamp}"
                )
            camera_poses = episode_payload["camera_poses"]
            if camera_name not in camera_poses:
                raise KeyError(f"camera pose cache is missing camera={camera_name!r} for episode={episode_index}")
            values.append([step_timestamp, *camera_poses[camera_name][position].tolist()])
        return np.asarray(values, dtype=np.float32).reshape(-1, self.tvi_dim)

    @staticmethod
    def _camera_pose(value: Any, *, camera_name: str, row_context: str) -> np.ndarray:
        try:
            pose = float_array(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"camera pose for {camera_name} at {row_context} must contain exactly 6 float values"
            ) from exc
        if pose.shape != (6,):
            raise ValueError(
                f"camera pose for {camera_name} at {row_context} must contain exactly 6 values, got shape {pose.shape}"
            )
        if not np.isfinite(pose).all():
            raise ValueError(f"camera pose for {camera_name} at {row_context} must contain only finite values")
        return pose.astype(np.float32)

    def _task(self, task_index: int) -> dict[str, Any]:
        try:
            row_position = self._task_row_by_index[int(task_index)]
        except KeyError as exc:
            raise KeyError(f"task_index={task_index} does not resolve in meta/tasks.parquet") from exc
        row = self.task_rows[row_position]
        if int(row["task_index"]) != int(task_index):
            raise ValueError(
                f"task physical row {row_position} contains task_index={row['task_index']}; "
                f"expected logical task_index={task_index}"
            )
        row.setdefault("task_subtype", None)
        row.setdefault("answer", None)
        return row

    def _load_dataset_statistics(self) -> dict[str, Any]:
        cache_path = self.root / "dataset_statistics.json"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"missing authoritative dataset statistics: {cache_path}; generate them during dataset conversion"
            )
        stats = read_dataset_statistics(cache_path)
        if self.dataset_statistics_key is None and self.dataset_key not in stats and len(stats) == 1:
            self.dataset_key = next(iter(stats))
        if self.dataset_key not in stats:
            raise KeyError(f"dataset_statistics.json does not contain {self.dataset_key!r}; keys={sorted(stats)}")
        block = stats[self.dataset_key]
        block.setdefault("action_mode", "anchor_relative_body_frame_xyz_yaw")
        block.setdefault("action_anchor", "current_frame_pose")
        block["state_mode"] = "variable_bats_history_relative_body_frame_actions"
        block.pop("state", None)
        block.pop("state_history_steps", None)
        return stats

    def save_dataset_statistics(self, save_path: str | Path) -> None:
        write_dataset_statistics(
            save_path,
            {self.checkpoint_statistics_key: self.dataset_statistics[self.dataset_key]},
        )

    def _action_stats(self) -> dict[str, Any]:
        return self.dataset_statistics[self.dataset_key]["action"]

    def _normalized_action(self, row: dict[str, Any]) -> np.ndarray:
        action = float_array(row["action"]).reshape(
            int(self.info["navvla"]["action_horizon"]),
            int(self.info["navvla"]["action_dim"]),
        )
        mask = np.asarray(as_list(row["action.padding_mask"]), dtype=bool)
        normalized = normalize_values(action, self._action_stats()).astype(np.float32)
        normalized[mask] = 0.0
        if self.action_extra_dim_mode == "none":
            return normalized
        return np.concatenate([normalized, self._action_path_progress(row, mask)], axis=-1).astype(np.float32)

    def _action_path_progress(self, row: dict[str, Any], mask: np.ndarray) -> np.ndarray:
        horizon = int(self.info["navvla"]["action_horizon"])
        episode_index = int(row["episode_index"])
        frame_index = int(row["frame_index"])
        future_frames = frame_index + np.arange(1, horizon + 1, dtype=np.int64)
        episode_payload = self._episode_payload(episode_index)
        frame_indices = episode_payload["frame_indices"]
        linear_values = episode_payload["linear_progress"]
        positions = np.searchsorted(frame_indices, future_frames, side="right") - 1
        progress = np.ones((horizon,), dtype=np.float32)
        valid = positions >= 0
        if valid.any():
            progress[valid] = linear_values[np.clip(positions[valid], 0, len(linear_values) - 1)]
        progress = np.power(np.clip(progress, 0.0, 1.0), self.action_path_progress_gamma).astype(np.float32)
        progress[mask] = 1.0
        return progress.reshape(-1, 1)

    def _history_relative_state(self, row: dict[str, Any], context: dict[str, Any], raw_state: np.ndarray) -> np.ndarray:
        history_steps = as_list(context["history_steps"])
        raw_chunks = np.zeros((len(history_steps), int(self.info["navvla"]["action_dim"])), dtype=np.float32)
        if not history_steps:
            return raw_chunks.reshape(-1)
        poses = [self._pose_for_history_step(int(row["episode_index"]), step) for step in history_steps] + [raw_state]
        chunks = np.stack(
            [body_frame_action_from_pose(poses[index - 1], poses[index]) for index in range(1, len(poses))],
            axis=0,
        )
        raw_chunks[-chunks.shape[0] :] = chunks
        state_stats = build_repeated_state_statistics(self._action_stats(), len(history_steps))
        normalized = normalize_values(raw_chunks.reshape(-1), state_stats).reshape(raw_chunks.shape)
        return normalized.reshape(-1).astype(np.float32)

    def _pose_for_history_step(self, episode_index: int, step: dict[str, Any]) -> np.ndarray:
        if "frame_index" in step:
            return self._pose_for_frame(episode_index, int(step["frame_index"]))
        if "timestamp" not in step:
            raise KeyError("history_steps entries must include timestamp for state lookup")
        lookup = self._episode_payload(episode_index)
        timestamps = lookup["timestamps"]
        position = int(np.searchsorted(timestamps, float(step["timestamp"]), side="left"))
        for candidate in (position, position - 1):
            if 0 <= candidate < len(timestamps) and abs(float(timestamps[candidate]) - float(step["timestamp"])) <= 1e-6:
                return np.asarray(lookup["states"][candidate], dtype=np.float32)
        raise KeyError(f"history timestamp does not resolve: episode={episode_index} timestamp={step['timestamp']}")

    def _pose_for_frame(self, episode_index: int, frame_index: int) -> np.ndarray:
        lookup = self._episode_payload(episode_index)
        frame_indices = lookup["frame_indices"]
        position = int(np.searchsorted(frame_indices, frame_index, side="left"))
        if position >= len(frame_indices) or int(frame_indices[position]) != frame_index:
            raise KeyError(f"history frame does not resolve: episode={episode_index} frame={frame_index}")
        return np.asarray(lookup["states"][position], dtype=np.float32)

    def _episode_payload(self, episode_index: int) -> dict[str, Any]:
        episode_index = int(episode_index)
        payload = self._episode_cache.pop(episode_index, None)
        if payload is not None:
            self._episode_cache[episode_index] = payload
            return payload
        try:
            episode = self._episode_range_by_index[episode_index]
        except KeyError as exc:
            raise KeyError(f"unknown episode_index={episode_index}") from exc
        columns = ["episode_index", "frame_index", "timestamp", "observation.state"]
        if uses_camera_pose_tvi(self.tvi_mode):
            columns.extend(
                f"observation.camera_pose.{camera_name}" for camera_name in self.required_cameras or list(self.cameras)
            )
        rows = self.data.read_range(
            episode.start,
            episode.length,
            columns=columns,
        )
        if any(int(row["episode_index"]) != episode_index for row in rows):
            raise ValueError(f"episode {episode_index} data rows are not contiguous at start={episode.start}")
        rows.sort(key=lambda row: (int(row["frame_index"]), float(row["timestamp"])))
        frame_indices = np.asarray([int(row["frame_index"]) for row in rows], dtype=np.int64)
        if len(frame_indices) > 1:
            duplicate_mask = frame_indices[1:] == frame_indices[:-1]
            if duplicate_mask.any():
                duplicate_frame = int(frame_indices[1:][duplicate_mask][0])
                raise ValueError(
                    f"duplicate frame_index in episode data: episode={episode_index} frame={duplicate_frame}"
                )
        timestamps = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
        states = np.stack([pose4(row["observation.state"]) for row in rows], axis=0)
        camera_poses: dict[str, np.ndarray] = {}
        if uses_camera_pose_tvi(self.tvi_mode):
            for camera_name in self.required_cameras or list(self.cameras):
                column = f"observation.camera_pose.{camera_name}"
                camera_poses[camera_name] = np.stack(
                    [
                        self._camera_pose(
                            row[column],
                            camera_name=camera_name,
                            row_context=(
                                f"cached data row episode={episode_index} "
                                f"frame={row['frame_index']} timestamp={row['timestamp']}"
                            ),
                        )
                        for row in rows
                    ],
                    axis=0,
                )
        positions = states[:, :3]
        if len(positions) <= 1:
            progress = np.ones((len(positions),), dtype=np.float32)
        else:
            cumulative = np.concatenate(
                [np.zeros((1,), dtype=np.float32), np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
            )
            total = float(cumulative[-1])
            progress = (
                np.clip(frame_indices.astype(np.float32) / float(max(int(frame_indices[-1]), 1)), 0.0, 1.0)
                if total <= 1e-6
                else np.clip(cumulative / total, 0.0, 1.0).astype(np.float32)
            )
        payload = {
            "frame_indices": frame_indices,
            "timestamps": timestamps,
            "states": states,
            "camera_poses": camera_poses,
            "goal_position": positions[-1].astype(np.float32),
            "linear_progress": progress.astype(np.float32),
        }
        self._episode_cache[episode_index] = payload
        while len(self._episode_cache) > EPISODE_CACHE_SIZE:
            self._episode_cache.popitem(last=False)
        return payload

    def _distance_to_goal(self, row: dict[str, Any], state: np.ndarray) -> float:
        goal = self._episode_payload(int(row["episode_index"]))["goal_position"]
        return float(np.linalg.norm(state[:3] - goal))


def _episode_ranges_from_metadata(episodes: pd.DataFrame, *, data_length: int) -> list[EpisodeRange]:
    output: list[EpisodeRange] = []
    start = 0
    for episode_index, row in episodes.iterrows():
        length = int(row["length"])
        if length <= 0:
            raise ValueError(f"episode {episode_index} has invalid length={length}")
        output.append(EpisodeRange(0, int(episode_index), start, length))
        start += length
    if start != int(data_length):
        raise ValueError(f"episode lengths sum to {start}, but data contains {data_length} rows")
    return output


def _filtered_episode_ranges(
    episode_ranges: list[EpisodeRange],
    sample_indices: np.ndarray,
) -> list[EpisodeRange]:
    output: list[EpisodeRange] = []
    for episode in episode_ranges:
        left = int(np.searchsorted(sample_indices, episode.start, side="left"))
        right = int(np.searchsorted(sample_indices, episode.start + episode.length, side="left"))
        if right > left:
            output.append(EpisodeRange(episode.dataset_index, episode.episode_index, left, right - left))
    return output


def _normalize_sample_indices(indices: list[int] | np.ndarray, *, length: int) -> np.ndarray:
    original = np.asarray(indices, dtype=np.int64).reshape(-1)
    normalized = original.copy()
    normalized[normalized < 0] += int(length)
    invalid_positions = np.flatnonzero((normalized < 0) | (normalized >= int(length)))
    if len(invalid_positions):
        invalid = int(original[int(invalid_positions[0])])
        raise IndexError(f"sample index {invalid} outside length {length}")
    return normalized


def _history_count_dtype(max_history_steps: int) -> np.dtype:
    maximum = int(max_history_steps)
    if maximum < np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if maximum < np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if maximum < np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)
