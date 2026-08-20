from __future__ import annotations

import io
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from starVLA.dataloader.airsim_utils import (
    action_normalization_modes,
    build_ego_relative_action_chunk,
    build_previous_step_action_chunk,
    build_stats,
    config_bool,
    normalize_array,
    repeated_action_modes,
    resize_image_tree,
    resolve_obs_image_size,
)

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("AirSimOpenFlyDataset requires pyarrow for parquet-backed image loading.") from exc


def collate_fn(batch):
    return batch


def _as_float_array(values: list[float], *, expected_dim: int, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (expected_dim,):
        raise ValueError(f"{field_name} must have shape ({expected_dim},), got {array.shape}")
    return array


@dataclass(frozen=True)
class _FrameRecord:
    frame_idx: int
    timestamp: float
    image_relpaths: tuple[str, ...]
    state: np.ndarray
    source_parquet: str | None
    source_image_path: str | None


@dataclass(frozen=True)
class _EpisodeRecord:
    episode_id: str
    instruction: str
    frames: list[_FrameRecord]


class EpisodeGroupedSampler(Sampler[int]):
    def __init__(self, episode_sample_indices: list[list[int]], *, shuffle: bool, seed: int = 0):
        self.episode_sample_indices = episode_sample_indices
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.episode_sample_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        episode_order = list(range(len(self.episode_sample_indices)))
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            episode_order = torch.randperm(len(self.episode_sample_indices), generator=generator).tolist()

        for episode_idx in episode_order:
            yield from self.episode_sample_indices[episode_idx]


class AirSimOpenFlyDataset(Dataset):
    """OpenFly AirSim dataset with trajectory shuffle and data-side normalization."""

    def __init__(self, cfg, split: str = "train"):
        self.cfg = cfg
        self.data_cfg = cfg.datasets.vla_data
        self.data_root = Path(self.data_cfg.data_root_dir)
        self.split = split
        self.dataset_key = self.data_cfg.get("data_mix", self.data_root.name)
        self.include_state = config_bool(self.data_cfg.get("include_state", True), True)
        self.normalize_action = config_bool(self.data_cfg.get("normalize_action", True), True)
        self.normalize_state = config_bool(self.data_cfg.get("normalize_state", True), True)
        self.include_terminal_stop_action = config_bool(
            self.data_cfg.get("include_terminal_stop_action", False),
            False,
        )
        self.action_type = self.data_cfg.get("action_type", "body_frame_xyz_yaw")
        self.base_dim = 4
        self.history_action_chunk_len = int(self.data_cfg.get("history_action_chunk_len", 2))
        self.history_image_frames = int(self.data_cfg.get("history_image_frames", 2))
        self.future_horizon = int(cfg.framework.action_model.action_horizon)
        self.action_dim = int(cfg.framework.action_model.action_dim)
        self.state_dim = int(cfg.framework.action_model.state_dim)
        self.obs_image_size = resolve_obs_image_size(self.data_cfg)
        self._parquet_image_cache: OrderedDict[str, dict[str, bytes]] = OrderedDict()
        self._max_cached_parquet_files = 1

        if self.action_dim != self.base_dim:
            raise ValueError(f"AirSimOpenFlyDataset expects action_dim={self.base_dim}, got {self.action_dim}")
        if self.future_horizon <= 0:
            raise ValueError(f"action_horizon must be > 0, got {self.future_horizon}")
        if self.history_action_chunk_len < 0:
            raise ValueError(f"history_action_chunk_len must be >= 0, got {self.history_action_chunk_len}")
        if self.history_image_frames < 0:
            raise ValueError(f"history_image_frames must be >= 0, got {self.history_image_frames}")
        expected_state_dim = self.history_action_chunk_len * self.base_dim if self.include_state else 0
        if self.include_state and self.state_dim != expected_state_dim:
            raise ValueError(
                f"AirSimOpenFlyDataset expects state_dim={expected_state_dim}, got {self.state_dim} "
                f"for history_action_chunk_len={self.history_action_chunk_len}"
            )

        self.episodes = self._load_split_episodes()
        self._balanced_keep_set = self._load_balanced_train_keep_set()
        self.sample_index = self._build_sample_index()
        self.episode_sample_indices = self._build_episode_sample_indices()
        self.dataset_statistics = self._load_or_compute_dataset_statistics()

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_idx, base_idx = self.sample_index[index]
        episode = self.episodes[episode_idx]
        images, _, _ = self._load_image_context(episode, base_idx)
        images = resize_image_tree(images, self.obs_image_size)
        raw_action = self._build_action_chunk(episode, base_idx)
        action = raw_action
        if self.normalize_action:
            action = normalize_array(action, self._action_stats())

        sample: dict[str, Any] = {
            "image": images,
            "lang": episode.instruction,
            "language": episode.instruction,
            "action": action.astype(np.float16),
        }
        if self.include_state:
            state = self._build_state_vector(episode, base_idx)
            if self.normalize_state:
                state = normalize_array(state, self._state_stats())
            sample["state"] = state[None, :].astype(np.float16)
        return sample

    def save_dataset_statistics(self, save_path: Path | str) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset_statistics, f, indent=2)

    def _action_stats(self) -> dict[str, Any]:
        return self.dataset_statistics[self.dataset_key]["action"]

    def _state_stats(self) -> dict[str, Any]:
        return self.dataset_statistics[self.dataset_key]["state"]

    def _load_or_compute_dataset_statistics(self) -> dict[str, dict[str, Any]]:
        cache_path = self.data_root / "dataset_statistics.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        dataset_statistics = self._compute_dataset_statistics()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(dataset_statistics, f, indent=2)
        return dataset_statistics

    def _load_split_episodes(self) -> list[_EpisodeRecord]:
        split_file = self.data_root / "splits" / f"{self.split}.txt"
        episodes_dir = self.data_root / "episodes"
        if split_file.exists():
            episode_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            episode_ids = sorted(path.stem for path in episodes_dir.glob("*.json"))
        if not episode_ids:
            raise FileNotFoundError(
                f"No episodes found for split '{self.split}' under {episodes_dir}. "
                f"Expected {split_file} or JSON files in episodes/."
            )

        episodes: list[_EpisodeRecord] = []
        for episode_id in episode_ids:
            episode_path = episodes_dir / f"{episode_id}.json"
            if not episode_path.exists():
                raise FileNotFoundError(f"Missing episode file: {episode_path}")
            payload = json.loads(episode_path.read_text(encoding="utf-8"))
            instruction = payload.get("instruction", "").strip()
            if not instruction:
                raise ValueError(f"Episode {episode_id} has empty instruction")
            frames = []
            for frame in payload.get("frames", []):
                frames.append(
                    _FrameRecord(
                        frame_idx=int(frame["frame_idx"]),
                        timestamp=float(frame.get("timestamp", frame["frame_idx"])),
                        image_relpaths=self._normalize_image_relpaths(
                            image_relpaths=frame.get("image_relpaths"),
                            image_relpath=frame.get("image_relpath"),
                        ),
                        state=_as_float_array(frame["state"], expected_dim=self.base_dim, field_name=f"{episode_id}.state"),
                        source_parquet=self._normalize_optional_string(frame.get("source_parquet")),
                        source_image_path=self._normalize_optional_string(frame.get("source_image_path")),
                    )
                )
            if len(frames) <= self.future_horizon:
                raise ValueError(
                    f"Episode {episode_id} has {len(frames)} frames, but needs more than "
                    f"{self.future_horizon} to produce one sample."
                )
            episodes.append(_EpisodeRecord(episode_id=episode_id, instruction=instruction, frames=frames))
        return episodes

    def _build_sample_index(self) -> list[tuple[int, int]]:
        sample_index: list[tuple[int, int]] = []
        for episode_idx, episode in enumerate(self.episodes):
            valid_base_count = len(episode.frames) - self.future_horizon
            base_indices = list(range(valid_base_count))
            terminal_idx = len(episode.frames) - 1
            if self.include_terminal_stop_action and terminal_idx not in base_indices:
                base_indices.append(terminal_idx)
            for base_idx in base_indices:
                if not self._frame_has_image(episode.frames[base_idx]):
                    continue
                is_terminal_stop = self._is_terminal_stop_sample(episode, base_idx)
                # Balanced keep files are generated from normal transitions; keep terminal stops per trajectory.
                if (
                    self._balanced_keep_set is not None
                    and not is_terminal_stop
                    and (episode.episode_id, base_idx) not in self._balanced_keep_set
                ):
                    continue
                sample_index.append((episode_idx, base_idx))
        if not sample_index:
            raise ValueError("No valid OpenFly training samples were generated.")
        return sample_index

    def _load_balanced_train_keep_set(self) -> set[tuple[str, int]] | None:
        if self.split != "train":
            return None
        keep_path = self.data_root / "balance" / "train_kept_samples.jsonl"
        if not keep_path.exists():
            return None
        keep_set: set[tuple[str, int]] = set()
        for line in keep_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            keep_set.add((str(payload["episode_id"]), int(payload["base_idx"])))
        return keep_set

    def _compute_dataset_statistics(self) -> dict[str, dict[str, Any]]:
        actions = np.stack(
            [self._build_action_chunk(self.episodes[episode_idx], base_idx) for episode_idx, base_idx in self.sample_index],
            axis=0,
        ).astype(np.float32)
        action_stats = build_stats(actions, action_normalization_modes(self.action_type), dim=self.action_dim)
        stats: dict[str, Any] = {
            "action": action_stats,
            "num_trajectories": len(self.episodes),
            "num_transitions": int(actions.shape[0]),
        }
        if self.include_state:
            # State chunks must stay on the same one-step action scale as the model targets.
            stats["state"] = self._build_state_stats_from_action_stats(action_stats)
        return {self.dataset_key: stats}

    def _build_state_stats_from_action_stats(self, action_stats: dict[str, Any]) -> dict[str, Any]:
        state_stats: dict[str, Any] = {}
        for key in ("mean", "std", "min", "max", "q01", "q99"):
            single_step = np.asarray(action_stats[key], dtype=np.float32).reshape(-1, self.base_dim)[0]
            state_stats[key] = np.tile(single_step, self.history_action_chunk_len).tolist()
        state_stats["normalization_modes"] = repeated_action_modes(self.history_action_chunk_len, self.action_type)
        state_stats["mask"] = [True] * self.state_dim
        state_stats["binary_mask"] = [False] * self.state_dim
        return state_stats

    def _build_episode_sample_indices(self) -> list[list[int]]:
        grouped_indices: list[list[int]] = [[] for _ in self.episodes]
        for dataset_index, (episode_idx, _) in enumerate(self.sample_index):
            grouped_indices[episode_idx].append(dataset_index)
        return [indices for indices in grouped_indices if indices]

    def _is_terminal_stop_sample(self, episode: _EpisodeRecord, base_idx: int) -> bool:
        return self.include_terminal_stop_action and base_idx == len(episode.frames) - 1

    def _build_action_chunk(self, episode: _EpisodeRecord, base_idx: int) -> np.ndarray:
        if self._is_terminal_stop_sample(episode, base_idx):
            return np.zeros((self.future_horizon, self.base_dim), dtype=np.float32)

        future_states = np.stack(
            [
                episode.frames[target_idx].state
                for target_idx in range(base_idx + 1, base_idx + 1 + self.future_horizon)
            ],
            axis=0,
        ).astype(np.float32)
        current_state = episode.frames[base_idx].state.astype(np.float32)
        if self.action_type == "next_state_xyz_yaw":
            return future_states
        if self.action_type in {"ego_relative_xyz_yaw", "body_frame_xyz_yaw"}:
            return build_previous_step_action_chunk(np.concatenate([current_state[None, :], future_states], axis=0))
        raise ValueError(f"Unsupported OpenFly action_type: {self.action_type}")

    def _build_state_vector(self, episode: _EpisodeRecord, base_idx: int) -> np.ndarray:
        history_action_chunks = np.zeros((self.history_action_chunk_len, self.base_dim), dtype=np.float32)
        if self.history_action_chunk_len == 0:
            return history_action_chunks.reshape(-1)

        chosen_indices = self._resolve_image_context_indices(episode, base_idx)
        selected_states = np.stack(
            [episode.frames[index].state.astype(np.float32) for index in chosen_indices],
            axis=0,
        ).astype(np.float32)
        if selected_states.shape[0] > 1:
            localized = build_previous_step_action_chunk(selected_states)
            localized = localized[-self.history_action_chunk_len :]
            history_action_chunks[-localized.shape[0] :] = localized
        return history_action_chunks.reshape(-1).astype(np.float32)

    def _load_image_context(self, episode: _EpisodeRecord, base_idx: int) -> tuple[list[Image.Image], list[str], list[int]]:
        chosen_indices = self._resolve_image_context_indices(episode, base_idx)
        history_count = len(chosen_indices) - 1
        if history_count <= 0:
            role_by_slot = ["current"]
        elif history_count == 1:
            role_by_slot = ["history_recent", "current"]
        else:
            role_by_slot = ["history_old"] * (history_count - 1) + ["history_recent", "current"]

        images: list[Image.Image] = []
        roles: list[str] = []
        for frame_index, role in zip(chosen_indices, role_by_slot, strict=True):
            loaded = self._load_frame_images(episode.frames[frame_index])
            images.extend(loaded)
            roles.extend([role] * len(loaded))
        return images, roles, chosen_indices

    def _resolve_image_context_indices(self, episode: _EpisodeRecord, base_idx: int) -> list[int]:
        candidate_positions = [
            index for index in range(base_idx) if self._frame_has_image(episode.frames[index])
        ]
        if self.data_cfg.get("keyframe_selection_mode", "openfly_action_change") == "openfly_action_change":
            if candidate_positions:
                motion_labels = self._build_history_motion_labels(episode, candidate_positions, base_idx)
                selected_positions = self._select_openfly_history_positions(
                    candidate_count=len(candidate_positions),
                    motion_labels=motion_labels,
                    current_pos=len(candidate_positions) - 1,
                )
                history_indices = [candidate_positions[pos] for pos in selected_positions]
            else:
                history_indices = []
            return history_indices[-self.history_image_frames :] + [base_idx]

        history_indices = candidate_positions[-self.history_image_frames :]
        return history_indices + [base_idx]

    def _build_history_motion_labels(
        self,
        episode: _EpisodeRecord,
        candidate_positions: list[int],
        base_idx: int,
    ) -> list[str]:
        if not candidate_positions:
            return []
        motion_labels: list[str] = []
        next_positions = candidate_positions[1:] + [base_idx]
        for current_index, next_index in zip(candidate_positions, next_positions, strict=True):
            delta = build_ego_relative_action_chunk(
                current_state=episode.frames[current_index].state.astype(np.float32),
                future_states=episode.frames[next_index].state.astype(np.float32)[None, :],
            )[0]
            motion_labels.append(self._classify_motion_delta(delta))
        return motion_labels

    @staticmethod
    def _classify_motion_delta(delta: np.ndarray) -> str:
        dx, dy, dz, dyaw = np.asarray(delta, dtype=np.float32)
        planar = float(np.hypot(dx, dy))
        yaw_mag = abs(float(dyaw))
        vertical_mag = abs(float(dz))
        if yaw_mag >= 0.2:
            return "turn_left" if dyaw > 0 else "turn_right"
        if vertical_mag > planar and vertical_mag > 1e-3:
            return "ascend" if dz > 0 else "descend"
        if planar > 1e-3:
            return "forward"
        return "slow_or_hover"

    @staticmethod
    def _find_first_action_change_position(motion_labels: list[str]) -> int:
        if len(motion_labels) < 2:
            return 0
        previous_label = motion_labels[0]
        for index, label in enumerate(motion_labels[1:], start=1):
            if label != previous_label:
                return index
        return 0

    def _select_openfly_history_positions(
        self,
        candidate_count: int,
        motion_labels: list[str],
        current_pos: int,
    ) -> list[int]:
        if candidate_count <= 1 or current_pos <= 0:
            return [0, 0]
        if current_pos == 1:
            return [0, 1]
        keypoint = self._find_first_action_change_position(motion_labels[: current_pos + 1])
        if keypoint == current_pos - 1:
            return [max(0, current_pos - 2), keypoint]
        if keypoint > 0:
            return [keypoint, current_pos - 1]
        return [max(0, current_pos - 2), current_pos - 1]

    @staticmethod
    def _normalize_image_relpaths(image_relpaths: Any, image_relpath: Any) -> tuple[str, ...]:
        if image_relpaths is not None:
            normalized = tuple(
                str(path).strip()
                for path in image_relpaths
                if path is not None and str(path).strip()
            )
            if normalized:
                return normalized
        if image_relpath is None:
            return ()
        normalized_single = str(image_relpath).strip()
        return (normalized_single,) if normalized_single else ()

    @staticmethod
    def _frame_has_image(frame: _FrameRecord) -> bool:
        return bool(frame.image_relpaths) or bool(frame.source_parquet and frame.source_image_path)

    def _load_images(self, image_relpaths: tuple[str, ...]) -> list[Image.Image]:
        if not image_relpaths:
            raise ValueError("Tried to load images from empty image_relpaths.")
        images = []
        for image_relpath in image_relpaths:
            image_path = self.data_root / image_relpath
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image file: {image_path}")
            images.append(Image.open(image_path).convert("RGB"))
        return images

    def _load_frame_images(self, frame: _FrameRecord) -> list[Image.Image]:
        if frame.image_relpaths:
            return self._load_images(frame.image_relpaths)
        if frame.source_parquet and frame.source_image_path:
            return [self._load_parquet_image(frame.source_parquet, frame.source_image_path)]
        raise ValueError("OpenFly frame has no image source.")

    def _load_parquet_image(self, parquet_path: str, image_path: str) -> Image.Image:
        if parquet_path not in self._parquet_image_cache:
            table = pq.read_table(parquet_path, columns=["image"])
            mapping: dict[str, bytes] = {}
            for row in table.to_pylist():
                image_info = row["image"]
                mapping[image_info["path"]] = image_info["bytes"]
            self._parquet_image_cache[parquet_path] = mapping
            self._parquet_image_cache.move_to_end(parquet_path)
            while len(self._parquet_image_cache) > self._max_cached_parquet_files:
                self._parquet_image_cache.popitem(last=False)
        else:
            self._parquet_image_cache.move_to_end(parquet_path)

        image_map = self._parquet_image_cache[parquet_path]
        if image_path not in image_map:
            raise FileNotFoundError(f"{image_path} not found in parquet image column for {parquet_path}")
        return Image.open(io.BytesIO(image_map[image_path])).convert("RGB")

    @staticmethod
    def _normalize_optional_string(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None
