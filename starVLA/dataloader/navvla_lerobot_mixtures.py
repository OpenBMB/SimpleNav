"""
NavVLA LeRobot dataset mixture registry.

Each entry maps a mixture name to a list of:
    (dataset_root, sampling_weight)

Paths may be absolute or relative to ``datasets.vla_data.data_root_dir``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from starVLA.dataloader.navvla_lerobot_datasets import NavVLALeRobotDataset
from tool.navvla.statistics import write_dataset_statistics

_INDEX_DATASET_SHIFT = 48
_INDEX_LOCAL_MASK = (1 << _INDEX_DATASET_SHIFT) - 1

# {mixture_name: [(dataset_root, sampling_weight), ...]}
NAVVLA_NAMED_MIXTURES: dict[str, list[tuple[str, float]]] = {
    "traveluav_opentrackvla": [
        ("traveluav/vln_train", 1.0),
        ("opentrackvla", 1.0),
    ],
}


def resolve_navvla_path(path: str | Path, data_root_dir: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(data_root_dir) / candidate


def encode_navvla_mixture_index(dataset_index: int, local_index: int) -> int:
    if dataset_index < 0:
        raise ValueError(f"dataset_index must be non-negative, got {dataset_index}")
    if local_index < 0 or local_index >= (1 << _INDEX_DATASET_SHIFT):
        raise ValueError(f"local_index out of range for mixture encoding: {local_index}")
    return (int(dataset_index) << _INDEX_DATASET_SHIFT) | int(local_index)


def decode_navvla_mixture_index(encoded_index: int) -> tuple[int, int]:
    encoded = int(encoded_index)
    return encoded >> _INDEX_DATASET_SHIFT, encoded & _INDEX_LOCAL_MASK


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "off"}
    return bool(value)


def _navvla_dataset_kwargs(data_cfg: Any) -> dict[str, Any]:
    return {
        "split": data_cfg.get("split", "train"),
        "required_cameras": list(data_cfg.get("required_cameras", ["front", "left", "right", "rear"])),
        "image_resize": tuple(data_cfg.image_resize) if data_cfg.get("image_resize", None) is not None else None,
        "visual_token_mode": data_cfg.get("visual_token_mode", "online_images"),
        "max_history_frames": int(data_cfg.get("max_history_frames", 8)),
        "stop_distance_positive_m": float(data_cfg.get("stop_distance_positive_m", 3.0)),
        "stop_distance_negative_m": float(data_cfg.get("stop_distance_negative_m", 10.0)),
        "use_platform_text": _as_bool(data_cfg.get("use_platform_text", False)),
        "use_state_text": _as_bool(data_cfg.get("use_state_text", False)),
    }


def make_navvla_single_dataset(*, dataset_root: str | Path, data_cfg: Any) -> NavVLALeRobotDataset:
    return NavVLALeRobotDataset(dataset_root=dataset_root, **_navvla_dataset_kwargs(data_cfg))


class NavVLAMixtureDataset(Dataset):
    """Weighted mixture of NavVLA LeRobot datasets with per-dataset normalization."""

    def __init__(
        self,
        datasets: list[NavVLALeRobotDataset],
        weights: list[float],
        *,
        mixture_name: str = "navvla_mixture",
    ) -> None:
        if not datasets:
            raise ValueError("NavVLAMixtureDataset requires at least one dataset")
        if len(datasets) != len(weights):
            raise ValueError("datasets and weights must have the same length")

        self.datasets = list(datasets)
        self.mixture_name = str(mixture_name)
        self.dataset_weights = np.asarray(weights, dtype=np.float64)
        if np.any(self.dataset_weights <= 0):
            raise ValueError(f"NavVLA mixture weights must be positive, got {weights}")
        self.dataset_weights = self.dataset_weights / self.dataset_weights.sum()
        self.dataset_keys = [dataset.dataset_key for dataset in self.datasets]

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_index, local_index = decode_navvla_mixture_index(index)
        if dataset_index < 0 or dataset_index >= len(self.datasets):
            raise IndexError(
                f"mixture dataset_index={dataset_index} out of range for mixture {self.mixture_name!r}"
            )
        dataset = self.datasets[dataset_index]
        if local_index < 0 or local_index >= len(dataset):
            raise IndexError(
                f"local_index={local_index} out of range for dataset {dataset.root} (len={len(dataset)})"
            )
        sample = dataset[local_index]
        metadata = dict(sample.get("metadata", {}) or {})
        metadata["unnorm_key"] = dataset.dataset_key
        metadata["mixture_name"] = self.mixture_name
        metadata["mixture_dataset_index"] = int(dataset_index)
        metadata["mixture_dataset_root"] = str(dataset.root)
        sample["metadata"] = metadata
        return sample

    def save_dataset_statistics(self, save_path: str | Path) -> None:
        merged: dict[str, Any] = {}
        for dataset in self.datasets:
            merged.update(dataset.dataset_statistics)
        write_dataset_statistics(save_path, merged)


class NavVLAEpisodeMixtureSampler(Sampler[int]):
    """Episode-level weighted mixture sampler that yields encoded mixture indices."""

    def __init__(
        self,
        mixture_dataset: NavVLAMixtureDataset,
        *,
        shuffle: bool,
        seed: int = 0,
        balance_dataset_weights: bool = True,
        balance_episode_weights: bool = True,
    ) -> None:
        if not isinstance(mixture_dataset, NavVLAMixtureDataset):
            raise TypeError("NavVLAEpisodeMixtureSampler requires NavVLAMixtureDataset")
        self.mixture_dataset = mixture_dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.balance_dataset_weights = bool(balance_dataset_weights)
        self.balance_episode_weights = bool(balance_episode_weights)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.mixture_dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _episode_entries(self) -> list[tuple[int, int, list[int], float]]:
        entries: list[tuple[int, int, list[int], float]] = []
        for dataset_index, dataset in enumerate(self.mixture_dataset.datasets):
            dataset_weight = float(self.mixture_dataset.dataset_weights[dataset_index])
            if self.balance_dataset_weights:
                dataset_weight *= float(len(dataset))
            for episode_index, frame_indices in dataset.episode_sample_indices.items():
                if not frame_indices:
                    continue
                episode_weight = dataset_weight
                if self.balance_episode_weights:
                    episode_weight *= float(len(frame_indices))
                entries.append((dataset_index, int(episode_index), list(frame_indices), episode_weight))
        if not entries:
            raise ValueError("NavVLAEpisodeMixtureSampler found no non-empty episodes")
        return entries

    def _build_epoch_plan(self) -> list[tuple[int, int]]:
        entries = self._episode_entries()
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            scores = torch.rand(len(entries), generator=generator).numpy()
            weights = np.asarray([entry[3] for entry in entries], dtype=np.float64)
            weights = np.maximum(weights, 1e-8)
            keys = scores ** (1.0 / weights)
            order = np.argsort(-keys)
        else:
            order = np.arange(len(entries))

        plan: list[tuple[int, int]] = []
        for entry_index in order:
            dataset_index, _episode_index, frame_indices, _ = entries[int(entry_index)]
            plan.extend((dataset_index, int(local_index)) for local_index in frame_indices)
        return plan

    def __iter__(self):
        for dataset_index, local_index in self._build_epoch_plan():
            yield encode_navvla_mixture_index(dataset_index, local_index)


def build_navvla_mixture_dataset(data_cfg: Any) -> NavVLAMixtureDataset:
    data_mix = data_cfg.get("data_mix")
    if data_mix not in NAVVLA_NAMED_MIXTURES:
        raise KeyError(
            f"Unknown NavVLA data_mix={data_mix!r}; "
            f"available mixtures: {sorted(NAVVLA_NAMED_MIXTURES)}"
        )

    data_root_dir = Path(data_cfg.data_root_dir)
    mixture_spec = NAVVLA_NAMED_MIXTURES[data_mix]
    datasets: list[NavVLALeRobotDataset] = []
    weights: list[float] = []
    seen_roots: set[str] = set()

    for dataset_root, weight in mixture_spec:
        resolved_root = resolve_navvla_path(dataset_root, data_root_dir)
        root_key = str(resolved_root.resolve())
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)

        dataset = make_navvla_single_dataset(dataset_root=resolved_root, data_cfg=data_cfg)
        if len(dataset) == 0:
            raise ValueError(f"NavVLA mixture dataset is empty: {resolved_root}")
        datasets.append(dataset)
        weights.append(float(weight))

    if not datasets:
        raise ValueError(f"NavVLA mixture {data_mix!r} contains no datasets")

    return NavVLAMixtureDataset(
        datasets=datasets,
        weights=weights,
        mixture_name=str(data_mix),
    )


def build_navvla_dataset(data_cfg: Any) -> NavVLALeRobotDataset | NavVLAMixtureDataset:
    data_mix = data_cfg.get("data_mix")
    if data_mix in NAVVLA_NAMED_MIXTURES:
        return build_navvla_mixture_dataset(data_cfg)
    return make_navvla_single_dataset(dataset_root=data_cfg.data_root_dir, data_cfg=data_cfg)
