from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from tool.navvla.statistics import write_dataset_statistics

from .dataset import NavVLACPMDataset
from .sampler import EpisodeRange


class NavVLACPMMixtureDataset(Dataset):
    def __init__(
        self,
        datasets: list[NavVLACPMDataset],
        *,
        mixture_name: str,
        dataset_statistics_keys: list[str],
        checkpoint_statistics_keys: list[str] | None = None,
    ) -> None:
        if not datasets:
            raise ValueError("NavVLACPMMixtureDataset requires at least one dataset")
        if len(datasets) != len(dataset_statistics_keys):
            raise ValueError("datasets and dataset_statistics_keys must have the same length")
        if checkpoint_statistics_keys is not None and len(datasets) != len(checkpoint_statistics_keys):
            raise ValueError("datasets and checkpoint_statistics_keys must have the same length")
        self.datasets = list(datasets)
        self.mixture_name = str(mixture_name)
        self.dataset_statistics_keys = [str(key) for key in dataset_statistics_keys]
        self.checkpoint_statistics_keys = [
            str(key)
            for key in (
                checkpoint_statistics_keys
                if checkpoint_statistics_keys is not None
                else dataset_statistics_keys
            )
        ]
        self._lengths = [len(dataset) for dataset in self.datasets]
        if any(length <= 0 for length in self._lengths):
            raise ValueError(f"all mixture datasets must be non-empty, got lengths={self._lengths}")
        self._offsets = np.cumsum([0] + self._lengths, dtype=np.int64)
        self.episode_ranges = [
            EpisodeRange(dataset_index, episode.episode_index, episode.start, episode.length)
            for dataset_index, dataset in enumerate(self.datasets)
            for episode in dataset.episode_ranges
        ]

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_index, sample_index = self._decode_index(index)
        sample = self.datasets[dataset_index][sample_index]
        metadata = dict(sample["metadata"])
        metadata.update(
            {
                "mixture": self.mixture_name,
                "mixture_dataset_index": int(dataset_index),
                "mixture_sample_index": int(sample_index),
                "mixture_dataset_root": str(self.datasets[dataset_index].root),
                "dataset_statistics_key": self.checkpoint_statistics_keys[dataset_index],
                "checkpoint_statistics_key": self.checkpoint_statistics_keys[dataset_index],
                "normalization_dataset_statistics_key": self.dataset_statistics_keys[dataset_index],
                "source_dataset_statistics_key": str(self.datasets[dataset_index].source_dataset_key),
            }
        )
        sample["metadata"] = metadata
        return sample

    def encode_sample_index(self, dataset_index: int, sample_index: int) -> int:
        encoded = self._encode_index(int(dataset_index), int(sample_index))
        if encoded is None:
            raise IndexError(f"dataset={dataset_index} sample={sample_index} does not resolve in mixture")
        return encoded

    def history_frame_capacity_for_dataset(self, dataset_index: int) -> int:
        dataset_index = int(dataset_index)
        if dataset_index < 0 or dataset_index >= len(self.datasets):
            raise IndexError(f"mixture dataset index {dataset_index} is out of range")
        return self.datasets[dataset_index].history_frame_capacity_for_dataset(0)

    def prepare_history_frame_counts(self) -> None:
        for dataset in self.datasets:
            dataset.prepare_history_frame_counts()

    def history_frame_count(self, index: int) -> int:
        return int(self.history_frame_counts([int(index)])[0])

    def history_frame_counts(self, indices: list[int] | np.ndarray) -> np.ndarray:
        encoded_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        output = np.empty(len(encoded_indices), dtype=np.int64)
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_position, encoded_index in enumerate(encoded_indices.tolist()):
            dataset_index, sample_index = self._decode_index(encoded_index)
            grouped.setdefault(dataset_index, []).append((output_position, sample_index))
        for dataset_index, entries in grouped.items():
            positions = np.asarray([position for position, _sample_index in entries], dtype=np.int64)
            sample_indices = np.asarray([sample_index for _position, sample_index in entries], dtype=np.int64)
            output[positions] = self.datasets[dataset_index].history_frame_counts(sample_indices)
        return output

    def _decode_index(self, index: int) -> tuple[int, int]:
        length = len(self)
        index = int(index)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(f"mixture sample index {index} out of range for length {length}")
        dataset_index = int(np.searchsorted(self._offsets, index, side="right") - 1)
        return dataset_index, int(index - self._offsets[dataset_index])

    def _encode_index(self, dataset_index: int, sample_index: int) -> int | None:
        if sample_index < 0 or sample_index >= self._lengths[dataset_index]:
            return None
        return int(self._offsets[dataset_index] + sample_index)

    def save_dataset_statistics(self, save_path: str | Path) -> None:
        combined: dict[str, Any] = {}
        for dataset_index, dataset in enumerate(self.datasets):
            requested_key = self.checkpoint_statistics_keys[dataset_index]
            statistics = dataset.dataset_statistics[dataset.dataset_key]
            if requested_key in combined and combined[requested_key] != statistics:
                raise ValueError(
                    f"checkpoint_statistics_key {requested_key!r} maps to inconsistent normalization statistics"
                )
            combined[requested_key] = statistics
        write_dataset_statistics(save_path, combined)
