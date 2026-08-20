from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import Any, Iterator

import numpy as np
from torch.utils.data import Sampler


ACTIVE_POOL_FACTOR = 8
FRAME_SHUFFLE_WINDOW = 16


@dataclass(frozen=True)
class EpisodeRange:
    dataset_index: int
    episode_index: int
    start: int
    length: int


@dataclass
class _ActiveEpisode:
    episode: EpisodeRange
    sample_indices: np.ndarray
    position: int = 0

    def take(self) -> int:
        value = int(self.sample_indices[self.position])
        self.position += 1
        return value

    @property
    def exhausted(self) -> bool:
        return self.position >= len(self.sample_indices)


class _ActivePoolEpisodeBatchSampler(Sampler[list[int]]):
    """Mix episodes globally while retaining bounded within-episode video locality."""

    def __init__(
        self,
        dataset: Any,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.episode_ranges = [
            EpisodeRange(
                dataset_index=int(value.dataset_index),
                episode_index=int(value.episode_index),
                start=int(value.start),
                length=int(value.length),
            )
            for value in dataset.episode_ranges
            if int(value.length) > 0
        ]
        if not self.episode_ranges:
            raise ValueError("dataset.episode_ranges must contain at least one non-empty episode")
        self._length_cache: dict[int, int] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._iter_batches()

    def __len__(self) -> int:
        cached = self._length_cache.get(self.epoch)
        if cached is None:
            cached = self._count_batches()
            self._length_cache[self.epoch] = cached
        return cached

    def _count_batches(self) -> int:
        rng = np.random.default_rng(self.seed + self.epoch)
        lengths = [episode.length for episode in self.episode_ranges]
        if self.shuffle:
            lengths = [lengths[index] for index in rng.permutation(len(lengths))]
        pending = deque(lengths)
        active = deque(pending.popleft() for _ in range(min(len(lengths), ACTIVE_POOL_FACTOR * self.batch_size)))
        batch_count = 0
        while active:
            batch_size = 0
            max_rounds = 1 if len(active) >= self.batch_size else 2
            for _round in range(max_rounds):
                round_size = len(active)
                for _ in range(round_size):
                    remaining = active.popleft() - 1
                    batch_size += 1
                    if remaining:
                        active.append(remaining)
                    elif pending:
                        active.append(pending.popleft())
                    if batch_size == self.batch_size:
                        break
                if batch_size == self.batch_size or not active:
                    break
            if batch_size == self.batch_size or not self.drop_last:
                batch_count += 1
        return batch_count

    def _iter_batches(self) -> Iterator[list[int]]:
        for batch in self._iter_indexed_batches():
            yield [index for index, _dataset_index, _episode_offset in batch]

    def _iter_indexed_batches(self) -> Iterator[list[tuple[int, int, int]]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        episodes = list(self.episode_ranges)
        if self.shuffle:
            episodes = [episodes[index] for index in rng.permutation(len(episodes))]
        pending = deque(episodes)
        active: deque[_ActiveEpisode] = deque()
        pool_size = min(len(episodes), ACTIVE_POOL_FACTOR * self.batch_size)
        for _ in range(pool_size):
            active.append(self._activate(pending.popleft(), rng))

        while active:
            batch: list[tuple[int, int, int]] = []
            max_rounds = 1 if len(active) >= self.batch_size else 2
            for _round in range(max_rounds):
                round_size = len(active)
                for _ in range(round_size):
                    state = active.popleft()
                    local_index = state.take()
                    batch.append(
                        (
                            int(self.dataset.encode_sample_index(state.episode.dataset_index, local_index)),
                            int(state.episode.dataset_index),
                            int(local_index - state.episode.start),
                        )
                    )
                    if state.exhausted:
                        if pending:
                            active.append(self._activate(pending.popleft(), rng))
                    else:
                        active.append(state)
                    if len(batch) == self.batch_size:
                        break
                if len(batch) == self.batch_size or not active:
                    break
            if batch and (len(batch) == self.batch_size or not self.drop_last):
                yield batch

    def _activate(self, episode: EpisodeRange, rng: np.random.Generator) -> _ActiveEpisode:
        indices = np.arange(episode.start, episode.start + episode.length, dtype=np.int64)
        if self.shuffle and len(indices) > 1:
            windows = [
                indices[start : start + FRAME_SHUFFLE_WINDOW].copy()
                for start in range(0, len(indices), FRAME_SHUFFLE_WINDOW)
            ]
            for window in windows:
                rng.shuffle(window)
            order = rng.permutation(len(windows))
            indices = np.concatenate([windows[index] for index in order])
        return _ActiveEpisode(episode=episode, sample_indices=indices)


class LengthBucketedEpisodeBatchSampler(Sampler[list[int]]):
    """Bucket active-pool samples by clipped history length across synchronized ranks."""

    def __init__(
        self,
        dataset: Any,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        drop_last: bool = False,
        bucket_width: int = 8,
        buffer_size: int = 1024,
        sync_group_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.bucket_width = int(bucket_width)
        self.buffer_size = int(buffer_size)
        self.sync_group_size = int(sync_group_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not self.shuffle:
            raise ValueError("length-bucketed sampling requires shuffle=True")
        if self.bucket_width <= 0:
            raise ValueError(f"bucket_width must be positive, got {bucket_width}")
        if self.sync_group_size <= 0:
            raise ValueError(f"sync_group_size must be positive, got {sync_group_size}")
        self.global_batch_size = self.batch_size * self.sync_group_size
        if self.buffer_size < self.global_batch_size:
            raise ValueError(
                "buffer_size must cover at least one synchronized global batch: "
                f"buffer_size={buffer_size}, global_batch_size={self.global_batch_size}"
            )
        prepare_history_counts = getattr(dataset, "prepare_history_frame_counts", None)
        history_count_getter = getattr(dataset, "history_frame_count", None)
        history_counts_getter = getattr(dataset, "history_frame_counts", None)
        if not callable(prepare_history_counts) or not callable(history_count_getter):
            raise TypeError(
                "length-bucketed sampling requires dataset.prepare_history_frame_counts() "
                "and dataset.history_frame_count()"
            )
        capacity_getter = getattr(dataset, "history_frame_capacity_for_dataset", None)
        dataset_indices = {int(value.dataset_index) for value in dataset.episode_ranges}
        self.history_frame_capacities = (
            {
                dataset_index: int(capacity_getter(dataset_index))
                for dataset_index in dataset_indices
            }
            if callable(capacity_getter)
            else {}
        )
        if any(value < 0 for value in self.history_frame_capacities.values()):
            raise ValueError(
                f"history frame capacities must be non-negative, got {self.history_frame_capacities}"
            )
        prepare_history_counts()
        self._history_frame_count = history_count_getter
        self._history_frame_counts = history_counts_getter if callable(history_counts_getter) else None
        self.epoch = 0
        self._source = _ActivePoolEpisodeBatchSampler(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seed,
            drop_last=False,
        )
        self._num_samples = sum(int(value.length) for value in dataset.episode_ranges)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._source.set_epoch(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch + 1_000_003)
        pending: dict[int, list[tuple[int, int]]] = {}
        source = (
            sample
            for batch in self._source._iter_indexed_batches()
            for sample in batch
        )
        while buffer := list(islice(source, self.buffer_size)):
            indices = [int(index) for index, _dataset_index, _episode_offset in buffer]
            if self._history_frame_counts is None:
                frame_counts = np.asarray(
                    [self._history_frame_count(index) for index in indices],
                    dtype=np.int64,
                ).reshape(-1)
            else:
                frame_counts = np.asarray(
                    self._history_frame_counts(indices),
                    dtype=np.int64,
                ).reshape(-1)
            if len(frame_counts) != len(indices):
                raise ValueError(
                    "dataset.history_frame_counts() returned an unexpected number of values: "
                    f"expected={len(indices)} actual={len(frame_counts)}"
                )
            for index, frame_count_value in zip(indices, frame_counts.tolist(), strict=True):
                frame_count = int(frame_count_value)
                if frame_count < 0:
                    raise ValueError(f"history frame count must be non-negative, got {frame_count}")
                bucket_id = frame_count // self.bucket_width
                pending.setdefault(bucket_id, []).append((int(index), frame_count))
            ready = self._take_ready_global_batches(pending, rng)
            rng.shuffle(ready)
            yield from self._yield_local_batches(ready)

        tail = [sample for bucket_id in sorted(pending) for sample in pending[bucket_id]]
        rng.shuffle(tail)
        tail.sort(key=lambda sample: sample[1])
        full_tail_size = (len(tail) // self.global_batch_size) * self.global_batch_size
        ready = [
            tail[start : start + self.global_batch_size]
            for start in range(0, full_tail_size, self.global_batch_size)
        ]
        for batch in ready:
            rng.shuffle(batch)
        rng.shuffle(ready)
        yield from self._yield_local_batches(ready)

        if not self.drop_last:
            remainder = tail[full_tail_size:]
            rng.shuffle(remainder)
            for start in range(0, len(remainder), self.batch_size):
                yield [index for index, _frame_count in remainder[start : start + self.batch_size]]

    def __len__(self) -> int:
        if self.drop_last:
            global_batches = self._num_samples // self.global_batch_size
            return global_batches * self.sync_group_size
        return (self._num_samples + self.batch_size - 1) // self.batch_size

    def _take_ready_global_batches(
        self,
        pending: dict[int, list[tuple[int, int]]],
        rng: np.random.Generator,
    ) -> list[list[tuple[int, int]]]:
        ready: list[list[tuple[int, int]]] = []
        for bucket_id in sorted(pending):
            samples = pending[bucket_id]
            rng.shuffle(samples)
            full_size = (len(samples) // self.global_batch_size) * self.global_batch_size
            ready.extend(
                samples[start : start + self.global_batch_size]
                for start in range(0, full_size, self.global_batch_size)
            )
            pending[bucket_id] = samples[full_size:]
        return ready

    def _yield_local_batches(
        self,
        global_batches: list[list[tuple[int, int]]],
    ) -> Iterator[list[int]]:
        for global_batch in global_batches:
            for start in range(0, len(global_batch), self.batch_size):
                yield [index for index, _frame_count in global_batch[start : start + self.batch_size]]
