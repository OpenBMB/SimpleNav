from __future__ import annotations

import copy
import json
import shutil
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from starVLA.dataloader.navvla_cpm_dataset import (
    NavVLACPMDataset,
    collate_navvla_cpm_batch,
)
from starVLA.dataloader.cpm_lerobot.collate import _collate_core, _pad_tvi
from conftest import tiny_navvla_episodes, tiny_navvla_spec
from tool.navvla.context_index import ContextIndexConfig, iter_context_refs, load_runtime_context_index, resolve_context_index_paths
from tool.navvla.lerobot_v3_writer import write_navvla_lerobot_dataset
from tool.navvla.schema import NavVLACameraSpec
from tool.navvla.visual_token_cache import (
    default_minicpm_v46_visual_token_profile,
    write_profile_mmap_npy_cache,
)


@dataclass(frozen=True)
class _SamplerEpisode:
    dataset_index: int
    episode_index: int
    start: int
    length: int


class _SamplerDataset:
    def __init__(self, lengths: list[list[int]]) -> None:
        self.episode_ranges = []
        self.history_count_requests: list[list[int]] = []
        self._offsets = np.cumsum([0] + [sum(values) for values in lengths], dtype=np.int64)
        for dataset_index, dataset_lengths in enumerate(lengths):
            start = 0
            for episode_index, length in enumerate(dataset_lengths):
                self.episode_ranges.append(
                    _SamplerEpisode(dataset_index, episode_index, start, int(length))
                )
                start += int(length)

    def encode_sample_index(self, dataset_index: int, sample_index: int) -> int:
        return int(self._offsets[dataset_index] + sample_index)

    def history_frame_capacity_for_dataset(self, dataset_index: int) -> int:
        if dataset_index < 0 or dataset_index >= len(self._offsets) - 1:
            raise IndexError(dataset_index)
        return 31

    def prepare_history_frame_counts(self) -> None:
        return None

    def history_frame_count(self, index: int) -> int:
        dataset_index = int(np.searchsorted(self._offsets, index, side="right") - 1)
        local_index = int(index - self._offsets[dataset_index])
        episodes = [item for item in self.episode_ranges if item.dataset_index == dataset_index]
        for episode in episodes:
            if episode.start <= local_index < episode.start + episode.length:
                return min(
                    local_index - episode.start,
                    self.history_frame_capacity_for_dataset(dataset_index),
                )
        raise AssertionError(index)

    def history_frame_counts(self, indices: list[int] | np.ndarray) -> np.ndarray:
        requested = [int(index) for index in indices]
        self.history_count_requests.append(requested)
        return np.asarray([self.history_frame_count(index) for index in requested], dtype=np.int64)

    def episode_for_index(self, index: int) -> tuple[int, int]:
        dataset_index = int(np.searchsorted(self._offsets, index, side="right") - 1)
        local_index = int(index - self._offsets[dataset_index])
        start = 0
        lengths = [item for item in self.episode_ranges if item.dataset_index == dataset_index]
        for episode in lengths:
            if start <= local_index < start + episode.length:
                return dataset_index, episode.episode_index
            start += episode.length
        raise AssertionError(index)


def test_length_bucketed_sampler_aligns_sync_groups_and_randomizes_epochs() -> None:
    from starVLA.dataloader.cpm_lerobot.sampler import LengthBucketedEpisodeBatchSampler

    dataset = _SamplerDataset([[32] * 8])
    sampler = LengthBucketedEpisodeBatchSampler(
        dataset,
        batch_size=4,
        shuffle=True,
        seed=17,
        drop_last=True,
        bucket_width=8,
        buffer_size=64,
        sync_group_size=2,
    )
    epoch_zero = list(sampler)
    assert epoch_zero == list(sampler)
    flattened = [index for batch in epoch_zero for index in batch]

    assert sorted(flattened) == list(range(256))
    assert len(flattened) == len(set(flattened))
    assert len(epoch_zero) == len(sampler) == 64
    for start in range(0, len(epoch_zero), sampler.sync_group_size):
        sync_group = epoch_zero[start : start + sampler.sync_group_size]
        bucket_ids = {
            dataset.history_frame_count(index) // sampler.bucket_width
            for batch in sync_group
            for index in batch
        }
        assert len(bucket_ids) == 1

    sampler.set_epoch(1)
    epoch_one = list(sampler)

    assert epoch_one != epoch_zero
    assert sorted(index for batch in epoch_one for index in batch) == list(range(256))


def test_length_bucketed_sampler_drops_only_one_global_tail() -> None:
    from starVLA.dataloader.cpm_lerobot.sampler import LengthBucketedEpisodeBatchSampler

    dataset = _SamplerDataset([[5, 7, 11, 12]])
    sampler = LengthBucketedEpisodeBatchSampler(
        dataset,
        batch_size=4,
        shuffle=True,
        seed=23,
        drop_last=True,
        bucket_width=8,
        buffer_size=16,
        sync_group_size=2,
    )
    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]

    assert len(batches) == len(sampler) == 8
    assert len(flattened) == 32
    assert len(flattened) == len(set(flattened))
    assert set(flattened).issubset(set(range(35)))


def test_length_bucketed_sampler_keeps_partial_buckets_across_buffers() -> None:
    from starVLA.dataloader.cpm_lerobot.sampler import LengthBucketedEpisodeBatchSampler

    dataset = _SamplerDataset([[12]])
    sampler = LengthBucketedEpisodeBatchSampler(
        dataset,
        batch_size=2,
        shuffle=True,
        seed=29,
        drop_last=True,
        bucket_width=8,
        buffer_size=6,
        sync_group_size=2,
    )
    ordered_buffers = [
        [
            (0, 0, 0),
            (1, 0, 1),
            (2, 0, 2),
            (3, 0, 3),
            (8, 0, 8),
            (9, 0, 9),
            (10, 0, 10),
            (11, 0, 11),
            (4, 0, 4),
            (5, 0, 5),
            (6, 0, 6),
            (7, 0, 7),
        ],
    ]
    sampler._source._iter_indexed_batches = lambda: iter(ordered_buffers)

    assert dataset.history_count_requests == []
    iterator = iter(sampler)
    batches = [next(iterator)]
    assert dataset.history_count_requests == [[0, 1, 2, 3, 8, 9]]
    batches.append(next(iterator))
    assert dataset.history_count_requests == [[0, 1, 2, 3, 8, 9]]
    batches.append(next(iterator))
    assert dataset.history_count_requests == [
        [0, 1, 2, 3, 8, 9],
        [10, 11, 4, 5, 6, 7],
    ]
    batches.extend(iterator)
    assert dataset.history_count_requests == [
        [0, 1, 2, 3, 8, 9],
        [10, 11, 4, 5, 6, 7],
    ]
    flattened = [index for batch in batches for index in batch]

    assert len(batches) == len(sampler) == 6
    assert sorted(flattened) == list(range(12))
    for start in range(0, len(batches), sampler.sync_group_size):
        bucket_ids = {
            dataset.history_frame_count(index) // sampler.bucket_width
            for batch in batches[start : start + sampler.sync_group_size]
            for index in batch
        }
        assert len(bucket_ids) == 1


def test_length_bucketed_sampler_crosses_buckets_only_in_epoch_tail() -> None:
    from starVLA.dataloader.cpm_lerobot.sampler import LengthBucketedEpisodeBatchSampler

    dataset = _SamplerDataset([[13, 3]])
    sampler = LengthBucketedEpisodeBatchSampler(
        dataset,
        batch_size=2,
        shuffle=True,
        seed=31,
        drop_last=True,
        bucket_width=8,
        buffer_size=4,
        sync_group_size=2,
    )

    batches = list(sampler)
    global_groups = [
        batches[start : start + sampler.sync_group_size]
        for start in range(0, len(batches), sampler.sync_group_size)
    ]
    bucket_sets = [
        {
            dataset.history_frame_count(index) // sampler.bucket_width
            for batch in group
            for index in batch
        }
        for group in global_groups
    ]

    assert len(batches) == len(sampler) == 8
    assert all(len(bucket_ids) == 1 for bucket_ids in bucket_sets[:-1])
    assert len(bucket_sets[-1]) == 2
    assert sorted(index for batch in batches for index in batch) == list(range(16))


def test_lazy_parquet_rows_reads_nested_values_with_bounded_row_group_cache(tmp_path: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.parquet import LazyParquetRows

    path = tmp_path / "rows.parquet"
    table = pa.table(
        {
            "index": list(range(12)),
            "action": [[[float(index), float(index + 1)]] for index in range(12)],
        }
    )
    pq.write_table(table, path, row_group_size=2)
    rows = LazyParquetRows(path, cache_size=4)

    assert rows[3] == {"index": 3, "action": [[3.0, 4.0]]}
    for index in (0, 2, 4, 6, 8, 10):
        assert rows[index]["index"] == index
    assert len(rows._row_group_cache) == 4


def test_compact_context_resolves_by_integer_data_index(tiny_navvla_dataset_root: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.context import CompactRuntimeContextIndex

    paths = resolve_context_index_paths(tiny_navvla_dataset_root, token_budget=1024)
    expected_context = load_runtime_context_index(paths)
    context = CompactRuntimeContextIndex(paths)
    assert not paths.refs_path.exists()

    for data_index in expected_context.meta["index"].astype(int).tolist():
        expected = expected_context.materialize_by_data_index(data_index)
        actual = context.materialize_by_data_index(data_index)
        for key in ("index", "bats_k", "history_frames", "long_memory_frames", "camera_names"):
            assert actual[key] == expected[key]


def test_compact_context_contains_only_frame_and_camera_arrays(tiny_navvla_dataset_root: Path) -> None:
    paths = resolve_context_index_paths(tiny_navvla_dataset_root, token_budget=1024)
    assert sorted(path.name for path in paths.arrays_path.glob("*.npy")) == [
        "history_camera_mask.npy",
        "history_frame_index.npy",
        "long_memory_camera_mask.npy",
        "long_memory_frame_index.npy",
    ]


def test_video_reader_cache_reuses_and_evicts_readers() -> None:
    from starVLA.dataloader.cpm_lerobot.video import VideoReaderCache

    class FakeCapture:
        instances: dict[str, "FakeCapture"] = {}

        def __init__(self, path: str) -> None:
            self.path = path
            self.position = 0
            self.released = False
            self.instances[path] = self

        def isOpened(self) -> bool:
            return True

        def set(self, _key: int, value: int) -> None:
            self.position = int(value)

        def read(self):
            self.position += 1
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    cache = VideoReaderCache(max_readers=12, capture_factory=FakeCapture)
    cache.read("video-0.mp4", 0)
    cache.read("video-0.mp4", 1)
    assert cache.open_count == 1
    for index in range(1, 13):
        cache.read(f"video-{index}.mp4", 0)

    assert len(cache._readers) == 12
    assert FakeCapture.instances["video-0.mp4"].released


def test_token_store_bounds_mmap_shards_and_dataset_does_not_own_store(
    tiny_navvla_dataset_root: Path,
    tmp_path: Path,
) -> None:
    from starVLA.dataloader.cpm_lerobot.cache import MiniCPMTokenStore

    _write_minicpm_profile_cache(tiny_navvla_dataset_root)
    dataset = NavVLACPMDataset(tiny_navvla_dataset_root)
    assert not hasattr(dataset, "token_store")

    store = MiniCPMTokenStore(tiny_navvla_dataset_root)
    paths = []
    for index in range(13):
        path = tmp_path / f"shard-{index}.npy"
        np.save(path, np.zeros((1, 4, 8), dtype=np.float16))
        paths.append(path)
        store._load_shard(path)
    assert len(store._mmap_shards) == 12
    assert str(paths[0]) not in store._mmap_shards


def test_cpm_online_images_loads_contiguous_history_without_visual_cache(
    tiny_navvla_dataset_root: Path,
) -> None:
    dataset = NavVLACPMDataset(
        tiny_navvla_dataset_root,
        visual_token_mode="online_images",
        history_sampling_mode="continuous",
        history_visual_tokens=4,
        current_visual_tokens=64,
        tvi_tokens=1,
        token_budget=1024,
        required_cameras=["front"],
        image_resize=(448, 448),
    )

    sample = dataset[2]
    batch = collate_navvla_cpm_batch([sample])

    assert sample["metadata"]["visual_token_mode"] == "online_images"
    assert sample["metadata"]["history_sampling_mode"] == "continuous_recent"
    assert sample["images"]["front"].size == (448, 448)
    assert len(sample["history_images"]["front"]) == 2
    assert all(image.size == (448, 448) for image in sample["history_images"]["front"])
    assert "history_cached_embeds" not in batch
    assert batch["history_images"]["front"][0] == sample["history_images"]["front"]


def test_cpm_online_continuous_history_does_not_require_context_index(
    tiny_navvla_dataset_root: Path,
) -> None:
    root = tiny_navvla_dataset_root
    (root / "meta/navvla_context_index_manifest.json").unlink()
    shutil.rmtree(root / "meta/context_index")
    shutil.rmtree(root / "cache/context_index_debug")

    dataset = NavVLACPMDataset(
        root,
        visual_token_mode="online_images",
        history_sampling_mode="continuous",
        history_visual_tokens=4,
        current_visual_tokens=64,
        tvi_tokens=1,
        token_budget=1024,
        required_cameras=["front"],
        image_resize=(448, 448),
        tvi_mode="time_yaw",
    )

    sample = dataset[2]

    assert dataset.context_index_paths is None
    assert dataset.runtime_context is None
    assert sample["metadata"]["context_index_path"] is None
    assert sample["metadata"]["context_index_key"].endswith("/online-continuous_recent-v1")
    assert sample["metadata"]["history_token_refs"] == [
        "episode-a/000000/front",
        "episode-a/000001/front",
    ]
    np.testing.assert_allclose(sample["history_tvi"], [[0.0, 0.0], [1.0, 0.0]])
    assert len(sample["history_images"]["front"]) == 2

    budget_limited_dataset = NavVLACPMDataset(
        root,
        visual_token_mode="online_images",
        history_sampling_mode="continuous",
        history_visual_tokens=4,
        current_visual_tokens=64,
        tvi_tokens=1,
        token_budget=76,
        required_cameras=["front"],
        image_resize=(448, 448),
        tvi_mode="time_yaw",
    )
    budget_limited_sample = budget_limited_dataset[2]
    assert budget_limited_sample["metadata"]["history_token_refs"] == ["episode-a/000001/front"]
    assert len(budget_limited_sample["history_images"]["front"]) == 1


def test_cpm_online_bats_history_does_not_require_context_index(
    tiny_navvla_dataset_root: Path,
) -> None:
    root = tiny_navvla_dataset_root
    (root / "meta/navvla_context_index_manifest.json").unlink()
    shutil.rmtree(root / "meta/context_index")
    shutil.rmtree(root / "cache/context_index_debug")

    dataset = NavVLACPMDataset(
        root,
        visual_token_mode="online_images",
        history_sampling_mode="bats",
        bats_k=0.0,
        use_dynamic_bats_k=False,
        token_budget=1024,
        required_cameras=["front"],
    )
    sample = dataset[2]

    assert dataset.context_index_paths is None
    assert dataset.runtime_context is None
    assert sample["metadata"]["context_index_path"] is None
    assert sample["metadata"]["context_index_key"].endswith("/f000002/online-bats-v1")
    assert sample["metadata"]["history_token_refs"] == [
        "episode-a/000000/front",
        "episode-a/000001/front",
    ]


def test_cpm_dynamic_bats_history_count_matches_materialized_context(
    tiny_navvla_dataset_root: Path,
) -> None:
    dataset = NavVLACPMDataset(
        tiny_navvla_dataset_root,
        visual_token_mode="online_images",
        history_sampling_mode="bats",
        use_dynamic_bats_k=True,
        token_budget=1024,
        required_cameras=["front"],
    )

    dataset.prepare_history_frame_counts()
    cached_counts = [dataset.history_frame_count(index) for index in range(len(dataset))]
    actual_counts = [
        len(dataset[index]["metadata"]["history_steps"])
        for index in range(len(dataset))
    ]

    assert cached_counts == actual_counts == [0, 1, 2]
    assert dataset.history_frame_capacity_for_dataset(0) > cached_counts[-1]
    assert dataset._history_frame_counts is not None
    assert dataset._history_frame_counts.dtype == np.uint8


def test_cpm_dynamic_bats_history_count_uses_actual_probability_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episodes = tiny_navvla_episodes(tmp_path / "images-dynamic-bats-counts")
    source_frame = episodes[0].frames[-1]
    frames = [
        replace(
            source_frame,
            frame_index=frame_index,
            timestamp=float(frame_index),
            source_frame_index=frame_index,
        )
        for frame_index in range(12)
    ]
    summary = write_navvla_lerobot_dataset(
        [replace(episodes[0], frames=frames)],
        output_root=tmp_path / "out-dynamic-bats-counts",
        spec=tiny_navvla_spec(dataset_name="dynamic_bats_counts"),
        overwrite=True,
        cache_workers=1,
        context_index_config=ContextIndexConfig(use_dynamic_bats_k=False, k=0.0),
    )
    dataset = NavVLACPMDataset(
        Path(summary["dataset_root"]),
        visual_token_mode="online_images",
        history_sampling_mode="bats",
        use_dynamic_bats_k=True,
        token_budget=84,
        required_cameras=["front"],
    )

    selected_anchors: list[int] = []
    original_select = dataset._select_bats_history

    def tracked_select(**kwargs):
        selected_anchors.append(int(kwargs["anchor_position"]))
        return original_select(**kwargs)

    monkeypatch.setattr(dataset, "_select_bats_history", tracked_select)

    dataset.prepare_history_frame_counts()
    assert selected_anchors == []
    assert dataset._history_frame_counts is not None
    assert dataset._history_frame_count_sentinel is not None
    assert np.all(dataset._history_frame_counts == dataset._history_frame_count_sentinel)

    cached_counts = dataset.history_frame_counts(np.arange(len(dataset))).tolist()
    assert selected_anchors == list(range(len(dataset)))
    assert dataset.history_frame_counts([6, 3, 6]).tolist() == [1, cached_counts[3], 1]
    assert selected_anchors == list(range(len(dataset)))
    actual_counts = [
        len(dataset[index]["metadata"]["history_steps"])
        for index in range(len(dataset))
    ]

    assert dataset.history_frame_capacity_for_dataset(0) == 2
    assert cached_counts == actual_counts
    assert cached_counts[6] == 1
    assert cached_counts[6] < dataset.history_frame_capacity_for_dataset(0)


def test_cpm_materialization_rejects_stale_history_count_cache(
    tiny_navvla_dataset_root: Path,
) -> None:
    dataset = NavVLACPMDataset(
        tiny_navvla_dataset_root,
        visual_token_mode="online_images",
        history_sampling_mode="bats",
        use_dynamic_bats_k=True,
        token_budget=1024,
        required_cameras=["front"],
    )
    dataset.prepare_history_frame_counts()
    assert dataset._history_frame_counts is not None
    dataset._history_frame_counts[2] = 1

    with pytest.raises(ValueError, match="cached history frame count differs"):
        dataset[2]


def test_cpm_bats_online_history_merges_compact_long_memory(
    tiny_navvla_dataset_root: Path,
) -> None:
    dataset = NavVLACPMDataset(
        tiny_navvla_dataset_root,
        visual_token_mode="online_images",
        history_sampling_mode="bats",
        bats_k=0.0,
        use_dynamic_bats_k=False,
        token_budget=1024,
        required_cameras=["front"],
        require_long_memory_tokens=True,
    )

    sample = dataset[2]

    assert sample["metadata"]["context_index_key"].endswith("/f000002/online-bats-v1")
    assert sample["metadata"]["history_token_refs"] == [
        "episode-a/000000/front",
        "episode-a/000001/front",
    ]
    assert sample["metadata"]["long_memory_token_refs"] == ["episode-a/000000/front"]


def test_cpm_online_uniform_continuous_history_ignores_context_and_spreads_frames(
    tiny_navvla_dataset_root: Path,
) -> None:
    dataset = NavVLACPMDataset(
        tiny_navvla_dataset_root,
        visual_token_mode="online_images",
        history_sampling_mode="continuous_uniform",
        history_visual_tokens=4,
        current_visual_tokens=64,
        tvi_tokens=1,
        token_budget=76,
        required_cameras=["front"],
    )

    sample = dataset[2]

    assert dataset.context_index_paths is None
    assert dataset.runtime_context is None
    assert sample["metadata"]["history_sampling_mode"] == "continuous_uniform"
    assert sample["metadata"]["history_token_refs"] == ["episode-a/000000/front"]


def test_bats_tvi_reads_updated_data_timestamps_without_rebuilding_context(
    tiny_navvla_dataset_root: Path,
) -> None:
    root = tiny_navvla_dataset_root
    context_path = resolve_context_index_paths(root, token_budget=1024).meta_path
    context_before = (context_path.stat().st_mtime_ns, context_path.read_bytes())
    data_path = next((root / "data").glob("chunk-*/part-*.parquet"))
    _replace_parquet_column(data_path, "timestamp", [0, 10, 20])

    sample = NavVLACPMDataset(
        root,
        visual_token_mode="online_images",
        history_sampling_mode="bats",
        required_cameras=["front"],
    )[2]

    np.testing.assert_allclose(sample["current_tvi"], [[20.0, 0.0]])
    np.testing.assert_allclose(sample["history_tvi"], [[0.0, 0.0], [10.0, 0.0]])
    assert (context_path.stat().st_mtime_ns, context_path.read_bytes()) == context_before


def _write_minicpm_profile_cache(root: Path) -> None:
    profile = default_minicpm_v46_visual_token_profile(encoder_ckpt="/tmp/minicpm")
    profile = profile.__class__(
        **{
            **profile.__dict__,
            "hidden_dim": 8,
            "dtype": "float16",
            "shard_size": 2,
        }
    )
    refs = sorted(set(iter_context_refs(root, token_budget=1024)))
    records = []
    for ref in refs:
        episode_id, frame_index, camera_name = ref.split("/", 2)
        token_value = float(int(frame_index) + 1)
        records.append(
            {
                "ref": ref,
                "image_embeds": np.full((4, 8), token_value, dtype=np.float16),
                "episode_id": episode_id,
                "trajectory_id": "traj-a",
                "frame_index": int(frame_index),
                "source_frame_index": int(frame_index),
                "data_index": int(frame_index),
                "camera_name": camera_name,
                "video_key": "front_image",
            }
        )
    write_profile_mmap_npy_cache(root, profile=profile, records=records)


def _build_tiny_cpm_root(tmp_path: Path, *, dataset_name: str) -> Path:
    summary = write_navvla_lerobot_dataset(
        tiny_navvla_episodes(tmp_path / f"images-{dataset_name}"),
        output_root=tmp_path / f"out-{dataset_name}",
        spec=tiny_navvla_spec(dataset_name=dataset_name),
        overwrite=True,
        cache_workers=1,
        context_index_config=ContextIndexConfig(
            use_dynamic_bats_k=False,
            k=0.0,
        ),
    )
    root = Path(summary["dataset_root"])
    _write_minicpm_profile_cache(root)
    return root


def _build_two_camera_cpm_root(tmp_path: Path, *, dataset_name: str) -> Path:
    episodes = tiny_navvla_episodes(tmp_path / f"images-{dataset_name}")
    front = episodes[0].cameras[0]
    down = NavVLACameraSpec(
        name="down",
        video_key="down_image",
        viewpoint_type="down",
        azimuth_rad=0.0,
    )
    frames = [
        replace(
            frame,
            media_paths={
                **frame.media_paths,
                down.video_key: frame.media_paths[front.video_key],
            },
        )
        for frame in episodes[0].frames
    ]
    summary = write_navvla_lerobot_dataset(
        [replace(episodes[0], frames=frames, cameras=[front, down])],
        output_root=tmp_path / f"out-{dataset_name}",
        spec=tiny_navvla_spec(dataset_name=dataset_name),
        overwrite=True,
        cache_workers=1,
        context_index_config=ContextIndexConfig(
            use_dynamic_bats_k=False,
            k=0.0,
        ),
    )
    return Path(summary["dataset_root"])


def _replace_parquet_column(path: Path, column: str, values: list[int]) -> None:
    table = pq.read_table(path)
    column_index = table.schema.get_field_index(column)
    field = table.schema.field(column_index)
    updated = table.set_column(column_index, field, pa.array(values, type=field.type))
    pq.write_table(updated, path)


def _add_camera_pose_columns(root: Path, poses_by_camera: dict[str, list[list[float]]]) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    for camera_name in poses_by_camera:
        info["features"][f"observation.camera_pose.{camera_name}"] = {
            "dtype": "float32",
            "names": ["x_world", "y_world", "z_world", "yaw_body", "roll_body", "pitch_body"],
            "shape": [6],
        }
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    offset = 0
    for data_path in sorted((root / "data").glob("chunk-*/part-*.parquet")):
        table = pq.read_table(data_path)
        row_count = table.num_rows
        updated = table
        for camera_name, poses in poses_by_camera.items():
            shard_values = poses[offset : offset + row_count]
            if len(shard_values) != row_count:
                raise AssertionError(f"camera {camera_name} has {len(poses)} poses for dataset rows")
            pose_type = (
                pa.list_(pa.float32(), 6) if all(len(pose) == 6 for pose in shard_values) else pa.list_(pa.float32())
            )
            field = pa.field(f"observation.camera_pose.{camera_name}", pose_type)
            updated = updated.append_column(field, pa.array(shard_values, type=pose_type))
        pq.write_table(updated, data_path)
        offset += row_count
    if any(len(poses) != offset for poses in poses_by_camera.values()):
        raise AssertionError(f"camera pose row counts do not match dataset length {offset}")


def _camera_pose_method_dataset() -> NavVLACPMDataset:
    dataset = NavVLACPMDataset.__new__(NavVLACPMDataset)
    dataset.tvi_mode = "time_camera_pose"
    dataset.tvi_dim = 7
    dataset.cameras = {
        "front": {"azimuth_rad": 0.0},
        "down": {"azimuth_rad": 0.0},
    }
    return dataset


def test_cpm_dataset_loads_exact_time_camera_pose_tvi_rows(tmp_path: Path) -> None:
    root = _build_tiny_cpm_root(tmp_path, dataset_name="camera_pose_tvi")
    front_poses = [
        [10.0, 11.0, 12.0, 0.1, 0.2, 0.3],
        [20.0, 21.0, 22.0, 0.4, 0.5, 0.6],
        [30.0, 31.0, 32.0, 0.7, 0.8, 0.9],
    ]
    _add_camera_pose_columns(root, {"front": front_poses})

    dataset = NavVLACPMDataset(
        root,
        tvi_mode="time_camera_pose",
        required_cameras=["front"],
        require_long_memory_tokens=True,
    )
    sample = dataset[2]

    assert dataset.tvi_dim == 7
    np.testing.assert_allclose(sample["current_tvi"], [[2.0, *front_poses[2]]])
    np.testing.assert_allclose(
        sample["history_tvi"],
        [[0.0, *front_poses[0]], [1.0, *front_poses[1]]],
    )
    np.testing.assert_allclose(sample["long_memory_source_tvi"], [[0.0, *front_poses[0]]])


def test_cpm_dataset_metric_camera_pose_mode_loads_exact_seven_dimensional_rows(tmp_path: Path) -> None:
    root = _build_tiny_cpm_root(tmp_path, dataset_name="metric_camera_pose_tvi")
    front_poses = [
        [10.0, 11.0, 12.0, 0.1, 0.2, 0.3],
        [20.0, 21.0, 22.0, 0.4, 0.5, 0.6],
        [30.0, 31.0, 32.0, 0.7, 0.8, 0.9],
    ]
    _add_camera_pose_columns(root, {"front": front_poses})

    dataset = NavVLACPMDataset(
        root,
        tvi_mode="metric_camera_pose",
        required_cameras=["front"],
        require_long_memory_tokens=True,
    )
    sample = dataset[2]

    assert dataset.tvi_dim == 7
    np.testing.assert_allclose(sample["current_tvi"], [[2.0, *front_poses[2]]])
    np.testing.assert_allclose(sample["history_tvi"], [[0.0, *front_poses[0]], [1.0, *front_poses[1]]])
    np.testing.assert_allclose(sample["long_memory_source_tvi"], [[0.0, *front_poses[0]]])


def test_cpm_dataset_camera_pose_tvi_empty_history_has_seven_columns(tmp_path: Path) -> None:
    root = _build_tiny_cpm_root(tmp_path, dataset_name="camera_pose_empty_history")
    _add_camera_pose_columns(
        root,
        {
            "front": [
                [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
                [4.0, 5.0, 6.0, 0.4, 0.5, 0.6],
                [7.0, 8.0, 9.0, 0.7, 0.8, 0.9],
            ]
        },
    )

    sample = NavVLACPMDataset(root, tvi_mode="time_camera_pose", required_cameras=["front"])[0]

    assert sample["history_tvi"].shape == (0, 7)


def test_cpm_dataset_default_tvi_remains_time_and_camera_azimuth(tmp_path: Path) -> None:
    root = _build_tiny_cpm_root(tmp_path, dataset_name="legacy_time_yaw")
    _add_camera_pose_columns(
        root,
        {
            "front": [
                [10.0, 11.0, 12.0, 0.1, 0.2, 0.3],
                [20.0, 21.0, 22.0, 0.4, 0.5, 0.6],
                [30.0, 31.0, 32.0, 0.7, 0.8, 0.9],
            ]
        },
    )

    dataset = NavVLACPMDataset(root, required_cameras=["front"])
    sample = dataset[2]

    assert dataset.tvi_mode == "time_yaw"
    assert dataset.tvi_dim == 2
    np.testing.assert_allclose(sample["current_tvi"], [[2.0, 0.0]])
    np.testing.assert_allclose(sample["history_tvi"], [[0.0, 0.0], [1.0, 0.0]])


def test_cpm_dataset_learned_token_tvi_uses_zero_placeholders(tmp_path: Path) -> None:
    root = _build_tiny_cpm_root(tmp_path, dataset_name="learned_token_tvi")
    dataset = NavVLACPMDataset(root, tvi_mode="learned_token", required_cameras=["front"])

    sample = dataset[2]

    assert dataset.tvi_dim == 2
    np.testing.assert_array_equal(sample["current_tvi"], np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(sample["history_tvi"], np.zeros((2, 2), dtype=np.float32))


def test_learned_token_context_tvi_does_not_read_time_or_camera_pose() -> None:
    dataset = NavVLACPMDataset.__new__(NavVLACPMDataset)
    dataset.tvi_mode = "learned_token"
    dataset.tvi_dim = 2
    context = {
        "history_blocks": [
            {"step_index": 99, "camera_name": "front"},
            {"step_index": 100, "camera_name": "left"},
        ],
    }

    history_tvi = dataset._context_tvi(context, prefix="history", episode_index=7)

    np.testing.assert_array_equal(history_tvi, np.zeros((2, 2), dtype=np.float32))


def test_camera_pose_tvi_methods_preserve_present_image_and_context_block_order() -> None:
    dataset = _camera_pose_method_dataset()
    row = {
        "episode_index": 4,
        "frame_index": 8,
        "observation.camera_pose.front": [10.0, 11.0, 12.0, 0.1, 0.2, 0.3],
        "observation.camera_pose.down": [20.0, 21.0, 22.0, 0.4, 0.5, 0.6],
    }
    images = OrderedDict([("down", object()), ("front", object())])
    payload = {
        "frame_indices": np.asarray([3, 7], dtype=np.int64),
        "timestamps": np.asarray([0.5, 1.5], dtype=np.float64),
        "camera_poses": {
            "front": np.asarray([[30.0, 31.0, 32.0, 0.7, 0.8, 0.9], [40.0, 41.0, 42.0, 1.0, 1.1, 1.2]]),
            "down": np.asarray([[50.0, 51.0, 52.0, 1.3, 1.4, 1.5], [60.0, 61.0, 62.0, 1.6, 1.7, 1.8]]),
        },
    }
    payload_calls: list[int] = []

    def episode_payload(episode_index: int) -> dict[str, object]:
        payload_calls.append(episode_index)
        return payload

    dataset._episode_payload = episode_payload
    context = {
        "history_steps": [{"timestamp": 1.5}, {"timestamp": 0.5}],
        "history_blocks": [
            {"step_index": 0, "camera_name": "down", "frame_index": 7},
            {"step_index": 1, "camera_name": "front", "frame_index": 3},
        ],
    }

    current_tvi = dataset._current_tvi(row, 2.5, images)
    history_tvi = dataset._context_tvi(context, prefix="history", episode_index=4)

    np.testing.assert_allclose(
        current_tvi,
        [[2.5, *row["observation.camera_pose.down"]], [2.5, *row["observation.camera_pose.front"]]],
    )
    np.testing.assert_allclose(
        history_tvi,
        [[1.5, *payload["camera_poses"]["down"][1]], [0.5, *payload["camera_poses"]["front"][0]]],
    )
    assert payload_calls == [4]


def test_cpm_dataset_rejects_unknown_tvi_mode_before_reading_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported TVI mode"):
        NavVLACPMDataset(tmp_path / "missing", tvi_mode="unknown")


def test_cpm_dataset_requires_selected_camera_pose_column_in_seven_dimensional_mode(
    tiny_navvla_dataset_root: Path,
) -> None:
    with pytest.raises(ValueError, match=r"observation\.camera_pose\.front"):
        NavVLACPMDataset(
            tiny_navvla_dataset_root,
            tvi_mode="time_camera_pose",
            required_cameras=["front"],
        )


def test_cpm_dataset_rejects_camera_pose_with_wrong_length(tmp_path: Path) -> None:
    root = _build_tiny_cpm_root(tmp_path, dataset_name="camera_pose_wrong_length")
    _add_camera_pose_columns(
        root,
        {
            "front": [
                [1.0, 2.0, 3.0, 0.1, 0.2],
                [4.0, 5.0, 6.0, 0.4, 0.5, 0.6],
                [7.0, 8.0, 9.0, 0.7, 0.8, 0.9],
            ]
        },
    )
    dataset = NavVLACPMDataset(root, tvi_mode="time_camera_pose", required_cameras=["front"])

    with pytest.raises(ValueError, match=r"front.*exactly 6"):
        dataset[0]


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_cpm_dataset_rejects_non_finite_camera_pose(tmp_path: Path, invalid_value: float) -> None:
    root = _build_tiny_cpm_root(tmp_path, dataset_name=f"camera_pose_non_finite_{invalid_value}")
    _add_camera_pose_columns(
        root,
        {
            "front": [
                [invalid_value, 2.0, 3.0, 0.1, 0.2, 0.3],
                [4.0, 5.0, 6.0, 0.4, 0.5, 0.6],
                [7.0, 8.0, 9.0, 0.7, 0.8, 0.9],
            ]
        },
    )
    dataset = NavVLACPMDataset(root, tvi_mode="time_camera_pose", required_cameras=["front"])

    with pytest.raises(ValueError, match=r"front.*finite"):
        dataset[0]


def test_camera_pose_context_tvi_requires_block_frame_index() -> None:
    dataset = _camera_pose_method_dataset()
    context = {
        "history_steps": [{"timestamp": 0.0}],
        "history_blocks": [{"step_index": 0, "camera_name": "front"}],
    }

    with pytest.raises(KeyError, match="frame_index"):
        dataset._context_tvi(context, prefix="history", episode_index=2)


def test_camera_pose_context_tvi_does_not_fall_back_outside_current_episode() -> None:
    dataset = _camera_pose_method_dataset()
    episode_payloads = {
        2: {
            "frame_indices": np.asarray([0], dtype=np.int64),
            "timestamps": np.asarray([0.0], dtype=np.float64),
            "camera_poses": {"front": np.zeros((1, 6), dtype=np.float32)},
        },
        3: {
            "frame_indices": np.asarray([9], dtype=np.int64),
            "timestamps": np.asarray([0.0], dtype=np.float64),
            "camera_poses": {"front": np.ones((1, 6), dtype=np.float32)},
        },
    }
    dataset._episode_payload = lambda episode_index: episode_payloads[episode_index]
    context = {
        "history_steps": [{"timestamp": 0.0}],
        "history_blocks": [{"step_index": 0, "camera_name": "front", "frame_index": 9}],
    }

    with pytest.raises(KeyError, match=r"episode=2.*frame=9"):
        dataset._context_tvi(context, prefix="history", episode_index=2)


def test_camera_pose_context_tvi_rejects_cached_timestamp_mismatch() -> None:
    dataset = _camera_pose_method_dataset()
    mismatched_timestamp = float(np.float32(1.000002))
    dataset._episode_payload = lambda episode_index: {
        "frame_indices": np.asarray([4], dtype=np.int64),
        "timestamps": np.asarray([mismatched_timestamp], dtype=np.float64),
        "camera_poses": {"front": np.zeros((1, 6), dtype=np.float32)},
    }
    context = {
        "history_steps": [{"timestamp": 1.0}],
        "history_blocks": [{"step_index": 0, "camera_name": "front", "frame_index": 4}],
    }

    with pytest.raises(ValueError, match=r"timestamp.*episode=2.*frame=4"):
        dataset._context_tvi(context, prefix="history", episode_index=2)


def test_camera_pose_context_tvi_accepts_float32_equivalent_timestamp() -> None:
    dataset = _camera_pose_method_dataset()
    data_timestamp = 123.456789
    context_timestamp = float(np.float32(data_timestamp))
    pose = np.asarray([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]], dtype=np.float32)
    dataset._episode_payload = lambda episode_index: {
        "frame_indices": np.asarray([4], dtype=np.int64),
        "timestamps": np.asarray([data_timestamp], dtype=np.float64),
        "camera_poses": {"front": pose},
    }
    context = {
        "history_steps": [{"timestamp": context_timestamp}],
        "history_blocks": [{"step_index": 0, "camera_name": "front", "frame_index": 4}],
    }

    history_tvi = dataset._context_tvi(context, prefix="history", episode_index=2)

    np.testing.assert_allclose(history_tvi, [[context_timestamp, *pose[0]]])


def test_episode_payload_rejects_duplicate_frame_indices(tiny_navvla_dataset_root: Path) -> None:
    data_path = next((tiny_navvla_dataset_root / "data").glob("chunk-*/part-*.parquet"))
    _replace_parquet_column(data_path, "frame_index", [0, 1, 1])
    dataset = NavVLACPMDataset(tiny_navvla_dataset_root)

    with pytest.raises(ValueError, match=r"duplicate frame_index.*episode=0.*frame=1"):
        dataset._episode_payload(0)


def test_build_cpm_dataset_forwards_camera_pose_tvi_mode_with_entry_override(tmp_path: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataset

    root = _build_tiny_cpm_root(tmp_path, dataset_name="builder_camera_pose_tvi")
    _add_camera_pose_columns(
        root,
        {
            "front": [
                [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
                [4.0, 5.0, 6.0, 0.4, 0.5, 0.6],
                [7.0, 8.0, 9.0, 0.7, 0.8, 0.9],
            ]
        },
    )
    config = OmegaConf.create(
        {
            "datasets": [
                {
                    "data_root_dir": str(root),
                    "dataset_statistics_key": "builder_camera_pose_tvi_vln_train",
                    "required_cameras": ["front"],
                    "tvi_mode": "time_camera_pose",
                }
            ],
            "tvi_mode": "time_yaw",
            "token_budget": 1024,
        }
    )

    top_level_dataset = build_cpm_dataset(
        OmegaConf.create(
            {
                "data_root_dir": str(root),
                "required_cameras": ["front"],
                "tvi_mode": "time_camera_pose",
                "token_budget": 1024,
            }
        )
    )
    override_dataset = build_cpm_dataset(config)

    assert top_level_dataset.tvi_mode == "time_camera_pose"
    assert top_level_dataset.tvi_dim == 7
    assert override_dataset.tvi_mode == "time_camera_pose"
    assert override_dataset.tvi_dim == 7


def test_cpm_dataset_supports_sparse_logical_indices(tmp_path: Path) -> None:
    root = _build_tiny_cpm_root(tmp_path, dataset_name="sparse_indices")
    data_path = next((root / "data").glob("chunk-*/part-*.parquet"))
    episode_path = next((root / "meta" / "episodes").glob("chunk-*/part-*.parquet"))
    context_path = resolve_context_index_paths(root, token_budget=1024).meta_path
    video_index_path = root / "meta" / "navvla_video_index.parquet"
    task_path = root / "meta" / "tasks.parquet"

    _replace_parquet_column(data_path, "index", [5, 10, 11])
    _replace_parquet_column(data_path, "episode_index", [9, 9, 9])
    _replace_parquet_column(data_path, "task_index", [7, 7, 7])
    _replace_parquet_column(context_path, "index", [5, 10, 11])
    _replace_parquet_column(video_index_path, "index", [5, 10, 11])
    _replace_parquet_column(episode_path, "episode_index", [9])
    _replace_parquet_column(episode_path, "task_index", [7])
    _replace_parquet_column(task_path, "task_index", [7])

    dataset = NavVLACPMDataset(root, token_budget=1024)
    sample = dataset[1]

    assert sample["lang"] == "go forward"
    assert sample["metadata"]["episode_index"] == 9
    assert sample["metadata"]["frame_index"] == 1
    assert sample["metadata"]["context_index_key"].endswith("/f000001/online-bats-v1")
    assert np.allclose(np.asarray(sample["images"]["front"])[0, 0], [40, 40, 40], atol=2)


def test_cpm_dataset_uses_tasks_parquet_as_the_only_task_metadata_source(tmp_path: Path) -> None:
    root = _build_tiny_cpm_root(tmp_path, dataset_name="parquet_task_metadata")
    task_path = root / "meta" / "tasks.parquet"
    task_table = pq.read_table(task_path)
    replacements = {
        "task_type": ["tracking"],
        "task_subtype": ["human_following"],
        "platform_text": ["Platform: tracking robot."],
        "dataset_source": ["tracking_fixture"],
        "answer": ["target"],
    }
    for column, values in replacements.items():
        column_index = task_table.schema.get_field_index(column)
        task_table = task_table.set_column(column_index, column, pa.array(values))
    pq.write_table(task_table, task_path)
    (root / "meta" / "navvla_tasks.jsonl").unlink()

    sample = NavVLACPMDataset(root, token_budget=1024)[0]

    assert sample["platform_text"] == "Platform: tracking robot."
    assert sample["qa_target"] == "target"
    assert sample["metadata"]["task_type"] == "tracking"
    assert sample["metadata"]["task_subtype"] == "human_following"
    assert sample["metadata"]["dataset_source"] == "tracking_fixture"


def test_cpm_dataset_requires_complete_tasks_parquet_metadata(tmp_path: Path) -> None:
    root = _build_tiny_cpm_root(tmp_path, dataset_name="missing_task_metadata")
    task_path = root / "meta" / "tasks.parquet"
    task_table = pq.read_table(task_path)
    pq.write_table(
        task_table.select([name for name in task_table.column_names if name != "platform_text"]),
        task_path,
    )

    with pytest.raises(ValueError, match="platform_text"):
        NavVLACPMDataset(root, token_budget=1024)


def test_cpm_dataloader_loads_long_memory_source_tokens(tiny_navvla_dataset_root: Path) -> None:
    _write_minicpm_profile_cache(tiny_navvla_dataset_root)
    dataset = NavVLACPMDataset(
        dataset_root=tiny_navvla_dataset_root,
        require_long_memory_tokens=True,
    )

    sample = dataset[2]
    batch = collate_navvla_cpm_batch([sample])

    assert dataset.allow_missing_long_memory is True
    assert sample["metadata"]["history_token_refs"]
    assert sample["metadata"]["long_memory_token_refs"] == ["episode-a/000000/front"]
    assert sample["metadata"]["long_memory_mask"] == [True]
    assert sample["long_memory_source_tvi"].shape == (1, 2)
    assert sample["long_memory_source_tvi"].tolist() == [[0.0, 0.0]]
    assert batch["long_memory_source_tokens"].shape == (1, 1, 4, 8)
    assert batch["long_memory_source_mask"].tolist() == [[True]]
    assert batch["long_memory_source_tvi"].shape == (1, 1, 2)
    assert batch["long_memory_source_tvi"][0].tolist() == [[0.0, 0.0]]
    assert np.all(batch["long_memory_source_tokens"][0, 0] == 1.0)


def test_cpm_dataloader_without_required_long_memory_skips_source_tokens(tiny_navvla_dataset_root: Path) -> None:
    _write_minicpm_profile_cache(tiny_navvla_dataset_root)
    dataset = NavVLACPMDataset(
        dataset_root=tiny_navvla_dataset_root,
    )

    sample = dataset[2]
    batch = collate_navvla_cpm_batch([sample])

    assert "long_memory_source_tvi" not in sample
    assert "long_memory_source_tokens" not in batch


def test_cpm_collate_pads_long_memory_source_tokens(tiny_navvla_dataset_root: Path) -> None:
    _write_minicpm_profile_cache(tiny_navvla_dataset_root)
    dataset = NavVLACPMDataset(
        dataset_root=tiny_navvla_dataset_root,
        require_long_memory_tokens=True,
    )

    batch = collate_navvla_cpm_batch([dataset[1], dataset[2]])

    assert batch["long_memory_source_tokens"].shape == (2, 1, 4, 8)
    assert batch["long_memory_source_mask"].tolist() == [[False], [True]]
    assert batch["long_memory_source_tvi"].shape == (2, 1, 2)
    assert batch["long_memory_source_tvi"][1].tolist() == [[0.0, 0.0]]


def test_pad_tvi_preserves_seven_columns_and_pads_only_rows() -> None:
    first = np.arange(14, dtype=np.float32).reshape(2, 7)
    second = np.arange(7, dtype=np.float32).reshape(1, 7) + 100.0

    padded = _pad_tvi([first, second], max_length=3)

    assert padded.shape == (2, 3, 7)
    np.testing.assert_array_equal(padded[0, :2], first)
    np.testing.assert_array_equal(padded[1, :1], second)
    np.testing.assert_array_equal(padded[:, 2], np.zeros((2, 7), dtype=np.float32))


def test_pad_tvi_uses_empty_array_width_when_max_length_is_zero() -> None:
    padded = _pad_tvi(
        [np.zeros((0, 7), dtype=np.float32), np.zeros((0, 7), dtype=np.float32)],
        max_length=0,
    )

    assert padded.shape == (2, 0, 7)


def test_pad_tvi_rejects_mixed_feature_widths() -> None:
    with pytest.raises(ValueError, match="TVI feature widths.*2.*7"):
        _pad_tvi(
            [np.zeros((1, 2), dtype=np.float32), np.zeros((0, 7), dtype=np.float32)],
            max_length=1,
        )


def test_pad_tvi_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="rank 2"):
        _pad_tvi([np.zeros((7,), dtype=np.float32)], max_length=1)


def test_pad_tvi_rejects_values_without_a_known_width() -> None:
    with pytest.raises(ValueError, match="known TVI feature width"):
        _pad_tvi([None, None], max_length=0)


def test_pad_tvi_rejects_negative_max_length() -> None:
    with pytest.raises(ValueError, match="max_length must be non-negative"):
        _pad_tvi([np.zeros((0, 7), dtype=np.float32)], max_length=-1)


def test_cpm_collate_core_preserves_nonempty_seven_dimensional_history_tvi() -> None:
    history_tvi = np.asarray(
        [
            [1.0, 10.0, 11.0, 12.0, 0.1, 0.2, 0.3],
            [2.0, 20.0, 21.0, 22.0, 0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    sample = {
        "images": {},
        "current_tvi": np.asarray([[3.0, 30.0, 31.0, 32.0, 0.7, 0.8, 0.9]], dtype=np.float32),
        "history_tvi": history_tvi,
        "history_mask": np.asarray([True, True], dtype=bool),
        "lang": "go",
        "platform_text": "uav",
        "action": np.zeros((1, 4), dtype=np.float32),
        "action_padding_mask": np.zeros((1,), dtype=bool),
        "distance_to_goal": 1.0,
        "qa_target": "",
        "metadata": {},
    }

    batch = _collate_core([sample])

    assert batch["history_tvi"].shape == (1, 2, 7)
    np.testing.assert_array_equal(batch["history_tvi"][0], history_tvi)


def test_cpm_require_long_memory_strict_filters_empty_source_rows(tiny_navvla_dataset_root: Path) -> None:
    _write_minicpm_profile_cache(tiny_navvla_dataset_root)
    dataset = NavVLACPMDataset(
        root=tiny_navvla_dataset_root,
        visual_token_mode="online_images",
        require_long_memory_tokens=True,
        allow_missing_long_memory=False,
    )

    assert len(dataset) == 1
    assert dataset.episode_sample_indices == {0: [0]}
    assert dataset._sample_indices.tolist() == [2]
    assert dataset.history_frame_count(0) == 2
    sample = dataset[0]
    assert len(sample["metadata"]["history_steps"]) == 2
    assert sample["metadata"]["long_memory_token_refs"] == ["episode-a/000000/front"]
    assert sample["metadata"]["long_memory_mask"] == [True]


def test_cpm_collate_keeps_mixed_dataset_refs_local(tmp_path: Path) -> None:
    root_a = _build_tiny_cpm_root(tmp_path, dataset_name="mix_a")
    root_b = _build_tiny_cpm_root(tmp_path, dataset_name="mix_b")
    dataset_a = NavVLACPMDataset(
        dataset_root=root_a,
        require_long_memory_tokens=True,
    )
    dataset_b = NavVLACPMDataset(
        dataset_root=root_b,
        require_long_memory_tokens=True,
    )

    sample_a = dataset_a[2]
    sample_b = dataset_b[2]
    batch = collate_navvla_cpm_batch([sample_a, sample_b])

    assert sample_a["metadata"]["long_memory_token_refs"] == ["episode-a/000000/front"]
    assert sample_b["metadata"]["long_memory_token_refs"] == ["episode-a/000000/front"]
    assert batch["long_memory_source_tokens"].shape == (2, 1, 4, 8)
    assert batch["long_memory_source_mask"].tolist() == [[True], [True]]


def test_cpm_lerobot_package_does_not_reference_legacy_dataloaders() -> None:
    from starVLA.dataloader import cpm_lerobot

    package_root = Path(cpm_lerobot.__file__).parent
    source = "\n".join(path.read_text(encoding="utf-8") for path in package_root.glob("*.py"))

    assert "navvla_lerobot_datasets" not in source
    assert "airsim_datasets" not in source
    assert "airsim_openfly_datasets" not in source
    assert "airsim_utils" not in source


def test_build_cpm_dataset_consumes_configured_dataset_list(tmp_path: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataset
    from starVLA.dataloader.cpm_lerobot.mixture import NavVLACPMMixtureDataset

    root_a = _build_tiny_cpm_root(tmp_path, dataset_name="mix_a")
    root_b = _build_tiny_cpm_root(tmp_path, dataset_name="mix_b")
    data_cfg = OmegaConf.create(
        {
            "datasets": [
                {"name": "a", "dataset_statistics_key": "mix_a_vln_train", "data_root_dir": str(root_a)},
                {"name": "b", "dataset_statistics_key": "mix_b_vln_train", "data_root_dir": str(root_b)},
            ],
            "data_mix": "tiny_mix",
            "split": "train",
            "token_budget": 1024,
            "include_state": True,
            "image_resize": [16, 16],
            "require_long_memory_tokens": True,
            "action_extra_dim_mode": "none",
        }
    )

    dataset = build_cpm_dataset(data_cfg)

    assert isinstance(dataset, NavVLACPMMixtureDataset)
    assert len(dataset) == sum(len(item) for item in dataset.datasets)
    assert dataset.dataset_statistics_keys == ["mix_a_vln_train", "mix_b_vln_train"]
    roots = {dataset[index]["metadata"]["mixture_dataset_root"] for index in range(len(dataset))}
    assert roots == {str(root_a), str(root_b)}


def test_cpm_mixture_uses_checkpoint_statistics_aliases(tmp_path: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataset

    root_a = _build_tiny_cpm_root(tmp_path, dataset_name="alias_a")
    root_b = _build_tiny_cpm_root(tmp_path, dataset_name="alias_b")
    dataset = build_cpm_dataset(
        OmegaConf.create(
            {
                "datasets": [
                    {
                        "name": "a",
                        "data_root_dir": str(root_a),
                        "dataset_statistics_key": "alias_a_vln_train",
                        "checkpoint_statistics_key": "dataset_a",
                    },
                    {
                        "name": "b",
                        "data_root_dir": str(root_b),
                        "dataset_statistics_key": "alias_b_vln_train",
                        "checkpoint_statistics_key": "dataset_b",
                    },
                ],
                "data_mix": "alias_mix",
                "split": "train",
                "token_budget": 1024,
            }
        )
    )

    metadata = dataset[0]["metadata"]
    assert metadata["dataset_statistics_key"] == "dataset_a"
    assert metadata["checkpoint_statistics_key"] == "dataset_a"
    assert metadata["normalization_dataset_statistics_key"] == "alias_a_vln_train"
    output_path = tmp_path / "aliased_statistics.json"
    dataset.save_dataset_statistics(output_path)
    assert set(json.loads(output_path.read_text(encoding="utf-8"))) == {"dataset_a", "dataset_b"}


def test_build_cpm_dataset_forwards_history_capacity_configuration(tmp_path: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataset

    root = _build_tiny_cpm_root(tmp_path, dataset_name="builder_capacity")
    dataset = build_cpm_dataset(
        OmegaConf.create(
            {
                "datasets": [
                    {
                        "data_root_dir": str(root),
                        "dataset_statistics_key": "builder_capacity_vln_train",
                        "max_online_history_frames": 7,
                        "bats_seed": 17,
                        "bats_epsilon": 0.2,
                        "bats_k": 3.0,
                        "use_dynamic_bats_k": False,
                        "budget_num_cameras": 1,
                        "current_wrapper_tokens": 2,
                        "history_wrapper_tokens": 0,
                    }
                ],
                "history_sampling_mode": "continuous",
                "token_budget": 1024,
                "required_cameras": ["front"],
            }
        )
    )

    assert dataset.max_online_history_frames == 7
    assert dataset.bats_seed == 17
    assert dataset.bats_epsilon == 0.2
    assert dataset.bats_k == 3.0
    assert dataset.use_dynamic_bats_k is False
    assert dataset.budget_num_cameras == 1
    assert dataset.current_wrapper_tokens == 2
    assert dataset.history_wrapper_tokens == 0


def test_cpm_dataset_rejects_missing_explicit_dataset_statistics_key(
    tiny_navvla_dataset_root: Path,
) -> None:
    with pytest.raises(KeyError, match="shared_abc_unnorm_key"):
        NavVLACPMDataset(
            tiny_navvla_dataset_root,
            dataset_statistics_key="shared_abc_unnorm_key",
        )


def test_cpm_dataset_removes_normalization_statistics_key(tiny_navvla_dataset_root: Path) -> None:
    with pytest.raises(TypeError, match="normalization_statistics_key"):
        NavVLACPMDataset(
            tiny_navvla_dataset_root,
            normalization_statistics_key="shared_abc_unnorm_key",
        )


def test_cpm_mixture_uses_and_exports_shared_dataset_statistics_key(tmp_path: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataset
    from starVLA.dataloader.cpm_lerobot.mixture import NavVLACPMMixtureDataset

    shared_key = "shared_abc_unnorm_key"
    root_a = _build_tiny_cpm_root(tmp_path, dataset_name="shared_a")
    root_b = _build_tiny_cpm_root(tmp_path, dataset_name="shared_b")
    source_statistics = json.loads((root_a / "dataset_statistics.json").read_text(encoding="utf-8"))
    shared_statistics = copy.deepcopy(next(iter(source_statistics.values())))
    shared_statistics["action"]["q01"] = [-1.0, -1.0, -1.0, -1.0]
    shared_statistics["action"]["q99"] = [1.0, 1.0, 1.0, 1.0]
    for root in (root_a, root_b):
        statistics_path = root / "dataset_statistics.json"
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        statistics[shared_key] = copy.deepcopy(shared_statistics)
        statistics_path.write_text(json.dumps(statistics), encoding="utf-8")

    data_cfg = OmegaConf.create(
        {
            "datasets": [
                {
                    "name": "a",
                    "data_root_dir": str(root_a),
                    "dataset_statistics_key": shared_key,
                },
                {
                    "name": "b",
                    "data_root_dir": str(root_b),
                    "dataset_statistics_key": shared_key,
                },
            ],
            "data_mix": "shared_mix",
            "split": "train",
            "token_budget": 1024,
        }
    )

    dataset = build_cpm_dataset(data_cfg)

    assert isinstance(dataset, NavVLACPMMixtureDataset)
    assert dataset.dataset_statistics_keys == [shared_key, shared_key]
    assert [item.dataset_key for item in dataset.datasets] == [shared_key, shared_key]
    assert [item._action_stats() for item in dataset.datasets] == [
        shared_statistics["action"],
        shared_statistics["action"],
    ]
    assert dataset[0]["metadata"]["source_dataset_statistics_key"] == "shared_a_train"
    output_path = tmp_path / "checkpoint_statistics.json"
    dataset.save_dataset_statistics(output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == {shared_key: shared_statistics}


def test_cpm_mixture_rejects_inconsistent_shared_dataset_statistics(tmp_path: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataset

    shared_key = "shared_abc_unnorm_key"
    root_a = _build_tiny_cpm_root(tmp_path, dataset_name="conflict_a")
    root_b = _build_tiny_cpm_root(tmp_path, dataset_name="conflict_b")
    source_statistics = json.loads((root_a / "dataset_statistics.json").read_text(encoding="utf-8"))
    shared_statistics = copy.deepcopy(next(iter(source_statistics.values())))
    for root in (root_a, root_b):
        statistics_path = root / "dataset_statistics.json"
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        statistics[shared_key] = copy.deepcopy(shared_statistics)
        statistics_path.write_text(json.dumps(statistics), encoding="utf-8")
    conflicting_path = root_b / "dataset_statistics.json"
    conflicting_statistics = json.loads(conflicting_path.read_text(encoding="utf-8"))
    conflicting_statistics[shared_key]["action"]["q99"] = [2.0, 2.0, 2.0, 2.0]
    conflicting_path.write_text(json.dumps(conflicting_statistics), encoding="utf-8")

    dataset = build_cpm_dataset(
        OmegaConf.create(
            {
                "datasets": [
                    {"name": "a", "data_root_dir": str(root_a), "dataset_statistics_key": shared_key},
                    {"name": "b", "data_root_dir": str(root_b), "dataset_statistics_key": shared_key},
                ],
                "data_mix": "conflict_mix",
                "split": "train",
                "token_budget": 1024,
            }
        )
    )

    with pytest.raises(ValueError, match="inconsistent normalization statistics"):
        dataset.save_dataset_statistics(tmp_path / "checkpoint_statistics.json")


def test_cpm_mixture_requires_dataset_statistics_key_per_entry(tmp_path: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataset

    root_a = _build_tiny_cpm_root(tmp_path, dataset_name="required_a")
    root_b = _build_tiny_cpm_root(tmp_path, dataset_name="required_b")

    with pytest.raises(KeyError, match=r"datasets\[0\].*dataset_statistics_key"):
        build_cpm_dataset(
            OmegaConf.create(
                {
                    "datasets": [
                        {"name": "a", "data_root_dir": str(root_a)},
                        {
                            "name": "b",
                            "data_root_dir": str(root_b),
                            "dataset_statistics_key": "required_b_train",
                        },
                    ],
                    "data_mix": "required_mix",
                    "split": "train",
                    "token_budget": 1024,
                }
            )
        )


def test_build_single_cpm_dataset_uses_dataset_statistics_key(tmp_path: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataset

    shared_key = "shared_abc_unnorm_key"
    root = _build_tiny_cpm_root(tmp_path, dataset_name="single_explicit")
    statistics_path = root / "dataset_statistics.json"
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    statistics[shared_key] = copy.deepcopy(next(iter(statistics.values())))
    statistics_path.write_text(json.dumps(statistics), encoding="utf-8")

    dataset = build_cpm_dataset(
        OmegaConf.create(
            {
                "data_root_dir": str(root),
                "dataset_statistics_key": shared_key,
                "split": "train",
                "token_budget": 1024,
            }
        )
    )

    assert dataset.dataset_key == shared_key


def test_build_cpm_dataloader_iterates_all_mixture_roots(tmp_path: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataloader

    root_a = _build_tiny_cpm_root(tmp_path, dataset_name="loader_a")
    root_b = _build_tiny_cpm_root(tmp_path, dataset_name="loader_b")
    data_cfg = OmegaConf.create(
        {
            "datasets": [
                {"name": "a", "dataset_statistics_key": "loader_a_vln_train", "data_root_dir": str(root_a)},
                {"name": "b", "dataset_statistics_key": "loader_b_vln_train", "data_root_dir": str(root_b)},
            ],
            "data_mix": "loader_mix",
            "split": "train",
            "token_budget": 1024,
            "include_state": False,
            "image_resize": [16, 16],
            "require_long_memory_tokens": True,
            "action_extra_dim_mode": "none",
            "per_device_batch_size": 2,
            "num_workers": 0,
            "shuffle": False,
        }
    )

    dataloader = build_cpm_dataloader(data_cfg, seed=7)
    seen_roots: set[str] = set()
    seen_statistics_keys: set[str] = set()
    for batch in dataloader:
        seen_roots.update(str(item["mixture_dataset_root"]) for item in batch["metadata"])
        seen_statistics_keys.update(str(item["dataset_statistics_key"]) for item in batch["metadata"])

    assert seen_roots == {str(root_a), str(root_b)}
    assert seen_statistics_keys == {"loader_a_vln_train", "loader_b_vln_train"}


def test_build_cpm_training_dataloader_drops_non_full_tail_batches(tmp_path: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataloader
    from starVLA.dataloader.cpm_lerobot.sampler import LengthBucketedEpisodeBatchSampler

    root = _build_tiny_cpm_root(tmp_path, dataset_name="drop_tail")
    data_cfg = OmegaConf.create(
        {
            "data_root_dir": str(root),
            "dataset_statistics_key": "drop_tail_vln_train",
            "split": "train",
            "token_budget": 1024,
            "per_device_batch_size": 2,
            "num_workers": 0,
            "shuffle": True,
        }
    )

    dataloader = build_cpm_dataloader(data_cfg, seed=7)
    batches = list(dataloader.batch_sampler)

    assert isinstance(dataloader.batch_sampler, LengthBucketedEpisodeBatchSampler)
    assert dataloader.batch_sampler.drop_last is True
    assert [len(batch) for batch in batches] == [2]


def test_build_cpm_training_dataloader_uses_world_size_for_length_buckets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataloader
    from starVLA.dataloader.cpm_lerobot.sampler import LengthBucketedEpisodeBatchSampler

    root = _build_tiny_cpm_root(tmp_path, dataset_name="length_bucketed_world_size")
    monkeypatch.setenv("WORLD_SIZE", "2")
    data_cfg = OmegaConf.create(
        {
            "data_root_dir": str(root),
            "dataset_statistics_key": "length_bucketed_world_size_vln_train",
            "split": "train",
            "history_sampling_mode": "continuous",
            "token_budget": 512,
            "per_device_batch_size": 1,
            "num_workers": 0,
            "shuffle": True,
            "length_bucket_width": 8,
            "length_bucket_buffer_size": 16,
        }
    )

    dataloader = build_cpm_dataloader(data_cfg, seed=7)

    assert isinstance(dataloader.batch_sampler, LengthBucketedEpisodeBatchSampler)
    assert dataloader.batch_sampler.bucket_width == 8
    assert dataloader.batch_sampler.buffer_size == 16
    assert dataloader.batch_sampler.sync_group_size == 2


def test_cpm_distributed_world_size_prefers_initialized_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starVLA.dataloader.cpm_lerobot import builder

    monkeypatch.setenv("WORLD_SIZE", "not-an-integer")
    monkeypatch.setattr(builder.dist, "is_available", lambda: True)
    monkeypatch.setattr(builder.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(builder.dist, "get_world_size", lambda: 4)

    assert builder._distributed_world_size() == 4


@pytest.mark.parametrize(
    ("world_size", "message"),
    [
        ("not-an-integer", "must be an integer"),
        ("0", "must be positive"),
        ("-2", "must be positive"),
    ],
)
def test_cpm_distributed_world_size_validates_environment(
    world_size: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starVLA.dataloader.cpm_lerobot import builder

    monkeypatch.setenv("WORLD_SIZE", world_size)
    monkeypatch.setattr(builder.dist, "is_available", lambda: True)
    monkeypatch.setattr(builder.dist, "is_initialized", lambda: False)

    with pytest.raises(ValueError, match=message):
        builder._distributed_world_size()


def test_build_cpm_length_bucketed_dataloader_supports_dynamic_bats(tmp_path: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataloader
    from starVLA.dataloader.cpm_lerobot.sampler import LengthBucketedEpisodeBatchSampler

    root_a = _build_tiny_cpm_root(tmp_path, dataset_name="length_bucketed_bats_a")
    root_b = _build_tiny_cpm_root(tmp_path, dataset_name="length_bucketed_bats_b")
    data_cfg = OmegaConf.create(
        {
            "datasets": [
                {
                    "data_root_dir": str(root_a),
                    "dataset_statistics_key": "length_bucketed_bats_a_vln_train",
                },
                {
                    "data_root_dir": str(root_b),
                    "dataset_statistics_key": "length_bucketed_bats_b_vln_train",
                    "visual_token_profile": "qwen3_5_test_profile",
                    "token_budget": 256,
                },
            ],
            "split": "train",
            "history_sampling_mode": "bats",
            "use_dynamic_bats_k": True,
            "token_budget": 512,
            "per_device_batch_size": 2,
            "num_workers": 0,
            "shuffle": True,
            "length_bucket_width": 8,
            "length_bucket_buffer_size": 16,
        }
    )

    dataloader = build_cpm_dataloader(data_cfg, seed=7)

    assert isinstance(dataloader.batch_sampler, LengthBucketedEpisodeBatchSampler)
    assert dataloader.batch_sampler.history_frame_capacities == {0: 55, 1: 37}
    assert [len(batch) for batch in dataloader.batch_sampler] == [2, 2, 2]


def test_cpm_mixture_history_counts_use_each_dataset_budget_and_camera_count(
    tmp_path: Path,
) -> None:
    from starVLA.dataloader.cpm_lerobot.mixture import NavVLACPMMixtureDataset

    root_a = _build_tiny_cpm_root(tmp_path, dataset_name="history_count_one_camera")
    root_b = _build_two_camera_cpm_root(tmp_path, dataset_name="history_count_two_cameras")
    dataset_a = NavVLACPMDataset(
        root_a,
        visual_token_mode="online_images",
        history_sampling_mode="bats",
        use_dynamic_bats_k=True,
        token_budget=84,
        required_cameras=["front"],
    )
    dataset_b = NavVLACPMDataset(
        root_b,
        visual_token_mode="online_images",
        history_sampling_mode="bats",
        use_dynamic_bats_k=True,
        token_budget=152,
        required_cameras=["front", "down"],
    )
    mixture = NavVLACPMMixtureDataset(
        [dataset_a, dataset_b],
        mixture_name="history_count_mix",
        dataset_statistics_keys=[dataset_a.dataset_key, dataset_b.dataset_key],
    )

    mixture.prepare_history_frame_counts()
    first_counts = [mixture.history_frame_count(index) for index in range(3)]
    second_counts = [mixture.history_frame_count(index) for index in range(3, 6)]

    assert dataset_a.history_frame_capacity_for_dataset(0) == 2
    assert dataset_b.history_frame_capacity_for_dataset(0) == 1
    assert first_counts == [0, 1, 2]
    assert second_counts == [0, 1, 1]
    assert first_counts == [
        len(mixture[index]["metadata"]["history_steps"])
        for index in range(3)
    ]
    assert second_counts == [
        len(mixture[index]["metadata"]["history_steps"])
        for index in range(3, 6)
    ]


def test_cpm_history_capacity_uses_profile_wrapper_tokens(tmp_path: Path) -> None:
    root = _build_tiny_cpm_root(tmp_path, dataset_name="wrapper_capacity")
    minicpm = NavVLACPMDataset(
        root,
        history_sampling_mode="continuous",
        token_budget=512,
        required_cameras=["front"],
    )
    qwen35 = NavVLACPMDataset(
        root,
        visual_token_profile="qwen3_5_test_profile",
        history_sampling_mode="continuous",
        token_budget=512,
        required_cameras=["front"],
    )

    assert minicpm.current_wrapper_tokens == 3
    assert minicpm.history_wrapper_tokens == 3
    assert minicpm.history_frame_capacity_for_dataset(0) == 55
    assert qwen35.current_wrapper_tokens == 2
    assert qwen35.history_wrapper_tokens == 0
    assert qwen35.history_frame_capacity_for_dataset(0) == 89


def test_build_cpm_length_bucketed_dataloader_rejects_fixed_k_bats(tmp_path: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataloader

    root = _build_tiny_cpm_root(tmp_path, dataset_name="length_bucketed_fixed_k_bats")
    data_cfg = OmegaConf.create(
        {
            "data_root_dir": str(root),
            "dataset_statistics_key": "length_bucketed_fixed_k_bats_vln_train",
            "split": "train",
            "history_sampling_mode": "bats",
            "use_dynamic_bats_k": False,
            "token_budget": 512,
            "per_device_batch_size": 2,
            "num_workers": 0,
            "shuffle": True,
            "length_bucket_width": 8,
            "length_bucket_buffer_size": 16,
        }
    )

    with pytest.raises(ValueError, match="requires use_dynamic_bats_k=True"):
        build_cpm_dataloader(data_cfg, seed=7)


def test_build_cpm_evaluation_dataloader_remains_sequential(tmp_path: Path) -> None:
    from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataloader

    root = _build_tiny_cpm_root(tmp_path, dataset_name="sequential_eval")
    dataloader = build_cpm_dataloader(
        OmegaConf.create(
            {
                "data_root_dir": str(root),
                "dataset_statistics_key": "sequential_eval_vln_train",
                "split": "train",
                "use_dynamic_bats_k": False,
                "token_budget": 1024,
                "per_device_batch_size": 2,
                "num_workers": 0,
                "shuffle": False,
            }
        ),
        seed=7,
    )

    assert dataloader.batch_size == 2
    assert dataloader.batch_sampler.batch_size == 2
    assert [list(batch) for batch in dataloader.batch_sampler] == [[0, 1], [2]]
