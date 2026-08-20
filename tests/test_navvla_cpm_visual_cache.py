from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from tool.navvla.cli import generate_visual_cache as gvc
from tool.navvla.cli.generate_visual_cache import load_existing_ref_rows_for_refs, load_history_refs, load_visual_cache_refs
from tool.navvla.context_index import ContextIndexConfig
from tool.navvla.lerobot_v3_writer import write_navvla_lerobot_dataset
from tool.navvla.profile_visual_cache import generate_profile_cache_parallel
from tool.navvla.visual_token_cache import (
    default_minicpm_v46_visual_token_profile,
    profile_cache_root,
    write_profile_mmap_npy_cache,
    write_profile_manifest,
)
from conftest import tiny_navvla_episodes, tiny_navvla_spec


def test_open_video_capture_retries_transient_open_failure(monkeypatch, tmp_path: Path) -> None:
    attempts = []

    class FakeCapture:
        def __init__(self, opened: bool) -> None:
            self.opened = opened
            self.released = False

        def isOpened(self) -> bool:
            return self.opened

        def release(self) -> None:
            self.released = True

    captures = [FakeCapture(False), FakeCapture(True)]

    def fake_video_capture(path: str) -> FakeCapture:
        attempts.append(path)
        return captures[len(attempts) - 1]

    monkeypatch.setattr(gvc.cv2, "VideoCapture", fake_video_capture)
    monkeypatch.setattr(gvc.time, "sleep", lambda _seconds: None)

    capture = gvc._open_video_capture(tmp_path / "episode.mp4")

    assert capture is captures[1]
    assert captures[0].released is True
    assert len(attempts) == 2


def test_minicpm_v46_profile_writes_image_embeds_only() -> None:
    profile = default_minicpm_v46_visual_token_profile(encoder_ckpt="/tmp/minicpm")

    assert profile.name == "minicpm_v46_pooled_history_4_mmap"
    assert profile.visual_head == "minicpm_v46_visual"
    assert profile.encoder_name == "MiniCPM-V-4.6"
    assert profile.encoder_ckpt == "/tmp/minicpm"
    assert profile.token_count == 4
    assert profile.file_format == "mmap_npy"
    assert profile.has_deepstack is False
    assert profile.array_keys == ["image_embeds"]


def test_minicpm_cache_pooling_preserves_the_spatial_grid() -> None:
    visual_tokens = torch.arange(16, dtype=torch.float32).reshape(16, 1)

    pooled = gvc._pool_minicpm_visual_tokens(
        visual_tokens,
        target_tokens=4,
        grid_height=4,
        grid_width=4,
    )

    torch.testing.assert_close(
        pooled,
        torch.tensor([[2.5], [4.5], [10.5], [12.5]], dtype=torch.float32),
    )


def test_load_history_refs_unions_bats_and_long_memory_refs(tmp_path: Path) -> None:
    summary = write_navvla_lerobot_dataset(
        tiny_navvla_episodes(tmp_path / "images-history-refs"),
        output_root=tmp_path / "out-history-refs",
        spec=tiny_navvla_spec(dataset_name="history_refs"),
        overwrite=True,
        cache_workers=1,
        write_visual_token_cache=False,
        context_index_config=ContextIndexConfig(use_dynamic_bats_k=False, k=0.0),
    )
    dataset_root = Path(summary["dataset_root"])

    assert load_history_refs(dataset_root, token_budget=1024) == [
        "episode-a/000000/front",
        "episode-a/000001/front",
    ]


def test_load_visual_cache_refs_keeps_frame_zero_for_bats_history(tmp_path: Path) -> None:
    summary = write_navvla_lerobot_dataset(
        tiny_navvla_episodes(tmp_path / "images-frame-zero"),
        output_root=tmp_path / "out-frame-zero",
        spec=tiny_navvla_spec(dataset_name="frame_zero"),
        overwrite=True,
        cache_workers=1,
        write_visual_token_cache=False,
        context_index_config=ContextIndexConfig(use_dynamic_bats_k=False, k=0.0),
    )
    dataset_root = Path(summary["dataset_root"])

    refs = load_visual_cache_refs(dataset_root)

    assert "episode-a/000000/front" in refs
    assert load_visual_cache_refs(dataset_root, camera_names=["front"]) == refs
    with pytest.raises(ValueError, match="unknown visual cache camera names"):
        load_visual_cache_refs(dataset_root, camera_names=["left"])


def test_minicpm_mmap_cache_writes_shards_without_ref_hash_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    profile = default_minicpm_v46_visual_token_profile(encoder_ckpt="/tmp/minicpm")
    profile = profile.__class__(**{**profile.__dict__, "hidden_dim": 8, "shard_size": 2})

    index_path = write_profile_mmap_npy_cache(
        dataset_root,
        profile=profile,
        records=[
            {
                "ref": "episode-a/000000/front",
                "image_embeds": np.full((4, 8), 1.0, dtype=np.float16),
                "episode_id": "episode-a",
                "trajectory_id": "traj-a",
                "frame_index": 0,
                "source_frame_index": 0,
                "data_index": 0,
                "camera_name": "front",
                "video_key": "front_image",
            },
            {
                "ref": "episode-a/000001/front",
                "image_embeds": np.full((4, 8), 2.0, dtype=np.float16),
                "episode_id": "episode-a",
                "trajectory_id": "traj-a",
                "frame_index": 1,
                "source_frame_index": 1,
                "data_index": 1,
                "camera_name": "front",
                "video_key": "front_image",
            },
        ],
    )

    root = profile_cache_root(dataset_root, profile.name)
    index = pd.read_parquet(index_path)
    assert index["ref"].tolist() == ["episode-a/000000/front", "episode-a/000001/front"]
    assert set(index["shard_path"]) == {"cache/visual_tokens/minicpm_v46_pooled_history_4_mmap/shards/image_embeds_000000.npy"}
    assert index["row_index"].tolist() == [0, 1]
    assert index["token_count"].tolist() == [4, 4]
    assert index["hidden_dim"].tolist() == [8, 8]
    assert not list((root / "tokens").glob("*.npz"))
    shard = np.load(dataset_root / index.loc[0, "shard_path"], mmap_mode="r")
    assert shard.shape == (2, 4, 8)
    assert np.asarray(shard[1]).mean() == 2.0


def _mmap_checkpoint_row(dataset_root: Path, profile_name: str, ref: str, shard_name: str) -> dict[str, object]:
    shard_path = dataset_root / "cache" / "visual_tokens" / profile_name / "shards" / shard_name
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    np.lib.format.open_memmap(shard_path, mode="w+", dtype=np.float16, shape=(1, 4, 8)).flush()
    return {
        "ref": ref,
        "path": shard_path.relative_to(dataset_root).as_posix(),
        "image_key": "episode-a/000000/front",
        "shard_path": shard_path.relative_to(dataset_root).as_posix(),
        "row_index": 0,
        "token_count": 4,
        "hidden_dim": 8,
        "dtype": "float16",
        "episode_id": "episode-a",
        "trajectory_id": "episode-a",
        "frame_index": 0,
        "source_frame_index": 0,
        "data_index": 0,
        "camera_name": "front",
        "video_key": "front_image",
    }


def test_load_existing_ref_rows_for_refs_recovers_minicpm_mmap_checkpoint_indexes(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    profile = default_minicpm_v46_visual_token_profile(encoder_ckpt="/tmp/minicpm")
    profile = profile.__class__(**{**profile.__dict__, "hidden_dim": 8})
    write_profile_manifest(dataset_root, profile)
    checkpoint_row = _mmap_checkpoint_row(dataset_root, profile.name, "checkpoint-ref", "rank_00000_image_embeds_000000.npy")

    gvc.write_mmap_checkpoint_index_rows(dataset_root, profile.name, [checkpoint_row])

    rows = load_existing_ref_rows_for_refs(dataset_root, profile.name, ["checkpoint-ref", "missing-ref"])

    assert [row["ref"] for row in rows] == ["checkpoint-ref"]
    assert rows[0]["shard_path"] == checkpoint_row["shard_path"]


def test_merge_distributed_index_rows_keeps_minicpm_checkpoint_rows_without_duplicates(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    profile = default_minicpm_v46_visual_token_profile(encoder_ckpt="/tmp/minicpm")
    profile = profile.__class__(**{**profile.__dict__, "hidden_dim": 8})
    write_profile_manifest(dataset_root, profile)
    checkpoint_row = _mmap_checkpoint_row(dataset_root, profile.name, "checkpoint-ref", "rank_00000_image_embeds_000000.npy")
    duplicate_checkpoint_row = _mmap_checkpoint_row(dataset_root, profile.name, "dup-ref", "rank_00000_image_embeds_000001.npy")
    gvc.write_mmap_checkpoint_index_rows(dataset_root, profile.name, [checkpoint_row])
    gvc.write_mmap_checkpoint_index_rows(dataset_root, profile.name, [duplicate_checkpoint_row])
    rank_rows = [
        _mmap_checkpoint_row(dataset_root, profile.name, "rank-ref", "rank_00001_image_embeds_000000.npy"),
        _mmap_checkpoint_row(dataset_root, profile.name, "checkpoint-ref", "rank_00001_image_embeds_000001.npy"),
    ]
    pd.DataFrame(rank_rows).to_parquet(gvc.distributed_rank_index_path(dataset_root, profile.name, rank=1), index=False)
    existing_rows = load_existing_ref_rows_for_refs(dataset_root, profile.name, ["checkpoint-ref", "dup-ref", "rank-ref"])

    merged = gvc.merge_distributed_index_rows(dataset_root, profile.name, existing_rows=existing_rows, world_size=2)

    assert [row["ref"] for row in merged] == ["checkpoint-ref", "dup-ref", "rank-ref"]
    assert next(row for row in merged if row["ref"] == "checkpoint-ref")["shard_path"] == checkpoint_row["shard_path"]


def test_rebuild_mmap_profile_index_from_checkpoint_sidecars(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    profile = default_minicpm_v46_visual_token_profile(encoder_ckpt="/tmp/minicpm")
    profile = profile.__class__(**{**profile.__dict__, "hidden_dim": 0})
    write_profile_manifest(dataset_root, profile)
    first_row = _mmap_checkpoint_row(dataset_root, profile.name, "first-ref", "rank_00000_image_embeds_000000.npy")
    second_row = _mmap_checkpoint_row(dataset_root, profile.name, "second-ref", "rank_00001_image_embeds_000000.npy")
    gvc.write_mmap_checkpoint_index_rows(dataset_root, profile.name, [first_row])
    gvc.write_mmap_checkpoint_index_rows(dataset_root, profile.name, [second_row])

    report = gvc.rebuild_mmap_profile_index(dataset_root, profile)

    index = pd.read_parquet(dataset_root / "cache" / "visual_tokens" / profile.name / "index.parquet")
    manifest = pd.read_json(dataset_root / "cache" / "visual_tokens" / profile.name / "manifest.json", typ="series")
    assert report["records"] == 2
    assert index["ref"].tolist() == ["first-ref", "second-ref"]
    assert manifest["hidden_dim"] == 8


def test_profile_with_minicpm_mmap_index_hidden_dim_infers_from_rows() -> None:
    profile = default_minicpm_v46_visual_token_profile(encoder_ckpt="/tmp/minicpm")

    updated = gvc.profile_with_mmap_index_hidden_dim(profile, [{"hidden_dim": 8}])

    assert updated.hidden_dim == 8


class _FakeMiniCPMCacheEncoder:
    def get_image_features_batch(self, images):
        return [
            (np.full((4, 8), float(index + 1), dtype=np.float16), None)
            for index, _image in enumerate(images)
        ]


def test_profile_cache_parallel_writes_minicpm_mmap_shards(tmp_path: Path) -> None:
    summary = write_navvla_lerobot_dataset(
        tiny_navvla_episodes(tmp_path / "images"),
        output_root=tmp_path / "out",
        spec=tiny_navvla_spec(dataset_name="mmap_cpm"),
        overwrite=True,
        cache_workers=1,
        write_visual_token_cache=False,
        context_index_config=ContextIndexConfig(
            use_dynamic_bats_k=False,
            k=0.0,
        ),
    )
    root = Path(summary["dataset_root"])
    profile = default_minicpm_v46_visual_token_profile(encoder_ckpt="/tmp/minicpm")
    profile = profile.__class__(**{**profile.__dict__, "hidden_dim": 8, "shard_size": 2})

    report = generate_profile_cache_parallel(
        root,
        profile=profile,
        encoder=_FakeMiniCPMCacheEncoder(),
        workers=1,
        batch_size=2,
    )

    index = pd.read_parquet(root / "cache" / "visual_tokens" / profile.name / "index.parquet")
    assert report["records"] == len(index)
    assert set(index["ref"]) == set(load_visual_cache_refs(root))
    assert len(index) > len(load_history_refs(root, token_budget=1024))
    assert set(index["ref"].str.count("/")) == {2}
    assert set(index["token_count"]) == {4}
    assert not list((root / "cache" / "visual_tokens" / profile.name / "tokens").glob("*.npz"))
    for shard_relpath in sorted(set(index["shard_path"])):
        shard = np.load(root / shard_relpath, mmap_mode="r")
        assert shard.shape[1:] == (4, 8)


class _FakeMiniCPMVisionOutput:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class _FakeMiniCPMModel:
    config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8), hidden_size=8)
    device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.model_id = model_id
        cls.kwargs = kwargs
        return cls()

    def eval(self):
        return self

    def to(self, device):
        self.device = torch.device(device)
        return self

    def get_image_features(self, pixel_values, target_sizes, downsample_mode="4x"):
        del pixel_values, target_sizes
        type(self).downsample_mode = downsample_mode
        first = torch.arange(64 * 8, dtype=torch.float32).view(64, 8)
        second = first + 1000.0
        return _FakeMiniCPMVisionOutput([first, second])


class _FakeMiniCPMBatch(dict):
    def to(self, device):
        self["device"] = device
        return self


class _FakeMiniCPMProcessor:
    @classmethod
    def from_pretrained(cls, model_id, trust_remote_code=True):
        cls.model_id = model_id
        cls.trust_remote_code = trust_remote_code
        return cls()

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0][0]["content"][0]["type"] == "image"
        assert kwargs["return_tensors"] == "pt"
        batch_size = len(messages)
        return _FakeMiniCPMBatch(
            {
                "pixel_values": torch.zeros((batch_size, 3, 8, 8), dtype=torch.float32),
                "target_sizes": torch.tensor([[8, 8]] * batch_size, dtype=torch.long),
            }
        )


def test_load_minicpm_encoder_outputs_image_embeds_only(monkeypatch) -> None:
    monkeypatch.setattr(gvc, "AutoModelForImageTextToText", _FakeMiniCPMModel)
    monkeypatch.setattr(gvc, "AutoProcessor", _FakeMiniCPMProcessor)
    monkeypatch.setattr(gvc.torch.cuda, "is_available", lambda: False)
    profile = default_minicpm_v46_visual_token_profile(encoder_ckpt="/tmp/minicpm")
    profile = profile.__class__(**{**profile.__dict__, "hidden_dim": 8})

    encoder = gvc.load_minicpm_v46_encoder(encoder_ckpt="/tmp/minicpm", profile=profile)
    outputs = encoder.get_image_features_batch([Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))])

    assert _FakeMiniCPMModel.model_id == "/tmp/minicpm"
    assert _FakeMiniCPMProcessor.model_id == "/tmp/minicpm"
    assert encoder.downsample_mode == "16x"
    assert encoder.processor.downsample_mode == "16x"
    assert _FakeMiniCPMModel.downsample_mode == "16x"
    assert len(outputs) == 2
    for image_embeds, deepstack_embeds in outputs:
        assert image_embeds.shape == (4, 8)
        assert deepstack_embeds is None
