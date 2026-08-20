from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _valid_sampled_data_row() -> dict[str, object]:
    return {
        "observation.state": [1.0, 2.0, 3.0, 4.0],
        "action": [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]],
        "action.padding_mask": [False, False],
        "timestamp": 0.0,
        "task_index": 0,
        "episode_index": 0,
        "frame_index": 0,
        "source_frame_index": 0,
        "index": 0,
        "next.done": False,
        "sample.action_available": True,
        "context.index_key": "dataset/train/episode/f000000/bats-v1",
    }


def test_validation_rejects_empty_or_invalid_sampled_data_fields() -> None:
    from tool.navvla.validation import _validate_state_contract

    info = {
        "features": {"observation.state": {"shape": [4]}},
        "navvla": {"state_dim": 4, "action_horizon": 2, "action_dim": 4},
    }
    _validate_state_contract(info, [_valid_sampled_data_row()])
    with pytest.raises(ValueError, match="context.index_key"):
        _validate_state_contract(info, [{**_valid_sampled_data_row(), "context.index_key": ""}])
    with pytest.raises(ValueError, match="action"):
        _validate_state_contract(info, [{**_valid_sampled_data_row(), "action": []}])


def test_validation_reports_full_parquet_inventory_and_sampled_content(
    tiny_navvla_dataset_root: Path,
    monkeypatch,
) -> None:
    from tool.navvla import validation

    real_read_parquet = validation.pd.read_parquet

    def guarded_read_parquet(path, *args, **kwargs):
        if tiny_navvla_dataset_root / "data" in Path(path).parents and kwargs.get("columns") is None:
            raise AssertionError("data shard content must be sampled through parquet row groups")
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(validation.pd, "read_parquet", guarded_read_parquet)

    report = validation.validate_navvla_lerobot_dataset(
        tiny_navvla_dataset_root,
        visual_token_mode="online_images",
        smoke_load=0,
    )

    assert report["artifacts"]["data_parquet"]["scope"] == "full"
    assert report["artifacts"]["data_parquet"]["checked"] == report["artifacts"]["data_parquet"]["total"]
    assert report["artifacts"]["data_rows"]["scope"] == "sampled"
    assert report["artifacts"]["data_rows"]["checked"] <= report["artifacts"]["data_rows"]["total"]
    assert report["artifacts"]["data_rows"]["sample_indices"] == sorted(
        report["artifacts"]["data_rows"]["sample_indices"]
    )


def test_validation_samples_cache_tensors_but_counts_all_index_rows(
    profile_cache_dataset_root: Path,
    monkeypatch,
) -> None:
    from tool.navvla import validation

    index_path = (
        profile_cache_dataset_root
        / "cache"
        / "visual_tokens"
        / "qwen3_vl_4b_pooled_history"
        / "index.parquet"
    )
    index = pd.read_parquet(index_path)
    expanded = pd.concat(
        [index, *[index.assign(ref=index["ref"].astype(str) + f"-copy-{copy}") for copy in range(20)]],
        ignore_index=True,
    )
    token_path = profile_cache_dataset_root / str(index.iloc[0]["path"])
    expanded["path"] = str(token_path.relative_to(profile_cache_dataset_root))
    expanded.to_parquet(index_path, index=False)

    opened = []
    real_load = np.load
    real_read_parquet = validation.pd.read_parquet

    def recording_load(path, *args, **kwargs):
        opened.append(Path(path))
        return real_load(path, *args, **kwargs)

    def guarded_read_parquet(path, *args, **kwargs):
        if Path(path) == index_path:
            raise AssertionError("cache index content must be sampled through parquet row groups")
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(validation.np, "load", recording_load)
    monkeypatch.setattr(validation.pd, "read_parquet", guarded_read_parquet)
    report = validation.validate_navvla_lerobot_dataset(
        profile_cache_dataset_root,
        visual_token_mode="cached_history_online_current",
        visual_token_profile="qwen3_vl_4b_pooled_history",
        cache_sample_size=5,
        smoke_load=0,
    )

    cache_index = report["artifacts"]["visual_cache_index"]
    cache_tensors = report["artifacts"]["visual_cache_tensors"]
    assert cache_index == {"scope": "full", "checked": len(expanded), "total": len(expanded)}
    assert cache_tensors["scope"] == "sampled"
    assert cache_tensors["checked"] == 5
    assert cache_tensors["total"] == len(expanded)
    cache_opens = [path for path in opened if "cache/visual_tokens" in str(path)]
    assert len(cache_opens) == 5


def test_validation_cache_sampling_is_deterministic(profile_cache_dataset_root: Path) -> None:
    from tool.navvla.validation import validate_navvla_lerobot_dataset

    kwargs = {
        "visual_token_mode": "cached_history_online_current",
        "visual_token_profile": "qwen3_vl_4b_pooled_history",
        "cache_sample_size": 2,
        "sample_seed": 17,
        "smoke_load": 0,
    }
    first = validate_navvla_lerobot_dataset(profile_cache_dataset_root, **kwargs)
    second = validate_navvla_lerobot_dataset(profile_cache_dataset_root, **kwargs)

    assert first["artifacts"]["visual_cache_tensors"]["sample_indices"] == second["artifacts"]["visual_cache_tensors"][
        "sample_indices"
    ]


def test_validation_rejects_empty_sampled_cache_fields(profile_cache_dataset_root: Path) -> None:
    from tool.navvla.validation import validate_navvla_lerobot_dataset

    index_path = (
        profile_cache_dataset_root
        / "cache"
        / "visual_tokens"
        / "qwen3_vl_4b_pooled_history"
        / "index.parquet"
    )
    index = pd.read_parquet(index_path)
    index.loc[0, "camera_name"] = ""
    index.to_parquet(index_path, index=False)

    with pytest.raises(ValueError, match="camera_name"):
        validate_navvla_lerobot_dataset(
            profile_cache_dataset_root,
            visual_token_mode="cached_history_online_current",
            visual_token_profile="qwen3_vl_4b_pooled_history",
            smoke_load=0,
        )


def test_validation_checks_bfloat16_bit_patterns_for_non_finite_values() -> None:
    from tool.navvla.validation import _visual_token_values_are_finite

    finite = np.asarray([0x3F80, 0xC000, 0x0000], dtype=np.uint16)
    non_finite = np.asarray([0x7F80, 0x7FC0], dtype=np.uint16)

    assert _visual_token_values_are_finite(finite, storage_encoding="bfloat16_bits")
    assert not _visual_token_values_are_finite(non_finite, storage_encoding="bfloat16_bits")


def test_validation_smoke_loads_qwen35_mmap_cache_with_cpm_reader(
    tiny_navvla_dataset_root: Path,
) -> None:
    import torch

    from starVLA.model.modules.qwen35_vision import bf16_to_numpy_bits
    from tool.navvla.cli.generate_visual_cache import load_visual_cache_refs
    from tool.navvla.validation import validate_navvla_lerobot_dataset
    from tool.navvla.visual_token_cache import (
        default_qwen35_pooled_history_visual_token_profile,
        write_profile_mmap_npy_cache,
    )

    profile = default_qwen35_pooled_history_visual_token_profile(encoder_ckpt="checkpoint")
    records = []
    for ref in load_visual_cache_refs(tiny_navvla_dataset_root, camera_names=["front"]):
        episode_id, frame_index, camera_name = ref.split("/", 2)
        records.append(
            {
                "ref": ref,
                "image_embeds": bf16_to_numpy_bits(torch.zeros((4, 8), dtype=torch.float32)),
                "episode_id": episode_id,
                "trajectory_id": "traj-a",
                "frame_index": int(frame_index),
                "source_frame_index": int(frame_index),
                "data_index": int(frame_index),
                "camera_name": camera_name,
                "video_key": "front_image",
                "grid_t": 1,
                "grid_h": 16,
                "grid_w": 16,
                "cache_stage": profile.cache_stage,
            }
        )
    write_profile_mmap_npy_cache(tiny_navvla_dataset_root, profile=profile, records=records)

    report = validate_navvla_lerobot_dataset(
        tiny_navvla_dataset_root,
        visual_token_mode="cached_history_online_current",
        visual_token_profile=profile.name,
        token_budget=1024,
        required_cameras=["front"],
        image_resize=(16, 16),
        smoke_load=2,
    )

    assert report["smoke_load"]["reader"] == "NavVLACPMDataset"
    assert report["smoke_load"]["collator"] == "NavVLACPMCollator"
    assert report["smoke_load"]["loaded_samples"] == 2
    assert report["smoke_load"]["history_cached_shape"][0] == 2
