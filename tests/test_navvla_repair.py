from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


def _remove_context_budget(root: Path, budget: int) -> None:
    manifest_path = root / "meta" / "navvla_context_index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["entries"][str(budget)]
    for key in ("meta_path", "arrays_path", "debug_path"):
        path = root / entry[key]
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def test_repair_dry_run_plans_missing_context_without_writes(tiny_navvla_dataset_root: Path) -> None:
    from tool.navvla.repair import repair_navvla_dataset

    root = tiny_navvla_dataset_root
    _remove_context_budget(root, 1024)

    report = repair_navvla_dataset(
        root, apply=False, token_budgets=(1024,), budget_num_cameras=1, history_camera_names=("front",)
    )

    assert report["applied"] is False
    assert [action["type"] for action in report["actions"]] == ["rebuild_context"]
    assert not (root / "meta/context_index/budget_1024/context_meta.parquet").exists()


def test_repair_apply_rebuilds_context_and_validates(tiny_navvla_dataset_root: Path) -> None:
    from tool.navvla.repair import repair_navvla_dataset

    root = tiny_navvla_dataset_root
    _remove_context_budget(root, 1024)

    report = repair_navvla_dataset(
        root, apply=True, token_budgets=(1024,), budget_num_cameras=1, history_camera_names=("front",)
    )

    assert report["applied"] is True
    assert report["validation"]["total_frames"] == 3
    assert (root / "meta/context_index/budget_1024/context_meta.parquet").exists()
    assert not (root / "meta/context_index/budget_1024/refs.parquet").exists()
    assert (root / "meta/context_index/budget_1024/context_arrays").is_dir()


def test_context_repair_uses_tasks_parquet_when_jsonl_is_absent(tiny_navvla_dataset_root: Path) -> None:
    from tool.navvla.repair import repair_navvla_dataset

    root = tiny_navvla_dataset_root
    (root / "meta/navvla_tasks.jsonl").unlink()
    _remove_context_budget(root, 1024)

    report = repair_navvla_dataset(
        root,
        apply=True,
        token_budgets=(1024,),
        budget_num_cameras=1,
        history_camera_names=("front",),
    )

    assert report["validation"]["total_frames"] == 3
    assert (root / "meta/context_index/budget_1024/context_meta.parquet").exists()


def test_context_repair_can_disable_long_memory(tiny_navvla_dataset_root: Path) -> None:
    from tool.navvla.context_index import load_runtime_context_index, resolve_context_index_paths
    from tool.navvla.repair import repair_navvla_dataset

    root = tiny_navvla_dataset_root
    _remove_context_budget(root, 1024)
    repair_navvla_dataset(
        root,
        apply=True,
        token_budgets=(1024,),
        budget_num_cameras=1,
        history_camera_names=("front",),
        include_long_memory=False,
    )
    runtime = load_runtime_context_index(resolve_context_index_paths(root, token_budget=1024))

    assert runtime.meta["long_memory_count"].astype(int).sum() == 0


def test_context_only_repair_does_not_rewrite_unrelated_dataset_files(tiny_navvla_dataset_root: Path) -> None:
    from tool.navvla.repair import repair_navvla_dataset

    root = tiny_navvla_dataset_root
    protected = [
        next((root / "data").glob("chunk-*/part-*.parquet")),
        next((root / "meta/episodes").glob("chunk-*/part-*.parquet")),
        root / "meta/navvla_video_index.parquet",
        root / "dataset_statistics.json",
        root / "meta/context_index/budget_2048/context_meta.parquet",
    ]
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in protected}
    _remove_context_budget(root, 1024)

    repair_navvla_dataset(
        root,
        apply=True,
        token_budgets=(1024,),
        budget_num_cameras=1,
        history_camera_names=("front",),
    )

    assert {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in protected} == before


def test_repair_regenerates_missing_dataset_statistics(tiny_navvla_dataset_root: Path) -> None:
    from tool.navvla.repair import repair_navvla_dataset

    root = tiny_navvla_dataset_root
    (root / "dataset_statistics.json").unlink()

    dry_run = repair_navvla_dataset(root, apply=False)
    assert "rebuild_statistics" in [action["type"] for action in dry_run["actions"]]

    applied = repair_navvla_dataset(root, apply=True)
    statistics = json.loads((root / "dataset_statistics.json").read_text(encoding="utf-8"))
    assert applied["validation"]["total_frames"] == 3
    assert statistics


def test_repair_rejects_missing_data_parquet(tiny_navvla_dataset_root: Path) -> None:
    import pytest

    from tool.navvla.repair import repair_navvla_dataset

    data_path = next((tiny_navvla_dataset_root / "data").glob("chunk-*/part-*.parquet"))
    data_path.unlink()

    with pytest.raises(FileNotFoundError, match="data parquet"):
        repair_navvla_dataset(tiny_navvla_dataset_root, apply=False)


def test_repair_recovers_mmap_cache_index_from_checkpoint_rows(tiny_navvla_dataset_root: Path) -> None:
    import pandas as pd

    from tool.navvla.cli import generate_visual_cache as gvc
    from tool.navvla.repair import repair_navvla_dataset
    from tool.navvla.visual_token_cache import default_minicpm_v46_visual_token_profile, write_profile_manifest

    root = tiny_navvla_dataset_root
    profile = default_minicpm_v46_visual_token_profile(encoder_ckpt="/tmp/minicpm")
    profile = profile.__class__(**{**profile.__dict__, "hidden_dim": 8})
    write_profile_manifest(root, profile)
    shard_path = root / "cache/visual_tokens" / profile.name / "shards/image_embeds_000000.npy"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(shard_path, np.ones((1, 4, 8), dtype=np.float16))
    row = {
        "ref": "episode-a/000000/front",
        "path": None,
        "image_key": "episode-a/000000/front",
        "shard_path": shard_path.relative_to(root).as_posix(),
        "row_index": 0,
        "token_count": 4,
        "hidden_dim": 8,
        "dtype": "float16",
        "episode_id": "episode-a",
        "trajectory_id": "traj-a",
        "frame_index": 0,
        "source_frame_index": 0,
        "data_index": 0,
        "camera_name": "front",
        "video_key": "front_image",
    }
    gvc.write_mmap_checkpoint_index_rows(root, profile.name, [row])

    dry_run = repair_navvla_dataset(root, apply=False)
    assert "rebuild_mmap_cache_index" in [action["type"] for action in dry_run["actions"]]

    repair_navvla_dataset(root, apply=True)
    index = pd.read_parquet(root / "cache/visual_tokens" / profile.name / "index.parquet")
    assert index["ref"].tolist() == ["episode-a/000000/front"]
