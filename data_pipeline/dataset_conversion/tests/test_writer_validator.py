from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from navvla_conversion.context_index import ContextIndexConfig
from navvla_conversion.derived_artifacts import finalize_derived_artifacts
from navvla_conversion.lerobot_v3_writer import write_navvla_lerobot_dataset
from navvla_conversion.validation import validate_navvla_lerobot_dataset

from conftest import tiny_episodes, tiny_spec


def test_conversion_writes_and_validates_core_artifacts(tiny_dataset_root: Path) -> None:
    report = validate_navvla_lerobot_dataset(tiny_dataset_root, check_media_decode="sampled")
    assert report["valid"] is True
    assert report["counts"] == {"frames": 3, "episodes": 1, "tasks": 1, "frame_metadata": 3}
    assert report["state_contract"]["world_pose_assumed"] is False
    assert not (tiny_dataset_root / "cache" / "visual_tokens").exists()

    data = pd.read_parquet(tiny_dataset_root / "data" / "chunk-000" / "part-000.parquet")
    assert data["action.padding_mask"].map(list).tolist() == [[False, False]] * 3
    first = np.stack(
        [np.asarray(step, dtype=float) for step in data.iloc[0]["action"]],
        axis=0,
    )
    np.testing.assert_allclose(first[1], np.zeros(4))


def test_writer_protects_existing_output(tmp_path: Path) -> None:
    kwargs = dict(
        episodes=tiny_episodes(tmp_path / "images"),
        output_root=tmp_path / "out",
        spec=tiny_spec(),
        write_workers=1,
        context_index_config=ContextIndexConfig(use_dynamic_bats_k=False, k=0.0),
    )
    write_navvla_lerobot_dataset(**kwargs)
    with pytest.raises(FileExistsError):
        write_navvla_lerobot_dataset(**kwargs)


def test_missing_media_fails_explicitly(tmp_path: Path) -> None:
    episodes = tiny_episodes(tmp_path / "images")
    Path(next(iter(episodes[0].frames[0].media_paths.values()))).unlink()
    with pytest.raises(FileNotFoundError):
        write_navvla_lerobot_dataset(
            episodes,
            output_root=tmp_path / "out",
            spec=tiny_spec(),
            write_workers=1,
            context_index_config=ContextIndexConfig(use_dynamic_bats_k=False, k=0.0),
        )


def test_enhanced_finalizer_rebuilds_only_core_derived_artifacts(tiny_dataset_root: Path) -> None:
    (tiny_dataset_root / "dataset_statistics.json").unlink()
    shutil.rmtree(tiny_dataset_root / "meta" / "context_index")
    shutil.rmtree(tiny_dataset_root / "cache" / "context_index_debug")
    (tiny_dataset_root / "meta" / "navvla_context_index_manifest.json").unlink()

    report = finalize_derived_artifacts(
        tiny_dataset_root,
        apply=True,
        token_budgets=(1024,),
        budget_num_cameras=1,
        history_camera_names=("front",),
    )
    assert report["validation"]["valid"] is True
    assert (tiny_dataset_root / "dataset_statistics.json").is_file()
    assert (tiny_dataset_root / "meta" / "navvla_context_index_manifest.json").is_file()


def test_validator_rejects_non_finite_state(tiny_dataset_root: Path) -> None:
    path = tiny_dataset_root / "data" / "chunk-000" / "part-000.parquet"
    data = pd.read_parquet(path)
    state = list(data.loc[0, "observation.state"])
    state[0] = float("nan")
    data.at[0, "observation.state"] = state
    data.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="invalid observation.state"):
        validate_navvla_lerobot_dataset(tiny_dataset_root)


def test_validator_rejects_episode_length_mismatch(tiny_dataset_root: Path) -> None:
    path = tiny_dataset_root / "meta" / "episodes" / "chunk-000" / "part-000.parquet"
    episodes = pd.read_parquet(path)
    episodes.loc[0, "length"] = int(episodes.loc[0, "length"]) + 1
    episodes.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="episode lengths"):
        validate_navvla_lerobot_dataset(tiny_dataset_root)


def test_validator_rejects_invalid_scene_identity(tiny_dataset_root: Path) -> None:
    path = tiny_dataset_root / "meta" / "episodes" / "chunk-000" / "part-000.parquet"
    episodes = pd.read_parquet(path)
    episodes.loc[0, "scene_id"] = ""
    episodes.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="invalid scene_id"):
        validate_navvla_lerobot_dataset(tiny_dataset_root)


def test_conversion_report_marks_no_visual_cache(tiny_dataset_root: Path) -> None:
    conversion = json.loads((tiny_dataset_root / "conversion_report.json").read_text(encoding="utf-8"))
    assert conversion["total_frames"] == 3
    assert conversion["visual_token_cache"]["status"] == "not_generated"
