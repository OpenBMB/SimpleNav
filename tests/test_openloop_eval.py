from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from starVLA.training.openloop_eval import (
    FixedDistributedIndexSampler,
    OpenLoopMetricAccumulator,
    build_openloop_eval_loaders,
    flatten_openloop_metrics,
    write_openloop_metrics,
)
from starVLA.training.train_starvla import VLATrainer
from tool.navvla.openloop_eval import (
    select_scene_stratified_episodes,
    select_target_source_indexes,
)


def test_scene_stratified_selection_is_exact_deterministic_and_covers_scenes() -> None:
    candidates = [
        {
            "episode_index": index,
            "episode_id": f"episode-{index:03d}",
            "scene_id": f"scene-{index % 5}",
        }
        for index in range(60)
    ]

    first = select_scene_stratified_episodes(
        candidates,
        count=20,
        seed=42,
        dataset_name="tiny",
        split="vln_val_seen",
    )
    second = select_scene_stratified_episodes(
        candidates,
        count=20,
        seed=42,
        dataset_name="tiny",
        split="vln_val_seen",
    )

    assert first == second
    assert len(first) == 20
    assert {row["scene_id"] for row in first} == {f"scene-{index}" for index in range(5)}


def test_target_selection_redistributes_short_episode_deficit() -> None:
    episodes = [
        {"episode_index": 0, "episode_id": "a", "scene_id": "scene-a"},
        {"episode_index": 1, "episode_id": "b", "scene_id": "scene-b"},
    ]
    selected = select_target_source_indexes(
        episodes,
        valid_frames_by_episode={0: [0, 1], 1: list(range(10, 20))},
        count=8,
    )

    assert len(selected) == 8
    assert len(set(selected)) == 8
    assert set([0, 1]).issubset(selected)


def test_fixed_distributed_sampler_has_no_overlap_or_omission() -> None:
    indices = list(range(17))
    partitions = [
        list(FixedDistributedIndexSampler(indices, rank=rank, world_size=4))
        for rank in range(4)
    ]

    assert sorted(value for partition in partitions for value in partition) == indices
    assert sum(len(partition) for partition in partitions) == len(
        {value for partition in partitions for value in partition}
    )


def test_openloop_eval_can_use_datasets_independent_from_training_mixture(
    monkeypatch,
    tmp_path,
) -> None:
    targets_root = tmp_path / "targets"
    targets_root.mkdir()
    for dataset_name, checkpoint_key in (("eval_a", "alias_a"), ("eval_b", "alias_b")):
        for split in ("vln_val_seen", "vln_val_unseen"):
            path = targets_root / f"{dataset_name}_{split}.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "checkpoint_statistics_key": checkpoint_key,
                        "index": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    built_configs = []

    class _Dataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            raise AssertionError(f"loader construction should not read sample {index}")

    def fake_build_cpm_dataset(config):
        built_configs.append(dict(config))
        return _Dataset()

    monkeypatch.setattr(
        "starVLA.training.openloop_eval.build_cpm_dataset",
        fake_build_cpm_dataset,
    )
    data_cfg = OmegaConf.create(
        {
            "datasets": [
                {
                    "name": "train_only",
                    "data_root_dir": "/train",
                    "dataset_statistics_key": "train_key",
                }
            ],
            "token_budget": 512,
        }
    )
    openloop_cfg = OmegaConf.create(
        {
            "enabled": True,
            "targets_root": str(targets_root),
            "num_workers": 0,
            "datasets": [
                {
                    "name": "eval_a",
                    "eval_root_dir": "/eval/a",
                    "dataset_statistics_key": "eval_key_a",
                    "checkpoint_statistics_key": "alias_a",
                    "required_cameras": ["front"],
                },
                {
                    "name": "eval_b",
                    "eval_root_dir": "/eval/b",
                    "dataset_statistics_key": "eval_key_b",
                    "checkpoint_statistics_key": "alias_b",
                    "required_cameras": ["front", "left"],
                },
            ],
        }
    )

    loaders = build_openloop_eval_loaders(data_cfg, openloop_cfg, rank=0, world_size=1)

    assert len(loaders) == 4
    assert {loader.dataset_name for loader in loaders} == {"eval_a", "eval_b"}
    assert {config["dataset_statistics_key"] for config in built_configs} == {
        "eval_key_a",
        "eval_key_b",
    }
    assert all(config["token_budget"] == 512 for config in built_configs)
    assert all(config["data_root_dir"].startswith("/eval/") for config in built_configs)


def test_openloop_metrics_use_wrapped_yaw_and_padding_mask() -> None:
    stats = {
        "q01": [-1.0, -1.0, -1.0, -np.pi],
        "q99": [1.0, 1.0, 1.0, np.pi],
        "normalization_modes": ["q01_q99"] * 4,
    }
    accumulator = OpenLoopMetricAccumulator()
    target = np.zeros((1, 8, 4), dtype=np.float32)
    predicted = np.zeros_like(target)
    predicted[0, 0, :3] = [1.0, 0.0, 0.0]
    predicted[0, 0, 3] = 2.0
    padding = np.ones((1, 8), dtype=bool)
    padding[0, 0] = False

    accumulator.update(
        predicted_normalized=predicted,
        target_normalized=target,
        padding_mask=padding,
        action_statistics=[stats],
    )
    metrics = accumulator.finalize(duration_seconds=0.5)

    assert metrics["valid_frames"] == 1
    assert metrics["valid_waypoints"] == 1
    assert metrics["translation_l2"]["mean"] == pytest.approx(1.0)
    assert metrics["yaw_abs_wrapped"]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["horizon"]["1"]["count"] == 1
    assert metrics["horizon"]["1"]["normalized_action_mse"] == pytest.approx(1.25)
    assert metrics["horizon"]["1"]["raw_mae"]["dx"] == pytest.approx(1.0)
    assert metrics["horizon"]["1"]["raw_rmse"]["dx"] == pytest.approx(1.0)
    assert metrics["horizon"]["2"]["count"] == 0
    assert metrics["horizon"]["2"]["normalized_action_mse"] is None


def test_openloop_metrics_identity_prediction_is_zero() -> None:
    stats = {
        "q01": [-2.0, -2.0, -2.0, -np.pi],
        "q99": [2.0, 2.0, 2.0, np.pi],
        "normalization_modes": ["q01_q99"] * 4,
    }
    actions = np.linspace(-0.5, 0.5, 2 * 8 * 4, dtype=np.float32).reshape(2, 8, 4)
    padding = np.zeros((2, 8), dtype=bool)
    accumulator = OpenLoopMetricAccumulator()

    accumulator.update(
        predicted_normalized=actions,
        target_normalized=actions,
        padding_mask=padding,
        action_statistics=[stats, stats],
    )
    metrics = accumulator.finalize(duration_seconds=0.1)

    assert metrics["normalized_action_mse"] == 0.0
    assert metrics["translation_l2"]["mean"] == 0.0
    assert metrics["yaw_abs_wrapped"]["mean"] == 0.0
    assert metrics["endpoint_translation_error"]["mean"] == 0.0


def test_flatten_openloop_metrics_selects_compact_analysis_metrics() -> None:
    split_metrics = {
        "normalized_action_mse": np.float64(0.25),
        "raw_mae": {"dx": 1.0, "dy": 2.0, "dz": 3.0, "dyaw": 4.0},
        "raw_rmse": {"dx": 5.0, "dy": 6.0, "dz": 7.0, "dyaw": 8.0},
        "translation_l2": {"mean": 1.5, "p50": 1.0, "p90": 3.0, "count": 800},
        "yaw_abs_wrapped": {"mean": 0.2, "p50": 0.1, "p90": 0.5, "count": 800},
        "endpoint_translation_error": {
            "mean": 2.5,
            "p50": 2.0,
            "p90": 5.0,
            "count": 800,
        },
        "horizon": {
            str(index): {
                "normalized_action_mse": index / 10.0,
                "raw_mae": {"dx": 1.0, "dy": 2.0, "dz": 3.0, "dyaw": 4.0},
                "raw_rmse": {"dx": 5.0, "dy": 6.0, "dz": 7.0, "dyaw": 8.0},
                "translation_l2_mean": float(index),
                "yaw_abs_mean": index / 100.0,
                "count": 100,
            }
            for index in range(1, 9)
        },
        "valid_frames": 800,
        "valid_waypoints": 6400,
        "duration_seconds": 10.0,
        "errors": 0,
    }
    report = {
        "macro_normalized_action_mse": 0.25,
        "datasets": {
            "openfly": {
                "vln_val_seen": split_metrics,
                "vln_val_unseen": split_metrics,
                "combined": split_metrics,
            }
        },
    }

    flattened = flatten_openloop_metrics(report)

    assert len(flattened) == 28
    assert flattened["openloop/macro_normalized_action_mse"] == 0.25
    assert flattened["openloop/openfly/vln_val_unseen/translation_l2/mean"] == 1.5
    assert flattened["openloop/openfly/combined/raw_mae/dz"] == 3.0
    assert flattened["openloop/openfly/combined/yaw_abs_wrapped/p90"] == 0.5
    assert flattened["openloop/openfly/combined/horizon/8/normalized_action_mse"] == 0.8
    assert "openloop/openfly/combined/raw_rmse/dx" not in flattened
    assert "openloop/openfly/combined/translation_l2/p50" not in flattened
    assert "openloop/openfly/combined/horizon/2/normalized_action_mse" not in flattened
    assert "openloop/openfly/combined/horizon/8/raw_mae/dx" not in flattened
    assert "openloop/openfly/combined/valid_frames" not in flattened
    assert "openloop/openfly/combined/duration_seconds" not in flattened


def test_write_openloop_metrics_preserves_full_local_report(tmp_path) -> None:
    report = {
        "macro_normalized_action_mse": 0.25,
        "datasets": {
            "openfly": {
                "combined": {
                    "raw_rmse": {"dx": 5.0},
                    "horizon": {"2": {"raw_mae": {"dx": 1.0}, "count": 100}},
                    "valid_frames": 800,
                    "duration_seconds": 10.0,
                }
            }
        },
    }

    path = write_openloop_metrics(tmp_path, step=100, metrics=report)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["step"] == 100
    assert saved["datasets"]["openfly"]["combined"]["raw_rmse"]["dx"] == 5.0
    assert saved["datasets"]["openfly"]["combined"]["horizon"]["2"]["count"] == 100
    assert saved["datasets"]["openfly"]["combined"]["valid_frames"] == 800
    assert saved["datasets"]["openfly"]["combined"]["duration_seconds"] == 10.0


def test_openloop_trigger_is_step_zero_then_every_eval_interval() -> None:
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            eval_interval=100,
            openloop_eval={
                "enabled": True,
                "run_at_step_zero": True,
            },
        )
    )
    trainer.openloop_eval_loaders = [object()]
    trainer.resume_state_checkpoint_path = None
    trainer._last_openloop_eval_step = None

    assert trainer._should_run_openloop_eval(step=0, fresh_only=True)
    assert trainer._should_run_openloop_eval(step=100)
    assert not trainer._should_run_openloop_eval(step=101)
    trainer._last_openloop_eval_step = 100
    assert not trainer._should_run_openloop_eval(step=100)


def test_openloop_step_zero_does_not_repeat_after_resume() -> None:
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            eval_interval=100,
            openloop_eval={
                "enabled": True,
                "run_at_step_zero": True,
            },
        )
    )
    trainer.openloop_eval_loaders = [object()]
    trainer.resume_state_checkpoint_path = "/checkpoint/steps_0_state"
    trainer._last_openloop_eval_step = None

    assert not trainer._should_run_openloop_eval(step=0, fresh_only=True)
