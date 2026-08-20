from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from NavVLAeval.common import config as config_module
from NavVLAeval.common.config import load_eval_config
from NavVLAeval.common.runner.worker import (
    _mean_adjacent_waypoint_translation,
    _waypoint_motion_stop_threshold,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "examples/NavVLA/train_files/qwen35"
TRAIN_CONFIG = TRAIN_DIR / "navvla_qwen35_cpm_vlnce.yaml"
TRAIN_SCRIPT = TRAIN_DIR / "train_qwen35_cpm_vlnce.sh"
R2R_CONFIG = REPO_ROOT / "NavVLAeval/vlnce/r2r/config_portable.yaml"
RXR_CONFIG = REPO_ROOT / "NavVLAeval/vlnce/rxr/config_portable.yaml"
QUICK_START = REPO_ROOT / "docs/vlnce_train_eval_readme.md"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_vlnce_training_uses_one_rxr_recipe_for_both_evaluations() -> None:
    config = _load_yaml(TRAIN_CONFIG)
    data = config["datasets"]["vla_data"]
    trainer = config["trainer"]

    assert TRAIN_SCRIPT.is_file()
    assert config["framework"]["name"] == "navvla_qwen35_cpm"
    assert config["framework"]["qwenvl"]["base_vlm"] == "local/models/Qwen3.5-4B"
    assert data["datasets"] == [
        {
            "name": "rxr",
            "dataset_statistics_key": "vln_train_train",
            "data_root_dir": "local/data/VLN-CE-Lerobot/RxR/vln_train",
            "required_cameras": ["front", "left", "right", "rear"],
        }
    ]
    assert data["per_device_batch_size"] == 4
    assert trainer["gradient_accumulation_steps"] == 8
    assert trainer["max_train_steps"] == 15716
    assert trainer["num_warmup_steps"] == 471
    assert trainer["save_interval"] == 500
    assert trainer["save_training_state"] is True


def test_vlnce_eval_configs_share_qwen35_checkpoint_and_protocol() -> None:
    r2r = _load_yaml(R2R_CONFIG)
    rxr = _load_yaml(RXR_CONFIG)

    for config in (r2r, rxr):
        assert config["model"]["framework_name"] == "navvla_qwen35_cpm"
        assert config["model"]["repo_root"] == "../../.."
        assert config["model"]["checkpoint"] == "../../../local/checkpoints/vlnce/checkpoints/steps_8000_pytorch_model.pt"
        assert config["model"]["config_overrides"]["framework"]["qwenvl"]["base_vlm"] == "../../../local/models/Qwen3.5-4B"
        assert config["dataset"]["required_cameras"] == ["front", "left", "right", "rear"]
        assert config["dataset"]["action_horizon"] == 8
        assert config["env"]["kwargs"]["continuous_control_mode"] == "collision_slide_pose_delta"
        assert config["env"]["kwargs"]["execute_waypoints_per_step"] == 8

    for config in (r2r, rxr):
        assert config["stop_rule"] == "mean_adjacent_waypoint_translation"
        assert config["stop_threshold"] == 0.03


def test_vlnce_relative_paths_resolve_from_each_eval_config(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "_validate_eval_config", lambda _cfg: None)
    r2r = load_eval_config(R2R_CONFIG)
    rxr = load_eval_config(RXR_CONFIG)

    expected_checkpoint = (REPO_ROOT / "local/checkpoints/vlnce/checkpoints/steps_8000_pytorch_model.pt").resolve()
    assert r2r.model.checkpoint == expected_checkpoint
    assert rxr.model.checkpoint == expected_checkpoint
    assert Path(r2r.model.kwargs["repo_root"]) == REPO_ROOT.resolve()
    assert Path(rxr.model.kwargs["repo_root"]) == REPO_ROOT.resolve()


def test_waypoint_motion_stop_is_opt_in_and_matches_final_threshold() -> None:
    assert _waypoint_motion_stop_threshold(SimpleNamespace(raw={})) is None
    assert _waypoint_motion_stop_threshold(
        SimpleNamespace(raw={"stop_rule": "mean_adjacent_waypoint_translation", "stop_threshold": 0.03})
    ) == 0.03

    stopped = np.asarray([[0.0, 0.0, 0.0, 0.0], [0.01, 0.0, 0.0, 0.1], [0.02, 0.0, 0.0, 0.2]])
    moving = np.asarray([[0.0, 0.0, 0.0, 0.0], [0.04, 0.0, 0.0, 0.1], [0.08, 0.0, 0.0, 0.2]])
    assert _mean_adjacent_waypoint_translation(stopped) <= 0.03
    assert _mean_adjacent_waypoint_translation(moving) > 0.03


def test_vlnce_quick_start_names_the_three_public_entrypoints() -> None:
    text = QUICK_START.read_text(encoding="utf-8")

    assert "train_qwen35_cpm_vlnce.sh" in text
    assert "NavVLAeval/vlnce/r2r/run_eval.sh" in text
    assert "NavVLAeval/vlnce/rxr/run_eval.sh" in text
    assert "mean_adjacent_waypoint_translation" in text
    assert "stop_threshold: 0.03" in text
