from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PORTABLE_TRAINING_CONFIG = (
    REPO_ROOT / "examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml"
)
PORTABLE_TRAINING_CONFIGS = {
    "openfly": PORTABLE_TRAINING_CONFIG,
    "aerialvln": REPO_ROOT
    / "examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_aerialvln_portable.yaml",
    "traveluav": REPO_ROOT
    / "examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_traveluav_portable.yaml",
}
PORTABLE_EVAL_CONFIG = REPO_ROOT / "NavVLAeval/openfly/config_portable.yaml"
PORTABLE_EVAL_CONFIGS = tuple(
    REPO_ROOT / relative
    for relative in (
        "NavVLAeval/traveluav/config_portable.yaml",
        "NavVLAeval/aerialvln/config_portable.yaml",
        "NavVLAeval/aerialvln/config_qwen35_tb1024_ph32_s_seen_stop_finalseg0p292_k2.yaml",
        "NavVLAeval/vlnce/r2r/config_portable.yaml",
        "NavVLAeval/vlnce/rxr/config_portable.yaml",
    )
)


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_portable_training_config_preserves_reference_experiment_contract() -> None:
    expected = {
        "openfly": {
            "per_device_batch_size": 5,
            "gradient_accumulation_steps": 6,
            "max_train_steps": 12502,
            "statistics_key": "vln_train_enhanced_lerobot_vln_train",
            "cameras": ["front"],
        },
        "aerialvln": {
            "per_device_batch_size": 3,
            "gradient_accumulation_steps": 10,
            "max_train_steps": 15338,
            "statistics_key": "vln_train_enhanced_lerobot_merged_remaining_20260813_vln_train",
            "cameras": ["front"],
        },
        "traveluav": {
            "per_device_batch_size": 6,
            "gradient_accumulation_steps": 5,
            "max_train_steps": 2593,
            "statistics_key": "traveluav",
            "cameras": ["front", "left", "right", "rear"],
        },
    }

    for name, path in PORTABLE_TRAINING_CONFIGS.items():
        portable = _load_yaml(path)
        data = portable["datasets"]["vla_data"]
        dataset = data["datasets"][0]
        trainer = portable["trainer"]
        launcher = portable["launcher"]

        assert portable["framework"]["name"] == "navvla_qwen35_cpm"
        assert portable["framework"]["qwenvl"]["attn_implementation"] == "flash_attention_2"
        assert portable["framework"]["qwenvl"]["action_placeholder_token"] == "<|fim_pad|>"
        assert portable["framework"]["action_model"]["action_horizon"] == 8
        assert data["token_budget"] == 1024
        assert data["visual_token_mode"] == "cached_history_online_current"
        assert data["per_device_batch_size"] == expected[name]["per_device_batch_size"]
        assert trainer["gradient_accumulation_steps"] == expected[name]["gradient_accumulation_steps"]
        assert trainer["max_train_steps"] == expected[name]["max_train_steps"]
        assert dataset["dataset_statistics_key"] == expected[name]["statistics_key"]
        assert dataset["required_cameras"] == expected[name]["cameras"]
        assert (
            data["per_device_batch_size"]
            * trainer["gradient_accumulation_steps"]
            * launcher["num_processes"]
            == 240
        )

        path_values = [
            portable["run_root_dir"],
            portable["framework"]["qwenvl"]["base_vlm"],
            portable["framework"]["navvla"]["visual_cache_encoder_ckpt"],
            dataset["data_root_dir"],
        ]
        assert all(not Path(value).is_absolute() for value in path_values)


def test_training_launcher_materializes_portable_paths_from_any_working_directory(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "examples/NavVLA/train_files/qwen35/run_train.sh"),
            str(PORTABLE_TRAINING_CONFIG),
            "--dry-run",
        ],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )

    assert f"output_dir: {REPO_ROOT}/local/results/" in completed.stdout
    assert "dry-run complete" in completed.stdout


def test_portable_eval_config_materializes_all_supported_model_paths(tmp_path: Path) -> None:
    source = _load_yaml(PORTABLE_EVAL_CONFIG)
    config_dir = tmp_path / "NavVLAeval/openfly"
    config_dir.mkdir(parents=True)
    local_root = tmp_path / "local"
    checkpoint = local_root / "checkpoints/openfly/final_model/pytorch_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (checkpoint.parent.parent / "dataset_statistics.json").write_text(
        json.dumps({"vln_train_enhanced_lerobot_vln_train": {"action": {}}}),
        encoding="utf-8",
    )
    (local_root / "data/OpenFly/openfly_env/splits").mkdir(parents=True)
    (local_root / "data/OpenFly/openfly_env/splits/seen.txt").touch()
    (local_root / "simulators/airsim_runtime").mkdir(parents=True)
    (local_root / "models/Qwen3.5-4B").mkdir(parents=True)
    config_path = config_dir / "config_portable.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")

    previous = Path.cwd()
    try:
        os.chdir(tmp_path.parent)
        from NavVLAeval.common.config import load_eval_config

        cfg = load_eval_config(config_path)
    finally:
        os.chdir(previous)

    assert cfg.model.checkpoint == checkpoint.resolve()
    assert cfg.model.kwargs["repo_root"] == str(tmp_path.resolve())
    assert cfg.model.kwargs["config_overrides"]["framework"]["qwenvl"]["base_vlm"] == str(
        (local_root / "models/Qwen3.5-4B").resolve()
    )


def test_portable_benchmark_configs_are_relative_and_complete() -> None:
    for path in (PORTABLE_EVAL_CONFIG, *PORTABLE_EVAL_CONFIGS):
        config = _load_yaml(path)
        assert set(config) >= {"benchmark", "input", "model", "dataset", "env", "parallel", "output"}
        text = path.read_text(encoding="utf-8")
        assert "/nfsdata/" not in text
        assert "/data1/" not in text
        assert "/home/" not in text
