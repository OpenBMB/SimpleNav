"""One model configuration and one checkpoint asset layout."""

import json
from dataclasses import asdict
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig


def merge_framework_config(default_config_cls, cfg):
    if cfg is None:
        cfg = OmegaConf.create({"framework": {}})
    elif isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    if isinstance(cfg, AccessTrackedConfig):
        accessed = cfg.framework._local_accessed.copy()
        framework = cfg.framework.unwrap()
        cfg.framework = OmegaConf.merge(asdict(default_config_cls()), framework)
        cfg.framework._local_accessed.update(accessed)
    else:
        cfg.framework = OmegaConf.merge(asdict(default_config_cls()), cfg.framework)
    return cfg


def read_mode_config(pretrained_checkpoint):
    checkpoint = Path(pretrained_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.suffix not in {".pt", ".safetensors"}:
        raise ValueError("Expected a .pt or .safetensors checkpoint")
    run = checkpoint.parents[1]
    config = OmegaConf.to_container(OmegaConf.load(run / "config.full.yaml"), resolve=True)
    statistics = json.loads((run / "dataset_statistics.json").read_text())
    return config, statistics
