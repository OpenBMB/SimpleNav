import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader
from starVLA.dataloader.cpm_vlm_datasets import make_cpm_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            # shuffle=True
        )        
        if is_rank_zero:
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "airsim_openfly_datasets":
        from starVLA.dataloader.airsim_openfly_datasets import (
            AirSimOpenFlyDataset,
            EpisodeGroupedSampler,
            collate_fn,
        )

        split = cfg.datasets.vla_data.get("split", "train")
        shuffle = cfg.datasets.vla_data.get("shuffle", split == "train")
        shuffle = shuffle not in ["False", "false", False]
        vla_dataset = AirSimOpenFlyDataset(cfg=cfg, split=split)
        sampler = None
        if shuffle:
            sampler = EpisodeGroupedSampler(
                vla_dataset.episode_sample_indices,
                shuffle=True,
                seed=int(getattr(cfg, "seed", 0)),
            )
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.datasets.vla_data.get("num_workers", 4),
            sampler=sampler,
            shuffle=False,
        )
        if is_rank_zero:
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "navvla_lerobot_datasets":
        from starVLA.dataloader.navvla_lerobot_datasets import (
            EpisodeGroupedSampler,
            NavVLALeRobotDataset,
            collate_navvla_batch,
        )

        data_cfg = cfg.datasets.vla_data
        vla_dataset = NavVLALeRobotDataset(
            data_cfg.data_root_dir,
            split=data_cfg.get("split", "train"),
            required_cameras=list(data_cfg.get("required_cameras", ["front", "left", "right", "rear"])),
            image_resize=tuple(data_cfg.image_resize) if data_cfg.get("image_resize", None) is not None else None,
            visual_token_mode=data_cfg.get("visual_token_mode", "online_images"),
            visual_token_profile=data_cfg.get("visual_token_profile", "qwen3_vl_4b_pooled_history"),
            token_budget=int(data_cfg.get("token_budget", 512)),
            stop_distance_positive_m=float(data_cfg.get("stop_distance_positive_m", 3.0)),
            stop_distance_negative_m=float(data_cfg.get("stop_distance_negative_m", 10.0)),
            include_state=data_cfg.get("include_state", False),
        )
        episode_sampling = data_cfg.get("episode_sampling", data_cfg.get("episode_segment_sampling", False))
        episode_sampling = episode_sampling not in ["False", "false", False]
        if episode_sampling:
            sampler = EpisodeGroupedSampler(
                vla_dataset.episode_sample_indices,
                shuffle=data_cfg.get("shuffle", False),
                seed=int(getattr(cfg, "seed", 0)),
            )
            vla_train_dataloader = DataLoader(
                vla_dataset,
                batch_size=data_cfg.per_device_batch_size,
                collate_fn=collate_navvla_batch,
                num_workers=data_cfg.get("num_workers", 4),
                sampler=sampler,
                shuffle=False,
            )
        else:
            vla_train_dataloader = DataLoader(
                vla_dataset,
                batch_size=data_cfg.per_device_batch_size,
                collate_fn=collate_navvla_batch,
                num_workers=data_cfg.get("num_workers", 4),
                shuffle=data_cfg.get("shuffle", False),
            )
        if is_rank_zero:
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "navvla_cpm_dataset":
        from starVLA.dataloader.cpm_lerobot import build_cpm_dataloader

        data_cfg = cfg.datasets.vla_data
        vla_train_dataloader = build_cpm_dataloader(data_cfg, seed=int(getattr(cfg, "seed", 0)))
        if is_rank_zero:
            output_dir = Path(cfg.output_dir)
            vla_train_dataloader.dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
    elif dataset_py == "cpm_vlm_datasets":
        vlm_data_module = make_cpm_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]

        return vlm_train_dataloader
