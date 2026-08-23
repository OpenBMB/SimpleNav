from pathlib import Path

import torch.distributed as dist


def build_dataloader(cfg, dataset_py="navvla_cpm_dataset"):
    """Build the Release 01 CPM LeRobot v3 training dataloader."""
    if dataset_py != "navvla_cpm_dataset":
        raise NotImplementedError(
            f"Release 01 only supports dataset_py='navvla_cpm_dataset', got {dataset_py!r}"
        )

    from starVLA.dataloader.cpm_lerobot import build_cpm_dataloader

    data_cfg = cfg.datasets.vla_data
    dataloader = build_cpm_dataloader(data_cfg, seed=int(getattr(cfg, "seed", 0)))
    is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
    if is_rank_zero:
        output_dir = Path(cfg.output_dir)
        dataloader.dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
    return dataloader
