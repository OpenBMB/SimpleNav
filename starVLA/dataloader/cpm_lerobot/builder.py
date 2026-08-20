from __future__ import annotations

import os
from typing import Any

import torch.distributed as dist
from torch.utils.data import DataLoader

from starVLA.model.modules.tvi import TIME_YAW_TVI_MODE
from tool.navvla.context_index import DEFAULT_CONTEXT_TOKEN_BUDGET
from tool.navvla.visual_token_cache import DEFAULT_MINICPM_V46_VISUAL_TOKEN_PROFILE

from .collate import NavVLACPMCollator
from .dataset import NavVLACPMDataset
from .mixture import NavVLACPMMixtureDataset
from .sampler import LengthBucketedEpisodeBatchSampler
from .utils import as_bool, as_list


def build_cpm_dataset(data_cfg: Any) -> NavVLACPMDataset | NavVLACPMMixtureDataset:
    entries = as_list(_get(data_cfg, "datasets", []))
    if not entries:
        root = _get(data_cfg, "data_root_dir")
        if root is None:
            raise KeyError("CPM data config requires data_root_dir or a non-empty datasets list")
        return _build_single(data_cfg, root=root)

    datasets: list[NavVLACPMDataset] = []
    statistics_keys: list[str] = []
    checkpoint_statistics_keys: list[str] = []
    for index, entry in enumerate(entries):
        root = _get(entry, "data_root_dir")
        if root is None:
            raise KeyError(f"CPM datasets[{index}] is missing data_root_dir")
        statistics_key = _get(entry, "dataset_statistics_key", None)
        if statistics_key is None:
            raise KeyError(f"CPM datasets[{index}] is missing dataset_statistics_key")
        datasets.append(_build_single(data_cfg, root=root, overrides=entry))
        statistics_keys.append(str(statistics_key))
        checkpoint_statistics_keys.append(str(_get(entry, "checkpoint_statistics_key", statistics_key)))
    if len(datasets) == 1:
        return datasets[0]
    return NavVLACPMMixtureDataset(
        datasets,
        mixture_name=str(_get(data_cfg, "data_mix", "navvla_cpm_mixture")),
        dataset_statistics_keys=statistics_keys,
        checkpoint_statistics_keys=checkpoint_statistics_keys,
    )


def build_cpm_dataloader(data_cfg: Any, *, seed: int = 0) -> DataLoader:
    dataset = build_cpm_dataset(data_cfg)
    shuffle = as_bool(_get(data_cfg, "shuffle", str(_get(data_cfg, "split", "train")) == "train"))
    batch_size = int(_get(data_cfg, "per_device_batch_size", 1))
    num_workers = int(_get(data_cfg, "num_workers", 0))
    common = {
        "dataset": dataset,
        "collate_fn": NavVLACPMCollator(),
        "num_workers": num_workers,
        "pin_memory": as_bool(_get(data_cfg, "pin_memory", False)),
    }
    if num_workers > 0:
        common.update({"persistent_workers": True, "prefetch_factor": 2})
    if shuffle:
        return DataLoader(
            **common,
            batch_sampler=LengthBucketedEpisodeBatchSampler(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                seed=int(seed),
                drop_last=True,
                bucket_width=int(_get(data_cfg, "length_bucket_width", 8)),
                buffer_size=int(_get(data_cfg, "length_bucket_buffer_size", 1024)),
                sync_group_size=_distributed_world_size(),
            ),
        )
    return DataLoader(**common, batch_size=batch_size, shuffle=False)


def _build_single(data_cfg: Any, *, root: Any, overrides: Any | None = None) -> NavVLACPMDataset:
    def value(key: str, default: Any = None) -> Any:
        return _get(overrides, key, _get(data_cfg, key, default))

    required_cameras = value("required_cameras", None)
    image_resize = value("image_resize", None)
    max_online_history_frames = value("max_online_history_frames", None)
    budget_num_cameras = value("budget_num_cameras", None)
    return NavVLACPMDataset(
        root,
        split=str(value("split", "train")),
        dataset_statistics_key=value("dataset_statistics_key", None),
        checkpoint_statistics_key=value("checkpoint_statistics_key", None),
        required_cameras=None if required_cameras is None else [str(item) for item in as_list(required_cameras)],
        image_resize=None if image_resize is None else tuple(int(item) for item in as_list(image_resize)),
        visual_token_mode=str(value("visual_token_mode", "cached_history_online_current")),
        visual_token_profile=str(value("visual_token_profile", DEFAULT_MINICPM_V46_VISUAL_TOKEN_PROFILE)),
        history_sampling_mode=str(value("history_sampling_mode", "bats")),
        max_online_history_frames=(
            None if max_online_history_frames is None else int(max_online_history_frames)
        ),
        token_budget=int(value("token_budget", DEFAULT_CONTEXT_TOKEN_BUDGET)),
        current_visual_tokens=int(value("current_visual_tokens", 64)),
        history_visual_tokens=int(value("history_visual_tokens", 4)),
        tvi_tokens=int(value("tvi_tokens", 1)),
        current_wrapper_tokens=value("current_wrapper_tokens", None),
        history_wrapper_tokens=value("history_wrapper_tokens", None),
        bats_seed=int(value("bats_seed", 42)),
        bats_epsilon=float(value("bats_epsilon", 0.1)),
        bats_k=float(value("bats_k", 4.0)),
        use_dynamic_bats_k=as_bool(value("use_dynamic_bats_k", True)),
        budget_num_cameras=None if budget_num_cameras is None else int(budget_num_cameras),
        include_state=as_bool(value("include_state", False)),
        require_long_memory_tokens=as_bool(value("require_long_memory_tokens", False)),
        allow_missing_long_memory=as_bool(value("allow_missing_long_memory", True)),
        action_extra_dim_mode=str(value("action_extra_dim_mode", "none")),
        action_path_progress_gamma=float(value("action_path_progress_gamma", 2.0)),
        tvi_mode=str(value("tvi_mode", TIME_YAW_TVI_MODE)),
    )


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    raw_world_size = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw_world_size)
    except ValueError as error:
        raise ValueError(f"WORLD_SIZE must be an integer, got {raw_world_size!r}") from error
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    return world_size
