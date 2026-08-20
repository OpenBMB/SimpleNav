from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataset
from starVLA.dataloader.cpm_lerobot.collate import NavVLACPMCollator
from tool.navvla.statistics import unnormalize_values, wrap_to_pi


@dataclass(frozen=True)
class OpenLoopEvalLoader:
    dataset_name: str
    split: str
    checkpoint_statistics_key: str
    targets_path: Path
    dataloader: DataLoader
    target_count: int


class FixedDistributedIndexSampler(Sampler[int]):
    def __init__(self, indices: Iterable[int], *, rank: int, world_size: int) -> None:
        values = [int(value) for value in indices]
        if len(values) != len(set(values)):
            raise ValueError("fixed evaluation indices must be unique")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} must be in [0, {world_size})")
        self.indices = values
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_indices = self.indices[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[int]:
        return iter(self.local_indices)

    def __len__(self) -> int:
        return len(self.local_indices)


class OpenLoopMetricAccumulator:
    def __init__(self, *, horizon: int = 8, action_dim: int = 4) -> None:
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.normalized_squared_error = 0.0
        self.normalized_valid_elements = 0
        self.raw_abs_sum = np.zeros((self.action_dim,), dtype=np.float64)
        self.raw_squared_sum = np.zeros((self.action_dim,), dtype=np.float64)
        self.raw_valid_steps = 0
        self.translation_l2: list[float] = []
        self.yaw_abs: list[float] = []
        self.horizon_normalized_squared_sum = np.zeros((self.horizon,), dtype=np.float64)
        self.horizon_normalized_elements = np.zeros((self.horizon,), dtype=np.int64)
        self.horizon_raw_abs_sum = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        self.horizon_raw_squared_sum = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        self.horizon_translation_sum = np.zeros((self.horizon,), dtype=np.float64)
        self.horizon_yaw_sum = np.zeros((self.horizon,), dtype=np.float64)
        self.horizon_count = np.zeros((self.horizon,), dtype=np.int64)
        self.endpoint_error: list[float] = []
        self.valid_frames = 0
        self.valid_waypoints = 0
        self.errors = 0

    def update(
        self,
        *,
        predicted_normalized: np.ndarray,
        target_normalized: np.ndarray,
        padding_mask: np.ndarray,
        action_statistics: Sequence[dict[str, Any]],
    ) -> None:
        predicted = np.asarray(predicted_normalized, dtype=np.float64)
        target = np.asarray(target_normalized, dtype=np.float64)
        padding = np.asarray(padding_mask, dtype=bool)
        if predicted.shape != target.shape:
            raise ValueError(f"prediction/target shape mismatch: {predicted.shape} vs {target.shape}")
        if predicted.ndim != 3 or predicted.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"expected actions [B,{self.horizon},{self.action_dim}], got {predicted.shape}"
            )
        if padding.shape != predicted.shape[:2]:
            raise ValueError(f"padding shape {padding.shape} does not match actions {predicted.shape}")
        if len(action_statistics) != predicted.shape[0]:
            raise ValueError("action_statistics length must match the batch size")

        valid = ~padding
        normalized_diff = predicted - target
        normalized_valid = np.broadcast_to(valid[..., None], predicted.shape)
        self.normalized_squared_error += float(np.square(normalized_diff)[normalized_valid].sum())
        self.normalized_valid_elements += int(normalized_valid.sum())

        for batch_index, stats in enumerate(action_statistics):
            sample_valid = valid[batch_index]
            if not sample_valid.any():
                continue
            raw_prediction = unnormalize_values(predicted[batch_index], stats).astype(np.float64)
            raw_target = unnormalize_values(target[batch_index], stats).astype(np.float64)
            raw_diff = raw_prediction - raw_target
            raw_diff[:, 3] = np.asarray(wrap_to_pi(raw_diff[:, 3]), dtype=np.float64)
            valid_diff = raw_diff[sample_valid]
            self.raw_abs_sum += np.abs(valid_diff).sum(axis=0)
            self.raw_squared_sum += np.square(valid_diff).sum(axis=0)
            self.raw_valid_steps += int(sample_valid.sum())

            translation = np.linalg.norm(valid_diff[:, :3], axis=-1)
            yaw = np.abs(valid_diff[:, 3])
            self.translation_l2.extend(translation.tolist())
            self.yaw_abs.extend(yaw.tolist())
            self.valid_frames += 1
            self.valid_waypoints += int(sample_valid.sum())
            valid_horizons = np.flatnonzero(sample_valid)
            for waypoint_position, horizon_index in enumerate(valid_horizons):
                normalized_horizon_diff = normalized_diff[batch_index, horizon_index]
                self.horizon_normalized_squared_sum[horizon_index] += float(
                    np.square(normalized_horizon_diff).sum()
                )
                self.horizon_normalized_elements[horizon_index] += self.action_dim
                self.horizon_raw_abs_sum[horizon_index] += np.abs(valid_diff[waypoint_position])
                self.horizon_raw_squared_sum[horizon_index] += np.square(
                    valid_diff[waypoint_position]
                )
                self.horizon_translation_sum[horizon_index] += translation[waypoint_position]
                self.horizon_yaw_sum[horizon_index] += yaw[waypoint_position]
                self.horizon_count[horizon_index] += 1
            endpoint_index = int(valid_horizons[-1])
            self.endpoint_error.append(float(np.linalg.norm(raw_diff[endpoint_index, :3])))

    def add_error(self, count: int = 1) -> None:
        self.errors += int(count)

    def distributed_payload(self) -> dict[str, Any]:
        return {
            "normalized_squared_error": self.normalized_squared_error,
            "normalized_valid_elements": self.normalized_valid_elements,
            "raw_abs_sum": self.raw_abs_sum.tolist(),
            "raw_squared_sum": self.raw_squared_sum.tolist(),
            "raw_valid_steps": self.raw_valid_steps,
            "translation_l2": list(self.translation_l2),
            "yaw_abs": list(self.yaw_abs),
            "horizon_normalized_squared_sum": self.horizon_normalized_squared_sum.tolist(),
            "horizon_normalized_elements": self.horizon_normalized_elements.tolist(),
            "horizon_raw_abs_sum": self.horizon_raw_abs_sum.tolist(),
            "horizon_raw_squared_sum": self.horizon_raw_squared_sum.tolist(),
            "horizon_translation_sum": self.horizon_translation_sum.tolist(),
            "horizon_yaw_sum": self.horizon_yaw_sum.tolist(),
            "horizon_count": self.horizon_count.tolist(),
            "endpoint_error": list(self.endpoint_error),
            "valid_frames": self.valid_frames,
            "valid_waypoints": self.valid_waypoints,
            "errors": self.errors,
        }

    @classmethod
    def merge(cls, payloads: Sequence[dict[str, Any]]) -> "OpenLoopMetricAccumulator":
        merged = cls()
        for payload in payloads:
            merged.normalized_squared_error += float(payload["normalized_squared_error"])
            merged.normalized_valid_elements += int(payload["normalized_valid_elements"])
            merged.raw_abs_sum += np.asarray(payload["raw_abs_sum"], dtype=np.float64)
            merged.raw_squared_sum += np.asarray(payload["raw_squared_sum"], dtype=np.float64)
            merged.raw_valid_steps += int(payload["raw_valid_steps"])
            merged.translation_l2.extend(float(value) for value in payload["translation_l2"])
            merged.yaw_abs.extend(float(value) for value in payload["yaw_abs"])
            merged.horizon_normalized_squared_sum += np.asarray(
                payload["horizon_normalized_squared_sum"], dtype=np.float64
            )
            merged.horizon_normalized_elements += np.asarray(
                payload["horizon_normalized_elements"], dtype=np.int64
            )
            merged.horizon_raw_abs_sum += np.asarray(
                payload["horizon_raw_abs_sum"], dtype=np.float64
            )
            merged.horizon_raw_squared_sum += np.asarray(
                payload["horizon_raw_squared_sum"], dtype=np.float64
            )
            merged.horizon_translation_sum += np.asarray(payload["horizon_translation_sum"], dtype=np.float64)
            merged.horizon_yaw_sum += np.asarray(payload["horizon_yaw_sum"], dtype=np.float64)
            merged.horizon_count += np.asarray(payload["horizon_count"], dtype=np.int64)
            merged.endpoint_error.extend(float(value) for value in payload["endpoint_error"])
            merged.valid_frames += int(payload["valid_frames"])
            merged.valid_waypoints += int(payload["valid_waypoints"])
            merged.errors += int(payload["errors"])
        return merged

    def finalize(self, *, duration_seconds: float) -> dict[str, Any]:
        if self.normalized_valid_elements <= 0:
            raise ValueError("open-loop evaluation contains no valid normalized action elements")
        if self.raw_valid_steps <= 0:
            raise ValueError("open-loop evaluation contains no valid raw action waypoints")
        dimension_names = ("dx", "dy", "dz", "dyaw")
        horizon = {}
        for index in range(self.horizon):
            count = int(self.horizon_count[index])
            normalized_elements = int(self.horizon_normalized_elements[index])
            horizon[str(index + 1)] = {
                "normalized_action_mse": (
                    self.horizon_normalized_squared_sum[index] / normalized_elements
                    if normalized_elements
                    else None
                ),
                "raw_mae": {
                    name: self.horizon_raw_abs_sum[index, dimension] / count if count else None
                    for dimension, name in enumerate(dimension_names)
                },
                "raw_rmse": {
                    name: (
                        math.sqrt(self.horizon_raw_squared_sum[index, dimension] / count)
                        if count
                        else None
                    )
                    for dimension, name in enumerate(dimension_names)
                },
                "translation_l2_mean": (
                    self.horizon_translation_sum[index] / count if count else None
                ),
                "yaw_abs_mean": self.horizon_yaw_sum[index] / count if count else None,
                "count": count,
            }
        return {
            "normalized_action_mse": self.normalized_squared_error / self.normalized_valid_elements,
            "raw_mae": {
                name: self.raw_abs_sum[index] / self.raw_valid_steps
                for index, name in enumerate(dimension_names)
            },
            "raw_rmse": {
                name: math.sqrt(self.raw_squared_sum[index] / self.raw_valid_steps)
                for index, name in enumerate(dimension_names)
            },
            "translation_l2": _distribution(self.translation_l2),
            "yaw_abs_wrapped": _distribution(self.yaw_abs),
            "horizon": horizon,
            "endpoint_translation_error": _distribution(self.endpoint_error),
            "valid_frames": int(self.valid_frames),
            "valid_waypoints": int(self.valid_waypoints),
            "duration_seconds": float(duration_seconds),
            "errors": int(self.errors),
        }


def build_openloop_eval_loaders(
    data_cfg: Any,
    openloop_cfg: Any,
    *,
    rank: int,
    world_size: int,
) -> list[OpenLoopEvalLoader]:
    if not _cfg_bool(openloop_cfg, "enabled", False):
        return []
    targets_root = Path(_cfg_get(openloop_cfg, "targets_root"))
    batch_size = int(_cfg_get(openloop_cfg, "per_device_batch_size", 4))
    num_workers = int(_cfg_get(openloop_cfg, "num_workers", 2))
    eval_entries = _cfg_get(openloop_cfg, "datasets", None)
    if eval_entries is None:
        eval_entries = _cfg_get(data_cfg, "datasets", [])
    loaders: list[OpenLoopEvalLoader] = []
    for entry in _cfg_list(eval_entries):
        dataset_name = str(_cfg_get(entry, "name", "")).strip()
        eval_root_dir = _cfg_get(entry, "eval_root_dir", None)
        if not dataset_name or eval_root_dir is None:
            raise KeyError("each open-loop dataset entry requires name and eval_root_dir")
        checkpoint_key = str(
            _cfg_get(entry, "checkpoint_statistics_key", _cfg_get(entry, "dataset_statistics_key"))
        )
        for split in ("vln_val_seen", "vln_val_unseen"):
            targets_path = targets_root / f"{dataset_name}_{split}.jsonl"
            target_rows = _read_jsonl(targets_path)
            mismatched_rows = [
                row
                for row in target_rows
                if str(row.get("dataset")) != dataset_name
                or str(row.get("split")) != split
                or str(row.get("checkpoint_statistics_key")) != checkpoint_key
            ]
            if mismatched_rows:
                raise ValueError(
                    f"target manifest metadata does not match {dataset_name}/{split}/"
                    f"{checkpoint_key}: {targets_path}"
                )
            target_indices = [int(row["index"]) for row in target_rows]
            eval_entry = _plain_mapping(entry)
            eval_entry.update(
                {
                    "data_root_dir": str(Path(eval_root_dir) / split),
                    "split": split,
                    "checkpoint_statistics_key": checkpoint_key,
                }
            )
            eval_data_cfg = _plain_mapping(data_cfg)
            eval_data_cfg.pop("datasets", None)
            eval_data_cfg.update(eval_entry)
            eval_data_cfg["shuffle"] = False
            dataset = build_cpm_dataset(eval_data_cfg)
            if max(target_indices, default=-1) >= len(dataset):
                raise IndexError(
                    f"target index exceeds {dataset_name}/{split} dataset length {len(dataset)}"
                )
            sampler = FixedDistributedIndexSampler(
                target_indices,
                rank=int(rank),
                world_size=int(world_size),
            )
            dataloader_kwargs = {
                "dataset": dataset,
                "batch_size": batch_size,
                "sampler": sampler,
                "shuffle": False,
                "collate_fn": NavVLACPMCollator(),
                "num_workers": num_workers,
                "pin_memory": bool(_cfg_get(data_cfg, "pin_memory", False)),
                "persistent_workers": num_workers > 0,
            }
            if num_workers > 0:
                dataloader_kwargs["prefetch_factor"] = 2
            dataloader = DataLoader(**dataloader_kwargs)
            loaders.append(
                OpenLoopEvalLoader(
                    dataset_name=dataset_name,
                    split=split,
                    checkpoint_statistics_key=checkpoint_key,
                    targets_path=targets_path,
                    dataloader=dataloader,
                    target_count=len(target_indices),
                )
            )
    return loaders


def run_openloop_eval_loader(
    *,
    model: Any,
    loader: OpenLoopEvalLoader,
) -> tuple[dict[str, Any], float]:
    started_at = time.perf_counter()
    accumulator = OpenLoopMetricAccumulator()
    for batch in loader.dataloader:
        output = model.predict_action(examples=batch, use_ddim=True, num_ddim_steps=20)
        metadata = list(batch["metadata"])
        accumulator.update(
            predicted_normalized=_to_numpy(output["normalized_actions"]),
            target_normalized=_to_numpy(batch["action"]),
            padding_mask=_to_numpy(batch["action_padding_mask"]),
            action_statistics=[item["action_statistics"] for item in metadata],
        )
    return accumulator.distributed_payload(), time.perf_counter() - started_at


def write_openloop_metrics(
    output_dir: str | Path,
    *,
    step: int,
    metrics: dict[str, Any],
) -> Path:
    path = Path(output_dir) / "openloop_eval.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": int(step), **metrics}, ensure_ascii=False) + "\n")
    return path


def flatten_openloop_metrics(metrics: Mapping[str, Any]) -> dict[str, float | int]:
    """Select a compact set of quality metrics for experiment tracker logging."""
    flattened: dict[str, float | int] = {}
    _add_scalar_metric(
        metrics.get("macro_normalized_action_mse"),
        key="openloop/macro_normalized_action_mse",
        output=flattened,
    )
    datasets = metrics.get("datasets", {})
    if not isinstance(datasets, Mapping):
        raise TypeError("open-loop report datasets must be mapping-like")
    for dataset_name, split_metrics in sorted(datasets.items()):
        if not isinstance(split_metrics, Mapping):
            raise TypeError(f"open-loop metrics for {dataset_name} must be mapping-like")
        for split_name, values in sorted(split_metrics.items()):
            prefix = f"openloop/{dataset_name}/{split_name}"
            for path in (
                ("normalized_action_mse",),
                ("translation_l2", "mean"),
                ("yaw_abs_wrapped", "mean"),
                ("endpoint_translation_error", "mean"),
            ):
                _copy_metric_path(values, path=path, prefix=prefix, output=flattened)
            if split_name != "combined":
                continue
            for path in (
                ("raw_mae", "dx"),
                ("raw_mae", "dy"),
                ("raw_mae", "dz"),
                ("translation_l2", "p90"),
                ("yaw_abs_wrapped", "p90"),
                ("endpoint_translation_error", "p90"),
            ):
                _copy_metric_path(values, path=path, prefix=prefix, output=flattened)
            for horizon in ("1", "4", "8"):
                for path in (
                    ("horizon", horizon, "normalized_action_mse"),
                    ("horizon", horizon, "translation_l2_mean"),
                    ("horizon", horizon, "yaw_abs_mean"),
                ):
                    _copy_metric_path(values, path=path, prefix=prefix, output=flattened)
    return flattened


def _copy_metric_path(
    metrics: Mapping[str, Any],
    *,
    path: Sequence[str],
    prefix: str,
    output: dict[str, float | int],
) -> None:
    value: Any = metrics
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return
        value = value[key]
    _add_scalar_metric(value, key=f"{prefix}/{'/'.join(path)}", output=output)


def _add_scalar_metric(
    value: Any,
    *,
    key: str,
    output: dict[str, float | int],
) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, np.integer)):
        output[key] = int(value)
        return
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        output[key] = float(value)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "p50": None, "p90": None, "count": 0}
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "count": int(array.size),
    }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"open-loop target manifest does not exist: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"open-loop target manifest is empty: {path}")
    return rows


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _cfg_bool(config: Any, key: str, default: bool) -> bool:
    value = _cfg_get(config, key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "off", "no"}
    return bool(value)


def _cfg_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def _plain_mapping(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        return dict(config.to_dict(resolve=True))
    if hasattr(config, "items"):
        return {str(key): value for key, value in config.items()}
    raise TypeError(f"expected mapping-like config, got {type(config).__name__}")
