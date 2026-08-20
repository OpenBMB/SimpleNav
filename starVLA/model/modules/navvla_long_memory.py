"""Framework-neutral orchestration around the NavVLA long-memory module."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def attach_navvla_long_memory_tokens(
    owner: Any,
    samples: list[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    long_memory_visual_tokens = owner._visual_token_budgets()[1]
    for sample in samples:
        if sample.get("long_memory_tokens") is not None:
            continue
        source_tokens = sample.get("long_memory_source_tokens")
        if source_tokens is None:
            continue
        source_tokens_tensor = torch.as_tensor(source_tokens, device=device, dtype=dtype)
        if long_memory_visual_tokens <= 0 or owner.long_memory_aggregator is None:
            raise ValueError("long_memory source tokens require long_memory_visual_tokens > 0")
        metadata = dict(sample.get("metadata", {}) or {})
        source_blocks = list(metadata.get("long_memory_blocks") or [])
        required_cameras = owner._sample_required_cameras(sample)
        source_tvi = torch.as_tensor(
            sample.get(
                "long_memory_source_tvi",
                np.zeros((int(source_tokens_tensor.shape[0]), owner.tvi_dim), dtype=np.float32),
            ),
            device=device,
            dtype=dtype,
        )
        if source_tvi.ndim != 2 or int(source_tvi.shape[1]) != owner.tvi_dim:
            raise ValueError(
                f"long_memory_source_tvi must have shape [N, {owner.tvi_dim}], got {tuple(source_tvi.shape)}"
            )
        source_mask = torch.as_tensor(
            sample.get(
                "long_memory_source_mask",
                np.ones((int(source_tokens_tensor.shape[0]),), dtype=bool),
            ),
            device=device,
            dtype=torch.bool,
        )
        source_slot_count = int(source_tokens_tensor.shape[0])
        source_block_count = len(source_blocks)
        if source_block_count > source_slot_count:
            raise ValueError(
                f"long_memory metadata has {source_block_count} blocks but only {source_slot_count} source token slots"
            )
        if source_block_count < source_slot_count:
            source_tokens_tensor = source_tokens_tensor[:source_block_count]
            source_tvi = source_tvi[:source_block_count]
            source_mask = source_mask.reshape(-1)[:source_block_count]
        source_count = int(source_tokens_tensor.shape[0])
        missing_long_memory = source_count == 0 or not bool(source_mask[:source_count].any().item())
        if missing_long_memory:
            camera_metadata = metadata.get("camera", {}) or {}
            if owner.tvi_dim == 2:
                dummy_tvi = [
                    [0.0, float((camera_metadata.get(name, {}) or {}).get("azimuth_rad", 0.0))]
                    for name in required_cameras
                ]
            else:
                dummy_tvi = np.zeros((len(required_cameras), owner.tvi_dim), dtype=np.float32)
            source_tokens_tensor = source_tokens_tensor.new_zeros(
                (
                    len(required_cameras),
                    owner.long_memory_aggregator.source_visual_tokens,
                    owner.hidden_size,
                )
            )
            source_tvi = torch.as_tensor(dummy_tvi, device=device, dtype=dtype)
            source_mask = torch.ones((len(required_cameras),), device=device, dtype=torch.bool)
            source_blocks = [
                {"step_index": 0, "camera_name": str(name), "missing_long_memory": True}
                for name in required_cameras
            ]
        tokens, tvi, blocks = owner.long_memory_aggregator.aggregate_sample(
            source_tokens=source_tokens_tensor,
            source_tvi=source_tvi,
            source_mask=source_mask,
            source_blocks=source_blocks,
            required_cameras=required_cameras,
        )
        if missing_long_memory:
            sample["_long_memory_zero_dependency"] = True
            for block in blocks:
                block["source_block_count"] = 0
                block["missing_long_memory"] = True
        sample["long_memory_tokens"] = tokens
        sample["long_memory_tvi"] = tvi.detach().to(torch.float32).cpu().numpy()
        metadata["long_memory_blocks"] = blocks
        sample["metadata"] = metadata


def compute_navvla_online_long_memory_updates(
    owner: Any,
    samples: list[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    if owner.long_memory_aggregator is None:
        return []
    updates: list[dict[str, Any]] = []
    for sample in samples:
        source_tokens = sample.get("online_long_memory_update_tokens")
        if source_tokens is None:
            continue
        source_tokens_tensor = torch.as_tensor(source_tokens, device=device, dtype=dtype)
        if int(source_tokens_tensor.shape[0]) == 0:
            continue
        metadata = dict(sample.get("metadata", {}) or {})
        previous_tokens_value = sample.get("long_memory_tokens")
        previous_tvi_value = sample.get("long_memory_tvi")
        previous_tokens = (
            None
            if previous_tokens_value is None
            else torch.as_tensor(previous_tokens_value, device=device, dtype=dtype)
        )
        previous_tvi = (
            None
            if previous_tvi_value is None
            else torch.as_tensor(previous_tvi_value, device=device, dtype=dtype)
        )
        source_tvi = torch.as_tensor(
            sample.get(
                "online_long_memory_update_tvi",
                np.zeros((int(source_tokens_tensor.shape[0]), owner.tvi_dim), dtype=np.float32),
            ),
            device=device,
            dtype=dtype,
        )
        source_mask = torch.as_tensor(
            sample.get(
                "online_long_memory_update_mask",
                np.ones((int(source_tokens_tensor.shape[0]),), dtype=bool),
            ),
            device=device,
            dtype=torch.bool,
        )
        tokens, tvi, blocks = owner.long_memory_aggregator.update_state(
            previous_tokens=previous_tokens,
            previous_tvi=previous_tvi,
            previous_blocks=list(metadata.get("long_memory_blocks") or []),
            source_tokens=source_tokens_tensor,
            source_tvi=source_tvi,
            source_mask=source_mask,
            source_blocks=list(metadata.get("online_long_memory_update_blocks") or []),
            required_cameras=owner._sample_required_cameras(sample),
        )
        updates.append(
            {
                "tokens": tokens.detach().to(torch.float16).cpu().numpy(),
                "tvi": tvi.detach().to(torch.float32).cpu().numpy(),
                "blocks": blocks,
                "frame_index": int(metadata["online_long_memory_update_frame_index"]),
            }
        )
    return updates


__all__ = ["attach_navvla_long_memory_tokens", "compute_navvla_online_long_memory_updates"]
