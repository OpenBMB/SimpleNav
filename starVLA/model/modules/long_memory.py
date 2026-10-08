from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class LongMemoryTokenAggregator(nn.Module):
    def __init__(
        self,
        *,
        source_visual_tokens: int = 4,
        long_memory_visual_tokens: int = 128,
        decay: float = 0.9,
        update_weight: float = 0.1,
        tvi_dim: int = 2,
    ) -> None:
        super().__init__()
        self.source_visual_tokens = int(source_visual_tokens)
        self.long_memory_visual_tokens = int(long_memory_visual_tokens)
        self.decay = float(decay)
        self.update_weight = float(update_weight)
        self.tvi_dim = int(tvi_dim)
        if self.source_visual_tokens <= 0:
            raise ValueError(f"source_visual_tokens must be positive, got {source_visual_tokens}")
        if self.long_memory_visual_tokens <= 0:
            raise ValueError(f"long_memory_visual_tokens must be positive, got {long_memory_visual_tokens}")
        if self.tvi_dim <= 0:
            raise ValueError(f"tvi_dim must be positive, got {tvi_dim}")
        self.projection = nn.Parameter(
            torch.empty(self.long_memory_visual_tokens, self.source_visual_tokens, dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.projection.zero_()
            for target_index in range(self.long_memory_visual_tokens):
                source_index = min(
                    self.source_visual_tokens - 1,
                    int(target_index * self.source_visual_tokens / self.long_memory_visual_tokens),
                )
                self.projection[target_index, source_index] = 1.0

    def project_source_tokens(self, source_tokens: torch.Tensor) -> torch.Tensor:
        if source_tokens.ndim != 3:
            raise ValueError(f"source_tokens must have shape [N, 4, hidden], got {tuple(source_tokens.shape)}")
        if int(source_tokens.shape[1]) != self.source_visual_tokens:
            raise ValueError(
                f"source token count {int(source_tokens.shape[1])} does not match "
                f"configured source_visual_tokens={self.source_visual_tokens}"
            )
        projection = self.projection.to(device=source_tokens.device, dtype=source_tokens.dtype)
        return torch.einsum("ls,nsh->nlh", projection, source_tokens)

    def zero_dependency(self, reference: torch.Tensor) -> torch.Tensor:
        if reference.ndim == 0:
            raise ValueError("reference tensor must expose a hidden dimension")
        hidden_dim = int(reference.shape[-1])
        source_tokens = reference.new_zeros((1, self.source_visual_tokens, hidden_dim))
        return self.project_source_tokens(source_tokens).sum() * reference.new_zeros(())

    def _recurrent_weights(self, count: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if count <= 0:
            return torch.empty((0,), device=device, dtype=dtype)
        exponents = torch.arange(count - 1, -1, -1, device=device, dtype=torch.float32)
        weights = torch.pow(torch.tensor(self.decay, device=device, dtype=torch.float32), exponents)
        if count > 1:
            weights[1:] = weights[1:] * float(self.update_weight)
        return weights.to(dtype=dtype)

    def _validate_tvi(
        self,
        value: torch.Tensor,
        *,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if value.ndim != 2 or int(value.shape[1]) != self.tvi_dim:
            raise ValueError(f"{name} must have shape [N, {self.tvi_dim}], got {tuple(value.shape)}")
        return value.to(device=device, dtype=dtype)

    def aggregate_sample(
        self,
        *,
        source_tokens: torch.Tensor,
        source_tvi: torch.Tensor,
        source_mask: torch.Tensor,
        source_blocks: list[dict[str, Any]],
        required_cameras: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        if source_tokens.ndim != 3:
            raise ValueError(f"source_tokens must have shape [N, 4, hidden], got {tuple(source_tokens.shape)}")
        source_count = int(source_tokens.shape[0])
        hidden_dim = int(source_tokens.shape[-1])
        source_tvi = self._validate_tvi(
            source_tvi,
            name="source_tvi",
            device=source_tokens.device,
            dtype=source_tokens.dtype,
        )
        source_mask = source_mask.to(device=source_tokens.device, dtype=torch.bool).reshape(-1)
        if int(source_tvi.shape[0]) < source_count:
            raise ValueError(
                f"source_tvi length {int(source_tvi.shape[0])} is shorter than source token count {source_count}"
            )
        if int(source_mask.shape[0]) < source_count:
            raise ValueError(
                f"source_mask length {int(source_mask.shape[0])} is shorter than source token count {source_count}"
            )
        if len(source_blocks) < source_count:
            raise ValueError(
                f"source_blocks length {len(source_blocks)} is shorter than source token count {source_count}"
            )

        token_outputs: list[torch.Tensor] = []
        tvi_outputs: list[torch.Tensor] = []
        output_blocks: list[dict[str, Any]] = []

        source_mask = source_mask[:source_count]
        for camera_name in required_cameras:
            camera_matches = torch.tensor(
                [str(block.get("camera_name", "")) == str(camera_name) for block in source_blocks[:source_count]],
                device=source_tokens.device,
                dtype=torch.bool,
            )
            indices = torch.nonzero(source_mask & camera_matches, as_tuple=False).flatten()
            source_block_count = int(indices.numel())
            if source_block_count <= 0:
                continue
            weights = self._recurrent_weights(
                source_block_count,
                device=source_tokens.device,
                dtype=source_tokens.dtype,
            )
            camera_source_tokens = source_tokens.index_select(0, indices)
            camera_tvi = source_tvi.index_select(0, indices)
            memory_source_tokens = (camera_source_tokens * weights.view(-1, 1, 1)).sum(dim=0, keepdim=True)
            memory = self.project_source_tokens(memory_source_tokens).squeeze(0)
            memory_tvi = (camera_tvi * weights.view(-1, 1)).sum(dim=0)
            token_outputs.append(memory)
            tvi_outputs.append(memory_tvi)
            last_block = source_blocks[int(indices[-1].item())]
            output_blocks.append(
                {
                    "step_index": int(last_block.get("step_index", int(indices[-1].item()))),
                    "camera_name": str(camera_name),
                    "source_block_count": source_block_count,
                }
            )

        if not token_outputs:
            return (
                source_tokens.new_zeros((0, self.long_memory_visual_tokens, hidden_dim)),
                source_tokens.new_zeros((0, self.tvi_dim)),
                [],
            )
        return torch.stack(token_outputs, dim=0), torch.stack(tvi_outputs, dim=0), output_blocks

    def update_state(
        self,
        *,
        previous_tokens: torch.Tensor | None,
        previous_tvi: torch.Tensor | None,
        previous_blocks: list[dict[str, Any]],
        source_tokens: torch.Tensor,
        source_tvi: torch.Tensor,
        source_mask: torch.Tensor,
        source_blocks: list[dict[str, Any]],
        required_cameras: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        if source_tokens.ndim != 3:
            raise ValueError(f"source_tokens must have shape [N, 4, hidden], got {tuple(source_tokens.shape)}")
        source_count = int(source_tokens.shape[0])
        hidden_dim = int(source_tokens.shape[-1])
        source_tvi = self._validate_tvi(
            source_tvi,
            name="source_tvi",
            device=source_tokens.device,
            dtype=source_tokens.dtype,
        )
        source_mask = source_mask.to(device=source_tokens.device, dtype=torch.bool).reshape(-1)
        if int(source_tvi.shape[0]) < source_count or int(source_mask.shape[0]) < source_count:
            raise ValueError("source_tvi and source_mask must cover every source token block")
        if len(source_blocks) < source_count:
            raise ValueError("source_blocks must cover every source token block")

        projected = (
            self.project_source_tokens(source_tokens)
            if source_count
            else source_tokens.new_zeros((0, self.long_memory_visual_tokens, hidden_dim))
        )
        previous_by_camera: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = {}
        if previous_tokens is not None:
            if previous_tokens.ndim != 3:
                raise ValueError(
                    f"previous_tokens must have shape [C, long_tokens, hidden], got {tuple(previous_tokens.shape)}"
                )
            previous_tokens = previous_tokens.to(device=source_tokens.device, dtype=source_tokens.dtype)
            if (
                int(previous_tokens.shape[1]) != self.long_memory_visual_tokens
                or int(previous_tokens.shape[2]) != hidden_dim
            ):
                raise ValueError("previous long-memory token shape does not match the configured aggregator")
            if previous_tvi is None:
                raise ValueError("previous_tvi is required when previous_tokens are provided")
            previous_tvi = self._validate_tvi(
                previous_tvi,
                name="previous_tvi",
                device=source_tokens.device,
                dtype=source_tokens.dtype,
            )
            if int(previous_tvi.shape[0]) < int(previous_tokens.shape[0]) or len(previous_blocks) < int(
                previous_tokens.shape[0]
            ):
                raise ValueError("previous_tvi and previous_blocks must cover every previous memory block")
            for index in range(int(previous_tokens.shape[0])):
                block = dict(previous_blocks[index])
                previous_by_camera[str(block.get("camera_name", ""))] = (
                    previous_tokens[index],
                    previous_tvi[index],
                    block,
                )

        token_outputs: list[torch.Tensor] = []
        tvi_outputs: list[torch.Tensor] = []
        output_blocks: list[dict[str, Any]] = []
        for camera_name in required_cameras:
            previous = previous_by_camera.get(str(camera_name))
            memory = None if previous is None else previous[0]
            memory_tvi = None if previous is None else previous[1]
            source_block_count = 0 if previous is None else int(previous[2].get("source_block_count", 1))
            last_block = None if previous is None else previous[2]
            indices = [
                index
                for index, block in enumerate(source_blocks[:source_count])
                if bool(source_mask[index].item()) and str(block.get("camera_name", "")) == str(camera_name)
            ]
            for index in indices:
                if memory is None:
                    memory = projected[index]
                    memory_tvi = source_tvi[index]
                else:
                    memory = self.decay * memory + self.update_weight * projected[index]
                    memory_tvi = self.decay * memory_tvi + self.update_weight * source_tvi[index]
                source_block_count += 1
                last_block = source_blocks[index]
            if memory is None or memory_tvi is None or last_block is None:
                continue
            token_outputs.append(memory)
            tvi_outputs.append(memory_tvi)
            output_blocks.append(
                {
                    "step_index": int(last_block.get("step_index", 0)),
                    "camera_name": str(camera_name),
                    "source_block_count": int(source_block_count),
                }
            )

        if not token_outputs:
            return (
                source_tokens.new_zeros((0, self.long_memory_visual_tokens, hidden_dim)),
                source_tokens.new_zeros((0, self.tvi_dim)),
                [],
            )
        return torch.stack(token_outputs, dim=0), torch.stack(tvi_outputs, dim=0), output_blocks


def attach_navvla_long_memory_tokens(
    samples: list[dict[str, Any]],
    *,
    aggregator,
    tvi_dim: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    for sample in samples:
        if sample.get("long_memory_tokens") is not None:
            continue
        source_tokens = sample.get("long_memory_source_tokens")
        if source_tokens is None:
            continue
        source_tokens_tensor = torch.as_tensor(source_tokens, device=device, dtype=dtype)
        if aggregator is None:
            raise ValueError("long_memory source tokens require long_memory_visual_tokens > 0")
        metadata = dict(sample.get("metadata", {}) or {})
        source_blocks = list(metadata.get("long_memory_blocks") or [])
        required_cameras = sample["metadata"]["required_cameras"]
        source_tvi = torch.as_tensor(
            sample.get(
                "long_memory_source_tvi",
                np.zeros((int(source_tokens_tensor.shape[0]), tvi_dim), dtype=np.float32),
            ),
            device=device,
            dtype=dtype,
        )
        if source_tvi.ndim != 2 or int(source_tvi.shape[1]) != tvi_dim:
            raise ValueError(f"long_memory_source_tvi must have shape [N, {tvi_dim}], got {tuple(source_tvi.shape)}")
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
            sample["_long_memory_zero_dependency"] = True
            continue
        tokens, tvi, blocks = aggregator.aggregate_sample(
            source_tokens=source_tokens_tensor,
            source_tvi=source_tvi,
            source_mask=source_mask,
            source_blocks=source_blocks,
            required_cameras=required_cameras,
        )
        sample["long_memory_tokens"] = tokens
        sample["long_memory_tvi"] = tvi.detach().to(torch.float32).cpu().numpy()
        metadata["long_memory_blocks"] = blocks
        sample["metadata"] = metadata


def compute_navvla_online_long_memory_updates(
    samples: list[dict[str, Any]],
    *,
    aggregator,
    tvi_dim: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    if aggregator is None:
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
            None if previous_tokens_value is None else torch.as_tensor(previous_tokens_value, device=device, dtype=dtype)
        )
        previous_tvi = (
            None if previous_tvi_value is None else torch.as_tensor(previous_tvi_value, device=device, dtype=dtype)
        )
        source_tvi = torch.as_tensor(
            sample.get(
                "online_long_memory_update_tvi",
                np.zeros((int(source_tokens_tensor.shape[0]), tvi_dim), dtype=np.float32),
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
        tokens, tvi, blocks = aggregator.update_state(
            previous_tokens=previous_tokens,
            previous_tvi=previous_tvi,
            previous_blocks=list(metadata.get("long_memory_blocks") or []),
            source_tokens=source_tokens_tensor,
            source_tvi=source_tvi,
            source_mask=source_mask,
            source_blocks=list(metadata.get("online_long_memory_update_blocks") or []),
            required_cameras=sample["metadata"]["required_cameras"],
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


__all__ = ["LongMemoryTokenAggregator", "attach_navvla_long_memory_tokens", "compute_navvla_online_long_memory_updates"]
