from __future__ import annotations

from typing import Any

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
            raise ValueError(
                f"{name} must have shape [N, {self.tvi_dim}], got {tuple(value.shape)}"
            )
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
            raise ValueError(f"source_tvi length {int(source_tvi.shape[0])} is shorter than source token count {source_count}")
        if int(source_mask.shape[0]) < source_count:
            raise ValueError(f"source_mask length {int(source_mask.shape[0])} is shorter than source token count {source_count}")
        if len(source_blocks) < source_count:
            raise ValueError(f"source_blocks length {len(source_blocks)} is shorter than source token count {source_count}")

        token_outputs: list[torch.Tensor] = []
        tvi_outputs: list[torch.Tensor] = []
        output_blocks: list[dict[str, Any]] = []

        source_mask = source_mask[:source_count]
        for camera_name in required_cameras:
            camera_matches = torch.tensor(
                [
                    str(block.get("camera_name", "")) == str(camera_name)
                    for block in source_blocks[:source_count]
                ],
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

        projected = self.project_source_tokens(source_tokens) if source_count else source_tokens.new_zeros(
            (0, self.long_memory_visual_tokens, hidden_dim)
        )
        previous_by_camera: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = {}
        if previous_tokens is not None:
            if previous_tokens.ndim != 3:
                raise ValueError(
                    f"previous_tokens must have shape [C, long_tokens, hidden], got {tuple(previous_tokens.shape)}"
                )
            previous_tokens = previous_tokens.to(device=source_tokens.device, dtype=source_tokens.dtype)
            if int(previous_tokens.shape[1]) != self.long_memory_visual_tokens or int(previous_tokens.shape[2]) != hidden_dim:
                raise ValueError("previous long-memory token shape does not match the configured aggregator")
            if previous_tvi is None:
                raise ValueError("previous_tvi is required when previous_tokens are provided")
            previous_tvi = self._validate_tvi(
                previous_tvi,
                name="previous_tvi",
                device=source_tokens.device,
                dtype=source_tokens.dtype,
            )
            if int(previous_tvi.shape[0]) < int(previous_tokens.shape[0]) or len(previous_blocks) < int(previous_tokens.shape[0]):
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
