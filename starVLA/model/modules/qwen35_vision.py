"""Shared Qwen3.5 visual preprocessing and BF16 cache utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

BFLOAT16_BITS_STORAGE_ENCODING = "bfloat16_bits"


def configure_qwen35_processor(processor: Any, input_resize: tuple[int, int]) -> None:
    width, height = (int(input_resize[0]), int(input_resize[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"Qwen3.5 input size must be positive, got {(width, height)}")
    image_processor = processor.image_processor
    pixels = width * height
    image_processor.size = {"shortest_edge": pixels, "longest_edge": pixels}
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = pixels
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = pixels


def qwen35_premerge_token_count(grid_thw: Any) -> int:
    grid = torch.as_tensor(grid_thw, dtype=torch.long).reshape(3)
    if (grid <= 0).any():
        raise ValueError(f"Qwen3.5 grid_thw values must be positive, got {grid.tolist()}")
    return int(grid.prod().item())


def qwen35_postmerge_token_count(grid_thw: Any, *, spatial_merge_size: int = 2) -> int:
    grid = torch.as_tensor(grid_thw, dtype=torch.long).reshape(3)
    merge = int(spatial_merge_size)
    if merge <= 0 or int(grid[1]) % merge or int(grid[2]) % merge:
        raise ValueError(f"Qwen3.5 grid {grid.tolist()} is not divisible by spatial_merge_size={merge}")
    return int(grid.prod().item() // (merge * merge))


def encode_qwen35_postmerge_one_by_one(
    visual: Any,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> list[torch.Tensor]:
    """Run each image independently for stable offline cache extraction."""
    pixel_chunks = torch.split(pixel_values, grid_thw.prod(-1).tolist())
    outputs: list[torch.Tensor] = []
    for pixel_chunk, grid in zip(pixel_chunks, grid_thw, strict=True):
        result = visual(pixel_chunk, grid_thw=grid.reshape(1, 3), return_dict=True)
        outputs.append(result.pooler_output)
    return outputs


def encode_qwen35_postmerge_batched(
    visual: Any,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> list[torch.Tensor]:
    """Encode packed images in one vision forward and restore per-image outputs."""
    if grid_thw.ndim != 2 or int(grid_thw.shape[1]) != 3:
        raise ValueError(f"Qwen3.5 grid_thw must have shape [images, 3], got {tuple(grid_thw.shape)}")

    grids = grid_thw.detach().to(device="cpu", dtype=torch.long).tolist()
    premerge_counts = [qwen35_premerge_token_count(grid) for grid in grids]
    expected_premerge = sum(premerge_counts)
    if pixel_values.ndim < 1 or int(pixel_values.shape[0]) != expected_premerge:
        raise ValueError(
            f"Qwen3.5 packed pixels must contain {expected_premerge} patches, got {tuple(pixel_values.shape)}"
        )
    if not grids:
        return []

    result = visual(pixel_values, grid_thw=grid_thw, return_dict=True)
    merged = result.pooler_output
    merge_size = int(visual.spatial_merge_size)
    postmerge_counts = [
        qwen35_postmerge_token_count(grid, spatial_merge_size=merge_size)
        for grid in grids
    ]
    expected_postmerge = sum(postmerge_counts)
    if merged.ndim != 2 or int(merged.shape[0]) != expected_postmerge:
        raise ValueError(
            "Qwen3.5 packed merger output must have shape "
            f"[{expected_postmerge}, hidden], got {tuple(merged.shape)}"
        )
    return list(torch.split(merged, postmerge_counts, dim=0))


def pool_qwen35_postmerge(
    tokens: torch.Tensor,
    grid_thw: Any,
    *,
    target_tokens: int = 4,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    grid = torch.as_tensor(grid_thw, device=tokens.device, dtype=torch.long).reshape(3)
    merge = int(spatial_merge_size)
    t, h, w = (int(value) for value in grid.tolist())
    merged_h, merged_w = h // merge, w // merge
    expected = t * merged_h * merged_w
    if tokens.ndim != 2 or int(tokens.shape[0]) != expected:
        raise ValueError(f"Qwen3.5 post-merger tokens must have shape [{expected}, hidden], got {tuple(tokens.shape)}")
    target = int(target_tokens)
    target_h = int(target**0.5)
    if target_h * target_h != target or t != 1:
        raise ValueError(f"Qwen3.5 2D history pool requires square target tokens and grid_t=1, got {target}, {t}")
    features = tokens.to(torch.float32).transpose(0, 1).reshape(1, int(tokens.shape[-1]), merged_h, merged_w)
    pooled = F.adaptive_avg_pool2d(features, (target_h, target_h))
    return pooled.reshape(int(tokens.shape[-1]), target).transpose(0, 1).contiguous()


def bf16_to_numpy_bits(tokens: torch.Tensor) -> np.ndarray:
    value = tokens.detach().to(torch.bfloat16).cpu().contiguous()
    return value.view(torch.uint16).numpy()


def numpy_bits_to_bf16(tokens: np.ndarray, device: torch.device) -> torch.Tensor:
    raw = np.ascontiguousarray(tokens, dtype=np.uint16)
    return torch.from_numpy(raw).view(torch.bfloat16).to(device)


def decode_qwen35_cache_tokens(
    tokens: Any,
    *,
    storage_encoding: str,
    device: torch.device,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    if str(storage_encoding) == BFLOAT16_BITS_STORAGE_ENCODING:
        if isinstance(tokens, torch.Tensor):
            if tokens.dtype != torch.uint16:
                raise TypeError(
                    "Qwen3.5 bfloat16_bits cache must arrive as torch.uint16, "
                    f"got {tokens.dtype}"
                )
            raw = tokens.detach().to(device="cpu").contiguous().numpy()
        else:
            raw = np.asarray(tokens)
            if raw.dtype != np.dtype(np.uint16):
                raise TypeError(
                    "Qwen3.5 bfloat16_bits cache must arrive as numpy uint16, "
                    f"got {raw.dtype}"
                )
        return numpy_bits_to_bf16(raw, device).to(dtype=model_dtype)
    return torch.as_tensor(tokens, device=device, dtype=model_dtype)


__all__ = [
    "BFLOAT16_BITS_STORAGE_ENCODING",
    "bf16_to_numpy_bits",
    "configure_qwen35_processor",
    "decode_qwen35_cache_tokens",
    "encode_qwen35_postmerge_batched",
    "encode_qwen35_postmerge_one_by_one",
    "numpy_bits_to_bf16",
    "pool_qwen35_postmerge",
    "qwen35_postmerge_token_count",
    "qwen35_premerge_token_count",
]
