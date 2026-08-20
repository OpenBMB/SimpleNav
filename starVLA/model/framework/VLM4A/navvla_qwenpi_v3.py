"""NavVLA TravelUAV-oriented QwenPI_v3 framework."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
import os
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import create_causal_mask

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.framework.VLM4A.QwenPI_v3 import QwenPI_v3DefaultConfig
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.bats import bats_keep_probability
from starVLA.model.modules.tvi import NavVLATVIEmbedding, sinusoidal_scalar_pe
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.vlm.QWen3 import IMAGE_TOKEN_INDEX
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

CAMERA_VIEWPOINT_TO_ID = {"front": 0, "left": 1, "right": 2, "rear": 3, "down": 4}
DEFAULT_NAVVLA_CAMERAS = ["front", "left", "right", "rear"]


def _viewpoint_id(camera_name: str) -> int:
    return CAMERA_VIEWPOINT_TO_ID.get(str(camera_name), 0)


def _image_token_spans(input_ids: torch.Tensor, image_token_id: int) -> list[tuple[int, int]]:
    positions = torch.nonzero(input_ids == image_token_id, as_tuple=False).flatten().tolist()
    if not positions:
        return []

    spans: list[tuple[int, int]] = []
    start = positions[0]
    previous = positions[0]
    for position in positions[1:]:
        if position == previous + 1:
            previous = position
            continue
        spans.append((start, previous + 1))
        start = position
        previous = position
    spans.append((start, previous + 1))
    return spans


def _grid_visual_token_count(grid: torch.Tensor | tuple[int, int, int], *, merge_size: int = 2) -> int:
    if isinstance(grid, torch.Tensor):
        temporal, height, width = [int(value) for value in grid.tolist()]
    else:
        temporal, height, width = [int(value) for value in grid]
    return max(1, int(temporal * height * width // (int(merge_size) ** 2)))


def _grid_shape_for_token_count(
    grid: torch.Tensor,
    target_tokens: int,
    *,
    merge_size: int = 2,
) -> tuple[int, int, int]:
    temporal, height, width = [int(value) for value in grid.tolist()]
    temporal = max(1, temporal)
    merge = max(1, int(merge_size))
    target = max(1, int(target_tokens))
    spatial_target = max(1, int(math.ceil(float(target) / float(temporal))))
    original_h_tokens = max(1, height // merge)
    original_w_tokens = max(1, width // merge)
    aspect = float(original_h_tokens) / float(max(1, original_w_tokens))
    target_h = max(1, int(round(math.sqrt(float(spatial_target) * aspect))))
    while target_h > 1 and spatial_target % target_h != 0:
        target_h -= 1
    target_w = max(1, int(math.ceil(float(spatial_target) / float(target_h))))
    return temporal, target_h * merge, target_w * merge


def _pool_visual_tokens_by_grid(
    visual_tokens: torch.Tensor,
    original_grid: torch.Tensor,
    *,
    target_tokens: int,
    merge_size: int = 2,
) -> torch.Tensor:
    temporal, height, width = [int(value) for value in original_grid.tolist()]
    merge = max(1, int(merge_size))
    token_h = max(1, height // merge)
    token_w = max(1, width // merge)
    target_grid = _grid_shape_for_token_count(original_grid, target_tokens, merge_size=merge)
    target_h = max(1, int(target_grid[1]) // merge)
    target_w = max(1, int(target_grid[2]) // merge)
    target_len = _grid_visual_token_count(target_grid, merge_size=merge)

    if int(visual_tokens.shape[0]) == int(target_len):
        return visual_tokens

    if temporal == 1 and token_h * token_w == visual_tokens.shape[0]:
        pooled = F.adaptive_avg_pool2d(
            visual_tokens.view(1, token_h, token_w, visual_tokens.shape[-1]).permute(0, 3, 1, 2).float(),
            (target_h, target_w),
        )
        return pooled.to(dtype=visual_tokens.dtype).permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)

    pooled = F.adaptive_avg_pool1d(
        visual_tokens.transpose(0, 1).unsqueeze(0).float(),
        target_len,
    )
    return pooled.to(dtype=visual_tokens.dtype).squeeze(0).transpose(0, 1)


def _as_numpy_bool(values: Any) -> np.ndarray:
    return np.asarray(values if values is not None else [], dtype=bool).reshape(-1)


def _as_numpy_tvi(values: Any) -> np.ndarray:
    array = np.asarray(values if values is not None else [], dtype=np.float32)
    return array.reshape(-1, 2) if array.size else np.zeros((0, 2), dtype=np.float32)


def _debug_memory_enabled() -> bool:
    return os.environ.get("NAVVLA_DEBUG_MEMORY", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_memory_summary(prefix: str, *, samples: list[dict[str, Any]], qwen_inputs: dict[str, Any] | None = None, blocks: list[dict[str, Any]] | None = None, vl_embs_list: list[torch.Tensor] | None = None) -> None:
    if not _debug_memory_enabled():
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        max_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        allocated_gb = reserved_gb = max_allocated_gb = 0.0
    sample_history_counts = [
        int(len((sample.get("metadata", {}) or {}).get("history_blocks", []) or []))
        for sample in samples
    ]
    sample_keys = [(sample.get("metadata", {}) or {}).get("context_index_key") for sample in samples]
    details = {
        "rank": rank,
        "prefix": prefix,
        "history_blocks_per_sample": sample_history_counts,
        "context_keys": sample_keys,
        "total_blocks": None if blocks is None else len(blocks),
        "cuda_allocated_gb": round(allocated_gb, 3),
        "cuda_reserved_gb": round(reserved_gb, 3),
        "cuda_max_allocated_gb": round(max_allocated_gb, 3),
    }
    if qwen_inputs is not None:
        input_ids = qwen_inputs.get("input_ids")
        attention_mask = qwen_inputs.get("attention_mask")
        pixel_values = qwen_inputs.get("pixel_values")
        image_grid_thw = qwen_inputs.get("image_grid_thw")
        details.update(
            {
                "input_ids_shape": None if input_ids is None else tuple(input_ids.shape),
                "attention_active": None if attention_mask is None else [int(x) for x in attention_mask.sum(dim=1).detach().cpu().tolist()],
                "pixel_values_shape": None if pixel_values is None else tuple(pixel_values.shape),
                "image_grid_rows": None if image_grid_thw is None else int(image_grid_thw.shape[0]),
            }
        )
    if vl_embs_list is not None:
        details.update(
            {
                "vl_layers": len(vl_embs_list),
                "vl_first_shape": None if not vl_embs_list else tuple(vl_embs_list[0].shape),
                "vl_last_shape": None if not vl_embs_list else tuple(vl_embs_list[-1].shape),
            }
        )
    print(f"[NAVVLA_DEBUG_MEMORY] {details}", flush=True)


def build_navvla_image_sequence(
    sample: dict[str, Any],
    *,
    required_cameras: list[str],
) -> tuple[list[Any], list[dict[str, Any]]]:
    images: list[Any] = []
    blocks: list[dict[str, Any]] = []

    history_images = sample.get("history_images", {}) or {}
    history_tvi = _as_numpy_tvi(sample.get("history_tvi"))
    history_mask = _as_numpy_bool(sample.get("history_mask"))
    metadata = sample.get("metadata", {}) or {}
    history_blocks = list(metadata.get("history_blocks") or [])
    camera_order = {camera_name: index for index, camera_name in enumerate(required_cameras)}

    if history_blocks:
        per_camera_cursor: dict[str, int] = defaultdict(int)
        ordered_records: list[tuple[int, int, int, Any, np.ndarray, str]] = []
        for block_index, block in enumerate(history_blocks):
            camera_name = str(block["camera_name"])
            image_index = per_camera_cursor[camera_name]
            per_camera_cursor[camera_name] += 1
            camera_images = history_images.get(camera_name, [])
            image = camera_images[image_index] if image_index < len(camera_images) else None
            step_index = int(block["step_index"])
            if camera_name not in camera_order:
                continue
            if block_index < len(history_mask) and not bool(history_mask[block_index]):
                continue
            if image is None:
                continue
            tvi = history_tvi[block_index] if block_index < len(history_tvi) else np.asarray([0.0, 0.0], dtype=np.float32)
            ordered_records.append((step_index, camera_order[camera_name], block_index, image, tvi, camera_name))

        for step_index, _camera_index, _block_index, image, tvi, camera_name in sorted(ordered_records):
            images.append(image)
            blocks.append(
                {
                    "is_history": True,
                    "camera_name": camera_name,
                    "time": float(tvi[0]),
                    "phi": float(tvi[1]),
                    "viewpoint_id": _viewpoint_id(camera_name),
                }
            )
    else:
        available_history = max((len(history_images.get(camera, [])) for camera in required_cameras), default=0)
        history_block_index = 0
        for history_index in range(available_history):
            for camera_name in required_cameras:
                camera_images = history_images.get(camera_name, [])
                if history_index >= len(camera_images):
                    continue
                if history_block_index < len(history_mask) and not bool(history_mask[history_block_index]):
                    history_block_index += 1
                    continue
                tvi = (
                    history_tvi[history_block_index]
                    if history_block_index < len(history_tvi)
                    else np.asarray([0.0, 0.0], dtype=np.float32)
                )
                images.append(camera_images[history_index])
                blocks.append(
                    {
                        "is_history": True,
                        "camera_name": camera_name,
                        "time": float(tvi[0]),
                        "phi": float(tvi[1]),
                        "viewpoint_id": _viewpoint_id(camera_name),
                    }
                )
                history_block_index += 1

    current_tvi = _as_numpy_tvi(sample.get("current_tvi"))
    present_current_cameras = [camera_name for camera_name in required_cameras if sample.get("images", {}).get(camera_name) is not None]
    current_tvi_by_camera = {
        camera_name: current_tvi[position]
        for position, camera_name in enumerate(present_current_cameras)
        if position < len(current_tvi)
    }
    for camera_name in required_cameras:
        image = sample.get("images", {}).get(camera_name)
        if image is None:
            continue
        tvi = current_tvi_by_camera.get(
            camera_name,
            np.asarray([float(metadata.get("timestamp", 0.0)), 0.0], dtype=np.float32),
        )
        images.append(image)
        blocks.append(
            {
                "is_history": False,
                "camera_name": camera_name,
                "time": float(tvi[0]),
                "phi": float(tvi[1]),
                "viewpoint_id": _viewpoint_id(camera_name),
            }
        )
    return images, blocks


def build_navvla_cached_history_image_sequence(
    sample: dict[str, Any],
    *,
    required_cameras: list[str],
) -> tuple[list[Any], list[dict[str, Any]]]:
    images: list[Any] = []
    blocks: list[dict[str, Any]] = []
    history_tvi = _as_numpy_tvi(sample.get("history_tvi"))
    history_mask = _as_numpy_bool(sample.get("history_mask"))
    cached_mask = _as_numpy_bool(sample.get("history_cached_mask"))
    metadata = sample.get("metadata", {}) or {}
    history_blocks = list(metadata.get("history_blocks") or [])
    camera_order = {camera_name: index for index, camera_name in enumerate(required_cameras)}
    placeholder = Image.new("RGB", (1, 1), color=(0, 0, 0))

    ordered_records: list[tuple[int, int, int, np.ndarray, str]] = []
    for block_index, block in enumerate(history_blocks):
        camera_name = str(block["camera_name"])
        if camera_name not in camera_order:
            continue
        if block_index < len(history_mask) and not bool(history_mask[block_index]):
            continue
        if block_index < len(cached_mask) and not bool(cached_mask[block_index]):
            continue
        step_index = int(block["step_index"])
        tvi = history_tvi[block_index] if block_index < len(history_tvi) else np.asarray([0.0, 0.0], dtype=np.float32)
        ordered_records.append((step_index, camera_order[camera_name], block_index, tvi, camera_name))

    for _step_index, _camera_index, block_index, tvi, camera_name in sorted(ordered_records):
        images.append(placeholder)
        blocks.append(
            {
                "is_history": True,
                "is_cached_history": True,
                "cached_history_index": int(block_index),
                "camera_name": camera_name,
                "time": float(tvi[0]),
                "phi": float(tvi[1]),
                "viewpoint_id": _viewpoint_id(camera_name),
                "sample": sample,
            }
        )

    current_tvi = _as_numpy_tvi(sample.get("current_tvi"))
    present_current_cameras = [camera_name for camera_name in required_cameras if sample.get("images", {}).get(camera_name) is not None]
    current_tvi_by_camera = {
        camera_name: current_tvi[position]
        for position, camera_name in enumerate(present_current_cameras)
        if position < len(current_tvi)
    }
    for camera_name in required_cameras:
        image = sample.get("images", {}).get(camera_name)
        if image is None:
            continue
        tvi = current_tvi_by_camera.get(
            camera_name,
            np.asarray([float(metadata.get("timestamp", 0.0)), 0.0], dtype=np.float32),
        )
        images.append(image)
        blocks.append(
            {
                "is_history": False,
                "is_cached_history": False,
                "camera_name": camera_name,
                "time": float(tvi[0]),
                "phi": float(tvi[1]),
                "viewpoint_id": _viewpoint_id(camera_name),
                "sample": sample,
            }
        )
    return images, blocks


def build_navvla_action_loss_mask(action_padding_mask: torch.Tensor, *, repeated_diffusion_steps: int) -> torch.Tensor:
    valid = ~action_padding_mask.to(dtype=torch.bool)
    return valid.repeat(int(repeated_diffusion_steps), 1)


def _infer_pad_token_id(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> int:
    if attention_mask is None:
        return 0
    inactive = ~attention_mask.to(dtype=torch.bool)
    pad_candidates = input_ids[inactive]
    if pad_candidates.numel() > 0:
        return int(pad_candidates[0].item())
    return 0


def _active_row_end(attention_mask: torch.Tensor | None, row_index: int, fallback_len: int) -> int:
    if attention_mask is None:
        return int(fallback_len)
    row_mask = attention_mask[row_index]
    if row_mask.ndim != 1:
        return int(fallback_len)
    active_positions = torch.nonzero(row_mask.to(dtype=torch.bool), as_tuple=False).flatten()
    if active_positions.numel() == 0:
        return 0
    return int(active_positions[-1].item()) + 1


def _active_slice(values: torch.Tensor, attention_mask: torch.Tensor | None, row_index: int, start: int, end: int) -> torch.Tensor:
    sliced = values[row_index, start:end]
    if attention_mask is None:
        return sliced
    active = attention_mask[row_index, start:end].to(dtype=torch.bool)
    return sliced[active]


def pool_navvla_history_qwen_inputs(
    qwen_inputs,
    blocks: list[dict[str, Any]],
    image_token_id: int = IMAGE_TOKEN_INDEX,
    *,
    history_visual_tokens: int = 4,
    current_visual_tokens: int = 64,
):
    qwen_inputs = dict(qwen_inputs)
    image_grid_thw = qwen_inputs.get("image_grid_thw", None)
    if image_grid_thw is None:
        qwen_inputs["_navvla_history_image_indices"] = set()
        return qwen_inputs

    input_ids = qwen_inputs["input_ids"]
    attention_mask = qwen_inputs.get("attention_mask", None)
    spans_per_row = [_image_token_spans(row, image_token_id) for row in input_ids]
    total_spans = sum(len(spans) for spans in spans_per_row)
    if total_spans != len(blocks):
        raise ValueError(f"Qwen image span count {total_spans} does not match NavVLA block count {len(blocks)}")
    if total_spans != int(image_grid_thw.shape[0]):
        raise ValueError(f"Qwen image span count {total_spans} does not match image_grid_thw rows {int(image_grid_thw.shape[0])}")

    pooled_grid = image_grid_thw.clone()
    history_indices = {index for index, block in enumerate(blocks) if bool(block.get("is_history", False))}
    target_tokens_by_image = [
        int(history_visual_tokens) if bool(block.get("is_history", False)) else int(current_visual_tokens)
        for block in blocks
    ]
    for image_index, target_tokens in enumerate(target_tokens_by_image):
        temporal, height, width = _grid_shape_for_token_count(image_grid_thw[image_index], target_tokens)
        pooled_grid[image_index, 0] = temporal
        pooled_grid[image_index, 1] = height
        pooled_grid[image_index, 2] = width

    rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    image_cursor = 0
    for row_index, spans in enumerate(spans_per_row):
        chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        cursor = 0
        row_end = _active_row_end(attention_mask, row_index, input_ids.shape[1])
        for start, end in spans:
            if start >= row_end or end > row_end:
                raise ValueError("Qwen image span extends beyond active attention_mask range")
            if start > cursor:
                active_chunk = _active_slice(input_ids, attention_mask, row_index, cursor, start)
                if active_chunk.numel() > 0:
                    chunks.append(active_chunk)
                    if attention_mask is not None:
                        mask_chunks.append(attention_mask.new_ones((active_chunk.shape[0],)))
            target_len = end - start
            target_len = _grid_visual_token_count(pooled_grid[image_cursor])
            chunks.append(input_ids.new_full((target_len,), image_token_id))
            if attention_mask is not None:
                mask_chunks.append(attention_mask.new_ones((target_len,)))
            cursor = end
            image_cursor += 1
        if cursor < row_end:
            active_tail = _active_slice(input_ids, attention_mask, row_index, cursor, row_end)
            if active_tail.numel() > 0:
                chunks.append(active_tail)
                if attention_mask is not None:
                    mask_chunks.append(attention_mask.new_ones((active_tail.shape[0],)))
        rows.append(torch.cat(chunks, dim=0))
        if attention_mask is not None:
            masks.append(torch.cat(mask_chunks, dim=0))

    max_len = max(row.shape[0] for row in rows)
    pad_token_id = _infer_pad_token_id(input_ids, attention_mask)
    padded_ids = input_ids.new_full((len(rows), max_len), pad_token_id)
    padded_mask = attention_mask.new_zeros((len(rows), max_len)) if attention_mask is not None else None
    for row_index, row in enumerate(rows):
        padded_ids[row_index, : row.shape[0]] = row
        if padded_mask is not None:
            padded_mask[row_index, : masks[row_index].shape[0]] = masks[row_index]

    qwen_inputs["input_ids"] = padded_ids
    if padded_mask is not None:
        qwen_inputs["attention_mask"] = padded_mask
    qwen_inputs["image_grid_thw"] = pooled_grid
    qwen_inputs["_navvla_original_image_grid_thw"] = image_grid_thw
    qwen_inputs["_navvla_history_image_indices"] = history_indices
    qwen_inputs["_navvla_history_visual_tokens"] = int(history_visual_tokens)
    qwen_inputs["_navvla_current_visual_tokens"] = int(current_visual_tokens)
    return qwen_inputs


def trim_qwen_inputs_to_online_images(qwen_inputs, online_image_indices: set[int]):
    qwen_inputs = dict(qwen_inputs)
    image_grid_thw = qwen_inputs.get("image_grid_thw", None)
    pixel_values = qwen_inputs.get("pixel_values", None)
    if image_grid_thw is None or pixel_values is None:
        return qwen_inputs

    original_image_grid_thw = qwen_inputs.get("_navvla_original_image_grid_thw", image_grid_thw)
    online_indices_sorted = sorted(int(index) for index in online_image_indices)
    if len(online_indices_sorted) == int(original_image_grid_thw.shape[0]):
        qwen_inputs["_navvla_online_image_grid_thw"] = original_image_grid_thw
        return qwen_inputs

    split_sizes = original_image_grid_thw.prod(-1).tolist()
    pixel_chunks = torch.split(pixel_values, split_sizes)
    if online_indices_sorted:
        qwen_inputs["pixel_values"] = torch.cat([pixel_chunks[index] for index in online_indices_sorted], dim=0)
        qwen_inputs["_navvla_online_image_grid_thw"] = original_image_grid_thw[online_indices_sorted]
    else:
        qwen_inputs["pixel_values"] = pixel_values[:0]
        qwen_inputs["_navvla_online_image_grid_thw"] = original_image_grid_thw[:0]
    qwen_inputs["_navvla_original_image_grid_thw"] = original_image_grid_thw
    return qwen_inputs


def insert_navvla_tvi_prefix_tokens(
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    blocks: list[dict[str, Any]],
    tvi_embeds: torch.Tensor,
    vision_start_token_id: int,
    image_token_id: int = IMAGE_TOKEN_INDEX,
    tvi_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    spans_per_row = [_image_token_spans(row, image_token_id) for row in input_ids]
    total_spans = sum(len(spans) for spans in spans_per_row)
    if total_spans != len(blocks):
        raise ValueError(f"Qwen image span count {total_spans} does not match NavVLA block count {len(blocks)}")
    if tvi_embeds.shape[0] != total_spans:
        raise ValueError(f"TVI embedding count {tvi_embeds.shape[0]} does not match image span count {total_spans}")
    if tvi_embeds.shape[-1] != inputs_embeds.shape[-1]:
        raise ValueError(
            f"TVI hidden size {tvi_embeds.shape[-1]} does not match input embedding size {inputs_embeds.shape[-1]}"
        )

    row_ids: list[torch.Tensor] = []
    row_embeds: list[torch.Tensor] = []
    row_masks: list[torch.Tensor] = []
    block_cursor = 0
    for row_index, spans in enumerate(spans_per_row):
        id_chunks: list[torch.Tensor] = []
        embed_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        cursor = 0
        row_end = _active_row_end(attention_mask, row_index, input_ids.shape[1])
        for start, _end in spans:
            vision_start = start - 1
            if _end > row_end:
                raise ValueError("Qwen image span extends beyond active attention_mask range")
            if vision_start < 0 or int(input_ids[row_index, vision_start].item()) != int(vision_start_token_id):
                raise ValueError("NavVLA TVI insertion expected vision_start immediately before each image span")
            if vision_start > cursor:
                active_ids = _active_slice(input_ids, attention_mask, row_index, cursor, vision_start)
                if active_ids.numel() > 0:
                    active_mask = attention_mask[row_index, cursor:vision_start].to(dtype=torch.bool) if attention_mask is not None else None
                    id_chunks.append(active_ids)
                    embed_chunks.append(
                        inputs_embeds[row_index, cursor:vision_start][active_mask]
                        if active_mask is not None
                        else inputs_embeds[row_index, cursor:vision_start]
                    )
                    if attention_mask is not None:
                        mask_chunks.append(attention_mask.new_ones((active_ids.shape[0],)))
            id_chunks.append(input_ids.new_full((1,), int(tvi_token_id)))
            embed_chunks.append(
                tvi_embeds[block_cursor].to(device=inputs_embeds.device, dtype=inputs_embeds.dtype).view(1, -1)
            )
            if attention_mask is not None:
                mask_chunks.append(attention_mask.new_ones((1,)))
            cursor = vision_start
            block_cursor += 1
        if cursor < row_end:
            active_ids = _active_slice(input_ids, attention_mask, row_index, cursor, row_end)
            if active_ids.numel() > 0:
                active_mask = attention_mask[row_index, cursor:row_end].to(dtype=torch.bool) if attention_mask is not None else None
                id_chunks.append(active_ids)
                embed_chunks.append(
                    inputs_embeds[row_index, cursor:row_end][active_mask]
                    if active_mask is not None
                    else inputs_embeds[row_index, cursor:row_end]
                )
                if attention_mask is not None:
                    mask_chunks.append(attention_mask.new_ones((active_ids.shape[0],)))
        row_ids.append(torch.cat(id_chunks, dim=0))
        row_embeds.append(torch.cat(embed_chunks, dim=0))
        if attention_mask is not None:
            row_masks.append(torch.cat(mask_chunks, dim=0))

    max_len = max(row.shape[0] for row in row_ids)
    pad_token_id = _infer_pad_token_id(input_ids, attention_mask)
    padded_ids = input_ids.new_full((len(row_ids), max_len), pad_token_id)
    padded_embeds = inputs_embeds.new_zeros((len(row_ids), max_len, inputs_embeds.shape[-1]))
    padded_mask = attention_mask.new_zeros((len(row_ids), max_len)) if attention_mask is not None else None
    for row_index, ids in enumerate(row_ids):
        padded_ids[row_index, : ids.shape[0]] = ids
        padded_embeds[row_index, : row_embeds[row_index].shape[0]] = row_embeds[row_index]
        if padded_mask is not None:
            padded_mask[row_index, : row_masks[row_index].shape[0]] = row_masks[row_index]

    visual_pos_masks = padded_ids == int(image_token_id)
    return padded_ids, padded_embeds, padded_mask, visual_pos_masks


def _split_qwen_image_features_by_grid(
    image_features: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    grid: torch.Tensor,
    *,
    merge_size: int = 2,
) -> list[torch.Tensor]:
    if isinstance(image_features, (tuple, list)):
        if len(image_features) != int(grid.shape[0]):
            raise ValueError(f"Qwen image feature chunks {len(image_features)} do not match image_grid_thw rows {int(grid.shape[0])}")
        return list(image_features)
    split_sizes = (grid.prod(-1) // int(merge_size) ** 2).tolist()
    return list(torch.split(image_features, split_sizes))


def _cached_history_feature_for_block(block: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sample = block.get("sample")
    if sample is None:
        raise KeyError("cached history block is missing source sample")
    history_index = int(block.get("cached_history_index", -1))
    cached = sample.get("history_cached_embeds")
    if cached is None:
        raise KeyError("sample is missing history_cached_embeds for cached history block")
    return torch.as_tensor(cached[history_index], device=device, dtype=dtype)


def _cached_history_deepstack_for_block(block: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    sample = block.get("sample")
    if sample is None:
        raise KeyError("cached history block is missing source sample")
    history_index = int(block.get("cached_history_index", -1))
    cached = sample.get("history_cached_deepstack_embeds")
    if cached is None:
        return None
    cached_slice = np.ascontiguousarray(cached[:, history_index])
    cached_tensor = torch.as_tensor(cached_slice, device=device, dtype=dtype)
    if cached_tensor.numel() == 0 or cached_tensor.shape[0] == 0:
        return None
    return cached_tensor


def _pool_navvla_image_features(
    image_features,
    original_grid: torch.Tensor,
    blocks: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
    *,
    history_visual_tokens: int = 4,
    current_visual_tokens: int = 64,
) -> list[torch.Tensor]:
    if len(image_features) != len(blocks):
        raise ValueError(f"Qwen image feature count {len(image_features)} does not match NavVLA block count {len(blocks)}")

    chunks: list[torch.Tensor] = []
    for image_index, tokens in enumerate(image_features):
        block = blocks[image_index]
        target_tokens = int(history_visual_tokens) if bool(block.get("is_history", False)) else int(current_visual_tokens)
        tokens = _pool_visual_tokens_by_grid(tokens, original_grid[image_index], target_tokens=target_tokens)
        chunks.append(tokens.to(device=device, dtype=dtype))
    return chunks


def _attention_mask_for_rope(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    if attention_mask is None or attention_mask.ndim != 4:
        return attention_mask
    rope_mask = torch.diagonal(attention_mask[:, 0], dim1=1, dim2=2)
    if rope_mask.dtype.is_floating_point:
        rope_mask = rope_mask / torch.finfo(rope_mask.dtype).min
        rope_mask = (1.0 - rope_mask).int()
    return rope_mask


def _run_qwen_language_model_with_hidden_states(
    language_model,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    visual_pos_masks: torch.Tensor | None,
    deepstack_visual_embeds: list[torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    causal_mask = create_causal_mask(
        config=language_model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    hidden_states = inputs_embeds
    all_hidden_states: list[torch.Tensor] = []
    position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        all_hidden_states.append(hidden_states)
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = language_model._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )
    hidden_states = language_model.norm(hidden_states)
    all_hidden_states.append(hidden_states)
    return hidden_states, tuple(all_hidden_states)


@dataclass
class NavVLAQwenPI_v3DefaultConfig(QwenPI_v3DefaultConfig):
    name: str = "navvla_qwenpi_v3"
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 4,
            "state_dim": 0,
            "action_horizon": 8,
            "repeated_diffusion_steps": 1,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 2048,
            "num_target_vision_tokens": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                "action_dit_hidden_dim": 1024,
                "dropout": 0.1,
                "final_dropout": False,
                "interleave_self_attention": False,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )
    navvla: dict = field(
        default_factory=lambda: {
            "required_cameras": DEFAULT_NAVVLA_CAMERAS,
            "history_policy": "bats",
            "bats_epsilon": 0.1,
            "bats_seed": 0,
            "history_visual_tokens": 4,
            "current_visual_tokens": 64,
        }
    )


@FRAMEWORK_REGISTRY.register("navvla_qwenpi_v3")
class NavVLA_QwenPI_v3(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(NavVLAQwenPI_v3DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        llm_hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        language_layers = getattr(self.qwen_vl_interface.model.model.language_model, "layers", None)
        num_vl_layers = len(language_layers) if language_layers is not None else 36
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        diffusion_model_cfg = self.config.framework.action_model.diffusion_model_cfg
        action_dit_hidden_dim = diffusion_model_cfg.get("action_dit_hidden_dim", None)
        if action_dit_hidden_dim is None:
            action_dit_hidden_dim = llm_hidden_size
        self.action_dit_hidden_dim = int(action_dit_hidden_dim)

        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=self.action_dit_hidden_dim,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)
        self.project_layers = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if llm_hidden_size == self.action_dit_hidden_dim
                    else nn.Sequential(
                        nn.LayerNorm(llm_hidden_size),
                        nn.Linear(llm_hidden_size, self.action_dit_hidden_dim),
                    )
                )
                for _ in range(self.num_action_dit_layers)
            ]
        )
        self.tvi_embedding = NavVLATVIEmbedding(hidden_size=llm_hidden_size)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _project_vl_hidden_for_action(self, vl_embs_list: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(vl_embs_list) != len(self.project_layers):
            raise ValueError(
                f"Layer number mismatch: got {len(vl_embs_list)} VL layers, "
                f"but project_layers has {len(self.project_layers)} layers."
            )
        return [projector(vl_hidden) for projector, vl_hidden in zip(self.project_layers, vl_embs_list)]

    def _samples_from_batch(self, examples):
        if isinstance(examples, list):
            return examples
        if examples is None:
            raise ValueError("NavVLA_QwenPI_v3.forward requires examples or a collated NavVLA batch.")

        batch_size = len(examples["lang"])
        metadata = list(examples.get("metadata", [{} for _ in range(batch_size)]))
        samples: list[dict[str, Any]] = []
        for index in range(batch_size):
            images = {
                camera: camera_batch[index]
                for camera, camera_batch in examples.get("images", {}).items()
                if index < len(camera_batch) and camera_batch[index] is not None
            }
            history_images = {
                camera: camera_batch[index] if index < len(camera_batch) else []
                for camera, camera_batch in examples.get("history_images", {}).items()
            }
            action = examples["action"][index] if "action" in examples else np.zeros((self.action_horizon, 4), dtype=np.float32)
            action_padding_mask = (
                examples["action_padding_mask"][index]
                if "action_padding_mask" in examples
                else np.zeros((self.action_horizon,), dtype=bool)
            )
            samples.append(
                {
                    "images": images,
                    "current_tvi": examples["current_tvi"][index],
                    "history_images": history_images,
                    "history_tvi": examples["history_tvi"][index],
                    "history_mask": examples["history_mask"][index],
                    **(
                        {"history_cached_embeds": examples["history_cached_embeds"][index]}
                        if "history_cached_embeds" in examples
                        else {}
                    ),
                    **(
                        {"history_cached_deepstack_embeds": examples["history_cached_deepstack_embeds"][index]}
                        if "history_cached_deepstack_embeds" in examples
                        else {}
                    ),
                    **(
                        {"history_cached_mask": examples["history_cached_mask"][index]}
                        if "history_cached_mask" in examples
                        else {}
                    ),
                    "lang": examples["lang"][index],
                    "platform_text": examples.get("platform_text", [""] * batch_size)[index],
                    **(
                        {"state": examples["state"][index]}
                        if "state" in examples
                        and (
                            "state_present" not in examples
                            or bool(examples["state_present"][index])
                        )
                        else {}
                    ),
                    "action": action,
                    "action_padding_mask": action_padding_mask,
                    "distance_to_goal": float(examples.get("distance_to_goal", [0.0] * batch_size)[index]),
                    "metadata": metadata[index],
                }
            )
        return samples

    def _required_cameras(self) -> list[str]:
        return list(self.config.framework.navvla.get("required_cameras", DEFAULT_NAVVLA_CAMERAS))

    def _build_qwen_inputs(self, samples: list[dict[str, Any]]):
        required_cameras = self._required_cameras()
        navvla_config = self.config.framework.navvla
        history_visual_tokens = int(navvla_config.get("history_visual_tokens", 4))
        current_visual_tokens = int(navvla_config.get("current_visual_tokens", 64))
        batch_images: list[list[Any]] = []
        instructions: list[str] = []
        all_blocks: list[dict[str, Any]] = []

        datasets_config = getattr(self.config, "datasets", None)
        vla_data_config = getattr(datasets_config, "vla_data", {}) if datasets_config is not None else {}
        if hasattr(vla_data_config, "get"):
            visual_token_mode = str(vla_data_config.get("visual_token_mode", "online_images"))
        else:
            visual_token_mode = str(getattr(vla_data_config, "visual_token_mode", "online_images"))
        for sample in samples:
            if visual_token_mode == "cached_history_online_current" or "history_cached_embeds" in sample:
                images, blocks = build_navvla_cached_history_image_sequence(
                    sample,
                    required_cameras=required_cameras,
                )
            else:
                images, blocks = build_navvla_image_sequence(
                    sample,
                    required_cameras=required_cameras,
                )
            batch_images.append([to_pil_preserve(image) for image in images])
            all_blocks.extend(blocks)
            platform_text = sample.get("platform_text", "")
            state = np.asarray(sample["state"], dtype=np.float32) if "state" in sample else np.zeros((0,), dtype=np.float32)
            if state.size:
                state_text = self.state2str_transform(state)
                instructions.append(f"{sample['lang']} {platform_text} [STATE] {state_text} [ACTION]".strip())
            else:
                instructions.append(f"{sample['lang']} {platform_text} [ACTION]".strip())

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        qwen_inputs = pool_navvla_history_qwen_inputs(
            qwen_inputs,
            all_blocks,
            image_token_id=IMAGE_TOKEN_INDEX,
            history_visual_tokens=history_visual_tokens,
            current_visual_tokens=current_visual_tokens,
        )
        online_image_indices = {
            index for index, block in enumerate(all_blocks) if not bool(block.get("is_cached_history", False))
        }
        qwen_inputs["_navvla_online_image_indices"] = online_image_indices
        qwen_inputs = trim_qwen_inputs_to_online_images(qwen_inputs, online_image_indices=online_image_indices)
        return qwen_inputs, all_blocks

    def _qwen_forward_for_action(self, qwen_inputs, blocks: list[dict[str, Any]]):
        qwen_model = self.qwen_vl_interface.model
        model = qwen_model.model
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs.get("attention_mask", None)
        pixel_values = qwen_inputs.get("pixel_values", None)
        image_grid_thw = qwen_inputs.get("image_grid_thw", None)
        original_image_grid_thw = qwen_inputs.get("_navvla_original_image_grid_thw", image_grid_thw)
        history_visual_tokens = int(qwen_inputs.get("_navvla_history_visual_tokens", 4))
        current_visual_tokens = int(qwen_inputs.get("_navvla_current_visual_tokens", 64))

        online_image_indices = set(qwen_inputs.get("_navvla_online_image_indices", range(len(blocks))))

        inputs_embeds = model.get_input_embeddings()(input_ids)
        image_mask = None
        deepstack_image_embeds = None
        if pixel_values is not None:
            if online_image_indices and len(online_image_indices) != len(blocks):
                online_indices_sorted = sorted(online_image_indices)
                online_original_grid = qwen_inputs.get("_navvla_online_image_grid_thw", original_image_grid_thw[online_indices_sorted])
                online_pixel_values = pixel_values
                image_embeds_raw, deepstack_raw = model.get_image_features(online_pixel_values, online_original_grid)
                online_feature_chunks = _split_qwen_image_features_by_grid(
                    image_embeds_raw,
                    online_original_grid,
                    merge_size=int(model.visual.spatial_merge_size),
                )
                online_pooled = _pool_navvla_image_features(
                    online_feature_chunks,
                    online_original_grid,
                    [blocks[index] for index in online_indices_sorted],
                    inputs_embeds.device,
                    inputs_embeds.dtype,
                    history_visual_tokens=history_visual_tokens,
                    current_visual_tokens=current_visual_tokens,
                )
                online_feature_by_index = dict(zip(online_indices_sorted, online_pooled))
                image_chunks: list[torch.Tensor] = []
                cached_deepstack_by_index: dict[int, torch.Tensor] = {}
                for block_index, block in enumerate(blocks):
                    if block_index in online_feature_by_index:
                        image_chunks.append(online_feature_by_index[block_index])
                    else:
                        image_chunks.append(_cached_history_feature_for_block(block, device=inputs_embeds.device, dtype=inputs_embeds.dtype))
                        cached_deepstack = _cached_history_deepstack_for_block(block, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                        if cached_deepstack is not None:
                            cached_deepstack_by_index[block_index] = cached_deepstack
                image_embeds = torch.cat(image_chunks, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                if deepstack_raw is not None or cached_deepstack_by_index:
                    pooled_deepstack: list[torch.Tensor] = []
                    online_deepstack_by_layer: list[dict[int, torch.Tensor]] = []
                    if deepstack_raw is not None:
                        for layer_embeds in deepstack_raw:
                            layer_chunks = _split_qwen_image_features_by_grid(
                                layer_embeds,
                                online_original_grid,
                                merge_size=int(model.visual.spatial_merge_size),
                            )
                            layer_chunks = _pool_navvla_image_features(
                                layer_chunks,
                                online_original_grid,
                                [blocks[index] for index in online_indices_sorted],
                                inputs_embeds.device,
                                inputs_embeds.dtype,
                                history_visual_tokens=history_visual_tokens,
                                current_visual_tokens=current_visual_tokens,
                            )
                            online_deepstack_by_layer.append(dict(zip(online_indices_sorted, layer_chunks)))
                    deepstack_layers = len(online_deepstack_by_layer)
                    if cached_deepstack_by_index:
                        deepstack_layers = max(deepstack_layers, next(iter(cached_deepstack_by_index.values())).shape[0])
                    for layer_index in range(deepstack_layers):
                        layer_chunks: list[torch.Tensor] = []
                        for block_index, block in enumerate(blocks):
                            if block_index in online_image_indices:
                                if layer_index >= len(online_deepstack_by_layer):
                                    raise ValueError("online deepstack layer count is smaller than required mixed layer count")
                                layer_chunks.append(online_deepstack_by_layer[layer_index][block_index])
                            else:
                                cached_deepstack = cached_deepstack_by_index.get(block_index)
                                if cached_deepstack is None or layer_index >= cached_deepstack.shape[0]:
                                    raise ValueError("cached history deepstack embeddings are required for mixed forward")
                                layer_chunks.append(cached_deepstack[layer_index])
                        pooled_deepstack.append(torch.cat(layer_chunks, dim=0))
                    deepstack_image_embeds = pooled_deepstack
            else:
                image_embeds, deepstack_image_embeds = model.get_image_features(pixel_values, original_image_grid_thw)
                image_embeds = _pool_navvla_image_features(
                    image_embeds,
                    original_image_grid_thw,
                    blocks,
                    inputs_embeds.device,
                    inputs_embeds.dtype,
                    history_visual_tokens=history_visual_tokens,
                    current_visual_tokens=current_visual_tokens,
                )
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                if deepstack_image_embeds is not None:
                    split_sizes = (original_image_grid_thw.prod(-1) // model.visual.spatial_merge_size**2).tolist()
                    pooled_deepstack: list[torch.Tensor] = []
                    for layer_embeds in deepstack_image_embeds:
                        layer_chunks = _split_qwen_image_features_by_grid(
                            layer_embeds,
                            original_image_grid_thw,
                            merge_size=int(model.visual.spatial_merge_size),
                        )
                        layer_chunks = _pool_navvla_image_features(
                            layer_chunks,
                            original_image_grid_thw,
                            blocks,
                            inputs_embeds.device,
                            inputs_embeds.dtype,
                            history_visual_tokens=history_visual_tokens,
                            current_visual_tokens=current_visual_tokens,
                        )
                        pooled_deepstack.append(torch.cat(layer_chunks, dim=0))
                    deepstack_image_embeds = pooled_deepstack

        visual_pos_masks = image_mask[..., 0] if image_mask is not None else None
        if blocks:
            tvi_values = torch.tensor(
                [[float(block["time"]), float(block["phi"])] for block in blocks],
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            tvi_embeds = self.tvi_embedding(tvi_values)
            input_ids, inputs_embeds, attention_mask, visual_pos_masks = insert_navvla_tvi_prefix_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                blocks=blocks,
                tvi_embeds=tvi_embeds,
                image_token_id=IMAGE_TOKEN_INDEX,
                vision_start_token_id=int(model.config.vision_start_token_id),
                tvi_token_id=_infer_pad_token_id(input_ids, attention_mask),
            )
        attention_mask_for_rope = _attention_mask_for_rope(attention_mask)
        position_ids, rope_deltas = model.get_rope_index(
            input_ids,
            image_grid_thw,
            None,
            attention_mask=attention_mask_for_rope,
        )
        model.rope_deltas = rope_deltas
        last_hidden_state, hidden_states = _run_qwen_language_model_with_hidden_states(
            model.language_model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_image_embeds,
        )
        return SimpleNamespace(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            rope_deltas=rope_deltas,
            attention_mask=attention_mask,
        )

    def forward(self, examples: list[dict] = None, **kwargs):
        samples = self._samples_from_batch(examples)
        qwen_inputs, blocks = self._build_qwen_inputs(samples)
        _debug_memory_summary("after_build_qwen_inputs", samples=samples, qwen_inputs=qwen_inputs, blocks=blocks)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self._qwen_forward_for_action(qwen_inputs, blocks)
            all_hidden = outputs.hidden_states
            vl_embs_list = list(all_hidden[-self.num_action_dit_layers :])
            vl_embs_list = self._project_vl_hidden_for_action(vl_embs_list)
            _debug_memory_summary("before_action_head", samples=samples, qwen_inputs=qwen_inputs, blocks=blocks, vl_embs_list=vl_embs_list)
            base_hidden = vl_embs_list[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.asarray([sample["action"] for sample in samples]), device=base_hidden.device, dtype=base_hidden.dtype)
            action_padding_mask = torch.tensor(
                np.asarray([sample["action_padding_mask"] for sample in samples]),
                device=base_hidden.device,
                dtype=torch.bool,
            )
            actions_target = actions[:, -self.action_horizon :, :]
            action_padding_mask = action_padding_mask[:, -self.action_horizon :]
            repeated_diffusion_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 1))
            actions_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [hidden.repeat(repeated_diffusion_steps, 1, 1) for hidden in vl_embs_list]
            loss_mask = build_navvla_action_loss_mask(action_padding_mask, repeated_diffusion_steps=repeated_diffusion_steps)
            action_loss = self.action_model(vl_embs_list_repeated, actions_repeated, None, loss_mask=loss_mask)
            total_loss = action_loss

        return {
            "action_loss": total_loss,
            "action_dit_loss": action_loss,
            "loss": total_loss,
        }

    @torch.inference_mode()
    def predict_action(self, examples: list[dict] = None, **kwargs):
        samples = self._samples_from_batch(examples)
        qwen_inputs, blocks = self._build_qwen_inputs(samples)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self._qwen_forward_for_action(qwen_inputs, blocks)
            all_hidden = outputs.hidden_states
            vl_embs_list = list(all_hidden[-self.num_action_dit_layers :])
            vl_embs_list = self._project_vl_hidden_for_action(vl_embs_list)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(vl_embs_list, None)

        response = {
            "normalized_actions": pred_actions.detach().cpu().numpy(),
        }
        return response

    def state2str_transform(self, state: np.ndarray) -> str:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        clipped = np.clip(state, -1.0, 1.0)
        discretized_state = np.digitize(clipped, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return " ".join(map(str, discretized_state.tolist()))
