"""NavVLA MiniCPM-V-4.6 + GR00T framework with TVI, BATS, and long-memory inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.long_memory import LongMemoryTokenAggregator
from starVLA.model.modules.navvla_context import (
    HistoryAugmentationConfig,
    action_model_accepts_padding_mask,
    as_numpy_tvi as _as_numpy_tvi,
    build_navvla_instruction,
    build_navvla_cached_visual_sequence,
    history_augmentation_probabilities,
    mask_history_tvi_embeddings,
    sample_required_cameras,
    samples_from_collated_batch,
    scatter_image_embeddings,
    target_visual_tokens_for_block as _target_visual_tokens_for_block,
)
from starVLA.model.modules.tvi import TIME_YAW_TVI_MODE, NavVLATVIEmbedding, get_tvi_input_dim
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


DEFAULT_HISTORY_VISUAL_TOKENS = 4
DEFAULT_LONG_MEMORY_SOURCE_VISUAL_TOKENS = 4
DEFAULT_LONG_MEMORY_VISUAL_TOKENS = 128
DEFAULT_CURRENT_VISUAL_TOKENS = 64
LONG_MEMORY_DECAY = 0.9
LONG_MEMORY_UPDATE_WEIGHT = 0.1


def _image_token_spans(input_ids: torch.Tensor, image_token_id: int) -> list[tuple[int, int]]:
    positions = torch.nonzero(input_ids == int(image_token_id), as_tuple=False).flatten().tolist()
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
    active_positions = torch.nonzero(row_mask.to(dtype=torch.bool), as_tuple=False).flatten()
    if active_positions.numel() == 0:
        return 0
    return int(active_positions[-1].item()) + 1


def _active_slice(
    values: torch.Tensor,
    attention_mask: torch.Tensor | None,
    row_index: int,
    start: int,
    end: int,
) -> torch.Tensor:
    sliced = values[row_index, start:end]
    if attention_mask is None:
        return sliced
    active = attention_mask[row_index, start:end].to(dtype=torch.bool)
    return sliced[active]


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _model_device_dtype(model: Any) -> tuple[torch.device, torch.dtype]:
    device = getattr(model, "device", None)
    dtype = getattr(model, "dtype", None)
    if device is None or dtype is None:
        try:
            first_param = next(model.parameters())
            device = first_param.device if device is None else device
            dtype = first_param.dtype if dtype is None else dtype
        except StopIteration:
            pass
    return torch.device("cpu") if device is None else device, torch.float32 if dtype is None else dtype


def _navigation_tokenizer(vlm_interface: Any) -> Any | None:
    processor = getattr(vlm_interface, "processor", None)
    if processor is None:
        return None
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def _resolve_vision_prefix_token_id(input_ids: torch.Tensor, image_token_id: int, vlm_interface: Any) -> int:
    observed: set[int] = set()
    for row in input_ids:
        for start, _end in _image_token_spans(row, image_token_id):
            if start <= 0:
                raise ValueError("NavVLA CPM TVI insertion requires a vision prefix token before each image span")
            observed.add(int(row[start - 1].item()))
    if len(observed) == 1:
        return next(iter(observed))

    model_config = getattr(getattr(vlm_interface, "model", None), "config", None)
    preferred = getattr(model_config, "vision_start_token_id", None)
    if preferred is not None and int(preferred) in observed:
        return int(preferred)
    if len(observed) > 1:
        raise ValueError(f"found multiple vision prefix token ids before image spans: {sorted(observed)}")
    raise ValueError("could not resolve MiniCPM vision prefix token id")


def _minicpm_image_slot_suffix_token_ids(
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    vlm_interface: Any,
) -> list[int]:
    for row in input_ids:
        spans = _image_token_spans(row, image_token_id)
        if len(spans) < 2:
            continue
        _start, end = spans[0]
        next_prefix = spans[1][0] - 1
        if next_prefix > end:
            suffix = [int(token) for token in row[end:next_prefix].tolist()]
            if suffix:
                return suffix

    tokenizer = _navigation_tokenizer(vlm_interface)
    if tokenizer is not None:
        image_end_token_id = tokenizer.convert_tokens_to_ids("</image>")
        if image_end_token_id is not None and image_end_token_id != getattr(tokenizer, "unk_token_id", None):
            newline_token_ids = [int(token_id) for token_id in tokenizer.encode("\n", add_special_tokens=False)]
            if not newline_token_ids or any(
                token_id == getattr(tokenizer, "unk_token_id", None) for token_id in newline_token_ids
            ):
                raise ValueError("could not resolve MiniCPM newline token ids for cached history image slots")
            fallback = [int(image_end_token_id), *newline_token_ids]
            return fallback

    for row in input_ids:
        spans = _image_token_spans(row, image_token_id)
        if spans:
            _start, end = spans[0]
            if end < int(row.shape[0]):
                return [int(row[end].item())]
    raise ValueError("could not resolve MiniCPM image slot suffix token ids")


def prepend_navvla_cached_minicpm_spans(
    minicpm_inputs: dict[str, Any],
    *,
    cached_spans_per_row: list[int],
    image_token_id: int,
    vision_prefix_token_id: int,
    image_slot_suffix_token_ids: list[int],
) -> dict[str, Any]:
    minicpm_inputs = dict(minicpm_inputs)
    input_ids = minicpm_inputs["input_ids"]
    attention_mask = minicpm_inputs.get("attention_mask")
    rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for row_index, cached_count in enumerate(cached_spans_per_row):
        row = input_ids[row_index]
        row_end = _active_row_end(attention_mask, row_index, row.shape[0])
        spans = _image_token_spans(row, image_token_id)
        if int(cached_count) <= 0:
            compact = _active_slice(input_ids, attention_mask, row_index, 0, row_end)
            rows.append(compact)
            if attention_mask is not None:
                masks.append(attention_mask.new_ones((compact.shape[0],)))
            continue
        if not spans:
            raise ValueError("cached history/long-memory spans require at least one current MiniCPM image slot")

        insert_at = spans[0][0] - 1
        if insert_at < 0 or int(row[insert_at].item()) != int(vision_prefix_token_id):
            raise ValueError("expected MiniCPM vision prefix token immediately before the first current image span")

        cached_ids: list[int] = []
        for _ in range(int(cached_count)):
            cached_ids.extend([int(vision_prefix_token_id), int(image_token_id), *image_slot_suffix_token_ids])
        cached_ids_tensor = row.new_tensor(cached_ids, dtype=row.dtype)

        before = _active_slice(input_ids, attention_mask, row_index, 0, insert_at)
        after = _active_slice(input_ids, attention_mask, row_index, insert_at, row_end)
        rows.append(torch.cat([before, cached_ids_tensor, after], dim=0))
        if attention_mask is not None:
            masks.append(attention_mask.new_ones((before.shape[0] + cached_ids_tensor.shape[0] + after.shape[0],)))

    max_len = max(row.shape[0] for row in rows)
    pad_token_id = _infer_pad_token_id(input_ids, attention_mask)
    padded_ids = input_ids.new_full((len(rows), max_len), pad_token_id)
    padded_mask = attention_mask.new_zeros((len(rows), max_len)) if attention_mask is not None else None
    for row_index, row in enumerate(rows):
        padded_ids[row_index, : row.shape[0]] = row
        if padded_mask is not None:
            padded_mask[row_index, : masks[row_index].shape[0]] = masks[row_index]

    minicpm_inputs["input_ids"] = padded_ids
    if padded_mask is not None:
        minicpm_inputs["attention_mask"] = padded_mask
    for stale_key in ("position_ids", "cache_position", "past_key_values"):
        minicpm_inputs.pop(stale_key, None)
    return minicpm_inputs


def pool_navvla_minicpm_inputs(
    minicpm_inputs: dict[str, Any],
    blocks: list[dict[str, Any]],
    image_token_id: int,
    *,
    history_visual_tokens: int,
    long_memory_visual_tokens: int,
    current_visual_tokens: int,
) -> dict[str, Any]:
    minicpm_inputs = dict(minicpm_inputs)
    input_ids = minicpm_inputs["input_ids"]
    attention_mask = minicpm_inputs.get("attention_mask")
    spans_per_row = [_image_token_spans(row, image_token_id) for row in input_ids]
    total_spans = sum(len(spans) for spans in spans_per_row)
    if total_spans != len(blocks):
        raise ValueError(f"MiniCPM image span count {total_spans} does not match NavVLA block count {len(blocks)}")

    target_tokens_by_block = [
        _target_visual_tokens_for_block(
            block,
            history_visual_tokens=history_visual_tokens,
            long_memory_visual_tokens=long_memory_visual_tokens,
            current_visual_tokens=current_visual_tokens,
        )
        for block in blocks
    ]

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
                raise ValueError("MiniCPM image span extends beyond active attention_mask range")
            if start > cursor:
                active_chunk = _active_slice(input_ids, attention_mask, row_index, cursor, start)
                if active_chunk.numel() > 0:
                    chunks.append(active_chunk)
                    if attention_mask is not None:
                        mask_chunks.append(attention_mask.new_ones((active_chunk.shape[0],)))
            target_len = int(target_tokens_by_block[image_cursor])
            chunks.append(input_ids.new_full((target_len,), int(image_token_id)))
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

    minicpm_inputs["input_ids"] = padded_ids
    if padded_mask is not None:
        minicpm_inputs["attention_mask"] = padded_mask
    minicpm_inputs["_navvla_history_visual_tokens"] = int(history_visual_tokens)
    minicpm_inputs["_navvla_long_memory_visual_tokens"] = int(long_memory_visual_tokens)
    minicpm_inputs["_navvla_current_visual_tokens"] = int(current_visual_tokens)
    return minicpm_inputs


def insert_navvla_tvi_prefix_tokens(
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    blocks: list[dict[str, Any]],
    tvi_embeds: torch.Tensor,
    vision_prefix_token_id: int,
    image_token_id: int,
    tvi_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    spans_per_row = [_image_token_spans(row, image_token_id) for row in input_ids]
    total_spans = sum(len(spans) for spans in spans_per_row)
    if total_spans != len(blocks):
        raise ValueError(f"MiniCPM image span count {total_spans} does not match NavVLA block count {len(blocks)}")
    if int(tvi_embeds.shape[0]) != total_spans:
        raise ValueError(f"TVI embedding count {tvi_embeds.shape[0]} does not match image span count {total_spans}")

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
        for start, end in spans:
            vision_prefix = start - 1
            if end > row_end:
                raise ValueError("MiniCPM image span extends beyond active attention_mask range")
            if vision_prefix < 0 or int(input_ids[row_index, vision_prefix].item()) != int(vision_prefix_token_id):
                raise ValueError("expected MiniCPM vision prefix token immediately before each image span")
            if vision_prefix > cursor:
                active_ids = _active_slice(input_ids, attention_mask, row_index, cursor, vision_prefix)
                if active_ids.numel() > 0:
                    active_mask = (
                        attention_mask[row_index, cursor:vision_prefix].to(dtype=torch.bool)
                        if attention_mask is not None
                        else None
                    )
                    id_chunks.append(active_ids)
                    embed_chunks.append(
                        inputs_embeds[row_index, cursor:vision_prefix][active_mask]
                        if active_mask is not None
                        else inputs_embeds[row_index, cursor:vision_prefix]
                    )
                    if attention_mask is not None:
                        mask_chunks.append(attention_mask.new_ones((active_ids.shape[0],)))
            id_chunks.append(input_ids.new_full((1,), int(tvi_token_id)))
            embed_chunks.append(
                tvi_embeds[block_cursor].to(device=inputs_embeds.device, dtype=inputs_embeds.dtype).view(1, -1)
            )
            if attention_mask is not None:
                mask_chunks.append(attention_mask.new_ones((1,)))
            cursor = vision_prefix
            block_cursor += 1
        if cursor < row_end:
            active_ids = _active_slice(input_ids, attention_mask, row_index, cursor, row_end)
            if active_ids.numel() > 0:
                active_mask = (
                    attention_mask[row_index, cursor:row_end].to(dtype=torch.bool)
                    if attention_mask is not None
                    else None
                )
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
    return padded_ids, padded_embeds, padded_mask


def pool_minicpm_vlm_inputs(
    minicpm_inputs: dict[str, Any],
    *,
    image_token_id: int,
    max_visual_tokens: int,
    original_visual_token_counts: list[int] | None = None,
) -> tuple[dict[str, Any], list[int]]:
    if int(max_visual_tokens) <= 0:
        raise ValueError(f"max_visual_tokens must be positive, got {max_visual_tokens}")

    pooled_inputs = dict(minicpm_inputs)
    input_ids = minicpm_inputs["input_ids"]
    attention_mask = minicpm_inputs.get("attention_mask")
    labels = minicpm_inputs.get("labels")
    spans_per_row = [_image_token_spans(row, image_token_id) for row in input_ids]
    span_lengths = [end - start for spans in spans_per_row for start, end in spans]
    if original_visual_token_counts is None:
        original_visual_token_counts = span_lengths
    if len(original_visual_token_counts) != len(span_lengths):
        raise ValueError(
            f"MiniCPM VLM visual feature count {len(original_visual_token_counts)} does not match image span count {len(span_lengths)}"
        )
    target_counts = [
        min(span_length, int(feature_count), int(max_visual_tokens))
        for span_length, feature_count in zip(span_lengths, original_visual_token_counts)
    ]

    rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    image_cursor = 0
    for row_index, spans in enumerate(spans_per_row):
        id_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        cursor = 0
        row_end = _active_row_end(attention_mask, row_index, input_ids.shape[1])
        for start, end in spans:
            if start >= row_end or end > row_end:
                raise ValueError("MiniCPM VLM image span extends beyond active attention_mask range")
            if start > cursor:
                active_ids = _active_slice(input_ids, attention_mask, row_index, cursor, start)
                if active_ids.numel() > 0:
                    id_chunks.append(active_ids)
                    if attention_mask is not None:
                        mask_chunks.append(attention_mask.new_ones((active_ids.shape[0],)))
                    if labels is not None:
                        label_chunks.append(_active_slice(labels, attention_mask, row_index, cursor, start))

            target_count = target_counts[image_cursor]
            id_chunks.append(input_ids.new_full((target_count,), int(image_token_id)))
            if attention_mask is not None:
                mask_chunks.append(attention_mask.new_ones((target_count,)))
            if labels is not None:
                label_chunks.append(labels[row_index, start : start + target_count])
            cursor = end
            image_cursor += 1

        if cursor < row_end:
            active_ids = _active_slice(input_ids, attention_mask, row_index, cursor, row_end)
            if active_ids.numel() > 0:
                id_chunks.append(active_ids)
                if attention_mask is not None:
                    mask_chunks.append(attention_mask.new_ones((active_ids.shape[0],)))
                if labels is not None:
                    label_chunks.append(_active_slice(labels, attention_mask, row_index, cursor, row_end))

        rows.append(torch.cat(id_chunks, dim=0))
        if attention_mask is not None:
            masks.append(torch.cat(mask_chunks, dim=0))
        if labels is not None:
            label_rows.append(torch.cat(label_chunks, dim=0))

    max_len = max(row.shape[0] for row in rows)
    pad_token_id = _infer_pad_token_id(input_ids, attention_mask)
    padded_ids = input_ids.new_full((len(rows), max_len), pad_token_id)
    padded_mask = attention_mask.new_zeros((len(rows), max_len)) if attention_mask is not None else None
    padded_labels = labels.new_full((len(rows), max_len), -100) if labels is not None else None
    for row_index, row in enumerate(rows):
        padded_ids[row_index, : row.shape[0]] = row
        if padded_mask is not None:
            padded_mask[row_index, : masks[row_index].shape[0]] = masks[row_index]
        if padded_labels is not None:
            padded_labels[row_index, : label_rows[row_index].shape[0]] = label_rows[row_index]

    pooled_inputs["input_ids"] = padded_ids
    if padded_mask is not None:
        pooled_inputs["attention_mask"] = padded_mask
    if padded_labels is not None:
        pooled_inputs["labels"] = padded_labels
    for stale_key in ("position_ids", "cache_position", "past_key_values"):
        pooled_inputs.pop(stale_key, None)
    return pooled_inputs, target_counts


def minicpm_token_hw(num_tokens: int, tgt_size: torch.Tensor | None = None) -> tuple[int, int]:
    if tgt_size is not None and int(tgt_size.numel()) >= 2:
        height = max(1, int(tgt_size.reshape(-1)[0].item()))
        width = max(1, int(tgt_size.reshape(-1)[1].item()))
        if height * width == int(num_tokens):
            return height, width
    side = max(1, int(round(math.sqrt(int(num_tokens)))))
    if side * side == int(num_tokens):
        return side, side
    width = max(1, int(num_tokens) // side)
    height = max(1, int(math.ceil(int(num_tokens) / width)))
    return height, width


def _pool_visual_tokens_to_count(visual_tokens: torch.Tensor, *, target_tokens: int) -> torch.Tensor:
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")
    if int(visual_tokens.shape[0]) == int(target_tokens):
        return visual_tokens
    pooled = F.adaptive_avg_pool1d(
        visual_tokens.transpose(0, 1).unsqueeze(0).float(),
        int(target_tokens),
    )
    return pooled.to(dtype=visual_tokens.dtype).squeeze(0).transpose(0, 1)


def _minicpm_grid_pool2d(
    visual_tokens: torch.Tensor,
    *,
    height: int,
    width: int,
    target_tokens: int,
) -> torch.Tensor:
    target_side = int(math.isqrt(int(target_tokens)))
    if target_side * target_side != int(target_tokens):
        return _pool_visual_tokens_to_count(visual_tokens, target_tokens=target_tokens)
    if visual_tokens.dim() == 2:
        visual_tokens = visual_tokens.unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(
        visual_tokens.view(visual_tokens.shape[0], height, width, visual_tokens.shape[-1]).permute(0, 3, 1, 2).float(),
        (target_side, target_side),
    )
    output = pooled.to(dtype=visual_tokens.dtype).permute(0, 2, 3, 1).flatten(1, 2)
    return output.squeeze(0) if output.shape[0] == 1 else output


def pool_minicpm_visual_tokens_to_count(
    visual_tokens: torch.Tensor,
    *,
    target_tokens: int,
    tgt_size: torch.Tensor | None = None,
) -> torch.Tensor:
    token_count = int(visual_tokens.shape[0])
    if tgt_size is not None and int(tgt_size.numel()) >= 2:
        height, width = minicpm_token_hw(token_count, tgt_size)
        if height * width == token_count:
            return _minicpm_grid_pool2d(
                visual_tokens,
                height=height,
                width=width,
                target_tokens=int(target_tokens),
            )
    return _pool_visual_tokens_to_count(visual_tokens, target_tokens=int(target_tokens))


def _flatten_minicpm_image_features(features: Any) -> torch.Tensor:
    pooler_output = getattr(features, "pooler_output", None)
    if pooler_output is not None:
        features = pooler_output
    if isinstance(features, (list, tuple)):
        chunks = [_flatten_minicpm_image_features(chunk) for chunk in features]
        return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0))
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"unsupported MiniCPM image feature type: {type(features)!r}")
    if features.dim() == 3:
        return features.reshape(-1, features.shape[-1])
    if features.dim() == 2:
        return features
    raise ValueError(f"expected MiniCPM image features with 2 or 3 dims, got shape {tuple(features.shape)}")


def _minicpm_feature_target_sizes(target_sizes: Any, *, downsample_mode: str | None) -> Any:
    if downsample_mode is None:
        return target_sizes
    spatial_factor_by_mode = {"4x": 2, "16x": 4}
    spatial_factor = spatial_factor_by_mode.get(str(downsample_mode))
    if spatial_factor is None:
        raise ValueError(f"unsupported MiniCPM downsample_mode={downsample_mode!r}")

    if isinstance(target_sizes, torch.Tensor):
        return torch.div(target_sizes, spatial_factor, rounding_mode="floor").clamp_min(1)
    if isinstance(target_sizes, np.ndarray):
        return np.maximum(target_sizes // spatial_factor, 1)
    if isinstance(target_sizes, (list, tuple)):
        adjusted = []
        for item in target_sizes:
            if isinstance(item, torch.Tensor):
                adjusted.append(torch.div(item, spatial_factor, rounding_mode="floor").clamp_min(1))
            elif isinstance(item, np.ndarray):
                adjusted.append(np.maximum(item // spatial_factor, 1))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                adjusted.append([max(1, int(item[0]) // spatial_factor), max(1, int(item[1]) // spatial_factor)])
        return adjusted
    return target_sizes


def _split_flat_minicpm_features_by_target_sizes(
    flat_features: torch.Tensor,
    target_sizes: Any,
    *,
    downsample_mode: str | None = None,
) -> list[torch.Tensor]:
    if target_sizes is None:
        return [flat_features]
    sizes = _minicpm_feature_target_sizes(target_sizes, downsample_mode=downsample_mode)
    if isinstance(sizes, torch.Tensor):
        sizes = sizes.detach().cpu().reshape(-1, 2).tolist()
    elif isinstance(sizes, np.ndarray):
        sizes = sizes.reshape(-1, 2).tolist()
    elif isinstance(sizes, (list, tuple)):
        normalized = []
        for item in sizes:
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().reshape(-1).tolist()
            elif isinstance(item, np.ndarray):
                item = item.reshape(-1).tolist()
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                normalized.append([int(item[0]), int(item[1])])
        sizes = normalized
    else:
        return [flat_features]

    num_images = len(sizes)
    if num_images <= 0:
        return [flat_features]
    total_tokens = int(flat_features.shape[0])
    expected_lengths = [max(1, int(height)) * max(1, int(width)) for height, width in sizes]
    if sum(expected_lengths) == total_tokens:
        chunks: list[torch.Tensor] = []
        offset = 0
        for length in expected_lengths:
            chunks.append(flat_features[offset : offset + length])
            offset += length
        return chunks
    if total_tokens % num_images == 0:
        tokens_per_image = total_tokens // num_images
        return [flat_features[index * tokens_per_image : (index + 1) * tokens_per_image] for index in range(num_images)]

    if len(sizes) == 1:
        return [flat_features]
    raise ValueError(f"cannot split MiniCPM features: feature tokens={total_tokens}, target_sizes={sizes}")


def _cached_tokens_for_block(block: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sample = block.get("sample")
    if sample is None:
        raise KeyError("cached visual block is missing source sample")
    if bool(block.get("is_long_memory", False)):
        index = int(block.get("long_memory_index", -1))
        cached = sample.get("long_memory_tokens")
        if cached is None:
            raise KeyError("sample is missing long_memory_tokens for long-memory block")
    else:
        index = int(block.get("cached_history_index", -1))
        cached = sample.get("history_cached_embeds")
        if cached is None:
            raise KeyError("sample is missing history_cached_embeds for history block")
    cached_value = cached[index]
    if isinstance(cached_value, torch.Tensor):
        return cached_value.to(device=device, dtype=dtype)
    return torch.as_tensor(cached_value, device=device, dtype=dtype)


@dataclass
class NavVLACPMDefaultConfig:
    name: str = "navvla_cpm"
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "openbmb/MiniCPM-V-4.6",
            "attn_implementation": "sdpa",
            "vl_hidden_dim": 1024,
            "downsample_mode": "4x",
            "max_slice_nums": 36,
            "use_image_id": False,
            "enable_thinking": False,
            "action_placeholder_token": "◆",
            "action_start_token": "▷",
            "action_end_token": "◯",
        }
    )
    navvla: dict = field(
        default_factory=lambda: {
            "tvi_mode": TIME_YAW_TVI_MODE,
            "use_platform_text": True,
            "history_visual_tokens": DEFAULT_HISTORY_VISUAL_TOKENS,
            "long_memory_source_visual_tokens": DEFAULT_LONG_MEMORY_SOURCE_VISUAL_TOKENS,
            "long_memory_visual_tokens": DEFAULT_LONG_MEMORY_VISUAL_TOKENS,
            "current_visual_tokens": DEFAULT_CURRENT_VISUAL_TOKENS,
            "long_memory_decay": LONG_MEMORY_DECAY,
            "long_memory_update_weight": LONG_MEMORY_UPDATE_WEIGHT,
            "action_placeholder_count": None,
            "history_augmentation": {
                "enabled": False,
                "shuffle": {"target_probability": 0.3, "warmup_end_ratio": 0.05},
                "tvi_mask": {
                    "target_probability": 0.1,
                    "warmup_start_ratio": 0.05,
                    "warmup_end_ratio": 0.15,
                },
            },
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 2048,
            "action_dim": 4,
            "state_dim": 0,
            "action_horizon": 8,
            "repeated_diffusion_steps": 2,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 8,
            "diffusion_model_cfg": {
                "cross_attention_dim": 1024,
                "dropout": 0.1,
                "final_dropout": False,
                "interleave_self_attention": False,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("navvla_cpm")
class NavVLA_CPM(baseframework):
    def __init__(self, config: Optional[dict] = None, **_kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(NavVLACPMDefaultConfig, config)
        nav_cfg = self.config.framework.navvla
        self.tvi_mode = str(nav_cfg.get("tvi_mode", TIME_YAW_TVI_MODE))
        self.tvi_dim = get_tvi_input_dim(self.tvi_mode)
        self.minicpm_vl_interface = get_vlm_model(config=self.config)
        hidden_size = int(self.minicpm_vl_interface.model.config.hidden_size)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = hidden_size
        self.config.framework.action_model.num_target_vision_tokens = int(self.config.framework.action_model.action_horizon)
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.tvi_embedding = NavVLATVIEmbedding(
            hidden_size=hidden_size,
            mode=self.tvi_mode,
            enable_mask_token=True,
        )
        augmentation_cfg = nav_cfg.get("history_augmentation", {}) or {}
        shuffle_cfg = augmentation_cfg.get("shuffle", {}) or {}
        mask_cfg = augmentation_cfg.get("tvi_mask", {}) or {}
        self.history_augmentation = HistoryAugmentationConfig(
            enabled=bool(augmentation_cfg.get("enabled", False)),
            shuffle_target_probability=float(shuffle_cfg.get("target_probability", 0.3)),
            shuffle_warmup_end_ratio=float(shuffle_cfg.get("warmup_end_ratio", 0.05)),
            tvi_mask_target_probability=float(mask_cfg.get("target_probability", 0.1)),
            tvi_mask_warmup_start_ratio=float(mask_cfg.get("warmup_start_ratio", 0.05)),
            tvi_mask_warmup_end_ratio=float(mask_cfg.get("warmup_end_ratio", 0.15)),
        )
        long_memory_visual_tokens = int(nav_cfg.get("long_memory_visual_tokens", DEFAULT_LONG_MEMORY_VISUAL_TOKENS))
        self.long_memory_aggregator = (
            LongMemoryTokenAggregator(
                source_visual_tokens=int(
                    nav_cfg.get("long_memory_source_visual_tokens", DEFAULT_LONG_MEMORY_SOURCE_VISUAL_TOKENS)
                ),
                long_memory_visual_tokens=long_memory_visual_tokens,
                decay=float(nav_cfg.get("long_memory_decay", LONG_MEMORY_DECAY)),
                update_weight=float(nav_cfg.get("long_memory_update_weight", LONG_MEMORY_UPDATE_WEIGHT)),
                tvi_dim=self.tvi_dim,
            )
            if long_memory_visual_tokens > 0
            else None
        )
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.action_dim = int(self.config.framework.action_model.action_dim)
        raw_action_placeholder_count = nav_cfg.get("action_placeholder_count", None)
        self.action_placeholder_count = (
            self.action_dim * self.action_horizon if raw_action_placeholder_count is None else int(raw_action_placeholder_count)
        )
        if self.action_placeholder_count <= 0:
            raise ValueError("framework.navvla.action_placeholder_count must be a positive integer")
        self.hidden_size = hidden_size

    def _visual_token_budgets(self) -> tuple[int, int, int]:
        nav_cfg = self.config.framework.navvla
        return (
            int(nav_cfg.get("history_visual_tokens", DEFAULT_HISTORY_VISUAL_TOKENS)),
            int(nav_cfg.get("long_memory_visual_tokens", DEFAULT_LONG_MEMORY_VISUAL_TOKENS)),
            int(nav_cfg.get("current_visual_tokens", DEFAULT_CURRENT_VISUAL_TOKENS)),
        )

    def forward_pooled_vlm(self, batch: dict[str, Any]) -> Any:
        model = self.minicpm_vl_interface.model
        device, dtype = _model_device_dtype(model)
        minicpm_inputs = dict(_move_to_device(batch, device))
        image_token_id = int(self.minicpm_vl_interface.IMAGE_TOKEN_INDEX)
        _history_tokens, _long_memory_tokens, current_visual_tokens = self._visual_token_budgets()

        feature_getter = getattr(model, "get_image_features", None) or getattr(
            getattr(model, "model", None), "get_image_features", None
        )
        if feature_getter is None:
            raise RuntimeError("MiniCPM checkpoint does not expose get_image_features()")

        target_sizes = minicpm_inputs.get("tgt_sizes", minicpm_inputs.get("target_sizes"))
        if target_sizes is None:
            raise ValueError("MiniCPM VLM pooling requires target_sizes to split per-image visual features")
        if "pixel_values" not in minicpm_inputs:
            raise ValueError("MiniCPM VLM pooling requires pixel_values")
        downsample_mode = str(getattr(self.minicpm_vl_interface, "downsample_mode", "4x"))
        flat_features = _flatten_minicpm_image_features(
            feature_getter(
                pixel_values=minicpm_inputs["pixel_values"],
                target_sizes=target_sizes,
                downsample_mode=downsample_mode,
            )
        )
        feature_target_sizes = _minicpm_feature_target_sizes(target_sizes, downsample_mode=downsample_mode)
        image_features = _split_flat_minicpm_features_by_target_sizes(
            flat_features,
            target_sizes,
            downsample_mode=downsample_mode,
        )

        pooled_inputs, target_counts = pool_minicpm_vlm_inputs(
            minicpm_inputs,
            image_token_id=image_token_id,
            max_visual_tokens=current_visual_tokens,
            original_visual_token_counts=[int(features.shape[0]) for features in image_features],
        )
        if len(image_features) != len(target_counts):
            raise ValueError(
                f"MiniCPM VLM image feature count {len(image_features)} does not match image span count {len(target_counts)}"
            )

        pooled_features: list[torch.Tensor] = []
        for image_index, (features, target_count) in enumerate(zip(image_features, target_counts)):
            tgt_size = None
            if feature_target_sizes is not None:
                tgt_size = feature_target_sizes[image_index]
            pooled_features.append(
                pool_minicpm_visual_tokens_to_count(
                    features.to(device=device, dtype=dtype),
                    target_tokens=target_count,
                    tgt_size=tgt_size,
                )
            )

        input_ids = pooled_inputs["input_ids"]
        inputs_embeds = model.get_input_embeddings()(input_ids)
        fused_features = torch.cat(pooled_features, dim=0) if pooled_features else inputs_embeds.new_zeros((0, inputs_embeds.shape[-1]))
        inputs_embeds = self._scatter_image_embeddings(inputs_embeds, input_ids, fused_features, image_token_id)
        model_kwargs: dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": pooled_inputs.get("attention_mask"),
            "labels": pooled_inputs.get("labels"),
            "return_dict": True,
            "downsample_mode": str(getattr(self.minicpm_vl_interface, "downsample_mode", "4x")),
        }
        return model(**model_kwargs)

    def forward_vlm(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        output = self.forward_pooled_vlm(batch)
        return {"vlm_loss": output.loss}

    def _samples_from_batch(self, examples: list[dict[str, Any]] | dict[str, Any] | None) -> list[dict[str, Any]]:
        samples = samples_from_collated_batch(examples)
        if samples and "minicpm_inputs" in examples:
            samples[0]["_collated_minicpm_inputs"] = examples["minicpm_inputs"]
        return samples

    def _attach_long_memory_tokens(self, samples: list[dict[str, Any]]) -> None:
        _history_visual_tokens, long_memory_visual_tokens, _current_visual_tokens = self._visual_token_budgets()
        device, dtype = _model_device_dtype(self.minicpm_vl_interface.model)
        for sample in samples:
            if sample.get("long_memory_tokens") is not None:
                continue
            source_tokens = sample.get("long_memory_source_tokens")
            if source_tokens is None:
                continue
            source_tokens_tensor = torch.as_tensor(source_tokens, device=device, dtype=dtype)
            if long_memory_visual_tokens <= 0 or self.long_memory_aggregator is None:
                raise ValueError("long_memory source tokens require framework.navvla.long_memory_visual_tokens > 0")

            metadata = dict(sample.get("metadata", {}) or {})
            source_blocks = list(metadata.get("long_memory_blocks") or [])
            required_cameras = self._sample_required_cameras(sample)
            source_tvi = torch.as_tensor(
                sample.get(
                    "long_memory_source_tvi",
                    np.zeros((int(source_tokens_tensor.shape[0]), self.tvi_dim), dtype=np.float32),
                ),
                device=device,
                dtype=dtype,
            )
            if source_tvi.ndim != 2 or int(source_tvi.shape[1]) != self.tvi_dim:
                raise ValueError(
                    f"long_memory_source_tvi must have shape [N, {self.tvi_dim}], "
                    f"got {tuple(source_tvi.shape)}"
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
                    f"long_memory metadata has {source_block_count} blocks but only "
                    f"{source_slot_count} source token slots"
                )
            if source_block_count < source_slot_count:
                source_tokens_tensor = source_tokens_tensor[:source_block_count]
                source_tvi = source_tvi[:source_block_count]
                source_mask = source_mask.reshape(-1)[:source_block_count]
            source_count = int(source_tokens_tensor.shape[0])
            missing_long_memory = source_count == 0 or not bool(source_mask[:source_count].any().item())
            if missing_long_memory:
                camera_metadata = metadata.get("camera", {}) or {}
                if self.tvi_dim == 2:
                    dummy_tvi = [
                        [
                            0.0,
                            float((camera_metadata.get(camera_name, {}) or {}).get("azimuth_rad", 0.0)),
                        ]
                        for camera_name in required_cameras
                    ]
                else:
                    dummy_tvi = np.zeros((len(required_cameras), self.tvi_dim), dtype=np.float32)
                source_tokens_tensor = source_tokens_tensor.new_zeros(
                    (
                        len(required_cameras),
                        self.long_memory_aggregator.source_visual_tokens,
                        self.hidden_size,
                    )
                )
                source_tvi = torch.as_tensor(dummy_tvi, device=device, dtype=dtype)
                source_mask = torch.ones((len(required_cameras),), device=device, dtype=torch.bool)
                source_blocks = [
                    {"step_index": 0, "camera_name": str(camera_name), "missing_long_memory": True}
                    for camera_name in required_cameras
                ]
            tokens, tvi, blocks = self.long_memory_aggregator.aggregate_sample(
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

    def _compute_online_long_memory_updates(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if getattr(self, "long_memory_aggregator", None) is None:
            return []
        device, dtype = _model_device_dtype(self.minicpm_vl_interface.model)
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
                    np.zeros((int(source_tokens_tensor.shape[0]), self.tvi_dim), dtype=np.float32),
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
            tokens, tvi, blocks = self.long_memory_aggregator.update_state(
                previous_tokens=previous_tokens,
                previous_tvi=previous_tvi,
                previous_blocks=list(metadata.get("long_memory_blocks") or []),
                source_tokens=source_tokens_tensor,
                source_tvi=source_tvi,
                source_mask=source_mask,
                source_blocks=list(metadata.get("online_long_memory_update_blocks") or []),
                required_cameras=self._sample_required_cameras(sample),
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

    def _build_instruction(self, sample: dict[str, Any]) -> str:
        return build_navvla_instruction(self, sample)

    def _sample_required_cameras(self, sample: dict[str, Any]) -> list[str]:
        return sample_required_cameras(sample)

    def _build_minicpm_inputs(
        self,
        samples: list[dict[str, Any]],
        *,
        history_shuffle_probability: float = 0.0,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        history_visual_tokens, long_memory_visual_tokens, current_visual_tokens = self._visual_token_budgets()
        batch_images: list[list[Any]] = []
        instructions: list[str] = []
        action_suffixes: list[str] = []
        cached_spans_per_row: list[int] = []
        all_blocks: list[dict[str, Any]] = []
        action_suffix = (
            self.minicpm_vl_interface.build_action_placeholder_suffix(self.action_placeholder_count)
            if hasattr(self.minicpm_vl_interface, "build_action_placeholder_suffix")
            else ""
        )
        for sample in samples:
            required_cameras = self._sample_required_cameras(sample)
            online_images, blocks = build_navvla_cached_visual_sequence(
                sample,
                required_cameras=required_cameras,
                history_shuffle_probability=history_shuffle_probability,
                tvi_dim=self.tvi_dim,
            )
            batch_images.append([to_pil_preserve(image) for image in online_images])
            cached_spans_per_row.append(sum(1 for block in blocks if bool(block.get("is_cached_history", False))))
            all_blocks.extend(blocks)
            instructions.append(self._build_instruction(sample))
            action_suffixes.append(action_suffix)

        model = self.minicpm_vl_interface.model
        minicpm_inputs = samples[0].get("_collated_minicpm_inputs") if samples else None
        if minicpm_inputs is not None:
            device, _dtype = _model_device_dtype(model)
            minicpm_inputs = dict(_move_to_device(minicpm_inputs, device))
        else:
            minicpm_inputs = self.minicpm_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                action_suffixes=action_suffixes,
            )

        minicpm_inputs["_nav_online_input_ids"] = minicpm_inputs["input_ids"].clone()
        target_sizes = minicpm_inputs.get("target_sizes")
        if target_sizes is not None and isinstance(target_sizes, torch.Tensor):
            minicpm_inputs["_nav_online_target_sizes"] = target_sizes.clone()
        elif target_sizes is not None:
            minicpm_inputs["_nav_online_target_sizes"] = target_sizes

        if any(int(count) > 0 for count in cached_spans_per_row):
            vision_prefix_token_id = _resolve_vision_prefix_token_id(
                minicpm_inputs["input_ids"],
                self.minicpm_vl_interface.IMAGE_TOKEN_INDEX,
                self.minicpm_vl_interface,
            )
            image_slot_suffix_token_ids = _minicpm_image_slot_suffix_token_ids(
                minicpm_inputs["_nav_online_input_ids"],
                image_token_id=self.minicpm_vl_interface.IMAGE_TOKEN_INDEX,
                vlm_interface=self.minicpm_vl_interface,
            )
            minicpm_inputs = prepend_navvla_cached_minicpm_spans(
                minicpm_inputs,
                cached_spans_per_row=cached_spans_per_row,
                image_token_id=self.minicpm_vl_interface.IMAGE_TOKEN_INDEX,
                vision_prefix_token_id=vision_prefix_token_id,
                image_slot_suffix_token_ids=image_slot_suffix_token_ids,
            )

        minicpm_inputs = pool_navvla_minicpm_inputs(
            minicpm_inputs,
            all_blocks,
            self.minicpm_vl_interface.IMAGE_TOKEN_INDEX,
            history_visual_tokens=history_visual_tokens,
            long_memory_visual_tokens=long_memory_visual_tokens,
            current_visual_tokens=current_visual_tokens,
        )
        return minicpm_inputs, all_blocks

    def _encode_minicpm_current_image_features(
        self,
        minicpm_inputs: dict[str, Any],
        *,
        online_block_count: int,
    ) -> list[torch.Tensor]:
        if online_block_count <= 0:
            return []
        model = self.minicpm_vl_interface.model
        getter = getattr(model, "get_image_features", None) or getattr(getattr(model, "model", None), "get_image_features", None)
        if getter is None:
            raise RuntimeError("MiniCPM checkpoint does not expose get_image_features()")
        device, _dtype = _model_device_dtype(model)
        feature_kwargs: dict[str, Any] = {"downsample_mode": str(getattr(self.minicpm_vl_interface, "downsample_mode", "4x"))}
        if "pixel_values" in minicpm_inputs:
            feature_kwargs["pixel_values"] = _move_to_device(minicpm_inputs["pixel_values"], device)
        for optional_key in ("image_grid_thw", "tgt_sizes", "target_sizes", "image_bound"):
            if optional_key in minicpm_inputs:
                feature_kwargs[optional_key] = _move_to_device(minicpm_inputs[optional_key], device)
        try:
            flat_features = _flatten_minicpm_image_features(getter(**feature_kwargs))
        except TypeError as exc:
            if "pixel_values" not in minicpm_inputs:
                raise RuntimeError("MiniCPM inputs are missing pixel_values for online current image encoding") from exc
            raise
        target_sizes = minicpm_inputs.get("_nav_online_target_sizes", minicpm_inputs.get("tgt_sizes", minicpm_inputs.get("target_sizes")))
        return _split_flat_minicpm_features_by_target_sizes(
            flat_features,
            target_sizes,
            downsample_mode=str(getattr(self.minicpm_vl_interface, "downsample_mode", "4x")),
        )

    @torch.inference_mode()
    def encode_history_images(self, images: list[Any]) -> list[np.ndarray]:
        images = list(images)
        if not images:
            return []
        pil_images = [to_pil_preserve(image) for image in images]
        minicpm_inputs = self.minicpm_vl_interface.build_qwenvl_inputs(
            images=[pil_images],
            instructions=[""],
            action_suffixes=[""],
        )
        target_sizes = minicpm_inputs.get("target_sizes")
        if target_sizes is not None and isinstance(target_sizes, torch.Tensor):
            minicpm_inputs["_nav_online_target_sizes"] = target_sizes.clone()
        elif target_sizes is not None:
            minicpm_inputs["_nav_online_target_sizes"] = target_sizes
        image_features = self._encode_minicpm_current_image_features(
            minicpm_inputs,
            online_block_count=len(pil_images),
        )
        if len(image_features) != len(pil_images):
            raise ValueError(
                f"MiniCPM encoded {len(image_features)} history images for {len(pil_images)} inputs"
            )

        history_visual_tokens, _long_memory_visual_tokens, _current_visual_tokens = self._visual_token_budgets()
        feature_target_sizes = minicpm_inputs.get(
            "_nav_online_target_sizes",
            minicpm_inputs.get("tgt_sizes", minicpm_inputs.get("target_sizes")),
        )
        cached: list[np.ndarray] = []
        for image_index, features in enumerate(image_features):
            tgt_size = None if feature_target_sizes is None else feature_target_sizes[image_index]
            pooled = pool_minicpm_visual_tokens_to_count(
                features,
                target_tokens=history_visual_tokens,
                tgt_size=tgt_size,
            )
            cached.append(pooled.detach().to(torch.float16).cpu().numpy())
        return cached

    def _fuse_cached_tokens(
        self,
        tokens: torch.Tensor,
        *,
        target_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tokens = tokens.to(device=device, dtype=dtype)
        if int(tokens.shape[-1]) != self.hidden_size:
            raise ValueError(f"cached visual hidden dim {tokens.shape[-1]} does not match VLM hidden dim {self.hidden_size}")
        if int(tokens.shape[0]) != int(target_tokens):
            raise ValueError(f"cached visual token count {tokens.shape[0]} does not match expected {target_tokens}")
        return tokens

    def _fuse_image_token_embeddings(
        self,
        minicpm_inputs: dict[str, Any],
        blocks: list[dict[str, Any]],
        *,
        capture_online_current_cache: bool = False,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        model = self.minicpm_vl_interface.model
        device, dtype = _model_device_dtype(model)
        input_ids = minicpm_inputs["input_ids"]
        image_token_id = self.minicpm_vl_interface.IMAGE_TOKEN_INDEX
        history_visual_tokens, long_memory_visual_tokens, current_visual_tokens = self._visual_token_budgets()
        online_indices = sorted(index for index, block in enumerate(blocks) if not bool(block.get("is_cached_history", False)))
        online_features = self._encode_minicpm_current_image_features(
            minicpm_inputs,
            online_block_count=len(online_indices),
        )
        online_target_sizes = minicpm_inputs.get("_nav_online_target_sizes", minicpm_inputs.get("tgt_sizes", minicpm_inputs.get("target_sizes")))
        if online_indices and len(online_features) != len(online_indices):
            raise ValueError(
                f"MiniCPM encoded current feature count {len(online_features)} does not match online image block count "
                f"{len(online_indices)}"
            )

        expected_image_tokens = int((input_ids == int(image_token_id)).sum().item())
        expected_budget = sum(
            _target_visual_tokens_for_block(
                block,
                history_visual_tokens=history_visual_tokens,
                long_memory_visual_tokens=long_memory_visual_tokens,
                current_visual_tokens=current_visual_tokens,
            )
            for block in blocks
        )
        if expected_image_tokens != expected_budget:
            raise ValueError(f"MiniCPM placeholder count {expected_image_tokens} does not match visual token budget {expected_budget}")

        online_features_by_index: dict[int, torch.Tensor] = {}
        online_current_cache_records: list[dict[str, Any]] = []
        for local_index, block_index in enumerate(online_indices):
            block = blocks[block_index]
            target_tokens = _target_visual_tokens_for_block(
                block,
                history_visual_tokens=history_visual_tokens,
                long_memory_visual_tokens=long_memory_visual_tokens,
                current_visual_tokens=current_visual_tokens,
            )
            tgt_size = None if online_target_sizes is None else online_target_sizes[local_index]
            pooled = pool_minicpm_visual_tokens_to_count(
                online_features[local_index].to(device=device, dtype=dtype),
                target_tokens=target_tokens,
                tgt_size=tgt_size,
            )
            online_features_by_index[block_index] = pooled.to(device=device, dtype=dtype)
            if capture_online_current_cache and not bool(block.get("is_history", False)):
                history_pooled = pool_minicpm_visual_tokens_to_count(
                    online_features[local_index].to(device=device, dtype=dtype),
                    target_tokens=history_visual_tokens,
                    tgt_size=tgt_size,
                )
                online_current_cache_records.append(
                    {
                        "camera_name": str(block["camera_name"]),
                        "frame_index": int(block.get("frame_index", 0)),
                        "tokens": history_pooled.detach().to(torch.float16).cpu().numpy(),
                    }
                )

        fused_chunks: list[torch.Tensor] = []
        for block_index, block in enumerate(blocks):
            target_tokens = _target_visual_tokens_for_block(
                block,
                history_visual_tokens=history_visual_tokens,
                long_memory_visual_tokens=long_memory_visual_tokens,
                current_visual_tokens=current_visual_tokens,
            )
            if block_index in online_features_by_index:
                fused_chunks.append(online_features_by_index[block_index])
                continue
            cached = _cached_tokens_for_block(block, device=device, dtype=dtype)
            fused_chunks.append(self._fuse_cached_tokens(cached, target_tokens=target_tokens, device=device, dtype=dtype))

        fused = torch.cat(fused_chunks, dim=0) if fused_chunks else torch.zeros((0, self.hidden_size), device=device, dtype=dtype)
        if int(fused.shape[0]) != expected_image_tokens:
            raise ValueError(f"fused visual token count {fused.shape[0]} does not match placeholders {expected_image_tokens}")
        return fused, online_current_cache_records

    @staticmethod
    def _scatter_image_embeddings(
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        image_embeddings: torch.Tensor,
        image_token_id: int,
    ) -> torch.Tensor:
        return scatter_image_embeddings(inputs_embeds, input_ids, image_embeddings, image_token_id)

    def _forward_backbone(
        self,
        minicpm_inputs: dict[str, Any],
        blocks: list[dict[str, Any]],
        *,
        capture_online_current_cache: bool = False,
        tvi_mask_probability: float = 0.0,
    ) -> tuple[Any, list[dict[str, Any]]]:
        model = self.minicpm_vl_interface.model
        input_ids = minicpm_inputs["input_ids"]
        attention_mask = minicpm_inputs.get("attention_mask")
        image_token_id = self.minicpm_vl_interface.IMAGE_TOKEN_INDEX
        inputs_embeds = model.get_input_embeddings()(input_ids)
        fused_image_embeds, online_current_cache_records = self._fuse_image_token_embeddings(
            minicpm_inputs,
            blocks,
            capture_online_current_cache=capture_online_current_cache,
        )
        inputs_embeds = self._scatter_image_embeddings(inputs_embeds, input_ids, fused_image_embeds, image_token_id)

        if blocks:
            for block_index, block in enumerate(blocks):
                if "tvi" not in block:
                    raise ValueError(f"visual block {block_index} is missing its TVI row")
            try:
                stacked_tvi = np.stack([block["tvi"] for block in blocks], axis=0)
                tvi_array = _as_numpy_tvi(stacked_tvi, tvi_dim=self.tvi_dim)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"visual block TVI rows must stack to shape [N, {self.tvi_dim}]"
                ) from exc
            if tuple(tvi_array.shape) != (len(blocks), self.tvi_dim):
                raise ValueError(
                    f"visual block TVI rows must have shape [{len(blocks)}, {self.tvi_dim}], "
                    f"got {tuple(tvi_array.shape)}"
                )
            tvi_values = torch.as_tensor(
                tvi_array,
                device=inputs_embeds.device,
                dtype=torch.float32,
            )
            tvi_embeds = self.tvi_embedding(tvi_values)
            tvi_embeds = self._mask_history_tvi_embeddings(
                tvi_embeds,
                blocks,
                probability=tvi_mask_probability,
            )
            vision_prefix_token_id = _resolve_vision_prefix_token_id(input_ids, image_token_id, self.minicpm_vl_interface)
            input_ids, inputs_embeds, attention_mask = insert_navvla_tvi_prefix_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                blocks=blocks,
                tvi_embeds=tvi_embeds,
                vision_prefix_token_id=vision_prefix_token_id,
                image_token_id=image_token_id,
                tvi_token_id=_infer_pad_token_id(input_ids, attention_mask),
            )
            minicpm_inputs["input_ids"] = input_ids
            if attention_mask is not None:
                minicpm_inputs["attention_mask"] = attention_mask
            for stale_key in ("position_ids", "cache_position", "past_key_values"):
                minicpm_inputs.pop(stale_key, None)

        model_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "return_dict": True,
            "downsample_mode": str(getattr(self.minicpm_vl_interface, "downsample_mode", "4x")),
        }
        for optional_key in ("position_ids", "cache_position", "past_key_values"):
            if optional_key in minicpm_inputs:
                model_kwargs[optional_key] = minicpm_inputs[optional_key]
        inner_model = getattr(model, "model", model)
        return inner_model(**model_kwargs), online_current_cache_records

    def _forward_vlm_for_action(
        self,
        samples: list[dict[str, Any]],
        *,
        capture_online_current_cache: bool = False,
        history_shuffle_probability: float = 0.0,
        tvi_mask_probability: float = 0.0,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        self._attach_long_memory_tokens(samples)
        minicpm_inputs, blocks = self._build_minicpm_inputs(
            samples,
            history_shuffle_probability=history_shuffle_probability,
        )
        outputs, online_current_cache_records = self._forward_backbone(
            minicpm_inputs,
            blocks,
            capture_online_current_cache=capture_online_current_cache,
            tvi_mask_probability=tvi_mask_probability,
        )
        last_hidden = outputs.last_hidden_state
        if hasattr(self.minicpm_vl_interface, "gather_action_placeholder_hidden_states"):
            action_hidden = self.minicpm_vl_interface.gather_action_placeholder_hidden_states(
                last_hidden,
                minicpm_inputs["input_ids"],
                num_placeholders=self.action_placeholder_count,
            )
        else:
            action_hidden = last_hidden
        if self.long_memory_aggregator is not None and any(
            bool(sample.pop("_long_memory_zero_dependency", False)) for sample in samples
        ):
            action_hidden = action_hidden + self.long_memory_aggregator.zero_dependency(action_hidden)
        return action_hidden, online_current_cache_records

    def _action_model_accepts_padding_mask(self) -> bool:
        return action_model_accepts_padding_mask(self.action_model)

    def _mask_history_tvi_embeddings(
        self,
        embeddings: torch.Tensor,
        blocks: list[dict[str, Any]],
        *,
        probability: float,
    ) -> torch.Tensor:
        return mask_history_tvi_embeddings(self, embeddings, blocks, probability=probability)

    def forward(
        self,
        examples: list[dict[str, Any]] | dict[str, Any] | None = None,
        *,
        training_step: int | None = None,
        total_training_steps: int | None = None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        samples = self._samples_from_batch(examples)
        if self.training:
            shuffle_probability, mask_probability = history_augmentation_probabilities(
                self.history_augmentation,
                training_step=training_step,
                total_training_steps=total_training_steps,
            )
        else:
            shuffle_probability, mask_probability = 0.0, 0.0
        vl_embs, _online_current_cache_records = self._forward_vlm_for_action(
            samples,
            history_shuffle_probability=shuffle_probability,
            tvi_mask_probability=mask_probability,
        )
        actions = torch.as_tensor(
            np.asarray([sample["action"] for sample in samples], dtype=np.float32),
            device=vl_embs.device,
            dtype=vl_embs.dtype,
        )
        if int(actions.shape[-1]) != self.action_dim:
            raise ValueError(f"NavVLA_CPM action dim mismatch: batch={int(actions.shape[-1])}, model={self.action_dim}")
        action_padding_mask = torch.as_tensor(
            np.asarray([sample["action_padding_mask"] for sample in samples], dtype=bool),
            device=vl_embs.device,
            dtype=torch.bool,
        )
        actions_target = actions[:, -self.action_horizon :, :].clone()
        action_padding_mask = action_padding_mask[:, -self.action_horizon :]
        actions_target = actions_target.masked_fill(action_padding_mask.unsqueeze(-1), 0.0)
        path_progress_rows = torch.as_tensor(
            [
                str(sample.get("metadata", {}).get("action_extra_dim_mode", "none")) == "path_progress"
                for sample in samples
            ],
            device=vl_embs.device,
            dtype=torch.bool,
        )
        if self.action_dim > 4 and bool(path_progress_rows.any().item()):
            progress_padding_mask = action_padding_mask & path_progress_rows.unsqueeze(-1)
            actions_target[..., 4] = torch.where(
                progress_padding_mask,
                torch.ones_like(actions_target[..., 4]),
                actions_target[..., 4],
            )
        repeated_diffusion_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 2))
        actions_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
        vl_embs_repeated = vl_embs.repeat(repeated_diffusion_steps, 1, 1)
        mask_repeated = action_padding_mask.repeat(repeated_diffusion_steps, 1)
        state = None
        if any("state" in sample for sample in samples):
            state = torch.as_tensor(
                np.asarray([sample.get("state", np.zeros((0,), dtype=np.float32)) for sample in samples], dtype=np.float32),
                device=vl_embs.device,
                dtype=vl_embs.dtype,
            )
            state = state.repeat(repeated_diffusion_steps, 1)
        if self._action_model_accepts_padding_mask():
            action_loss = self.action_model(vl_embs_repeated, actions_repeated, state, action_padding_mask=mask_repeated)
        else:
            action_loss = self.action_model(vl_embs_repeated, actions_repeated, state)
        return {"action_loss": action_loss, "loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: list[dict[str, Any]] | dict[str, Any] | None = None, **_kwargs) -> dict[str, np.ndarray]:
        samples = self._samples_from_batch(examples)
        tvi_mask_probability = float(_kwargs.get("tvi_mask_probability", 0.0))
        if not 0.0 <= tvi_mask_probability <= 1.0:
            raise ValueError("tvi_mask_probability must be between 0 and 1")
        vl_embs, online_current_cache_records = self._forward_vlm_for_action(
            samples,
            capture_online_current_cache=True,
            history_shuffle_probability=0.0,
            tvi_mask_probability=tvi_mask_probability,
        )
        try:
            action_param = next(self.action_model.parameters())
            action_device = action_param.device
            action_dtype = action_param.dtype
        except StopIteration:
            action_device = vl_embs.device
            action_dtype = vl_embs.dtype
        vl_embs = vl_embs.to(device=action_device, dtype=action_dtype)
        state = None
        if any("state" in sample for sample in samples):
            state = torch.as_tensor(
                np.asarray([sample.get("state", np.zeros((0,), dtype=np.float32)) for sample in samples], dtype=np.float32),
                device=action_device,
                dtype=action_dtype,
            )
        pred_actions = self.action_model.predict_action(vl_embs, state)
        online_long_memory_updates = self._compute_online_long_memory_updates(samples)
        return {
            "normalized_actions": pred_actions.detach().to(torch.float32).cpu().numpy(),
            "metadata": {
                "online_current_visual_tokens": online_current_cache_records,
                "online_long_memory_updates": online_long_memory_updates,
            },
        }
