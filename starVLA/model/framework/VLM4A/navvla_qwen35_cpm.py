"""NavVLA CPM outer architecture with a Qwen3.5-VL backbone."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.long_memory import LongMemoryTokenAggregator
from starVLA.model.modules.navvla_context import (
    HistoryAugmentationConfig,
    as_numpy_tvi,
    build_navvla_cached_visual_sequence,
    build_navvla_instruction,
    forward_navvla_action,
    mask_history_tvi_embeddings,
    predict_navvla_action,
    sample_required_cameras,
    samples_from_collated_batch,
    scatter_image_embeddings,
    target_visual_tokens_for_block,
)
from starVLA.model.modules.navvla_long_memory import (
    attach_navvla_long_memory_tokens,
    compute_navvla_online_long_memory_updates,
)
from starVLA.model.modules.qwen35_vision import (
    BFLOAT16_BITS_STORAGE_ENCODING,
    bf16_to_numpy_bits,
    configure_qwen35_processor,
    decode_qwen35_cache_tokens,
    encode_qwen35_postmerge_batched,
    encode_qwen35_postmerge_one_by_one,
    pool_qwen35_postmerge,
    qwen35_postmerge_token_count,
)
from starVLA.model.modules.tvi import TIME_YAW_TVI_MODE, NavVLATVIEmbedding, get_tvi_input_dim
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from tool.navvla.visual_token_cache import (
    DEFAULT_QWEN35_POOLED_HISTORY_VISUAL_TOKEN_PROFILE,
    QWEN35_POOLED_HISTORY_CACHE_STAGE,
)

QWEN35_LONG_MEMORY_SOURCE_POOLED_STAGE = "navvla_long_memory_source_pooled"


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


def _model_device_dtype(model: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters())
    return parameter.device, parameter.dtype


def _active_row(values: torch.Tensor, attention_mask: torch.Tensor | None, row_index: int) -> torch.Tensor:
    if attention_mask is None:
        return values[row_index]
    return values[row_index][attention_mask[row_index].to(dtype=torch.bool)]


def _left_pad_rows(
    rows: list[torch.Tensor],
    *,
    pad_value: int,
    template: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(row.shape[0]) for row in rows)
    padded = template.new_full((len(rows), max_len), int(pad_value))
    mask = template.new_zeros((len(rows), max_len))
    for row_index, row in enumerate(rows):
        padded[row_index, -int(row.shape[0]) :] = row
        mask[row_index, -int(row.shape[0]) :] = 1
    return padded, mask


def _rewrite_qwen35_visual_spans(
    qwen_inputs: dict[str, Any],
    blocks: list[dict[str, Any]],
    *,
    image_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    history_visual_tokens: int,
    long_memory_visual_tokens: int,
    current_visual_tokens: int,
    merge_size: int,
    pad_token_id: int,
) -> dict[str, Any]:
    output = dict(qwen_inputs)
    input_ids = output["input_ids"]
    attention_mask = output.get("attention_mask")
    token_types = output.get("mm_token_type_ids")
    raw_grid = output.get("image_grid_thw")
    if token_types is None or raw_grid is None:
        raise ValueError("Qwen3.5 processor inputs require mm_token_type_ids and image_grid_thw")

    rows: list[torch.Tensor] = []
    type_rows: list[torch.Tensor] = []
    tvi_mask_rows: list[torch.Tensor] = []
    context_grids: list[tuple[int, int, int]] = []
    image_cursor = 0
    for row_index in range(int(input_ids.shape[0])):
        row_ids = _active_row(input_ids, attention_mask, row_index)
        row_types = _active_row(token_types, attention_mask, row_index)
        spans = _image_token_spans(row_ids, image_token_id)
        id_chunks: list[torch.Tensor] = []
        type_chunks: list[torch.Tensor] = []
        tvi_mask_chunks: list[torch.Tensor] = []
        cursor = 0
        for start, end in spans:
            if image_cursor >= len(blocks):
                raise ValueError("Qwen3.5 processor produced more image spans than NavVLA visual blocks")
            vision_start = start - 1
            if vision_start < cursor or int(row_ids[vision_start].item()) != int(vision_start_token_id):
                raise ValueError("Qwen3.5 image span must start immediately after vision_start")
            if end >= int(row_ids.shape[0]) or int(row_ids[end].item()) != int(vision_end_token_id):
                raise ValueError("Qwen3.5 image span must end immediately before vision_end")
            block = blocks[image_cursor]
            target = target_visual_tokens_for_block(
                block,
                history_visual_tokens=history_visual_tokens,
                long_memory_visual_tokens=long_memory_visual_tokens,
                current_visual_tokens=current_visual_tokens,
            )
            context_grid = _grid_shape_for_token_count(raw_grid[image_cursor], target, merge_size=merge_size)
            if qwen35_postmerge_token_count(context_grid, spatial_merge_size=merge_size) != target:
                raise ValueError(f"cannot represent target visual token count {target} as a Qwen3.5 M-RoPE grid")
            prefix_ids = row_ids[cursor:vision_start]
            prefix_types = row_types[cursor:vision_start]
            id_chunks.extend((prefix_ids, row_ids.new_full((1,), pad_token_id)))
            type_chunks.extend((prefix_types, row_types.new_zeros((1,))))
            tvi_mask_chunks.extend(
                (
                    row_types.new_zeros((prefix_ids.shape[0],)),
                    row_types.new_ones((1,)),
                )
            )
            if not bool(block.get("is_cached_history", False)):
                id_chunks.append(row_ids[vision_start:start])
                type_chunks.append(row_types[vision_start:start])
                tvi_mask_chunks.append(row_types.new_zeros((start - vision_start,)))
            id_chunks.append(row_ids.new_full((target,), image_token_id))
            type_chunks.append(row_types.new_full((target,), 1))
            tvi_mask_chunks.append(row_types.new_zeros((target,)))
            context_grids.append(context_grid)
            cursor = end + 1 if bool(block.get("is_cached_history", False)) else end
            image_cursor += 1
        tail_ids = row_ids[cursor:]
        id_chunks.append(tail_ids)
        type_chunks.append(row_types[cursor:])
        tvi_mask_chunks.append(row_types.new_zeros((tail_ids.shape[0],)))
        rows.append(torch.cat(id_chunks))
        type_rows.append(torch.cat(type_chunks))
        tvi_mask_rows.append(torch.cat(tvi_mask_chunks))
    if image_cursor != len(blocks) or image_cursor != int(raw_grid.shape[0]):
        raise ValueError(
            "Qwen3.5 image alignment mismatch: "
            f"spans={image_cursor}, blocks={len(blocks)}, grids={int(raw_grid.shape[0])}"
        )

    inactive = input_ids[~attention_mask.to(dtype=torch.bool)] if attention_mask is not None else input_ids[:0]
    if inactive.numel():
        pad_token_id = int(inactive[0].item())
    output["input_ids"], output["attention_mask"] = _left_pad_rows(
        rows, pad_value=pad_token_id, template=input_ids
    )
    output["mm_token_type_ids"], _ = _left_pad_rows(type_rows, pad_value=0, template=token_types)
    padded_tvi_mask, _ = _left_pad_rows(tvi_mask_rows, pad_value=0, template=token_types)
    output["_nav_tvi_mask"] = padded_tvi_mask.to(dtype=torch.bool)
    output["_nav_context_image_grid_thw"] = raw_grid.new_tensor(context_grids)
    output["_nav_original_image_grid_thw"] = raw_grid.clone()

    online_indices = [index for index, block in enumerate(blocks) if not bool(block.get("is_cached_history", False))]
    pixel_values = output.get("pixel_values")
    if pixel_values is not None:
        pixel_chunks = torch.split(pixel_values, raw_grid.prod(-1).tolist())
        output["pixel_values"] = (
            torch.cat([pixel_chunks[index] for index in online_indices], dim=0)
            if online_indices
            else pixel_values[:0]
        )
    output["_nav_online_indices"] = online_indices
    output["_nav_online_image_grid_thw"] = raw_grid[online_indices] if online_indices else raw_grid[:0]
    output["image_grid_thw"] = output["_nav_context_image_grid_thw"]
    return output


def _insert_qwen35_cached_visual_spans(
    qwen_inputs: dict[str, Any],
    blocks: list[dict[str, Any]],
    *,
    sample_block_counts: list[int],
    image_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    history_visual_tokens: int,
    long_memory_visual_tokens: int,
    current_visual_tokens: int,
    merge_size: int,
    pad_token_id: int,
) -> dict[str, Any]:
    """Recreate cached-image spans on CPU, then reuse the canonical visual rewrite."""
    output = dict(qwen_inputs)
    input_ids = output["input_ids"]
    attention_mask = output.get("attention_mask")
    token_types = output.get("mm_token_type_ids")
    raw_grid = output.get("image_grid_thw")
    if token_types is None or raw_grid is None:
        raise ValueError("Qwen3.5 processor inputs require mm_token_type_ids and image_grid_thw")
    batch_size = int(input_ids.shape[0])
    if len(sample_block_counts) != batch_size:
        raise ValueError("Qwen3.5 cached visual metadata must match the processor batch size")

    rows: list[torch.Tensor] = []
    type_rows: list[torch.Tensor] = []
    expanded_grids: list[torch.Tensor] = []
    online_pixel_values = output.pop("pixel_values")
    image_cursor = 0
    block_cursor = 0
    for row_index in range(batch_size):
        row_ids = _active_row(input_ids, attention_mask, row_index)
        row_types = _active_row(token_types, attention_mask, row_index)
        spans = _image_token_spans(row_ids, image_token_id)
        row_block_count = int(sample_block_counts[row_index])
        row_blocks = blocks[block_cursor : block_cursor + row_block_count]
        if len(row_blocks) != row_block_count:
            raise ValueError("Qwen3.5 cached visual metadata exceeds the available blocks")
        row_processor_indices = [
            index
            for index, block in enumerate(row_blocks)
            if not bool(block.get("is_cached_history", False))
        ]
        if len(spans) != len(row_processor_indices):
            raise ValueError("Qwen3.5 processor image spans do not match online visual block indices")
        if row_blocks and not spans:
            raise ValueError("Qwen3.5 cached visual reconstruction requires at least one online image")

        span_by_block: dict[int, tuple[int, int, torch.Tensor]] = {}
        previous_end: int | None = None
        for relative_index, (start, end) in zip(row_processor_indices, spans, strict=True):
            if image_cursor >= int(raw_grid.shape[0]):
                raise ValueError("Qwen3.5 processor produced fewer grids than image spans")
            vision_start = start - 1
            if vision_start < 0 or int(row_ids[vision_start].item()) != int(vision_start_token_id):
                raise ValueError("Qwen3.5 image span must start immediately after vision_start")
            if end >= int(row_ids.shape[0]) or int(row_ids[end].item()) != int(vision_end_token_id):
                raise ValueError("Qwen3.5 image span must end immediately before vision_end")
            if previous_end is not None and vision_start != previous_end + 1:
                raise ValueError("Qwen3.5 online image spans must be adjacent")
            span_by_block[relative_index] = (
                vision_start,
                end,
                raw_grid[image_cursor],
            )
            previous_end = end
            image_cursor += 1

        if not spans:
            rows.append(row_ids)
            type_rows.append(row_types)
            block_cursor += row_block_count
            continue
        reference_grid = raw_grid[image_cursor - len(spans)]
        if any(
            not torch.equal(raw_grid[index], reference_grid)
            for index in range(image_cursor - len(spans), image_cursor)
        ) and len(spans) != row_block_count:
            raise ValueError("Qwen3.5 cached visual reconstruction requires a common online image grid")
        first_vision_start = spans[0][0] - 1
        id_chunks = [row_ids[:first_vision_start]]
        type_chunks = [row_types[:first_vision_start]]
        for relative_index in range(len(row_blocks)):
            span = span_by_block.get(relative_index)
            block_grid = reference_grid if span is None else span[2]
            if span is not None:
                vision_start, vision_end, _ = span
                id_chunks.append(row_ids[vision_start : vision_end + 1])
                type_chunks.append(row_types[vision_start : vision_end + 1])
            else:
                image_tokens = qwen35_postmerge_token_count(block_grid, spatial_merge_size=merge_size)
                id_chunks.append(row_ids.new_full((1,), vision_start_token_id))
                id_chunks.append(row_ids.new_full((image_tokens,), image_token_id))
                id_chunks.append(row_ids.new_full((1,), vision_end_token_id))
                type_chunks.append(row_types.new_zeros((1,)))
                type_chunks.append(row_types.new_ones((image_tokens,)))
                type_chunks.append(row_types.new_zeros((1,)))
            expanded_grids.append(block_grid)
        tail_start = spans[-1][1] + 1
        id_chunks.append(row_ids[tail_start:])
        type_chunks.append(row_types[tail_start:])
        rows.append(torch.cat(id_chunks))
        type_rows.append(torch.cat(type_chunks))
        block_cursor += row_block_count

    if block_cursor != len(blocks) or image_cursor != int(raw_grid.shape[0]):
        raise ValueError("Qwen3.5 cached visual alignment mismatch")
    inactive = input_ids[~attention_mask.to(dtype=torch.bool)] if attention_mask is not None else input_ids[:0]
    if inactive.numel():
        pad_token_id = int(inactive[0].item())
    output["input_ids"], output["attention_mask"] = _left_pad_rows(
        rows, pad_value=pad_token_id, template=input_ids
    )
    output["mm_token_type_ids"], _ = _left_pad_rows(type_rows, pad_value=0, template=token_types)
    output["image_grid_thw"] = torch.stack(expanded_grids, dim=0)
    rewritten = _rewrite_qwen35_visual_spans(
        output,
        blocks,
        image_token_id=image_token_id,
        vision_start_token_id=vision_start_token_id,
        vision_end_token_id=vision_end_token_id,
        history_visual_tokens=history_visual_tokens,
        long_memory_visual_tokens=long_memory_visual_tokens,
        current_visual_tokens=current_visual_tokens,
        merge_size=merge_size,
        pad_token_id=pad_token_id,
    )
    rewritten["pixel_values"] = online_pixel_values
    return rewritten


def _apply_qwen35_tvi_embeddings(
    *,
    inputs_embeds: torch.Tensor,
    tvi_mask: torch.Tensor,
    tvi_embeds: torch.Tensor,
) -> torch.Tensor:
    if tuple(tvi_mask.shape) != tuple(inputs_embeds.shape[:2]):
        raise ValueError(
            f"Qwen3.5 TVI mask shape {tuple(tvi_mask.shape)} does not match input shape {tuple(inputs_embeds.shape[:2])}"
        )
    if int(tvi_mask.sum().item()) != int(tvi_embeds.shape[0]):
        raise ValueError(
            f"TVI count {int(tvi_embeds.shape[0])} does not match Qwen3.5 TVI slots {int(tvi_mask.sum().item())}"
        )
    output = inputs_embeds.clone()
    output[tvi_mask.to(device=output.device, dtype=torch.bool)] = tvi_embeds.to(
        device=output.device,
        dtype=output.dtype,
    )
    return output


@dataclass
class NavVLAQwen35CPMDefaultConfig:
    name: str = "navvla_qwen35_cpm"
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "Qwen/Qwen3.5-4B",
            "attn_implementation": "flash_attention_2",
            "max_text_tokens": 2048,
            "action_placeholder_token": "<|fim_pad|>",
            "action_start_token": "<|fim_prefix|>",
            "action_end_token": "<|fim_suffix|>",
        }
    )
    navvla: dict = field(
        default_factory=lambda: {
            "tvi_mode": TIME_YAW_TVI_MODE,
            "use_platform_text": True,
            "history_visual_tokens": 4,
            "long_memory_source_visual_tokens": 4,
            "long_memory_visual_tokens": 128,
            "current_visual_tokens": 64,
            "long_memory_decay": 0.9,
            "long_memory_update_weight": 0.1,
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
            "visual_token_profile": DEFAULT_QWEN35_POOLED_HISTORY_VISUAL_TOKEN_PROFILE,
            "visual_cache_stage": QWEN35_POOLED_HISTORY_CACHE_STAGE,
            "visual_cache_input_resize": [256, 256],
            "visual_cache_encoder_ckpt": "Qwen/Qwen3.5-4B",
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 2560,
            "hidden_size": 2560,
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
                "cross_attention_dim": 2560,
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


@FRAMEWORK_REGISTRY.register("navvla_qwen35_cpm")
class NavVLA_Qwen35_CPM(baseframework):
    def __init__(self, config: Optional[dict] = None, **_kwargs: Any) -> None:
        super().__init__()
        self.config = merge_framework_config(NavVLAQwen35CPMDefaultConfig, config)
        nav_cfg = self.config.framework.navvla
        self.tvi_mode = str(nav_cfg.get("tvi_mode", TIME_YAW_TVI_MODE))
        self.tvi_dim = get_tvi_input_dim(self.tvi_mode)
        self.qwen35_vl_interface = get_vlm_model(config=self.config)
        resize = tuple(int(value) for value in nav_cfg.get("visual_cache_input_resize", [256, 256]))
        configure_qwen35_processor(self.qwen35_vl_interface.processor, resize)
        hidden_size = int(self.qwen35_vl_interface.model.config.text_config.hidden_size)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = hidden_size
        self.config.framework.action_model.num_target_vision_tokens = int(
            self.config.framework.action_model.action_horizon
        )
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
        long_memory_visual_tokens = int(nav_cfg.get("long_memory_visual_tokens", 128))
        self.long_memory_aggregator = (
            LongMemoryTokenAggregator(
                source_visual_tokens=int(nav_cfg.get("long_memory_source_visual_tokens", 4)),
                long_memory_visual_tokens=long_memory_visual_tokens,
                decay=float(nav_cfg.get("long_memory_decay", 0.9)),
                update_weight=float(nav_cfg.get("long_memory_update_weight", 0.1)),
                tvi_dim=self.tvi_dim,
            )
            if long_memory_visual_tokens > 0
            else None
        )
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.action_dim = int(self.config.framework.action_model.action_dim)
        configured_placeholders = nav_cfg.get("action_placeholder_count", None)
        self.action_placeholder_count = (
            self.action_dim * self.action_horizon
            if configured_placeholders is None
            else int(configured_placeholders)
        )
        if self.action_placeholder_count <= 0:
            raise ValueError("framework.navvla.action_placeholder_count must be positive")
        self.hidden_size = hidden_size
        model_type = str(getattr(self.qwen35_vl_interface.model.config, "model_type", ""))
        if "qwen3_5" not in model_type:
            raise ValueError(f"navvla_qwen35_cpm requires a Qwen3.5 checkpoint, got model_type={model_type!r}")

    def train(self, mode: bool = True):
        super().train(mode)
        self.qwen35_vl_interface.model.model.visual.eval()
        return self

    def _visual_token_budgets(self) -> tuple[int, int, int]:
        nav_cfg = self.config.framework.navvla
        return (
            int(nav_cfg.get("history_visual_tokens", 4)),
            int(nav_cfg.get("long_memory_visual_tokens", 128)),
            int(nav_cfg.get("current_visual_tokens", 64)),
        )

    def _samples_from_batch(self, examples):
        return samples_from_collated_batch(
            examples,
            extra_keys=(
                "history_cached_grid_thw",
                "history_cached_cache_stage",
                "history_cached_encoder_ckpt",
                "history_cached_storage_encoding",
                "long_memory_source_grid_thw",
                "long_memory_source_cache_stage",
                "long_memory_source_encoder_ckpt",
                "long_memory_source_storage_encoding",
                "online_long_memory_update_grid_thw",
                "online_long_memory_update_cache_stage",
                "online_long_memory_update_encoder_ckpt",
                "online_long_memory_update_storage_encoding",
            ),
        )

    def forward_vlm(self, batch) -> dict[str, torch.Tensor]:
        outputs = self.qwen35_vl_interface(**batch)
        return {"vlm_loss": outputs.loss}

    def _validate_sample_profile(self, sample: dict[str, Any]) -> None:
        metadata = sample.get("metadata", {}) or {}
        actual = str(metadata.get("visual_token_profile", ""))
        expected = str(self.config.framework.navvla.get("visual_token_profile", ""))
        if actual and expected and actual != expected:
            raise ValueError(f"Qwen3.5 visual_token_profile mismatch: batch={actual!r}, model={expected!r}")
        expected_encoder = str(
            self.config.framework.navvla.get(
                "visual_cache_encoder_ckpt", self.config.framework.qwenvl.get("base_vlm", "")
            )
        )
        for key in (
            "history_cached_encoder_ckpt",
            "long_memory_source_encoder_ckpt",
            "online_long_memory_update_encoder_ckpt",
        ):
            actual_encoder = str(sample.get(key, ""))
            if actual_encoder and expected_encoder and actual_encoder != expected_encoder:
                raise ValueError(
                    f"Qwen3.5 cache encoder mismatch for {key}: cache={actual_encoder!r}, model={expected_encoder!r}"
                )

    def _prepare_visual_image(self, image: Any) -> Image.Image:
        value = to_pil_preserve(image)
        resize = self.config.framework.navvla.get("visual_cache_input_resize", [256, 256])
        width, height = [int(item) for item in resize]
        if value.size != (width, height):
            value = value.resize((width, height), Image.Resampling.BICUBIC)
        return value

    def _validate_postmerge_grid(self, grid: Any) -> None:
        expected = self._visual_token_budgets()[2]
        merge_size = int(self.qwen35_vl_interface.model.model.visual.spatial_merge_size)
        actual = qwen35_postmerge_token_count(grid, spatial_merge_size=merge_size)
        if actual != expected:
            raise ValueError(
                f"Qwen3.5 preprocessing produced {actual} post-merge tokens, expected {expected}; "
                "training, offline cache, and online cache must use the same visual_cache_input_resize"
            )

    def _pool_postmerge(
        self,
        tokens: Any,
        grid_thw: Any,
        *,
        target_tokens: int,
    ) -> torch.Tensor:
        model = self.qwen35_vl_interface.model.model
        visual = model.visual
        device, dtype = _model_device_dtype(self.qwen35_vl_interface.model)
        value = torch.as_tensor(tokens, device=device, dtype=dtype)
        grid = torch.as_tensor(grid_thw, device=device, dtype=torch.long).reshape(3)
        self._validate_postmerge_grid(grid)
        expected = qwen35_postmerge_token_count(grid, spatial_merge_size=int(visual.spatial_merge_size))
        if value.ndim != 2 or int(value.shape[0]) != expected:
            raise ValueError(
                f"Qwen3.5 post-merge cache must have shape [{expected}, llm_hidden], got {tuple(value.shape)}"
            )
        if int(value.shape[-1]) != self.hidden_size:
            raise ValueError(
                f"Qwen3.5 post-merge hidden dim {value.shape[-1]} does not match LLM hidden dim {self.hidden_size}"
            )
        pooled = pool_qwen35_postmerge(
            value,
            grid,
            target_tokens=int(target_tokens),
            spatial_merge_size=int(visual.spatial_merge_size),
        )
        return pooled.to(device=device, dtype=dtype)

    def _cached_pooled_history_tokens(
        self,
        tokens: Any,
        grid_thw: Any,
        *,
        storage_encoding: str,
    ) -> torch.Tensor:
        device, dtype = _model_device_dtype(self.qwen35_vl_interface.model)
        value = decode_qwen35_cache_tokens(
            tokens,
            storage_encoding=storage_encoding,
            device=device,
            model_dtype=dtype,
        )
        grid = torch.as_tensor(grid_thw, device=device, dtype=torch.long).reshape(3)
        qwen35_postmerge_token_count(
            grid,
            spatial_merge_size=int(self.qwen35_vl_interface.model.model.visual.spatial_merge_size),
        )
        expected_tokens = self._visual_token_budgets()[0]
        expected_shape = (expected_tokens, self.hidden_size)
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"Qwen3.5 pooled-history cache shape {tuple(value.shape)} != {expected_shape}")
        return value

    def _convert_source_cache(self, sample: dict[str, Any], *, prefix: str) -> None:
        tokens_key = f"{prefix}_tokens"
        grid_key = f"{prefix}_grid_thw"
        stage_key = f"{prefix}_cache_stage"
        encoding_key = f"{prefix}_storage_encoding"
        source = sample.get(tokens_key)
        if source is None:
            return
        stage = sample.get(stage_key, QWEN35_POOLED_HISTORY_CACHE_STAGE)
        if isinstance(stage, (list, tuple, np.ndarray)):
            values = np.asarray(stage).reshape(-1).tolist()
            stage = values[0] if values else QWEN35_POOLED_HISTORY_CACHE_STAGE
        if str(stage) == QWEN35_LONG_MEMORY_SOURCE_POOLED_STAGE:
            return
        if str(stage) != QWEN35_POOLED_HISTORY_CACHE_STAGE:
            raise ValueError(f"unsupported Qwen3.5 cache stage {stage!r} for {prefix}")
        grids = sample.get(grid_key)
        if grids is None:
            raise ValueError(f"{prefix} requires grid_thw for Qwen3.5 pooled-history cache")
        source_array = np.asarray(source)
        grid_array = np.asarray(grids).reshape(-1, 3)
        if int(source_array.shape[0]) != int(grid_array.shape[0]):
            raise ValueError(f"{prefix} token/grid row mismatch: {source_array.shape[0]} != {grid_array.shape[0]}")
        target = int(self.config.framework.navvla.get("long_memory_source_visual_tokens", 4))
        cache_tokens = self._visual_token_budgets()[0]
        if target != cache_tokens:
            raise ValueError(
                f"long_memory_source_visual_tokens={target} must equal cached pooled-history tokens={cache_tokens}"
            )
        converted = [
            self._cached_pooled_history_tokens(
                value,
                grid,
                storage_encoding=str(sample.get(encoding_key, "")),
            )
            for value, grid in zip(source_array, grid_array, strict=True)
        ]
        if converted:
            sample[tokens_key] = torch.stack(converted, dim=0)
        else:
            device, dtype = _model_device_dtype(self.qwen35_vl_interface.model)
            sample[tokens_key] = torch.zeros((0, target, self.hidden_size), device=device, dtype=dtype)
        sample[stage_key] = QWEN35_LONG_MEMORY_SOURCE_POOLED_STAGE

    def _attach_long_memory_tokens(self, samples: list[dict[str, Any]]) -> None:
        for sample in samples:
            self._validate_sample_profile(sample)
            if sample.get("long_memory_tokens") is None:
                self._convert_source_cache(sample, prefix="long_memory_source")
        device, dtype = _model_device_dtype(self.qwen35_vl_interface.model)
        attach_navvla_long_memory_tokens(self, samples, device=device, dtype=dtype)

    def _compute_online_long_memory_updates(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for sample in samples:
            self._validate_sample_profile(sample)
            self._convert_source_cache(sample, prefix="online_long_memory_update")
        device, dtype = _model_device_dtype(self.qwen35_vl_interface.model)
        return compute_navvla_online_long_memory_updates(self, samples, device=device, dtype=dtype)

    def _build_instruction(self, sample: dict[str, Any]) -> str:
        return build_navvla_instruction(self, sample)

    def _sample_required_cameras(self, sample: dict[str, Any]) -> list[str]:
        return sample_required_cameras(sample)

    @staticmethod
    def _scatter_image_embeddings(
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        image_embeddings: torch.Tensor,
        image_token_id: int,
    ) -> torch.Tensor:
        return scatter_image_embeddings(inputs_embeds, input_ids, image_embeddings, image_token_id)

    def _mask_history_tvi_embeddings(
        self,
        embeddings: torch.Tensor,
        blocks: list[dict[str, Any]],
        *,
        probability: float,
    ) -> torch.Tensor:
        return mask_history_tvi_embeddings(self, embeddings, blocks, probability=probability)

    def _build_qwen35_inputs(
        self,
        samples: list[dict[str, Any]],
        *,
        history_shuffle_probability: float,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        history_tokens, long_tokens, current_tokens = self._visual_token_budgets()
        batch_images: list[list[Image.Image]] = []
        instructions: list[str] = []
        action_suffixes: list[str] = []
        blocks: list[dict[str, Any]] = []
        sample_block_counts: list[int] = []
        action_suffix = self.qwen35_vl_interface.build_action_placeholder_suffix(self.action_placeholder_count)
        for sample in samples:
            self._validate_sample_profile(sample)
            online_images, sample_blocks = build_navvla_cached_visual_sequence(
                sample,
                required_cameras=self._sample_required_cameras(sample),
                history_shuffle_probability=history_shuffle_probability,
                tvi_dim=self.tvi_dim,
            )
            online_iter = iter(online_images)
            sample_images: list[Image.Image] = []
            for block in sample_blocks:
                if bool(block.get("is_cached_history", False)):
                    continue
                sample_images.append(self._prepare_visual_image(next(online_iter)))
            try:
                next(online_iter)
            except StopIteration:
                pass
            else:
                raise ValueError("NavVLA visual sequence produced more online images than online visual blocks")
            batch_images.append(sample_images)
            instructions.append(self._build_instruction(sample))
            action_suffixes.append(action_suffix)
            blocks.extend(sample_blocks)
            sample_block_counts.append(len(sample_blocks))
        qwen_inputs = self.qwen35_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            action_suffixes=action_suffixes,
            move_to_device=False,
        )
        model = self.qwen35_vl_interface.model.model
        qwen_inputs = _insert_qwen35_cached_visual_spans(
            dict(qwen_inputs),
            blocks,
            sample_block_counts=sample_block_counts,
            image_token_id=int(model.config.image_token_id),
            vision_start_token_id=int(model.config.vision_start_token_id),
            vision_end_token_id=int(model.config.vision_end_token_id),
            history_visual_tokens=history_tokens,
            long_memory_visual_tokens=long_tokens,
            current_visual_tokens=current_tokens,
            merge_size=int(model.visual.spatial_merge_size),
            pad_token_id=int(self.qwen35_vl_interface.processor.tokenizer.pad_token_id or 0),
        )
        return {
            key: value.to(self.qwen35_vl_interface.model.device) if isinstance(value, torch.Tensor) else value
            for key, value in qwen_inputs.items()
        }, blocks

    def _encode_online_postmerge(self, qwen_inputs: dict[str, Any]) -> list[torch.Tensor]:
        grids = qwen_inputs["_nav_online_image_grid_thw"]
        if int(grids.shape[0]) == 0:
            return []
        model = self.qwen35_vl_interface.model.model
        return encode_qwen35_postmerge_batched(
            model.visual,
            qwen_inputs["pixel_values"].to(dtype=model.visual.dtype),
            grids,
        )

    @torch.inference_mode()
    def encode_history_images(self, images: list[Any]) -> list[dict[str, Any]]:
        images = [self._prepare_visual_image(image) for image in images]
        if not images:
            return []
        messages = [
            [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": ""}]}]
            for image in images
        ]
        inputs = self.qwen35_vl_interface.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.qwen35_vl_interface.model.device)
        model = self.qwen35_vl_interface.model.model
        grids = inputs["image_grid_thw"]
        chunks = encode_qwen35_postmerge_one_by_one(
            model.visual,
            inputs["pixel_values"].to(dtype=model.visual.dtype),
            grids,
        )
        for grid in grids:
            self._validate_postmerge_grid(grid)
        profile = str(
            self.config.framework.navvla.get(
                "visual_token_profile", DEFAULT_QWEN35_POOLED_HISTORY_VISUAL_TOKEN_PROFILE
            )
        )
        cache_tokens = self._visual_token_budgets()[0]
        return [
            {
                "tokens": bf16_to_numpy_bits(
                    self._pool_postmerge(chunk, grid, target_tokens=cache_tokens)
                ),
                "grid_thw": grid.detach().to(torch.int64).cpu().numpy(),
                "cache_stage": QWEN35_POOLED_HISTORY_CACHE_STAGE,
                "storage_encoding": BFLOAT16_BITS_STORAGE_ENCODING,
                "visual_token_profile": profile,
                "encoder_ckpt": str(
                    self.config.framework.navvla.get(
                        "visual_cache_encoder_ckpt", self.config.framework.qwenvl.get("base_vlm", "")
                    )
                ),
            }
            for chunk, grid in zip(chunks, grids, strict=True)
        ]

    def _fuse_qwen35_visual_tokens(
        self,
        qwen_inputs: dict[str, Any],
        blocks: list[dict[str, Any]],
        *,
        capture_online_current_cache: bool,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        history_tokens, long_tokens, current_tokens = self._visual_token_budgets()
        cache_tokens = history_tokens
        online_indices = list(qwen_inputs["_nav_online_indices"])
        online_grids = qwen_inputs["_nav_online_image_grid_thw"]
        online_postmerge = self._encode_online_postmerge(qwen_inputs)
        if len(online_postmerge) != len(online_indices):
            raise ValueError("Qwen3.5 online post-merge feature count does not match online visual blocks")
        online_by_index = dict(zip(online_indices, zip(online_postmerge, online_grids, strict=True), strict=True))
        chunks: list[torch.Tensor] = []
        records: list[dict[str, Any]] = []
        for block_index, block in enumerate(blocks):
            target = target_visual_tokens_for_block(
                block,
                history_visual_tokens=history_tokens,
                long_memory_visual_tokens=long_tokens,
                current_visual_tokens=current_tokens,
            )
            if block_index in online_by_index:
                postmerge, grid = online_by_index[block_index]
                self._validate_postmerge_grid(grid)
                if bool(block.get("is_history", False)):
                    chunks.append(self._pool_postmerge(postmerge, grid, target_tokens=target))
                else:
                    if tuple(postmerge.shape) != (current_tokens, self.hidden_size):
                        raise ValueError(
                            f"Qwen3.5 current merger output {tuple(postmerge.shape)} != "
                            f"{(current_tokens, self.hidden_size)}"
                        )
                    chunks.append(postmerge)
                if capture_online_current_cache and not bool(block.get("is_history", False)):
                    records.append(
                        {
                            "camera_name": str(block["camera_name"]),
                            "frame_index": int(block.get("frame_index", 0)),
                            "tokens": bf16_to_numpy_bits(
                                self._pool_postmerge(postmerge, grid, target_tokens=history_tokens)
                            ),
                            "grid_thw": grid.detach().to(torch.int64).cpu().numpy(),
                            "cache_stage": QWEN35_POOLED_HISTORY_CACHE_STAGE,
                            "storage_encoding": BFLOAT16_BITS_STORAGE_ENCODING,
                            "visual_token_profile": str(
                                self.config.framework.navvla.get(
                                    "visual_token_profile", DEFAULT_QWEN35_POOLED_HISTORY_VISUAL_TOKEN_PROFILE
                                )
                            ),
                            "encoder_ckpt": str(
                                self.config.framework.navvla.get(
                                    "visual_cache_encoder_ckpt", self.config.framework.qwenvl.get("base_vlm", "")
                                )
                            ),
                        }
                    )
                continue
            if bool(block.get("is_long_memory", False)):
                device, dtype = _model_device_dtype(self.qwen35_vl_interface.model)
                cached = torch.as_tensor(
                    block["sample"]["long_memory_tokens"][int(block["long_memory_index"])],
                    device=device,
                    dtype=dtype,
                )
                if tuple(cached.shape) != (target, self.hidden_size):
                    raise ValueError(
                        f"Qwen3.5 long-memory token shape {tuple(cached.shape)} "
                        f"!= {(target, self.hidden_size)}"
                    )
                chunks.append(cached)
                continue
            sample = block["sample"]
            history_index = int(block["cached_history_index"])
            stage = sample.get("history_cached_cache_stage", QWEN35_POOLED_HISTORY_CACHE_STAGE)
            if isinstance(stage, (list, tuple, np.ndarray)):
                stage = np.asarray(stage).reshape(-1)[0]
            if str(stage) != QWEN35_POOLED_HISTORY_CACHE_STAGE:
                raise ValueError(f"history cache must use stage {QWEN35_POOLED_HISTORY_CACHE_STAGE!r}, got {stage!r}")
            grids = sample.get("history_cached_grid_thw")
            if grids is None:
                raise ValueError("history_cached_grid_thw is required for Qwen3.5 pooled-history cache")
            if target != cache_tokens:
                raise ValueError(f"cached history target={target} must equal cache token count={cache_tokens}")
            chunks.append(
                self._cached_pooled_history_tokens(
                    sample["history_cached_embeds"][history_index],
                    grids[history_index],
                    storage_encoding=str(sample.get("history_cached_storage_encoding", "")),
                )
            )
        device, dtype = _model_device_dtype(self.qwen35_vl_interface.model)
        return (
            torch.cat(chunks, dim=0) if chunks else torch.zeros((0, self.hidden_size), device=device, dtype=dtype),
            records,
        )

    def _forward_qwen35_backbone(
        self,
        qwen_inputs: dict[str, Any],
        blocks: list[dict[str, Any]],
        *,
        capture_online_current_cache: bool,
        tvi_mask_probability: float,
    ) -> tuple[Any, list[dict[str, Any]]]:
        model = self.qwen35_vl_interface.model.model
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs["attention_mask"]
        token_types = qwen_inputs["mm_token_type_ids"]
        inputs_embeds = model.get_input_embeddings()(input_ids)
        image_embeds, records = self._fuse_qwen35_visual_tokens(
            qwen_inputs, blocks, capture_online_current_cache=capture_online_current_cache
        )
        inputs_embeds = self._scatter_image_embeddings(
            inputs_embeds, input_ids, image_embeds, int(model.config.image_token_id)
        )
        if blocks:
            tvi = torch.as_tensor(
                as_numpy_tvi(np.stack([block["tvi"] for block in blocks]), tvi_dim=self.tvi_dim),
                device=inputs_embeds.device,
                dtype=torch.float32,
            )
            tvi_embeds = self._mask_history_tvi_embeddings(
                self.tvi_embedding(tvi), blocks, probability=tvi_mask_probability
            )
            inputs_embeds = _apply_qwen35_tvi_embeddings(
                inputs_embeds=inputs_embeds,
                tvi_mask=qwen_inputs["_nav_tvi_mask"],
                tvi_embeds=tvi_embeds,
            )
        position_ids = model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=qwen_inputs["_nav_context_image_grid_thw"],
            attention_mask=attention_mask,
            mm_token_type_ids=token_types,
        )
        outputs = model.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        qwen_inputs["input_ids"] = input_ids
        qwen_inputs["attention_mask"] = attention_mask
        qwen_inputs["mm_token_type_ids"] = token_types
        return outputs, records

    def _forward_vlm_for_action(
        self,
        samples: list[dict[str, Any]],
        *,
        capture_online_current_cache: bool = False,
        history_shuffle_probability: float = 0.0,
        tvi_mask_probability: float = 0.0,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        self._attach_long_memory_tokens(samples)
        qwen_inputs, blocks = self._build_qwen35_inputs(
            samples, history_shuffle_probability=history_shuffle_probability
        )
        outputs, records = self._forward_qwen35_backbone(
            qwen_inputs,
            blocks,
            capture_online_current_cache=capture_online_current_cache,
            tvi_mask_probability=tvi_mask_probability,
        )
        action_hidden = self.qwen35_vl_interface.gather_action_placeholder_hidden_states(
            outputs.last_hidden_state,
            qwen_inputs["input_ids"],
            num_placeholders=self.action_placeholder_count,
        )
        if self.long_memory_aggregator is not None and any(
            bool(sample.pop("_long_memory_zero_dependency", False)) for sample in samples
        ):
            action_hidden = action_hidden + self.long_memory_aggregator.zero_dependency(action_hidden)
        return action_hidden, records

    def forward(
        self,
        examples: list[dict[str, Any]] | dict[str, Any] | None = None,
        *,
        training_step: int | None = None,
        total_training_steps: int | None = None,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        return forward_navvla_action(
            self,
            examples,
            training_step=training_step,
            total_training_steps=total_training_steps,
        )

    @torch.inference_mode()
    def predict_action(
        self,
        examples: list[dict[str, Any]] | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return predict_navvla_action(
            self,
            examples,
            tvi_mask_probability=float(kwargs.get("tvi_mask_probability", 0.0)),
        )
