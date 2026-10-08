# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2026].
# Design and Merged by [Jinhui YE / HKUST University] in [2026].

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from starVLA.model.modules.input_context import (
    as_numpy_tvi,
    build_navvla_cached_visual_sequence,
    gather_action_queries,
    image_token_spans,
    mask_history_tvi_embeddings,
    navigation_messages,
    rgb_image,
    scatter_image_embeddings,
    target_visual_tokens_for_block,
    validate_single_token,
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
from starVLA.model.modules.tvi import get_tvi_input_dim
from starVLA.training.trainer_utils import initialize_overwatch
from tool.navvla.visual_token_cache import (
    DEFAULT_QWEN35_POOLED_HISTORY_VISUAL_TOKEN_PROFILE,
    QWEN35_POOLED_HISTORY_CACHE_STAGE,
)

QWEN35_LONG_MEMORY_SOURCE_POOLED_STAGE = "navvla_long_memory_source_pooled"


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


def _insert_qwen35_cached_visual_spans(
    qwen_inputs,
    blocks,
    *,
    sample_block_counts,
    image_token_id,
    vision_start_token_id,
    vision_end_token_id,
    history_visual_tokens,
    long_memory_visual_tokens,
    current_visual_tokens,
    merge_size,
    pad_token_id,
):
    """Build final visual/TVI spans once, preserving the native online pixels and grids."""
    output = dict(qwen_inputs)
    input_ids, mask = output["input_ids"], output["attention_mask"]
    token_types, raw_grid = output["mm_token_type_ids"], output["image_grid_thw"]
    rows, type_rows, tvi_rows, original_grids, context_grids, online_indices = [], [], [], [], [], []
    image_cursor = block_cursor = 0
    for row_index, count in enumerate(sample_block_counts):
        ids = _active_row(input_ids, mask, row_index)
        types = _active_row(token_types, mask, row_index)
        row_blocks = blocks[block_cursor : block_cursor + count]
        spans = image_token_spans(ids, image_token_id)
        online_count = sum(not block["is_cached_history"] for block in row_blocks)
        if not spans or len(spans) != online_count:
            raise ValueError("Qwen processor spans must match online blocks, including a current image")
        row_grids = raw_grid[image_cursor : image_cursor + online_count]
        if online_count != count and not torch.all(row_grids == row_grids[0]):
            raise ValueError("Qwen cached visual context requires a common online image grid")
        for i, (start, end) in enumerate(spans):
            if ids[start - 1] != vision_start_token_id or ids[end] != vision_end_token_id:
                raise ValueError("Qwen processor image spans require native vision delimiters")
            if i and start - 1 != spans[i - 1][1] + 1:
                raise ValueError("Qwen online image spans must be adjacent")
        prefix = spans[0][0] - 1
        id_chunks, type_chunks = [ids[:prefix]], [types[:prefix]]
        tvi_chunks = [types.new_zeros(prefix)]
        online_local = 0
        for local, block in enumerate(row_blocks):
            cached = block["is_cached_history"]
            grid = row_grids[0] if cached else row_grids[online_local]
            target = target_visual_tokens_for_block(
                block,
                history_visual_tokens=history_visual_tokens,
                long_memory_visual_tokens=long_memory_visual_tokens,
                current_visual_tokens=current_visual_tokens,
            )
            context_grid = _grid_shape_for_token_count(grid, target, merge_size=merge_size)
            if qwen35_postmerge_token_count(context_grid, spatial_merge_size=merge_size) != target:
                raise ValueError(f"Cannot represent {target} tokens as a Qwen M-RoPE grid")
            slot = [pad_token_id] + ([] if cached else [vision_start_token_id])
            slot += [image_token_id] * target + ([] if cached else [vision_end_token_id])
            slot_types = [0] * (1 if cached else 2) + [1] * target + ([] if cached else [0])
            id_chunks.append(ids.new_tensor(slot))
            type_chunks.append(types.new_tensor(slot_types))
            tvi_chunks.append(types.new_tensor([1] + [0] * (len(slot) - 1)))
            original_grids.append(grid)
            context_grids.append(context_grid)
            if not cached:
                online_indices.append(block_cursor + local)
                online_local += 1
        tail = spans[-1][1] + 1
        rows.append(torch.cat([*id_chunks, ids[tail:]]))
        type_rows.append(torch.cat([*type_chunks, types[tail:]]))
        tvi_rows.append(torch.cat([*tvi_chunks, types.new_zeros(len(ids) - tail)]))
        image_cursor += online_count
        block_cursor += count
    if block_cursor != len(blocks) or image_cursor != len(raw_grid) or len(rows) != len(input_ids):
        raise ValueError("Qwen batch, visual blocks and image grids must align")
    output["input_ids"], output["attention_mask"] = _left_pad_rows(rows, pad_value=pad_token_id, template=input_ids)
    output["mm_token_type_ids"], _ = _left_pad_rows(type_rows, pad_value=0, template=token_types)
    tvi_mask, _ = _left_pad_rows(tvi_rows, pad_value=0, template=token_types)
    output["_nav_tvi_mask"] = tvi_mask.bool()
    output["_nav_context_image_grid_thw"] = raw_grid.new_tensor(context_grids)
    output["_nav_original_image_grid_thw"] = torch.stack(original_grids)
    output["_nav_online_indices"] = online_indices
    output["_nav_online_image_grid_thw"] = raw_grid
    output["image_grid_thw"] = output["_nav_context_image_grid_thw"]
    return output


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


logger = initialize_overwatch(__name__)

DEFAULT_ACTION_PLACEHOLDER_TOKEN = "<|fim_pad|>"
DEFAULT_ACTION_START_TOKEN = "<|fim_prefix|>"
DEFAULT_ACTION_END_TOKEN = "<|fim_suffix|>"


class _QWen3_5_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3.5-VL (Qwen3_5ForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3.5-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3.5-4B

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        trainer_config = config.get("trainer", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3.5-4B")
        attn_implementation = qwenvl_config.get("attn_implementation", "flash_attention_2")
        enable_gradient_checkpointing = bool(trainer_config.get("enable_gradient_checkpointing", False))
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError as exc:
                raise ImportError("Qwen3.5 requires flash_attn when attn_implementation=flash_attention_2") from exc

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        )
        if enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.text_config.use_cache = False
            logger.info("Qwen3.5 gradient checkpointing enabled (use_reentrant=False)")

        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3.5 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        tokenizer = self.processor.tokenizer
        placeholder = str(qwenvl_config.get("action_placeholder_token", DEFAULT_ACTION_PLACEHOLDER_TOKEN))
        action_start = str(qwenvl_config.get("action_start_token", DEFAULT_ACTION_START_TOKEN))
        action_end = str(qwenvl_config.get("action_end_token", DEFAULT_ACTION_END_TOKEN))
        self.action_placeholder_token, self.action_placeholder_token_id = validate_single_token(
            tokenizer, placeholder, role="placeholder"
        )
        self.action_start_token, self.action_start_token_id = validate_single_token(
            tokenizer, action_start, role="start"
        )
        self.action_end_token, self.action_end_token_id = validate_single_token(tokenizer, action_end, role="end")

        self.IMAGE_TOKEN_INDEX = int(self.model.config.image_token_id)

        self.tvi_dim = get_tvi_input_dim(str(config.framework.navvla.tvi_mode))
        self.hidden_size = int(self.model.config.text_config.hidden_size)
        self.action_placeholder_count = int(
            config.framework.navvla.get("action_placeholder_count")
            or config.framework.action_model.action_dim * config.framework.action_model.action_horizon
        )
        configure_qwen35_processor(self.processor, tuple(config.framework.navvla.visual_cache_input_resize))

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_action_placeholder_suffix(self, num_placeholders: int) -> str:
        suffix = self.action_start_token + self.action_placeholder_token * int(num_placeholders) + self.action_end_token
        token_ids = self.processor.tokenizer.encode(suffix, add_special_tokens=False)
        if token_ids.count(self.action_placeholder_token_id) != int(num_placeholders):
            raise ValueError(
                "Qwen3.5 tokenizer did not preserve the repeated action-placeholder suffix as single tokens"
            )
        return suffix

    def build_qwenvl_inputs(self, images, instructions, action_suffixes, *, move_to_device=True):
        messages = navigation_messages(
            self.processor.tokenizer,
            images,
            instructions,
            action_suffixes,
            max_text_tokens=int(self.config.framework.qwenvl.max_text_tokens),
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
            add_vision_id=False,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device) if move_to_device else inputs

    def prepare_samples(self, samples):
        if self.training and any("history_cached_embeds" in s or "long_memory_source_tokens" in s for s in samples):
            if any(p.requires_grad for p in self.model.model.visual.parameters()):
                raise ValueError("Cached visual inputs require a frozen visual encoder")
        for sample in samples:
            self._validate_sample_profile(sample)
            for prefix in ("long_memory_source", "online_long_memory_update"):
                self._convert_source_cache(sample, prefix=prefix)

    def _visual_token_budgets(self) -> tuple[int, int, int]:
        nav_cfg = self.config.framework.navvla
        return (
            int(nav_cfg.get("history_visual_tokens", 4)),
            int(nav_cfg.get("long_memory_visual_tokens", 128)),
            int(nav_cfg.get("current_visual_tokens", 64)),
        )

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
        value = rgb_image(image)
        resize = self.config.framework.navvla.get("visual_cache_input_resize", [256, 256])
        width, height = [int(item) for item in resize]
        if value.size != (width, height):
            value = value.resize((width, height), Image.Resampling.BICUBIC)
        return value

    def _validate_postmerge_grid(self, grid: Any) -> None:
        expected = self._visual_token_budgets()[2]
        merge_size = int(self.model.model.visual.spatial_merge_size)
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
        model = self.model.model
        visual = model.visual
        device, dtype = _model_device_dtype(self.model)
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
        device, dtype = _model_device_dtype(self.model)
        value = decode_qwen35_cache_tokens(
            tokens,
            storage_encoding=storage_encoding,
            device=device,
            model_dtype=dtype,
        )
        grid = torch.as_tensor(grid_thw, device=device, dtype=torch.long).reshape(3)
        qwen35_postmerge_token_count(
            grid,
            spatial_merge_size=int(self.model.model.visual.spatial_merge_size),
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
            device, dtype = _model_device_dtype(self.model)
            sample[tokens_key] = torch.zeros((0, target, self.hidden_size), device=device, dtype=dtype)
        sample[stage_key] = QWEN35_LONG_MEMORY_SOURCE_POOLED_STAGE

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
        action_suffix = self.build_action_placeholder_suffix(self.action_placeholder_count)
        for sample in samples:
            online_images, sample_blocks = build_navvla_cached_visual_sequence(
                sample,
                required_cameras=sample["metadata"]["required_cameras"],
                history_shuffle_probability=history_shuffle_probability,
                tvi_dim=self.tvi_dim,
            )
            batch_images.append([self._prepare_visual_image(image) for image in online_images])
            instructions.append(sample["instruction_text"])
            action_suffixes.append(action_suffix)
            blocks.extend(sample_blocks)
            sample_block_counts.append(len(sample_blocks))
        qwen_inputs = self.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            action_suffixes=action_suffixes,
            move_to_device=False,
        )
        model = self.model.model
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
            pad_token_id=int(self.processor.tokenizer.pad_token_id or 0),
        )
        return {
            key: value.to(self.model.device) if isinstance(value, torch.Tensor) else value
            for key, value in qwen_inputs.items()
        }, blocks

    def _encode_online_postmerge(self, qwen_inputs: dict[str, Any]) -> list[torch.Tensor]:
        grids = qwen_inputs["_nav_online_image_grid_thw"]
        if int(grids.shape[0]) == 0:
            return []
        model = self.model.model
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
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        model = self.model.model
        grids = inputs["image_grid_thw"]
        chunks = encode_qwen35_postmerge_one_by_one(
            model.visual,
            inputs["pixel_values"].to(dtype=model.visual.dtype),
            grids,
        )
        for grid in grids:
            self._validate_postmerge_grid(grid)
        profile = str(
            self.config.framework.navvla.get("visual_token_profile", DEFAULT_QWEN35_POOLED_HISTORY_VISUAL_TOKEN_PROFILE)
        )
        cache_tokens = self._visual_token_budgets()[0]
        return [
            {
                "tokens": bf16_to_numpy_bits(self._pool_postmerge(chunk, grid, target_tokens=cache_tokens)),
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
                device, dtype = _model_device_dtype(self.model)
                cached = torch.as_tensor(
                    block["sample"]["long_memory_tokens"][int(block["long_memory_index"])],
                    device=device,
                    dtype=dtype,
                )
                if tuple(cached.shape) != (target, self.hidden_size):
                    raise ValueError(
                        f"Qwen3.5 long-memory token shape {tuple(cached.shape)} != {(target, self.hidden_size)}"
                    )
                chunks.append(cached)
                continue
            sample = block["sample"]
            history_index = int(block["cached_history_index"])
            stage = sample.get("history_cached_cache_stage", QWEN35_POOLED_HISTORY_CACHE_STAGE)
            if str(stage) != QWEN35_POOLED_HISTORY_CACHE_STAGE:
                raise ValueError(f"history cache must use stage {QWEN35_POOLED_HISTORY_CACHE_STAGE!r}, got {stage!r}")
            grids = sample.get("history_cached_grid_thw")
            if grids is None:
                raise ValueError("history_cached_grid_thw is required for Qwen3.5 pooled-history cache")
            chunks.append(
                self._cached_pooled_history_tokens(
                    sample["history_cached_embeds"][history_index],
                    grids[history_index],
                    storage_encoding=str(sample.get("history_cached_storage_encoding", "")),
                )
            )
        device, dtype = _model_device_dtype(self.model)
        return (
            torch.cat(chunks, dim=0) if chunks else torch.zeros((0, self.hidden_size), device=device, dtype=dtype),
            records,
        )

    def _forward_qwen35_backbone(
        self,
        qwen_inputs: dict[str, Any],
        blocks: list[dict[str, Any]],
        *,
        tvi_embedding,
        capture_online_current_cache: bool,
        tvi_mask_probability: float,
    ) -> tuple[Any, list[dict[str, Any]]]:
        model = self.model.model
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs["attention_mask"]
        token_types = qwen_inputs["mm_token_type_ids"]
        inputs_embeds = model.get_input_embeddings()(input_ids)
        image_embeds, records = self._fuse_qwen35_visual_tokens(
            qwen_inputs, blocks, capture_online_current_cache=capture_online_current_cache
        )
        inputs_embeds = scatter_image_embeddings(
            inputs_embeds, input_ids, image_embeds, int(model.config.image_token_id)
        )
        if blocks:
            tvi = torch.as_tensor(
                as_numpy_tvi(np.stack([block["tvi"] for block in blocks]), tvi_dim=self.tvi_dim),
                device=inputs_embeds.device,
                dtype=torch.float32,
            )
            tvi_embeds = mask_history_tvi_embeddings(
                tvi_embedding, tvi_embedding(tvi), blocks, probability=tvi_mask_probability
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

    def encode_context(
        self,
        samples: list[dict[str, Any]],
        *,
        tvi_embedding,
        capture_online_current_cache: bool = False,
        history_shuffle_probability: float = 0.0,
        tvi_mask_probability: float = 0.0,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        qwen_inputs, blocks = self._build_qwen35_inputs(samples, history_shuffle_probability=history_shuffle_probability)
        outputs, records = self._forward_qwen35_backbone(
            qwen_inputs,
            blocks,
            capture_online_current_cache=capture_online_current_cache,
            tvi_mask_probability=tvi_mask_probability,
            tvi_embedding=tvi_embedding,
        )
        action_hidden = gather_action_queries(
            outputs.last_hidden_state,
            qwen_inputs["input_ids"],
            token_id=self.action_placeholder_token_id,
            num_placeholders=self.action_placeholder_count,
        )
        return action_hidden, records

    def freeze_visual_eval(self):
        if not any(p.requires_grad for p in self.model.model.visual.parameters()):
            self.model.model.visual.eval()
