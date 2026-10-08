# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForImageTextToText, AutoProcessor

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
from starVLA.model.modules.tvi import get_tvi_input_dim


def _model_device_dtype(model):
    parameter = next(model.parameters())
    return parameter.device, parameter.dtype


def _pad_rows(rows, pad_token_id):
    ids = pad_sequence(rows, batch_first=True, padding_value=pad_token_id)
    mask = pad_sequence([torch.ones_like(row) for row in rows], batch_first=True)
    return ids, mask


def _navigation_spans(inputs, blocks, counts, *, image_token_id, prefix_id, suffix_ids, budgets, pad_token_id):
    """Insert cached slots and resize online slots in a single pass."""
    output = dict(inputs)
    rows, block_cursor = [], 0
    for ids, mask, count in zip(inputs["input_ids"], inputs["attention_mask"], counts, strict=True):
        row = ids[mask.bool()]
        spans = image_token_spans(row, image_token_id)
        row_blocks = blocks[block_cursor : block_cursor + count]
        if len(spans) != sum(not block["is_cached_history"] for block in row_blocks) or not spans:
            raise ValueError("MiniCPM processor spans must match the online image blocks")
        cursor = spans[0][0] - 1
        chunks = [row[:cursor]]
        online = 0
        for block in row_blocks:
            target = target_visual_tokens_for_block(block, **budgets)
            if block["is_cached_history"]:
                chunks.append(row.new_tensor([prefix_id] + [image_token_id] * target + suffix_ids))
            else:
                start, end = spans[online]
                if row[start - 1] != prefix_id:
                    raise ValueError("MiniCPM processor span must start after its native image prefix")
                chunks.extend([row[cursor:start], row.new_full((target,), image_token_id)])
                cursor, online = end, online + 1
        rows.append(torch.cat([*chunks, row[cursor:]]))
        block_cursor += count
    if block_cursor != len(blocks):
        raise ValueError("MiniCPM visual metadata does not cover the batch")
    output["input_ids"], output["attention_mask"] = _pad_rows(rows, pad_token_id)
    return output


def insert_navvla_tvi_prefix_tokens(
    *,
    input_ids,
    inputs_embeds,
    attention_mask,
    blocks,
    tvi_embeds,
    vision_prefix_token_id,
    image_token_id,
    tvi_token_id=0,
):
    ids_rows, embed_rows, block_cursor = [], [], 0
    for ids, embeds, mask in zip(input_ids, inputs_embeds, attention_mask, strict=True):
        row, embeddings = ids[mask.bool()], embeds[mask.bool()]
        ids_chunks, embed_chunks, cursor = [], [], 0
        for start, end in image_token_spans(row, image_token_id):
            prefix = start - 1
            if row[prefix] != vision_prefix_token_id:
                raise ValueError("MiniCPM TVI must precede the native image prefix")
            ids_chunks.extend([row[cursor:prefix], row.new_tensor([tvi_token_id])])
            embed_chunks.extend([embeddings[cursor:prefix], tvi_embeds[block_cursor : block_cursor + 1].to(embeddings)])
            cursor = prefix
            block_cursor += 1
        ids_rows.append(torch.cat([*ids_chunks, row[cursor:]]))
        embed_rows.append(torch.cat([*embed_chunks, embeddings[cursor:]]))
    if block_cursor != len(blocks) or block_cursor != len(tvi_embeds):
        raise ValueError("MiniCPM image spans, blocks and TVI embeddings must align")
    padded_ids, padded_mask = _pad_rows(ids_rows, tvi_token_id)
    return padded_ids, pad_sequence(embed_rows, batch_first=True), padded_mask


def pool_minicpm_vlm_inputs(inputs, *, image_token_id, max_visual_tokens, original_visual_token_counts, pad_token_id):
    rows, label_rows, target_counts = [], [], []
    labels = inputs.get("labels")
    cursor_image = 0
    for index, (ids, mask) in enumerate(zip(inputs["input_ids"], inputs["attention_mask"], strict=True)):
        row = ids[mask.bool()]
        row_labels = None if labels is None else labels[index][mask.bool()]
        chunks, label_chunks, cursor = [], [], 0
        for start, end in image_token_spans(row, image_token_id):
            if end - start != original_visual_token_counts[cursor_image]:
                raise ValueError("MiniCPM processor image span does not match native feature count")
            target = min(end - start, max_visual_tokens)
            chunks.extend([row[cursor:start], row.new_full((target,), image_token_id)])
            if labels is not None:
                label_chunks.extend([row_labels[cursor:start], row_labels[start : start + target]])
            target_counts.append(target)
            cursor, cursor_image = end, cursor_image + 1
        rows.append(torch.cat([*chunks, row[cursor:]]))
        if labels is not None:
            label_rows.append(torch.cat([*label_chunks, row_labels[cursor:]]))
    if cursor_image != len(original_visual_token_counts):
        raise ValueError("MiniCPM image feature count does not match processor spans")
    output = dict(inputs)
    output["input_ids"], output["attention_mask"] = _pad_rows(rows, pad_token_id)
    if labels is not None:
        output["labels"] = pad_sequence(label_rows, batch_first=True, padding_value=-100)
    return output, target_counts


def pool_minicpm_visual_tokens_to_count(tokens, *, target_tokens, tgt_size):
    height, width = (int(value) for value in tgt_size)
    if height * width != len(tokens) or target_tokens <= 0:
        raise ValueError(
            f"MiniCPM visual grid {height}x{width} does not match {len(tokens)} tokens or budget is invalid"
        )
    if len(tokens) == target_tokens:
        return tokens
    side = math.isqrt(target_tokens)
    if side * side == target_tokens:
        grid = tokens.view(height, width, -1).permute(2, 0, 1).unsqueeze(0).float()
        return F.adaptive_avg_pool2d(grid, (side, side)).squeeze(0).permute(1, 2, 0).flatten(0, 1).to(tokens.dtype)
    pooled = F.adaptive_avg_pool1d(tokens.T.unsqueeze(0).float(), target_tokens)
    return pooled.squeeze(0).T.to(tokens.dtype)


def _cached_tokens_for_block(block, *, device, dtype):
    if block.get("is_long_memory", False):
        tokens = block["sample"]["long_memory_tokens"][block["long_memory_index"]]
    else:
        tokens = block["sample"]["history_cached_embeds"][block["cached_history_index"]]
    return torch.as_tensor(tokens, device=device, dtype=dtype)


class _MiniCPM_VL_Interface(nn.Module):
    """Wrapper around MiniCPM-V-4.6 with the StarVLA VLM interface."""

    def __init__(self, config: Optional[dict] = None, **_kwargs: Any) -> None:
        super().__init__()
        minicpm_config = config.framework.get("qwenvl", {})
        model_id = minicpm_config.get("base_vlm", "openbmb/MiniCPM-V-4.6")
        attn_implementation = minicpm_config.get("attn_implementation", "sdpa")
        self.downsample_mode = minicpm_config.get("downsample_mode", "4x")
        self.max_slice_nums = int(minicpm_config.get("max_slice_nums", 36))
        self.use_image_id = bool(minicpm_config.get("use_image_id", False))
        self.enable_thinking = bool(minicpm_config.get("enable_thinking", False))
        self.max_text_tokens = int(minicpm_config.max_text_tokens)
        if self.downsample_mode not in {"4x", "16x"}:
            raise ValueError(f"Unsupported MiniCPM downsample_mode: {self.downsample_mode}")

        model_kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "trust_remote_code": True}
        if attn_implementation != "eager":
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.processor.downsample_mode = self.downsample_mode
        self.processor.max_slice_nums = self.max_slice_nums
        self.processor.tokenizer.padding_side = "left"

        self.model.config.hidden_size = int(self.model.config.text_config.hidden_size)

        self.config = config
        self.IMAGE_TOKEN_INDEX = int(self.model.config.image_token_id)
        self.vision_prefix_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_start_token)
        self.image_slot_suffix_ids = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_end_token)]
        self.image_slot_suffix_ids += self.processor.tokenizer.encode("\n", add_special_tokens=False)
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        tokenizer = self.processor.tokenizer
        placeholder = str(minicpm_config.get("action_placeholder_token", "◆"))
        action_start = str(minicpm_config.get("action_start_token", "▷"))
        action_end = str(minicpm_config.get("action_end_token", "◯"))
        self.action_placeholder_token, self.action_placeholder_token_id = validate_single_token(
            tokenizer, placeholder, role="placeholder"
        )
        self.action_start_token, self.action_start_token_id = validate_single_token(
            tokenizer, action_start, role="start"
        )
        self.action_end_token, self.action_end_token_id = validate_single_token(tokenizer, action_end, role="end")
        self.tvi_dim = get_tvi_input_dim(str(config.framework.navvla.tvi_mode))
        self.hidden_size = int(self.model.config.hidden_size)
        self.action_placeholder_count = int(
            config.framework.navvla.get("action_placeholder_count")
            or config.framework.action_model.action_dim * config.framework.action_model.action_horizon
        )

    def build_action_placeholder_suffix(self, num_placeholders: int) -> str:
        placeholders = self.action_placeholder_token * int(num_placeholders)
        return f"{self.action_start_token}{placeholders}{self.action_end_token}"

    def build_qwenvl_inputs(self, images, instructions, action_suffixes):
        messages = navigation_messages(
            self.processor.tokenizer,
            images,
            instructions,
            action_suffixes,
            max_text_tokens=self.max_text_tokens,
        )
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            downsample_mode=self.downsample_mode,
            max_slice_nums=self.max_slice_nums,
            use_image_id=self.use_image_id,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        ).to(self.model.device)

    def generate(self, **kwargs: Any) -> Any:
        downsample_mode = kwargs.pop("downsample_mode", self.downsample_mode)
        return self.model.generate(**kwargs, downsample_mode=downsample_mode)

    def prepare_samples(self, samples):
        if self.training and any("history_cached_embeds" in s or "long_memory_source_tokens" in s for s in samples):
            if any(
                p.requires_grad
                for module in (self.model.model.vision_tower, self.model.model.merger)
                for p in module.parameters()
            ):
                raise ValueError("Cached visual inputs require a frozen vision tower and merger")

    def _visual_token_budgets(self) -> tuple[int, int, int]:
        nav_cfg = self.config.framework.navvla
        return (
            int(nav_cfg.get("history_visual_tokens", 4)),
            int(nav_cfg.get("long_memory_visual_tokens", 128)),
            int(nav_cfg.get("current_visual_tokens", 64)),
        )

    def _image_features(self, inputs):
        features = self.model.get_image_features(
            pixel_values=inputs["pixel_values"],
            target_sizes=inputs["target_sizes"],
            downsample_mode=self.downsample_mode,
        ).pooler_output
        sizes = inputs["target_sizes"] // (2 if self.downsample_mode == "4x" else 4)
        if len(features) != len(sizes):
            raise ValueError("MiniCPM native features must contain one tensor per image grid")
        return features, sizes

    def forward(self, **batch):
        """VLM training shares the navigation current-image token budget."""
        device, dtype = _model_device_dtype(self.model)
        inputs = {key: value.to(device) for key, value in batch.items()}
        features, sizes = self._image_features(inputs)
        pooled, counts = pool_minicpm_vlm_inputs(
            inputs,
            image_token_id=self.IMAGE_TOKEN_INDEX,
            max_visual_tokens=self._visual_token_budgets()[2],
            original_visual_token_counts=[len(value) for value in features],
            pad_token_id=self.pad_token_id,
        )
        image_embeds = torch.cat(
            [
                pool_minicpm_visual_tokens_to_count(value.to(dtype), target_tokens=count, tgt_size=size)
                for value, count, size in zip(features, counts, sizes, strict=True)
            ]
        )
        ids = pooled["input_ids"]
        embeds = scatter_image_embeddings(
            self.model.get_input_embeddings()(ids), ids, image_embeds, self.IMAGE_TOKEN_INDEX
        )
        return self.model(
            inputs_embeds=embeds,
            attention_mask=pooled["attention_mask"],
            labels=pooled["labels"],
            use_cache=False,
            return_dict=True,
            downsample_mode=self.downsample_mode,
        )

    def _build_minicpm_inputs(self, samples, *, history_shuffle_probability=0.0):
        batch_images, instructions, suffixes, counts, blocks = [], [], [], [], []
        suffix = self.build_action_placeholder_suffix(self.action_placeholder_count)
        for sample in samples:
            images, sample_blocks = build_navvla_cached_visual_sequence(
                sample,
                required_cameras=sample["metadata"]["required_cameras"],
                history_shuffle_probability=history_shuffle_probability,
                tvi_dim=self.tvi_dim,
            )
            batch_images.append([rgb_image(image) for image in images])
            instructions.append(sample["instruction_text"])
            suffixes.append(suffix)
            counts.append(len(sample_blocks))
            blocks.extend(sample_blocks)
        inputs = self.build_qwenvl_inputs(batch_images, instructions, suffixes)
        budgets = dict(
            zip(
                ("history_visual_tokens", "long_memory_visual_tokens", "current_visual_tokens"),
                self._visual_token_budgets(),
            )
        )
        return _navigation_spans(
            inputs,
            blocks,
            counts,
            image_token_id=self.IMAGE_TOKEN_INDEX,
            prefix_id=self.vision_prefix_token_id,
            suffix_ids=self.image_slot_suffix_ids,
            budgets=budgets,
            pad_token_id=self.pad_token_id,
        ), blocks

    @torch.inference_mode()
    def encode_history_images(self, images):
        if not images:
            return []
        inputs = self.build_qwenvl_inputs([[rgb_image(image) for image in images]], [""], [""])
        features, sizes = self._image_features(inputs)
        if len(features) != len(images):
            raise ValueError("MiniCPM history input must produce exactly one visual grid per image")
        return [
            {
                "tokens": pool_minicpm_visual_tokens_to_count(
                    value,
                    target_tokens=self._visual_token_budgets()[0],
                    tgt_size=size,
                )
                .to(torch.float16)
                .cpu()
                .numpy()
            }
            for value, size in zip(features, sizes, strict=True)
        ]

    def _fuse_image_token_embeddings(self, inputs, blocks, *, capture_online_current_cache=False):
        device, dtype = _model_device_dtype(self.model)
        budgets = dict(
            zip(
                ("history_visual_tokens", "long_memory_visual_tokens", "current_visual_tokens"),
                self._visual_token_budgets(),
            )
        )
        online_count = sum(not block["is_cached_history"] for block in blocks)
        features, sizes = self._image_features(inputs) if online_count else ([], [])
        if len(features) != online_count:
            raise ValueError("MiniCPM online blocks must match native image feature groups")
        online = iter(zip(features, sizes, strict=True))
        fused, records = [], []
        for block in blocks:
            target = target_visual_tokens_for_block(block, **budgets)
            if block["is_cached_history"]:
                tokens = _cached_tokens_for_block(block, device=device, dtype=dtype)
                if tokens.shape != (target, self.hidden_size):
                    raise ValueError("MiniCPM cached token shape does not match the configured budget/hidden size")
            else:
                value, size = next(online)
                tokens = pool_minicpm_visual_tokens_to_count(value.to(dtype), target_tokens=target, tgt_size=size)
                if capture_online_current_cache and not block["is_history"]:
                    cached = pool_minicpm_visual_tokens_to_count(
                        value, target_tokens=budgets["history_visual_tokens"], tgt_size=size
                    )
                    records.append(
                        {
                            "camera_name": block["camera_name"],
                            "frame_index": block["frame_index"],
                            "tokens": cached.detach().to(torch.float16).cpu().numpy(),
                        }
                    )
            fused.append(tokens)
        return torch.cat(fused), records

    def _forward_backbone(
        self,
        minicpm_inputs: dict[str, Any],
        blocks: list[dict[str, Any]],
        *,
        tvi_embedding,
        capture_online_current_cache: bool = False,
        tvi_mask_probability: float = 0.0,
    ) -> tuple[Any, list[dict[str, Any]]]:
        model = self.model
        input_ids = minicpm_inputs["input_ids"]
        attention_mask = minicpm_inputs.get("attention_mask")
        image_token_id = self.IMAGE_TOKEN_INDEX
        inputs_embeds = model.get_input_embeddings()(input_ids)
        fused_image_embeds, online_current_cache_records = self._fuse_image_token_embeddings(
            minicpm_inputs,
            blocks,
            capture_online_current_cache=capture_online_current_cache,
        )
        inputs_embeds = scatter_image_embeddings(inputs_embeds, input_ids, fused_image_embeds, image_token_id)

        if blocks:
            tvi_array = as_numpy_tvi(np.stack([block["tvi"] for block in blocks]), tvi_dim=self.tvi_dim)
            tvi_values = torch.as_tensor(
                tvi_array,
                device=inputs_embeds.device,
                dtype=torch.float32,
            )
            tvi_embeds = tvi_embedding(tvi_values)
            tvi_embeds = mask_history_tvi_embeddings(
                tvi_embedding,
                tvi_embeds,
                blocks,
                probability=tvi_mask_probability,
            )
            input_ids, inputs_embeds, attention_mask = insert_navvla_tvi_prefix_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                blocks=blocks,
                tvi_embeds=tvi_embeds,
                vision_prefix_token_id=self.vision_prefix_token_id,
                image_token_id=image_token_id,
                tvi_token_id=self.pad_token_id,
            )
            minicpm_inputs["input_ids"] = input_ids
            if attention_mask is not None:
                minicpm_inputs["attention_mask"] = attention_mask
        return model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            downsample_mode=self.downsample_mode,
            use_cache=False,
        ), online_current_cache_records

    def encode_context(
        self,
        samples: list[dict[str, Any]],
        *,
        tvi_embedding,
        capture_online_current_cache: bool = False,
        history_shuffle_probability: float = 0.0,
        tvi_mask_probability: float = 0.0,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        minicpm_inputs, blocks = self._build_minicpm_inputs(
            samples,
            history_shuffle_probability=history_shuffle_probability,
        )
        outputs, online_current_cache_records = self._forward_backbone(
            minicpm_inputs,
            blocks,
            capture_online_current_cache=capture_online_current_cache,
            tvi_mask_probability=tvi_mask_probability,
            tvi_embedding=tvi_embedding,
        )
        last_hidden = outputs.last_hidden_state
        action_hidden = gather_action_queries(
            last_hidden,
            minicpm_inputs["input_ids"],
            token_id=self.action_placeholder_token_id,
            num_placeholders=self.action_placeholder_count,
        )
        return action_hidden, online_current_cache_records

    def freeze_visual_eval(self):
        if not any(p.requires_grad for p in self.model.model.vision_tower.parameters()):
            self.model.model.vision_tower.eval()
