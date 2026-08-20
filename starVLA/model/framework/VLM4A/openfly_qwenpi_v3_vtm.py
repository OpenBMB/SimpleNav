# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""OpenFly-specialized QwenPI_v3 with pre-LLM visual token merge."""

from dataclasses import dataclass, field
import math
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import create_causal_mask

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.vlm.QWen3 import IMAGE_TOKEN_INDEX
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


def _image_token_spans(input_ids: torch.Tensor, image_token_id: int) -> list[tuple[int, int]]:
    positions = torch.nonzero(input_ids == image_token_id, as_tuple=False).flatten().tolist()
    if not positions:
        return []

    spans: list[tuple[int, int]] = []
    start = positions[0]
    prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        spans.append((start, prev + 1))
        start = pos
        prev = pos
    spans.append((start, prev + 1))
    return spans


def _merge_single_openfly_hidden(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    image_token_id: int,
    attention_mask: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if attention_mask is not None:
        active = attention_mask.to(dtype=torch.bool)
        hidden = hidden[active]
        input_ids = input_ids[active]

    spans = _image_token_spans(input_ids, image_token_id)
    if len(spans) <= 1:
        return hidden

    chunks: list[torch.Tensor] = []
    cursor = 0
    current_span = spans[-1]
    for span_idx, (start, end) in enumerate(spans):
        if start > cursor:
            chunks.append(hidden[cursor:start])
        image_hidden = hidden[start:end]
        if (start, end) == current_span:
            chunks.append(image_hidden)
        else:
            chunks.append(_pool_history_image_hidden_by_half_grid(image_hidden, image_grid_thw, span_idx))
        cursor = end
    if cursor < hidden.shape[0]:
        chunks.append(hidden[cursor:])
    return torch.cat(chunks, dim=0)


def _pool_history_image_hidden_by_half_grid(
    image_hidden: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
    span_idx: int,
) -> torch.Tensor:
    if image_grid_thw is None or span_idx >= image_grid_thw.shape[0]:
        return image_hidden.mean(dim=0, keepdim=True)

    grid = image_grid_thw[span_idx].tolist()
    if len(grid) != 3:
        return image_hidden.mean(dim=0, keepdim=True)
    temporal, height, width = [int(v) for v in grid]
    token_h = max(1, height // 2)
    token_w = max(1, width // 2)
    if temporal != 1 or token_h * token_w != image_hidden.shape[0]:
        return image_hidden.mean(dim=0, keepdim=True)

    target_h = max(1, math.ceil(token_h / 2))
    target_w = max(1, math.ceil(token_w / 2))
    pooled = F.adaptive_avg_pool2d(
        image_hidden.view(1, token_h, token_w, image_hidden.shape[-1]).permute(0, 3, 1, 2).float(),
        (target_h, target_w),
    )
    return pooled.to(dtype=image_hidden.dtype).permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)


def merge_openfly_visual_tokens_for_action_hidden(
    vl_embs_list: List[torch.Tensor],
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    image_token_id: int = IMAGE_TOKEN_INDEX,
) -> List[torch.Tensor]:
    """Pool OpenFly history-image tokens by halving each spatial token grid.

    The current OpenFly dataloader orders images as history frames followed by
    current frame. The last image span is kept at full token length. Earlier
    image spans are pooled to roughly one quarter of their original tokens by
    halving token-grid height and width. Existing text padding is removed
    before pooling, and the merged batch is padded only to the longest merged
    sample because the current action head consumes fixed-size tensors.
    """

    if not vl_embs_list:
        return vl_embs_list

    image_grids_by_row = _split_image_grids_by_row(image_grid_thw, input_ids, image_token_id)
    merged_layers: list[torch.Tensor] = []
    for layer_hidden in vl_embs_list:
        samples = [
            _merge_single_openfly_hidden(
                layer_hidden[row],
                input_ids[row],
                image_token_id,
                attention_mask[row] if attention_mask is not None else None,
                image_grids_by_row[row],
            )
            for row in range(layer_hidden.shape[0])
        ]
        max_len = max(sample.shape[0] for sample in samples)
        if all(sample.shape[0] == max_len for sample in samples):
            merged_layers.append(torch.stack(samples, dim=0))
            continue

        padded = layer_hidden.new_zeros((len(samples), max_len, layer_hidden.shape[-1]))
        for row, sample in enumerate(samples):
            padded[row, : sample.shape[0]] = sample
        merged_layers.append(padded)
    return merged_layers


def _split_image_grids_by_row(
    image_grid_thw: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    image_token_id: int,
) -> list[Optional[torch.Tensor]]:
    if image_grid_thw is None:
        return [None for _ in range(input_ids.shape[0])]
    if image_grid_thw.ndim == 3:
        return [image_grid_thw[row] for row in range(input_ids.shape[0])]
    if image_grid_thw.ndim != 2:
        return [None for _ in range(input_ids.shape[0])]

    grids_by_row: list[Optional[torch.Tensor]] = []
    cursor = 0
    for row in range(input_ids.shape[0]):
        span_count = len(_image_token_spans(input_ids[row], image_token_id))
        end = cursor + span_count
        if end <= image_grid_thw.shape[0]:
            grids_by_row.append(image_grid_thw[cursor:end])
        else:
            grids_by_row.append(None)
        cursor = end
    return grids_by_row


def _pool_history_visual_sequence_by_half_grid(
    visual_tokens: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_index: int,
) -> torch.Tensor:
    grid = image_grid_thw[image_index].tolist()
    if len(grid) != 3:
        return visual_tokens
    temporal, height, width = [int(v) for v in grid]
    token_h = max(1, height // 2)
    token_w = max(1, width // 2)
    if temporal != 1 or token_h * token_w != visual_tokens.shape[0]:
        return visual_tokens

    target_h = max(1, math.ceil(token_h / 2))
    target_w = max(1, math.ceil(token_w / 2))
    pooled = F.adaptive_avg_pool2d(
        visual_tokens.view(1, token_h, token_w, visual_tokens.shape[-1]).permute(0, 3, 1, 2).float(),
        (target_h, target_w),
    )
    return pooled.to(dtype=visual_tokens.dtype).permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)


def _pool_openfly_history_visual_features(
    visual_features,
    image_grid_thw: torch.Tensor,
    current_image_indices: Optional[set[int]] = None,
    keep_current: bool = True,
):
    pooled = []
    current_image_indices = current_image_indices or {int(image_grid_thw.shape[0]) - 1}
    for image_index, tokens in enumerate(visual_features):
        if keep_current and image_index in current_image_indices:
            pooled.append(tokens)
        else:
            pooled.append(_pool_history_visual_sequence_by_half_grid(tokens, image_grid_thw, image_index))
    return pooled


def _replace_image_token_spans_for_openfly_vtm(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    keep_current: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    image_cursor = 0
    for row in range(input_ids.shape[0]):
        spans = _image_token_spans(input_ids[row], image_token_id)
        chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        cursor = 0
        current_local_idx = len(spans) - 1
        for local_idx, (start, end) in enumerate(spans):
            if start > cursor:
                chunks.append(input_ids[row, cursor:start])
                if attention_mask is not None:
                    mask_chunks.append(attention_mask[row, cursor:start])
            original_len = end - start
            if keep_current and local_idx == current_local_idx:
                target_len = original_len
            else:
                # image_grid_thw is already the pooled grid here; Qwen uses merge_size=2.
                target_len = int(image_grid_thw[image_cursor].prod().item() // 4)
            chunks.append(input_ids.new_full((target_len,), image_token_id))
            if attention_mask is not None:
                mask_chunks.append(attention_mask.new_ones((target_len,)))
            cursor = end
            image_cursor += 1
        if cursor < input_ids.shape[1]:
            chunks.append(input_ids[row, cursor:])
            if attention_mask is not None:
                mask_chunks.append(attention_mask[row, cursor:])
        rows.append(torch.cat(chunks, dim=0))
        if attention_mask is not None:
            mask_chunks_cat = torch.cat(mask_chunks, dim=0)
            masks.append(mask_chunks_cat)

    max_len = max(row.shape[0] for row in rows)
    pad_token_id = 0
    if attention_mask is not None:
        active_ids = input_ids[attention_mask.to(dtype=torch.bool)]
        if active_ids.numel() < input_ids.numel():
            pad_candidates = input_ids[~attention_mask.to(dtype=torch.bool)]
            if pad_candidates.numel() > 0:
                pad_token_id = int(pad_candidates[0].item())
    padded_ids = input_ids.new_full((len(rows), max_len), pad_token_id)
    padded_mask = attention_mask.new_zeros((len(rows), max_len)) if attention_mask is not None else None
    for row_idx, row_ids in enumerate(rows):
        padded_ids[row_idx, : row_ids.shape[0]] = row_ids
        if padded_mask is not None:
            padded_mask[row_idx, : masks[row_idx].shape[0]] = masks[row_idx]
    return padded_ids, padded_mask


def _current_image_indices_from_input_ids(input_ids: torch.Tensor, image_token_id: int) -> set[int]:
    current_indices: set[int] = set()
    image_cursor = 0
    for row in range(input_ids.shape[0]):
        spans = _image_token_spans(input_ids[row], image_token_id)
        if spans:
            current_indices.add(image_cursor + len(spans) - 1)
        image_cursor += len(spans)
    return current_indices


def _apply_openfly_pre_llm_vtm_to_qwen_inputs(
    qwen_inputs,
    image_token_id: int = IMAGE_TOKEN_INDEX,
):
    image_grid_thw = qwen_inputs.get("image_grid_thw", None)
    if image_grid_thw is None or int(image_grid_thw.shape[0]) <= 1:
        return qwen_inputs

    current_image_indices = _current_image_indices_from_input_ids(qwen_inputs["input_ids"], image_token_id)
    pooled_image_grid_thw = image_grid_thw.clone()
    for image_index in range(int(image_grid_thw.shape[0])):
        if image_index in current_image_indices:
            continue
        temporal, height, width = [int(v) for v in image_grid_thw[image_index].tolist()]
        token_h = max(1, height // 2)
        token_w = max(1, width // 2)
        target_h = max(1, math.ceil(token_h / 2))
        target_w = max(1, math.ceil(token_w / 2))
        pooled_image_grid_thw[image_index, 0] = temporal
        pooled_image_grid_thw[image_index, 1] = target_h * 2
        pooled_image_grid_thw[image_index, 2] = target_w * 2
    input_ids, attention_mask = _replace_image_token_spans_for_openfly_vtm(
        qwen_inputs["input_ids"],
        qwen_inputs.get("attention_mask", None),
        pooled_image_grid_thw,
        image_token_id,
    )
    qwen_inputs["_openfly_original_image_grid_thw"] = image_grid_thw
    qwen_inputs["_openfly_current_image_indices"] = current_image_indices
    qwen_inputs["input_ids"] = input_ids
    qwen_inputs["image_grid_thw"] = pooled_image_grid_thw
    if attention_mask is not None:
        qwen_inputs["attention_mask"] = attention_mask
    qwen_inputs["_openfly_pre_llm_vtm_applied"] = True
    return qwen_inputs


def _prepare_openfly_pre_llm_vtm_qwen_inputs(
    qwen_inputs,
    image_token_id: int = IMAGE_TOKEN_INDEX,
):
    if qwen_inputs.get("pixel_values", None) is None or qwen_inputs.get("image_grid_thw", None) is None:
        return qwen_inputs
    return _apply_openfly_pre_llm_vtm_to_qwen_inputs(qwen_inputs, image_token_id=image_token_id)


def _forward_qwen_with_openfly_pre_llm_vtm(qwen_model, qwen_inputs, *, output_hidden_states: bool, return_dict: bool):
    qwen_inputs = _prepare_openfly_pre_llm_vtm_qwen_inputs(qwen_inputs)
    model = qwen_model.model
    input_ids = qwen_inputs["input_ids"]
    attention_mask = qwen_inputs.get("attention_mask", None)
    pixel_values = qwen_inputs.get("pixel_values", None)
    image_grid_thw = qwen_inputs.get("image_grid_thw", None)
    original_image_grid_thw = qwen_inputs.get("_openfly_original_image_grid_thw", image_grid_thw)
    current_image_indices = qwen_inputs.get("_openfly_current_image_indices", None)

    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_mask = None
    deepstack_image_embeds = None
    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = model.get_image_features(pixel_values, original_image_grid_thw)
        image_embeds = _pool_openfly_history_visual_features(
            image_embeds,
            original_image_grid_thw,
            current_image_indices=current_image_indices,
        )
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if deepstack_image_embeds is not None:
            pooled_deepstack = []
            for layer_embeds in deepstack_image_embeds:
                split_sizes = (original_image_grid_thw.prod(-1) // model.visual.spatial_merge_size**2).tolist()
                layer_chunks = torch.split(layer_embeds, split_sizes)
                pooled_deepstack.append(
                    torch.cat(
                        _pool_openfly_history_visual_features(
                            layer_chunks,
                            original_image_grid_thw,
                            current_image_indices=current_image_indices,
                        ),
                        dim=0,
                    )
                )
            deepstack_image_embeds = pooled_deepstack

    if image_mask is not None:
        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = deepstack_image_embeds
    else:
        visual_pos_masks = None
        deepstack_visual_embeds = None

    attention_mask_tensor = attention_mask
    if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
        attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
        if attention_mask_tensor.dtype.is_floating_point:
            attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
            attention_mask_tensor = (1.0 - attention_mask_tensor).int()
    position_ids, rope_deltas = model.get_rope_index(
        input_ids,
        image_grid_thw,
        None,
        attention_mask=attention_mask_tensor,
    )
    model.rope_deltas = rope_deltas

    last_hidden_state, hidden_states = _run_qwen_language_model_with_hidden_states(
        model.language_model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )
    return SimpleNamespace(last_hidden_state=last_hidden_state, hidden_states=hidden_states, rope_deltas=rope_deltas)


def _run_qwen_language_model_with_hidden_states(
    language_model,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    visual_pos_masks: Optional[torch.Tensor],
    deepstack_visual_embeds: Optional[list[torch.Tensor]],
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
class OpenFlyQwenPI_v3VTMDefaultConfig:
    name: str = "openfly_qwenpi_v3_vtm"
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
            "num_vl_layers": 36,
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 4,
            "state_dim": 8,
            "action_horizon": 1,
            "repeated_diffusion_steps": 2,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                "action_dit_hidden_dim": 1024,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )
    visual_token_merge: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "stage": "pre_llm",
            "image_token_id": IMAGE_TOKEN_INDEX,
        }
    )


@FRAMEWORK_REGISTRY.register("openfly_qwenpi_v3_vtm")
class OpenFly_QwenPI_v3_VTM(baseframework):
    """QwenPI_v3 variant that compresses OpenFly history visual tokens before the LLM."""

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = merge_framework_config(OpenFlyQwenPI_v3VTMDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        num_vl_layers, llm_hidden_size = 36, self.qwen_vl_interface.model.config.hidden_size
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
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _project_vl_hidden_for_action(self, vl_embs_list: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(vl_embs_list) != len(self.project_layers):
            raise ValueError(
                f"Layer number mismatch: got {len(vl_embs_list)} VL layers, "
                f"but project_layers has {len(self.project_layers)} layers."
            )
        return [proj(vl_h) for proj, vl_h in zip(self.project_layers, vl_embs_list)]

    def _merge_vl_hidden_for_action(self, vl_embs_list: List[torch.Tensor], qwen_inputs) -> List[torch.Tensor]:
        merge_cfg = self.config.framework.get("visual_token_merge", {})
        if not merge_cfg.get("enabled", True):
            return vl_embs_list
        if merge_cfg.get("stage", "pre_llm") == "pre_llm":
            return vl_embs_list
        return merge_openfly_visual_tokens_for_action_hidden(
            vl_embs_list,
            input_ids=qwen_inputs["input_ids"],
            attention_mask=qwen_inputs.get("attention_mask", None),
            image_grid_thw=qwen_inputs.get("image_grid_thw", None),
            image_token_id=int(merge_cfg.get("image_token_id", IMAGE_TOKEN_INDEX)),
        )

    def _qwen_forward_for_action(self, qwen_inputs):
        merge_cfg = self.config.framework.get("visual_token_merge", {})
        if merge_cfg.get("enabled", True) and merge_cfg.get("stage", "pre_llm") == "pre_llm":
            return _forward_qwen_with_openfly_pre_llm_vtm(
                self.qwen_vl_interface.model,
                qwen_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        return self.qwen_vl_interface(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )
        state = None

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self._qwen_forward_for_action(qwen_inputs)
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = self.num_action_dit_layers
            vl_embs_list = list(all_hidden[-expected_layers:])
            vl_embs_list = self._merge_vl_hidden_for_action(vl_embs_list, qwen_inputs)
            vl_embs_list = self._project_vl_hidden_for_action(vl_embs_list)
            base_hidden = vl_embs_list[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]
            repeated_diffusion_steps = int(
                self.config.framework.action_model.get("repeated_diffusion_steps", 2)
            )

            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)

            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [hidden.repeat(repeated_diffusion_steps, 1, 1) for hidden in vl_embs_list]
            state_repeated = state.repeat(repeated_diffusion_steps, 1, 1) if state is not None else None
            action_loss = self.action_model(vl_embs_list_repeated, actions_target_repeated, state_repeated)
            total_loss = action_loss

        return {
            "action_loss": total_loss,
            "action_dit_loss": action_loss,
            "loss": total_loss,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )
        state = None

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self._qwen_forward_for_action(qwen_inputs)
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = self.num_action_dit_layers
            vl_embs_list = list(all_hidden[-expected_layers:])
            vl_embs_list = self._merge_vl_hidden_for_action(vl_embs_list, qwen_inputs)
            vl_embs_list = self._project_vl_hidden_for_action(vl_embs_list)
            base_hidden = vl_embs_list[-1]

        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(vl_embs_list, state)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {
            "normalized_actions": normalized_actions,
        }

    def state2str_transform(self, state: np.ndarray) -> str:
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return " ".join(map(str, discretized_state))

    def add_discretized_state_to_instruction(self, instructions: List[str], states: List[np.ndarray]) -> List[str]:
        updated_instructions = []
        for instr, state in zip(instructions, states):
            state_str = self.state2str_transform(state[0])
            updated_instructions.append(f"{instr} [STATE] {state_str} [ACTION]")
        return updated_instructions


if __name__ == "__main__":
    import argparse
    import os

    import debugpy
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/AirSim/train_files/starvla_train_openfly_qwenpi_v3_vtm.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()
    if os.getenv("DEBUG_MODE", "0") == "1":
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    model = OpenFly_QwenPI_v3_VTM(cfg)
    print(model)
