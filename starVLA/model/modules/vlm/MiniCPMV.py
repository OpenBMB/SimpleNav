# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:  # pragma: no cover
    AutoModelForImageTextToText = None
    AutoProcessor = None


IMAGE_TOKEN_INDEX = 248056
VIDEO_TOKEN_INDEX = 248057
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
DEFAULT_DOWNSAMPLE_MODE = "4x"
DEFAULT_MAX_SLICE_NUMS = 36
DEFAULT_USE_IMAGE_ID = False
DEFAULT_ENABLE_THINKING = False
DEFAULT_ACTION_PLACEHOLDER_TOKEN = "◆"
DEFAULT_ACTION_START_TOKEN = "▷"
DEFAULT_ACTION_END_TOKEN = "◯"
DEFAULT_MAX_TEXT_TOKENS = 2048
IGNORE_INDEX = -100


def _load_custom_token_map(model_id: str) -> Dict[str, int]:
    map_path = os.path.join(model_id, "added_custom_token_id_map.json")
    if not os.path.isfile(map_path):
        return {}
    with open(map_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def configure_minicpm_processor(processor: Any, config: Any) -> None:
    minicpm_config = config.framework.get("qwenvl", {})
    processor.downsample_mode = minicpm_config.get("downsample_mode", DEFAULT_DOWNSAMPLE_MODE)
    processor.max_slice_nums = int(minicpm_config.get("max_slice_nums", DEFAULT_MAX_SLICE_NUMS))


def _resolve_max_text_tokens(minicpm_config: Any) -> int:
    max_text_tokens = minicpm_config.get("max_text_tokens", None)
    if max_text_tokens is None:
        max_text_tokens = minicpm_config.get("max_seq_len", DEFAULT_MAX_TEXT_TOKENS)
    max_text_tokens = int(max_text_tokens)
    if max_text_tokens <= 0:
        raise ValueError(f"MiniCPM max text tokens must be positive, got {max_text_tokens}")
    return max_text_tokens


def _truncate_text_tokens(tokenizer: Any, text: str, *, max_text_tokens: int) -> str:
    token_ids = tokenizer.encode(str(text), add_special_tokens=False)
    if len(token_ids) <= int(max_text_tokens):
        return str(text)
    return tokenizer.decode(token_ids[: int(max_text_tokens)], skip_special_tokens=False)


def _validate_single_token(tokenizer: Any, token: str, *, role: str) -> tuple[str, int]:
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if len(encoded) != 1:
        raise ValueError(f"MiniCPM action {role} token {token!r} must encode to one token, got ids={encoded}")
    return token, int(encoded[0])


def _resolve_vision_token_id(
    processor: Any,
    *,
    token_attr: str,
    default_token: str,
    expected_id: int,
    token_name: str,
) -> int:
    tokenizer = processor.tokenizer
    token = getattr(tokenizer, token_attr, None) or default_token
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"failed to resolve MiniCPM {token_name} token id for {token!r}")
    if int(token_id) != int(expected_id):
        return int(token_id)
    return int(expected_id)


class _MiniCPM_VL_Interface(nn.Module):
    """Wrapper around MiniCPM-V-4.6 with the StarVLA VLM interface."""

    def __init__(self, config: Optional[dict] = None, **_kwargs: Any) -> None:
        super().__init__()
        if AutoModelForImageTextToText is None or AutoProcessor is None:
            raise ImportError("transformers with AutoModelForImageTextToText is required for MiniCPM-V-4.6")

        minicpm_config = config.framework.get("qwenvl", {})
        model_id = minicpm_config.get("base_vlm", "openbmb/MiniCPM-V-4.6")
        attn_implementation = minicpm_config.get("attn_implementation", "sdpa")
        self.downsample_mode = minicpm_config.get("downsample_mode", DEFAULT_DOWNSAMPLE_MODE)
        self.max_slice_nums = int(minicpm_config.get("max_slice_nums", DEFAULT_MAX_SLICE_NUMS))
        self.use_image_id = bool(minicpm_config.get("use_image_id", DEFAULT_USE_IMAGE_ID))
        self.enable_thinking = bool(minicpm_config.get("enable_thinking", DEFAULT_ENABLE_THINKING))
        self.max_text_tokens = _resolve_max_text_tokens(minicpm_config)

        model_kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "trust_remote_code": True}
        if attn_implementation != "eager":
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        configure_minicpm_processor(self.processor, config)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        text_config = getattr(self.model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "hidden_size"):
            self.model.config.hidden_size = int(text_config.hidden_size)

        self.config = config
        self.custom_token_map = _load_custom_token_map(str(model_id))
        self.IMAGE_TOKEN_INDEX = _resolve_vision_token_id(
            self.processor,
            token_attr="image_token",
            default_token=DEFAULT_IMAGE_TOKEN,
            expected_id=IMAGE_TOKEN_INDEX,
            token_name="image",
        )
        self.VIDEO_TOKEN_INDEX = _resolve_vision_token_id(
            self.processor,
            token_attr="video_token",
            default_token=DEFAULT_VIDEO_TOKEN,
            expected_id=VIDEO_TOKEN_INDEX,
            token_name="video",
        )
        tokenizer = self.processor.tokenizer
        placeholder = str(minicpm_config.get("action_placeholder_token", DEFAULT_ACTION_PLACEHOLDER_TOKEN))
        action_start = str(minicpm_config.get("action_start_token", DEFAULT_ACTION_START_TOKEN))
        action_end = str(minicpm_config.get("action_end_token", DEFAULT_ACTION_END_TOKEN))
        self.action_placeholder_token, self.action_placeholder_token_id = _validate_single_token(
            tokenizer, placeholder, role="placeholder"
        )
        self.action_start_token, self.action_start_token_id = _validate_single_token(tokenizer, action_start, role="start")
        self.action_end_token, self.action_end_token_id = _validate_single_token(tokenizer, action_end, role="end")

    def build_action_placeholder_suffix(self, num_placeholders: int) -> str:
        placeholders = self.action_placeholder_token * int(num_placeholders)
        return f"{self.action_start_token}{placeholders}{self.action_end_token}"

    def gather_action_placeholder_hidden_states(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        num_placeholders: int,
        action_token_id: int | None = None,
    ) -> torch.Tensor:
        token_id = int(action_token_id if action_token_id is not None else self.action_placeholder_token_id)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        mask = input_ids == token_id
        counts = mask.sum(dim=1)
        if (counts < int(num_placeholders)).any():
            insufficient = (counts < int(num_placeholders)).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"expected at least {num_placeholders} action placeholder tokens; "
                f"insufficient samples={insufficient}, counts={counts.tolist()}"
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        masked = torch.where(mask, positions, torch.full_like(positions, -1))
        selected = masked.topk(k=int(num_placeholders), dim=-1).values.sort(dim=-1).values
        gather_index = selected.unsqueeze(-1).expand(-1, -1, hidden_dim)
        return hidden_states.gather(dim=1, index=gather_index)

    def build_qwenvl_inputs(
        self,
        images: list[list[Any]],
        instructions: list[str],
        solutions: list[str] | None = None,
        action_suffixes: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        if len(images) != len(instructions):
            raise ValueError("images and instructions must have the same batch length")
        if action_suffixes is not None and len(action_suffixes) != len(instructions):
            raise ValueError("action_suffixes must have the same batch length as instructions")

        messages: list[list[dict[str, Any]]] = []
        tokenizer = self.processor.tokenizer
        for index, (sample_images, instruction) in enumerate(zip(images, instructions)):
            content = [{"type": "image", "image": image} for image in sample_images]
            content.append(
                {
                    "type": "text",
                    "text": _truncate_text_tokens(
                        tokenizer,
                        str(instruction),
                        max_text_tokens=self.max_text_tokens,
                    ),
                }
            )
            message = [{"role": "user", "content": content}]
            if solutions is not None:
                message.append({"role": "assistant", "content": [{"type": "text", "text": str(solutions[index])}]})
            elif action_suffixes is not None:
                message.append({"role": "assistant", "content": [{"type": "text", "text": str(action_suffixes[index])}]})
            messages.append(message)

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=solutions is None and action_suffixes is None,
            return_dict=True,
            return_tensors="pt",
            downsample_mode=self.downsample_mode,
            max_slice_nums=self.max_slice_nums,
            use_image_id=kwargs.pop("use_image_id", self.use_image_id),
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        )
        if solutions is not None:
            labels = batch_inputs["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels
        return batch_inputs.to(self.model.device)

    def forward(self, **kwargs: Any) -> Any:
        downsample_mode = kwargs.pop("downsample_mode", self.downsample_mode)
        return self.model(**kwargs, downsample_mode=downsample_mode)

    def generate(self, **kwargs: Any) -> Any:
        downsample_mode = kwargs.pop("downsample_mode", self.downsample_mode)
        return self.model.generate(**kwargs, downsample_mode=downsample_mode)
