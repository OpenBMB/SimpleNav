# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2026].
# Design and Merged by [Jinhui YE / HKUST University] in [2026].

from typing import Any, Optional

import torch
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as import_error:
    raise ImportError(
        "Qwen3.5 model class is unavailable. Please install transformers >= 5.2.0 or check your transformers version."
    ) from import_error

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 248056
VIDEO_TOKEN_INDEX = 248057
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_ACTION_PLACEHOLDER_TOKEN = "<|fim_pad|>"
DEFAULT_ACTION_START_TOKEN = "<|fim_prefix|>"
DEFAULT_ACTION_END_TOKEN = "<|fim_suffix|>"

_ACTION_TOKEN_MIN = 248077  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    248077 + 2047
)  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


def _validate_single_token(tokenizer: Any, token: str, *, role: str) -> tuple[str, int]:
    token_ids = tokenizer.encode(str(token), add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Qwen3.5 action {role} token {token!r} must encode to exactly one tokenizer token, got {token_ids}"
        )
    return str(token), int(token_ids[0])


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
        enable_gradient_checkpointing = bool(
            trainer_config.get("enable_gradient_checkpointing", False)
        )
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
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
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
        self.action_placeholder_token, self.action_placeholder_token_id = _validate_single_token(
            tokenizer, placeholder, role="placeholder"
        )
        self.action_start_token, self.action_start_token_id = _validate_single_token(
            tokenizer, action_start, role="start"
        )
        self.action_end_token, self.action_end_token_id = _validate_single_token(
            tokenizer, action_end, role="end"
        )

        self.IMAGE_TOKEN_INDEX = int(self.model.config.image_token_id)

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def _truncate_text(self, text: str) -> str:
        max_tokens = int(self.config.framework.qwenvl.get("max_text_tokens", 2048))
        if max_tokens <= 0:
            raise ValueError(f"framework.qwenvl.max_text_tokens must be positive, got {max_tokens}")
        token_ids = self.processor.tokenizer.encode(str(text), add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return str(text)
        return self.processor.tokenizer.decode(
            token_ids[:max_tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

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
        suffix = (
            self.action_start_token
            + self.action_placeholder_token * int(num_placeholders)
            + self.action_end_token
        )
        token_ids = self.processor.tokenizer.encode(suffix, add_special_tokens=False)
        if token_ids.count(self.action_placeholder_token_id) != int(num_placeholders):
            raise ValueError(
                "Qwen3.5 tokenizer did not preserve the repeated action-placeholder suffix as single tokens"
            )
        return suffix

    def gather_action_placeholder_hidden_states(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        num_placeholders: int,
        action_token_id: int | None = None,
    ) -> torch.Tensor:
        token_id = int(action_token_id if action_token_id is not None else self.action_placeholder_token_id)
        mask = input_ids == token_id
        counts = mask.sum(dim=1)
        if (counts < int(num_placeholders)).any():
            insufficient = (counts < int(num_placeholders)).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"expected at least {num_placeholders} action placeholder tokens; "
                f"insufficient samples={insufficient}, counts={counts.tolist()}"
            )
        batch_size, seq_len, hidden_dim = hidden_states.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        selected = torch.where(mask, positions, torch.full_like(positions, -1)).topk(
            k=int(num_placeholders), dim=-1
        ).values.sort(dim=-1).values
        return hidden_states.gather(dim=1, index=selected.unsqueeze(-1).expand(-1, -1, hidden_dim))

    def build_qwenvl_inputs(
        self,
        images,
        instructions,
        solutions=None,
        action_suffixes=None,
        *,
        move_to_device: bool = True,
        **kwargs,
    ):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow the official Qwen3.5 multimodal format: https://huggingface.co/Qwen/Qwen3.5-4B
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        if action_suffixes is not None and len(action_suffixes) != len(instructions):
            raise ValueError("action_suffixes must have the same batch length as instructions")
        for sample_index, (imgs, instruction) in enumerate(zip(images, instructions)):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            prompt = self._truncate_text(prompt)

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            elif action_suffixes is not None:
                msg.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": str(action_suffixes[sample_index])}],
                    }
                )
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=solutions is None and action_suffixes is None,
            add_vision_id=False,
            return_dict=True,
            return_tensors="pt",
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    logger.warning(
                        "No action token found in sequence; please check action-tokenized tokenizer in "
                        "starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md"
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device) if move_to_device else batch_inputs


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3.5-4B"
    qwen_vl = _QWen3_5_VL_Interface(cfg)
    pass
