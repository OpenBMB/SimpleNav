"""SimpleNav: shared history context, pluggable backbone and action head."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from starVLA.model.framework import FRAMEWORK_REGISTRY
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model
from starVLA.model.modules.input_context import (
    HistoryAugmentationConfig,
    build_navvla_instruction,
    history_augmentation_probabilities,
    samples_from_collated_batch,
)
from starVLA.model.modules.long_memory import (
    LongMemoryTokenAggregator,
    attach_navvla_long_memory_tokens,
    compute_navvla_online_long_memory_updates,
)
from starVLA.model.modules.tvi import TIME_YAW_TVI_MODE, NavVLATVIEmbedding, get_tvi_input_dim
from starVLA.model.modules.vlm import get_vlm_model


@dataclass
class SimpleNavConfig:
    name: str = "simplenav"
    qwenvl: dict = field(
        default_factory=lambda: {
            "type": "qwen35",
            "base_vlm": "Qwen/Qwen3.5-4B",
            "attn_implementation": "flash_attention_2",
            "max_text_tokens": 2048,
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
            "visual_token_profile": "qwen3_5_4b_postmerge_pool4_256_mmap",
            "visual_cache_stage": "vit_postmerge_pool4",
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


@FRAMEWORK_REGISTRY.register("simplenav")
class SimpleNav(baseframework):
    def __init__(self, config: Optional[dict] = None, **_kwargs: Any) -> None:
        super().__init__()
        self.config = merge_framework_config(SimpleNavConfig, config)
        nav_cfg = self.config.framework.navvla
        self.tvi_mode = str(nav_cfg.get("tvi_mode", TIME_YAW_TVI_MODE))
        self.tvi_dim = get_tvi_input_dim(self.tvi_mode)
        self.backbone = get_vlm_model(config=self.config)
        hidden_size = self.backbone.hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = hidden_size
        self.config.framework.action_model.num_target_vision_tokens = int(
            self.config.framework.action_model.action_horizon
        )
        self.action_model = get_action_model(config=self.config)
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
            self.action_dim * self.action_horizon if configured_placeholders is None else int(configured_placeholders)
        )
        if self.action_placeholder_count <= 0:
            raise ValueError("framework.navvla.action_placeholder_count must be positive")
        self.hidden_size = hidden_size

    def train(self, mode=True):
        super().train(mode)
        self.backbone.freeze_visual_eval()
        return self

    def forward_vlm(self, batch):
        return {"vlm_loss": self.backbone(**batch).loss}

    def _samples(self, examples):
        keys = tuple(
            f"{prefix}_{suffix}"
            for prefix in ("history_cached", "long_memory_source", "online_long_memory_update")
            for suffix in ("grid_thw", "cache_stage", "encoder_ckpt", "storage_encoding")
        )
        samples = [dict(sample) for sample in samples_from_collated_batch(examples, extra_keys=keys)]
        for sample in samples:
            sample["instruction_text"] = build_navvla_instruction(
                sample,
                use_platform_text=bool(self.config.framework.navvla.use_platform_text),
                prompt_template=self.config.framework.navvla.get("prompt_template", "{instruction}"),
            )
        return samples

    def _memory_kwargs(self):
        parameter = next(self.backbone.parameters())
        return dict(
            aggregator=self.long_memory_aggregator,
            tvi_dim=self.tvi_dim,
            hidden_size=self.hidden_size,
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def _condition(self, samples, *, capture=False, shuffle=0.0, mask=0.0):
        self.backbone.prepare_samples(samples)
        attach_navvla_long_memory_tokens(samples, **self._memory_kwargs())
        condition, records = self.backbone.encode_context(
            samples,
            tvi_embedding=self.tvi_embedding,
            capture_online_current_cache=capture,
            history_shuffle_probability=shuffle,
            tvi_mask_probability=mask,
        )
        if self.long_memory_aggregator is not None and any(
            s.pop("_long_memory_zero_dependency", False) for s in samples
        ):
            condition = condition + self.long_memory_aggregator.zero_dependency(condition)
        return condition, records

    def forward(self, examples=None, *, training_step=None, total_training_steps=None):
        samples = self._samples(examples)
        shuffle, mask = (
            history_augmentation_probabilities(
                self.history_augmentation, training_step=training_step, total_training_steps=total_training_steps
            )
            if self.training
            else (0.0, 0.0)
        )
        condition, _ = self._condition(samples, shuffle=shuffle, mask=mask)
        target = torch.as_tensor(
            np.asarray([s["action"] for s in samples]), device=condition.device, dtype=condition.dtype
        )[:, -self.action_horizon :]
        padding = torch.as_tensor(
            np.asarray([s["action_padding_mask"] for s in samples]), device=condition.device, dtype=torch.bool
        )[:, -self.action_horizon :]
        # Preserve existing zero-target supervision explicitly; masking is an opt-in training objective.
        if self.config.framework.action_model.get("padding_loss", "zero_target") == "zero_target":
            target = target.masked_fill(padding.unsqueeze(-1), 0.0)
            for i, sample in enumerate(samples):
                if self.action_dim > 4 and sample["metadata"].get("action_extra_dim_mode") == "path_progress":
                    target[i, padding[i], 4] = 1.0
        state = self._state(samples, condition)
        loss = self.action_model.loss(condition, target, state=state, action_padding_mask=padding)
        return {"action_loss": loss, "loss": loss}

    def _state(self, samples, condition):
        if not int(self.config.framework.action_model.state_dim):
            return None
        state = torch.as_tensor(
            np.asarray([s["state"] for s in samples]), device=condition.device, dtype=condition.dtype
        )
        if state.shape != (len(samples), int(self.config.framework.action_model.state_dim)):
            raise ValueError(f"State shape does not match the model: {state.shape}")
        return state

    @torch.inference_mode()
    def predict_action(self, examples=None, *, tvi_mask_probability=0.0, generator=None):
        samples = self._samples(examples)
        condition, records = self._condition(samples, capture=True, mask=tvi_mask_probability)
        parameter = next(self.action_model.parameters())
        condition = condition.to(device=parameter.device, dtype=parameter.dtype)
        actions = self.action_model.predict_action(condition, self._state(samples, condition), generator=generator)
        updates = compute_navvla_online_long_memory_updates(samples, **self._memory_kwargs())
        return {
            "normalized_actions": actions.float().cpu().numpy(),
            "metadata": {"online_current_visual_tokens": records, "online_long_memory_updates": updates},
        }

    def new_session(self, *, episode_id, statistics_key, seed):
        from starVLA.model.modules.input_context import ModelSession

        key = statistics_key
        if key is None:
            if len(self.input_profiles) != 1:
                raise ValueError("statistics_key is required for models with multiple input profiles")
            key = next(iter(self.input_profiles))
        return ModelSession(episode_id, self.input_profiles[key], self.norm_stats[key]["action"], int(seed))

    @torch.inference_mode()
    def predict_session(self, session, *, frames, instruction):
        from tool.navvla.statistics import unnormalize_values

        session.append(frames)
        profile = session.profile
        if profile["visual_token_mode"] != "online_images" or profile["require_long_memory_tokens"]:
            pending = [
                (f, camera)
                for f in session.frames[:-1]
                for camera in profile["cameras"]
                if camera not in f.get("cache", {})
            ]
            if pending:
                encoded = self.backbone.encode_history_images([f["images"][camera] for f, camera in pending])
                for (frame, camera), record in zip(pending, encoded, strict=True):
                    frame.setdefault("cache", {})[camera] = record
        sample = session.sample(instruction)
        generator = torch.Generator(device=next(self.action_model.parameters()).device)
        generator.manual_seed(session.seed + session.predictions)
        output = self.predict_action([sample], generator=generator)
        session.predictions += 1
        current = session.frames[-1]
        records = output["metadata"]["online_current_visual_tokens"]
        current["cache"] = {r["camera_name"]: r for r in records}
        updates = output["metadata"]["online_long_memory_updates"]
        if updates:
            session.long_memory = updates[-1]
            session.memory_frame_indices.add(int(updates[-1]["frame_index"]))
        # Frozen caches retain metadata/poses without retaining all episode RGB arrays.
        if profile["visual_token_mode"] != "online_images":
            for frame in session.frames[:-1]:
                frame.pop("images", None)
        normalized = output["normalized_actions"][0]
        actions = normalized.copy()
        stats_dim = len(session.action_stats["q01"])
        actions[:, :stats_dim] = unnormalize_values(normalized[:, :stats_dim], session.action_stats)
        return {
            "actions": actions,
            "stop": bool(output.get("stop", False)),
            "action_spec": "anchor_relative_body_frame_xyz_yaw",
        }
