"""Shared input construction for offline readers and live model sessions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from starVLA.model.modules.bats import (
    BATSSelectionResult,
    online_bats_history_budget,
    select_bats_history,
    select_long_memory_candidate,
)
from starVLA.model.modules.tvi import LEARNED_TOKEN_TVI_MODE, tvi_rows
from tool.navvla.statistics import body_frame_action_from_pose, normalize_values


def rgb_image(image):
    """Image boundary shared by live sessions and native backbone processors."""
    if isinstance(image, Image.Image):
        return image if image.mode == "RGB" else image.convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError("Images must be PIL images or uint8 RGB/RGBA arrays")
    return Image.fromarray(array).convert("RGB")


def select_history(candidates, *, anchor_frame_index, episode_id, profile):
    costs = {
        key: profile[key]
        for key in (
            "token_budget",
            "current_visual_tokens",
            "history_visual_tokens",
            "tvi_tokens",
            "current_wrapper_tokens",
            "history_wrapper_tokens",
        )
    }
    cameras = profile.get("budget_num_cameras", len(profile["cameras"]))
    if profile["history_sampling_mode"] == "bats":
        return select_bats_history(
            candidates=candidates,
            anchor_frame_index=anchor_frame_index,
            episode_id=episode_id,
            dataset_name=profile["dataset_name"],
            seed=profile["bats_seed"],
            epsilon=profile["bats_epsilon"],
            k=profile["bats_k"],
            use_dynamic_bats_k=profile["use_dynamic_bats_k"],
            sampling_mode="priority_capped",
            budget_num_cameras=cameras,
            **costs,
        )
    capacity = online_bats_history_budget(budget_num_cameras=cameras, **costs)
    if capacity == 0:
        selected = []
    elif profile["history_sampling_mode"] == "continuous_uniform" and len(candidates) > capacity:
        selected = [candidates[i][1] for i in np.linspace(0, len(candidates) - 1, capacity, dtype=int)]
    else:
        selected = [item for _, item in candidates[-capacity:]]
    return BATSSelectionResult(selected, list(selected), profile["bats_k"], capacity)


def history_state(poses, current_pose, *, action_stats, state_dim):
    """Selected real poses to a fixed-size state, shared by training and inference."""
    if state_dim <= 0 or state_dim % 4:
        raise ValueError("state_dim must be a positive multiple of four")
    count = state_dim // 4
    poses = [np.asarray(p, dtype=np.float32) for p in poses[-count:]] + [np.asarray(current_pose, dtype=np.float32)]
    state = np.zeros((count, 4), dtype=np.float32)
    if len(poses) > 1:
        actions = np.stack([body_frame_action_from_pose(a, b) for a, b in zip(poses, poses[1:])])
        state[-len(actions) :] = normalize_values(actions, action_stats)
    return state.reshape(-1)


def input_sample(
    *, images, current_tvi, history_tvi, history_mask, instruction, platform_text, metadata, history_images=None
):
    sample = dict(
        images=images,
        current_tvi=current_tvi,
        history_tvi=history_tvi,
        history_mask=np.asarray(history_mask, dtype=bool),
        lang=instruction,
        platform_text=platform_text,
        metadata=metadata,
    )
    if history_images is not None:
        sample["history_images"] = history_images
    return sample


@dataclass
class ModelSession:
    episode_id: str
    profile: dict[str, Any]
    action_stats: dict[str, Any]
    seed: int
    frames: list[dict[str, Any]] = field(default_factory=list)
    time_origin: float | None = None
    pose_origin: np.ndarray | None = None
    predictions: int = 0
    long_memory: dict[str, Any] | None = None
    memory_frame_indices: set[int] = field(default_factory=set)

    def append(self, frames):
        if not frames:
            raise ValueError("predict requires at least one new observation")
        previous = self.frames[-1]["frame_id"] if self.frames else -1
        previous_time = self.frames[-1]["timestamp_s"] if self.frames else -float("inf")
        converted = []
        for frame in frames:
            if frame["frame_id"] <= previous or frame["timestamp_s"] < previous_time:
                raise ValueError("Frames must have increasing IDs and monotonic timestamps")
            images = {}
            for camera in self.profile["cameras"]:
                image = rgb_image(frame["images"][camera])
                if self.profile["image_resize"] is not None:
                    image = image.resize(tuple(self.profile["image_resize"]))
                images[camera] = image
            if self.time_origin is None:
                self.time_origin = float(frame["timestamp_s"])
                self.pose_origin = np.asarray(frame["body_pose"], dtype=np.float32)
            converted.append(
                {
                    **frame,
                    "images": images,
                    "frame_index": frame["frame_id"],
                    "relative_time": float(frame["timestamp_s"]) - self.time_origin,
                }
            )
            previous, previous_time = frame["frame_id"], frame["timestamp_s"]
        self.frames.extend(converted)

    def sample(self, instruction):
        current = self.frames[-1]
        profile = self.profile
        selection = select_history(
            [(frame["frame_id"], frame) for frame in self.frames[:-1]],
            anchor_frame_index=current["frame_id"],
            episode_id=self.episode_id,
            profile=profile,
        )
        selected = selection.selected
        cameras = list(profile["cameras"])
        blocks = [
            {"step_index": i, "frame_index": frame["frame_id"], "camera_name": camera}
            for i, frame in enumerate(selected)
            for camera in cameras
        ]
        history_tvi = (
            np.concatenate([self._tvi(frame) for frame in selected], axis=0) if selected else self._tvi(current)[:0]
        )
        sample = input_sample(
            images=current["images"],
            current_tvi=self._tvi(current),
            history_tvi=history_tvi,
            history_mask=np.ones(len(blocks), dtype=bool),
            instruction=instruction,
            platform_text=profile["platform_text"],
            metadata={
                "required_cameras": cameras,
                "frame_index": current["frame_id"],
                "timestamp": current["timestamp_s"],
                "history_steps": [{"frame_index": f["frame_id"], "timestamp": f["timestamp_s"]} for f in selected],
                "history_blocks": blocks,
                "visual_token_profile": profile["visual_token_profile"],
                "action_extra_dim_mode": profile["action_extra_dim_mode"],
            },
        )
        if selected:
            if profile["visual_token_mode"] == "online_images":
                sample["history_images"] = {camera: [f["images"][camera] for f in selected] for camera in cameras}
            else:
                self._cache_fields(sample, "history_cached", [f["cache"][c] for f in selected for c in cameras])
        if profile["include_state"]:
            sample["state"] = history_state(
                [f["body_pose"] for f in selected],
                current["body_pose"],
                action_stats=self.action_stats,
                state_dim=profile["state_dim"],
            )
        if self.long_memory is not None:
            sample["long_memory_tokens"] = self.long_memory["tokens"]
            sample["long_memory_tvi"] = self.long_memory["tvi"]
            sample["metadata"]["long_memory_blocks"] = self.long_memory["blocks"]
        if profile["require_long_memory_tokens"]:
            candidate = select_long_memory_candidate(
                selection.ranked_selected, memory_frame_indices=self.memory_frame_indices
            )
            if candidate is not None:
                self._cache_fields(sample, "online_long_memory_update", [candidate["cache"][c] for c in cameras])
                sample["online_long_memory_update_tvi"] = self._tvi(candidate)
                sample["metadata"]["online_long_memory_update_frame_index"] = candidate["frame_id"]
                sample["metadata"]["online_long_memory_update_blocks"] = [
                    {"step_index": 0, "frame_index": candidate["frame_id"], "camera_name": c} for c in cameras
                ]
        return sample

    def _tvi(self, frame):
        cameras = self.profile["cameras"]
        return tvi_rows(
            mode=self.profile["tvi_mode"],
            timestamps=[frame["relative_time"]] * len(cameras),
            azimuths=[v["azimuth_rad"] for v in cameras.values()],
            camera_poses=self._camera_poses(frame)
            if self.profile["tvi_mode"] in {"time_camera_pose", "metric_camera_pose"}
            else None,
        )

    def _camera_poses(self, frame):
        from scipy.spatial.transform import Rotation

        poses = np.asarray([frame["camera_poses"][c] for c in self.profile["cameras"]], dtype=np.float32).copy()
        if self.profile.get("camera_pose_position_frame", "world") == "episode_relative_first_body_aligned":
            delta = poses[:, :3] - self.pose_origin[:3]
            c, s = np.cos(self.pose_origin[3]), np.sin(self.pose_origin[3])
            poses[:, :3] = delta @ np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        if self.profile.get("camera_pose_rotation_frame", "world") == "body_mount":
            body = np.asarray(frame["body_rotation"])[[0, 2, 1]]
            world = Rotation.from_euler("ZYX", poses[:, [3, 5, 4]])
            mount = (Rotation.from_euler("ZYX", body).inv() * world).as_euler("ZYX")
            poses[:, 3:] = mount[:, [0, 2, 1]]
        return poses

    @staticmethod
    def _cache_fields(sample, prefix, records):
        tokens_key = "history_cached_embeds" if prefix == "history_cached" else f"{prefix}_tokens"
        sample[tokens_key] = np.stack([r["tokens"] for r in records])
        sample[f"{prefix}_mask"] = np.ones(len(records), dtype=bool)
        for key in ("cache_stage", "storage_encoding", "encoder_ckpt"):
            if key in records[0]:
                sample[f"{prefix}_{key}"] = records[0][key]
        if "grid_thw" in records[0]:
            sample[f"{prefix}_grid_thw"] = np.stack([r["grid_thw"] for r in records])


@dataclass(frozen=True)
class HistoryAugmentationConfig:
    enabled: bool = False
    shuffle_target_probability: float = 0.3
    shuffle_warmup_end_ratio: float = 0.05
    tvi_mask_target_probability: float = 0.1
    tvi_mask_warmup_start_ratio: float = 0.05
    tvi_mask_warmup_end_ratio: float = 0.15

    def __post_init__(self) -> None:
        values = {
            "shuffle_target_probability": self.shuffle_target_probability,
            "shuffle_warmup_end_ratio": self.shuffle_warmup_end_ratio,
            "tvi_mask_target_probability": self.tvi_mask_target_probability,
            "tvi_mask_warmup_start_ratio": self.tvi_mask_warmup_start_ratio,
            "tvi_mask_warmup_end_ratio": self.tvi_mask_warmup_end_ratio,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"history augmentation {name} must be finite and within [0, 1], got {value}")
        if self.shuffle_warmup_end_ratio > self.tvi_mask_warmup_start_ratio:
            raise ValueError("history augmentation shuffle warmup must end before TVI mask warmup starts")
        if self.tvi_mask_warmup_start_ratio > self.tvi_mask_warmup_end_ratio:
            raise ValueError("history augmentation TVI mask warmup start must not exceed its end")


def history_augmentation_probabilities(
    config: HistoryAugmentationConfig,
    *,
    training_step: int | None = None,
    total_training_steps: int | None = None,
) -> tuple[float, float]:
    if not config.enabled:
        return 0.0, 0.0
    if training_step is None or int(training_step) < 0:
        raise ValueError(f"training_step must be a non-negative integer, got {training_step}")
    if total_training_steps is None or int(total_training_steps) <= 0:
        raise ValueError(f"total_training_steps must be a positive integer, got {total_training_steps}")
    progress = min(1.0, max(0.0, float(training_step) / float(total_training_steps)))
    shuffle_probability = config.shuffle_target_probability
    if config.shuffle_warmup_end_ratio:
        shuffle_probability *= min(1.0, progress / config.shuffle_warmup_end_ratio)
    if progress < config.tvi_mask_warmup_start_ratio:
        mask_probability = 0.0
    elif config.tvi_mask_warmup_end_ratio == config.tvi_mask_warmup_start_ratio:
        mask_probability = config.tvi_mask_target_probability
    else:
        mask_probability = config.tvi_mask_target_probability * min(
            1.0,
            (progress - config.tvi_mask_warmup_start_ratio)
            / (config.tvi_mask_warmup_end_ratio - config.tvi_mask_warmup_start_ratio),
        )
    return float(shuffle_probability), float(mask_probability)


def as_numpy_tvi(values: Any, *, tvi_dim: int) -> np.ndarray:
    if int(tvi_dim) <= 0:
        raise ValueError(f"tvi_dim must be positive, got {tvi_dim}")
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, int(tvi_dim)), dtype=np.float32)
    if array.ndim != 2 or int(array.shape[1]) != int(tvi_dim):
        raise ValueError(f"TVI values must have shape [N, {tvi_dim}], got {tuple(array.shape)}")
    return array


def target_visual_tokens_for_block(
    block: dict[str, Any],
    *,
    history_visual_tokens: int,
    long_memory_visual_tokens: int,
    current_visual_tokens: int,
) -> int:
    if bool(block.get("is_long_memory", False)):
        return int(long_memory_visual_tokens)
    if bool(block.get("is_history", False)):
        return int(history_visual_tokens)
    return int(current_visual_tokens)


def samples_from_collated_batch(
    examples: list[dict[str, Any]] | dict[str, Any] | None,
    *,
    extra_keys: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if isinstance(examples, list):
        return examples
    if examples is None:
        raise ValueError("NavVLA forward requires examples or a collated batch")
    batch_size = len(examples["lang"])
    shared_keys = (
        "history_cached_embeds",
        "history_cached_mask",
        "long_memory_source_tokens",
        "long_memory_source_mask",
        "long_memory_source_tvi",
        "long_memory_tokens",
        "long_memory_tvi",
        "online_long_memory_update_tokens",
        "online_long_memory_update_tvi",
        "online_long_memory_update_mask",
        "state",
        "action",
        "action_padding_mask",
    )
    samples: list[dict[str, Any]] = []
    for index in range(batch_size):
        sample = {
            key: examples[key][index]
            for key in (
                "current_tvi",
                "history_tvi",
                "history_mask",
                "lang",
                "platform_text",
                "metadata",
            )
        }
        sample["images"] = {camera: values[index] for camera, values in examples["images"].items()}
        if "history_images" in examples:
            sample["history_images"] = {
                camera: camera_batch[index] for camera, camera_batch in examples["history_images"].items()
            }
        for key in (*shared_keys, *extra_keys):
            if key in examples:
                sample[key] = examples[key][index]
        samples.append(sample)
    return samples


def build_navvla_instruction(
    sample: dict[str, Any], *, use_platform_text: bool = True, prompt_template: str = "{instruction}"
) -> str:
    parts: list[str] = []
    if use_platform_text:
        platform = str(sample["platform_text"]).strip()
        if platform:
            parts.append(platform)
    parts.append(str(sample["lang"]).strip())
    return prompt_template.replace("{instruction}", " ".join(part for part in parts if part).strip())


def scatter_image_embeddings(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    image_embeddings: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    mask = input_ids == int(image_token_id)
    if int(mask.sum().item()) != int(image_embeddings.shape[0]):
        raise ValueError(
            f"image token count mismatch: placeholders={int(mask.sum().item())}, "
            f"embeddings={int(image_embeddings.shape[0])}"
        )
    scattered = inputs_embeds.clone()
    scattered[mask] = image_embeddings.to(device=scattered.device, dtype=scattered.dtype)
    return scattered


def mask_history_tvi_embeddings(
    tvi_embedding,
    embeddings: torch.Tensor,
    blocks: list[dict[str, Any]],
    *,
    probability: float,
) -> torch.Tensor:
    if probability <= 0.0 or not blocks or tvi_embedding.mode == LEARNED_TOKEN_TVI_MODE:
        return embeddings
    eligible = torch.tensor(
        [bool(block.get("is_history", False)) and not bool(block.get("is_long_memory", False)) for block in blocks],
        device=embeddings.device,
        dtype=torch.bool,
    )
    sampled = torch.rand((len(blocks),), device=embeddings.device) < float(probability)
    return tvi_embedding.replace_masked_rows(embeddings, eligible & sampled)


def build_navvla_cached_visual_sequence(
    sample,
    *,
    required_cameras,
    history_shuffle_probability=0.0,
    generator=None,
    tvi_dim=2,
):
    """Order explicit memory/history/current blocks; never infer missing cameras or TVI."""
    metadata = sample["metadata"]
    camera_order = {camera: index for index, camera in enumerate(required_cameras)}
    images, blocks = [], []
    if "long_memory_tokens" in sample:
        memory = sample["long_memory_tokens"]
        memory_tvi = as_numpy_tvi(sample["long_memory_tvi"], tvi_dim=tvi_dim)
        memory_blocks = metadata["long_memory_blocks"]
        if len(memory) != len(memory_blocks) or len(memory_tvi) != len(memory_blocks):
            raise ValueError("Long-memory tokens, TVI and block metadata must align")
        order = sorted(
            range(len(memory_blocks)),
            key=lambda i: (
                memory_blocks[i]["step_index"],
                camera_order[memory_blocks[i]["camera_name"]],
            ),
        )
        for index in order:
            blocks.append(
                {
                    **memory_blocks[index],
                    "is_history": True,
                    "is_cached_history": True,
                    "is_long_memory": True,
                    "long_memory_index": index,
                    "tvi": memory_tvi[index],
                    "sample": sample,
                }
            )
    history_tvi = as_numpy_tvi(sample["history_tvi"], tvi_dim=tvi_dim)
    history_blocks = metadata["history_blocks"]
    history_mask = sample["history_mask"]
    cached = "history_cached_embeds" in sample
    cache_mask = sample["history_cached_mask"] if cached else history_mask
    if min(len(history_tvi), len(history_mask), len(cache_mask)) < len(history_blocks):
        raise ValueError("History TVI and masks must cover every block")
    order = sorted(
        (i for i in range(len(history_blocks)) if history_mask[i] and cache_mask[i]),
        key=lambda i: (history_blocks[i]["step_index"], camera_order[history_blocks[i]["camera_name"]]),
    )
    if len(order) > 1 and history_shuffle_probability > 0:
        if torch.rand((), generator=generator).item() < history_shuffle_probability:
            order = [order[i] for i in torch.randperm(len(order), generator=generator).tolist()]
    for index in order:
        block = history_blocks[index]
        if not cached:
            image = sample["history_images"][block["camera_name"]][block["step_index"]]
            if image is None:
                raise ValueError("Unmasked history block is missing its image")
            images.append(image)
        blocks.append(
            {
                **block,
                "is_history": True,
                "is_cached_history": cached,
                "cached_history_index": index,
                "tvi": history_tvi[index],
                "sample": sample,
            }
        )
    current_tvi = as_numpy_tvi(sample["current_tvi"], tvi_dim=tvi_dim)
    if len(current_tvi) != len(required_cameras):
        raise ValueError("Current TVI must contain one row per required camera")
    for camera, tvi in zip(required_cameras, current_tvi, strict=True):
        image = sample["images"][camera]
        if image is None:
            raise ValueError(f"Current image is missing for required camera {camera}")
        images.append(image)
        blocks.append(
            {
                "camera_name": camera,
                "frame_index": metadata["frame_index"],
                "is_history": False,
                "is_cached_history": False,
                "tvi": tvi,
                "sample": sample,
            }
        )
    return images, blocks


def image_token_spans(row, token_id):
    positions = torch.nonzero(row == token_id, as_tuple=False).flatten().tolist()
    if not positions:
        return []
    starts = [positions[0]] + [b for a, b in zip(positions, positions[1:]) if b != a + 1]
    ends = [a + 1 for a, b in zip(positions, positions[1:]) if b != a + 1] + [positions[-1] + 1]
    return list(zip(starts, ends))


def gather_action_queries(hidden_states, input_ids, *, token_id, num_placeholders):
    mask = input_ids == token_id
    if (mask.sum(dim=1) < num_placeholders).any():
        raise ValueError(f"Every sample must contain at least {num_placeholders} action query tokens")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
    selected = torch.where(mask, positions, -1).topk(num_placeholders, dim=-1).values.sort(dim=-1).values
    return hidden_states.gather(1, selected.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]))


def validate_single_token(tokenizer, token, *, role):
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if len(encoded) != 1:
        raise ValueError(f"Action {role} token {token!r} must encode to one token, got {encoded}")
    return token, int(encoded[0])


def navigation_messages(tokenizer, images, instructions, action_suffixes, *, max_text_tokens):
    messages = []
    for sample_images, instruction, suffix in zip(images, instructions, action_suffixes, strict=True):
        tokens = tokenizer.encode(instruction, add_special_tokens=False)
        if len(tokens) > max_text_tokens:
            instruction = tokenizer.decode(
                tokens[:max_text_tokens], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        content = [{"type": "image", "image": image} for image in sample_images]
        content.append({"type": "text", "text": instruction})
        messages.append(
            [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": suffix}]},
            ]
        )
    return messages
