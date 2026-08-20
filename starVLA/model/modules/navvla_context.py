"""VLM-agnostic NavVLA context ordering and augmentation helpers."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from starVLA.model.modules.tvi import LEARNED_TOKEN_TVI_MODE

CAMERA_VIEWPOINT_TO_ID = {"front": 0, "left": 1, "right": 2, "rear": 3, "down": 4}


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
    if values is None:
        return np.zeros((0, int(tvi_dim)), dtype=np.float32)
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
    metadata = list(examples.get("metadata", [{} for _ in range(batch_size)]))
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
    )
    samples: list[dict[str, Any]] = []
    for index in range(batch_size):
        images = {
            camera: camera_batch[index]
            for camera, camera_batch in examples.get("images", {}).items()
            if index < len(camera_batch) and camera_batch[index] is not None
        }
        sample = {
            "images": images,
            "current_tvi": examples["current_tvi"][index],
            "history_tvi": examples["history_tvi"][index],
            "history_mask": examples["history_mask"][index]
            if "history_mask" in examples
            else np.ones((0,), dtype=bool),
            "lang": examples["lang"][index],
            "platform_text": examples.get("platform_text", [""] * batch_size)[index],
            "action": examples["action"][index],
            "action_padding_mask": examples["action_padding_mask"][index],
            "metadata": metadata[index],
        }
        if "history_images" in examples:
            sample["history_images"] = {
                camera: camera_batch[index] for camera, camera_batch in examples["history_images"].items()
            }
        for key in (*shared_keys, *extra_keys):
            if key in examples:
                sample[key] = examples[key][index]
        samples.append(sample)
    return samples


def build_navvla_instruction(owner: Any, sample: dict[str, Any]) -> str:
    parts: list[str] = []
    if bool(owner.config.framework.navvla.get("use_platform_text", True)):
        platform = str(sample.get("platform_text", "")).strip()
        if platform:
            parts.append(platform)
    parts.append(str(sample.get("lang", "")).strip())
    return " ".join(part for part in parts if part).strip()


def sample_required_cameras(sample: dict[str, Any]) -> list[str]:
    metadata = sample.get("metadata", {}) or {}
    cameras = metadata.get("required_cameras")
    if cameras:
        return [str(camera_name) for camera_name in cameras]
    ordered_cameras: list[str] = []
    seen: set[str] = set()
    for camera_name in sample.get("images", {}).keys():
        value = str(camera_name)
        if value and value not in seen:
            seen.add(value)
            ordered_cameras.append(value)
    for block in list(metadata.get("history_blocks") or []) + list(metadata.get("long_memory_blocks") or []):
        value = str(block.get("camera_name", ""))
        if value and value not in seen:
            seen.add(value)
            ordered_cameras.append(value)
    if not ordered_cameras:
        raise ValueError("NavVLA sample is missing required_cameras and visual camera metadata")
    return ordered_cameras


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


def action_model_accepts_padding_mask(action_model: Any) -> bool:
    try:
        signature = inspect.signature(action_model.forward)
    except (TypeError, ValueError):
        return False
    return "action_padding_mask" in signature.parameters


def mask_history_tvi_embeddings(
    owner: Any,
    embeddings: torch.Tensor,
    blocks: list[dict[str, Any]],
    *,
    probability: float,
) -> torch.Tensor:
    if (
        probability <= 0.0
        or not blocks
        or getattr(owner.tvi_embedding, "mode", None) == LEARNED_TOKEN_TVI_MODE
    ):
        return embeddings
    eligible = torch.tensor(
        [bool(block.get("is_history", False)) and not bool(block.get("is_long_memory", False)) for block in blocks],
        device=embeddings.device,
        dtype=torch.bool,
    )
    sampled = torch.rand((len(blocks),), device=embeddings.device) < float(probability)
    return owner.tvi_embedding.replace_masked_rows(embeddings, eligible & sampled)


def forward_navvla_action(
    owner: Any,
    examples: list[dict[str, Any]] | dict[str, Any] | None,
    *,
    training_step: int | None,
    total_training_steps: int | None,
) -> dict[str, torch.Tensor]:
    samples = owner._samples_from_batch(examples)
    if owner.training:
        shuffle_probability, mask_probability = history_augmentation_probabilities(
            owner.history_augmentation,
            training_step=training_step,
            total_training_steps=total_training_steps,
        )
    else:
        shuffle_probability, mask_probability = 0.0, 0.0
    vl_embs, _records = owner._forward_vlm_for_action(
        samples,
        history_shuffle_probability=shuffle_probability,
        tvi_mask_probability=mask_probability,
    )
    actions = torch.as_tensor(
        np.asarray([sample["action"] for sample in samples], dtype=np.float32),
        device=vl_embs.device,
        dtype=vl_embs.dtype,
    )
    if int(actions.shape[-1]) != owner.action_dim:
        raise ValueError(f"NavVLA action dim mismatch: batch={int(actions.shape[-1])}, model={owner.action_dim}")
    action_padding_mask = torch.as_tensor(
        np.asarray([sample["action_padding_mask"] for sample in samples], dtype=bool),
        device=vl_embs.device,
        dtype=torch.bool,
    )
    actions_target = actions[:, -owner.action_horizon :, :].clone()
    action_padding_mask = action_padding_mask[:, -owner.action_horizon :]
    actions_target = actions_target.masked_fill(action_padding_mask.unsqueeze(-1), 0.0)
    path_progress_rows = torch.as_tensor(
        [
            str(sample.get("metadata", {}).get("action_extra_dim_mode", "none")) == "path_progress"
            for sample in samples
        ],
        device=vl_embs.device,
        dtype=torch.bool,
    )
    if owner.action_dim > 4 and bool(path_progress_rows.any().item()):
        progress_padding_mask = action_padding_mask & path_progress_rows.unsqueeze(-1)
        actions_target[..., 4] = torch.where(
            progress_padding_mask,
            torch.ones_like(actions_target[..., 4]),
            actions_target[..., 4],
        )
    repeats = int(owner.config.framework.action_model.get("repeated_diffusion_steps", 2))
    actions_repeated = actions_target.repeat(repeats, 1, 1)
    vl_embs_repeated = vl_embs.repeat(repeats, 1, 1)
    mask_repeated = action_padding_mask.repeat(repeats, 1)
    state = None
    if any("state" in sample for sample in samples):
        state = torch.as_tensor(
            np.asarray([sample.get("state", np.zeros((0,), dtype=np.float32)) for sample in samples], dtype=np.float32),
            device=vl_embs.device,
            dtype=vl_embs.dtype,
        ).repeat(repeats, 1)
    if action_model_accepts_padding_mask(owner.action_model):
        action_loss = owner.action_model(
            vl_embs_repeated,
            actions_repeated,
            state,
            action_padding_mask=mask_repeated,
        )
    else:
        action_loss = owner.action_model(vl_embs_repeated, actions_repeated, state)
    return {"action_loss": action_loss, "loss": action_loss}


def predict_navvla_action(
    owner: Any,
    examples: list[dict[str, Any]] | dict[str, Any] | None,
    *,
    tvi_mask_probability: float,
) -> dict[str, Any]:
    samples = owner._samples_from_batch(examples)
    if not 0.0 <= float(tvi_mask_probability) <= 1.0:
        raise ValueError("tvi_mask_probability must be between 0 and 1")
    vl_embs, records = owner._forward_vlm_for_action(
        samples,
        capture_online_current_cache=True,
        history_shuffle_probability=0.0,
        tvi_mask_probability=float(tvi_mask_probability),
    )
    try:
        action_param = next(owner.action_model.parameters())
        action_device, action_dtype = action_param.device, action_param.dtype
    except StopIteration:
        action_device, action_dtype = vl_embs.device, vl_embs.dtype
    vl_embs = vl_embs.to(device=action_device, dtype=action_dtype)
    state = None
    if any("state" in sample for sample in samples):
        state = torch.as_tensor(
            np.asarray([sample.get("state", np.zeros((0,), dtype=np.float32)) for sample in samples], dtype=np.float32),
            device=action_device,
            dtype=action_dtype,
        )
    predictions = owner.action_model.predict_action(vl_embs, state)
    return {
        "normalized_actions": predictions.detach().to(torch.float32).cpu().numpy(),
        "metadata": {
            "online_current_visual_tokens": records,
            "online_long_memory_updates": owner._compute_online_long_memory_updates(samples),
        },
    }


@dataclass(frozen=True)
class _HistoryVisualRecord:
    step_index: int
    camera_index: int
    block_index: int
    camera_name: str
    frame_index: int
    tvi: np.ndarray
    image: Any | None
    is_cached_history: bool


def _as_numpy_bool(values: Any) -> np.ndarray:
    return np.asarray(values if values is not None else [], dtype=bool).reshape(-1)


def _tvi_yaw_index(tvi_dim: int) -> int:
    if int(tvi_dim) == 2:
        return 1
    if int(tvi_dim) == 7:
        return 4
    raise ValueError(f"TVI yaw compatibility is defined only for widths 2 and 7, got {tvi_dim}")


def _fallback_tvi_row(*, time: float, tvi_dim: int) -> np.ndarray:
    row = np.zeros((int(tvi_dim),), dtype=np.float32)
    row[0] = float(time)
    return row


def _history_image_for_block(
    history_images: dict[str, Any],
    *,
    camera_name: str,
    step_index: int,
    block_index: int,
) -> Any | None:
    camera_images = history_images.get(camera_name)
    if camera_images is None:
        return None
    if isinstance(camera_images, np.ndarray) and camera_images.ndim >= 4:
        if 0 <= int(step_index) < int(camera_images.shape[0]):
            return camera_images[int(step_index)]
        if 0 <= int(block_index) < int(camera_images.shape[0]):
            return camera_images[int(block_index)]
        return None
    if isinstance(camera_images, (list, tuple)):
        if 0 <= int(step_index) < len(camera_images):
            return camera_images[int(step_index)]
        if 0 <= int(block_index) < len(camera_images):
            return camera_images[int(block_index)]
    return None


def build_navvla_cached_visual_sequence(
    sample: dict[str, Any],
    *,
    required_cameras: list[str],
    history_shuffle_probability: float = 0.0,
    generator: torch.Generator | None = None,
    tvi_dim: int = 2,
) -> tuple[list[Any], list[dict[str, Any]]]:
    if not math.isfinite(float(history_shuffle_probability)) or not 0.0 <= float(history_shuffle_probability) <= 1.0:
        raise ValueError(
            f"history_shuffle_probability must be finite and within [0, 1], got {history_shuffle_probability}"
        )
    online_images: list[Any] = []
    blocks: list[dict[str, Any]] = []
    metadata = sample.get("metadata", {}) or {}
    camera_order = {camera_name: index for index, camera_name in enumerate(required_cameras)}
    yaw_index = _tvi_yaw_index(tvi_dim)

    long_tokens = sample.get("long_memory_tokens")
    if long_tokens is not None:
        if isinstance(long_tokens, torch.Tensor):
            long_shape = tuple(long_tokens.shape)
            long_size = int(long_tokens.numel())
        else:
            long_array = np.asarray(long_tokens)
            long_shape = tuple(long_array.shape)
            long_size = int(long_array.size)
        long_tvi = as_numpy_tvi(sample.get("long_memory_tvi"), tvi_dim=tvi_dim)
        long_blocks = list(metadata.get("long_memory_blocks") or [])
        long_records: list[tuple[int, int, int, str, np.ndarray]] = []
        if len(long_shape) == 3 and long_size:
            for block_index in range(int(long_shape[0])):
                block_meta = long_blocks[block_index] if block_index < len(long_blocks) else {}
                camera_name = str(block_meta.get("camera_name", required_cameras[block_index % len(required_cameras)]))
                if camera_name not in camera_order:
                    continue
                step_index = int(
                    block_meta.get(
                        "step_index",
                        block_meta.get("max_frame_index", block_meta.get("long_memory_max_frame_index", block_index)),
                    )
                )
                tvi = (
                    long_tvi[block_index]
                    if block_index < len(long_tvi)
                    else _fallback_tvi_row(time=float(step_index), tvi_dim=tvi_dim)
                )
                long_records.append((step_index, camera_order[camera_name], block_index, camera_name, tvi))
        for _step_index, _camera_index, block_index, camera_name, tvi in sorted(long_records):
            block_tvi = np.asarray(tvi, dtype=np.float32).copy()
            blocks.append(
                {
                    "is_history": True,
                    "is_cached_history": True,
                    "is_long_memory": True,
                    "long_memory_index": int(block_index),
                    "camera_name": camera_name,
                    "tvi": block_tvi,
                    "time": float(block_tvi[0]),
                    "phi": float(block_tvi[yaw_index]),
                    "viewpoint_id": CAMERA_VIEWPOINT_TO_ID.get(camera_name, 0),
                    "sample": sample,
                }
            )

    history_tvi = as_numpy_tvi(sample.get("history_tvi"), tvi_dim=tvi_dim)
    history_mask = _as_numpy_bool(sample.get("history_mask"))
    cached_mask = _as_numpy_bool(sample.get("history_cached_mask"))
    history_blocks = list(metadata.get("history_blocks") or [])
    history_images = sample.get("history_images") or {}
    cached_history = sample.get("history_cached_embeds")
    has_cached_history = cached_history is not None
    history_records: list[_HistoryVisualRecord] = []
    if history_blocks:
        for block_index, block in enumerate(history_blocks):
            camera_name = str(block["camera_name"])
            if camera_name not in camera_order:
                continue
            if block_index < len(history_mask) and not bool(history_mask[block_index]):
                continue
            if block_index < len(cached_mask) and not bool(cached_mask[block_index]):
                continue
            step_index = int(block.get("step_index", block_index))
            if block_index >= len(history_tvi):
                raise ValueError(f"ordinary history block {block_index} is missing its TVI row")
            image = None
            if not has_cached_history:
                image = _history_image_for_block(
                    history_images,
                    camera_name=camera_name,
                    step_index=step_index,
                    block_index=block_index,
                )
                if image is None:
                    continue
            history_records.append(
                _HistoryVisualRecord(
                    step_index=step_index,
                    camera_index=camera_order[camera_name],
                    block_index=block_index,
                    camera_name=camera_name,
                    frame_index=int(block.get("frame_index", step_index)),
                    tvi=history_tvi[block_index],
                    image=image,
                    is_cached_history=has_cached_history,
                )
            )
    elif cached_history is not None:
        cached_array = np.asarray(cached_history)
        if cached_array.ndim == 3:
            for block_index in range(int(cached_array.shape[0])):
                if block_index < len(cached_mask) and not bool(cached_mask[block_index]):
                    continue
                camera_name = required_cameras[block_index % len(required_cameras)]
                if block_index >= len(history_tvi):
                    raise ValueError(f"ordinary history block {block_index} is missing its TVI row")
                history_records.append(
                    _HistoryVisualRecord(
                        step_index=block_index,
                        camera_index=camera_order[camera_name],
                        block_index=block_index,
                        camera_name=camera_name,
                        frame_index=block_index,
                        tvi=history_tvi[block_index],
                        image=None,
                        is_cached_history=True,
                    )
                )
    history_records = sorted(
        history_records,
        key=lambda record: (record.step_index, record.camera_index, record.block_index),
    )
    if len(history_records) >= 2 and history_shuffle_probability > 0.0:
        if torch.rand((), generator=generator).item() < history_shuffle_probability:
            order = torch.randperm(len(history_records), generator=generator).tolist()
            history_records = [history_records[index] for index in order]
    for record in history_records:
        if not record.is_cached_history:
            online_images.append(record.image)
        block_tvi = np.asarray(record.tvi, dtype=np.float32).copy()
        blocks.append(
            {
                "is_history": True,
                "is_cached_history": bool(record.is_cached_history),
                "cached_history_index": int(record.block_index),
                "camera_name": record.camera_name,
                "frame_index": int(record.frame_index),
                "tvi": block_tvi,
                "time": float(block_tvi[0]),
                "phi": float(block_tvi[yaw_index]),
                "viewpoint_id": CAMERA_VIEWPOINT_TO_ID.get(record.camera_name, 0),
                "sample": sample,
            }
        )

    current_tvi = as_numpy_tvi(sample.get("current_tvi"), tvi_dim=tvi_dim)
    present_current_cameras = [
        camera_name for camera_name in required_cameras if sample.get("images", {}).get(camera_name) is not None
    ]
    if int(tvi_dim) == 7 and len(current_tvi) != len(present_current_cameras):
        raise ValueError(
            f"current_tvi row count {len(current_tvi)} does not match "
            f"present current camera count {len(present_current_cameras)}"
        )
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
            _fallback_tvi_row(time=float(metadata.get("timestamp", 0.0)), tvi_dim=tvi_dim),
        )
        block_tvi = np.asarray(tvi, dtype=np.float32).copy()
        online_images.append(image)
        blocks.append(
            {
                "is_history": False,
                "is_cached_history": False,
                "camera_name": camera_name,
                "frame_index": int(metadata.get("frame_index", metadata.get("index", 0))),
                "tvi": block_tvi,
                "time": float(block_tvi[0]),
                "phi": float(block_tvi[yaw_index]),
                "viewpoint_id": CAMERA_VIEWPOINT_TO_ID.get(camera_name, 0),
                "sample": sample,
            }
        )
    return online_images, blocks


__all__ = [
    "HistoryAugmentationConfig",
    "action_model_accepts_padding_mask",
    "as_numpy_tvi",
    "build_navvla_cached_visual_sequence",
    "build_navvla_instruction",
    "forward_navvla_action",
    "history_augmentation_probabilities",
    "mask_history_tvi_embeddings",
    "predict_navvla_action",
    "sample_required_cameras",
    "samples_from_collated_batch",
    "scatter_image_embeddings",
    "target_visual_tokens_for_block",
]
