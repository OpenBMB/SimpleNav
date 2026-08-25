from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from tool.navvla.visual_token_cache import DEFAULT_MINICPM_V46_VISUAL_TOKEN_PROFILE

from .cache import Qwen35PooledHistoryTokenStore, VisualTokenBatch, open_visual_token_store
from .utils import as_list


TOKEN_STORE_CACHE_SIZE = 4


class NavVLACPMCollator:
    """Collate CPM samples while keeping visual-token stores out of samples."""

    def __init__(self) -> None:
        self._stores: OrderedDict[str, Any] = OrderedDict()

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("cannot collate an empty NavVLA CPM batch")
        collated = _collate_core(batch)
        visual_modes = {str((sample.get("metadata", {}) or {}).get("visual_token_mode")) for sample in batch}
        if len(visual_modes) != 1:
            raise ValueError(f"CPM batch mixes visual_token_mode values: {sorted(visual_modes)}")
        visual_mode = next(iter(visual_modes))
        visual_profiles = {
            str(
                (sample.get("metadata", {}) or {}).get(
                    "visual_token_profile", DEFAULT_MINICPM_V46_VISUAL_TOKEN_PROFILE
                )
            )
            for sample in batch
        }
        if visual_mode == "cached_history_online_current" and (len(visual_profiles) != 1 or not next(iter(visual_profiles))):
            raise ValueError(f"CPM cached-token batch must use one non-empty visual_token_profile: {sorted(visual_profiles)}")
        if visual_mode == "online_images":
            cameras = sorted({camera for sample in batch for camera in sample.get("history_images", {})})
            collated["history_images"] = {
                camera: [sample.get("history_images", {}).get(camera, []) for sample in batch]
                for camera in cameras
            }
        elif visual_mode == "cached_history_online_current":
            history_batches, history_grids, cache_stage, encoder_ckpt, storage_encoding = self._load_token_batches(
                batch, metadata_key="history_token_refs"
            )
            history_tokens, history_mask = _pad_token_batches(history_batches)
            collated.update(
                {
                    "history_cached_embeds": history_tokens,
                    "history_cached_deepstack_embeds": np.zeros(
                        (len(batch), 0, history_tokens.shape[1], *history_tokens.shape[2:]),
                        dtype=history_tokens.dtype,
                    ),
                    "history_cached_mask": history_mask,
                }
            )
            if cache_stage:
                collated["history_cached_grid_thw"] = _pad_grid_batches(
                    history_grids, max_length=history_tokens.shape[1]
                )
                collated["history_cached_cache_stage"] = [cache_stage] * len(batch)
                collated["history_cached_encoder_ckpt"] = [encoder_ckpt] * len(batch)
                collated["history_cached_storage_encoding"] = [storage_encoding] * len(batch)
        else:
            raise ValueError(f"unsupported visual_token_mode={visual_mode!r}")

        if any("long_memory_source_tvi" in sample for sample in batch):
            long_batches, long_grids, long_cache_stage, long_encoder_ckpt, long_storage_encoding = self._load_token_batches(
                batch, metadata_key="long_memory_token_refs"
            )
            source_tokens, source_present = _pad_token_batches(long_batches)
            source_mask = np.zeros_like(source_present)
            for batch_index, sample in enumerate(batch):
                length = int(long_batches[batch_index].shape[0])
                if not length:
                    continue
                configured = as_list(sample["metadata"].get("long_memory_mask", []))
                source_mask[batch_index, :length] = (
                    np.asarray(configured[:length], dtype=bool) if configured else True
                )
            collated["long_memory_source_tokens"] = source_tokens
            collated["long_memory_source_mask"] = source_mask
            collated["long_memory_source_tvi"] = _pad_tvi(
                [sample.get("long_memory_source_tvi") for sample in batch],
                max_length=source_tokens.shape[1],
            )
            if long_cache_stage:
                collated["long_memory_source_grid_thw"] = _pad_grid_batches(
                    long_grids, max_length=source_tokens.shape[1]
                )
                collated["long_memory_source_cache_stage"] = [long_cache_stage] * len(batch)
                collated["long_memory_source_encoder_ckpt"] = [long_encoder_ckpt] * len(batch)
                collated["long_memory_source_storage_encoding"] = [long_storage_encoding] * len(batch)
        return collated

    def _load_token_batches(
        self, batch: list[dict[str, Any]], *, metadata_key: str
    ) -> tuple[list[np.ndarray], list[np.ndarray | None], str, str, str]:
        outputs: list[VisualTokenBatch | None] = [None] * len(batch)
        groups: dict[tuple[str, str], tuple[list[int], list[list[str]]]] = {}
        for batch_index, sample in enumerate(batch):
            metadata = sample.get("metadata", {}) or {}
            root = str(metadata.get("dataset_root", "")).strip()
            if not root:
                raise KeyError("CPM sample metadata is missing dataset_root")
            profile = str(
                metadata.get("visual_token_profile", DEFAULT_MINICPM_V46_VISUAL_TOKEN_PROFILE)
            ).strip()
            if not profile:
                raise KeyError("CPM sample metadata is missing visual_token_profile")
            indices, refs = groups.setdefault((root, profile), ([], []))
            indices.append(batch_index)
            refs.append([str(ref) for ref in as_list(metadata.get(metadata_key, []))])
        for (root, profile), (indices, ref_batches) in groups.items():
            store_key = f"{root}\0{profile}"
            store = self._stores.pop(store_key, None)
            if store is None:
                store = open_visual_token_store(Path(root), profile=profile)
            self._stores[store_key] = store
            while len(self._stores) > TOKEN_STORE_CACHE_SIZE:
                _old_root, old_store = self._stores.popitem(last=False)
                old_store.close()
            records = (
                store.load_ref_record_batches(ref_batches)
                if isinstance(store, Qwen35PooledHistoryTokenStore)
                else [VisualTokenBatch(tokens=tokens) for tokens in store.load_ref_batches(ref_batches)]
            )
            for batch_index, record in zip(indices, records, strict=True):
                outputs[batch_index] = record
        if any(value is None for value in outputs):
            raise RuntimeError("failed to load one or more CPM token batches")
        records = [value for value in outputs if value is not None]
        stages = {record.cache_stage for record in records if record.cache_stage}
        encoder_ckpts = {record.encoder_ckpt for record in records if record.encoder_ckpt}
        storage_encodings = {record.storage_encoding for record in records}
        if len(stages) > 1:
            raise ValueError(f"CPM batch mixes visual cache stages: {sorted(stages)}")
        if len(encoder_ckpts) > 1:
            raise ValueError(f"CPM batch mixes visual cache encoders: {sorted(encoder_ckpts)}")
        if len(storage_encodings) > 1:
            raise ValueError(f"CPM batch mixes visual cache storage encodings: {sorted(storage_encodings)}")
        return (
            [record.tokens for record in records],
            [record.grid_thw for record in records],
            next(iter(stages), ""),
            next(iter(encoder_ckpts), ""),
            next(iter(storage_encodings), ""),
        )


_DEFAULT_COLLATOR = NavVLACPMCollator()


def collate_navvla_cpm_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return _DEFAULT_COLLATOR(batch)


def _collate_core(batch: list[dict[str, Any]]) -> dict[str, Any]:
    all_cameras = sorted({camera for sample in batch for camera in sample["images"]})
    images = {camera: [sample["images"].get(camera) for sample in batch] for camera in all_cameras}
    max_history = max((int(sample["history_tvi"].shape[0]) for sample in batch), default=0)
    output: dict[str, Any] = {
        "images": images,
        "image_masks": {
            camera: np.asarray([sample["images"].get(camera) is not None for sample in batch], dtype=bool)
            for camera in all_cameras
        },
        "current_tvi": [sample["current_tvi"] for sample in batch],
        "history_tokens": np.zeros((len(batch), max_history, 1, 3), dtype=np.float32),
        "history_tvi": _pad_tvi([sample["history_tvi"] for sample in batch], max_length=max_history),
        "history_mask": _pad_bool([sample["history_mask"] for sample in batch], max_length=max_history),
        "lang": [sample["lang"] for sample in batch],
        "platform_text": [sample["platform_text"] for sample in batch],
        "action": np.stack([sample["action"] for sample in batch], axis=0),
        "action_padding_mask": np.stack([sample["action_padding_mask"] for sample in batch], axis=0),
        "distance_to_goal": np.asarray([sample["distance_to_goal"] for sample in batch], dtype=np.float32),
        "qa_target": [sample["qa_target"] for sample in batch],
        "metadata": [sample["metadata"] for sample in batch],
    }
    if any("state" in sample for sample in batch):
        max_state = max((int(sample["state"].shape[0]) for sample in batch if "state" in sample), default=0)
        state = np.zeros((len(batch), max_state), dtype=np.float32)
        padding = np.ones((len(batch), max_state), dtype=bool)
        present = np.zeros((len(batch),), dtype=bool)
        for batch_index, sample in enumerate(batch):
            if "state" not in sample:
                continue
            length = int(sample["state"].shape[0])
            present[batch_index] = True
            if length:
                state[batch_index, -length:] = sample["state"]
                padding[batch_index, -length:] = False
        output.update({"state": state, "state_padding_mask": padding, "state_present": present})
    return output


def _pad_token_batches(batches: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    max_length = max((int(value.shape[0]) for value in batches), default=0)
    token_shape = next((tuple(value.shape[1:]) for value in batches if value.shape[0]), None)
    if token_shape is None:
        token_shape = next((tuple(value.shape[1:]) for value in batches), (4, 0))
    dtype = next((value.dtype for value in batches), np.dtype(np.float16))
    output = np.zeros((len(batches), max_length, *token_shape), dtype=dtype)
    mask = np.zeros((len(batches), max_length), dtype=bool)
    for batch_index, value in enumerate(batches):
        length = int(value.shape[0])
        if length:
            output[batch_index, :length] = value
            mask[batch_index, :length] = True
    return output, mask


def _pad_grid_batches(values: list[np.ndarray | None], *, max_length: int) -> np.ndarray:
    output = np.zeros((len(values), max_length, 3), dtype=np.int64)
    for batch_index, value in enumerate(values):
        if value is None:
            raise ValueError("spatial visual-token batches require grid_thw metadata")
        array = np.asarray(value, dtype=np.int64).reshape(-1, 3)
        length = min(int(array.shape[0]), max_length)
        output[batch_index, :length] = array[:length]
    return output


def _pad_tvi(values: list[np.ndarray | None], *, max_length: int) -> np.ndarray:
    if max_length < 0:
        raise ValueError(f"max_length must be non-negative, got {max_length}")
    arrays: list[np.ndarray | None] = []
    widths: set[int] = set()
    for value in values:
        if value is None:
            arrays.append(None)
            continue
        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError(f"TVI arrays must have rank 2, got shape {tuple(array.shape)}")
        arrays.append(array)
        widths.add(int(array.shape[1]))
    if not widths:
        raise ValueError("cannot pad TVI values without a known TVI feature width")
    if len(widths) != 1:
        raise ValueError(f"inconsistent TVI feature widths: {sorted(widths)}")
    width = next(iter(widths))
    output = np.zeros((len(values), max_length, width), dtype=np.float32)
    for batch_index, array in enumerate(arrays):
        if array is None:
            continue
        length = min(int(array.shape[0]), max_length)
        output[batch_index, :length] = array[:length]
    return output


def _pad_bool(values: list[np.ndarray], *, max_length: int) -> np.ndarray:
    output = np.zeros((len(values), max_length), dtype=bool)
    for batch_index, value in enumerate(values):
        length = min(int(value.shape[0]), max_length)
        output[batch_index, :length] = value[:length]
    return output
