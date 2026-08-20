from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from rich.progress import Progress
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tool.navvla.validation import validate_navvla_lerobot_dataset
from tool.navvla.context_index import (
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    available_context_token_budgets,
    iter_context_refs,
)
from tool.navvla.visual_token_cache import (
    DEFAULT_MINICPM_V46_VISUAL_TOKEN_PROFILE,
    MMAP_NPY_VISUAL_TOKEN_FORMAT,
    MINICPM_V46_VISUAL_HEAD,
    MMapNpyProfileShardWriter,
    NPZ_VISUAL_TOKEN_FORMAT,
    PROFILE_VISUAL_TOKEN_INDEX_COLUMNS,
    QWEN35_POOLED_HISTORY_CACHE_STAGE,
    QWEN35_POOLED_HISTORY_VISUAL_HEAD,
    VisualTokenProfile,
    profile_cache_root,
    stable_ref_hash,
    write_profile_index,
    write_profile_manifest,
    write_profile_token_record,
)
from starVLA.model.modules.qwen35_vision import (
    BFLOAT16_BITS_STORAGE_ENCODING,
    bf16_to_numpy_bits,
    configure_qwen35_processor,
    encode_qwen35_postmerge_one_by_one,
    pool_qwen35_postmerge,
)


class _SuppressProcessorKwargsWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict" not in record.getMessage()


logging.getLogger("transformers").addFilter(_SuppressProcessorKwargsWarning())


ImageBatch = tuple[list[str], list[dict[str, Any]], list[Image.Image]]
_PREFETCH_DONE = object()


class _PrefetchError:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.traceback = exc.__traceback__


class LazyEncoderUnavailable(RuntimeError):
    pass


class _ProgressReporter:
    """Emit bounded progress bars that remain useful after stdout is redirected."""

    def __init__(self, description: str, *, total: int) -> None:
        self.description = str(description)
        self.total = max(0, int(total))
        self.completed = 0
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at
        self.next_percent = 5
        self._report()

    def advance(self, amount: int) -> None:
        self.completed = min(self.total, self.completed + max(0, int(amount)))
        now = time.monotonic()
        percent = 100 if self.total == 0 else int(100 * self.completed / self.total)
        if self.completed == self.total or percent >= self.next_percent or now - self.last_report_at >= 30.0:
            self._report(now=now)
            while self.next_percent <= percent:
                self.next_percent += 5

    def _report(self, *, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        elapsed = now - self.started_at
        percent = 100 if self.total == 0 else int(100 * self.completed / self.total)
        filled = min(20, percent // 5)
        bar = "#" * filled + "-" * (20 - filled)
        eta = "?"
        if self.completed > 0 and self.completed < self.total:
            eta = f"{elapsed * (self.total - self.completed) / self.completed:.0f}s"
        print(
            f"{self.description}: [{bar}] {percent:3d}% "
            f"({self.completed}/{self.total}) elapsed={elapsed:.0f}s eta={eta}",
            flush=True,
        )
        self.last_report_at = now


def find_missing_cache_refs(refs: Iterable[str], *, existing_refs: set[str]) -> list[str]:
    seen: set[str] = set()
    missing: list[str] = []
    for ref in refs:
        ref = str(ref)
        if ref in seen:
            continue
        seen.add(ref)
        if ref not in existing_refs:
            missing.append(ref)
    return missing


def load_history_refs(dataset_root: Path, *, token_budget: int | None = DEFAULT_CONTEXT_TOKEN_BUDGET) -> list[str]:
    dataset_root = Path(dataset_root)
    budgets = [int(token_budget)] if token_budget is not None else available_context_token_budgets(dataset_root)
    refs: list[str] = []
    seen: set[str] = set()
    for budget in budgets:
        for ref in iter_context_refs(dataset_root, token_budget=budget):
            if ref in seen:
                continue
            seen.add(ref)
            refs.append(ref)
    return refs


def load_visual_cache_refs(dataset_root: Path, *, camera_names: Iterable[str] | None = None) -> list[str]:
    """Return every available source video frame once, independent of BATS context."""
    dataset_root = Path(dataset_root)
    data_paths = sorted((dataset_root / "data").glob("chunk-*/part-*.parquet"))
    episode_paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/part-*.parquet"))
    if not data_paths or not episode_paths:
        raise FileNotFoundError(f"missing data or episode parquet shards under {dataset_root}")
    data = pd.concat(
        [pd.read_parquet(path, columns=["index", "episode_index", "frame_index"]) for path in data_paths],
        ignore_index=True,
    )
    episodes = pd.concat(
        [pd.read_parquet(path, columns=["episode_index", "episode_id"]) for path in episode_paths],
        ignore_index=True,
    )
    video_index_path = dataset_root / "meta" / "navvla_video_index.parquet"
    video_index = pd.read_parquet(video_index_path, columns=["index", "video_key", "available"])
    cameras = json.loads((dataset_root / "meta" / "navvla_cameras.json").read_text(encoding="utf-8"))
    selected_cameras = None if camera_names is None else {str(name) for name in camera_names}
    unknown_cameras = set() if selected_cameras is None else selected_cameras - set(cameras)
    if unknown_cameras:
        raise ValueError(f"unknown visual cache camera names: {sorted(unknown_cameras)}")
    camera_name_by_video_key = {
        str(camera["video_key"]): str(camera_name) for camera_name, camera in cameras.items()
    }
    episode_id_by_index = {
        int(row.episode_index): str(row.episode_id) for row in episodes.itertuples(index=False)
    }
    data_by_index = {
        int(row.index): (int(row.episode_index), int(row.frame_index)) for row in data.itertuples(index=False)
    }
    refs: list[str] = []
    for row in video_index.itertuples(index=False):
        if not bool(row.available):
            continue
        camera_name = camera_name_by_video_key.get(str(row.video_key))
        if selected_cameras is not None and camera_name not in selected_cameras:
            continue
        data_row = data_by_index.get(int(row.index))
        if camera_name is None or data_row is None:
            continue
        episode_index, frame_index = data_row
        refs.append(f"{episode_id_by_index[episode_index]}/{frame_index:06d}/{camera_name}")
    return refs


def iter_history_refs(dataset_root: Path, *, token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET) -> Iterable[str]:
    yield from iter_context_refs(Path(dataset_root), token_budget=int(token_budget))


def load_existing_refs(dataset_root: Path, profile_name: str) -> set[str]:
    return {str(row["ref"]) for row in load_existing_ref_rows_for_refs(dataset_root, profile_name, [])}


def load_existing_ref_rows_for_refs(dataset_root: Path, profile_name: str, refs: Iterable[str]) -> list[dict[str, object]]:
    """Recover existing cache rows from index.parquet and orphan token files."""
    dataset_root = Path(dataset_root)
    rows_by_ref: dict[str, dict[str, object]] = {}
    manifest_path = profile_cache_root(dataset_root, profile_name) / "manifest.json"
    profile_format = NPZ_VISUAL_TOKEN_FORMAT
    if manifest_path.exists():
        profile_format = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("file_format", NPZ_VISUAL_TOKEN_FORMAT))
    shard_exists: dict[str, bool] = {}
    for row in _read_existing_index_rows(dataset_root, profile_name, shard_exists=shard_exists):
        ref = str(row.get("ref", ""))
        if ref:
            rows_by_ref[ref] = row

    if profile_format == MMAP_NPY_VISUAL_TOKEN_FORMAT:
        for row in _read_mmap_checkpoint_index_rows(dataset_root, profile_name, shard_exists=shard_exists):
            ref = str(row.get("ref", ""))
            if ref and ref not in rows_by_ref:
                rows_by_ref[ref] = row
        for row in _read_distributed_rank_index_rows(dataset_root, profile_name, shard_exists=shard_exists):
            ref = str(row.get("ref", ""))
            if ref and ref not in rows_by_ref:
                rows_by_ref[ref] = row
        return list(rows_by_ref.values())

    token_root = profile_cache_root(dataset_root, profile_name) / "tokens"
    if token_root.exists():
        existing_hashes = {path.stem for path in token_root.glob("*.npz")}
        seen_refs = set(rows_by_ref)
        for ref in refs:
            ref = str(ref)
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            token_hash = stable_ref_hash(ref)
            if token_hash in existing_hashes:
                token_path = token_root / f"{token_hash}.npz"
                rows_by_ref[ref] = {"ref": ref, "path": token_path.relative_to(dataset_root).as_posix()}
    return list(rows_by_ref.values())


def extract_qwen3_visual_tokens(image: Image.Image, *, encoder: Any) -> tuple[Any, Any]:
    image_embeds, deepstack_embeds = encoder.get_image_features(image)
    return image_embeds, deepstack_embeds


def extract_qwen3_visual_tokens_batch(images: list[Image.Image], *, encoder: Any) -> list[tuple[Any, Any]]:
    if hasattr(encoder, "get_image_features_batch"):
        return list(encoder.get_image_features_batch(images))
    return [extract_qwen3_visual_tokens(image, encoder=encoder) for image in images]


def unpack_visual_token_output(output: Any) -> tuple[Any, Any, dict[str, Any]]:
    if not isinstance(output, (tuple, list)) or len(output) not in (2, 3):
        raise TypeError("visual token encoder output must be (image_embeds, deepstack_embeds[, metadata])")
    image_embeds, deepstack_embeds = output[:2]
    metadata = {} if len(output) == 2 else dict(output[2] or {})
    return image_embeds, deepstack_embeds, metadata


def load_qwen3_encoder(*, encoder_ckpt: str, profile: VisualTokenProfile) -> Any:
    return Qwen3VisualTokenEncoder(encoder_ckpt=encoder_ckpt, profile=profile)


def load_minicpm_v46_encoder(*, encoder_ckpt: str, profile: VisualTokenProfile) -> Any:
    return MiniCPMV46VisualTokenEncoder(encoder_ckpt=encoder_ckpt, profile=profile)


def load_visual_encoder(*, encoder_ckpt: str, profile: VisualTokenProfile) -> Any:
    if profile.visual_head == MINICPM_V46_VISUAL_HEAD:
        return load_minicpm_v46_encoder(encoder_ckpt=encoder_ckpt, profile=profile)
    if profile.visual_head == QWEN35_POOLED_HISTORY_VISUAL_HEAD:
        return Qwen35PooledHistoryVisualTokenEncoder(encoder_ckpt=encoder_ckpt, profile=profile)
    return load_qwen3_encoder(encoder_ckpt=encoder_ckpt, profile=profile)


class Qwen35PooledHistoryVisualTokenEncoder:
    def __init__(self, *, encoder_ckpt: str, profile: VisualTokenProfile) -> None:
        if not encoder_ckpt:
            raise LazyEncoderUnavailable("--encoder-ckpt is required for Qwen3.5 visual cache generation")
        if Qwen3_5ForConditionalGeneration is None:
            raise LazyEncoderUnavailable("Qwen3.5 cache generation requires transformers with Qwen3_5 support")
        if profile.cache_stage != QWEN35_POOLED_HISTORY_CACHE_STAGE:
            raise ValueError(
                f"Qwen3.5 pooled-history encoder requires cache_stage={QWEN35_POOLED_HISTORY_CACHE_STAGE!r}"
            )
        self.profile = profile
        if not torch.cuda.is_available():
            raise LazyEncoderUnavailable("Qwen3.5 FlashAttention 2 cache generation requires CUDA")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device("cuda", local_rank)
        torch.cuda.set_device(self.device)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            encoder_ckpt,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(encoder_ckpt)
        if profile.input_resize is None:
            raise ValueError("Qwen3.5 cache profile requires input_resize")
        configure_qwen35_processor(self.processor, profile.input_resize)
        visual = self.model.model.visual
        if profile.patch_size is not None and int(profile.patch_size) != int(visual.patch_size):
            raise ValueError(f"cache profile patch_size={profile.patch_size} does not match checkpoint={visual.patch_size}")
        if profile.spatial_merge_size is not None and int(profile.spatial_merge_size) != int(visual.spatial_merge_size):
            raise ValueError(
                f"cache profile spatial_merge_size={profile.spatial_merge_size} does not match checkpoint={visual.spatial_merge_size}"
            )

    @torch.inference_mode()
    def get_image_features(self, image: Image.Image):
        return self.get_image_features_batch([image])[0]

    @torch.inference_mode()
    def get_image_features_batch(self, images: list[Image.Image]):
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
        ).to(self.device)
        grid_thw = inputs["image_grid_thw"]
        chunks = encode_qwen35_postmerge_one_by_one(
            self.model.model.visual,
            inputs["pixel_values"].to(dtype=self.model.model.visual.dtype),
            grid_thw,
        )
        merge_size = int(self.model.model.visual.spatial_merge_size)
        outputs = []
        for chunk, grid in zip(chunks, grid_thw, strict=True):
            grid_values = [int(value) for value in grid.tolist()]
            expected_postmerge = int(grid.prod().item()) // (merge_size**2)
            if int(chunk.shape[0]) != expected_postmerge:
                raise ValueError(
                    f"Qwen3.5 merger output for grid={grid_values} must contain {expected_postmerge} tokens, "
                    f"got {int(chunk.shape[0])}"
                )
            pooled = pool_qwen35_postmerge(
                chunk,
                grid,
                target_tokens=int(self.profile.token_count),
                spatial_merge_size=merge_size,
            )
            outputs.append(
                (
                    bf16_to_numpy_bits(pooled),
                    None,
                    {
                        "grid_t": grid_values[0],
                        "grid_h": grid_values[1],
                        "grid_w": grid_values[2],
                        "cache_stage": QWEN35_POOLED_HISTORY_CACHE_STAGE,
                    },
                )
            )
        return outputs


class Qwen3VisualTokenEncoder:
    def __init__(self, *, encoder_ckpt: str, profile: VisualTokenProfile) -> None:
        if not encoder_ckpt:
            raise LazyEncoderUnavailable("--encoder-ckpt is required for Qwen3 visual cache generation")
        self.profile = profile
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            self.device = torch.device("cuda", local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            encoder_ckpt,
            dtype=dtype,
            attn_implementation="sdpa",
            ignore_mismatched_sizes=True,
        ).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(encoder_ckpt)

    @torch.inference_mode()
    def get_image_features(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
        return self.get_image_features_batch([image])[0]

    @torch.inference_mode()
    def get_image_features_batch(self, images: list[Image.Image]) -> list[tuple[np.ndarray, np.ndarray | None]]:
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
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        image_grid_thw = inputs["image_grid_thw"]
        pixel_values = inputs["pixel_values"]
        image_features, deepstack_features = self.model.model.get_image_features(pixel_values, image_grid_thw)
        merge_size = int(self.model.model.visual.spatial_merge_size)
        chunks = _split_qwen_image_features_by_grid(image_features, image_grid_thw, merge_size=merge_size)
        deepstack_chunks_by_layer = []
        if deepstack_features is not None and self.profile.has_deepstack:
            for layer_features in list(deepstack_features)[: int(self.profile.deepstack_layers)]:
                deepstack_chunks_by_layer.append(
                    _split_qwen_image_features_by_grid(layer_features, image_grid_thw, merge_size=merge_size)
                )
        outputs: list[tuple[np.ndarray, np.ndarray | None]] = []
        for image_index, chunk in enumerate(chunks):
            pooled = _pool_visual_tokens_by_grid(
                chunk,
                image_grid_thw[image_index],
                target_tokens=int(self.profile.token_count),
                merge_size=merge_size,
            )
            deepstack_pooled = None
            if deepstack_chunks_by_layer:
                layers = [
                    _pool_visual_tokens_by_grid(
                        layer_chunks[image_index],
                        image_grid_thw[image_index],
                        target_tokens=int(self.profile.token_count),
                        merge_size=merge_size,
                    )
                    for layer_chunks in deepstack_chunks_by_layer
                ]
                deepstack_pooled = torch.stack(layers, dim=0) if layers else None
            outputs.append(
                (
                    pooled.detach().to("cpu", dtype=torch.float32).numpy(),
                    None if deepstack_pooled is None else deepstack_pooled.detach().to("cpu", dtype=torch.float32).numpy(),
                )
            )
        return outputs


class MiniCPMV46VisualTokenEncoder:
    def __init__(self, *, encoder_ckpt: str, profile: VisualTokenProfile) -> None:
        if not encoder_ckpt:
            raise LazyEncoderUnavailable("--encoder-ckpt is required for MiniCPM visual cache generation")
        self.profile = profile
        self.downsample_mode = "16x"
        self.max_slice_nums = 36
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            self.device = torch.device("cuda", local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        model_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            "trust_remote_code": True,
        }
        self.model = AutoModelForImageTextToText.from_pretrained(encoder_ckpt, **model_kwargs).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(encoder_ckpt, trust_remote_code=True)
        self.processor.downsample_mode = self.downsample_mode
        self.processor.max_slice_nums = self.max_slice_nums

    def _resolve_image_feature_getter(self):
        for candidate in (self.model, getattr(self.model, "model", None)):
            if candidate is None:
                continue
            getter = getattr(candidate, "get_image_features", None)
            if getter is not None:
                return getter
        raise RuntimeError(
            "MiniCPM checkpoint does not expose get_image_features(); "
            "requires transformers with MiniCPM-V-4.6 support."
        )

    @torch.inference_mode()
    def get_image_features(self, image: Image.Image) -> tuple[np.ndarray, None]:
        return self.get_image_features_batch([image])[0]

    @torch.inference_mode()
    def get_image_features_batch(self, images: list[Image.Image]) -> list[tuple[np.ndarray, None]]:
        try:
            return self._get_image_features_batch(images)
        except BaseException as exc:
            if not _is_torch_cuda_oom(exc) or len(images) <= 1:
                raise
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            mid = max(1, len(images) // 2)
            print(
                f"MiniCPM visual forward OOM for batch_size={len(images)}; "
                f"retrying as {mid}+{len(images) - mid}",
                flush=True,
            )
            return self.get_image_features_batch(images[:mid]) + self.get_image_features_batch(images[mid:])

    def _get_image_features_batch(self, images: list[Image.Image]) -> list[tuple[np.ndarray, None]]:
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
            return_dict=True,
            return_tensors="pt",
            downsample_mode=self.downsample_mode,
            max_slice_nums=self.max_slice_nums,
        ).to(self.device)
        getter = self._resolve_image_feature_getter()
        pixel_values = inputs["pixel_values"].to(self.device)
        target_sizes = inputs.get("target_sizes")
        if target_sizes is not None:
            target_sizes = target_sizes.to(self.device)
        try:
            vision_output = getter(pixel_values, target_sizes, downsample_mode=self.downsample_mode)
        except TypeError:
            vision_output = getter(pixel_values, target_sizes)
        pooler_output = getattr(vision_output, "pooler_output", vision_output)
        chunks = _split_minicpm_pooler_output(pooler_output, batch_size=len(images))
        outputs: list[tuple[np.ndarray, None]] = []
        for image_index, chunk in enumerate(chunks):
            grid_height, grid_width = _minicpm_feature_grid_shape(
                int(chunk.shape[0]),
                target_size=None if target_sizes is None else target_sizes[image_index],
                downsample_mode=self.downsample_mode,
            )
            pooled = _pool_minicpm_visual_tokens(
                chunk,
                target_tokens=int(self.profile.token_count),
                grid_height=grid_height,
                grid_width=grid_width,
            )
            outputs.append((pooled.detach().to("cpu", dtype=torch.float32).numpy(), None))
        return outputs


def _is_torch_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return "cuda" in message and "out of memory" in message
    return False


def _split_minicpm_pooler_output(pooler_output: Any, *, batch_size: int) -> list[torch.Tensor]:
    if isinstance(pooler_output, (list, tuple)):
        chunks = [_flatten_minicpm_feature_chunk(chunk) for chunk in pooler_output]
        return _group_minicpm_feature_chunks(chunks, batch_size=batch_size)
    if not isinstance(pooler_output, torch.Tensor):
        raise TypeError(f"unsupported MiniCPM pooler_output type: {type(pooler_output)!r}")
    if pooler_output.ndim == 3:
        chunks = [_flatten_minicpm_feature_chunk(chunk) for chunk in pooler_output]
        return _group_minicpm_feature_chunks(chunks, batch_size=batch_size)
    if pooler_output.ndim == 2 and int(batch_size) == 1:
        return [_flatten_minicpm_feature_chunk(pooler_output)]
    raise ValueError(f"cannot split MiniCPM pooler_output shape {tuple(pooler_output.shape)} for batch size {batch_size}")


def _group_minicpm_feature_chunks(chunks: list[torch.Tensor], *, batch_size: int) -> list[torch.Tensor]:
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if len(chunks) == batch_size:
        return chunks
    if len(chunks) % batch_size != 0:
        raise ValueError(f"MiniCPM feature chunks {len(chunks)} do not divide batch size {batch_size}")
    chunks_per_image = len(chunks) // batch_size
    if chunks_per_image <= 0:
        raise ValueError(f"MiniCPM feature chunks {len(chunks)} do not match batch size {batch_size}")
    return [
        torch.cat(chunks[start : start + chunks_per_image], dim=0)
        for start in range(0, len(chunks), chunks_per_image)
    ]


def _flatten_minicpm_feature_chunk(chunk: torch.Tensor) -> torch.Tensor:
    if chunk.ndim == 3 and int(chunk.shape[0]) == 1:
        chunk = chunk.squeeze(0)
    if chunk.ndim != 2:
        raise ValueError(f"MiniCPM feature chunk must have shape [tokens, hidden], got {tuple(chunk.shape)}")
    return chunk


def _minicpm_feature_grid_shape(
    token_count: int,
    *,
    target_size: torch.Tensor | None,
    downsample_mode: str,
) -> tuple[int, int]:
    if target_size is not None and int(target_size.numel()) >= 2:
        height, width = [max(1, int(value)) for value in target_size.reshape(-1)[:2].tolist()]
        for factor in (1, {"4x": 2, "16x": 4}.get(downsample_mode, 1)):
            grid_height = max(1, height // factor)
            grid_width = max(1, width // factor)
            if grid_height * grid_width == token_count:
                return grid_height, grid_width
        aspect_ratio = float(height) / float(width)
    else:
        aspect_ratio = 1.0

    grid_height = max(1, int(round(math.sqrt(float(token_count) * aspect_ratio))))
    while grid_height > 1 and token_count % grid_height != 0:
        grid_height -= 1
    return grid_height, token_count // grid_height


def _pool_minicpm_visual_tokens(
    visual_tokens: torch.Tensor,
    *,
    target_tokens: int,
    grid_height: int,
    grid_width: int,
) -> torch.Tensor:
    target_len = max(1, int(target_tokens))
    if int(visual_tokens.shape[0]) == target_len:
        return visual_tokens
    if int(grid_height) * int(grid_width) != int(visual_tokens.shape[0]):
        raise ValueError(
            f"MiniCPM visual grid {grid_height}x{grid_width} does not match {int(visual_tokens.shape[0])} tokens"
        )
    target_height = max(1, int(round(math.sqrt(float(target_len) * float(grid_height) / float(grid_width)))))
    while target_height > 1 and target_len % target_height != 0:
        target_height -= 1
    target_width = max(1, int(math.ceil(float(target_len) / float(target_height))))
    pooled = F.adaptive_avg_pool2d(
        visual_tokens.view(1, grid_height, grid_width, visual_tokens.shape[-1]).permute(0, 3, 1, 2).float(),
        (target_height, target_width),
    )
    return pooled.to(dtype=visual_tokens.dtype).permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)[:target_len]


def _split_qwen_image_features_by_grid(
    image_features: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    grid: torch.Tensor,
    *,
    merge_size: int = 2,
) -> list[torch.Tensor]:
    if isinstance(image_features, (tuple, list)):
        if len(image_features) != int(grid.shape[0]):
            raise ValueError(f"Qwen image feature chunks {len(image_features)} do not match image_grid_thw rows {int(grid.shape[0])}")
        return list(image_features)
    split_sizes = (grid.prod(-1) // int(merge_size) ** 2).tolist()
    return list(torch.split(image_features, split_sizes))


def _pool_visual_tokens_by_grid(
    visual_tokens: torch.Tensor,
    original_grid: torch.Tensor,
    *,
    target_tokens: int,
    merge_size: int = 2,
) -> torch.Tensor:
    temporal, height, width = [int(value) for value in original_grid.tolist()]
    merge = max(1, int(merge_size))
    token_h = max(1, height // merge)
    token_w = max(1, width // merge)
    target_len = max(1, int(target_tokens))
    if int(visual_tokens.shape[0]) == target_len:
        return visual_tokens
    if temporal == 1 and token_h * token_w == visual_tokens.shape[0]:
        target_h = max(1, int(round(np.sqrt(float(target_len) * float(token_h) / float(max(1, token_w))))))
        while target_h > 1 and target_len % target_h != 0:
            target_h -= 1
        target_w = max(1, int(np.ceil(float(target_len) / float(target_h))))
        pooled = F.adaptive_avg_pool2d(
            visual_tokens.view(1, token_h, token_w, visual_tokens.shape[-1]).permute(0, 3, 1, 2).float(),
            (target_h, target_w),
        )
        return pooled.to(dtype=visual_tokens.dtype).permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)[:target_len]
    pooled = F.adaptive_avg_pool1d(visual_tokens.transpose(0, 1).unsqueeze(0).float(), target_len)
    return pooled.to(dtype=visual_tokens.dtype).squeeze(0).transpose(0, 1)


def generate_profile_cache(
    dataset_root: Path,
    *,
    profile: VisualTokenProfile,
    refs: Iterable[str],
    encoder: Any,
    skip_existing: bool = False,
    progress_description: str | None = None,
    batch_size: int = 1,
    prefetch_batches: int = 2,
    input_resize: tuple[int, int] | None = None,
    shard_prefix: str = "image_embeds",
    mmap_flush_callback: Callable[[list[dict[str, object]]], None] | None = None,
    metadata_by_ref: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, object]]:
    existing_rows = load_existing_ref_rows_for_refs(dataset_root, profile.name, refs) if skip_existing else []
    existing_refs = {str(row["ref"]) for row in existing_rows}
    rows = list(existing_rows)
    missing_refs = find_missing_cache_refs(refs, existing_refs=existing_refs)
    if metadata_by_ref is None:
        data, episodes, video_index, cameras, info = _load_dataset_lookup_tables(dataset_root)
        lookup = build_ref_metadata_lookup(data=data, episodes=episodes, video_index=video_index)
        metadata_by_ref = {
            ref: resolve_ref_metadata(ref, cameras=cameras, lookup=lookup) for ref in missing_refs
        }
        lookup_report = (
            f"episodes={len(lookup['episode_by_id'])} frames={len(lookup['data_by_episode_frame'])} "
            f"video_rows={len(lookup['video_by_data_key'])}"
        )
    else:
        info = json.loads((Path(dataset_root) / "meta" / "info.json").read_text(encoding="utf-8"))
        missing_metadata_refs = [ref for ref in missing_refs if ref not in metadata_by_ref]
        if missing_metadata_refs:
            raise KeyError(f"pre-resolved metadata is missing {len(missing_metadata_refs)} cache refs")
        metadata_by_ref = {ref: metadata_by_ref[ref] for ref in missing_refs}
        lookup_report = "source=distributed_work_plan"
    print(
        f"prepared lookup tables: refs={len(missing_refs)} {lookup_report}",
        flush=True,
    )
    print(f"resolved metadata for {len(metadata_by_ref)} refs; ordering by video", flush=True)
    ordered_refs = order_refs_by_video(missing_refs, metadata_by_ref=metadata_by_ref)
    description = progress_description or f"visual-cache {profile.name}"
    progress_reporter = _ProgressReporter(description, total=len(ordered_refs))
    mmap_writer = (
        MMapNpyProfileShardWriter(
            dataset_root,
            profile=profile,
            shard_prefix=shard_prefix,
            on_flush=mmap_flush_callback,
        )
        if profile.file_format == MMAP_NPY_VISUAL_TOKEN_FORMAT
        else None
    )
    with Progress() as progress:
        task = progress.add_task(description, total=len(ordered_refs))
        for batch_refs, batch_metas, batch_images in iter_prefetched_batches(
            iter_video_ordered_image_batches(
                dataset_root,
                refs=ordered_refs,
                metadata_by_ref=metadata_by_ref,
                info=info,
                batch_size=batch_size,
                input_resize=input_resize,
            ),
            max_prefetch_batches=prefetch_batches,
        ):
            batch_outputs = extract_qwen3_visual_tokens_batch(batch_images, encoder=encoder)
            if len(batch_outputs) != len(batch_refs):
                raise ValueError(f"encoder batch output length {len(batch_outputs)} does not match input batch {len(batch_refs)}")
            for ref, meta, output in zip(batch_refs, batch_metas, batch_outputs):
                image_embeds, deepstack_embeds, output_metadata = unpack_visual_token_output(output)
                row_metadata = {
                    "episode_id": meta["episode_id"],
                    "trajectory_id": meta["trajectory_id"],
                    "frame_index": meta["frame_index"],
                    "source_frame_index": meta["source_frame_index"],
                    "data_index": meta["data_index"],
                    "camera_name": meta["camera_name"],
                    "video_key": meta["video_key"],
                    "dataset_name": meta["dataset_name"],
                    "split": meta["split"],
                    **output_metadata,
                }
                if mmap_writer is not None:
                    if deepstack_embeds is not None:
                        raise ValueError(f"profile {profile.name} mmap_npy cache does not support deepstack_embeds")
                    mmap_writer.add(ref=ref, image_embeds=np.asarray(image_embeds), metadata=row_metadata)
                else:
                    record = write_profile_token_record(
                        dataset_root,
                        profile=profile,
                        ref=ref,
                        image_embeds=np.asarray(image_embeds),
                        deepstack_embeds=None if deepstack_embeds is None else np.asarray(deepstack_embeds),
                    )
                    rows.append({"ref": record.ref, "path": record.path, **row_metadata})
            progress.advance(task, advance=len(batch_refs))
            progress_reporter.advance(len(batch_refs))
    if mmap_writer is not None:
        rows.extend(mmap_writer.close())
    return rows


def build_ref_metadata_lookup(
    *,
    data: pd.DataFrame,
    episodes: pd.DataFrame,
    video_index: pd.DataFrame,
) -> dict[str, dict[Any, dict[str, Any]]]:
    episode_by_split_key: dict[tuple[str, str], dict[str, Any]] = {}
    episode_by_id: dict[str, dict[str, Any]] = {}
    for row in episodes.to_dict("records"):
        episode_id = str(row["episode_id"])
        split = str(row.get("split", ""))
        episode_by_split_key[(episode_id, split)] = row
        episode_by_id.setdefault(episode_id, row)

    data_by_episode_frame: dict[tuple[int, int], dict[str, Any]] = {}
    for row in data.to_dict("records"):
        data_by_episode_frame[(int(row["episode_index"]), int(row["frame_index"]))] = row

    video_by_data_key: dict[tuple[int, str], dict[str, Any]] = {}
    for row in video_index.to_dict("records"):
        video_by_data_key[(int(row["index"]), str(row["video_key"]))] = row

    return {
        "episode_by_split_key": episode_by_split_key,
        "episode_by_id": episode_by_id,
        "data_by_episode_frame": data_by_episode_frame,
        "video_by_data_key": video_by_data_key,
    }


def resolve_ref_metadata(
    ref: str,
    *,
    data: pd.DataFrame | None = None,
    episodes: pd.DataFrame | None = None,
    video_index: pd.DataFrame | None = None,
    cameras: dict[str, Any],
    lookup: dict[str, dict[Any, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    parts = ref.split("/")
    if len(parts) == 3:
        dataset_name = ""
        split = ""
        episode_id, frame_index_text, camera_name = parts
    elif len(parts) == 5:
        dataset_name, split, episode_id, frame_index_text, camera_name = parts
    else:
        raise ValueError(f"history token ref must be episode_id/frame_index/camera_name, got {ref!r}")
    if not frame_index_text.isdigit():
        raise ValueError(f"history token ref frame_index must be zero-padded digits, got {ref!r}")
    frame_index = int(frame_index_text)
    if lookup is None:
        if data is None or episodes is None or video_index is None:
            raise ValueError("resolve_ref_metadata requires either lookup or data/episodes/video_index")
        lookup = build_ref_metadata_lookup(data=data, episodes=episodes, video_index=video_index)
    episode_row = (
        lookup["episode_by_split_key"].get((episode_id, split))
        if split
        else None
    ) or lookup["episode_by_id"].get(episode_id)
    if episode_row is None:
        raise KeyError(f"history token ref episode_id does not resolve: {ref}")
    if not split:
        split = str(episode_row.get("split", ""))
    episode_index = int(episode_row["episode_index"])
    data_row = lookup["data_by_episode_frame"].get((episode_index, frame_index))
    if data_row is None:
        raise KeyError(f"history token ref frame does not resolve: {ref}")
    if camera_name not in cameras:
        raise KeyError(f"history token ref camera does not resolve: {ref}")
    video_key = str(cameras[camera_name]["video_key"])
    video_row = lookup["video_by_data_key"].get((int(data_row["index"]), video_key))
    if video_row is None:
        raise KeyError(f"history token ref video row does not resolve: {ref}")
    if not bool(video_row["available"]):
        raise FileNotFoundError(f"history token ref has no available video frame: {ref}")
    return {
        "dataset_name": dataset_name,
        "split": split,
        "episode_id": episode_id,
        "trajectory_id": str(episode_row.get("trajectory_id") or episode_id),
        "frame_index": frame_index,
        "source_frame_index": int(data_row.get("source_frame_index", -1)),
        "data_index": int(data_row["index"]),
        "camera_name": camera_name,
        "video_key": video_key,
        "chunk_index": int(video_row.get("chunk_index", 0)),
        "file_index": int(video_row.get("file_index", 0)),
        "video_frame_index": int(video_row["video_frame_index"]),
    }


def read_ref_image(dataset_root: Path, *, meta: dict[str, Any], info: dict[str, Any]) -> Image.Image:
    path = video_path_for_meta(dataset_root, meta=meta, info=info)
    cap = _open_video_capture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(meta["video_frame_index"]))
        ok, frame = cap.read()
        if not ok:
            raise IndexError(f"failed to read frame {meta['video_frame_index']} from {path}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


def video_path_for_meta(dataset_root: Path, *, meta: dict[str, Any], info: dict[str, Any]) -> Path:
    pattern = info["video_path"].get(str(meta["video_key"]))
    if pattern is None:
        raise KeyError(f"video_path is missing key {meta['video_key']!r}")
    return dataset_root / pattern.format(chunk_index=int(meta["chunk_index"]), file_index=int(meta["file_index"]))


def _open_video_capture(path: Path, *, attempts: int = 3) -> cv2.VideoCapture:
    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        capture = cv2.VideoCapture(str(path))
        if capture.isOpened():
            return capture
        capture.release()
        if attempt + 1 < attempts:
            time.sleep(0.25 * (attempt + 1))
    raise FileNotFoundError(f"video does not open after {attempts} attempts: {path}")


def order_refs_by_video(refs: Iterable[str], *, metadata_by_ref: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        [str(ref) for ref in refs],
        key=lambda ref: (
            str(metadata_by_ref[ref]["video_key"]),
            int(metadata_by_ref[ref]["chunk_index"]),
            int(metadata_by_ref[ref]["file_index"]),
            int(metadata_by_ref[ref]["video_frame_index"]),
            ref,
        ),
    )


def iter_prefetched_batches(
    batches: Iterable[ImageBatch],
    *,
    max_prefetch_batches: int = 2,
) -> Iterable[ImageBatch]:
    if int(max_prefetch_batches) <= 0:
        yield from batches
        return

    batch_queue: queue.Queue[object] = queue.Queue(maxsize=max(1, int(max_prefetch_batches)))
    stop_event = threading.Event()

    def put_until_stopped(item: object) -> None:
        while not stop_event.is_set():
            try:
                batch_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def worker() -> None:
        try:
            for batch in batches:
                if stop_event.is_set():
                    break
                put_until_stopped(batch)
        except BaseException as exc:
            put_until_stopped(_PrefetchError(exc))
        finally:
            put_until_stopped(_PREFETCH_DONE)

    thread = threading.Thread(target=worker, name="navvla-visual-cache-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = batch_queue.get()
            if item is _PREFETCH_DONE:
                return
            if isinstance(item, _PrefetchError):
                raise item.exc.with_traceback(item.traceback)
            yield item  # type: ignore[misc]
    finally:
        stop_event.set()
        thread.join(timeout=1.0)


def iter_video_ordered_image_batches(
    dataset_root: Path,
    *,
    refs: list[str],
    metadata_by_ref: dict[str, dict[str, Any]],
    info: dict[str, Any],
    batch_size: int,
    input_resize: tuple[int, int] | None = None,
) -> Iterable[tuple[list[str], list[dict[str, Any]], list[Image.Image]]]:
    current_video: Path | None = None
    cap: cv2.VideoCapture | None = None
    current_pos: int | None = None
    batch_refs: list[str] = []
    batch_metas: list[dict[str, Any]] = []
    batch_images: list[Image.Image] = []

    def flush() -> tuple[list[str], list[dict[str, Any]], list[Image.Image]] | None:
        nonlocal batch_refs, batch_metas, batch_images
        if not batch_refs:
            return None
        payload = (batch_refs, batch_metas, batch_images)
        batch_refs, batch_metas, batch_images = [], [], []
        return payload

    try:
        for ref in refs:
            meta = metadata_by_ref[ref]
            video_path = video_path_for_meta(dataset_root, meta=meta, info=info)
            if current_video != video_path:
                payload = flush()
                if payload is not None:
                    yield payload
                if cap is not None:
                    cap.release()
                cap = _open_video_capture(video_path)
                current_video = video_path
                current_pos = 0

            assert cap is not None
            target_frame = int(meta["video_frame_index"])
            if current_pos is None or target_frame < current_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                current_pos = target_frame
            while current_pos < target_frame:
                ok, _ = cap.read()
                if not ok:
                    raise IndexError(f"failed to skip to frame {target_frame} from {video_path}")
                current_pos += 1
            ok, frame = cap.read()
            if not ok:
                raise IndexError(f"failed to read frame {target_frame} from {video_path}")
            current_pos = target_frame + 1
            batch_refs.append(ref)
            batch_metas.append(meta)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            batch_images.append(_resize_input_image(image, input_resize))
            if len(batch_refs) >= max(1, int(batch_size)):
                payload = flush()
                if payload is not None:
                    yield payload
        payload = flush()
        if payload is not None:
            yield payload
    finally:
        if cap is not None:
            cap.release()


def _parse_input_resize(value: str) -> tuple[int, int]:
    width_text, separator, height_text = str(value).lower().partition("x")
    if not separator:
        raise argparse.ArgumentTypeError("--input-resize must use WIDTHxHEIGHT, for example 597x336")
    try:
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--input-resize width and height must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("--input-resize width and height must be positive")
    return width, height


def _resize_input_image(image: Image.Image, size: tuple[int, int] | None) -> Image.Image:
    if size is None:
        return image
    width, height = size
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.BICUBIC)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate profile-based NavVLA visual token cache")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--profile", default=DEFAULT_MINICPM_V46_VISUAL_TOKEN_PROFILE)
    parser.add_argument("--visual-head", default=MINICPM_V46_VISUAL_HEAD)
    parser.add_argument("--encoder-name", default="MiniCPM-V-4.6")
    parser.add_argument("--encoder-ckpt", default="")
    parser.add_argument("--token-level", default="pooled_history")
    parser.add_argument("--token-count", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--file-format", default=MMAP_NPY_VISUAL_TOKEN_FORMAT)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--deepstack-layers", type=int, default=0)
    parser.add_argument("--validate-before", action="store_true")
    parser.add_argument("--validate-after", action="store_true")
    parser.add_argument("--token-budget", type=int, default=DEFAULT_CONTEXT_TOKEN_BUDGET)
    parser.add_argument("--all-token-budgets", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rebuild-index-only",
        action="store_true",
        help="For mmap_npy caches, rebuild index.parquet from existing shards/checkpoint indexes and exit.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=None,
        help="Cache only these camera names. By default every available dataset camera is cached.",
    )
    parser.add_argument(
        "--input-resize",
        type=_parse_input_resize,
        default=(448, 448),
        metavar="WIDTHxHEIGHT",
        help="Resize decoded frames before visual encoding (default: 448x448). Existing refs skipped by --skip-existing are not changed.",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=int(os.environ.get("NAVVLA_VISUAL_CACHE_PREFETCH_BATCHES", "2")),
        help="Number of decoded image batches to prefetch per rank; set 0 to disable.",
    )
    return parser.parse_args(argv)


def distributed_rank_index_path(dataset_root: Path, profile_name: str, *, rank: int) -> Path:
    return profile_cache_root(dataset_root, profile_name) / f"index.rank{int(rank):05d}.parquet"


def distributed_rank_work_plan_path(dataset_root: Path, profile_name: str, *, rank: int) -> Path:
    return profile_cache_root(dataset_root, profile_name) / f"work_plan.rank{int(rank):05d}.parquet"


def _write_distributed_rank_work_plans(
    dataset_root: Path,
    profile_name: str,
    *,
    rank_refs: list[list[str]],
    metadata_by_ref: dict[str, dict[str, Any]],
) -> None:
    for rank, refs in enumerate(rank_refs):
        rows = [{"ref": ref, **metadata_by_ref[ref]} for ref in refs]
        path = distributed_rank_work_plan_path(dataset_root, profile_name, rank=rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        try:
            frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ref"])
            frame.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)


def _read_distributed_rank_work_plan(dataset_root: Path, profile_name: str, *, rank: int) -> dict[str, dict[str, Any]]:
    path = distributed_rank_work_plan_path(dataset_root, profile_name, rank=rank)
    if not path.exists():
        raise FileNotFoundError(f"missing distributed work plan: {path}")
    rows = pd.read_parquet(path).to_dict("records")
    return {str(row.pop("ref")): row for row in rows}


def mmap_checkpoint_index_root(dataset_root: Path, profile_name: str) -> Path:
    return profile_cache_root(dataset_root, profile_name) / "checkpoint_indexes"


def mmap_checkpoint_index_path_for_shard(dataset_root: Path, profile_name: str, shard_path: str | Path) -> Path:
    shard_name = Path(str(shard_path)).stem
    return mmap_checkpoint_index_root(dataset_root, profile_name) / f"index.{shard_name}.parquet"


def _ordered_index_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered_rows: list[dict[str, object]] = []
    for row in rows:
        ordered = {column: row.get(column) for column in PROFILE_VISUAL_TOKEN_INDEX_COLUMNS}
        for key, value in row.items():
            if key not in ordered:
                ordered[key] = value
        ordered_rows.append(ordered)
    return ordered_rows


def profile_with_mmap_index_hidden_dim(profile: VisualTokenProfile, rows: list[dict[str, object]]) -> VisualTokenProfile:
    if profile.file_format != MMAP_NPY_VISUAL_TOKEN_FORMAT or int(profile.hidden_dim) > 0:
        return profile
    hidden_dims = {int(row["hidden_dim"]) for row in rows if row.get("hidden_dim") is not None}
    if not hidden_dims:
        return profile
    if len(hidden_dims) != 1:
        raise ValueError(f"mmap checkpoint/index rows disagree on hidden_dim: {sorted(hidden_dims)}")
    return VisualTokenProfile(**{**profile.__dict__, "hidden_dim": next(iter(hidden_dims))})


def _deduplicate_index_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    deduplicated: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        ref = str(row.get("ref", ""))
        if not ref or ref in seen:
            continue
        deduplicated.append(row)
        seen.add(ref)
    return deduplicated


def recover_mmap_profile_index_rows(dataset_root: Path, profile_name: str) -> list[dict[str, object]]:
    dataset_root = Path(dataset_root)
    root = profile_cache_root(dataset_root, profile_name)
    rows: list[dict[str, object]] = []
    rows.extend(_read_mmap_index_rows_from_paths(dataset_root, [root / "index.parquet"]))
    rows.extend(_read_mmap_checkpoint_index_rows(dataset_root, profile_name))
    rows.extend(_read_distributed_rank_index_rows(dataset_root, profile_name))
    return _deduplicate_index_rows(rows)


def rebuild_mmap_profile_index(dataset_root: Path, profile: VisualTokenProfile) -> dict[str, object]:
    if profile.file_format != MMAP_NPY_VISUAL_TOKEN_FORMAT:
        raise ValueError(f"profile {profile.name} must use {MMAP_NPY_VISUAL_TOKEN_FORMAT} file_format")
    rows = recover_mmap_profile_index_rows(Path(dataset_root), profile.name)
    profile = profile_with_mmap_index_hidden_dim(profile, rows)
    index_path = write_profile_index(dataset_root, profile.name, rows)
    write_profile_manifest(dataset_root, profile)
    return {
        "profile": profile.name,
        "index": str(index_path),
        "records": len(rows),
        "source": "mmap_recoverable_index",
    }


def write_mmap_checkpoint_index_rows(dataset_root: Path, profile_name: str, rows: list[dict[str, object]]) -> Path | None:
    if not rows:
        return None
    shard_paths = {str(row.get("shard_path", "")) for row in rows}
    if len(shard_paths) != 1 or not next(iter(shard_paths)):
        raise ValueError("mmap checkpoint index rows must all point to one shard_path")
    index_path = mmap_checkpoint_index_path_for_shard(dataset_root, profile_name, next(iter(shard_paths)))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_name(f".{index_path.name}.tmp.{os.getpid()}")
    try:
        pd.DataFrame(_ordered_index_rows(rows)).to_parquet(tmp_path, index=False)
        os.replace(tmp_path, index_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return index_path


def shard_refs_for_rank(refs: Iterable[str], *, rank: int, world_size: int) -> list[str]:
    rank = int(rank)
    world_size = int(world_size)
    if world_size <= 1:
        return list(refs)
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return [ref for index, ref in enumerate(refs) if index % world_size == rank]


def _episode_shard_key(ref: str) -> tuple[str, ...]:
    parts = str(ref).split("/")
    if len(parts) == 3:
        episode_id, _frame_index, _camera_name = parts
        return (episode_id,)
    if len(parts) == 5:
        dataset_name, split, episode_id, _frame_index, _camera_name = parts
        return (dataset_name, split, episode_id)
    return ("__ref__", str(ref))


def shard_refs_by_episode_for_rank(refs: Iterable[str], *, rank: int, world_size: int) -> list[str]:
    world_size = int(world_size)
    refs = [str(ref) for ref in refs]
    if world_size <= 1:
        return refs
    rank = int(rank)
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    return shard_refs_by_episode(refs, world_size=world_size)[rank]


def shard_refs_by_episode(refs: Iterable[str], *, world_size: int) -> list[list[str]]:
    world_size = int(world_size)
    refs = [str(ref) for ref in refs]
    if world_size <= 1:
        return [refs]

    groups: dict[tuple[str, ...], list[str]] = {}
    for ref in refs:
        groups.setdefault(_episode_shard_key(ref), []).append(ref)

    rank_groups: list[list[tuple[tuple[str, ...], list[str]]]] = [[] for _ in range(world_size)]
    rank_loads = [0 for _ in range(world_size)]
    for key, group_refs in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        target_rank = min(range(world_size), key=lambda candidate: (rank_loads[candidate], candidate))
        rank_groups[target_rank].append((key, group_refs))
        rank_loads[target_rank] += len(group_refs)

    return [
        [ref for _key, group_refs in sorted(groups_for_rank, key=lambda item: item[0]) for ref in group_refs]
        for groups_for_rank in rank_groups
    ]


def merge_distributed_index_rows(dataset_root: Path, profile_name: str, *, existing_rows: list[dict[str, object]], world_size: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in existing_rows:
        ref = str(row.get("ref", ""))
        if ref and ref not in seen:
            rows.append(row)
            seen.add(ref)
    for rank in range(int(world_size)):
        rank_path = distributed_rank_index_path(dataset_root, profile_name, rank=rank)
        if not rank_path.exists():
            continue
        for row in pd.read_parquet(rank_path).to_dict("records"):
            ref = str(row.get("ref", ""))
            if ref and ref not in seen:
                rows.append(row)
                seen.add(ref)
    return rows


def _distributed_env() -> tuple[int, int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))


def _distributed_barrier(world_size: int) -> None:
    if int(world_size) <= 1:
        return
    import torch.distributed as dist

    if not dist.is_available():
        return
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        timeout_seconds = int(os.environ.get("NAVVLA_DIST_TIMEOUT_SECONDS", str(24 * 60 * 60)))
        dist.init_process_group(backend=backend, timeout=dt.timedelta(seconds=timeout_seconds))
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
    else:
        dist.barrier()


def _cleanup_rank_indexes(dataset_root: Path, profile_name: str, *, world_size: int) -> None:
    for rank in range(int(world_size)):
        path = distributed_rank_index_path(dataset_root, profile_name, rank=rank)
        if path.exists():
            path.unlink()


def _cleanup_rank_work_plans(dataset_root: Path, profile_name: str, *, world_size: int) -> None:
    for rank in range(int(world_size)):
        path = distributed_rank_work_plan_path(dataset_root, profile_name, rank=rank)
        if path.exists():
            path.unlink()


def _cleanup_mmap_checkpoint_indexes(dataset_root: Path, profile_name: str) -> None:
    root = mmap_checkpoint_index_root(dataset_root, profile_name)
    if not root.exists():
        return
    for path in root.glob("*.parquet"):
        path.unlink()
    try:
        root.rmdir()
    except OSError:
        pass


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rank, world_size, _local_rank = _distributed_env()
    is_distributed = world_size > 1
    is_rank0 = rank == 0
    selected_token_budget = None if args.all_token_budgets else int(args.token_budget)
    profile = VisualTokenProfile(
        name=args.profile,
        visual_head=args.visual_head,
        encoder_name=args.encoder_name,
        encoder_ckpt=str(args.encoder_ckpt),
        token_level=args.token_level,
        token_count=args.token_count,
        hidden_dim=args.hidden_dim,
        dtype=args.dtype,
        has_deepstack=args.deepstack_layers > 0,
        deepstack_layers=args.deepstack_layers,
        file_format=args.file_format,
        shard_size=int(args.shard_size),
        cache_stage=(QWEN35_POOLED_HISTORY_CACHE_STAGE if args.visual_head == QWEN35_POOLED_HISTORY_VISUAL_HEAD else ""),
        input_resize=None if args.input_resize is None else tuple(int(value) for value in args.input_resize),
        patch_size=16 if args.visual_head == QWEN35_POOLED_HISTORY_VISUAL_HEAD else None,
        spatial_merge_size=2 if args.visual_head == QWEN35_POOLED_HISTORY_VISUAL_HEAD else None,
        storage_encoding=(
            BFLOAT16_BITS_STORAGE_ENCODING
            if args.visual_head == QWEN35_POOLED_HISTORY_VISUAL_HEAD
            else ""
        ),
    )
    if args.visual_head == QWEN35_POOLED_HISTORY_VISUAL_HEAD:
        if str(args.token_level) != QWEN35_POOLED_HISTORY_CACHE_STAGE:
            raise ValueError(
                f"Qwen3.5 pooled-history profile requires --token-level {QWEN35_POOLED_HISTORY_CACHE_STAGE}"
            )
        if tuple(args.input_resize) != (256, 256) or int(args.token_count) != 4 or str(args.dtype) != "uint16":
            raise ValueError(
                "Qwen3.5 pooled-history profile requires --input-resize 256x256, --token-count 4, "
                "and --dtype uint16; "
                "this keeps fixed-shape mmap rows aligned with training preprocessing"
            )
    if args.rebuild_index_only:
        if is_distributed and not is_rank0:
            return
        print(json.dumps({"rank": rank, "world_size": world_size, **rebuild_mmap_profile_index(args.dataset_root, profile)}, indent=2), flush=True)
        return

    if args.validate_before and is_rank0:
        print(f"rank=0 validating dataset before cache generation: {args.dataset_root}", flush=True)
        if args.all_token_budgets:
            for budget in available_context_token_budgets(args.dataset_root):
                validate_navvla_lerobot_dataset(
                    args.dataset_root, visual_token_mode="online_images", token_budget=int(budget)
                )
        else:
            validate_navvla_lerobot_dataset(
                args.dataset_root,
                visual_token_mode="online_images",
                token_budget=int(args.token_budget),
            )
        print("rank=0 validate-before completed", flush=True)
    _distributed_barrier(world_size)

    existing_rows_for_refs: list[dict[str, object]] = []
    rank_metadata_by_ref: dict[str, dict[str, Any]] | None = None
    if is_distributed:
        if is_rank0:
            planning_started = time.monotonic()
            print("rank=0 loading visual cache refs", flush=True)
            refs = load_visual_cache_refs(args.dataset_root, camera_names=args.camera_names)
            refs_seconds = time.monotonic() - planning_started
            print(f"rank=0 loaded visual cache refs: {len(refs)} seconds={refs_seconds:.1f}", flush=True)
            existing_started = time.monotonic()
            existing_rows_for_refs = (
                load_existing_ref_rows_for_refs(args.dataset_root, args.profile, refs) if args.skip_existing else []
            )
            existing_refs = {str(row["ref"]) for row in existing_rows_for_refs}
            missing_refs = find_missing_cache_refs(refs, existing_refs=existing_refs)
            if args.limit is not None:
                missing_refs = missing_refs[: args.limit]
            rank_refs_by_rank = shard_refs_by_episode(missing_refs, world_size=world_size)
            if missing_refs:
                data, episodes, video_index, cameras, _info = _load_dataset_lookup_tables(args.dataset_root)
                lookup = build_ref_metadata_lookup(data=data, episodes=episodes, video_index=video_index)
                metadata_by_ref = {
                    ref: resolve_ref_metadata(ref, cameras=cameras, lookup=lookup) for ref in missing_refs
                }
            else:
                metadata_by_ref = {}
            _write_distributed_rank_work_plans(
                args.dataset_root,
                args.profile,
                rank_refs=rank_refs_by_rank,
                metadata_by_ref=metadata_by_ref,
            )
            print(
                f"rank=0 history_refs={len(refs)} existing_refs={len(existing_refs)} "
                f"missing_refs={len(missing_refs)} profile={args.profile} world_size={world_size} "
                f"existing_and_plan_seconds={time.monotonic() - existing_started:.1f}",
                flush=True,
            )
        _distributed_barrier(world_size)
        rank_metadata_by_ref = _read_distributed_rank_work_plan(args.dataset_root, args.profile, rank=rank)
        rank_refs = list(rank_metadata_by_ref)
        print(f"rank={rank} rank_missing_refs={len(rank_refs)} profile={args.profile} shard_strategy=episode", flush=True)
    else:
        print("loading visual cache refs", flush=True)
        refs = load_visual_cache_refs(args.dataset_root, camera_names=args.camera_names)
        print(f"loaded visual cache refs: {len(refs)}", flush=True)
        existing_rows_for_refs = load_existing_ref_rows_for_refs(args.dataset_root, args.profile, refs) if args.skip_existing else []
        existing_refs = {str(row["ref"]) for row in existing_rows_for_refs}
        missing_refs = find_missing_cache_refs(refs, existing_refs=existing_refs)
        if args.limit is not None:
            missing_refs = missing_refs[: args.limit]
        rank_refs = missing_refs
        print(
            f"history_refs={len(refs)} existing_refs={len(existing_refs)} "
            f"missing_refs={len(missing_refs)} profile={args.profile} world_size={world_size}",
            flush=True,
        )

    if args.dry_run:
        return
    if not args.encoder_ckpt:
        raise ValueError("--encoder-ckpt is required unless --dry-run or --rebuild-index-only is used")

    if is_rank0:
        write_profile_manifest(args.dataset_root, profile)
        if is_distributed:
            _cleanup_rank_indexes(args.dataset_root, profile.name, world_size=world_size)
    _distributed_barrier(world_size)

    if is_distributed:
        if rank_refs:
            encoder = load_visual_encoder(encoder_ckpt=str(args.encoder_ckpt), profile=profile)

            def flush_mmap_checkpoint_rows(rows: list[dict[str, object]]) -> None:
                write_mmap_checkpoint_index_rows(args.dataset_root, profile.name, rows)

            rank_rows = generate_profile_cache(
                args.dataset_root,
                profile=profile,
                refs=rank_refs,
                encoder=encoder,
                skip_existing=False,
                progress_description=f"visual-cache rank {rank}/{world_size} {profile.name}",
                batch_size=args.batch_size,
                prefetch_batches=args.prefetch_batches,
                input_resize=args.input_resize,
                shard_prefix=f"rank_{rank:05d}_image_embeds",
                mmap_flush_callback=(
                    flush_mmap_checkpoint_rows
                    if profile.file_format == MMAP_NPY_VISUAL_TOKEN_FORMAT
                    else None
                ),
                metadata_by_ref=rank_metadata_by_ref,
            )
        else:
            rank_rows = []
        rank_path = distributed_rank_index_path(args.dataset_root, profile.name, rank=rank)
        rank_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rank_rows).to_parquet(rank_path, index=False)
        report = {
            "generated_by_rank": True,
            "profile": profile.name,
            "rank_index": str(rank_path),
            "records": len(rank_rows),
            "rank": rank,
            "world_size": world_size,
            "batch_size": int(args.batch_size),
            "prefetch_batches": int(args.prefetch_batches),
            "input_resize": None if args.input_resize is None else f"{args.input_resize[0]}x{args.input_resize[1]}",
        }
    else:
        encoder = load_visual_encoder(encoder_ckpt=str(args.encoder_ckpt), profile=profile)
        from tool.navvla.profile_visual_cache import generate_profile_cache_parallel

        report = generate_profile_cache_parallel(
            args.dataset_root,
            profile=profile,
            refs=rank_refs,
            encoder=encoder,
            skip_existing=args.skip_existing,
            batch_size=args.batch_size,
            prefetch_batches=args.prefetch_batches,
            input_resize=args.input_resize,
        )
    print(json.dumps({"rank": rank, "world_size": world_size, **report}, indent=2), flush=True)
    _distributed_barrier(world_size)

    if is_distributed and is_rank0:
        if profile.file_format == MMAP_NPY_VISUAL_TOKEN_FORMAT:
            final_report = rebuild_mmap_profile_index(args.dataset_root, profile)
            index_path = Path(str(final_report["index"]))
            merged_rows = recover_mmap_profile_index_rows(args.dataset_root, profile.name)
            profile = profile_with_mmap_index_hidden_dim(profile, merged_rows)
        else:
            existing_rows = existing_rows_for_refs if args.skip_existing else []
            merged_rows = merge_distributed_index_rows(args.dataset_root, profile.name, existing_rows=existing_rows, world_size=world_size)
            profile = profile_with_mmap_index_hidden_dim(profile, merged_rows)
            index_path = write_profile_index(args.dataset_root, profile.name, merged_rows)
            write_profile_manifest(args.dataset_root, profile)
        _cleanup_rank_indexes(args.dataset_root, profile.name, world_size=world_size)
        _cleanup_mmap_checkpoint_indexes(args.dataset_root, profile.name)
        _cleanup_rank_work_plans(args.dataset_root, profile.name, world_size=world_size)
        final_report = {
            "distributed": True,
            "profile": profile.name,
            "index": str(index_path),
            "records": len(merged_rows),
            "world_size": world_size,
        }
        if args.validate_after:
            if args.all_token_budgets:
                final_report["validation"] = {
                    str(budget): validate_navvla_lerobot_dataset(
                        args.dataset_root,
                        visual_token_mode="cached_history_online_current",
                        visual_token_profile=profile.name,
                        token_budget=int(budget),
                    )
                    for budget in available_context_token_budgets(args.dataset_root)
                }
            else:
                final_report["validation"] = validate_navvla_lerobot_dataset(
                    args.dataset_root,
                    visual_token_mode="cached_history_online_current",
                    visual_token_profile=profile.name,
                    token_budget=int(args.token_budget),
                )
        print(json.dumps(final_report, indent=2), flush=True)
    elif not is_distributed and args.validate_after:
        if args.all_token_budgets:
            validation = {
                str(budget): validate_navvla_lerobot_dataset(
                    args.dataset_root,
                    visual_token_mode="cached_history_online_current",
                    visual_token_profile=profile.name,
                    token_budget=int(budget),
                )
                for budget in available_context_token_budgets(args.dataset_root)
            }
        else:
            validation = validate_navvla_lerobot_dataset(
                args.dataset_root,
                visual_token_mode="cached_history_online_current",
                visual_token_profile=profile.name,
                token_budget=int(args.token_budget),
            )
        print(json.dumps({"validation": validation}, indent=2), flush=True)


def _load_dataset_lookup_tables(dataset_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    data = _read_parquet_shards(dataset_root / "data")
    episodes = _read_parquet_shards(dataset_root / "meta" / "episodes")
    video_index = pd.read_parquet(dataset_root / "meta" / "navvla_video_index.parquet")
    cameras = json.loads((dataset_root / "meta" / "navvla_cameras.json").read_text(encoding="utf-8"))
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    return data, episodes, video_index, cameras, info


def _read_existing_index_rows(
    dataset_root: Path,
    profile_name: str,
    *,
    shard_exists: dict[str, bool] | None = None,
) -> list[dict[str, object]]:
    index_path = profile_cache_root(dataset_root, profile_name) / "index.parquet"
    if not index_path.exists():
        return []
    manifest_path = profile_cache_root(dataset_root, profile_name) / "manifest.json"
    profile_format = NPZ_VISUAL_TOKEN_FORMAT
    if manifest_path.exists():
        profile_format = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("file_format", NPZ_VISUAL_TOKEN_FORMAT))
    rows = []
    for row in pd.read_parquet(index_path).to_dict("records"):
        if profile_format == MMAP_NPY_VISUAL_TOKEN_FORMAT:
            if _mmap_row_points_to_existing_shard(dataset_root, row, shard_exists=shard_exists):
                rows.append(row)
        else:
            token_path = dataset_root / str(row.get("path", ""))
            if token_path.exists() and token_path.suffix == ".npz":
                rows.append(row)
    return rows


def _mmap_row_points_to_existing_shard(
    dataset_root: Path,
    row: dict[str, object],
    *,
    shard_exists: dict[str, bool] | None = None,
) -> bool:
    shard_path = str(row.get("shard_path", ""))
    if not shard_path:
        return False
    if shard_exists is not None and shard_path in shard_exists:
        return shard_exists[shard_path]
    token_path = dataset_root / shard_path
    exists = token_path.exists() and token_path.suffix == ".npy"
    if shard_exists is not None:
        shard_exists[shard_path] = exists
    return exists


def _read_mmap_index_rows_from_paths(
    dataset_root: Path,
    paths: Iterable[Path],
    *,
    shard_exists: dict[str, bool] | None = None,
) -> list[dict[str, object]]:
    shard_exists = {} if shard_exists is None else shard_exists
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        if not path.exists():
            continue
        for row in pd.read_parquet(path).to_dict("records"):
            if _mmap_row_points_to_existing_shard(dataset_root, row, shard_exists=shard_exists):
                rows.append(row)
    return rows


def _read_mmap_checkpoint_index_rows(
    dataset_root: Path,
    profile_name: str,
    *,
    shard_exists: dict[str, bool] | None = None,
) -> list[dict[str, object]]:
    root = mmap_checkpoint_index_root(dataset_root, profile_name)
    if not root.exists():
        return []
    return _read_mmap_index_rows_from_paths(dataset_root, root.glob("*.parquet"), shard_exists=shard_exists)


def _read_distributed_rank_index_rows(
    dataset_root: Path,
    profile_name: str,
    *,
    shard_exists: dict[str, bool] | None = None,
) -> list[dict[str, object]]:
    root = profile_cache_root(dataset_root, profile_name)
    if not root.exists():
        return []
    return _read_mmap_index_rows_from_paths(dataset_root, root.glob("index.rank*.parquet"), shard_exists=shard_exists)


def _read_parquet_shards(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("chunk-*/part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards found under {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


if __name__ == "__main__":
    main()
