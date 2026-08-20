from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

QWEN3_VL_VISUAL_HEAD = "qwen3_vl_visual"
QWEN35_POOLED_HISTORY_VISUAL_HEAD = "qwen3_5_postmerge_pool4"
MINICPM_V46_VISUAL_HEAD = "minicpm_v46_visual"
SMOKE_VISUAL_HEAD = "smoke_token"
DEFAULT_VISUAL_TOKEN_PROFILE = "qwen3_vl_4b_pooled_history"
DEFAULT_MINICPM_V46_VISUAL_TOKEN_PROFILE = "minicpm_v46_pooled_history_4_mmap"
DEFAULT_QWEN35_POOLED_HISTORY_VISUAL_TOKEN_PROFILE = "qwen3_5_4b_postmerge_pool4_256_mmap"
QWEN35_POOLED_HISTORY_CACHE_STAGE = "vit_postmerge_pool4"
NPZ_VISUAL_TOKEN_FORMAT = "npz"
MMAP_NPY_VISUAL_TOKEN_FORMAT = "mmap_npy"
PROFILE_VISUAL_TOKEN_INDEX_COLUMNS = [
    "ref",
    "path",
    "image_key",
    "shard_path",
    "row_index",
    "token_count",
    "hidden_dim",
    "dtype",
    "episode_id",
    "trajectory_id",
    "frame_index",
    "source_frame_index",
    "data_index",
    "camera_name",
    "video_key",
    "grid_t",
    "grid_h",
    "grid_w",
    "cache_stage",
]


@dataclass(frozen=True)
class VisualTokenProfile:
    name: str
    visual_head: str = QWEN3_VL_VISUAL_HEAD
    encoder_name: str = "Qwen3-VL-4B-Instruct"
    encoder_ckpt: str = ""
    token_level: str = "pooled_history"
    token_count: int = 4
    hidden_dim: int = 2560
    dtype: str = "float16"
    has_deepstack: bool = True
    deepstack_layers: int = 3
    schema_version: str = "1.0"
    file_format: str = NPZ_VISUAL_TOKEN_FORMAT
    shard_size: int = 8192
    cache_stage: str = ""
    input_resize: tuple[int, int] | None = None
    patch_size: int | None = None
    spatial_merge_size: int | None = None
    storage_encoding: str = ""

    @property
    def array_keys(self) -> list[str]:
        return ["image_embeds", "deepstack_embeds"] if self.has_deepstack else ["image_embeds"]


@dataclass(frozen=True)
class VisualTokenRecord:
    ref: str
    path: str


def stable_ref_hash(ref: str) -> str:
    return hashlib.sha256(ref.encode("utf-8")).hexdigest()[:24]


def profile_cache_root(dataset_root: str | Path, profile_name: str) -> Path:
    return Path(dataset_root) / "cache" / "visual_tokens" / str(profile_name)


def default_visual_token_profile(*, encoder_ckpt: str = "") -> VisualTokenProfile:
    return VisualTokenProfile(name=DEFAULT_VISUAL_TOKEN_PROFILE, encoder_ckpt=str(encoder_ckpt))


def default_minicpm_v46_visual_token_profile(*, encoder_ckpt: str = "") -> VisualTokenProfile:
    return VisualTokenProfile(
        name=DEFAULT_MINICPM_V46_VISUAL_TOKEN_PROFILE,
        visual_head=MINICPM_V46_VISUAL_HEAD,
        encoder_name="MiniCPM-V-4.6",
        encoder_ckpt=str(encoder_ckpt),
        token_level="pooled_history",
        token_count=4,
        hidden_dim=0,
        dtype="float16",
        has_deepstack=False,
        deepstack_layers=0,
        file_format=MMAP_NPY_VISUAL_TOKEN_FORMAT,
    )


def default_qwen35_pooled_history_visual_token_profile(*, encoder_ckpt: str = "") -> VisualTokenProfile:
    return VisualTokenProfile(
        name=DEFAULT_QWEN35_POOLED_HISTORY_VISUAL_TOKEN_PROFILE,
        visual_head=QWEN35_POOLED_HISTORY_VISUAL_HEAD,
        encoder_name="Qwen3.5-4B",
        encoder_ckpt=str(encoder_ckpt),
        token_level=QWEN35_POOLED_HISTORY_CACHE_STAGE,
        token_count=4,
        hidden_dim=0,
        dtype="uint16",
        has_deepstack=False,
        deepstack_layers=0,
        schema_version="2.0",
        file_format=MMAP_NPY_VISUAL_TOKEN_FORMAT,
        cache_stage=QWEN35_POOLED_HISTORY_CACHE_STAGE,
        input_resize=(256, 256),
        patch_size=16,
        spatial_merge_size=2,
        shard_size=256,
        storage_encoding="bfloat16_bits",
    )


def write_profile_manifest(dataset_root: str | Path, profile: VisualTokenProfile) -> Path:
    root = profile_cache_root(dataset_root, profile.name)
    root.mkdir(parents=True, exist_ok=True)
    payload = asdict(profile)
    payload["array_keys"] = profile.array_keys
    path = root / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_profile_token_record(
    dataset_root: str | Path,
    *,
    profile: VisualTokenProfile,
    ref: str,
    image_embeds: np.ndarray,
    deepstack_embeds: np.ndarray | None,
) -> VisualTokenRecord:
    if profile.file_format != "npz":
        raise ValueError(f"profile {profile.name} must use npz file_format")
    if profile.has_deepstack and deepstack_embeds is None:
        raise ValueError(f"profile {profile.name} requires deepstack_embeds")
    if not profile.has_deepstack and deepstack_embeds is not None:
        raise ValueError(f"profile {profile.name} does not declare deepstack_embeds")
    image_array = np.asarray(image_embeds)
    if image_array.ndim != 2:
        raise ValueError(f"image_embeds must have shape [tokens, hidden_dim], got {image_array.shape}")
    arrays: dict[str, np.ndarray] = {"image_embeds": image_array.astype(profile.dtype)}
    if deepstack_embeds is not None:
        deepstack_array = np.asarray(deepstack_embeds)
        if deepstack_array.ndim != 3:
            raise ValueError(f"deepstack_embeds must have shape [layers, tokens, hidden_dim], got {deepstack_array.shape}")
        arrays["deepstack_embeds"] = deepstack_array.astype(profile.dtype)
    token_relpath = Path("cache") / "visual_tokens" / profile.name / "tokens" / f"{stable_ref_hash(ref)}.npz"
    token_path = Path(dataset_root) / token_relpath
    token_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(token_path, **arrays)
    return VisualTokenRecord(ref=ref, path=token_relpath.as_posix())


def _profile_with_inferred_hidden_dim(profile: VisualTokenProfile, hidden_dim: int | None) -> VisualTokenProfile:
    if hidden_dim is None or int(profile.hidden_dim) == int(hidden_dim):
        return profile
    if int(profile.hidden_dim) not in (0, int(hidden_dim)):
        raise ValueError(f"profile hidden_dim={profile.hidden_dim} does not match cache hidden_dim={hidden_dim}")
    return VisualTokenProfile(**{**asdict(profile), "hidden_dim": int(hidden_dim)})


def _image_key_from_row(row: dict[str, Any]) -> str:
    if row.get("image_key"):
        return str(row["image_key"])
    return f"{row.get('episode_id')}/{int(row.get('frame_index')):06d}/{row.get('camera_name')}"


class MMapNpyProfileShardWriter:
    def __init__(
        self,
        dataset_root: str | Path,
        *,
        profile: VisualTokenProfile,
        shard_size: int | None = None,
        shard_prefix: str = "image_embeds",
        on_flush: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> None:
        if profile.file_format != MMAP_NPY_VISUAL_TOKEN_FORMAT:
            raise ValueError(f"profile {profile.name} must use {MMAP_NPY_VISUAL_TOKEN_FORMAT} file_format")
        if profile.has_deepstack:
            raise ValueError(f"profile {profile.name} mmap_npy cache supports image_embeds only")
        self.dataset_root = Path(dataset_root)
        self.profile = profile
        self.shard_size = int(shard_size if shard_size is not None else profile.shard_size)
        if self.shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {self.shard_size}")
        self.shard_prefix = str(shard_prefix)
        self.shard_root = profile_cache_root(self.dataset_root, profile.name) / "shards"
        self.shard_root.mkdir(parents=True, exist_ok=True)
        self.next_shard_index = self._next_shard_index()
        self.buffer: list[np.ndarray] = []
        self.pending_rows: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []
        self.hidden_dim: int | None = int(profile.hidden_dim) if int(profile.hidden_dim) > 0 else None
        self.on_flush = on_flush

    def _next_shard_index(self) -> int:
        max_index = -1
        for path in self.shard_root.glob(f"{self.shard_prefix}_*.npy"):
            suffix = path.stem.removeprefix(f"{self.shard_prefix}_")
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
        return max_index + 1

    def add(self, *, ref: str, image_embeds: np.ndarray, metadata: dict[str, Any]) -> None:
        if len(self.buffer) >= self.shard_size:
            self.flush()
        image_array = np.asarray(image_embeds)
        expected_dtype = np.dtype(self.profile.dtype)
        if self.profile.storage_encoding:
            if image_array.dtype != expected_dtype:
                raise TypeError(
                    f"encoded cache profile {self.profile.name} requires pre-encoded {expected_dtype} arrays, "
                    f"got {image_array.dtype}"
                )
        else:
            image_array = image_array.astype(expected_dtype, copy=False)
        if image_array.ndim != 2:
            raise ValueError(f"image_embeds must have shape [tokens, hidden_dim], got {image_array.shape}")
        if int(image_array.shape[0]) != int(self.profile.token_count):
            raise ValueError(
                f"profile {self.profile.name} expects {self.profile.token_count} tokens, "
                f"got image_embeds shape {image_array.shape}"
            )
        hidden_dim = int(image_array.shape[1])
        if self.hidden_dim is None:
            self.hidden_dim = hidden_dim
        elif int(self.hidden_dim) != hidden_dim:
            raise ValueError(f"all mmap_npy records must share hidden_dim={self.hidden_dim}, got {hidden_dim}")
        shard_relpath = (
            Path("cache")
            / "visual_tokens"
            / self.profile.name
            / "shards"
            / f"{self.shard_prefix}_{self.next_shard_index:06d}.npy"
        )
        row_index = len(self.buffer)
        row = {
            "ref": str(ref),
            "path": shard_relpath.as_posix(),
            "image_key": _image_key_from_row(metadata),
            "shard_path": shard_relpath.as_posix(),
            "row_index": int(row_index),
            "token_count": int(image_array.shape[0]),
            "hidden_dim": hidden_dim,
            "dtype": str(image_array.dtype),
            **metadata,
        }
        self.buffer.append(image_array)
        self.pending_rows.append(row)

    def flush(self) -> None:
        if not self.buffer:
            return
        shard_relpath = Path(str(self.pending_rows[0]["shard_path"]))
        shard_path = self.dataset_root / shard_relpath
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        stacked_shape = (len(self.buffer), int(self.profile.token_count), int(self.hidden_dim or 0))
        shard = np.lib.format.open_memmap(
            shard_path,
            mode="w+",
            dtype=np.dtype(self.profile.dtype),
            shape=stacked_shape,
        )
        for row_index, image_array in enumerate(self.buffer):
            shard[row_index] = image_array
        shard.flush()
        flushed_rows = list(self.pending_rows)
        if self.on_flush is not None:
            self.on_flush(flushed_rows)
        self.rows.extend(flushed_rows)
        self.buffer = []
        self.pending_rows = []
        self.next_shard_index += 1

    def close(self) -> list[dict[str, Any]]:
        self.flush()
        write_profile_manifest(self.dataset_root, _profile_with_inferred_hidden_dim(self.profile, self.hidden_dim))
        return list(self.rows)


def write_profile_mmap_npy_cache(
    dataset_root: str | Path,
    *,
    profile: VisualTokenProfile,
    records: list[dict[str, Any]],
    shard_prefix: str = "image_embeds",
) -> Path:
    writer = MMapNpyProfileShardWriter(dataset_root, profile=profile, shard_prefix=shard_prefix)
    for record in records:
        payload = dict(record)
        image_embeds = payload.pop("image_embeds")
        ref = str(payload.pop("ref"))
        writer.add(ref=ref, image_embeds=np.asarray(image_embeds), metadata=payload)
    rows = writer.close()
    return write_profile_index(dataset_root, profile.name, rows)


def write_profile_index(dataset_root: str | Path, profile_name: str, rows: list[dict[str, object]]) -> Path:
    root = profile_cache_root(dataset_root, profile_name)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "index.parquet"
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame: pd.DataFrame
    if not rows:
        frame = pd.DataFrame(columns=PROFILE_VISUAL_TOKEN_INDEX_COLUMNS)
        try:
            frame.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return path
    ordered_rows = []
    for row in rows:
        ordered = {column: row.get(column) for column in PROFILE_VISUAL_TOKEN_INDEX_COLUMNS}
        for key, value in row.items():
            if key not in ordered:
                ordered[key] = value
        ordered_rows.append(ordered)
    frame = pd.DataFrame(ordered_rows)
    try:
        frame.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


@dataclass(frozen=True)
class VisualTokenCacheRecord:
    ref: str
    path: Path
    visual_head: str
    token_count: int | None = None
    hidden_dim: int | None = None
    dtype: str | None = None
    has_deepstack: bool = False
    deepstack_layers: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenCacheManifest:
    manifest_path: Path
    ref_to_path: dict[str, Path]
    ref_to_record: dict[str, VisualTokenCacheRecord] = field(default_factory=dict)


def _record_from_manifest_json(dataset_root: Path, payload: dict[str, Any]) -> VisualTokenCacheRecord:
    ref = str(payload["ref"])
    path = dataset_root / str(payload["path"])
    visual_head = str(payload.get("visual_head", SMOKE_VISUAL_HEAD))
    token_count = payload.get("token_count")
    hidden_dim = payload.get("hidden_dim")
    deepstack_layers = int(payload.get("deepstack_layers", 0) or 0)
    record = VisualTokenCacheRecord(
        ref=ref,
        path=path,
        visual_head=visual_head,
        token_count=int(token_count) if token_count is not None else None,
        hidden_dim=int(hidden_dim) if hidden_dim is not None else None,
        dtype=str(payload["dtype"]) if "dtype" in payload else None,
        has_deepstack=bool(payload.get("has_deepstack", deepstack_layers > 0)),
        deepstack_layers=deepstack_layers,
        metadata=dict(payload),
    )
    return record


def _path_for_ref(ref: str, *, suffix: str) -> Path:
    return Path("cache") / "visual_tokens" / "tokens" / f"{stable_ref_hash(ref)[:16]}{suffix}"


def write_visual_cache_record(
    dataset_root: Path,
    *,
    manifest_path: Path,
    ref: str,
    image_embeds: np.ndarray,
    deepstack_embeds: np.ndarray | None,
    visual_head: str,
    encoder_name: str,
    encoder_ckpt: str,
    token_level: str,
    extra_metadata: dict[str, Any] | None = None,
) -> VisualTokenCacheRecord:
    image_array = np.asarray(image_embeds)
    if image_array.ndim != 2:
        raise ValueError(f"image_embeds must have shape [tokens, hidden_dim], got {image_array.shape}")
    deepstack_array = None if deepstack_embeds is None else np.asarray(deepstack_embeds)
    if deepstack_array is not None and deepstack_array.ndim != 3:
        raise ValueError(f"deepstack_embeds must have shape [layers, tokens, hidden_dim], got {deepstack_array.shape}")
    if deepstack_array is not None and tuple(deepstack_array.shape[1:]) != tuple(image_array.shape):
        raise ValueError(
            "deepstack_embeds trailing shape must match image_embeds, "
            f"got {deepstack_array.shape[1:]} vs {image_array.shape}"
        )

    rel_path = _path_for_ref(ref, suffix=".npz")
    token_path = dataset_root / rel_path
    token_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"image_embeds": image_array}
    if deepstack_array is not None:
        payload["deepstack_embeds"] = deepstack_array
    np.savez(token_path, **payload)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    record_payload: dict[str, Any] = {
        "ref": ref,
        "path": str(rel_path),
        "visual_head": str(visual_head),
        "encoder_name": str(encoder_name),
        "encoder_ckpt": str(encoder_ckpt),
        "token_level": str(token_level),
        "token_count": int(image_array.shape[0]),
        "hidden_dim": int(image_array.shape[1]),
        "dtype": str(image_array.dtype),
        "has_deepstack": deepstack_array is not None,
        "deepstack_layers": int(deepstack_array.shape[0]) if deepstack_array is not None else 0,
    }
    if extra_metadata:
        record_payload.update(extra_metadata)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record_payload, ensure_ascii=True) + "\n")
    return _record_from_manifest_json(dataset_root, record_payload)


def read_token_manifest(dataset_root: Path) -> TokenCacheManifest:
    return read_visual_token_manifest(dataset_root, expected_visual_head=None)


def read_visual_token_manifest(dataset_root: Path, *, expected_visual_head: str | None = None) -> TokenCacheManifest:
    manifest_path = dataset_root / "cache" / "visual_tokens" / "manifest.jsonl"
    ref_to_path: dict[str, Path] = {}
    ref_to_record: dict[str, VisualTokenCacheRecord] = {}
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing visual token cache manifest: {manifest_path}")
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        visual_head = str(payload.get("visual_head", SMOKE_VISUAL_HEAD))
        if expected_visual_head is not None and visual_head != str(expected_visual_head):
            continue
        record = _record_from_manifest_json(dataset_root, payload)
        ref_to_path[record.ref] = record.path
        ref_to_record[record.ref] = record
    return TokenCacheManifest(manifest_path=manifest_path, ref_to_path=ref_to_path, ref_to_record=ref_to_record)


def validate_token_refs(refs: list[str], manifest: TokenCacheManifest) -> None:
    for ref in refs:
        if ref not in manifest.ref_to_path:
            raise KeyError(f"visual token ref is not in manifest: {ref}")
        path = manifest.ref_to_path[ref]
        if not path.exists():
            raise FileNotFoundError(f"missing visual token cache file for {ref}: {path}")
