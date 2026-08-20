from __future__ import annotations

import json
from bisect import bisect_left
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from starVLA.model.modules.qwen35_vision import BFLOAT16_BITS_STORAGE_ENCODING
from tool.navvla.visual_token_cache import (
    DEFAULT_MINICPM_V46_VISUAL_TOKEN_PROFILE,
    MINICPM_V46_VISUAL_HEAD,
    MMAP_NPY_VISUAL_TOKEN_FORMAT,
    QWEN35_POOLED_HISTORY_CACHE_STAGE,
    QWEN35_POOLED_HISTORY_VISUAL_HEAD,
    profile_cache_root,
)

TOKEN_SHARD_CACHE_SIZE = 12


@dataclass(frozen=True)
class VisualTokenBatch:
    tokens: np.ndarray
    grid_thw: np.ndarray | None = None
    cache_stage: str = ""
    encoder_ckpt: str = ""
    storage_encoding: str = ""


class _CompactTokenIndex:
    def __init__(self, path: Path, *, columns: list[str] | None = None) -> None:
        columns = columns or ["ref", "shard_path", "row_index", "token_count", "hidden_dim"]
        table = pq.read_table(path, columns=columns).sort_by([("ref", "ascending")])
        self.table = table.combine_chunks()
        self.refs = self.table["ref"].chunk(0)

    def __getitem__(self, ref: str) -> dict[str, object]:
        value = str(ref)
        position = bisect_left(self.refs, value, key=lambda scalar: scalar.as_py())
        if position >= len(self.refs) or self.refs[position].as_py() != value:
            raise KeyError(value)
        return self.table.slice(position, 1).to_pylist()[0]


class MiniCPMTokenStore:
    def __init__(
        self,
        dataset_root: str | Path,
        *,
        profile: str = DEFAULT_MINICPM_V46_VISUAL_TOKEN_PROFILE,
    ) -> None:
        self.root = Path(dataset_root)
        self.profile = str(profile)
        self._mmap_shards: OrderedDict[str, np.ndarray] = OrderedDict()
        self.token_count = 4
        self.hidden_dim = 0
        self.dtype = np.dtype(np.float16)
        self.ref_index = self._load_index()

    def _load_index(self) -> _CompactTokenIndex:
        profile_root = profile_cache_root(self.root, self.profile)
        manifest_path = profile_root / "manifest.json"
        index_path = profile_root / "index.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing visual token profile manifest: {manifest_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"missing visual token profile index: {index_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("file_format") != MMAP_NPY_VISUAL_TOKEN_FORMAT:
            raise ValueError(f"MiniCPM token store requires file_format={MMAP_NPY_VISUAL_TOKEN_FORMAT!r}")
        if manifest.get("visual_head") != MINICPM_V46_VISUAL_HEAD:
            raise ValueError(
                f"MiniCPM token store requires visual_head={MINICPM_V46_VISUAL_HEAD!r}, "
                f"got {manifest.get('visual_head')!r}"
            )
        array_keys = set(manifest.get("array_keys", []))
        if array_keys != {"image_embeds"}:
            raise ValueError(f"MiniCPM token store requires only image_embeds, got {sorted(array_keys)}")
        self.token_count = int(manifest.get("token_count", 4) or 4)
        self.hidden_dim = int(manifest.get("hidden_dim", 0) or 0)
        self.dtype = np.dtype(manifest.get("dtype", "float16"))
        required = {"ref", "shard_path", "row_index", "token_count", "hidden_dim"}
        missing = required - set(pq.read_schema(index_path).names)
        if missing:
            raise ValueError(f"visual token profile {self.profile} index is missing columns: {sorted(missing)}")
        return _CompactTokenIndex(index_path)

    def empty(self) -> np.ndarray:
        return np.zeros((0, self.token_count, self.hidden_dim), dtype=self.dtype)

    def load_ref_batches(self, ref_batches: list[list[str]]) -> list[np.ndarray]:
        unique_refs: list[str] = []
        seen: set[str] = set()
        for refs in ref_batches:
            for ref_value in refs:
                ref = str(ref_value)
                if ref not in seen:
                    seen.add(ref)
                    unique_refs.append(ref)
        if not unique_refs:
            return [self.empty() for _ in ref_batches]

        shard_groups: dict[str, list[tuple[str, int, int, int]]] = {}
        for ref in unique_refs:
            try:
                row = self.ref_index[ref]
            except KeyError as exc:
                raise KeyError(f"visual token ref {ref!r} is missing from profile {self.profile}") from exc
            shard_path = str(row["shard_path"])
            if Path(shard_path).suffix != ".npy":
                raise ValueError(f"cached MiniCPM shard must be .npy for {ref}: {shard_path}")
            shard_groups.setdefault(shard_path, []).append(
                (ref, int(row["row_index"]), int(row["token_count"]), int(row["hidden_dim"]))
            )

        loaded: dict[str, np.ndarray] = {}
        for shard_path, entries in shard_groups.items():
            shard = self._load_shard(self.root / shard_path)
            row_indices = np.asarray([entry[1] for entry in entries], dtype=np.int64)
            if (row_indices < 0).any() or (row_indices >= int(shard.shape[0])).any():
                bad = int(row_indices[(row_indices < 0) | (row_indices >= int(shard.shape[0]))][0])
                raise IndexError(f"visual token row_index={bad} is outside shard shape {shard.shape}: {shard_path}")
            batch = np.asarray(shard[row_indices])
            if batch.ndim != 3:
                raise ValueError(f"image_embeds batch must have shape [N, tokens, hidden_dim], got {batch.shape}")
            for offset, (ref, _row_index, expected_tokens, expected_hidden) in enumerate(entries):
                value = np.asarray(batch[offset])
                if value.shape != (expected_tokens, expected_hidden):
                    raise ValueError(
                        f"visual token ref {ref!r} shape mismatch: "
                        f"index=({expected_tokens}, {expected_hidden}) shard={value.shape}"
                    )
                loaded[ref] = value

        outputs = []
        for refs in ref_batches:
            outputs.append(self.empty() if not refs else np.stack([loaded[str(ref)] for ref in refs], axis=0))
        return outputs

    def _load_shard(self, path: Path) -> np.ndarray:
        key = str(path)
        shard = self._mmap_shards.pop(key, None)
        if shard is None:
            if not path.exists():
                raise FileNotFoundError(f"missing visual token mmap shard: {path}")
            shard = np.load(path, mmap_mode="r", allow_pickle=False)
            if shard.dtype != self.dtype:
                raise TypeError(
                    f"visual token shard dtype {shard.dtype} does not match manifest dtype {self.dtype}: {path}"
                )
        self._mmap_shards[key] = shard
        while len(self._mmap_shards) > TOKEN_SHARD_CACHE_SIZE:
            _old_key, old_shard = self._mmap_shards.popitem(last=False)
            mmap = getattr(old_shard, "_mmap", None)
            if mmap is not None:
                mmap.close()
        return shard

    def close(self) -> None:
        while self._mmap_shards:
            _key, shard = self._mmap_shards.popitem(last=False)
            mmap = getattr(shard, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __del__(self) -> None:
        self.close()


class Qwen35PooledHistoryTokenStore(MiniCPMTokenStore):
    def _load_index(self) -> _CompactTokenIndex:
        profile_root = profile_cache_root(self.root, self.profile)
        manifest_path = profile_root / "manifest.json"
        index_path = profile_root / "index.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing visual token profile manifest: {manifest_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"missing visual token profile index: {index_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("file_format") != MMAP_NPY_VISUAL_TOKEN_FORMAT:
            raise ValueError(f"Qwen3.5 pooled-history token store requires file_format={MMAP_NPY_VISUAL_TOKEN_FORMAT!r}")
        if manifest.get("visual_head") != QWEN35_POOLED_HISTORY_VISUAL_HEAD:
            raise ValueError(
                f"Qwen3.5 pooled-history token store requires visual_head={QWEN35_POOLED_HISTORY_VISUAL_HEAD!r}, "
                f"got {manifest.get('visual_head')!r}"
            )
        if manifest.get("cache_stage") != QWEN35_POOLED_HISTORY_CACHE_STAGE:
            raise ValueError(
                f"Qwen3.5 cache stage must be {QWEN35_POOLED_HISTORY_CACHE_STAGE!r}, "
                f"got {manifest.get('cache_stage')!r}"
            )
        if set(manifest.get("array_keys", [])) != {"image_embeds"}:
            raise ValueError("Qwen3.5 pooled-history token store requires only image_embeds")
        self.token_count = int(manifest.get("token_count", 0) or 0)
        if self.token_count != 4:
            raise ValueError(f"Qwen3.5 pooled-history cache requires token_count=4, got {self.token_count}")
        self.hidden_dim = int(manifest.get("hidden_dim", 0) or 0)
        self.dtype = np.dtype(manifest.get("dtype", "float16"))
        self.storage_encoding = str(manifest.get("storage_encoding", ""))
        if self.storage_encoding == BFLOAT16_BITS_STORAGE_ENCODING and self.dtype != np.dtype(np.uint16):
            raise ValueError("Qwen3.5 bfloat16_bits cache must use dtype=uint16")
        self.encoder_ckpt = str(manifest.get("encoder_ckpt", ""))
        self.spatial_merge_size = int(manifest.get("spatial_merge_size", 2) or 2)
        required = {
            "ref", "shard_path", "row_index", "token_count", "hidden_dim", "grid_t", "grid_h", "grid_w", "cache_stage"
        }
        missing = required - set(pq.read_schema(index_path).names)
        if missing:
            raise ValueError(f"Qwen3.5 visual token profile {self.profile} index is missing columns: {sorted(missing)}")
        return _CompactTokenIndex(index_path, columns=sorted(required))

    def load_ref_record_batches(self, ref_batches: list[list[str]]) -> list[VisualTokenBatch]:
        token_batches = super().load_ref_batches(ref_batches)
        outputs: list[VisualTokenBatch] = []
        for refs, tokens in zip(ref_batches, token_batches, strict=True):
            grids: list[list[int]] = []
            for ref in refs:
                row = self.ref_index[str(ref)]
                stage = str(row["cache_stage"])
                if stage != QWEN35_POOLED_HISTORY_CACHE_STAGE:
                    raise ValueError(f"visual token ref {ref!r} has incompatible cache_stage={stage!r}")
                grid = [int(row["grid_t"]), int(row["grid_h"]), int(row["grid_w"])]
                if any(value <= 0 for value in grid) or int(row["token_count"]) != self.token_count:
                    raise ValueError(
                        f"visual token ref {ref!r} must describe a positive source grid and a "
                        f"{self.token_count}-token pooled row, got grid={grid}, token_count={row['token_count']}"
                    )
                grids.append(grid)
            outputs.append(
                VisualTokenBatch(
                    tokens=tokens,
                    grid_thw=np.asarray(grids, dtype=np.int64).reshape(-1, 3),
                    cache_stage=QWEN35_POOLED_HISTORY_CACHE_STAGE,
                    encoder_ckpt=self.encoder_ckpt,
                    storage_encoding=self.storage_encoding,
                )
            )
        return outputs


def open_visual_token_store(dataset_root: str | Path, *, profile: str):
    manifest_path = profile_cache_root(dataset_root, profile) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing visual token profile manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    visual_head = str(manifest.get("visual_head", ""))
    if visual_head == MINICPM_V46_VISUAL_HEAD:
        return MiniCPMTokenStore(dataset_root, profile=profile)
    if visual_head == QWEN35_POOLED_HISTORY_VISUAL_HEAD:
        return Qwen35PooledHistoryTokenStore(dataset_root, profile=profile)
    raise ValueError(f"unsupported CPM visual token head {visual_head!r} in {manifest_path}")
