from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tool.navvla.context_index import ContextIndexResult

from .parquet import LazyParquetRows


_META_COLUMNS = (
    "index",
    "bats_k",
    "history_offset",
    "history_count",
    "long_memory_offset",
    "long_memory_count",
)


class CompactRuntimeContextIndex:
    """Read the BATS-only compact context without materializing parquet tables."""

    def __init__(self, result: ContextIndexResult) -> None:
        self.result = result
        self.meta = LazyParquetRows(result.meta_path, columns=_META_COLUMNS)
        self.arrays = self._load_arrays(result.arrays_path)
        self.camera_names = list(result.camera_names)

    def __len__(self) -> int:
        return len(self.meta)

    def materialize_by_data_index(self, data_index: int) -> dict[str, Any]:
        return self.materialize_by_row_position(int(data_index), expected_index=int(data_index))

    def materialize_by_row_position(self, row_position: int, *, expected_index: int) -> dict[str, Any]:
        payload = dict(self.meta[int(row_position)])
        if int(payload["index"]) != int(expected_index):
            raise ValueError(
                f"context meta physical row {row_position} contains index={payload['index']}; "
                f"expected logical index={expected_index}"
            )
        payload["history_frames"] = self._frames(
            "history",
            offset=int(payload["history_offset"]),
            count=int(payload["history_count"]),
        )
        payload["long_memory_frames"] = self._frames(
            "long_memory",
            offset=int(payload["long_memory_offset"]),
            count=int(payload["long_memory_count"]),
        )
        payload["camera_names"] = list(self.camera_names)
        payload["context_policy_version"] = self.result.context_policy_version
        return payload

    def _frames(self, prefix: str, *, offset: int, count: int) -> list[dict[str, int]]:
        end = int(offset) + int(count)
        frame_indices = self.arrays[f"{prefix}_frame_index"][offset:end]
        camera_masks = self.arrays[f"{prefix}_camera_mask"][offset:end]
        return [
            {"frame_index": int(frame_index), "camera_mask": int(camera_mask)}
            for frame_index, camera_mask in zip(frame_indices, camera_masks, strict=True)
        ]

    @staticmethod
    def _load_arrays(path: Path) -> dict[str, np.ndarray]:
        if not path.is_dir():
            raise FileNotFoundError(f"missing context arrays directory: {path}")
        required = {
            "history_frame_index",
            "history_camera_mask",
            "long_memory_frame_index",
            "long_memory_camera_mask",
        }
        arrays = {
            array_path.stem: np.load(array_path, mmap_mode="r", allow_pickle=False)
            for array_path in sorted(path.glob("*.npy"))
        }
        missing = required - set(arrays)
        if missing:
            raise FileNotFoundError(f"context arrays are missing: {sorted(missing)}")
        return arrays
