from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


ROW_GROUP_CACHE_SIZE = 4


class LazyParquetRows:
    """Random row access backed by a small per-process row-group cache."""

    def __init__(
        self,
        source: str | Path | Sequence[str | Path],
        *,
        columns: Sequence[str] | None = None,
        optional_columns: Sequence[str] = (),
        cache_size: int = ROW_GROUP_CACHE_SIZE,
    ) -> None:
        self.paths = self._resolve_paths(source)
        requested_columns = None if columns is None else tuple(columns)
        optional = set(optional_columns)
        self.cache_size = int(cache_size)
        if self.cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {cache_size}")
        available_sets: list[set[str]] = []
        self._row_groups: list[tuple[int, int, int]] = []
        self._row_group_ends: list[int] = []
        total = 0
        for file_index, path in enumerate(self.paths):
            parquet_file = pq.ParquetFile(path)
            try:
                available_sets.append(set(parquet_file.schema_arrow.names))
                for row_group_index in range(parquet_file.metadata.num_row_groups):
                    count = int(parquet_file.metadata.row_group(row_group_index).num_rows)
                    self._row_groups.append((file_index, row_group_index, total))
                    total += count
                    self._row_group_ends.append(total)
            finally:
                parquet_file.close()
        if requested_columns is None:
            self.columns = None
        else:
            common_columns = set.intersection(*available_sets)
            missing = set(requested_columns) - common_columns - optional
            if missing:
                raise ValueError(f"parquet source is missing required columns: {sorted(missing)}")
            self.columns = tuple(column for column in requested_columns if column in common_columns)
        self._length = total
        self._row_group_cache: OrderedDict[tuple[int, int], pa.Table] = OrderedDict()

    @staticmethod
    def _resolve_paths(source: str | Path | Sequence[str | Path]) -> list[Path]:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                paths = sorted(path.glob("chunk-*/part-*.parquet"))
            else:
                paths = [path]
        else:
            paths = sorted(Path(value) for value in source)
        if not paths:
            raise FileNotFoundError(f"no parquet files found for {source}")
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing parquet files: {missing}")
        return paths

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, object]:
        index = self._normalize_index(index)
        group_position = bisect_right(self._row_group_ends, index)
        file_index, row_group_index, group_start = self._row_groups[group_position]
        table = self._load_row_group(file_index, row_group_index)
        return table.slice(index - group_start, 1).to_pylist()[0]

    def read_range(
        self,
        start: int,
        length: int,
        *,
        columns: Iterable[str] | None = None,
    ) -> list[dict[str, object]]:
        requested = None if columns is None else tuple(columns)
        if requested is not None and self.columns is not None and not set(requested).issubset(self.columns):
            raise ValueError(f"requested columns {requested} are not in accessor columns {self.columns}")
        rows = [self[index] for index in range(int(start), int(start) + int(length))]
        if requested is None:
            return rows
        return [{key: row[key] for key in requested} for row in rows]

    def _normalize_index(self, index: int) -> int:
        index = int(index)
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(f"parquet row {index} outside length {self._length}")
        return index

    def _load_row_group(self, file_index: int, row_group_index: int) -> pa.Table:
        key = (file_index, row_group_index)
        table = self._row_group_cache.pop(key, None)
        if table is None:
            parquet_file = pq.ParquetFile(self.paths[file_index])
            try:
                table = parquet_file.read_row_group(
                    row_group_index,
                    columns=None if self.columns is None else list(self.columns),
                )
            finally:
                parquet_file.close()
        self._row_group_cache[key] = table
        while len(self._row_group_cache) > self.cache_size:
            self._row_group_cache.popitem(last=False)
        return table
