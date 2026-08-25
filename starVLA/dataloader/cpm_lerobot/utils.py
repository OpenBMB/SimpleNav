from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return list(value)


def as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "off"}
    return bool(value)


def float_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return np.asarray([float_array(item) for item in value], dtype=np.float32)
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray(
            [float_array(item) if isinstance(item, (list, np.ndarray)) else item for item in value],
            dtype=np.float32,
        )
    return np.asarray(value, dtype=np.float32)


def pose4(value: Any) -> np.ndarray:
    state = float_array(value).reshape(-1)
    if state.shape[0] < 4:
        raise ValueError(f"NavVLA observation.state requires at least 4 values, got shape {state.shape}")
    return state[:4].astype(np.float32)


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return int(value)


def optional_float(row: pd.Series | dict[str, Any], key: str) -> float | None:
    if key not in row:
        return None
    value = row[key]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def read_parquet_shards(root: Path, *, columns: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    paths = sorted(root.glob("chunk-*/part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards found under {root}")
    frames = []
    for path in paths:
        read_columns = None
        if columns is not None:
            available = set(pq.read_schema(path).names)
            read_columns = [column for column in columns if column in available]
        frames.append(pd.read_parquet(path, columns=read_columns))
    return pd.concat(frames, ignore_index=True)
