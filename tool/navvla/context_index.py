from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tool.navvla.compute_bats_k import (
    BATSBudgetConfig,
    BATSRowBudget,
    compute_bats_row_budget,
    history_frame_capacity,
    visual_block_token_cost,
)
from tool.navvla.schema import NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame
from tool.navvla.visual_token_cache import TokenCacheManifest


DEFAULT_CONTEXT_TOKEN_BUDGETS = (1024, 2048)
DEFAULT_CONTEXT_TOKEN_BUDGET = 1024
CONTEXT_INDEX_MANIFEST = "navvla_context_index_manifest.json"
CONTEXT_INDEX_VERSION = 2
CONTEXT_PARQUET_ROW_GROUP_SIZE = 131_072


class _ProgressReporter:
    """Emit bounded, log-friendly progress updates for long context builds."""

    def __init__(self, description: str, *, total: int) -> None:
        self.description = str(description)
        self.total = max(0, int(total))
        self.completed = 0
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at
        self.next_percent = 5
        self._report(force=True)

    def advance(self, amount: int) -> None:
        self.completed = min(self.total, self.completed + max(0, int(amount)))
        now = time.monotonic()
        percent = 100 if self.total == 0 else int(100 * self.completed / self.total)
        if self.completed == self.total or percent >= self.next_percent or now - self.last_report_at >= 30.0:
            self._report(now=now)
            while self.next_percent <= percent:
                self.next_percent += 5

    def _report(self, *, force: bool = False, now: float | None = None) -> None:
        del force
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


@dataclass(frozen=True)
class ContextIndexConfig:
    epsilon: float = 0.1
    k: float = 4.0
    use_dynamic_bats_k: bool = True
    bats_token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET
    current_visual_tokens: int = 64
    history_visual_tokens: int = 4
    tvi_tokens: int = 1
    seed: int = 42
    budget_num_cameras: int | None = None
    history_camera_names: tuple[str, ...] | None = None
    include_long_memory: bool = True


def build_history_frame_ref(
    *,
    dataset_name: str,
    split: str,
    episode_id: str,
    frame_index: int,
    camera_name: str,
) -> str:
    del dataset_name, split
    return f"{episode_id}/{int(frame_index):06d}/{camera_name}"


@dataclass(frozen=True)
class ContextIndexResult:
    context_dir: Path
    meta_path: Path
    arrays_path: Path
    debug_path: Path
    token_budget: int | None = None
    camera_names: tuple[str, ...] = ()
    context_policy_version: str = "bats-v1"

    @property
    def refs_path(self) -> Path:
        """Legacy path retained for callers that need to reject old context roots."""
        return self.context_dir / "refs.parquet"


@dataclass
class RuntimeContextIndex:
    result: ContextIndexResult
    meta: pd.DataFrame
    arrays: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        self.meta_by_data_index = self.meta.set_index("index", drop=False)
        self.camera_names = list(self.result.camera_names)

    def materialize_by_data_index(self, data_index: int) -> dict[str, object]:
        return self.materialize_meta_row(self.meta_by_data_index.loc[int(data_index)])

    def materialize_meta_row(self, row: pd.Series | Mapping[str, object]) -> dict[str, object]:
        payload = dict(row)
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


def normalize_context_token_budgets(token_budgets: list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    values = DEFAULT_CONTEXT_TOKEN_BUDGETS if token_budgets is None else tuple(token_budgets)
    normalized: list[int] = []
    for value in values:
        budget = int(value)
        if budget <= 0:
            raise ValueError(f"context token budgets must be positive, got {value}")
        if budget not in normalized:
            normalized.append(budget)
    if not normalized:
        raise ValueError("at least one context token budget is required")
    return tuple(normalized)


def context_budget_tag(token_budget: int) -> str:
    budget = int(token_budget)
    if budget <= 0:
        raise ValueError(f"context token budget must be positive, got {token_budget}")
    return f"budget_{budget}"


def budget_context_index_paths(output_root: Path, *, split: str, token_budget: int) -> ContextIndexResult:
    tag = context_budget_tag(token_budget)
    context_dir = output_root / "meta" / "context_index" / tag
    return ContextIndexResult(
        context_dir=context_dir,
        meta_path=context_dir / "context_meta.parquet",
        arrays_path=context_dir / "context_arrays",
        debug_path=output_root / "cache" / "context_index_debug" / tag / f"{split}.parquet",
        token_budget=int(token_budget),
    )


def context_index_manifest_path(root: str | Path) -> Path:
    return Path(root) / "meta" / CONTEXT_INDEX_MANIFEST


def write_context_index_manifest(
    output_root: Path,
    *,
    split: str,
    results: Mapping[int, ContextIndexResult],
    default_token_budget: int,
) -> Path:
    budgets = sorted(int(budget) for budget in results)
    default_budget = int(default_token_budget)
    if default_budget not in results:
        raise ValueError(f"default context token budget {default_budget} is not in generated budgets {budgets}")
    entries: dict[str, dict[str, object]] = {}
    for budget in budgets:
        result = results[budget]
        entries[str(budget)] = {
            "token_budget": int(budget),
            "selection_policy": "bats",
            "context_policy_version": str(result.context_policy_version),
            "camera_names": list(result.camera_names),
            "context_dir": result.context_dir.relative_to(output_root).as_posix(),
            "meta_path": result.meta_path.relative_to(output_root).as_posix(),
            "arrays_path": result.arrays_path.relative_to(output_root).as_posix(),
            "debug_path": result.debug_path.relative_to(output_root).as_posix(),
        }
    payload = {
        "version": CONTEXT_INDEX_VERSION,
        "split": split,
        "selection_policy": "bats",
        "default_token_budget": default_budget,
        "available_token_budgets": budgets,
        "entries": entries,
    }
    path = context_index_manifest_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_context_index_manifest(root: str | Path) -> dict | None:
    path = context_index_manifest_path(root)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != CONTEXT_INDEX_VERSION:
        raise ValueError(
            f"unsupported context index version {payload.get('version')!r}; "
            f"rebuild context with version {CONTEXT_INDEX_VERSION}"
        )
    if str(payload.get("selection_policy")) != "bats":
        raise ValueError("context index selection_policy must be 'bats'")
    return payload


def available_context_token_budgets(root: str | Path) -> list[int]:
    manifest = read_context_index_manifest(root)
    if manifest is None:
        raise FileNotFoundError(f"missing context index manifest: {context_index_manifest_path(root)}")
    return [int(value) for value in manifest.get("available_token_budgets", [])]


def resolve_context_index_paths(root: str | Path, *, token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET) -> ContextIndexResult:
    root = Path(root)
    budget = int(token_budget)
    manifest = read_context_index_manifest(root)
    if manifest is None:
        raise FileNotFoundError(f"missing context index manifest: {context_index_manifest_path(root)}")
    entry = (manifest.get("entries") or {}).get(str(budget))
    if entry is None:
        available = [int(value) for value in manifest.get("available_token_budgets", [])]
        raise ValueError(f"context token budget {budget} is not available; available={available}")
    if str(entry.get("selection_policy")) != "bats":
        raise ValueError(f"context budget {budget} is not a BATS context")
    return ContextIndexResult(
        context_dir=root / str(entry["context_dir"]),
        meta_path=root / str(entry["meta_path"]),
        arrays_path=root / str(entry["arrays_path"]),
        debug_path=root / str(entry["debug_path"]),
        token_budget=budget,
        camera_names=tuple(str(value) for value in entry.get("camera_names", [])),
        context_policy_version=str(entry.get("context_policy_version", "bats-v1")),
    )


def load_runtime_context_index(result: ContextIndexResult) -> RuntimeContextIndex:
    return RuntimeContextIndex(
        result=result,
        meta=pd.read_parquet(result.meta_path),
        arrays=_load_context_arrays_dir(result.arrays_path, mmap_mode="r"),
    )


_CONTEXT_ARRAY_DTYPES = {
    "history_frame_index": np.int64,
    "history_camera_mask": np.uint64,
    "long_memory_frame_index": np.int64,
    "long_memory_camera_mask": np.uint64,
}


def _load_context_arrays_dir(path: Path, *, mmap_mode: str | None = None) -> dict[str, np.ndarray]:
    if not path.is_dir():
        raise FileNotFoundError(f"missing context arrays directory: {path}")
    arrays: dict[str, np.ndarray] = {}
    for name, dtype in _CONTEXT_ARRAY_DTYPES.items():
        array_path = path / f"{name}.npy"
        if not array_path.exists():
            raise FileNotFoundError(f"missing context array file: {array_path}")
        array = np.load(array_path, mmap_mode=mmap_mode, allow_pickle=False)
        if np.dtype(array.dtype) != np.dtype(dtype):
            raise ValueError(f"context array {array_path} has dtype {array.dtype}, expected {np.dtype(dtype)}")
        arrays[name] = array
    return arrays


def iter_parquet_batches(
    path: str | Path,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    batch_size: int = CONTEXT_PARQUET_ROW_GROUP_SIZE,
) -> Iterator[pd.DataFrame]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=int(batch_size), columns=list(columns) if columns is not None else None):
        yield batch.to_pandas()


def read_parquet_in_batches(
    path: str | Path,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    batch_size: int = CONTEXT_PARQUET_ROW_GROUP_SIZE,
) -> pd.DataFrame:
    batches = list(iter_parquet_batches(path, columns=columns, batch_size=batch_size))
    if not batches:
        return pd.DataFrame()
    return pd.concat(batches, ignore_index=True)


def rewrite_parquet_row_groups(path: str | Path, *, row_group_size: int = CONTEXT_PARQUET_ROW_GROUP_SIZE) -> None:
    path = Path(path)
    source = pq.ParquetFile(path)
    temporary = path.with_name(f".{path.name}.rowgroups-{os.getpid()}")
    writer: pq.ParquetWriter | None = None
    try:
        for batch in source.iter_batches(batch_size=int(row_group_size)):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema)
            writer.write_table(table, row_group_size=int(row_group_size))
        if writer is None:
            writer = pq.ParquetWriter(temporary, source.schema_arrow)
        writer.close()
        writer = None
        os.replace(temporary, path)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def keep_probability(history_frame_index: int, anchor_frame_index: int, *, epsilon: float, k: float) -> float:
    if anchor_frame_index <= 0:
        return 0.0
    exponent = float(k) * (int(history_frame_index) - int(anchor_frame_index)) / int(anchor_frame_index)
    decay = 0.0 if exponent <= -745.0 else math.exp(min(exponent, 709.0))
    return float(min(1.0, max(0.0, (1.0 - float(epsilon)) * decay + float(epsilon))))


def _positive_int(value: int, *, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return integer


def _resolve_budget_num_cameras(config: ContextIndexConfig, episode: NavVLAEpisode) -> int:
    if config.budget_num_cameras is not None:
        return _positive_int(config.budget_num_cameras, name="budget_num_cameras")
    return max(1, len(episode.cameras))


def _history_cameras_for_episode(episode: NavVLAEpisode, config: ContextIndexConfig) -> list:
    if config.history_camera_names is None:
        cameras = list(episode.cameras)
    else:
        by_name = {camera.name: camera for camera in episode.cameras}
        missing = [name for name in config.history_camera_names if name not in by_name]
        if missing:
            raise ValueError(f"episode {episode.episode_id} is missing history cameras: {missing}")
        cameras = [by_name[name] for name in config.history_camera_names]
    if len(cameras) > 64:
        raise ValueError(f"compact context camera masks support at most 64 cameras, got {len(cameras)}")
    return cameras


def _history_frame_token_cost(config: ContextIndexConfig, *, budget_num_cameras: int) -> int:
    return int(budget_num_cameras) * visual_block_token_cost(
        visual_tokens=config.history_visual_tokens,
        tvi_tokens=config.tvi_tokens,
    )


def _max_history_frames_for_budget(config: ContextIndexConfig, *, budget_num_cameras: int) -> int:
    return history_frame_capacity(
        token_budget=config.bats_token_budget,
        num_cameras=budget_num_cameras,
        current_visual_tokens=config.current_visual_tokens,
        history_visual_tokens=config.history_visual_tokens,
        tvi_tokens=config.tvi_tokens,
    )


def _independent_draw(
    *,
    seed: int,
    dataset_name: str,
    episode_id: str,
    anchor_frame_index: int,
    history_frame_index: int,
) -> float:
    return random.Random(
        f"{int(seed)}:{dataset_name}:{episode_id}:{int(anchor_frame_index)}:{int(history_frame_index)}"
    ).random()


def _select_history(
    frames: list[NavVLAFrame],
    anchor_pos: int,
    *,
    episode: NavVLAEpisode,
    spec: NavVLADatasetSpec,
    config: ContextIndexConfig,
    bats_k: float,
    budget_num_cameras: int,
) -> tuple[list[NavVLAFrame], list[float], list[float], list[NavVLAFrame]]:
    if anchor_pos <= 0:
        return [], [], [], []
    max_history_frames = _max_history_frames_for_budget(config, budget_num_cameras=budget_num_cameras)
    if max_history_frames <= 0:
        return [], [], [], []

    anchor = frames[anchor_pos]
    sampled: list[tuple[float, int, float, float, NavVLAFrame]] = []
    for frame in frames[:anchor_pos]:
        probability = keep_probability(
            frame.frame_index,
            anchor.frame_index,
            epsilon=config.epsilon,
            k=bats_k,
        )
        draw = _independent_draw(
            seed=config.seed,
            dataset_name=spec.dataset_name,
            episode_id=episode.episode_id,
            anchor_frame_index=anchor.frame_index,
            history_frame_index=frame.frame_index,
        )
        if draw < probability:
            priority = draw / probability if probability > 0.0 else math.inf
            sampled.append((priority, int(frame.frame_index), probability, draw, frame))

    if len(sampled) > max_history_frames:
        sampled = sorted(sampled, key=lambda item: (item[0], item[1]))[:max_history_frames]
    sampled.sort(key=lambda item: item[1])
    selected = [frame for _priority, _frame_index, _probability, _draw, frame in sampled]
    probabilities = [probability for _priority, _frame_index, probability, _draw, _frame in sampled]
    draws = [draw for _priority, _frame_index, _probability, draw, _frame in sampled]
    ranked = [
        frame
        for _priority, _frame_index, _probability, _draw, frame in sorted(
            sampled,
            key=lambda item: (item[2], item[1]),
            reverse=True,
        )
    ]
    return selected, probabilities, draws, ranked


def _long_memory_candidate_frame_index(candidate: NavVLAFrame | Mapping[str, object]) -> int:
    if isinstance(candidate, Mapping):
        return int(candidate["frame_index"])
    return int(candidate.frame_index)


def _select_long_memory_candidate(
    ranked_candidates: list[NavVLAFrame] | list[Mapping[str, object]],
    *,
    memory_frame_indices: set[int],
) -> NavVLAFrame | Mapping[str, object] | None:
    memory_indices = {int(index) for index in memory_frame_indices}
    memory_max = max(memory_indices) if memory_indices else None
    for candidate in reversed(ranked_candidates):
        frame_index = _long_memory_candidate_frame_index(candidate)
        if frame_index in memory_indices:
            continue
        if memory_max is not None and frame_index <= memory_max:
            continue
        return candidate
    return None


def _budget_config(config: ContextIndexConfig) -> BATSBudgetConfig:
    return BATSBudgetConfig(
        token_budget=int(config.bats_token_budget),
        epsilon=float(config.epsilon),
        current_visual_tokens=int(config.current_visual_tokens),
        history_visual_tokens=int(config.history_visual_tokens),
        tvi_tokens=int(config.tvi_tokens),
    )


def _row_budget(anchor_pos: int, *, camera_count: int, config: ContextIndexConfig) -> BATSRowBudget:
    if not bool(config.use_dynamic_bats_k):
        target_frames = min(float(anchor_pos), _budget_config(config).target_history_frames(num_cameras=camera_count))
        return BATSRowBudget(
            k=float(config.k),
            target_frames=target_frames,
            expected_frames=float("nan"),
            budget_feasible=True,
        )
    return compute_bats_row_budget(anchor_pos, num_cameras=camera_count, config=_budget_config(config))


def _frame_camera_mask(frame: NavVLAFrame, cameras: list) -> int:
    mask = 0
    for camera_index, camera in enumerate(cameras):
        if camera.video_key in frame.media_paths:
            mask |= 1 << camera_index
    return mask


def _selected_frame_rows(frames: list[NavVLAFrame], cameras: list) -> list[dict[str, int]]:
    rows = []
    for frame in frames:
        camera_mask = _frame_camera_mask(frame, cameras)
        if camera_mask:
            rows.append({"frame_index": int(frame.frame_index), "camera_mask": int(camera_mask)})
    return rows


def _iter_context_row_batches(
    episodes: list[NavVLAEpisode],
    *,
    spec: NavVLADatasetSpec,
    config: ContextIndexConfig,
    batch_size: int,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    debug_rows: list[dict[str, object]] = []
    global_index = 0
    expected_camera_names: tuple[str, ...] | None = None
    for episode in episodes:
        frames = list(episode.frames)
        budget_num_cameras = _resolve_budget_num_cameras(config, episode)
        history_cameras = _history_cameras_for_episode(episode, config)
        camera_names = tuple(str(camera.name) for camera in history_cameras)
        if expected_camera_names is None:
            expected_camera_names = camera_names
        elif camera_names != expected_camera_names:
            raise ValueError(
                f"history camera ordering differs across episodes: {expected_camera_names} vs {camera_names}"
            )
        long_memory_frames: list[NavVLAFrame] = []
        long_memory_frame_indices: set[int] = set()
        for anchor_pos, frame in enumerate(frames):
            bats_budget = _row_budget(anchor_pos, camera_count=budget_num_cameras, config=config)
            selected, probabilities, draws, ranked_selected = _select_history(
                frames,
                anchor_pos,
                episode=episode,
                spec=spec,
                config=config,
                bats_k=bats_budget.k,
                budget_num_cameras=budget_num_cameras,
            )
            history_rows = _selected_frame_rows(selected, history_cameras)
            long_memory_rows = _selected_frame_rows(long_memory_frames, history_cameras)
            data_index = global_index if frame.data_index is None else int(frame.data_index)
            rows.append(
                {
                    "index": data_index,
                    "bats_k": float(bats_budget.k),
                    "history_frames": history_rows,
                    "long_memory_frames": long_memory_rows,
                }
            )
            debug_rows.append(
                {
                    "index": data_index,
                    "split": episode.split,
                    "selected_history_frame_index": [int(value["frame_index"]) for value in history_rows],
                    "keep_probability": [float(value) for value in probabilities],
                    "random_draw": [float(value) for value in draws],
                    "token_count_before": int(anchor_pos) * int(budget_num_cameras),
                    "bats_k": float(bats_budget.k),
                    "bats_expected_frames": float(bats_budget.expected_frames),
                    "bats_target_frames": float(bats_budget.target_frames),
                    "bats_budget_tokens": int(config.bats_token_budget),
                    "bats_epsilon": float(config.epsilon),
                    "bats_num_cameras": int(budget_num_cameras),
                    "bats_history_num_cameras": int(len(history_cameras)),
                    "bats_budget_feasible": bool(bats_budget.budget_feasible),
                    "history_selection_policy": "bats_independent_bernoulli",
                }
            )
            if config.include_long_memory:
                candidate = _select_long_memory_candidate(
                    ranked_selected,
                    memory_frame_indices=long_memory_frame_indices,
                )
                if candidate is not None and _frame_camera_mask(candidate, history_cameras):
                    frame_index = _long_memory_candidate_frame_index(candidate)
                    long_memory_frame_indices.add(frame_index)
                    long_memory_frames.append(candidate)  # type: ignore[arg-type]
            global_index += 1
            if len(rows) >= int(batch_size):
                yield pd.DataFrame(rows), pd.DataFrame(debug_rows)
                rows = []
                debug_rows = []
    if rows:
        yield pd.DataFrame(rows), pd.DataFrame(debug_rows)


_CONTEXT_META_COLUMNS = [
    "index",
    "bats_k",
    "history_offset",
    "history_count",
    "long_memory_offset",
    "long_memory_count",
]


def _context_frames(row: Mapping[str, object], prefix: str, camera_names: list[str]) -> list[dict[str, int]]:
    direct = row.get(f"{prefix}_frames")
    if isinstance(direct, np.ndarray):
        direct = direct.tolist()
    if isinstance(direct, (list, tuple)):
        return [
            {"frame_index": int(value["frame_index"]), "camera_mask": int(value["camera_mask"])}
            for value in direct
        ]

    steps = _context_list(row.get(f"{prefix}_steps"))
    blocks = _context_list(row.get(f"{prefix}_blocks"))
    refs = _context_list(row.get(f"{prefix}_token_refs"))
    per_step: list[dict[str, int]] = [
        {"frame_index": int(step.get("frame_index", -1)), "camera_mask": 0} for step in steps
    ]
    camera_id = {name: index for index, name in enumerate(camera_names)}
    for block_index, block in enumerate(blocks):
        step_index = int(block["step_index"])
        camera_name = str(block["camera_name"])
        if camera_name not in camera_id:
            camera_id[camera_name] = len(camera_names)
            camera_names.append(camera_name)
        frame_index = int(block.get("frame_index", -1))
        if frame_index < 0 and block_index < len(refs):
            parts = str(refs[block_index]).split("/", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                frame_index = int(parts[1])
        if step_index >= len(per_step):
            per_step.extend({"frame_index": -1, "camera_mask": 0} for _ in range(step_index + 1 - len(per_step)))
        if per_step[step_index]["frame_index"] < 0:
            per_step[step_index]["frame_index"] = frame_index
        elif frame_index >= 0 and per_step[step_index]["frame_index"] != frame_index:
            raise ValueError(f"{prefix} step {step_index} resolves to multiple frame indices")
        per_step[step_index]["camera_mask"] |= 1 << camera_id[camera_name]
    return [value for value in per_step if value["frame_index"] >= 0 and value["camera_mask"]]


def write_runtime_context_index(result: ContextIndexResult, *, context: pd.DataFrame, remove_legacy: bool = True) -> None:
    write_runtime_context_index_batches(result, context_batches=(context,), remove_legacy=remove_legacy)


def write_runtime_context_index_batches(
    result: ContextIndexResult,
    *,
    context_batches: Iterator[pd.DataFrame] | tuple[pd.DataFrame, ...] | list[pd.DataFrame],
    remove_legacy: bool = True,
) -> None:
    result.context_dir.mkdir(parents=True, exist_ok=True)
    camera_names = list(result.camera_names)
    raw_dir = result.context_dir / f".context-arrays-{os.getpid()}"
    shutil.rmtree(raw_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True)
    raw_paths = {name: raw_dir / f"{name}.bin" for name in _CONTEXT_ARRAY_DTYPES}
    raw_files = {name: path.open("wb") for name, path in raw_paths.items()}
    counts = {name: 0 for name in _CONTEXT_ARRAY_DTYPES}
    meta_writer: pq.ParquetWriter | None = None
    try:
        for context in context_batches:
            if context.empty:
                continue
            meta_rows: list[dict[str, object]] = []
            chunks: dict[str, list[int]] = {name: [] for name in _CONTEXT_ARRAY_DTYPES}
            for row in context.to_dict("records"):
                history = _context_frames(row, "history", camera_names)
                long_memory = _context_frames(row, "long_memory", camera_names)
                history_offset = counts["history_frame_index"] + len(chunks["history_frame_index"])
                long_memory_offset = counts["long_memory_frame_index"] + len(chunks["long_memory_frame_index"])
                for value in history:
                    chunks["history_frame_index"].append(int(value["frame_index"]))
                    chunks["history_camera_mask"].append(int(value["camera_mask"]))
                for value in long_memory:
                    chunks["long_memory_frame_index"].append(int(value["frame_index"]))
                    chunks["long_memory_camera_mask"].append(int(value["camera_mask"]))
                meta_rows.append(
                    {
                        "index": int(row["index"]),
                        "bats_k": float(row.get("bats_k", float("nan"))),
                        "history_offset": int(history_offset),
                        "history_count": int(len(history)),
                        "long_memory_offset": int(long_memory_offset),
                        "long_memory_count": int(len(long_memory)),
                    }
                )
            table = pa.Table.from_pandas(pd.DataFrame(meta_rows, columns=_CONTEXT_META_COLUMNS), preserve_index=False)
            if meta_writer is None:
                meta_writer = pq.ParquetWriter(result.meta_path, table.schema)
            meta_writer.write_table(table, row_group_size=CONTEXT_PARQUET_ROW_GROUP_SIZE)
            for name, values in chunks.items():
                array = np.asarray(values, dtype=_CONTEXT_ARRAY_DTYPES[name])
                if array.size:
                    array.tofile(raw_files[name])
                counts[name] += int(array.size)
    finally:
        if meta_writer is not None:
            meta_writer.close()
        for handle in raw_files.values():
            handle.close()

    if meta_writer is None:
        pd.DataFrame(columns=_CONTEXT_META_COLUMNS).to_parquet(result.meta_path, index=False)
    temporary_arrays = result.arrays_path.with_name(f".{result.arrays_path.name}.tmp-{os.getpid()}")
    shutil.rmtree(temporary_arrays, ignore_errors=True)
    temporary_arrays.mkdir(parents=True)
    try:
        for name, dtype_value in _CONTEXT_ARRAY_DTYPES.items():
            dtype = np.dtype(dtype_value)
            count = int(counts[name])
            output = np.lib.format.open_memmap(
                temporary_arrays / f"{name}.npy",
                mode="w+",
                dtype=dtype,
                shape=(count,),
            )
            if count:
                source = np.memmap(raw_paths[name], mode="r", dtype=dtype, shape=(count,))
                output[:] = source[:]
                del source
            output.flush()
            del output
        shutil.rmtree(result.arrays_path, ignore_errors=True)
        os.replace(temporary_arrays, result.arrays_path)
    finally:
        shutil.rmtree(temporary_arrays, ignore_errors=True)
        shutil.rmtree(raw_dir, ignore_errors=True)
    if remove_legacy:
        result.refs_path.unlink(missing_ok=True)
        legacy_npz = result.context_dir / "context_arrays.npz"
        legacy_npz.unlink(missing_ok=True)


def _remove_context_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _publish_context_directory(temp_path: Path, final_path: Path) -> None:
    backup_path = final_path.with_name(f".{final_path.name}.backup-{os.getpid()}")
    _remove_context_path(backup_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        final_path.rename(backup_path)
    try:
        temp_path.rename(final_path)
    except BaseException:
        if backup_path.exists() and not final_path.exists():
            backup_path.rename(final_path)
        raise
    _remove_context_path(backup_path)


def build_context_index_streaming(
    episodes: list[NavVLAEpisode],
    *,
    spec: NavVLADatasetSpec,
    output_root: Path,
    config: ContextIndexConfig,
    cache_manifest: TokenCacheManifest | None,
    output_token_budget: int,
    batch_size: int = 4096,
    progress_description: str | None = None,
) -> ContextIndexResult:
    del cache_manifest
    if not episodes:
        raise ValueError("episodes must be non-empty")
    cameras = _history_cameras_for_episode(episodes[0], config)
    camera_names = tuple(str(camera.name) for camera in cameras)
    result = replace(
        budget_context_index_paths(output_root, split=spec.split, token_budget=output_token_budget),
        camera_names=camera_names,
        context_policy_version=str(spec.context_policy_version),
    )
    temp_dir = result.context_dir.with_name(f".{result.context_dir.name}.tmp-{os.getpid()}")
    temp_debug = result.debug_path.with_name(f".{result.debug_path.name}.tmp-{os.getpid()}")
    _remove_context_path(temp_dir)
    temp_result = replace(
        result,
        context_dir=temp_dir,
        meta_path=temp_dir / result.meta_path.name,
        arrays_path=temp_dir / result.arrays_path.name,
    )
    temp_result.context_dir.mkdir(parents=True, exist_ok=True)
    result.debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_writer: pq.ParquetWriter | None = None
    progress = (
        _ProgressReporter(progress_description, total=sum(len(episode.frames) for episode in episodes))
        if progress_description is not None
        else None
    )

    def context_batches() -> Iterator[pd.DataFrame]:
        nonlocal debug_writer
        for context, debug in _iter_context_row_batches(
            episodes,
            spec=spec,
            config=config,
            batch_size=int(batch_size),
        ):
            if not debug.empty:
                table = pa.Table.from_pandas(debug, preserve_index=False)
                if debug_writer is None:
                    debug_writer = pq.ParquetWriter(temp_debug, table.schema)
                debug_writer.write_table(table, row_group_size=CONTEXT_PARQUET_ROW_GROUP_SIZE)
            if progress is not None:
                progress.advance(len(context))
            yield context

    try:
        write_runtime_context_index_batches(temp_result, context_batches=context_batches())
        if debug_writer is not None:
            debug_writer.close()
            debug_writer = None
        else:
            pd.DataFrame(columns=["index"]).to_parquet(temp_debug, index=False)
        _publish_context_directory(temp_result.context_dir, result.context_dir)
        os.replace(temp_debug, result.debug_path)
    except BaseException:
        if debug_writer is not None:
            debug_writer.close()
        _remove_context_path(temp_result.context_dir)
        temp_debug.unlink(missing_ok=True)
        raise
    return result


def build_context_index(
    episodes: list[NavVLAEpisode],
    *,
    spec: NavVLADatasetSpec,
    output_root: Path,
    config: ContextIndexConfig,
    cache_manifest: TokenCacheManifest | None,
    output_token_budget: int,
) -> ContextIndexResult:
    frame_count = sum(len(episode.frames) for episode in episodes)
    return build_context_index_streaming(
        episodes,
        spec=spec,
        output_root=output_root,
        config=config,
        cache_manifest=cache_manifest,
        output_token_budget=output_token_budget,
        batch_size=max(1, frame_count),
    )


def build_context_indexes_streaming(
    episodes: list[NavVLAEpisode],
    *,
    spec: NavVLADatasetSpec,
    output_root: Path,
    config: ContextIndexConfig,
    cache_manifest: TokenCacheManifest | None,
    token_budgets: list[int] | tuple[int, ...] | None = None,
    batch_size: int = 4096,
    progress_description: str | None = None,
) -> dict[int, ContextIndexResult]:
    budgets = normalize_context_token_budgets(token_budgets)
    results: dict[int, ContextIndexResult] = {}
    for budget in budgets:
        result = build_context_index_streaming(
            episodes,
            spec=spec,
            output_root=output_root,
            config=replace(config, bats_token_budget=int(budget)),
            cache_manifest=cache_manifest,
            output_token_budget=int(budget),
            batch_size=int(batch_size),
            progress_description=(
                f"{progress_description} budget={int(budget)}"
                if progress_description is not None
                else None
            ),
        )
        results[int(budget)] = result
    write_context_index_manifest(
        output_root,
        split=spec.split,
        results=results,
        default_token_budget=int(budgets[0]),
    )
    return results


def build_context_indexes(
    episodes: list[NavVLAEpisode],
    *,
    spec: NavVLADatasetSpec,
    output_root: Path,
    config: ContextIndexConfig,
    cache_manifest: TokenCacheManifest | None,
    token_budgets: list[int] | tuple[int, ...] | None = None,
) -> dict[int, ContextIndexResult]:
    return build_context_indexes_streaming(
        episodes,
        spec=spec,
        output_root=output_root,
        config=config,
        cache_manifest=cache_manifest,
        token_budgets=token_budgets,
    )


def iter_context_refs(
    dataset_root: str | Path,
    *,
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
) -> Iterator[str]:
    root = Path(dataset_root)
    result = resolve_context_index_paths(root, token_budget=int(token_budget))
    runtime = load_runtime_context_index(result)
    data_paths = sorted((root / "data").glob("chunk-*/part-*.parquet"))
    episode_paths = sorted((root / "meta" / "episodes").glob("chunk-*/part-*.parquet"))
    data = pd.concat(
        [pd.read_parquet(path, columns=["index", "episode_index"]) for path in data_paths],
        ignore_index=True,
    )
    episodes = pd.concat(
        [pd.read_parquet(path, columns=["episode_index", "episode_id"]) for path in episode_paths],
        ignore_index=True,
    )
    episode_id_by_index = {
        int(row.episode_index): str(row.episode_id) for row in episodes.itertuples(index=False)
    }
    max_data_index = int(data["index"].max()) if not data.empty else -1
    episode_index_by_data_index = np.full(max_data_index + 1, -1, dtype=np.int64)
    episode_index_by_data_index[data["index"].to_numpy(dtype=np.int64)] = data["episode_index"].to_numpy(dtype=np.int64)
    camera_names = runtime.camera_names
    meta_columns = ["index", "history_offset", "history_count", "long_memory_offset", "long_memory_count"]
    for data_index, history_offset, history_count, long_memory_offset, long_memory_count in runtime.meta[
        meta_columns
    ].itertuples(index=False, name=None):
        episode_index = int(episode_index_by_data_index[int(data_index)])
        if episode_index < 0:
            raise KeyError(f"context data index does not resolve to an episode: {data_index}")
        episode_id = episode_id_by_index[episode_index]
        for prefix, offset, count in (
            ("history", int(history_offset), int(history_count)),
            ("long_memory", int(long_memory_offset), int(long_memory_count)),
        ):
            frame_indices = runtime.arrays[f"{prefix}_frame_index"][offset : offset + count]
            camera_masks = runtime.arrays[f"{prefix}_camera_mask"][offset : offset + count]
            for frame_index, camera_mask in zip(frame_indices, camera_masks, strict=True):
                for camera_index, camera_name in enumerate(camera_names):
                    if int(camera_mask) & (1 << camera_index):
                        yield build_history_frame_ref(
                            dataset_name="",
                            split="",
                            episode_id=episode_id,
                            frame_index=int(frame_index),
                            camera_name=camera_name,
                        )


def _context_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if pd.isna(value):
        return []
    return list(value)  # type: ignore[arg-type]
