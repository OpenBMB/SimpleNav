"""Read-only discovery and episode extraction for LeRobot VLN datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass(frozen=True)
class EpisodeMetadata:
    """The fields needed to select and locate one source episode."""

    episode_index: int
    episode_id: str
    scene_id: str
    length: int
    data_chunk_index: int
    data_file_index: int
    trajectory_id: str = ""
    task_index: int = 0
    tasks: tuple[str, ...] = ()


def _episode_metadata_files(root: Path) -> list[Path]:
    return sorted((root / "meta" / "episodes").glob("chunk-*/part-*.parquet"))


def discover_train_splits(dataset_root: str | Path) -> list[Path]:
    """Return active ``vln_train`` roots below *dataset_root* in stable order."""

    root = Path(dataset_root)
    discovered: list[Path] = []
    for info_path in root.rglob("info.json"):
        try:
            relative_parts = info_path.relative_to(root).parts
        except ValueError:
            continue
        if any(part.startswith(".") or _is_backup_or_temp_component(part) for part in relative_parts):
            continue
        if "vln_train_enhanced" in relative_parts:
            continue
        if len(relative_parts) < 3 or relative_parts[-3:] != (
            "vln_train",
            "meta",
            "info.json",
        ):
            continue
        discovered.append(info_path.parent.parent)
    return sorted(discovered)


def _is_backup_or_temp_component(part: str) -> bool:
    lowered = part.lower()
    tokens = ("backup", "old", "temp", "tmp", "scratch")
    return lowered in tokens or any(token in lowered for token in ("_backup", "backup_", "_temp", "temp_"))


def read_episode_metadata(train_root: str | Path) -> list[EpisodeMetadata]:
    """Read all episode records from ``meta/episodes`` without changing them."""

    root = Path(train_root)
    episode_files = _episode_metadata_files(root)
    records: list[EpisodeMetadata] = []
    required_columns = [
        "episode_index",
        "episode_id",
        "scene_id",
        "length",
        "data/chunk_index",
        "data/file_index",
    ]
    for path in episode_files:
        available = set(pq.ParquetFile(path).schema_arrow.names)
        columns = required_columns + [
            column
            for column in ("trajectory_id", "task_index", "tasks")
            if column in available
        ]
        table = pq.read_table(path, columns=columns)
        records.extend(_metadata_from_row(row) for row in table.to_pylist())
        del table
    return sorted(records, key=lambda item: item.episode_index)


def iter_episode_tables(
    train_root: str | Path, episode_indices: set[int] | None = None
):
    """Yield complete episode tables while opening each shared data parquet once."""

    root = Path(train_root)
    info = _read_json(root / "meta" / "info.json")
    template = info.get(
        "data_path", "data/chunk-{chunk_index:03d}/part-{file_index:03d}.parquet"
    )
    records = read_episode_metadata(root)
    if episode_indices is not None:
        allowed = set(int(index) for index in episode_indices)
        records = [episode for episode in records if episode.episode_index in allowed]
    grouped: dict[tuple[int, int], list[EpisodeMetadata]] = {}
    for episode in records:
        grouped.setdefault(
            (episode.data_chunk_index, episode.data_file_index), []
        ).append(episode)
    for chunk_index, file_index in sorted(grouped):
        relative_path = template.format(
            chunk_index=chunk_index,
            file_index=file_index,
        )
        data_path = _contained_path(root, relative_path)
        available_columns = set(pq.ParquetFile(data_path).schema_arrow.names)
        columns = ["episode_index", "frame_index", "observation.state"]
        if "index" in available_columns:
            columns.append("index")
        table = pq.read_table(
            data_path,
            columns=columns,
        )
        episode_values = table.column("episode_index")
        for episode in sorted(grouped[(chunk_index, file_index)], key=lambda item: item.episode_index):
            episode_table = table.filter(pc.equal(episode_values, episode.episode_index))
            if episode_table.num_rows != episode.length:
                raise ValueError(
                    f"episode {episode.episode_index} expected {episode.length} rows, "
                    f"found {episode_table.num_rows} in {data_path}"
                )
            if "frame_index" in episode_table.column_names and episode_table.num_rows > 1:
                episode_table = pc.take(
                    episode_table,
                    pc.sort_indices(episode_table, sort_keys=[("frame_index", "ascending")]),
                )
            if "frame_index" in episode_table.column_names:
                frame_indices = episode_table.column("frame_index").to_pylist()
                if frame_indices != list(range(episode.length)):
                    raise ValueError(
                        f"episode {episode.episode_index} has non-contiguous frame_index: "
                        f"expected 0..{episode.length - 1}"
                    )
            yield episode, episode_table


def extract_episode_rows(
    train_root: str | Path, episode: EpisodeMetadata
) -> pa.Table:
    """Read and return one complete episode from its metadata-indexed data file."""

    root = Path(train_root)
    info = _read_json(root / "meta" / "info.json")
    template = info.get(
        "data_path", "data/chunk-{chunk_index:03d}/part-{file_index:03d}.parquet"
    )
    relative_path = template.format(
        chunk_index=episode.data_chunk_index,
        file_index=episode.data_file_index,
    )
    data_path = _contained_path(root, relative_path)
    table = pq.read_table(
        data_path, filters=[("episode_index", "=", episode.episode_index)]
    )
    if table.num_rows != episode.length:
        raise ValueError(
            f"episode {episode.episode_index} expected {episode.length} rows, "
            f"found {table.num_rows} in {data_path}"
        )
    if "frame_index" in table.column_names and table.num_rows > 1:
        table = pc.take(table, pc.sort_indices(table, sort_keys=[("frame_index", "ascending")]))
    if "frame_index" in table.column_names:
        frame_indices = table.column("frame_index").to_pylist()
        if frame_indices != list(range(episode.length)):
            raise ValueError(
                f"episode {episode.episode_index} has non-contiguous frame_index: "
                f"expected 0..{episode.length - 1}"
            )
    return table


def _metadata_from_row(row: dict[str, Any]) -> EpisodeMetadata:
    return EpisodeMetadata(
        episode_index=int(row["episode_index"]),
        episode_id=str(row.get("episode_id", row["episode_index"])),
        scene_id=str(row.get("scene_id", "")),
        length=int(row["length"]),
        data_chunk_index=int(row["data/chunk_index"]),
        data_file_index=int(row["data/file_index"]),
        trajectory_id=str(row.get("trajectory_id", "")),
        task_index=int(row.get("task_index", 0)),
        tasks=tuple(str(item) for item in (row.get("tasks") or ())),
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _contained_path(root: Path, relative_path: str | Path) -> Path:
    root_resolved = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"data path escapes train root: {relative_path}")
    return candidate
