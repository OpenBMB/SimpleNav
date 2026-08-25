from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from NavVLAeval.common.config import InputConfig
from NavVLAeval.common.types import EvalEpisode


class VLNCEInputAdapter:
    """Load VLN-CE R2R/RxR json.gz splits as NavVLAeval EvalEpisode records."""

    def load_episodes(self, cfg: InputConfig, *, max_samples: int | None) -> list[EvalEpisode]:
        if cfg.path is None:
            raise ValueError("input.path is required for VLN-CE input")
        if not cfg.namespace:
            raise ValueError("input.namespace is required for VLN-CE input")
        task_name = str(cfg.raw.get("task_name") or cfg.raw.get("benchmark_name") or _task_from_input_type(cfg.type)).lower()
        split = str(cfg.raw.get("split") or "val_unseen")
        roles = tuple(str(role) for role in (cfg.raw.get("roles") or ("guide",)))
        languages = tuple(str(language) for language in (cfg.raw.get("languages") or ()))
        records = _load_records(
            data_root=cfg.path,
            task_name=task_name,
            split=split,
            roles=roles,
            languages=languages,
            max_samples=max_samples,
        )
        episodes: list[EvalEpisode] = []
        for record in records:
            payload = dict(record)
            source_id = str(payload["episode_id"])
            instruction = str(payload.get("instruction") or "").strip()
            scene_id = _scene_name(payload)
            payload.update(
                {
                    "task_name": task_name,
                    "split": split,
                    "instruction": instruction,
                    "gt_path_length": _reference_path_length(payload.get("reference_path") or []),
                }
            )
            episodes.append(
                EvalEpisode(
                    episode_uid=f"{cfg.namespace}:{source_id}",
                    source_episode_id=source_id,
                    scene_id=scene_id,
                    instruction=instruction,
                    source=f"vlnce_{task_name}",
                    input_namespace=cfg.namespace,
                    input_root=str(cfg.path),
                    payload=payload,
                )
            )
        return episodes

    def fingerprint(self, cfg: InputConfig) -> str:
        if cfg.path is None:
            raise ValueError("input.path is required for VLN-CE fingerprint")
        task_name = str(cfg.raw.get("task_name") or cfg.raw.get("benchmark_name") or _task_from_input_type(cfg.type)).lower()
        split = str(cfg.raw.get("split") or "val_unseen")
        roles = tuple(str(role) for role in (cfg.raw.get("roles") or ("guide",)))
        digest = hashlib.sha256()
        digest.update(str(cfg.namespace).encode("utf-8"))
        digest.update(task_name.encode("utf-8"))
        digest.update(split.encode("utf-8"))
        for path in _split_paths(cfg.path, task_name=task_name, split=split, roles=roles):
            resolved_path = _existing_json_or_json_gz(path)
            digest.update(str(resolved_path.relative_to(cfg.path)).encode("utf-8"))
            digest.update(resolved_path.read_bytes())
        return digest.hexdigest()


VLNCER2RInputAdapter = VLNCEInputAdapter
VLNCERxRInputAdapter = VLNCEInputAdapter


class NavVLALeRobotVLNCEInputAdapter:
    """Load NavVLA-LeRobot v3 roots as VLN-CE EvalEpisode records."""

    def load_episodes(self, cfg: InputConfig, *, max_samples: int | None) -> list[EvalEpisode]:
        if cfg.path is None:
            raise ValueError("input.path is required for NavVLA-LeRobot VLN-CE input")
        episode_paths = sorted((cfg.path / "meta" / "episodes").glob("chunk-*/part-*.parquet"))
        if not episode_paths:
            fallback_path = cfg.path / "meta" / "episodes" / "chunk-000" / "part-000.parquet"
            raise FileNotFoundError(f"NavVLA-LeRobot episodes parquet does not exist: {fallback_path}")
        import pandas as pd

        table = pd.concat((pd.read_parquet(path) for path in episode_paths), ignore_index=True).sort_values("episode_index")
        if max_samples is not None:
            table = table.head(int(max_samples))
        task_text_by_index = _lerobot_task_text_by_index(cfg.path)
        task_name = str(cfg.raw.get("task_name") or _task_from_input_type(cfg.type)).lower()
        namespace = cfg.namespace or str(cfg.path.name)
        result: list[EvalEpisode] = []
        for _idx, row in table.iterrows():
            episode_index = int(row["episode_index"])
            source_id = str(row.get("episode_id") or episode_index)
            task_index = int(row.get("task_index", episode_index))
            instruction = task_text_by_index.get(task_index) or _first_task_text(row.get("tasks"))
            payload = {
                "task_name": task_name,
                "split": str(row.get("split") or cfg.raw.get("split") or ""),
                "episode_index": episode_index,
                "episode_id": source_id,
                "trajectory_id": str(row.get("trajectory_id") or source_id),
                "task_index": task_index,
                "lerobot_root": str(cfg.path),
                "gt_path_length": float(row.get("gt_path_length", 0.0) or 0.0),
            }
            result.append(
                EvalEpisode(
                    episode_uid=f"{namespace}:{source_id}",
                    source_episode_id=source_id,
                    scene_id=str(row.get("scene_id") or ""),
                    instruction=str(instruction or ""),
                    source=f"vlnce_{task_name}_lerobot",
                    input_namespace=namespace,
                    input_root=str(cfg.path),
                    payload=payload,
                )
            )
        return result

    def fingerprint(self, cfg: InputConfig) -> str:
        if cfg.path is None:
            raise ValueError("input.path is required for NavVLA-LeRobot VLN-CE fingerprint")
        digest = hashlib.sha256()
        episode_paths = sorted((cfg.path / "meta" / "episodes").glob("chunk-*/part-*.parquet"))
        for rel in ("meta/info.json", "meta/tasks.parquet"):
            path = cfg.path / rel
            if path.exists():
                digest.update(rel.encode("utf-8"))
                digest.update(path.read_bytes())
        for path in episode_paths:
            rel = path.relative_to(cfg.path).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()


VLNCELerobotInputAdapter = NavVLALeRobotVLNCEInputAdapter


def _lerobot_task_text_by_index(root: Path) -> dict[int, str]:
    import pandas as pd

    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}
    table = pd.read_parquet(tasks_path).reset_index()
    if "task_index" not in table.columns:
        return {}
    text_column = "task" if "task" in table.columns else table.columns[0]
    return {int(row["task_index"]): str(row[text_column]) for _idx, row in table.iterrows()}


def _first_task_text(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return str(value or "")



def _load_records(
    *,
    data_root: Path,
    task_name: str,
    split: str,
    roles: tuple[str, ...],
    languages: tuple[str, ...],
    max_samples: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _split_paths(data_root, task_name=task_name, split=split, roles=roles):
        role = _rxr_role_from_path(path, split=split) if task_name == "rxr" else None
        with _open_json_or_json_gz(path) as handle:
            payload = json.load(handle)
        for episode in payload.get("episodes", []):
            converted = dict(episode)
            converted["episode_id"] = str(converted["episode_id"])
            language = _instruction_language(converted) if task_name == "rxr" else ""
            converted["instruction"] = _instruction_text(converted)
            if task_name == "rxr":
                if languages and language not in languages:
                    continue
                converted["language"] = language
                converted["role"] = role
            records.append(converted)
            if max_samples is not None and len(records) >= int(max_samples):
                return records
    return records


def _split_paths(data_root: Path, *, task_name: str, split: str, roles: tuple[str, ...]) -> list[Path]:
    if task_name == "rxr":
        return [data_root / "datasets" / "RxR_VLNCE_v0" / split / f"{split}_{role}.json.gz" for role in roles]
    if task_name == "r2r":
        return [data_root / "datasets" / "R2R_VLNCE_v1-3_preprocessed" / split / f"{split}.json.gz"]
    raise ValueError(f"Unsupported VLN-CE task_name: {task_name!r}")


def _existing_json_or_json_gz(path: Path) -> Path:
    if path.exists():
        return path
    if str(path).endswith(".json.gz"):
        plain = Path(str(path)[: -len(".gz")])
        if plain.exists():
            return plain
    raise FileNotFoundError(f"VLN-CE split file does not exist: {path}")


def _open_json_or_json_gz(path: Path):
    resolved = _existing_json_or_json_gz(path)
    if str(resolved).endswith(".gz"):
        return gzip.open(resolved, "rt", encoding="utf-8")
    return resolved.open("r", encoding="utf-8")


def _task_from_input_type(input_type: str) -> str:
    lowered = str(input_type).lower()
    if "rxr" in lowered:
        return "rxr"
    return "r2r"


def _instruction_text(episode: dict[str, Any]) -> str:
    instruction = episode.get("instruction") or {}
    if isinstance(instruction, dict):
        return str(instruction.get("instruction_text") or "")
    return str(instruction)


def _instruction_language(episode: dict[str, Any]) -> str:
    instruction = episode.get("instruction") or {}
    if isinstance(instruction, dict):
        return str(instruction.get("language") or "")
    return str(episode.get("language") or "")


def _scene_name(episode: dict[str, Any]) -> str:
    scene_id = str(episode.get("scene_id") or "")
    stem = Path(scene_id).stem
    if stem:
        return stem
    parent = Path(scene_id).parent.name
    if parent:
        return parent
    raise ValueError(f"VLN-CE episode {episode.get('episode_id')} is missing scene_id")


def _reference_path_length(reference_path: list[Any]) -> float:
    if len(reference_path) < 2:
        return 0.0
    points = np.asarray(reference_path, dtype=np.float32)
    return float(np.linalg.norm(points[1:, :3] - points[:-1, :3], axis=1).sum())


def _rxr_role_from_path(path: Path, *, split: str) -> str:
    prefix = f"{split}_"
    stem = path.name.removesuffix(".json.gz")
    return stem[len(prefix) :] if stem.startswith(prefix) else stem
