from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from NavVLAeval.common.config import InputConfig
from NavVLAeval.common.types import EvalEpisode


class TravelUAVJsonInputAdapter:
    def load_episodes(self, cfg: InputConfig, *, max_samples: int | None) -> list[EvalEpisode]:
        if not cfg.namespace:
            raise ValueError("input.namespace is required for TravelUAV JSON input")
        if cfg.path is None:
            raise ValueError("input.path is required for TravelUAV JSON input")
        payload = json.loads(cfg.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"TravelUAV eval_json must contain a list, got {type(payload).__name__}")
        allowed_scene_ids = _allowed_scene_ids(cfg.raw)
        episodes = []
        for index, record in enumerate(payload):
            if not isinstance(record, dict):
                raise ValueError(f"TravelUAV eval record {index} must be an object")
            source_id = _required_source_id(record, record_index=index)
            scene_id = str(record.get("scene_id") or record.get("env_name") or "").strip()
            if not scene_id:
                raise ValueError(f"TravelUAV eval record {source_id} is missing scene_id")
            if allowed_scene_ids and scene_id not in allowed_scene_ids:
                continue
            instruction = str(record.get("instruction") or record.get("gpt_instruction") or "").strip()
            if not instruction:
                raise ValueError(f"TravelUAV eval record {source_id} is missing instruction")
            env_name = str(record.get("env_name") or "").strip()
            if not env_name:
                raise ValueError(f"TravelUAV eval record {source_id} is missing env_name")
            enriched = dict(record)
            enriched.setdefault("scene_id", scene_id)
            episode_uid = f"{cfg.namespace}:{source_id}"
            episodes.append(
                EvalEpisode(
                    episode_uid=episode_uid,
                    source_episode_id=source_id,
                    scene_id=scene_id,
                    instruction=instruction,
                    source="eval_json",
                    input_namespace=cfg.namespace,
                    input_root=str(cfg.path),
                    payload=enriched,
                )
            )
            if max_samples is not None and len(episodes) >= int(max_samples):
                break
        return episodes

    def fingerprint(self, cfg: InputConfig) -> str:
        if cfg.path is None:
            raise ValueError("input.path is required for TravelUAV JSON fingerprint")
        digest = hashlib.sha256()
        digest.update(str(cfg.namespace).encode("utf-8"))
        digest.update(cfg.path.read_bytes())
        _update_scene_filter_fingerprint(digest, cfg.raw)
        return digest.hexdigest()


class TravelUAVLeRobotV3InputAdapter:
    def load_episodes(self, cfg: InputConfig, *, max_samples: int | None) -> list[EvalEpisode]:
        episodes: list[EvalEpisode] = []
        allowed_scene_ids = _allowed_scene_ids(cfg.raw)
        for namespace, root in _lerobot_roots(cfg):
            episodes.extend(
                self._load_root(
                    namespace=namespace,
                    root=root,
                    max_samples=None,
                    allowed_scene_ids=allowed_scene_ids,
                )
            )
            if max_samples is not None and len(episodes) >= int(max_samples):
                return episodes[: int(max_samples)]
        return episodes

    def fingerprint(self, cfg: InputConfig) -> str:
        digest = hashlib.sha256()
        for namespace, root in _lerobot_roots(cfg):
            digest.update(namespace.encode("utf-8"))
            digest.update(str(root).encode("utf-8"))
            for path in _required_lerobot_files(root):
                digest.update(str(path.relative_to(root)).encode("utf-8"))
                digest.update(path.read_bytes())
        _update_scene_filter_fingerprint(digest, cfg.raw)
        return digest.hexdigest()

    def _load_root(
        self,
        *,
        namespace: str,
        root: Path,
        max_samples: int | None,
        allowed_scene_ids: set[str],
    ) -> list[EvalEpisode]:
        episodes_table = _read_parquet_shards(root / "meta" / "episodes")
        data_table = _read_parquet_shards(root / "data")
        _require_columns(
            episodes_table,
            {"episode_index", "episode_id", "scene_id", "tasks", "length"},
            label=f"{root}/meta/episodes",
        )
        _require_columns(
            data_table,
            {"episode_index", "frame_index", "observation.state"},
            label=f"{root}/data",
        )
        data_table = _attach_frame_source_metadata(root, data_table)
        episodes_table = episodes_table.sort_values("episode_index").reset_index(drop=True)
        data_by_episode = {
            int(episode_index): group.sort_values("frame_index")
            for episode_index, group in data_table.groupby("episode_index", sort=False)
        }
        episode_sidecar_keys = _episode_sidecar_keys(episodes_table)
        benchmark_sidecar = _read_benchmark_sidecar(root)
        object_description_path = _object_description_path(root)
        unmatched_sidecar_keys = sorted(set(benchmark_sidecar) - episode_sidecar_keys)
        if unmatched_sidecar_keys:
            raise ValueError(f"unmatched benchmark sidecar records in {root}: {unmatched_sidecar_keys[:3]}")
        missing_sidecar_keys = sorted(episode_sidecar_keys - set(benchmark_sidecar))
        if benchmark_sidecar and missing_sidecar_keys:
            raise ValueError(f"missing benchmark sidecar records in {root}: {missing_sidecar_keys[:3]}")

        episodes: list[EvalEpisode] = []
        for row in episodes_table.to_dict("records"):
            episode_index = int(row["episode_index"])
            source_id = str(row["episode_id"]).strip()
            if not source_id:
                raise ValueError(f"TravelUAV LeRobot episode at index {episode_index} is missing episode_id")
            scene_id = str(row["scene_id"] or "").strip()
            if not scene_id:
                raise ValueError(f"TravelUAV LeRobot episode {source_id} is missing scene_id")
            if allowed_scene_ids and scene_id not in allowed_scene_ids:
                continue
            frame_rows = data_by_episode.get(episode_index)
            if frame_rows is None or frame_rows.empty:
                raise ValueError(f"TravelUAV LeRobot episode {source_id} has no data rows")
            episode_uid = f"{namespace}:{source_id}"
            frame_records = frame_rows.to_dict("records")
            trajectory_raw = [
                _source_world_pose(frame_row, episode_uid=episode_uid)
                for frame_row in frame_records
            ]
            trajectory = [pose[:3] for pose in trajectory_raw]
            frames = [
                _frame_metadata(frame_row, pose=pose)
                for frame_row, pose in zip(frame_records, trajectory_raw)
            ]
            payload = {
                "env_name": scene_id,
                "scene_id": scene_id,
                "episode_index": episode_index,
                "task_index": _optional_int(row.get("task_index")),
                "trajectory_id": str(row.get("trajectory_id") or ""),
                "instruction": _first_task(row.get("tasks"), episode_uid=f"{namespace}:{source_id}"),
                "length": int(row["length"]),
                "start_pose": trajectory_raw[0],
                "goal_position": trajectory[-1],
                "trajectory": trajectory,
                "trajectory_raw_detailed": trajectory_raw,
                "frames": frames,
                "source_lerobot_root": str(root),
                "object_description_path": str(object_description_path),
            }
            sidecar_key = (source_id, int(payload["task_index"]), str(payload["trajectory_id"]), scene_id)
            sidecar_metadata = benchmark_sidecar.get(sidecar_key)
            if sidecar_metadata is None:
                sidecar_metadata = _frame_source_metadata(frame_rows)
                if sidecar_metadata is not None:
                    payload["source_metadata"] = sidecar_metadata
            elif sidecar_metadata is not None:
                _merge_benchmark_metadata(payload, sidecar_metadata)
            instruction = _benchmark_instruction(payload, episode_uid=episode_uid)
            episodes.append(
                EvalEpisode(
                    episode_uid=episode_uid,
                    source_episode_id=source_id,
                    scene_id=scene_id,
                    instruction=instruction,
                    source="navvla_lerobot_v3",
                    input_namespace=namespace,
                    input_root=str(root),
                    payload=payload,
                )
            )
            if max_samples is not None and len(episodes) >= int(max_samples):
                break
        return episodes


def _lerobot_roots(cfg: InputConfig) -> list[tuple[str, Path]]:
    if cfg.roots:
        return [(root.namespace, root.path) for root in cfg.roots]
    if not cfg.namespace:
        raise ValueError("input.namespace is required for TravelUAV LeRobot v3 input")
    if cfg.data_root is None:
        raise ValueError("input.data_root is required for TravelUAV LeRobot v3 input")
    return [(cfg.namespace, cfg.data_root)]


def _allowed_scene_ids(raw: dict[str, Any]) -> set[str]:
    raw_value = raw.get("scene_ids", raw.get("scene_id"))
    if raw_value is None:
        return set()
    values = raw_value if isinstance(raw_value, (list, tuple, set)) else [raw_value]
    return {str(value).strip() for value in values if str(value).strip()}


def _update_scene_filter_fingerprint(digest: Any, raw: dict[str, Any]) -> None:
    scene_ids = sorted(_allowed_scene_ids(raw))
    if scene_ids:
        digest.update(json.dumps(scene_ids, sort_keys=True).encode("utf-8"))


def _required_source_id(record: dict[str, Any], *, record_index: int) -> str:
    for key in ("episode_id", "sample_id", "id"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"TravelUAV eval record {record_index} is missing explicit episode_id")


def _required_lerobot_files(root: Path) -> list[Path]:
    files = []
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        files.append(info_path)
    object_description_path = _object_description_path(root)
    if object_description_path.exists():
        files.append(object_description_path)
    frame_metadata_path = root / "meta" / "navvla_frame_metadata.jsonl"
    if frame_metadata_path.exists():
        files.append(frame_metadata_path)
    files.extend(sorted((root / "meta" / "episodes").glob("chunk-*/part-*.parquet")))
    files.extend(sorted((root / "data").glob("chunk-*/part-*.parquet")))
    sidecar_path = root / "meta" / "navvla_benchmark_episodes.jsonl"
    if sidecar_path.exists():
        files.append(sidecar_path)
    if not files:
        raise FileNotFoundError(f"TravelUAV LeRobot root has no readable metadata/data shards: {root}")
    return files


def _read_benchmark_sidecar(root: Path) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    path = root / "meta" / "navvla_benchmark_episodes.jsonl"
    if not path.exists():
        return {}
    records: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"benchmark sidecar line {line_number} must be an object: {path}")
        if str(record.get("benchmark") or "") != "traveluav":
            raise ValueError(f"benchmark sidecar line {line_number} has unsupported benchmark: {path}")
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"benchmark sidecar line {line_number} is missing metadata object: {path}")
        key = (
            str(record.get("episode_id") or "").strip(),
            int(record["task_index"]),
            str(record.get("trajectory_id") or "").strip(),
            str(record.get("scene_id") or "").strip(),
        )
        if not all((key[0], key[2], key[3])):
            raise ValueError(f"benchmark sidecar line {line_number} has an empty join key: {path}")
        if key in records:
            raise ValueError(f"duplicate benchmark sidecar key {key}: {path}")
        records[key] = dict(metadata)
    return records


def _object_description_path(root: Path) -> Path:
    path = root / "meta" / "object_description.json"
    if not path.exists():
        return root / "meta" / "object_description.json"
    return path


def _episode_sidecar_keys(episodes_table: pd.DataFrame) -> set[tuple[str, int, str, str]]:
    keys: set[tuple[str, int, str, str]] = set()
    for row in episodes_table.to_dict("records"):
        episode_index = int(row["episode_index"])
        episode_id = str(row["episode_id"]).strip()
        if not episode_id:
            raise ValueError(f"TravelUAV LeRobot episode at index {episode_index} is missing episode_id")
        scene_id = str(row["scene_id"] or "").strip()
        if not scene_id:
            raise ValueError(f"TravelUAV LeRobot episode {episode_id} is missing scene_id")
        task_index = _optional_int(row.get("task_index"))
        if task_index is None:
            raise ValueError(f"TravelUAV LeRobot episode {episode_id} is missing task_index")
        trajectory_id = str(row.get("trajectory_id") or "").strip()
        if not trajectory_id:
            raise ValueError(f"TravelUAV LeRobot episode {episode_id} is missing trajectory_id")
        key = (episode_id, int(task_index), trajectory_id, scene_id)
        if key in keys:
            raise ValueError(f"duplicate TravelUAV LeRobot episode sidecar key: {key}")
        keys.add(key)
    return keys


def _attach_frame_source_metadata(root: Path, data_table: pd.DataFrame) -> pd.DataFrame:
    if "source_metadata" in data_table.columns:
        return data_table
    path = root / "meta" / "navvla_frame_metadata.jsonl"
    if not path.exists():
        return data_table
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return data_table
    metadata = pd.DataFrame(records)
    if "index" not in metadata.columns or "source_metadata" not in metadata.columns or "index" not in data_table.columns:
        return data_table
    return data_table.merge(metadata[["index", "source_metadata"]], on="index", how="left")


def _frame_source_metadata(frame_rows: pd.DataFrame) -> dict[str, Any] | None:
    if "source_metadata" not in frame_rows.columns:
        return None
    for value in frame_rows["source_metadata"]:
        value = _coerce_source_metadata(value)
        if value is None:
            continue
        return value
    return None


def _merge_benchmark_metadata(payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    payload["benchmark_metadata"] = dict(metadata)


def _coerce_source_metadata(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, str):
        if not value.strip():
            return None
        value = json.loads(value)
    if isinstance(value, dict):
        return value
    return None


def _benchmark_instruction(payload: dict[str, Any], *, episode_uid: str) -> str:
    metadata = payload.get("benchmark_metadata")
    metadata_instruction = metadata.get("instruction") if isinstance(metadata, dict) else None
    instruction = str(metadata_instruction or payload.get("instruction") or "").strip()
    if not instruction:
        raise ValueError(f"TravelUAV LeRobot episode {episode_uid} is missing benchmark instruction")
    return instruction


def _read_parquet_shards(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("chunk-*/part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet shards found under {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _require_columns(table: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _first_task(value: Any, *, episode_uid: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    if isinstance(value, (list, tuple)):
        for item in value:
            text = str(item).strip()
            if text:
                return text
    if hasattr(value, "tolist"):
        return _first_task(value.tolist(), episode_uid=episode_uid)
    raise ValueError(f"TravelUAV LeRobot episode {episode_uid} is missing instruction")


def _source_world_pose(row: dict[str, Any], *, episode_uid: str) -> list[float]:
    metadata = _coerce_source_metadata(row.get("source_metadata"))
    source_state = metadata.get("source_state") if metadata is not None else None
    if hasattr(source_state, "tolist"):
        source_state = source_state.tolist()
    if source_state is None:
        frame_index = row.get("frame_index")
        raise ValueError(
            f"TravelUAV LeRobot episode {episode_uid} frame {frame_index} "
            "is missing absolute source_state in source_metadata"
        )
    if len(source_state) >= 6:
        values = [source_state[0], source_state[1], source_state[2], source_state[5]]
    elif len(source_state) == 4:
        values = list(source_state)
    else:
        raise ValueError(
            f"TravelUAV LeRobot episode {episode_uid} source_state must contain "
            f"[x,y,z,yaw] or [x,y,z,roll,pitch,yaw], got length {len(source_state)}"
        )
    pose = [float(value) for value in values]
    if not all(math.isfinite(value) for value in pose):
        raise ValueError(f"TravelUAV LeRobot episode {episode_uid} source_state contains non-finite values")
    return pose


def _frame_metadata(row: dict[str, Any], *, pose: list[float]) -> dict[str, Any]:
    frame = {
        "frame_index": int(row["frame_index"]),
        "timestamp": float(row["timestamp"]) if row.get("timestamp") is not None else None,
        "state": pose,
    }
    context_key = str(row.get("context.index_key") or "").strip()
    if context_key:
        frame["context_index_key"] = context_key
    if row.get("sample.action_available") is not None:
        frame["action_available"] = bool(row["sample.action_available"])
    if row.get("index") is not None:
        frame["index"] = int(row["index"])
    return frame


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
