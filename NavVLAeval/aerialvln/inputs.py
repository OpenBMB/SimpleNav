from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from NavVLAeval.common.config import InputConfig
from NavVLAeval.common.types import EvalEpisode


class AerialVLNJsonInputAdapter:
    def load_episodes(self, cfg: InputConfig, *, max_samples: int | None) -> list[EvalEpisode]:
        episodes: list[EvalEpisode] = []
        for namespace, path in _json_roots(cfg):
            episodes.extend(
                self._load_root(
                    namespace=namespace,
                    path=path,
                    raw=cfg.raw,
                    max_samples=None,
                )
            )
            if max_samples is not None and len(episodes) >= int(max_samples):
                return episodes[: int(max_samples)]
        return episodes

    def fingerprint(self, cfg: InputConfig) -> str:
        digest = hashlib.sha256()
        for namespace, path in _json_roots(cfg):
            digest.update(namespace.encode("utf-8"))
            digest.update(str(path).encode("utf-8"))
            digest.update(path.read_bytes())
        scene_ids = cfg.raw.get("scene_ids")
        if scene_ids is not None:
            digest.update(json.dumps(scene_ids, sort_keys=True).encode("utf-8"))
        episode_ids = sorted(_allowed_episode_ids(cfg.raw))
        if episode_ids:
            digest.update(json.dumps(episode_ids).encode("utf-8"))
        return digest.hexdigest()

    def _load_root(
        self,
        *,
        namespace: str,
        path,
        raw: dict[str, Any],
        max_samples: int | None,
    ) -> list[EvalEpisode]:
        payload = _load_payload(path)
        records = _episode_records(payload, source=str(path))
        allowed_scene_ids = _allowed_scene_ids(raw)
        allowed_episode_ids = _allowed_episode_ids(raw)
        env_name_prefix = str(raw.get("env_name_prefix") or "env_")

        episodes: list[EvalEpisode] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"AerialVLN episode {index} must be an object")
            source_id = _source_episode_id(record, index=index)
            if allowed_episode_ids and source_id not in allowed_episode_ids:
                continue
            scene_id = _scene_id(record, episode_id=source_id)
            if allowed_scene_ids and scene_id not in allowed_scene_ids:
                continue
            instruction = _instruction(record, episode_id=source_id)
            start_pose = _start_pose(record, episode_id=source_id)
            reference_path = _reference_path(record, episode_id=source_id)
            goal_position = _goal_position(record, reference_path=reference_path, episode_id=source_id)
            env_name = _env_name(record, scene_id=scene_id, prefix=env_name_prefix)

            enriched = dict(record)
            enriched.update(
                {
                    "scene_id": scene_id,
                    "env_name": env_name,
                    "instruction_text": instruction,
                    "start_pose": start_pose,
                    "goal_position": goal_position,
                    "reference_path_m": reference_path,
                    "trajectory": [point[:3] for point in reference_path],
                }
            )
            episode_uid = f"{namespace}:{source_id}"
            episodes.append(
                EvalEpisode(
                    episode_uid=episode_uid,
                    source_episode_id=source_id,
                    scene_id=scene_id,
                    instruction=instruction,
                    source="aerialvln_json",
                    input_namespace=namespace,
                    input_root=str(path),
                    payload=enriched,
                )
            )
            if max_samples is not None and len(episodes) >= int(max_samples):
                break
        return episodes


def _json_roots(cfg: InputConfig):
    if cfg.roots:
        return [(root.namespace, root.path) for root in cfg.roots]
    if not cfg.namespace:
        raise ValueError("input.namespace is required for AerialVLN JSON input")
    if cfg.path is None:
        raise ValueError("input.path is required for AerialVLN JSON input")
    return [(cfg.namespace, cfg.path)]


def _episode_records(payload: Any, *, source: str) -> list[Any]:
    if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        return list(payload["episodes"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError(f"AerialVLN JSON must contain an episodes list: {source}")


def _load_payload(path) -> Any:
    if path.suffix.lower() != ".jsonl":
        return json.loads(path.read_text(encoding="utf-8"))

    records: list[Any] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid AerialVLN JSONL record at {path}:{line_number}") from exc
    return records


def _allowed_scene_ids(raw: dict[str, Any]) -> set[str]:
    raw_value = raw.get("scene_ids", raw.get("scene_id"))
    if raw_value is None:
        return set()
    values = raw_value if isinstance(raw_value, (list, tuple, set)) else [raw_value]
    return {str(value).strip().removeprefix("env_") for value in values if str(value).strip()}


def _allowed_episode_ids(raw: dict[str, Any]) -> set[str]:
    raw_value = raw.get("episode_ids")
    if raw_value is None:
        return set()
    values = raw_value if isinstance(raw_value, (list, tuple, set)) else [raw_value]
    return {str(value).strip() for value in values if str(value).strip()}


def _source_episode_id(record: dict[str, Any], *, index: int) -> str:
    for key in ("episode_id", "id", "trajectory_id"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"AerialVLN episode {index} is missing episode_id")


def _scene_id(record: dict[str, Any], *, episode_id: str) -> str:
    value = str(record.get("scene_id") or record.get("scene") or "").strip()
    if not value:
        raise ValueError(f"AerialVLN episode {episode_id} is missing scene_id")
    return value.removeprefix("env_")


def _instruction(record: dict[str, Any], *, episode_id: str) -> str:
    container = record.get("instruction")
    if isinstance(container, dict):
        text = str(container.get("instruction_text") or container.get("text") or "").strip()
    else:
        text = str(record.get("instruction_text") or container or "").strip()
    if not text:
        raise ValueError(f"AerialVLN episode {episode_id} is missing instruction text")
    return text


def _start_pose(record: dict[str, Any], *, episode_id: str) -> list[float]:
    position = record.get("start_position") or record.get("start")
    if position is None or len(position) < 3:
        raise ValueError(f"AerialVLN episode {episode_id} is missing start_position")
    rotation = record.get("start_rotation")
    yaw = _yaw_from_quaternion_wxyz(rotation) if rotation is not None else 0.0
    return [float(position[0]), float(position[1]), float(position[2]), float(yaw)]


def _reference_path(record: dict[str, Any], *, episode_id: str) -> list[list[float]]:
    raw_path = record.get("reference_path") or record.get("trajectory")
    if not isinstance(raw_path, list) or not raw_path:
        raise ValueError(f"AerialVLN episode {episode_id} is missing non-empty reference_path")
    path: list[list[float]] = []
    for point_index, point in enumerate(raw_path):
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            raise ValueError(f"AerialVLN episode {episode_id} reference_path[{point_index}] must have xyz")
        yaw = float(point[5]) if len(point) >= 6 else (float(point[3]) if len(point) >= 4 else 0.0)
        path.append([float(point[0]), float(point[1]), float(point[2]), yaw])
    return path


def _goal_position(
    record: dict[str, Any],
    *,
    reference_path: list[list[float]],
    episode_id: str,
) -> list[float]:
    goals = record.get("goals")
    if isinstance(goals, list) and goals:
        first_goal = goals[0]
        if isinstance(first_goal, dict):
            position = first_goal.get("position")
        else:
            position = first_goal
        if position is not None and len(position) >= 3:
            return [float(position[0]), float(position[1]), float(position[2])]
    if reference_path:
        return [float(value) for value in reference_path[-1][:3]]
    raise ValueError(f"AerialVLN episode {episode_id} is missing goal position")


def _env_name(record: dict[str, Any], *, scene_id: str, prefix: str) -> str:
    value = str(record.get("env_name") or "").strip()
    if value:
        return value
    if str(scene_id).startswith(prefix):
        return str(scene_id)
    return f"{prefix}{scene_id}"


def _yaw_from_quaternion_wxyz(rotation: Any) -> float:
    if not isinstance(rotation, (list, tuple)) or len(rotation) < 4:
        raise ValueError("AerialVLN start_rotation must be [w, x, y, z]")
    w = float(rotation[0])
    x = float(rotation[1])
    y = float(rotation[2])
    z = float(rotation[3])
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
