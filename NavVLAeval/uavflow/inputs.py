from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from NavVLAeval.common.config import InputConfig
from NavVLAeval.common.simulators.unrealzoo.coordinates import preprocessed_uavflow_pose_cm_to_nav_m
from NavVLAeval.common.types import EvalEpisode


class UAVFlowJsonInputAdapter:
    def load_episodes(self, cfg: InputConfig, *, max_samples: int | None) -> list[EvalEpisode]:
        if not cfg.namespace:
            raise ValueError("input.namespace is required for UAV-Flow JSON input")
        if cfg.path is None:
            raise ValueError("input.path is required for UAV-Flow JSON input")
        paths = _json_paths(cfg.path)
        instruction_type = str(cfg.raw.get("instruction_type") or "instruction").strip()
        scene_id = str(cfg.raw.get("scene_id") or "DowntownWest").strip()
        episodes: list[EvalEpisode] = []
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError(f"UAV-Flow task JSON must contain an object: {path}")
            instruction = _instruction(record, instruction_type=instruction_type, path=path)
            source_id = path.stem
            payload = _payload(record, path=path, instruction=instruction, scene_id=scene_id)
            episodes.append(
                EvalEpisode(
                    episode_uid=f"{cfg.namespace}:{source_id}",
                    source_episode_id=source_id,
                    scene_id=scene_id,
                    instruction=instruction,
                    source="uavflow_json",
                    input_namespace=cfg.namespace,
                    input_root=str(cfg.path),
                    payload=payload,
                )
            )
            if max_samples is not None and len(episodes) >= int(max_samples):
                break
        return episodes

    def fingerprint(self, cfg: InputConfig) -> str:
        if cfg.path is None:
            raise ValueError("input.path is required for UAV-Flow JSON fingerprint")
        digest = hashlib.sha256()
        digest.update(str(cfg.namespace).encode("utf-8"))
        for path in _json_paths(cfg.path):
            digest.update(str(path.name).encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()


def _json_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"UAV-Flow input path does not exist: {path}")
    paths = sorted(path.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"UAV-Flow input directory contains no JSON files: {path}")
    return paths


def _instruction(record: dict[str, Any], *, instruction_type: str, path: Path) -> str:
    candidates = [instruction_type]
    if instruction_type != "instruction":
        candidates.append("instruction")
    if instruction_type != "instruction_unified":
        candidates.append("instruction_unified")
    for key in candidates:
        value = str(record.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"UAV-Flow task JSON has no instruction: {path}")


def _payload(record: dict[str, Any], *, path: Path, instruction: str, scene_id: str) -> dict[str, Any]:
    initial_pos = _required_pose(record, "initial_pos", path)
    end_pos = _required_pose(record, "end_pos", path)
    preprocessed = record.get("reference_path_preprocessed")
    if not isinstance(preprocessed, list) or not preprocessed:
        raise ValueError(f"UAV-Flow task JSON is missing reference_path_preprocessed: {path}")
    payload = dict(record)
    payload.update(
        {
            "scene_id": scene_id,
            "env_name": scene_id,
            "task_json_path": str(path),
            "instruction": instruction,
            "initial_pos_cm": initial_pos,
            "end_pos_cm": end_pos,
            "reference_path_preprocessed_cm": preprocessed,
            "reference_path_preprocessed_m": [preprocessed_uavflow_pose_cm_to_nav_m(item) for item in preprocessed],
        }
    )
    if "target_pos" in record:
        payload["target_pos_cm"] = record["target_pos"]
    if "obj_id" in record and "use_obj" in record:
        obj_pos = record.get("target_pos", record.get("obj_pos"))
        if obj_pos is not None:
            payload["object"] = {
                "obj_id": record["obj_id"],
                "use_obj": record["use_obj"],
                "obj_pos": obj_pos[:3],
                "obj_rot": record.get("obj_rot", obj_pos[3:] if isinstance(obj_pos, list) and len(obj_pos) >= 6 else [0, 0, 0]),
            }
    return payload


def _required_pose(record: dict[str, Any], key: str, path: Path) -> list[float]:
    value = record.get(key)
    if not isinstance(value, list) or len(value) < 6:
        raise ValueError(f"UAV-Flow task JSON is missing {key}: {path}")
    return [float(item) for item in value[:6]]
