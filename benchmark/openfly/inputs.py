from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from benchmark.common.config import InputConfig
from benchmark.common.types import EvalEpisode


class OpenFlyStarVLAInputAdapter:
    def load_episodes(self, cfg: InputConfig, *, max_samples: int | None) -> list[EvalEpisode]:
        if not cfg.namespace:
            raise ValueError("input.namespace is required for OpenFly input")
        if cfg.data_root is None:
            raise ValueError("input.data_root is required for OpenFly input")
        if not cfg.split:
            raise ValueError("input.split is required for OpenFly input")
        split_path = cfg.data_root / "splits" / f"{cfg.split}.txt"
        episodes_dir = cfg.data_root / "episodes"
        annotation_path = cfg.data_root / "Annotation" / f"{cfg.split}.json"
        source_z_sign = _source_z_sign(cfg.raw)
        if not split_path.exists() and annotation_path.exists():
            return self._load_annotation_episodes(
                annotation_path,
                cfg=cfg,
                max_samples=max_samples,
                source_z_sign=source_z_sign,
            )
        if not split_path.exists():
            raise FileNotFoundError(f"OpenFly split file does not exist: {split_path}")
        source_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        allowed_scene_ids = _allowed_scene_ids(cfg.raw)
        episodes = []
        for source_id in source_ids:
            payload = self._load_payload(episodes_dir / f"{source_id}.json", source_z_sign=source_z_sign)
            scene_id = _explicit_scene_id(payload=payload, cfg=cfg, source_id=source_id)
            if not scene_id:
                raise ValueError(f"OpenFly episode is missing scene_id: {source_id}")
            if allowed_scene_ids and scene_id not in allowed_scene_ids:
                continue
            instruction = str(payload.get("instruction") or payload.get("gpt_instruction") or "").strip()
            if not instruction:
                raise ValueError(f"OpenFly episode is missing instruction: {source_id}")
            env_name = str(payload.get("env_name") or scene_id).strip()
            episode_uid = f"{cfg.namespace}:{source_id}"
            payload["scene_id"] = scene_id
            payload["env_name"] = env_name
            episodes.append(
                EvalEpisode(
                    episode_uid=episode_uid,
                    source_episode_id=source_id,
                    scene_id=scene_id,
                    instruction=instruction,
                    source="starvla_episode_json",
                    input_namespace=cfg.namespace,
                    input_root=str(cfg.data_root),
                    payload=payload,
                )
            )
            if max_samples is not None and len(episodes) >= int(max_samples):
                break
        return episodes

    def _load_payload(self, episode_path: Path, *, source_z_sign: float = 1.0) -> dict[str, Any]:
        if not episode_path.exists():
            raise FileNotFoundError(f"Missing OpenFly episode file: {episode_path}")
        payload = json.loads(episode_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"OpenFly episode must contain an object: {episode_path}")
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"OpenFly episode is missing non-empty frames: {episode_path}")
        positions = []
        yaw = []
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise ValueError(f"OpenFly episode frame {index} must be an object: {episode_path}")
            state = frame.get("state")
            if state is None or len(state) < 4:
                raise ValueError(f"OpenFly episode frame {index} is missing 4D state: {episode_path}")
            position, heading = _source_pose_to_canonical(state[:4], source_z_sign=source_z_sign)
            positions.append(position)
            yaw.append(heading)
        converted = dict(payload)
        converted["pos"] = positions
        converted["reference_points"] = positions
        converted["yaw"] = yaw
        return converted

    def _load_annotation_episodes(
        self,
        annotation_path: Path,
        *,
        cfg: InputConfig,
        max_samples: int | None,
        source_z_sign: float,
    ) -> list[EvalEpisode]:
        records = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"OpenFly annotation file must contain a list: {annotation_path}")
        allowed_scene_ids = _allowed_scene_ids(cfg.raw)
        episodes = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"OpenFly annotation record {index} must be an object: {annotation_path}")
            source_id = str(record.get("episode_id") or record.get("id") or f"{index:06d}")
            scene_id = _explicit_scene_id(payload=record, cfg=cfg, source_id=source_id)
            if not scene_id:
                image_path = str(record.get("image_path") or "").strip()
                scene_id = image_path.split("/", 1)[0] if image_path else ""
            if not scene_id:
                raise ValueError(f"OpenFly annotation record {index} is missing scene_id")
            if allowed_scene_ids and scene_id not in allowed_scene_ids:
                continue
            instruction = str(record.get("instruction") or record.get("gpt_instruction") or "").strip()
            if not instruction:
                raise ValueError(f"OpenFly annotation record {index} is missing instruction")
            positions = record.get("pos") or record.get("positions")
            yaw = record.get("yaw")
            if not positions or not yaw:
                raise ValueError(f"OpenFly annotation record {index} is missing pos/yaw")
            payload = dict(record)
            payload["scene_id"] = scene_id
            payload["env_name"] = str(record.get("env_name") or scene_id)
            converted_poses = [
                _source_pose_to_canonical([*position[:3], heading], source_z_sign=source_z_sign)
                for position, heading in zip(positions, yaw)
            ]
            payload["pos"] = [position for position, _heading in converted_poses]
            payload["reference_points"] = payload["pos"]
            payload["yaw"] = [heading for _position, heading in converted_poses]
            episodes.append(
                EvalEpisode(
                    episode_uid=f"{cfg.namespace}:{source_id}",
                    source_episode_id=source_id,
                    scene_id=scene_id,
                    instruction=instruction,
                    source="openfly_annotation_json",
                    input_namespace=str(cfg.namespace),
                    input_root=str(annotation_path),
                    payload=payload,
                )
            )
            if max_samples is not None and len(episodes) >= int(max_samples):
                break
        return episodes


def _explicit_scene_id(*, payload: dict[str, Any], cfg: InputConfig, source_id: str) -> str:
    scene_id = str(payload.get("scene_id") or payload.get("env_name") or "").strip()
    if scene_id:
        return scene_id
    by_episode = cfg.raw.get("scene_id_by_episode")
    if isinstance(by_episode, dict):
        scene_id = str(by_episode.get(source_id) or "").strip()
        if scene_id:
            return scene_id
    by_split = cfg.raw.get("scene_id_by_split")
    if isinstance(by_split, dict) and cfg.split:
        scene_id = str(by_split.get(cfg.split) or "").strip()
        if scene_id:
            return scene_id
    return ""


def _source_pose_to_canonical(values: Any, *, source_z_sign: float = 1.0) -> tuple[list[float], float]:
    if len(values) < 4:
        raise ValueError(f"OpenFly source pose must have 4 values, got {values!r}")
    x, y, z, yaw = [float(value) for value in values[:4]]
    canonical_yaw = (-yaw + math.pi) % (2 * math.pi) - math.pi
    return [x, -y, z * source_z_sign], canonical_yaw


def _source_z_sign(raw: dict[str, Any]) -> float:
    value = raw.get("source_z_sign", 1.0)
    try:
        sign = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"input.source_z_sign must be either -1 or 1, got {value!r}") from exc
    if sign not in {-1.0, 1.0}:
        raise ValueError(f"input.source_z_sign must be either -1 or 1, got {value!r}")
    return sign


def _allowed_scene_ids(raw: dict[str, Any]) -> set[str]:
    raw_value = raw.get("scene_ids", raw.get("scene_id"))
    if raw_value is None:
        return set()
    values = raw_value if isinstance(raw_value, (list, tuple, set)) else [raw_value]
    return {str(value).strip() for value in values if str(value).strip()}
