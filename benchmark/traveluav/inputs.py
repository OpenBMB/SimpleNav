"""Normalize TravelUAV JSON and LeRobot exports at the input boundary."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from benchmark.common.types import EvalEpisode


def _vector(value, width, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (width,) or not np.isfinite(vector).all():
        raise ValueError(f"TravelUAV {name} must be a finite vector of width {width}")
    return vector.tolist()


def _instruction_description(instruction):
    # The raw dataset embeds stage/heading commands around the target description.
    text = instruction.split("\n\nInstruction: ", 1)[-1]
    text = re.sub(r"^Fly\s+.+?\s+and find the target\.\s*", "", text, count=1, flags=re.I | re.S)
    text = re.sub(r"^.*?degrees from you\.\s*", "", text, count=1, flags=re.I | re.S)
    return re.sub(r"\s*Please control the drone.*$", "", text, flags=re.I | re.S).strip()


def normalize_payload(record):
    """Canonical JSON fields; only source readers translate their storage formats."""
    trajectory = np.asarray(record["trajectory"], dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or not len(trajectory) or not np.isfinite(trajectory).all():
        raise ValueError("TravelUAV trajectory must be a nonempty finite [N, 3] array")
    goal = _vector(record["goal_position"], 3, "goal_position")
    object_info = record.get("object")
    if object_info is None and record.get("object_name"):
        object_info = {"asset_name": record["object_name"], "pose": goal + [0, 0, 0, 1]}
    if object_info is not None:
        object_info = {
            "asset_name": str(object_info["asset_name"]),
            "pose": _vector(object_info["pose"], 7, "object.pose"),
            "scale": _vector(object_info.get("scale", [1, 1, 1]), 3, "object.scale"),
        }
    description = record.get("object_description", "") or record.get("object_desc", "")
    return {
        "env_name": record["env_name"],
        "start_pose": _vector(record["start_pose"], 4, "start_pose"),
        "goal_position": goal,
        "trajectory": trajectory.tolist(),
        "reference_points": trajectory.tolist(),
        "gt_path_length": float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum()),
        "object": object_info,
        "object_desc": record.get("object_desc", "").strip(),
        "object_description": description.strip() or _instruction_description(record["instruction"]),
    }


class TravelUAVJsonInputAdapter:
    def load_episodes(self, cfg, *, max_samples):
        records = json.loads(cfg.path.read_text())
        if not isinstance(records, list):
            raise ValueError("TravelUAV eval JSON must contain a list of canonical episode records")
        allowed = set(cfg.raw.get("scene_ids", []))
        episodes = []
        for record in records:
            scene = str(record["env_name"])
            if allowed and scene not in allowed:
                continue
            source_id = str(record["episode_id"])
            episodes.append(
                EvalEpisode(
                    f"{cfg.namespace}:{source_id}",
                    source_id,
                    scene,
                    record["instruction"],
                    "eval_json",
                    cfg.namespace,
                    str(cfg.path),
                    normalize_payload(record),
                )
            )
            if max_samples is not None and len(episodes) >= max_samples:
                break
        return episodes


class TravelUAVLeRobotV3InputAdapter:
    def load_episodes(self, cfg, *, max_samples):
        roots = [(item.namespace, item.path) for item in cfg.roots] if cfg.roots else [(cfg.namespace, cfg.data_root)]
        episodes = []
        for namespace, root in roots:
            remaining = None if max_samples is None else max_samples - len(episodes)
            episodes.extend(self._load_root(namespace, root, set(cfg.raw.get("scene_ids", [])), remaining))
            if max_samples is not None and len(episodes) >= max_samples:
                break
        return episodes

    def _load_root(self, namespace, root, allowed, limit):
        table = _parquet_shards(root / "meta/episodes").sort_values("episode_index")
        data = _parquet_shards(root / "data")
        if "source_metadata" not in data:
            metadata = pd.read_json(root / "meta/navvla_frame_metadata.jsonl", lines=True)
            data = data.merge(metadata[["index", "source_metadata"]], on="index", how="left", validate="one_to_one")
        grouped = {int(index): frames.sort_values("frame_index") for index, frames in data.groupby("episode_index")}
        sidecar = _benchmark_sidecar(root)
        if sidecar:
            keys = [_sidecar_key(row) for row in table.to_dict("records")]
            if len(set(keys)) != len(keys) or set(keys) != set(sidecar):
                raise ValueError(f"TravelUAV benchmark sidecar must match episode keys one-to-one: {root}")
        episodes = []
        for row in table.to_dict("records"):
            scene, source_id = str(row["scene_id"]), str(row["episode_id"])
            if allowed and scene not in allowed:
                continue
            frames = grouped[int(row["episode_index"])].to_dict("records")
            source = [_source_metadata(frame["source_metadata"]) for frame in frames]
            poses = [_source_pose(metadata["source_state"]) for metadata in source]
            metadata = sidecar[_sidecar_key(row)] if sidecar else source[0]
            instruction = metadata["instruction"] if "instruction" in metadata else row["tasks"][0]
            goal = metadata.get("goal_position", metadata.get("target", poses[-1][:3]))
            # The source converter stores mark.json target as xyz or an x/y/z object.
            if isinstance(goal, dict):
                goal = [goal[axis] for axis in ("x", "y", "z")]
            payload = normalize_payload(
                {
                    **metadata,
                    "instruction": instruction,
                    "env_name": scene,
                    "start_pose": poses[0],
                    "trajectory": [pose[:3] for pose in poses],
                    "goal_position": goal,
                }
            )
            episodes.append(
                EvalEpisode(
                    f"{namespace}:{source_id}",
                    source_id,
                    scene,
                    instruction,
                    "navvla_lerobot_v3",
                    namespace,
                    str(root),
                    payload,
                )
            )
            if limit is not None and len(episodes) >= limit:
                break
        return episodes


def _parquet_shards(path):
    files = sorted(path.glob("chunk-*/part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot parquet shards under {path}")
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def _source_metadata(value):
    # LeRobot stores this column either as a JSON string or an Arrow struct.
    return json.loads(value) if isinstance(value, str) else value


def _source_pose(values):
    if len(values) == 4:
        return _vector(values, 4, "source_state")
    if len(values) == 6:
        return _vector([values[0], values[1], values[2], values[5]], 4, "source_state")
    raise ValueError("TravelUAV source_state must be xyz/yaw or xyz/roll/pitch/yaw")


def _sidecar_key(record):
    return (str(record["episode_id"]), int(record["task_index"]), str(record["trajectory_id"]), str(record["scene_id"]))


def _benchmark_sidecar(root):
    path = root / "meta/navvla_benchmark_episodes.jsonl"
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = _sidecar_key(record)
        if record["benchmark"] != "traveluav" or key in records:
            raise ValueError(f"Invalid or duplicate TravelUAV benchmark sidecar entry {key}: {path}")
        records[key] = record["metadata"]
    return records
