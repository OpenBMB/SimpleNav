"""Public readers for the enhanced trajectory/render interchange format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_episodes(package_dir: str | Path) -> Iterator[dict]:
    path = Path(package_dir) / "trajectories" / "episodes.jsonl"
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            episode = json.loads(line)
            if "episode_id" not in episode or "reference_path" not in episode:
                raise ValueError(f"invalid episode at {path}:{line_number}")
            yield episode


def iter_render_requests(package_dir: str | Path) -> Iterator[dict]:
    path = Path(package_dir) / "render" / "render_requests.jsonl"
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            request = json.loads(line)
            required = (
                "request_id",
                "scene_id",
                "waypoint_index",
                "position_xyz",
                "orientation_quaternion_wxyz",
                "camera_key",
                "expected_image_relpath",
            )
            missing = [key for key in required if key not in request]
            if missing:
                raise ValueError(
                    f"invalid render request at {path}:{line_number}; missing {missing}"
                )
            yield request
