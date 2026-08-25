from dataclasses import dataclass
from pathlib import Path
import re

from waypoint_collector.requests import iter_render_requests


def scene_sort_key(scene_id):
    parts = re.split(r"(\d+)", str(scene_id))
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in parts
    )


@dataclass(frozen=True)
class RequestIndexSummary:
    total_requests: int
    available_requests: int
    missing_requests: int
    total_episodes: int
    available_episodes: int
    missing_episodes: int
    scene_ids: tuple


def _build_request_index(request_path, state, skipped_scenes=(),
                         shard_target_requests=50000, image_width=224,
                         image_height=224):
    request_path = Path(request_path)
    skipped_scenes = {str(scene_id) for scene_id in skipped_scenes}
    seen_episode_ids = set()
    scene_ids = set()
    total_requests = 0
    missing_requests = 0
    total_episodes = 0
    missing_episodes = 0
    current = None
    last_source_episode_index = -1
    scene_shard_numbers = {}
    scene_shard_counts = {}

    def finish_episode(item):
        nonlocal total_episodes, missing_episodes
        if item is None:
            return
        scene_id = item["scene_id"]
        shard_number = scene_shard_numbers.get(scene_id, 0)
        shard_count = scene_shard_counts.get(scene_id, 0)
        if shard_count and shard_count + item["request_count"] > int(shard_target_requests):
            shard_number += 1
            shard_count = 0
        scene_shard_numbers[scene_id] = shard_number
        scene_shard_counts[scene_id] = shard_count + item["request_count"]
        state.add_episode(
            episode_id=item["episode_id"],
            scene_id=item["scene_id"],
            source_episode_index=item["source_episode_index"],
            byte_start=item["byte_start"],
            byte_end=item["byte_end"],
            request_count=item["request_count"],
            start_index=item["start_index"],
            shard_id="scene-{}-shard-{:04d}".format(scene_id, shard_number),
        )
        total_episodes += 1
        if item["scene_id"] in skipped_scenes:
            state.mark_missing_scene(item["episode_id"])
            missing_episodes += 1

    for request in iter_render_requests(
        request_path, expected_width=image_width, expected_height=image_height
    ):
        total_requests += 1
        scene_ids.add(request.scene_id)
        if request.scene_id in skipped_scenes:
            missing_requests += 1
        if current is None or request.episode_id != current["episode_id"]:
            finish_episode(current)
            if request.episode_id in seen_episode_ids:
                raise ValueError(
                    "episode {} is not contiguous in render request file".format(
                        request.episode_id
                    )
                )
            seen_episode_ids.add(request.episode_id)
            if request.source_episode_index <= last_source_episode_index:
                raise ValueError(
                    "source_episode_index must increase between episodes"
                )
            last_source_episode_index = request.source_episode_index
            current = {
                "episode_id": request.episode_id,
                "scene_id": request.scene_id,
                "source_episode_index": request.source_episode_index,
                "byte_start": request.byte_start,
                "byte_end": request.byte_end,
                "request_count": 1,
                "start_index": request.index,
                "last_image_index": request.image_index,
                "last_waypoint_index": request.waypoint_index,
            }
            if request.image_index != 0:
                raise ValueError("first image_index in episode must be zero")
            continue
        if request.scene_id != current["scene_id"]:
            raise ValueError("scene_id changed within episode {}".format(request.episode_id))
        if request.source_episode_index != current["source_episode_index"]:
            raise ValueError("source_episode_index changed within episode {}".format(request.episode_id))
        if request.image_index != current["last_image_index"] + 1:
            raise ValueError("image_index is not contiguous within episode {}".format(request.episode_id))
        if request.waypoint_index <= current["last_waypoint_index"]:
            raise ValueError("waypoint_index is not increasing within episode {}".format(request.episode_id))
        current["byte_end"] = request.byte_end
        current["request_count"] += 1
        current["last_image_index"] = request.image_index
        current["last_waypoint_index"] = request.waypoint_index
    finish_episode(current)
    if total_requests == 0:
        raise ValueError("render request file is empty")

    summary = RequestIndexSummary(
        total_requests=total_requests,
        available_requests=total_requests - missing_requests,
        missing_requests=missing_requests,
        total_episodes=total_episodes,
        available_episodes=total_episodes - missing_episodes,
        missing_episodes=missing_episodes,
        scene_ids=tuple(sorted(scene_ids, key=scene_sort_key)),
    )
    for key, value in summary.__dict__.items():
        state.set_metadata(key, ",".join(value) if isinstance(value, tuple) else value)
    state.set_metadata("request_path", str(request_path.resolve()))
    state.set_metadata("request_size", request_path.stat().st_size)
    state.set_metadata("request_mtime_ns", request_path.stat().st_mtime_ns)
    state.set_metadata("image_width", int(image_width))
    state.set_metadata("image_height", int(image_height))
    return summary


def build_request_index(request_path, state, skipped_scenes=(),
                        shard_target_requests=50000, image_width=224,
                        image_height=224):
    if any(True for _ in state.iter_jobs()):
        raise RuntimeError("request index state must be empty before indexing")
    try:
        return _build_request_index(
            request_path, state, skipped_scenes=skipped_scenes,
            shard_target_requests=shard_target_requests,
            image_width=image_width, image_height=image_height,
        )
    except Exception:
        state.clear_request_index()
        raise
