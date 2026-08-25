from pathlib import Path


EPISODES_PER_FILE = 20
FILES_PER_CHUNK = 50


def part_location(source_episode_index):
    source_episode_index = int(source_episode_index)
    if source_episode_index < 0:
        raise ValueError("source_episode_index must be non-negative")
    global_part = source_episode_index // EPISODES_PER_FILE
    return global_part // FILES_PER_CHUNK, global_part % FILES_PER_CHUNK


def video_relative_path(video_key, source_episode_index):
    chunk_index, file_index = part_location(source_episode_index)
    return Path("videos") / video_key / "chunk-{:03d}".format(chunk_index) / "part-{:03d}.mp4".format(file_index)


def video_key_for_view(view):
    if view not in ("front", "back", "left", "right"):
        raise ValueError("unsupported view: {}".format(view))
    return "{}_image".format(view)
