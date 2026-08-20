from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from waypoint_collector.layout import video_key_for_view
from waypoint_collector.video import (
    assemble_part_video,
    probe_video,
    validate_episode_commit,
)


@dataclass(frozen=True)
class AssemblySummary:
    video_file_count: int
    available_frame_count: int
    part_count: int


def assemble_videos(state, episode_video_root, final_root, views,
                    skipped_scenes=("1",), assembly_workers=1):
    episode_video_root = Path(episode_video_root)
    final_root = Path(final_root)
    views = tuple(views)
    assembly_workers = int(assembly_workers)
    if assembly_workers < 1:
        raise ValueError("assembly_workers must be positive")
    skipped_scenes = {str(scene_id) for scene_id in skipped_scenes}
    groups = defaultdict(list)
    available_request_count = 0
    for job in state.iter_jobs():
        if job.scene_id in skipped_scenes:
            if state.job_status(job.episode_id) != "missing_scene":
                raise RuntimeError("skipped episode {} has invalid state".format(job.episode_id))
            continue
        if state.job_status(job.episode_id) != "complete":
            raise RuntimeError("episode {} is not complete".format(job.episode_id))
        if not validate_episode_commit(
            episode_video_root, job.episode_id, views, job.request_count,
            probe_videos=False,
        ):
            raise RuntimeError(
                "episode {} has invalid episode commit".format(job.episode_id)
            )
        global_part = job.source_episode_index // 20
        chunk_index = global_part // 50
        file_index = global_part % 50
        groups[(chunk_index, file_index)].append(job)
        available_request_count += job.request_count

    tasks = []
    for (chunk_index, file_index), jobs in sorted(groups.items()):
        jobs.sort(key=lambda item: item.source_episode_index)
        expected_frames = sum(job.request_count for job in jobs)
        for view in views:
            inputs = [
                episode_video_root / "episodes" / view /
                "{}.mp4".format(job.episode_id)
                for job in jobs
            ]
            for job, path in zip(jobs, inputs):
                if not path.is_file():
                    raise FileNotFoundError(
                        "completed episode {} is missing {} video {}".format(
                            job.episode_id, view, path
                        )
                    )
            output = (
                final_root / "videos" / video_key_for_view(view) /
                "chunk-{:03d}".format(chunk_index) /
                "part-{:03d}.mp4".format(file_index)
            )
            tasks.append((inputs, output, expected_frames))

    def assemble_task(inputs, output, expected_frames):
        if output.exists():
            if probe_video(output)["frame_count"] != expected_frames:
                raise RuntimeError("existing part has wrong frame count: {}".format(output))
        else:
            assemble_part_video(inputs, output)

    if assembly_workers == 1:
        for task in tasks:
            assemble_task(*task)
    else:
        with ThreadPoolExecutor(max_workers=assembly_workers) as executor:
            futures = [executor.submit(assemble_task, *task) for task in tasks]
            for future in as_completed(futures):
                future.result()
    return AssemblySummary(
        video_file_count=len(tasks),
        available_frame_count=available_request_count * len(views),
        part_count=len(groups),
    )
