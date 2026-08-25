from dataclasses import dataclass
from pathlib import Path

from waypoint_collector.video import probe_video


@dataclass(frozen=True)
class ValidationSummary:
    total_index_rows: int
    available_index_rows: int
    missing_index_rows: int
    video_file_count: int
    frame_metadata_rows: int
    camera_parameter_rows: int


def _line_count(path):
    count = 0
    with Path(path).open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def validate_artifacts(final_root, state, views, skipped_scenes=("1",),
                       deep_video_probe=False, image_width=224,
                       image_height=224):
    import pyarrow.parquet as pq

    final_root = Path(final_root)
    views = tuple(views)
    skipped_scenes = {str(scene_id) for scene_id in skipped_scenes}
    meta_root = final_root / "meta"
    index_path = meta_root / "navvla_video_index.parquet"
    expected_schema = (
        "index", "video_key", "available", "video_frame_index",
        "chunk_index", "file_index",
    )
    parquet_file = pq.ParquetFile(index_path)
    if tuple(parquet_file.schema_arrow.names) != expected_schema:
        raise RuntimeError("video index schema mismatch")
    expected_next_frame = {}
    total_rows = available_rows = missing_rows = 0
    for batch in parquet_file.iter_batches(batch_size=100000):
        data = batch.to_pydict()
        for index in range(batch.num_rows):
            total_rows += 1
            key = (
                data["video_key"][index], data["chunk_index"][index],
                data["file_index"][index],
            )
            if data["available"][index]:
                expected = expected_next_frame.get(key, 0)
                actual = data["video_frame_index"][index]
                if actual != expected:
                    raise RuntimeError(
                        "non-contiguous frame index for {}: got {}, expected {}".format(
                            key, actual, expected
                        )
                    )
                expected_next_frame[key] = expected + 1
                available_rows += 1
            else:
                if data["video_frame_index"][index] != -1:
                    raise RuntimeError("unavailable row must use video_frame_index=-1")
                missing_rows += 1

    expected_total_requests = sum(job.request_count for job in state.iter_jobs())
    expected_missing_requests = sum(
        job.request_count for job in state.iter_jobs()
        if job.scene_id in skipped_scenes
    )
    expected_total_rows = expected_total_requests * len(views)
    expected_missing_rows = expected_missing_requests * len(views)
    if total_rows != expected_total_rows:
        raise RuntimeError("video index total row count mismatch")
    if missing_rows != expected_missing_rows:
        raise RuntimeError("video index missing row count mismatch")
    if available_rows != expected_total_rows - expected_missing_rows:
        raise RuntimeError("video index available row count mismatch")

    expected_paths = set()
    for (video_key, chunk_index, file_index), frame_count in expected_next_frame.items():
        path = (
            final_root / "videos" / video_key /
            "chunk-{:03d}".format(chunk_index) /
            "part-{:03d}.mp4".format(file_index)
        )
        expected_paths.add(path.resolve())
        if not path.is_file() or path.is_symlink() or path.stat().st_size < 1:
            raise RuntimeError("published video is missing or empty: {}".format(path))
        if deep_video_probe:
            info = probe_video(path)
            required = {
                "codec_name": "h264", "profile": "High", "pix_fmt": "yuv420p",
                "width": int(image_width), "height": int(image_height),
                "avg_frame_rate": "1/1",
            }
            for key, expected in required.items():
                if info.get(key) != expected:
                    raise RuntimeError(
                        "{} has {}={}, expected {}".format(path, key, info.get(key), expected)
                    )
            if info["frame_count"] != frame_count:
                raise RuntimeError(
                    "{} has {} frames, index expects {}".format(
                        path, info["frame_count"], frame_count
                    )
                )
    actual_paths = {path.resolve() for path in (final_root / "videos").rglob("*.mp4")}
    if actual_paths != expected_paths:
        raise RuntimeError("published video set does not match index")

    frame_metadata_path = meta_root / "navvla_multiview_frame_metadata.jsonl"
    camera_path = meta_root / "navvla_episode_camera_parameters.jsonl"
    frame_metadata_rows = _line_count(frame_metadata_path)
    camera_parameter_rows = _line_count(camera_path)
    if frame_metadata_rows != expected_total_requests:
        raise RuntimeError("frame metadata row count mismatch")
    expected_camera_rows = sum(1 for _ in state.iter_jobs()) * len(views)
    if camera_parameter_rows != expected_camera_rows:
        raise RuntimeError("camera parameter row count mismatch")
    if not (meta_root / "navvla_cameras.json").is_file():
        raise RuntimeError("static camera metadata is missing")
    return ValidationSummary(
        total_index_rows=total_rows,
        available_index_rows=available_rows,
        missing_index_rows=missing_rows,
        video_file_count=len(actual_paths),
        frame_metadata_rows=frame_metadata_rows,
        camera_parameter_rows=camera_parameter_rows,
    )
