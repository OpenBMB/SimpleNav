from dataclasses import dataclass
import json
import os
from pathlib import Path

from waypoint_collector.cameras import episode_camera_rows, static_camera_metadata
from waypoint_collector.layout import part_location, video_key_for_view
from waypoint_collector.requests import iter_render_requests


@dataclass(frozen=True)
class MetadataSummary:
    total_rows: int
    available_rows: int
    missing_rows: int
    frame_metadata_rows: int
    camera_parameter_rows: int


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.partial".format(path.name))
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def write_metadata_artifacts(request_path, state, meta_root, views,
                             camera_seed=1, skipped_scenes=("1",),
                             batch_size=100000, image_width=224,
                             image_height=224):
    import pyarrow as pa
    import pyarrow.parquet as pq

    request_path = Path(request_path)
    meta_root = Path(meta_root)
    meta_root.mkdir(parents=True, exist_ok=True)
    views = tuple(views)
    skipped_scenes = {str(scene_id) for scene_id in skipped_scenes}
    for job in state.iter_jobs():
        expected = "missing_scene" if job.scene_id in skipped_scenes else "complete"
        actual = state.job_status(job.episode_id)
        if actual != expected:
            raise RuntimeError(
                "episode {} is {}, expected {} before metadata assembly".format(
                    job.episode_id, actual, expected
                )
            )

    schema = pa.schema([
        ("index", pa.int64()),
        ("video_key", pa.string()),
        ("available", pa.bool_()),
        ("video_frame_index", pa.int64()),
        ("chunk_index", pa.int64()),
        ("file_index", pa.int64()),
    ])
    final_parquet = meta_root / "navvla_video_index.parquet"
    temporary_parquet = meta_root / ".navvla_video_index.parquet.partial"
    frame_metadata_path = meta_root / "navvla_multiview_frame_metadata.jsonl"
    temporary_frame_metadata = meta_root / ".navvla_multiview_frame_metadata.jsonl.partial"
    counters = {}
    buffer = {field.name: [] for field in schema}
    total_rows = available_rows = missing_rows = frame_rows = 0

    def flush(writer):
        if not buffer["index"]:
            return
        writer.write_table(pa.Table.from_pydict(buffer, schema=schema))
        for values in buffer.values():
            values.clear()

    writer = pq.ParquetWriter(
        str(temporary_parquet), schema=schema, compression="zstd"
    )
    try:
        with temporary_frame_metadata.open("w", encoding="utf-8") as frame_handle:
            for request in iter_render_requests(
                request_path, expected_width=image_width,
                expected_height=image_height,
            ):
                available = request.scene_id not in skipped_scenes
                status = "available" if available else "missing_scene"
                frame_handle.write(
                    json.dumps(request.metadata_payload(status), separators=(",", ":")) + "\n"
                )
                frame_rows += 1
                chunk_index, file_index = part_location(request.source_episode_index)
                for view in views:
                    video_key = video_key_for_view(view)
                    counter_key = (video_key, chunk_index, file_index)
                    if available:
                        frame_index = counters.get(counter_key, 0)
                        counters[counter_key] = frame_index + 1
                        available_rows += 1
                    else:
                        frame_index = -1
                        missing_rows += 1
                    values = (
                        request.index, video_key, available, frame_index,
                        chunk_index, file_index,
                    )
                    for field, value in zip(schema, values):
                        buffer[field.name].append(value)
                    total_rows += 1
                if len(buffer["index"]) >= int(batch_size):
                    flush(writer)
            flush(writer)
    except Exception:
        writer.close()
        temporary_parquet.unlink(missing_ok=True)
        temporary_frame_metadata.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        os.replace(str(temporary_parquet), str(final_parquet))
        os.replace(str(temporary_frame_metadata), str(frame_metadata_path))

    camera_path = meta_root / "navvla_episode_camera_parameters.jsonl"
    temporary_camera = meta_root / ".navvla_episode_camera_parameters.jsonl.partial"
    camera_rows = 0
    with temporary_camera.open("w", encoding="utf-8") as handle:
        for job in state.iter_jobs():
            render_status = (
                "missing_scene" if job.scene_id in skipped_scenes else "complete"
            )
            for row in episode_camera_rows(
                episode_id=job.episode_id,
                scene_id=job.scene_id,
                seed=camera_seed,
                views=views,
                render_status=render_status,
                zero_position_delta_views=job.zero_position_camera_fallback_views,
            ):
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                camera_rows += 1
    os.replace(str(temporary_camera), str(camera_path))
    _atomic_json(
        meta_root / "navvla_cameras.json",
        static_camera_metadata(
            views, image_width=image_width, image_height=image_height
        ),
    )
    return MetadataSummary(
        total_rows=total_rows,
        available_rows=available_rows,
        missing_rows=missing_rows,
        frame_metadata_rows=frame_rows,
        camera_parameter_rows=camera_rows,
    )
