import json
import os
from pathlib import Path
import tempfile


def build_updated_manifest(original_manifest, total_episodes,
                           available_episodes, missing_episodes,
                           total_requests, available_requests,
                           missing_requests, skipped_scenes, views,
                           image_width=224, image_height=224):
    manifest = dict(original_manifest)
    manifest.update({
        "trajectory_only": False,
        "complete_lerobot_split": False,
        "image_status": "partial" if missing_requests else "collected",
        "missing_scene_ids": [str(item) for item in skipped_scenes],
        "available_episode_count": int(available_episodes),
        "missing_episode_count": int(missing_episodes),
        "available_render_request_count": int(available_requests),
        "missing_render_request_count": int(missing_requests),
        "render_request_count": int(total_requests),
        "episode_count": int(total_episodes),
        "views": [str(view) for view in views],
        "video_index_file": "meta/navvla_video_index.parquet",
        "multiview_frame_metadata_file": "meta/navvla_multiview_frame_metadata.jsonl",
        "episode_camera_parameters_file": "meta/navvla_episode_camera_parameters.jsonl",
        "image_width": int(image_width),
        "image_height": int(image_height),
        "image_channels": 3,
    })
    return manifest


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".manifest-",
            suffix=".json", dir=str(path.parent), delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def publish_artifacts(staging_final_root, package_dir, manifest):
    staging_final_root = Path(staging_final_root)
    package_dir = Path(package_dir)
    staged_videos = staging_final_root / "videos"
    staged_meta = staging_final_root / "meta"
    final_videos = package_dir / "videos"
    final_meta = package_dir / "meta"
    package_dir.mkdir(parents=True, exist_ok=True)
    journal_path = package_dir / ".waypoint_publish.json"
    if journal_path.is_file():
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if (
            journal.get("staging_final_root") != str(staging_final_root.resolve())
            or journal.get("manifest") != manifest
        ):
            raise RuntimeError("publish journal does not match this staging run")
        metadata_files = tuple(journal.get("metadata_files", ()))
        if not metadata_files:
            names = set()
            if staged_meta.is_dir():
                names.update(path.name for path in staged_meta.iterdir())
            if final_meta.is_dir():
                names.update(path.name for path in final_meta.iterdir())
            metadata_files = tuple(sorted(names))
    else:
        if not staged_videos.is_dir():
            raise FileNotFoundError("staged videos directory is missing")
        if not staged_meta.is_dir():
            raise FileNotFoundError("staged metadata directory is missing")
        if final_videos.exists():
            raise FileExistsError("published videos directory already exists: {}".format(final_videos))
        metadata_files = tuple(sorted(path.name for path in staged_meta.iterdir()))
        for name in metadata_files:
            target = final_meta / name
            if target.exists():
                raise FileExistsError("published metadata already exists: {}".format(target))
        journal = {
            "staging_final_root": str(staging_final_root.resolve()),
            "metadata_files": list(metadata_files),
            "manifest": manifest,
        }
        _atomic_write_json(journal_path, journal)

    final_meta.mkdir(parents=True, exist_ok=True)
    if staged_videos.is_dir():
        if final_videos.exists():
            raise FileExistsError("both staged and published videos exist")
        os.replace(str(staged_videos), str(final_videos))
    elif not final_videos.is_dir():
        raise FileNotFoundError("neither staged nor published videos exist")
    for name in metadata_files:
        staged_file = staged_meta / name
        target = final_meta / name
        if staged_file.exists() and target.exists():
            raise FileExistsError("both staged and published metadata exist: {}".format(name))
        if staged_file.exists():
            os.replace(str(staged_file), str(target))
        elif not target.exists():
            raise FileNotFoundError("metadata is missing from staging and package: {}".format(name))
    _atomic_write_json(package_dir / "manifest.json", manifest)
    journal_path.unlink(missing_ok=True)
