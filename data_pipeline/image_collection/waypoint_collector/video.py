import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid


EPISODE_COMMIT_SCHEMA_VERSION = 1
EXPECTED_VIDEO_FORMAT = {
    "codec_name": "h264",
    "profile": "High",
    "pix_fmt": "yuv420p",
    "width": 224,
    "height": 224,
    "avg_frame_rate": "1/1",
}


def expected_video_format(image_width=224, image_height=224):
    result = dict(EXPECTED_VIDEO_FORMAT)
    result["width"] = int(image_width)
    result["height"] = int(image_height)
    return result


class VideoEncodingError(RuntimeError):
    pass


def _path_component(value, label):
    value = str(value)
    if not value or value in (".", "..") or Path(value).name != value:
        raise ValueError("{} must be one path component".format(label))
    return value


def episode_marker_path(output_root, episode_id):
    episode_id = _path_component(episode_id, "episode_id")
    return Path(output_root) / "commits" / "{}.json".format(episode_id)


def episode_video_paths(output_root, episode_id, views):
    episode_id = _path_component(episode_id, "episode_id")
    return {
        _path_component(view, "view"): Path(output_root) / "episodes" /
        _path_component(view, "view") / "{}.mp4".format(episode_id)
        for view in views
    }


def _video_matches(path, expected_frames, image_width=224, image_height=224):
    if not path.is_file() or path.is_symlink():
        return False
    try:
        info = probe_video(path)
    except Exception:
        return False
    return (
        all(
            info.get(key) == value
            for key, value in expected_video_format(
                image_width, image_height
            ).items()
        )
        and info.get("frame_count") == int(expected_frames)
    )


def validate_episode_videos(output_root, episode_id, views, expected_frames,
                            image_width=224, image_height=224):
    try:
        paths = episode_video_paths(output_root, episode_id, views)
    except (TypeError, ValueError):
        return False
    return bool(paths) and all(
        _video_matches(
            path, expected_frames, image_width=image_width,
            image_height=image_height,
        )
        for path in paths.values()
    )


def legacy_episode_files_present(output_root, episode_id, views):
    """Check legacy complete outputs without decoding every video stream."""
    try:
        paths = episode_video_paths(output_root, episode_id, views)
    except (TypeError, ValueError):
        return False
    try:
        return bool(paths) and all(
            path.is_file() and not path.is_symlink() and path.stat().st_size > 0
            for path in paths.values()
        )
    except OSError:
        return False


def write_episode_commit_marker(output_root, episode_id, views, frame_counts,
                                attempt_id=None):
    output_root = Path(output_root)
    episode_id = _path_component(episode_id, "episode_id")
    views = tuple(_path_component(view, "view") for view in views)
    attempt_id = str(uuid.UUID(str(attempt_id or uuid.uuid4())))
    if set(frame_counts) != set(views):
        raise ValueError("frame counts do not match commit views")
    videos = {}
    for view in views:
        count = frame_counts[view]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("frame count must be a positive integer")
        videos[view] = {
            "path": "episodes/{}/{}.mp4".format(view, episode_id),
            "frame_count": count,
        }
    payload = {
        "schema_version": EPISODE_COMMIT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "episode_id": episode_id,
        "videos": videos,
    }
    marker_path = episode_marker_path(output_root, episode_id)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = marker_path.with_name(".{}.partial".format(marker_path.name))
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(marker_path))
    finally:
        temporary_path.unlink(missing_ok=True)
    return marker_path


def validate_episode_commit(output_root, episode_id, views, expected_frames,
                            probe_videos=True, image_width=224,
                            image_height=224):
    output_root = Path(output_root)
    try:
        episode_id = _path_component(episode_id, "episode_id")
        views = tuple(_path_component(view, "view") for view in views)
        marker_path = episode_marker_path(output_root, episode_id)
        if not marker_path.is_file() or marker_path.is_symlink():
            return False
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if not isinstance(marker, dict) or set(marker) != {
            "schema_version", "attempt_id", "episode_id", "videos"
        }:
            return False
        if (
            isinstance(marker["schema_version"], bool)
            or not isinstance(marker["schema_version"], int)
            or marker["schema_version"] != EPISODE_COMMIT_SCHEMA_VERSION
        ):
            return False
        if marker["episode_id"] != episode_id:
            return False
        attempt_id = marker["attempt_id"]
        if not isinstance(attempt_id, str) or str(uuid.UUID(attempt_id)) != attempt_id:
            return False
        videos = marker["videos"]
        if not isinstance(videos, dict) or set(videos) != set(views):
            return False
        expected_frames = int(expected_frames)
        if expected_frames < 1:
            return False
        root_resolved = output_root.resolve()
        for view in views:
            item = videos[view]
            expected_relative = "episodes/{}/{}.mp4".format(view, episode_id)
            if not isinstance(item, dict) or set(item) != {"path", "frame_count"}:
                return False
            if item["path"] != expected_relative:
                return False
            count = item["frame_count"]
            if isinstance(count, bool) or not isinstance(count, int):
                return False
            if count != expected_frames:
                return False
            video_path = output_root / Path(expected_relative)
            try:
                video_path.resolve().relative_to(root_resolved)
            except ValueError:
                return False
            if probe_videos and not _video_matches(
                video_path, expected_frames, image_width=image_width,
                image_height=image_height,
            ):
                return False
            if (
                not probe_videos
                and (
                    not video_path.is_file()
                    or video_path.is_symlink()
                    or video_path.stat().st_size < 1
                )
            ):
                return False
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _remove_path(path):
    path = Path(path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def remove_episode_transient_artifacts(output_root, episode_id, views):
    output_root = Path(output_root)
    episode_id = _path_component(episode_id, "episode_id")
    attempt_root = output_root / "attempts" / episode_id
    if attempt_root.exists() or attempt_root.is_symlink():
        _remove_path(attempt_root)
    for path in episode_video_paths(output_root, episode_id, views).values():
        partial = path.with_name(".{}.partial.mp4".format(path.stem))
        if partial.exists() or partial.is_symlink():
            _remove_path(partial)
    marker = episode_marker_path(output_root, episode_id)
    marker_partial = marker.with_name(".{}.partial".format(marker.name))
    if marker_partial.exists() or marker_partial.is_symlink():
        _remove_path(marker_partial)


def remove_episode_artifacts(output_root, episode_id, views):
    remove_episode_transient_artifacts(output_root, episode_id, views)
    marker = episode_marker_path(output_root, episode_id)
    if marker.exists() or marker.is_symlink():
        _remove_path(marker)
    for path in episode_video_paths(output_root, episode_id, views).values():
        if path.exists() or path.is_symlink():
            _remove_path(path)


class RawVideoWriter:
    def __init__(self, output_path, ffmpeg="ffmpeg", image_width=224,
                 image_height=224):
        self.output_path = Path(output_path)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.image_width % 2 or self.image_height % 2:
            raise ValueError("image dimensions must be even for YUV420P video")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", "{}x{}".format(
                    self.image_width, self.image_height
                ), "-framerate", "1",
                "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-profile:v", "high", "-pix_fmt", "yuv420p",
                "-crf", "23", "-preset", "medium",
                "-movflags", "+faststart", str(self.output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.frame_count = 0

    def append(self, frame):
        expected_shape = (self.image_height, self.image_width, 3)
        if frame.shape != expected_shape or str(frame.dtype) != "uint8":
            raise ValueError(
                "video frame must be uint8 {}x{}x3".format(
                    self.image_width, self.image_height
                )
            )
        if self.process.poll() is not None:
            error = self.process.stderr.read().decode("utf-8", errors="replace")
            raise VideoEncodingError("ffmpeg exited early: {}".format(error))
        self.process.stdin.write(frame.tobytes(order="C"))
        self.frame_count += 1

    def close(self):
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            error = self.process.stderr.read().decode("utf-8", errors="replace")
            return_code = self.process.wait()
        finally:
            self.process.stderr.close()
        if return_code != 0:
            raise VideoEncodingError(
                "ffmpeg failed with code {}: {}".format(return_code, error)
            )

    def abort(self):
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.communicate(timeout=5)
        except Exception:
            pass
        self.output_path.unlink(missing_ok=True)


class EpisodeVideoSink:
    def __init__(self, output_root, episode_id, views, ffmpeg="ffmpeg",
                 image_width=224, image_height=224):
        self.output_root = Path(output_root)
        self.episode_id = str(episode_id)
        self.views = tuple(views)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        if not self.views:
            raise ValueError("at least one video view is required")
        self.final_paths = episode_video_paths(
            self.output_root, self.episode_id, self.views
        )
        existing = [path for path in self.final_paths.values() if path.exists()]
        marker_path = episode_marker_path(self.output_root, self.episode_id)
        if marker_path.exists():
            existing.append(marker_path)
        if existing:
            raise FileExistsError("episode video already exists: {}".format(existing[0]))
        self.attempt_id = str(uuid.uuid4())
        self.attempt_dir = (
            self.output_root / "attempts" / self.episode_id / self.attempt_id
        )
        self.temporary_paths = {
            view: self.attempt_dir / "{}.mp4".format(view)
            for view in self.views
        }
        self.committed = False
        self.writers = {}
        try:
            for view in self.views:
                self.writers[view] = RawVideoWriter(
                    self.temporary_paths[view], ffmpeg=ffmpeg,
                    image_width=self.image_width,
                    image_height=self.image_height,
                )
        except Exception:
            self.abort()
            raise

    def append(self, request, frames):
        if set(frames) != set(self.views):
            raise ValueError("frame views do not match episode sink")
        for view in self.views:
            self.writers[view].append(frames[view])

    def commit(self):
        try:
            for writer in self.writers.values():
                writer.close()
            counts = {writer.frame_count for writer in self.writers.values()}
            if len(counts) != 1 or counts == {0}:
                raise VideoEncodingError("episode views have unequal or zero frames")
            expected_frames = next(iter(counts))
            for view, path in self.temporary_paths.items():
                info = probe_video(path)
                if any(
                    info.get(key) != value
                    for key, value in expected_video_format(
                        self.image_width, self.image_height
                    ).items()
                ):
                    raise VideoEncodingError(
                        "episode {} {} video format mismatch".format(
                            self.episode_id, view
                        )
                    )
                if info["frame_count"] != expected_frames:
                    raise VideoEncodingError(
                        "episode {} {} frame count mismatch".format(
                            self.episode_id, view
                        )
                    )
            for path in self.final_paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            for view in self.views:
                os.replace(str(self.temporary_paths[view]), str(self.final_paths[view]))
            write_episode_commit_marker(
                self.output_root,
                self.episode_id,
                self.views,
                {view: expected_frames for view in self.views},
                attempt_id=self.attempt_id,
            )
            self.committed = True
            if self.attempt_dir.exists():
                self.attempt_dir.rmdir()
            episode_attempt_root = self.attempt_dir.parent
            if episode_attempt_root.exists() and not any(episode_attempt_root.iterdir()):
                episode_attempt_root.rmdir()
        except Exception:
            self.abort()
            raise

    def abort(self):
        if self.committed:
            return
        for writer in self.writers.values():
            try:
                writer.abort()
            except Exception:
                pass
        for path in tuple(self.temporary_paths.values()) + tuple(self.final_paths.values()):
            path.unlink(missing_ok=True)
        marker = episode_marker_path(self.output_root, self.episode_id)
        marker.unlink(missing_ok=True)
        if self.attempt_dir.exists():
            shutil.rmtree(self.attempt_dir)
        episode_attempt_root = self.attempt_dir.parent
        if episode_attempt_root.exists() and not any(episode_attempt_root.iterdir()):
            episode_attempt_root.rmdir()


def probe_video(path, ffprobe="ffprobe"):
    completed = subprocess.run(
        [
            ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt,width,height,avg_frame_rate,nb_read_frames,nb_frames",
            "-of", "json", str(path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if completed.returncode != 0:
        raise VideoEncodingError(
            "ffprobe failed for {}: {}".format(path, completed.stderr)
        )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise VideoEncodingError("{} does not contain one video stream".format(path))
    stream = streams[0]
    frame_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    stream["frame_count"] = int(frame_count)
    return stream


def assemble_part_video(episode_paths, output_path, ffmpeg="ffmpeg"):
    episode_paths = tuple(Path(path).resolve() for path in episode_paths)
    if not episode_paths:
        raise ValueError("cannot assemble an empty part")
    for path in episode_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(".{}.partial.mp4".format(output_path.stem))
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt",
        dir=str(output_path.parent), delete=False,
    ) as handle:
        concat_path = Path(handle.name)
        for path in episode_paths:
            escaped = str(path).replace("'", "'\\''")
            handle.write("file '{}'\n".format(escaped))
    try:
        completed = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat_path),
                "-c", "copy", "-movflags", "+faststart", str(temporary_output),
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        if completed.returncode != 0:
            raise VideoEncodingError(
                "ffmpeg concat failed for {}: {}".format(
                    output_path, completed.stderr
                )
            )
        os.replace(str(temporary_output), str(output_path))
    finally:
        concat_path.unlink(missing_ok=True)
        temporary_output.unlink(missing_ok=True)
    return output_path
