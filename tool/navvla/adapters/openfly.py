from __future__ import annotations

import json
import math
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

from tool.navvla.adapters.base import NavVLASourceAdapter, register_adapter
from tool.navvla.context_index import ContextIndexConfig
from tool.navvla.workers import resolve_workers as resolve_load_workers
from tool.navvla.lerobot_v3_writer import write_navvla_lerobot_dataset
from tool.navvla.schema import NavVLACameraSpec, NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec
from tool.navvla.statistics import body_frame_action_from_pose


SOURCE_SPLIT_TO_ANNOTATION = {
    "train": "train.json",
    "vln_train": "train.json",
    "seen": "seen.json",
    "val_seen": "seen.json",
    "vln_val_seen": "seen.json",
    "unseen": "unseen.json",
    "val_unseen": "unseen.json",
    "vln_val_unseen": "unseen.json",
}
ANNOTATION_TO_TARGET_SPLIT = {
    "train.json": "vln_train",
    "seen.json": "vln_val_seen",
    "unseen.json": "vln_val_unseen",
}
OPENFLY_CAMERA = NavVLACameraSpec(
    name="front",
    video_key="front_image",
    viewpoint_type="front",
    azimuth_rad=0.0,
    calibration_status="unknown",
)
PLATFORM_TEXT = "Platform: UAV. Task: instruction-conditioned navigation. Action: local 3D waypoints (dx, dy, dz, dyaw)."
OPENFLY_CONTEXT_INDEX_CONFIG = ContextIndexConfig(
    bats_token_budget=1024,
    budget_num_cameras=1,
    history_camera_names=("front",),
)
OPENFLY_IMAGE_SIZE = (646, 480)


class OpenFlyIncompleteSourceError(ValueError):
    def __init__(
        self,
        *,
        image_path: str,
        source_parquet: Path,
        source_image_path: str,
        frame_index: int,
        reason: str,
    ) -> None:
        super().__init__(f"{reason} for {image_path} frame {frame_index} in {source_parquet}")
        self.image_path = image_path
        self.source_parquet = source_parquet
        self.source_image_path = source_image_path
        self.frame_index = frame_index
        self.reason = reason

    def report(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "source_parquet": str(self.source_parquet),
            "source_image_path": self.source_image_path,
            "frame_index": int(self.frame_index),
            "reason": self.reason,
        }


class OpenFlyCachedMediaError(ValueError):
    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class OpenFlyAdapter(NavVLASourceAdapter):
    name = "openfly"

    def __init__(
        self,
        *,
        annotation_root: str | Path | None = None,
        traj_root: str | Path | None = None,
        media_cache_root: str | Path | None = None,
        fps: float = 5.0,
        action_horizon: int = 8,
        scene_prefixes: tuple[str, ...] | list[str] | None = None,
        fail_on_missing_source: bool = False,
        reuse_media_cache: bool = False,
    ) -> None:
        self.annotation_root = Path(annotation_root) if annotation_root is not None else None
        self.traj_root = Path(traj_root) if traj_root is not None else None
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.scene_prefixes = tuple(prefix for prefix in (scene_prefixes or ()) if prefix)
        self.fail_on_missing_source = bool(fail_on_missing_source)
        self.reuse_media_cache = bool(reuse_media_cache)
        self.summary: dict[str, Any] = {}
        self.load_workers: int | None = None

    def configure(
        self,
        *,
        annotation_root: str | Path | None = None,
        traj_root: str | Path | None = None,
        media_cache_root: str | Path | None = None,
        fps: float = 5.0,
        action_horizon: int = 8,
        scene_prefixes: tuple[str, ...] | list[str] | None = None,
        fail_on_missing_source: bool = False,
        reuse_media_cache: bool = False,
        load_workers: int | None = None,
        **kwargs: Any,
    ) -> "OpenFlyAdapter":
        super().configure(**kwargs)
        self.annotation_root = Path(annotation_root) if annotation_root is not None else None
        self.traj_root = Path(traj_root) if traj_root is not None else None
        self.media_cache_root = Path(media_cache_root) if media_cache_root is not None else None
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        self.scene_prefixes = tuple(prefix for prefix in (scene_prefixes or ()) if prefix)
        self.fail_on_missing_source = bool(fail_on_missing_source)
        self.reuse_media_cache = bool(reuse_media_cache)
        self.load_workers = load_workers
        return self

    def load_episodes(
        self,
        source_root: str | Path,
        *,
        split: str = "train",
        max_episodes: int | None = None,
        load_workers: int | None = None,
    ) -> list[NavVLAEpisode]:
        source_root = Path(source_root)
        annotation_root = resolve_annotation_root(source_root, annotation_root=self.annotation_root)
        traj_root = resolve_traj_root(source_root, traj_root=self.traj_root)
        annotation_file = annotation_file_for_split(annotation_root, split)
        target_split = target_split_name(split)
        records = load_annotation_records(annotation_file)
        records, skipped_prefix_records = filter_records_by_prefix(records, self.scene_prefixes)
        media_cache_root = resolve_media_cache_root(source_root, media_cache_root=self.media_cache_root)

        jobs: list[tuple[Any, ...]] = []
        missing_source_examples: list[dict[str, Any]] = []
        missing_source_records = 0
        for record in records:
            parquet_path = parquet_path_for_record(traj_root, record)
            if not parquet_path.exists():
                missing = {"image_path": str(record.get("image_path") or ""), "source_parquet": str(parquet_path)}
                if self.fail_on_missing_source:
                    raise FileNotFoundError(f"Missing OpenFly parquet: {parquet_path}")
                missing_source_records += 1
                if len(missing_source_examples) < 20:
                    missing_source_examples.append(missing)
                continue
            task_index = len(jobs)
            if max_episodes is not None and task_index >= int(max_episodes):
                break
            episode_id = f"{task_index:06d}"
            jobs.append(
                (
                    str(source_root),
                    str(parquet_path),
                    str(media_cache_root),
                    target_split,
                    episode_id,
                    task_index,
                    float(self.fps),
                    int(self.action_horizon),
                    bool(self.reuse_media_cache),
                    record,
                )
            )

        if not jobs:
            raise FileNotFoundError(f"no convertible OpenFly episodes found in {annotation_file}")

        workers = resolve_load_workers(load_workers)
        if workers == 1 or len(jobs) == 1:
            results = [_load_openfly_episode_job(job) for job in jobs]
        else:
            max_workers = min(workers, len(jobs))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_load_openfly_episode_job, jobs, chunksize=8))

        episodes: list[NavVLAEpisode] = []
        incomplete_source_records = 0
        incomplete_source_examples: list[dict[str, Any]] = []
        for episode, incomplete in results:
            if incomplete is not None:
                incomplete_source_records += 1
                if len(incomplete_source_examples) < 20:
                    incomplete_source_examples.append(incomplete)
                continue
            if episode is not None:
                episodes.append(episode)

        self.summary = {
            "source_root": str(source_root),
            "annotation_root": str(annotation_root),
            "traj_root": str(traj_root),
            "annotation_file": str(annotation_file),
            "source_records": len(records),
            "skipped_prefix_records": skipped_prefix_records,
            "converted_episodes": len(episodes),
            "converted_frames": sum(len(episode.frames) for episode in episodes),
            "missing_source_records": missing_source_records,
            "missing_source_examples": missing_source_examples,
            "incomplete_source_records": incomplete_source_records,
            "incomplete_source_examples": incomplete_source_examples,
            "target_split": target_split,
            "media_cache_root": str(media_cache_root),
            "reuse_media_cache": self.reuse_media_cache,
        }
        return episodes

    def convert(
        self,
        *,
        source_root: str | Path,
        output_root: str | Path,
        dataset_name: str,
        max_episodes: int | None,
        fps: float,
        action_horizon: int,
        overwrite: bool,
        control_frequency_hz: float | None = None,
        repair_existing: bool = False,
        split: str = "train",
        context_policy_version: str = "bats-v1",
        cache_policy_version: str = "smoke-coarse-v1",
        cache_workers: int | None = None,
        load_workers: int | None = None,
        write_visual_token_cache: bool = True,
        visual_token_profile: Any | None = None,
        visual_token_encoder: Any | None = None,
        visual_token_encoder_factory: Any | None = None,
        episodes_per_file: int = 20,
        files_per_chunk: int = 50,
    ) -> dict[str, Any]:
        self.fps = float(fps)
        self.action_horizon = int(action_horizon)
        target_split = target_split_name(split)
        episodes = self.load_episodes(
            source_root,
            split=split,
            max_episodes=max_episodes,
            load_workers=self.load_workers if load_workers is None else load_workers,
        )
        spec = NavVLADatasetSpec(
            dataset_name=dataset_name,
            fps=fps,
            control_frequency_hz=float(control_frequency_hz) if control_frequency_hz is not None else float(fps),
            action_horizon=action_horizon,
            action_dim=4,
            state_dim=4,
            context_policy_version=context_policy_version,
            cache_policy_version=cache_policy_version,
            split=target_split,
            episodes_per_file=episodes_per_file,
            files_per_chunk=files_per_chunk,
        )
        summary = write_navvla_lerobot_dataset(
            episodes,
            output_root=Path(output_root),
            spec=spec,
            overwrite=overwrite,
            repair_existing=repair_existing,
            cache_workers=cache_workers,
            write_visual_token_cache=write_visual_token_cache,
            visual_token_profile=visual_token_profile,
            visual_token_encoder=visual_token_encoder,
            visual_token_encoder_factory=visual_token_encoder_factory,
            context_index_config=OPENFLY_CONTEXT_INDEX_CONFIG,
        )
        summary["adapter_summary"] = self.summary
        summary["openfly_filter_report"] = str(write_filter_report(Path(summary["dataset_root"]), self.summary))
        return summary


def normalize_split(split: str) -> str:
    normalized = split.strip()
    if normalized not in SOURCE_SPLIT_TO_ANNOTATION:
        raise ValueError(f"unsupported OpenFly split: {split}")
    return normalized


def annotation_file_for_split(annotation_root: Path, split: str) -> Path:
    filename = SOURCE_SPLIT_TO_ANNOTATION[normalize_split(split)]
    path = annotation_root / filename
    if not path.is_file():
        raise FileNotFoundError(f"OpenFly annotation file not found: {path}")
    return path


def target_split_name(split: str) -> str:
    filename = SOURCE_SPLIT_TO_ANNOTATION[normalize_split(split)]
    return ANNOTATION_TO_TARGET_SPLIT[filename]


def resolve_annotation_root(source_root: Path, *, annotation_root: str | Path | None) -> Path:
    if annotation_root is not None:
        return Path(annotation_root)
    candidate = source_root / "Annotation"
    if candidate.is_dir():
        return candidate
    return source_root


def resolve_traj_root(source_root: Path, *, traj_root: str | Path | None) -> Path:
    if traj_root is not None:
        return Path(traj_root)
    candidate = source_root / "traj"
    if candidate.is_dir():
        return candidate
    return source_root


def resolve_media_cache_root(source_root: Path, *, media_cache_root: str | Path | None) -> Path:
    if media_cache_root is not None:
        return Path(media_cache_root)
    return source_root / ".navvla_media_cache" / "openfly"


def load_annotation_records(annotation_file: Path) -> list[dict[str, Any]]:
    records = json.loads(annotation_file.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {annotation_file}, got {type(records).__name__}")
    return records


def filter_records_by_prefix(records: list[dict[str, Any]], scene_prefixes: tuple[str, ...]) -> tuple[list[dict[str, Any]], int]:
    if not scene_prefixes:
        return records, 0
    selected = [record for record in records if str(record.get("image_path") or "").startswith(scene_prefixes)]
    return selected, len(records) - len(selected)


def parquet_path_for_record(traj_root: Path, record: dict[str, Any]) -> Path:
    image_path = str(record.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("OpenFly annotation record has empty image_path")
    return traj_root / f"{image_path}.parquet"


def _load_openfly_episode_job(job: tuple[Any, ...]) -> tuple[NavVLAEpisode | None, dict[str, Any] | None]:
    (
        source_root_str,
        parquet_path_str,
        media_cache_root_str,
        target_split,
        episode_id,
        task_index,
        fps,
        action_horizon,
        reuse_media_cache,
        record,
    ) = job
    try:
        episode = build_openfly_episode(
            source_root=Path(source_root_str),
            parquet_path=Path(parquet_path_str),
            media_cache_root=Path(media_cache_root_str),
            target_split=str(target_split),
            episode_id=str(episode_id),
            task_index=int(task_index),
            fps=float(fps),
            action_horizon=int(action_horizon),
            reuse_media_cache=bool(reuse_media_cache),
            record=record,
        )
    except OpenFlyIncompleteSourceError as exc:
        return None, exc.report()
    return episode, None


def build_openfly_episode(
    *,
    source_root: Path,
    parquet_path: Path,
    media_cache_root: Path,
    target_split: str,
    episode_id: str,
    task_index: int,
    fps: float,
    action_horizon: int,
    reuse_media_cache: bool,
    record: dict[str, Any],
) -> NavVLAEpisode:
    length = validate_record_lengths(record)
    image_path = str(record["image_path"])
    instruction = str(record.get("gpt_instruction") or "").strip()
    if not instruction:
        raise ValueError(f"OpenFly record has empty gpt_instruction: {image_path}")
    states = [
        state4(pos, yaw, image_path=image_path, frame_index=frame_index)
        for frame_index, (pos, yaw) in enumerate(zip(record["pos"], record["yaw"], strict=True))
    ]
    source_rows = load_parquet_image_rows(parquet_path, reuse_media_cache=reuse_media_cache)
    scene_id, task_subtype = scene_and_subtype_from_image_path(image_path)
    task = NavVLATaskSpec(
        task_index=task_index,
        instruction=instruction,
        task_type="navigation",
        task_subtype=task_subtype,
        platform_text=PLATFORM_TEXT,
        dataset_source="openfly",
        scene_id=scene_id,
        metadata={"source_image_path": image_path},
    )

    frames: list[NavVLAFrame] = []
    for frame_position in range(length):
        source_image_id = str(record["index_list"][frame_position])
        source_image_path = f"{source_image_id}.png"
        parquet_row = source_rows.get(source_image_path, {})
        if not reuse_media_cache and not parquet_row:
            raise OpenFlyIncompleteSourceError(
                image_path=image_path,
                source_parquet=parquet_path,
                source_image_path=source_image_path,
                frame_index=frame_position,
                reason="missing parquet image row",
            )
        image_value = None if reuse_media_cache else parquet_row.get("image")
        if not reuse_media_cache and not isinstance(image_value, dict):
            raise OpenFlyIncompleteSourceError(
                image_path=image_path,
                source_parquet=parquet_path,
                source_image_path=source_image_path,
                frame_index=frame_position,
                reason="missing image struct",
            )
        if not reuse_media_cache and not image_value.get("bytes"):
            raise OpenFlyIncompleteSourceError(
                image_path=image_path,
                source_parquet=parquet_path,
                source_image_path=source_image_path,
                frame_index=frame_position,
                reason="missing image bytes",
            )
        try:
            media_path = materialize_openfly_image(
                image_value,
                media_cache_root=media_cache_root,
                target_split=target_split,
                episode_id=episode_id,
                frame_index=frame_position,
                reuse_media_cache=reuse_media_cache,
            )
        except OpenFlyCachedMediaError as exc:
            if reuse_media_cache:
                raise OpenFlyIncompleteSourceError(
                    image_path=image_path,
                    source_parquet=parquet_path,
                    source_image_path=source_image_path,
                    frame_index=frame_position,
                    reason=exc.reason,
                ) from exc
            raise
        source_metadata = {
            "source_dataset": "openfly",
            "source_image_path": source_image_path,
            "source_image_id": source_image_id,
            "source_annotation_image_path": image_path,
            "source_annotation_action_id": int(record["action"][frame_position]),
            "source_parquet": str(parquet_path),
            "source_parquet_frame_index": parquet_row.get("frame_index"),
            "source_parquet_action_type": parquet_row.get("action_type"),
            "source_parquet_action_value": parquet_row.get("action_value"),
            "source_scene_id": scene_id,
            "source_task_subtype": task_subtype,
        }
        frames.append(
            NavVLAFrame(
                frame_index=frame_position,
                timestamp=float(frame_position) / float(fps),
                media_paths={"front_image": media_path},
                state=states[frame_position],
                action=action_chunk_for_frame(states, frame_idx=frame_position, horizon=action_horizon),
                action_available=True,
                source_frame_index=frame_position,
                source_metadata=source_metadata,
            )
        )

    return NavVLAEpisode(
        episode_id=episode_id,
        trajectory_id=image_path,
        task=task,
        frames=frames,
        cameras=[OPENFLY_CAMERA],
        split=target_split,
    )


def validate_record_lengths(record: dict[str, Any]) -> int:
    lengths = {
        "action": len(record.get("action", [])),
        "index_list": len(record.get("index_list", [])),
        "pos": len(record.get("pos", [])),
        "yaw": len(record.get("yaw", [])),
    }
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"Mismatched OpenFly entry lengths for {record.get('image_path')}: {lengths}")
    length = unique_lengths.pop()
    if length < 2:
        raise ValueError(f"OpenFly entry {record.get('image_path')} has only {length} aligned frames")
    return length


def load_parquet_image_rows(parquet_path: Path, *, reuse_media_cache: bool) -> dict[str, dict[str, Any]]:
    columns = ["image_id", "frame_index", "action_type", "action_value"]
    if not reuse_media_cache:
        columns.insert(0, "image")
    table = pq.read_table(parquet_path, columns=columns)
    rows: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        image_info = row.get("image") if not reuse_media_cache else None
        image_path = image_source_path(image_info)
        if image_path is None:
            image_path = f"{row['image_id']}.png"
        rows[str(image_path)] = row
    return rows


def state4(pos: Any, yaw: Any, *, image_path: str, frame_index: int) -> list[float]:
    values = list(pos or [])
    if len(values) not in {3, 4}:
        raise ValueError(f"OpenFly pos must have length 3 or 4 at {image_path} frame {frame_index}, got {len(values)}")
    if len(values) == 4 and abs(float(values[3]) - float(yaw)) > 1e-5:
        raise ValueError(
            f"OpenFly pos[3] must match yaw at {image_path} frame {frame_index}: pos[3]={values[3]} yaw={yaw}"
        )
    return [_clean_float(values[0]), _clean_float(values[1]), _clean_float(values[2]), _clean_float(yaw)]


def action_chunk_for_frame(poses: list[list[float]], *, frame_idx: int, horizon: int) -> list[list[float]]:
    current = poses[frame_idx]
    chunk = []
    for future_idx in range(frame_idx + 1, min(len(poses), frame_idx + 1 + horizon)):
        action = body_frame_action_from_pose(current, poses[future_idx]).astype(float).tolist()
        chunk.append([_clean_float(value) for value in action])
    return chunk


def materialize_openfly_image(
    image_value: Any,
    *,
    media_cache_root: Path,
    target_split: str,
    episode_id: str,
    frame_index: int,
    reuse_media_cache: bool,
) -> Path:
    path = media_cache_root / target_split / episode_id / f"{frame_index:06d}.png"
    if reuse_media_cache:
        if not path.exists() or path.stat().st_size <= 0:
            raise OpenFlyCachedMediaError(
                f"missing cached OpenFly image for episode {episode_id} frame {frame_index}: {path}",
                reason="missing cached media",
            )
        try:
            with Image.open(path) as image:
                if image.size != OPENFLY_IMAGE_SIZE:
                    raise OpenFlyCachedMediaError(
                        f"stale cached OpenFly image size for episode {episode_id} frame {frame_index}: "
                        f"{image.size} != {OPENFLY_IMAGE_SIZE} at {path}",
                        reason="stale cached media size",
                    )
        except OpenFlyCachedMediaError:
            raise
        except Exception as exc:
            raise OpenFlyCachedMediaError(
                f"invalid cached OpenFly image for episode {episode_id} frame {frame_index}: {path}",
                reason="invalid cached media",
            ) from exc
        return path
    if not isinstance(image_value, dict):
        raise ValueError(f"expected OpenFly image struct for episode {episode_id} frame {frame_index}")
    image_bytes = image_value.get("bytes")
    if not image_bytes:
        raise ValueError(f"missing OpenFly image bytes for episode {episode_id} frame {frame_index}")
    image_bytes = normalize_openfly_png_bytes(image_bytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == image_bytes:
        return path
    path.write_bytes(image_bytes)
    return path


def normalize_openfly_png_bytes(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
        if rgb.size != OPENFLY_IMAGE_SIZE:
            rgb = rgb.resize(OPENFLY_IMAGE_SIZE, Image.Resampling.BICUBIC)
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
        return buffer.getvalue()


def image_source_path(image_value: Any) -> str | None:
    if not isinstance(image_value, dict):
        return None
    value = image_value.get("path")
    return None if value is None else str(value)


def scene_and_subtype_from_image_path(image_path: str) -> tuple[str, str]:
    parts = image_path.split("/")
    scene_id = parts[0] if parts and parts[0] else "openfly"
    if len(parts) >= 3 and parts[1] == "astar_data":
        return scene_id, parts[2]
    if len(parts) >= 2:
        return scene_id, parts[1]
    return scene_id, "openfly"


def write_filter_report(dataset_root: Path, summary: dict[str, Any]) -> Path:
    report_path = dataset_root / "meta" / "openfly_filter_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return report_path


def _clean_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"OpenFly numeric value must be finite, got {value}")
    return 0.0 if abs(value) < 1e-7 else value


register_adapter(OpenFlyAdapter())
