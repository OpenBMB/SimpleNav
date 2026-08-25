from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

import numpy as np
from PIL import Image

from NavVLAeval.common.types import EvalEpisode, StepState


_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_REQUIRED_EVAL_INFO_FIELDS = {
    "benchmark",
    "run_name",
    "config_sha256",
    "input_fingerprint",
    "episode_uid",
    "source_episode_id",
    "input_namespace",
    "input_root",
    "scene_id",
    "status",
    "failure",
}


@dataclass(frozen=True)
class ImageCoercionResult:
    image: np.ndarray | None
    reason: str | None
    shape: tuple[int, ...] | None


@dataclass(frozen=True)
class RunIdentity:
    benchmark: str
    run_name: str
    config_sha256: str
    input_fingerprint: str


@dataclass(frozen=True)
class EvalInfoRecord:
    path: Path
    payload: dict[str, Any] | None
    valid: bool
    error: str | None = None


@dataclass(frozen=True)
class ArtifactStore:
    run_root: Path

    @property
    def config_path(self) -> Path:
        return self.run_root / "config.yaml"

    @property
    def run_plan_path(self) -> Path:
        return self.run_root / "run_plan.json"

    @property
    def summary_path(self) -> Path:
        return self.run_root / "summary.json"

    def worker_plan_path(self, index: int) -> Path:
        return self.run_root / "worker_plans" / f"worker_{int(index)}.json"

    def worker_log_path(self, index: int) -> Path:
        return self.run_root / "worker_logs" / f"worker_{int(index)}.log"

    def settings_root(self, *, worker_index: int, gpu_id: int, port: int) -> Path:
        return self.run_root / "settings" / f"worker_{int(worker_index)}_gpu_{int(gpu_id)}_port_{int(port)}"

    def episode_dir(self, episode: EvalEpisode) -> Path:
        return (
            self.run_root
            / "logs"
            / sanitize_path_component(episode.scene_id)
            / sanitize_path_component(episode.input_namespace)
            / sanitize_path_component(episode.source_episode_id)
        )

    def episode_eval_info_path(self, episode: EvalEpisode) -> Path:
        return self.episode_dir(episode) / "eval_info.json"


class RunLock:
    def __init__(self, path: Path, fd: int):
        self.path = path
        self._fd = fd

    def release(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        self.path.unlink(missing_ok=True)


def sanitize_path_component(value: str) -> str:
    sanitized = _SAFE_COMPONENT_RE.sub("_", str(value).strip())
    sanitized = sanitized.strip("._")
    if not sanitized:
        raise ValueError(f"path component sanitizes to empty: {value!r}")
    if "/" in sanitized:
        raise ValueError(f"path component contains slash after sanitize: {value!r}")
    return sanitized


def validate_sanitized_episode_paths(episodes: list[EvalEpisode]) -> None:
    seen: dict[tuple[str, str], tuple[str, str]] = {}
    for episode in episodes:
        sanitized = (sanitize_path_component(episode.scene_id), sanitize_path_component(episode.episode_uid))
        raw = (episode.scene_id, episode.episode_uid)
        previous = seen.get(sanitized)
        if previous is not None and previous != raw:
            raise ValueError(f"sanitize collision for episode paths: {previous!r} and {raw!r}")
        seen[sanitized] = raw


def write_json_atomic(path: Path, payload: Mapping[str, Any], *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=sort_keys), encoding="utf-8")
    tmp.replace(path)


def json_safe_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    return value


def _coerce_rgb_image_with_diagnostics(image: Any) -> ImageCoercionResult:
    if image is None:
        return ImageCoercionResult(image=None, reason="image is None", shape=None)
    array = np.asarray(image)
    shape = tuple(int(dim) for dim in array.shape)
    if array.size == 0:
        return ImageCoercionResult(image=None, reason="empty image array", shape=shape)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim != 3 or array.shape[-1] not in {1, 3, 4}:
        return ImageCoercionResult(image=None, reason="invalid image shape", shape=shape)
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            return ImageCoercionResult(image=None, reason="non-finite float image", shape=shape)
        if float(np.max(array)) <= 1.0:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return ImageCoercionResult(image=array[:, :, :3], reason=None, shape=shape)


def common_step_payload(state: StepState, *, benchmark_specific: Mapping[str, Any] | None = None) -> dict[str, Any]:
    termination = state.termination
    payload: dict[str, Any] = {
        "waypoint_index": int(state.artifact_step_index),
        "state": _pose_to_list(state.pose_after),
        "action_waypoints": _array_to_list(state.world_waypoints),
        "executed_action_count": int(state.executed_action_count),
        "distance": float(state.distance_after),
        "time": _observation_time(state.post_observation, fallback=_observation_time(state.pre_observation)),
        "done": bool(termination.done) if termination is not None else False,
        "success": bool(termination.success) if termination is not None else False,
        "termination_reason": termination.reason if termination is not None else "running",
        "benchmark_specific": dict(benchmark_specific or {}),
    }
    diagnostics = _artifact_diagnostics(state.diagnostics)
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload


def _artifact_diagnostics(diagnostics: Mapping[str, Any] | None) -> dict[str, Any]:
    """Persist only rollout evidence useful for debugging, not large raw plans."""
    source = dict(diagnostics or {})
    result = {
        key: source[key]
        for key in (
            "action_execution_mode",
            "original_waypoint_count",
            "executed_waypoint_count",
            "selected_waypoint_indices",
            "completed_waypoint_count",
            "attempted_waypoint_count",
            "captured_action_observation_count",
            "history_update",
            "collision",
            "collision_reason",
            "actual_pose",
            "actual_waypoint_poses",
            "waypoint_control",
            "pose_mismatch_count",
            "pose_mismatches",
            "model_input",
        )
        if key in source
    }
    return result


def extract_observation_images(
    observation: Mapping[str, Any],
    *,
    image_cameras: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    allowed = set(image_cameras) if image_cameras is not None else None
    images: dict[str, Any] = {}
    if "image" in observation:
        images["image"] = observation["image"]
    camera_images = observation.get("images")
    if isinstance(camera_images, Mapping):
        images.update({str(key): value for key, value in camera_images.items()})
    episode = observation.get("traveluav_episode")
    if isinstance(episode, Mapping):
        rgb_images = episode.get("rgb")
        folders = ["frontcamera", "leftcamera", "rightcamera", "rearcamera", "downcamera"]
        if isinstance(rgb_images, (list, tuple)):
            for camera_name, image in zip(folders, rgb_images):
                images[camera_name] = image
    if allowed is not None:
        images = {key: value for key, value in images.items() if key in allowed}
    return images


def _pose_to_list(pose: Any) -> list[float] | None:
    if pose is None:
        return None
    if hasattr(pose, "as_array"):
        return [float(value) for value in pose.as_array().tolist()]
    array = np.asarray(pose, dtype=np.float32).reshape(-1)
    return [float(value) for value in array.tolist()]


def _array_to_list(value: Any) -> Any:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).tolist()


def _observation_time(observation: Mapping[str, Any], *, fallback: Any = None) -> Any:
    for key in ("time", "timestamp", "time_stamp"):
        if key in observation:
            return observation[key]
    episode = observation.get("traveluav_episode")
    if isinstance(episode, Mapping):
        sensors = episode.get("sensors")
        state = sensors.get("state") if isinstance(sensors, Mapping) else None
        if isinstance(state, Mapping):
            timestamp = state.get("timestamp", state.get("time_stamp"))
            if timestamp is not None:
                return float(timestamp) / 1_000_000_000.0
    return fallback


class EpisodeArtifactWriter:
    def __init__(self, store: ArtifactStore, episode: EvalEpisode):
        self.store = store
        self.episode = episode
        self.episode_dir = store.episode_dir(episode)
        self.diagnostics: dict[str, Any] = {"warnings": []}
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self._archive_previous_attempt()
        (self.episode_dir / "data").mkdir(exist_ok=True)

    def add_warning(self, message: str) -> None:
        self.diagnostics.setdefault("warnings", []).append(str(message))

    def write_eval_info(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["paths"] = self.artifact_paths()
        warnings = self.diagnostics.get("warnings")
        if warnings:
            diagnostics = dict(payload.get("diagnostics") or {})
            diagnostics["warnings"] = list(warnings)
            payload["diagnostics"] = diagnostics
        write_json_atomic(
            self.store.episode_eval_info_path(self.episode),
            json_safe_value(payload),
            sort_keys=False,
        )

    def write_step_json(self, relative_path: str | Path, payload: Mapping[str, Any]) -> None:
        write_json_atomic(self._resolve_episode_path(relative_path), json_safe_value(payload))

    def write_common_step_artifacts(
        self,
        *,
        state: StepState,
        benchmark_specific: Mapping[str, Any] | None = None,
        save_images: bool = True,
        image_cameras: tuple[str, ...] | None = None,
        action_observation_image_policy: str = "step",
    ) -> None:
        step_name = f"{int(state.artifact_step_index):06d}"
        payload = common_step_payload(state, benchmark_specific=benchmark_specific)
        self.write_step_json(Path("data") / f"{step_name}.json", payload)
        if not save_images:
            return
        policy = str(action_observation_image_policy)
        if policy in {"step", "both"}:
            for camera_name, image in extract_observation_images(state.post_observation, image_cameras=image_cameras).items():
                self.write_image(Path(camera_name) / f"{step_name}.png", image)
        if policy in {"action", "both"}:
            for action_offset, observation in enumerate(state.action_observations):
                action_name = f"{int(state.artifact_step_index) + int(action_offset) + 1:06d}"
                for camera_name, image in extract_observation_images(observation, image_cameras=image_cameras).items():
                    self.write_image(Path(camera_name) / "actions" / f"{action_name}.png", image)

    def write_image(self, relative_path: str | Path, image: Any) -> bool:
        path = self._resolve_episode_path(relative_path)
        result = _coerce_rgb_image_with_diagnostics(image)
        if result.image is None:
            self.add_warning(f"{result.reason} for {relative_path}: shape={result.shape}")
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.image).save(path)
        return True

    def _resolve_episode_path(self, relative_path: str | Path) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError(f"episode artifact path must be relative: {relative_path}")
        episode_root = self.episode_dir.resolve()
        resolved = (self.episode_dir / path).resolve()
        if resolved != episode_root and episode_root not in resolved.parents:
            raise ValueError(f"episode artifact path escapes episode_dir: {relative_path}")
        return resolved

    def artifact_paths(self) -> dict[str, Any]:
        image_dirs = []
        for path in sorted(self.episode_dir.iterdir()):
            if not path.is_dir() or path.name in {"attempts", "data"}:
                continue
            if any(child.is_file() for child in path.iterdir()):
                image_dirs.append(path.name)
        return {
            "data": "data",
            "image_dirs": image_dirs,
        }

    def _archive_previous_attempt(self) -> None:
        eval_info_path = self.store.episode_eval_info_path(self.episode)
        if not eval_info_path.exists():
            return
        payload = json.loads(eval_info_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"existing eval_info must be an object before retry: {eval_info_path}")
        attempt_id = str(payload.get("attempt_id") or "").strip()
        if not attempt_id:
            raise ValueError(f"existing eval_info is missing attempt_id before retry: {eval_info_path}")
        attempt_dir = self.episode_dir / "attempts" / _safe_attempt_id(attempt_id)
        if attempt_dir.exists():
            raise FileExistsError(f"attempt archive already exists: {attempt_dir}")
        attempt_dir.mkdir(parents=True)
        for name in ("eval_info.json",):
            source = self.episode_dir / name
            if source.exists():
                shutil.move(str(source), str(attempt_dir / name))
        for source_dir in sorted(path for path in self.episode_dir.iterdir() if path.is_dir() and path.name != "attempts"):
            if source_dir.exists():
                shutil.move(str(source_dir), str(attempt_dir / source_dir.name))


def _safe_attempt_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    if not safe:
        raise ValueError(f"attempt_id sanitizes to empty: {value!r}")
    return safe


def scan_eval_infos(run_root: Path) -> list[EvalInfoRecord]:
    records = []
    for path in sorted((Path(run_root) / "logs").glob("**/eval_info.json")):
        if "attempts" in path.relative_to(Path(run_root) / "logs").parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            records.append(EvalInfoRecord(path=path, payload=None, valid=False, error=f"invalid JSON: {exc}"))
            continue
        if not isinstance(payload, dict):
            records.append(EvalInfoRecord(path=path, payload=None, valid=False, error="eval_info must be an object"))
            continue
        missing = sorted(field for field in _REQUIRED_EVAL_INFO_FIELDS if field not in payload)
        if missing:
            records.append(
                EvalInfoRecord(
                    path=path,
                    payload=payload,
                    valid=False,
                    error=f"missing required fields: {missing}",
                )
            )
            continue
        records.append(EvalInfoRecord(path=path, payload=payload, valid=True))
    return records


def is_completed_skip_candidate(eval_info: Mapping[str, Any], episode: EvalEpisode, identity: RunIdentity) -> bool:
    expected = {
        "benchmark": identity.benchmark,
        "run_name": identity.run_name,
        "config_sha256": identity.config_sha256,
        "input_fingerprint": identity.input_fingerprint,
        "episode_uid": episode.episode_uid,
        "source_episode_id": episode.source_episode_id,
        "input_namespace": episode.input_namespace,
        "input_root": episode.input_root,
        "scene_id": episode.scene_id,
        "status": "completed",
        "failure": None,
    }
    return all(eval_info.get(key) == value for key, value in expected.items())


def acquire_run_lock(run_root: Path) -> RunLock:
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / "run.lock"
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"run.lock already exists: {lock_path}") from exc
    os.write(fd, str(os.getpid()).encode("utf-8"))
    return RunLock(lock_path, fd)
