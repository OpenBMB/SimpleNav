from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from airsim_plugin.camera_views import CameraSpec


@dataclass(frozen=True)
class CameraPose6D:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float


@dataclass(frozen=True)
class CameraPoseRecord:
    view: str
    name: str
    base: CameraPose6D
    delta: CameraPose6D
    final: CameraPose6D
    fov_degrees: float = 90.0


CAMERA_POSE_NOISE_MANIFEST_NAME = "camera_pose_noise_manifest.json"


def build_camera_pose_noise_manifest(
    split: str,
    selected_camera_views: Sequence[str],
    mode: str,
    seed: int,
    xyz_max: float,
    yaw_pitch_max: float,
    roll_max: float,
    rgb_width: int,
    rgb_height: int,
    camera_specs: Sequence[CameraSpec],
) -> Dict[str, object]:
    if mode not in ("none", "episode"):
        raise ValueError("unsupported camera pose noise mode: {}".format(mode))
    xyz_max, yaw_pitch_max, roll_max = validate_camera_pose_noise_limits(
        xyz_max=xyz_max,
        yaw_pitch_max=yaw_pitch_max,
        roll_max=roll_max,
    )
    return {
        "schema_version": 1,
        "split": str(split),
        "selected_camera_views": list(selected_camera_views),
        "mode": str(mode),
        "seed": int(seed),
        "noise_limits": {
            "xyz_m": xyz_max,
            "yaw_pitch_degrees": yaw_pitch_max,
            "roll_degrees": roll_max,
        },
        "rgb": {"width": int(rgb_width), "height": int(rgb_height)},
        "base_cameras": [
            {
                "view": spec.view,
                "name": spec.name,
                "x": spec.x,
                "y": spec.y,
                "z": spec.z,
                "yaw": spec.yaw,
                "pitch": spec.pitch,
                "roll": spec.roll,
            }
            for spec in camera_specs
        ],
    }


def _manifest_mismatches(
    existing: Mapping[str, object], requested: Mapping[str, object]
) -> Tuple[str, ...]:
    keys = sorted(set(existing) | set(requested))
    return tuple(key for key in keys if existing.get(key) != requested.get(key))


def ensure_camera_pose_noise_manifest(
    output_root: Union[str, Path], manifest: Mapping[str, object]
) -> Optional[Path]:
    if manifest.get("mode") == "none":
        return None

    output_root = Path(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.parent / (
        ".{}.camera_pose_noise_manifest.lock".format(output_root.name)
    )
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / CAMERA_POSE_NOISE_MANIFEST_NAME
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            requested = dict(manifest)
            mismatches = _manifest_mismatches(existing, requested)
            if mismatches:
                raise ValueError(
                    "camera pose noise manifest mismatch: {}".format(
                        "; ".join(
                            "{} existing={!r} requested={!r}".format(
                                key, existing.get(key), requested.get(key)
                            )
                            for key in mismatches
                        )
                    )
                )
            return manifest_path

        if any(output_root.iterdir()):
            raise ValueError(
                "non-empty collection root has no camera pose noise manifest: {}".format(
                    output_root
                )
            )

        payload = json.dumps(
            dict(manifest), indent=2, sort_keys=True
        ).encode("utf-8") + b"\n"
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".camera-pose-noise-manifest-",
                dir=str(output_root),
                delete=False,
            ) as temporary_handle:
                temporary_path = Path(temporary_handle.name)
                temporary_handle.write(payload)
                temporary_handle.flush()
                os.fsync(temporary_handle.fileno())
            os.replace(str(temporary_path), str(manifest_path))
            directory_fd = os.open(str(output_root), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return manifest_path


def _derived_seed(
    seed: int,
    episode_id: Union[str, int],
    camera_name: str,
) -> int:
    seed_parts = [str(seed), "episode", str(episode_id), camera_name]
    digest = hashlib.sha256("\x1f".join(seed_parts).encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big")


def _add_pose(base: CameraPose6D, delta: CameraPose6D) -> CameraPose6D:
    return CameraPose6D(
        x=base.x + delta.x,
        y=base.y + delta.y,
        z=base.z + delta.z,
        yaw=base.yaw + delta.yaw,
        pitch=base.pitch + delta.pitch,
        roll=base.roll + delta.roll,
    )


def sample_camera_pose_records(
    camera_specs: Sequence[CameraSpec],
    mode: str,
    seed: int,
    episode_id: Union[str, int],
    xyz_max: float = 0.1,
    yaw_pitch_max: float = 10.0,
    roll_max: float = 3.0,
    fov_min_degrees: float = 90.0,
    fov_max_degrees: float = 90.0,
) -> Tuple[CameraPoseRecord, ...]:
    if mode != "episode":
        raise ValueError("unsupported camera pose noise mode: {}".format(mode))
    xyz_max, yaw_pitch_max, roll_max = validate_camera_pose_noise_limits(
        xyz_max=xyz_max,
        yaw_pitch_max=yaw_pitch_max,
        roll_max=roll_max,
    )
    fov_min_degrees, fov_max_degrees = validate_camera_fov_limits(
        fov_min_degrees, fov_max_degrees
    )

    records = []
    for spec in camera_specs:
        rng = random.Random(
            _derived_seed(seed, episode_id, spec.name)
        )
        base = CameraPose6D(
            x=spec.x,
            y=spec.y,
            z=spec.z,
            yaw=spec.yaw,
            pitch=spec.pitch,
            roll=spec.roll,
        )
        delta = CameraPose6D(
            x=rng.uniform(-xyz_max, xyz_max),
            y=rng.uniform(-xyz_max, xyz_max),
            z=rng.uniform(-xyz_max, xyz_max),
            yaw=rng.uniform(-yaw_pitch_max, yaw_pitch_max),
            pitch=rng.uniform(-yaw_pitch_max, yaw_pitch_max),
            roll=rng.uniform(-roll_max, roll_max),
        )
        fov_degrees = rng.uniform(fov_min_degrees, fov_max_degrees)
        records.append(
            CameraPoseRecord(
                view=spec.view,
                name=spec.name,
                base=base,
                delta=delta,
                final=_add_pose(base, delta),
                fov_degrees=fov_degrees,
            )
        )
    return tuple(records)


def camera_pose_metadata_key(
    episode_id: Union[str, int], frame_index: int, view: str
) -> str:
    return "{}_{}_{}_camera_pose".format(episode_id, frame_index, view)


def validate_camera_pose_noise_limits(
    xyz_max: float = 0.1,
    yaw_pitch_max: float = 10.0,
    roll_max: float = 3.0,
) -> Tuple[float, float, float]:
    limits = (float(xyz_max), float(yaw_pitch_max), float(roll_max))
    if any(limit < 0.0 for limit in limits):
        raise ValueError("camera pose noise limits must be non-negative")
    return limits


def validate_camera_fov_limits(
    fov_min_degrees: float,
    fov_max_degrees: float,
) -> Tuple[float, float]:
    limits = (float(fov_min_degrees), float(fov_max_degrees))
    if (
        not all(math.isfinite(value) for value in limits)
        or limits[0] <= 0.0
        or limits[0] > limits[1]
    ):
        raise ValueError("camera FOV limits must be positive and ordered")
    return limits


def _pose_payload(pose: CameraPose6D) -> Dict[str, float]:
    return {
        "x": pose.x,
        "y": pose.y,
        "z": pose.z,
        "yaw": pose.yaw,
        "pitch": pose.pitch,
        "roll": pose.roll,
    }


def camera_pose_metadata_payload(
    episode_id: Union[str, int],
    trajectory_id: Union[str, int],
    frame_index: int,
    mode: str,
    seed: int,
    record: CameraPoseRecord,
) -> Dict[str, object]:
    if mode != "episode":
        raise ValueError("unsupported camera pose noise mode: {}".format(mode))
    return {
        "episode_id": episode_id,
        "trajectory_id": trajectory_id,
        "frame": frame_index,
        "view": record.view,
        "mode": mode,
        "seed": seed,
        "delta": _pose_payload(record.delta),
        "final": _pose_payload(record.final),
    }


def rendered_observation_identity(
    mode: str,
    episode_id: Union[str, int],
    trajectory_id: Union[str, int],
) -> Union[str, int]:
    if mode not in ("none", "episode"):
        raise ValueError("unsupported camera pose noise mode: {}".format(mode))
    return trajectory_id if mode == "none" else episode_id


def collect_rgb_identity(
    mode: str,
    episode_id: Union[str, int],
    trajectory_id: Union[str, int],
) -> Union[str, int]:
    return rendered_observation_identity(mode, episode_id, trajectory_id)
