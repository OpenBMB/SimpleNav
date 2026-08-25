from dataclasses import dataclass

import numpy as np

from vln_aug.actions import wrap_angle


@dataclass
class CameraFrame:
    rgb: np.ndarray
    receipt: str


@dataclass(frozen=True)
class RenderProvenance:
    backend_type: str
    backend_name: str
    scene_id: str
    render_call_id: str


@dataclass
class RenderBatch:
    requested_pose: np.ndarray
    returned_pose: np.ndarray
    frames: dict[str, CameraFrame]
    provenance: RenderProvenance


def validate_publishable_render_batch(
    batch: RenderBatch,
    expected_cameras: dict[str, tuple[int, int]],
    expected_scene_id: str,
    translation_tolerance_m: float = 1e-3,
    yaw_tolerance_rad: float = 1e-3,
) -> None:
    if batch.provenance.backend_type != "REAL":
        raise ValueError("only a real renderer backend may publish training data")
    if not batch.provenance.backend_name or not batch.provenance.render_call_id:
        raise ValueError("renderer provenance is incomplete")
    if batch.provenance.scene_id != expected_scene_id:
        raise ValueError("renderer scene does not match episode scene")

    requested = np.asarray(batch.requested_pose, dtype=float)
    returned = np.asarray(batch.returned_pose, dtype=float)
    if requested.shape != (4,) or returned.shape != (4,):
        raise ValueError("renderer poses must be [x, y, z, yaw]")
    if np.linalg.norm(returned[:3] - requested[:3]) > translation_tolerance_m:
        raise ValueError("renderer returned a different translation")
    if abs(float(wrap_angle(returned[3] - requested[3]))) > yaw_tolerance_rad:
        raise ValueError("renderer returned a different yaw")

    if set(batch.frames) != set(expected_cameras):
        raise ValueError("renderer camera set is incomplete or unexpected")
    receipts = set()
    for name, (height, width) in expected_cameras.items():
        frame = batch.frames[name]
        image = np.asarray(frame.rgb)
        if image.shape != (height, width, 3) or image.dtype != np.uint8:
            raise ValueError(f"camera {name} has wrong image shape or dtype")
        if not frame.receipt or frame.receipt in receipts:
            raise ValueError("camera frames require unique fresh receipts")
        receipts.add(frame.receipt)
