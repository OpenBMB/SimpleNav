from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union


COLLECT_RGB_CAMERA_VIEWS = ("front", "back", "left", "right")
_VIEW_TO_CAMERA = {
    "front": ("front_0", 0.25, 0.0, 0.0, 0, 0.0, 0.0),
    "back": ("back_0", -0.25, 0.0, 0.0, 180, 0.0, 0.0),
    "left": ("left_0", 0.0, -0.25, 0.0, -90, 0.0, 0.0),
    "right": ("right_0", 0.0, 0.25, 0.0, 90, 0.0, 0.0),
}


@dataclass(frozen=True)
class CameraSpec:
    view: str
    name: str
    yaw: int
    x: float = 0.5
    y: float = 0.0
    z: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    fov_degrees: float = 90.0


def parse_camera_views(value: Optional[str]) -> Tuple[str, ...]:
    if value is None or str(value).strip() == "":
        return ("front",)

    raw_value = str(value).strip().lower()
    if raw_value == "all":
        return COLLECT_RGB_CAMERA_VIEWS

    views = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    if not views:
        raise ValueError("camera view selection is empty")
    duplicates = sorted({view for view in views if views.count(view) > 1})
    if duplicates:
        raise ValueError("duplicate camera view(s): {}".format(",".join(duplicates)))
    unknown = [view for view in views if view not in _VIEW_TO_CAMERA]
    if unknown:
        raise ValueError("unknown camera view(s): {}".format(",".join(unknown)))
    return views


def camera_specs(
    views: Union[str, Sequence[str]] = COLLECT_RGB_CAMERA_VIEWS,
) -> Tuple[CameraSpec, ...]:
    if isinstance(views, str):
        views = parse_camera_views(views)
    specs = []
    for view in views:
        name, x, y, z, yaw, pitch, roll = _VIEW_TO_CAMERA[view]
        specs.append(
            CameraSpec(
                view=view,
                name=name,
                yaw=yaw,
                x=x,
                y=y,
                z=z,
                pitch=pitch,
                roll=roll,
            )
        )
    return tuple(specs)


def collect_rgb_key(trajectory_id: str, step: int, view: str) -> str:
    if view == "front":
        return "{}_{}_rgb".format(trajectory_id, step)
    return "{}_{}_{}_rgb".format(trajectory_id, step, view)
