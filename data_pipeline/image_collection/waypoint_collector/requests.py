from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple


def _number_tuple(value, length, field_name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError("{} must contain {} numbers".format(field_name, length))
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("{} must contain finite numbers".format(field_name))
    return result


@dataclass(frozen=True)
class RenderRequest:
    index: int
    request_id: str
    episode_id: str
    trajectory_id: str
    source_episode_index: int
    scene_id: str
    image_index: int
    waypoint_index: int
    timestamp: float
    position_xyz: Tuple[float, float, float]
    orientation_quaternion_wxyz: Tuple[float, float, float, float]
    expected_width: int
    expected_height: int
    expected_channels: int
    byte_start: Optional[int] = None
    byte_end: Optional[int] = None

    @property
    def orientation_quaternion_xyzw(self):
        w, x, y, z = self.orientation_quaternion_wxyz
        return x, y, z, w

    @classmethod
    def from_payload(cls, payload, index, byte_start=None, byte_end=None,
                     normalize_quaternion=False, expected_width=224,
                     expected_height=224, expected_channels=3):
        required = (
            "request_id", "episode_id", "trajectory_id", "source_episode_index",
            "scene_id", "image_index", "waypoint_index", "timestamp",
            "position_xyz", "orientation_quaternion_wxyz", "expected_height",
            "expected_width", "expected_channels",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError("render request missing fields: {}".format(", ".join(missing)))
        dimensions = (
            int(payload["expected_width"]), int(payload["expected_height"]),
            int(payload["expected_channels"]),
        )
        expected_dimensions = (
            int(expected_width), int(expected_height), int(expected_channels)
        )
        if dimensions != expected_dimensions:
            raise ValueError(
                "render request must describe {}x{}x{} RGB".format(
                    expected_dimensions[0], expected_dimensions[1],
                    expected_dimensions[2],
                )
            )
        position = _number_tuple(payload["position_xyz"], 3, "position_xyz")
        quaternion = _number_tuple(
            payload["orientation_quaternion_wxyz"], 4,
            "orientation_quaternion_wxyz",
        )
        norm = math.sqrt(sum(component * component for component in quaternion))
        if normalize_quaternion:
            if norm <= 1e-12:
                raise ValueError("orientation must be a unit quaternion")
            quaternion = tuple(component / norm for component in quaternion)
        elif abs(norm - 1.0) > 1e-4:
            raise ValueError("orientation must be a unit quaternion")
        return cls(
            index=int(index),
            request_id=str(payload["request_id"]),
            episode_id=str(payload["episode_id"]),
            trajectory_id=str(payload["trajectory_id"]),
            source_episode_index=int(payload["source_episode_index"]),
            scene_id=str(payload["scene_id"]),
            image_index=int(payload["image_index"]),
            waypoint_index=int(payload["waypoint_index"]),
            timestamp=float(payload["timestamp"]),
            position_xyz=position,
            orientation_quaternion_wxyz=quaternion,
            expected_width=dimensions[0],
            expected_height=dimensions[1],
            expected_channels=dimensions[2],
            byte_start=byte_start,
            byte_end=byte_end,
        )

    def metadata_payload(self, status):
        return {
            "index": self.index,
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "source_episode_index": self.source_episode_index,
            "scene_id": self.scene_id,
            "image_index": self.image_index,
            "waypoint_index": self.waypoint_index,
            "timestamp": self.timestamp,
            "position_xyz": list(self.position_xyz),
            "orientation_quaternion_wxyz": list(self.orientation_quaternion_wxyz),
            "expected_width": self.expected_width,
            "expected_height": self.expected_height,
            "expected_channels": self.expected_channels,
            "status": status,
        }


def iter_render_requests(path, byte_start=0, byte_end=None, start_index=0,
                         expected_width=224, expected_height=224,
                         expected_channels=3):
    path = Path(path)
    with path.open("rb") as handle:
        handle.seek(int(byte_start))
        index = int(start_index)
        while True:
            line_start = handle.tell()
            if byte_end is not None and line_start >= int(byte_end):
                break
            line = handle.readline()
            if not line:
                break
            line_end = handle.tell()
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "invalid JSON at byte {} in {}: {}".format(
                        line_start, path, error
                    )
                ) from error
            yield RenderRequest.from_payload(
                payload, index=index, byte_start=line_start, byte_end=line_end,
                expected_width=expected_width, expected_height=expected_height,
                expected_channels=expected_channels,
            )
            index += 1


def validate_episode_sequence(requests: Sequence[RenderRequest]):
    if not requests:
        raise ValueError("episode has no render requests")
    episode_id = requests[0].episode_id
    source_episode_index = requests[0].source_episode_index
    for expected_image_index, request in enumerate(requests):
        if request.episode_id != episode_id:
            raise ValueError("byte range contains multiple episodes")
        if request.source_episode_index != source_episode_index:
            raise ValueError("source episode index changed within episode")
        if request.image_index != expected_image_index:
            raise ValueError("image_index is not contiguous within episode")
        if expected_image_index and request.waypoint_index <= requests[expected_image_index - 1].waypoint_index:
            raise ValueError("waypoint_index is not strictly increasing")
    return True
