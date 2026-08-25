from __future__ import annotations

from time import sleep
from typing import Any

import numpy as np

from NavVLAeval.common.simulators.airsim.images import decode_depth_response, decode_scene_response


TRAVELUAV_CAMERA_NAMES = ["FrontCamera", "LeftCamera", "RightCamera", "RearCamera", "DownCamera"]
TRAVELUAV_RECORD_CAMERA_INDICES = (0, 4)


class AirSimObservationBuilder:
    def __init__(self, *, profile: str, camera_name: str) -> None:
        self.profile = str(profile or "openfly")
        self.camera_name = str(camera_name or "front")
        self.last_movement_result: dict[str, Any] = {"collision": False, "collision_reason": None}
        self.target_position: np.ndarray | None = None

    def build(self, *, client: Any, airsim: Any) -> dict[str, Any]:
        if self.profile == "traveluav":
            return self._build_traveluav_observation(client=client, airsim=airsim)
        if self.profile == "aerialvln":
            return self._build_aerialvln_observation(client=client, airsim=airsim)
        return self._build_openfly_observation(client=client, airsim=airsim)

    def _build_openfly_observation(self, *, client: Any, airsim: Any) -> dict[str, Any]:
        requests = [airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)]
        responses = _sim_get_images_with_valid_scenes(
            client,
            requests=requests,
            scene_response_indices=[0],
            label="openfly",
        )
        return {"image": decode_scene_response(responses[0])}

    def _build_aerialvln_observation(self, *, client: Any, airsim: Any) -> dict[str, Any]:
        camera_name = self.camera_name or "front_0"
        requests = [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)]
        responses = _sim_get_images_with_valid_scenes(
            client,
            requests=requests,
            scene_response_indices=[0],
            label="aerialvln",
            vehicle_name="Drone_1",
        )
        return {"image": decode_scene_response(responses[0]), "navvla_eval": {}}

    def _build_traveluav_observation(self, *, client: Any, airsim: Any) -> dict[str, Any]:
        requests = []
        for camera_name in TRAVELUAV_CAMERA_NAMES:
            requests.append(airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False))
            requests.append(airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True, False))
        responses = _sim_get_images_with_valid_scenes(
            client,
            requests=requests,
            scene_response_indices=[2 * index for index in range(len(TRAVELUAV_CAMERA_NAMES))],
            label="traveluav",
        )
        rgbs = [decode_scene_response(responses[2 * index]) for index in range(len(TRAVELUAV_CAMERA_NAMES))]
        depths = [decode_depth_response(airsim, responses[2 * index + 1]) for index in range(len(TRAVELUAV_CAMERA_NAMES))]
        rgb_record = [rgbs[index] for index in TRAVELUAV_RECORD_CAMERA_INDICES]
        depth_record = [depths[index] for index in TRAVELUAV_RECORD_CAMERA_INDICES]
        state_info = state_info_from_multirotor(client)
        imu_info = imu_info_from_client(client)
        traveluav_episode = {
            "instruction": "",
            "rgb": rgbs,
            "depth": depths,
            "rgb_record": rgb_record,
            "depth_record": depth_record,
            "sensors": {"state": state_info, "imu": imu_info},
        }
        traveluav_episode["sensors"]["state"]["movement"] = dict(self.last_movement_result)
        return {
            "image": rgbs[0],
            "state": np.asarray([0.0] * 16, dtype=np.float32),
            "traveluav_episode": traveluav_episode,
            "target_position": self.target_position if self.target_position is not None else np.zeros(3, dtype=np.float32),
        }

def _sim_get_images_with_valid_scenes(
    client: Any,
    *,
    requests: list[Any],
    scene_response_indices: list[int],
    label: str,
    vehicle_name: str | None = None,
    max_attempts: int = 10,
    retry_delay_sec: float = 0.1,
) -> list[Any]:
    responses = []
    for attempt in range(1, max_attempts + 1):
        if vehicle_name:
            responses = client.simGetImages(requests=requests, vehicle_name=vehicle_name)
        else:
            responses = client.simGetImages(requests=requests)
        invalid = [
            index
            for index in scene_response_indices
            if index >= len(responses) or not _valid_scene_response(responses[index])
        ]
        if not invalid:
            return responses
        if attempt < max_attempts:
            sleep(float(retry_delay_sec))
    raise RuntimeError(f"AirSim {label} simGetImages returned empty scene responses at indices {invalid}")


def _valid_scene_response(response: Any) -> bool:
    width = int(getattr(response, "width", 0) or 0)
    height = int(getattr(response, "height", 0) or 0)
    if width <= 0 or height <= 0:
        return False
    data = getattr(response, "image_data_uint8", b"")
    return len(data) >= width * height * 3


def state_info_from_multirotor(client: Any) -> dict[str, Any]:
    data = client.getMultirotorState(vehicle_name="")
    collision_info = client.simGetCollisionInfo(vehicle_name="")
    return {
        "collision": {
            "has_collided": bool(collision_info.has_collided),
            "object_name": getattr(collision_info, "object_name", getattr(data.collision, "object_name", "")),
        },
        "gps_location": [
            data.gps_location.latitude,
            data.gps_location.longitude,
            data.gps_location.altitude,
        ],
        "timestamp": data.timestamp,
        "position": list(data.kinematics_estimated.position),
        "linear_velocity": list(data.kinematics_estimated.linear_velocity),
        "linear_acceleration": list(data.kinematics_estimated.linear_acceleration),
        "orientation": list(data.kinematics_estimated.orientation),
        "angular_velocity": list(data.kinematics_estimated.angular_velocity),
        "angular_acceleration": list(data.kinematics_estimated.angular_acceleration),
    }


def imu_info_from_client(client: Any) -> dict[str, Any]:
    data = client.getImuData(imu_name="Imu", vehicle_name="")
    orientation = data.orientation
    q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
    rotation_matrix = np.array(
        [
            [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q3 * q0), 2 * (q1 * q3 + q2 * q0)],
            [2 * (q1 * q2 + q3 * q0), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q1 * q0)],
            [2 * (q1 * q3 - q2 * q0), 2 * (q2 * q3 + q1 * q0), 1 - 2 * (q1 * q1 + q2 * q2)],
        ],
        dtype=np.float32,
    ).tolist()
    return {
        "time_stamp": data.time_stamp,
        "rotation": rotation_matrix,
        "orientation": list(data.orientation),
        "linear_acceleration": list(data.linear_acceleration),
        "angular_velocity": list(data.angular_velocity),
    }
