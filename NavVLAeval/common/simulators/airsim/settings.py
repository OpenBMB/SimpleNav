from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any


def _capture_settings(width: int, height: int) -> list[dict[str, Any]]:
    return [
        {
            "ImageType": 0,
            "Width": int(width),
            "Height": int(height),
            "FOV_Degrees": 90,
            "AutoExposureMaxBrightness": 1,
            "AutoExposureMinBrightness": 0.03,
        },
        {
            "ImageType": 2,
            "Width": int(width),
            "Height": int(height),
            "FOV_Degrees": 90,
            "AutoExposureMaxBrightness": 1,
            "AutoExposureMinBrightness": 0.03,
        },
    ]


def _camera(x: float, y: float, z: float, pitch: float, roll: float, yaw: float, *, width: int, height: int) -> dict[str, Any]:
    return {
        "X": x,
        "Y": y,
        "Z": z,
        "Pitch": pitch,
        "Roll": roll,
        "Yaw": yaw,
        "CaptureSettings": _capture_settings(width, height),
    }


def _aerialvln_front_camera() -> dict[str, Any]:
    return {
        "X": 0.5,
        "Y": 0,
        "Z": 0,
        "Pitch": 0,
        "Roll": 0,
        "Yaw": 0,
        "CaptureSettings": [
            {
                "ImageType": 0,
                "Width": 448,
                "Height": 448,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03,
            },
            {
                "ImageType": 2,
                "Width": 448,
                "Height": 448,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03,
            },
            {
                "ImageType": 3,
                "Width": 448,
                "Height": 448,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03,
            },
        ],
    }


OPENFLY_CAMERAS = {
    "front_custom": _camera(1, 0, 0, 0, 0, 0, width=448, height=448),
    "0": _camera(1, 0, 0, 0, 0, 0, width=448, height=448),
}

TRAVELUAV_CAMERAS = {
    "FrontCamera": _camera(1, 0, 0, 0, 0, 0, width=384, height=256),
    "RearCamera": _camera(-1, 0, 0, 0, 0, 180, width=384, height=256),
    "LeftCamera": _camera(0, -1, 0, 0, 0, -90, width=384, height=256),
    "RightCamera": _camera(0, 1, 0, 0, 0, 90, width=384, height=256),
    "DownCamera": _camera(0, 0, 0, -90, 0, 0, width=384, height=256),
    "FrontCameraRecord": _camera(1, 0, 0, 0, 0, 0, width=1024, height=1024),
    "DownCameraRecord": _camera(0, 0, 0, -90, 0, 0, width=1024, height=1024),
}

TRAVELUAV_EXTERNAL_CAMERAS = {
    "SmoothRecordCamera": _camera(0, 0, 0, 0, 0, 0, width=1024, height=1024),
}

AERIALVLN_CAMERAS = {
    "front_0": _aerialvln_front_camera(),
}


def _recording_settings(
    *,
    folder: str | Path | None,
    camera_name: str | None,
    interval: float | None,
) -> dict[str, Any]:
    cameras = []
    if camera_name:
        cameras.append(
            {
                "CameraName": str(camera_name),
                "ImageType": 0,
                "PixelsAsFloat": False,
                "Compress": True,
            }
        )
    recording = {
        "RecordInterval": float(interval if interval is not None else 1.0),
        "RecordOnMove": False,
        "Enabled": False,
        "Cameras": cameras,
    }
    if folder is not None:
        recording["Folder"] = str(folder)
    return recording


def build_airsim_settings(
    *,
    api_server_port: int,
    profile: str = "openfly",
    recording_folder: str | Path | None = None,
    recording_camera_name: str | None = None,
    recording_interval: float | None = None,
    camera_resolution_overrides: dict[str, tuple[int, int]] | None = None,
    external_camera_resolution_overrides: dict[str, tuple[int, int]] | None = None,
    clock_speed: int | float | None = None,
    view_mode: str | None = None,
) -> dict[str, Any]:
    external_cameras: dict[str, Any] = {}
    if profile == "traveluav":
        cameras = copy.deepcopy(TRAVELUAV_CAMERAS)
        _apply_camera_resolution_overrides(cameras, camera_resolution_overrides)
        if external_camera_resolution_overrides or recording_camera_name in TRAVELUAV_EXTERNAL_CAMERAS:
            external_cameras = copy.deepcopy(TRAVELUAV_EXTERNAL_CAMERAS)
            _apply_camera_resolution_overrides(external_cameras, external_camera_resolution_overrides)
        resolved_clock_speed = 10 if clock_speed is None else clock_speed
        extra = {
            "PhysiceEngineName": "ExternalPhysicsEngine",
            "Recording": _recording_settings(
                folder=recording_folder,
                camera_name=recording_camera_name,
                interval=recording_interval,
            ),
        }
        sensors = {
            "Imu": {
                "SensorType": 2,
                "Enabled": True,
                "AngularRandomWalk": 0.3,
                "GyroBiasStabilityTau": 500,
                "GyroBiasStability": 4.6,
                "VelocityRandomWalk": 0.24,
                "AccelBiasStabilityTau": 800,
                "AccelBiasStability": 36,
            }
        }
        sim_mode = "Multirotor"
        vehicle_type = "SimpleFlight"
    elif profile == "openfly":
        cameras = copy.deepcopy(OPENFLY_CAMERAS)
        _apply_camera_resolution_overrides(cameras, camera_resolution_overrides)
        resolved_clock_speed = 1 if clock_speed is None else clock_speed
        extra = {}
        sensors = {}
        sim_mode = "Multirotor"
        vehicle_type = "SimpleFlight"
    elif profile == "aerialvln":
        cameras = copy.deepcopy(AERIALVLN_CAMERAS)
        _apply_camera_resolution_overrides(cameras, camera_resolution_overrides)
        resolved_clock_speed = 1 if clock_speed is None else clock_speed
        extra = {
            "Recording": _recording_settings(
                folder=recording_folder,
                camera_name=recording_camera_name,
                interval=recording_interval if recording_interval is not None else 0.001,
            ),
            "SubWindows": [],
        }
        sensors = {}
        sim_mode = "Multirotor"
        vehicle_type = "SimpleFlight"
    else:
        raise ValueError(f"unsupported AirSim settings profile: {profile}")

    settings = {
        "SeeDocsAt": "https://microsoft.github.io/AirSim/settings/",
        "SettingsVersion": 1.2,
        "SimMode": sim_mode,
        "ClockSpeed": resolved_clock_speed,
        "ViewMode": str(view_mode or "NoDisplay"),
        "ApiServerPort": int(api_server_port),
        "Vehicles": {
            "Drone_1": {
                "VehicleType": vehicle_type,
                "UseSerial": False,
                "LockStep": True,
                "AutoCreate": True,
                "X": 0,
                "Y": 0,
                "Z": 0,
                "Roll": 0,
                "Pitch": 0,
                "Yaw": 0,
                "Cameras": cameras,
            }
        },
    }
    if sensors:
        settings["Vehicles"]["Drone_1"]["Sensors"] = sensors
    if external_cameras:
        settings["ExternalCameras"] = external_cameras
    settings.update(extra)
    return settings


def _apply_camera_resolution_overrides(
    cameras: dict[str, Any],
    overrides: dict[str, tuple[int, int]] | None,
) -> None:
    if not overrides:
        return
    for camera_name, dimensions in overrides.items():
        if camera_name not in cameras:
            raise KeyError(f"cannot override missing AirSim camera: {camera_name}")
        width, height = dimensions
        for capture_setting in cameras[camera_name].get("CaptureSettings", []):
            capture_setting["Width"] = int(width)
            capture_setting["Height"] = int(height)


def write_airsim_settings(
    settings_path: str | Path,
    *,
    api_server_port: int,
    profile: str = "openfly",
    recording_folder: str | Path | None = None,
    recording_camera_name: str | None = None,
    recording_interval: float | None = None,
    camera_resolution_overrides: dict[str, tuple[int, int]] | None = None,
    external_camera_resolution_overrides: dict[str, tuple[int, int]] | None = None,
    clock_speed: int | float | None = None,
    view_mode: str | None = None,
) -> None:
    settings_path = Path(settings_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings = build_airsim_settings(
        api_server_port=api_server_port,
        profile=profile,
        recording_folder=recording_folder,
        recording_camera_name=recording_camera_name,
        recording_interval=recording_interval,
        camera_resolution_overrides=camera_resolution_overrides,
        external_camera_resolution_overrides=external_camera_resolution_overrides,
        clock_speed=clock_speed,
        view_mode=view_mode,
    )
    tmp_path = settings_path.with_name(f"{settings_path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    tmp_path.replace(settings_path)
