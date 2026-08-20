import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from msgpackrpc.error import TimeoutError as RpcTimeoutError, TransportError

from airsim_plugin.camera_views import camera_specs


class AirSimSessionUnavailableError(RuntimeError):
    pass


def _call_airsim(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except (RpcTimeoutError, TransportError, ConnectionError) as error:
        raise AirSimSessionUnavailableError(str(error)) from error


def quaternion_angular_error_degrees(first_xyzw, second_xyzw):
    first = np.asarray(first_xyzw, dtype=np.float64)
    second = np.asarray(second_xyzw, dtype=np.float64)
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return float("inf")
    dot = abs(float(np.dot(first / first_norm, second / second_norm)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _xyz(vector):
    return np.asarray([
        getattr(vector, "x_val", getattr(vector, "x", 0.0)),
        getattr(vector, "y_val", getattr(vector, "y", 0.0)),
        getattr(vector, "z_val", getattr(vector, "z", 0.0)),
    ], dtype=np.float64)


def _xyzw(quaternion):
    return np.asarray([
        getattr(quaternion, "x_val", getattr(quaternion, "x", 0.0)),
        getattr(quaternion, "y_val", getattr(quaternion, "y", 0.0)),
        getattr(quaternion, "z_val", getattr(quaternion, "z", 0.0)),
        getattr(quaternion, "w_val", getattr(quaternion, "w", 0.0)),
    ], dtype=np.float64)


class DirectAirSimSession:
    def __init__(self, client, airsim_module, channel_order="rgb",
                 image_width=224, image_height=224):
        if channel_order not in ("rgb", "bgr"):
            raise ValueError("channel_order must be rgb or bgr for rendering")
        self.client = client
        self.airsim = airsim_module
        self.channel_order = channel_order
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")

    def apply_camera_records(self, records):
        for record in records:
            final = record.final
            pose = self.airsim.Pose(
                self.airsim.Vector3r(final.x, final.y, final.z),
                self.airsim.to_quaternion(
                    math.radians(final.pitch),
                    math.radians(final.roll),
                    math.radians(final.yaw),
                ),
            )
            _call_airsim(
                self.client.simSetCameraPose,
                record.name, pose, vehicle_name="Drone_1",
            )
            _call_airsim(
                self.client.simSetCameraFov,
                record.name, float(record.fov_degrees), vehicle_name="Drone_1"
            )

    def set_vehicle_pose(self, request):
        x, y, z = request.position_xyz
        qx, qy, qz, qw = request.orientation_quaternion_xyzw
        pose = self.airsim.Pose(
            self.airsim.Vector3r(x, y, z),
            self.airsim.Quaternionr(qx, qy, qz, qw),
        )
        _call_airsim(
            self.client.simSetVehiclePose,
            pose=pose, ignore_collision=True, vehicle_name="Drone_1"
        )

    def verify_vehicle_pose(self, request):
        pose = _call_airsim(
            self.client.simGetVehiclePose, vehicle_name="Drone_1"
        )
        position_error = float(np.linalg.norm(
            _xyz(pose.position) - np.asarray(request.position_xyz, dtype=np.float64)
        ))
        rotation_error = quaternion_angular_error_degrees(
            _xyzw(pose.orientation), request.orientation_quaternion_xyzw
        )
        return position_error, rotation_error

    def capture_rgb(self, views):
        specs = camera_specs(tuple(views))
        requests = [
            self.airsim.ImageRequest(
                spec.name, self.airsim.ImageType.Scene,
                pixels_as_float=False, compress=False,
            )
            for spec in specs
        ]
        responses = _call_airsim(
            self.client.simGetImages, requests, vehicle_name="Drone_1"
        )
        if len(responses) != len(specs):
            raise RuntimeError(
                "AirSim returned {} images for {} views".format(
                    len(responses), len(specs)
                )
            )
        frames = {}
        for spec, response in zip(specs, responses):
            if (
                response.width != self.image_width
                or response.height != self.image_height
            ):
                raise RuntimeError(
                    "{} returned {}x{} instead of {}x{}".format(
                        spec.view, response.width, response.height,
                        self.image_width, self.image_height,
                    )
                )
            flat = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            expected_bytes = self.image_width * self.image_height * 3
            if flat.size != expected_bytes:
                raise RuntimeError(
                    "{} returned {} RGB bytes".format(spec.view, flat.size)
                )
            frame = flat.reshape(self.image_height, self.image_width, 3).copy()
            if self.channel_order == "bgr":
                frame = frame[:, :, ::-1].copy()
            frames[spec.view] = frame
        return frames

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


class AirSimServerRuntime:
    """Own one scene server process and reopen a single scene on demand."""

    def __init__(self, repository_root, env_root, gpu, control_port, log_path,
                 startup_timeout=180, image_width=224, image_height=224):
        self.repository_root = Path(repository_root).resolve()
        self.env_root = Path(env_root).resolve()
        self.gpu = int(gpu)
        self.control_port = int(control_port)
        self.log_path = Path(log_path)
        self.startup_timeout = float(startup_timeout)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.process = None
        self.log_handle = None
        self.rpc_client = None

    def start(self):
        import msgpackrpc

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("ab", buffering=0)
        server_script = self.repository_root / "airsim_plugin" / "AirVLNSimulatorServerTool.py"
        self.process = subprocess.Popen(
            [
                sys.executable, str(server_script),
                "--gpus", str(self.gpu),
                "--port", str(self.control_port),
                "--env-root", str(self.env_root),
                "--image-width", str(self.image_width),
                "--image-height", str(self.image_height),
            ],
            cwd=str(self.repository_root),
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.startup_timeout
        last_error = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "AirVLN scene server exited with code {}".format(
                        self.process.returncode
                    )
                )
            try:
                client = msgpackrpc.Client(
                    msgpackrpc.Address("127.0.0.1", self.control_port), timeout=5
                )
                if client.call("ping"):
                    client.close()
                    self.rpc_client = msgpackrpc.Client(
                        msgpackrpc.Address("127.0.0.1", self.control_port),
                        timeout=max(300, int(self.startup_timeout)),
                    )
                    return self
            except Exception as error:
                last_error = error
            time.sleep(1)
        raise RuntimeError("AirVLN scene server did not become ready: {}".format(last_error))

    def open_scene(self, scene_id, channel_order="rgb"):
        import airsim

        if self.rpc_client is None:
            self.start()
        result = self.rpc_client.call(
            "reopen_scenes", "127.0.0.1", [str(scene_id)]
        )
        if not result or not result[0]:
            raise RuntimeError("failed to open scene {}".format(scene_id))
        ip, ports = result[1]
        if isinstance(ip, bytes):
            ip = ip.decode("utf-8")
        if len(ports) != 1:
            raise RuntimeError("scene server returned invalid AirSim ports")
        client = airsim.VehicleClient(
            ip=str(ip), port=int(ports[0]), timeout_value=60
        )
        client.confirmConnection()
        return DirectAirSimSession(
            client, airsim, channel_order=channel_order,
            image_width=self.image_width, image_height=self.image_height,
        )

    def close(self, suppress_unavailable_scene_rpc=False):
        first_error = None

        def record_cleanup_error(error):
            nonlocal first_error
            if first_error is None:
                first_error = error

        rpc_client = self.rpc_client
        self.rpc_client = None
        if rpc_client is not None:
            try:
                result = rpc_client.call("close_scenes", "127.0.0.1")
                if not result:
                    raise RuntimeError("close_scenes returned false")
            except Exception as error:
                if not (
                    suppress_unavailable_scene_rpc
                    and isinstance(error, (RpcTimeoutError, TransportError, ConnectionError))
                ):
                    record_cleanup_error(error)
            try:
                rpc_client.close()
            except Exception as error:
                record_cleanup_error(error)

        process = self.process
        self.process = None
        if process is not None:
            try:
                running = process.poll() is None
            except Exception as error:
                record_cleanup_error(error)
                running = True
            if running:
                try:
                    process.terminate()
                except Exception as error:
                    record_cleanup_error(error)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except Exception as error:
                        record_cleanup_error(error)
                    try:
                        process.wait(timeout=10)
                    except Exception as error:
                        record_cleanup_error(error)
                except Exception as error:
                    record_cleanup_error(error)

        log_handle = self.log_handle
        self.log_handle = None
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception as error:
                record_cleanup_error(error)

        if first_error is not None:
            raise first_error
