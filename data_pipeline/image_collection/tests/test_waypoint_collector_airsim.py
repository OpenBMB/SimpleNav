from pathlib import Path
import types
import tempfile
import unittest

from msgpackrpc import error as rpc_error
import numpy as np

from waypoint_collector import airsim_session
from waypoint_collector.airsim_session import AirSimServerRuntime, DirectAirSimSession
from airsim_plugin import AirVLNSimulatorServerTool as server_tool


class Vector3r:
    def __init__(self, x, y, z):
        self.x_val, self.y_val, self.z_val = x, y, z


class Quaternionr:
    def __init__(self, x, y, z, w):
        self.x_val, self.y_val, self.z_val, self.w_val = x, y, z, w


class Pose:
    def __init__(self, position_val, orientation_val):
        self.position = position_val
        self.orientation = orientation_val


class ImageRequest:
    def __init__(self, camera_name, image_type, pixels_as_float, compress):
        self.camera_name = camera_name
        self.image_type = image_type
        self.pixels_as_float = pixels_as_float
        self.compress = compress


class Response:
    def __init__(self, value, width=224, height=224):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = value
        frame[0, 0, 1] = value + 1
        self.width = width
        self.height = height
        self.image_data_uint8 = frame.tobytes()


class FakeClient:
    def __init__(self, width=224, height=224):
        self.pose = None
        self.width = width
        self.height = height
        self.camera_poses = []
        self.camera_fovs = []
        self.requests = []

    def simSetVehiclePose(self, pose, ignore_collision, vehicle_name):
        self.pose = pose

    def simGetVehiclePose(self, vehicle_name):
        return self.pose

    def simSetCameraPose(self, name, pose, vehicle_name):
        self.camera_poses.append((name, pose))

    def simSetCameraFov(self, name, fov, vehicle_name):
        self.camera_fovs.append((name, fov))

    def simGetImages(self, requests, vehicle_name):
        self.requests.append(requests)
        return [
            Response(index + 10, self.width, self.height)
            for index, _ in enumerate(requests)
        ]

    def close(self):
        pass


class FailingClient(FakeClient):
    def __init__(self, method_name, error):
        super().__init__()
        self.method_name = method_name
        self.error = error

    def _fail(self, method_name):
        if self.method_name == method_name:
            raise self.error

    def simSetVehiclePose(self, pose, ignore_collision, vehicle_name):
        self._fail("simSetVehiclePose")
        return super().simSetVehiclePose(pose, ignore_collision, vehicle_name)

    def simGetVehiclePose(self, vehicle_name):
        self._fail("simGetVehiclePose")
        return super().simGetVehiclePose(vehicle_name)

    def simSetCameraPose(self, name, pose, vehicle_name):
        self._fail("simSetCameraPose")
        return super().simSetCameraPose(name, pose, vehicle_name)

    def simSetCameraFov(self, name, fov, vehicle_name):
        self._fail("simSetCameraFov")
        return super().simSetCameraFov(name, fov, vehicle_name)

    def simGetImages(self, requests, vehicle_name):
        self._fail("simGetImages")
        return super().simGetImages(requests, vehicle_name)


class Record:
    def __init__(self, name, final, fov):
        self.name = name
        self.final = final
        self.fov_degrees = fov


class Request:
    position_xyz = (1.0, 2.0, -3.0)
    orientation_quaternion_xyzw = (0.1, 0.2, 0.3, 0.9)


class FakeRpcClient:
    def __init__(self, close_scenes_result=True, call_error=None):
        self.close_scenes_result = close_scenes_result
        self.call_error = call_error
        self.calls = []
        self.closed = False

    def call(self, method, *args):
        self.calls.append((method, args))
        if self.call_error is not None:
            raise self.call_error
        return self.close_scenes_result

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.wait_calls = []
        self.killed = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        self.wait_calls.append(timeout)

    def kill(self):
        self.killed = True


class FakeLogHandle:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class DirectAirSimSessionTests(unittest.TestCase):
    def setUp(self):
        self.airsim = types.SimpleNamespace(
            Pose=Pose,
            Vector3r=Vector3r,
            Quaternionr=Quaternionr,
            ImageRequest=ImageRequest,
            ImageType=types.SimpleNamespace(Scene=0),
            to_quaternion=lambda pitch, roll, yaw: Quaternionr(pitch, roll, yaw, 1.0),
        )
        self.client = FakeClient()
        self.session = DirectAirSimSession(self.client, self.airsim, channel_order="rgb")

    def _invoke(self, session, method_name):
        if method_name == "simSetCameraPose":
            final = types.SimpleNamespace(x=0.5, y=0, z=0, pitch=1, roll=2, yaw=3)
            return session.apply_camera_records([Record("front_0", final, 91)])
        if method_name == "simSetCameraFov":
            final = types.SimpleNamespace(x=0.5, y=0, z=0, pitch=1, roll=2, yaw=3)
            return session.apply_camera_records([Record("front_0", final, 91)])
        if method_name == "simSetVehiclePose":
            return session.set_vehicle_pose(Request())
        if method_name == "simGetVehiclePose":
            session.client.pose = Pose(Vector3r(1.0, 2.0, -3.0), Quaternionr(0.1, 0.2, 0.3, 0.9))
            return session.verify_vehicle_pose(Request())
        if method_name == "simGetImages":
            return session.capture_rgb(("front",))
        raise AssertionError("unknown method {}".format(method_name))

    def test_wraps_established_session_transport_failures_with_cause(self):
        cases = (
            ("simSetCameraPose", rpc_error.TimeoutError),
            ("simSetCameraFov", rpc_error.TransportError),
            ("simSetVehiclePose", ConnectionError),
            ("simGetVehiclePose", rpc_error.TimeoutError),
            ("simGetImages", rpc_error.TransportError),
        )
        for method_name, error_type in cases:
            with self.subTest(method_name=method_name, error_type=error_type.__name__):
                original = error_type("session unavailable")
                session = DirectAirSimSession(
                    FailingClient(method_name, original), self.airsim,
                    channel_order="rgb",
                )
                with self.assertRaises(
                    airsim_session.AirSimSessionUnavailableError
                ) as caught:
                    self._invoke(session, method_name)
                self.assertIs(caught.exception.__cause__, original)

    def test_does_not_wrap_rpc_call_or_argument_errors(self):
        for error_type in (rpc_error.CallError, rpc_error.ArgumentError):
            with self.subTest(error_type=error_type.__name__):
                original = error_type("bad rpc call")
                session = DirectAirSimSession(
                    FailingClient("simGetImages", original), self.airsim,
                    channel_order="rgb",
                )
                with self.assertRaises(error_type) as caught:
                    session.capture_rgb(("front",))
                self.assertIs(caught.exception, original)

    def test_does_not_wrap_ordinary_established_session_errors(self):
        original = ValueError("programming error")
        session = DirectAirSimSession(
            FailingClient("simSetVehiclePose", original), self.airsim,
            channel_order="rgb",
        )

        with self.assertRaises(ValueError) as caught:
            session.set_vehicle_pose(Request())

        self.assertIs(caught.exception, original)

    def test_sets_vehicle_pose_using_xyzw_order_and_verifies_sign_equivalence(self):
        self.session.set_vehicle_pose(Request())
        pose = self.client.pose
        self.assertEqual(
            (pose.orientation.x_val, pose.orientation.y_val,
             pose.orientation.z_val, pose.orientation.w_val),
            Request.orientation_quaternion_xyzw,
        )
        position_error, rotation_error = self.session.verify_vehicle_pose(Request())
        self.assertAlmostEqual(position_error, 0.0)
        self.assertAlmostEqual(rotation_error, 0.0)

    def test_applies_camera_pose_then_fov_and_requests_only_four_rgb_images(self):
        final = types.SimpleNamespace(x=0.5, y=0, z=0, pitch=1, roll=2, yaw=3)
        records = [Record("front_0", final, 91), Record("back_0", final, 119)]

        self.session.apply_camera_records(records)
        frames = self.session.capture_rgb(("front", "back", "left", "right"))

        self.assertEqual([item[0] for item in self.client.camera_poses], ["front_0", "back_0"])
        self.assertEqual(self.client.camera_fovs, [("front_0", 91), ("back_0", 119)])
        self.assertEqual(len(self.client.requests[0]), 4)
        self.assertTrue(all(request.image_type == 0 for request in self.client.requests[0]))
        self.assertEqual(set(frames), {"front", "back", "left", "right"})

    def test_captures_native_448_rgb_frames(self):
        client = FakeClient(width=448, height=448)
        session = DirectAirSimSession(
            client, self.airsim, channel_order="rgb",
            image_width=448, image_height=448,
        )

        frames = session.capture_rgb(("front",))

        self.assertEqual(frames["front"].shape, (448, 448, 3))


class AirSimServerRuntimeTests(unittest.TestCase):

    def test_creates_native_448_rgb_capture_settings(self):
        settings = server_tool.create_drones(image_width=448, image_height=448)

        rgb = settings["CameraDefaults"]["CaptureSettings"][0]
        self.assertEqual((rgb["Width"], rgb["Height"]), (448, 448))
        front_rgb = settings["Vehicles"]["Drone_1"]["Cameras"]["front_0"][
            "CaptureSettings"
        ][0]
        self.assertEqual((front_rgb["Width"], front_rgb["Height"]), (448, 448))

    def test_scene_identifier_decodes_msgpack_bytes(self):
        self.assertEqual(
            server_tool.scene_identifier(b"env_airsim_18"),
            "env_airsim_18",
        )

    def test_detects_only_non_airvln_embedded_runtime_settings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            standard = root / "LinuxNoEditor/AirVLN/Binaries/Linux/settings.json"
            custom = root / "LinuxNoEditor/shanghai/Binaries/Linux/settings.json"
            for path in (standard, custom):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")

            self.assertIsNone(server_tool.embedded_runtime_settings_path(standard))
            self.assertEqual(
                server_tool.embedded_runtime_settings_path(custom), custom
            )

    def test_temporary_runtime_settings_override_restores_original_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = Path(temporary_directory) / "settings.json"
            settings.write_text('{"original": true}\n', encoding="utf-8")

            override = server_tool.override_runtime_settings(
                settings, '{"ApiServerPort": 33001}\n'
            )
            self.assertEqual(
                settings.read_text(encoding="utf-8"), '{"ApiServerPort": 33001}\n'
            )

            server_tool.restore_runtime_settings(override)
            self.assertEqual(
                settings.read_text(encoding="utf-8"), '{"original": true}\n'
            )
    def _runtime_with_resources(self, rpc_client):
        runtime = AirSimServerRuntime(".", ".", 0, 30000, "runtime.log")
        runtime.rpc_client = rpc_client
        runtime.process = FakeProcess()
        runtime.log_handle = FakeLogHandle()
        return runtime

    def test_close_rejects_false_close_scenes_result_after_local_cleanup(self):
        rpc_client = FakeRpcClient(close_scenes_result=False)
        runtime = self._runtime_with_resources(rpc_client)
        process = runtime.process
        log_handle = runtime.log_handle

        with self.assertRaisesRegex(RuntimeError, "close_scenes returned false"):
            runtime.close()

        self.assertTrue(rpc_client.closed)
        self.assertTrue(process.terminated)
        self.assertEqual(process.wait_calls, [10])
        self.assertTrue(log_handle.closed)
        self.assertIsNone(runtime.rpc_client)
        self.assertIsNone(runtime.process)
        self.assertIsNone(runtime.log_handle)

    def test_close_reraises_rpc_failure_after_local_cleanup(self):
        original = rpc_error.TransportError("controller disconnected")
        rpc_client = FakeRpcClient(call_error=original)
        runtime = self._runtime_with_resources(rpc_client)
        process = runtime.process
        log_handle = runtime.log_handle

        with self.assertRaises(rpc_error.TransportError) as caught:
            runtime.close()

        self.assertIs(caught.exception, original)
        self.assertTrue(rpc_client.closed)
        self.assertTrue(process.terminated)
        self.assertEqual(process.wait_calls, [10])
        self.assertTrue(log_handle.closed)
        self.assertIsNone(runtime.rpc_client)
        self.assertIsNone(runtime.process)
        self.assertIsNone(runtime.log_handle)

    def test_recovery_close_ignores_unavailable_controller_after_local_cleanup(self):
        rpc_client = FakeRpcClient(call_error=rpc_error.TimeoutError("Request timed out"))
        runtime = self._runtime_with_resources(rpc_client)
        process = runtime.process
        log_handle = runtime.log_handle

        runtime.close(suppress_unavailable_scene_rpc=True)

        self.assertTrue(rpc_client.closed)
        self.assertTrue(process.terminated)
        self.assertEqual(process.wait_calls, [10])
        self.assertTrue(log_handle.closed)
        self.assertIsNone(runtime.rpc_client)
        self.assertIsNone(runtime.process)
        self.assertIsNone(runtime.log_handle)


class ServerEnvRootContractTests(unittest.TestCase):
    def test_server_accepts_explicit_environment_root(self):
        source = (Path(__file__).parents[1] / "airsim_plugin" / "AirVLNSimulatorServerTool.py").read_text()
        self.assertIn('"--env-root"', source)
        self.assertIn("args.env_root", source)


if __name__ == "__main__":
    unittest.main()
