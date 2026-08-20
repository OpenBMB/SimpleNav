import argparse
import threading
import msgpackrpc
from pathlib import Path
import glob
import time
import os
import json
import sys
import subprocess
import errno
import signal
import copy
import socket
import tempfile
import fcntl

try:
    from airsim_plugin.camera_views import camera_specs
except ModuleNotFoundError:
    from camera_views import camera_specs


AIRSIM_SETTINGS_TEMPLATE = {
    "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
    "SettingsVersion": 1.2,
    "SimMode": "ComputerVision", # ComputerVision / Multirotor
    "ViewMode": "NoDisplay", # Fpv / NoDisplay
    "ClockSpeed": 1,
    # "LocalHostIp": "127.0.0.1",
    # "ApiServerPort": 10000,
    "CameraDefaults": {
        "CaptureSettings": [
            {
                "ImageType": 0,
                "Width": 224,
                "Height": 224,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03
            },
            {
                "ImageType": 2,
                "Width": 256,
                "Height": 256,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03
            },
            {
                "ImageType": 3,
                "Width": 256,
                "Height": 256,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03
            }
        ],
        "X": 0,
        "Y": 0,
        "Z": 0,
        "Pitch": 0,
        "Roll": 0,
        "Yaw": 0
    },
    "Recording": {
        "RecordInterval": 0.001,
        "Enabled": False,
        "Cameras": []
    },
    "SubWindows": [],
    "Vehicles": {}
}

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


def create_drones(drone_num_per_env=1, show_scene=False, uav_mode=False,
                  image_width=224, image_height=224) -> dict:
    airsim_settings = copy.deepcopy(AIRSIM_SETTINGS_TEMPLATE)
    image_width = int(image_width)
    image_height = int(image_height)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    airsim_settings["CameraDefaults"]["CaptureSettings"][0]["Width"] = image_width
    airsim_settings["CameraDefaults"]["CaptureSettings"][0]["Height"] = image_height

    if show_scene == True:
        airsim_settings['ViewMode'] = 'Fpv'
    else:
        airsim_settings['ViewMode'] = 'NoDisplay'

    if uav_mode == True:
        airsim_settings['SimMode'] = 'Multirotor'
        airsim_settings['PhysicsEngineName'] = 'ExternalPhysicsEngine'
    else:
        airsim_settings['SimMode'] = 'ComputerVision'


    # create drone objects
    for i in range(drone_num_per_env):
        drone_name = 'Drone_' + str(i+1)

        airsim_settings['Vehicles'][str(drone_name)] = {}

        cameras = {}
        for camera in camera_specs():
            cameras[camera.name] = {
                "CaptureSettings": copy.deepcopy(airsim_settings["CameraDefaults"]["CaptureSettings"]),
                "X": camera.x, "Y": camera.y, "Z": camera.z,
                "Pitch": camera.pitch, "Roll": camera.roll, "Yaw": camera.yaw
            }

        drone = {
            "VehicleType": "ComputerVision",
            "Cameras": cameras,
            "X": 0, "Y": 0, "Z": 0,
            "Pitch": 0, "Roll": 0, "Yaw": 0
        }

        if airsim_settings['SimMode'] == 'ComputerVision':
            drone['VehicleType'] = 'ComputerVision'
        elif airsim_settings['SimMode'] == 'Multirotor':
            drone['VehicleType'] = 'SimpleFlight'
        else:
            raise NotImplementedError

        airsim_settings['Vehicles'][str(drone_name)] = copy.deepcopy(drone)

    return airsim_settings


def scene_identifier(scene_id):
    if isinstance(scene_id, bytes):
        return scene_id.decode("utf-8")
    return str(scene_id)


def embedded_runtime_settings_path(settings_path):
    settings_path = Path(settings_path)
    return None if "AirVLN" in settings_path.parts else settings_path


def override_runtime_settings(settings_path, content):
    settings_path = Path(settings_path)
    backup_handle = tempfile.NamedTemporaryFile(
        prefix=".settings-backup-", dir=str(settings_path.parent), delete=False
    )
    backup_path = Path(backup_handle.name)
    try:
        with settings_path.open("rb") as source:
            backup_handle.write(source.read())
        backup_handle.flush()
        os.fsync(backup_handle.fileno())
    finally:
        backup_handle.close()
    temporary_path = settings_path.with_name(".{}.partial".format(settings_path.name))
    try:
        with temporary_path.open("w", encoding="utf-8") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(str(temporary_path), str(settings_path))
    except Exception:
        backup_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_path.unlink(missing_ok=True)
    return settings_path, backup_path


def restore_runtime_settings(override):
    settings_path, backup_path = override
    try:
        os.replace(str(backup_path), str(settings_path))
    finally:
        Path(backup_path).unlink(missing_ok=True)


def scene_settings_path(
    cwd_dir: Path, control_port: int, scene_index: int
) -> Path:
    return (
        Path(cwd_dir)
        / 'airsim_plugin/settings'
        / 'port_{}'.format(int(control_port))
        / str(int(scene_index) + 1)
        / 'settings.json'
    )


def pid_exists(pid) -> bool:
    """
    Check whether pid exists in the current process table.
    UNIX only.
    """
    if pid < 0:
        return False

    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH == No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM clearly means there's a process to deny access to
            return True
        else:
            # According to "man 2 kill" possible error values are
            # (EINVAL, EPERM, ESRCH)
            raise
    else:
        return True


def FromPortGetPid(port: int):
    try:
        completed = subprocess.run(
            ["ss", "-ltnp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception as e:
        print(
            "{}\t{}\t{}".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                'FromPortGetPid',
                e,
            )
        )
        return None
    except:
        return None

    expected_suffix = ":{}".format(int(port))
    for line in completed.stdout.splitlines():
        columns = line.split()
        if len(columns) < 4 or not columns[3].endswith(expected_suffix):
            continue
        marker = "pid="
        marker_index = line.find(marker)
        if marker_index < 0:
            continue
        pid_text = line[marker_index + len(marker):].split(",", 1)[0]
        try:
            return int(pid_text)
        except ValueError:
            return None
    return None


def KillPid(pid) -> None:
    if pid is None or not isinstance(pid, int):
        print('pid is not int')
        return

    while pid_exists(pid):
        try:
            # os.system(("kill -9 {}".format(pid)))
            os.kill(pid, signal.SIGKILL)
        except Exception as e:
            pass
        time.sleep(0.5)

    return


def KillPorts(ports) -> None:
    threads = []

    def _kill_port(index, port):
        pid = FromPortGetPid(port)
        KillPid(pid)

    for index, port in enumerate(ports):
        thread = threading.Thread(target=_kill_port, args=(index, port))
        threads.append(thread)
    for thread in threads:
        thread.setDaemon(True)
        thread.start()
    for thread in threads:
        thread.join()
    threads = []

    return


def KillAirVLN() -> None:
    subprocess_execute = "pkill -9 AirVLN"

    try:
        p = subprocess.Popen(
            subprocess_execute,
            stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True,
        )
    except Exception as e:
        print(
            "{}\t{}\t{}".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                'KillAirVLN',
                e,
            )
        )
        return
    except:
        return

    try:
        # os.system(("kill -9 {}".format(p.pid)))
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass

    time.sleep(1)
    return


class EventHandler(object):
    def __init__(self):
        scene_ports = []
        for i in range(1000):
            scene_ports.append(
                int(args.port) + (i+1)
            )
        self.scene_ports = scene_ports

        scene_gpus = []
        while len(scene_gpus) < 100:
            scene_gpus += GPU_IDS.copy()
        self.scene_gpus = scene_gpus

        self.scene_used_ports = []
        self.scene_processes = []
        self.runtime_overrides = []
        self.runtime_locks = []

    def _restore_runtime_settings(self):
        first_error = None
        for override in reversed(self.runtime_overrides):
            try:
                restore_runtime_settings(override)
            except Exception as error:
                if first_error is None:
                    first_error = error
        self.runtime_overrides = []
        for lock_handle in self.runtime_locks:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        self.runtime_locks = []
        if first_error is not None:
            raise first_error

    def _close_managed_processes(self):
        for process in self.scene_processes:
            if process is None:
                continue
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.scene_processes = []

    def ping(self) -> bool:
        return True

    def _open_scenes(self, ip: str , scen_ids: list):
        print(
            "{}\tSTART closing scenes ".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        self._close_managed_processes()
        KillPorts(self.scene_used_ports)
        self.scene_used_ports = []
        self._restore_runtime_settings()
        # KillAirVLN()
        print(
            "{}\tEND closing scenes ".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )


        # Occupied airsim port 1
        ports = []
        index = 0
        while len(ports) < len(scen_ids):
            pid = FromPortGetPid(self.scene_ports[index])
            if pid is None or not isinstance(pid, int):
                ports.append(self.scene_ports[index])
            index += 1

        KillPorts(ports)


        # Occupied GPU 2
        gpus = [self.scene_gpus[index] for index in range(len(scen_ids))]


        # search scene path 3
        choose_env_exe_paths = []
        for raw_scene_id in scen_ids:
            scen_id = scene_identifier(raw_scene_id)
            if scen_id.lower() == 'none':
                choose_env_exe_paths.append(None)
                continue

            scene_names = [scen_id]
            if not scen_id.startswith("env_"):
                scene_names.append('env_{}'.format(scen_id))
            launchers = ('AirVLN.sh', 'start.sh')
            candidates = [
                SEARCH_ENVs_PATH / scene_name / 'LinuxNoEditor' / launcher
                for scene_name in scene_names
                for launcher in launchers
            ]
            launcher = next((path for path in candidates if path.is_file()), None)
            if launcher is None:
                print(f'can not find scene file: {scen_id}; checked {candidates}')
                raise KeyError
            choose_env_exe_paths.append(str(launcher))


        p_s = []
        pending_overrides = []
        pending_locks = []
        for index in range(len(scen_ids)):
            # airsim settings 4
            airsim_settings = create_drones(
                image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT
            )
            airsim_settings['ApiServerPort'] = int(ports[index])
            airsim_settings_write_content = json.dumps(airsim_settings)
            settings_path = scene_settings_path(CWD_DIR, PORT, index)
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(settings_path), 'w', encoding='utf-8') as dump_f:
                dump_f.write(airsim_settings_write_content)


            # open scene 5
            if choose_env_exe_paths[index] is None:
                p_s.append(None)
                continue
            else:
                launcher_path = Path(choose_env_exe_paths[index])
                embedded_settings = next(
                    (
                        embedded_runtime_settings_path(path)
                        for path in launcher_path.parent.glob("*/Binaries/Linux/settings.json")
                        if embedded_runtime_settings_path(path) is not None
                    ),
                    None,
                )
                if embedded_settings is not None:
                    lock_path = embedded_settings.with_name(
                        ".{}.collector.lock".format(embedded_settings.name)
                    )
                    lock_handle = lock_path.open("a+")
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                    try:
                        pending_overrides.append(
                            override_runtime_settings(
                                embedded_settings, airsim_settings_write_content
                            )
                        )
                    except Exception:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                        lock_handle.close()
                        raise
                    pending_locks.append(lock_handle)
                subprocess_execute = [
                    "bash",
                    choose_env_exe_paths[index],
                    "-RenderOffscreen",
                    "-NoSound",
                    "-NoVSync",
                    "-GraphicsAdapter={}".format(gpus[index]),
                    "--settings",
                    str(settings_path),
                ]
                scene_log_path = settings_path.parent / "scene_{}.log".format(
                    scen_ids[index]
                )

                try:
                    with open(str(scene_log_path), "ab", buffering=0) as log_handle:
                        p = subprocess.Popen(
                            subprocess_execute,
                            stdin=None,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    p_s.append(p)
                except Exception as e:
                    print(
                        "{}\t{}".format(
                            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                            e,
                        )
                    )
                    return False, None
                except:
                    return False, None
        # Check scene readiness without piping Unreal stdout. Leaving an
        # unread PIPE here can block the renderer once its log buffer fills.
        threads = []
        scene_ready = [False for _ in p_s]

        def _check_scene(index, p):
            if p is None:
                print(
                    "{}\tOpening {}-th scene (scene {})\tgpu:{}".format(
                        str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                        index,
                        None,
                        gpus[index],
                    )
                )
                scene_ready[index] = True
                return

            deadline = time.time() + 180
            while time.time() < deadline:
                if p.poll() is not None:
                    return
                try:
                    connection = socket.create_connection(
                        (ip, ports[index]), timeout=1
                    )
                except OSError:
                    time.sleep(1)
                    continue
                else:
                    connection.close()
                    scene_ready[index] = True
                    break

            print(
                "{}\tOpening {}-th scene (scene {})\tgpu:{}".format(
                    str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                    index,
                    scen_ids[index],
                    gpus[index],
                )
            )
            return

        for index, p in enumerate(p_s):
            thread = threading.Thread(target=_check_scene, args=(index, p))
            threads.append(thread)
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        threads = []

        if not all(scene_ready):
            for process in p_s:
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
            KillPorts(ports)
            for override in reversed(pending_overrides):
                restore_runtime_settings(override)
            for lock_handle in pending_locks:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()
            return False, None

        # ChangeNice(ports)

        self.scene_used_ports += copy.deepcopy(ports)
        self.scene_processes = p_s
        self.runtime_overrides = pending_overrides
        self.runtime_locks = pending_locks

        return True, (ip, ports)

    def reopen_scenes(self, ip: str, scen_ids: list):
        print(
            "{}\tSTART reopen_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        try:
            result = self._open_scenes(ip, scen_ids)
        except Exception as e:
            print(e)
            result = False, None
        print(
            "{}\tEND reopen_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        return result

    def close_scenes(self, ip: str) -> bool:
        print(
            "{}\tSTART close_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )

        try:
            self._close_managed_processes()
            KillPorts(self.scene_used_ports)
            self.scene_used_ports = []
            self._restore_runtime_settings()
            # KillPorts(self.scene_ports)
            # KillAirVLN()

            result = True
        except Exception as e:
            print(e)
            result = False

        print(
            "{}\tEND close_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        return result


def serve_background(server, daemon=False):
    def _start_server(server):
        server.start()
        server.close()

    t = threading.Thread(target=_start_server, args=(server,))
    t.setDaemon(daemon)
    t.start()
    return t


def serve(daemon=False):
    try:
        server = msgpackrpc.Server(EventHandler())
        addr = msgpackrpc.Address(HOST, PORT)
        server.listen(addr)

        thread = serve_background(server, daemon)

        return addr, server, thread
    except Exception as err:
        print(err)
        pass


if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=str,
        default='0',
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help='server port'
    )
    parser.add_argument(
        "--env-root",
        type=str,
        default=None,
        help='directory containing extracted env_<scene_id> folders',
    )
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    args = parser.parse_args()


    HOST = '127.0.0.1'
    PORT = int(args.port)
    IMAGE_WIDTH = int(args.image_width)
    IMAGE_HEIGHT = int(args.image_height)
    if IMAGE_WIDTH <= 0 or IMAGE_HEIGHT <= 0:
        raise ValueError("image dimensions must be positive")

    CWD_DIR = Path(str(os.getcwd())).resolve()
    PROJECT_ROOT_DIR = CWD_DIR.parent
    SEARCH_ENVs_PATH = (
        Path(args.env_root).expanduser().resolve()
        if args.env_root is not None
        else PROJECT_ROOT_DIR / 'ENVs'
    )
    assert os.path.isdir(str(SEARCH_ENVs_PATH)), (
        'environment root does not exist: {}'.format(SEARCH_ENVs_PATH)
    )

    gpu_list = []
    gpus = str(args.gpus).split(',')
    for gpu in gpus:
        gpu_list.append(int(gpu.strip()))
    GPU_IDS = gpu_list.copy()


    addr, server, thread = serve()
    print(f"start listening \t{addr._host}:{addr._port}")
