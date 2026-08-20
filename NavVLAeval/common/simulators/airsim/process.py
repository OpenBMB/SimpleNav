from __future__ import annotations

import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

XVFB_RUN_PREFIX = ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24"]

TRAVELUAV_SCENE_PATHS = {
    "NYC_dev": ("closeloop_envs", "NYCEnvironmentMegapa"),
    "NYCEnvironmentMegapa": ("closeloop_envs", "NYCEnvironmentMegapa"),
    "TropicalIsland": ("closeloop_envs", "TropicalIsland"),
    "NewYorkCity": ("closeloop_envs", "NewYorkCity"),
    "ModernCityMap": ("closeloop_envs", "ModernCityMap"),
    "ModularPark": ("closeloop_envs", "ModularPark"),
    "ModularEuropean": ("closeloop_envs", "ModularEuropean"),
    "Carla_Town01": ("carla_town_envs/Town01/LinuxNoEditor", "CarlaUE4"),
    "Carla_Town02": ("carla_town_envs/Town02/LinuxNoEditor", "CarlaUE4"),
    "Carla_Town03": ("carla_town_envs/Town03/LinuxNoEditor", "CarlaUE4"),
    "Carla_Town04": ("carla_town_envs/Town04/LinuxNoEditor", "CarlaUE4"),
    "Carla_Town05": ("carla_town_envs/Town05/LinuxNoEditor", "CarlaUE4"),
    "Carla_Town06": ("carla_town_envs/Town06/LinuxNoEditor", "CarlaUE4"),
    "Carla_Town07": ("carla_town_envs/Town07/LinuxNoEditor", "CarlaUE4"),
    "Carla_Town10HD": ("carla_town_envs/Town10HD/LinuxNoEditor", "CarlaUE4"),
    "Carla_Town15": ("carla_town_envs/Town15/LinuxNoEditor", "CarlaUE4"),
}


@dataclass(frozen=True)
class AirSimLaunchConfig:
    start_script: Path
    physical_gpu_id: int
    settings_path: Path
    ue_args: list[str]
    settings_argument_style: str = "equals"


def build_airsim_launch_command(config: AirSimLaunchConfig) -> list[str]:
    settings_path = config.settings_path.resolve()
    command = [
        *XVFB_RUN_PREFIX,
        "bash",
        str(config.start_script),
        "-RenderOffscreen",
        "-NoSound",
        "-NoVSync",
        f"-GraphicsAdapter={int(config.physical_gpu_id)}",
    ]
    if config.settings_argument_style == "space":
        command.extend(["--settings", str(settings_path)])
    elif config.settings_argument_style == "equals":
        command.append(f"-settings={settings_path}")
    else:
        raise ValueError(f"unsupported AirSim settings argument style: {config.settings_argument_style}")
    for arg in config.ue_args:
        if arg not in command:
            command.append(arg)
    return command


def build_airsim_launch_env(render_lib_root: str | Path, *, physical_gpu_id: int | None = None) -> dict[str, str]:
    render_lib_root = Path(render_lib_root)
    required_files = [
        render_lib_root / "etc" / "nvidia_icd.json",
        render_lib_root / "etc" / "10_nvidia.json",
        render_lib_root / "lib",
    ]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"missing AirSim Vulkan render dependency: {path}")
    env = os.environ.copy()
    current_ld_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{render_lib_root / 'lib'}:{current_ld_library_path}"
    env["VK_DRIVER_FILES"] = str(render_lib_root / "etc" / "nvidia_icd.json")
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(render_lib_root / "etc" / "10_nvidia.json")
    if physical_gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu_id))
    return env


def copytree_with_hardlinks(src: Path, dst: Path) -> None:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, copy_function=os.link)
    _detach_launcher_chmod_targets(src, dst)


_CHMOD_PROJECT_EXECUTABLE_PATTERN = re.compile(
    r"^\s*chmod\s+\+x\s+[\"']?\$(?:\{UE4_PROJECT_ROOT\}|UE4_PROJECT_ROOT)/(?P<path>[^\"']+?)[\"']?\s*$"
)


def _detach_launcher_chmod_targets(src: Path, dst: Path) -> None:
    """Give worker launchers private inodes for binaries they chmod at startup."""
    for source_script in src.rglob("start.sh"):
        relative_script = source_script.relative_to(src)
        worker_script = dst / relative_script
        if not worker_script.is_file():
            continue
        for line in source_script.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _CHMOD_PROJECT_EXECUTABLE_PATTERN.match(line)
            if match is None:
                continue
            relative_target = Path(match.group("path"))
            if relative_target.is_absolute() or ".." in relative_target.parts:
                raise ValueError(f"unsafe executable path in AirSim launcher {source_script}: {relative_target}")
            source_target = source_script.parent / relative_target
            worker_target = worker_script.parent / relative_target
            if not source_target.is_file() or not worker_target.is_file():
                raise FileNotFoundError(
                    f"AirSim launcher executable does not exist: source={source_target}, worker={worker_target}"
                )
            source_stat = source_target.stat()
            worker_stat = worker_target.stat()
            if (source_stat.st_dev, source_stat.st_ino) != (worker_stat.st_dev, worker_stat.st_ino):
                continue
            fd, temporary_name = tempfile.mkstemp(prefix=f".{worker_target.name}.", dir=worker_target.parent)
            os.close(fd)
            temporary_path = Path(temporary_name)
            try:
                shutil.copy2(source_target, temporary_path)
                temporary_path.replace(worker_target)
            finally:
                temporary_path.unlink(missing_ok=True)


def resolve_binary_settings_path(env_dir: Path) -> Path:
    return env_dir / "LinuxNoEditor" / "AirVLN" / "Binaries" / "Linux" / "settings.json"


def resolve_airsim_start_script(env_root: str | Path, env_name: str, *, layout: str = "openfly") -> Path:
    env_root = Path(env_root)
    if layout == "traveluav":
        if env_name in TRAVELUAV_SCENE_PATHS:
            rel_dir, bash_name = TRAVELUAV_SCENE_PATHS[env_name]
            return env_root / rel_dir / f"{bash_name}.sh"
        direct = env_root / env_name / f"{env_name}.sh"
        if direct.exists():
            return direct
        matches = sorted(env_root.glob(f"**/{env_name}.sh"))
        if matches:
            return matches[0]
        return direct
    if layout == "openfly":
        return env_root / env_name / "LinuxNoEditor" / "start.sh"
    if layout == "aerialvln":
        candidates = [str(env_name).strip()]
        if candidates[0] and not candidates[0].startswith("env_"):
            candidates.append(f"env_{candidates[0]}")
        for candidate in candidates:
            direct = env_root / candidate / "LinuxNoEditor" / "AirVLN.sh"
            if direct.exists():
                return direct
        return env_root / candidates[-1] / "LinuxNoEditor" / "AirVLN.sh"
    raise ValueError(f"unsupported AirSim env layout: {layout}")


def pid_for_listening_port(port: int) -> int | None:
    for command in (["ss", "-ltnp"], ["netstat", "-nlp"]):
        try:
            output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
        except Exception:
            continue
        for line in output.splitlines():
            if f":{int(port)}" not in line:
                continue
            match = re.search(r"pid=(\d+)", line)
            if match:
                return int(match.group(1))
            tail = line.strip().split()[-1].split("/")[0]
            try:
                return int(tail)
            except Exception:
                continue
    return None


def kill_pid(pid: int | None) -> None:
    if pid is None:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def kill_process_group(pid: int | None) -> None:
    if pid is None:
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(0.5)
    except OSError:
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except OSError:
        pass
