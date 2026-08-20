from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


class SceneArchiveError(RuntimeError):
    pass


def _scene_source_names(scene_id):
    scene_id = str(scene_id)
    return (scene_id,) if scene_id.startswith("env_") else (
        scene_id, "env_{}".format(scene_id),
    )


def _scene_launcher(source):
    source = Path(source)
    for name in ("AirVLN.sh", "start.sh"):
        candidate = source / "LinuxNoEditor" / name
        if candidate.is_file():
            return candidate
    return None


def scene_launcher_path(source_root, scene_id):
    source = scene_source_path(source_root, scene_id)
    launcher = _scene_launcher(source)
    if launcher is None:
        raise SceneArchiveError(
            "scene {} has no LinuxNoEditor/AirVLN.sh or start.sh launcher".format(
                scene_id
            )
        )
    return launcher


def scene_runtime_settings_path(source_root, scene_id):
    launcher = scene_launcher_path(source_root, scene_id)
    if launcher.name != "start.sh":
        return None
    candidates = tuple(launcher.parent.glob("*/Binaries/Linux/settings.json"))
    if len(candidates) != 1:
        raise SceneArchiveError(
            "scene {} must contain exactly one custom runtime settings file".format(
                scene_id
            )
        )
    return candidates[0]


def required_scene_members(scene_id):
    prefix = "env_{}/LinuxNoEditor".format(scene_id)
    return (
        "{}/AirVLN.sh".format(prefix),
        "{}/AirVLN/Binaries/Linux/AirVLN-Linux-Shipping".format(prefix),
        "{}/AirVLN/Content/Paks/AirVLN-LinuxNoEditor.pak".format(prefix),
    )


@dataclass(frozen=True)
class SceneArchiveInspection:
    scene_id: str
    archive_path: Path
    complete: bool
    missing_members: tuple


@dataclass(frozen=True)
class PreparedScene:
    scene_id: str
    scene_root: Path
    reused: bool
    sha256: str


def inspect_scene_archive(archive_path, scene_id):
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        return SceneArchiveInspection(str(scene_id), archive_path, False,
                                      required_scene_members(scene_id))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile) as error:
        raise SceneArchiveError("invalid scene archive {}: {}".format(archive_path, error)) from error
    missing = tuple(member for member in required_scene_members(scene_id) if member not in names)
    return SceneArchiveInspection(str(scene_id), archive_path, not missing, missing)


def scene_source_path(source_root, scene_id):
    source_root = Path(source_root)
    names = _scene_source_names(scene_id)
    for name in names:
        extracted = source_root / name
        if extracted.is_dir():
            return extracted
    for name in names:
        archive = source_root / "{}.zip".format(name)
        if archive.is_file():
            return archive
    return source_root / "{}.zip".format(names[-1])


def inspect_scene_source(source_root, scene_id):
    source = scene_source_path(source_root, scene_id)
    if source.is_file():
        return inspect_scene_archive(source, scene_id)
    launcher = _scene_launcher(source)
    missing = () if launcher is not None else (
        "{}/LinuxNoEditor/{{AirVLN.sh,start.sh}}".format(source.name),
    )
    return SceneArchiveInspection(str(scene_id), source, not missing, missing)


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def scene_archive_uncompressed_size(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        return sum(info.file_size for info in archive.infolist())


def _safe_extract(archive, destination):
    destination = Path(destination).resolve()
    for info in archive.infolist():
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise SceneArchiveError("unsafe ZIP member: {}".format(info.filename))
        target = (destination / Path(*member.parts)).resolve()
        if destination != target and destination not in target.parents:
            raise SceneArchiveError("unsafe ZIP member: {}".format(info.filename))
    archive.extractall(str(destination))


def prepare_scene_archive(archive_path, cache_root, scene_id):
    archive_path = Path(archive_path)
    cache_root = Path(cache_root)
    inspection = inspect_scene_archive(archive_path, scene_id)
    if not inspection.complete:
        raise SceneArchiveError(
            "scene {} archive is missing: {}".format(
                scene_id, ", ".join(inspection.missing_members)
            )
        )
    stat = archive_path.stat()
    scene_root = cache_root / "env_{}".format(scene_id)
    manifest_path = scene_root / ".archive_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("archive_size") == stat.st_size and
                    manifest.get("archive_mtime_ns") == stat.st_mtime_ns and
                    all((cache_root / member).is_file() for member in required_scene_members(scene_id))):
                return PreparedScene(str(scene_id), scene_root, True, manifest["sha256"])
        except (OSError, ValueError, KeyError):
            pass
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(archive_path)
    temporary_root = Path(tempfile.mkdtemp(
        prefix=".env_{}-".format(scene_id), dir=str(cache_root)
    ))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, temporary_root)
        extracted_scene = temporary_root / "env_{}".format(scene_id)
        if not extracted_scene.is_dir():
            raise SceneArchiveError("scene archive did not create {}".format(extracted_scene.name))
        for member in required_scene_members(scene_id):
            if not (temporary_root / member).is_file():
                raise SceneArchiveError("extracted scene is missing {}".format(member))
        for relative_path in (
            Path("LinuxNoEditor/AirVLN.sh"),
            Path("LinuxNoEditor/AirVLN/Binaries/Linux/AirVLN-Linux-Shipping"),
        ):
            executable = extracted_scene / relative_path
            executable.chmod(executable.stat().st_mode | 0o755)
        payload = {
            "scene_id": str(scene_id),
            "archive_path": str(archive_path.resolve()),
            "archive_size": stat.st_size,
            "archive_mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }
        (extracted_scene / ".archive_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        previous_root = cache_root / ".env_{}.previous-{}".format(
            scene_id, os.getpid()
        )
        if previous_root.exists():
            shutil.rmtree(previous_root)
        if scene_root.exists():
            os.replace(str(scene_root), str(previous_root))
        try:
            os.replace(str(extracted_scene), str(scene_root))
        except Exception:
            if previous_root.exists() and not scene_root.exists():
                os.replace(str(previous_root), str(scene_root))
            raise
        else:
            shutil.rmtree(previous_root, ignore_errors=True)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return PreparedScene(str(scene_id), scene_root, False, digest)


def prepare_scene_source(source_root, cache_root, scene_id):
    source = scene_source_path(source_root, scene_id)
    if source.is_file():
        return prepare_scene_archive(source, cache_root, scene_id)

    inspection = inspect_scene_source(source_root, scene_id)
    if not inspection.complete:
        raise SceneArchiveError(
            "scene {} directory is missing: {}".format(
                scene_id, ", ".join(inspection.missing_members)
            )
        )
    launcher = _scene_launcher(source)
    if launcher is None:
        raise SceneArchiveError("scene {} has no supported launcher".format(scene_id))
    launcher.chmod(launcher.stat().st_mode | 0o755)

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_scene = cache_root / source.name
    if cache_scene.resolve() == source.resolve():
        return PreparedScene(str(scene_id), cache_scene, True, "preextracted")
    if cache_scene.is_symlink() and cache_scene.resolve() == source.resolve():
        return PreparedScene(str(scene_id), cache_scene, True, "preextracted")
    if cache_scene.exists() or cache_scene.is_symlink():
        raise SceneArchiveError(
            "scene cache path already exists and does not match source: {}".format(
                cache_scene
            )
        )
    temporary_link = cache_root / ".{}.link-{}".format(source.name, os.getpid())
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(source.resolve(), target_is_directory=True)
    os.replace(str(temporary_link), str(cache_scene))
    return PreparedScene(str(scene_id), cache_scene, True, "preextracted")


def _hardlink_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _make_runtime_settings_private(source_root, worker_root, scene_id):
    source_settings = scene_runtime_settings_path(source_root, scene_id)
    worker_settings = scene_runtime_settings_path(worker_root, scene_id)
    if source_settings is None or worker_settings is None:
        return
    source_stat = source_settings.stat()
    worker_stat = worker_settings.stat()
    if (
        source_stat.st_dev != worker_stat.st_dev
        or source_stat.st_ino != worker_stat.st_ino
    ):
        return
    temporary = worker_settings.with_name(
        ".{}.collector-private-{}".format(worker_settings.name, os.getpid())
    )
    temporary.unlink(missing_ok=True)
    shutil.copy2(source_settings, temporary)
    os.replace(str(temporary), str(worker_settings))


def prepare_isolated_scene_source(source_root, worker_root, scene_id):
    """Hard-link immutable scene assets and privatize mutable runtime settings."""
    source_root = Path(source_root)
    worker_root = Path(worker_root)
    source = scene_source_path(source_root, scene_id)
    if not source.is_dir():
        raise SceneArchiveError(
            "isolated scene source must already be prepared: {}".format(source)
        )
    worker_root.mkdir(parents=True, exist_ok=True)
    target = worker_root / source.name
    if target.exists() or target.is_symlink():
        if target.is_symlink() or _scene_launcher(target) is None:
            raise SceneArchiveError(
                "isolated scene cache is invalid: {}".format(target)
            )
        _make_runtime_settings_private(source_root, worker_root, scene_id)
        return PreparedScene(str(scene_id), target, True, "hardlinked")

    temporary = Path(tempfile.mkdtemp(
        prefix=".{}-".format(source.name), dir=str(worker_root)
    ))
    temporary_scene = temporary / source.name
    try:
        shutil.copytree(
            source,
            temporary_scene,
            symlinks=False,
            copy_function=_hardlink_or_copy,
        )
        os.replace(str(temporary_scene), str(target))
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    _make_runtime_settings_private(source_root, worker_root, scene_id)
    return PreparedScene(str(scene_id), target, False, "hardlinked")
