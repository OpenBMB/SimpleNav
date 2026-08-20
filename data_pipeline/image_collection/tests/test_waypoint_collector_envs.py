import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

from waypoint_collector.envs import (
    SceneArchiveError,
    inspect_scene_archive,
    inspect_scene_source,
    prepare_isolated_scene_source,
    prepare_scene_archive,
    prepare_scene_source,
    scene_launcher_path,
    scene_runtime_settings_path,
)


REQUIRED = (
    "AirVLN.sh",
    "AirVLN/Binaries/Linux/AirVLN-Linux-Shipping",
    "AirVLN/Content/Paks/AirVLN-LinuxNoEditor.pak",
)


def make_archive(path, scene_id, members=REQUIRED):
    prefix = "env_{}/LinuxNoEditor/".format(scene_id)
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(prefix + member, member.encode("utf-8"))


def make_extracted_scene(root, scene_id, members=REQUIRED):
    scene_root = Path(root) / "env_{}".format(scene_id)
    for member in members:
        path = scene_root / "LinuxNoEditor" / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(member.encode("utf-8"))
    return scene_root


class SceneArchiveTests(unittest.TestCase):
    def test_detects_missing_required_scene_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "env_1.zip"
            make_archive(archive_path, 1, members=REQUIRED[2:])

            result = inspect_scene_archive(archive_path, scene_id="1")

            self.assertFalse(result.complete)
            self.assertIn("env_1/LinuxNoEditor/AirVLN.sh", result.missing_members)

    def test_extracts_valid_archive_atomically_and_reuses_matching_cache(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "env_5.zip"
            cache_root = root / "cache"
            make_archive(archive_path, 5)

            first = prepare_scene_archive(archive_path, cache_root, scene_id="5")
            second = prepare_scene_archive(archive_path, cache_root, scene_id="5")

            self.assertEqual(first.scene_root, cache_root / "env_5")
            self.assertFalse(first.reused)
            self.assertTrue(second.reused)
            manifest = json.loads((first.scene_root / ".archive_manifest.json").read_text())
            self.assertEqual(manifest["scene_id"], "5")
            self.assertEqual(len(manifest["sha256"]), 64)
            self.assertTrue(os.access(
                first.scene_root / "LinuxNoEditor/AirVLN.sh", os.X_OK
            ))
            self.assertTrue(os.access(
                first.scene_root / "LinuxNoEditor/AirVLN/Binaries/Linux/AirVLN-Linux-Shipping",
                os.X_OK,
            ))

    def test_invalid_non_skipped_archive_blocks_preparation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "env_2.zip"
            make_archive(archive_path, 2, members=REQUIRED[:1])

            with self.assertRaises(SceneArchiveError):
                prepare_scene_archive(archive_path, root / "cache", scene_id="2")

    def test_reuses_preextracted_scene_without_copying_it(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "sources"
            scene_root = make_extracted_scene(source_root, 5)
            cache_root = root / "cache"

            first = prepare_scene_source(source_root, cache_root, scene_id="5")
            second = prepare_scene_source(source_root, cache_root, scene_id="5")

            self.assertTrue(first.reused)
            self.assertEqual(first.scene_root.resolve(), scene_root.resolve())
            self.assertTrue((cache_root / "env_5").is_symlink())
            self.assertEqual(second.scene_root, first.scene_root)

    def test_recognizes_openfly_named_scene_and_start_launcher(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene = root / "env_airsim_18" / "LinuxNoEditor"
            launcher = scene / "start.sh"
            executable = scene / "AirVLN/Binaries/Linux/AirVLN-Linux-Shipping"
            pak = scene / "AirVLN/Content/Paks/AirVLN-LinuxNoEditor.pak"
            for path in (launcher, executable, pak):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder", encoding="utf-8")

            inspection = inspect_scene_source(root, "env_airsim_18")
            prepared = prepare_scene_source(root, root / "cache", "env_airsim_18")

            self.assertTrue(inspection.complete)
            self.assertEqual(scene_launcher_path(root, "env_airsim_18"), launcher)
            self.assertEqual(prepared.scene_root.resolve(), (root / "env_airsim_18").resolve())

    def test_finds_custom_openfly_runtime_settings_next_to_start_launcher(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scene = root / "env_airsim_sh" / "LinuxNoEditor"
            launcher = scene / "start.sh"
            settings = scene / "shanghai/Binaries/Linux/settings.json"
            launcher.parent.mkdir(parents=True)
            launcher.write_text("#!/bin/sh\n", encoding="utf-8")
            settings.parent.mkdir(parents=True)
            settings.write_text("{}\n", encoding="utf-8")

            self.assertEqual(
                scene_runtime_settings_path(root, "env_airsim_sh"), settings
            )

    def test_prepares_isolated_worker_scene_with_private_runtime_settings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "sources"
            scene = source_root / "env_airsim_18" / "LinuxNoEditor"
            launcher = scene / "start.sh"
            executable = scene / "city/Binaries/Linux/AirVLN-Linux-Shipping"
            settings = scene / "city/Binaries/Linux/settings.json"
            pak = scene / "city/Content/Paks/AirVLN-LinuxNoEditor.pak"
            for path, contents in (
                (launcher, "#!/bin/sh\n"),
                (executable, "binary"),
                (settings, "{\"port\": 30000}\n"),
                (pak, "assets"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")

            worker_root = root / "worker-4"
            first = prepare_isolated_scene_source(
                source_root, worker_root, "env_airsim_18"
            )
            second = prepare_isolated_scene_source(
                source_root, worker_root, "env_airsim_18"
            )

            isolated_scene = worker_root / "env_airsim_18" / "LinuxNoEditor"
            isolated_settings = (
                isolated_scene / "city/Binaries/Linux/settings.json"
            )
            isolated_pak = (
                isolated_scene / "city/Content/Paks/AirVLN-LinuxNoEditor.pak"
            )
            self.assertEqual(first.scene_root, worker_root / "env_airsim_18")
            self.assertFalse(first.reused)
            self.assertTrue(second.reused)
            self.assertTrue((isolated_scene / "start.sh").is_file())
            self.assertEqual(isolated_settings.read_text(), settings.read_text())
            self.assertNotEqual(isolated_settings.stat().st_ino, settings.stat().st_ino)
            self.assertEqual(isolated_pak.stat().st_ino, pak.stat().st_ino)


if __name__ == "__main__":
    unittest.main()
