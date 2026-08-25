import tempfile
import unittest
from pathlib import Path

from vln_aug.safety import (
    assert_reports_outside_sources,
    assert_safe_output_path,
    build_sha256_manifest,
    refuse_existing_output,
)


class SafetyTests(unittest.TestCase):
    def test_rejects_source_split_as_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "A_lerobot" / "vln_train"
            source.mkdir(parents=True)
            with self.assertRaises(ValueError):
                assert_safe_output_path(source, source)

    def test_rejects_output_inside_source_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "A_lerobot" / "vln_train"
            source.mkdir(parents=True)
            with self.assertRaises(ValueError):
                assert_safe_output_path(source, source / "enhanced")

    def test_accepts_required_sibling_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "A_lerobot" / "vln_train"
            source.mkdir(parents=True)
            output = source.parent / "vln_train_enhanced"
            assert_safe_output_path(source, output)

    def test_rejects_wrong_sibling_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "A_lerobot" / "vln_train"
            source.mkdir(parents=True)
            with self.assertRaises(ValueError):
                assert_safe_output_path(source, source.parent / "augmented")

    def test_refuses_existing_enhanced_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "A_lerobot" / "vln_train"
            source.mkdir(parents=True)
            output = source.parent / "vln_train_enhanced"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                refuse_existing_output(output)

    def test_sha256_manifest_detects_changed_source_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "vln_train"
            (source / "meta").mkdir(parents=True)
            payload = source / "meta" / "info.json"
            payload.write_bytes(b"abc")
            before = build_sha256_manifest(source)
            payload.write_bytes(b"abd")
            after = build_sha256_manifest(source)
            self.assertNotEqual(before, after)

    def test_rejects_reports_directory_inside_any_source_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "A" / "vln_train"
            source.mkdir(parents=True)
            with self.assertRaises(ValueError):
                assert_reports_outside_sources(source / "generated_reports", [source])

    def test_accepts_tool_owned_reports_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "A" / "vln_train"
            source.mkdir(parents=True)
            assert_reports_outside_sources(Path(tmp) / "tool" / "reports", [source])


if __name__ == "__main__":
    unittest.main()
