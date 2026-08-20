import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from vln_aug.render_requests import validate_rendered_images


class RenderRequestValidationTests(unittest.TestCase):
    def _write_manifest(self, root: Path):
        request = {
            "request_id": "dataset/ep/frame/front",
            "scene_id": "scene",
            "frame_index": 0,
            "camera_key": "front",
            "expected_height": 8,
            "expected_width": 12,
            "expected_channels": 3,
            "expected_image_relpath": "rendered_images/frame_000000/front.png",
            "body_pose_xyz_yaw": [0.0, 0.0, 0.0, 0.0],
            "coordinate_metadata": {"state_mode": "absolute"},
            "camera_metadata": {"viewpoint_type": "front"},
        }
        path = root / "render_requests_0p2hz.jsonl"
        path.write_text(json.dumps(request) + "\n", encoding="utf-8")
        return path

    def test_accepts_complete_images_with_expected_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_manifest(root)
            image_path = root / "rendered_images/frame_000000/front.png"
            image_path.parent.mkdir(parents=True)
            Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(image_path)

            report = validate_rendered_images(manifest, root)

            self.assertTrue(report["complete"])
            self.assertEqual(report["valid_count"], 1)

    def test_rejects_missing_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = validate_rendered_images(self._write_manifest(root), root)
            self.assertFalse(report["complete"])
            self.assertEqual(report["missing_count"], 1)

    def test_rejects_wrong_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._write_manifest(root)
            image_path = root / "rendered_images/frame_000000/front.png"
            image_path.parent.mkdir(parents=True)
            Image.fromarray(np.zeros((7, 12, 3), dtype=np.uint8)).save(image_path)
            report = validate_rendered_images(manifest, root)
            self.assertFalse(report["complete"])
            self.assertEqual(report["invalid_count"], 1)

    def test_empty_manifest_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "render_requests_0p2hz.jsonl"
            manifest.write_text("", encoding="utf-8")
            report = validate_rendered_images(manifest, root)
            self.assertFalse(report["complete"])

    def test_rejects_duplicate_request_ids_or_output_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = {
                "request_id": "duplicate",
                "scene_id": "scene",
                "frame_index": 0,
                "camera_key": "front",
                "expected_height": 8,
                "expected_width": 12,
                "expected_channels": 3,
                "expected_image_relpath": "same.png",
                "body_pose_xyz_yaw": [0, 0, 0, 0],
                "coordinate_metadata": {"state_mode": "absolute"},
                "camera_metadata": {"viewpoint_type": "front"},
            }
            manifest = root / "render_requests_0p2hz.jsonl"
            manifest.write_text(json.dumps(request) + "\n" + json.dumps(request) + "\n")
            Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(root / "same.png")
            report = validate_rendered_images(manifest, root)
            self.assertFalse(report["complete"])
            self.assertGreaterEqual(report["invalid_count"], 1)


if __name__ == "__main__":
    unittest.main()
