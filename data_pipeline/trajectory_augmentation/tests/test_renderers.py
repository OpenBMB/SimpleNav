import unittest

import numpy as np

from vln_aug.renderers.base import (
    CameraFrame,
    RenderBatch,
    RenderProvenance,
    validate_publishable_render_batch,
)


class RendererGateTests(unittest.TestCase):
    def _batch(self, backend_type="REAL", returned_pose=None):
        requested = np.array([1.0, 2.0, 3.0, 0.25])
        if returned_pose is None:
            returned_pose = requested.copy()
        return RenderBatch(
            requested_pose=requested,
            returned_pose=np.asarray(returned_pose),
            frames={
                "front": CameraFrame(
                    rgb=np.zeros((8, 12, 3), dtype=np.uint8),
                    receipt="fresh-front-0001",
                )
            },
            provenance=RenderProvenance(
                backend_type=backend_type,
                backend_name="example-simulator",
                scene_id="scene-a",
                render_call_id="call-0001",
            ),
        )

    def test_accepts_real_fresh_complete_render(self):
        validate_publishable_render_batch(
            self._batch(),
            expected_cameras={"front": (8, 12)},
            expected_scene_id="scene-a",
        )

    def test_rejects_mock_or_preview_backend(self):
        with self.assertRaises(ValueError):
            validate_publishable_render_batch(
                self._batch(backend_type="PREVIEW"),
                expected_cameras={"front": (8, 12)},
                expected_scene_id="scene-a",
            )

    def test_rejects_missing_camera(self):
        batch = self._batch()
        batch.frames.clear()
        with self.assertRaises(ValueError):
            validate_publishable_render_batch(
                batch,
                expected_cameras={"front": (8, 12)},
                expected_scene_id="scene-a",
            )

    def test_rejects_wrong_dimensions(self):
        with self.assertRaises(ValueError):
            validate_publishable_render_batch(
                self._batch(),
                expected_cameras={"front": (9, 12)},
                expected_scene_id="scene-a",
            )

    def test_rejects_render_pose_mismatch(self):
        with self.assertRaises(ValueError):
            validate_publishable_render_batch(
                self._batch(returned_pose=[2.0, 2.0, 3.0, 0.25]),
                expected_cameras={"front": (8, 12)},
                expected_scene_id="scene-a",
                translation_tolerance_m=0.01,
            )

    def test_rejects_scene_mismatch(self):
        with self.assertRaises(ValueError):
            validate_publishable_render_batch(
                self._batch(),
                expected_cameras={"front": (8, 12)},
                expected_scene_id="scene-b",
            )


if __name__ == "__main__":
    unittest.main()
