import tempfile
import unittest
from pathlib import Path

from vln_aug.pipeline import RendererUnavailable, preview_without_publish


class NoRendererIntegrationTests(unittest.TestCase):
    def test_unavailable_renderer_creates_preview_only_and_no_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "A_lerobot" / "vln_train"
            source.mkdir(parents=True)
            reports = root / "reports"

            with self.assertRaises(RendererUnavailable):
                preview_without_publish(
                    source_split=source,
                    reports_dir=reports,
                    renderer_available=False,
                )

            self.assertTrue((reports / "capability.json").is_file())
            self.assertTrue((reports / "selection.json").is_file())
            self.assertTrue((reports / "metrics.json").is_file())
            self.assertTrue((reports / "trajectory_preview.png").is_file())
            self.assertFalse((source.parent / "vln_train_enhanced").exists())
            self.assertEqual(list(source.parent.glob(".vln_train_enhanced.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
