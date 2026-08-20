import tempfile
import unittest
from pathlib import Path

from vln_aug.redraw import require_delivery_report_dir


class RedrawSafetyTests(unittest.TestCase):
    def test_accepts_dataset_directory_inside_delivery_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "reports_delivery"
            report = root / "dataset_a"
            report.mkdir(parents=True)
            self.assertEqual(require_delivery_report_dir(report, root), report.resolve())

    def test_rejects_directory_outside_delivery_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "reports_delivery"
            outside = base / "reports_verified" / "dataset_a"
            root.mkdir()
            outside.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "reports_delivery"):
                require_delivery_report_dir(outside, root)

    def test_rejects_delivery_root_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "reports_delivery"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "dataset subdirectory"):
                require_delivery_report_dir(root, root)


if __name__ == "__main__":
    unittest.main()
