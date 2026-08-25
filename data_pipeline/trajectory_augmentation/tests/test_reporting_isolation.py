import unittest
from pathlib import Path

from vln_aug.reporting import build_isolated_preview_command


class ReportingIsolationTests(unittest.TestCase):
    def test_builds_one_dataset_subprocess_command(self):
        command = build_isolated_preview_command(
            python_executable="/usr/bin/python3",
            dataset_root=Path("/datasets"),
            train_split=Path("/datasets/A/vln_train"),
            reports_dir=Path("/reports"),
        )
        self.assertEqual(
            command,
            [
                "/usr/bin/python3",
                "-m",
                "vln_aug.cli",
                "preview-one",
                "--dataset-root",
                "/datasets",
                "--train-split",
                "/datasets/A/vln_train",
                "--reports-dir",
                "/reports",
            ],
        )


if __name__ == "__main__":
    unittest.main()
