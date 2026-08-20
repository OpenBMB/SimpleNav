import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vln_aug.safety import assert_reports_outside_sources, assert_safe_output_path


class RendererUnavailable(RuntimeError):
    pass


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def preview_without_publish(source_split: Path, reports_dir: Path, renderer_available: bool) -> None:
    source = source_split.resolve()
    output = source.parent / f"{source.name}_enhanced"
    assert_safe_output_path(source, output)
    reports = reports_dir.resolve()
    assert_reports_outside_sources(reports, [source])
    reports.mkdir(parents=True, exist_ok=True)
    _write_json(reports / "capability.json", {"renderer_available": bool(renderer_available)})
    _write_json(reports / "selection.json", {"episodes": []})
    _write_json(reports / "metrics.json", {"status": "preview_only"})
    figure, axis = plt.subplots(figsize=(4, 3))
    axis.plot(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    axis.set_title("Trajectory preview unavailable")
    figure.tight_layout()
    figure.savefig(reports / "trajectory_preview.png", dpi=100)
    plt.close(figure)
    if not renderer_available:
        raise RendererUnavailable("real renderer or scene is unavailable; no enhanced split was published")
