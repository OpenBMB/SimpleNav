import argparse
import json
from pathlib import Path

from vln_aug.lerobot_io import extract_episode_rows, read_episode_metadata
from vln_aug.reporting import _stable_seed, _trajectory_from_table
from vln_aug.trajectory import TrajectoryConfig, smooth_and_retime
from vln_aug.visualize import plot_trajectory_comparison


DEFAULT_DELIVERY_ROOT = Path(__file__).resolve().parents[1] / "reports_delivery"


def require_delivery_report_dir(report_dir: Path, delivery_root: Path) -> Path:
    report = Path(report_dir).resolve()
    root = Path(delivery_root).resolve()
    if report == root:
        raise ValueError("report_dir must be a dataset subdirectory of reports_delivery")
    if not report.is_relative_to(root):
        raise ValueError(f"report_dir must be inside reports_delivery: {root}")
    return report


def redraw_dataset_report(
    report_dir: Path, delivery_root: Path = DEFAULT_DELIVERY_ROOT
) -> list[Path]:
    report = require_delivery_report_dir(report_dir, delivery_root)
    metrics_path = report / "metrics.json"
    capability_path = report / "capability.json"
    dataset_key = report.name
    capability = json.loads(capability_path.read_text(encoding="utf-8"))
    train_split = Path(capability["train_split"])
    metadata = {episode.episode_index: episode for episode in read_episode_metadata(train_split)}
    redrawn = []
    for item in json.loads(metrics_path.read_text(encoding="utf-8"))["episodes"]:
        if item["status"] != "preview_generated":
            continue
        episode_index = int(item["episode"]["episode_index"])
        episode = metadata[episode_index]
        source_table = extract_episode_rows(train_split, episode)
        trajectory = smooth_and_retime(
            _trajectory_from_table(source_table),
            TrajectoryConfig(),
            seed=_stable_seed(dataset_key, episode_index),
        )
        output = report / f"episode_{episode_index:06d}_comparison.png"
        plot_trajectory_comparison(
            trajectory,
            output,
            title=f"{dataset_key} episode {episode_index} scene={episode.scene_id}",
        )
        redrawn.append(output)
    return redrawn


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args(argv)
    outputs = redraw_dataset_report(args.report_dir)
    print(json.dumps({"redrawn": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
