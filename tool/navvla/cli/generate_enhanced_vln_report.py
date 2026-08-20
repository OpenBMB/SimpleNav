from __future__ import annotations

import argparse
import json
from pathlib import Path

from tool.navvla.enhanced_vln_report import generate_episode_report, generate_standalone_episode_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an offline HTML report for one enhanced VLN episode")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="write one self-contained .html with four MP4 videos embedded as base64",
    )
    args = parser.parse_args()
    if args.standalone:
        report = generate_standalone_episode_report(
            args.dataset_root,
            output_html=args.output,
            episode_index=args.episode_index,
        )
    else:
        report = generate_episode_report(
            args.dataset_root,
            output_dir=args.output,
            episode_index=args.episode_index,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
