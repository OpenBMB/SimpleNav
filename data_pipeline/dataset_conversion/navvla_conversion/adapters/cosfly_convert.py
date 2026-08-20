from __future__ import annotations

import argparse
import json
from pathlib import Path

from navvla_conversion.adapters._cosfly_splits import build_cosfly_split_manifest
from navvla_conversion.adapters.cosfly import CosFlyAdapter
from navvla_conversion.validation import validate_navvla_lerobot_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and convert CosFly v7 with its isolated adapter.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-manifest")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)

    convert = subparsers.add_parser("convert")
    convert.add_argument("--source-root", type=Path, required=True)
    convert.add_argument("--output-root", type=Path, required=True)
    convert.add_argument("--split-manifest", type=Path, required=True)
    convert.add_argument("--split", choices=["train", "seen", "unseen"], required=True)
    convert.add_argument("--max-episodes", type=int, default=None)
    convert.add_argument("--load-workers", type=int, default=1)
    convert.add_argument(
        "--write-workers",
        "--cache-workers",
        dest="write_workers",
        type=int,
        default=None,
    )
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument("--validate", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-manifest":
        manifest = build_cosfly_split_manifest(
            args.source_root,
            output_path=args.output,
            seed=args.seed,
        )
        print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
        return
    adapter = CosFlyAdapter(
        fps=2.0,
        action_horizon=8,
        split_manifest_path=args.split_manifest,
        load_workers=args.load_workers,
    )
    summary = adapter.convert(
        source_root=args.source_root,
        output_root=args.output_root,
        dataset_name=args.split,
        max_episodes=args.max_episodes,
        fps=2.0,
        control_frequency_hz=2.0,
        action_horizon=8,
        overwrite=args.overwrite,
        split=args.split,
        write_workers=args.write_workers,
        write_visual_token_cache=False,
    )
    if args.validate:
        summary["validation"] = validate_navvla_lerobot_dataset(
            summary["dataset_root"],
            token_budget=2048,
        )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
