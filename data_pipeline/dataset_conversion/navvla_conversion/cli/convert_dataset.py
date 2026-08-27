from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from navvla_conversion.adapters import get_adapter
from navvla_conversion.validation import validate_navvla_lerobot_dataset

ADAPTER_CHOICES = (
    "traveluav",
    "aerialvln",
    "vlnce_rendered",
    "flight",
    "indooruav",
    "huge",
    "embodiednav",
    "enhanced_vln",
    "openfly",
    "openscene",
    "nuscenes",
)


def add_common_convert_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter", choices=ADAPTER_CHOICES, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="vln_train")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None, help="Image/video frame frequency in Hz.")
    parser.add_argument("--control-frequency-hz", type=float, default=None, help="Action/control frequency in Hz.")
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--split", default="train")
    parser.add_argument("--context-policy-version", default="bats-v1")
    parser.add_argument("--cache-policy-version", default="smoke-coarse-v1")
    parser.add_argument(
        "--write-workers",
        "--cache-workers",
        dest="write_workers",
        type=int,
        default=None,
        help="Parallel workers used to prepare shards and encode/copy videos.",
    )
    parser.add_argument("--episodes-per-file", type=int, default=20)
    parser.add_argument("--files-per-chunk", type=int, default=50)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--overwrite", action="store_true")
    resume_group.add_argument(
        "--repair-existing",
        action="store_true",
        help="Resume/repair an existing output root by reusing complete episode shards and rewriting missing or invalid episode shards.",
    )
    parser.add_argument("--media-cache-root", type=Path, default=None)
    parser.add_argument("--load-workers", type=int, default=None)
    parser.add_argument("--reuse-media-cache", action="store_true")
    parser.add_argument("--fail-on-missing-media", action="store_true")
    parser.add_argument("--extracted-root", type=Path, default=None)
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--traj-root", type=Path, default=None)
    parser.add_argument("--scene-prefix", action="append", dest="scene_prefixes", default=None)
    parser.add_argument("--fail-on-missing-source", action="store_true")
    parser.add_argument("--dataset-version", default="v1.0-trainval")
    parser.add_argument("--validate", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a registered NavVLA source dataset to NavVLA LeRobot v3 format."
    )
    add_common_convert_args(parser)
    return parser


def convert_from_args(args: argparse.Namespace) -> dict[str, Any]:
    default_fps = {
        "aerialvln": 1.0,
        "vlnce_rendered": 1.0,
        "flight": 1.0,
        "indooruav": 10.0,
        "huge": 5.0,
        "embodiednav": 1.0,
        "enhanced_vln": 1.0,
        "openfly": 5.0,
        "openscene": 2.0,
        "nuscenes": 2.0,
    }.get(args.adapter, 0.2)
    fps = float(args.fps) if args.fps is not None else default_fps
    control_frequency_hz = (
        float(args.control_frequency_hz)
        if args.control_frequency_hz is not None
        else (
            2.0
            if args.adapter == "flight"
            else (
                fps
                if args.adapter
                in {
                    "aerialvln",
                    "vlnce_rendered",
                    "indooruav",
                    "huge",
                    "embodiednav",
                    "enhanced_vln",
                    "openfly",
                    "openscene",
                    "nuscenes",
                }
                else 1.0
            )
        )
    )
    adapter = get_adapter(args.adapter)
    configure_kwargs: dict[str, Any] = {"fps": fps, "action_horizon": args.action_horizon}
    if args.adapter == "aerialvln":
        configure_kwargs.update(
            media_cache_root=args.media_cache_root,
            reuse_media_cache=args.reuse_media_cache,
        )
    if args.adapter == "enhanced_vln":
        configure_kwargs["load_workers"] = args.load_workers
    if args.adapter in {"flight", "huge", "embodiednav", "openfly", "openscene"}:
        configure_kwargs.update(media_cache_root=args.media_cache_root, load_workers=args.load_workers)
    if args.adapter in {"huge", "embodiednav", "openfly", "openscene"}:
        configure_kwargs["reuse_media_cache"] = args.reuse_media_cache
    if args.adapter in {"flight", "openscene"}:
        configure_kwargs["fail_on_missing_media"] = args.fail_on_missing_media
    if args.adapter == "indooruav":
        configure_kwargs["extracted_root"] = args.extracted_root
    if args.adapter == "openfly":
        configure_kwargs.update(
            annotation_root=args.annotation_root,
            traj_root=args.traj_root,
            scene_prefixes=tuple(args.scene_prefixes or ()),
            fail_on_missing_source=args.fail_on_missing_source,
        )
    if args.adapter == "nuscenes":
        configure_kwargs["dataset_version"] = args.dataset_version
    adapter = adapter.configure(**configure_kwargs)
    source_root = args.source_root
    dataset_name = args.dataset_name
    if dataset_name == "vln_train" and args.adapter == "openscene":
        from navvla_conversion.adapters.openscene import default_dataset_name

        dataset_name = default_dataset_name(args.split)
    if dataset_name == "vln_train" and args.adapter == "nuscenes":
        from navvla_conversion.adapters.nuscenes import default_dataset_name

        dataset_name = default_dataset_name(args.split)
    summary = adapter.convert(
        source_root=source_root,
        output_root=args.output_root,
        dataset_name=dataset_name,
        max_episodes=args.max_episodes,
        fps=fps,
        control_frequency_hz=control_frequency_hz,
        action_horizon=args.action_horizon,
        overwrite=args.overwrite,
        repair_existing=args.repair_existing,
        split=args.split,
        context_policy_version=args.context_policy_version,
        cache_policy_version=args.cache_policy_version,
        write_workers=args.write_workers,
        write_visual_token_cache=False,
        episodes_per_file=args.episodes_per_file,
        files_per_chunk=args.files_per_chunk,
    )
    if args.validate:
        summary["validation"] = validate_navvla_lerobot_dataset(summary["dataset_root"])
    return summary

def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(convert_from_args(args), indent=2))


if __name__ == "__main__":
    main()
