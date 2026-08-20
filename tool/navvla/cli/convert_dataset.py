from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tool.navvla.adapters import get_adapter
from tool.navvla.cli.generate_visual_cache import load_qwen3_encoder
from tool.navvla.validation import validate_navvla_lerobot_dataset
from tool.navvla.visual_token_cache import DEFAULT_VISUAL_TOKEN_PROFILE, VisualTokenProfile

ADAPTER_CHOICES = (
    "traveluav",
    "uav_flow",
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
    parser.add_argument("--cache-workers", type=int, default=None)
    parser.add_argument("--episodes-per-file", type=int, default=20)
    parser.add_argument("--files-per-chunk", type=int, default=50)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--overwrite", action="store_true")
    resume_group.add_argument(
        "--repair-existing",
        action="store_true",
        help="Resume/repair an existing output root by reusing complete episode shards and rewriting missing or invalid episode shards.",
    )
    parser.add_argument("--no-visual-token-cache", action="store_true")
    parser.add_argument("--visual-token-profile", default=DEFAULT_VISUAL_TOKEN_PROFILE)
    parser.add_argument("--visual-token-encoder-name", default="Qwen3-VL-4B-Instruct")
    parser.add_argument("--visual-token-encoder-ckpt", default="")
    parser.add_argument("--visual-token-head", default="qwen3_vl_visual")
    parser.add_argument("--visual-token-level", default="pooled_history")
    parser.add_argument("--visual-token-count", type=int, default=4)
    parser.add_argument("--visual-token-hidden-dim", type=int, default=2560)
    parser.add_argument("--visual-token-dtype", default="float16")
    parser.add_argument("--visual-token-deepstack-layers", type=int, default=3)
    parser.add_argument("--validate", action="store_true")


def add_uav_flow_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", choices=["real", "sim"], default="real")
    parser.add_argument("--media-cache-root", type=Path, default=None)
    parser.add_argument("--instruction-field", choices=["instruction", "instruction_unified"], default="instruction")
    parser.add_argument("--load-workers", type=int, default=None)
    parser.add_argument("--source-root-is-family-root", action="store_true")
    parser.add_argument("--reuse-media-cache", action="store_true")
    parser.add_argument("--fail-on-missing-media", action="store_true")
    parser.add_argument("--extracted-root", type=Path, default=None)
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--traj-root", type=Path, default=None)
    parser.add_argument("--scene-prefix", action="append", dest="scene_prefixes", default=None)
    parser.add_argument("--fail-on-missing-source", action="store_true")
    parser.add_argument("--dataset-version", default="v1.0-trainval")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a registered NavVLA source dataset to NavVLA LeRobot v3 format."
    )
    add_common_convert_args(parser)
    add_uav_flow_args(parser)
    return parser


def convert_from_args(args: argparse.Namespace) -> dict[str, Any]:
    default_fps = {
        "uav_flow": 5.0,
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
                    "uav_flow",
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
    visual_token_profile = _visual_token_profile_from_args(args) if not args.no_visual_token_cache else None
    visual_token_encoder_factory = (
        (lambda: load_qwen3_encoder(encoder_ckpt=str(args.visual_token_encoder_ckpt), profile=visual_token_profile))
        if visual_token_profile is not None
        else None
    )
    adapter = get_adapter(args.adapter)
    configure_kwargs: dict[str, Any] = {"fps": fps, "action_horizon": args.action_horizon}
    if args.adapter in {"aerialvln", "uav_flow"}:
        configure_kwargs.update(
            media_cache_root=args.media_cache_root,
            reuse_media_cache=args.reuse_media_cache,
        )
    if args.adapter == "uav_flow":
        configure_kwargs.update(
            variant=args.variant,
            instruction_field=args.instruction_field,
            load_workers=args.load_workers,
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
    if args.adapter == "uav_flow" and args.source_root_is_family_root:
        from tool.navvla.adapters.uav_flow import resolve_uav_flow_source_root

        source_root = resolve_uav_flow_source_root(source_root, args.variant)
    dataset_name = args.dataset_name
    if dataset_name == "vln_train" and args.adapter == "openscene":
        from tool.navvla.adapters.openscene import default_dataset_name

        dataset_name = default_dataset_name(args.split)
    if dataset_name == "vln_train" and args.adapter == "nuscenes":
        from tool.navvla.adapters.nuscenes import default_dataset_name

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
        cache_workers=args.cache_workers,
        write_visual_token_cache=not args.no_visual_token_cache,
        visual_token_profile=visual_token_profile,
        visual_token_encoder_factory=visual_token_encoder_factory,
        episodes_per_file=args.episodes_per_file,
        files_per_chunk=args.files_per_chunk,
    )
    if args.validate:
        summary["validation"] = validate_navvla_lerobot_dataset(summary["dataset_root"])
    return summary


def _visual_token_profile_from_args(args: argparse.Namespace) -> VisualTokenProfile:
    return VisualTokenProfile(
        name=args.visual_token_profile,
        visual_head=args.visual_token_head,
        encoder_name=args.visual_token_encoder_name,
        encoder_ckpt=str(args.visual_token_encoder_ckpt),
        token_level=args.visual_token_level,
        token_count=args.visual_token_count,
        hidden_dim=args.visual_token_hidden_dim,
        dtype=args.visual_token_dtype,
        has_deepstack=args.visual_token_deepstack_layers > 0,
        deepstack_layers=args.visual_token_deepstack_layers,
    )


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(convert_from_args(args), indent=2))


if __name__ == "__main__":
    main()
