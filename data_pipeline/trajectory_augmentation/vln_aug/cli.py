import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vln-augment",
        description="Non-destructive VLN train trajectory augmentation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "select", "preview", "enhance", "validate"):
        command = subparsers.add_parser(name)
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument(
            "--reports-dir",
            type=Path,
            default=Path(__file__).resolve().parents[1] / "reports",
        )
    preview_one = subparsers.add_parser("preview-one", help=argparse.SUPPRESS)
    preview_one.add_argument("--dataset-root", type=Path, required=True)
    preview_one.add_argument("--train-split", type=Path, required=True)
    preview_one.add_argument("--reports-dir", type=Path, required=True)
    export = subparsers.add_parser(
        "export-trajectories",
        help="enhance frame poses and export an AerialVLN-format trajectory-only package",
    )
    export.add_argument("--train-split", type=Path, required=True)
    export.add_argument("--dataset-key", required=True)
    export.add_argument("--output-dir", type=Path)
    export.add_argument("--sample-episode-indices", default="")
    export.add_argument("--include-episode-indices", default="")
    export.add_argument("--image-stride-choices", default="1,3,5")
    export.add_argument(
        "--image-stride-policy",
        choices=("fixed-per-episode", "deterministic-random-per-interval"),
        default="fixed-per-episode",
    )
    export.add_argument("--image-interval-seed", type=int, default=0)
    export.add_argument("--image-width", type=int, default=224)
    export.add_argument("--image-height", type=int, default=224)
    export.add_argument("--retain-fraction", type=float)
    export.add_argument("--exclude-scene-ids", default="")
    export.add_argument("--include-scene-ids", default="")
    export.add_argument("--selection-seed", type=int, default=0)
    export.add_argument("--balanced-image-strides", action="store_true")
    export.add_argument("--eligible-package", type=Path)
    export.add_argument("--sample-per-stride", type=int, default=0)
    export.add_argument(
        "--world-pose-source",
        choices=(
            "auto",
            "original-json",
            "openfly-annotation",
            "frame-metadata",
            "observation-state",
        ),
        default="auto",
    )
    export.add_argument("--original-trajectory-json", type=Path)
    export.add_argument("--world-pose-metadata", type=Path)
    export.add_argument("--turn-speed-min-factor", type=float, default=0.55)
    export.add_argument("--turn-curvature-start", type=float, default=0.08)
    export.add_argument("--turn-curvature-full", type=float, default=0.45)
    export.add_argument("--turn-smoothing-multiplier", type=float, default=2.0)
    export.add_argument("--disable-turn-slowdown", action="store_true")
    export.add_argument("--allow-preview-output", action="store_true")
    validate_package = subparsers.add_parser(
        "validate-trajectory-package",
        help="validate final trajectories and direct-render requests",
    )
    validate_package.add_argument("--package-dir", type=Path, required=True)
    for command_name in ("validate-profile", "export-profile"):
        profile_command = subparsers.add_parser(
            command_name,
            help="validate or run a portable dataset augmentation profile",
        )
        profile_command.add_argument("--profile", type=Path, required=True)
        profile_command.add_argument("--dataset-root", type=Path, required=True)
        if command_name == "export-profile":
            profile_command.add_argument("--dry-run", action="store_true")
    enhance = subparsers.choices["enhance"]
    enhance.add_argument("--renderer", required=True)
    enhance.add_argument("--renderer-config", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"inspect", "select", "preview"}:
        from vln_aug.reporting import run_isolated_preview_report

        summary = run_isolated_preview_report(args.dataset_root, args.reports_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "preview-one":
        from vln_aug.reporting import run_one_split_preview

        summary = run_one_split_preview(args.dataset_root, args.train_split, args.reports_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "export-trajectories":
        from vln_aug.aerialvln_export import export_train_split
        from vln_aug.lightweight_subset import read_eligible_episode_indices
        from vln_aug.trajectory import TrajectoryConfig

        source = args.train_split.resolve()
        output = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else source.parent / f"{source.name}_enhanced"
        )
        sample_indices = {
            int(value.strip())
            for value in args.sample_episode_indices.split(",")
            if value.strip()
        }
        include_indices = {
            int(value.strip())
            for value in args.include_episode_indices.split(",")
            if value.strip()
        }
        image_stride_choices = tuple(
            int(value.strip())
            for value in args.image_stride_choices.split(",")
            if value.strip()
        )
        excluded_scene_ids = {
            value.strip()
            for value in args.exclude_scene_ids.split(",")
            if value.strip()
        }
        included_scene_ids = {
            value.strip()
            for value in args.include_scene_ids.split(",")
            if value.strip()
        }
        eligible_episode_indices = (
            read_eligible_episode_indices(args.eligible_package.resolve())
            if args.eligible_package is not None
            else None
        )
        trajectory_config = TrajectoryConfig(
            turn_slowdown_enabled=not args.disable_turn_slowdown,
            turn_speed_min_factor=args.turn_speed_min_factor,
            turn_curvature_start_rad_per_m=args.turn_curvature_start,
            turn_curvature_full_rad_per_m=args.turn_curvature_full,
            turn_smoothing_multiplier=args.turn_smoothing_multiplier,
        )
        summary = export_train_split(
            source_split=source,
            output_dir=output,
            dataset_key=args.dataset_key,
            sample_episode_indices=sample_indices,
            include_episode_indices=include_indices or None,
            require_enhanced_sibling=not args.allow_preview_output,
            config=trajectory_config,
            image_stride_choices=image_stride_choices,
            image_stride_policy=args.image_stride_policy,
            image_interval_seed=args.image_interval_seed,
            render_image_width=args.image_width,
            render_image_height=args.image_height,
            retain_fraction=args.retain_fraction,
            excluded_scene_ids=excluded_scene_ids,
            include_scene_ids=included_scene_ids,
            selection_seed=args.selection_seed,
            balanced_image_strides=args.balanced_image_strides,
            eligible_episode_indices=eligible_episode_indices,
            sample_per_stride=args.sample_per_stride,
            world_pose_source=args.world_pose_source,
            original_trajectory_json=args.original_trajectory_json,
            world_pose_metadata_path=args.world_pose_metadata,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-trajectory-package":
        from vln_aug.aerialvln_export import validate_trajectory_package

        report = validate_trajectory_package(args.package_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["valid"] else 1
    if args.command in {"validate-profile", "export-profile"}:
        from vln_aug.aerialvln_export import export_train_split
        from vln_aug.dataset_profile import (
            build_export_plan,
            load_dataset_profile,
            validate_export_plan,
        )

        profile = load_dataset_profile(args.profile)
        plan = build_export_plan(profile, dataset_root=args.dataset_root)
        preflight = validate_export_plan(plan)
        if args.command == "validate-profile" or args.dry_run:
            print(json.dumps(preflight, ensure_ascii=False, indent=2))
            return 0 if preflight["valid"] else 1
        if not preflight["valid"]:
            raise SystemExit("dataset profile preflight failed: " + "; ".join(preflight["errors"]))
        summary = export_train_split(
            source_split=plan.source_split,
            output_dir=plan.output_dir,
            dataset_key=plan.dataset_key,
            **plan.export_kwargs,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "enhance":
        raise SystemExit("enhance requires a configured real renderer adapter; run inspect/preview first")
    if args.command == "validate":
        raise SystemExit("validate expects an already published enhanced split")
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
