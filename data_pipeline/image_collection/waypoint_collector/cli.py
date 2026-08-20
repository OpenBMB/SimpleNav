import argparse


COMMANDS = (
    "preflight", "prepare-envs", "pilot", "render", "assemble",
    "validate", "publish", "run",
)


def _common_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--env-archive-root", required=True)
    parser.add_argument("--env-cache-root", required=True)
    parser.add_argument(
        "--worker-env-cache-roots",
        default="",
        help=(
            "comma-separated environment roots, one per render worker; "
            "used when a custom UE4 scene must run concurrently on multiple GPUs"
        ),
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-scene", default="1")
    parser.add_argument("--views", default="front,back,left,right")
    parser.add_argument("--camera-seed", type=int, default=1)
    parser.add_argument("--channel-order", choices=("rgb", "bgr"), default="rgb")
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--run-id", default="waypoint-v1")
    parser.add_argument("--state-root", default=None)
    parser.add_argument("--base-control-port", type=int, default=31000)
    parser.add_argument("--frame-attempts", type=int, default=10)
    parser.add_argument("--failed-episode-retry-rounds", type=int, default=3)
    parser.add_argument("--estimated-output-gib", type=float, default=250.0)
    parser.add_argument("--space-safety-factor", type=float, default=1.5)
    parser.add_argument(
        "--assembly-workers", type=int, default=1,
        help="number of independent video-part assembly tasks to run concurrently",
    )
    parser.add_argument(
        "--deep-video-validation", action="store_true",
        help="fully decode final part videos during validation (slow audit mode)",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m waypoint_collector",
        description="Collect four-view RGB videos from absolute waypoint requests.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()
    for command in COMMANDS:
        subparsers.add_parser(command, parents=[common])
    return parser


def main(argv=None):
    from waypoint_collector.pipeline import CollectorPipeline

    args = build_parser().parse_args(argv)
    pipeline = CollectorPipeline.from_args(args)
    return pipeline.execute(args.command)
