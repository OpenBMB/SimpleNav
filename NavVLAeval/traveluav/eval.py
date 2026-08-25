from __future__ import annotations

import argparse
import json
from pathlib import Path

from NavVLAeval.common.config import load_eval_config
from NavVLAeval.common.runner.parallel_runner import run_eval_from_config


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or dry-run TravelUAV evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_argparser().parse_args(argv)
    cfg = load_eval_config(args.config, overrides=args.override)
    summary = run_eval_from_config(cfg, dry_run=args.dry_run, repo_root=Path.cwd())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
