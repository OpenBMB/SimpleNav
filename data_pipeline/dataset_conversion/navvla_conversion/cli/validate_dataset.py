from __future__ import annotations

import argparse
import json
from pathlib import Path

from navvla_conversion.context_index import DEFAULT_CONTEXT_TOKEN_BUDGET, available_context_token_budgets
from navvla_conversion.validation import MEDIA_DECODE_MODES, validate_navvla_lerobot_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate standalone NavVLA LeRobot v3 artifacts.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--data-rows-per-shard", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_CONTEXT_TOKEN_BUDGET)
    parser.add_argument("--all-token-budgets", action="store_true")
    parser.add_argument("--check-media-decode", choices=sorted(MEDIA_DECODE_MODES), default="none")
    parser.add_argument("--media-decode-sample", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    budgets = (
        available_context_token_budgets(args.dataset_root)
        if args.all_token_budgets
        else [int(args.token_budget)]
    )
    if not budgets:
        raise FileNotFoundError(f"no current-format context budgets found under {args.dataset_root}")
    reports = {
        str(budget): validate_navvla_lerobot_dataset(
            args.dataset_root,
            data_rows_per_shard=args.data_rows_per_shard,
            sample_seed=args.sample_seed,
            token_budget=int(budget),
            check_media_decode=args.check_media_decode,
            media_decode_sample=args.media_decode_sample,
        )
        for budget in budgets
    }
    payload = reports if args.all_token_budgets else reports[str(budgets[0])]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
