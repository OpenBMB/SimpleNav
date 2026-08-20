from __future__ import annotations

import argparse
import json
from pathlib import Path

from tool.navvla.context_index import DEFAULT_CONTEXT_TOKEN_BUDGET, available_context_token_budgets
from tool.navvla.validation import MEDIA_DECODE_MODES
from tool.navvla.validation import validate_navvla_lerobot_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate current-format NavVLA LeRobot artifacts.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--visual-token-mode",
        default="online_images",
        choices=["online_images", "offline_cache", "cached_history_online_current"],
    )
    parser.add_argument("--visual-token-profile", default="qwen3_vl_4b_pooled_history")
    parser.add_argument("--cache-sample-size", type=int, default=32)
    parser.add_argument("--data-rows-per-shard", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_CONTEXT_TOKEN_BUDGET)
    parser.add_argument("--all-token-budgets", action="store_true")
    parser.add_argument("--check-media-decode", choices=sorted(MEDIA_DECODE_MODES), default="none")
    parser.add_argument("--media-decode-sample", type=int, default=3)
    parser.add_argument("--smoke-load", type=int, default=0)
    parser.add_argument("--smoke-load-all", action="store_true")
    parser.add_argument("--required-cameras", nargs="+", default=None)
    parser.add_argument("--image-resize", nargs=2, type=int, default=None, metavar=("WIDTH", "HEIGHT"))
    args = parser.parse_args()
    image_resize = tuple(args.image_resize) if args.image_resize is not None else None
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
            visual_token_mode=args.visual_token_mode,
            visual_token_profile=args.visual_token_profile,
            cache_sample_size=args.cache_sample_size,
            data_rows_per_shard=args.data_rows_per_shard,
            sample_seed=args.sample_seed,
            token_budget=int(budget),
            check_media_decode=args.check_media_decode,
            media_decode_sample=args.media_decode_sample,
            smoke_load=args.smoke_load,
            smoke_load_all=args.smoke_load_all,
            required_cameras=args.required_cameras,
            image_resize=image_resize,
        )
        for budget in budgets
    }
    payload = reports if args.all_token_budgets else reports[str(budgets[0])]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
