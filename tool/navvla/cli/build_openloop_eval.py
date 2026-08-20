from __future__ import annotations

import argparse
import json
from pathlib import Path

from tool.navvla.openloop_eval import (
    DEFAULT_OPENLOOP_EVAL_ROOT,
    DEFAULT_QWEN35_ENCODER,
    build_openloop_eval_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed five-dataset NavVLA open-loop eval suite.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OPENLOOP_EVAL_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes-per-split", type=int, default=100)
    parser.add_argument("--targets-per-split", type=int, default=400)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--encoder-ckpt", type=Path, default=DEFAULT_QWEN35_ENCODER)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--cache-prefetch-batches", type=int, default=2)
    parser.add_argument("--skip-visual-cache", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = build_openloop_eval_suite(
        output_root=args.output_root,
        seed=args.seed,
        episodes_per_split=args.episodes_per_split,
        targets_per_split=args.targets_per_split,
        token_budget=args.token_budget,
        overwrite=args.overwrite,
        generate_visual_cache=not args.skip_visual_cache,
        encoder_ckpt=args.encoder_ckpt,
        cache_batch_size=args.cache_batch_size,
        cache_prefetch_batches=args.cache_prefetch_batches,
        validate=not args.skip_validation,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
