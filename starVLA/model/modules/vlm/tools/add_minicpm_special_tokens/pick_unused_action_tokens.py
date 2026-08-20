# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

import argparse
import json
import os
from typing import Dict, List

from transformers import AutoTokenizer


def collect_action_token_ids(
    tokenizer,
    *,
    prefix: str = "a",
    count: int = 2048,
) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for index in range(count):
        token = f"{prefix}{index}"
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"{token!r} is not a single token: ids={token_ids}")
        mapping[token] = int(token_ids[0])
    return mapping


def validate_single_token(tokenizer, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{token!r} is not a single token: ids={token_ids}")
    token_id = int(token_ids[0])
    print(f"[OK] {token!r} -> id={token_id}, decode={tokenizer.decode([token_id])!r}")
    return token_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate MiniCPM-V-4.6 action placeholder tokens as single vocabulary entries "
            "without resizing embeddings."
        )
    )
    parser.add_argument("--model-id", default="openbmb/MiniCPM-V-4.6", help="HF Hub model ID or local path")
    parser.add_argument("--save-dir", required=True, help="Directory to write unused_action_token_id_map.json")
    parser.add_argument("--token", default="", help="Single placeholder token to validate, e.g. ◆")
    parser.add_argument("--prefix", default="a", help="Token prefix for batch mode, e.g. a -> a0, a1, ...")
    parser.add_argument("--count", type=int, default=2048, help="Number of prefixed tokens to validate in batch mode")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if args.token:
        token = args.token.strip()
        token_id = validate_single_token(tokenizer, token)
        mapping = {token: token_id}
        payload = {
            "model_id": args.model_id,
            "action_placeholder_token": token,
            "action_placeholder_token_id": token_id,
            "token_to_id": mapping,
        }
    else:
        mapping = collect_action_token_ids(tokenizer, prefix=args.prefix, count=args.count)
        token_ids = list(mapping.values())
        payload = {
            "model_id": args.model_id,
            "action_token_prefix": args.prefix,
            "action_token_count": args.count,
            "action_token_min": min(token_ids),
            "action_token_max": max(token_ids),
            "token_to_id": mapping,
        }

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "unused_action_token_id_map.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved action token map to: {out_path}")
    if args.token:
        print(f"[INFO] action_placeholder_token={payload['action_placeholder_token']!r}, id={payload['action_placeholder_token_id']}")
    else:
        print(f"[INFO] action_token_min={payload['action_token_min']}, action_token_max={payload['action_token_max']}")
        print(f"[INFO] example: {args.prefix}0 -> {mapping[f'{args.prefix}0']}")


if __name__ == "__main__":
    main()
