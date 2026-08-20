# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


def add_new_tokens(
    model,
    tokenizer,
    new_tokens: List[str],
    init_strategy: str = "avg",
    as_special: bool = True,
) -> Tuple[Dict[str, int], int, int, int]:
    vocab = tokenizer.get_vocab()
    to_add_tokens = [t for t in new_tokens if t not in vocab]

    old_embed = model.get_input_embeddings()
    old_embed_size = old_embed.weight.shape[0]

    added_now = 0
    if to_add_tokens:
        if as_special:
            added_now = tokenizer.add_special_tokens({"additional_special_tokens": to_add_tokens})
        else:
            added_now = tokenizer.add_tokens(to_add_tokens)

    target_size = old_embed_size + added_now
    token_start_idx = old_embed_size
    token_end_idx = old_embed_size - 1
    if target_size > old_embed_size:
        model.resize_token_embeddings(target_size)
        new_embed = model.get_input_embeddings()
        with torch.no_grad():
            if init_strategy == "avg":
                ref_vec = old_embed.weight.mean(dim=0, keepdim=True)
                for idx in range(old_embed_size, target_size):
                    new_embed.weight[idx].copy_(ref_vec[0])
            elif init_strategy == "zero":
                for idx in range(old_embed_size, target_size):
                    new_embed.weight[idx].zero_()
            elif init_strategy == "normal":
                for idx in range(old_embed_size, target_size):
                    nn.init.normal_(new_embed.weight[idx], mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unknown init_strategy: {init_strategy}")
        token_end_idx = target_size - 1

    mapping = {t: tokenizer.convert_tokens_to_ids(t) for t in new_tokens}
    return mapping, added_now, token_start_idx, token_end_idx


def save_bundle(
    model,
    tokenizer,
    mapping: Dict[str, int],
    save_dir: str,
    processor_src: str | None = None,
):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "added_custom_token_id_map.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved model + tokenizer to: {save_dir}")

    try:
        src = processor_src or save_dir
        processor = AutoProcessor.from_pretrained(src, trust_remote_code=True)
        processor.tokenizer = tokenizer
        processor.save_pretrained(save_dir)
        print(f"[OK] AutoProcessor saved to: {save_dir}")
    except Exception as exc:
        print(f"[WARN] Failed to save AutoProcessor: {exc}")


def reload_and_check(save_dir: str, tokens: List[str]) -> bool:
    tok = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
    vocab = tok.get_vocab()
    missing = [t for t in tokens if t not in vocab]
    if missing:
        print(f"[WARN] Still missing after reload: {missing}")
        return False

    for token in tokens:
        ids = tok.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            print(f"[WARN] {token!r} is not a single token after reload: ids={ids}")
            return False

    print("[OK] Reload check passed, all tokens exist as single-token entries.")
    return True


def parse_tokens(args) -> List[str]:
    tokens: List[str] = []
    if args.tokens:
        tokens.extend([t.strip() for t in args.tokens.split(",") if t.strip()])
    if args.tokens_file:
        with open(args.tokens_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens.append(line)

    seen = set()
    ordered = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def resolve_device_map(device: str):
    if device == "auto":
        return "auto"
    if device == "cpu":
        return {"": "cpu"}
    return device


def main():
    parser = argparse.ArgumentParser(
        description="Add special tokens to openbmb/MiniCPM-V-4.6 and save to a local directory."
    )
    parser.add_argument("--model-id", default="openbmb/MiniCPM-V-4.6", help="HF Hub model ID or local path")
    parser.add_argument("--save-dir", required=True, help="Output directory to save")
    parser.add_argument("--tokens", default="", help="Comma-separated tokens")
    parser.add_argument("--tokens-file", help="Text file containing tokens to add (one per line)")
    parser.add_argument(
        "--init-strategy",
        default="avg",
        choices=["avg", "normal", "zero"],
        help="Initialization strategy for newly added embeddings",
    )
    parser.add_argument("--as-special", action="store_true", help="Whether to add as special tokens")
    parser.add_argument("--no-as-special", dest="as_special", action="store_false")
    parser.set_defaults(as_special=True)
    parser.add_argument("--padding-side", default="left", choices=["left", "right"])
    parser.add_argument("--device", default="cuda", help="cuda / cpu / mps / auto")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention backend when loading MiniCPM-V-4.6",
    )
    args = parser.parse_args()

    tokens = parse_tokens(args)
    if not tokens:
        print("No tokens provided, use --tokens or --tokens-file")
        return

    print(f"[INFO] Tokens to process: {tokens}")
    print(f"[INFO] Loading model: {args.model_id}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = args.padding_side

    model_kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if args.attn_implementation != "eager":
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.device != "cuda":
        model_kwargs["device_map"] = resolve_device_map(args.device)

    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    processor.tokenizer.padding_side = args.padding_side

    print(f"[DEBUG] tokenizer.vocab_size(base) = {tokenizer.vocab_size}")
    print(f"[DEBUG] len(tokenizer)(total)     = {len(tokenizer)}")
    print(f"[DEBUG] model.embed_size(before)  = {model.get_input_embeddings().weight.shape[0]}")

    mapping, added, token_start_idx, token_end_idx = add_new_tokens(
        model=model,
        tokenizer=tokenizer,
        new_tokens=tokens,
        init_strategy=args.init_strategy,
        as_special=args.as_special,
    )

    save_bundle(model, tokenizer, mapping, args.save_dir, processor_src=args.model_id)
    reload_and_check(args.save_dir, tokens)

    print(f"[INFO] Newly added to tokenizer: {added}")
    print(f"[INFO] Token mapping: {mapping}")
    print(f"[INFO] New token embed idx range: [{token_start_idx}, {token_end_idx}]")
    print(f"[DEBUG] model.embed_size(after)   = {model.get_input_embeddings().weight.shape[0]}")


if __name__ == "__main__":
    main()
