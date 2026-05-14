#!/usr/bin/env python3
"""Micro-benchmark EBFT dense 4D strided-mask cache construction.

This measures mask/position-id construction only. It is useful for quickly
checking whether ``EBFT_CACHE_STRIDED_MASKS=1`` is active before running a full
training smoke.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from contextlib import contextmanager
from pathlib import Path
import sys
import time
from typing import Iterator

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / "openrlhf" / "models" / "utils.py"
spec = importlib.util.spec_from_file_location("openrlhf_models_utils", UTILS_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {UTILS_PATH}")
utils = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = utils
spec.loader.exec_module(utils)

build_strided_attention_mask_and_positions = utils.build_strided_attention_mask_and_positions
clear_strided_mask_cache = utils.clear_strided_mask_cache
get_strided_mask_cache_stats = utils.get_strided_mask_cache_stats


@contextmanager
def temporary_env(**updates: str) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_loop(args: argparse.Namespace, *, cache: bool) -> dict[str, object]:
    device = torch.device(args.device)
    doc_ids = torch.zeros((args.batch_size, args.full_sequence_length), dtype=torch.long, device=device)
    clear_strided_mask_cache()
    with temporary_env(
        EBFT_CACHE_STRIDED_MASKS="1" if cache else "0",
        EBFT_CACHE_STRIDED_MASKS_MAX_MB=str(args.cache_max_mb),
    ):
        for _ in range(args.warmup):
            build_strided_attention_mask_and_positions(
                full_sequence_length=args.full_sequence_length,
                prompt_length=args.prompt_length,
                context_length=args.context_length,
                generation_step=args.generation_step,
                max_generation_length=args.max_generation_length,
                stride=args.stride,
                num_blocks=args.num_blocks,
                device=device,
                doc_ids=doc_ids,
                document_masking=False,
                dtype=torch.float32,
            )
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.iters):
            build_strided_attention_mask_and_positions(
                full_sequence_length=args.full_sequence_length,
                prompt_length=args.prompt_length,
                context_length=args.context_length,
                generation_step=args.generation_step,
                max_generation_length=args.max_generation_length,
                stride=args.stride,
                num_blocks=args.num_blocks,
                device=device,
                doc_ids=doc_ids,
                document_masking=False,
                dtype=torch.float32,
            )
        synchronize(device)
        elapsed = time.perf_counter() - start
        return {
            "cache": cache,
            "iters": args.iters,
            "elapsed_sec": elapsed,
            "sec_per_call": elapsed / args.iters,
            "cache_stats": get_strided_mask_cache_stats(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--full-sequence-length", type=int, default=288)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--generation-step", type=int, default=2)
    parser.add_argument("--max-generation-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--cache-max-mb", type=float, default=1024)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    off = run_loop(args, cache=False)
    on = run_loop(args, cache=True)
    result = {
        "schema": "ebft_densemask_cache_microbenchmark_v1",
        "cache_off": off,
        "cache_on": on,
        "speedup": off["sec_per_call"] / on["sec_per_call"] if on["sec_per_call"] else None,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
