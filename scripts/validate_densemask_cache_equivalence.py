#!/usr/bin/env python3
"""Validate that EBFT dense strided-mask caching is representation-equivalent.

This script compares ``build_strided_attention_mask_and_positions`` with cache
disabled and enabled. It intentionally tests the current production dense 4D
mask path only; it does not exercise any sparse / BlockMask experiment.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path
import sys
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
def temporary_env(**updates: str | None) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_doc_ids(batch: int, length: int, *, device: torch.device, variant: int = 0) -> torch.Tensor:
    base = torch.arange(length, device=device).unsqueeze(0).expand(batch, -1)
    if variant == 0:
        return torch.zeros((batch, length), dtype=torch.long, device=device)
    if variant == 1:
        return (base // 5).long()
    return ((base + torch.arange(batch, device=device).unsqueeze(1)) // 3).long()


def build_once(
    *,
    cache: bool,
    device: torch.device,
    doc_ids: torch.Tensor,
    dtype: torch.dtype,
    document_masking: bool,
    shape: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    with temporary_env(EBFT_CACHE_STRIDED_MASKS="1" if cache else "0"):
        return build_strided_attention_mask_and_positions(
            full_sequence_length=shape["full_sequence_length"],
            prompt_length=shape["prompt_length"],
            context_length=shape["context_length"],
            generation_step=shape["generation_step"],
            max_generation_length=shape["max_generation_length"],
            stride=shape["stride"],
            num_blocks=shape["num_blocks"],
            device=device,
            doc_ids=doc_ids,
            document_masking=document_masking,
            dtype=dtype,
        )


def assert_equal(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    if not torch.equal(lhs, rhs):
        diff = (lhs != rhs).sum().item()
        raise AssertionError(f"{name} mismatch: {diff} entries differ")


def run_case(device: torch.device, dtype: torch.dtype, shape: dict[str, int]) -> None:
    batch = shape["batch"]
    full_len = shape["full_sequence_length"]

    clear_strided_mask_cache()
    doc_ids = make_doc_ids(batch, full_len, device=device, variant=0)
    ref_mask, ref_pos = build_once(
        cache=False,
        device=device,
        doc_ids=doc_ids,
        dtype=dtype,
        document_masking=False,
        shape=shape,
    )

    cached_mask_1, cached_pos_1 = build_once(
        cache=True,
        device=device,
        doc_ids=doc_ids,
        dtype=dtype,
        document_masking=False,
        shape=shape,
    )
    # Different doc IDs must still hit the same cache entry when document masking
    # is disabled, because the dense mask is shape-only in this path.
    cached_mask_2, cached_pos_2 = build_once(
        cache=True,
        device=device,
        doc_ids=make_doc_ids(batch, full_len, device=device, variant=2),
        dtype=dtype,
        document_masking=False,
        shape=shape,
    )

    assert_equal("non_document_mask", ref_mask, cached_mask_1)
    assert_equal("non_document_pos", ref_pos, cached_pos_1)
    assert_equal("non_document_cached_mask_hit", ref_mask, cached_mask_2)
    assert_equal("non_document_cached_pos_hit", ref_pos, cached_pos_2)

    stats = get_strided_mask_cache_stats()
    if stats["stores"] != 1 or stats["hits"] < 1 or stats["entries"] != 1:
        raise AssertionError(f"unexpected cache stats for non-document path: {stats}")

    clear_strided_mask_cache()
    doc_ids = make_doc_ids(batch, full_len, device=device, variant=1)
    ref_mask, ref_pos = build_once(
        cache=False,
        device=device,
        doc_ids=doc_ids,
        dtype=dtype,
        document_masking=True,
        shape=shape,
    )
    cached_mask, cached_pos = build_once(
        cache=True,
        device=device,
        doc_ids=doc_ids,
        dtype=dtype,
        document_masking=True,
        shape=shape,
    )
    assert_equal("document_mask", ref_mask, cached_mask)
    assert_equal("document_pos", ref_pos, cached_pos)
    stats = get_strided_mask_cache_stats()
    if stats["entries"] != 0 or stats["stores"] != 0:
        raise AssertionError(f"document-masking path should not cache: {stats}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--include-bf16", action="store_true", help="also test bfloat16 masks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    shapes = [
        {
            "batch": 1,
            "full_sequence_length": 12,
            "prompt_length": 8,
            "context_length": 4,
            "generation_step": 2,
            "max_generation_length": 4,
            "stride": 2,
            "num_blocks": 2,
        },
        {
            "batch": 3,
            "full_sequence_length": 28,
            "prompt_length": 16,
            "context_length": 4,
            "generation_step": 3,
            "max_generation_length": 8,
            "stride": 2,
            "num_blocks": 4,
        },
    ]
    dtypes = [torch.float32]
    if args.include_bf16:
        dtypes.append(torch.bfloat16)

    for dtype in dtypes:
        for shape in shapes:
            run_case(device=device, dtype=dtype, shape=shape)
    print("PASS: dense strided-mask cache is equivalent for tested shapes.")


if __name__ == "__main__":
    main()
