#!/usr/bin/env python3
"""Benchmark remote teacher throughput against single or multi-worker vLLM.

This script mirrors the training-side remote teacher path:
- load real AoPS questions from disk
- build the same teacher provider abstraction used by training
- issue prompt batches with n_samples=M completions per prompt
- report prompts/s and completions/s for each measured iteration

By default it avoids cache so the numbers reflect server throughput rather
than warm-cache hits.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from statistics import mean
from types import SimpleNamespace

from datasets import DatasetDict, load_from_disk

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from openrlhf.utils.teacher_provider import build_teacher_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark teacher provider throughput")
    parser.add_argument(
        "--prompt-data",
        default="/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict",
        help="Path to HF dataset directory",
    )
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument("--input-key", default="question", help="Question column name")
    parser.add_argument(
        "--teacher-api-base",
        required=True,
        help="One URL or comma-separated list of worker URLs",
    )
    parser.add_argument("--teacher-model-name", default="qwen3.5-27b")
    parser.add_argument("--teacher-api-key", default="teacher-local")
    parser.add_argument(
        "--teacher-api-style",
        default="completions",
        choices=["completions", "chat_completions"],
    )
    parser.add_argument("--teacher-remote-batch-size", type=int, default=8)
    parser.add_argument("--teacher-timeout", type=int, default=180)
    parser.add_argument("--teacher-max-retries", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--prompts-per-iter", type=int, default=32)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measure-iters", type=int, default=3)
    parser.add_argument(
        "--prompt-offset",
        type=int,
        default=0,
        help="Start offset inside the unique-question list",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="/mnt/data/models/qwen3.5-27b",
        help="Tokenizer path used when truncating prompts to training length",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=256,
        help="Truncate each question to this many tokens to mirror training prompt_max_len",
    )
    parser.add_argument(
        "--teacher-system-prompt-text",
        default="You are a precise assistant. produce a correct and well-reasoned answer. Step by step when necessary. Keep reasoning sufficient. Final answer is clearly stated.",
    )
    parser.add_argument("--teacher-system-prompt-id", default="v1-balanced")
    parser.add_argument(
        "--teacher-cache-enable",
        action="store_true",
        help="Enable provider cache during benchmarking",
    )
    parser.add_argument("--teacher-cache-dir", default=None)
    return parser.parse_args()


def load_unique_questions(
    prompt_data: str,
    split: str,
    input_key: str,
    tokenizer_path: str,
    max_prompt_tokens: int,
) -> list[str]:
    ds = load_from_disk(prompt_data)
    if isinstance(ds, DatasetDict):
        if split not in ds:
            raise ValueError(f"split {split!r} not found in dataset; available={list(ds.keys())}")
        ds = ds[split]
    questions = ds[input_key]
    if max_prompt_tokens > 0:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        truncated_questions = []
        for text in questions:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) > max_prompt_tokens:
                token_ids = token_ids[:max_prompt_tokens]
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
            truncated_questions.append(text)
        questions = truncated_questions
    unique_questions = sorted(set(questions))
    if not unique_questions:
        raise ValueError("no questions found in dataset")
    return unique_questions


def make_provider(args: argparse.Namespace):
    ns = SimpleNamespace(
        teacher_backend="remote",
        teacher_api_base=args.teacher_api_base,
        teacher_model_name=args.teacher_model_name,
        teacher_api_key=args.teacher_api_key,
        teacher_api_style=args.teacher_api_style,
        teacher_timeout=args.teacher_timeout,
        teacher_max_retries=args.teacher_max_retries,
        teacher_remote_batch_size=args.teacher_remote_batch_size,
        teacher_cache_enable=args.teacher_cache_enable,
        teacher_cache_dir=args.teacher_cache_dir,
        save_path="/tmp",
        teacher_system_prompt_text=args.teacher_system_prompt_text,
        teacher_system_prompt_id=args.teacher_system_prompt_id,
    )
    return build_teacher_provider(ns)


def collect_prompt_windows(
    questions: list[str],
    prompts_per_iter: int,
    total_iters: int,
    prompt_offset: int,
) -> list[list[str]]:
    needed = prompts_per_iter * total_iters
    if len(questions) < needed:
        raise ValueError(
            f"need at least {needed} unique questions, only have {len(questions)}"
        )
    start = prompt_offset
    end = start + needed
    if end > len(questions):
        raise ValueError(
            f"prompt_offset={prompt_offset} with needed={needed} exceeds dataset size={len(questions)}"
        )
    chosen = questions[start:end]
    return [
        chosen[i * prompts_per_iter : (i + 1) * prompts_per_iter]
        for i in range(total_iters)
    ]


def main() -> int:
    args = parse_args()
    questions = load_unique_questions(
        args.prompt_data,
        args.split,
        args.input_key,
        args.tokenizer_path,
        args.max_prompt_tokens,
    )
    total_iters = args.warmup_iters + args.measure_iters
    prompt_windows = collect_prompt_windows(
        questions,
        prompts_per_iter=args.prompts_per_iter,
        total_iters=total_iters,
        prompt_offset=args.prompt_offset,
    )
    provider = make_provider(args)
    if provider is None:
        raise RuntimeError("failed to build remote teacher provider")

    worker_count = len([u for u in args.teacher_api_base.split(",") if u.strip()])
    print("Teacher benchmark configuration")
    print(f"- workers: {worker_count}")
    print(f"- api_base: {args.teacher_api_base}")
    print(f"- prompts_per_iter: {args.prompts_per_iter}")
    print(f"- max_prompt_tokens: {args.max_prompt_tokens}")
    print(f"- teacher_remote_batch_size: {args.teacher_remote_batch_size}")
    print(f"- n_samples: {args.n_samples}")
    print(f"- max_new_tokens: {args.max_new_tokens}")
    print(f"- temperature/top_p: {args.temperature}/{args.top_p}")
    print(f"- cache_enabled: {args.teacher_cache_enable}")
    print()

    measured_prompts_per_sec: list[float] = []
    measured_completions_per_sec: list[float] = []

    for iter_idx, prompts in enumerate(prompt_windows):
        t0 = time.perf_counter()
        outputs = provider.sample_targets(
            prompts,
            args.n_samples,
            args.temperature,
            args.top_p,
            args.max_new_tokens,
        )
        elapsed = time.perf_counter() - t0
        prompts_per_sec = len(prompts) / max(elapsed, 1e-9)
        completions = len(prompts) * args.n_samples
        completions_per_sec = completions / max(elapsed, 1e-9)
        phase = "warmup" if iter_idx < args.warmup_iters else "measure"
        print(
            f"[{phase}] iter={iter_idx + 1}/{total_iters} "
            f"elapsed={elapsed:.2f}s "
            f"prompts/s={prompts_per_sec:.2f} "
            f"completions/s={completions_per_sec:.2f}"
        )
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"provider returned {len(outputs)} prompt outputs, expected {len(prompts)}"
            )
        if phase == "measure":
            measured_prompts_per_sec.append(prompts_per_sec)
            measured_completions_per_sec.append(completions_per_sec)

    print()
    print("Summary")
    print(f"- avg prompts/s: {mean(measured_prompts_per_sec):.2f}")
    print(f"- avg completions/s: {mean(measured_completions_per_sec):.2f}")
    print(f"- best prompts/s: {max(measured_prompts_per_sec):.2f}")
    print(f"- best completions/s: {max(measured_completions_per_sec):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
