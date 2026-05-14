#!/usr/bin/env python3
"""Pre-fill the teacher completion SQLite cache before training.

Loads the HF dataset, extracts unique questions, and calls the remote
teacher API to populate the cache.  Supports resume: already-cached
prompts are skipped automatically.

No GPU or Ray required -- runs on CPU only.

Usage:
    python scripts/warmup_teacher_cache.py \
        --prompt_data /mnt/data/data/aops/aops_qa_hf_dict \
        --input_key question --split train \
        --cache_dir /root/outputs/teacher_cache_shared \
        --teacher_api_base http://172.17.0.26:8000/v1 \
        --teacher_model_name qwen-122b \
        --n_samples 2 --temperature 0.7 --top_p 0.95 --max_new_tokens 512
"""

import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from openrlhf.utils.teacher_provider import RemoteTeacherProvider, TeacherCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Teacher cache warmup")

    p.add_argument("--prompt_data", type=str, required=True,
                   help="Path to HF dataset (load_from_disk)")
    p.add_argument("--input_key", type=str, default="question",
                   help="Column name for the question text")
    p.add_argument("--split", type=str, default="train",
                   help="Dataset split to use")

    p.add_argument("--cache_dir", type=str, required=True,
                   help="SQLite cache directory (same as training --teacher_cache_dir)")

    p.add_argument("--teacher_api_base", type=str, required=True)
    p.add_argument("--teacher_model_name", type=str, required=True)
    p.add_argument("--teacher_api_key", type=str, default="EMPTY")
    p.add_argument("--teacher_api_style", type=str, default="completions",
                   choices=["completions", "chat_completions"])

    p.add_argument("--n_samples", type=int, default=2, help="M: completions per question")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=512)

    p.add_argument("--max_samples", type=int, default=0,
                   help="Limit unique questions to warmup (0 = all)")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Concurrent HTTP requests")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--max_retries", type=int, default=3)

    p.add_argument("--system_prompt_text", type=str, default="",
                   help="System prompt / prefix text injected into every teacher request. "
                        "Must be identical to SYSTEM_PROMPT_TEXT used during training.")
    p.add_argument("--system_prompt_id", type=str, default="",
                   help="Short opaque ID embedded in cache keys (e.g. 'v1-balanced'). "
                        "Must match training. Changing this invalidates existing cache entries.")

    return p.parse_args()


def load_unique_questions(prompt_data: str, split: str, input_key: str,
                          max_samples: int = 0):
    from datasets import load_from_disk, DatasetDict

    logger.info("Loading dataset from %s ...", prompt_data)
    ds = load_from_disk(prompt_data)

    if isinstance(ds, DatasetDict):
        if split in ds:
            ds = ds[split]
        elif "train" in ds:
            logger.warning("Split %r not found, falling back to 'train'", split)
            ds = ds["train"]
        else:
            raise ValueError(f"Split {split!r} not in dataset: {list(ds.keys())}")

    if max_samples > 0 and max_samples < len(ds):
        logger.info("Limiting dataset to first %d rows (max_samples)", max_samples)
        ds = ds.select(range(max_samples))

    all_questions = ds[input_key]
    unique = sorted(set(all_questions))
    logger.info("Total rows: %d, unique questions: %d", len(all_questions), len(unique))
    return unique


def count_cached(cache, questions, model_name, n_samples, temperature, top_p, max_new_tokens,
                  api_style="completions", system_prompt_id=""):
    """Count how many questions are already in the cache."""
    cached = 0
    for q in questions:
        if cache.get(q, model_name, n_samples, temperature, top_p, max_new_tokens,
                     api_style=api_style, system_prompt_id=system_prompt_id) is not None:
            cached += 1
    return cached


def main():
    args = parse_args()

    questions = load_unique_questions(args.prompt_data, args.split, args.input_key,
                                      args.max_samples)
    total = len(questions)

    if total == 0:
        logger.warning("No questions found -- nothing to do.")
        return

    cache = TeacherCache(args.cache_dir)
    logger.info("Cache dir: %s", args.cache_dir)

    uncached = []
    pre_cached = 0
    for q in questions:
        if cache.get(q, args.teacher_model_name, args.n_samples,
                     args.temperature, args.top_p, args.max_new_tokens,
                     api_style=args.teacher_api_style,
                     system_prompt_id=args.system_prompt_id) is not None:
            pre_cached += 1
        else:
            uncached.append(q)

    logger.info("Already cached: %d / %d (%.1f%%)", pre_cached, total, 100.0 * pre_cached / total)

    if not uncached:
        logger.info("All questions already cached. Nothing to do.")
        return

    logger.info("Questions to fetch: %d", len(uncached))

    provider = RemoteTeacherProvider(
        api_base=args.teacher_api_base,
        model_name=args.teacher_model_name,
        api_key=args.teacher_api_key,
        api_style=args.teacher_api_style,
        timeout=args.timeout,
        max_retries=args.max_retries,
        batch_size=args.batch_size,
        cache=cache,
        system_prompt_text=args.system_prompt_text,
        system_prompt_id=args.system_prompt_id,
    )
    if args.system_prompt_id:
        logger.info("System prompt: id=%r, text=%r", args.system_prompt_id,
                    args.system_prompt_text[:80] + ("..." if len(args.system_prompt_text) > 80 else ""))

    bs = args.batch_size
    n_batches = (len(uncached) + bs - 1) // bs
    failed = []
    fetched = 0
    t0 = time.time()

    for batch_idx in range(n_batches):
        batch = uncached[batch_idx * bs : (batch_idx + 1) * bs]
        try:
            provider.sample_targets(
                batch, args.n_samples,
                args.temperature, args.top_p, args.max_new_tokens,
            )
            fetched += len(batch)
        except Exception as e:
            logger.error("Batch %d/%d failed: %s", batch_idx + 1, n_batches, e)
            for q in batch:
                try:
                    provider.sample_targets(
                        [q], args.n_samples,
                        args.temperature, args.top_p, args.max_new_tokens,
                    )
                    fetched += 1
                except Exception as e2:
                    logger.error("Single prompt failed (len=%d): %s", len(q), e2)
                    failed.append(q)

        elapsed = time.time() - t0
        done = batch_idx + 1
        rate = elapsed / done if done else 0
        eta = rate * (n_batches - done)
        logger.info(
            "Progress: %d/%d batches | %d/%d fetched | %.0fs elapsed | ETA %.0fs",
            done, n_batches, fetched, len(uncached), elapsed, eta,
        )

    elapsed_total = time.time() - t0

    post_cached = count_cached(
        cache, questions, args.teacher_model_name,
        args.n_samples, args.temperature, args.top_p, args.max_new_tokens,
        api_style=args.teacher_api_style,
        system_prompt_id=args.system_prompt_id,
    )

    logger.info("=" * 60)
    logger.info("Warmup complete")
    logger.info("  Total unique questions: %d", total)
    logger.info("  Previously cached:      %d", pre_cached)
    logger.info("  Fetched this run:       %d", fetched)
    logger.info("  Failed:                 %d", len(failed))
    logger.info("  Final cache coverage:   %d / %d (%.1f%%)",
                post_cached, total, 100.0 * post_cached / total)
    logger.info("  Elapsed:                %.1fs", elapsed_total)
    logger.info("  Cache DB:               %s",
                os.path.join(args.cache_dir, "teacher_cache.db"))
    logger.info("=" * 60)

    if failed:
        fail_path = os.path.join(args.cache_dir, "warmup_failed.txt")
        with open(fail_path, "w") as f:
            for q in failed:
                f.write(q + "\n")
        logger.warning("Failed prompts saved to %s", fail_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
