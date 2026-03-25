#!/usr/bin/env python3
"""Export teacher completion SQLite cache to a HF dataset.

Since the SQLite cache key is a SHA256 hash (cannot be reversed to get the
original question text), this script re-loads the original HF dataset,
iterates over unique questions, looks up each one in the cache using the
same hashing logic, and writes matches to a new HF dataset.

The output dataset has columns:
    question (str): the original question text
    teacher_completions (List[str]): M teacher completion strings

Usage:
    python scripts/export_teacher_cache_to_dataset.py \
        --prompt_data /mnt/data/data/aops/aops_qa_hf_dict \
        --input_key question --split train \
        --cache_dir /mnt/data/data/aops/teacher_cache_n_samples_2 \
        --model_name qwen-122b \
        --n_samples 2 --temperature 0.7 --top_p 0.95 --max_new_tokens 512 \
        --output_dir /mnt/data/data/aops/teacher_dataset_n_samples_2
"""

import argparse
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from openrlhf.utils.teacher_provider import TeacherCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Export teacher cache to HF dataset")

    p.add_argument("--prompt_data", type=str, required=True,
                   help="Path to original HF dataset (load_from_disk)")
    p.add_argument("--input_key", type=str, default="question",
                   help="Column name for the question text")
    p.add_argument("--split", type=str, default="train",
                   help="Dataset split to use")

    p.add_argument("--cache_dir", type=str, required=True,
                   help="SQLite cache directory containing teacher_cache.db")

    p.add_argument("--model_name", type=str, required=True,
                   help="Model name used during warmup (for cache key)")
    p.add_argument("--n_samples", type=int, required=True,
                   help="n_samples used during warmup (for cache key)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=512)

    p.add_argument("--output_dir", type=str, required=True,
                   help="Output path for the HF dataset (save_to_disk)")

    return p.parse_args()


def load_unique_questions(prompt_data: str, split: str, input_key: str):
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

    all_questions = ds[input_key]
    unique = sorted(set(all_questions))
    logger.info("Total rows: %d, unique questions: %d", len(all_questions), len(unique))
    return unique


def main():
    args = parse_args()

    questions = load_unique_questions(args.prompt_data, args.split, args.input_key)
    if not questions:
        logger.warning("No questions found -- nothing to export.")
        return

    cache = TeacherCache(args.cache_dir)
    logger.info("Cache DB: %s", cache.db_path)

    rows = []
    missing = 0
    for q in questions:
        completions = cache.get(
            q, args.model_name, args.n_samples,
            args.temperature, args.top_p, args.max_new_tokens,
        )
        if completions is not None:
            rows.append({"question": q, "teacher_completions": completions})
        else:
            missing += 1

    logger.info("Exported: %d, missing in cache: %d, total unique: %d",
                len(rows), missing, len(questions))

    if not rows:
        logger.error("No completions found in cache -- nothing to export. "
                     "Check that model_name/n_samples/temperature/top_p/max_new_tokens "
                     "match the warmup parameters.")
        sys.exit(1)

    if missing > 0:
        logger.warning("%d questions had no cached completions and were skipped.", missing)

    from datasets import Dataset
    out_ds = Dataset.from_list(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    out_ds.save_to_disk(args.output_dir)
    logger.info("HF dataset saved to %s  (%d rows)", args.output_dir, len(out_ds))


if __name__ == "__main__":
    main()
