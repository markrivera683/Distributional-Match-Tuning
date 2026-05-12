#!/usr/bin/env python3
"""Prepare OpenCodeInstruct train and code eval JSONL files for EBFT scripts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return count


def prepare_train(source: str, output: Path, sample_count: int, force: bool) -> None:
    existing = jsonl_count(output)
    if existing >= sample_count and not force:
        print(f"[prepare] train exists: {output} ({existing} rows)")
        return

    from datasets import load_dataset

    ds = load_dataset(source, split=f"train[:{sample_count}]")

    def rows():
        for idx, row in enumerate(ds):
            question = str(row.get("input", "")).strip()
            answer = str(row.get("output", "")).strip()
            if not question or not answer:
                continue
            yield {
                "source_idx": idx,
                "id": row.get("id", idx),
                "question": question,
                "answer": answer,
                "input": question,
                "output": answer,
                "domain": row.get("domain", ""),
            }

    count = write_jsonl(output, rows())
    print(f"[prepare] train wrote: {output} ({count} rows)")


def prepare_mbpp(source: str, output: Path, force: bool) -> None:
    if output.exists() and not force:
        print(f"[prepare] mbpp exists: {output} ({jsonl_count(output)} rows)")
        return

    source_path = Path(source)
    if source_path.is_dir():
        source_path = source_path / "data" / "mbpp.jsonl"

    def rows():
        with source_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                tests = row.get("test_list") or []
                prompt = row.get("text", "")
                question = (
                    f"{prompt}\n\n"
                    "Write the Python solution. The solution should satisfy these tests:\n"
                    + "\n".join(str(t) for t in tests)
                ).strip()
                yield {
                    "source_idx": idx,
                    "dataset": "mbpp",
                    "task_id": row.get("task_id", idx),
                    "question": question,
                    "answer": row.get("code", ""),
                    "tests": tests,
                }

    count = write_jsonl(output, rows())
    print(f"[prepare] mbpp wrote: {output} ({count} rows)")


def prepare_humaneval(source: str, output: Path, force: bool) -> None:
    if output.exists() and not force:
        print(f"[prepare] humaneval exists: {output} ({jsonl_count(output)} rows)")
        return

    from datasets import load_dataset

    ds = load_dataset(source, split="test")

    def rows():
        for idx, row in enumerate(ds):
            yield {
                "source_idx": idx,
                "dataset": "humaneval",
                "task_id": row.get("task_id", idx),
                "question": row.get("prompt", ""),
                "answer": row.get("canonical_solution", ""),
                "test": row.get("test", ""),
                "entry_point": row.get("entry_point", ""),
            }

    count = write_jsonl(output, rows())
    print(f"[prepare] humaneval wrote: {output} ({count} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/mnt/data/ebft-distribution-new/outputs/diff_dataset_prepared")
    parser.add_argument("--train-source", default="/mnt/data/OpenCodeInstruct")
    parser.add_argument("--mbpp-source", default="/mnt/data/mbpp")
    parser.add_argument("--humaneval-source", default="/mnt/data/humaneval")
    parser.add_argument("--train-samples", type=int, default=100000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    prepare_train(args.train_source, out / "opencodeinstruct_qa_100k.jsonl", args.train_samples, args.force)
    prepare_mbpp(args.mbpp_source, out / "mbpp_eval_qa.jsonl", args.force)
    prepare_humaneval(args.humaneval_source, out / "humaneval_eval_qa.jsonl", args.force)

    manifest = out / "manifest.env"
    manifest.write_text(
        "\n".join(
            [
                f"TRAIN_DATA={out / 'opencodeinstruct_qa_100k.jsonl'}",
                f"MBPP_EVAL_DATA={out / 'mbpp_eval_qa.jsonl'}",
                f"HUMANEVAL_EVAL_DATA={out / 'humaneval_eval_qa.jsonl'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[prepare] manifest: {manifest}")


if __name__ == "__main__":
    main()
