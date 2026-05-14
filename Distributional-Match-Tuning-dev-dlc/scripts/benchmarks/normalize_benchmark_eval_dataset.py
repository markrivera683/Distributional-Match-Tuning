#!/usr/bin/env python3
"""
Normalize local benchmark datasets into a common JSONL schema.

Output schema:
  {"question": <prompt_text>, "answer": <gold_answer>, "benchmark": <name>, "source_id": <id>}

This lets batch_inference and analyze_eval_results.py share one stable
input_key/label_key pair across all local benchmark folders.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset


def infer_benchmark_name(dataset_path: str, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    return Path(dataset_path.rstrip("/")).name


def extract_last_boxed(solution: str) -> str | None:
    marker = r"\boxed{"
    start = solution.rfind(marker)
    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    chars: list[str] = []

    while i < len(solution):
        ch = solution[i]
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(ch)
        else:
            chars.append(ch)
        i += 1

    return None


def normalize_answer(row: dict[str, Any], benchmark_name: str) -> str:
    if "answer" in row and row["answer"] is not None:
        return str(row["answer"]).strip()

    if "final_answer" in row and row["final_answer"] is not None:
        final_answer = row["final_answer"]
        if isinstance(final_answer, list):
            if len(final_answer) != 1:
                raise ValueError(
                    f"{benchmark_name}: expected single final_answer item, got {len(final_answer)} for row {row.get('id')}"
                )
            final_answer = final_answer[0]
        return str(final_answer).strip()

    if "solution" in row and row["solution"] is not None and "problem" in row:
        boxed = extract_last_boxed(str(row["solution"]))
        if boxed is None:
            raise ValueError(
                f"{benchmark_name}: could not extract boxed answer from solution for row {row.get('id')}"
            )
        return boxed

    raise ValueError(
        f"{benchmark_name}: unsupported answer schema; available keys={sorted(row.keys())}"
    )


def normalize_question(row: dict[str, Any], benchmark_name: str) -> str:
    if "question" in row and row["question"] is not None:
        return str(row["question"]).strip()

    if "problem" in row and row["problem"] is not None:
        return str(row["problem"]).strip()

    raise ValueError(
        f"{benchmark_name}: unsupported prompt schema; available keys={sorted(row.keys())}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize a local benchmark dataset into question/answer JSONL.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Local benchmark dataset folder")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--benchmark_name", type=str, default=None, help="Optional benchmark name override")
    args = parser.parse_args()

    benchmark_name = infer_benchmark_name(args.dataset_path, args.benchmark_name)

    ds = load_dataset(args.dataset_path, split="test")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    normalized_rows = []
    for idx, row in enumerate(ds):
        question = normalize_question(row, benchmark_name)
        answer = normalize_answer(row, benchmark_name)
        if not question:
            raise ValueError(f"{benchmark_name}: empty normalized question at idx={idx}")
        if not answer:
            raise ValueError(f"{benchmark_name}: empty normalized answer at idx={idx}")

        source_id = row.get("id", row.get("unique_id", idx))
        normalized_rows.append(
            {
                "question": question,
                "answer": answer,
                "benchmark": benchmark_name,
                "source_id": source_id,
            }
        )

    with open(args.output_path, "w", encoding="utf-8") as f:
        for row in normalized_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sample = normalized_rows[0] if normalized_rows else {}
    print(
        f"[normalize] benchmark={benchmark_name} rows={len(normalized_rows)} "
        f"output={args.output_path}"
    )
    if sample:
        print(
            "[normalize] sample: "
            f"question={sample['question'][:120]!r} answer={sample['answer'][:80]!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
