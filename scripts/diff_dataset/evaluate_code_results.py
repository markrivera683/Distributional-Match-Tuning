#!/usr/bin/env python3
"""Execute HumanEval/MBPP unit tests against generated JSONL completions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_benchmark_helpers(repo_root: Path):
    helper_path = repo_root / "scripts" / "benchmarks" / "run_code_generation_benchmarks.py"
    spec = importlib.util.spec_from_file_location("code_bench_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_def_signature(code: str) -> tuple[str | None, str | None]:
    for line in str(code or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            return stripped[4:].split("(", 1)[0].strip(), stripped
    return None, None


def evaluate_humaneval(source_rows, outputs_by_idx, helpers, timeout: int):
    details = []
    correct = 0
    for idx, row in enumerate(source_rows):
        output_row = outputs_by_idx.get(idx)
        if output_row is None:
            ok, err, completion = False, "missing_output", ""
        else:
            completion = output_row.get("output", "")
            ok, err = helpers.evaluate_humaneval_completion(
                row.get("question", ""),
                completion,
                row.get("test", ""),
                row.get("entry_point"),
                timeout,
            )
        correct += int(ok)
        details.append(
            {
                "benchmark": "HumanEval",
                "source_idx": idx,
                "task_id": row.get("task_id", idx),
                "is_correct": bool(ok),
                "error_type": err,
                "output_chars": len(completion or ""),
            }
        )
    return correct, details


def evaluate_mbpp(source_rows, outputs_by_idx, helpers, timeout: int):
    details = []
    correct = 0
    for idx, row in enumerate(source_rows):
        output_row = outputs_by_idx.get(idx)
        function_name, function_signature = first_def_signature(row.get("answer", ""))
        request = {
            "prompt_for_model": row.get("question", ""),
            "function_name": function_name,
            "function_signature": function_signature,
            "helper_code": "",
            "unit_tests": row.get("tests") or [],
        }
        if output_row is None:
            ok, err, completion = False, "missing_output", ""
        else:
            completion = output_row.get("output", "")
            ok, err = helpers.evaluate_mbpp_completion(request, completion, timeout)
        correct += int(ok)
        details.append(
            {
                "benchmark": "MBPP",
                "source_idx": idx,
                "task_id": row.get("task_id", idx),
                "is_correct": bool(ok),
                "error_type": err,
                "output_chars": len(completion or ""),
            }
        )
    return correct, details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--source_jsonl", required=True)
    parser.add_argument("--results_jsonl", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--details_jsonl", required=True)
    parser.add_argument("--timeout_seconds", type=int, default=10)
    parser.add_argument("--repo_root", default="/mnt/data/ebft-distribution-new/code")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    source_rows = load_jsonl(Path(args.source_jsonl))
    result_rows = load_jsonl(Path(args.results_jsonl))
    outputs_by_idx = {}
    for row in result_rows:
        source_idx = int(row.get("source_idx", row.get("idx", len(outputs_by_idx))))
        outputs_by_idx.setdefault(source_idx, row)

    helpers = load_benchmark_helpers(Path(args.repo_root))
    if args.benchmark == "humaneval":
        correct, details = evaluate_humaneval(source_rows, outputs_by_idx, helpers, args.timeout_seconds)
        benchmark_name = "HumanEval"
    else:
        correct, details = evaluate_mbpp(source_rows, outputs_by_idx, helpers, args.timeout_seconds)
        benchmark_name = "MBPP"

    total = len(source_rows)
    summary = {
        "benchmark": benchmark_name,
        "total": total,
        "result_rows": len(result_rows),
        "matched_outputs": len(outputs_by_idx),
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "accuracy_pct": round(correct / total * 100, 2) if total else 0.0,
        "temperature": args.temperature,
        "n_samples": 1,
        "max_new_tokens": args.max_new_tokens,
        "timeout_seconds": args.timeout_seconds,
        "error_counts": dict(Counter(d["error_type"] or "ok" for d in details)),
        "source_jsonl": args.source_jsonl,
        "results_jsonl": args.results_jsonl,
    }

    report_path = Path(args.report_json)
    details_path = Path(args.details_jsonl)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"summary": summary, "details_path": str(details_path)}, indent=2) + "\n")
    with details_path.open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
