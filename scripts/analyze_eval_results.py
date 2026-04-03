#!/usr/bin/env python3
"""
Post-eval analysis: reads batch_inference output (eval_results.jsonl) and the
original eval dataset, joins on prompt text, runs math_verify accuracy check,
and writes a detailed JSON report + prints a summary.

Usage:
  python scripts/analyze_eval_results.py \
      --eval_results /root/outputs/run_xxx/eval_results.jsonl \
      --eval_dataset /mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl \
      --input_key question --label_key answer \
      [--report_path /root/outputs/run_xxx/eval_analysis.json]
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from math_verify import parse, verify
    from openrlhf.utils.math_verifier import get_llm_answer, verify_llm_answer
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_eval_dataset(path, input_key, label_key):
    """Load gold labels from the eval dataset file or HF-format folder."""
    if path.endswith(".jsonl") or path.endswith(".json"):
        rows = load_jsonl(path)
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset(path, split="test")
            rows = [dict(r) for r in ds]
        except Exception:
            rows = load_jsonl(os.path.join(path, "test.jsonl"))

    lookup = {}
    for r in rows:
        prompt_text = r.get(input_key, "")
        if prompt_text:
            key = prompt_text.strip()[:500]
            lookup[key] = r.get(label_key, "")
    return lookup


def classify_output(model_output, gold_answer):
    """Classify a single (model_output, gold_answer) pair."""
    if not HAS_MATH_VERIFY:
        return None, "no_math_verify", "math_verify not installed"

    if not gold_answer or not gold_answer.strip():
        return None, "missing_gold", "Gold answer is empty"

    gold_boxed = parse(f"\\boxed{{{gold_answer}}}")
    if not gold_boxed:
        gold_boxed = parse(gold_answer)
    if not gold_boxed:
        return None, "unparseable_gold", f"Cannot parse gold: {gold_answer[:80]}"

    if not model_output or not model_output.strip():
        return False, "empty_output", "Model produced no output"

    pred, resp_type = get_llm_answer(model_output)

    if not pred:
        if len(model_output.strip()) < 30:
            return False, "too_short", "Output too short, no parseable answer"
        return False, "no_answer_extracted", "Cannot extract answer from model output"

    try:
        correct = verify(pred, gold_boxed)
    except Exception:
        correct = False

    if correct:
        return True, "correct", ""

    raw_gold = parse(gold_answer)
    if raw_gold:
        try:
            if verify(pred, raw_gold):
                return True, "correct_raw_match", "Matches raw (non-boxed) gold"
        except Exception:
            pass

    out_lower = model_output.lower()
    has_steps = bool(re.search(
        r'step\s*\d|first|then|therefore|thus|hence|so\s+we|let\s+', out_lower))
    has_eq = bool(re.search(r'[=<>]', model_output))
    has_boxed = "\\boxed" in model_output

    if has_boxed:
        return False, "wrong_answer", f"Has \\boxed but answer is wrong (resp_type={resp_type})"
    if has_steps and has_eq:
        return False, "reasoning_incomplete", "Has reasoning steps but wrong/no final answer"
    if has_eq and not has_steps:
        return False, "calculation_error", "Has equations but answer is wrong"

    return False, "no_reasoning", "No recognisable reasoning towards the answer"


def main():
    parser = argparse.ArgumentParser(description="Analyze eval_results.jsonl")
    parser.add_argument("--eval_results", type=str, required=True,
                        help="Path to eval_results.jsonl from batch_inference")
    parser.add_argument("--eval_dataset", type=str, required=True,
                        help="Path to eval dataset (jsonl/json/HF folder)")
    parser.add_argument("--input_key", type=str, default="question")
    parser.add_argument("--label_key", type=str, default="answer")
    parser.add_argument("--report_path", type=str, default=None,
                        help="Output JSON report path (default: eval_analysis.json next to eval_results)")
    args = parser.parse_args()

    if not os.path.isfile(args.eval_results):
        print(f"[ERROR] eval_results not found: {args.eval_results}")
        sys.exit(1)

    if args.report_path is None:
        base_dir = os.path.dirname(args.eval_results)
        args.report_path = os.path.join(base_dir, "eval_analysis.json")

    print("=" * 70)
    print("  Post-Eval Analysis")
    print("=" * 70)

    print(f"\n[1] Loading eval results: {args.eval_results}")
    results = load_jsonl(args.eval_results)
    print(f"    Loaded {len(results)} predictions")

    print(f"\n[2] Loading gold labels: {args.eval_dataset}")
    gold_lookup = load_eval_dataset(args.eval_dataset, args.input_key, args.label_key)
    print(f"    Loaded {len(gold_lookup)} gold entries")

    print(f"\n[3] Matching predictions to gold answers ...")
    matched = 0
    unmatched = 0
    records = []

    for i, r in enumerate(results):
        prompt = r.get("input", "")
        model_output = r.get("output", "")
        key = prompt.strip()[:500]
        gold = gold_lookup.get(key, None)

        if gold is None:
            unmatched += 1
            records.append({
                "idx": i,
                "prompt": prompt[:200],
                "model_output": model_output[:500],
                "gold_answer": None,
                "is_correct": None,
                "category": "unmatched",
                "detail": "Could not find gold answer for this prompt",
            })
            continue

        matched += 1
        correct, category, detail = classify_output(model_output, gold)
        records.append({
            "idx": i,
            "prompt": prompt[:200],
            "model_output": model_output[:500],
            "gold_answer": gold,
            "is_correct": correct,
            "category": category,
            "detail": detail,
        })

    print(f"    Matched: {matched}, Unmatched: {unmatched}")

    evaluated = [r for r in records if r["is_correct"] is not None]
    n_correct = sum(1 for r in evaluated if r["is_correct"])
    n_evaluated = len(evaluated)

    cats = Counter(r["category"] for r in records)

    output_lengths = []
    for r in results:
        out = r.get("output", "")
        output_lengths.append(len(out))
    avg_output_len = sum(output_lengths) / max(1, len(output_lengths))
    empty_outputs = sum(1 for l in output_lengths if l < 5)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total predictions:      {len(results)}")
    print(f"  Matched to gold:        {matched}")
    print(f"  Evaluated (parseable):  {n_evaluated}")
    if n_evaluated > 0:
        acc = n_correct / n_evaluated * 100
        print(f"  Correct:                {n_correct}/{n_evaluated} ({acc:.1f}%)")
    else:
        acc = 0.0
        print(f"  Correct:                N/A (no evaluable samples)")
    print(f"  Avg output length:      {avg_output_len:.0f} chars")
    print(f"  Empty/very short (<5):  {empty_outputs}")

    print(f"\n  Category breakdown:")
    for cat, cnt in cats.most_common():
        pct = cnt / len(records) * 100
        print(f"    {cat:30s}  {cnt:4d}  ({pct:5.1f}%)")

    if any(r["category"] == "too_short" or r["category"] == "empty_output"
           for r in records):
        short_count = sum(1 for r in records
                         if r["category"] in ("too_short", "empty_output"))
        print(f"\n  [WARNING] {short_count} samples had too-short/empty output.")
        print(f"            Consider increasing POST_EVAL_MAX_NEW_TOKENS.")

    print(f"\n  Sample outputs per category:")
    shown = set()
    for cat_name in ["correct", "correct_raw_match", "wrong_answer",
                     "reasoning_incomplete", "calculation_error",
                     "no_answer_extracted", "too_short", "empty_output",
                     "no_reasoning", "unparseable_gold", "unmatched"]:
        for r in records:
            if r["category"] == cat_name and cat_name not in shown:
                shown.add(cat_name)
                print(f"\n    [{cat_name}]")
                print(f"      Q: {r['prompt'][:120]}")
                print(f"      Gold: {str(r['gold_answer'])[:80]}")
                print(f"      Model: {r['model_output'][:200]}")
                break

    summary = {
        "total_predictions": len(results),
        "matched": matched,
        "unmatched": unmatched,
        "evaluated": n_evaluated,
        "correct": n_correct,
        "accuracy_pct": round(acc, 2),
        "avg_output_length_chars": round(avg_output_len, 1),
        "empty_or_very_short": empty_outputs,
        "categories": dict(cats),
        "math_verify_available": HAS_MATH_VERIFY,
    }

    report = {
        "summary": summary,
        "records": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.report_path)), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Full report saved to: {args.report_path}")
    print("=" * 70)

    return 0 if n_evaluated == 0 or acc >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
