#!/usr/bin/env python3
"""Standalone "overall / final" report builder for supplement_2rounds runs.

Why this exists
---------------
G1/G2/G3/baseline/Teacher.sh under scripts/supplement_2rounds/ now finish
with a `build_final_report` step that emits
`eval_analysis_<TAG>_final_<TS>.json` -- the merged stage1 + stage2 retry
result, the canonical "overall" number for the run. Older runs (e.g.
g1_rebase_0427_1553/supplement_logs/) predate that step, so the final
report is missing and people have to look at stage1.json + stage2.json
separately, which is misleading (stage2 is a *subset retry*, not an
independent run).

This script reproduces the same merge logic on the existing per-stage JSON
analyses, so old runs can be brought up to date without regenerating
anything from the GPU.

Usage:
  python scripts/build_final_eval_report.py \\
    --stage1 /path/to/eval_analysis_<TAG>_stage1_<TS>.v2.json \\
    --stage2 /path/to/eval_analysis_<TAG>_stage2_<TS>.v2.json \\
    --retry_meta /path/to/eval_retry_subset_meta_<TAG>_<TS>.json \\
    --out /path/to/eval_analysis_<TAG>_final_<TS>.v2.json

The merge logic is intentionally identical to G1.sh:build_final_report() so
the two paths produce byte-comparable summaries given the same inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter


def _gsi(rec):
    """Coerce a record's source_idx/idx to int, or None when unparseable."""
    v = rec.get("source_idx", rec.get("idx"))
    try:
        return None if v is None else int(v)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--stage1", required=True,
                    help="Path to stage1 analysis JSON (v2 format preferred).")
    ap.add_argument("--stage2", required=True,
                    help="Path to stage2 (retry) analysis JSON.")
    ap.add_argument("--retry_meta", required=True,
                    help="Path to eval_retry_subset_meta_*.json.")
    ap.add_argument("--out", required=True,
                    help="Output path for the merged final report JSON.")
    args = ap.parse_args()

    fr = json.load(open(args.stage1, "r", encoding="utf-8"))
    sr = json.load(open(args.stage2, "r", encoding="utf-8"))
    rm = json.load(open(args.retry_meta, "r", encoding="utf-8"))

    frc = fr.get("records", [])
    src = sr.get("records", [])
    fs = fr.get("summary", {})
    ss = sr.get("summary", {})

    # Step 1: build `final` = copy of stage1 records, then overwrite each
    # retried position with the stage2 record. This is the same priority as
    # G1.sh's build_final_report: stage2 wins when present.
    final = [dict(r) for r in frc]
    retried_idx = set()
    second_by_idx = {}
    for r in src:
        si = _gsi(r)
        if si is None:
            continue
        retried_idx.add(si)
        second_by_idx[si] = r
        if 0 <= si < len(final):
            merged = dict(frc[si])
            merged["first_pass"] = dict(frc[si])
            merged["second_pass"] = dict(r)
            merged["is_correct"] = r.get("is_correct")
            merged["category"] = r.get("category")
            merged["retry_applied"] = True
            final[si] = merged
    for i, rec in enumerate(final):
        rec.setdefault("retry_applied", i in retried_idx)

    # Step 2: aggregate "final" accuracy.
    evaluated = [r for r in final if r.get("is_correct") is not None]
    n_evaluated = len(evaluated)
    n_correct = sum(1 for r in evaluated if r.get("is_correct"))
    accuracy_pct = round(n_correct / n_evaluated * 100, 2) if n_evaluated else 0.0

    # Step 3: retry deltas (how many wrong→correct, how many still wrong).
    n_improved = sum(
        1 for i in retried_idx
        if 0 <= i < len(frc) and 0 <= i < len(final)
        and frc[i].get("is_correct") is not True
        and final[i].get("is_correct") is True
    )
    n_still_wrong = sum(
        1 for i in retried_idx
        if 0 <= i < len(final) and final[i].get("is_correct") is not True
    )

    # Step 4: oracle union (best-of-both-passes upper bound).
    both = s1_only = s2_only = ev_oracle = 0
    for i in range(len(final)):
        fc = frc[i].get("is_correct") if i < len(frc) else None
        sec = second_by_idx.get(i)
        sc = None if sec is None else sec.get("is_correct")
        if fc is not None or sc is not None:
            ev_oracle += 1
        if fc is True and sc is True:
            both += 1
        elif fc is True and sc is not True:
            s1_only += 1
        elif sc is True and fc is not True:
            s2_only += 1
    oracle_correct = both + s1_only + s2_only
    oracle_acc = round(oracle_correct / ev_oracle * 100, 2) if ev_oracle else 0.0

    # Step 5: final category breakdown.
    cats = Counter(r.get("category") for r in final)
    cats_pct = {k: round(v / max(1, len(final)) * 100, 2) for k, v in cats.items()}

    # Step 6: carry over the stage-level diagnostics so a single look at the
    # final.json answers "what was generated, how, and what failure modes
    # dominated" without needing to crack open the stage files.
    n1 = fs.get("total_predictions", 0)
    n2 = ss.get("total_predictions", 0)
    pe1 = fs.get("pure_eos_count", 0) or 0
    pe2 = ss.get("pure_eos_count", 0) or 0
    hc1 = fs.get("hit_max_new_tokens_count", 0) or 0
    hc2 = ss.get("hit_max_new_tokens_count", 0) or 0

    summary = {
        "total_predictions": len(final),
        "evaluated": n_evaluated,
        "correct_final": n_correct,
        "accuracy_pct_final": accuracy_pct,

        "first_pass_correct": fs.get("correct"),
        "first_pass_accuracy_pct": fs.get("accuracy_pct"),

        "second_pass_retry_count": rm.get("retry_count", len(retried_idx)),
        "retry_improved_to_correct": n_improved,
        "retry_still_incorrect": n_still_wrong,

        "oracle_union_evaluated": ev_oracle,
        "oracle_union_correct": oracle_correct,
        "oracle_union_accuracy_pct": oracle_acc,
        "oracle_both_correct": both,
        "oracle_stage1_only_correct": s1_only,
        "oracle_stage2_only_correct": s2_only,

        "stage1_pure_eos": pe1,
        "stage1_pure_eos_pct": round(pe1 / max(1, n1) * 100, 2),
        "stage2_pure_eos": pe2,
        "stage2_pure_eos_pct": round(pe2 / max(1, n2) * 100, 2),
        "stage1_hit_cap": hc1,
        "stage1_hit_cap_pct": round(hc1 / max(1, n1) * 100, 2),
        "stage2_hit_cap": hc2,
        "stage2_hit_cap_pct": round(hc2 / max(1, n2) * 100, 2),
        "stage1_max_new_tokens": fs.get("max_new_tokens"),
        "stage2_max_new_tokens": ss.get("max_new_tokens"),
        "stage1_avg_tokens": fs.get("avg_output_length_tokens"),
        "stage2_avg_tokens": ss.get("avg_output_length_tokens"),

        "final_categories": dict(cats),
        "final_categories_pct": cats_pct,
    }

    out = {
        "summary": summary,
        "first_pass_report_path": args.stage1,
        "second_pass_report_path": args.stage2,
        "retry_metadata": rm,
        "records": final,
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Pretty printout — same shape as the message the human will quote in PRs.
    print("=" * 70)
    print(f"  OVERALL / FINAL MERGED REPORT")
    print(f"  stage1: {args.stage1}")
    print(f"  stage2: {args.stage2}")
    print("=" * 70)
    print(f"  Total predictions:        {summary['total_predictions']}")
    print(f"  Evaluated:                {summary['evaluated']}")
    print()
    print(f"  ── ROUND-1 (stage1, cap={summary['stage1_max_new_tokens']}) ──")
    print(f"    Correct:                {summary['first_pass_correct']}")
    print(f"    Accuracy:               {summary['first_pass_accuracy_pct']}%")
    print(f"    Pure-EOS (immediate):   {summary['stage1_pure_eos']} ({summary['stage1_pure_eos_pct']}%)")
    print(f"    Hit max_new_tokens:     {summary['stage1_hit_cap']} ({summary['stage1_hit_cap_pct']}%)")
    print(f"    Avg output tokens:      {summary['stage1_avg_tokens']}")
    print()
    print(f"  ── ROUND-2 (stage2 retry on {summary['second_pass_retry_count']} wrong samples, cap={summary['stage2_max_new_tokens']}) ──")
    print(f"    Improved wrong→correct: {summary['retry_improved_to_correct']}")
    print(f"    Still wrong after retry:{summary['retry_still_incorrect']}")
    print(f"    Pure-EOS (immediate):   {summary['stage2_pure_eos']} ({summary['stage2_pure_eos_pct']}%)")
    print(f"    Hit max_new_tokens:     {summary['stage2_hit_cap']} ({summary['stage2_hit_cap_pct']}%)")
    print(f"    Avg output tokens:      {summary['stage2_avg_tokens']}")
    print()
    print(f"  ── FINAL (replace stage1 wrong with stage2 result) ──")
    print(f"    Correct:                {summary['correct_final']}")
    print(f"    Accuracy:               {summary['accuracy_pct_final']}%")
    print()
    print(f"  ── ORACLE UNION (best of stage1 ∪ stage2) ──")
    print(f"    Correct:                {summary['oracle_union_correct']}")
    print(f"    Accuracy:               {summary['oracle_union_accuracy_pct']}%")
    print(f"    Both stage1 & stage2:   {summary['oracle_both_correct']}")
    print(f"    Stage1 only correct:    {summary['oracle_stage1_only_correct']}")
    print(f"    Stage2 only correct:    {summary['oracle_stage2_only_correct']}")
    print()
    print(f"  ── FINAL category breakdown ──")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        pct = cats_pct.get(k, 0.0)
        print(f"    {k!s:35s} {v:5d}  ({pct:5.1f}%)")
    print()
    print(f"  Saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
