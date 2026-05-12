#!/usr/bin/env python3
"""Token-level distill-quality evaluator for ebft-trained student models.

Why this script
---------------
ebft trains the student via next-token / sliding-window feature-matching loss
on packed (Q + A) streams. There is **no** supervision teaching the model to
emit `\\boxed{...}`, follow chat templates, or commit to "Final Answer:"
markers. So `correct%` on AOPS test_qa.jsonl (which requires the model to
sample a complete answer ending in `\\boxed{X}`) systematically *under*-
measures distill quality: the model can match the teacher's distribution
beautifully and still score near zero on that metric (the
g1_rebase_0427_1553 run measured 3-6% correct yet "committed to an answer"
in 60-80% of cases).

This script measures distill quality with metrics that are *aligned with the
training objective*:

  - **Mean answer-token NLL** (and perplexity = exp(NLL)). Lower = student
    matches the gold continuation distribution better. This is what gets
    optimized (modulo feature-map shenanigans) at every ebft step.
  - **Top-k accuracy at answer positions** for k in {1, 5, 10}: fraction of
    gold answer tokens that fall in the student's top-k. Robust to the gold
    being one of several reasonable continuations.
  - **NLL by position-in-answer quintile** (q1=first 20%, q5=last 20%): if
    the student's distill quality decays with answer depth, the q5 NLL will
    be noticeably worse than q1's. Useful when the training packed answers
    aggressively and only the head of each (Q + A) saw clean training signal.

These are all student-side metrics (no teacher needed at eval time), so this
runs on a single A100 and finishes in minutes for a 4B model.

Usage
-----
    python scripts/eval_distill_quality.py \\
        --pretrain /mnt/data/ebft-distribution-new/outputs/g1_rebase_0427_1553/model \\
        --dataset /mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl \\
        --input_key question --label_key solution \\
        --max_samples 500 --max_seq_len 2048 \\
        --report_path /mnt/data/ebft-distribution-new/outputs/g1_rebase_0427_1553/distill_quality.json

Compare across runs by pointing --pretrain at G1, G2, G3, baseline, Teacher
in turn (with the same dataset / max_samples / max_seq_len) and stacking the
resulting JSON reports.

Notes
-----
- We default to `label_key=solution` (long teacher reasoning) rather than the
  short `answer` field, because the long solution is what the student was
  actually trained to predict in the packed Q+A stream.
- We deliberately do NOT apply a chat template; G1/G2/G3 were all trained
  with `--no_chat_template --pretrain_mode`, so the inference-time format
  must match (raw Q + raw A concatenation).
- Tokens beyond `max_seq_len` are truncated tail-first (drop end of answer
  rather than question), and samples whose answer ends up <2 tokens are
  skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_rows(path: str, max_samples: int) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pretrain", required=True,
                        help="Path to HF model checkpoint (the run's `model/` dir).")
    parser.add_argument("--dataset", required=True,
                        help="Path to test_qa.jsonl (or any jsonl with question/solution).")
    parser.add_argument("--input_key", default="question")
    parser.add_argument("--label_key", default="solution",
                        help="Key whose value is the gold continuation. "
                             "Default 'solution' (long reasoning) rather than 'answer' (short).")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Cap on # of test rows to evaluate (0 = all).")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="Truncate (Q+A) tokens to this length (drop tail).")
    parser.add_argument("--report_path", required=True,
                        help="Output JSON path for the distill_quality summary.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--top_ks", default="1,5,10",
                        help="Comma-separated top-k values to evaluate.")
    parser.add_argument("--num_quintiles", type=int, default=5,
                        help="# of position buckets for per-position NLL stratification.")
    parser.add_argument("--micro_batch_size", type=int, default=1,
                        help="# of samples per forward pass. 1 is safest for varied lengths; "
                             "raise if all your sequences are about the same size.")
    args = parser.parse_args()

    top_ks = sorted({int(k.strip()) for k in args.top_ks.split(",") if k.strip()})
    max_topk = max(top_ks) if top_ks else 1
    n_buckets = max(1, args.num_quintiles)

    # Lazy-import heavy deps after argparse so --help is fast and import errors
    # surface only when actually needed.
    from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: WPS433

    print("=" * 70)
    print("  Distill-Quality Eval")
    print("=" * 70)
    print(f"  Model:          {args.pretrain}")
    print(f"  Dataset:        {args.dataset}")
    print(f"  input_key:      {args.input_key}")
    print(f"  label_key:      {args.label_key}")
    print(f"  max_samples:    {args.max_samples}")
    print(f"  max_seq_len:    {args.max_seq_len}")
    print(f"  device:         {args.device}")
    print(f"  dtype:          {args.torch_dtype}")
    print(f"  top_ks:         {top_ks}")

    rows = load_rows(args.dataset, args.max_samples)
    print(f"\n[1] Loaded {len(rows)} rows")

    print(f"\n[2] Loading tokenizer + model ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]
    # Many ebft runs save with custom configs; use trust_remote_code so
    # student-specific architectures (Gemma-4 E4B etc.) load cleanly.
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrain,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(args.device).eval()
    print(f"    Loaded in {time.time() - t0:.1f}s")

    # Aggregators
    nll_sum = 0.0
    nll_count = 0
    topk_correct = {k: 0 for k in top_ks}
    bucket_nll_sum = [0.0] * n_buckets
    bucket_nll_count = [0] * n_buckets
    n_evaluated = 0
    n_skipped = 0
    skipped_reasons: dict = {}

    # Per-sample records (small footprint: NLL + counts only) so the user can
    # post-hoc bucket by question difficulty / answer length / etc.
    per_sample: List[dict] = []

    print(f"\n[3] Running forward passes ...")

    pbar = tqdm(rows, desc="distill-eval", dynamic_ncols=True)
    for ridx, r in enumerate(pbar):
        q = (r.get(args.input_key) or "").strip()
        a = (r.get(args.label_key) or "").strip()
        if not q or not a:
            n_skipped += 1
            skipped_reasons["empty_q_or_a"] = skipped_reasons.get("empty_q_or_a", 0) + 1
            continue

        # Tokenize without special tokens — ebft training uses bare Q+A
        # concatenation joined by <eos>. We don't insert <eos> here because
        # we're scoring the model's prediction of A given Q (no separator
        # was conditioned on at training time inside a packed chunk).
        q_ids = tokenizer.encode(q, add_special_tokens=False)
        a_ids = tokenizer.encode(a, add_special_tokens=False)

        if len(q_ids) >= args.max_seq_len - 1:
            n_skipped += 1
            skipped_reasons["q_too_long"] = skipped_reasons.get("q_too_long", 0) + 1
            continue

        full_ids = q_ids + a_ids
        if len(full_ids) > args.max_seq_len:
            full_ids = full_ids[: args.max_seq_len]
            a_ids = full_ids[len(q_ids):]

        if len(a_ids) < 2:
            n_skipped += 1
            skipped_reasons["a_too_short"] = skipped_reasons.get("a_too_short", 0) + 1
            continue

        input_ids = torch.tensor(full_ids, dtype=torch.long, device=args.device).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids).logits[0]  # (S, V)
        # Token at position t is predicted by logits at position t-1.
        # Answer positions are q_len .. q_len + a_len - 1.
        # Predicting positions: logits[q_len-1 : q_len-1 + a_len].
        q_len = len(q_ids)
        a_len = len(a_ids)
        pred_logits = logits[q_len - 1 : q_len - 1 + a_len]  # (a_len, V)
        target = torch.tensor(a_ids, dtype=torch.long, device=args.device)

        # Cross-entropy in float32 to avoid bfloat16 precision blowup on
        # large-V softmaxes (Gemma-4 vocab is 262144).
        log_probs = F.log_softmax(pred_logits.float(), dim=-1)
        nll_per_pos = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (a_len,)
        sample_nll = nll_per_pos.sum().item()

        # Top-k accuracy: take top max_topk indices once, slice for each k.
        top_idx = pred_logits.topk(max_topk, dim=-1).indices  # (a_len, max_topk)
        eq_mat = (top_idx == target.unsqueeze(-1))
        sample_topk_hits = {}
        for k in top_ks:
            hits = eq_mat[:, :k].any(dim=-1).sum().item()
            topk_correct[k] += hits
            sample_topk_hits[k] = hits

        # Position-bucket NLL (uniform binning over a_len).
        # bucket_idx = floor(i / a_len * n_buckets), clamped to [0, n_buckets-1].
        for i, nll in enumerate(nll_per_pos.tolist()):
            b = min(int(i / a_len * n_buckets), n_buckets - 1)
            bucket_nll_sum[b] += nll
            bucket_nll_count[b] += 1

        nll_sum += sample_nll
        nll_count += a_len
        n_evaluated += 1

        per_sample.append({
            "row_idx": ridx,
            "q_tokens": len(q_ids),
            "a_tokens": a_len,
            "nll_sum": sample_nll,
            "nll_mean": sample_nll / a_len,
            **{f"top{k}_hits": sample_topk_hits[k] for k in top_ks},
        })

        # Light-touch progress summary in postfix.
        if (ridx + 1) % 10 == 0:
            running = nll_sum / max(1, nll_count)
            pbar.set_postfix(
                evaluated=n_evaluated,
                skipped=n_skipped,
                ppl=f"{math.exp(min(running, 30)):.2f}",
            )

    if n_evaluated == 0:
        print("[ERROR] No samples were evaluated. Check --dataset / --label_key / --max_seq_len.")
        return 1

    mean_nll = nll_sum / nll_count
    summary = {
        "model_path": args.pretrain,
        "dataset_path": args.dataset,
        "input_key": args.input_key,
        "label_key": args.label_key,
        "max_seq_len": args.max_seq_len,
        "torch_dtype": args.torch_dtype,
        "n_rows_loaded": len(rows),
        "n_evaluated": n_evaluated,
        "n_skipped": n_skipped,
        "skipped_reasons": skipped_reasons,
        "n_answer_tokens_total": nll_count,
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 30.0)),
        "topk_accuracy": {
            f"top{k}": topk_correct[k] / nll_count for k in top_ks
        },
        "nll_by_position_bucket": [
            {
                "bucket": b + 1,
                "n_tokens": bucket_nll_count[b],
                "mean_nll": (
                    bucket_nll_sum[b] / bucket_nll_count[b]
                    if bucket_nll_count[b] > 0 else None
                ),
            }
            for b in range(n_buckets)
        ],
    }

    print(f"\n{'=' * 70}")
    print(f"  Distill-Quality Summary")
    print(f"{'=' * 70}")
    print(f"  Evaluated:              {n_evaluated} / {len(rows)} rows  (skipped: {n_skipped})")
    print(f"  Answer tokens scored:   {nll_count}")
    print(f"  Mean answer-token NLL:  {mean_nll:.4f}")
    print(f"  Perplexity:             {summary['perplexity']:.3f}")
    print(f"  Top-k accuracy:")
    for k in top_ks:
        acc = topk_correct[k] / nll_count
        print(f"    top-{k:<3d}              {acc * 100:.2f}%   ({topk_correct[k]}/{nll_count})")
    print(f"  NLL by answer-position bucket:")
    for b in range(n_buckets):
        n = bucket_nll_count[b]
        if n == 0:
            print(f"    q{b + 1}: (no tokens)")
            continue
        bnll = bucket_nll_sum[b] / n
        print(f"    q{b + 1} ({n:>6d} tokens):  NLL={bnll:.4f}  PPL={math.exp(min(bnll, 30)):.2f}")
    if skipped_reasons:
        print(f"  Skipped reasons:        {skipped_reasons}")

    out_dir = os.path.dirname(os.path.abspath(args.report_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_sample": per_sample}, f, indent=2)
    print(f"\n  Full report saved to: {args.report_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
