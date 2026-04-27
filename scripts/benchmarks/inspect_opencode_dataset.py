#!/usr/bin/env python3
"""Inspect the `sjelassi/opencode-instruct_100k_200tok` dataset used by EBFT
training.

Goals (driven by reward-vs-HumanEval analysis):

  1. Tokenize every answer with the actor tokenizer (Qwen2.5-1.5B) and report
     the length distribution (min / quartiles / 95p / 99p / max), plus the
     fraction of answers whose length pile up around the suspected 200-token
     cap.

  2. Print ~30 random (question, answer) samples so we can eyeball whether the
     answers are:
       - cut off mid-code,
       - mostly explanation text instead of pure code,
       - ending with `<|endoftext|>` / obvious truncation markers.

Outputs:
  outputs/dataset_inspection/opencode-instruct_100k_200tok/length_stats.json
  outputs/dataset_inspection/opencode-instruct_100k_200tok/length_hist.png
  outputs/dataset_inspection/opencode-instruct_100k_200tok/samples.md
  outputs/dataset_inspection/opencode-instruct_100k_200tok/report.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path("/root/code/Distributional-Match-Tuning")
DEFAULT_OUT = REPO_ROOT / "outputs/dataset_inspection/opencode-instruct_100k_200tok"

# Heuristic markers that suggest the answer was cut mid-stream.
TRUNC_PATTERNS = [
    re.compile(r"<\|endoftext\|>"),
    re.compile(r"<\|im_end\|>"),
    re.compile(r"<\|EOS\|>", re.IGNORECASE),
    re.compile(r"\\.\\.\\.$"),
    re.compile(r"\bTODO\b", re.IGNORECASE),
]

# Detect "code-y" tail: the last non-empty line ends with a token that suggests
# code is unfinished (open paren / colon / comma / backslash / unclosed string).
UNFINISHED_TAIL = re.compile(r"[\(\[\{,:\\]\s*$|`{1,2}[^`]*$")


def _split_tail(text: str, n_chars: int = 80) -> str:
    return text[-n_chars:].replace("\n", "\\n")


def detect_truncation(answer: str) -> Dict[str, Any]:
    """Return per-sample truncation signals."""
    tail = answer.rstrip()
    flags = {
        "has_special_token": any(p.search(answer) for p in TRUNC_PATTERNS),
        "ends_unfinished_code": bool(UNFINISHED_TAIL.search(tail)),
        # crude check: does the answer end with a closed function/class/return?
        "ends_with_period_or_close": bool(re.search(r"[\)\]\}\.\"\']\s*$", tail))
        and not tail.endswith(("(", "{", "[", ",", ":", "\\")),
        # explanation-heavy: starts/ends with English prose lines?
        "tail_is_prose": bool(re.search(r"[A-Za-z]\s+[A-Za-z]+\s+[A-Za-z]+[\.\!\?]\s*$", tail)),
    }
    flags["tail"] = _split_tail(tail)
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sjelassi/opencode-instruct_100k_200tok")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Cap the number of rows we tokenize per split (5k is enough for a stable distribution).",
    )
    parser.add_argument("--n-print", type=int, default=30, help="How many samples to dump as markdown.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--tokenizer", default="/root/model")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"[inspect] loading tokenizer from {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print(f"[inspect] loading dataset {args.dataset}")
    full = load_dataset(args.dataset)
    print(f"[inspect] splits available: {list(full.keys())}")

    rng = random.Random(args.seed)

    overall_stats: Dict[str, Any] = {"dataset": args.dataset, "tokenizer": args.tokenizer, "splits": {}}
    sample_md_lines: List[str] = [
        f"# Random samples from {args.dataset}",
        "",
        f"Tokenizer: `{args.tokenizer}`  |  Seed: {args.seed}",
        "",
    ]

    # We'll combine token-length lists across splits for the histogram, but
    # report stats per-split.
    all_lengths: Dict[str, List[int]] = {}

    for split in args.splits:
        if split not in full:
            print(f"[inspect] skip split={split} (not in dataset)")
            continue
        ds = full[split]
        n_total = len(ds)
        n_use = min(args.max_samples, n_total)
        print(f"[inspect] split={split} total={n_total} using={n_use}")

        # Tokenize answers WITHOUT special tokens — we want raw answer length.
        sub = ds.select(range(n_use))
        answers = [str(ex.get("answer", "") or "") for ex in sub]
        questions = [str(ex.get("question", "") or "") for ex in sub]

        # Batched tokenize for speed
        enc = tok(
            answers,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        lengths = [len(x) for x in enc]
        all_lengths[split] = lengths

        # Char length too — useful sanity check (token vs char mismatch hint)
        char_lengths = [len(a) for a in answers]

        # Truncation heuristics on *all* used rows
        trunc = [detect_truncation(a) for a in answers]
        trunc_counts = Counter()
        for t in trunc:
            for k, v in t.items():
                if isinstance(v, bool) and v:
                    trunc_counts[k] += 1

        def pct(x: int) -> str:
            return f"{100.0 * x / n_use:.1f}%"

        # Length histogram bands
        bins = [(0, 50), (50, 100), (100, 150), (150, 180), (180, 195),
                (195, 200), (200, 201), (201, 205), (205, 220), (220, 256), (256, 10000)]
        band_counts = {f"[{lo},{hi})": sum(1 for L in lengths if lo <= L < hi) for lo, hi in bins}

        stats = {
            "n_used": n_use,
            "n_total": n_total,
            "token_length": {
                "min": min(lengths),
                "p25": int(statistics.quantiles(lengths, n=4)[0]),
                "median": int(statistics.median(lengths)),
                "mean": round(statistics.mean(lengths), 1),
                "p75": int(statistics.quantiles(lengths, n=4)[2]),
                "p90": int(statistics.quantiles(lengths, n=10)[8]),
                "p95": int(statistics.quantiles(lengths, n=20)[18]),
                "p99": int(statistics.quantiles(lengths, n=100)[98]),
                "max": max(lengths),
                "frac_le_50": pct(sum(1 for L in lengths if L <= 50)),
                "frac_in_195_205": pct(sum(1 for L in lengths if 195 <= L <= 205)),
                "frac_eq_200": pct(sum(1 for L in lengths if L == 200)),
                "frac_gt_200": pct(sum(1 for L in lengths if L > 200)),
                "bands": band_counts,
            },
            "char_length": {
                "median": int(statistics.median(char_lengths)),
                "mean": round(statistics.mean(char_lengths), 1),
                "p95": int(statistics.quantiles(char_lengths, n=20)[18]),
                "max": max(char_lengths),
            },
            "truncation_signals": {
                "n_with_special_token_string": trunc_counts.get("has_special_token", 0),
                "n_ends_unfinished_code": trunc_counts.get("ends_unfinished_code", 0),
                "n_ends_with_period_or_close": trunc_counts.get("ends_with_period_or_close", 0),
                "n_tail_is_prose": trunc_counts.get("tail_is_prose", 0),
                "pct_with_special_token_string": pct(trunc_counts.get("has_special_token", 0)),
                "pct_ends_unfinished_code": pct(trunc_counts.get("ends_unfinished_code", 0)),
                "pct_ends_with_period_or_close": pct(trunc_counts.get("ends_with_period_or_close", 0)),
                "pct_tail_is_prose": pct(trunc_counts.get("tail_is_prose", 0)),
            },
        }
        overall_stats["splits"][split] = stats

        # ----------- random samples for this split -----------
        idxs = sorted(rng.sample(range(n_use), min(args.n_print, n_use)))
        sample_md_lines.append(f"## split = {split} ({len(idxs)} random samples)")
        sample_md_lines.append("")
        for i in idxs:
            q = questions[i].strip()
            a = answers[i]
            L_tok = lengths[i]
            L_char = len(a)
            t = trunc[i]
            sample_md_lines.extend([
                f"### sample idx={i} (token_len={L_tok}, char_len={L_char})",
                f"- has_special_token={t['has_special_token']}  ends_unfinished_code={t['ends_unfinished_code']}  ends_closed={t['ends_with_period_or_close']}  tail_is_prose={t['tail_is_prose']}",
                f"- tail (last 80 chars, escaped): `{t['tail']}`",
                "",
                "**question:**",
                "",
                "```",
                (q[:600] + (" …[truncated]" if len(q) > 600 else "")),
                "```",
                "",
                "**answer:**",
                "",
                "```",
                a,
                "```",
                "",
            ])

    # ---------------- write reports ----------------
    stats_path = args.out_dir / "length_stats.json"
    samples_path = args.out_dir / "samples.md"
    report_path = args.out_dir / "report.md"

    stats_path.write_text(json.dumps(overall_stats, indent=2))
    samples_path.write_text("\n".join(sample_md_lines))

    # Top-level human report
    rep = ["# opencode-instruct_100k_200tok — length / truncation inspection", ""]
    for split, st in overall_stats["splits"].items():
        L = st["token_length"]
        T = st["truncation_signals"]
        rep += [
            f"## split = {split}  (used {st['n_used']:,} / {st['n_total']:,})",
            "",
            "### Answer token length (Qwen2.5 tokenizer)",
            "",
            f"- min / median / mean / max  =  **{L['min']} / {L['median']} / {L['mean']} / {L['max']}**",
            f"- p25 / p75 / p90 / p95 / p99  =  **{L['p25']} / {L['p75']} / {L['p90']} / {L['p95']} / {L['p99']}**",
            f"- fraction ≤ 50 tokens          : **{L['frac_le_50']}**",
            f"- fraction in [195, 205]        : **{L['frac_in_195_205']}**   ← **stack near the suspected 200 cap**",
            f"- fraction == 200 exactly       : **{L['frac_eq_200']}**",
            f"- fraction > 200                : **{L['frac_gt_200']}**",
            "",
            "Histogram bands:",
            "",
        ]
        for band, c in L["bands"].items():
            pct_ = 100.0 * c / st["n_used"]
            bar = "█" * max(1, int(round(pct_)))
            rep.append(f"- {band:>12} : {c:>5}  ({pct_:5.1f}%)  {bar}")
        rep += [
            "",
            "### Truncation / style heuristics",
            "",
            f"- contains `<|endoftext|>` / `<|im_end|>` literal in answer text : **{T['pct_with_special_token_string']}** ({T['n_with_special_token_string']})",
            f"- last non-blank char looks like unfinished code (`(`, `{{`, `,`, `:`, `\\\\`, …) : **{T['pct_ends_unfinished_code']}** ({T['n_ends_unfinished_code']})",
            f"- last non-blank char is a closed token (`)`, `]`, `}}`, `.`, quote) : **{T['pct_ends_with_period_or_close']}** ({T['n_ends_with_period_or_close']})",
            f"- last sentence looks like English prose                          : **{T['pct_tail_is_prose']}** ({T['n_tail_is_prose']})",
            "",
        ]

    report_path.write_text("\n".join(rep))
    print(f"\n[inspect] wrote {stats_path}")
    print(f"[inspect] wrote {samples_path}")
    print(f"[inspect] wrote {report_path}")
    print()
    print("\n".join(rep))

    # ---------------- histogram plot ----------------
    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[inspect] matplotlib unavailable, skipping plot ({exc})")
        return

    fig, axes = plt.subplots(1, len(all_lengths), figsize=(6 * len(all_lengths), 4), squeeze=False)
    for ax, (split, lengths) in zip(axes[0], all_lengths.items()):
        ax.hist(lengths, bins=range(0, max(lengths) + 5, 5), color="steelblue", edgecolor="black", linewidth=0.3)
        ax.axvline(200, color="red", linestyle="--", linewidth=1.0, label="200 tok cap?")
        ax.set_title(f"{split}  (n={len(lengths):,})")
        ax.set_xlabel("answer token length")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Answer length distribution — {args.dataset}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = args.out_dir / "length_hist.png"
    fig.savefig(out_png, dpi=130)
    print(f"[inspect] wrote {out_png}")


if __name__ == "__main__":
    main()
