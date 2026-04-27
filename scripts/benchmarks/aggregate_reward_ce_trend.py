#!/usr/bin/env python3
"""Aggregate per-checkpoint reward / CE results into a single trend table + plot.

Reads:    outputs/paperqa_ebft_trend_seed43/offline_reward_ce_trend/step{N}.json
Writes:   outputs/paperqa_ebft_trend_seed43/offline_reward_ce_trend/trend_summary.json
          outputs/paperqa_ebft_trend_seed43/offline_reward_ce_trend/trend_summary.md
          outputs/paperqa_ebft_trend_seed43/offline_reward_ce_trend/trend.png   (if matplotlib available)

Also overlays the HumanEval / MBPP greedy_accuracy from
outputs/paperqa_ebft_trend_seed43/offline_benchmarks/trend_summary.json
so we can visually correlate "reward in EBFT space" with "downstream code accuracy".
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path("/root/code/Distributional-Match-Tuning")
RUN_DIR = REPO_ROOT / "outputs/paperqa_ebft_trend_seed43"
REWARD_DIR = RUN_DIR / "offline_reward_ce_trend"
BENCH_TREND = RUN_DIR / "offline_benchmarks/trend_summary.json"


METRIC_KEYS = [
    "reward_pass1_gt",
    "reward_pass1_diversity",
    "reward_pass1_effective",
    "reward_pass1",
    "reward_passk_gt",
    "reward_passk_diversity",
    "reward_passk_effective",
    "reward_passk",
    "full_ce_loss",
    "full_perplexity",
    "answer_ce_loss",
    "answer_perplexity",
]


def parse_step_from_filename(name: str) -> Optional[int]:
    if name == "step0_baseline.json":
        return 0
    m = re.match(r"step(\d+)\.json$", name)
    return int(m.group(1)) if m else None


def load_one(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    return {k: metrics.get(k) for k in METRIC_KEYS}


def load_humaneval_mbpp_trend(bench_trend: Path = BENCH_TREND) -> Dict[int, Dict[str, float]]:
    """Return {step -> {humaneval, mbpp}} from a benchmarks trend_summary.json.

    The file currently has the flat schema:
        [{"checkpoint": "global_step38_hf", "global_step": 38,
          "HumanEval": 0.35..., "MBPP": 0.0}, ...]
    """
    if not bench_trend.exists():
        return {}
    with bench_trend.open() as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("entries", [])

    def _get(entry: Dict[str, Any], *keys: str) -> Optional[float]:
        for k in keys:
            if k in entry:
                v = entry[k]
                if isinstance(v, dict):
                    v = v.get("greedy_accuracy") or v.get("pass@1") or v.get("pass1")
                if v is not None:
                    return float(v)
        return None

    out: Dict[int, Dict[str, float]] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        step = entry.get("global_step") or entry.get("step")
        if step is None:
            continue
        out[int(step)] = {
            "humaneval_pass1": _get(entry, "HumanEval", "humaneval"),
            "mbpp_pass1": _get(entry, "MBPP", "mbpp"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-dir", type=Path, default=REWARD_DIR)
    parser.add_argument("--out-prefix", type=str, default="trend_summary")
    parser.add_argument("--bench-trend", type=Path, default=BENCH_TREND,
                        help="Path to offline benchmarks trend_summary.json (HumanEval/MBPP)")
    parser.add_argument("--plot-name", type=str, default="trend.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    files = sorted(args.reward_dir.glob("step*.json"))
    if not files:
        raise SystemExit(f"no step*.json found under {args.reward_dir}")

    rows: List[Dict[str, Any]] = []
    for f in files:
        step = parse_step_from_filename(f.name)
        if step is None:
            continue
        m = load_one(f)
        m["step"] = step
        rows.append(m)
    rows.sort(key=lambda r: r["step"])

    # Merge in HumanEval / MBPP if available (allow partial / nearest match)
    bench = load_humaneval_mbpp_trend(args.bench_trend)
    for r in rows:
        b = bench.get(r["step"], {})
        r["humaneval_pass1"] = b.get("humaneval_pass1")
        r["mbpp_pass1"] = b.get("mbpp_pass1")

    out_json = args.reward_dir / f"{args.out_prefix}.json"
    out_md = args.reward_dir / f"{args.out_prefix}.md"
    out_png = args.reward_dir / args.plot_name

    with out_json.open("w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"wrote {out_json}")

    # ---------------- Markdown table ------------------
    headers = [
        "step",
        "align(p@1)", "div(p@1)", "eff(p@1)",
        "ans_ppl", "full_ppl",
        "HumEv", "MBPP",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    def fmt(x, n=4):
        if x is None:
            return "-"
        if isinstance(x, float):
            return f"{x:.{n}f}"
        return str(x)
    for r in rows:
        lines.append("| " + " | ".join([
            str(r["step"]),
            fmt(r["reward_pass1_gt"]),
            fmt(r["reward_pass1_diversity"]),
            fmt(r["reward_pass1_effective"]),
            fmt(r["answer_perplexity"]),
            fmt(r["full_perplexity"]),
            fmt(r["humaneval_pass1"], 3),
            fmt(r["mbpp_pass1"], 3),
        ]) + " |")
    md = "\n".join(lines) + "\n"
    with out_md.open("w") as f:
        f.write(md)
    print(f"wrote {out_md}")
    print()
    print(md)

    # ---------------- Plot ------------------
    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skipping plot: {exc}")
        return

    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("paperqa_ebft_trend_seed43 — reward / CE / downstream trend", fontsize=12)

    # (0,0) alignment + diversity
    ax = axes[0, 0]
    ax.plot(steps, [r["reward_pass1_gt"] for r in rows], "o-", label="alignment (gt)")
    ax.plot(steps, [r["reward_pass1_diversity"] for r in rows], "s-", label="diversity (penalty)")
    ax.plot(steps, [r["reward_pass1_effective"] for r in rows], "^-", label="effective")
    ax.set_title("Reward (pass@1)")
    ax.set_xlabel("global step")
    ax.set_ylabel("reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) PPL
    ax = axes[0, 1]
    ax.plot(steps, [r["answer_perplexity"] for r in rows], "o-", label="answer PPL")
    ax.plot(steps, [r["full_perplexity"] for r in rows], "s-", label="full PPL")
    ax.set_title("Perplexity")
    ax.set_xlabel("global step")
    ax.set_ylabel("PPL")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) downstream HumanEval / MBPP
    ax = axes[1, 0]
    he = [r["humaneval_pass1"] for r in rows]
    mb = [r["mbpp_pass1"] for r in rows]
    if any(v is not None for v in he):
        ax.plot(steps, he, "o-", label="HumanEval pass@1")
    if any(v is not None for v in mb):
        ax.plot(steps, mb, "s-", label="MBPP pass@1")
    ax.set_title("Downstream code accuracy")
    ax.set_xlabel("global step")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) reward vs HumanEval scatter (with step labels)
    ax = axes[1, 1]
    pts = [(r["reward_pass1_gt"], r["humaneval_pass1"], r["step"])
           for r in rows
           if r["reward_pass1_gt"] is not None and r["humaneval_pass1"] is not None]
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts])
        for x, y, s in pts:
            ax.annotate(f"step{s}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_xlabel("alignment reward (pass@1)")
        ax.set_ylabel("HumanEval pass@1")
        ax.set_title("Reward vs downstream (each dot = ckpt)")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
