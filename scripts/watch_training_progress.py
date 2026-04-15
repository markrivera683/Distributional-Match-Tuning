#!/usr/bin/env python3
"""Watch or summarize training progress for EBFT/G* runs.

Examples:
  python scripts/watch_training_progress.py
  python scripts/watch_training_progress.py --watch
  python scripts/watch_training_progress.py --run-dir /root/outputs/g3_rebase_xxx
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import os
import re
import shutil
import sys
import time
from pathlib import Path


DEFAULT_ROOTS = [
    "/root/outputs",
    "/mnt/data/ebft-teacher-distribution/outputs_g3_0.99",
    "/mnt/data/ebft-teacher-distribution/outputs2",
    "/mnt/data/ebft-teacher-distribution/outputs",
]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MAX_STEPS_RE = re.compile(r"max[_ ]steps[:=]\s*(\d+)", re.IGNORECASE)
GLOBAL_STEP_RE = re.compile(r"Global step (\d+): (.+)$")
PREFETCH_RE = re.compile(
    r"\[Prefetch\] step=(\d+) hit_rate=([0-9.]+)% mem=(\d+) sqlite=(\d+) fallback=(\d+) queue=(\d+) pending=(\d+)"
)
EVAL_RE = re.compile(r"Evaluation completed .* global_step (\d+), eval_metrics: (.+)$")
RUNNING_EVAL_RE = re.compile(r"Running .*evaluation .* step (\d+)", re.IGNORECASE)
ERROR_RE = re.compile(r"\b(ERROR|WARN|Traceback|Exception)\b")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def human_age(ts: float | None) -> str:
    if ts is None:
        return "unknown"
    delta = max(0, int(time.time() - ts))
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


def fmt_ratio(current: int | None, total: int | None) -> str:
    if current is None and total is None:
        return "unknown"
    if current is None:
        return f"? / {total}"
    if total in (None, 0):
        return str(current)
    return f"{current} / {total} ({current / total * 100:.1f}%)"


def read_head(path: Path, max_bytes: int = 256 * 1024) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read(max_bytes)


def read_tail(path: Path, max_bytes: int = 512 * 1024) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        data = f.read().decode("utf-8", errors="replace")
    return data


def latest_file(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def parse_metrics_dict(raw: str) -> dict:
    try:
        obj = ast.literal_eval(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def summarize_metrics(metrics: dict) -> str:
    if not metrics:
        return "n/a"
    preferred = [
        "critic_loss",
        "actor_loss",
        "loss",
        "reward",
        "reward_pass1",
        "reward_passk",
        "full_ce_loss",
        "full_perplexity",
        "mse",
    ]
    parts = []
    for key in preferred:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
    if parts:
        return ", ".join(parts)
    keys = list(metrics.keys())[:5]
    for key in keys:
        value = metrics[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def detect_run_dirs(roots: list[Path]) -> list[Path]:
    runs: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if (child / "train.log").exists() or (child / "ray_job.log").exists():
                runs.append(child)
    runs.sort(key=lambda p: latest_mtime_for_run(p) or 0, reverse=True)
    return runs


def latest_mtime_for_run(run_dir: Path) -> float | None:
    candidates = [
        run_dir / "train.log",
        run_dir / "ray_job.log",
        run_dir / "status.txt",
    ]
    supplement_logs = run_dir / "supplement_logs"
    if supplement_logs.is_dir():
        candidates.extend(p for p in supplement_logs.iterdir() if p.is_file())
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None
    return max(p.stat().st_mtime for p in existing)


def classify_status(last_train_mtime: float | None, last_eval_mtime: float | None, has_status: bool) -> str:
    now = time.time()
    if has_status:
        return "completed"
    if last_eval_mtime and now - last_eval_mtime < 180:
        return "evaluating"
    if last_train_mtime and now - last_train_mtime < 180:
        return "training"
    if last_train_mtime:
        return "idle/stale"
    return "unknown"


def collect_recent_interesting_lines(paths: list[Path], limit: int = 8) -> list[str]:
    interesting: list[str] = []
    for path in paths:
        text = read_tail(path, max_bytes=256 * 1024)
        for raw in text.splitlines():
            line = strip_ansi(raw).strip()
            if not line:
                continue
            m_global = GLOBAL_STEP_RE.search(line)
            m_eval = EVAL_RE.search(line)
            if m_global:
                step = int(m_global.group(1))
                metrics = parse_metrics_dict(m_global.group(2))
                interesting.append(f"Global step {step} | {summarize_metrics(metrics)}")
                continue
            if m_eval:
                step = int(m_eval.group(1))
                metrics = parse_metrics_dict(m_eval.group(2))
                interesting.append(f"Eval complete @ step {step} | {summarize_metrics(metrics)}")
                continue
            if (
                PREFETCH_RE.search(line)
                or RUNNING_EVAL_RE.search(line)
                or "[done]" in line
                or ERROR_RE.search(line)
            ):
                interesting.append(line[:220] + ("..." if len(line) > 220 else ""))
    if len(interesting) > limit:
        interesting = interesting[-limit:]
    return interesting


def build_summary(run_dir: Path) -> dict:
    train_log = run_dir / "train.log"
    ray_job_log = run_dir / "ray_job.log"
    supplement_logs = run_dir / "supplement_logs"
    status_file = run_dir / "status.txt"

    latest_eval_log = None
    latest_eval_output = None
    latest_eval_report = None
    if supplement_logs.is_dir():
        latest_eval_log = latest_file(list(supplement_logs.glob("eval_*.log")))
        latest_eval_output = latest_file(list(supplement_logs.glob("eval_results_*.jsonl")))
        latest_eval_report = latest_file(list(supplement_logs.glob("eval_analysis_*.json")))

    head = strip_ansi(read_head(train_log))
    tail = strip_ansi(read_tail(train_log))

    max_steps = None
    for match in MAX_STEPS_RE.finditer(head):
        max_steps = int(match.group(1))
        break

    last_global_step = None
    last_global_metrics = {}
    for line in tail.splitlines():
        m = GLOBAL_STEP_RE.search(line)
        if m:
            last_global_step = int(m.group(1))
            last_global_metrics = parse_metrics_dict(m.group(2))

    last_prefetch = None
    for line in tail.splitlines():
        m = PREFETCH_RE.search(line)
        if m:
            last_prefetch = {
                "step": int(m.group(1)),
                "hit_rate": float(m.group(2)),
                "mem": int(m.group(3)),
                "sqlite": int(m.group(4)),
                "fallback": int(m.group(5)),
                "queue": int(m.group(6)),
                "pending": int(m.group(7)),
            }

    last_eval = None
    for line in tail.splitlines():
        m = EVAL_RE.search(line)
        if m:
            last_eval = {
                "step": int(m.group(1)),
                "metrics": parse_metrics_dict(m.group(2)),
            }

    running_eval_step = None
    for line in tail.splitlines():
        m = RUNNING_EVAL_RE.search(line)
        if m:
            running_eval_step = int(m.group(1))

    ckpt_dirs = sorted(
        [p.name for p in (run_dir / "model" / "ckpt").glob("global_step*_hf") if p.is_dir()],
        key=lambda name: int(re.search(r"global_step(\d+)", name).group(1)) if re.search(r"global_step(\d+)", name) else -1,
    )

    recent_lines = collect_recent_interesting_lines(
        [p for p in [train_log, ray_job_log, latest_eval_log] if isinstance(p, Path) and p.exists()]
    )

    return {
        "run_dir": run_dir,
        "train_log": train_log,
        "ray_job_log": ray_job_log,
        "latest_eval_log": latest_eval_log,
        "latest_eval_output": latest_eval_output,
        "latest_eval_report": latest_eval_report,
        "max_steps": max_steps,
        "last_global_step": last_global_step,
        "last_global_metrics": last_global_metrics,
        "last_prefetch": last_prefetch,
        "last_eval": last_eval,
        "running_eval_step": running_eval_step,
        "ckpt_dirs": ckpt_dirs,
        "last_train_mtime": train_log.stat().st_mtime if train_log.exists() else None,
        "last_eval_mtime": latest_eval_log.stat().st_mtime if latest_eval_log else None,
        "status": classify_status(
            train_log.stat().st_mtime if train_log.exists() else None,
            latest_eval_log.stat().st_mtime if latest_eval_log else None,
            status_file.exists(),
        ),
        "recent_lines": recent_lines,
    }


def render(summary: dict) -> str:
    run_dir: Path = summary["run_dir"]
    lines = []
    lines.append(f"Run Dir:        {run_dir}")
    lines.append(f"Status:         {summary['status']}")
    lines.append(f"Train Log:      {summary['train_log']}")
    lines.append(
        f"Last Update:    {human_age(summary['last_train_mtime'])}"
        + (
            f" | Last Eval: {human_age(summary['last_eval_mtime'])}"
            if summary["last_eval_mtime"]
            else ""
        )
    )
    lines.append(f"Progress:       {fmt_ratio(summary['last_global_step'], summary['max_steps'])}")

    if summary["last_global_step"] is not None:
        lines.append(f"Latest Metrics: {summarize_metrics(summary['last_global_metrics'])}")
    if summary["last_prefetch"]:
        pf = summary["last_prefetch"]
        lines.append(
            "Prefetch:       "
            f"step={pf['step']} hit={pf['hit_rate']:.1f}% mem={pf['mem']} sqlite={pf['sqlite']} "
            f"fallback={pf['fallback']} queue={pf['queue']} pending={pf['pending']}"
        )
    if summary["running_eval_step"] is not None:
        lines.append(f"Eval Running:   step {summary['running_eval_step']}")
    if summary["last_eval"]:
        lines.append(
            f"Last Eval:      step {summary['last_eval']['step']} | "
            f"{summarize_metrics(summary['last_eval']['metrics'])}"
        )
    if summary["ckpt_dirs"]:
        lines.append(f"Checkpoints:    {', '.join(summary['ckpt_dirs'][-5:])}")
    else:
        lines.append("Checkpoints:    none yet")

    if summary["latest_eval_output"]:
        lines.append(f"Eval Output:    {summary['latest_eval_output']}")
    if summary["latest_eval_report"]:
        lines.append(f"Eval Report:    {summary['latest_eval_report']}")

    if summary["recent_lines"]:
        lines.append("")
        lines.append("Recent Interesting Lines:")
        for line in summary["recent_lines"]:
            lines.append(f"  {line}")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show current training progress for EBFT runs.")
    parser.add_argument("--run-dir", type=str, default=None, help="Explicit run directory to inspect.")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Search root for auto-detect. Can be passed multiple times.",
    )
    parser.add_argument("--watch", action="store_true", help="Continuously refresh the summary.")
    parser.add_argument("--interval", type=float, default=10.0, help="Refresh interval in seconds for --watch.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = [Path(p) for p in (args.root or DEFAULT_ROOTS)]

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_dir():
            print(f"[ERROR] run dir not found: {run_dir}", file=sys.stderr)
            return 1
    else:
        candidates = detect_run_dirs(roots)
        if not candidates:
            searched = ", ".join(str(p) for p in roots)
            print(f"[ERROR] no run directories found under: {searched}", file=sys.stderr)
            return 1
        run_dir = candidates[0]

    while True:
        summary = build_summary(run_dir)
        if args.watch:
            width = shutil.get_terminal_size((120, 30)).columns
            print("\033[2J\033[H", end="")
            print(f"{' Training Progress ':=^{width}}")
            print(f"Refreshed: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(render(summary))
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
