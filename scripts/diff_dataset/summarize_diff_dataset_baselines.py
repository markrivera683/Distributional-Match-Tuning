#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def duration_to_seconds(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def read_final_status(run_dir: Path) -> dict[str, str]:
    status_path = run_dir / "final_status.env"
    status: dict[str, str] = {}
    if not status_path.exists():
        return status
    for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            status[key.strip()] = value.strip()
    return status


def summarize_run(run_dir: Path) -> dict[str, str]:
    log_path = run_dir / "train.log"
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    status = read_final_status(run_dir)

    max_steps = ""
    match = re.search(r"\bmax_steps=(\d+)\b", text) or re.search(r"\bmax steps:\s*(\d+)", text)
    if match:
        max_steps = match.group(1)

    episode_seconds = None
    seconds_per_step = None
    for match in re.finditer(r"Episode \[1/1\].*?\[([0-9:]+)<00:00,\s*([0-9.]+)s/it\]", text):
        episode_seconds = duration_to_seconds(match.group(1))
        seconds_per_step = float(match.group(2))

    make_exp = [duration_to_seconds(match.group(1)) for match in re.finditer(r"Experience making completed in ([0-9:]+)", text)]
    def parse_train_durations(actor_name: str) -> list[float]:
        durations: list[float] = []
        current_actor = ""
        for line in text.replace("\r", "\n").splitlines():
            if "EBFTPolicyModelActor" in line:
                current_actor = "EBFTPolicyModelActor"
            elif "EBFTCriticModelActor" in line:
                current_actor = "EBFTCriticModelActor"
            if current_actor != actor_name or "Train epoch [1/1]" not in line or "100%" not in line:
                continue
            matches = re.findall(r"\[([0-9:]+)<00:00", line)
            if matches:
                durations.append(duration_to_seconds(matches[-1]))
        return durations

    actor_train = parse_train_durations("EBFTPolicyModelActor")
    critic_train = parse_train_durations("EBFTCriticModelActor")
    save_shards = [float(match.group(1)) for match in re.finditer(r"Writing model shards:.*?\[00:([0-9.]+)<00:00", text)]

    make_exp_avg = mean(make_exp)
    actor_avg = mean(actor_train)
    critic_avg = mean(critic_train)
    residual = None
    if seconds_per_step is not None:
        residual = seconds_per_step
        for value in (make_exp_avg, actor_avg, critic_avg):
            if value is not None:
                residual -= value

    def fmt(value: float | None) -> str:
        return "" if value is None else f"{value:.3f}"

    return {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "train_rc": status.get("TRAIN_RC", ""),
        "eval_rc": status.get("EVAL_RC", ""),
        "final_rc": status.get("FINAL_RC", ""),
        "max_steps": max_steps,
        "epoch_wall_clock_sec": fmt(episode_seconds),
        "seconds_per_step": fmt(seconds_per_step),
        "make_experience_avg_sec": fmt(make_exp_avg),
        "actor_train_avg_sec": fmt(actor_avg),
        "critic_train_avg_sec": fmt(critic_avg),
        "residual_rollout_other_avg_sec": fmt(residual),
        "save_shard_avg_sec": fmt(mean(save_shards)),
        "make_experience_count": str(len(make_exp)),
        "actor_train_count": str(len(actor_train)),
        "critic_train_count": str(len(critic_train)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize existing diff-dataset EBFT run logs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    rows = [summarize_run(Path(item)) for item in args.run_dirs]
    keys = [
        "run",
        "run_dir",
        "train_rc",
        "eval_rc",
        "final_rc",
        "max_steps",
        "epoch_wall_clock_sec",
        "seconds_per_step",
        "make_experience_avg_sec",
        "actor_train_avg_sec",
        "critic_train_avg_sec",
        "residual_rollout_other_avg_sec",
        "save_shard_avg_sec",
        "make_experience_count",
        "actor_train_count",
        "critic_train_count",
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for row in rows:
            f.write("\t".join(row.get(key, "") for key in keys) + "\n")
    print(f"[summary] wrote {out}")


if __name__ == "__main__":
    main()
