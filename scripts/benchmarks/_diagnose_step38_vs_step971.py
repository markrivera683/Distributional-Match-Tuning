#!/usr/bin/env python3
"""One-off diagnostic: side-by-side HumanEval generation comparison between
the earliest checkpoint (step38, best HumanEval) and the latest (step971,
worst HumanEval) for the paper QA EBFT trend run.

Outputs:
    diagnose_step38_vs_step971.md  — human-readable side-by-side dump
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


HUMANEVAL_TASKS = [
    # 5 regressed (step38 PASS -> step971 FAIL syntax) — most informative
    "HumanEval/2",
    "HumanEval/3",
    "HumanEval/7",
    "HumanEval/12",
    "HumanEval/13",
    # 3 both-pass — sanity check (does step971 still produce sane code on easy ones?)
    "HumanEval/9",
    "HumanEval/11",
    "HumanEval/23",
    # 2 both-fail — out-of-scope baseline
    "HumanEval/0",
    "HumanEval/1",
]

CKPTS = {
    "step38": "/root/code/Distributional-Match-Tuning/outputs/paperqa_ebft_trend_seed43/checkpoints/global_step38_hf",
    "step971": "/root/code/Distributional-Match-Tuning/outputs/paperqa_ebft_trend_seed43/checkpoints/global_step971_hf",
}

OUT_PATH = Path("/root/code/Distributional-Match-Tuning/outputs/paperqa_ebft_trend_seed43/offline_benchmarks/diagnose_step38_vs_step971.md")
MAX_NEW_TOKENS = 512
PROMPT_MAX_LEN = 1024


def truncate_at_stop(text: str, stops: list[str]) -> str:
    if not stops:
        return text
    cut = len(text)
    for s in stops:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and idx < cut:
            cut = idx
    return text[:cut]


def generate_for_ckpt(ckpt_path: str, tasks_with_prompts: list[tuple[str, str, list[str]]]):
    print(f"[gen] loading {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    results = {}
    prompts = [p for _, p, _ in tasks_with_prompts]
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=PROMPT_MAX_LEN,
    )
    input_ids = enc["input_ids"].to("cuda")
    attention_mask = enc["attention_mask"].to("cuda")
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            top_p=1.0,
            repetition_penalty=1.0,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    out = out[:, prompt_len:]
    decoded = tokenizer.batch_decode(out, skip_special_tokens=False)

    for (tid, _, stops), text in zip(tasks_with_prompts, decoded):
        results[tid] = truncate_at_stop(text, stops)

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--humaneval_dataset", default="openai/openai_humaneval")
    parser.add_argument("--humaneval_split", default="test")
    args = parser.parse_args()

    ds = load_dataset(args.humaneval_dataset, split=args.humaneval_split)
    by_id = {row["task_id"]: row for row in ds}

    HUMANEVAL_STOPS = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
    tasks_with_prompts = [
        (tid, by_id[tid]["prompt"], HUMANEVAL_STOPS) for tid in HUMANEVAL_TASKS
    ]

    all_outputs = {}
    for label, ckpt in CKPTS.items():
        all_outputs[label] = generate_for_ckpt(ckpt, tasks_with_prompts)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        f.write("# step38 vs step971 — HumanEval greedy generation diff\n\n")
        f.write("Per task: prompt (truncated), step38 output, step971 output.\n\n")
        for tid, _, _ in tasks_with_prompts:
            row = by_id[tid]
            f.write(f"---\n\n## {tid}\n\n")
            prompt_tail = row["prompt"][-600:]
            f.write("### Prompt (last 600 chars)\n\n")
            f.write("```python\n" + prompt_tail + "\n```\n\n")
            for label in ("step38", "step971"):
                gen = all_outputs[label].get(tid, "")
                f.write(f"### {label} generation\n\n")
                f.write("```python\n" + gen + "\n```\n\n")

    print(f"[done] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
