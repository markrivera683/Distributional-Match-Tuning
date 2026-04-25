#!/usr/bin/env python3
"""Dump raw, untruncated greedy generations from step971 on a handful of
HumanEval tasks, plus extract any fenced ```python ... ``` block we can find.

Goal: tell apart the WEAK vs STRONG version of the format-mismatch hypothesis.
  - WEAK : even after extracting code from any wrapper, the function body is wrong.
  - STRONG: the function body inside the wrapper is structurally correct; benchmark
            failures are due to wrapper / parsing.

Outputs:
  outputs/paperqa_ebft_trend_seed43/offline_benchmarks/diagnose_step971_raw.md
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = "/root/code/Distributional-Match-Tuning/outputs/paperqa_ebft_trend_seed43/checkpoints/global_step971_hf"
OUT = Path("/root/code/Distributional-Match-Tuning/outputs/paperqa_ebft_trend_seed43/offline_benchmarks/diagnose_step971_raw.md")

# 5 representative tasks: a few easy / mid / harder
TASK_IDS = [
    "HumanEval/2",   # truncate floats
    "HumanEval/3",   # below_zero
    "HumanEval/7",   # filter_by_substring
    "HumanEval/9",   # rolling_max  (was both-pass earlier)
    "HumanEval/13",  # greatest_common_divisor
]

MAX_NEW_TOKENS = 512
PROMPT_MAX_LEN = 1024

FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def main() -> None:
    print(f"[load] {CKPT}")
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        CKPT, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()

    ds = load_dataset("openai/openai_humaneval", split="test")
    by_id = {row["task_id"]: row for row in ds}

    prompts = [by_id[t]["prompt"] for t in TASK_IDS]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=PROMPT_MAX_LEN)
    input_ids = enc["input_ids"].to("cuda")
    attn = enc["attention_mask"].to("cuda")
    p_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
    out = out[:, p_len:]
    decoded = tok.batch_decode(out, skip_special_tokens=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write("# step971 raw HumanEval greedy generations (no post-processing)\n\n")
        f.write("Per task we dump:\n")
        f.write("1. **prompt tail** — last ~400 chars\n")
        f.write("2. **raw generation** — exactly what the model emitted, no stop strings, no special-token stripping (well, we set skip_special_tokens=True only)\n")
        f.write("3. **fenced code extracted** — content between ```python ``` blocks if any\n\n")
        for tid, gen in zip(TASK_IDS, decoded):
            row = by_id[tid]
            f.write(f"---\n\n## {tid}\n\n")
            f.write("### Prompt (last 400 chars)\n\n")
            f.write("```python\n" + row["prompt"][-400:] + "\n```\n\n")
            f.write("### Raw generation\n\n")
            f.write("```\n" + gen + "\n```\n\n")
            fences = FENCE_RE.findall(gen)
            if fences:
                f.write("### Extracted fenced code blocks (count={})\n\n".format(len(fences)))
                for i, code in enumerate(fences):
                    f.write(f"#### block {i+1}\n\n")
                    f.write("```python\n" + code.rstrip() + "\n```\n\n")
            else:
                f.write("### Extracted fenced code blocks\n\n_(none found)_\n\n")

    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
