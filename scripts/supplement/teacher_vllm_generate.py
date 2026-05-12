#!/usr/bin/env python3
"""Minimal vLLM-based generator for teacher-model supplement eval.

This intentionally avoids the project's `batch_inference.py` so that teacher
evaluation can run from the lightweight `.teacherVenv` environment without
needing the full training dependency stack.
"""

import argparse
import json
import os
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dataset_rows(path: str) -> list[dict[str, Any]]:
    if os.path.isdir(path):
        for name in ("test.jsonl", "test_qa.jsonl", "eval.jsonl"):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return load_jsonl(candidate)
        raise FileNotFoundError(f"No supported eval file found under directory: {path}")

    if path.endswith(".jsonl"):
        return load_jsonl(path)

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("test", "data", "rows"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported JSON dataset structure in: {path}")

    raise ValueError(f"Unsupported dataset path: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal vLLM generator for teacher eval")
    parser.add_argument("--pretrain", required=True, help="Model path or HF id")
    parser.add_argument("--dataset", required=True, help="JSONL / JSON dataset path")
    parser.add_argument("--input_key", default="question")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--prompt_max_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=5328)
    parser.add_argument("--best_of_n", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    args = parser.parse_args()

    rows = load_dataset_rows(args.dataset)
    rows = rows[: min(args.max_samples, len(rows))]

    prompts = []
    for row in tqdm(rows, desc="Preparing prompts"):
        prompt = row.get(args.input_key, "")
        if args.input_template:
            prompt = args.input_template.format(prompt)
        prompts.append(prompt)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)
    model_prompts = []
    truncated_count = 0
    for prompt in tqdm(prompts, desc="Truncating prompts"):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) > args.prompt_max_len:
            token_ids = token_ids[: args.prompt_max_len]
            prompt_for_model = tokenizer.decode(token_ids, skip_special_tokens=False)
            truncated_count += 1
        else:
            prompt_for_model = prompt
        model_prompts.append(prompt_for_model)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print("=" * 70)
    print("Minimal Teacher vLLM Generation")
    print("=" * 70)
    print(f"Model path:             {args.pretrain}")
    print(f"Dataset path:           {args.dataset}")
    print(f"Loaded prompts:         {len(prompts)}")
    print(f"Tensor parallel size:   {args.tp_size}")
    print(f"Max num seqs:           {args.max_num_seqs}")
    print(f"Prompt max len:         {args.prompt_max_len}")
    print(f"Prompts truncated:      {truncated_count}")
    print(f"Max new tokens:         {args.max_new_tokens}")
    print(f"Output path:            {args.output_path}")
    print("=" * 70)

    llm = LLM(
        model=args.pretrain,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    repeated_prompts = model_prompts * args.best_of_n
    repeated_original_prompts = prompts * args.best_of_n
    outputs = llm.generate(repeated_prompts, sampling_params)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for original_prompt, output in zip(repeated_original_prompts, outputs):
            record = {
                "input": original_prompt,
                "output": output.outputs[0].text if output.outputs else "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(outputs)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
