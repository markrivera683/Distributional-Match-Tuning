#!/usr/bin/env python3
"""Chunked vLLM generator with explicit real-time progress reporting."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return

    target_dir = os.path.dirname(path) or "."
    os.makedirs(target_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".progress-", suffix=".json", dir=target_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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


def chunked(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def ensure_qwen35_config_registered() -> None:
    """Register qwen3.5 config aliases for older transformers builds."""
    try:
        from transformers import AutoConfig

        AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)
        AutoConfig.register("qwen3_5", Qwen3_5Config, exist_ok=True)
    except Exception:
        # Best-effort only; continue if registration API differs.
        pass


def qwen35_hf_overrides(cfg: Any) -> Any:
    """Normalize qwen3_5_text config for vLLM multimodal registry.

    Some model checkpoints expose a text-only HF config (`qwen3_5_text`), while
    vLLM renderer paths expect the wrapper config type (`Qwen3_5Config`).
    """
    try:
        model_type = getattr(cfg, "model_type", None)
        if model_type == "qwen3_5_text" and not isinstance(cfg, Qwen3_5Config):
            text_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
            # Remove generic metadata keys that do not belong to text_config.
            text_cfg.pop("model_type", None)
            text_cfg.pop("transformers_version", None)
            wrapped = Qwen3_5Config(text_config=text_cfg)
            # Preserve architecture hint expected by vLLM model registry.
            arch = getattr(cfg, "architectures", None)
            if arch:
                wrapped.architectures = list(arch)
            elif text_cfg.get("architectures"):
                wrapped.architectures = list(text_cfg["architectures"])
            else:
                wrapped.architectures = ["Qwen3_5ForCausalLM"]
            return wrapped
    except Exception:
        pass
    return cfg


def ensure_qwen35_preprocessor_files(model_path: str) -> None:
    """Ensure qwen3.5 preprocessor json files exist for vLLM multimodal init."""
    if not os.path.isdir(model_path):
        return

    preproc = os.path.join(model_path, "preprocessor_config.json")
    video_preproc = os.path.join(model_path, "video_preprocessor_config.json")
    if os.path.exists(preproc):
        return

    # Common local base-model paths used in this repo.
    candidates = [
        "/mnt/data/teacher_model/models/Qwen3.5-0.8B",
        "/mnt/data/models/Qwen3.5-0.8B",
        "/mnt/data/models/qwen3.5-0.8b",
    ]

    for base in candidates:
        src_preproc = os.path.join(base, "preprocessor_config.json")
        if os.path.exists(src_preproc):
            shutil.copy2(src_preproc, preproc)
            src_video = os.path.join(base, "video_preprocessor_config.json")
            if os.path.exists(src_video) and not os.path.exists(video_preproc):
                shutil.copy2(src_video, video_preproc)
            print(f"[compat] copied preprocessor config from: {base}")
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunked vLLM generator with progress")
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
    parser.add_argument(
        "--progress_batch_size",
        type=int,
        default=16,
        help="How many prompts to submit per vLLM generate() call",
    )
    parser.add_argument(
        "--progress_json_path",
        type=str,
        default=None,
        help="Optional JSON file updated with the latest eval progress",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    args = parser.parse_args()

    if args.progress_batch_size < 1:
        raise ValueError("--progress_batch_size must be >= 1")

    progress_state: dict[str, Any] = {
        "status": "initializing",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "model_path": args.pretrain,
        "dataset_path": args.dataset,
        "output_path": args.output_path,
        "input_key": args.input_key,
        "prompt_max_len": args.prompt_max_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_samples": args.max_samples,
        "best_of_n": args.best_of_n,
        "tp_size": args.tp_size,
        "max_num_seqs": args.max_num_seqs,
        "progress_batch_size": args.progress_batch_size,
        "enable_prefix_caching": args.enable_prefix_caching,
        "total_prompts": 0,
        "total_outputs": 0,
        "written": 0,
        "remaining_outputs": 0,
        "percent": 0.0,
        "chunk_idx": 0,
        "chunk_count": 0,
        "rate_samples_per_sec": 0.0,
        "elapsed_seconds": 0.0,
    }
    write_json_atomic(args.progress_json_path, progress_state)

    try:
        rows = load_dataset_rows(args.dataset)
        rows = rows[: min(args.max_samples, len(rows))]

        prompts = []
        source_indices = []
        for idx, row in enumerate(tqdm(rows, desc="Preparing prompts", dynamic_ncols=True)):
            prompt = row.get(args.input_key, "")
            if args.input_template:
                prompt = args.input_template.format(prompt)
            prompts.append(prompt)
            source_idx = row.get("source_idx", idx)
            try:
                source_idx = int(source_idx)
            except Exception:
                source_idx = idx
            source_indices.append(source_idx)

        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        progress_state.update(status="loading_model", updated_at=now_iso())
        write_json_atomic(args.progress_json_path, progress_state)

        ensure_qwen35_config_registered()
        ensure_qwen35_preprocessor_files(args.pretrain)
        llm = LLM(
            model=args.pretrain,
            tensor_parallel_size=args.tp_size,
            trust_remote_code=True,
            seed=args.seed,
            max_num_seqs=args.max_num_seqs,
            enable_prefix_caching=args.enable_prefix_caching,
            hf_overrides=qwen35_hf_overrides,
        )

        tokenizer = llm.get_tokenizer()
        model_prompts = []
        truncated_count = 0
        for prompt in tqdm(prompts, desc="Truncating prompts", dynamic_ncols=True):
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(token_ids) > args.prompt_max_len:
                token_ids = token_ids[: args.prompt_max_len]
                prompt_for_model = tokenizer.decode(token_ids, skip_special_tokens=False)
                truncated_count += 1
            else:
                prompt_for_model = prompt
            model_prompts.append(prompt_for_model)

        total_outputs = len(model_prompts) * args.best_of_n
        chunk_count = (len(model_prompts) + args.progress_batch_size - 1) // args.progress_batch_size

        progress_state.update(
            status="generating",
            updated_at=now_iso(),
            total_prompts=len(model_prompts),
            total_outputs=total_outputs,
            remaining_outputs=total_outputs,
            chunk_count=chunk_count,
            prompts_truncated=truncated_count,
        )
        write_json_atomic(args.progress_json_path, progress_state)

        print("=" * 72)
        print("vLLM Generation With Progress")
        print("=" * 72)
        print(f"Model path:             {args.pretrain}")
        print(f"Dataset path:           {args.dataset}")
        print(f"Loaded prompts:         {len(prompts)}")
        print(f"Expected outputs:       {total_outputs}")
        print(f"Tensor parallel size:   {args.tp_size}")
        print(f"Max num seqs:           {args.max_num_seqs}")
        print(f"Progress batch size:    {args.progress_batch_size}")
        print(f"Prompt max len:         {args.prompt_max_len}")
        print(f"Prompts truncated:      {truncated_count}")
        print(f"Max new tokens:         {args.max_new_tokens}")
        print(f"Output path:            {args.output_path}")
        if args.progress_json_path:
            print(f"Progress JSON:          {args.progress_json_path}")
        print("=" * 72)

        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

        started_at = time.time()
        written = 0
        progress_state.update(status="running", updated_at=now_iso())
        write_json_atomic(args.progress_json_path, progress_state)

        progress = tqdm(total=total_outputs, desc="Generating", unit="sample", dynamic_ncols=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            for chunk_idx, (start_idx, model_chunk_prompts) in enumerate(
                chunked(model_prompts, args.progress_batch_size), start=1
            ):
                original_chunk_prompts = prompts[start_idx : start_idx + len(model_chunk_prompts)]
                chunk_source_indices = source_indices[start_idx : start_idx + len(model_chunk_prompts)]
                repeated_prompts = model_chunk_prompts * args.best_of_n
                repeated_original_prompts = original_chunk_prompts * args.best_of_n
                repeated_source_indices = chunk_source_indices * args.best_of_n
                repeated_attempt_indices = []
                for attempt_idx in range(args.best_of_n):
                    repeated_attempt_indices.extend([attempt_idx] * len(chunk_source_indices))

                outputs = llm.generate(repeated_prompts, sampling_params)

                for original_prompt, source_idx, attempt_idx, output in zip(
                    repeated_original_prompts,
                    repeated_source_indices,
                    repeated_attempt_indices,
                    outputs,
                ):
                    record = {
                        "source_idx": source_idx,
                        "attempt_idx": attempt_idx,
                        "input": original_prompt,
                        "output": output.outputs[0].text if output.outputs else "",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                f.flush()

                written += len(outputs)
                elapsed = max(time.time() - started_at, 1e-6)
                remaining_outputs = max(total_outputs - written, 0)
                percent = (written / total_outputs * 100.0) if total_outputs else 100.0

                progress.update(len(outputs))
                progress.set_postfix(
                    chunk=f"{chunk_idx}/{chunk_count}",
                    written=written,
                    rate=f"{written / elapsed:.2f}/s",
                )

                progress_state.update(
                    status="running",
                    updated_at=now_iso(),
                    written=written,
                    remaining_outputs=remaining_outputs,
                    percent=percent,
                    chunk_idx=chunk_idx,
                    last_chunk_outputs=len(outputs),
                    rate_samples_per_sec=written / elapsed,
                    elapsed_seconds=elapsed,
                )
                write_json_atomic(args.progress_json_path, progress_state)

        progress.close()
        progress_state.update(
            status="completed",
            updated_at=now_iso(),
            finished_at=now_iso(),
            written=written,
            remaining_outputs=0,
            percent=100.0,
            chunk_idx=chunk_count,
            rate_samples_per_sec=(written / max(time.time() - started_at, 1e-6)) if written else 0.0,
            elapsed_seconds=max(time.time() - started_at, 0.0),
        )
        write_json_atomic(args.progress_json_path, progress_state)
        print(f"[done] wrote {written} rows to {args.output_path}")
    except Exception as exc:
        progress_state.update(
            status="failed",
            updated_at=now_iso(),
            finished_at=now_iso(),
            error=str(exc),
        )
        write_json_atomic(args.progress_json_path, progress_state)
        raise


if __name__ == "__main__":
    main()
