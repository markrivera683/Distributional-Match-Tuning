#!/usr/bin/env python3
"""Chunked vLLM generator with explicit real-time progress reporting."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from functools import partial
from typing import Any

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

from qwen35_text_only_shim import (
    QWEN35_TEXT_ONLY_SHIM_ARCH,
    ensure_qwen35_text_only_shim_registered,
    prepare_qwen35_text_only_shim_env,
)


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


def append_rank_suffix(path: str | None, rank: int) -> str | None:
    if path is None:
        return None
    root, ext = os.path.splitext(path)
    return f"{root}.rank{rank}{ext}"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def record_sort_key(record: dict[str, Any]) -> tuple[int, int]:
    return (
        safe_int(record.get("source_idx", record.get("idx", 0))),
        safe_int(record.get("attempt_idx", 0)),
    )


def next_json_record(handle) -> dict[str, Any] | None:
    for line in handle:
        line = line.strip()
        if line:
            return json.loads(line)
    return None


def load_json_file_with_retries(
    path: str, attempts: int = 5, initial_delay_seconds: float = 0.05
) -> dict[str, Any] | None:
    delay_seconds = initial_delay_seconds
    last_error: Exception | None = None

    for attempt_idx in range(attempts):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                raise json.JSONDecodeError("Empty JSON content", content, 0)
            return json.loads(content)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt_idx == attempts - 1:
                break
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 0.5)

    print(
        f"[warn] skipping unreadable progress shard after {attempts} attempts: "
        f"{path} ({last_error})"
    )
    return None


def merge_rank_output_shards(output_path: str, dp_size: int) -> int:
    rank_paths = [append_rank_suffix(output_path, rank) for rank in range(dp_size)]
    handles: dict[int, Any] = {}
    heap: list[tuple[tuple[int, int], int, dict[str, Any]]] = []
    merged_count = 0

    try:
        for rank, rank_path in enumerate(rank_paths):
            assert rank_path is not None
            if not os.path.isfile(rank_path):
                raise FileNotFoundError(f"Missing DP shard output: {rank_path}")
            handle = open(rank_path, "r", encoding="utf-8")
            handles[rank] = handle
            record = next_json_record(handle)
            if record is not None:
                heapq.heappush(heap, (record_sort_key(record), rank, record))

        with open(output_path, "w", encoding="utf-8") as out:
            while heap:
                _, rank, record = heapq.heappop(heap)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                merged_count += 1
                record = next_json_record(handles[rank])
                if record is not None:
                    heapq.heappush(heap, (record_sort_key(record), rank, record))
    finally:
        for handle in handles.values():
            handle.close()

    return merged_count


def aggregate_progress_from_rank_files(
    progress_json_path: str | None,
    dp_size: int,
    base_state: dict[str, Any],
) -> dict[str, Any]:
    merged_state = dict(base_state)
    if not progress_json_path or dp_size <= 1:
        return merged_state

    local_states: list[dict[str, Any]] = []
    for rank in range(dp_size):
        rank_path = append_rank_suffix(progress_json_path, rank)
        if rank_path and os.path.isfile(rank_path):
            state = load_json_file_with_retries(rank_path)
            if state is not None:
                local_states.append(state)

    if not local_states:
        return merged_state

    total_outputs = safe_int(merged_state.get("total_outputs", 0))
    written = sum(safe_int(state.get("written", 0)) for state in local_states)
    remaining_outputs = max(total_outputs - written, 0)
    statuses = [str(state.get("status", "unknown")) for state in local_states]

    merged_state.update(
        written=written,
        remaining_outputs=remaining_outputs,
        percent=(written / total_outputs * 100.0) if total_outputs else 100.0,
        chunk_idx=sum(safe_int(state.get("chunk_idx", 0)) for state in local_states),
        chunk_count=sum(safe_int(state.get("chunk_count", 0)) for state in local_states),
        last_chunk_outputs=sum(
            safe_int(state.get("last_chunk_outputs", 0)) for state in local_states
        ),
        prompts_truncated=sum(
            safe_int(state.get("prompts_truncated", 0)) for state in local_states
        ),
        rate_samples_per_sec=sum(
            float(state.get("rate_samples_per_sec", 0.0) or 0.0) for state in local_states
        ),
        elapsed_seconds=max(
            float(state.get("elapsed_seconds", 0.0) or 0.0) for state in local_states
        ),
        dp_local_statuses=statuses,
        dp_reporting_ranks=len(local_states),
    )

    if any(status == "failed" for status in statuses):
        merged_state["status"] = "failed"
        for state in local_states:
            if state.get("status") == "failed":
                merged_state["error"] = state.get("error")
                break
    elif len(local_states) == dp_size and all(status == "completed" for status in statuses):
        merged_state["status"] = "completed"
        merged_state["finished_at"] = now_iso()
    elif any(status == "running" for status in statuses):
        merged_state["status"] = "running"
    elif any(status == "generating" for status in statuses):
        merged_state["status"] = "generating"
    elif any(status == "loading_model" for status in statuses):
        merged_state["status"] = "loading_model"

    return merged_state


def wait_for_rank_done_markers(output_path: str, dp_size: int, timeout_seconds: int) -> None:
    deadline = time.time() + max(timeout_seconds, 1)
    while True:
        all_ready = True
        for rank in range(dp_size):
            rank_path = append_rank_suffix(output_path, rank)
            marker_path = f"{rank_path}.done"
            if not os.path.isfile(marker_path):
                all_ready = False
                break
        if all_ready:
            return
        if time.time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for DP rank completion markers for: {output_path}"
            )
        time.sleep(1.0)


def cleanup_rank_artifacts(path: str | None, dp_size: int) -> None:
    if not path or dp_size <= 1:
        return
    for rank in range(dp_size):
        rank_path = append_rank_suffix(path, rank)
        if rank_path and os.path.exists(rank_path):
            os.remove(rank_path)
        marker_path = f"{rank_path}.done" if rank_path else None
        if marker_path and os.path.exists(marker_path):
            os.remove(marker_path)


def ensure_qwen35_config_registered() -> None:
    """Register qwen3.5 config aliases for older transformers builds."""
    try:
        from transformers import AutoConfig

        AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)
        AutoConfig.register("qwen3_5", Qwen3_5Config, exist_ok=True)
    except Exception:
        # Best-effort only; continue if registration API differs.
        pass

def qwen35_hf_overrides(cfg: Any, *, enable_text_only_shim: bool = False) -> Any:
    """Normalize qwen3_5_text config for vLLM multimodal registry.

    Some model checkpoints expose a text-only HF config (`qwen3_5_text`), while
    vLLM renderer paths expect the wrapper config type (`Qwen3_5Config`).
    """
    try:
        model_type = getattr(cfg, "model_type", None)
        if enable_text_only_shim and model_type == "qwen3_5":
            ensure_qwen35_text_only_shim_registered()
            cfg.architectures = [QWEN35_TEXT_ONLY_SHIM_ARCH]
            return cfg

        if model_type == "qwen3_5_text" and not isinstance(cfg, Qwen3_5Config):
            text_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
            # Remove generic metadata keys that do not belong to text_config.
            text_cfg.pop("model_type", None)
            text_cfg.pop("transformers_version", None)
            wrapped = Qwen3_5Config(text_config=text_cfg)
            # Preserve architecture hint expected by vLLM model registry.
            if enable_text_only_shim:
                ensure_qwen35_text_only_shim_registered()
                wrapped.architectures = [QWEN35_TEXT_ONLY_SHIM_ARCH]
            else:
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


def load_local_model_config(model_path: str) -> dict[str, Any] | None:
    """Load a local HuggingFace config.json when available."""
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
    elif os.path.isfile(model_path) and model_path.endswith(".json"):
        config_path = model_path
    else:
        return None

    if not os.path.isfile(config_path):
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Expected JSON object in config file: {config_path}")
    return config


def inspect_qwen35_conditional_config(model_path: str) -> dict[str, Any] | None:
    config = load_local_model_config(model_path)
    if config is None:
        return None

    architectures = [str(arch) for arch in (config.get("architectures") or []) if arch]
    model_type = str(config.get("model_type") or "")
    vision_config = config.get("vision_config") or {}
    text_config = config.get("text_config") or {}
    vision_heads = vision_config.get("num_heads")
    conditional_arches = {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
    is_qwen35_conditional = (
        model_type == "qwen3_5"
        and isinstance(vision_config, dict)
        and isinstance(text_config, dict)
        and isinstance(vision_heads, int)
        and (
            any(arch in conditional_arches for arch in architectures)
            or "vision_start_token_id" in config
            or "image_token_id" in config
        )
    )
    if not is_qwen35_conditional:
        return None

    return {
        "architectures": architectures,
        "model_type": model_type,
        "vision_heads": vision_heads,
        "text_config": text_config,
    }


def qwen35_text_only_tp_is_compatible(
    text_config: dict[str, Any], tp_size: int
) -> tuple[bool, str]:
    num_attention_heads = text_config.get("num_attention_heads")
    num_key_value_heads = text_config.get("num_key_value_heads")
    linear_num_key_heads = text_config.get("linear_num_key_heads")
    linear_num_value_heads = text_config.get("linear_num_value_heads")
    linear_key_head_dim = text_config.get("linear_key_head_dim")
    linear_value_head_dim = text_config.get("linear_value_head_dim")

    if not isinstance(num_attention_heads, int) or num_attention_heads <= 0:
        return False, f"invalid text num_attention_heads={num_attention_heads!r}"
    if num_attention_heads % tp_size != 0:
        return False, f"text num_attention_heads={num_attention_heads} is not divisible by tp_size={tp_size}"

    if not isinstance(num_key_value_heads, int) or num_key_value_heads <= 0:
        return False, f"invalid text num_key_value_heads={num_key_value_heads!r}"
    if num_key_value_heads >= tp_size:
        if num_key_value_heads % tp_size != 0:
            return False, (
                f"text num_key_value_heads={num_key_value_heads} is not divisible by "
                f"tp_size={tp_size}"
            )
    elif tp_size % num_key_value_heads != 0:
        return False, (
            f"text num_key_value_heads={num_key_value_heads} cannot be replicated across "
            f"tp_size={tp_size}"
        )

    if not isinstance(linear_num_key_heads, int) or linear_num_key_heads <= 0:
        return False, f"invalid text linear_num_key_heads={linear_num_key_heads!r}"
    if not isinstance(linear_num_value_heads, int) or linear_num_value_heads <= 0:
        return False, f"invalid text linear_num_value_heads={linear_num_value_heads!r}"
    if linear_num_value_heads % tp_size != 0:
        return False, (
            f"text linear_num_value_heads={linear_num_value_heads} is not divisible by "
            f"tp_size={tp_size}"
        )

    if not isinstance(linear_key_head_dim, int) or linear_key_head_dim <= 0:
        return False, f"invalid text linear_key_head_dim={linear_key_head_dim!r}"
    if not isinstance(linear_value_head_dim, int) or linear_value_head_dim <= 0:
        return False, f"invalid text linear_value_head_dim={linear_value_head_dim!r}"

    conv_dim = (
        linear_key_head_dim * linear_num_key_heads * 2
        + linear_value_head_dim * linear_num_value_heads
    )
    if conv_dim % tp_size != 0:
        return False, f"text gated-delta conv_dim={conv_dim} is not divisible by tp_size={tp_size}"

    return True, ""


def fail_fast_on_qwen35_tp_incompatibility(
    model_path: str, tp_size: int, *, enable_text_only_shim: bool = False
) -> dict[str, Any] | None:
    """Raise a clear error before LLM init for known Qwen3.5 TP mismatches."""
    info = inspect_qwen35_conditional_config(model_path)
    if info is None:
        return

    architectures = info["architectures"]
    model_type = info["model_type"]
    vision_heads = info["vision_heads"]

    if vision_heads % tp_size == 0:
        return

    text_compatible, text_reason = qwen35_text_only_tp_is_compatible(
        info["text_config"], tp_size
    )
    if enable_text_only_shim and text_compatible:
        return {
            "architectures": architectures,
            "model_type": model_type,
            "vision_heads": vision_heads,
            "tp_size": tp_size,
        }

    arch_hint = ", ".join(architectures) if architectures else "unknown"
    text_only_hint = (
        "A repo-local text-only shim was also considered, but the text tower "
        f"is not TP-compatible here: {text_reason}\n"
        if text_reason
        else ""
    )
    raise ValueError(
        "Qwen3.5 conditional / multimodal TP compatibility check failed before "
        "LLM initialization.\n"
        f"Detected load path: model_type={model_type}, architectures={arch_hint}\n"
        f"Visual attention heads: {vision_heads}\n"
        f"Requested tp_size: {tp_size}\n"
        f"Why this fails: vLLM loads this checkpoint as a Qwen3.5 conditional / "
        f"multimodal model and initializes self.visual before generation, so the "
        f"visual attention heads must satisfy vision_heads % tp_size == 0. "
        f"Here {vision_heads} % {tp_size} != 0.\n"
        "This is a vLLM / Qwen3.5 conditional generation loading limitation, "
        "not a shell orchestration problem.\n"
        f"{text_only_hint}"
        "If you only want this repo path to run, change TP to a divisor of the "
        "visual head count.\n"
        "If you need to keep TP=8, that requires a text-only shim to bypass the "
        "visual tower."
    )


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
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--merge_timeout_seconds", type=int, default=3600)
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
    if args.tp_size < 1:
        raise ValueError("--tp_size must be >= 1")
    if args.dp_size < 1:
        raise ValueError("--dp_size must be >= 1")
    if args.pp_size < 1:
        raise ValueError("--pp_size must be >= 1")

    torch_rank = safe_int(os.environ.get("RANK"), 0)
    is_primary_process = torch_rank == 0
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
        "dp_size": args.dp_size,
        "pp_size": args.pp_size,
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
    if args.dp_size == 1 or is_primary_process:
        write_json_atomic(args.progress_json_path, progress_state)

    try:
        rows = load_dataset_rows(args.dataset)
        rows = rows[: min(args.max_samples, len(rows))]

        prompts = []
        source_indices = []
        for idx, row in enumerate(
            tqdm(rows, desc="Preparing prompts", dynamic_ncols=True, disable=not is_primary_process)
        ):
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
        if args.dp_size == 1 or is_primary_process:
            write_json_atomic(args.progress_json_path, progress_state)

        ensure_qwen35_config_registered()
        qwen35_text_only_shim = fail_fast_on_qwen35_tp_incompatibility(
            args.pretrain, args.tp_size, enable_text_only_shim=True
        )
        if qwen35_text_only_shim:
            prepare_qwen35_text_only_shim_env()
            progress_state["compat_mode"] = "qwen35_text_only_shim"
            if is_primary_process:
                arch_hint = ", ".join(qwen35_text_only_shim["architectures"]) or "unknown"
                print(
                    "[compat] enabling repo-local Qwen3.5 text-only shim to bypass "
                    "the visual tower"
                )
                print(
                    f"[compat] detected load path: model_type={qwen35_text_only_shim['model_type']}, "
                    f"architectures={arch_hint}"
                )
                print(
                    f"[compat] visual heads={qwen35_text_only_shim['vision_heads']}, "
                    f"tp_size={qwen35_text_only_shim['tp_size']}"
                )
        ensure_qwen35_preprocessor_files(args.pretrain)
        llm_kwargs: dict[str, Any] = dict(
            model=args.pretrain,
            tensor_parallel_size=args.tp_size,
            trust_remote_code=True,
            seed=args.seed,
            max_num_seqs=args.max_num_seqs,
            enable_prefix_caching=args.enable_prefix_caching,
            hf_overrides=partial(
                qwen35_hf_overrides,
                enable_text_only_shim=bool(qwen35_text_only_shim),
            ),
        )
        if args.pp_size > 1:
            llm_kwargs["pipeline_parallel_size"] = args.pp_size
        if args.dp_size > 1:
            llm_kwargs["data_parallel_size"] = args.dp_size
            llm_kwargs["distributed_executor_backend"] = "external_launcher"
        if args.gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
        llm = LLM(**llm_kwargs)

        dp_rank = 0
        dp_size = 1
        if args.dp_size > 1:
            parallel_config = llm.llm_engine.vllm_config.parallel_config
            dp_rank = parallel_config.data_parallel_rank
            dp_size = parallel_config.data_parallel_size

        tp_group_size = max(args.tp_size * args.pp_size, 1)
        is_dp_writer = torch_rank % tp_group_size == 0
        local_output_path = (
            append_rank_suffix(args.output_path, dp_rank) if dp_size > 1 else args.output_path
        )
        local_progress_json_path = (
            append_rank_suffix(args.progress_json_path, dp_rank)
            if dp_size > 1 and is_dp_writer
            else (args.progress_json_path if dp_size == 1 else None)
        )

        global_prompts = prompts
        global_source_indices = source_indices
        if dp_size > 1:
            selected_indices = [
                idx for idx in range(len(global_prompts)) if idx % dp_size == dp_rank
            ]
            prompts = [global_prompts[idx] for idx in selected_indices]
            source_indices = [global_source_indices[idx] for idx in selected_indices]

        tokenizer = llm.get_tokenizer()
        model_prompts = []
        truncated_count = 0
        for prompt in tqdm(
            prompts,
            desc="Truncating prompts",
            dynamic_ncols=True,
            disable=not is_primary_process,
        ):
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(token_ids) > args.prompt_max_len:
                token_ids = token_ids[: args.prompt_max_len]
                prompt_for_model = tokenizer.decode(token_ids, skip_special_tokens=False)
                truncated_count += 1
            else:
                prompt_for_model = prompt
            model_prompts.append(prompt_for_model)

        total_outputs = len(global_prompts) * args.best_of_n
        local_total_outputs = len(model_prompts) * args.best_of_n
        chunk_count = (len(model_prompts) + args.progress_batch_size - 1) // args.progress_batch_size

        progress_state.update(
            status="generating",
            updated_at=now_iso(),
            total_prompts=len(global_prompts),
            total_outputs=total_outputs,
            remaining_outputs=total_outputs,
            chunk_count=chunk_count,
            prompts_truncated=truncated_count if dp_size == 1 else 0,
            local_prompts=len(model_prompts),
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
            global_rank=torch_rank,
        )
        local_progress_state = dict(progress_state)
        local_progress_state["prompts_truncated"] = truncated_count
        if dp_size == 1:
            write_json_atomic(args.progress_json_path, local_progress_state)
        elif is_dp_writer and local_progress_json_path:
            write_json_atomic(local_progress_json_path, local_progress_state)
            if is_primary_process and args.progress_json_path:
                write_json_atomic(
                    args.progress_json_path,
                    aggregate_progress_from_rank_files(
                        args.progress_json_path, dp_size, progress_state
                    ),
                )

        if is_primary_process:
            print("=" * 72)
            print("vLLM Generation With Progress")
            print("=" * 72)
            print(f"Model path:             {args.pretrain}")
            print(f"Dataset path:           {args.dataset}")
            print(f"Loaded prompts:         {len(global_prompts)}")
            print(f"Expected outputs:       {total_outputs}")
            print(f"Tensor parallel size:   {args.tp_size}")
            print(f"Data parallel size:     {dp_size}")
            print(f"Max num seqs:           {args.max_num_seqs}")
            print(f"Progress batch size:    {args.progress_batch_size}")
            print(f"Prompt max len:         {args.prompt_max_len}")
            print(f"Prompts truncated:      {truncated_count}")
            print(f"Local prompts(rank0):   {len(model_prompts)}")
            print(f"Max new tokens:         {args.max_new_tokens}")
            print(f"Output path:            {args.output_path}")
            if dp_size > 1:
                print(f"Writer shard path:      {local_output_path}")
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
        if dp_size == 1:
            write_json_atomic(args.progress_json_path, progress_state)
        elif is_dp_writer and local_progress_json_path:
            local_progress_state["status"] = "running"
            local_progress_state["updated_at"] = now_iso()
            write_json_atomic(local_progress_json_path, local_progress_state)
            if is_primary_process and args.progress_json_path:
                write_json_atomic(
                    args.progress_json_path,
                    aggregate_progress_from_rank_files(
                        args.progress_json_path, dp_size, progress_state
                    ),
                )

        progress = tqdm(
            total=total_outputs if dp_size > 1 else local_total_outputs,
            desc="Generating",
            unit="sample",
            dynamic_ncols=True,
            disable=not is_primary_process,
        )
        progress_written = 0
        output_path_for_rank = (
            local_output_path if (dp_size == 1 or is_dp_writer) else os.devnull
        )
        with open(output_path_for_rank, "w", encoding="utf-8") as f:
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

                if dp_size == 1 or is_dp_writer:
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
                if dp_size == 1:
                    remaining_outputs = max(local_total_outputs - written, 0)
                    percent = (written / local_total_outputs * 100.0) if local_total_outputs else 100.0
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
                elif is_dp_writer and local_progress_json_path:
                    local_progress_state.update(
                        status="running",
                        updated_at=now_iso(),
                        written=written,
                        remaining_outputs=max(local_total_outputs - written, 0),
                        percent=(written / local_total_outputs * 100.0) if local_total_outputs else 100.0,
                        chunk_idx=chunk_idx,
                        chunk_count=chunk_count,
                        last_chunk_outputs=len(outputs),
                        rate_samples_per_sec=written / elapsed,
                        elapsed_seconds=elapsed,
                        local_prompts=len(model_prompts),
                        local_total_outputs=local_total_outputs,
                    )
                    write_json_atomic(local_progress_json_path, local_progress_state)
                    if is_primary_process and args.progress_json_path:
                        aggregate_state = aggregate_progress_from_rank_files(
                            args.progress_json_path, dp_size, progress_state
                        )
                        write_json_atomic(args.progress_json_path, aggregate_state)
                        new_written = safe_int(aggregate_state.get("written", 0))
                        progress.update(max(new_written - progress_written, 0))
                        progress_written = new_written
                        progress.set_postfix(
                            chunk=f"{safe_int(aggregate_state.get('chunk_idx', 0))}/"
                                  f"{safe_int(aggregate_state.get('chunk_count', 0))}",
                            written=new_written,
                            rate=f"{float(aggregate_state.get('rate_samples_per_sec', 0.0) or 0.0):.2f}/s",
                        )

        progress.close()
        if dp_size == 1:
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
        else:
            if is_dp_writer and local_progress_json_path:
                local_progress_state.update(
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
                write_json_atomic(local_progress_json_path, local_progress_state)
                with open(f"{output_path_for_rank}.done", "w", encoding="utf-8") as marker:
                    marker.write("done\n")

            if is_primary_process:
                wait_for_rank_done_markers(args.output_path, dp_size, args.merge_timeout_seconds)
                merged_count = merge_rank_output_shards(args.output_path, dp_size)
                aggregate_state = aggregate_progress_from_rank_files(
                    args.progress_json_path, dp_size, progress_state
                )
                aggregate_state.update(
                    status="completed",
                    updated_at=now_iso(),
                    finished_at=now_iso(),
                    written=merged_count,
                    remaining_outputs=max(total_outputs - merged_count, 0),
                    percent=(merged_count / total_outputs * 100.0) if total_outputs else 100.0,
                )
                if args.progress_json_path:
                    write_json_atomic(args.progress_json_path, aggregate_state)
                cleanup_rank_artifacts(args.output_path, dp_size)
                cleanup_rank_artifacts(args.progress_json_path, dp_size)
                print(f"[done] merged {merged_count} rows to {args.output_path}")
    except Exception as exc:
        progress_state.update(
            status="failed",
            updated_at=now_iso(),
            finished_at=now_iso(),
            error=str(exc),
        )
        if args.dp_size == 1:
            write_json_atomic(args.progress_json_path, progress_state)
        else:
            global_rank = safe_int(os.environ.get("RANK"), 0)
            tp_group_size = max(args.tp_size * args.pp_size, 1)
            dp_rank = global_rank // tp_group_size
            is_dp_writer = global_rank % tp_group_size == 0
            rank_progress_path = append_rank_suffix(args.progress_json_path, dp_rank)
            if is_dp_writer and rank_progress_path:
                write_json_atomic(rank_progress_path, progress_state)
            if is_primary_process and args.progress_json_path:
                write_json_atomic(args.progress_json_path, progress_state)
        raise


if __name__ == "__main__":
    main()
