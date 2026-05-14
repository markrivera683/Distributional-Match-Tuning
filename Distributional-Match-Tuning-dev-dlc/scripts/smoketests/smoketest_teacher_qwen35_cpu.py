#!/usr/bin/env python3
"""CPU-only smoke test for the teacher venv + local Qwen3.5-27B model.

This script is intentionally lightweight:
- does not instantiate a vLLM engine
- does not load model weights into memory
- does not require CUDA

It validates that the current Python environment can:
- import the expected vLLM / transformers modules
- parse the local model config
- load tokenizer / processor metadata
- see all safetensors shards referenced by the index file
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as importlib_metadata
import json
import os
import sys
import warnings
from pathlib import Path


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def require(condition: bool, msg: str) -> None:
    if not condition:
        raise RuntimeError(msg)


def file_size_gb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only teacher smoke test")
    parser.add_argument(
        "--model-path",
        default="/mnt/data/models/qwen3.5-27b",
        help="Local Hugging Face model directory",
    )
    args = parser.parse_args()

    # Keep this script CPU-only.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    warnings.filterwarnings("ignore", category=FutureWarning)

    model_path = Path(args.model_path)
    info(f"model path: {model_path}")
    require(model_path.is_dir(), f"model directory not found: {model_path}")

    vllm_version = importlib_metadata.version("vllm")
    transformers_version = importlib_metadata.version("transformers")
    torch_version = importlib_metadata.version("torch")

    info(f"vllm={vllm_version}")
    info(f"transformers={transformers_version}")
    info(f"torch={torch_version}")
    if transformers_version.startswith("5."):
        warn(
            "transformers is a v5 build. This is expected for modern Qwen3.5 support, "
            "but note that the installed vLLM wheel metadata still advertises transformers<5."
        )

    required_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ]
    for name in required_files:
        path = model_path / name
        require(path.is_file(), f"missing required file: {path}")
        ok(f"found {name}")

    raw_cfg = json.loads((model_path / "config.json").read_text())
    require(raw_cfg.get("model_type") == "qwen3_5", f"unexpected model_type: {raw_cfg.get('model_type')}")
    archs = raw_cfg.get("architectures") or []
    require("Qwen3_5ForConditionalGeneration" in archs, f"unexpected architectures: {archs}")
    ok("config.json advertises Qwen3.5 conditional generation")

    text_cfg = raw_cfg.get("text_config") or {}
    require(text_cfg.get("model_type") == "qwen3_5_text", "unexpected text_config.model_type")
    ok("text_config model_type is qwen3_5_text")

    shard_index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_map = shard_index.get("weight_map") or {}
    require(weight_map, "empty weight_map in model.safetensors.index.json")
    shard_names = sorted(set(weight_map.values()))
    require(shard_names, "no shard files referenced by safetensors index")
    ok(f"safetensors index references {len(shard_names)} shard files")

    missing_shards = [name for name in shard_names if not (model_path / name).is_file()]
    require(not missing_shards, f"missing shard files: {missing_shards[:5]}")
    total_shard_gb = sum(file_size_gb(model_path / name) for name in shard_names)
    ok(f"all shard files exist ({total_shard_gb:.1f} GiB total)")

    transformers = importlib.import_module("transformers")
    AutoConfig = getattr(transformers, "AutoConfig")
    AutoTokenizer = getattr(transformers, "AutoTokenizer")
    AutoProcessor = getattr(transformers, "AutoProcessor")

    cfg = AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    require(getattr(cfg, "model_type", None) == "qwen3_5", "AutoConfig loaded an unexpected model_type")
    ok("transformers AutoConfig can load the local Qwen3.5 config")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello from smoke test."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    require(isinstance(rendered, str) and rendered, "tokenizer chat template render failed")
    ok("transformers AutoTokenizer loads and renders chat template")

    try:
        processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        ok(f"transformers AutoProcessor loads ({processor.__class__.__name__})")
    except Exception as exc:
        warn(f"AutoProcessor load failed (non-fatal for text-only serving): {exc}")

    importlib.import_module("vllm")
    importlib.import_module("vllm.model_executor.models.qwen3_5")
    importlib.import_module("vllm.transformers_utils.configs.qwen3_5")
    ok("vLLM imports the Qwen3.5 implementation modules")

    print()
    print("Summary")
    print("- Environment import path looks usable for a Qwen3.5-27B teacher.")
    print("- This script did NOT instantiate vLLM or touch CUDA, so it is not a serving proof.")
    print("- The next validation step, if needed, is a tiny serve/import test with GPU enabled.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        fail(str(exc))
        raise SystemExit(1)
