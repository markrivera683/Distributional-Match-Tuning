#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ARGS = {
    "num_layers": 36,
    "hidden_size": 2560,
    "ffn_hidden_size": 9728,
    "vocab_size": 151936,
    "num_attention_heads": 32,
    "num_query_groups": 8,
    "kv_channels": 128,
    "normalization": "RMSNorm",
    "norm_epsilon": 1e-6,
    "rotary_base": 1000000,
}


def module_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    status: dict[str, Any] = {"available": spec is not None, "origin": spec.origin if spec else None, "version": None}
    if spec is None:
        return status
    for dist_name in (name, name.replace("_", "-")):
        try:
            status["version"] = importlib.metadata.version(dist_name)
            break
        except importlib.metadata.PackageNotFoundError:
            pass
    return status


def load_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {"available": False, "error": f"missing {config_path}"}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = data.get("text_config") if isinstance(data.get("text_config"), dict) else data

    def pick(*names: str) -> Any:
        for name in names:
            if name in cfg:
                return cfg[name]
            if name in data:
                return data[name]
        return None

    return {
        "available": True,
        "model_type": pick("model_type"),
        "num_hidden_layers": pick("num_hidden_layers"),
        "hidden_size": pick("hidden_size"),
        "intermediate_size": pick("intermediate_size"),
        "vocab_size": pick("vocab_size"),
        "num_attention_heads": pick("num_attention_heads"),
        "num_key_value_heads": pick("num_key_value_heads"),
        "rms_norm_eps": pick("rms_norm_eps"),
        "rope_theta": pick("rope_theta") or (pick("rope_parameters") or {}).get("rope_theta") if isinstance(pick("rope_parameters"), dict) else pick("rope_theta"),
        "head_dim": pick("head_dim"),
        "layer_types": pick("layer_types"),
        "torch_dtype": pick("torch_dtype") or pick("dtype"),
    }


def infer_model_args(config: dict[str, Any]) -> dict[str, Any]:
    if not config.get("available"):
        return DEFAULT_MODEL_ARGS.copy()
    hidden = config.get("hidden_size") or DEFAULT_MODEL_ARGS["hidden_size"]
    heads = config.get("num_attention_heads") or DEFAULT_MODEL_ARGS["num_attention_heads"]
    return {
        "num_layers": config.get("num_hidden_layers") or DEFAULT_MODEL_ARGS["num_layers"],
        "hidden_size": hidden,
        "ffn_hidden_size": config.get("intermediate_size") or DEFAULT_MODEL_ARGS["ffn_hidden_size"],
        "vocab_size": config.get("vocab_size") or DEFAULT_MODEL_ARGS["vocab_size"],
        "num_attention_heads": heads,
        "num_query_groups": config.get("num_key_value_heads") or DEFAULT_MODEL_ARGS["num_query_groups"],
        "kv_channels": hidden // heads,
        "normalization": "RMSNorm",
        "norm_epsilon": config.get("rms_norm_eps") or DEFAULT_MODEL_ARGS["norm_epsilon"],
        "rotary_base": config.get("rope_theta") or DEFAULT_MODEL_ARGS["rotary_base"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit slime/Megatron/SGLang runtime availability.")
    parser.add_argument("--model-path", default="/mnt/data/models/Qwen3.5-4B")
    parser.add_argument("--slime-root", default=os.environ.get("SLIME_ROOT", "/root/slime"))
    parser.add_argument("--megatron-path", default=os.environ.get("MEGATRON_PATH", "/root/Megatron-LM"))
    parser.add_argument("--output", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.model_path))
    model_args = infer_model_args(config)
    report = {
        "python": sys.executable,
        "python_version": sys.version,
        "commands": {name: shutil.which(name) for name in ("slime", "sglang", "ray", "python")},
        "modules": {name: module_status(name) for name in ("slime", "sglang", "megatron")},
        "paths": {
            "slime_root": args.slime_root,
            "slime_train_py_exists": (Path(args.slime_root) / "train.py").exists(),
            "megatron_path": args.megatron_path,
            "megatron_path_exists": Path(args.megatron_path).exists(),
        },
        "hf_config": config,
        "recommended_model_args": model_args,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.strict and (not report["paths"]["slime_train_py_exists"] or not report["paths"]["megatron_path_exists"]):
        raise SystemExit("slime or Megatron runtime is missing")


if __name__ == "__main__":
    main()
