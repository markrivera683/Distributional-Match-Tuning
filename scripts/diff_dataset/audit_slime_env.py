#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
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
    head_dim = config.get("head_dim")
    return {
        "num_layers": config.get("num_hidden_layers") or DEFAULT_MODEL_ARGS["num_layers"],
        "hidden_size": hidden,
        "ffn_hidden_size": config.get("intermediate_size") or DEFAULT_MODEL_ARGS["ffn_hidden_size"],
        "vocab_size": config.get("vocab_size") or DEFAULT_MODEL_ARGS["vocab_size"],
        "num_attention_heads": heads,
        "num_query_groups": config.get("num_key_value_heads") or DEFAULT_MODEL_ARGS["num_query_groups"],
        "kv_channels": head_dim or hidden // heads,
        "normalization": "RMSNorm",
        "norm_epsilon": config.get("rms_norm_eps") or DEFAULT_MODEL_ARGS["norm_epsilon"],
        "rotary_base": config.get("rope_theta") or DEFAULT_MODEL_ARGS["rotary_base"],
    }


def load_model_args_script(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "error": "missing model args script"}
    command = f"source {str(path)!r}; printf '%s\\n' \"${{MODEL_ARGS[@]}}\""
    try:
        result = subprocess.run(["bash", "--noprofile", "--norc", "-c", command], check=True, capture_output=True, text=True)
    except Exception as exc:
        return {"available": False, "path": str(path), "error": str(exc)}
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    parsed: dict[str, Any] = {"available": True, "path": str(path), "raw": values}
    idx = 0
    while idx < len(values):
        item = values[idx]
        if item.startswith("--"):
            key = item[2:].replace("-", "_")
            if idx + 1 < len(values) and not values[idx + 1].startswith("--"):
                parsed[key] = values[idx + 1]
                idx += 2
            else:
                parsed[key] = True
                idx += 1
        else:
            idx += 1
    return parsed


def checkpoint_status(path: Path) -> dict[str, Any]:
    latest_step = None
    latest_dir = None
    latest_file = path / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        latest_step = latest_file.read_text(encoding="utf-8").strip()
        if latest_step:
            try:
                latest_dir = path / f"iter_{int(latest_step):07d}"
            except ValueError:
                latest_dir = path / latest_step
    return {
        "path": str(path),
        "exists": path.exists(),
        "common_pt": (path / "common.pt").exists(),
        "latest_checkpointed_iteration": (path / "latest_checkpointed_iteration.txt").exists(),
        "latest_step": latest_step,
        "latest_dir": str(latest_dir) if latest_dir else None,
        "latest_dir_exists": latest_dir.exists() if latest_dir else False,
        "latest_dir_common_pt": (latest_dir / "common.pt").exists() if latest_dir else False,
        "metadata": (path / ".metadata").exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit slime/Megatron/SGLang runtime availability.")
    parser.add_argument("--model-path", default="/mnt/data/models/Qwen3.5-4B")
    parser.add_argument("--slime-root", default=os.environ.get("SLIME_ROOT", "/root/slime"))
    parser.add_argument("--megatron-path", default=os.environ.get("MEGATRON_PATH", "/root/Megatron-LM"))
    parser.add_argument("--model-args-script", default=os.environ.get("MODEL_ARGS_SCRIPT", ""))
    parser.add_argument("--ref-load", default=os.environ.get("REF_LOAD", ""))
    parser.add_argument("--output", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    slime_root = Path(args.slime_root)
    model_path = Path(args.model_path)
    model_name = model_path.name
    model_args_script = Path(args.model_args_script) if args.model_args_script else slime_root / "slime" / "scripts" / "models" / f"{model_name}.sh"
    ref_load = Path(args.ref_load) if args.ref_load else Path("/root/slime_runtime/checkpoints") / f"{model_name}_torch_dist"

    config = load_config(Path(args.model_path))
    model_args = infer_model_args(config)
    scripted_args = load_model_args_script(model_args_script)
    report = {
        "python": sys.executable,
        "python_version": sys.version,
        "commands": {name: shutil.which(name) for name in ("slime", "sglang", "ray", "python")},
        "modules": {name: module_status(name) for name in ("slime", "sglang", "megatron")},
        "paths": {
            "slime_root": args.slime_root,
            "slime_train_py_exists": (slime_root / "train.py").exists(),
            "slime_train_async_py_exists": (slime_root / "train_async.py").exists(),
            "convert_hf_to_torch_dist_exists": (slime_root / "tools" / "convert_hf_to_torch_dist.py").exists(),
            "convert_torch_dist_to_hf_exists": (slime_root / "tools" / "convert_torch_dist_to_hf.py").exists(),
            "megatron_path": args.megatron_path,
            "megatron_path_exists": Path(args.megatron_path).exists(),
            "megatron_package_exists": (Path(args.megatron_path) / "megatron").exists(),
        },
        "hf_config": config,
        "inferred_model_args": model_args,
        "model_args_script": scripted_args,
        "ref_load": checkpoint_status(ref_load),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.strict:
        failures = []
        if not report["paths"]["slime_train_py_exists"]:
            failures.append("missing slime train.py")
        if not report["paths"]["megatron_path_exists"]:
            failures.append("missing Megatron path")
        if not scripted_args.get("available"):
            failures.append("missing MODEL_ARGS_SCRIPT")
        if not report["ref_load"]["exists"]:
            failures.append("missing REF_LOAD")
        if failures:
            raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
