"""Custom slime reward functions for EBFT-style feature rewards.

This module is intentionally conservative: it can be used as a normal slime
``--custom-rm-path`` reward, while preserving a simple exact/non-empty fallback
when no feature model is configured.

Environment knobs used by the module:
- EBFT_RM_MODE: exact | nonempty | pointwise | cf_l1oo
- EBFT_FEATURE_MODEL_PATH: HF model used as the frozen feature extractor
- EBFT_TOKENIZER_PATH: optional tokenizer path; defaults to feature model path
- EBFT_EMBED_METHOD: last_token | mean_pooling
- EBFT_CF_TARGET_MODE: single | vicinal | teacher
- EBFT_CF_*: cf_num_freqs, cf_sigma, cf_seed, cf_alpha, cf_beta, cf_reward_scale,
  cf_target_num_refs, cf_target_std, cf_target_seed

For CF teacher mode, samples may provide metadata["teacher_responses"] as a list
of strings. The current scaffold embeds those teacher responses with the same
feature model and passes them to openrlhf.utils.embedding_utils.get_cf_l1oo_rewards.
"""

from __future__ import annotations

import asyncio
import os
import re
from functools import lru_cache
from typing import Any, Iterable


def _get_field(sample: Any, name: str, default: Any = None) -> Any:
    if hasattr(sample, name):
        value = getattr(sample, name)
        if value is not None:
            return value
    if isinstance(sample, dict) and sample.get(name) is not None:
        return sample.get(name)
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict) and metadata.get(name) is not None:
        return metadata.get(name)
    if isinstance(sample, dict) and isinstance(sample.get("metadata"), dict):
        return sample["metadata"].get(name, default)
    return default


def _sample_response(sample: Any) -> str:
    for name in ("response", "output", "answer", "completion"):
        value = _get_field(sample, name)
        if value is not None:
            return str(value)
    return ""


def _sample_label(sample: Any) -> str:
    for name in ("label", "answer", "reference", "target"):
        value = _get_field(sample, name)
        if value is not None:
            return str(value)
    return ""


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _exact_reward(sample: Any) -> float:
    label = _normalize_text(_sample_label(sample))
    response = _normalize_text(_sample_response(sample))
    if not label or not response:
        return 0.0
    return 1.0 if label in response or response.endswith(label) else 0.0


def _nonempty_reward(sample: Any) -> float:
    response = _sample_response(sample).strip()
    return 1.0 if response else 0.0


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@lru_cache(maxsize=1)
def _load_feature_stack() -> tuple[Any, Any, Any]:
    feature_model_path = os.environ.get("EBFT_FEATURE_MODEL_PATH", "").strip()
    if not feature_model_path:
        raise RuntimeError("EBFT_FEATURE_MODEL_PATH is required for feature rewards")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = os.environ.get("EBFT_TOKENIZER_PATH", feature_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        feature_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return torch, tokenizer, model


def _embed_texts(texts: list[str]):
    torch, tokenizer, model = _load_feature_stack()
    device = next(model.parameters()).device
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=_env_int("EBFT_FEATURE_MAX_LEN", 2048))
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1].float()
    attn = inputs.get("attention_mask")
    embed_method = os.environ.get("EBFT_EMBED_METHOD", "last_token")
    if embed_method == "mean_pooling":
        weights = attn.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    lengths = attn.sum(dim=1).clamp_min(1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _feature_rewards(samples: list[Any]) -> list[float]:
    import torch
    from openrlhf.utils.embedding_utils import get_alignment_rewards, get_cf_l1oo_rewards, get_diversity_rewards

    n_samples = _env_int("EBFT_N_SAMPLES_PER_PROMPT", int(os.environ.get("N_SAMPLES_PER_PROMPT", 4)))
    mode = os.environ.get("EBFT_RM_MODE", "pointwise")
    rewards: list[float] = []

    for group in _chunks(samples, n_samples):
        if not group:
            continue
        if len(group) != n_samples:
            rewards.extend(_exact_reward(sample) for sample in group)
            continue

        responses = [_sample_response(sample) for sample in group]
        labels = [_sample_label(sample) for sample in group]
        label = labels[0] if labels else ""
        if not label or any(not response for response in responses):
            rewards.extend(_exact_reward(sample) for sample in group)
            continue

        gen_vec = _embed_texts(responses)
        gt_vec = _embed_texts([label] * len(group))
        gen_embedding = gen_vec.reshape(1, 1, len(group), 1, gen_vec.shape[-1])
        gt_embedding = gt_vec.reshape(1, 1, len(group), 1, gt_vec.shape[-1])

        if mode == "cf_l1oo":
            teacher_embedding = None
            teacher_responses = _get_field(group[0], "teacher_responses", []) or []
            if os.environ.get("EBFT_CF_TARGET_MODE", "single") == "teacher" and teacher_responses:
                teacher_vec = _embed_texts([str(item) for item in teacher_responses])
                teacher_embedding = teacher_vec.reshape(1, 1, teacher_vec.shape[0], 1, teacher_vec.shape[-1])
            reward_tensor = get_cf_l1oo_rewards(
                gen_embedding,
                gt_embedding,
                cf_num_freqs=_env_int("EBFT_CF_NUM_FREQS", 128),
                cf_sigma=_env_float("EBFT_CF_SIGMA", 1.0),
                cf_seed=_env_int("EBFT_CF_SEED", 43),
                cf_alpha=_env_float("EBFT_CF_ALPHA", 0.5),
                cf_beta=_env_float("EBFT_CF_BETA", 0.5),
                cf_reward_scale=_env_float("EBFT_CF_REWARD_SCALE", 1.0),
                cf_target_mode=os.environ.get("EBFT_CF_TARGET_MODE", "single"),
                cf_target_num_refs=_env_int("EBFT_CF_TARGET_NUM_REFS", 1),
                cf_target_std=_env_float("EBFT_CF_TARGET_STD", 0.05),
                cf_target_seed=_env_int("EBFT_CF_TARGET_SEED", 43),
                teacher_embedding=teacher_embedding,
                cf_teacher_lambda=_env_float("EBFT_CF_TEACHER_LAMBDA", 0.0),
            )
        else:
            align = get_alignment_rewards(gen_embedding, gt_embedding) * 2.0
            div = get_diversity_rewards(gen_embedding, per_token=False) * 2.0
            reward_tensor = align - div

        rewards.extend(float(x) for x in reward_tensor.reshape(-1).detach().cpu())
    return rewards


async def custom_rm(args: Any, sample: Any) -> float:
    mode = os.environ.get("EBFT_RM_MODE", "exact")
    if mode == "nonempty":
        return _nonempty_reward(sample)
    if mode in {"pointwise", "cf_l1oo"} and os.environ.get("EBFT_FEATURE_MODEL_PATH"):
        return _feature_rewards([sample])[0]
    return _exact_reward(sample)


async def batched_custom_rm(args: Any, samples: list[Any]) -> list[float]:
    mode = os.environ.get("EBFT_RM_MODE", "exact")
    if mode == "nonempty":
        return [_nonempty_reward(sample) for sample in samples]
    if mode in {"pointwise", "cf_l1oo"} and os.environ.get("EBFT_FEATURE_MODEL_PATH"):
        return await asyncio.to_thread(_feature_rewards, samples)
    return [_exact_reward(sample) for sample in samples]
