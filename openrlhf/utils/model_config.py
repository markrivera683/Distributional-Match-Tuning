"""HF config helpers for plain causal-LM and multimodal-wrapper checkpoints.

Newer HF model families (e.g. Gemma 4, Qwen 3.5 with `*ForConditionalGeneration`)
expose a *wrapper* config that nests the language stack under ``text_config``
and reserves the top level for cross-modal metadata. As a result, the wrapper
config does NOT carry attributes like ``hidden_size`` directly; reading them
the old-fashioned ``model.config.hidden_size`` raises ``AttributeError``.

This module centralizes a tiny resolver so every call-site that historically
assumed a flat decoder-only config (``LlamaConfig`` / ``Qwen2Config`` /
``Gemma2Config`` / ...) keeps working AND newer wrapper configs are supported
transparently.

Behavior is intentionally non-breaking for plain configs: when ``hidden_size``
is present at the top level, that value is returned unchanged.
"""
from __future__ import annotations

from typing import Any


def resolve_text_config(config: Any) -> Any:
    """Return the inner text-stack config for both flat and wrapper configs.

    Resolution order:
    1. If ``config`` already carries ``hidden_size`` at the top level, treat it
       as a flat decoder-only config and return it as-is.
    2. Else, if ``config.text_config`` exists, return it.
    3. Else, return ``config`` unchanged (caller may still inspect it).
    """
    if config is None:
        return config
    if getattr(config, "hidden_size", None) is not None:
        return config
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        return text_cfg
    return config


def resolve_text_hidden_size(config: Any) -> int:
    """Return the language-stack ``hidden_size`` for any HF config schema.

    Supports:
      * Flat decoder-only configs (LLaMA / Qwen2 / Gemma2 / Mistral / ...):
        returns ``config.hidden_size`` directly.
      * Multimodal-wrapper configs (Gemma 4 / Qwen 3.5 conditional / ...):
        returns ``config.text_config.hidden_size``.

    Raises:
      AttributeError: if neither location yields an integer hidden size. The
        message includes the config type name to make miswired call-sites easy
        to spot in the trace.
    """
    if config is None:
        raise AttributeError("resolve_text_hidden_size: config is None")

    hs = getattr(config, "hidden_size", None)
    if hs is not None:
        return int(hs)

    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        inner = getattr(text_cfg, "hidden_size", None)
        if inner is not None:
            return int(inner)

    raise AttributeError(
        "resolve_text_hidden_size: cannot find hidden_size on config of type "
        f"{type(config).__name__}; tried .hidden_size and .text_config.hidden_size"
    )


def freeze_unused_multimodal_modules(
    model: Any,
    sub_module_names: tuple = (
        "vision_tower",
        "audio_tower",
        "embed_vision",
        "embed_audio",
        "multi_modal_projector",
    ),
) -> dict:
    """Freeze multimodal sub-modules that don't participate in text-only training.

    Multimodal wrappers (Gemma 4 ``Gemma4ForConditionalGeneration``, Qwen 3.5
    ``Qwen3VLForConditionalGeneration``, LLaVA, ...) ship a vision and/or audio
    backbone whose ``nn.Parameter`` tensors carry ``requires_grad=True`` by
    default. In text-only RLHF/SFT, those sub-modules' parameters are never
    used in the forward pass, but they still appear in ``model.parameters()``
    and therefore in DeepSpeed ZeRO-3's parameter book-keeping.

    On DeepSpeed >= 0.18.x, the first backward then crashes inside
    ``count_used_parameters_in_backward`` (``deepspeed/runtime/utils.py``
    around line 1461) -> ``torch.autograd.graph._get_grad_fn_or_grad_acc``
    with::

        AttributeError: 'NoneType' object has no attribute 'next_functions'

    because for unused params ``param.view_as(param).grad_fn`` is ``None`` and
    the helper unconditionally dereferences ``.next_functions[0][0]``.

    DS's own filters at ``utils.py:1455`` (``not param.requires_grad: continue``)
    and ``stage3.py:1314`` (only adds ``param.requires_grad`` params to the
    tracked list) explicitly skip frozen tensors, so flipping
    ``requires_grad=False`` on every parameter inside the unused towers is
    sufficient: those params no longer enter the partition / used-counter
    pipeline at all.

    This helper walks the module tree and freezes every sub-module whose
    *local* attribute name (last path component) matches ``sub_module_names``.
    Calling it on a flat decoder-only model (LLaMA / Qwen2 / Gemma2 / ...) is
    a no-op. Idempotent: safe to call multiple times.

    Args:
        model: a ``torch.nn.Module`` (typically the inner HF transformers
            model, e.g. ``Actor.model`` or ``Critic.model``).
        sub_module_names: attribute basenames that, anywhere in the module
            tree, should be frozen. Default covers Gemma 4 / Qwen 3.5 /
            LLaVA-style vision+audio backbones.

    Returns:
        Dict mapping the resolved attribute path (e.g. ``"model.vision_tower"``)
        to the number of parameters frozen at that path. Empty dict if nothing
        matched (model is not multimodal, or towers were already frozen).
    """
    target_set = set(sub_module_names)
    frozen: dict = {}

    for path, sub in model.named_modules():
        local_name = path.rsplit(".", 1)[-1] if path else ""
        if local_name not in target_set:
            continue
        n_touched = 0
        n_params = 0
        for p in sub.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                n_touched += 1
                # Under HfDeepSpeedConfig(ZeRO-3) init, ``p.numel()`` reports
                # the LOCAL partition shard size (and may be 0 on some ranks
                # before all_gather), so it is unreliable as a presence check.
                # ``ds_numel`` is the original full numel saved by DeepSpeed
                # at partition time. Fall back to ``numel()`` for non-ZeRO
                # tensors.
                n_params += int(getattr(p, "ds_numel", 0)) or int(p.numel())
        # Record the entry whenever we actually flipped at least one
        # ``requires_grad`` flag, regardless of whether ds_numel was
        # populated. This ensures the freeze report appears even when
        # ZeRO-3 partitioning makes the per-rank numel zero.
        if n_touched > 0:
            frozen[path] = n_params

    return frozen


__all__ = [
    "resolve_text_config",
    "resolve_text_hidden_size",
    "freeze_unused_multimodal_modules",
]
