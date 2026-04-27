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


__all__ = ["resolve_text_config", "resolve_text_hidden_size"]
