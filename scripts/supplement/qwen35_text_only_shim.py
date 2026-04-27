from __future__ import annotations

from functools import wraps
from math import prod
import os

import torch
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.utils.torch_utils import get_dtype_size, get_kv_cache_torch_dtype

QWEN35_TEXT_ONLY_SHIM_ARCH = "RepoQwen3_5TextOnlyForCausalLM"
QWEN35_TEXT_ONLY_SHIM_ENV = "REPO_QWEN35_TEXT_ONLY_SHIM"


class Qwen35TextOnlyShimForCausalLM(Qwen3_5ForCausalLM):
    """Text-only loader for Qwen3.5 conditional checkpoints."""

    supports_mrope = True
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": None,
            "model.language_model.": "model.",
        }
    )

    def load_weights(self, weights):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        hf_config = vllm_config.model_config.hf_text_config
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator,
        )

        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise ValueError(
                "Qwen3.5 text-only shim does not support multimodal features."
            )
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).repeat(3, 1), 0


def ensure_qwen35_text_only_shim_registered() -> None:
    """Register the repo-local text-only Qwen3.5 shim with vLLM."""
    if QWEN35_TEXT_ONLY_SHIM_ARCH in ModelRegistry.get_supported_archs():
        return
    ModelRegistry.register_model(
        QWEN35_TEXT_ONLY_SHIM_ARCH,
        "qwen35_text_only_shim:Qwen35TextOnlyShimForCausalLM",
    )


def _is_qwen35_text_only_shim_config(vllm_config) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return False
    architectures = list(getattr(model_config, "architectures", []) or [])
    return QWEN35_TEXT_ONLY_SHIM_ARCH in architectures


def _compute_attention_page_size_bytes(vllm_config) -> int:
    text_cfg = vllm_config.model_config.hf_text_config
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    kv_cache_dtype = get_kv_cache_torch_dtype(
        vllm_config.cache_config.cache_dtype,
        vllm_config.model_config.dtype,
    )
    local_num_kv_heads = max(1, text_cfg.num_key_value_heads // tp_size)
    return (
        vllm_config.cache_config.block_size
        * local_num_kv_heads
        * (text_cfg.head_dim + text_cfg.head_dim)
        * get_dtype_size(kv_cache_dtype)
    )


def _compute_mamba_page_size_bytes(vllm_config) -> int:
    text_cfg = vllm_config.model_config.hf_text_config
    num_spec = (
        vllm_config.speculative_config.num_speculative_tokens
        if vllm_config.speculative_config
        else 0
    )
    state_shapes = MambaStateShapeCalculator.gated_delta_net_state_shape(
        vllm_config.parallel_config.tensor_parallel_size,
        text_cfg.linear_num_key_heads,
        text_cfg.linear_num_value_heads,
        text_cfg.linear_key_head_dim,
        text_cfg.linear_value_head_dim,
        text_cfg.linear_conv_kernel_dim,
        num_spec,
    )
    state_dtypes = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        vllm_config.model_config.dtype,
        vllm_config.cache_config.mamba_cache_dtype,
        vllm_config.cache_config.mamba_ssm_cache_dtype,
    )
    return sum(
        prod(shape) * get_dtype_size(dtype)
        for shape, dtype in zip(state_shapes, state_dtypes)
    )


def maybe_pad_qwen35_text_only_mamba_page_size(vllm_config) -> None:
    """Pad Mamba page size so hybrid Qwen3.5 TP=8 can unify KV cache pages."""
    if not _is_qwen35_text_only_shim_config(vllm_config):
        return

    attention_page_size = _compute_attention_page_size_bytes(vllm_config)
    mamba_page_size = _compute_mamba_page_size_bytes(vllm_config)
    if attention_page_size <= 0 or mamba_page_size <= 0:
        return
    if mamba_page_size % attention_page_size == 0:
        return

    padded_page_size = (
        (mamba_page_size + attention_page_size - 1) // attention_page_size
    ) * attention_page_size
    current_padded = vllm_config.cache_config.mamba_page_size_padded
    if current_padded is not None and current_padded >= padded_page_size:
        return

    vllm_config.cache_config.mamba_page_size_padded = padded_page_size
    print(
        "[compat] padding qwen3.5 text-only mamba page size "
        f"from {mamba_page_size} to {padded_page_size} bytes "
        f"to align with attention page size {attention_page_size}",
        flush=True,
    )


def ensure_qwen35_text_only_runtime_patch_installed() -> None:
    """Patch EngineArgs.create_engine_config to apply shim-specific cache fixes."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.models.config import HybridAttentionMambaModelConfig
    from vllm.v1.kv_cache_interface import AttentionSpec
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(
        EngineArgs.create_engine_config,
        "_qwen35_text_only_runtime_patch_installed",
        False,
    ):
        return

    original_create_engine_config = EngineArgs.create_engine_config

    @wraps(original_create_engine_config)
    def patched_create_engine_config(self, *args, **kwargs):
        vllm_config = original_create_engine_config(self, *args, **kwargs)
        if _is_qwen35_text_only_shim_config(vllm_config):
            HybridAttentionMambaModelConfig.verify_and_update_config(vllm_config)
            maybe_pad_qwen35_text_only_mamba_page_size(vllm_config)
        return vllm_config

    patched_create_engine_config._qwen35_text_only_runtime_patch_installed = True
    EngineArgs.create_engine_config = patched_create_engine_config

    if getattr(
        GPUModelRunner._update_hybrid_attention_mamba_layout,
        "_qwen35_text_only_runtime_patch_installed",
        False,
    ):
        return

    original_update_hybrid_layout = GPUModelRunner._update_hybrid_attention_mamba_layout

    @wraps(original_update_hybrid_layout)
    def patched_update_hybrid_attention_mamba_layout(self, kv_caches):
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                kv_cache = kv_caches[layer_name]
                if not isinstance(kv_cache_spec, AttentionSpec) or kv_cache.shape[0] != 2:
                    continue

                needs_relayout = kv_cache.shape[1] != 2
                if not needs_relayout:
                    probe_shape = group.backend.get_kv_cache_shape(
                        3,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=self.cache_config.cache_dtype,
                    )
                    needs_relayout = probe_shape[0] == 2

                if needs_relayout:
                    hidden_size = kv_cache.shape[2:].numel()
                    kv_cache.as_strided_(
                        size=kv_cache.shape,
                        stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
                    )

    patched_update_hybrid_attention_mamba_layout._qwen35_text_only_runtime_patch_installed = True
    GPUModelRunner._update_hybrid_attention_mamba_layout = (
        patched_update_hybrid_attention_mamba_layout
    )


def prepare_qwen35_text_only_shim_env() -> None:
    """Expose the shim module to fresh Python worker processes."""
    shim_dir = os.path.dirname(os.path.abspath(__file__))
    pythonpath_entries = [
        entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry
    ]
    if shim_dir not in pythonpath_entries:
        pythonpath_entries.insert(0, shim_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    os.environ[QWEN35_TEXT_ONLY_SHIM_ENV] = "1"
    ensure_qwen35_text_only_shim_registered()
    ensure_qwen35_text_only_runtime_patch_installed()
