from contextlib import nullcontext
from typing import Optional
import math
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import torch.nn.functional as F

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _unwrap_attention_target(model):
    target = model
    while hasattr(target, "module"):
        target = target.module
    return target


class _TemporaryAttentionImplementation:
    def __init__(self, model, override_impl: Optional[str]):
        self.model = _unwrap_attention_target(model)
        self.override_impl = override_impl
        self.original_impl = None

    def __enter__(self):
        if self.override_impl is None:
            return self.model
        self.original_impl = getattr(self.model.config, "_attn_implementation", None)
        self.model.config._attn_implementation = self.override_impl
        return self.model

    def __exit__(self, exc_type, exc, tb):
        if self.override_impl is not None:
            self.model.config._attn_implementation = self.original_impl
        return False


class Float32Linear(nn.Linear):
    """
    A linear layer that converts input to float32 before applying the linear operation.
    """
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            return F.linear(x.float(), self.weight.float(), 
                            None if self.bias is None else self.bias.float())


class ResidualBottleneckFeatureAdapter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int = 64, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.down_proj(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x


class Critic(nn.Module):
    """
    Base class for Critic models in reinforcement learning.

    This class serves as a foundation for implementing various critic models, which are responsible for evaluating the quality of the actions taken by the actor.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        use_flash_attention_2 (bool, optional): Whether to utilize Flash Attention 2.0 for improved performance. Defaults to False.
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel for the model. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        temperature=1.0,
        use_liger_kernel=False,
        critic_sequence_level="token",
        gen_len=Optional[int],
        hidden_state_method="last_only",
        feature_adapter_enable: bool = False,
        feature_adapter_type: str = "residual_bottleneck",
        feature_adapter_rank: int = 64,
        feature_adapter_dropout: float = 0.0,
        feature_adapter_unfreeze_layers: int = 0,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.critic_sequence_level = critic_sequence_level
        self.feature_adapter_enable = bool(feature_adapter_enable)
        self.feature_adapter_type = feature_adapter_type
        self.feature_adapter_unfreeze_layers = int(feature_adapter_unfreeze_layers)
        # Debug printing in this module can easily spam logs (and break Ray Jobs log streaming).
        # Keep it off by default; enable only when explicitly debugging.
        self.debug = bool(debug)

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            if hidden_state_method == "last_only":
                num_layers = 1
            elif hidden_state_method == "mean":
                num_layers = 1
            elif hidden_state_method == "middle":
                num_layers = 1 
            elif hidden_state_method == "middle_concat":
                num_layers = 2
            elif hidden_state_method == "middle_stack":
                num_layers = 2
            elif hidden_state_method == "concat":
                num_layers = 3
            elif hidden_state_method == "stack":
                num_layers = 3
            elif hidden_state_method.startswith("layer_"):
                num_layers = 1
            elif hidden_state_method.startswith("concat_layers_"):
                try:
                    # Extract layer indices from the string (e.g., "concat_layers_5_6" -> [5, 6])
                    layer_indices_str = hidden_state_method.replace("concat_layers_", "")
                    layer_indices = [int(x) for x in layer_indices_str.split("_")]

                    num_layers = len(layer_indices)
                except (IndexError, ValueError) as e:
                    raise ValueError(f"Invalid layer specification in '{hidden_state_method}': {e}")
            elif hidden_state_method.startswith("stack_layers_"):
                try:
                    # Extract layer indices from the string (e.g., "concat_layers_5_6" -> [5, 6])
                    layer_indices_str = hidden_state_method.replace("stack_layers_", "")
                    layer_indices = [int(x) for x in layer_indices_str.split("_")]

                    num_layers = len(layer_indices)
                except (IndexError, ValueError) as e:
                    raise ValueError(f"Invalid layer specification in '{hidden_state_method}': {e}")
            else:
                raise ValueError(f"Invalid hidden_state_method: {hidden_state_method}")

            if hidden_state_method in {"middle_concat", "concat"} or hidden_state_method.startswith("concat_layers_"):
                feature_hidden_dim = num_layers * self.model.config.hidden_size
            else:
                feature_hidden_dim = self.model.config.hidden_size

            if self.feature_adapter_enable:
                if self.feature_adapter_type != "residual_bottleneck":
                    raise ValueError(f"Unsupported feature_adapter_type: {self.feature_adapter_type}")
                self.feature_adapter = ResidualBottleneckFeatureAdapter(
                    hidden_dim=feature_hidden_dim,
                    rank=int(feature_adapter_rank),
                    dropout=float(feature_adapter_dropout),
                )
            else:
                self.feature_adapter = nn.Identity()

            # Determine classifier head input dimension
            if self.critic_sequence_level == "concat":
                assert gen_len is not None, "gen_len must be provided for 'concat'"
                d_in = gen_len * num_layers * self.model.config.hidden_size

            elif self.critic_sequence_level in ["token", "mean_pooling", "last_token"]:
                d_in = num_layers * self.model.config.hidden_size

            else:
                raise ValueError(f"Unknown critic_sequence_level: {self.critic_sequence_level}")

            self.pre_head_norm = nn.LayerNorm(d_in, elementwise_affine=False)
            self.classifier_head = Float32Linear(d_in, 1, bias=True)

            # Small weight/bias init → logits ≈ 0 → P(ŷ=1) ≈ 0.5 at init
            with torch.no_grad():
                nn.init.normal_(self.classifier_head.weight, mean=0.0, std=5e-3)
                nn.init.normal_(self.classifier_head.bias, mean=0.0, std=5e-3)

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                if self.debug:
                    logger.info("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples

            if self.feature_adapter_enable:
                # Freeze the entire backbone first.
                for param in self.model.parameters():
                    param.requires_grad = False

                # 2-full: selectively unfreeze the top-N transformer layers.
                # feature_adapter_unfreeze_layers=0  → 2-lite (frozen backbone, adapter+head only)
                # feature_adapter_unfreeze_layers>0  → 2-full (top-N layers + adapter + head trainable)
                if self.feature_adapter_unfreeze_layers > 0:
                    self._unfreeze_top_layers(self.model, self.feature_adapter_unfreeze_layers)
        else:
            self.model = pretrain_or_model
            self.feature_adapter = nn.Identity()
            self.feature_adapter_unfreeze_layers = 0

    @staticmethod
    def _unfreeze_top_layers(model: nn.Module, n_layers: int) -> None:
        """Unfreeze the top-N transformer layers of the backbone for 2-full adaptation.

        Looks for a ``layers`` / ``blocks`` attribute in the common positions used
        by HuggingFace decoder models (model.layers, model.model.layers, etc.).
        Falls back to unfreezing the final LM head (norm + lm_head) when the
        transformer stack cannot be located automatically.

        Args:
            model: The HuggingFace causal-LM module.
            n_layers: Number of transformer layers to unfreeze from the top.
        """
        # Candidates for the transformer layer list, ordered by likelihood.
        _layer_list = None
        for attr_path in (
            "model.layers",       # Llama / Qwen / Mistral
            "model.model.layers",  # wrapped models
            "transformer.h",       # GPT-2 / Falcon
            "model.blocks",        # some custom models
            "blocks",
            "layers",
        ):
            obj = model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                    _layer_list = obj
                    break
            except AttributeError:
                continue

        if _layer_list is None:
            logger.warning(
                "[2-full] Could not locate transformer layer list in backbone. "
                "Falling back to unfreezing final norm and lm_head only."
            )
            # Unfreeze final norm and lm_head as a fallback.
            for attr in ("model.norm", "model.model.norm", "lm_head"):
                obj = model
                try:
                    for part in attr.split("."):
                        obj = getattr(obj, part)
                    for param in obj.parameters():
                        param.requires_grad = True
                except AttributeError:
                    pass
            return

        total = len(_layer_list)
        unfreeze_from = max(0, total - n_layers)
        unfrozen_count = 0
        for i, layer in enumerate(_layer_list):
            if i >= unfreeze_from:
                for param in layer.parameters():
                    param.requires_grad = True
                    unfrozen_count += 1

        # Also unfreeze the final layer norm (if present) for full feature quality.
        for attr in ("model.norm", "model.model.norm", "transformer.ln_f"):
            obj = model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                for param in obj.parameters():
                    param.requires_grad = True
            except AttributeError:
                pass

        logger.info(
            "[2-full] Unfroze top %d / %d transformer layers (indices %d..%d), "
            "%d parameters set to trainable.",
            min(n_layers, total), total, unfreeze_from, total - 1, unfrozen_count,
        )

    def _apply_feature_adapter(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.feature_adapter_enable:
            return hidden_states
        return self.feature_adapter(hidden_states)

    def _apply_head(self, h):
        """
        h is shaped:
        - (B, num_blocks, D) for concat / mean_pooling / last_token
        - or (B, num_blocks, seq_len, D) for token mode
        """
        return self.classifier_head(h.float())

    @property
    def config(self):
        """Return the config of the model for compatibility with save_model."""
        return self.model.config
    
    def select_last_answer_token(self, gen_hidden_states, gen_qa_mask, qa_masking=False):
        # gen_hidden_states: (B, NB, G, NF, H)
        # gen_qa_mask:       (B, NB, G, NF, 1) with 1=answer, 0=question
        B, NB, G, NF, H = gen_hidden_states.shape
        device = gen_hidden_states.device

        if not qa_masking:
            gen_qa_mask = torch.ones_like(gen_qa_mask)

        # (B, NB, G, NF)
        mask = gen_qa_mask.squeeze(-1).to(torch.bool)

        # Index of each timestep along G
        time_idx = torch.arange(G, device=device).view(1, 1, G, 1).expand(B, NB, G, NF)

        # Mask non-answer positions to -1, then amax over time gives the last answer token index.
        # Positions with no answer token at all remain -1 → zeroed out below.
        last_idx = time_idx.masked_fill(~mask, -1).amax(dim=2)  # (B, NB, NF)

        # Build gather index tensor for dim=2 (time)
        safe_idx = last_idx.clamp_min(0).unsqueeze(2).unsqueeze(-1).expand(B, NB, 1, NF, H)
        out = gen_hidden_states.gather(dim=2, index=safe_idx)   # (B, NB, 1, NF, H)

        # Zero out where there was no answer
        no_ans = last_idx.eq(-1).unsqueeze(2).unsqueeze(-1).expand_as(out)
        out = out.masked_fill(no_ans, 0.0)

        return out
    
    def forward(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.LongTensor] = None,
        normalize_hidden_states: bool = True,
        return_classifier_logits: bool = False,
        context_length: Optional[int] = None,
        prompt_length: Optional[int] = None,
        generate_max_len: Optional[int] = None,
        stride: Optional[int] = None,
        num_blocks: Optional[int] = None,
        hidden_state_method: Optional[str] = "last_only",
        return_dtype: torch.dtype = torch.float32,
        qa_masks: Optional[torch.Tensor] = None,
        qa_masking: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the critic model.

        Args:
            sequences: Input token sequences [batch_size, seq_length]
            attention_mask: Attention mask for the transformer
            pos_ids: Position IDs for custom positional encoding
        Returns:
            Model output (last hidden state).
        """

        override_impl = None
        attention_target = _unwrap_attention_target(self.model)
        if (
            attention_mask is not None
            and attention_mask.dim() == 4
            and getattr(attention_target.config, "_attn_implementation", None) == "flash_attention_2"
        ):
            # HF FlashAttention2 does not support EBFT's dense 4D additive masks.
            override_impl = "eager"

        attention_impl_context = (
            _TemporaryAttentionImplementation(self.model, override_impl)
            if override_impl is not None
            else nullcontext(self.model)
        )

        with attention_impl_context:
            output = self.model(
                sequences,
                attention_mask=attention_mask,
                position_ids=pos_ids,
                output_hidden_states=True,
            )

        # Parse hidden_state_method to support flexible layer extraction
        if hidden_state_method == "last_only":
            hidden_states = output.hidden_states[-1] # bsz, fsl, features -> bsz, fsl, 1, features
            hidden_states = hidden_states.unsqueeze(-2)
        elif hidden_state_method == "mean":
            # mean pooling over all layers
            hidden_states = torch.mean(torch.stack(output.hidden_states[1:]), dim=0) # exclude embedding layer
            hidden_states = hidden_states.unsqueeze(-2)
        elif hidden_state_method == "middle":
            num_layers = len(output.hidden_states) - 1  # exclude embedding layer
            mid1 = num_layers // 2
            mid2 = mid1 + 1
            hidden_states = torch.mean(torch.stack([output.hidden_states[mid1], output.hidden_states[mid2]]), dim=0)
            hidden_states = hidden_states.unsqueeze(-2)
        elif hidden_state_method == "middle_concat":
            num_layers = len(output.hidden_states) - 1  # exclude embedding layer
            mid1 = num_layers // 2
            mid2 = mid1 + 1
            hidden_states = torch.cat([output.hidden_states[mid1], output.hidden_states[mid2]], dim=-1)
            hidden_states = hidden_states.unsqueeze(-2)
        elif hidden_state_method == "middle_stack":
            num_layers = len(output.hidden_states) - 1  # exclude embedding layer
            mid1 = num_layers // 2
            mid2 = mid1 + 1
            hidden_states = torch.stack([output.hidden_states[mid1], output.hidden_states[mid2]], dim=-2)
            
        elif hidden_state_method == "concat":
            num_layers = len(output.hidden_states) - 1 # exclude embedding layer
            idxs = [
                max(1, min(num_layers, math.floor(num_layers * 0.25))),
                max(1, min(num_layers, math.floor(num_layers * 0.50))),
                max(1, min(num_layers, math.floor(num_layers * 0.75))),
            ]
            selected = [output.hidden_states[i] for i in idxs]
            hidden_states = torch.cat(selected, dim=-1) # (bsz, full seq len, num_layers, features)
            hidden_states = hidden_states.unsqueeze(-2)
        elif hidden_state_method == "stack":
            num_layers = len(output.hidden_states) - 1 # exclude embedding layer
            idxs = [
                max(1, min(num_layers, math.floor(num_layers * 0.25))),
                max(1, min(num_layers, math.floor(num_layers * 0.50))),
                max(1, min(num_layers, math.floor(num_layers * 0.75))),
            ]
            selected = [output.hidden_states[i] for i in idxs]
            hidden_states = torch.stack(selected, dim=-2) # (bsz, full seq len, num_layers, features)
        elif hidden_state_method.startswith("layer_"):
            # Single layer extraction: "layer_5", "layer_6", etc.
            try:
                layer_idx = int(hidden_state_method.split("_")[1])
                # Note: hidden_states[0] is embedding layer, so layer_1 is hidden_states[1]
                hidden_states = output.hidden_states[layer_idx]
                hidden_states = hidden_states.unsqueeze(-2)
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid layer specification in '{hidden_state_method}': {e}")
        elif hidden_state_method.startswith("concat_layers_"):
            # Multi-layer concatenation: "concat_layers_5_6", "concat_layers_3_4_5_6", etc.
            try:
                # Extract layer indices from the string (e.g., "concat_layers_5_6" -> [5, 6])
                layer_indices_str = hidden_state_method.replace("concat_layers_", "")
                layer_indices = [int(x) for x in layer_indices_str.split("_")]

                # Extract the specified layers
                selected_layers = [output.hidden_states[idx] for idx in layer_indices]

                # Concatenate along the feature dimension
                hidden_states = torch.cat(selected_layers, dim=-1)
                if self.debug:
                    logger.info(f"Concatenating layers {layer_indices} -> output shape: {hidden_states.shape}")
                hidden_states = hidden_states.unsqueeze(-2)
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid layer specification in '{hidden_state_method}': {e}")
        elif hidden_state_method.startswith("stack_layers_"):
            # Multi-layer stacking: "stack_layers_5_6", "stack_layers_3_4_5_6", etc.
            try:
                # Extract layer indices from the string (e.g., "stack_layers_5_6" -> [5, 6])
                layer_indices_str = hidden_state_method.replace("stack_layers_", "")
                layer_indices = [int(x) for x in layer_indices_str.split("_")]

                # Extract the specified layers
                selected_layers = [output.hidden_states[idx] for idx in layer_indices]

                # Concatenate along the feature dimension
                hidden_states = torch.stack(selected_layers, dim=-2)
                if self.debug:
                    logger.info(f"Stacking layers {layer_indices} -> output shape: {hidden_states.shape}")
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid layer specification in '{hidden_state_method}': {e}")
        else:
            raise ValueError(f"Invalid hidden_state_method: {hidden_state_method}")

        # Normalize per-token representations so dot products behave like cosine similarity.
        hidden_states = hidden_states / hidden_states.norm(dim=-1, keepdim=True)
        hidden_states = self._apply_feature_adapter(hidden_states)

        # ------------------------------------------------------------
        # Prompt-conditioned anchor (optional consumer)
        # ------------------------------------------------------------
        # For each block k, the "prompt anchor" is the last token of that block's
        # context window: index = (context_length - 1) + k * stride.
        #
        # IMPORTANT: This anchor should *not* be QA-masked away, because it is meant to
        # represent the prompt/prefix (which may include question tokens).
        #
        # Shape: (B, num_blocks, num_feat, hidden_size)
        if context_length is not None and stride is not None and num_blocks is not None:
            if context_length < 1:
                raise ValueError(f"context_length must be >= 1 to form a prompt anchor, got {context_length}")
            anchor_positions = (
                torch.arange(num_blocks, device=hidden_states.device, dtype=torch.long) * int(stride)
                + (int(context_length) - 1)
            )
            prompt_hidden_states = hidden_states[:, anchor_positions, :, :]
        else:
            prompt_hidden_states = None

        # Mask out hidden states where qa_mask is 0 (hidden states shape: BSZ, FULL_SEQ_LEN, NUM_FEAT, FEAT_DIM)
        if not qa_masking:
            qa_masks = torch.ones_like(qa_masks)
        hidden_states = hidden_states * qa_masks.unsqueeze(-1).unsqueeze(-1)

        assert context_length is not None and prompt_length is not None and generate_max_len is not None and stride is not None and num_blocks is not None, "context_length, prompt_length, generate_max_len, stride, and num_blocks must be provided"

        # Split full-sequence hidden states into GT region (context_length..prompt_length)
        # and generated region (prompt_length..end). Both are (B, T, NF, H).
        gt_hidden_states = hidden_states[:, context_length:prompt_length, :, :] # (batch_size, prompt_length, num_feat, hidden_size)
        gen_hidden_states = hidden_states[:, prompt_length:, :, :] # (batch_size, generate_max_len * num_blocks, num_feat, hidden_size)
        gt_qa_mask = qa_masks[:, context_length:prompt_length].view(gt_hidden_states.shape[0], gt_hidden_states.shape[1], 1, 1).repeat(1, 1, gt_hidden_states.shape[2], 1)
        gen_qa_mask = qa_masks[:, prompt_length:].view(gen_hidden_states.shape[0], gen_hidden_states.shape[1], 1, 1).repeat(1, 1, gen_hidden_states.shape[2], 1)

        # Slide a window of size generate_max_len with step stride over the GT region,
        # then permute to (B, num_blocks, gen_len, NF, H).
        gt_hidden_states = gt_hidden_states.unfold(-3, generate_max_len, stride).permute(0, 1, 4, 2, 3) # (batch_size, num_blocks, gen_len, num_feat, hidden_size)
        gt_qa_mask = gt_qa_mask.unfold(-3, generate_max_len, stride).permute(0, 1, 4, 2, 3)

        # Generated tokens are interleaved as (B, gen_len*num_blocks, NF, H) — deinterleave to
        # (B, num_blocks, gen_len, NF, H) by reshaping then transposing the block/time axes.
        gen_hidden_states = gen_hidden_states.reshape(gen_hidden_states.shape[0], generate_max_len, num_blocks, gen_hidden_states.shape[-2], gen_hidden_states.shape[-1]).transpose(-3,-4) # (batch_size, num_blocks, generate_max_len, num_feat, hidden_size)
        gen_qa_mask = gen_qa_mask.reshape(gen_hidden_states.shape[0], generate_max_len, num_blocks, gen_hidden_states.shape[-2], 1).transpose(-3,-4)

        # shape coming in is (bsz, num_blocks, gen_len, num_feat, hidden_size)
        if self.critic_sequence_level == "concat":
            gt_logits = self._apply_head(gt_hidden_states.reshape(gt_hidden_states.shape[0], gt_hidden_states.shape[1], -1))
            gen_logits = self._apply_head(gen_hidden_states.reshape(gen_hidden_states.shape[0], gen_hidden_states.shape[1], -1))
        
        elif self.critic_sequence_level == "mean_pooling":
            gt_pooled = gt_hidden_states.mean(dim=2).reshape(gt_hidden_states.shape[0], gt_hidden_states.shape[1], -1)
            gen_pooled = gen_hidden_states.mean(dim=2).reshape(gen_hidden_states.shape[0], gen_hidden_states.shape[1], -1)
            gt_logits = self._apply_head(gt_pooled)
            gen_logits = self._apply_head(gen_pooled)

        elif self.critic_sequence_level == "token":
            # (batch_size, num_blocks, generate_max_len, num_feat * hidden_size) -> (batch_size, num_blocks, generate_max_len, 1)
            gt_logits = self._apply_head(gt_hidden_states.reshape(*gt_hidden_states.shape[:3], -1))
            gen_logits = self._apply_head(gen_hidden_states.reshape(*gen_hidden_states.shape[:3], -1))
        
        elif self.critic_sequence_level == "last_token":
            # Select the last answer token per block (groomed), then apply the classifier head.
            gen_last_answer_hidden = self.select_last_answer_token(gen_hidden_states, gen_qa_mask, qa_masking=qa_masking)
            gt_last_answer_hidden = self.select_last_answer_token(gt_hidden_states, gt_qa_mask, qa_masking=qa_masking)
            gt_logits = self._apply_head(gt_last_answer_hidden.reshape(gt_last_answer_hidden.shape[0], gt_last_answer_hidden.shape[1], -1))
            gen_logits = self._apply_head(gen_last_answer_hidden.reshape(gen_last_answer_hidden.shape[0], gen_last_answer_hidden.shape[1], -1))

        if return_classifier_logits:
            return (
                gt_hidden_states.to(return_dtype),
                gen_hidden_states.to(return_dtype),
                gt_logits.to(return_dtype),
                gen_logits.to(return_dtype),
                prompt_hidden_states.to(return_dtype) if prompt_hidden_states is not None else None,
            )
        else:
            gen_probs = torch.sigmoid(gen_logits.squeeze(-1))
            return hidden_states.to(return_dtype), gen_probs.to(return_dtype)
        

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
    
    def save_pretrained(self, output_dir: str, state_dict=None, **kwargs):
        """Save the model to a directory in HuggingFace format.

        The HF backbone is saved via its own ``save_pretrained`` (or a
        ``pytorch_model.bin`` fallback).  Critic-specific modules
        (``feature_adapter``, ``classifier_head``, ``pre_head_norm``) are
        stored separately as ``critic_head.pt`` so that the backbone
        checkpoint stays loadable by plain ``AutoModelForCausalLM`` while
        the full critic state can still be restored.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir, state_dict=state_dict, **kwargs)
        else:
            save_path = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(state_dict if state_dict is not None else self.model.state_dict(), save_path)

        head_state: dict = {}
        for module_name in ("feature_adapter", "classifier_head", "pre_head_norm"):
            module = getattr(self, module_name, None)
            if module is None or isinstance(module, nn.Identity):
                continue
            for k, v in module.state_dict().items():
                head_state[f"{module_name}.{k}"] = v
        if head_state:
            torch.save(head_state, os.path.join(output_dir, "critic_head.pt"))
            logger.info(
                f"[Critic] Saved {len(head_state)} critic-head tensors to {output_dir}/critic_head.pt"
            )
