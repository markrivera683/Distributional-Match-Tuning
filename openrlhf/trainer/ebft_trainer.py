import os
import time
from abc import ABC
from datetime import timedelta
import numpy as np
import ray
import math
import torch
from tqdm import tqdm
from typing import List
import torch.nn.functional as F
from collections import Counter

from openrlhf.datasets import SequenceDataset, QADataset, CodePromptDataset, HumanEvalDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.datasets.prompt_lookahead import PromptLookahead
from openrlhf.models import EmbeddingLoss
from openrlhf.trainer.ebft_eval_mixin import EBFTEvalMixin
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController, AdaptiveCEController, AdaptiveRLController, FixedCEController, FixedRLController
from openrlhf.trainer.ppo_utils.ebft_experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer
from openrlhf.utils.embedding_utils import (
    whiten_embeddings_batched,
    get_alignment_rewards,
    get_diversity_rewards
)
from datasets import load_dataset
from datasets import concatenate_datasets
try:
    from datatrove.utils.dataset import DatatroveFolderDataset
except ImportError:
    DatatroveFolderDataset = None

logger = init_logger(__name__)


class BaseEBFTTrainer(EBFTEvalMixin, ABC):
    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_groups: List[RayActorGroup],
        reference_model_group: RayActorGroup,
        teacher_model_group: RayActorGroup = None,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__()

        self.strategy = strategy
        self.args = strategy.args

        self.tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not self.args.disable_fast_tokenizer)
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_groups = reward_model_groups or []
        self.reference_model_group = reference_model_group
        self.teacher_model_group = teacher_model_group
        self.dataloader_pin_memory = dataloader_pin_memory

        self.primary_reward_model_group = self._get_primary_reward_model_group()
        self.primary_reward_pretrain = (
            getattr(self.primary_reward_model_group, "pretrain_name", None) if self.primary_reward_model_group else None
        )
 
        self.generate_kwargs = generate_kwargs

        self.max_epochs = self.args.max_epochs
        self.remote_rm_url = self.args.remote_rm_url
        self.init_kl_coef = self.args.init_kl_coef
        self.ce_loss_coef = self.args.ce_loss_coef
        self.kl_loss_coef = self.args.kl_loss_coef
        self.rl_loss_coef = self.args.rl_loss_coef
        self.kl_loss_decay_steps = self.args.kl_loss_decay_steps
        self.ce_loss_decay_steps = self.args.ce_loss_decay_steps
        self.rl_loss_warmup_steps = self.args.rl_loss_warmup_steps
        self.rl_loss_warmup_start = self.args.rl_loss_warmup_start

        self.prompt_split = prompt_split
        self.eval_split = eval_split

        # Init dummy variables
        self.prompts_dataloader = None
        self.eval_dataloader = None
        self.max_steps = None

        self.samples_generator = None
        self.experience_maker = None
        self.remote_reward_model = None
 
        from openrlhf.trainer.ppo_utils.ebft_experience_maker import SamplesGenerator

        self.generator_cls = SamplesGenerator

        # NOTE: Translation metrics (COMET) are initialized lazily inside MT evaluation.
        # This avoids downloading/loading COMET for non-translation eval tasks (e.g., XSum).

    def _init_wandb(self):
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        self.generated_samples_table = None
        if self.strategy.args.use_wandb:
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.strategy.args.use_wandb)
            wandb.init(
                entity=self.strategy.args.wandb_org,
                project=self.strategy.args.wandb_project,
                group=self.strategy.args.wandb_group,
                name=self.strategy.args.wandb_run_name,
                config=self.strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)
            self.generated_samples_table = wandb.Table(columns=["global_step", "text", "reward"])

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, self.strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)
    

    def _reward_model_group_label(self, group, default="reward_model"):
        if group is None:
            return default
        for attr in ("pretrain_name", "name"):
            value = getattr(group, attr, None)
            if isinstance(value, str) and value:
                return value
        actor_type = getattr(group, "ray_actor_type", None)
        if actor_type is not None:
            return actor_type.__name__
        return default
    
    @staticmethod
    def _is_comet_group(group):
        if group is None:
            return False
        if hasattr(group, "is_comet"):
            return bool(group.is_comet)
        group_name = getattr(group, "pretrain_name", None) or getattr(group, "name", None)
        if isinstance(group_name, str) and "comet" in group_name.lower():
            return True
        actor_type = getattr(group, "ray_actor_type", None)
        if actor_type and "comet" in actor_type.__name__.lower():
            return True
        return False

    def _get_primary_reward_model_group(self):
        for group in self.reward_model_groups or []:
            if not self._is_comet_group(group):
                return group
        return None

    def _ensure_primary_reward_model_group(self):
        if self.reward_model_groups and self.primary_reward_model_group is None:
            raise ValueError(
                "At least one non-COMET reward model must be provided when specifying --reward_pretrain. "
                "Place the non-COMET model at the beginning of the comma-separated list."
            )
        return self.primary_reward_model_group  # may be None if no reward groups configured


    def fit(self):
        raise NotImplementedError("fit method is not implemented")

    def critic_train(self, global_steps):
        status = {}
        if self.critic_model_group is not None:
            # sync for deepspeed_enable_sleep
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="reload_states"))

            critic_status_ref = self.critic_model_group.async_run_method(method_name="fit")

            if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
                status.update(ray.get(critic_status_ref)[0])
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="offload_states"))

        if self.critic_model_group and not self.strategy.args.colocate_all_models:
            status.update(ray.get(critic_status_ref)[0])

        return status

    def ppo_train(self, global_steps, status=None):
        if status is None:
            status = {}

        # actor model training
        if self.strategy.args.deepspeed_enable_sleep:
            ray.get(self.actor_model_group.async_run_method(method_name="reload_states"))

        actor_status_ref = self.actor_model_group.async_run_method(method_name="fit", rl_ctl=self.rl_ctl.value, ce_ctl=self.ce_ctl.value, kl_ctl=self.kl_ctl.value)
        status.update(ray.get(actor_status_ref)[0])

        if self.strategy.args.deepspeed_enable_sleep:
            ray.get(self.actor_model_group.async_run_method(method_name="offload_states"))

        return status

 
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None:
                # Add generated samples to wandb using Table
                if "generated_samples" in logs_dict:
                    # https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
                    new_table = self._wandb.Table(
                        columns=self.generated_samples_table.columns, data=self.generated_samples_table.data
                    )
                    new_table.add_data(global_step, *logs_dict.pop("generated_samples"))
                    self.generated_samples_table = new_table
                    self._wandb.log({"train/generated_samples": new_table})
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None:
                for k, v in logs_dict.items():
                    if k == "generated_samples":
                        # Record generated samples in TensorBoard using simple text format
                        text, reward = v
                        formatted_text = f"Sample:\n{text}\n\nReward: {reward:.4f}"
                        self._tensorboard.add_text("train/generated_samples", formatted_text, global_step)
                    else:
                        self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        if global_step % args.eval_down_steps == 0:
            eval_ds = str(getattr(self.args, "eval_dataset", "") or "").lower()
            if "swallow_code" in eval_ds:
                self.evaluate_downstream_humaneval(
                    self.humaneval_dataloader,
                    global_step,
                    args.eval_generate_max_len,
                    args.eval_temperature_down,
                    args.eval_n_samples_per_prompt_down,
                )
            elif "gsm8k" in eval_ds or "math" in eval_ds or "aops" in eval_ds:
                self.evaluate_downstream_gsm8k_math(self.eval_downstream_dataloader, global_step, args.eval_generate_max_len, args.eval_temperature_down, args.eval_n_samples_per_prompt_down)
            elif "opencode-instruct_100k_200tok" in eval_ds:
                self.evaluate_downstream_opencode(self.eval_downstream_dataloader, global_step, args.eval_generate_max_len, args.eval_temperature_down, args.eval_n_samples_per_prompt_down)
            else:
                self.evaluate_downstream_translation(self.eval_downstream_dataloader, global_step, args.eval_generate_max_len, args.eval_temperature_down, args.eval_n_samples_per_prompt_down)
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step, args.eval_temperature, args.eval_n_samples_per_prompt)
        

        # save ckpt
        should_save = False
        if getattr(args, "steps_to_save", None) is not None:
            if global_step in args.steps_to_save:
                should_save = True
        elif global_step % args.save_steps == 0:
            should_save = True

        if should_save:
            tag = f"global_step{global_step}"
            ref = self.actor_model_group.async_run_method(
                method_name="save_checkpoint", tag=tag, client_states=client_states
            )
            if self.critic_model_group is not None:
                ref.extend(self.critic_model_group.async_run_method(method_name="save_checkpoint", tag=tag))
            ray.get(ref)

    def evaluate(self, eval_dataloader, global_step, temperature=0.6, n_samples_per_prompt=1):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts
            global_step: Current training step for logging
            temperature: Sampling temperature for generation
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Wake up vLLM engines if sleeping
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        with torch.no_grad():
            # Collect all evaluation sequences
            all_sequences = []
            all_doc_ids = []
            all_qa_masks = []
            for sequence, doc_ids, qa_masks in eval_dataloader:
                all_sequences.extend(sequence)
                all_doc_ids.extend(doc_ids)
                all_qa_masks.extend(qa_masks)

            # Generate samples with specified temperature and number of samples per prompt
            generate_kwargs = self.generate_kwargs.copy()
            generate_kwargs["temperature"] = temperature
            generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt

            eval_samples_list = self.samples_generator.generate_samples(
                all_sequences, all_doc_ids, all_qa_masks, **generate_kwargs
            )

            # Extract sequence data using new field names
            stride = self.args.stride
            context_length = self.args.context_max_len
            n_samples = n_samples_per_prompt
            generate_length = self.args.generate_max_len
            prompts_list = [s.prompts for s in eval_samples_list]  # Original full prompts (X)
            full_sequences_list = [s.full_sequences for s in eval_samples_list]  # Full sequences (prompt + generated)
            doc_ids_list = [s.doc_ids[:self.args.prompt_max_len] if s.doc_ids.dim() == 1 else s.doc_ids[:,:self.args.prompt_max_len] for s in eval_samples_list]
            qa_masks_list = [s.qa_masks for s in eval_samples_list]

            if not prompts_list:
                logger.info("Evaluation skipped: eval dataloader is empty")
                return

            # Calculate sequence lengths
            prompt_length = prompts_list[0].shape[1]  # Length of original prompt
            num_blocks = (prompt_length - generate_length - context_length) // stride + 1  # Number of prediction blocks

            logs = {}
            critic_hidden_states_ref = self.critic_model_group.async_run_method_batch(
                method_name="forward",
                sequences=full_sequences_list,
                prompt_length=[prompt_length] * len(full_sequences_list),
                context_length=[context_length] * len(full_sequences_list),
                generate_max_len=[generate_length] * len(full_sequences_list),
                stride=[stride] * len(full_sequences_list),
                num_blocks=[num_blocks] * len(full_sequences_list),
                hidden_state_method=[self.args.hidden_state_method] * len(full_sequences_list),
                doc_ids=doc_ids_list,
                document_masking=[self.args.document_masking] * len(full_sequences_list),
                qa_masks=qa_masks_list,
                qa_masking=[self.args.qa_masking] * len(full_sequences_list),
            )
            ray.get(critic_hidden_states_ref)
            duplicate_factor = self.args.ring_attn_size * self.args.ds_tensor_parallel_size
            # Deduplicate across tensor-parallel / ring-attn replica ranks, then flatten micro-batches.
            critic_hidden_states_list = sum(ray.get(critic_hidden_states_ref)[::duplicate_factor], []) # list of num_batches tensors of shape (batch_size, full_sequence_length,hidden_size)

            critic_hidden_states_list = [entry[0] for entry in critic_hidden_states_list]
            critic_hidden_states_tensor = torch.stack(critic_hidden_states_list)

            # Slice critic output into GT region (context..prompt) and generated region (prompt..end).
            # Shape before slicing: (num_batches, batch_size, full_seq_len, num_feat, hidden_size)
            gt_embedding = critic_hidden_states_tensor[:, :, context_length:prompt_length, :, :] # (num_batches, batch_size, prompt_length, num_feat, hidden_size)
            gen_embedding = critic_hidden_states_tensor[:, :, prompt_length:, :, :] # (num_batches, batch_size, generate_length, num_feat, hidden_size)

            # Slide a window of size generate_length with step stride over the GT region →
            # (num_batches, batch_size, num_blocks, generate_length, num_feat, hidden_size)
            gt_embedding = gt_embedding.unfold(-3, generate_length, stride).permute(0, 1, 2, 5, 3, 4)
            # Deinterleave generated tokens from (B, gen_len*num_blocks, NF, H) →
            # (num_batches, batch_size, num_blocks, generate_length, num_feat, hidden_size)
            gen_embedding = gen_embedding.reshape(gen_embedding.shape[0], gen_embedding.shape[1], generate_length, num_blocks, gen_embedding.shape[-2], gen_embedding.shape[-1]).transpose(-3,-4)

            num_micro_batches = gt_embedding.shape[0]
            num_groups = gt_embedding.shape[1] // n_samples # num groups per micro batch
            num_feat = gt_embedding.shape[-2]
            # Split the batch dimension into (num_groups, n_samples) so each group = one prompt's samples.
            gt_embedding = gt_embedding.reshape(num_micro_batches, num_groups, n_samples, num_blocks, generate_length, num_feat, gt_embedding.shape[-1])
            gen_embedding = gen_embedding.reshape(num_micro_batches, num_groups, n_samples, num_blocks, generate_length, num_feat, gen_embedding.shape[-1])
            if self.args.embed_method == "mean_pooling":
                gt_embedding = torch.mean(gt_embedding, dim=-3, keepdim=True) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, 1, num_feat, hidden_size)
                gen_embedding = torch.mean(gen_embedding, dim=-3, keepdim=True) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, 1, num_feat, hidden_size)
            elif self.args.embed_method == "last_token":
                gt_embedding = gt_embedding[:,:,:,:,-1,:,:].unsqueeze(-3) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, 1, num_feat, hidden_size)
                gen_embedding = gen_embedding[:,:,:,:,-1,:,:].unsqueeze(-3) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, 1, num_feat, hidden_size)
            elif self.args.embed_method == "concat":
                gt_embedding = gt_embedding.transpose(-2, -3).reshape(num_micro_batches, num_groups, n_samples, num_blocks, 1, num_feat, generate_length * gt_embedding.shape[-1]) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, 1, num_feat, generate_length * hidden_size)
                gen_embedding = gen_embedding.transpose(-2, -3).reshape(num_micro_batches, num_groups, n_samples, num_blocks, 1, num_feat, generate_length * gen_embedding.shape[-1]) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, 1, num_feat, generate_length * hidden_size)
            elif self.args.embed_method == "token":
                # shape of gt_embedding and gen_embedding should be (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, generate_length, hidden_size)
                pass # already have token-level embeddings
            else:
                raise ValueError(f"Unknown embed_method: {self.args.embed_method}")

            if self.args.use_whitening: #FIX FOR TOKEN EMBED
                gen_embedding, gt_embedding = whiten_embeddings_batched(gen_embedding, gt_embedding, whiten_tol=1e-5, normalize=False)

            # Fuse num_feat and hidden_size into a single embedding dimension for downstream losses.
            gt_embedding = gt_embedding.reshape(
                gt_embedding.shape[0], gt_embedding.shape[1], gt_embedding.shape[2],
                gt_embedding.shape[3], gt_embedding.shape[4], gt_embedding.shape[5] * gt_embedding.shape[6]
            )
            gen_embedding = gen_embedding.reshape(
                gen_embedding.shape[0], gen_embedding.shape[1], gen_embedding.shape[2],
                gen_embedding.shape[3], gen_embedding.shape[4], gen_embedding.shape[5] * gen_embedding.shape[6]
            )

            if self.args.embed_method != "token":
                gt_embedding = gt_embedding.squeeze(-2) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, num_features)
                gen_embedding = gen_embedding.squeeze(-2) # (num_micro_batches, num_groups, num_seq/num_groups, num_blocks, num_features)

            with torch.no_grad():
                loss_fn = EmbeddingLoss()
                eval_critic_loss = loss_fn(gen_embedding, gt_embedding)
                logs.update({
                    "eval_critic_loss": eval_critic_loss.item(),
                })
            gt_rewards_tensor = get_alignment_rewards(gen_embedding, gt_embedding)
            diversity_rewards_tensor = get_diversity_rewards(gen_embedding, per_token=(self.args.embed_method == "token"))
            gt_rewards_tensor = gt_rewards_tensor.reshape(gt_rewards_tensor.shape[0], -1, gt_rewards_tensor.shape[-1])
            diversity_rewards_tensor = diversity_rewards_tensor.reshape(diversity_rewards_tensor.shape[0], -1, diversity_rewards_tensor.shape[-1])

            gt_rewards_tensor *= 2
            diversity_rewards_tensor *= 2

            # As in ebft_experience_maker.py: only apply coefficients when building the combined reward signal.
            rewards_tensor = self.args.alignment_rew_coef * gt_rewards_tensor - self.args.diversity_rew_coef * diversity_rewards_tensor
            effective_rewards_tensor = self.args.alignment_rew_coef * gt_rewards_tensor - self.args.diversity_rew_coef * diversity_rewards_tensor / 2
            # Flatten (num_micro_batches, num_groups, num_blocks) → (num_prompts * num_blocks,),
            # then group into (num_prompts, n_samples_per_prompt) for pass@k computation.
            rewards = torch.cat([r.transpose(1,0).reshape(-1) for r in rewards_tensor], dim=0).reshape(-1, n_samples_per_prompt)
            effective_rewards = torch.cat([r.transpose(1,0).reshape(-1) for r in effective_rewards_tensor], dim=0).reshape(-1, n_samples_per_prompt)

            # Log raw (unscaled) components.
            diversity_rewards = torch.cat([r.transpose(1,0).reshape(-1) for r in diversity_rewards_tensor], dim=0).reshape(-1, n_samples_per_prompt)
            gt_rewards = torch.cat([r.transpose(1,0).reshape(-1) for r in gt_rewards_tensor], dim=0).reshape(-1, n_samples_per_prompt)

            # Calculate pass@k and pass@1 metrics
            if n_samples_per_prompt > 1:
                passk = rewards.max(dim=1).values.mean().item()  # Best reward per prompt
                passk_effective = effective_rewards.max(dim=1).values.mean().item()
                passk_diversity = diversity_rewards.max(dim=1).values.mean().item()
                passk_gt = gt_rewards.max(dim=1).values.mean().item()
            else:
                passk = rewards.mean().item()
                passk_effective = effective_rewards.mean().item()
                passk_diversity = diversity_rewards.mean().item()
                passk_gt = gt_rewards.mean().item()

            pass1 = rewards.mean().item()  # Average reward across all samples
            pass1_effective = effective_rewards.mean().item()
            pass1_diversity = diversity_rewards.mean().item()
            pass1_gt = gt_rewards.mean().item()

            logs.update({"reward_passk": passk, "reward_pass1": pass1,              
                         "reward_passk_effective": passk_effective, "reward_pass1_effective": pass1_effective, "reward_passk_diversity": passk_diversity, "reward_passk_gt": passk_gt, "reward_pass1_diversity": pass1_diversity, "reward_pass1_gt": pass1_gt})

            # Compute full-sequence perplexity locally from stats
            ppl_stats_local = self.experience_maker.make_ppls_experience_batch(eval_samples_list)

            squared_loss = self.experience_maker.make_squared_loss_experience_batch(eval_samples_list)

            full_ce_loss_sum  = float(ppl_stats_local["full_ce_loss_sum"].detach().cpu())
            full_ce_token_sum  = float(ppl_stats_local["full_ce_token_sum"].detach().cpu())
            squared_loss = float(squared_loss["mse"].detach().cpu())
            # chunk_ppls = float(ppl_stats_local["chunk_ppls"])

            if full_ce_loss_sum > 0.0:
                ce_loss = full_ce_loss_sum  / full_ce_token_sum
                ce_ppl  = math.exp(ce_loss)
            else:
                ce_loss, ce_ppl = float("nan"), float("nan")

            logs.update({
                "full_ce_loss":   ce_loss,
                "full_perplexity": ce_ppl,
                "mse": squared_loss,
            })
            

            # Log to wandb/tensorboard
            if self._wandb is not None:
                logs_to_log = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs_to_log)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    
    def prepare_datasets(self):
        args = self.args
        strategy = self.strategy
        
        
        # Local directory: first try HuggingFace load_from_disk (supports both DatasetDict and Dataset).
        # If that fails, fall back to Datatrove tokenized folder format.
        if os.path.isdir(args.prompt_data):
            from datasets import load_from_disk, DatasetDict

            hf_local_data = None
            hf_load_error = None
            try:
                hf_local_data = load_from_disk(args.prompt_data)
            except Exception as e:
                hf_load_error = e

            if hf_local_data is not None:
                if isinstance(hf_local_data, DatasetDict):
                    if self.prompt_split in hf_local_data:
                        train_data = hf_local_data[self.prompt_split]
                    elif "train" in hf_local_data:
                        logger.warning(
                            "prompt_split '%s' not found in local DatasetDict at %s; falling back to 'train' split.",
                            self.prompt_split,
                            args.prompt_data,
                        )
                        train_data = hf_local_data["train"]
                    else:
                        raise ValueError(
                            f"Local DatasetDict at {args.prompt_data} does not contain split "
                            f"'{self.prompt_split}' or 'train'. Available splits: {list(hf_local_data.keys())}"
                        )
                else:
                    # Single-split Dataset saved via Dataset.save_to_disk
                    train_data = hf_local_data

                prompts_dataset = QADataset(
                    train_data,
                    self.tokenizer,
                    strategy,
                    max_samples=args.max_samples,
                    separate_prompt_label=False,
                    seq_len=args.prompt_max_len,
                )
            else:
                if DatatroveFolderDataset is None:
                    raise RuntimeError(
                        f"prompt_data directory '{args.prompt_data}' is not a HuggingFace dataset loadable via "
                        f"load_from_disk, and Datatrove is not installed. Original load_from_disk error: {hf_load_error}"
                    )

                train_data = DatatroveFolderDataset(
                    folder_path=args.prompt_data,
                    seq_len=args.prompt_max_len,
                    token_size=(2 if self.tokenizer.vocab_size < 65535 else 4),
                    shuffle=True,
                    seed=args.seed,
                    # return_positions=False,      # we don’t need them
                )
                prompts_dataset = SequenceDataset(
                    train_data,
                    self.tokenizer,
                    max_samples=args.max_samples,
                )
        elif args.prompt_data.endswith('.jsonl') or args.prompt_data.endswith('.json') or args.prompt_data.endswith('.parquet'):
            train_data = blending_datasets(args.prompt_data, None, strategy, dataset_split=self.prompt_split)
            # NOTE: max_samples will be applied AFTER packing inside QADataset, not to individual examples
            prompts_dataset = QADataset(
                train_data,
                self.tokenizer,
                strategy,
                max_samples=args.max_samples,
                separate_prompt_label=False,
                seq_len=args.prompt_max_len,
            )
        elif os.path.exists(args.prompt_data) and len(args.prompt_data.split("/")) > 2 and not args.prompt_data.endswith('.json'):
            if DatatroveFolderDataset is None:
                raise RuntimeError(
                    "DatatroveFolderDataset is unavailable because datatrove is not installed, "
                    f"but prompt_data='{args.prompt_data}' was inferred as a Datatrove folder path."
                )

            train_data = DatatroveFolderDataset(
                folder_path=args.prompt_data,
                seq_len=args.prompt_max_len,
                token_size=(2 if self.tokenizer.vocab_size < 65535 else 4),
                shuffle=True,
                seed=args.seed,
                # return_positions=False,      # we don’t need them
            )
            prompts_dataset = SequenceDataset(
                train_data,
                self.tokenizer,
                max_samples=args.max_samples,
            )
                   
            
        else:
            # prepare datasets
            if args.prompt_data == "openai/gsm8k":
                train_data = load_dataset(args.prompt_data, name='main')[self.prompt_split]
            else:
                train_data = load_dataset(args.prompt_data)[self.prompt_split]
            # NOTE: max_samples will be applied AFTER packing inside QADataset, not to individual examples
            prompts_dataset = QADataset(
                train_data, 
                self.tokenizer,
                strategy,
                max_samples=args.max_samples,
                separate_prompt_label=False,
                seq_len=args.prompt_max_len,
                )

        self.raw_question_texts = getattr(prompts_dataset, "prompts", None)
        if self.raw_question_texts:
            logger.info(
                "[DATA] Stored %d raw question texts (indexed by doc_id) for teacher queries",
                len(self.raw_question_texts),
            )

        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset,
            args.rollout_batch_size,
            self.dataloader_pin_memory,  # Use the configurable pin_memory setting
            True,
            collate_fn=prompts_dataset.collate_fn,
            drop_last=True,
        )

        eval_dataloader = None
        eval_downstream_dataloader = None
        humaneval_dataloader = None
        if getattr(args, "eval_dataset", None):
             # prepare datasets
            if args.eval_dataset == "openai/gsm8k":
                eval_data = load_dataset(args.eval_dataset, name='main')[self.eval_split]
            elif "swallow_code" in args.eval_dataset:
                eval_data = blending_datasets(args.eval_dataset, None, strategy, dataset_split=self.eval_split)
            elif args.eval_dataset.endswith('.jsonl') or args.eval_dataset.endswith('.json') or args.eval_dataset.endswith('.parquet'):
                # Local file support using blending_datasets
                eval_data = blending_datasets(args.eval_dataset, None, strategy, dataset_split=self.eval_split)
            else:
                eval_data = load_dataset(args.eval_dataset)[self.eval_split]

            # NOTE: max_samples will be applied AFTER packing inside QADataset, not to individual examples
            eval_dataset = QADataset(
                eval_data, 
                self.tokenizer,
                strategy,
                max_samples=args.eval_max_samples,
                separate_prompt_label=False,
                seq_len=args.prompt_max_len,
                )

            humaneval_dataloader = None
            if "opencode-instruct_100k_200tok" in args.eval_dataset or "swallow_code" in args.eval_dataset:


                eval_downstream_dataset = CodePromptDataset(eval_data, self.tokenizer, strategy)


                humaneval_data = blending_datasets("openai/openai_humaneval", None, strategy, dataset_split="test")
                humaneval_data = humaneval_data.select(range(min(args.eval_down_max_samples, len(humaneval_data))))
                humaneval_dataset = HumanEvalDataset(humaneval_data, self.tokenizer, strategy)
                humaneval_dataloader = strategy.setup_dataloader(
                    humaneval_dataset,
                    args.eval_down_batch_size,
                    False,
                    False,
                    collate_fn=humaneval_dataset.collate_fn,
                    drop_last=False,
                )

            else:
                eval_downstream_dataset = QADataset(
                    eval_data,
                    self.tokenizer,
                    strategy,
                    args.eval_down_max_samples,
                    separate_prompt_label=True
                )

            eval_dataloader = strategy.setup_dataloader(
                eval_dataset,
                args.eval_batch_size,
                True,
                False,
                collate_fn=eval_dataset.collate_fn,
                drop_last=False,
            )

            eval_downstream_dataloader = strategy.setup_dataloader(
                eval_downstream_dataset,
                args.eval_batch_size,
                True,
                False,
                collate_fn=eval_downstream_dataset.collate_fn,
                drop_last=False,
            )
        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_downstream_dataloader = eval_downstream_dataloader
        self.humaneval_dataloader = humaneval_dataloader

        # Lookahead helper for the teacher prefetch path. Computes the
        # next-K-batches' unique doc_ids without consuming the dataloader,
        # so background threads can prefetch teacher answers during
        # ppo_train(). Cheap to construct (no work done until first peek).
        # Only useful when the dataset path populated `.doc_ids` and
        # `.prompts` (QADataset packed mode). For SequenceDataset / other
        # paths PromptLookahead.peek_prompts_at_offset() will return [].
        try:
            self.prompt_lookahead = PromptLookahead(
                dataset=prompts_dataset,
                sampler=self.prompts_dataloader.sampler,
                batch_size=args.rollout_batch_size,
                drop_last=True,
            )
            logger.info(
                "[PromptLookahead] init OK: %d total prompts indexed by doc_id",
                self.prompt_lookahead.total_prompts(),
            )
        except Exception as exc:
            # Don't fail trainer init just because lookahead can't be built;
            # prefetch will gracefully degrade to no-op.
            logger.warning("[PromptLookahead] init failed: %s — prefetch disabled", exc)
            self.prompt_lookahead = None
        self.max_steps = (
            len(prompts_dataset)
            * args.n_samples_per_prompt
            // args.train_batch_size
            * args.num_episodes
            * args.max_epochs
        )
        logger.info("max_steps=%s", self.max_steps)

    def get_max_steps(self):
        return self.max_steps
    

@ray.remote
class EBFTTrainer(BaseEBFTTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO) / REINFORCE++ / GRPO / RLOO and their variants.
    Single Controller with Multiple ActorGroups
    """

    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_groups: List[RayActorGroup],
        reference_model_group: RayActorGroup,
        teacher_model_group: RayActorGroup = None,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_groups,
            reference_model_group,
            teacher_model_group=teacher_model_group,
            dataloader_pin_memory=dataloader_pin_memory,
            prompt_split=prompt_split,
            eval_split=eval_split,
            **generate_kwargs,
        )

        if self.kl_loss_coef:
            self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.kl_loss_coef, self.kl_loss_decay_steps)
        else:
            self.kl_ctl = FixedKLController(self.init_kl_coef)

        if self.ce_loss_decay_steps:
            self.ce_ctl = AdaptiveCEController(self.ce_loss_coef, self.ce_loss_decay_steps)
        else:
            self.ce_ctl = FixedCEController(self.ce_loss_coef)

        if self.rl_loss_warmup_steps and self.rl_loss_warmup_start:
            self.rl_ctl = AdaptiveRLController(self.rl_loss_coef, self.rl_loss_warmup_start, self.rl_loss_warmup_steps)

        else:
            self.rl_ctl = FixedRLController(self.rl_loss_coef)

        if self.args.remote_rm_url:
            from openrlhf.utils.remote_rm_utils import RemoteRewardModel

            self.remote_reward_model = RemoteRewardModel.remote(self.args, self.remote_rm_url)

        self.samples_generator = self.generator_cls(
            self.actor_model_group,
            self.strategy,
            self.tokenizer,
        )

        self.teacher_generator = (
            self.generator_cls(self.teacher_model_group, self.strategy, self.tokenizer)
            if self.teacher_model_group is not None
            else None
        )

        from openrlhf.utils.teacher_provider import build_teacher_provider
        from openrlhf.utils.teacher_prefetch import wrap_provider_with_prefetch
        self.teacher_provider = build_teacher_provider(self.args)
        self.teacher_provider = wrap_provider_with_prefetch(
            self.teacher_provider, self.args
        )

        teacher_backend = getattr(self.args, "teacher_backend", "local")

        if teacher_backend == "remote":
            assert self.teacher_provider is not None, (
                "teacher_backend=remote but build_teacher_provider returned None"
            )
            assert self.teacher_model_group is None, (
                "teacher_backend=remote but teacher_model_group is not None — "
                "local teacher model must not be loaded when using remote backend"
            )

        if teacher_backend == "dataset":
            assert self.teacher_provider is not None, (
                "teacher_backend=dataset but build_teacher_provider returned None"
            )
            assert self.teacher_model_group is None, (
                "teacher_backend=dataset but teacher_model_group is not None — "
                "local teacher model must not be loaded when using dataset backend"
            )

        logger.info(
            f"[TEACHER-VERIFY] teacher_backend={teacher_backend}, "
            f"teacher_model_group={'PRESENT' if self.teacher_model_group else 'None'}, "
            f"teacher_generator={'PRESENT' if self.teacher_generator else 'None'}, "
            f"teacher_provider={'PRESENT' if self.teacher_provider else 'None'}"
        )

        primary_reward_group = self._ensure_primary_reward_model_group()

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            primary_reward_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            self.tokenizer,
            teacher_samples_generator=self.teacher_generator,
            teacher_provider=self.teacher_provider,
            raw_question_texts=getattr(self, "raw_question_texts", None),
        )
        logger.info(
            f"[TEACHER-VERIFY] RemoteExperienceMaker: "
            f"teacher_samples_generator={'PRESENT' if self.experience_maker.teacher_samples_generator else 'None'}, "
            f"teacher_provider={'PRESENT' if self.experience_maker.teacher_provider else 'None'}"
        )

        self.prepare_datasets()
        self._init_wandb()

        # get eval and save steps
        if self.args.eval_steps == -1:
            self.args.eval_steps = float("inf")  # do not evaluate
        if self.args.eval_down_steps == -1:
            self.args.eval_down_steps = float("inf")  # do not evaluate
        if self.args.save_steps == -1:
            self.args.save_steps = float("inf")  # do not save ckpt unless another trigger is configured

        steps_to_save = None
        save_epoch_fractions = getattr(self.args, "save_epoch_fractions", None) or []
        stop_after_epoch_fraction = getattr(self.args, "stop_after_epoch_fraction", None)

        if save_epoch_fractions:
            steps_to_save = self._build_fractional_save_steps(self.max_steps, save_epoch_fractions)
        elif math.isinf(self.args.save_steps):
            if self.args.save_log_scale_count != -1:
                total_steps = self.args.num_episodes * self.prompts_dataloader.__len__()
                logspace = np.logspace(-2.1, 0, self.args.save_log_scale_count) * total_steps
                steps_to_save = [int(n) for n in logspace]
            else:
                save_even_count = getattr(self.args, "save_even_count", 0)
                if save_even_count and save_even_count > 0:
                    steps_to_save = self._build_evenly_spaced_save_steps(self.max_steps, save_even_count)

        stop_after_steps = None
        if stop_after_epoch_fraction is not None:
            stop_after_steps = self._fraction_to_step(self.max_steps, stop_after_epoch_fraction)
            if steps_to_save is not None:
                steps_to_save = [step for step in steps_to_save if step <= stop_after_steps]
                steps_to_save = sorted(set(steps_to_save + [stop_after_steps]))

        self.args.steps_to_save = steps_to_save
        self.args.stop_after_steps = stop_after_steps

        logger.info(f"max steps: {self.max_steps}")
        logger.info(f"save steps: {self.args.steps_to_save}")
        logger.info(f"stop after steps: {self.args.stop_after_steps}")



    def _build_evenly_spaced_save_steps(self, total_steps: int, count: int) -> List[int]:
        if total_steps <= 0 or count <= 0:
            return []
        return [max(1, math.floor(i * total_steps / count)) for i in range(1, count + 1)]

    def _fraction_to_step(self, total_steps: int, fraction: float) -> int:
        if total_steps <= 0:
            return 0
        return max(1, min(total_steps, math.floor(total_steps * fraction)))

    def _build_fractional_save_steps(self, total_steps: int, fractions: List[float]) -> List[int]:
        if total_steps <= 0 or not fractions:
            return []
        return sorted({self._fraction_to_step(total_steps, fraction) for fraction in fractions if 0 < fraction <= 1})

    def fit( 
        self,
    ) -> None:
        args = self.args

        # broadcast init checkpoint to vllm
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_actor_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            logger.info(f"checkpoint_states: {checkpoint_states}")
            self._broadcast_to_vllm()
        else:
            checkpoint_states = {"global_step": 0, "episode": 0, "data_loader_state_dict": {}}

        # Restore step and start_epoch
        steps = checkpoint_states["global_step"] + 1
        episode = checkpoint_states["episode"]
        data_loader_state_dict = checkpoint_states["data_loader_state_dict"]
        if data_loader_state_dict:
            self.prompts_dataloader.load_state_dict(data_loader_state_dict)



        # Perform initial evaluation at step 0 (before training starts)
        #
        # Gating contract: only run the step-0 initial eval if the user has
        # configured at least one in-training eval to actually fire during
        # this run. The check is:
        #   (1) eval_steps must not be the "disabled" sentinel (inf, set by
        #       the eval_steps == -1 normalization above), AND
        #   (2) eval_steps must be a positive int (defensive against 0 /
        #       negative values that would never trigger the modulo check
        #       at line 269), AND
        #   (3) eval_steps must be <= self.max_steps; otherwise the user
        #       has effectively disabled in-training eval by setting a value
        #       larger than the total training steps (e.g. EVAL_STEPS=999999
        #       with TARGET_STEPS=500), and we should NOT silently run a
        #       full step-0 eval anyway.
        #
        # Why the max_steps guard matters: under ZeRO-3 colocate_actor_ref
        # the step-0 evaluate() funnels samples through SamplesGenerator
        # which round-robins one Ray RPC per micro-batch across actor ranks
        # that share an NCCL world. If the per-actor RPC count is uneven
        # (e.g. eval_max_samples * eval_n_samples_per_prompt /
        # micro_rollout_batch_size not divisible by actor_gpus), the actors
        # with extra RPCs hang inside _ALLGATHER_BASE waiting for actors
        # that already returned, and watchdog kills the run after 1h.
        # Treating "big finite eval_steps" as disabled keeps that footgun
        # out of misconfigured launchers (see scripts/run_G2_rebase*.sh).
        if steps == 1:
            run_initial_down_eval = (
                not math.isinf(self.args.eval_down_steps)
                and self.args.eval_down_steps > 0
                and self.args.eval_down_steps <= self.max_steps
            )
            if run_initial_down_eval:
                logger.info("🔍 Running initial evaluation at step 0...")
                eval_ds = str(getattr(self.args, "eval_dataset", "") or "").lower()
                if "swallow_code" in eval_ds:
                    self.evaluate_downstream_humaneval(
                        self.humaneval_dataloader,
                        0,
                        args.eval_generate_max_len,
                        args.eval_temperature_down,
                        args.eval_n_samples_per_prompt_down,
                    )
                elif "gsm8k" in eval_ds or "math" in eval_ds or "aops" in eval_ds:
                    self.evaluate_downstream_gsm8k_math(self.eval_downstream_dataloader, 0, args.eval_generate_max_len, args.eval_temperature_down, args.eval_n_samples_per_prompt_down)
                elif "opencode-instruct_100k_200tok" in eval_ds:
                    self.evaluate_downstream_opencode(self.eval_downstream_dataloader, 0, args.eval_generate_max_len, args.eval_temperature_down, args.eval_n_samples_per_prompt_down)
                else:
                     self.evaluate_downstream_translation(self.eval_downstream_dataloader, 0, args.eval_generate_max_len, args.eval_temperature_down, args.eval_n_samples_per_prompt_down)
            run_initial_eval = (
                not math.isinf(self.args.eval_steps)
                and self.args.eval_steps > 0
                and self.args.eval_steps <= self.max_steps
            )
            if run_initial_eval:
                logger.info("🔍 Running initial evaluation at step 0...")
                self.evaluate(self.eval_dataloader, 0, args.eval_temperature, args.eval_n_samples_per_prompt)
            elif not math.isinf(self.args.eval_steps):
                logger.info(
                    "Skipping step-0 initial eval: eval_steps=%s is outside (0, max_steps=%s]; "
                    "treating as disabled. Pass --eval_steps -1 to silence this message.",
                    self.args.eval_steps, self.max_steps,
                )



        stop_training = False
        for episode in range(episode, args.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=False,
            )

            # Make sure the sampler reseeds for this epoch (re-shuffles).
            # Trainer's previous behaviour relied on the implicit `epoch=0`
            # never advancing; we now explicitly call set_epoch so
            # PromptLookahead picks up the new permutation.
            try:
                self.prompts_dataloader.sampler.set_epoch(episode)
            except Exception:
                pass

            # Counter for "how many batches have we consumed in THIS
            # episode on this rank so far". Used by the teacher prefetch
            # lookahead to peek the next batch's prompts without
            # consuming the dataloader. Resets to 0 each episode.
            #
            # NOTE: on a checkpoint resume mid-epoch, the dataloader
            # restores via state_dict and yields only the remaining
            # batches, but this counter starts at 0. That means for the
            # first few resumed steps the lookahead will look at the
            # WRONG slice of the index list (off by `pre_resume_batches`).
            # Side effect: a handful of wasted prefetches that get cached
            # but never consumed. Not a correctness bug — sample_targets
            # always falls back to the synchronous path on cache miss.
            _within_epoch_batch_idx = 0

            filtered_samples = []
            number_of_samples = 0
            for sequence_ids, doc_ids, qa_masks in self.prompts_dataloader:
                remote_reward_model = self.remote_reward_model if self.args.dynamic_filtering else None

                rollout_samples = self.samples_generator.generate_samples(
                    sequence_ids, doc_ids, qa_masks, remote_reward_model=remote_reward_model, **self.generate_kwargs
                )
                pbar.update()
 
                # dynamic filtering
                pass_rate = None
                if self.args.dynamic_filtering:
                    number_of_samples += len(rollout_samples)
                    # Group individual samples into batches of n_samples size
                    for i in range(0, len(rollout_samples), self.args.n_samples_per_prompt):
                        batch_samples = rollout_samples[i : i + self.args.n_samples_per_prompt]
                        if len(batch_samples) < self.args.n_samples_per_prompt:
                            continue

                        # Calculate average reward for this batch of samples
                        avg_reward = sum(sample.scores[0].item() for sample in batch_samples) / len(batch_samples)

                        # Check if average reward is within the specified range
                        min_reward, max_reward = self.args.dynamic_filtering_reward_range
                        if min_reward + 1e-6 < avg_reward < max_reward - 1e-6:
                            filtered_samples.extend(batch_samples)

                    # Continue sampling if filtered samples are insufficient
                    if len(filtered_samples) / self.args.n_samples_per_prompt < self.args.rollout_batch_size:
                        logger.info(
                            f"filtered_samples {len(filtered_samples) / self.args.n_samples_per_prompt} < rollout_batch_size {self.args.rollout_batch_size}, continue sampling"
                        )
                        continue

                    pass_rate = len(filtered_samples) / number_of_samples * 100
                    logger.info(
                        f"Dynamic filtering pass rate: {pass_rate:.2f}% ({len(filtered_samples)}/{number_of_samples})"
                    )
                    rollout_samples = filtered_samples[: self.args.rollout_batch_size * self.args.n_samples_per_prompt]
                    filtered_samples = []
                    number_of_samples = 0



                # critic train
                status = None
                if self.critic_model_group is not None:
                    experiences = self.experience_maker.assign_sample_indices(rollout_samples)
                    # flush buffer before appending new experiences
                    ray.get(self.critic_model_group.async_run_method(method_name="flush_buffer"))
                    refs = self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences)
                    status = self.critic_train(steps)

                # make experience batch with the updated critic model
                experiences = self.experience_maker.make_experience_batch(rollout_samples)

                # ── Teacher prefetch: lookahead into NEXT K batches ──
                # While GPU runs ppo_train below (~10s on 0.8B student),
                # background threads fetch teacher completions for the
                # prompts of step t+1 .. t+prefetch_depth. Then those
                # prompts are already in the mem queue / SQLite cache
                # when next step's make_experience_batch calls
                # sample_targets, hiding teacher latency.
                #
                # The PREVIOUS implementation (pre 2026-04-29) scheduled
                # the CURRENT step's prompts (which sample_targets above
                # had just queried), so the mem-queue hit rate was ~0%
                # and only SQLite cross-episode hits were saved. Fixed
                # by going through PromptLookahead.peek_prompts_at_offset.
                #
                # Guarded by --enable_teacher_prefetch; zero cost when off.
                _tp = self.teacher_provider
                if (
                    getattr(args, "enable_teacher_prefetch", False)
                    and _tp is not None
                    and hasattr(_tp, "schedule_prefetch")
                    and getattr(self, "prompt_lookahead", None) is not None
                ):
                    _depth = int(getattr(args, "prefetch_depth", 4))
                    # Number of sampler indices consumed on this rank
                    # AFTER processing this current step (we just finished
                    # iter (_within_epoch_batch_idx + 1) of the inner loop).
                    _consumed_so_far = (_within_epoch_batch_idx + 1) * args.rollout_batch_size
                    _prefetch_prompts = self.prompt_lookahead.peek_prompts_at_offset(
                        num_consumed_in_epoch_on_this_rank=_consumed_so_far,
                        num_steps=_depth,
                    )
                    if _prefetch_prompts:
                        _n_submitted = _tp.schedule_prefetch(_prefetch_prompts)
                        _pf_status = _tp.status() if hasattr(_tp, "status") else {}
                        logger.info(
                            "[Prefetch] step=%d depth=%d unique_prompts=%d submitted=%d "
                            "(mem_q=%d sqlite_h=%d fallback=%d)",
                            steps, _depth, len(_prefetch_prompts), _n_submitted,
                            _pf_status.get("prefetch_mem_hits", 0),
                            _pf_status.get("prefetch_sqlite_hits", 0),
                            _pf_status.get("prefetch_fallback", 0),
                        )


                # balance experiences across dp
                if args.use_dynamic_batch:
                    experiences = balance_experiences(experiences, args)

                # Ensure replay buffer is empty before appending new experiences
                ray.get(self.actor_model_group.async_run_method(method_name="flush_buffer"))
                refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)

              
                ray.get(refs)


                status = self.ppo_train(steps, status=status)

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

                self.rl_ctl.update(steps)
                self.ce_ctl.update(steps)

                # Add generated samples to status dictionary
                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = pass_rate
                logger.info(f"✨ Global step {steps}: {status}")

                # Log prefetch stats alongside step status
                if (
                    getattr(args, "enable_teacher_prefetch", False)
                    and self.teacher_provider is not None
                    and hasattr(self.teacher_provider, "status")
                ):
                    _pf = self.teacher_provider.status()
                    logger.info(
                        "[Prefetch] step=%d hit_rate=%.1f%% "
                        "mem=%d sqlite=%d fallback=%d queue=%d pending=%d",
                        steps,
                        _pf.get("prefetch_hit_rate", 0.0) * 100,
                        _pf.get("prefetch_mem_hits", 0),
                        _pf.get("prefetch_sqlite_hits", 0),
                        _pf.get("prefetch_fallback", 0),
                        _pf.get("prefetch_queue_size", 0),
                        _pf.get("prefetch_pending", 0),
                    )
                    status.update(_pf)
 
                # logs/checkpoints
                client_states = {
                    "global_step": steps,
                    "episode": episode,
                    "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                }
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                if getattr(args, "stop_after_steps", None) is not None and steps >= args.stop_after_steps:
                    logger.info(
                        "Reached stop_after_steps=%s (epoch fraction=%s); stopping training early.",
                        args.stop_after_steps,
                        getattr(args, "stop_after_epoch_fraction", None),
                    )
                    stop_training = True
                    break

                steps = steps + 1
                _within_epoch_batch_idx += 1

            if stop_training:
                break

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()
