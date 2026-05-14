#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/Energy/ebft_openrlhf_stepwise}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/RWICL/models/Qwen3-1.7B}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Energy/ebft_openrlhf_stepwise/runs/step0_vanilla_qwen3_smoke}"
SAVE_PATH="${SAVE_PATH:-${RUN_ROOT}/model}"
TB_ROOT="${TB_ROOT:-${RUN_ROOT}/tensorboard}"
RUN_NAME="${RUN_NAME:-step0_vanilla_qwen3_smoke}"

mkdir -p "${RUN_ROOT}" "${SAVE_PATH}" "${TB_ROOT}"

exec > >(tee -a "${RUN_ROOT}/train.log") 2>&1

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Starting Step 0 baseline run"
echo "REPO_ROOT=${REPO_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "SAVE_PATH=${SAVE_PATH}"
echo "TB_ROOT=${TB_ROOT}"

cd "${REPO_ROOT}"

python -m openrlhf.cli.train_ebft_ray \
  --bf16 \
  --flash_attn \
  --adam_offload \
  --pretrain_mode \
  --no_chat_template \
  --disable_ds_ckpt \
  --colocate_all_models \
  --use_kl_loss \
  --use_whitening \
  --enable_ema \
  --pretrain "${MODEL_PATH}" \
  --critic_pretrain "${MODEL_PATH}" \
  --prompt_data openai/gsm8k \
  --eval_dataset openai/gsm8k \
  --input_key question \
  --label_key answer \
  --output_key answer \
  --prompt_split train \
  --eval_split test \
  --prompt_max_len 128 \
  --context_max_len 8 \
  --generate_max_len 8 \
  --stride 8 \
  --n_samples_per_prompt 4 \
  --rollout_batch_size 1 \
  --train_batch_size 4 \
  --micro_train_batch_size 4 \
  --micro_rollout_batch_size 4 \
  --micro_reward_batch_size 2 \
  --max_samples 4 \
  --eval_max_samples 4 \
  --eval_down_max_samples 4 \
  --max_epochs 1 \
  --num_episodes 1 \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 1 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 1 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 1 \
  --reward_num_nodes 1 \
  --reward_num_gpus_per_node 1 \
  --advantage_estimator rloo \
  --init_kl_coef 0.0 \
  --kl_estimator k2 \
  --temperature 0.6 \
  --top_p 1.0 \
  --ce_loss_coef 0.03 \
  --rl_loss_coef 1.0 \
  --diversity_rew_coef 0.5 \
  --alignment_rew_coef 1.0 \
  --critic_learning_rate 0 \
  --critic_lr_head 0 \
  --critic_classifier_loss_coef 0.0 \
  --actor_learning_rate 1e-6 \
  --zero_stage 2 \
  --lr_warmup_ratio 0.03 \
  --lr_scheduler constant_with_warmup \
  --critic_lr_scheduler constant_with_warmup \
  --seed 43 \
  --ema_beta 0.9 \
  --hidden_state_method concat \
  --embed_method last_token \
  --critic_sequence_level last_token \
  --classifier_sequence_selection closest \
  --eval_generate_max_len 64 \
  --eval_n_samples_per_prompt 4 \
  --eval_n_samples_per_prompt_down 4 \
  --eval_steps 1 \
  --eval_down_steps 1 \
  --save_steps -1 \
  --save_log_scale_count -1 \
  --save_even_count 0 \
  --logging_steps 1 \
  --use_tensorboard "${TB_ROOT}" \
  --save_path "${SAVE_PATH}" \
  --ckpt_path "${SAVE_PATH}/ckpt" \
  --wandb_run_name "${RUN_NAME}"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Step 0 baseline run finished"
