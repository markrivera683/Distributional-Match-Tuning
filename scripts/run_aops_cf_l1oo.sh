#!/usr/bin/env bash
set -euo pipefail

# AoPS Ours Script - CF L1OO Reward
# Usage: MAX_SAMPLES=64 NUM_EPISODES=1 bash scripts/run_aops_cf_l1oo.sh

# This script is identical to run_aops_baseline.sh except for DISTRIBUTION_REWARD_TYPE
# This ensures a fair comparison between baseline and ours

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/.cache/huggingface/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.995}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENRLHF_RAY_OBJECT_STORE_MEMORY_BYTES="${OPENRLHF_RAY_OBJECT_STORE_MEMORY_BYTES:-8589934592}"

# Update paths for current environment
REPO_ROOT="${REPO_ROOT:-/mnt/workspace/EBFT/ebft_only_bundle_march19/repo/ebft_openrlhf_u1_ncfm}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/EBFT/ebft_only_bundle_march19/model/Qwen2___5-1___5B}"

# AoPS Data Paths
TRAIN_DATA="${TRAIN_DATA:-/mnt/aops_train_processed}"
EVAL_DATA="${EVAL_DATA:-/mnt/workspace/data/AoPS-Instruct-test/LiveAoPSBench-2024.jsonl}"

# Reward Type: cf_l1oo for ours (this is the ONLY difference from baseline)
DISTRIBUTION_REWARD_TYPE="${DISTRIBUTION_REWARD_TYPE:-cf_l1oo}"

# Feature Map Settings
FEATURE_MAP_TYPE="${FEATURE_MAP_TYPE:-identity}"
RFF_NUM_FEATURES="${RFF_NUM_FEATURES:-128}"
RFF_SIGMA="${RFF_SIGMA:-1.0}"
RFF_SEED="${RFF_SEED:-43}"

# CF Settings
CF_NUM_FREQS="${CF_NUM_FREQS:-128}"
CF_SIGMA="${CF_SIGMA:-1.0}"
CF_SEED="${CF_SEED:-43}"
CF_ALPHA="${CF_ALPHA:-0.5}"
CF_BETA="${CF_BETA:-0.5}"
CF_REWARD_SCALE="${CF_REWARD_SCALE:-1.0}"
CF_TARGET_MODE="${CF_TARGET_MODE:-single}"  # Single target, no vicinal
CF_TARGET_NUM_REFS="${CF_TARGET_NUM_REFS:-1}"
CF_TARGET_STD="${CF_TARGET_STD:-0.05}"
CF_TARGET_SEED="${CF_TARGET_SEED:-43}"

# Architecture: sequence-level only
EMBED_METHOD="${EMBED_METHOD:-last_token}"
CRITIC_SEQUENCE_LEVEL="${CRITIC_SEQUENCE_LEVEL:-last_token}"

# Disable 2nd order features
FEATURE_ADAPTER_ENABLE="${FEATURE_ADAPTER_ENABLE:-0}"
FEATURE_ADAPTER_TYPE="${FEATURE_ADAPTER_TYPE:-residual_bottleneck}"
FEATURE_ADAPTER_RANK="${FEATURE_ADAPTER_RANK:-64}"
FEATURE_ADAPTER_DROPOUT="${FEATURE_ADAPTER_DROPOUT:-0.0}"
CRITIC_DIRECT_DISCREPANCY_COEF="${CRITIC_DIRECT_DISCREPANCY_COEF:-0.0}"
CRITIC_DIRECT_DISCREPANCY_TARGET="${CRITIC_DIRECT_DISCREPANCY_TARGET:-ema_gt}"

# Loss coefficients (identical to baseline)
CE_LOSS_COEF="${CE_LOSS_COEF:-0.03}"
DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-0.5}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"
CRITIC_LEARNING_RATE="${CRITIC_LEARNING_RATE:-0}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-0}"
CRITIC_CLASSIFIER_LOSS_COEF="${CRITIC_CLASSIFIER_LOSS_COEF:-0.0}"

# Training settings
MAX_SAMPLES="${MAX_SAMPLES:--1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
EVAL_DOWN_MAX_SAMPLES="${EVAL_DOWN_MAX_SAMPLES:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_DOWN_BATCH_SIZE="${EVAL_DOWN_BATCH_SIZE:-128}"
EVAL_STEPS="${EVAL_STEPS:-100}"
EVAL_DOWN_STEPS="${EVAL_DOWN_STEPS:-50}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-512}"
EVAL_N_SAMPLES_PER_PROMPT="${EVAL_N_SAMPLES_PER_PROMPT:-4}"
EVAL_N_SAMPLES_PER_PROMPT_DOWN="${EVAL_N_SAMPLES_PER_PROMPT_DOWN:-4}"
GLOBAL_SEED="${GLOBAL_SEED:-43}"
NUM_EPISODES="${NUM_EPISODES:-2}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-0}"
GRADIENT_CHECKPOINTING_USE_REENTRANT="${GRADIENT_CHECKPOINTING_USE_REENTRANT:-0}"
EMA_BETA="${EMA_BETA:-0.9}"

RUN_TAG="${RUN_TAG:-aops_ours_${DISTRIBUTION_REWARD_TYPE}_seed${GLOBAL_SEED}}"
RUN_ROOT="${RUN_ROOT:-/mnt/workspace/runs/${RUN_TAG}}"
SAVE_PATH="${SAVE_PATH:-${RUN_ROOT}/model}"
TB_ROOT="${TB_ROOT:-${RUN_ROOT}/tensorboard}"
RUN_NAME="${RUN_NAME:-${RUN_TAG}}"

mkdir -p "${RUN_ROOT}" "${SAVE_PATH}" "${TB_ROOT}"
exec > >(tee -a "${RUN_ROOT}/train.log") 2>&1

has_model_weights() {
  find "${MODEL_PATH}" -maxdepth 2 -type f \
    \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \) | grep -q .
}

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing MODEL_PATH: ${MODEL_PATH}"
  exit 2
fi

if ! has_model_weights; then
  echo "MODEL_PATH exists but no model weight files were found: ${MODEL_PATH}"
  exit 3
fi

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Starting AoPS cf_l1oo run"
echo "REPO_ROOT=${REPO_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_DATA=${TRAIN_DATA}"
echo "EVAL_DATA=${EVAL_DATA}"
echo "DISTRIBUTION_REWARD_TYPE=${DISTRIBUTION_REWARD_TYPE}"
echo "EMBED_METHOD=${EMBED_METHOD}"
echo "CF_TARGET_MODE=${CF_TARGET_MODE}"

cd "${REPO_ROOT}"

EXTRA_ARGS=()
if [[ "${ENABLE_GRADIENT_CHECKPOINTING}" == "1" ]]; then
  EXTRA_ARGS+=(--gradient_checkpointing)
fi
if [[ "${GRADIENT_CHECKPOINTING_USE_REENTRANT}" == "1" ]]; then
  EXTRA_ARGS+=(--gradient_checkpointing_use_reentrant)
fi
if [[ "${FEATURE_ADAPTER_ENABLE}" == "1" ]]; then
  EXTRA_ARGS+=(--feature_adapter_enable)
fi

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
  --feature_map_type "${FEATURE_MAP_TYPE}" \
  --rff_num_features "${RFF_NUM_FEATURES}" \
  --rff_sigma "${RFF_SIGMA}" \
  --rff_seed "${RFF_SEED}" \
  --distribution_reward_type "${DISTRIBUTION_REWARD_TYPE}" \
  --cf_num_freqs "${CF_NUM_FREQS}" \
  --cf_sigma "${CF_SIGMA}" \
  --cf_seed "${CF_SEED}" \
  --cf_alpha "${CF_ALPHA}" \
  --cf_beta "${CF_BETA}" \
  --cf_reward_scale "${CF_REWARD_SCALE}" \
  --cf_target_mode "${CF_TARGET_MODE}" \
  --cf_target_num_refs "${CF_TARGET_NUM_REFS}" \
  --cf_target_std "${CF_TARGET_STD}" \
  --cf_target_seed "${CF_TARGET_SEED}" \
  --feature_adapter_type "${FEATURE_ADAPTER_TYPE}" \
  --feature_adapter_rank "${FEATURE_ADAPTER_RANK}" \
  --feature_adapter_dropout "${FEATURE_ADAPTER_DROPOUT}" \
  --critic_direct_discrepancy_coef "${CRITIC_DIRECT_DISCREPANCY_COEF}" \
  --critic_direct_discrepancy_target "${CRITIC_DIRECT_DISCREPANCY_TARGET}" \
  --pretrain "${MODEL_PATH}" \
  --critic_pretrain "${MODEL_PATH}" \
  --prompt_data "${TRAIN_DATA}" \
  --eval_dataset "${EVAL_DATA}" \
  --input_key question \
  --label_key answer \
  --output_key answer \
  --prompt_split train \
  --eval_split test \
  --prompt_max_len 1024 \
  --context_max_len 8 \
  --generate_max_len 8 \
  --stride 8 \
  --n_samples_per_prompt 4 \
  --rollout_batch_size 16 \
  --train_batch_size 64 \
  --micro_train_batch_size 8 \
  --micro_rollout_batch_size 8 \
  --micro_reward_batch_size 8 \
  --max_samples "${MAX_SAMPLES}" \
  --eval_max_samples "${EVAL_MAX_SAMPLES}" \
  --eval_down_max_samples "${EVAL_DOWN_MAX_SAMPLES}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --eval_down_batch_size "${EVAL_DOWN_BATCH_SIZE}" \
  --max_epochs 1 \
  --num_episodes "${NUM_EPISODES}" \
  --ref_num_nodes "${REF_NUM_NODES:-1}" \
  --ref_num_gpus_per_node "${REF_NUM_GPUS_PER_NODE:-1}" \
  --critic_num_nodes "${CRITIC_NUM_NODES:-1}" \
  --critic_num_gpus_per_node "${CRITIC_NUM_GPUS_PER_NODE:-1}" \
  --actor_num_nodes "${ACTOR_NUM_NODES:-1}" \
  --actor_num_gpus_per_node "${ACTOR_NUM_GPUS_PER_NODE:-1}" \
  --reward_num_nodes "${REWARD_NUM_NODES:-1}" \
  --reward_num_gpus_per_node "${REWARD_NUM_GPUS_PER_NODE:-1}" \
  --advantage_estimator rloo \
  --init_kl_coef 0.0 \
  --kl_estimator k2 \
  --temperature 0.6 \
  --top_p 1.0 \
  --ce_loss_coef "${CE_LOSS_COEF}" \
  --rl_loss_coef 1.0 \
  --diversity_rew_coef "${DIVERSITY_REW_COEF}" \
  --alignment_rew_coef "${ALIGNMENT_REW_COEF}" \
  --critic_learning_rate "${CRITIC_LEARNING_RATE}" \
  --critic_lr_head "${CRITIC_LR_HEAD}" \
  --critic_classifier_loss_coef "${CRITIC_CLASSIFIER_LOSS_COEF}" \
  --actor_learning_rate 1e-6 \
  --zero_stage 2 \
  --lr_warmup_ratio 0.03 \
  --lr_scheduler constant_with_warmup \
  --critic_lr_scheduler constant_with_warmup \
  --seed "${GLOBAL_SEED}" \
  --ema_beta "${EMA_BETA}" \
  --hidden_state_method concat \
  --embed_method "${EMBED_METHOD}" \
  --critic_sequence_level "${CRITIC_SEQUENCE_LEVEL}" \
  --classifier_sequence_selection closest \
  --eval_generate_max_len "${EVAL_GENERATE_MAX_LEN}" \
  --eval_n_samples_per_prompt "${EVAL_N_SAMPLES_PER_PROMPT}" \
  --eval_n_samples_per_prompt_down "${EVAL_N_SAMPLES_PER_PROMPT_DOWN}" \
  --eval_steps "${EVAL_STEPS}" \
  --eval_down_steps "${EVAL_DOWN_STEPS}" \
  --save_steps -1 \
  --save_log_scale_count 10 \
  --logging_steps 1 \
  --use_tensorboard "${TB_ROOT}" \
  --save_path "${SAVE_PATH}" \
  --ckpt_path "${SAVE_PATH}/ckpt" \
  --wandb_run_name "${RUN_NAME}" \
  "${EXTRA_ARGS[@]}"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] AoPS cf_l1oo run finished"
