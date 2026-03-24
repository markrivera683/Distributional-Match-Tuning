#!/usr/bin/env bash
set -euo pipefail

# Group 3: Dist+EMA (加入 distribution + EMA/feature network 第二点)
# - distribution_reward_type = cf_l1oo
# - sequence/block-level
# - 开启第二大点 2-lite：
#   - feature_adapter_enable = 1
#   - frozen backbone
#   - EMA target critic
# - 当前阶段不引入 token-level
# - 当前阶段不引入 vicinal target
# - 当前阶段不引入 direct discrepancy 额外项

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

# Paths
REPO_ROOT="${REPO_ROOT:-/mnt/workspace/EBFT/ebft_only_bundle_march19/repo/ebft_openrlhf_u1_ncfm}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/EBFT/ebft_only_bundle_march19/model/Qwen2___5-1___5B}"
TRAIN_DATA="${TRAIN_DATA:-/mnt/aops_train_processed}"
EVAL_DATA="${EVAL_DATA:-/mnt/workspace/data/AoPS-Instruct-test/LiveAoPSBench-2024.jsonl}"

# Group 3 Specific Parameters
DISTRIBUTION_REWARD_TYPE="cf_l1oo"
FEATURE_ADAPTER_ENABLE=1
FEATURE_ADAPTER_TYPE="residual_bottleneck"
FEATURE_ADAPTER_RANK=64
FEATURE_ADAPTER_DROPOUT=0.0
CRITIC_DIRECT_DISCREPANCY_COEF=0.0
EMBED_METHOD="last_token"
CF_TARGET_MODE="single"

# Critic enabled for Group 3 (train adapter)
CRITIC_LEARNING_RATE="${CRITIC_LEARNING_RATE:-9e-6}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-1e-5}"
CRITIC_CLASSIFIER_LOSS_COEF="${CRITIC_CLASSIFIER_LOSS_COEF:-1.0}"

# Common Parameters
FEATURE_MAP_TYPE="${FEATURE_MAP_TYPE:-identity}"
RFF_NUM_FEATURES="${RFF_NUM_FEATURES:-128}"
RFF_SIGMA="${RFF_SIGMA:-1.0}"
RFF_SEED="${RFF_SEED:-43}"
CF_NUM_FREQS="${CF_NUM_FREQS:-128}"
CF_SIGMA="${CF_SIGMA:-1.0}"
CF_SEED="${CF_SEED:-43}"
CF_ALPHA="${CF_ALPHA:-0.5}"
CF_BETA="${CF_BETA:-0.5}"
CF_REWARD_SCALE="${CF_REWARD_SCALE:-1.0}"
CF_TARGET_NUM_REFS="${CF_TARGET_NUM_REFS:-1}"
CF_TARGET_STD="${CF_TARGET_STD:-0.05}"
CF_TARGET_SEED="${CF_TARGET_SEED:-43}"
CRITIC_SEQUENCE_LEVEL="last_token"
CE_LOSS_COEF="${CE_LOSS_COEF:-0.03}"
DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-0.5}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"
EMA_BETA="${EMA_BETA:-0.99}"

# Training Settings
MAX_SAMPLES="${MAX_SAMPLES:-64}"
NUM_EPISODES="${NUM_EPISODES:-1}"

echo "TRAINING CONFIG:"
echo "  MAX_SAMPLES=${MAX_SAMPLES}"
echo "  NUM_EPISODES=${NUM_EPISODES}"
echo "  PROMPT_MAX_LEN=256 (hardcoded for speed)"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:--1}"
# Set to -1 to skip initial evaluation for fast diagnostic runs
EVAL_STEPS="${EVAL_STEPS:--1}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-512}"
GLOBAL_SEED="${GLOBAL_SEED:-43}"

# Group 3 Smoke: use 0 warmup to ensure adapter starts training immediately
CRITIC_LR_WARMUP_RATIO="${CRITIC_LR_WARMUP_RATIO:-0.03}"

# Run Configuration
RUN_TAG="aops_g3_distema_seed${GLOBAL_SEED}"
RUN_ROOT="${RUN_ROOT:-/mnt/workspace/runs/${RUN_TAG}}"
SAVE_PATH="${SAVE_PATH:-${RUN_ROOT}/model}"
TB_ROOT="${TB_ROOT:-${RUN_ROOT}/tensorboard}"

mkdir -p "${RUN_ROOT}" "${SAVE_PATH}" "${TB_ROOT}"
exec > >(tee -a "${RUN_ROOT}/train.log") 2>&1

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Starting AoPS Group 3: Dist+EMA"
echo "DISTRIBUTION_REWARD_TYPE=${DISTRIBUTION_REWARD_TYPE}"
echo "FEATURE_ADAPTER_ENABLE=${FEATURE_ADAPTER_ENABLE}"
echo "CRITIC_LEARNING_RATE=${CRITIC_LEARNING_RATE}"
echo "CRITIC_CLASSIFIER_LOSS_COEF=${CRITIC_CLASSIFIER_LOSS_COEF}"
echo "ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-1}"
echo "CRITIC_NUM_GPUS_PER_NODE=${CRITIC_NUM_GPUS_PER_NODE:-1}"

cd "${REPO_ROOT}"

EXTRA_ARGS=()
if [[ "${FEATURE_ADAPTER_ENABLE}" == "1" ]]; then
  EXTRA_ARGS+=(--feature_adapter_enable)
fi

python -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --adam_offload --pretrain_mode --no_chat_template \
  --disable_ds_ckpt --colocate_all_models --use_kl_loss --use_whitening --enable_ema \
  --distribution_reward_type "${DISTRIBUTION_REWARD_TYPE}" \
  --feature_map_type "${FEATURE_MAP_TYPE}" \
  --rff_num_features "${RFF_NUM_FEATURES}" \
  --rff_sigma "${RFF_SIGMA}" \
  --rff_seed "${RFF_SEED}" \
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
  --embed_method "${EMBED_METHOD}" \
  --critic_sequence_level "${CRITIC_SEQUENCE_LEVEL}" \
  --critic_learning_rate "${CRITIC_LEARNING_RATE}" \
  --critic_lr_head "${CRITIC_LR_HEAD}" \
  --critic_classifier_loss_coef "${CRITIC_CLASSIFIER_LOSS_COEF}" \
  --critic_direct_discrepancy_coef "${CRITIC_DIRECT_DISCREPANCY_COEF}" \
  --feature_adapter_type "${FEATURE_ADAPTER_TYPE}" \
  --feature_adapter_rank "${FEATURE_ADAPTER_RANK}" \
  --feature_adapter_dropout "${FEATURE_ADAPTER_DROPOUT}" \
  --ema_beta "${EMA_BETA}" \
  --ce_loss_coef "${CE_LOSS_COEF}" \
  --diversity_rew_coef "${DIVERSITY_REW_COEF}" \
  --alignment_rew_coef "${ALIGNMENT_REW_COEF}" \
  --pretrain "${MODEL_PATH}" \
  --critic_pretrain "${MODEL_PATH}" \
  --prompt_data "${TRAIN_DATA}" \
  --eval_dataset "${EVAL_DATA}" \
  --input_key question --label_key answer --output_key answer \
  --prompt_split train --eval_split test \
  --prompt_max_len 256 --context_max_len 8 --generate_max_len 8 --stride 8 \
  --n_samples_per_prompt 4 --rollout_batch_size 64 --train_batch_size 256 \
  --micro_train_batch_size 16 --micro_rollout_batch_size 16 --micro_reward_batch_size 16 \
  --max_samples "${MAX_SAMPLES}" --num_episodes "${NUM_EPISODES}" --max_epochs 1 \
  --actor_num_nodes 1 --actor_num_gpus_per_node "${ACTOR_NUM_GPUS_PER_NODE:-1}" \
  --critic_num_nodes 1 --critic_num_gpus_per_node "${CRITIC_NUM_GPUS_PER_NODE:-1}" \
  --ref_num_nodes 1 --ref_num_gpus_per_node "${REF_NUM_GPUS_PER_NODE:-1}" \
  --reward_num_nodes 1 --reward_num_gpus_per_node "${REWARD_NUM_GPUS_PER_NODE:-1}" \
  --advantage_estimator rloo --init_kl_coef 0.0 --kl_estimator k2 \
  --temperature 0.6 --top_p 1.0 --actor_learning_rate 1e-6 \
  --zero_stage 2 --lr_warmup_ratio 0.03 --lr_scheduler constant_with_warmup \
  --critic_lr_warmup_ratio "${CRITIC_LR_WARMUP_RATIO}" \
  --seed "${GLOBAL_SEED}" --eval_steps "${EVAL_STEPS}" \
  --eval_max_samples "${EVAL_MAX_SAMPLES}" --eval_generate_max_len "${EVAL_GENERATE_MAX_LEN}" \
  --save_steps -1 --logging_steps 1 --use_tensorboard "${TB_ROOT}" \
  --save_path "${SAVE_PATH}" --ckpt_path "${SAVE_PATH}/ckpt" \
  --wandb_run_name "${RUN_TAG}" \
  "${EXTRA_ARGS[@]}"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Group 3 Dist+EMA finished"
