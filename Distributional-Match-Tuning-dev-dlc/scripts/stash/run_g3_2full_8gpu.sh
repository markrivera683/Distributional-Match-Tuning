#!/usr/bin/env bash
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════╗
# ║  G3 Feature Adapter 2-full — 8×A100                            ║
# ║  Backbone top-N layers UNFROZEN + residual bottleneck adapter   ║
# ║  + distributional CF-L1OO reward (same as G2/G3-lite)          ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Difference vs. G3-lite (2-lite):
#   G3-lite:  --feature_adapter_unfreeze_layers 0  (backbone frozen)
#   G3-2full: --feature_adapter_unfreeze_layers N  (top-N backbone layers + adapter + head trainable)
#
# Recommended starting point: UNFREEZE_LAYERS=4 (top-4 transformer blocks).
# Increase for more capacity; decrease toward 0 for stability.
#
# Usage:
#   bash scripts/run_g3_2full_8gpu.sh
#
# Override any variable via env, e.g.:
#   UNFREEZE_LAYERS=8 CRITIC_LR=5e-6 bash scripts/run_g3_2full_8gpu.sh

# ====================================================================
# 1. GPU ALLOCATION
# ====================================================================
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
CRITIC_GPUS="${CRITIC_GPUS:-4}"
REF_GPUS="${REF_GPUS:-4}"          # colocated with actor
REWARD_GPUS="${REWARD_GPUS:-4}"    # colocated with critic

# ====================================================================
# 2. 2-FULL SPECIFIC SETTINGS
# ====================================================================
# Number of top backbone layers to unfreeze (0 = 2-lite / frozen backbone)
UNFREEZE_LAYERS="${UNFREEZE_LAYERS:-4}"

# Feature adapter hyperparams
FEATURE_ADAPTER_RANK="${FEATURE_ADAPTER_RANK:-64}"
FEATURE_ADAPTER_DROPOUT="${FEATURE_ADAPTER_DROPOUT:-0.0}"

# Critic LR split:
#   CRITIC_LR       -> backbone unfrozen layers (keep small to avoid catastrophic forgetting)
#   CRITIC_LR_HEAD  -> adapter + classifier head (can be higher)
CRITIC_LR="${CRITIC_LR:-1e-6}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-5e-5}"

# ====================================================================
# 3. REWARD FUNCTION (CF-L1OO, same as G2/G3-lite)
# ====================================================================
DISTRIBUTION_REWARD_TYPE="cf_l1oo"
CF_TARGET_MODE="single"
CF_TEACHER_LAMBDA=0.0
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-2}"
CF_NUM_FREQS="${CF_NUM_FREQS:-128}"
CF_SIGMA="${CF_SIGMA:-1.0}"
CF_SEED="${CF_SEED:-43}"
CF_ALPHA="${CF_ALPHA:-0.5}"
CF_BETA="${CF_BETA:-0.5}"
CF_REWARD_SCALE="${CF_REWARD_SCALE:-1.0}"
FEATURE_MAP_TYPE="${FEATURE_MAP_TYPE:-identity}"
RFF_NUM_FEATURES="${RFF_NUM_FEATURES:-128}"
RFF_SIGMA="${RFF_SIGMA:-1.0}"
RFF_SEED="${RFF_SEED:-43}"
CF_TARGET_NUM_REFS="${CF_TARGET_NUM_REFS:-1}"
CF_TARGET_STD="${CF_TARGET_STD:-0.05}"
CF_TARGET_SEED="${CF_TARGET_SEED:-43}"

# Auxiliary direct-discrepancy loss on critic features (0 = disabled)
CRITIC_DIRECT_DISCREPANCY_COEF="${CRITIC_DIRECT_DISCREPANCY_COEF:-0.0}"
CRITIC_DIRECT_DISCREPANCY_TARGET="${CRITIC_DIRECT_DISCREPANCY_TARGET:-ema_gt}"

# ====================================================================
# 4. MODEL & DATA PATHS
# ====================================================================
REPO_ROOT="${REPO_ROOT:-/root/code/data/Distributional-Match-Tuning}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/code/Qwen3.5-2B}"
TRAIN_DATA="${TRAIN_DATA:-/mnt/data/code/data/aops/aops_qa_hf_dict}"
EVAL_DATA="${EVAL_DATA:-/mnt/data/code/data/aops/test_qa.jsonl}"
INPUT_KEY="question"
LABEL_KEY="answer"
OUTPUT_KEY="answer"

# ====================================================================
# 5. TRAINING BUDGET & BATCH SIZES
# ====================================================================
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"          # 4 x 64 = 256
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-4}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-4}"
MICRO_REWARD_BATCH_SIZE="${MICRO_REWARD_BATCH_SIZE:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-46000}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-256}"
CONTEXT_MAX_LEN="${CONTEXT_MAX_LEN:-8}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-8}"
STRIDE="${STRIDE:-8}"

# ====================================================================
# 6. LOSS COEFFICIENTS & OPTIMIZER
# ====================================================================
CE_LOSS_COEF="${CE_LOSS_COEF:-0.03}"
DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-0.5}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"
EMA_BETA="${EMA_BETA:-0.9}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
CRITIC_CLASSIFIER_LOSS_COEF="${CRITIC_CLASSIFIER_LOSS_COEF:-0.0}"
EMBED_METHOD="${EMBED_METHOD:-last_token}"
CRITIC_SEQUENCE_LEVEL="${CRITIC_SEQUENCE_LEVEL:-last_token}"
GLOBAL_SEED="${GLOBAL_SEED:-43}"

# ====================================================================
# 7. OUTPUT DIRECTORY & LOGGING
# ====================================================================
RUN_TAG="g3_2full_L${UNFREEZE_LAYERS}_8gpu_$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
SAVE_PATH="${RUN_ROOT}/model"
TB_ROOT="${RUN_ROOT}/tensorboard"

EVAL_STEPS="${EVAL_STEPS:--1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-50}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-8}"

# ====================================================================
# ENVIRONMENT
# ====================================================================
export CUDA_VISIBLE_DEVICES
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
export PYTHONUNBUFFERED=1

# ====================================================================
# PRE-FLIGHT
# ====================================================================
mkdir -p "${RUN_ROOT}" "${SAVE_PATH}" "${TB_ROOT}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  G3 Feature Adapter 2-full                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""
echo "  [Model & Data]"
echo "    Model:                   ${MODEL_PATH}"
echo "    Train data:              ${TRAIN_DATA}"
echo "    Eval data:               ${EVAL_DATA}"
echo ""
echo "  [2-full Adapter Config]"
echo "    feature_adapter_enable:  true"
echo "    feature_adapter_type:    residual_bottleneck"
echo "    feature_adapter_rank:    ${FEATURE_ADAPTER_RANK}"
echo "    feature_adapter_dropout: ${FEATURE_ADAPTER_DROPOUT}"
echo "    unfreeze_layers:         ${UNFREEZE_LAYERS}  (top backbone layers trainable)"
echo ""
echo "  [LR]"
echo "    actor_lr:                ${ACTOR_LR}"
echo "    critic_lr (backbone):    ${CRITIC_LR}"
echo "    critic_lr_head:          ${CRITIC_LR_HEAD}  (adapter + classifier head)"
echo ""
echo "  [Reward]"
echo "    reward_type:             ${DISTRIBUTION_REWARD_TYPE}"
echo "    target_mode:             ${CF_TARGET_MODE}"
echo "    cf_freqs:                ${CF_NUM_FREQS}"
echo "    reward_scale:            ${CF_REWARD_SCALE}"
echo "    direct_discrepancy_coef: ${CRITIC_DIRECT_DISCREPANCY_COEF}"
echo ""
echo "  [Training Budget]"
echo "    max_samples:             ${MAX_SAMPLES}"
echo "    rollout_batch:           ${ROLLOUT_BATCH_SIZE}"
echo "    train_batch:             ${TRAIN_BATCH_SIZE}"
echo "    n_samples/prompt:        ${N_SAMPLES_PER_PROMPT}"
echo ""
echo "  [GPU Allocation]"
echo "    devices:                 ${CUDA_VISIBLE_DEVICES}"
echo "    actor:                   ${ACTOR_GPUS}  (+ ref colocated)"
echo "    critic:                  ${CRITIC_GPUS}  (+ reward colocated)"
echo ""
echo "  [Output]"
echo "    run dir:                 ${RUN_ROOT}"
echo "    tensorboard:             ${TB_ROOT}"
echo "────────────────────────────────────────────────────────────────"

ray stop --force 2>/dev/null || true
sleep 2
cd "${REPO_ROOT}"

python -m openrlhf.cli.train_ebft_ray \
  --bf16 --adam_offload --pretrain_mode --no_chat_template \
  --disable_ds_ckpt --colocate_actor_ref --colocate_critic_reward \
  --use_kl_loss --use_whitening --enable_ema \
  --feature_adapter_enable \
  --feature_adapter_type residual_bottleneck \
  --feature_adapter_rank "${FEATURE_ADAPTER_RANK}" \
  --feature_adapter_dropout "${FEATURE_ADAPTER_DROPOUT}" \
  --feature_adapter_unfreeze_layers "${UNFREEZE_LAYERS}" \
  \
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
  --cf_teacher_lambda "${CF_TEACHER_LAMBDA}" \
  --cf_teacher_n_samples "${CF_TEACHER_N_SAMPLES}" \
  \
  --embed_method "${EMBED_METHOD}" \
  --critic_sequence_level "${CRITIC_SEQUENCE_LEVEL}" \
  --critic_learning_rate "${CRITIC_LR}" \
  --critic_lr_head "${CRITIC_LR_HEAD}" \
  --critic_classifier_loss_coef "${CRITIC_CLASSIFIER_LOSS_COEF}" \
  --critic_direct_discrepancy_coef "${CRITIC_DIRECT_DISCREPANCY_COEF}" \
  --critic_direct_discrepancy_target "${CRITIC_DIRECT_DISCREPANCY_TARGET}" \
  --ema_beta "${EMA_BETA}" \
  --ce_loss_coef "${CE_LOSS_COEF}" \
  --diversity_rew_coef "${DIVERSITY_REW_COEF}" \
  --alignment_rew_coef "${ALIGNMENT_REW_COEF}" \
  \
  --pretrain "${MODEL_PATH}" \
  --critic_pretrain "${MODEL_PATH}" \
  --prompt_data "${TRAIN_DATA}" \
  --eval_dataset "${EVAL_DATA}" \
  --input_key "${INPUT_KEY}" --label_key "${LABEL_KEY}" --output_key "${OUTPUT_KEY}" \
  --prompt_split train --eval_split test \
  \
  --prompt_max_len "${PROMPT_MAX_LEN}" \
  --context_max_len "${CONTEXT_MAX_LEN}" \
  --generate_max_len "${GENERATE_MAX_LEN}" \
  --stride "${STRIDE}" \
  --n_samples_per_prompt "${N_SAMPLES_PER_PROMPT}" \
  --rollout_batch_size "${ROLLOUT_BATCH_SIZE}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --micro_train_batch_size "${MICRO_TRAIN_BATCH_SIZE}" \
  --micro_rollout_batch_size "${MICRO_ROLLOUT_BATCH_SIZE}" \
  --micro_reward_batch_size "${MICRO_REWARD_BATCH_SIZE}" \
  --max_samples "${MAX_SAMPLES}" \
  --num_episodes "${NUM_EPISODES}" \
  --max_epochs "${MAX_EPOCHS}" \
  \
  --actor_num_nodes 1 --actor_num_gpus_per_node "${ACTOR_GPUS}" \
  --critic_num_nodes 1 --critic_num_gpus_per_node "${CRITIC_GPUS}" \
  --ref_num_nodes 1 --ref_num_gpus_per_node "${REF_GPUS}" \
  --reward_num_nodes 1 --reward_num_gpus_per_node "${REWARD_GPUS}" \
  \
  --advantage_estimator rloo --init_kl_coef 0.0 --kl_estimator k2 \
  --temperature 0.6 --top_p 1.0 \
  --actor_learning_rate "${ACTOR_LR}" \
  --zero_stage 2 --lr_warmup_ratio 0.03 --lr_scheduler constant_with_warmup \
  --seed "${GLOBAL_SEED}" \
  \
  --eval_steps "${EVAL_STEPS}" \
  --eval_max_samples "${EVAL_MAX_SAMPLES}" \
  --eval_generate_max_len "${EVAL_GENERATE_MAX_LEN}" \
  --save_steps -1 --logging_steps 1 \
  --use_tensorboard "${TB_ROOT}" \
  --save_path "${SAVE_PATH}" --ckpt_path "${SAVE_PATH}/ckpt" \
  --wandb_run_name "${RUN_TAG}" \
  2>&1 | tee "${RUN_ROOT}/train.log"

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')  FINISHED"
echo "  Logs:        ${RUN_ROOT}/train.log"
echo "  TensorBoard: ${TB_ROOT}"
echo "  Checkpoints: ${SAVE_PATH}"
echo "────────────────────────────────────────────────────────────────" 