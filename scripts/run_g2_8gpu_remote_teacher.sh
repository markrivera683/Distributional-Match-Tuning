#!/usr/bin/env bash
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════╗
# ║  G2: Distributional Match Tuning with Remote Teacher Target     ║
# ║  8×A100 · cf_l1oo · remote teacher empirical target measure     ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Target measure:  nu_c = (1-λ)·δ(GT) + λ·(1/M)·Σ δ(teacher_i)
# Reward:          cf_l1oo leave-one-out marginal contribution
# Teacher:         external OpenAI-compatible API (e.g. vLLM serving qwen-122b)
#
# Usage:
#   bash scripts/run_g2_8gpu_remote_teacher.sh
#
# Override any variable via env, e.g.:
#   CF_TEACHER_LAMBDA=0.8 MAX_SAMPLES=10000 bash scripts/run_g2_8gpu_remote_teacher.sh

# ====================================================================
# 1. GPU ALLOCATION
# ====================================================================
# actor + ref colocated on first group; critic + reward on second group
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
CRITIC_GPUS="${CRITIC_GPUS:-4}"
REF_GPUS="${REF_GPUS:-4}"          # colocated with actor
REWARD_GPUS="${REWARD_GPUS:-4}"    # colocated with critic

# ====================================================================
# 2. REMOTE TEACHER ENDPOINT
# ====================================================================
TEACHER_API_BASE="${TEACHER_API_BASE:-http://172.17.0.26:8000/v1}"   # vLLM / OpenAI-compatible URL
TEACHER_MODEL="${TEACHER_MODEL:-qwen-122b}"                          # model name served at the endpoint
TEACHER_API_KEY="${TEACHER_API_KEY:-teacher-local}"                   # bearer token (use "EMPTY" if none)
TEACHER_API_STYLE="${TEACHER_API_STYLE:-chat_completions}"           # chat_completions | completions
TEACHER_BACKEND="${TEACHER_BACKEND:-remote}"                          # remote = HTTP API, local = Ray actor

# ====================================================================
# 3. TEACHER TARGET DISTRIBUTION
# ====================================================================
# λ: mixing weight. 0→GT only, 1→teacher only, 0.5→equal mix
CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.7}"
# M: number of independent teacher completions per question (support points in teacher measure)
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-8}"
# Generation params for teacher completions
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-512}"   # full-answer length (NOT 8-token block)
# Robustness
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-180}"                 # seconds per HTTP request
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-3}"
TEACHER_REMOTE_BATCH_SIZE="${TEACHER_REMOTE_BATCH_SIZE:-4}"  # concurrent HTTP requests
# Cache: SQLite disk cache for teacher completions (avoids re-fetching same questions)
TEACHER_CACHE_ENABLE="${TEACHER_CACHE_ENABLE:-true}"      # set "false" to force fresh API calls every step

# ====================================================================
# 2b. TEACHER CACHE WARMUP
# ====================================================================
# Set RUN_WARMUP=true to pre-fill the cache before training starts.
# This calls warmup_teacher_cache.py which loads the dataset, extracts
# all unique questions, and populates the SQLite cache via the teacher
# API.  Subsequent training steps will be 100% cache hits.
RUN_WARMUP="${RUN_WARMUP:-true}"
WARMUP_BATCH_SIZE="${WARMUP_BATCH_SIZE:-64}"              # concurrent HTTP requests during warmup

# ====================================================================
# 4. REWARD FUNCTION (CF-L1OO Distributional Matching)
# ====================================================================
DISTRIBUTION_REWARD_TYPE="cf_l1oo"   # cf_l1oo = characteristic function LOO distributional reward
CF_TARGET_MODE="teacher"             # teacher = use remote teacher target measure
# CF frequency-space parameters
CF_NUM_FREQS="${CF_NUM_FREQS:-128}"  # number of random Fourier frequencies
CF_SIGMA="${CF_SIGMA:-1.0}"          # bandwidth of the RFF kernel
CF_SEED="${CF_SEED:-43}"
CF_ALPHA="${CF_ALPHA:-0.5}"          # amplitude weight in CF loss
CF_BETA="${CF_BETA:-0.5}"            # phase weight in CF loss
CF_REWARD_SCALE="${CF_REWARD_SCALE:-1.0}"
# Feature map (identity = raw hidden states; rff = random Fourier features)
FEATURE_MAP_TYPE="${FEATURE_MAP_TYPE:-identity}"
RFF_NUM_FEATURES="${RFF_NUM_FEATURES:-128}"
RFF_SIGMA="${RFF_SIGMA:-1.0}"
RFF_SEED="${RFF_SEED:-43}"
# Vicinal target params (only used when cf_target_mode=vicinal, listed for completeness)
CF_TARGET_NUM_REFS="${CF_TARGET_NUM_REFS:-1}"
CF_TARGET_STD="${CF_TARGET_STD:-0.05}"
CF_TARGET_SEED="${CF_TARGET_SEED:-43}"

# ====================================================================
# 5. MODEL & DATA PATHS
# ====================================================================
REPO_ROOT="${REPO_ROOT:-/root/code/data/Distributional-Match-Tuning}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3.5-2B}"
TRAIN_DATA="${TRAIN_DATA:-/mnt/data/data/aops/aops_qa_hf_dict}"
EVAL_DATA="${EVAL_DATA:-/mnt/data/data/aops/test_qa.jsonl}"
INPUT_KEY="question"
LABEL_KEY="answer"
OUTPUT_KEY="answer"

# ====================================================================
# 6. TRAINING BUDGET & BATCH SIZES
# ====================================================================
# Constraints (must satisfy both):
#   DeepSpeed:  train_batch_size % (micro_train_batch_size × actor_gpus) == 0
#   ED:         train_batch_size == n_samples_per_prompt × rollout_batch_size
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"          # 4 × 64 = 256
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-4}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-4}"
MICRO_REWARD_BATCH_SIZE="${MICRO_REWARD_BATCH_SIZE:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-46000}"   # total training prompts (≈500 global steps × 64 rollout + margin)
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-256}"
CONTEXT_MAX_LEN="${CONTEXT_MAX_LEN:-8}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-8}"
STRIDE="${STRIDE:-8}"

# ====================================================================
# 7. LOSS COEFFICIENTS & OPTIMIZER
# ====================================================================
CE_LOSS_COEF="${CE_LOSS_COEF:-0.03}"          # cross-entropy auxiliary loss weight
DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-0.5}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"
EMA_BETA="${EMA_BETA:-0.9}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
CRITIC_LR="${CRITIC_LR:-0}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-0}"
CRITIC_CLASSIFIER_LOSS_COEF="${CRITIC_CLASSIFIER_LOSS_COEF:-0.0}"
CRITIC_DIRECT_DISCREPANCY_COEF="${CRITIC_DIRECT_DISCREPANCY_COEF:-0.0}"
EMBED_METHOD="${EMBED_METHOD:-last_token}"
CRITIC_SEQUENCE_LEVEL="${CRITIC_SEQUENCE_LEVEL:-last_token}"
GLOBAL_SEED="${GLOBAL_SEED:-43}"

# ====================================================================
# 8. OUTPUT DIRECTORY & LOGGING
# ====================================================================
RUN_TAG="g2_8gpu_remote_teacher_$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
SAVE_PATH="${RUN_ROOT}/model"
TB_ROOT="${RUN_ROOT}/tensorboard"
CACHE_DIR="${CACHE_DIR:-/root/outputs/teacher_cache_shared}"

# Eval (disabled by default — GatedDeltaNet OOMs on long generation)
EVAL_STEPS="${EVAL_STEPS:--1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-50}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-8}"

# ====================================================================
# ENVIRONMENT (generally no need to change)
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
mkdir -p "${RUN_ROOT}" "${SAVE_PATH}" "${TB_ROOT}" "${CACHE_DIR}"

# Resolve cache flag
CACHE_FLAGS=()
if [[ "${TEACHER_CACHE_ENABLE}" == "true" ]]; then
  CACHE_FLAGS=(--teacher_cache_enable --teacher_cache_dir "${CACHE_DIR}")
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  G2: Remote Teacher Distributional Match Tuning             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""
echo "  [Model & Data]"
echo "    Model:             ${MODEL_PATH}"
echo "    Train data:        ${TRAIN_DATA}"
echo "    Eval data:         ${EVAL_DATA}"
echo ""
echo "  [Remote Teacher]"
echo "    API base:          ${TEACHER_API_BASE}"
echo "    Model:             ${TEACHER_MODEL}"
echo "    API key:           ${TEACHER_API_KEY}"
echo "    API style:         ${TEACHER_API_STYLE}"
echo "    Temperature:       ${TEACHER_TEMPERATURE}"
echo "    Top-p:             ${TEACHER_TOP_P}"
echo "    Max new tokens:    ${TEACHER_MAX_NEW_TOKENS}"
echo "    Cache enabled:     ${TEACHER_CACHE_ENABLE}"
echo "    Cache dir:         ${CACHE_DIR}"
echo "    Run warmup:        ${RUN_WARMUP}"
echo ""
echo "  [Target Distribution]"
echo "    Reward type:       ${DISTRIBUTION_REWARD_TYPE}"
echo "    Target mode:       ${CF_TARGET_MODE}"
echo "    Lambda (teacher):  ${CF_TEACHER_LAMBDA}"
echo "    M (teacher samples): ${CF_TEACHER_N_SAMPLES}"
echo "    CF freqs:          ${CF_NUM_FREQS}"
echo "    CF sigma:          ${CF_SIGMA}"
echo "    Reward scale:      ${CF_REWARD_SCALE}"
echo ""
echo "  [Training Budget]"
echo "    Max samples:       ${MAX_SAMPLES}"
echo "    Episodes:          ${NUM_EPISODES}"
echo "    Rollout batch:     ${ROLLOUT_BATCH_SIZE}"
echo "    Train batch:       ${TRAIN_BATCH_SIZE}"
echo "    N samples/prompt:  ${N_SAMPLES_PER_PROMPT}"
echo ""
echo "  [GPU Allocation]"
echo "    Devices:           ${CUDA_VISIBLE_DEVICES}"
echo "    Actor:             ${ACTOR_GPUS} (+ ref colocated)"
echo "    Critic:            ${CRITIC_GPUS} (+ reward colocated)"
echo ""
echo "  [Output]"
echo "    Run dir:           ${RUN_ROOT}"
echo "    TensorBoard:       ${TB_ROOT}"
echo "────────────────────────────────────────────────────────────────"

# ── Optional cache warmup (before Ray / GPU init) ─────────────────
if [[ "${RUN_WARMUP}" == "true" ]]; then
  echo ""
  echo "  [Warmup] Pre-filling teacher cache ..."
  cd "${REPO_ROOT}"
  python scripts/warmup_teacher_cache.py \
    --prompt_data "${TRAIN_DATA}" \
    --input_key "${INPUT_KEY}" \
    --split train \
    --cache_dir "${CACHE_DIR}" \
    --teacher_api_base "${TEACHER_API_BASE}" \
    --teacher_model_name "${TEACHER_MODEL}" \
    --teacher_api_key "${TEACHER_API_KEY}" \
    --teacher_api_style "${TEACHER_API_STYLE}" \
    --n_samples "${CF_TEACHER_N_SAMPLES}" \
    --temperature "${TEACHER_TEMPERATURE}" \
    --top_p "${TEACHER_TOP_P}" \
    --max_new_tokens "${TEACHER_MAX_NEW_TOKENS}" \
    --max_samples "${MAX_SAMPLES}" \
    --batch_size "${WARMUP_BATCH_SIZE}" \
    --timeout "${TEACHER_TIMEOUT}" \
    --max_retries "${TEACHER_MAX_RETRIES}"
  echo "  [Warmup] Done."
  echo ""
fi

ray stop --force 2>/dev/null || true
sleep 2
cd "${REPO_ROOT}"

python -m openrlhf.cli.train_ebft_ray \
  --bf16 --adam_offload --pretrain_mode --no_chat_template \
  --disable_ds_ckpt --colocate_actor_ref --colocate_critic_reward \
  --use_kl_loss --use_whitening --enable_ema \
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
  --teacher_backend "${TEACHER_BACKEND}" \
  --teacher_api_base "${TEACHER_API_BASE}" \
  --teacher_api_key "${TEACHER_API_KEY}" \
  --teacher_api_style "${TEACHER_API_STYLE}" \
  --teacher_model_name "${TEACHER_MODEL}" \
  --teacher_timeout "${TEACHER_TIMEOUT}" \
  --teacher_max_retries "${TEACHER_MAX_RETRIES}" \
  --teacher_remote_batch_size "${TEACHER_REMOTE_BATCH_SIZE}" \
  --teacher_temperature "${TEACHER_TEMPERATURE}" \
  --teacher_top_p "${TEACHER_TOP_P}" \
  --teacher_max_new_tokens "${TEACHER_MAX_NEW_TOKENS}" \
  "${CACHE_FLAGS[@]}" \
  \
  --embed_method "${EMBED_METHOD}" \
  --critic_sequence_level "${CRITIC_SEQUENCE_LEVEL}" \
  --critic_learning_rate "${CRITIC_LR}" \
  --critic_lr_head "${CRITIC_LR_HEAD}" \
  --critic_classifier_loss_coef "${CRITIC_CLASSIFIER_LOSS_COEF}" \
  --critic_direct_discrepancy_coef "${CRITIC_DIRECT_DISCREPANCY_COEF}" \
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
echo "  Teacher cache: ${CACHE_DIR}"
echo "────────────────────────────────────────────────────────────────"
