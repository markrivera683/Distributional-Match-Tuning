#!/usr/bin/env bash
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════╗
# ║  G3-lite: Feature Adapter + EMA — Remote Teacher Target        ║
# ║  8×A100 · cf_l1oo · backbone FROZEN · adapter+head trainable   ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Contrast pair:  scripts/run_g2_8gpu_remote_teacher.sh
#
# Differences vs G2:
#   G2:       critic_lr=0, critic_lr_head=0  (everything frozen)
#   G3-lite:  critic_lr=0 (backbone frozen), critic_lr_head>0
#             feature_adapter_enable=true, feature_adapter_unfreeze_layers=0
#             adapter + classifier head are trainable via critic_lr_head
#             EMA provides stable target geometry
#
# Usage:
#   bash scripts/run_g3_lite_remote_teacher.sh
#
# Override any variable via env, e.g.:
#   CRITIC_LR_HEAD=1e-4 CF_TEACHER_LAMBDA=0.8 bash scripts/run_g3_lite_remote_teacher.sh

# ====================================================================
# 1. GPU ALLOCATION
# ====================================================================
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
CRITIC_GPUS="${CRITIC_GPUS:-4}"
REF_GPUS="${REF_GPUS:-4}"          # colocated with actor
REWARD_GPUS="${REWARD_GPUS:-4}"    # colocated with critic

# ====================================================================
# 2. TEACHER MODE: online vs offline  (identical to G2)
# ====================================================================
TEACHER_MODE="${TEACHER_MODE:-online}"   # online | offline
TEACHER_DATASET_PATH="${TEACHER_DATASET_PATH:-/mnt/data/data/aops/teacher_dataset_n_samples_4}"

# ====================================================================
# 2b. REMOTE TEACHER ENDPOINT
# ====================================================================
TEACHER_NUM_WORKERS="${TEACHER_NUM_WORKERS:-4}"
TEACHER_API_BASE_0="${TEACHER_API_BASE_0:-http://172.17.0.26:8000/v1}"
TEACHER_API_BASE_1="${TEACHER_API_BASE_1:-http://172.17.0.27:8000/v1}"
TEACHER_API_BASE_2="${TEACHER_API_BASE_2:-http://172.17.0.28:8000/v1}"
TEACHER_API_BASE_3="${TEACHER_API_BASE_3:-http://172.17.0.29:8000/v1}"

TEACHER_MODEL="${TEACHER_MODEL:-qwen-122b}"
TEACHER_API_KEY="${TEACHER_API_KEY:-teacher-local}"
TEACHER_API_STYLE="${TEACHER_API_STYLE:-completions}"

_build_api_base() {
  local n=$1
  local bases=()
  for i in $(seq 0 $((n - 1))); do
    local var="TEACHER_API_BASE_${i}"
    bases+=("${!var}")
  done
  local IFS=,
  echo "${bases[*]}"
}
TEACHER_API_BASE="$(_build_api_base "${TEACHER_NUM_WORKERS}")"

TEACHER_REMOTE_BATCH_SIZE="${TEACHER_REMOTE_BATCH_SIZE:-$(( TEACHER_NUM_WORKERS * 24 ))}"

# ====================================================================
# 3. TEACHER TARGET DISTRIBUTION  (identical to G2)
# ====================================================================
CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.6}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-16}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-512}"
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-180}"
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-3}"
TEACHER_CACHE_ENABLE="${TEACHER_CACHE_ENABLE:-true}"

# ====================================================================
# 2c. TEACHER CACHE WARMUP
# ====================================================================
RUN_WARMUP="${RUN_WARMUP:-true}"
WARMUP_BATCH_SIZE="${WARMUP_BATCH_SIZE:-64}"

# ====================================================================
# 2d. TEACHER SYSTEM PROMPT
# ====================================================================
SYSTEM_PROMPT_TEXT="${SYSTEM_PROMPT_TEXT:-You are a precise assistant. produce a correct and well-reasoned answer. Step by step when necessary. Keep reasoning sufficient. Final answer is clearly stated.}"
SYSTEM_PROMPT_ID="${SYSTEM_PROMPT_ID:-v1-balanced}"
SYSTEM_PROMPT_VERSION="${SYSTEM_PROMPT_VERSION:-1.0}"

# ====================================================================
# 4. REWARD FUNCTION  (identical to G2)
# ====================================================================
DISTRIBUTION_REWARD_TYPE="cf_l1oo"
CF_TARGET_MODE="teacher"
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

# ====================================================================
# 5. G3-lite SPECIFIC: FEATURE ADAPTER  ← KEY DIFFERENCE vs G2
# ====================================================================
# backbone is fully frozen (feature_adapter_unfreeze_layers=0)
# only the residual bottleneck adapter + classifier head are trainable
FEATURE_ADAPTER_RANK="${FEATURE_ADAPTER_RANK:-64}"
FEATURE_ADAPTER_DROPOUT="${FEATURE_ADAPTER_DROPOUT:-0.0}"
# UNFREEZE_LAYERS=0 → 2-lite (backbone frozen, adapter+head only)
# UNFREEZE_LAYERS>0 → 2-full (use run_g3_2full_remote_teacher.sh instead)
UNFREEZE_LAYERS="${UNFREEZE_LAYERS:-0}"

# Critic LR:
#   critic_learning_rate = 0      → backbone frozen
#   critic_lr_head       = 5e-5   → adapter + classifier head trainable
CRITIC_LR="${CRITIC_LR:-0}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-5e-5}"
CRITIC_CLASSIFIER_LOSS_COEF="${CRITIC_CLASSIFIER_LOSS_COEF:-0.0}"

# ── EMA target geometry: direct discrepancy loss ─────────────────────
# Core of G3: train the adapter to align online gen-embedding toward
# the EMA critic's stable GT embedding (ema_gt branch).
#   coef=0.0  → disabled (degenerates to G2 with adapter but no geometry signal)
#   coef>0.0  → EMA GT geometry drives adapter via CF discrepancy loss
#
# EMA branch (critic_direct_discrepancy_target=ema_gt) uses the slow-moving
# EMA critic forward pass as the target, so the target geometry drifts
# much more slowly than the online branch — this is the key stabilizer.
#
# Recommended starting range: 0.05 ~ 0.2
# Increase if adapter barely moves; decrease if training diverges.
CRITIC_DIRECT_DISCREPANCY_COEF="${CRITIC_DIRECT_DISCREPANCY_COEF:-0.1}"
CRITIC_DIRECT_DISCREPANCY_TARGET="${CRITIC_DIRECT_DISCREPANCY_TARGET:-ema_gt}"

# ====================================================================
# 6. MODEL & DATA PATHS
# ====================================================================
REPO_ROOT="${REPO_ROOT:-/root/code/data/Distributional-Match-Tuning}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3.5-2B}"
TRAIN_DATA="${TRAIN_DATA:-/mnt/data/data/aops/aops_qa_hf_dict}"
EVAL_DATA="${EVAL_DATA:-/mnt/data/data/aops/test_qa.jsonl}"
INPUT_KEY="question"
LABEL_KEY="answer"
OUTPUT_KEY="answer"

# ====================================================================
# 7. TRAINING BUDGET & BATCH SIZES  (identical to G2)
# ====================================================================
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
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
# 8. LOSS COEFFICIENTS & OPTIMIZER  (identical to G2 except critic LRs)
# ====================================================================
CE_LOSS_COEF="${CE_LOSS_COEF:-0.03}"
DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-0.5}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"
EMA_BETA="${EMA_BETA:-0.9}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
EMBED_METHOD="${EMBED_METHOD:-last_token}"
CRITIC_SEQUENCE_LEVEL="${CRITIC_SEQUENCE_LEVEL:-last_token}"
GLOBAL_SEED="${GLOBAL_SEED:-43}"

# ====================================================================
# 9. OUTPUT DIRECTORY & LOGGING
# ====================================================================
RUN_TAG="g3_lite_remote_teacher_$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
SAVE_PATH="${RUN_ROOT}/model"
TB_ROOT="${RUN_ROOT}/tensorboard"
CACHE_DIR="${CACHE_DIR:-/root/outputs/teacher_cache_shared}"

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

# Resolve TEACHER_BACKEND and associated flags from TEACHER_MODE
if [[ "${TEACHER_MODE}" == "offline" ]]; then
  TEACHER_BACKEND="dataset"
  if [[ -z "${TEACHER_DATASET_PATH}" || ! -e "${TEACHER_DATASET_PATH}" ]]; then
    echo "[ERROR] TEACHER_MODE=offline but TEACHER_DATASET_PATH not found: '${TEACHER_DATASET_PATH}'"
    exit 1
  fi
  TEACHER_FLAGS=(--teacher_backend dataset --teacher_dataset_path "${TEACHER_DATASET_PATH}")
  CACHE_FLAGS=()
  RUN_WARMUP="false"
  echo "[Teacher] Mode: OFFLINE — reading from dataset: ${TEACHER_DATASET_PATH}"
else
  TEACHER_BACKEND="remote"
  mkdir -p "${CACHE_DIR}"
  TEACHER_FLAGS=(
    --teacher_backend remote
    --teacher_api_base "${TEACHER_API_BASE}"
    --teacher_api_key "${TEACHER_API_KEY}"
    --teacher_api_style "${TEACHER_API_STYLE}"
    --teacher_model_name "${TEACHER_MODEL}"
    --teacher_timeout "${TEACHER_TIMEOUT}"
    --teacher_max_retries "${TEACHER_MAX_RETRIES}"
    --teacher_remote_batch_size "${TEACHER_REMOTE_BATCH_SIZE}"
  )
  CACHE_FLAGS=()
  if [[ "${TEACHER_CACHE_ENABLE}" == "true" ]]; then
    CACHE_FLAGS=(--teacher_cache_enable --teacher_cache_dir "${CACHE_DIR}")
  fi
  echo "[Teacher] Mode: ONLINE — API: ${TEACHER_API_BASE} (${TEACHER_NUM_WORKERS} worker(s))"
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  G3-lite: Feature Adapter + EMA — Remote Teacher Target     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""
echo "  [Model & Data]"
echo "    Model:               ${MODEL_PATH}"
echo "    Train data:          ${TRAIN_DATA}"
echo "    Eval data:           ${EVAL_DATA}"
echo ""
echo "  [Feature Adapter + EMA Discrepancy  <- vs G2: frozen critic]"
echo "    feature_adapter:          ENABLED (residual_bottleneck)"
echo "    adapter_rank:             ${FEATURE_ADAPTER_RANK}"
echo "    adapter_dropout:          ${FEATURE_ADAPTER_DROPOUT}"
echo "    unfreeze_layers:          ${UNFREEZE_LAYERS}  (0=backbone frozen / 2-lite)"
echo "    critic_lr_backbone:       ${CRITIC_LR}  (frozen)"
echo "    critic_lr_head:           ${CRITIC_LR_HEAD}  (adapter + classifier head)"
echo "    ema_beta:                 ${EMA_BETA}  (EMA slow-moving target)"
echo "    direct_discrepancy_coef:  ${CRITIC_DIRECT_DISCREPANCY_COEF}  (0=disabled, >0=EMA geometry loss)"
echo "    direct_discrepancy_target:${CRITIC_DIRECT_DISCREPANCY_TARGET}  (ema_gt=use EMA forward as target)"
echo ""
echo "  [Teacher]"
echo "    Mode:                ${TEACHER_MODE}"
if [[ "${TEACHER_MODE}" == "offline" ]]; then
echo "    Dataset path:        ${TEACHER_DATASET_PATH}"
else
echo "    Num workers:         ${TEACHER_NUM_WORKERS}"
echo "    API base(s):         ${TEACHER_API_BASE}"
echo "    Model:               ${TEACHER_MODEL}"
echo "    Concurrency:         ${TEACHER_REMOTE_BATCH_SIZE}"
echo "    Run warmup:          ${RUN_WARMUP}"
echo "    Cache dir:           ${CACHE_DIR}"
fi
echo ""
echo "  [Target Distribution]"
echo "    Reward type:         ${DISTRIBUTION_REWARD_TYPE}"
echo "    Target mode:         ${CF_TARGET_MODE}"
echo "    Lambda (teacher):    ${CF_TEACHER_LAMBDA}"
echo "    M (teacher samples): ${CF_TEACHER_N_SAMPLES}"
echo ""
echo "  [Training Budget]"
echo "    Max samples:         ${MAX_SAMPLES}"
echo "    Rollout batch:       ${ROLLOUT_BATCH_SIZE}"
echo "    Train batch:         ${TRAIN_BATCH_SIZE}"
echo ""
echo "  [GPU Allocation]"
echo "    Devices:             ${CUDA_VISIBLE_DEVICES}"
echo "    Actor:               ${ACTOR_GPUS}  (+ ref colocated)"
echo "    Critic:              ${CRITIC_GPUS}  (+ reward colocated)"
echo ""
echo "  [Output]"
echo "    Run dir:             ${RUN_ROOT}"
echo "    TensorBoard:         ${TB_ROOT}"
echo "────────────────────────────────────────────────────────────────"

# Optional cache warmup
if [[ "${RUN_WARMUP}" == "true" && "${TEACHER_MODE}" == "online" ]]; then
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
    --max_retries "${TEACHER_MAX_RETRIES}" \
    --system_prompt_text "${SYSTEM_PROMPT_TEXT}" \
    --system_prompt_id "${SYSTEM_PROMPT_ID}"
  echo "  [Warmup] Done."
  echo ""
elif [[ "${TEACHER_MODE}" == "offline" ]]; then
  echo "  [Warmup] Skipped (TEACHER_MODE=offline)"
fi

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
  "${TEACHER_FLAGS[@]}" \
  --teacher_temperature "${TEACHER_TEMPERATURE}" \
  --teacher_top_p "${TEACHER_TOP_P}" \
  --teacher_max_new_tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher_system_prompt_text "${SYSTEM_PROMPT_TEXT}" \
  --teacher_system_prompt_id "${SYSTEM_PROMPT_ID}" \
  "${CACHE_FLAGS[@]}" \
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
echo "  Teacher cache: ${CACHE_DIR}"
echo "────────────────────────────────────────────────────────────────"
  