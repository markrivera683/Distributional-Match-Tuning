#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------
# Local single-machine EBFT + AoPS + remote teacher
# ---------------------------------------------------------------------
# Usage:
#   bash scripts/run_train_local_qwen35_2b_aops_remote_teacher.sh
#
# ==================== EDIT THESE FIRST ====================
# [MUST] MODEL / DATA
  MODEL_PATH=/mnt/data/Qwen3.5-2B
  PROMPT_DATA=/mnt/data/data/aops/aops_qa_hf
  EVAL_DATASET=/mnt/data/data/aops/test_qa.jsonl
#
# [MUST] REMOTE TEACHER ENDPOINT
  TEACHER_API_BASE=http://172.17.0.26:8000/v1
  TEACHER_MODEL=qwen-122b
  TEACHER_API_KEY=teacher-local
#
# [CORE] TARGET MEASURE MIXING
  CF_TEACHER_LAMBDA=0.5
  CF_TEACHER_N_SAMPLES=2
#
# [CORE] GPU ASSIGNMENT
  CUDA_VISIBLE_DEVICES=0
  ACTOR_NUM_GPUS_PER_NODE=1
  CRITIC_NUM_GPUS_PER_NODE=1
  REF_NUM_GPUS_PER_NODE=1
  REWARD_NUM_GPUS_PER_NODE=1

# [CORE] OUTPUT DIR
  OUTPUT_ROOT=/root/code/data/outputs/ebft_local_qwen35_2b_aops_remote_teacher
# ==========================================================

REPO_DIR="${REPO_DIR:-/root/code/data/Distributional-Match-Tuning}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3.5-2B}"

# AoPS processed datasets (generated from your previous conversion step)
PROMPT_DATA="${PROMPT_DATA:-/mnt/data/data/aops/aops_qa_hf}"
EVAL_DATASET="${EVAL_DATASET:-/mnt/data/data/aops/test_qa.jsonl}"
INPUT_KEY="${INPUT_KEY:-question}"
LABEL_KEY="${LABEL_KEY:-answer}"
OUTPUT_KEY="${OUTPUT_KEY:-answer}"
PROMPT_SPLIT="${PROMPT_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

# Remote teacher config
TEACHER_API_BASE="${TEACHER_API_BASE:-http://172.17.0.26:8000/v1}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen-122b}"
TEACHER_API_KEY="${TEACHER_API_KEY:-teacher-local}"
TEACHER_API_STYLE="${TEACHER_API_STYLE:-chat_completions}"
CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.5}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-2}"
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-180}"
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-3}"
TEACHER_REMOTE_BATCH_SIZE="${TEACHER_REMOTE_BATCH_SIZE:-4}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"

# GPU assignment (keep near top for quick editing)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
CRITIC_NUM_GPUS_PER_NODE="${CRITIC_NUM_GPUS_PER_NODE:-4}"
REF_NUM_GPUS_PER_NODE="${REF_NUM_GPUS_PER_NODE:-1}"
REWARD_NUM_GPUS_PER_NODE="${REWARD_NUM_GPUS_PER_NODE:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/code/data/outputs/ebft_local_qwen35_2b_aops_remote_teacher}"
RUN_NAME="${RUN_NAME:-aops_remote_teacher_$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
TB_DIR="${RUN_DIR}/tensorboard"
SAVE_DIR="${RUN_DIR}/model"
CACHE_DIR="${RUN_DIR}/teacher_cache"

MAX_SAMPLES="${MAX_SAMPLES:-64}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-16}"
EVAL_DOWN_MAX_SAMPLES="${EVAL_DOWN_MAX_SAMPLES:-16}"
NUM_EPISODES="${NUM_EPISODES:-1}"
SEED="${SEED:-43}"

# Runtime / rollout (all tunable)
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-128}"
CONTEXT_MAX_LEN="${CONTEXT_MAX_LEN:-8}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-8}"
STRIDE="${STRIDE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-4}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-4}"
MICRO_REWARD_BATCH_SIZE="${MICRO_REWARD_BATCH_SIZE:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"

# RL / optimization (all tunable)
ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-rloo}"
INIT_KL_COEF="${INIT_KL_COEF:-0.0}"
KL_ESTIMATOR="${KL_ESTIMATOR:-k2}"
ACTOR_TEMPERATURE="${ACTOR_TEMPERATURE:-0.6}"
ACTOR_TOP_P="${ACTOR_TOP_P:-1.0}"
CE_LOSS_COEF="${CE_LOSS_COEF:-0.03}"
RL_LOSS_COEF="${RL_LOSS_COEF:-1.0}"
DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-0.0}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"
CRITIC_LEARNING_RATE="${CRITIC_LEARNING_RATE:-0}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-0}"
CRITIC_CLASSIFIER_LOSS_COEF="${CRITIC_CLASSIFIER_LOSS_COEF:-0.0}"
ACTOR_LEARNING_RATE="${ACTOR_LEARNING_RATE:-1e-6}"
ZERO_STAGE="${ZERO_STAGE:-2}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.03}"
LR_SCHEDULER="${LR_SCHEDULER:-constant_with_warmup}"
CRITIC_LR_SCHEDULER="${CRITIC_LR_SCHEDULER:-constant_with_warmup}"
EMA_BETA="${EMA_BETA:-0.9}"
HIDDEN_STATE_METHOD="${HIDDEN_STATE_METHOD:-concat}"
EMBED_METHOD="${EMBED_METHOD:-last_token}"
CRITIC_SEQUENCE_LEVEL="${CRITIC_SEQUENCE_LEVEL:-last_token}"
CLASSIFIER_SEQUENCE_SELECTION="${CLASSIFIER_SEQUENCE_SELECTION:-closest}"
EVAL_STEPS="${EVAL_STEPS:--1}"
EVAL_DOWN_STEPS="${EVAL_DOWN_STEPS:--1}"
SAVE_STEPS="${SAVE_STEPS:--1}"
SAVE_LOG_SCALE_COUNT="${SAVE_LOG_SCALE_COUNT:--1}"
SAVE_EVEN_COUNT="${SAVE_EVEN_COUNT:-0}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[ERROR] REPO_DIR does not exist: ${REPO_DIR}"
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}"
  exit 1
fi
if [[ ! -e "${PROMPT_DATA}" ]]; then
  echo "[ERROR] PROMPT_DATA does not exist: ${PROMPT_DATA}"
  exit 1
fi
if [[ ! -e "${EVAL_DATASET}" ]]; then
  echo "[ERROR] EVAL_DATASET does not exist: ${EVAL_DATASET}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENRLHF_RAY_OBJECT_STORE_MEMORY_BYTES="${OPENRLHF_RAY_OBJECT_STORE_MEMORY_BYTES:-4294967296}"
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_DIR}" "${TB_DIR}" "${SAVE_DIR}" "${CACHE_DIR}"
exec > >(tee -a "${RUN_DIR}/train.log") 2>&1

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Starting AoPS remote-teacher run"
echo "MODEL_PATH=${MODEL_PATH}"
echo "PROMPT_DATA=${PROMPT_DATA}"
echo "EVAL_DATASET=${EVAL_DATASET}"
echo "TEACHER_API_BASE=${TEACHER_API_BASE}"
echo "TEACHER_MODEL=${TEACHER_MODEL}"
echo "CF_TEACHER_LAMBDA=${CF_TEACHER_LAMBDA}"
echo "CF_TEACHER_N_SAMPLES=${CF_TEACHER_N_SAMPLES}"
echo "TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE}"
echo "TEACHER_TOP_P=${TEACHER_TOP_P}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE}"
echo "CRITIC_NUM_GPUS_PER_NODE=${CRITIC_NUM_GPUS_PER_NODE}"
echo "REF_NUM_GPUS_PER_NODE=${REF_NUM_GPUS_PER_NODE}"
echo "REWARD_NUM_GPUS_PER_NODE=${REWARD_NUM_GPUS_PER_NODE}"
echo "PROMPT_MAX_LEN=${PROMPT_MAX_LEN}, CONTEXT_MAX_LEN=${CONTEXT_MAX_LEN}, GENERATE_MAX_LEN=${GENERATE_MAX_LEN}"
echo "ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}, TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
echo "MICRO_TRAIN_BATCH_SIZE=${MICRO_TRAIN_BATCH_SIZE}, MICRO_ROLLOUT_BATCH_SIZE=${MICRO_ROLLOUT_BATCH_SIZE}, MICRO_REWARD_BATCH_SIZE=${MICRO_REWARD_BATCH_SIZE}"
echo "NUM_EPISODES=${NUM_EPISODES}, MAX_EPOCHS=${MAX_EPOCHS}, MAX_SAMPLES=${MAX_SAMPLES}"
echo "RUN_DIR=${RUN_DIR}"

cd "${REPO_DIR}"
ray stop --force >/dev/null 2>&1 || true

# Optional quick connectivity check (non-fatal if models endpoint is unavailable).
curl -s -m 10 -H "Authorization: Bearer ${TEACHER_API_KEY}" "${TEACHER_API_BASE}/models" >/dev/null \
  && echo "[Teacher Check] API reachable: ${TEACHER_API_BASE}" \
  || echo "[Teacher Check] WARNING: models endpoint check failed, training may fail if teacher API is unreachable."

python -m openrlhf.cli.train_ebft_ray \
  --bf16 \
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
  --prompt_data "${PROMPT_DATA}" \
  --eval_dataset "${EVAL_DATASET}" \
  --input_key "${INPUT_KEY}" \
  --label_key "${LABEL_KEY}" \
  --output_key "${OUTPUT_KEY}" \
  --prompt_split "${PROMPT_SPLIT}" \
  --eval_split "${EVAL_SPLIT}" \
  --distribution_reward_type cf_l1oo \
  --cf_target_mode teacher \
  --cf_teacher_lambda "${CF_TEACHER_LAMBDA}" \
  --cf_teacher_n_samples "${CF_TEACHER_N_SAMPLES}" \
  --cf_num_freqs 128 \
  --cf_sigma 1.0 \
  --cf_seed 43 \
  --cf_alpha 0.5 \
  --cf_beta 0.5 \
  --cf_reward_scale 1.0 \
  --teacher_backend remote \
  --teacher_api_base "${TEACHER_API_BASE}" \
  --teacher_api_key "${TEACHER_API_KEY}" \
  --teacher_api_style "${TEACHER_API_STYLE}" \
  --teacher_model_name "${TEACHER_MODEL}" \
  --teacher_timeout "${TEACHER_TIMEOUT}" \
  --teacher_max_retries "${TEACHER_MAX_RETRIES}" \
  --teacher_remote_batch_size "${TEACHER_REMOTE_BATCH_SIZE}" \
  --teacher_temperature "${TEACHER_TEMPERATURE}" \
  --teacher_top_p "${TEACHER_TOP_P}" \
  --teacher_cache_enable \
  --teacher_cache_dir "${CACHE_DIR}" \
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
  --eval_max_samples "${EVAL_MAX_SAMPLES}" \
  --eval_down_max_samples "${EVAL_DOWN_MAX_SAMPLES}" \
  --max_epochs "${MAX_EPOCHS}" \
  --num_episodes "${NUM_EPISODES}" \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node "${REF_NUM_GPUS_PER_NODE}" \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node "${CRITIC_NUM_GPUS_PER_NODE}" \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --reward_num_nodes 1 \
  --reward_num_gpus_per_node "${REWARD_NUM_GPUS_PER_NODE}" \
  --advantage_estimator "${ADVANTAGE_ESTIMATOR}" \
  --init_kl_coef "${INIT_KL_COEF}" \
  --kl_estimator "${KL_ESTIMATOR}" \
  --temperature "${ACTOR_TEMPERATURE}" \
  --top_p "${ACTOR_TOP_P}" \
  --ce_loss_coef "${CE_LOSS_COEF}" \
  --rl_loss_coef "${RL_LOSS_COEF}" \
  --diversity_rew_coef "${DIVERSITY_REW_COEF}" \
  --alignment_rew_coef "${ALIGNMENT_REW_COEF}" \
  --critic_learning_rate "${CRITIC_LEARNING_RATE}" \
  --critic_lr_head "${CRITIC_LR_HEAD}" \
  --critic_classifier_loss_coef "${CRITIC_CLASSIFIER_LOSS_COEF}" \
  --actor_learning_rate "${ACTOR_LEARNING_RATE}" \
  --zero_stage "${ZERO_STAGE}" \
  --lr_warmup_ratio "${LR_WARMUP_RATIO}" \
  --lr_scheduler "${LR_SCHEDULER}" \
  --critic_lr_scheduler "${CRITIC_LR_SCHEDULER}" \
  --seed "${SEED}" \
  --ema_beta "${EMA_BETA}" \
  --hidden_state_method "${HIDDEN_STATE_METHOD}" \
  --embed_method "${EMBED_METHOD}" \
  --critic_sequence_level "${CRITIC_SEQUENCE_LEVEL}" \
  --classifier_sequence_selection "${CLASSIFIER_SEQUENCE_SELECTION}" \
  --eval_steps "${EVAL_STEPS}" \
  --eval_down_steps "${EVAL_DOWN_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_log_scale_count "${SAVE_LOG_SCALE_COUNT}" \
  --save_even_count "${SAVE_EVEN_COUNT}" \
  --logging_steps "${LOGGING_STEPS}" \
  --use_tensorboard "${TB_DIR}" \
  --save_path "${SAVE_DIR}" \
  --ckpt_path "${SAVE_DIR}/ckpt" \
  --wandb_run_name "${RUN_NAME}"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Training finished"
echo "Logs: ${RUN_DIR}/train.log"
echo "TensorBoard: ${TB_DIR}"
echo "Checkpoints: ${SAVE_DIR}"
