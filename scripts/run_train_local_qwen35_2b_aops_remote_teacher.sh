#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------
# Local single-machine EBFT + AoPS + remote teacher
# ---------------------------------------------------------------------
# Usage:
#   bash scripts/run_train_local_qwen35_2b_aops_remote_teacher.sh
#
# Required env for your remote teacher endpoint (override as needed):
#   TEACHER_API_BASE=http://172.17.0.26:8000/v1
#   TEACHER_MODEL=qwen-122b
#   TEACHER_API_KEY=teacher-local
#
# Optional:
#   MODEL_PATH=/mnt/data/Qwen3.5-2B
#   PROMPT_DATA=/mnt/data/data/aops/aops_qa_hf
#   EVAL_DATASET=/mnt/data/data/aops/test_qa.jsonl
# ---------------------------------------------------------------------

REPO_DIR="${REPO_DIR:-/root/code/Distributional-Match-Tuning}"
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

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs/ebft_local_qwen35_2b_aops_remote_teacher}"
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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
  --max_samples "${MAX_SAMPLES}" \
  --eval_max_samples "${EVAL_MAX_SAMPLES}" \
  --eval_down_max_samples "${EVAL_DOWN_MAX_SAMPLES}" \
  --max_epochs 1 \
  --num_episodes "${NUM_EPISODES}" \
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
  --diversity_rew_coef 0.0 \
  --alignment_rew_coef 1.0 \
  --critic_learning_rate 0 \
  --critic_lr_head 0 \
  --critic_classifier_loss_coef 0.0 \
  --actor_learning_rate 1e-6 \
  --zero_stage 2 \
  --lr_warmup_ratio 0.03 \
  --lr_scheduler constant_with_warmup \
  --critic_lr_scheduler constant_with_warmup \
  --seed "${SEED}" \
  --ema_beta 0.9 \
  --hidden_state_method concat \
  --embed_method last_token \
  --critic_sequence_level last_token \
  --classifier_sequence_selection closest \
  --eval_steps -1 \
  --eval_down_steps -1 \
  --save_steps -1 \
  --save_log_scale_count -1 \
  --save_even_count 0 \
  --logging_steps 1 \
  --use_tensorboard "${TB_DIR}" \
  --save_path "${SAVE_DIR}" \
  --ckpt_path "${SAVE_DIR}/ckpt" \
  --wandb_run_name "${RUN_NAME}"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Training finished"
echo "Logs: ${RUN_DIR}/train.log"
echo "TensorBoard: ${TB_DIR}"
echo "Checkpoints: ${SAVE_DIR}"
