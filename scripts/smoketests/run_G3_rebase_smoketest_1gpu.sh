#!/usr/bin/env bash
# 1-GPU smoke test for G3 rebase pipeline.
# This script intentionally uses tiny settings and colocates all models.
set -euo pipefail

TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-0}"
STUDENT_CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES:-0}"

ACTOR_GPUS="${ACTOR_GPUS:-1}"
CRITIC_GPUS="${CRITIC_GPUS:-1}"
REF_GPUS="${REF_GPUS:-1}"
REWARD_GPUS="${REWARD_GPUS:-1}"

LAUNCH_TEACHER="${LAUNCH_TEACHER:-true}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/mnt/data/teacher_model/models/Qwen3.5-0.8B}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-Qwen3.5-0.8B}"
TEACHER_PORT="${TEACHER_PORT:-18041}"
TEACHER_API_KEY="${TEACHER_API_KEY:-teacher-local}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
TEACHER_DTYPE="${TEACHER_DTYPE:-auto}"
TEACHER_MAX_NUM_SEQS="${TEACHER_MAX_NUM_SEQS:-4}"
TEACHER_MAX_BATCHED_TOKENS="${TEACHER_MAX_BATCHED_TOKENS:-4096}"
TEACHER_GPU_MEMORY_UTIL="${TEACHER_GPU_MEMORY_UTIL:-0.45}"
TEACHER_WAIT_SECONDS="${TEACHER_WAIT_SECONDS:-180}"
TEACHER_API_BASE="${TEACHER_API_BASE:-http://127.0.0.1:${TEACHER_PORT}/v1}"

REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Matching-Tuning}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/teacher_model/models/Qwen3.5-0.8B}"
DEFAULT_TRAIN_DATA="/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
TRAIN_DATA="${TRAIN_DATA:-${DEFAULT_TRAIN_DATA}}"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"
TEACHER_VENV="${TEACHER_VENV:-${REPO_ROOT}/.teacherVenv}"
STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"

N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE))}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-${N_SAMPLES_PER_PROMPT}}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-2}"
MICRO_REWARD_BATCH_SIZE="${MICRO_REWARD_BATCH_SIZE:-1}"

PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-64}"
CONTEXT_MAX_LEN="${CONTEXT_MAX_LEN:-8}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-8}"
STRIDE="${STRIDE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"

CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.6}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-2}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-64}"
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-120}"
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-2}"
TEACHER_REMOTE_BATCH_SIZE="${TEACHER_REMOTE_BATCH_SIZE:-2}"
TEACHER_SYSTEM_PROMPT_TEXT="${TEACHER_SYSTEM_PROMPT_TEXT:-You are a precise assistant. produce a correct and well-reasoned answer.}"
TEACHER_SYSTEM_PROMPT_ID="${TEACHER_SYSTEM_PROMPT_ID:-smoke-v1}"
TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-/root/outputs/teacher_cache_smoke_1gpu}"

ENABLE_TEACHER_PREFETCH="${ENABLE_TEACHER_PREFETCH:-false}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-1}"
PREFETCH_MAX_WORKERS="${PREFETCH_MAX_WORKERS:-1}"
ENABLE_FLASH_ATTN="${ENABLE_FLASH_ATTN:-false}"

# G3-specific knobs
FEATURE_ADAPTER_RANK="${FEATURE_ADAPTER_RANK:-32}"
FEATURE_ADAPTER_DROPOUT="${FEATURE_ADAPTER_DROPOUT:-0.0}"
UNFREEZE_LAYERS="${UNFREEZE_LAYERS:-0}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
CE_LOSS_COEF="${CE_LOSS_COEF:-0.03}"
EMA_BETA="${EMA_BETA:-0.9}"
CRITIC_LR="${CRITIC_LR:-0}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-5e-5}"
CRITIC_CLASSIFIER_LOSS_COEF="${CRITIC_CLASSIFIER_LOSS_COEF:-0.0}"
CRITIC_DIRECT_DISCREPANCY_COEF="${CRITIC_DIRECT_DISCREPANCY_COEF:-0.1}"
CRITIC_DIRECT_DISCREPANCY_TARGET="${CRITIC_DIRECT_DISCREPANCY_TARGET:-ema_gt}"
DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-0.5}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"

# Align with run_G3_rebase.sh (scaled down for smoke)
EVAL_STEPS="${EVAL_STEPS:-1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-2}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-${GENERATE_MAX_LEN}}"
SAVE_STEPS="${SAVE_STEPS:-1}"
SAVE_EVEN_COUNT="${SAVE_EVEN_COUNT:-2}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
POST_EVAL_NPROC="${POST_EVAL_NPROC:-1}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-4}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-64}"
POST_EVAL_MAX_NEW_TOKENS="${POST_EVAL_MAX_NEW_TOKENS:-256}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_MICRO_BATCH_SIZE="${POST_EVAL_MICRO_BATCH_SIZE:-2}"
POST_EVAL_MASTER_PORT="${POST_EVAL_MASTER_PORT:-29513}"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

TEACHER_VLLM_BIN="${TEACHER_VLLM_BIN:-${TEACHER_VENV}/bin/vllm}"
if [[ ! -x "${TEACHER_VLLM_BIN}" ]]; then
  echo "[ERROR] TEACHER_VLLM_BIN not executable: ${TEACHER_VLLM_BIN}"
  echo "        expected teacher env: ${TEACHER_VENV}"
  exit 1
fi

STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"
if [[ ! -x "${STUDENT_PYTHON_BIN}" ]]; then
  echo "[ERROR] STUDENT_PYTHON_BIN not executable: ${STUDENT_PYTHON_BIN}"
  echo "        expected student env: ${STUDENT_VENV}"
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi
if [[ ! -d "${TEACHER_MODEL_PATH}" ]]; then
  echo "[ERROR] TEACHER_MODEL_PATH not found: ${TEACHER_MODEL_PATH}"
  exit 1
fi
if [[ "${TRAIN_DATA}" == "${DEFAULT_TRAIN_DATA}" && ! -e "${TRAIN_DATA}" && -f "${FALLBACK_LOCAL_DATA}" ]]; then
  echo "[WARN] TRAIN_DATA default not found, fallback to ${FALLBACK_LOCAL_DATA}"
  TRAIN_DATA="${FALLBACK_LOCAL_DATA}"
fi
if [[ "${EVAL_DATA}" == "${DEFAULT_EVAL_DATA}" && ! -e "${EVAL_DATA}" && -f "${FALLBACK_LOCAL_DATA}" ]]; then
  echo "[WARN] EVAL_DATA default not found, fallback to ${FALLBACK_LOCAL_DATA}"
  EVAL_DATA="${FALLBACK_LOCAL_DATA}"
fi
if [[ "${TRAIN_DATA}" == /* && ! -e "${TRAIN_DATA}" ]]; then
  echo "[ERROR] TRAIN_DATA not found: ${TRAIN_DATA}"
  exit 1
fi
if [[ "${EVAL_DATA}" == /* && ! -e "${EVAL_DATA}" ]]; then
  echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"
  exit 1
fi
if (( TRAIN_BATCH_SIZE != N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE )); then
  echo "[ERROR] TRAIN_BATCH_SIZE must equal N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE"
  exit 1
fi
if (( TRAIN_BATCH_SIZE % (MICRO_TRAIN_BATCH_SIZE * ACTOR_GPUS) != 0 )); then
  echo "[ERROR] train_batch_size % (micro_train_batch_size * actor_gpus) != 0"
  exit 1
fi
if (( MICRO_TRAIN_BATCH_SIZE < N_SAMPLES_PER_PROMPT || MICRO_TRAIN_BATCH_SIZE % N_SAMPLES_PER_PROMPT != 0 )); then
  echo "[ERROR] MICRO_TRAIN_BATCH_SIZE must be >= N_SAMPLES_PER_PROMPT and divisible by it"
  echo "        got MICRO_TRAIN_BATCH_SIZE=${MICRO_TRAIN_BATCH_SIZE}, N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}"
  exit 1
fi
if (( MICRO_ROLLOUT_BATCH_SIZE % N_SAMPLES_PER_PROMPT != 0 )); then
  echo "[ERROR] MICRO_ROLLOUT_BATCH_SIZE must be divisible by N_SAMPLES_PER_PROMPT"
  exit 1
fi

RUN_NAME="${RUN_NAME:-smoke_g3_1gpu_$(date +%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs/smoketest_1gpu}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SAVE_PATH="${RUN_DIR}/model"
TB_DIR="${RUN_DIR}/tensorboard"
TEACHER_LOG="${RUN_DIR}/teacher.log"
mkdir -p "${RUN_DIR}" "${SAVE_PATH}" "${TB_DIR}" "${TEACHER_CACHE_DIR}"
POST_EVAL_OUTPUT_PATH="${POST_EVAL_OUTPUT_PATH:-${RUN_DIR}/eval_results.jsonl}"
POST_EVAL_LOG_PATH="${POST_EVAL_LOG_PATH:-${RUN_DIR}/eval.log}"

echo "========== G3 rebase 1GPU SMOKETEST =========="
echo "RUN_DIR:         ${RUN_DIR}"
echo "Teacher GPU(s):  ${TEACHER_CUDA_VISIBLE_DEVICES} (tp=${TEACHER_TP_SIZE})"
echo "Student GPU(s):  ${STUDENT_CUDA_VISIBLE_DEVICES}"
echo "Model path:      ${MODEL_PATH}"
echo "Teacher model:   ${TEACHER_MODEL_PATH}"
echo "Train data:      ${TRAIN_DATA}"
echo "Eval data:       ${EVAL_DATA}"
echo "Teacher vLLM:    ${TEACHER_VLLM_BIN}"
echo "Student python:  ${STUDENT_PYTHON_BIN}"
echo "FlashAttention:  ${ENABLE_FLASH_ATTN}"
echo "======================================="

wait_for_teacher() {
  local waited=0
  until curl -sf "http://127.0.0.1:${TEACHER_PORT}/health" >/dev/null; do
    if [[ -n "${TEACHER_PID}" ]] && ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
      echo "[ERROR] Teacher process exited before health check passed."
      echo "        Check log: ${TEACHER_LOG}"
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
    if (( waited >= TEACHER_WAIT_SECONDS )); then
      echo "[ERROR] Teacher health check timeout (${TEACHER_WAIT_SECONDS}s)."
      echo "        Check log: ${TEACHER_LOG}"
      return 1
    fi
  done
}

TEACHER_PID=""
cleanup() {
  if [[ -n "${TEACHER_PID}" ]] && kill -0 "${TEACHER_PID}" 2>/dev/null; then
    echo "[cleanup] stopping teacher pid=${TEACHER_PID}"
    kill "${TEACHER_PID}" || true
  fi
}
trap cleanup EXIT

if [[ "${LAUNCH_TEACHER}" == "true" ]]; then
  CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}" \
  "${TEACHER_VLLM_BIN}" serve "${TEACHER_MODEL_PATH}" \
    --served-model-name "${TEACHER_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${TEACHER_PORT}" \
    --tensor-parallel-size "${TEACHER_TP_SIZE}" \
    --dtype "${TEACHER_DTYPE}" \
    --api-key "${TEACHER_API_KEY}" \
    --generation-config vllm \
    --max-num-seqs "${TEACHER_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${TEACHER_MAX_BATCHED_TOKENS}" \
    --enable-chunked-prefill \
    --gpu-memory-utilization "${TEACHER_GPU_MEMORY_UTIL}" \
    > "${TEACHER_LOG}" 2>&1 &
  TEACHER_PID=$!
  echo "[teacher] pid=${TEACHER_PID}, log=${TEACHER_LOG}"
fi

wait_for_teacher
echo "[teacher] health check passed."

PREFETCH_FLAGS=()
if [[ "${ENABLE_TEACHER_PREFETCH}" == "true" ]]; then
  PREFETCH_FLAGS=(
    --enable_teacher_prefetch
    --prefetch_depth "${PREFETCH_DEPTH}"
    --prefetch_max_workers "${PREFETCH_MAX_WORKERS}"
  )
fi

FLASH_ATTN_FLAGS=()
if [[ "${ENABLE_FLASH_ATTN}" == "true" ]]; then
  FLASH_ATTN_FLAGS=(--flash_attn)
fi

ray stop --force 2>/dev/null || true
sleep 2
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}" \
"${STUDENT_PYTHON_BIN}" -m openrlhf.cli.train_ebft_ray \
  --bf16 "${FLASH_ATTN_FLAGS[@]}" --pretrain_mode --no_chat_template \
  --disable_ds_ckpt --colocate_all_models \
  --gradient_checkpointing --use_kl_loss --use_whitening --enable_ema \
  --feature_adapter_enable \
  --feature_adapter_type residual_bottleneck \
  --feature_adapter_rank "${FEATURE_ADAPTER_RANK}" \
  --feature_adapter_dropout "${FEATURE_ADAPTER_DROPOUT}" \
  --feature_adapter_unfreeze_layers "${UNFREEZE_LAYERS}" \
  \
  --distribution_reward_type cf_l1oo \
  --feature_map_type identity --rff_num_features 64 --rff_sigma 1.0 --rff_seed 43 \
  --cf_num_freqs 32 --cf_sigma 1.0 --cf_seed 43 --cf_alpha 0.5 --cf_beta 0.5 --cf_reward_scale 1.0 \
  --cf_target_mode teacher --cf_teacher_lambda "${CF_TEACHER_LAMBDA}" --cf_teacher_n_samples "${CF_TEACHER_N_SAMPLES}" \
  \
  --teacher_backend remote \
  --teacher_api_base "${TEACHER_API_BASE}" \
  --teacher_api_key "${TEACHER_API_KEY}" \
  --teacher_api_style completions \
  --teacher_model_name "${TEACHER_MODEL_NAME}" \
  --teacher_timeout "${TEACHER_TIMEOUT}" \
  --teacher_max_retries "${TEACHER_MAX_RETRIES}" \
  --teacher_remote_batch_size "${TEACHER_REMOTE_BATCH_SIZE}" \
  --teacher_temperature "${TEACHER_TEMPERATURE}" \
  --teacher_top_p "${TEACHER_TOP_P}" \
  --teacher_max_new_tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher_system_prompt_text "${TEACHER_SYSTEM_PROMPT_TEXT}" \
  --teacher_system_prompt_id "${TEACHER_SYSTEM_PROMPT_ID}" \
  --teacher_cache_enable --teacher_cache_dir "${TEACHER_CACHE_DIR}" \
  "${PREFETCH_FLAGS[@]}" \
  \
  --embed_method last_token --critic_sequence_level last_token \
  --critic_learning_rate "${CRITIC_LR}" \
  --critic_lr_head "${CRITIC_LR_HEAD}" \
  --critic_classifier_loss_coef "${CRITIC_CLASSIFIER_LOSS_COEF}" \
  --critic_direct_discrepancy_coef "${CRITIC_DIRECT_DISCREPANCY_COEF}" \
  --critic_direct_discrepancy_target "${CRITIC_DIRECT_DISCREPANCY_TARGET}" \
  --ema_beta "${EMA_BETA}" \
  --ce_loss_coef "${CE_LOSS_COEF}" \
  --diversity_rew_coef "${DIVERSITY_REW_COEF}" \
  --alignment_rew_coef "${ALIGNMENT_REW_COEF}" \
  --actor_learning_rate "${ACTOR_LR}" \
  \
  --pretrain "${MODEL_PATH}" --critic_pretrain "${MODEL_PATH}" \
  --prompt_data "${TRAIN_DATA}" --eval_dataset "${EVAL_DATA}" \
  --input_key question --label_key answer --output_key answer \
  --prompt_split train --eval_split test \
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
  --zero_stage 2 --lr_warmup_ratio 0.03 --critic_lr_warmup_ratio 0.0 \
  --seed 43 \
  --eval_steps "${EVAL_STEPS}" \
  --eval_max_samples "${EVAL_MAX_SAMPLES}" \
  --eval_generate_max_len "${EVAL_GENERATE_MAX_LEN}" \
  --logging_steps 1 \
  --save_steps "${SAVE_STEPS}" --save_even_count "${SAVE_EVEN_COUNT}" --save_hf_ckpt \
  --use_tensorboard "${TB_DIR}" \
  --save_path "${SAVE_PATH}" --ckpt_path "${SAVE_PATH}/ckpt" \
  --wandb_run_name "${RUN_NAME}" \
  2>&1 | tee "${RUN_DIR}/train.log"

ray stop --force 2>/dev/null || true

echo ""
echo "──────────────────────────────────────────────────"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')  TRAINING FINISHED"
echo "  Logs:        ${RUN_DIR}/train.log"
echo "  Checkpoints: ${SAVE_PATH}"

if [[ "${EVAL_AFTER_TRAIN}" == "true" ]]; then
  if [[ -n "${TEACHER_PID}" ]] && kill -0 "${TEACHER_PID}" 2>/dev/null; then
    echo "[post-eval] stopping teacher to free teacher GPU memory..."
    kill "${TEACHER_PID}" || true
    wait "${TEACHER_PID}" 2>/dev/null || true
    TEACHER_PID=""
  fi

  echo ""
  echo "[post-eval] Running generation eval on ${EVAL_DATA} ..."
  CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}" \
  "${STUDENT_PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node "${POST_EVAL_NPROC}" --master_port "${POST_EVAL_MASTER_PORT}" \
    -m openrlhf.cli.batch_inference \
    --eval_task generate \
    --pretrain "${SAVE_PATH}" \
    --dataset "${EVAL_DATA}" \
    --input_key question \
    --output_path "${POST_EVAL_OUTPUT_PATH}" \
    --prompt_max_len "${POST_EVAL_PROMPT_MAX_LEN}" \
    --max_new_tokens "${POST_EVAL_MAX_NEW_TOKENS}" \
    --temperature "${POST_EVAL_TEMPERATURE}" \
    --top_p "${POST_EVAL_TOP_P}" \
    --max_samples "${POST_EVAL_MAX_SAMPLES}" \
    --micro_batch_size "${POST_EVAL_MICRO_BATCH_SIZE}" \
    --bf16 \
    2>&1 | tee "${POST_EVAL_LOG_PATH}"
  echo "[post-eval] Saved: ${POST_EVAL_OUTPUT_PATH}"
  echo "[post-eval] Log:   ${POST_EVAL_LOG_PATH}"

  echo ""
  echo "[analysis] Running eval analysis ..."
  "${STUDENT_PYTHON_BIN}" "${REPO_ROOT}/scripts/analyze_eval_results.py" \
    --eval_results "${POST_EVAL_OUTPUT_PATH}" \
    --eval_dataset "${EVAL_DATA}" \
    --input_key question --label_key answer \
    --report_path "${RUN_DIR}/eval_analysis.json" \
    2>&1 | tee "${RUN_DIR}/eval_analysis.log"
fi

echo "──────────────────────────────────────────────────"
echo "G3 rebase 1GPU smoke test completed at $(date)" > "${RUN_DIR}/status.txt"
echo "[done] logs: ${RUN_DIR}"
