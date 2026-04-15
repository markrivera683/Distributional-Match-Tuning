#!/usr/bin/env bash
# AOPS G3: single-node 8 GPU, online teacher + student training
# G3 vs G2: enables EMA + feature adapter + trainable critic head
# Manual resource split:
#   - teacher GPU ids: TEACHER_CUDA_VISIBLE_DEVICES
#   - student GPU ids: STUDENT_CUDA_VISIBLE_DEVICES
#   - student actor/critic GPU counts: ACTOR_GPUS / CRITIC_GPUS
#   - ref/reward follow actor/critic automatically (colocate)
set -euo pipefail

# Override examples:
#   TARGET_STEPS=500 bash scripts/run_G3_rebase.sh
#   MAX_SAMPLES=10000 bash scripts/run_G3_rebase.sh

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "$csv" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"$csv"
}

# ====================================================================
# 0) MANUAL GPU ASSIGNMENT (edit these first)
# ====================================================================
TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
STUDENT_CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES:-6,7}"

ACTOR_GPUS="${ACTOR_GPUS:-1}"
CRITIC_GPUS="${CRITIC_GPUS:-1}"
REF_GPUS="${ACTOR_GPUS}"
REWARD_GPUS="${CRITIC_GPUS}"

# ====================================================================
# 1) TEACHER SERVING CONFIG (local vLLM)
# ====================================================================
LAUNCH_TEACHER="${LAUNCH_TEACHER:-true}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/mnt/data/models/qwen3.5-27b}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-qwen3.5-27b}"
TEACHER_BASE_PORT="${TEACHER_BASE_PORT:-8004}"
TEACHER_API_KEY="${TEACHER_API_KEY:-teacher-local}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
TEACHER_DTYPE="${TEACHER_DTYPE:-bfloat16}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-2048}"
TEACHER_MAX_NUM_SEQS="${TEACHER_MAX_NUM_SEQS:-32}"
TEACHER_MAX_BATCHED_TOKENS="${TEACHER_MAX_BATCHED_TOKENS:-16384}"
TEACHER_GPU_MEMORY_UTIL="${TEACHER_GPU_MEMORY_UTIL:-0.96}"
TEACHER_WAIT_SECONDS="${TEACHER_WAIT_SECONDS:-1800}"

# Multi-worker: one vLLM worker per TP_SIZE GPUs, each on its own port.
IFS=',' read -r -a _TEACHER_GPU_IDS <<< "${TEACHER_CUDA_VISIBLE_DEVICES}"
TEACHER_WORKER_COUNT=$(( ${#_TEACHER_GPU_IDS[@]} / TEACHER_TP_SIZE ))
_DEFAULT_API_URLS=""
for (( _i=0; _i<TEACHER_WORKER_COUNT; _i++ )); do
  _port=$(( TEACHER_BASE_PORT + _i ))
  [[ -n "${_DEFAULT_API_URLS}" ]] && _DEFAULT_API_URLS="${_DEFAULT_API_URLS},"
  _DEFAULT_API_URLS="${_DEFAULT_API_URLS}http://127.0.0.1:${_port}/v1"
done
TEACHER_API_BASE="${TEACHER_API_BASE:-${_DEFAULT_API_URLS}}"

# ====================================================================
# 2) TRAINING DATA / MODEL PATHS
# ====================================================================
REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Matching-Tuning}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/teacher_model/models/Qwen3.5-0.8B}"
DEFAULT_TRAIN_DATA="/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
TRAIN_DATA="${TRAIN_DATA:-${DEFAULT_TRAIN_DATA}}"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"
TEACHER_VENV="${TEACHER_VENV:-${REPO_ROOT}/.teacherVenv}"
STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"

# ====================================================================
# 3) TRAINING KNOBS
# ====================================================================
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE))}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-4}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-4}"
MICRO_REWARD_BATCH_SIZE="${MICRO_REWARD_BATCH_SIZE:-4}"

PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-384}"
CONTEXT_MAX_LEN="${CONTEXT_MAX_LEN:-8}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-8}"
STRIDE="${STRIDE:-8}"

NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
TARGET_STEPS="${TARGET_STEPS:-500}"
# Trainer computes:
#   max_steps = len(prompts_dataset) * n_samples_per_prompt / train_batch_size * num_episodes * max_epochs
# So we back-solve max_samples to hit TARGET_STEPS by default.
DEFAULT_MAX_SAMPLES="$((TARGET_STEPS * TRAIN_BATCH_SIZE / N_SAMPLES_PER_PROMPT / NUM_EPISODES / MAX_EPOCHS))"
MAX_SAMPLES="${MAX_SAMPLES:-${DEFAULT_MAX_SAMPLES}}"

# ── Teacher target distribution ──
CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.6}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-8}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-768}"
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-200}"
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-3}"
TEACHER_REMOTE_BATCH_SIZE="${TEACHER_REMOTE_BATCH_SIZE:-48}"
TEACHER_SYSTEM_PROMPT_TEXT="${TEACHER_SYSTEM_PROMPT_TEXT:-You are a precise assistant. produce a correct and well-reasoned answer. Step by step when necessary. Keep reasoning sufficient. Final answer is clearly stated.}"
TEACHER_SYSTEM_PROMPT_ID="${TEACHER_SYSTEM_PROMPT_ID:-v1-balanced}"
TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-/root/outputs/teacher_cache_shared}"

# ── Teacher prefetch ──
ENABLE_TEACHER_PREFETCH="${ENABLE_TEACHER_PREFETCH:-true}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-2}"
PREFETCH_MAX_WORKERS="${PREFETCH_MAX_WORKERS:-6}"

# ── Eval / Checkpoint ──
EVAL_STEPS="${EVAL_STEPS:-100}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-50}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-${GENERATE_MAX_LEN}}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_EVEN_COUNT="${SAVE_EVEN_COUNT:-10}"

EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
POST_EVAL_NPROC="${POST_EVAL_NPROC:-8}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
POST_EVAL_MAX_NEW_TOKENS="${POST_EVAL_MAX_NEW_TOKENS:-8192}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_MICRO_BATCH_SIZE="${POST_EVAL_MICRO_BATCH_SIZE:-64}"
POST_EVAL_MASTER_PORT="${POST_EVAL_MASTER_PORT:-29501}"

# ====================================================================
# 4) G3-SPECIFIC: FEATURE ADAPTER + EMA  ← KEY DIFFERENCE vs G2
# ====================================================================
FEATURE_ADAPTER_RANK="${FEATURE_ADAPTER_RANK:-64}"
FEATURE_ADAPTER_DROPOUT="${FEATURE_ADAPTER_DROPOUT:-0.0}"
UNFREEZE_LAYERS="${UNFREEZE_LAYERS:-0}"

ACTOR_LR="${ACTOR_LR:-1e-6}"
CE_LOSS_COEF="${CE_LOSS_COEF:-0.03}"
EMA_BETA="${EMA_BETA:-0.99}"

CRITIC_LR="${CRITIC_LR:-0}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-5e-5}"
CRITIC_CLASSIFIER_LOSS_COEF="${CRITIC_CLASSIFIER_LOSS_COEF:-0.0}"
CRITIC_DIRECT_DISCREPANCY_COEF="${CRITIC_DIRECT_DISCREPANCY_COEF:-0.1}"
CRITIC_DIRECT_DISCREPANCY_TARGET="${CRITIC_DIRECT_DISCREPANCY_TARGET:-ema_gt}"

DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-0.5}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"

# ====================================================================
# 5) ENV / RUN DIR
# ====================================================================
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

RUN_NAME="${RUN_NAME:-g3_rebase_$(date +%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SAVE_PATH="${RUN_DIR}/model"
TB_DIR="${RUN_DIR}/tensorboard"
TEACHER_LOG_DIR="${RUN_DIR}/teacher_logs"
mkdir -p "$RUN_DIR" "$SAVE_PATH" "$TB_DIR" "$TEACHER_CACHE_DIR" "$TEACHER_LOG_DIR"
POST_EVAL_OUTPUT_PATH="${POST_EVAL_OUTPUT_PATH:-${RUN_DIR}/eval_results.jsonl}"
POST_EVAL_LOG_PATH="${POST_EVAL_LOG_PATH:-${RUN_DIR}/eval.log}"

# ====================================================================
# 6) SANITY CHECK
# ====================================================================
teacher_gpu_count="$(count_csv_items "$TEACHER_CUDA_VISIBLE_DEVICES")"
student_gpu_count="$(count_csv_items "$STUDENT_CUDA_VISIBLE_DEVICES")"

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

if (( teacher_gpu_count < TEACHER_TP_SIZE )); then
  echo "[ERROR] teacher GPU count (${teacher_gpu_count}) < TEACHER_TP_SIZE (${TEACHER_TP_SIZE})"
  exit 1
fi
if (( teacher_gpu_count % TEACHER_TP_SIZE != 0 )); then
  echo "[ERROR] teacher GPU count (${teacher_gpu_count}) not divisible by TEACHER_TP_SIZE (${TEACHER_TP_SIZE})"
  exit 1
fi

if (( ACTOR_GPUS + CRITIC_GPUS > student_gpu_count )); then
  echo "[ERROR] ACTOR_GPUS(${ACTOR_GPUS}) + CRITIC_GPUS(${CRITIC_GPUS}) > student GPU count(${student_gpu_count})"
  exit 1
fi

if (( TRAIN_BATCH_SIZE != N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE )); then
  echo "[ERROR] TRAIN_BATCH_SIZE must equal N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE"
  echo "        got ${TRAIN_BATCH_SIZE} vs ${N_SAMPLES_PER_PROMPT} * ${ROLLOUT_BATCH_SIZE}"
  exit 1
fi

if (( TRAIN_BATCH_SIZE % (MICRO_TRAIN_BATCH_SIZE * ACTOR_GPUS) != 0 )); then
  echo "[ERROR] train_batch_size % (micro_train_batch_size * actor_gpus) != 0"
  echo "        ${TRAIN_BATCH_SIZE} % (${MICRO_TRAIN_BATCH_SIZE} * ${ACTOR_GPUS}) != 0"
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

echo "========== AOPS G3 online-teacher run =========="
echo "RUN_DIR:                    ${RUN_DIR}"
echo "Teacher GPUs:               ${TEACHER_CUDA_VISIBLE_DEVICES} (count=${teacher_gpu_count})"
echo "Teacher workers:            ${TEACHER_WORKER_COUNT} x TP${TEACHER_TP_SIZE}"
echo "Teacher max_model_len:      ${TEACHER_MAX_MODEL_LEN}"
echo "Teacher max_num_seqs:       ${TEACHER_MAX_NUM_SEQS}"
echo "Teacher max_batched_tokens: ${TEACHER_MAX_BATCHED_TOKENS}"
echo "Teacher remote_batch_size:  ${TEACHER_REMOTE_BATCH_SIZE}"
echo "Student GPUs:               ${STUDENT_CUDA_VISIBLE_DEVICES} (count=${student_gpu_count})"
echo "Actor/Critic GPUs:          ${ACTOR_GPUS}/${CRITIC_GPUS}"
echo "Ref/Reward GPUs (colocate): ${REF_GPUS}/${REWARD_GPUS}"
echo "Teacher API:                ${TEACHER_API_BASE}"
echo "Teacher model:              ${TEACHER_MODEL_NAME}"
echo "Train data:                 ${TRAIN_DATA}"
echo "Eval data:                  ${EVAL_DATA}"
echo "target_steps:               ${TARGET_STEPS}"
echo "max_samples:                ${MAX_SAMPLES}"
echo "Teacher vLLM bin:           ${TEACHER_VLLM_BIN}"
echo "Student python:             ${STUDENT_PYTHON_BIN}"
echo ""
echo "[G3 specific]"
echo "  enable_ema:               true"
echo "  ema_beta:                 ${EMA_BETA}"
echo "  feature_adapter:          rank=${FEATURE_ADAPTER_RANK} dropout=${FEATURE_ADAPTER_DROPOUT}"
echo "  unfreeze_layers:          ${UNFREEZE_LAYERS}"
echo "  critic_lr (backbone):     ${CRITIC_LR}"
echo "  critic_lr_head:           ${CRITIC_LR_HEAD}"
echo "  discrepancy_coef:         ${CRITIC_DIRECT_DISCREPANCY_COEF}"
echo "  discrepancy_target:       ${CRITIC_DIRECT_DISCREPANCY_TARGET}"
echo "================================================="

# ====================================================================
# 7) TEACHER LAUNCH + HEALTH CHECK
# ====================================================================
declare -a TEACHER_PIDS=()
declare -a TEACHER_PORTS=()
declare -a TEACHER_WORKER_LOGS=()

cleanup() {
  for _pid in "${TEACHER_PIDS[@]:-}"; do
    if [[ -n "${_pid}" ]] && kill -0 "${_pid}" 2>/dev/null; then
      echo "[cleanup] stopping teacher pid=${_pid}"
      kill "${_pid}" || true
    fi
  done
  for _pid in "${TEACHER_PIDS[@]:-}"; do
    [[ -n "${_pid}" ]] && wait "${_pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

wait_for_teacher_worker() {
  local idx="$1" pid="$2" port="$3" log="$4" waited=0
  until curl -sf "http://127.0.0.1:${port}/health" >/dev/null; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[ERROR] Teacher worker ${idx} exited before health check. Log: ${log}"
      return 1
    fi
    sleep 3
    waited=$((waited + 3))
    if (( waited >= TEACHER_WAIT_SECONDS )); then
      echo "[ERROR] Teacher worker ${idx} health timeout (${TEACHER_WAIT_SECONDS}s). Log: ${log}"
      return 1
    fi
  done
  return 0
}

if [[ "${LAUNCH_TEACHER}" == "true" ]]; then
  echo "[teacher] launching ${TEACHER_WORKER_COUNT} workers (TP=${TEACHER_TP_SIZE}) ..."
  for (( _w=0; _w<TEACHER_WORKER_COUNT; _w++ )); do
    _port=$(( TEACHER_BASE_PORT + _w ))
    _gpu_start=$(( _w * TEACHER_TP_SIZE ))
    _worker_gpus=""
    for (( _g=_gpu_start; _g<_gpu_start+TEACHER_TP_SIZE; _g++ )); do
      [[ -n "${_worker_gpus}" ]] && _worker_gpus="${_worker_gpus},"
      _worker_gpus="${_worker_gpus}${_TEACHER_GPU_IDS[$_g]}"
    done
    _log="${TEACHER_LOG_DIR}/worker_${_w}.log"

    CUDA_VISIBLE_DEVICES="${_worker_gpus}" \
    "${TEACHER_VLLM_BIN}" serve "${TEACHER_MODEL_PATH}" \
      --served-model-name "${TEACHER_MODEL_NAME}" \
      --host 0.0.0.0 \
      --port "${_port}" \
      --tensor-parallel-size "${TEACHER_TP_SIZE}" \
      --dtype "${TEACHER_DTYPE}" \
      --api-key "${TEACHER_API_KEY}" \
      --generation-config vllm \
      --max-model-len "${TEACHER_MAX_MODEL_LEN}" \
      --max-num-seqs "${TEACHER_MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${TEACHER_MAX_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${TEACHER_GPU_MEMORY_UTIL}" \
      --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
      --enable-chunked-prefill \
      > "${_log}" 2>&1 &

    TEACHER_PIDS+=("$!")
    TEACHER_PORTS+=("${_port}")
    TEACHER_WORKER_LOGS+=("${_log}")
    echo "[teacher] worker ${_w}: GPU ${_worker_gpus}, port ${_port}, log ${_log}"
  done

  for (( _w=0; _w<TEACHER_WORKER_COUNT; _w++ )); do
    wait_for_teacher_worker "${_w}" "${TEACHER_PIDS[$_w]}" "${TEACHER_PORTS[$_w]}" "${TEACHER_WORKER_LOGS[$_w]}"
    echo "[teacher] worker ${_w} healthy."
  done
  echo "[teacher] all ${TEACHER_WORKER_COUNT} workers ready."
fi

# ====================================================================
# 8) TRAIN
# ====================================================================
PREFETCH_FLAGS=()
if [[ "${ENABLE_TEACHER_PREFETCH}" == "true" ]]; then
  PREFETCH_FLAGS=(
    --enable_teacher_prefetch
    --prefetch_depth "${PREFETCH_DEPTH}"
    --prefetch_max_workers "${PREFETCH_MAX_WORKERS}"
  )
fi

ray stop --force 2>/dev/null || true
sleep 2
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}" \
"${STUDENT_PYTHON_BIN}" -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --pretrain_mode --no_chat_template \
  --disable_ds_ckpt --colocate_actor_ref --colocate_critic_reward \
  --gradient_checkpointing --use_kl_loss --use_whitening --enable_ema \
  --feature_adapter_enable \
  --feature_adapter_type residual_bottleneck \
  --feature_adapter_rank "${FEATURE_ADAPTER_RANK}" \
  --feature_adapter_dropout "${FEATURE_ADAPTER_DROPOUT}" \
  --feature_adapter_unfreeze_layers "${UNFREEZE_LAYERS}" \
  \
  --distribution_reward_type cf_l1oo \
  --feature_map_type identity --rff_num_features 128 --rff_sigma 1.0 --rff_seed 43 \
  --cf_num_freqs 128 --cf_sigma 1.0 --cf_seed 43 --cf_alpha 0.5 --cf_beta 0.5 --cf_reward_scale 1.0 \
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
  --logging_steps 10 \
  --save_steps "${SAVE_STEPS}" --save_even_count "${SAVE_EVEN_COUNT}" --save_hf_ckpt \
  --use_tensorboard "${TB_DIR}" \
  --save_path "${SAVE_PATH}" --ckpt_path "${SAVE_PATH}/ckpt" \
  --wandb_run_name "${RUN_NAME}" \
  2>&1 | tee "${RUN_DIR}/train.log"

# ====================================================================
# POST-TRAINING EVAL
# ====================================================================
ray stop --force 2>/dev/null || true

echo ""
echo "──────────────────────────────────────────────────"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')  TRAINING FINISHED"
echo "  Logs:        ${RUN_DIR}/train.log"
echo "  TensorBoard: ${TB_DIR}"
echo "  Checkpoints: ${SAVE_PATH}"

if [[ "${EVAL_AFTER_TRAIN}" == "true" ]]; then
  if (( ${#TEACHER_PIDS[@]} > 0 )); then
    echo "[post-eval] stopping ${#TEACHER_PIDS[@]} teacher workers to free GPU memory..."
    for _pid in "${TEACHER_PIDS[@]}"; do
      kill "${_pid}" 2>/dev/null || true
    done
    for _pid in "${TEACHER_PIDS[@]}"; do
      wait "${_pid}" 2>/dev/null || true
    done
    TEACHER_PIDS=()
  fi

  echo ""
  echo "[post-eval] Running generation eval on ${EVAL_DATA} ..."
  CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
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
echo "G3 online teacher run completed at $(date)" > "${RUN_DIR}/status.txt"
echo "[done] logs: ${RUN_DIR}"
