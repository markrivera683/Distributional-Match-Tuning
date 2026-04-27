#!/usr/bin/env bash
set -euo pipefail

# Launch 4 persistent single-GPU vLLM workers for Qwen3.5-27B and keep them
# alive until this script is terminated.

REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Match-Tuning}"
TEACHER_VENV="${TEACHER_VENV:-${REPO_ROOT}/.teacherVenv}"
TEACHER_VLLM_BIN="${TEACHER_VLLM_BIN:-${TEACHER_VENV}/bin/vllm}"

MODEL_PATH="${MODEL_PATH:-/mnt/data/models/qwen3.5-27b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-27b}"
API_KEY="${API_KEY:-teacher-local}"

WORKER_GPUS_CSV="${WORKER_GPUS_CSV:-4,5,6,7}"
BASE_PORT="${BASE_PORT:-8304}"

DTYPE="${DTYPE:-bfloat16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-128}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
WAIT_SECONDS="${WAIT_SECONDS:-600}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs/teacher_launch_qwen35_multi}"
RUN_NAME="${RUN_NAME:-qwen35_vllm_4worker_live_$(date +%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED=1

if [[ ! -x "${TEACHER_VLLM_BIN}" ]]; then
  echo "[ERROR] TEACHER_VLLM_BIN not executable: ${TEACHER_VLLM_BIN}"
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

IFS=',' read -r -a WORKER_GPUS <<< "${WORKER_GPUS_CSV}"
if (( ${#WORKER_GPUS[@]} != 4 )); then
  echo "[ERROR] WORKER_GPUS_CSV must contain exactly 4 GPU ids, got: ${WORKER_GPUS_CSV}"
  exit 1
fi

declare -a PIDS=()
declare -a PORTS=()
declare -a LOGS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[cleanup] stopping pid=${pid}"
      kill "${pid}" || true
    fi
  done
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

wait_for_worker() {
  local worker_idx="$1"
  local pid="${PIDS[$worker_idx]}"
  local port="${PORTS[$worker_idx]}"
  local log_path="${LOGS[$worker_idx]}"
  local waited=0

  until curl -sf "http://127.0.0.1:${port}/health" >/dev/null; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[worker ${worker_idx}] exited before health check passed"
      echo "[worker ${worker_idx}] log: ${log_path}"
      return 1
    fi
    sleep 3
    waited=$((waited + 3))
    if (( waited >= WAIT_SECONDS )); then
      echo "[worker ${worker_idx}] health check timeout (${WAIT_SECONDS}s)"
      echo "[worker ${worker_idx}] log: ${log_path}"
      return 1
    fi
  done
}

echo "========== Qwen3.5-27B 4-worker LIVE LAUNCH =========="
echo "RUN_DIR:              ${RUN_DIR}"
echo "WORKER_GPUS_CSV:      ${WORKER_GPUS_CSV}"
echo "BASE_PORT:            ${BASE_PORT}"
echo "MAX_MODEL_LEN:        ${MAX_MODEL_LEN}"
echo "MAX_NUM_SEQS:         ${MAX_NUM_SEQS}"
echo "MAX_BATCHED_TOKENS:   ${MAX_BATCHED_TOKENS}"
echo "GPU_MEMORY_UTIL:      ${GPU_MEMORY_UTIL}"
echo "======================================================"

for idx in "${!WORKER_GPUS[@]}"; do
  port=$((BASE_PORT + idx))
  worker_dir="${RUN_DIR}/worker_${idx}"
  log_path="${worker_dir}/teacher.log"
  mkdir -p "${worker_dir}"
  echo "[worker ${idx}] launching on GPU ${WORKER_GPUS[$idx]}, port ${port}"
  CUDA_VISIBLE_DEVICES="${WORKER_GPUS[$idx]}" \
  "${TEACHER_VLLM_BIN}" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${port}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --dtype "${DTYPE}" \
    --api-key "${API_KEY}" \
    --generation-config vllm \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
    --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
    --enable-chunked-prefill \
    > "${log_path}" 2>&1 &
  PIDS+=("$!")
  PORTS+=("${port}")
  LOGS+=("${log_path}")
done

echo
for idx in "${!WORKER_GPUS[@]}"; do
  wait_for_worker "${idx}"
  echo "[worker ${idx}] healthy on GPU ${WORKER_GPUS[$idx]} port ${PORTS[$idx]}"
done

echo
echo "Teacher API bases:"
for port in "${PORTS[@]}"; do
  echo "- http://127.0.0.1:${port}/v1"
done
echo
echo "[live] workers are ready; keeping process alive."

while true; do
  sleep 30
done
