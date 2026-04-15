#!/usr/bin/env bash
set -euo pipefail

# Minimal GPU smoke test for serving Qwen3.5-27B via vLLM.
# Default behavior:
# - uses a single idle GPU (default: GPU 7)
# - launches a local OpenAI-compatible server
# - checks /health and /v1/models
# - sends one tiny /v1/completions request
# - cleans up the server process automatically

REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Matching-Tuning}"
TEACHER_VENV="${TEACHER_VENV:-${REPO_ROOT}/.teacherVenv}"
TEACHER_VLLM_BIN="${TEACHER_VLLM_BIN:-${TEACHER_VENV}/bin/vllm}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/qwen3.5-27b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-27b}"
PORT="${PORT:-8017}"
API_KEY="${API_KEY:-teacher-local}"

DTYPE="${DTYPE:-bfloat16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-512}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.60}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs/smoketest_teacher_qwen35}"
RUN_NAME="${RUN_NAME:-qwen35_vllm_smoke_$(date +%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_PATH="${RUN_DIR}/teacher.log"
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

echo "========== Qwen3.5-27B vLLM 1GPU SMOKETEST =========="
echo "RUN_DIR:                ${RUN_DIR}"
echo "CUDA_VISIBLE_DEVICES:   ${CUDA_VISIBLE_DEVICES}"
echo "MODEL_PATH:             ${MODEL_PATH}"
echo "SERVED_MODEL_NAME:      ${SERVED_MODEL_NAME}"
echo "TEACHER_VLLM_BIN:       ${TEACHER_VLLM_BIN}"
echo "PORT:                   ${PORT}"
echo "DTYPE:                  ${DTYPE}"
echo "TP_SIZE:                ${TP_SIZE}"
echo "MAX_MODEL_LEN:          ${MAX_MODEL_LEN}"
echo "MAX_NUM_SEQS:           ${MAX_NUM_SEQS}"
echo "MAX_BATCHED_TOKENS:     ${MAX_BATCHED_TOKENS}"
echo "GPU_MEMORY_UTIL:        ${GPU_MEMORY_UTIL}"
echo "LOG_PATH:               ${LOG_PATH}"
echo "===================================================="

TEACHER_PID=""
cleanup() {
  if [[ -n "${TEACHER_PID}" ]] && kill -0 "${TEACHER_PID}" 2>/dev/null; then
    echo "[cleanup] stopping teacher pid=${TEACHER_PID}"
    kill "${TEACHER_PID}" || true
    wait "${TEACHER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_teacher() {
  local waited=0
  until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; do
    if [[ -n "${TEACHER_PID}" ]] && ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
      echo "[ERROR] Teacher exited before health check passed."
      echo "        Check log: ${LOG_PATH}"
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
    if (( waited >= WAIT_SECONDS )); then
      echo "[ERROR] Teacher health check timeout (${WAIT_SECONDS}s)."
      echo "        Check log: ${LOG_PATH}"
      return 1
    fi
  done
}

echo "[teacher] launching vLLM on GPU ${CUDA_VISIBLE_DEVICES} ..."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${TEACHER_VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
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
  > "${LOG_PATH}" 2>&1 &
TEACHER_PID=$!
echo "[teacher] pid=${TEACHER_PID}"

wait_for_teacher
echo "[teacher] health check passed."

echo "[probe] /v1/models"
curl -sf \
  -H "Authorization: Bearer ${API_KEY}" \
  "http://127.0.0.1:${PORT}/v1/models" | python -m json.tool

echo "[probe] /v1/completions"
curl -sf \
  -X POST "http://127.0.0.1:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "{
    \"model\": \"${SERVED_MODEL_NAME}\",
    \"prompt\": \"Question: What is 2+2?\\nAnswer:\",
    \"max_tokens\": 16,
    \"temperature\": 0.0,
    \"top_p\": 1.0,
    \"n\": 1
  }" | python -m json.tool

echo "[done] smoke test passed. log: ${LOG_PATH}"
