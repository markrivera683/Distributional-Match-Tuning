#!/usr/bin/env bash
set -euo pipefail

# Smoketest for the G3 multi-worker teacher setup.
# Phase 1 — Feasibility: launch workers, health-check, single-request probe.
# Phase 2 — High concurrency: run benchmark_teacher_multiworker.py at batch_size=32.
#
# Usage:
#   bash scripts/smoketests/smoketest_G3_multiworker.sh
#   WORKER_GPUS_CSV=0,1,2,3 bash scripts/smoketests/smoketest_G3_multiworker.sh

REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Match-Tuning}"
TEACHER_VENV="${TEACHER_VENV:-${REPO_ROOT}/.teacherVenv}"
STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"
TEACHER_VLLM_BIN="${TEACHER_VLLM_BIN:-${TEACHER_VENV}/bin/vllm}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"

MODEL_PATH="${MODEL_PATH:-/mnt/data/models/qwen3.5-27b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-27b}"
API_KEY="${API_KEY:-teacher-local}"

WORKER_GPUS_CSV="${WORKER_GPUS_CSV:-0,1,2,3}"
BASE_PORT="${BASE_PORT:-8604}"

DTYPE="${DTYPE:-bfloat16}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
WAIT_SECONDS="${WAIT_SECONDS:-600}"

BENCH_BATCH_SIZE="${BENCH_BATCH_SIZE:-32}"
BENCH_PROMPTS_PER_ITER="${BENCH_PROMPTS_PER_ITER:-32}"
BENCH_MEASURE_ITERS="${BENCH_MEASURE_ITERS:-2}"
BENCH_MAX_NEW_TOKENS="${BENCH_MAX_NEW_TOKENS:-256}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs/smoketest_G3_multiworker}"
RUN_NAME="${RUN_NAME:-g3_smoke_$(date +%m%d_%H%M%S)}"
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
  echo "[ERROR] TEACHER_VLLM_BIN not executable: ${TEACHER_VLLM_BIN}"; exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 1
fi

IFS=',' read -r -a WORKER_GPUS <<< "${WORKER_GPUS_CSV}"
WORKER_COUNT="${#WORKER_GPUS[@]}"

declare -a PIDS=()
declare -a PORTS=()
declare -a LOGS=()

cleanup() {
  echo ""
  echo "[cleanup] stopping all teacher workers ..."
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "=============================================="
echo "  G3 Multi-Worker Teacher Smoketest"
echo "=============================================="
echo "RUN_DIR:              ${RUN_DIR}"
echo "WORKER_GPUS_CSV:      ${WORKER_GPUS_CSV} (${WORKER_COUNT} workers)"
echo "BASE_PORT:            ${BASE_PORT}"
echo "MAX_MODEL_LEN:        ${MAX_MODEL_LEN}"
echo "MAX_NUM_SEQS:         ${MAX_NUM_SEQS}"
echo "MAX_BATCHED_TOKENS:   ${MAX_BATCHED_TOKENS}"
echo "GPU_MEMORY_UTIL:      ${GPU_MEMORY_UTIL}"
echo "BENCH_BATCH_SIZE:     ${BENCH_BATCH_SIZE}"
echo "=============================================="
echo ""

# ── Phase 1: Feasibility ─────────────────────────────────────────────
echo "══════════════════════════════════════════════"
echo "  PHASE 1: Feasibility — Launch & Probe"
echo "══════════════════════════════════════════════"

for idx in "${!WORKER_GPUS[@]}"; do
  port=$((BASE_PORT + idx))
  gpu_id="${WORKER_GPUS[$idx]}"
  log="${RUN_DIR}/worker_${idx}.log"

  echo "[worker ${idx}] launching on GPU ${gpu_id}, port ${port}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
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
    > "${log}" 2>&1 &

  PIDS+=("$!")
  PORTS+=("${port}")
  LOGS+=("${log}")
done

echo ""
echo "[phase1] waiting for all ${WORKER_COUNT} workers ..."
for idx in "${!WORKER_GPUS[@]}"; do
  pid="${PIDS[$idx]}"
  port="${PORTS[$idx]}"
  log="${LOGS[$idx]}"
  waited=0
  while true; do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[worker ${idx}] healthy (GPU ${WORKER_GPUS[$idx]}, port ${port})"
      break
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[FAIL] worker ${idx} exited. Log: ${log}"
      echo ""; echo "=== Last 20 lines of ${log} ==="; tail -20 "${log}" || true
      exit 1
    fi
    sleep 3
    waited=$((waited + 3))
    if (( waited >= WAIT_SECONDS )); then
      echo "[FAIL] worker ${idx} health timeout (${WAIT_SECONDS}s). Log: ${log}"
      exit 1
    fi
  done
done

echo ""
echo "[phase1] probing each worker with a single completion ..."
probe_pass=0
for idx in "${!WORKER_GPUS[@]}"; do
  port="${PORTS[$idx]}"
  resp=$(curl -sf -X POST "http://127.0.0.1:${port}/v1/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "{
      \"model\": \"${SERVED_MODEL_NAME}\",
      \"prompt\": \"Question: What is 2+2?\\nAnswer:\",
      \"max_tokens\": 16,
      \"temperature\": 0.0,
      \"n\": 1
    }" 2>&1) || true

  if echo "${resp}" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['text']" 2>/dev/null; then
    echo "[worker ${idx}] probe PASSED"
    probe_pass=$((probe_pass + 1))
  else
    echo "[worker ${idx}] probe FAILED — response: ${resp:0:200}"
  fi
done

echo ""
if (( probe_pass != WORKER_COUNT )); then
  echo "[FAIL] Phase 1: only ${probe_pass}/${WORKER_COUNT} workers passed probe."
  exit 1
fi
echo "[PASS] Phase 1: all ${WORKER_COUNT} workers launched and probed successfully."

# ── Phase 2: High Concurrency Benchmark ──────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  PHASE 2: High Concurrency — Benchmark"
echo "══════════════════════════════════════════════"

API_URLS=""
for port in "${PORTS[@]}"; do
  [[ -n "${API_URLS}" ]] && API_URLS="${API_URLS},"
  API_URLS="${API_URLS}http://127.0.0.1:${port}/v1"
done

BENCH_SCRIPT="${REPO_ROOT}/scripts/smoketests/benchmark_teacher_multiworker.py"
if [[ ! -f "${BENCH_SCRIPT}" ]]; then
  echo "[WARN] benchmark script not found: ${BENCH_SCRIPT}"
  echo "[SKIP] Phase 2 skipped."
  echo ""
  echo "[DONE] G3 smoketest — Phase 1 passed, Phase 2 skipped."
  exit 0
fi

echo "[phase2] running benchmark: batch_size=${BENCH_BATCH_SIZE}, prompts/iter=${BENCH_PROMPTS_PER_ITER}, iters=${BENCH_MEASURE_ITERS}"
echo "[phase2] API URLs: ${API_URLS}"
echo ""

"${STUDENT_PYTHON_BIN}" "${BENCH_SCRIPT}" \
  --teacher-api-base "${API_URLS}" \
  --teacher-model-name "${SERVED_MODEL_NAME}" \
  --teacher-api-key "${API_KEY}" \
  --teacher-remote-batch-size "${BENCH_BATCH_SIZE}" \
  --prompts-per-iter "${BENCH_PROMPTS_PER_ITER}" \
  --warmup-iters 1 \
  --measure-iters "${BENCH_MEASURE_ITERS}" \
  --max-new-tokens "${BENCH_MAX_NEW_TOKENS}" \
  --max-prompt-tokens 256 \
  --tokenizer-path "${MODEL_PATH}" \
  2>&1 | tee "${RUN_DIR}/benchmark.log"

bench_exit=${PIPESTATUS[0]}
echo ""
if (( bench_exit == 0 )); then
  echo "[PASS] Phase 2: benchmark completed successfully."
else
  echo "[FAIL] Phase 2: benchmark exited with code ${bench_exit}."
  exit 1
fi

echo ""
echo "=============================================="
echo "  G3 Multi-Worker Smoketest PASSED"
echo "  Logs: ${RUN_DIR}"
echo "=============================================="
