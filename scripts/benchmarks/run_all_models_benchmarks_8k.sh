#!/usr/bin/env bash
# Sequential launcher for all 4 benchmark-eval variants:
#   raw base -> G1 rebase -> G2 rebase -> G3 rebase
#
# Required:
#   G1_RUN_DIR=/root/outputs/g1_rebase_xxx
#   G2_RUN_DIR=/root/outputs/g2_rebase_xxx
#   G3_RUN_DIR=/root/outputs/g3_rebase_xxx
#
# Optional:
#   RAW_BASE_MODEL_PATH=/path/to/Qwen3.5-0.8B
#   RAW_BASE_RUN_DIR=/root/outputs/raw_base_benchmarks_xxx
#   SKIP_RAW_BASE=true
#   SKIP_G1=true
#   SKIP_G2=true
#   SKIP_G3=true
#
# Shared benchmark env vars are forwarded automatically, e.g.:
#   BENCHMARKS
#   POST_EVAL_MAX_NEW_TOKENS
#   POST_EVAL_PROMPT_MAX_LEN
#   VLLM_TP_SIZE
#   VLLM_MAX_NUM_SEQS
#   VLLM_PROGRESS_BATCH_SIZE
#   VLLM_GPU_MEMORY_UTILIZATION
#   VLLM_ENABLE_PREFIX_CACHING
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
LAUNCH_TS="${LAUNCH_TS:-$(date +%m%d_%H%M)}"

RAW_BASE_SCRIPT="${RAW_BASE_SCRIPT:-${SCRIPT_DIR}/raw_base_benchmarks_8k.sh}"
G1_SCRIPT="${G1_SCRIPT:-${SCRIPT_DIR}/G1_rebase_benchmarks_8k.sh}"
G2_SCRIPT="${G2_SCRIPT:-${SCRIPT_DIR}/G2_rebase_benchmarks_8k.sh}"
G3_SCRIPT="${G3_SCRIPT:-${SCRIPT_DIR}/G3_rebase_benchmarks_8k.sh}"

RAW_BASE_MODEL_PATH="${RAW_BASE_MODEL_PATH:-${1:-}}"
G1_RUN_DIR="${G1_RUN_DIR:-${2:-}}"
G2_RUN_DIR="${G2_RUN_DIR:-${3:-}}"
G3_RUN_DIR="${G3_RUN_DIR:-${4:-}}"

SKIP_RAW_BASE="${SKIP_RAW_BASE:-false}"
SKIP_G1="${SKIP_G1:-false}"
SKIP_G2="${SKIP_G2:-false}"
SKIP_G3="${SKIP_G3:-false}"

LAUNCH_LOG_ROOT="${LAUNCH_LOG_ROOT:-/root/outputs/benchmark_launchers}"
LAUNCH_LOG_PATH="${LAUNCH_LOG_PATH:-${LAUNCH_LOG_ROOT}/run_all_models_benchmarks_8k_${LAUNCH_TS}.log}"
mkdir -p "${LAUNCH_LOG_ROOT}"
exec > >(tee -a "${LAUNCH_LOG_PATH}") 2>&1

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] script not found: ${path}" >&2
    exit 1
  fi
}

require_dir_if_needed() {
  local label="$1"
  local path="$2"
  if [[ -z "${path}" ]]; then
    echo "[ERROR] ${label} is required" >&2
    exit 1
  fi
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] ${label} not found: ${path}" >&2
    exit 1
  fi
}

run_stage() {
  local label="$1"
  shift
  echo ""
  echo "================================================================"
  echo "  START ${label}"
  echo "================================================================"
  echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  "$@"
  echo ""
  echo "  FINISH ${label}"
  echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "================================================================"
}

require_file "${RAW_BASE_SCRIPT}"
require_file "${G1_SCRIPT}"
require_file "${G2_SCRIPT}"
require_file "${G3_SCRIPT}"

if [[ "${SKIP_G1}" != "true" ]]; then
  require_dir_if_needed "G1_RUN_DIR" "${G1_RUN_DIR}"
fi
if [[ "${SKIP_G2}" != "true" ]]; then
  require_dir_if_needed "G2_RUN_DIR" "${G2_RUN_DIR}"
fi
if [[ "${SKIP_G3}" != "true" ]]; then
  require_dir_if_needed "G3_RUN_DIR" "${G3_RUN_DIR}"
fi

RAW_BASE_RUN_DIR="${RAW_BASE_RUN_DIR:-/root/outputs/raw_base_benchmarks_${LAUNCH_TS}}"

echo "================================================================"
echo "  Run All Models Benchmark 8K"
echo "================================================================"
echo "  repo_root:                 ${REPO_ROOT}"
echo "  launch_log_path:           ${LAUNCH_LOG_PATH}"
echo "  benchmarks:                ${BENCHMARKS:-aime24,aime25,amc23,math500,minervamath,olympiadbench}"
echo "  prompt_max_len:            ${POST_EVAL_PROMPT_MAX_LEN:-512}"
echo "  max_new_tokens:            ${POST_EVAL_MAX_NEW_TOKENS:-8192}"
echo "  vllm_tp_size:              ${VLLM_TP_SIZE:-<auto>}"
echo "  vllm_max_num_seqs:         ${VLLM_MAX_NUM_SEQS:-64}"
echo "  vllm_progress_batch_size:  ${VLLM_PROGRESS_BATCH_SIZE:-16}"
echo "  vllm_gpu_mem_util:         ${VLLM_GPU_MEMORY_UTILIZATION:-<default>}"
echo "  skip_raw_base:             ${SKIP_RAW_BASE}"
echo "  skip_g1:                   ${SKIP_G1}"
echo "  skip_g2:                   ${SKIP_G2}"
echo "  skip_g3:                   ${SKIP_G3}"
echo "  raw_base_model_path:       ${RAW_BASE_MODEL_PATH:-<auto-detect in child script>}"
echo "  raw_base_run_dir:          ${RAW_BASE_RUN_DIR}"
echo "  g1_run_dir:                ${G1_RUN_DIR:-<skipped>}"
echo "  g2_run_dir:                ${G2_RUN_DIR:-<skipped>}"
echo "  g3_run_dir:                ${G3_RUN_DIR:-<skipped>}"
echo "================================================================"

if [[ "${SKIP_RAW_BASE}" != "true" ]]; then
  if [[ -n "${RAW_BASE_MODEL_PATH}" ]]; then
    run_stage "RAW_BASE" bash "${RAW_BASE_SCRIPT}" "${RAW_BASE_MODEL_PATH}" "${RAW_BASE_RUN_DIR}"
  else
    run_stage "RAW_BASE" bash "${RAW_BASE_SCRIPT}" "" "${RAW_BASE_RUN_DIR}"
  fi
fi

if [[ "${SKIP_G1}" != "true" ]]; then
  run_stage "G1_REBASE" bash "${G1_SCRIPT}" "${G1_RUN_DIR}"
fi

if [[ "${SKIP_G2}" != "true" ]]; then
  run_stage "G2_REBASE" bash "${G2_SCRIPT}" "${G2_RUN_DIR}"
fi

if [[ "${SKIP_G3}" != "true" ]]; then
  run_stage "G3_REBASE" bash "${G3_SCRIPT}" "${G3_RUN_DIR}"
fi

echo ""
echo "[all-done] launch_log: ${LAUNCH_LOG_PATH}"
echo "[all-done] raw_base_logs: ${RAW_BASE_RUN_DIR}/benchmark_logs"
if [[ -n "${G1_RUN_DIR}" ]]; then
  echo "[all-done] g1_logs: ${G1_RUN_DIR}/benchmark_logs"
fi
if [[ -n "${G2_RUN_DIR}" ]]; then
  echo "[all-done] g2_logs: ${G2_RUN_DIR}/benchmark_logs"
fi
if [[ -n "${G3_RUN_DIR}" ]]; then
  echo "[all-done] g3_logs: ${G3_RUN_DIR}/benchmark_logs"
fi
