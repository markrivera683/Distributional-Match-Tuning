#!/usr/bin/env bash
# 8k benchmark completion suite for the raw Qwen3.5-0.8B base model.
#
# Usage:
#   MODEL_PATH=/path/to/model bash scripts/benchmarks/raw_base_benchmarks_8k.sh
#   bash scripts/benchmarks/raw_base_benchmarks_8k.sh /path/to/model
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SUITE_TS="${SUITE_TS:-$(date +%m%d_%H%M)}"

MODEL_PATH="${MODEL_PATH:-${1:-}}"
RUN_DIR="${RUN_DIR:-${2:-/root/outputs/raw_base_benchmarks_${SUITE_TS}}}"

if [[ -z "${MODEL_PATH}" ]]; then
  for _candidate in \
    "/mnt/data/models/Qwen3.5-0.8B" \
    "/mnt/data/models/qwen3.5-0.8b" \
    "/mnt/data/teacher_model/models/Qwen3.5-0.8B" \
    "/mnt/data/teacher_model/models/qwen3.5-0.8b"
  do
    if [[ -e "${_candidate}" ]]; then
      MODEL_PATH="${_candidate}"
      break
    fi
  done
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[ERROR] Could not auto-detect raw base checkpoint." >&2
  echo "        Pass MODEL_PATH=/path/to/model or use the first positional arg." >&2
  exit 1
fi

MODEL_LABEL="${MODEL_LABEL:-raw_base}"
LOAD_MODEL_PATH="${LOAD_MODEL_PATH:-${MODEL_PATH}}"
STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
BENCHMARK_LOG_ROOT="${BENCHMARK_LOG_ROOT:-${RUN_DIR}/benchmark_logs}"

source "${SCRIPT_DIR}/benchmark_eval_common.sh"
run_benchmark_eval_suite
