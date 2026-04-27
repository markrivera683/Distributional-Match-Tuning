#!/usr/bin/env bash
# 8k benchmark completion suite for a G3 rebase checkpoint.
#
# Usage:
#   RUN_DIR=/root/outputs/g3_rebase_xxx bash scripts/benchmarks/G3_rebase_benchmarks_8k.sh
#   bash scripts/benchmarks/G3_rebase_benchmarks_8k.sh /root/outputs/g3_rebase_xxx
#
# Note:
#   This benchmark suite now uses single-node vLLM inference through
#   scripts/benchmarks/benchmark_eval_common.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SUITE_TS="${SUITE_TS:-$(date +%m%d_%H%M)}"

RUN_DIR="${RUN_DIR:-${1:-}}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: RUN_DIR=/path/to/run bash scripts/benchmarks/G3_rebase_benchmarks_8k.sh" >&2
  echo "   or: bash scripts/benchmarks/G3_rebase_benchmarks_8k.sh /path/to/run" >&2
  exit 1
fi

MODEL_LABEL="${MODEL_LABEL:-g3_rebase}"
SAVE_PATH="${SAVE_PATH:-${RUN_DIR}/model}"
LOAD_MODEL_PATH="${LOAD_MODEL_PATH:-${SAVE_PATH}}"
STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${STUDENT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
BENCHMARK_LOG_ROOT="${BENCHMARK_LOG_ROOT:-${RUN_DIR}/benchmark_logs}"

source "${SCRIPT_DIR}/benchmark_eval_common.sh"
run_benchmark_eval_suite
