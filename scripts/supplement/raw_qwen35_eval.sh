#!/usr/bin/env bash
# Standalone raw Qwen3.5-0.8B Base eval.
# Usage:
#   MODEL_PATH=/mnt/data/teacher_model/models/Qwen3.5-0.8B
#   bash scripts/supplement/raw_qwen35_eval.sh /mnt/data/teacher_model/models/Qwen3.5-0.8B
#   RUN_DIR=/root/outputs/raw_qwen35_eval bash scripts/supplement/raw_qwen35_eval.sh /path/to/Qwen3.5-0.8B
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Match-Tuning}"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"
STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"

SCRIPT_NAME="$(basename "$0" .sh)"
TS="${TS:-$(date +%m%d_%H%M)}"
MODEL_PATH="${MODEL_PATH:-${1:-}}"
RUN_DIR="${RUN_DIR:-${2:-/root/outputs/raw_qwen35_eval_${TS}}}"
EVAL_TAG="${EVAL_TAG:-raw_base}"

MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
POST_EVAL_NPROC="${POST_EVAL_NPROC:-8}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
POST_EVAL_MAX_NEW_TOKENS="${POST_EVAL_MAX_NEW_TOKENS:-8192}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_MICRO_BATCH_SIZE="${POST_EVAL_MICRO_BATCH_SIZE:-128}"
POST_EVAL_MASTER_PORT="${POST_EVAL_MASTER_PORT:-29513}"

LOG_DIR="${LOG_DIR:-${RUN_DIR}/supplement_logs}"
SCRIPT_LOG_PATH="${SCRIPT_LOG_PATH:-${LOG_DIR}/${SCRIPT_NAME}_${EVAL_TAG}_${TS}.log}"
POST_EVAL_OUTPUT_PATH="${POST_EVAL_OUTPUT_PATH:-${LOG_DIR}/eval_results_${EVAL_TAG}_${TS}.jsonl}"
POST_EVAL_LOG_PATH="${POST_EVAL_LOG_PATH:-${LOG_DIR}/eval_${EVAL_TAG}_${TS}.log}"
ANALYSIS_REPORT_PATH="${ANALYSIS_REPORT_PATH:-${LOG_DIR}/eval_analysis_${EVAL_TAG}_${TS}.json}"
ANALYSIS_LOG_PATH="${ANALYSIS_LOG_PATH:-${LOG_DIR}/eval_analysis_${EVAL_TAG}_${TS}.log}"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

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

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "$(dirname "${POST_EVAL_OUTPUT_PATH}")" "$(dirname "${ANALYSIS_REPORT_PATH}")"
exec > >(tee -a "${SCRIPT_LOG_PATH}") 2>&1

if [[ ! -x "${STUDENT_PYTHON_BIN}" ]]; then
  echo "[ERROR] STUDENT_PYTHON_BIN not executable: ${STUDENT_PYTHON_BIN}"
  exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[ERROR] Could not auto-detect raw Qwen3.5-0.8B Base checkpoint."
  echo "        Please pass MODEL_PATH=/path/to/Qwen3.5-0.8B or use the first positional arg."
  exit 1
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -e "${EVAL_DATA}" ]]; then
  echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"
  exit 1
fi

cd "${REPO_ROOT}"

echo "========== Raw Qwen3.5-0.8B Base Eval =========="
echo "RUN_DIR:                       ${RUN_DIR}"
echo "MODEL_PATH:                    ${MODEL_PATH}"
echo "LOAD_MODEL_PATH:               ${MODEL_PATH}"
echo "EVAL_DATA:                     ${EVAL_DATA}"
echo "SCRIPT_LOG_PATH:               ${SCRIPT_LOG_PATH}"
echo "POST_EVAL_LOG_PATH:            ${POST_EVAL_LOG_PATH}"
echo "ANALYSIS_LOG_PATH:             ${ANALYSIS_LOG_PATH}"
echo "MODEL_CUDA_VISIBLE_DEVICES:    ${MODEL_CUDA_VISIBLE_DEVICES}"
echo "POST_EVAL_PROMPT_MAX_LEN:      ${POST_EVAL_PROMPT_MAX_LEN}"
echo "POST_EVAL_MAX_NEW_TOKENS:      ${POST_EVAL_MAX_NEW_TOKENS}"
echo "POST_EVAL_MAX_SAMPLES:         ${POST_EVAL_MAX_SAMPLES}"
echo "OUTPUT_PATH:                   ${POST_EVAL_OUTPUT_PATH}"
echo "=============================================="

echo "[load-model] batch_inference will load raw base weights from: ${MODEL_PATH}"
CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES}" \
"${STUDENT_PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node "${POST_EVAL_NPROC}" --master_port "${POST_EVAL_MASTER_PORT}" \
  -m openrlhf.cli.batch_inference \
  --eval_task generate \
  --pretrain "${MODEL_PATH}" \
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

"${STUDENT_PYTHON_BIN}" "${REPO_ROOT}/scripts/analyze_eval_results.py" \
  --eval_results "${POST_EVAL_OUTPUT_PATH}" \
  --eval_dataset "${EVAL_DATA}" \
  --input_key question --label_key answer \
  --report_path "${ANALYSIS_REPORT_PATH}" \
  2>&1 | tee "${ANALYSIS_LOG_PATH}"

echo "[analysis] Report: ${ANALYSIS_REPORT_PATH}"
echo "[analysis] Log:    ${ANALYSIS_LOG_PATH}"
echo "[script]   Log:    ${SCRIPT_LOG_PATH}"
