#!/usr/bin/env bash
# Standalone teacher-model eval via vLLM.
# Designed to be directly comparable with the four supplement eval scripts:
# raw base / G1 / G2 / G3.
# Usage:
#   MODEL_PATH=/mnt/data/models/qwen3.5-27b \
#   bash scripts/supplement/teacher_qwen35_vllm_eval.sh
#   RUN_DIR=/root/outputs/teacher_qwen35_eval_run \
#   bash scripts/supplement/teacher_qwen35_vllm_eval.sh /mnt/data/models/qwen3.5-27b
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Match-Tuning}"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"

TEACHER_VENV="${TEACHER_VENV:-${REPO_ROOT}/.teacherVenv}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
ANALYSIS_VENV="${ANALYSIS_VENV:-${REPO_ROOT}/.venv}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"

SCRIPT_NAME="$(basename "$0" .sh)"
TS="${TS:-$(date +%m%d_%H%M)}"
MODEL_PATH="${MODEL_PATH:-${1:-}}"
RUN_DIR="${RUN_DIR:-${2:-/root/outputs/teacher_qwen35_eval_run}}"
EVAL_TAG="${EVAL_TAG:-teacher_vllm}"

MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
POST_EVAL_MAX_NEW_TOKENS="${POST_EVAL_MAX_NEW_TOKENS:-1536}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY:-1.0}"
POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N:-1}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_SEED="${VLLM_SEED:-1234}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-}"

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
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

if [[ -z "${MODEL_PATH}" ]]; then
  for _candidate in \
    "/mnt/data/models/qwen3.5-27b" \
    "/mnt/data/models/Qwen3.5-27B" \
    "/mnt/data/teacher_model/models/qwen3.5-27b" \
    "/mnt/data/teacher_model/models/Qwen3.5-27B"
  do
    if [[ -e "${_candidate}" ]]; then
      MODEL_PATH="${_candidate}"
      break
    fi
  done
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "$(dirname "${POST_EVAL_OUTPUT_PATH}")" "$(dirname "${ANALYSIS_REPORT_PATH}")"
exec > >(tee -a "${SCRIPT_LOG_PATH}") 2>&1

if [[ ! -x "${TEACHER_PYTHON_BIN}" ]]; then
  echo "[ERROR] TEACHER_PYTHON_BIN not executable: ${TEACHER_PYTHON_BIN}"
  exit 1
fi

if [[ ! -x "${ANALYSIS_PYTHON_BIN}" ]]; then
  echo "[ERROR] ANALYSIS_PYTHON_BIN not executable: ${ANALYSIS_PYTHON_BIN}"
  exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[ERROR] Could not auto-detect Qwen3.5-27B teacher checkpoint."
  echo "        Please pass MODEL_PATH=/path/to/qwen3.5-27b or use the first positional arg."
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

IFS=',' read -r -a _VISIBLE_GPUS <<< "${MODEL_CUDA_VISIBLE_DEVICES}"
if (( ${#_VISIBLE_GPUS[@]} == 0 )); then
  echo "[ERROR] MODEL_CUDA_VISIBLE_DEVICES is empty"
  exit 1
fi

DEFAULT_VLLM_TP_SIZE="${#_VISIBLE_GPUS[@]}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-${DEFAULT_VLLM_TP_SIZE}}"
if (( VLLM_TP_SIZE < 1 )); then
  echo "[ERROR] VLLM_TP_SIZE must be >= 1, got: ${VLLM_TP_SIZE}"
  exit 1
fi
if (( VLLM_TP_SIZE > ${#_VISIBLE_GPUS[@]} )); then
  echo "[ERROR] VLLM_TP_SIZE=${VLLM_TP_SIZE} exceeds visible GPU count=${#_VISIBLE_GPUS[@]}"
  exit 1
fi

cd "${REPO_ROOT}"

echo "========== Teacher Qwen3.5-27B vLLM Eval =========="
echo "RUN_DIR:                         ${RUN_DIR}"
echo "MODEL_PATH:                      ${MODEL_PATH}"
echo "LOAD_MODEL_PATH:                 ${MODEL_PATH}"
echo "EVAL_DATA:                       ${EVAL_DATA}"
echo "SCRIPT_LOG_PATH:                 ${SCRIPT_LOG_PATH}"
echo "POST_EVAL_LOG_PATH:              ${POST_EVAL_LOG_PATH}"
echo "ANALYSIS_LOG_PATH:               ${ANALYSIS_LOG_PATH}"
echo "MODEL_CUDA_VISIBLE_DEVICES:      ${MODEL_CUDA_VISIBLE_DEVICES}"
echo "VLLM_TP_SIZE:                    ${VLLM_TP_SIZE}"
echo "VLLM_MAX_NUM_SEQS:               ${VLLM_MAX_NUM_SEQS}"
echo "VLLM_ENABLE_PREFIX_CACHING:      ${VLLM_ENABLE_PREFIX_CACHING}"
echo "POST_EVAL_PROMPT_MAX_LEN:        ${POST_EVAL_PROMPT_MAX_LEN}"
echo "POST_EVAL_MAX_NEW_TOKENS:        ${POST_EVAL_MAX_NEW_TOKENS}"
echo "POST_EVAL_MAX_SAMPLES:           ${POST_EVAL_MAX_SAMPLES}"
echo "POST_EVAL_BEST_OF_N:             ${POST_EVAL_BEST_OF_N}"
echo "OUTPUT_PATH:                     ${POST_EVAL_OUTPUT_PATH}"
echo "==================================================="

echo "[load-model] standalone vLLM helper will load teacher weights from: ${MODEL_PATH}"
VLLM_CMD=(
  "${TEACHER_PYTHON_BIN}" "${REPO_ROOT}/scripts/supplement/teacher_vllm_generate.py"
  --pretrain "${MODEL_PATH}"
  --dataset "${EVAL_DATA}"
  --input_key question
  --output_path "${POST_EVAL_OUTPUT_PATH}"
  --prompt_max_len "${POST_EVAL_PROMPT_MAX_LEN}"
  --max_new_tokens "${POST_EVAL_MAX_NEW_TOKENS}"
  --temperature "${POST_EVAL_TEMPERATURE}"
  --top_p "${POST_EVAL_TOP_P}"
  --repetition_penalty "${POST_EVAL_REPETITION_PENALTY}"
  --max_samples "${POST_EVAL_MAX_SAMPLES}"
  --best_of_n "${POST_EVAL_BEST_OF_N}"
  --tp_size "${VLLM_TP_SIZE}"
  --max_num_seqs "${VLLM_MAX_NUM_SEQS}"
  --seed "${VLLM_SEED}"
)

if [[ -n "${INPUT_TEMPLATE}" ]]; then
  VLLM_CMD+=(--input_template "${INPUT_TEMPLATE}")
fi

if [[ "${VLLM_ENABLE_PREFIX_CACHING}" == "true" ]]; then
  VLLM_CMD+=(--enable_prefix_caching)
fi

CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES}" \
"${VLLM_CMD[@]}" \
  2>&1 | tee "${POST_EVAL_LOG_PATH}"

echo "[post-eval] Saved: ${POST_EVAL_OUTPUT_PATH}"
echo "[post-eval] Log:   ${POST_EVAL_LOG_PATH}"

"${ANALYSIS_PYTHON_BIN}" "${REPO_ROOT}/scripts/analyze_eval_results.py" \
  --eval_results "${POST_EVAL_OUTPUT_PATH}" \
  --eval_dataset "${EVAL_DATA}" \
  --input_key question --label_key answer \
  --report_path "${ANALYSIS_REPORT_PATH}" \
  2>&1 | tee "${ANALYSIS_LOG_PATH}"

echo "[analysis] Report: ${ANALYSIS_REPORT_PATH}"
echo "[analysis] Log:    ${ANALYSIS_LOG_PATH}"
echo "[script]   Log:    ${SCRIPT_LOG_PATH}"
