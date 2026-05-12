#!/usr/bin/env bash
# Single-pass code-generation eval for one MBPP or HumanEval JSONL dataset.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/diff_dataset/_common.sh
source "${SCRIPT_DIR}/_common.sh"

RUN_DIR="${RUN_DIR:-${1:-}}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: RUN_DIR=/path/to/run EVAL_DATA=/path/to/eval.jsonl CODE_BENCHMARK=mbpp bash scripts/diff_dataset/posteval_code_pass1.sh"
  echo "   or: bash scripts/diff_dataset/posteval_code_pass1.sh /path/to/run"
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/model}"
if [[ -z "${EVAL_DATA:-}" ]]; then
  echo "[ERROR] EVAL_DATA must be set"
  exit 1
fi

infer_code_benchmark() {
  local raw="${1:-}"
  local lower
  lower="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
  case "${lower}" in
    mbpp|*mbpp*) echo "mbpp" ;;
    humaneval|human_eval|*humaneval*|*human_eval*) echo "humaneval" ;;
    *) echo "" ;;
  esac
}

CODE_BENCHMARK="${CODE_BENCHMARK:-$(infer_code_benchmark "${EVAL_TAG:-}")}"
if [[ -z "${CODE_BENCHMARK}" ]]; then
  CODE_BENCHMARK="$(infer_code_benchmark "$(basename "${EVAL_DATA}")")"
fi
if [[ "${CODE_BENCHMARK}" != "mbpp" && "${CODE_BENCHMARK}" != "humaneval" ]]; then
  echo "[ERROR] cannot infer CODE_BENCHMARK from EVAL_TAG='${EVAL_TAG:-}' or EVAL_DATA='${EVAL_DATA}'"
  echo "        set CODE_BENCHMARK=mbpp or CODE_BENCHMARK=humaneval"
  exit 1
fi

_DEFAULT_TEACHER_VENV="/mnt/workspace/venvs/.teacherVenv"
[[ -d "${_DEFAULT_TEACHER_VENV}" ]] || _DEFAULT_TEACHER_VENV="${REPO_ROOT}/.teacherVenv"
TEACHER_VENV="${TEACHER_VENV:-${_DEFAULT_TEACHER_VENV}}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"

_DEFAULT_ANALYSIS_VENV="/mnt/workspace/venvs/.venv"
[[ -d "${_DEFAULT_ANALYSIS_VENV}" ]] || _DEFAULT_ANALYSIS_VENV="${REPO_ROOT}/.venv"
ANALYSIS_VENV="${ANALYSIS_VENV:-${_DEFAULT_ANALYSIS_VENV}}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"
PROGRESS_HELPER="${PROGRESS_HELPER:-${REPO_ROOT}/scripts/supplement/vllm_generate_progress.py}"

MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a _VISIBLE_GPUS <<< "${MODEL_CUDA_VISIBLE_DEVICES}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-${#_VISIBLE_GPUS[@]}}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_PROGRESS_BATCH_SIZE="${VLLM_PROGRESS_BATCH_SIZE:-128}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_SEED="${VLLM_SEED:-1234}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"

CODE_EVAL_MAX_NEW_TOKENS="${CODE_EVAL_MAX_NEW_TOKENS:-1024}"
CODE_EVAL_TEMPERATURE="${CODE_EVAL_TEMPERATURE:-0.0}"
CODE_EVAL_TOP_P="${CODE_EVAL_TOP_P:-1.0}"
CODE_EVAL_REPETITION_PENALTY="${CODE_EVAL_REPETITION_PENALTY:-1.0}"
CODE_EVAL_TIMEOUT_SECONDS="${CODE_EVAL_TIMEOUT_SECONDS:-10}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-}"

EVAL_TAG="${EVAL_TAG:-${CODE_BENCHMARK}_pass1}"
TS="${TS:-$(date +%m%d_%H%M)}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/supplement_logs/${CODE_BENCHMARK}}"
SCRIPT_LOG_PATH="${SCRIPT_LOG_PATH:-${LOG_DIR}/code_pass1_${EVAL_TAG}_${TS}.log}"
OUTPUT_PATH="${OUTPUT_PATH:-${LOG_DIR}/eval_results_${EVAL_TAG}_${TS}.jsonl}"
GEN_LOG_PATH="${GEN_LOG_PATH:-${LOG_DIR}/eval_generate_${EVAL_TAG}_${TS}.log}"
REPORT_JSON="${REPORT_JSON:-${LOG_DIR}/code_eval_report_${EVAL_TAG}_${TS}.json}"
DETAILS_JSONL="${DETAILS_JSONL:-${LOG_DIR}/code_eval_details_${EVAL_TAG}_${TS}.jsonl}"
ANALYSIS_LOG_PATH="${ANALYSIS_LOG_PATH:-${LOG_DIR}/code_eval_${EVAL_TAG}_${TS}.log}"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# Reuse the same vLLM retry / cleanup safeguards as the existing post-eval scripts.
# shellcheck source=scripts/supplement_2rounds/_vllm_runtime.sh
source "${REPO_ROOT}/scripts/supplement_2rounds/_vllm_runtime.sh"

for _bin in "${TEACHER_PYTHON_BIN}" "${ANALYSIS_PYTHON_BIN}"; do
  [[ -x "${_bin}" ]] || { echo "[ERROR] Not executable: ${_bin}"; exit 1; }
done
[[ -d "${RUN_DIR}" ]] || { echo "[ERROR] RUN_DIR not found: ${RUN_DIR}"; exit 1; }
[[ -e "${MODEL_PATH}" ]] || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 1; }
[[ -e "${EVAL_DATA}" ]] || { echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"; exit 1; }
[[ -f "${PROGRESS_HELPER}" ]] || { echo "[ERROR] PROGRESS_HELPER not found: ${PROGRESS_HELPER}"; exit 1; }

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${SCRIPT_LOG_PATH}") 2>&1
cd "${REPO_ROOT}"

echo "========== Code Pass@1 Eval =========="
echo "RUN_DIR:                    ${RUN_DIR}"
echo "MODEL_PATH:                 ${MODEL_PATH}"
echo "CODE_BENCHMARK:             ${CODE_BENCHMARK}"
echo "EVAL_DATA:                  ${EVAL_DATA}"
echo "OUTPUT_PATH:                ${OUTPUT_PATH}"
echo "REPORT_JSON:                ${REPORT_JSON}"
echo "DETAILS_JSONL:              ${DETAILS_JSONL}"
echo "MODEL_CUDA_VISIBLE_DEVICES: ${MODEL_CUDA_VISIBLE_DEVICES}"
echo "VLLM_TP_SIZE:               ${VLLM_TP_SIZE}"
echo "POST_EVAL_MAX_SAMPLES:      ${POST_EVAL_MAX_SAMPLES}"
echo "CODE_EVAL_MAX_NEW_TOKENS:   ${CODE_EVAL_MAX_NEW_TOKENS}"
echo "CODE_EVAL_TEMPERATURE:      ${CODE_EVAL_TEMPERATURE}"
echo "======================================"

vllm_cmd=(
  env "CUDA_VISIBLE_DEVICES=${MODEL_CUDA_VISIBLE_DEVICES}"
  "${TEACHER_PYTHON_BIN}" "${PROGRESS_HELPER}"
  --pretrain "${MODEL_PATH}"
  --dataset "${EVAL_DATA}"
  --input_key question
  --output_path "${OUTPUT_PATH}"
  --prompt_max_len "${POST_EVAL_PROMPT_MAX_LEN}"
  --max_new_tokens "${CODE_EVAL_MAX_NEW_TOKENS}"
  --temperature "${CODE_EVAL_TEMPERATURE}"
  --top_p "${CODE_EVAL_TOP_P}"
  --repetition_penalty "${CODE_EVAL_REPETITION_PENALTY}"
  --max_samples "${POST_EVAL_MAX_SAMPLES}"
  --best_of_n 1
  --tp_size "${VLLM_TP_SIZE}"
  --max_num_seqs "${VLLM_MAX_NUM_SEQS}"
  --progress_batch_size "${VLLM_PROGRESS_BATCH_SIZE}"
  --seed "${VLLM_SEED}"
)
[[ -n "${INPUT_TEMPLATE}" ]] && vllm_cmd+=(--input_template "${INPUT_TEMPLATE}")
[[ "${VLLM_ENABLE_PREFIX_CACHING}" == "true" ]] && vllm_cmd+=(--enable_prefix_caching)
[[ -n "${VLLM_GPU_MEMORY_UTILIZATION}" ]] && vllm_cmd+=(--gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}")

run_vllm_generation_with_retry "code-pass1-${CODE_BENCHMARK}" "${GEN_LOG_PATH}" "${OUTPUT_PATH}" "${EVAL_DATA}" "${vllm_cmd[@]}"

echo ""
echo "===== Executing code benchmark tests ====="
"${ANALYSIS_PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_code_results.py" \
  --benchmark "${CODE_BENCHMARK}" \
  --source_jsonl "${EVAL_DATA}" \
  --results_jsonl "${OUTPUT_PATH}" \
  --report_json "${REPORT_JSON}" \
  --details_jsonl "${DETAILS_JSONL}" \
  --timeout_seconds "${CODE_EVAL_TIMEOUT_SECONDS}" \
  --repo_root "${REPO_ROOT}" \
  --temperature "${CODE_EVAL_TEMPERATURE}" \
  --max_new_tokens "${CODE_EVAL_MAX_NEW_TOKENS}" \
  2>&1 | tee "${ANALYSIS_LOG_PATH}"

echo ""
echo "========== Code pass@1 eval done =========="
echo "Results: ${OUTPUT_PATH}"
echo "Report:  ${REPORT_JSON}"
echo "Details: ${DETAILS_JSONL}"
