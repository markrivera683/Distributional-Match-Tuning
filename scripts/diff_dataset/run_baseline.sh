#!/usr/bin/env bash
# Standalone baseline: evaluate Qwen3.5-4B directly on MBPP + HumanEval.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# ── Baseline identity ────────────────────────────────────────────────────────
EXPERIMENT="${EXPERIMENT:-baseline}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-4B}"

# ── Dataset preparation ──────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-/mnt/data/ebft-distribution-new/outputs/diff_dataset_prepared}"
TRAIN_SAMPLE_POOL="${TRAIN_SAMPLE_POOL:-100000}"
TRAIN_DATA="${TRAIN_DATA:-${PREPARED_DATA_DIR}/opencodeinstruct_qa_100k.jsonl}"
MBPP_EVAL_DATA="${MBPP_EVAL_DATA:-${PREPARED_DATA_DIR}/mbpp_eval_qa.jsonl}"
HUMANEVAL_EVAL_DATA="${HUMANEVAL_EVAL_DATA:-${PREPARED_DATA_DIR}/humaneval_eval_qa.jsonl}"
POST_EVAL_DATASETS="${POST_EVAL_DATASETS:-mbpp:${MBPP_EVAL_DATA},humaneval:${HUMANEVAL_EVAL_DATA}}"
PREPARE_DIFF_DATASETS_FORCE="${PREPARE_DIFF_DATASETS_FORCE:-}"

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs/diff_dataset}"
RUN_NAME="${RUN_NAME:-diff_baseline_qwen35_4b_$(date +%m%d_%H%M)}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
TS="${TS:-$(date +%m%d_%H%M)}"

# ── Runtime / environment ────────────────────────────────────────────────────
_DEFAULT_TEACHER_VENV="/mnt/workspace/venvs/.teacherVenv"
[[ -d "${_DEFAULT_TEACHER_VENV}" ]] || _DEFAULT_TEACHER_VENV="${REPO_ROOT}/.teacherVenv"
TEACHER_VENV="${TEACHER_VENV:-${_DEFAULT_TEACHER_VENV}}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"

_DEFAULT_ANALYSIS_VENV="/mnt/workspace/venvs/.venv"
[[ -d "${_DEFAULT_ANALYSIS_VENV}" ]] || _DEFAULT_ANALYSIS_VENV="${REPO_ROOT}/.venv"
ANALYSIS_VENV="${ANALYSIS_VENV:-${_DEFAULT_ANALYSIS_VENV}}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# ── vLLM generation hyperparameters ──────────────────────────────────────────
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a _VISIBLE_GPUS <<< "${MODEL_CUDA_VISIBLE_DEVICES}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-${#_VISIBLE_GPUS[@]}}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_PROGRESS_BATCH_SIZE="${VLLM_PROGRESS_BATCH_SIZE:-128}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_SEED="${VLLM_SEED:-1234}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
PROGRESS_HELPER="${PROGRESS_HELPER:-${REPO_ROOT}/scripts/supplement/vllm_generate_progress.py}"

# ── Code benchmark hyperparameters ───────────────────────────────────────────
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
CODE_EVAL_MAX_NEW_TOKENS="${CODE_EVAL_MAX_NEW_TOKENS:-1024}"
CODE_EVAL_TEMPERATURE="${CODE_EVAL_TEMPERATURE:-0.0}"
CODE_EVAL_TOP_P="${CODE_EVAL_TOP_P:-1.0}"
CODE_EVAL_REPETITION_PENALTY="${CODE_EVAL_REPETITION_PENALTY:-1.0}"
CODE_EVAL_TIMEOUT_SECONDS="${CODE_EVAL_TIMEOUT_SECONDS:-10}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-}"

prepare_diff_datasets() {
  local prepare_cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_code_datasets.py"
    --output-dir "${PREPARED_DATA_DIR}"
    --train-samples "${TRAIN_SAMPLE_POOL}"
  )
  [[ -n "${PREPARE_DIFF_DATASETS_FORCE}" ]] && prepare_cmd+=(--force)
  "${prepare_cmd[@]}"
}

run_code_eval_one_dataset() {
  local benchmark="$1"
  local eval_data="$2"
  local log_dir="${RUN_DIR}/supplement_logs/${benchmark}"
  local eval_tag="${benchmark}_pass1"
  local output_path="${log_dir}/eval_results_${eval_tag}_${TS}.jsonl"
  local gen_log_path="${log_dir}/eval_generate_${eval_tag}_${TS}.log"
  local report_json="${log_dir}/code_eval_report_${eval_tag}_${TS}.json"
  local details_jsonl="${log_dir}/code_eval_details_${eval_tag}_${TS}.jsonl"
  local analysis_log_path="${log_dir}/code_eval_${eval_tag}_${TS}.log"

  mkdir -p "${log_dir}"

  echo
  echo "===== post-eval ${benchmark}: ${eval_data} ====="
  echo "LOG_DIR:                        ${log_dir}"
  echo "OUTPUT_PATH:                    ${output_path}"
  echo "REPORT_JSON:                    ${report_json}"
  echo "DETAILS_JSONL:                  ${details_jsonl}"

  local vllm_cmd=(
    env "CUDA_VISIBLE_DEVICES=${MODEL_CUDA_VISIBLE_DEVICES}"
    "${TEACHER_PYTHON_BIN}" "${PROGRESS_HELPER}"
    --pretrain "${MODEL_PATH}"
    --dataset "${eval_data}"
    --input_key question
    --output_path "${output_path}"
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

  "${vllm_cmd[@]}" 2>&1 | tee "${gen_log_path}"

  "${ANALYSIS_PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_code_results.py" \
    --benchmark "${benchmark}" \
    --source_jsonl "${eval_data}" \
    --results_jsonl "${output_path}" \
    --report_json "${report_json}" \
    --details_jsonl "${details_jsonl}" \
    --timeout_seconds "${CODE_EVAL_TIMEOUT_SECONDS}" \
    --repo_root "${REPO_ROOT}" \
    --temperature "${CODE_EVAL_TEMPERATURE}" \
    --max_new_tokens "${CODE_EVAL_MAX_NEW_TOKENS}" \
    2>&1 | tee "${analysis_log_path}"
}

run_eval_dataset_loop() {
  IFS=',' read -r -a eval_specs <<< "${POST_EVAL_DATASETS}"
  for spec in "${eval_specs[@]}"; do
    local name="${spec%%:*}"
    local data_path="${spec#*:}"
    if [[ -z "${name}" || -z "${data_path}" || "${name}" == "${data_path}" ]]; then
      echo "[ERROR] invalid POST_EVAL_DATASETS entry: ${spec}"
      exit 1
    fi
    run_code_eval_one_dataset "${name}" "${data_path}"
  done
}

mkdir -p "${RUN_DIR}"

echo "========== Diff-Dataset Baseline =========="
echo "[recipe]"
echo "EXPERIMENT:                     ${EXPERIMENT}"
echo "DESCRIPTION:                    evaluate the base model directly; no RL/SFT training"
echo "TRAINING_ENABLED:               false"
echo "TEACHER_ENABLED:                false"
echo "EBFT_CF_L1OO:                   disabled"
echo "EMA:                            disabled"
echo "FEATURE_ADAPTER:                disabled"
echo "FEATURE_NETWORK_UNFREEZE:       disabled"
echo "CRITIC_TRAINING:                disabled"
echo "ACTOR_TRAINING:                 disabled"
echo "CLASSIFIER_LOSS:                disabled"
echo "DIRECT_DISCREPANCY_LOSS:        disabled"
echo "DIVERSITY_ALIGNMENT_REWARD:     disabled"
echo "BEST_OF_N:                      1"
echo
echo "[paths]"
echo "REPO_ROOT:                      ${REPO_ROOT}"
echo "RUN_NAME:                       ${RUN_NAME}"
echo "RUN_DIR:                        ${RUN_DIR}"
echo "OUTPUT_ROOT:                    ${OUTPUT_ROOT}"
echo "MODEL_PATH:                     ${MODEL_PATH}"
echo "PREPARED_DATA_DIR:              ${PREPARED_DATA_DIR}"
echo
echo "[dataset preparation]"
echo "TRAIN_SAMPLE_POOL:              ${TRAIN_SAMPLE_POOL}"
echo "TRAIN_DATA:                     ${TRAIN_DATA}"
echo "MBPP_EVAL_DATA:                 ${MBPP_EVAL_DATA}"
echo "HUMANEVAL_EVAL_DATA:            ${HUMANEVAL_EVAL_DATA}"
echo "POST_EVAL_DATASETS:             ${POST_EVAL_DATASETS}"
echo "POST_EVAL_MAX_SAMPLES:          ${POST_EVAL_MAX_SAMPLES}"
echo "NOTE:                           TRAIN_DATA is prepared for comparability but not trained on"
echo
echo "[runtime]"
echo "PYTHON_BIN:                     ${PYTHON_BIN}"
echo "TEACHER_VENV:                   ${TEACHER_VENV}"
echo "TEACHER_PYTHON_BIN:             ${TEACHER_PYTHON_BIN}"
echo "ANALYSIS_VENV:                  ${ANALYSIS_VENV}"
echo "ANALYSIS_PYTHON_BIN:            ${ANALYSIS_PYTHON_BIN}"
echo "HF_HOME:                        ${HF_HOME}"
echo "HF_HUB_OFFLINE:                 ${HF_HUB_OFFLINE}"
echo "HF_DATASETS_OFFLINE:            ${HF_DATASETS_OFFLINE}"
echo "HF_HUB_DISABLE_XET:             ${HF_HUB_DISABLE_XET}"
echo "VLLM_WORKER_MULTIPROC_METHOD:   ${VLLM_WORKER_MULTIPROC_METHOD}"
echo
echo "[vllm]"
echo "MODEL_CUDA_VISIBLE_DEVICES:     ${MODEL_CUDA_VISIBLE_DEVICES}"
echo "VLLM_TP_SIZE:                   ${VLLM_TP_SIZE}"
echo "VLLM_MAX_NUM_SEQS:              ${VLLM_MAX_NUM_SEQS}"
echo "VLLM_PROGRESS_BATCH_SIZE:       ${VLLM_PROGRESS_BATCH_SIZE}"
echo "VLLM_ENABLE_PREFIX_CACHING:     ${VLLM_ENABLE_PREFIX_CACHING}"
echo "VLLM_SEED:                      ${VLLM_SEED}"
echo "VLLM_GPU_MEMORY_UTILIZATION:    ${VLLM_GPU_MEMORY_UTILIZATION:-unset}"
echo "PROGRESS_HELPER:                ${PROGRESS_HELPER}"
echo
echo "[code generation eval]"
echo "POST_EVAL_PROMPT_MAX_LEN:       ${POST_EVAL_PROMPT_MAX_LEN}"
echo "CODE_EVAL_MAX_NEW_TOKENS:       ${CODE_EVAL_MAX_NEW_TOKENS}"
echo "CODE_EVAL_TEMPERATURE:          ${CODE_EVAL_TEMPERATURE}"
echo "CODE_EVAL_TOP_P:                ${CODE_EVAL_TOP_P}"
echo "CODE_EVAL_REPETITION_PENALTY:   ${CODE_EVAL_REPETITION_PENALTY}"
echo "CODE_EVAL_TIMEOUT_SECONDS:      ${CODE_EVAL_TIMEOUT_SECONDS}"
echo "INPUT_TEMPLATE:                 ${INPUT_TEMPLATE:-unset}"
echo "==========================================="

for required_path in "${MODEL_PATH}" "${PROGRESS_HELPER}"; do
  [[ -e "${required_path}" ]] || { echo "[ERROR] not found: ${required_path}"; exit 1; }
done
for required_bin in "${TEACHER_PYTHON_BIN}" "${ANALYSIS_PYTHON_BIN}"; do
  [[ -x "${required_bin}" ]] || { echo "[ERROR] not executable: ${required_bin}"; exit 1; }
done

prepare_diff_datasets
run_eval_dataset_loop

echo "Diff-dataset baseline completed at $(date)" > "${RUN_DIR}/status.txt"
echo "[done] logs: ${RUN_DIR}/supplement_logs"
