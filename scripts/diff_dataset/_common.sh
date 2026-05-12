#!/usr/bin/env bash
# Common defaults for the OpenCodeInstruct / MBPP / HumanEval mirror scripts.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-4B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/mnt/data/models/qwen3.5-27b}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-qwen3.5-27b}"

PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-/mnt/data/ebft-distribution-new/outputs/diff_dataset_prepared}"
TRAIN_SAMPLE_POOL="${TRAIN_SAMPLE_POOL:-100000}"

TRAIN_DATA="${TRAIN_DATA:-${PREPARED_DATA_DIR}/opencodeinstruct_qa_100k.jsonl}"
MBPP_EVAL_DATA="${MBPP_EVAL_DATA:-${PREPARED_DATA_DIR}/mbpp_eval_qa.jsonl}"
HUMANEVAL_EVAL_DATA="${HUMANEVAL_EVAL_DATA:-${PREPARED_DATA_DIR}/humaneval_eval_qa.jsonl}"
POST_EVAL_DATASETS="${POST_EVAL_DATASETS:-mbpp:${MBPP_EVAL_DATA},humaneval:${HUMANEVAL_EVAL_DATA}}"

TARGET_STEPS="${TARGET_STEPS:-500}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE))}"
# Keep ~500 PPO steps by default. The prepared train file is 100k rows; this
# cap controls how many prompts the current recipe consumes in one run.
MAX_SAMPLES="${MAX_SAMPLES:-$((TARGET_STEPS * TRAIN_BATCH_SIZE / N_SAMPLES_PER_PROMPT))}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs/diff_dataset}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
CODE_POST_EVAL_WORKER="${CODE_POST_EVAL_WORKER:-${SCRIPT_DIR}/posteval_code_pass1.sh}"
CODE_EVAL_MAX_NEW_TOKENS="${CODE_EVAL_MAX_NEW_TOKENS:-1024}"
CODE_EVAL_TEMPERATURE="${CODE_EVAL_TEMPERATURE:-0.0}"
CODE_EVAL_TOP_P="${CODE_EVAL_TOP_P:-1.0}"
CODE_EVAL_REPETITION_PENALTY="${CODE_EVAL_REPETITION_PENALTY:-1.0}"
CODE_EVAL_TIMEOUT_SECONDS="${CODE_EVAL_TIMEOUT_SECONDS:-10}"

prepare_diff_datasets() {
  local node_rank="${PET_NODE_RANK:-${RANK:-0}}"
  local world_size="${PET_WORLD_SIZE:-${WORLD_SIZE:-1}}"
  local manifest="${PREPARED_DATA_DIR}/manifest.env"

  if (( world_size > 1 && node_rank > 0 )); then
    echo "[prepare] DLC worker rank=${node_rank}: waiting for prepared datasets at ${PREPARED_DATA_DIR}"
    local waited=0
    local wait_seconds="${PREPARE_DIFF_DATASETS_WAIT_SECONDS:-1800}"
    while [[ ! -s "${manifest}" || ! -s "${TRAIN_DATA}" || ! -s "${MBPP_EVAL_DATA}" || ! -s "${HUMANEVAL_EVAL_DATA}" ]]; do
      sleep 5
      waited=$((waited + 5))
      if (( waited >= wait_seconds )); then
        echo "[ERROR] prepared datasets not ready after ${wait_seconds}s"
        echo "        manifest: ${manifest}"
        echo "        train:    ${TRAIN_DATA}"
        echo "        mbpp:     ${MBPP_EVAL_DATA}"
        echo "        humaneval:${HUMANEVAL_EVAL_DATA}"
        exit 1
      fi
    done
    echo "[prepare] datasets ready after ${waited}s"
    return 0
  fi

  "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/prepare_code_datasets.py" \
    --output-dir "${PREPARED_DATA_DIR}" \
    --train-samples "${TRAIN_SAMPLE_POOL}" \
    ${PREPARE_DIFF_DATASETS_FORCE:+--force}
}

run_eval_dataset_loop() {
  local worker_script="${1:-${CODE_POST_EVAL_WORKER}}"
  local run_dir="$2"
  local model_path="$3"

  IFS=',' read -r -a _eval_specs <<< "${POST_EVAL_DATASETS}"
  for spec in "${_eval_specs[@]}"; do
    local name="${spec%%:*}"
    local data_path="${spec#*:}"
    if [[ -z "${name}" || -z "${data_path}" || "${name}" == "${data_path}" ]]; then
      echo "[ERROR] invalid POST_EVAL_DATASETS entry: ${spec}"
      exit 1
    fi
    echo "===== post-eval ${name}: ${data_path} ====="
    RUN_DIR="${run_dir}" \
    MODEL_PATH="${model_path}" \
    EVAL_DATA="${data_path}" \
    LOG_DIR="${run_dir}/supplement_logs/${name}" \
    EVAL_TAG="${name}_pass1" \
    CODE_BENCHMARK="${name}" \
    POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES}" \
    CODE_EVAL_MAX_NEW_TOKENS="${CODE_EVAL_MAX_NEW_TOKENS}" \
    CODE_EVAL_TEMPERATURE="${CODE_EVAL_TEMPERATURE}" \
    CODE_EVAL_TOP_P="${CODE_EVAL_TOP_P}" \
    CODE_EVAL_REPETITION_PENALTY="${CODE_EVAL_REPETITION_PENALTY}" \
    CODE_EVAL_TIMEOUT_SECONDS="${CODE_EVAL_TIMEOUT_SECONDS}" \
    bash "${worker_script}" "${run_dir}"
  done
}
