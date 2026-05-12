#!/usr/bin/env bash
# Run MBPP and HumanEval pass@1 code post-eval after one G2/G3 2-node training job.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/diff_dataset/_common.sh
source "${SCRIPT_DIR}/_common.sh"

RUN_DIR="${RUN_DIR:-${1:-}}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: RUN_DIR=/path/to/run bash scripts/diff_dataset/posteval_code_2node.sh"
  echo "   or: bash scripts/diff_dataset/posteval_code_2node.sh /path/to/run"
  exit 1
fi

CODE_POST_EVAL_WORKER="${CODE_POST_EVAL_WORKER:-${SCRIPT_DIR}/posteval_code_pass1.sh}"
if [[ ! -f "${CODE_POST_EVAL_WORKER}" ]]; then
  echo "[ERROR] CODE_POST_EVAL_WORKER not found: ${CODE_POST_EVAL_WORKER}"
  exit 1
fi

prepare_diff_datasets

MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/model}"

echo "========== Code Post-Eval Dataset Loop =========="
echo "RUN_DIR:                ${RUN_DIR}"
echo "MODEL_PATH:             ${MODEL_PATH}"
echo "CODE_POST_EVAL_WORKER:  ${CODE_POST_EVAL_WORKER}"
echo "POST_EVAL_DATASETS:     ${POST_EVAL_DATASETS}"
echo "POSTEVAL_DISPATCH:      ${POSTEVAL_WORKER_DISPATCH:-ssh}"
echo "================================================="

IFS=',' read -r -a _eval_specs <<< "${POST_EVAL_DATASETS}"
for spec in "${_eval_specs[@]}"; do
  name="${spec%%:*}"
  data_path="${spec#*:}"
  if [[ -z "${name}" || -z "${data_path}" || "${name}" == "${data_path}" ]]; then
    echo "[ERROR] invalid POST_EVAL_DATASETS entry: ${spec}"
    exit 1
  fi

  echo ""
  echo "===== post-eval ${name}: ${data_path} ====="
  RUN_DIR="${RUN_DIR}" \
  MODEL_PATH="${MODEL_PATH}" \
  EVAL_DATA="${data_path}" \
  LOG_DIR="${RUN_DIR}/supplement_logs/${name}" \
  EVAL_TAG="${name}_pass1" \
  CODE_BENCHMARK="${name}" \
  POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES}" \
  CODE_EVAL_MAX_NEW_TOKENS="${CODE_EVAL_MAX_NEW_TOKENS}" \
  CODE_EVAL_TEMPERATURE="${CODE_EVAL_TEMPERATURE}" \
  CODE_EVAL_TOP_P="${CODE_EVAL_TOP_P}" \
  CODE_EVAL_REPETITION_PENALTY="${CODE_EVAL_REPETITION_PENALTY}" \
  CODE_EVAL_TIMEOUT_SECONDS="${CODE_EVAL_TIMEOUT_SECONDS}" \
  bash "${CODE_POST_EVAL_WORKER}" "${RUN_DIR}"
done

if [[ "${POSTEVAL_WORKER_DISPATCH:-ssh}" == "rendezvous" ]]; then
  # The per-dataset eval scripts intentionally did not release the DLC worker.
  # Mark completion once both datasets have finished.
  # shellcheck source=scripts/supplement_2rounds/_rendezvous_dlc.sh
  source "${REPO_ROOT}/scripts/supplement_2rounds/_rendezvous_dlc.sh"
  rdv_init_root "${RUN_DIR}"
  rdv_mark_complete
fi

echo ""
echo "========== Code post-eval datasets completed =========="
