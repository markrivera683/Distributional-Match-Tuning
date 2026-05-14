#!/usr/bin/env bash
# Convert a slime/Megatron checkpoint back to HF and evaluate it with the existing code benchmark harness.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MCORE_CHECKPOINT="${MCORE_CHECKPOINT:-/mnt/data/ebft-distribution-new/outputs/diff_dataset/slime_qwen35_4b_run}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/data/models/Qwen3.5-4B}"
HF_OUTPUT="${HF_OUTPUT:-${MCORE_CHECKPOINT%/}_hf}"
CKPT_STEP="${CKPT_STEP:-}"
CONVERT_TO_HF="${CONVERT_TO_HF:-true}"

RUN_NAME="${RUN_NAME:-slime_posteval_$(date +%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs/diff_dataset/code_eval_repeats}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
SLIME_MODEL_LABEL="${SLIME_MODEL_LABEL:-slime_qwen35_4b}"

if [[ "${CONVERT_TO_HF}" == "true" ]]; then
  MCORE_CHECKPOINT="${MCORE_CHECKPOINT}" \
  HF_CHECKPOINT="${HF_CHECKPOINT}" \
  HF_OUTPUT="${HF_OUTPUT}" \
  CKPT_STEP="${CKPT_STEP}" \
  bash "${SCRIPT_DIR}/convert_slime_checkpoint.sh" mcore_to_hf "${MCORE_CHECKPOINT}"
fi

if [[ ! -f "${HF_OUTPUT}/config.json" || ! -f "${HF_OUTPUT}/tokenizer_config.json" ]]; then
  echo "[ERROR] HF_OUTPUT is not a valid HF model directory: ${HF_OUTPUT}" >&2
  echo "        Expected config.json and tokenizer_config.json." >&2
  exit 1
fi

if [[ "${EVAL_ONLY_SLIME:-false}" == "true" ]]; then
  ONLY_MODEL_SPECS_VALUE="${SLIME_MODEL_LABEL}|${HF_OUTPUT}"
else
  ONLY_MODEL_SPECS_VALUE="${ONLY_MODEL_SPECS:-}"
fi

SLIME_MODEL_PATH="${HF_OUTPUT}" \
SLIME_MODEL_LABEL="${SLIME_MODEL_LABEL}" \
ONLY_MODEL_SPECS="${ONLY_MODEL_SPECS_VALUE}" \
RUN_DIR="${RUN_DIR}" \
bash "${SCRIPT_DIR}/run_code_eval_pass16_once_baseline_g1_g2_g3.sh"
