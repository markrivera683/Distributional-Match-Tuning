#!/usr/bin/env bash
# Stage 1: synchronous slime engineering baseline.
# No EBFT custom reward and no speed knobs beyond the minimal Megatron/SGLang path.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="${SLIME_ROOT:-/mnt/data/distribution-matching-slime/code/slime-0.2.4}"
MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-${SLIME_ROOT}/slime/scripts/models/qwen3.5-4B.sh}"

export TRAIN_DRIVER="${TRAIN_DRIVER:-train.py}"
export SLIME_ROOT
export SOURCE_SLIME_ENV="${SOURCE_SLIME_ENV:-false}"
export COLOCATE="${COLOCATE:-true}"
export USE_EBFT_CUSTOM_RM="${USE_EBFT_CUSTOM_RM:-false}"
export CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-}"
export RM_TYPE="${RM_TYPE:-deepscaler}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-false}"
export BALANCE_DATA="${BALANCE_DATA:-false}"
export ENABLE_SLIME_EVAL="${ENABLE_SLIME_EVAL:-false}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export RUN_NAME="${RUN_NAME:-slime_baseline_sync_qwen35_4b_$(date +%m%d_%H%M)}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs/diff_dataset}"
export LOAD_PATH="${LOAD_PATH:-${OUTPUT_ROOT}/${RUN_NAME}/mcore}"
export SAVE_PATH="${SAVE_PATH:-${LOAD_PATH}}"
export MODEL_ARGS_SCRIPT

bash "${SCRIPT_DIR}/run_slime_gspo_1node_once.sh"
