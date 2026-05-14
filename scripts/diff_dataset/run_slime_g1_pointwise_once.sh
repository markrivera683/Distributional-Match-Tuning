#!/usr/bin/env bash
# Stage 2: slime G1-like pointwise reward, still synchronous.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="${SLIME_ROOT:-/mnt/data/distribution-matching-slime/code/slime-0.2.4}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-4B}"
MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-${SLIME_ROOT}/slime/scripts/models/qwen3.5-4B.sh}"

export TRAIN_DRIVER="${TRAIN_DRIVER:-train.py}"
export SLIME_ROOT
export SOURCE_SLIME_ENV="${SOURCE_SLIME_ENV:-false}"
export COLOCATE="${COLOCATE:-true}"
export USE_EBFT_CUSTOM_RM="${USE_EBFT_CUSTOM_RM:-true}"
export GROUP_RM="${GROUP_RM:-true}"
export EBFT_RM_MODE="${EBFT_RM_MODE:-pointwise}"
export EBFT_FEATURE_MODEL_PATH="${EBFT_FEATURE_MODEL_PATH:-${MODEL_PATH}}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-false}"
export BALANCE_DATA="${BALANCE_DATA:-false}"
export ENABLE_SLIME_EVAL="${ENABLE_SLIME_EVAL:-false}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export RUN_NAME="${RUN_NAME:-slime_g1_pointwise_qwen35_4b_$(date +%m%d_%H%M)}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs/diff_dataset}"
export LOAD_PATH="${LOAD_PATH:-${OUTPUT_ROOT}/${RUN_NAME}/mcore}"
export SAVE_PATH="${SAVE_PATH:-${LOAD_PATH}}"
export MODEL_PATH
export MODEL_ARGS_SCRIPT

bash "${SCRIPT_DIR}/run_slime_gspo_1node_once.sh"
