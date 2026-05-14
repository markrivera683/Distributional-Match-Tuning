#!/usr/bin/env bash
# Stage 4: OPD pilot on top of the async slime configuration.
# This script expects an already running SGLang teacher server.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_IP="${TEACHER_IP:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
RM_URL="${RM_URL:-http://${TEACHER_IP}:${TEACHER_PORT}/generate}"

export USE_OPD="${USE_OPD:-true}"
export SOURCE_SLIME_ENV="${SOURCE_SLIME_ENV:-false}"
export OPD_TYPE="${OPD_TYPE:-sglang}"
export OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
export USE_EBFT_CUSTOM_RM="${USE_EBFT_CUSTOM_RM:-false}"
export CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-slime.rollout.on_policy_distillation.reward_func}"
export CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-slime.rollout.on_policy_distillation.post_process_rewards}"
export GROUP_RM="${GROUP_RM:-false}"
export RM_URL
export RUN_NAME="${RUN_NAME:-slime_opd_sglang_qwen35_4b_$(date +%m%d_%H%M)}"

bash "${SCRIPT_DIR}/run_slime_async_g1_once.sh"
