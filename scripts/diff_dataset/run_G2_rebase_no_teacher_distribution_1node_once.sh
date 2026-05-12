#!/usr/bin/env bash
# 1-node launcher for G2 no-teacher distribution-only ablation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Force the underlying launcher into its single-node fallback path.
unset PET_NODE_RANK PET_MASTER_ADDR PET_WORLD_SIZE RANK MASTER_ADDR WORLD_SIZE
unset HEAD_NODE WORKER_NODE HEAD_NODE_IP WORKER_NODE_IP WORKER_SSH_HOST

export HEAD_STUDENT_CUDA_VISIBLE_DEVICES="${HEAD_STUDENT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WORKER_STUDENT_CUDA_VISIBLE_DEVICES="${WORKER_STUDENT_CUDA_VISIBLE_DEVICES:-}"
export ACTOR_GPUS="${ACTOR_GPUS:-4}"
export CRITIC_GPUS="${CRITIC_GPUS:-4}"
export REF_GPUS="${REF_GPUS:-${ACTOR_GPUS}}"
export REWARD_GPUS="${REWARD_GPUS:-${CRITIC_GPUS}}"

export RUN_NAME="${RUN_NAME:-diff_g2_no_teacher_distribution_qwen35_4b_1node_$(date +%m%d_%H%M)}"

exec bash "${SCRIPT_DIR}/run_G2_rebase_no_teacher_distribution_2node_once.sh" "$@"
