#!/usr/bin/env bash
# Single-node 8-GPU full-epoch launcher for the G2 rebase recipe.
#
# This is a thin wrapper around run_G2_rebase_2node_once.sh. The underlying
# launcher already has a single-node path; this wrapper pins that mode and
# changes the default "once" sample cap to a full train-split epoch.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Force the underlying launcher into its local single-node fallback path.
unset PET_NODE_RANK PET_MASTER_ADDR PET_WORLD_SIZE RANK MASTER_ADDR WORLD_SIZE
unset HEAD_NODE WORKER_NODE HEAD_NODE_IP WORKER_NODE_IP WORKER_SSH_HOST

# One 8-GPU node layout for G2 with a local 27B teacher:
#   teacher vLLM workers: GPUs 0-5
#   student actor/critic: GPUs 6-7
# Override these env vars if you provide a different local/external teacher
# layout, but avoid overlapping teacher and student GPUs.
export HEAD_TEACHER_CUDA_VISIBLE_DEVICES="${HEAD_TEACHER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export HEAD_STUDENT_CUDA_VISIBLE_DEVICES="${HEAD_STUDENT_CUDA_VISIBLE_DEVICES:-6,7}"
export WORKER_TEACHER_CUDA_VISIBLE_DEVICES="${WORKER_TEACHER_CUDA_VISIBLE_DEVICES:-}"
export WORKER_STUDENT_CUDA_VISIBLE_DEVICES="${WORKER_STUDENT_CUDA_VISIBLE_DEVICES:-}"

export ACTOR_GPUS="${ACTOR_GPUS:-1}"
export CRITIC_GPUS="${CRITIC_GPUS:-1}"
export REF_GPUS="${REF_GPUS:-${ACTOR_GPUS}}"
export REWARD_GPUS="${REWARD_GPUS:-${CRITIC_GPUS}}"

# Full epoch over TRAIN_DATA: train_ebft_ray treats -1 as no max-sample cap.
export NUM_EPISODES="${NUM_EPISODES:-1}"
export MAX_EPOCHS="${MAX_EPOCHS:-1}"
export MAX_SAMPLES="${MAX_SAMPLES:--1}"
export TARGET_STEPS="${TARGET_STEPS:-0}"

export RUN_NAME="${RUN_NAME:-diff_g2_qwen35_4b_1node_full_epoch_$(date +%m%d_%H%M)}"

exec bash "${SCRIPT_DIR}/run_G2_rebase_2node_once.sh" "$@"
