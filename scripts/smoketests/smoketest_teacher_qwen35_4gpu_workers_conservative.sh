#!/usr/bin/env bash
set -euo pipefail

# Conservative 4-worker smoke test wrapper for Qwen3.5-27B.
# This version aims to maximize the chance that each single-GPU worker
# can finish KV cache initialization and become healthy.

export WORKER_GPUS_CSV="${WORKER_GPUS_CSV:-4,5,6,7}"
export BASE_PORT="${BASE_PORT:-8204}"

# Key memory-saving defaults:
export GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-128}"

# Keep other defaults explicit for reproducibility.
export MODEL_PATH="${MODEL_PATH:-/mnt/data/models/qwen3.5-27b}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-27b}"
export DTYPE="${DTYPE:-bfloat16}"
export TP_SIZE="${TP_SIZE:-1}"
export WAIT_SECONDS="${WAIT_SECONDS:-600}"

echo "Using conservative 4-worker settings:"
echo "- WORKER_GPUS_CSV=${WORKER_GPUS_CSV}"
echo "- BASE_PORT=${BASE_PORT}"
echo "- GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL}"
echo "- MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "- MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "- MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS}"
echo

exec bash "/root/code/Distributional-Matching-Tuning/scripts/smoketests/smoketest_teacher_qwen35_4gpu_workers.sh"
