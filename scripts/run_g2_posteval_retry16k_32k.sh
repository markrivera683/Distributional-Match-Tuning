#!/usr/bin/env bash
# Post-eval for g2_online_teacher_8gpu_0411_0652.
# Two-stage vLLM eval: 16k first pass, then 32k retry on incorrect prompts.
# Uses the local .teacherVenv for vLLM generation and .venv for analysis.
#
# Usage:
#   bash scripts/run_g2_posteval_retry16k_32k.sh
#
# Wait for the current g3train tmux to finish before running this.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/data/ebft-distribution-new/code}"
# NOTE: MODEL_PATH below is the original training-run checkpoint location
# from the old environment; override via env when re-running on a new
# machine, e.g. MODEL_PATH=/mnt/data/.../some_run/model bash scripts/...
MODEL_PATH="${MODEL_PATH:-/root/outputs/g2_online_teacher_8gpu_0411_0652/model}"
EVAL_DATA="${EVAL_DATA:-/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl}"

# Venvs live on local ext4 (ossfs2 can't host venv symlinks). See
# scripts/setup_env.sh for the bootstrap that creates and snapshots them.
TEACHER_VENV="${TEACHER_VENV:-/mnt/workspace/venvs/.teacherVenv}"
ANALYSIS_VENV="${ANALYSIS_VENV:-/mnt/workspace/venvs/.venv}"

# HF blobs go on persistent OSS (model weights survive container restart;
# downloads are tmp+rename, OSS-safe).
export HF_HOME="${HF_HOME:-/mnt/data/ebft-distribution-new/caches/hf}"
# Compile caches MUST be on local ext4: ossfs2 rejects "seek + write into
# existing file" with EINVAL, which fuse mis-reports as 'No space left on
# device'. That kills g++/nvcc when emitting .o (FusedAdam, fused_adan,
# ...) and triton when emitting .cubin/.so. Cost of being on local ext4:
# ~30-60s recompile after a container restart.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/workspace/.torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/workspace/.triton_cache}"

# Reduce CUDA OOM under tight memory budgets. RLHF batches reshape every
# PPO step (rollout vs train, variable seq lens), so PyTorch's default
# fixed-size segments fragment fast. expandable_segments lets the
# allocator grow segments on demand and typically frees 1-2 GiB of
# headroom on an 80GB A100. PyTorch suggests this in the OOM message.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
FIRST_PASS_MAX_NEW_TOKENS="${FIRST_PASS_MAX_NEW_TOKENS:-16384}"
SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS:-32768}"

TS="$(date +%m%d_%H%M)"
RUN_DIR="${RUN_DIR:-/root/outputs/g2_0652_posteval_retry16k_to_32k_${TS}}"

export REPO_ROOT
export MODEL_PATH
export EVAL_DATA
export TEACHER_VENV
export TEACHER_PYTHON_BIN="${TEACHER_VENV}/bin/python"
export ANALYSIS_VENV
export ANALYSIS_PYTHON_BIN="${ANALYSIS_VENV}/bin/python"
export MODEL_CUDA_VISIBLE_DEVICES
export VLLM_TP_SIZE
export FIRST_PASS_MAX_NEW_TOKENS
export SECOND_PASS_MAX_NEW_TOKENS
export RUN_DIR
export EVAL_TAG="g2_0652_retry16k_to_32k"
export CURRENT_PROGRESS_POINTER="${REPO_ROOT}/.g2_posteval_current.env"

exec bash "${REPO_ROOT}/scripts/dlc_eval/dlc_baseline_eval.sh"
