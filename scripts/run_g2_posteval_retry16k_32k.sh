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

REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Match-Tuning}"
MODEL_PATH="${MODEL_PATH:-/root/outputs/g2_online_teacher_8gpu_0411_0652/model}"
EVAL_DATA="${EVAL_DATA:-/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl}"

TEACHER_VENV="${TEACHER_VENV:-${REPO_ROOT}/.teacherVenv}"
ANALYSIS_VENV="${ANALYSIS_VENV:-${REPO_ROOT}/.venv}"

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
