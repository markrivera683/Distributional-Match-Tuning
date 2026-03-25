#!/usr/bin/env bash
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Teacher Cache Warmup                                           ║
# ║  Pre-fill SQLite cache with remote teacher completions           ║
# ║  No GPU / Ray required — CPU + network only                     ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Populates the teacher completion cache so that the subsequent
# training run (run_g2_8gpu_remote_teacher.sh) achieves 100% cache hit
# and incurs zero network latency during training.
#
# IMPORTANT: all teacher generation parameters below MUST match the
# values used in the training script, otherwise the cache keys will
# differ and training will still call the API.
#
# Usage:
#   bash scripts/run_warmup_teacher_cache.sh
#
# Override any variable via env, e.g.:
#   TEACHER_API_BASE=http://10.0.0.5:8000/v1 bash scripts/run_warmup_teacher_cache.sh

# ====================================================================
# 1. REMOTE TEACHER ENDPOINT  (must match training script)
# ====================================================================
TEACHER_API_BASE="${TEACHER_API_BASE:-http://172.17.0.26:8000/v1}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen-122b}"
TEACHER_API_KEY="${TEACHER_API_KEY:-teacher-local}"
TEACHER_API_STYLE="${TEACHER_API_STYLE:-chat_completions}"

# ====================================================================
# 2. TEACHER GENERATION PARAMS  (must match training script)
# ====================================================================
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-2}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-512}"

# ====================================================================
# 3. ROBUSTNESS
# ====================================================================
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-180}"
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-3}"
WARMUP_BATCH_SIZE="${WARMUP_BATCH_SIZE:-16}"

# ====================================================================
# 4. DATA
# ====================================================================
REPO_ROOT="${REPO_ROOT:-/root/code/data/Distributional-Match-Tuning}"
TRAIN_DATA="${TRAIN_DATA:-/mnt/data/data/aops/aops_qa_hf_dict}"
INPUT_KEY="${INPUT_KEY:-question}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-46000}"                       # 0 = all rows; match training MAX_SAMPLES

# ====================================================================
# 5. CACHE DIRECTORY  (training script must use the same path)
# ====================================================================
CACHE_DIR="${CACHE_DIR:-/root/outputs/teacher_cache_shared}"

# ====================================================================
# ENVIRONMENT
# ====================================================================
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/.cache/huggingface/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1

# ====================================================================
# RUN
# ====================================================================
mkdir -p "${CACHE_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Teacher Cache Warmup                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""
echo "  [Teacher API]"
echo "    API base:          ${TEACHER_API_BASE}"
echo "    Model:             ${TEACHER_MODEL}"
echo "    API style:         ${TEACHER_API_STYLE}"
echo ""
echo "  [Generation Params]"
echo "    M (n_samples):     ${CF_TEACHER_N_SAMPLES}"
echo "    Temperature:       ${TEACHER_TEMPERATURE}"
echo "    Top-p:             ${TEACHER_TOP_P}"
echo "    Max new tokens:    ${TEACHER_MAX_NEW_TOKENS}"
echo ""
echo "  [Data]"
echo "    Dataset:           ${TRAIN_DATA}"
echo "    Input key:         ${INPUT_KEY}"
echo "    Split:             ${SPLIT}"
echo "    Max samples:       ${MAX_SAMPLES}"
echo ""
echo "  [Cache]"
echo "    Cache dir:         ${CACHE_DIR}"
echo "    Batch size:        ${WARMUP_BATCH_SIZE}"
echo "────────────────────────────────────────────────────────────────"

cd "${REPO_ROOT}"

python scripts/warmup_teacher_cache.py \
  --prompt_data "${TRAIN_DATA}" \
  --input_key "${INPUT_KEY}" \
  --split "${SPLIT}" \
  --cache_dir "${CACHE_DIR}" \
  --teacher_api_base "${TEACHER_API_BASE}" \
  --teacher_model_name "${TEACHER_MODEL}" \
  --teacher_api_key "${TEACHER_API_KEY}" \
  --teacher_api_style "${TEACHER_API_STYLE}" \
  --n_samples "${CF_TEACHER_N_SAMPLES}" \
  --temperature "${TEACHER_TEMPERATURE}" \
  --top_p "${TEACHER_TOP_P}" \
  --max_new_tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --max_samples "${MAX_SAMPLES}" \
  --batch_size "${WARMUP_BATCH_SIZE}" \
  --timeout "${TEACHER_TIMEOUT}" \
  --max_retries "${TEACHER_MAX_RETRIES}"

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')  WARMUP FINISHED"
echo "  Cache dir: ${CACHE_DIR}"
echo "────────────────────────────────────────────────────────────────"
