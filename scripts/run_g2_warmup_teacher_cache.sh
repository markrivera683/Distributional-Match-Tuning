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
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-512}"

# ====================================================================
# 3. ROBUSTNESS
# ====================================================================
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-180}"
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-3}"
WARMUP_BATCH_SIZE="${WARMUP_BATCH_SIZE:-32}"

# ====================================================================
# 4. DATA
# ====================================================================
REPO_ROOT="${REPO_ROOT:-/root/code/data/Distributional-Match-Tuning}"
TRAIN_DATA="${TRAIN_DATA:-/mnt/data/data/aops/aops_qa_hf_dict}"
INPUT_KEY="${INPUT_KEY:-question}"
SPLIT="${SPLIT:-train}"

# ====================================================================
# 5. CACHE / DATASET DIRECTORIES
# ====================================================================
# SQLite cache must live on a real filesystem (ossfs/FUSE doesn't support SQLite locking)
CACHE_DIR_PREFIX="/root/teacher_cache_n_samples"
# Exported HF dataset can go on ossfs
DATASET_DIR_PREFIX="/mnt/data/data/aops/teacher_dataset_n_samples"

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
# RUN 1: n_samples = 2
# ====================================================================
CACHE_DIR_1="${CACHE_DIR_PREFIX}_2"
mkdir -p "${CACHE_DIR_1}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Teacher Cache Warmup  [n_samples = 2]                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""
echo "  [Teacher API]"
echo "    API base:          ${TEACHER_API_BASE}"
echo "    Model:             ${TEACHER_MODEL}"
echo "    API style:         ${TEACHER_API_STYLE}"
echo ""
echo "  [Generation Params]"
echo "    M (n_samples):     2"
echo "    Temperature:       ${TEACHER_TEMPERATURE}"
echo "    Top-p:             ${TEACHER_TOP_P}"
echo "    Max new tokens:    ${TEACHER_MAX_NEW_TOKENS}"
echo ""
echo "  [Data]"
echo "    Dataset:           ${TRAIN_DATA}"
echo "    Input key:         ${INPUT_KEY}"
echo "    Split:             ${SPLIT}"
echo ""
echo "  [Cache]"
echo "    Cache dir:         ${CACHE_DIR_1}"
echo "    Batch size:        ${WARMUP_BATCH_SIZE}"
echo "────────────────────────────────────────────────────────────────"

cd "${REPO_ROOT}"

python scripts/warmup_teacher_cache.py \
  --prompt_data "${TRAIN_DATA}" \
  --input_key "${INPUT_KEY}" \
  --split "${SPLIT}" \
  --cache_dir "${CACHE_DIR_1}" \
  --teacher_api_base "${TEACHER_API_BASE}" \
  --teacher_model_name "${TEACHER_MODEL}" \
  --teacher_api_key "${TEACHER_API_KEY}" \
  --teacher_api_style "${TEACHER_API_STYLE}" \
  --n_samples 2 \
  --temperature "${TEACHER_TEMPERATURE}" \
  --top_p "${TEACHER_TOP_P}" \
  --max_new_tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --batch_size "${WARMUP_BATCH_SIZE}" \
  --timeout "${TEACHER_TIMEOUT}" \
  --max_retries "${TEACHER_MAX_RETRIES}"

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')  WARMUP FINISHED [n_samples = 2]"
echo "  Cache dir: ${CACHE_DIR_1}"
echo "────────────────────────────────────────────────────────────────"

# ── Export cache to HF dataset ──
DATASET_DIR_1="${DATASET_DIR_PREFIX}_2"
echo ""
echo "  Exporting cache -> HF dataset: ${DATASET_DIR_1}"
python scripts/export_teacher_cache_to_dataset.py \
  --prompt_data "${TRAIN_DATA}" \
  --input_key "${INPUT_KEY}" \
  --split "${SPLIT}" \
  --cache_dir "${CACHE_DIR_1}" \
  --model_name "${TEACHER_MODEL}" \
  --n_samples 2 \
  --temperature "${TEACHER_TEMPERATURE}" \
  --top_p "${TEACHER_TOP_P}" \
  --max_new_tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --output_dir "${DATASET_DIR_1}"
echo "  Export done: ${DATASET_DIR_1}"

# ====================================================================
# RUN 2: n_samples = 4
# ====================================================================
CACHE_DIR_2="${CACHE_DIR_PREFIX}_4"
mkdir -p "${CACHE_DIR_2}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Teacher Cache Warmup  [n_samples = 4]                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""
echo "  [Teacher API]"
echo "    API base:          ${TEACHER_API_BASE}"
echo "    Model:             ${TEACHER_MODEL}"
echo "    API style:         ${TEACHER_API_STYLE}"
echo ""
echo "  [Generation Params]"
echo "    M (n_samples):     4"
echo "    Temperature:       ${TEACHER_TEMPERATURE}"
echo "    Top-p:             ${TEACHER_TOP_P}"
echo "    Max new tokens:    ${TEACHER_MAX_NEW_TOKENS}"
echo ""
echo "  [Data]"
echo "    Dataset:           ${TRAIN_DATA}"
echo "    Input key:         ${INPUT_KEY}"
echo "    Split:             ${SPLIT}"
echo ""
echo "  [Cache]"
echo "    Cache dir:         ${CACHE_DIR_2}"
echo "    Batch size:        ${WARMUP_BATCH_SIZE}"
echo "────────────────────────────────────────────────────────────────"

python scripts/warmup_teacher_cache.py \
  --prompt_data "${TRAIN_DATA}" \
  --input_key "${INPUT_KEY}" \
  --split "${SPLIT}" \
  --cache_dir "${CACHE_DIR_2}" \
  --teacher_api_base "${TEACHER_API_BASE}" \
  --teacher_model_name "${TEACHER_MODEL}" \
  --teacher_api_key "${TEACHER_API_KEY}" \
  --teacher_api_style "${TEACHER_API_STYLE}" \
  --n_samples 4 \
  --temperature "${TEACHER_TEMPERATURE}" \
  --top_p "${TEACHER_TOP_P}" \
  --max_new_tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --batch_size "${WARMUP_BATCH_SIZE}" \
  --timeout "${TEACHER_TIMEOUT}" \
  --max_retries "${TEACHER_MAX_RETRIES}"

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')  WARMUP FINISHED [n_samples = 4]"
echo "  Cache dir: ${CACHE_DIR_2}"
echo "────────────────────────────────────────────────────────────────"

# ── Export cache to HF dataset ──
DATASET_DIR_2="${DATASET_DIR_PREFIX}_4"
echo ""
echo "  Exporting cache -> HF dataset: ${DATASET_DIR_2}"
python scripts/export_teacher_cache_to_dataset.py \
  --prompt_data "${TRAIN_DATA}" \
  --input_key "${INPUT_KEY}" \
  --split "${SPLIT}" \
  --cache_dir "${CACHE_DIR_2}" \
  --model_name "${TEACHER_MODEL}" \
  --n_samples 4 \
  --temperature "${TEACHER_TEMPERATURE}" \
  --top_p "${TEACHER_TOP_P}" \
  --max_new_tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --output_dir "${DATASET_DIR_2}"
echo "  Export done: ${DATASET_DIR_2}"
