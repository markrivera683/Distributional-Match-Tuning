#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  Baseline — pretrained model, NO RL training                    ║
# ║  16k/32k two-round eval, same protocol as G1/G2/G3 post-eval    ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# WHAT THIS SCRIPT DOES:
#   - Runs the two-round vLLM eval (16k first pass → 32k retry on incorrect)
#     directly against MODEL_PATH (the base / pretrained checkpoint).
#   - No Ray, no training, no Critic. Pure inference-only baseline number.
#   - Outputs land under ${RUN_DIR}/supplement_logs/, in the same layout
#     as the G1/G2/G3 runs, so accuracy numbers are directly comparable.
#
# CONTROLLED VARIABLES vs G1/G2/G3:
#   Baseline:  no RL update on actor; MODEL_PATH stays fixed.
#   G1/G2/G3:  same eval protocol applied to a trained checkpoint.
#
# Usage:  bash scripts/run_baseline.sh
# Override any variable via env, e.g.:
#   MODEL_PATH=/mnt/data/models/qwen3.5-0.8b bash scripts/run_baseline.sh
#   POST_EVAL_MAX_SAMPLES=512 bash scripts/run_baseline.sh
#   MODEL_CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_baseline.sh
set -euo pipefail

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "$csv" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"$csv"
}

# ====================================================================
# 1) MODEL / DATA PATHS  (kept identical to G1/G2/G3 defaults so the
#    baseline number drops into the same comparison table)
# ====================================================================
REPO_ROOT="${REPO_ROOT:-/mnt/data/ebft-distribution-new/code}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/gemma-4-E4B/}"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
FALLBACK_LOCAL_DATA="${FALLBACK_LOCAL_DATA:-}"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"

# Venvs live on local ext4 (ossfs2 can't host venv symlinks). See
# scripts/setup_env.sh for the bootstrap that creates and snapshots them.
STUDENT_VENV="${STUDENT_VENV:-/mnt/workspace/venvs/.venv}"
TEACHER_VENV="${TEACHER_VENV:-/mnt/workspace/venvs/.teacherVenv}"

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
ANALYSIS_VENV="${ANALYSIS_VENV:-${STUDENT_VENV}}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"

# ====================================================================
# 2) GPU / vLLM
#    No Ray, no training. All visible GPUs go to the vLLM engine.
# ====================================================================
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"

# ====================================================================
# 3) TWO-ROUND EVAL KNOBS  (identical defaults to scripts/run_G1_rebase.sh)
# ====================================================================
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
FIRST_PASS_MAX_NEW_TOKENS="${FIRST_PASS_MAX_NEW_TOKENS:-16384}"
SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS:-32768}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY:-1.0}"
POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N:-1}"
# vLLM concurrency knobs. See scripts/supplement_2rounds/baseline.sh for the
# full HOL-blocking rationale that motivated raising these from the legacy
# {32, hardcoded-16} defaults to {256, 256}.
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_PROGRESS_BATCH_SIZE="${VLLM_PROGRESS_BATCH_SIZE:-256}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_SEED="${VLLM_SEED:-1234}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-}"

# ====================================================================
# 4) ENV / RUN DIR
# ====================================================================
# HF_HOME is exported above (section 1) with DSW-specific defaults; do not
# redeclare here or the upper value would be silently shadowed if a user
# pre-exported it.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED=1

RUN_NAME="${RUN_NAME:-baseline_$(date +%m%d_%H%M)}"
# Outputs go on persistent OSS so eval artifacts survive container restart
# (matches G1/G2/G3 scripts; previously /root/outputs was rootfs and lost on
# rebuild).
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
mkdir -p "${RUN_DIR}"

# ====================================================================
# 5) SANITY CHECK
# ====================================================================
vllm_gpu_count="$(count_csv_items "${MODEL_CUDA_VISIBLE_DEVICES}")"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-${vllm_gpu_count}}"

if (( vllm_gpu_count == 0 )); then
  echo "[ERROR] MODEL_CUDA_VISIBLE_DEVICES is empty: '${MODEL_CUDA_VISIBLE_DEVICES}'"
  exit 1
fi
if (( VLLM_TP_SIZE < 1 )); then
  echo "[ERROR] VLLM_TP_SIZE must be >= 1, got: ${VLLM_TP_SIZE}"
  exit 1
fi
if (( VLLM_TP_SIZE > vllm_gpu_count )); then
  echo "[ERROR] VLLM_TP_SIZE=${VLLM_TP_SIZE} exceeds visible GPU count=${vllm_gpu_count}"
  echo "        MODEL_CUDA_VISIBLE_DEVICES=${MODEL_CUDA_VISIBLE_DEVICES}"
  exit 1
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

if [[ "${EVAL_DATA}" == "${DEFAULT_EVAL_DATA}" && -n "${FALLBACK_LOCAL_DATA}" && ! -e "${EVAL_DATA}" && -f "${FALLBACK_LOCAL_DATA}" ]]; then
  echo "[WARN] EVAL_DATA default not found, fallback to ${FALLBACK_LOCAL_DATA}"
  EVAL_DATA="${FALLBACK_LOCAL_DATA}"
fi
if [[ "${EVAL_DATA}" == /* && ! -e "${EVAL_DATA}" ]]; then
  echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"
  exit 1
fi

for _bin in "${TEACHER_PYTHON_BIN}" "${ANALYSIS_PYTHON_BIN}"; do
  if [[ ! -x "${_bin}" ]]; then
    echo "[ERROR] python not executable: ${_bin}"
    echo "        teacher venv: ${TEACHER_VENV}"
    echo "        analysis venv: ${ANALYSIS_VENV}"
    echo "        run scripts/setup_env.sh first."
    exit 1
  fi
done

WORKER_SCRIPT="${WORKER_SCRIPT:-${REPO_ROOT}/scripts/supplement_2rounds/baseline.sh}"
if [[ ! -f "${WORKER_SCRIPT}" ]]; then
  echo "[ERROR] worker script not found: ${WORKER_SCRIPT}"
  exit 1
fi

# ====================================================================
# 6) ECHO CONFIG
# ====================================================================
echo "========== Baseline (no training, two-round eval) =========="
echo "RUN_DIR:                      ${RUN_DIR}"
echo "MODEL_PATH:                   ${MODEL_PATH}"
echo "EVAL_DATA:                    ${EVAL_DATA}"
echo "MODEL_CUDA_VISIBLE_DEVICES:   ${MODEL_CUDA_VISIBLE_DEVICES}"
echo "VLLM_TP_SIZE:                 ${VLLM_TP_SIZE}"
echo "VLLM_MAX_NUM_SEQS:            ${VLLM_MAX_NUM_SEQS}"
echo "VLLM_PROGRESS_BATCH_SIZE:     ${VLLM_PROGRESS_BATCH_SIZE}"
echo "POST_EVAL_MAX_SAMPLES:        ${POST_EVAL_MAX_SAMPLES}"
echo "FIRST_PASS_MAX_NEW_TOKENS:    ${FIRST_PASS_MAX_NEW_TOKENS}"
echo "SECOND_PASS_MAX_NEW_TOKENS:   ${SECOND_PASS_MAX_NEW_TOKENS}"
echo "Teacher python:               ${TEACHER_PYTHON_BIN}"
echo "Analysis python:              ${ANALYSIS_PYTHON_BIN}"
echo "Worker:                       ${WORKER_SCRIPT}"
echo "============================================================="

# ====================================================================
# 7) RUN TWO-ROUND EVAL
#    All artifacts (stage1/stage2 jsonl, per-stage analysis json,
#    final merged report, oracle-union stats) land in:
#        ${RUN_DIR}/supplement_logs/
# ====================================================================
RUN_DIR="${RUN_DIR}" \
MODEL_PATH="${MODEL_PATH}" \
EVAL_DATA="${EVAL_DATA}" \
TEACHER_VENV="${TEACHER_VENV}" \
ANALYSIS_VENV="${ANALYSIS_VENV}" \
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN}" \
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN}" \
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES}" \
VLLM_TP_SIZE="${VLLM_TP_SIZE}" \
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}" \
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES}" \
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN}" \
FIRST_PASS_MAX_NEW_TOKENS="${FIRST_PASS_MAX_NEW_TOKENS}" \
SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS}" \
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE}" \
POST_EVAL_TOP_P="${POST_EVAL_TOP_P}" \
POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY}" \
POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N}" \
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
VLLM_PROGRESS_BATCH_SIZE="${VLLM_PROGRESS_BATCH_SIZE}" \
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING}" \
VLLM_SEED="${VLLM_SEED}" \
INPUT_TEMPLATE="${INPUT_TEMPLATE}" \
bash "${WORKER_SCRIPT}" "${RUN_DIR}"

echo "──────────────────────────────────────────────────"
echo "Baseline run completed at $(date)" > "${RUN_DIR}/status.txt"
echo "[done] logs: ${RUN_DIR}/supplement_logs"
