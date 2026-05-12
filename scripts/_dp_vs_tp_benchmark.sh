#!/usr/bin/env bash
# DP=8/TP=1 vs DP=1/TP=8 wall-clock benchmark for the same vLLM workload.
#
# Why: a 4B model under TP=8 was empirically running at MBW=25% / power=200W (50% TDP),
# suggesting NCCL all-reduce overhead dominates. The theoretical fix is DP=8 TP=1
# (each GPU runs the full model independently, splits prompts via vLLM's
# data_parallel rank), but theory != reality on this cluster's NVLink/topology.
# This script gets actual wall-clock numbers for both configurations on the
# same prompts so we can decide before changing baseline.sh / G1.sh / etc.
#
# Usage:
#   bash scripts/_dp_vs_tp_benchmark.sh [N_SAMPLES] [MAX_NEW_TOKENS]
# Defaults: 100 samples, 4096 max_new_tokens (smaller than full eval cap so the
# benchmark itself finishes in a few minutes, but long enough that decode-bound
# behaviour dominates rather than just prefill).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROGRESS_HELPER="${REPO_ROOT}/scripts/supplement/vllm_generate_progress.py"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-/mnt/workspace/venvs/.teacherVenv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/gemma-4-E4B/}"
EVAL_DATA="${EVAL_DATA:-/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl}"

N_SAMPLES="${1:-${N_SAMPLES:-100}}"
MAX_NEW_TOKENS="${2:-${MAX_NEW_TOKENS:-4096}}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
PROGRESS_BATCH_SIZE="${PROGRESS_BATCH_SIZE:-256}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-1234}"

TS="$(date +%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-/tmp/dp_vs_tp_bench_${TS}}"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo " DP=8/TP=1  vs  DP=1/TP=8  benchmark"
echo "============================================================"
echo "  Model:           ${MODEL_PATH}"
echo "  Dataset:         ${EVAL_DATA}"
echo "  N_SAMPLES:       ${N_SAMPLES}"
echo "  MAX_NEW_TOKENS:  ${MAX_NEW_TOKENS}"
echo "  PROMPT_MAX_LEN:  ${PROMPT_MAX_LEN}"
echo "  MAX_NUM_SEQS:    ${MAX_NUM_SEQS}"
echo "  out_dir:         ${OUT_DIR}"
echo "============================================================"

# Common vLLM env (matches scripts/supplement_2rounds/_vllm_runtime.sh defaults).
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

run_one() {
  local label="$1"; shift
  local out_jsonl="${OUT_DIR}/eval_${label}.jsonl"
  local log_path="${OUT_DIR}/log_${label}.txt"
  local time_path="${OUT_DIR}/time_${label}.txt"

  echo ""
  echo "============================================================"
  echo " [${label}] starting  ($(date +%T))"
  echo "============================================================"

  # /usr/bin/time isn't available on this image, so we use date+%s.%N for
  # wall-clock and ignore RSS/CPU% (which we don't need for the comparison).
  local t0
  t0=$(date +%s.%N)
  bash -c "$*" 2>&1 | tee "${log_path}" | tail -100
  local rc=$?
  local t1
  t1=$(date +%s.%N)
  local elapsed_sec
  elapsed_sec=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
  printf "wall_seconds=%s\nexit_code=%s\n" "${elapsed_sec}" "${rc}" > "${time_path}"

  echo ""
  echo "[${label}] done at $(date +%T)  wall=${elapsed_sec}s  rc=${rc}"
  echo "  jsonl size: $(wc -l < "${out_jsonl}" 2>/dev/null || echo 0) lines"
}

# ---------- Configuration A: TP=8, DP=1 (current baseline.sh default) ----------
CONFIG_A_CMD="env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ${TEACHER_PYTHON_BIN} ${PROGRESS_HELPER} \
  --pretrain ${MODEL_PATH} \
  --dataset ${EVAL_DATA} \
  --input_key question \
  --output_path ${OUT_DIR}/eval_TP8_DP1.jsonl \
  --prompt_max_len ${PROMPT_MAX_LEN} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --temperature ${TEMPERATURE} \
  --top_p ${TOP_P} \
  --repetition_penalty 1.0 \
  --max_samples ${N_SAMPLES} \
  --best_of_n 1 \
  --tp_size 8 \
  --max_num_seqs ${MAX_NUM_SEQS} \
  --progress_batch_size ${PROGRESS_BATCH_SIZE} \
  --seed ${SEED}"

run_one "TP8_DP1" "${CONFIG_A_CMD}"

# ---------- Configuration B: TP=1, DP=8 (vllm built-in DP via torchrun) ----------
# vllm_generate_progress.py uses distributed_executor_backend="external_launcher"
# when --dp_size > 1, so we MUST launch via torchrun. Each rank runs an LLM with
# its own GPU; rank dispatches prompts via idx % dp_size == dp_rank, then merges.
CONFIG_B_CMD="env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  $(dirname ${TEACHER_PYTHON_BIN})/torchrun --nproc_per_node=8 --master_port=29550 \
  ${PROGRESS_HELPER} \
  --pretrain ${MODEL_PATH} \
  --dataset ${EVAL_DATA} \
  --input_key question \
  --output_path ${OUT_DIR}/eval_TP1_DP8.jsonl \
  --prompt_max_len ${PROMPT_MAX_LEN} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --temperature ${TEMPERATURE} \
  --top_p ${TOP_P} \
  --repetition_penalty 1.0 \
  --max_samples ${N_SAMPLES} \
  --best_of_n 1 \
  --tp_size 1 \
  --dp_size 8 \
  --max_num_seqs ${MAX_NUM_SEQS} \
  --progress_batch_size ${PROGRESS_BATCH_SIZE} \
  --seed ${SEED}"

run_one "TP1_DP8" "${CONFIG_B_CMD}"

echo ""
echo "============================================================"
echo " Final wall-clock comparison"
echo "============================================================"
"${TEACHER_PYTHON_BIN}" - <<PY
import os
out = "${OUT_DIR}"
rows = []
for label in ['TP8_DP1', 'TP1_DP8']:
    tp = os.path.join(out, f'time_{label}.txt')
    jp = os.path.join(out, f'eval_{label}.jsonl')
    if not os.path.exists(tp):
        continue
    sec = None; rc = None
    for line in open(tp):
        if line.startswith('wall_seconds='):
            sec = float(line.strip().split('=', 1)[1])
        if line.startswith('exit_code='):
            rc = int(line.strip().split('=', 1)[1])
    n = sum(1 for _ in open(jp)) if os.path.exists(jp) else 0
    rows.append((label, sec, n, rc))

print(f"{'config':<10} {'wall(s)':>10} {'samples':>8} {'samp/s':>8} {'rc':>3}")
print('-' * 50)
for label, sec, n, rc in rows:
    rate = (n/sec) if (sec and sec > 0) else 0.0
    print(f'{label:<10} {sec:>10.2f} {n:>8d} {rate:>8.3f} {rc:>3d}')

if len(rows) == 2 and rows[0][1] and rows[1][1]:
    a_label, a_sec, _, _ = rows[0]
    b_label, b_sec, _, _ = rows[1]
    speedup = a_sec / b_sec
    if speedup > 1:
        print(f'\\n  → {b_label} is {speedup:.2f}x FASTER than {a_label}')
    else:
        print(f'\\n  → {a_label} is {1/speedup:.2f}x FASTER than {b_label}')
PY

echo ""
echo "Artefacts:    ${OUT_DIR}"
