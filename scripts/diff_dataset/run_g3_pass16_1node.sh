#!/usr/bin/env bash
# Run 1-node code-generation benchmark pass@16 for an already-trained G3 run.
#
# Defaults target the recovered DLC run:
#   /mnt/data/ebft-distribution-new/outputs/diff_dataset/g3_dlc8pd4wa5cqtgur
#
# Usage:
#   bash scripts/diff_dataset/run_g3_pass16_1node.sh
#   RUN_DIR=/path/to/run bash scripts/diff_dataset/run_g3_pass16_1node.sh
#   bash scripts/diff_dataset/run_g3_pass16_1node.sh /path/to/run

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

RUN_DIR="${RUN_DIR:-${1:-/mnt/data/ebft-distribution-new/outputs/diff_dataset/g3_dlc8pd4wa5cqtgur}}"
MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/model}"

# If this command is submitted to a multi-pod DLC job, keep it 1-node by
# letting only rank 0 run the benchmark.
NODE_RANK="${PET_NODE_RANK:-${RANK:-0}}"
if [[ "${NODE_RANK}" != "0" ]]; then
  echo "[pass16-1node] rank=${NODE_RANK}; skipping because this is a 1-node eval."
  exit 0
fi

SETUP_ENV_SCRIPT="${SETUP_ENV_SCRIPT:-${REPO_ROOT}/scripts/setup_env.sh}"
RUN_SETUP_ENV="${RUN_SETUP_ENV:-true}"
if [[ "${RUN_SETUP_ENV}" == "true" ]]; then
  [[ -f "${SETUP_ENV_SCRIPT}" ]] || { echo "[ERROR] setup_env.sh not found: ${SETUP_ENV_SCRIPT}"; exit 1; }
  bash "${SETUP_ENV_SCRIPT}"
fi

TEACHER_VENV="${TEACHER_VENV:-/mnt/workspace/venvs/.teacherVenv}"
CODE_BENCHMARK_PYTHON_BIN="${CODE_BENCHMARK_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-${REPO_ROOT}/scripts/benchmarks/run_code_generation_benchmarks.py}"

PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-/mnt/data/ebft-distribution-new/outputs/diff_dataset_prepared}"
HUMANEVAL_EVAL_DATA="${HUMANEVAL_EVAL_DATA:-${PREPARED_DATA_DIR}/humaneval_eval_qa.jsonl}"
MBPP_EVAL_DATA="${MBPP_EVAL_DATA:-${PREPARED_DATA_DIR}/mbpp_eval_qa.jsonl}"

MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a _VISIBLE_GPUS <<< "${MODEL_CUDA_VISIBLE_DEVICES}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-${#_VISIBLE_GPUS[@]}}"

TS="${TS:-$(date +%m%d_%H%M)}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/code_benchmarks/pass16_temp06_${TS}}"

# 0 means no prompt truncation in run_code_generation_benchmarks.py.
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GREEDY_TEMPERATURE="${GREEDY_TEMPERATURE:-0.0}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
N_SAMPLES="${N_SAMPLES:-16}"
PASSK_LIST="${PASSK_LIST:-16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GREEDY_BATCH_SIZE="${GREEDY_BATCH_SIZE:-16}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-4}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-10}"
MAX_SAMPLES_PER_BENCHMARK="${MAX_SAMPLES_PER_BENCHMARK:-0}"
BENCHMARKS="${BENCHMARKS:-humaneval,mbpp}"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600000}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
[[ "${NCCL_P2P_DISABLE:-}" == "1" ]] && unset NCCL_P2P_DISABLE
export NCCL_NET_GDR_DISABLE="${NCCL_NET_GDR_DISABLE:-1}"

[[ -d "${RUN_DIR}" ]] || { echo "[ERROR] RUN_DIR not found: ${RUN_DIR}"; exit 1; }
[[ -d "${MODEL_PATH}" ]] || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 1; }
[[ -x "${CODE_BENCHMARK_PYTHON_BIN}" ]] || { echo "[ERROR] python not executable: ${CODE_BENCHMARK_PYTHON_BIN}"; exit 1; }
[[ -f "${BENCHMARK_SCRIPT}" ]] || { echo "[ERROR] benchmark script not found: ${BENCHMARK_SCRIPT}"; exit 1; }
[[ -e "${HUMANEVAL_EVAL_DATA}" ]] || { echo "[ERROR] HumanEval data not found: ${HUMANEVAL_EVAL_DATA}"; exit 1; }
[[ -e "${MBPP_EVAL_DATA}" ]] || { echo "[ERROR] MBPP data not found: ${MBPP_EVAL_DATA}"; exit 1; }

mkdir -p "${OUT_DIR}"

if [[ "${CLEANUP_LEFTOVER_VLLM:-true}" == "true" ]]; then
  echo "[pass16-1node] cleaning leftover vLLM/Ray processes on this node..."
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f 'vllm.v1.engine.core' 2>/dev/null || true
  pkill -9 -f 'multiproc_executor' 2>/dev/null || true
  pkill -9 -f 'vllm_generate_progress' 2>/dev/null || true
  pkill -9 -f 'EngineCore' 2>/dev/null || true
  sleep 2
fi

echo "================================================================"
echo "  G3 pass@16 code benchmark (1 node)"
echo "  RUN_DIR:        ${RUN_DIR}"
echo "  MODEL_PATH:     ${MODEL_PATH}"
echo "  OUT_DIR:        ${OUT_DIR}"
echo "  BENCHMARKS:     ${BENCHMARKS}"
echo "  GPUs / TP:      ${MODEL_CUDA_VISIBLE_DEVICES} / ${VLLM_TP_SIZE}"
echo "  n_samples:      ${N_SAMPLES}"
echo "  passk_list:     ${PASSK_LIST}"
echo "  sample_temp:    ${SAMPLE_TEMPERATURE}"
echo "  max_new_tokens: ${MAX_NEW_TOKENS}"
echo "================================================================"

CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES}" \
"${CODE_BENCHMARK_PYTHON_BIN}" "${BENCHMARK_SCRIPT}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUT_DIR}" \
  --benchmarks "${BENCHMARKS}" \
  --backend vllm \
  --humaneval_dataset "${HUMANEVAL_EVAL_DATA}" \
  --humaneval_split test \
  --mbpp_dataset "${MBPP_EVAL_DATA}" \
  --mbpp_split test \
  --prompt_max_len "${PROMPT_MAX_LEN}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --greedy_temperature "${GREEDY_TEMPERATURE}" \
  --sample_temperature "${SAMPLE_TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --repetition_penalty "${REPETITION_PENALTY}" \
  --n_samples "${N_SAMPLES}" \
  --passk_list "${PASSK_LIST}" \
  --tp_size "${VLLM_TP_SIZE}" \
  --max_num_seqs "${MAX_NUM_SEQS}" \
  --greedy_batch_size "${GREEDY_BATCH_SIZE}" \
  --sample_batch_size "${SAMPLE_BATCH_SIZE}" \
  --max_samples_per_benchmark "${MAX_SAMPLES_PER_BENCHMARK}" \
  --timeout_seconds "${TIMEOUT_SECONDS}" \
  2>&1 | tee "${OUT_DIR}/run.log"

echo "[pass16-1node] summary: ${OUT_DIR}/benchmark_summary.json"
echo "[pass16-1node] details: ${OUT_DIR}/benchmark_details.jsonl"
