#!/usr/bin/env bash
# Standalone G3 post-eval rerun for an existing checkpoint.
# Usage:
RUN_DIR=/root/outputs/g3_rebase_0407_1338
#   bash scripts/supplement/G3_eval.sh /root/outputs/g3_rebase_0407_1338
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Matching-Tuning}"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"
STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"

RUN_DIR="${RUN_DIR:-${1:-}}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: RUN_DIR=/path/to/run bash scripts/supplement/G3_eval.sh"
  echo "   or: bash scripts/supplement/G3_eval.sh /path/to/run"
  exit 1
fi

SAVE_PATH="${SAVE_PATH:-${RUN_DIR}/model}"
EVAL_TAG="${EVAL_TAG:-supplement}"
SCRIPT_NAME="$(basename "$0" .sh)"
TS="${TS:-$(date +%m%d_%H%M)}"

STUDENT_CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
POST_EVAL_NPROC="${POST_EVAL_NPROC:-8}"
POST_EVAL_NNODES="${POST_EVAL_NNODES:-1}"
POST_EVAL_NPROC_PER_NODE="${POST_EVAL_NPROC_PER_NODE:-${POST_EVAL_NPROC}}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
POST_EVAL_MAX_NEW_TOKENS="${POST_EVAL_MAX_NEW_TOKENS:-8192}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_MICRO_BATCH_SIZE="${POST_EVAL_MICRO_BATCH_SIZE:-128}"
POST_EVAL_MASTER_PORT="${POST_EVAL_MASTER_PORT:-29512}"
HEAD_NODE="${HEAD_NODE:-}"
HEAD_NODE_IP="${HEAD_NODE_IP:-}"
WORKER_NODE="${WORKER_NODE:-}"
WORKER_NODE_IP="${WORKER_NODE_IP:-}"
SSH_USER="${SSH_USER:-}"
SSH_OPTS="${SSH_OPTS:-}"
WORKER_SSH_HOST="${WORKER_SSH_HOST:-}"
POST_EVAL_HEAD_CUDA_VISIBLE_DEVICES="${POST_EVAL_HEAD_CUDA_VISIBLE_DEVICES:-${STUDENT_CUDA_VISIBLE_DEVICES}}"
POST_EVAL_WORKER_CUDA_VISIBLE_DEVICES="${POST_EVAL_WORKER_CUDA_VISIBLE_DEVICES:-${STUDENT_CUDA_VISIBLE_DEVICES}}"

LOG_DIR="${LOG_DIR:-${RUN_DIR}/supplement_logs}"
SCRIPT_LOG_PATH="${SCRIPT_LOG_PATH:-${LOG_DIR}/${SCRIPT_NAME}_${EVAL_TAG}_${TS}.log}"
POST_EVAL_OUTPUT_PATH="${POST_EVAL_OUTPUT_PATH:-${LOG_DIR}/eval_results_${EVAL_TAG}_${TS}.jsonl}"
POST_EVAL_LOG_PATH="${POST_EVAL_LOG_PATH:-${LOG_DIR}/eval_${EVAL_TAG}_${TS}.log}"
ANALYSIS_REPORT_PATH="${ANALYSIS_REPORT_PATH:-${LOG_DIR}/eval_analysis_${EVAL_TAG}_${TS}.json}"
ANALYSIS_LOG_PATH="${ANALYSIS_LOG_PATH:-${LOG_DIR}/eval_analysis_${EVAL_TAG}_${TS}.log}"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

resolve_host_ip() {
  local host="$1"
  local ip=""
  local waited=0
  local resolve_wait_seconds="${HOST_RESOLVE_WAIT_SECONDS:-60}"
  local resolve_retry_seconds="${HOST_RESOLVE_RETRY_SECONDS:-2}"

  if [[ "${host}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    echo "${host}"
    return 0
  fi

  while true; do
    ip="$(getent ahostsv4 "${host}" | awk 'NR==1 {print $1}')"
    if [[ -n "${ip}" ]]; then
      echo "${ip}"
      return 0
    fi
    if (( waited >= resolve_wait_seconds )); then
      echo "[ERROR] failed to resolve IPv4 for host: ${host}"
      exit 1
    fi
    sleep "${resolve_retry_seconds}"
    waited=$((waited + resolve_retry_seconds))
  done
}

if [[ ! -x "${STUDENT_PYTHON_BIN}" ]]; then
  echo "[ERROR] STUDENT_PYTHON_BIN not executable: ${STUDENT_PYTHON_BIN}"
  exit 1
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[ERROR] RUN_DIR not found: ${RUN_DIR}"
  exit 1
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${SCRIPT_LOG_PATH}") 2>&1

if [[ ! -e "${SAVE_PATH}" ]]; then
  echo "[ERROR] SAVE_PATH not found: ${SAVE_PATH}"
  exit 1
fi

if [[ ! -e "${EVAL_DATA}" ]]; then
  echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"
  exit 1
fi

mkdir -p "$(dirname "${POST_EVAL_OUTPUT_PATH}")" "$(dirname "${ANALYSIS_REPORT_PATH}")"
cd "${REPO_ROOT}"

if (( POST_EVAL_NNODES > 1 )); then
  if [[ -z "${WORKER_NODE}" ]]; then
    echo "[ERROR] POST_EVAL_NNODES=${POST_EVAL_NNODES} but WORKER_NODE is not set"
    exit 1
  fi
  if [[ -z "${HEAD_NODE}" ]]; then
    HEAD_NODE="$(hostname -s 2>/dev/null || hostname)"
  fi
  if [[ -z "${HEAD_NODE_IP}" ]]; then
    HEAD_NODE_IP="$(resolve_host_ip "${HEAD_NODE}")"
  fi
  if [[ -z "${WORKER_NODE_IP}" ]]; then
    WORKER_NODE_IP="$(resolve_host_ip "${WORKER_NODE}")"
  fi
  WORKER_SSH_HOST="${WORKER_SSH_HOST:-${WORKER_NODE_IP}}"
  if [[ -n "${SSH_USER}" ]]; then
    WORKER_SSH_TARGET="${SSH_USER}@${WORKER_SSH_HOST}"
  else
    WORKER_SSH_TARGET="${WORKER_SSH_HOST}"
  fi
fi

echo "========== G3 Supplement Eval =========="
echo "RUN_DIR:                      ${RUN_DIR}"
echo "SAVE_PATH:                    ${SAVE_PATH}"
echo "LOAD_MODEL_PATH:              ${SAVE_PATH}"
echo "EVAL_DATA:                    ${EVAL_DATA}"
echo "SCRIPT_LOG_PATH:              ${SCRIPT_LOG_PATH}"
echo "POST_EVAL_LOG_PATH:           ${POST_EVAL_LOG_PATH}"
echo "ANALYSIS_LOG_PATH:            ${ANALYSIS_LOG_PATH}"
echo "STUDENT_CUDA_VISIBLE_DEVICES: ${STUDENT_CUDA_VISIBLE_DEVICES}"
echo "POST_EVAL_NNODES:             ${POST_EVAL_NNODES}"
echo "POST_EVAL_NPROC_PER_NODE:     ${POST_EVAL_NPROC_PER_NODE}"
if (( POST_EVAL_NNODES > 1 )); then
echo "HEAD_NODE / IP:               ${HEAD_NODE} / ${HEAD_NODE_IP}"
echo "WORKER_NODE / IP:             ${WORKER_NODE} / ${WORKER_NODE_IP}"
echo "HEAD EVAL CUDA:               ${POST_EVAL_HEAD_CUDA_VISIBLE_DEVICES}"
echo "WORKER EVAL CUDA:             ${POST_EVAL_WORKER_CUDA_VISIBLE_DEVICES}"
fi
echo "POST_EVAL_PROMPT_MAX_LEN:     ${POST_EVAL_PROMPT_MAX_LEN}"
echo "POST_EVAL_MAX_NEW_TOKENS:     ${POST_EVAL_MAX_NEW_TOKENS}"
echo "POST_EVAL_MAX_SAMPLES:        ${POST_EVAL_MAX_SAMPLES}"
echo "OUTPUT_PATH:                  ${POST_EVAL_OUTPUT_PATH}"
echo "========================================"

echo "[load-model] batch_inference will load actor weights from: ${SAVE_PATH}"
if (( POST_EVAL_NNODES > 1 )); then
  REMOTE_POST_EVAL_LOG_PATH="${LOG_DIR}/eval_remote_${EVAL_TAG}_${TS}.log"

  ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -s -- \
    '${REPO_ROOT}' \
    '${STUDENT_PYTHON_BIN}' \
    '${SAVE_PATH}' \
    '${EVAL_DATA}' \
    '${POST_EVAL_OUTPUT_PATH}' \
    '${POST_EVAL_PROMPT_MAX_LEN}' \
    '${POST_EVAL_MAX_NEW_TOKENS}' \
    '${POST_EVAL_TEMPERATURE}' \
    '${POST_EVAL_TOP_P}' \
    '${POST_EVAL_MAX_SAMPLES}' \
    '${POST_EVAL_MICRO_BATCH_SIZE}' \
    '${POST_EVAL_MASTER_PORT}' \
    '${HEAD_NODE_IP}' \
    '${POST_EVAL_NNODES}' \
    '${POST_EVAL_NPROC_PER_NODE}' \
    '${POST_EVAL_WORKER_CUDA_VISIBLE_DEVICES}' \
    '${HF_HOME}' \
    '${HF_HUB_OFFLINE}' \
    '${HF_DATASETS_OFFLINE}' \
    '${HF_HUB_DISABLE_XET}'" <<'EOF' > "${REMOTE_POST_EVAL_LOG_PATH}" 2>&1 &
set -euo pipefail
REPO_ROOT="$1"
STUDENT_PYTHON_BIN="$2"
SAVE_PATH="$3"
EVAL_DATA="$4"
POST_EVAL_OUTPUT_PATH="$5"
POST_EVAL_PROMPT_MAX_LEN="$6"
POST_EVAL_MAX_NEW_TOKENS="$7"
POST_EVAL_TEMPERATURE="$8"
POST_EVAL_TOP_P="$9"
POST_EVAL_MAX_SAMPLES="${10}"
POST_EVAL_MICRO_BATCH_SIZE="${11}"
POST_EVAL_MASTER_PORT="${12}"
HEAD_NODE_IP="${13}"
POST_EVAL_NNODES="${14}"
POST_EVAL_NPROC_PER_NODE="${15}"
POST_EVAL_WORKER_CUDA_VISIBLE_DEVICES="${16}"
HF_HOME="${17}"
HF_HUB_OFFLINE="${18}"
HF_DATASETS_OFFLINE="${19}"
HF_HUB_DISABLE_XET="${20}"

export HF_HOME HF_HUB_OFFLINE HF_DATASETS_OFFLINE HF_HUB_DISABLE_XET TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "${REPO_ROOT}"
mkdir -p "$(dirname "${POST_EVAL_OUTPUT_PATH}")"
CUDA_VISIBLE_DEVICES="${POST_EVAL_WORKER_CUDA_VISIBLE_DEVICES}" \
"${STUDENT_PYTHON_BIN}" -m torch.distributed.run \
  --nnodes "${POST_EVAL_NNODES}" \
  --nproc_per_node "${POST_EVAL_NPROC_PER_NODE}" \
  --node_rank 1 \
  --master_addr "${HEAD_NODE_IP}" \
  --master_port "${POST_EVAL_MASTER_PORT}" \
  -m openrlhf.cli.batch_inference \
  --eval_task generate \
  --pretrain "${SAVE_PATH}" \
  --dataset "${EVAL_DATA}" \
  --input_key question \
  --output_path "${POST_EVAL_OUTPUT_PATH}" \
  --prompt_max_len "${POST_EVAL_PROMPT_MAX_LEN}" \
  --max_new_tokens "${POST_EVAL_MAX_NEW_TOKENS}" \
  --temperature "${POST_EVAL_TEMPERATURE}" \
  --top_p "${POST_EVAL_TOP_P}" \
  --max_samples "${POST_EVAL_MAX_SAMPLES}" \
  --micro_batch_size "${POST_EVAL_MICRO_BATCH_SIZE}" \
  --bf16
EOF
  REMOTE_EVAL_PID=$!

  set +e
  CUDA_VISIBLE_DEVICES="${POST_EVAL_HEAD_CUDA_VISIBLE_DEVICES}" \
  "${STUDENT_PYTHON_BIN}" -m torch.distributed.run \
    --nnodes "${POST_EVAL_NNODES}" \
    --nproc_per_node "${POST_EVAL_NPROC_PER_NODE}" \
    --node_rank 0 \
    --master_addr "${HEAD_NODE_IP}" \
    --master_port "${POST_EVAL_MASTER_PORT}" \
    -m openrlhf.cli.batch_inference \
    --eval_task generate \
    --pretrain "${SAVE_PATH}" \
    --dataset "${EVAL_DATA}" \
    --input_key question \
    --output_path "${POST_EVAL_OUTPUT_PATH}" \
    --prompt_max_len "${POST_EVAL_PROMPT_MAX_LEN}" \
    --max_new_tokens "${POST_EVAL_MAX_NEW_TOKENS}" \
    --temperature "${POST_EVAL_TEMPERATURE}" \
    --top_p "${POST_EVAL_TOP_P}" \
    --max_samples "${POST_EVAL_MAX_SAMPLES}" \
    --micro_batch_size "${POST_EVAL_MICRO_BATCH_SIZE}" \
    --bf16 \
    2>&1 | tee "${POST_EVAL_LOG_PATH}"
  LOCAL_EVAL_RC=$?
  wait "${REMOTE_EVAL_PID}"
  REMOTE_EVAL_RC=$?
  set -e

  if (( LOCAL_EVAL_RC != 0 || REMOTE_EVAL_RC != 0 )); then
    echo "[ERROR] multinode post-eval failed: local_rc=${LOCAL_EVAL_RC}, remote_rc=${REMOTE_EVAL_RC}"
    echo "[ERROR] remote eval log: ${REMOTE_POST_EVAL_LOG_PATH}"
    exit 1
  fi
else
  CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}" \
  "${STUDENT_PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node "${POST_EVAL_NPROC_PER_NODE}" --master_port "${POST_EVAL_MASTER_PORT}" \
    -m openrlhf.cli.batch_inference \
    --eval_task generate \
    --pretrain "${SAVE_PATH}" \
    --dataset "${EVAL_DATA}" \
    --input_key question \
    --output_path "${POST_EVAL_OUTPUT_PATH}" \
    --prompt_max_len "${POST_EVAL_PROMPT_MAX_LEN}" \
    --max_new_tokens "${POST_EVAL_MAX_NEW_TOKENS}" \
    --temperature "${POST_EVAL_TEMPERATURE}" \
    --top_p "${POST_EVAL_TOP_P}" \
    --max_samples "${POST_EVAL_MAX_SAMPLES}" \
    --micro_batch_size "${POST_EVAL_MICRO_BATCH_SIZE}" \
    --bf16 \
    2>&1 | tee "${POST_EVAL_LOG_PATH}"
fi

echo "[post-eval] Saved: ${POST_EVAL_OUTPUT_PATH}"
echo "[post-eval] Log:   ${POST_EVAL_LOG_PATH}"

"${STUDENT_PYTHON_BIN}" "${REPO_ROOT}/scripts/analyze_eval_results.py" \
  --eval_results "${POST_EVAL_OUTPUT_PATH}" \
  --eval_dataset "${EVAL_DATA}" \
  --input_key question --label_key answer \
  --report_path "${ANALYSIS_REPORT_PATH}" \
  2>&1 | tee "${ANALYSIS_LOG_PATH}"

echo "[analysis] Report: ${ANALYSIS_REPORT_PATH}"
echo "[analysis] Log:    ${ANALYSIS_LOG_PATH}"
echo "[script]   Log:    ${SCRIPT_LOG_PATH}"
