#!/usr/bin/env bash
# Start a lightweight SGLang teacher server for slime OPD experiments.

set -euo pipefail

TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/mnt/data/models/Qwen3.5-4B}"
TEACHER_IP="${TEACHER_IP:-127.0.0.1}"
TEACHER_HOST="${TEACHER_HOST:-0.0.0.0}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-7}"
TEACHER_TP="${TEACHER_TP:-1}"
TEACHER_MEM_FRACTION_STATIC="${TEACHER_MEM_FRACTION_STATIC:-0.6}"
TEACHER_LOG_FILE="${TEACHER_LOG_FILE:-/mnt/data/ebft-distribution-new/outputs/diff_dataset/slime_opd_teacher_sglang.log}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$(dirname "${TEACHER_LOG_FILE}")"

CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${TEACHER_MODEL_PATH}" \
  --host "${TEACHER_HOST}" \
  --port "${TEACHER_PORT}" \
  --tp "${TEACHER_TP}" \
  --chunked-prefill-size 4096 \
  --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}" \
  > "${TEACHER_LOG_FILE}" 2>&1 &

echo "[teacher] pid=$!"
echo "[teacher] log=${TEACHER_LOG_FILE}"
echo "[teacher] waiting for http://${TEACHER_IP}:${TEACHER_PORT}/health_generate"
until curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" >/dev/null; do
  sleep 5
done
curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info"
echo
echo "[teacher] ready"
