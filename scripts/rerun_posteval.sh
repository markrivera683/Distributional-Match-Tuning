#!/usr/bin/env bash
# rerun_posteval.sh
#
# Re-run the two-round 16k/32k vLLM post-eval against an *already-trained*
# run directory, and (optionally) archive the run outputs afterwards. Use
# this when run_G{2,3}_rebase_2node_once.sh finished training successfully
# but the post-eval step failed (e.g. the DLC ssh-to-worker bug that caused
# G{2,3}_2node.sh to try ``ssh <head-ip>`` and die with
# "Connection refused"; see run_G{2,3}_rebase_2node_once.sh commit notes
# around ``WORKER_SSH_TARGET=""``).
#
# This script bypasses G{2,3}_2node.sh's ssh dispatcher entirely and always
# runs the single-node (head, TP=8) variant G{2,3}.sh, so it works in DLC
# multi-pod mode, DSW single-node, and plain ssh environments alike.
#
# Usage:
#   bash scripts/rerun_posteval.sh /path/to/run_dir
#   RUN_DIR=/path/to/run_dir bash scripts/rerun_posteval.sh
#
# Auto-detection:
#   - VARIANT (g1/g2/g3) inferred from RUN_DIR basename (g2_..., g3_...),
#     or can be forced via VARIANT=g3 / --variant g3.
#   - MODEL_PATH defaults to ${RUN_DIR}/model.
#   - ARCHIVE_OUTPUT_ROOT defaults to the same outputs_g{N}_0.99 path used
#     by run_G{1,2,3}_rebase_2node_once.sh; override to skip or redirect.
#
# Common overrides (same knobs as the original runner):
#   MODEL_CUDA_VISIBLE_DEVICES   (default 0,1,2,3,4,5,6,7)
#   VLLM_TP_SIZE                 (default: count of MODEL_CUDA_VISIBLE_DEVICES)
#   POST_EVAL_MAX_SAMPLES        (default 5328)
#   FIRST_PASS_MAX_NEW_TOKENS    (default 16384)
#   SECOND_PASS_MAX_NEW_TOKENS   (default 32768)
#   VLLM_MAX_NUM_SEQS            (default 256)
#   EVAL_DATA                    (default /mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl)
#   DO_ARCHIVE                   true|false (default true)
#   ARCHIVE_OUTPUT_ROOT          destination for ``mv RUN_DIR ...``; leave
#                                empty or DO_ARCHIVE=false to skip.
#
# Exit codes:
#   0  post-eval completed and (if requested) archive succeeded
#   2  usage / RUN_DIR not found
#   3  variant could not be inferred
#   $EVAL_RC  whatever G{1,2,3}.sh returned
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# -----------------------------------------------------------------------------
# Arg parsing: positional RUN_DIR, plus --variant / --no-archive flags.
# -----------------------------------------------------------------------------
VARIANT="${VARIANT:-}"
DO_ARCHIVE="${DO_ARCHIVE:-true}"

while (( $# > 0 )); do
  case "$1" in
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --variant=*)
      VARIANT="${1#--variant=}"
      shift
      ;;
    --no-archive)
      DO_ARCHIVE="false"
      shift
      ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *)
      if [[ -z "${RUN_DIR:-}" ]]; then
        RUN_DIR="$1"
      else
        echo "[rerun_posteval] unexpected extra argument: $1" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: bash scripts/rerun_posteval.sh /path/to/run_dir" >&2
  echo "   or: RUN_DIR=/path/to/run_dir bash scripts/rerun_posteval.sh" >&2
  exit 2
fi
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[rerun_posteval] ERROR: RUN_DIR not found: ${RUN_DIR}" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# Variant detection.
# -----------------------------------------------------------------------------
if [[ -z "${VARIANT}" ]]; then
  _basename="$(basename "${RUN_DIR}")"
  case "${_basename,,}" in
    g1_*|*_g1_*) VARIANT="g1" ;;
    g2_*|*_g2_*) VARIANT="g2" ;;
    g3_*|*_g3_*) VARIANT="g3" ;;
    *)
      echo "[rerun_posteval] ERROR: could not infer variant (g1/g2/g3) from RUN_DIR basename '${_basename}'." >&2
      echo "                 Pass --variant g1|g2|g3 explicitly." >&2
      exit 3
      ;;
  esac
fi
VARIANT="${VARIANT,,}"
case "${VARIANT}" in
  g1|g2|g3) ;;
  *)
    echo "[rerun_posteval] ERROR: invalid variant '${VARIANT}' (expected g1, g2, or g3)." >&2
    exit 3
    ;;
esac

VARIANT_UPPER="${VARIANT^^}"
POST_EVAL_SCRIPT="${POST_EVAL_SCRIPT:-${REPO_ROOT}/scripts/supplement_2rounds/${VARIANT_UPPER}.sh}"
if [[ ! -f "${POST_EVAL_SCRIPT}" ]]; then
  echo "[rerun_posteval] ERROR: eval script not found: ${POST_EVAL_SCRIPT}" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# Sensible defaults for the single-node eval. These mirror the defaults in
# run_G{2,3}_rebase_2node_once.sh so results are comparable with a successful
# 2-node run's post-eval.
# -----------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/model}"
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[rerun_posteval] ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  echo "                 (expected a saved HF checkpoint directory)" >&2
  exit 2
fi

export MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a _VISIBLE_GPUS <<< "${MODEL_CUDA_VISIBLE_DEVICES}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-${#_VISIBLE_GPUS[@]}}"

export EVAL_DATA="${EVAL_DATA:-/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl}"
export POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
export POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
export FIRST_PASS_MAX_NEW_TOKENS="${FIRST_PASS_MAX_NEW_TOKENS:-16384}"
export SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS:-32768}"
export POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
export POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
export POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY:-1.0}"
export POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N:-1}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
export VLLM_PROGRESS_BATCH_SIZE="${VLLM_PROGRESS_BATCH_SIZE:-256}"
export VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
export VLLM_SEED="${VLLM_SEED:-1234}"

# Tag the re-run so output files don't collide with the partial stage1
# attempt that the original run may have already written into
# supplement_logs/.
export EVAL_TAG="${EVAL_TAG:-post_train_rerun_$(date +%m%d_%H%M)}"
export LOG_DIR="${LOG_DIR:-${RUN_DIR}/supplement_logs}"
mkdir -p "${LOG_DIR}"

# NCCL safety (same envs as the runner; harmless if already set).
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
[[ "${NCCL_P2P_DISABLE:-}" == "1" ]] && unset NCCL_P2P_DISABLE
export NCCL_NET_GDR_DISABLE="${NCCL_NET_GDR_DISABLE:-1}"

# Venvs. Prefer the canonical locations created by setup_env.sh; fall back
# to the legacy in-repo ones if that layout is still in use.
_DEFAULT_TEACHER_VENV="/mnt/workspace/venvs/.teacherVenv"
[[ -d "${_DEFAULT_TEACHER_VENV}" ]] || _DEFAULT_TEACHER_VENV="${REPO_ROOT}/.teacherVenv"
export TEACHER_VENV="${TEACHER_VENV:-${_DEFAULT_TEACHER_VENV}}"
export TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
_DEFAULT_ANALYSIS_VENV="/mnt/workspace/venvs/.venv"
[[ -d "${_DEFAULT_ANALYSIS_VENV}" ]] || _DEFAULT_ANALYSIS_VENV="${REPO_ROOT}/.venv"
export ANALYSIS_VENV="${ANALYSIS_VENV:-${_DEFAULT_ANALYSIS_VENV}}"
export ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"

export REPO_ROOT
export RUN_DIR MODEL_PATH

# Default archive root mirrors run_G{N}_rebase_2node_once.sh
# (outputs_g{N}_0.99). Leave empty or DO_ARCHIVE=false to skip.
case "${VARIANT}" in
  g1) ARCHIVE_OUTPUT_ROOT="${ARCHIVE_OUTPUT_ROOT:-/mnt/data/ebft-teacher-distribution/outputs_g1_0.99}" ;;
  g2) ARCHIVE_OUTPUT_ROOT="${ARCHIVE_OUTPUT_ROOT:-/mnt/data/ebft-teacher-distribution/outputs_g2_0.99}" ;;
  g3) ARCHIVE_OUTPUT_ROOT="${ARCHIVE_OUTPUT_ROOT:-/mnt/data/ebft-teacher-distribution/outputs_g3_0.99}" ;;
esac

echo "================================================================"
echo "===== rerun_posteval.sh (${VARIANT_UPPER}) ====="
echo "RUN_DIR:               ${RUN_DIR}"
echo "MODEL_PATH:            ${MODEL_PATH}"
echo "EVAL_DATA:             ${EVAL_DATA}"
echo "EVAL_TAG:              ${EVAL_TAG}"
echo "LOG_DIR:               ${LOG_DIR}"
echo "POST_EVAL_SCRIPT:      ${POST_EVAL_SCRIPT}"
echo "TP size / GPUs:        ${VLLM_TP_SIZE} / ${MODEL_CUDA_VISIBLE_DEVICES}"
echo "first-pass max_new:    ${FIRST_PASS_MAX_NEW_TOKENS}"
echo "second-pass max_new:   ${SECOND_PASS_MAX_NEW_TOKENS}"
echo "POST_EVAL_MAX_SAMPLES: ${POST_EVAL_MAX_SAMPLES}"
echo "DO_ARCHIVE:            ${DO_ARCHIVE}"
echo "ARCHIVE_OUTPUT_ROOT:   ${ARCHIVE_OUTPUT_ROOT:-<skip>}"
echo "================================================================"
echo ""

# -----------------------------------------------------------------------------
# Sanity: warn if GPUs look busy; running eval on a card already hosting
# a 27B teacher will either OOM immediately or run at ~1/10 speed.
# -----------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  _busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -cE '^[0-9]' || true)"
  if (( _busy > 0 )); then
    echo "[rerun_posteval] WARNING: ${_busy} CUDA process(es) still running on this node."
    echo "                 Post-eval will contend for GPU memory; consider \`ray stop --force\` /"
    echo "                 \`pkill -f vllm\` before retrying if it OOMs."
    echo ""
  fi
fi

# -----------------------------------------------------------------------------
# Kick off the eval. Do NOT pass --exit-on-fail via set -e here so we can
# still reach the archive step even if the user wants to archive a partial
# result on failure; we capture EVAL_RC explicitly.
# -----------------------------------------------------------------------------
set +e
bash "${POST_EVAL_SCRIPT}" "${RUN_DIR}"
EVAL_RC=$?
set -e

if (( EVAL_RC != 0 )); then
  echo "[rerun_posteval] ERROR: post-eval script exited ${EVAL_RC}. See ${LOG_DIR}/ for details."
else
  echo "[rerun_posteval] post-eval completed OK."
fi

# -----------------------------------------------------------------------------
# Archive (optional). Same semantics as run_G{N}_rebase_2node_once.sh:
# ``mv RUN_DIR <target_root>/<basename(RUN_DIR)>``. Skip on failure unless
# the user explicitly wants to archive a partial result (not the default).
# -----------------------------------------------------------------------------
if [[ "${DO_ARCHIVE}" == "true" && -n "${ARCHIVE_OUTPUT_ROOT:-}" ]]; then
  if (( EVAL_RC != 0 )); then
    echo "[rerun_posteval] archive skipped because EVAL_RC=${EVAL_RC}."
    echo "                 Re-run with DO_ARCHIVE=false to suppress this message,"
    echo "                 or manually ``mv '${RUN_DIR}' '${ARCHIVE_OUTPUT_ROOT}/'`` if you"
    echo "                 really want to keep the partial output."
  else
    mkdir -p "${ARCHIVE_OUTPUT_ROOT}"
    _target="${ARCHIVE_OUTPUT_ROOT}/$(basename "${RUN_DIR}")"
    if [[ -e "${_target}" ]]; then
      _target="${_target}_$(date +%m%d_%H%M%S)"
    fi
    echo "[rerun_posteval] archiving: mv '${RUN_DIR}' '${_target}'"
    mv "${RUN_DIR}" "${_target}"
    echo "[rerun_posteval] archive done: ${_target}"
  fi
fi

exit "${EVAL_RC}"
