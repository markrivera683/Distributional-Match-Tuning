#!/usr/bin/env bash
# rerun_posteval_2nodes.sh
#
# Stand-alone 2-node post-eval against an *already-trained* run dir. Does
# NOT re-run training. Uses the rendezvous protocol from
# scripts/supplement_2rounds/_rendezvous_dlc.sh so it works on DLC
# (no-sshd pods) without any SSH. Runs the same two-round 16k/32k vLLM
# pipeline as G{2,3}_2node.sh.
#
# Usage (run on BOTH master and worker pods with identical args):
#   bash scripts/rerun_posteval_2nodes.sh /path/to/run_dir
#   RUN_DIR=/path/to/run_dir bash scripts/rerun_posteval_2nodes.sh
#
# Role detection
# --------------
# * PET_NODE_RANK or RANK env var present (set by PAI DLC pytorch-job):
#     rank 0 -> master    (runs G{2,3}_2node.sh + drives the pipeline)
#     rank n -> worker n  (enters posteval_worker_watch, parks on OSS)
# * Neither env set, no DSW ssh plumbing provided:
#     fallback to single-node rerun_posteval.sh (head TP=8 only).
#
# Variant detection
# -----------------
# Inferred from RUN_DIR basename: g1_*, g2_*, g3_* -> G1 / G2 / G3.
# Override with --variant g2|g3 or VARIANT=g2 env var.
#
# Options
# -------
#   --variant g1|g2|g3        force the variant
#   --no-archive              don't move RUN_DIR to outputs_g{N}_0.99/ after
#   --master-only             treat this pod as the master even without
#                              DLC env (e.g. quick testing); no worker
#                              rendezvous, no 2-node parallelism.
#   --worker-only             treat this pod as a worker watcher; does not
#                              run G{2,3}_2node.sh. Useful if you want to
#                              manually orchestrate.
#
# Env overrides (same defaults as runner / G{2,3}_2node.sh):
#   POSTEVAL_WORKER_CUDA_VISIBLE_DEVICES   default 0,1,2,3,4,5,6,7
#   POSTEVAL_WORKER_VLLM_TP_SIZE           default: visible GPU count
#   MODEL_CUDA_VISIBLE_DEVICES             (master side) default same
#   VLLM_TP_SIZE                           (master side) default 8
#   POST_EVAL_MAX_SAMPLES                  default 5328
#   FIRST_PASS_MAX_NEW_TOKENS              default 16384
#   SECOND_PASS_MAX_NEW_TOKENS             default 32768
#   VLLM_MAX_NUM_SEQS                      default 256
#   EVAL_DATA                              default .../aops/test_qa.jsonl
#   DO_ARCHIVE                             true|false (default true, master-only)
#   ARCHIVE_OUTPUT_ROOT                    default outputs_g{N}_0.99/
#   POSTEVAL_RDV_POLL_SECS                 default 5
#   POSTEVAL_RDV_MASTER_TIMEOUT            default 7200 (2h)
#   POSTEVAL_RDV_WORKER_TIMEOUT            default 10800 (3h)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# -----------------------------------------------------------------------------
# Arg parsing
# -----------------------------------------------------------------------------
VARIANT="${VARIANT:-}"
DO_ARCHIVE="${DO_ARCHIVE:-true}"
FORCE_ROLE=""
while (( $# > 0 )); do
  case "$1" in
    --variant)      VARIANT="$2"; shift 2 ;;
    --variant=*)    VARIANT="${1#--variant=}"; shift ;;
    --no-archive)   DO_ARCHIVE="false"; shift ;;
    --master-only)  FORCE_ROLE="master_only"; shift ;;
    --worker-only)  FORCE_ROLE="worker_only"; shift ;;
    -h|--help)      sed -n '2,50p' "$0"; exit 0 ;;
    *)
      if [[ -z "${RUN_DIR:-}" ]]; then
        RUN_DIR="$1"
      else
        echo "[rerun_posteval_2nodes] unexpected arg: $1" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: bash scripts/rerun_posteval_2nodes.sh /path/to/run_dir" >&2
  echo "       (run the SAME command on both master pod and worker pod)" >&2
  exit 2
fi
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[rerun_posteval_2nodes] ERROR: RUN_DIR not found: ${RUN_DIR}" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# Variant detection
# -----------------------------------------------------------------------------
if [[ -z "${VARIANT}" ]]; then
  _basename="$(basename "${RUN_DIR}")"
  case "${_basename,,}" in
    g1_*|*_g1_*) VARIANT="g1" ;;
    g2_*|*_g2_*) VARIANT="g2" ;;
    g3_*|*_g3_*) VARIANT="g3" ;;
    *)
      echo "[rerun_posteval_2nodes] ERROR: cannot infer variant from '${_basename}'; pass --variant g1|g2|g3" >&2
      exit 3
      ;;
  esac
fi
VARIANT="${VARIANT,,}"
case "${VARIANT}" in g1|g2|g3) ;; *) echo "invalid variant '${VARIANT}'"; exit 3 ;; esac
VARIANT_UPPER="${VARIANT^^}"

# -----------------------------------------------------------------------------
# Role detection (mirrors run_G{2,3}_rebase_2node_once.sh env handling)
# -----------------------------------------------------------------------------
DLC_MODE="false"
DLC_NODE_RANK=""
DLC_WORLD_SIZE="${WORLD_SIZE:-${PET_WORLD_SIZE:-1}}"
if [[ -n "${PET_NODE_RANK:-}" ]]; then
  DLC_MODE="true"
  DLC_NODE_RANK="${PET_NODE_RANK}"
elif [[ -n "${RANK:-}" && "${DLC_WORLD_SIZE:-1}" -gt 1 ]]; then
  DLC_MODE="true"
  DLC_NODE_RANK="${RANK}"
fi

ROLE=""
case "${FORCE_ROLE}" in
  master_only)
    ROLE="master_only"
    ;;
  worker_only)
    ROLE="worker"
    ;;
  "")
    if [[ "${DLC_MODE}" == "true" ]]; then
      if [[ "${DLC_NODE_RANK}" == "0" ]]; then
        ROLE="master"
      else
        ROLE="worker"
      fi
    else
      ROLE="single_node"
    fi
    ;;
esac

echo "================================================================"
echo "  rerun_posteval_2nodes.sh  (variant=${VARIANT_UPPER}, role=${ROLE})"
echo "  RUN_DIR:   ${RUN_DIR}"
echo "  DLC_MODE:  ${DLC_MODE}  rank=${DLC_NODE_RANK:-<none>}  world=${DLC_WORLD_SIZE}"
echo "================================================================"

# Venvs -- same resolution as runner / G{2,3}.sh
_DEFAULT_TEACHER_VENV="/mnt/workspace/venvs/.teacherVenv"
[[ -d "${_DEFAULT_TEACHER_VENV}" ]] || _DEFAULT_TEACHER_VENV="${REPO_ROOT}/.teacherVenv"
TEACHER_VENV="${TEACHER_VENV:-${_DEFAULT_TEACHER_VENV}}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
_DEFAULT_ANALYSIS_VENV="/mnt/workspace/venvs/.venv"
[[ -d "${_DEFAULT_ANALYSIS_VENV}" ]] || _DEFAULT_ANALYSIS_VENV="${REPO_ROOT}/.venv"
ANALYSIS_VENV="${ANALYSIS_VENV:-${_DEFAULT_ANALYSIS_VENV}}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"
PROGRESS_HELPER="${PROGRESS_HELPER:-${REPO_ROOT}/scripts/supplement/vllm_generate_progress.py}"

for _bin in "${TEACHER_PYTHON_BIN}" "${ANALYSIS_PYTHON_BIN}"; do
  [[ -x "${_bin}" ]] || { echo "[rerun_posteval_2nodes] ERROR: not executable: ${_bin}"; exit 2; }
done
export REPO_ROOT TEACHER_VENV TEACHER_PYTHON_BIN ANALYSIS_VENV ANALYSIS_PYTHON_BIN PROGRESS_HELPER

# -----------------------------------------------------------------------------
# Cleanup any leftover GPU / ray processes from a previous failed attempt.
# AIMaster's restart-on-failure loop that prompted this script often leaves
# orphaned vLLM / ray / teacher workers still holding GPU memory; letting
# them survive guarantees the next attempt CUDA-OOMs at engine init.
# -----------------------------------------------------------------------------
_cleanup_leftover_procs() {
  echo "[rerun_posteval_2nodes] cleaning up leftover GPU processes..."
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f 'vllm.v1.engine.core' 2>/dev/null || true
  pkill -9 -f 'multiproc_executor' 2>/dev/null || true
  pkill -9 -f 'vllm_generate_progress' 2>/dev/null || true
  pkill -9 -f 'vllm serve' 2>/dev/null || true
  pkill -9 -f 'EngineCore' 2>/dev/null || true
  sleep 2
  # Wait up to 20s for GPU memory to drain
  if command -v nvidia-smi >/dev/null 2>&1; then
    local waited=0
    while (( waited < 20 )); do
      local max_used
      max_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk 'BEGIN{m=0} {if($1+0>m) m=$1+0} END{print m+0}')"
      if (( max_used <= 2048 )); then
        echo "[rerun_posteval_2nodes] GPUs drained (max=${max_used} MiB)"
        break
      fi
      sleep 2
      waited=$(( waited + 2 ))
    done
  fi
}
_cleanup_leftover_procs

# -----------------------------------------------------------------------------
# Dispatch by role
# -----------------------------------------------------------------------------
case "${ROLE}" in
  single_node)
    echo "[rerun_posteval_2nodes] single-node mode (no DLC env, no --master/worker-only)"
    echo "[rerun_posteval_2nodes] delegating to rerun_posteval.sh (single-node TP=8 eval)"
    if [[ "${DO_ARCHIVE}" == "false" ]]; then
      exec bash "${SCRIPT_DIR}/rerun_posteval.sh" --no-archive --variant "${VARIANT}" "${RUN_DIR}"
    else
      exec bash "${SCRIPT_DIR}/rerun_posteval.sh" --variant "${VARIANT}" "${RUN_DIR}"
    fi
    ;;

  master|master_only)
    MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/model}"
    [[ -d "${MODEL_PATH}" ]] || { echo "[master] MODEL_PATH missing: ${MODEL_PATH}"; exit 2; }

    export MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    IFS=',' read -r -a _HEAD_GPUS <<< "${MODEL_CUDA_VISIBLE_DEVICES}"
    export VLLM_TP_SIZE="${VLLM_TP_SIZE:-${#_HEAD_GPUS[@]}}"

    # Head / worker shard GPU config consumed by G{2,3}_2node.sh. In
    # rendezvous mode both shards run TP=8 on their own pod's 8 GPUs.
    export HEAD_CUDA_VISIBLE_DEVICES="${HEAD_CUDA_VISIBLE_DEVICES:-${MODEL_CUDA_VISIBLE_DEVICES}}"
    export HEAD_VLLM_TP_SIZE="${HEAD_VLLM_TP_SIZE:-${VLLM_TP_SIZE}}"
    export WORKER_CUDA_VISIBLE_DEVICES="${WORKER_CUDA_VISIBLE_DEVICES:-${MODEL_CUDA_VISIBLE_DEVICES}}"
    export WORKER_VLLM_TP_SIZE="${WORKER_VLLM_TP_SIZE:-${VLLM_TP_SIZE}}"

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

    # NCCL safety nets (needed by _vllm_runtime.sh pre-flight on this cluster)
    export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
    [[ "${NCCL_P2P_DISABLE:-}" == "1" ]] && unset NCCL_P2P_DISABLE
    export NCCL_NET_GDR_DISABLE="${NCCL_NET_GDR_DISABLE:-1}"

    # Unique tag so re-run outputs don't collide with any half-written
    # files from the original runner attempt.
    export EVAL_TAG="${EVAL_TAG:-post_train_rerun_$(date +%m%d_%H%M)}"
    export LOG_DIR="${LOG_DIR:-${RUN_DIR}/supplement_logs}"
    mkdir -p "${LOG_DIR}"

    if [[ "${ROLE}" == "master_only" ]]; then
      # Still 2node script, but WORKER_SSH_TARGET empty + dispatch=ssh =>
      # G{2,3}_2node.sh falls back to single-node G{2,3}.sh inside the
      # same process. That way --master-only is a no-worker dry run.
      unset POSTEVAL_WORKER_DISPATCH
      WORKER_SSH_TARGET=""
    else
      # Real 2-node rendezvous: master writes request files into RUN_DIR,
      # worker side (sibling invocation of this script) picks them up.
      export POSTEVAL_WORKER_DISPATCH="rendezvous"
      WORKER_SSH_TARGET=""
    fi
    export WORKER_SSH_TARGET

    export RUN_DIR MODEL_PATH

    POST_EVAL_SCRIPT="${REPO_ROOT}/scripts/supplement_2rounds/${VARIANT_UPPER}_2node.sh"
    if [[ ! -f "${POST_EVAL_SCRIPT}" ]]; then
      echo "[master] ERROR: ${POST_EVAL_SCRIPT} not found" >&2
      exit 2
    fi

    echo "[master] launching ${POST_EVAL_SCRIPT} (dispatch=${POSTEVAL_WORKER_DISPATCH:-ssh-fallback}, worker=${ROLE})"
    set +e
    bash "${POST_EVAL_SCRIPT}" "${RUN_DIR}"
    EVAL_RC=$?
    set -e
    if (( EVAL_RC != 0 )); then
      echo "[master] ERROR: post-eval returned ${EVAL_RC}"
    else
      echo "[master] post-eval OK"
    fi

    # Archive (master only). Skip on failure unless DO_ARCHIVE=true AND
    # the user really wants a partial archive (not supported here;
    # simpler to say: only archive on success).
    if [[ "${DO_ARCHIVE}" == "true" && "${EVAL_RC}" == "0" ]]; then
      case "${VARIANT}" in
        g1) ARCHIVE_OUTPUT_ROOT="${ARCHIVE_OUTPUT_ROOT:-/mnt/data/ebft-teacher-distribution/outputs_g1_0.99}" ;;
        g2) ARCHIVE_OUTPUT_ROOT="${ARCHIVE_OUTPUT_ROOT:-/mnt/data/ebft-teacher-distribution/outputs_g2_0.99}" ;;
        g3) ARCHIVE_OUTPUT_ROOT="${ARCHIVE_OUTPUT_ROOT:-/mnt/data/ebft-teacher-distribution/outputs_g3_0.99}" ;;
      esac
      if [[ -n "${ARCHIVE_OUTPUT_ROOT}" ]]; then
        mkdir -p "${ARCHIVE_OUTPUT_ROOT}"
        _target="${ARCHIVE_OUTPUT_ROOT}/$(basename "${RUN_DIR}")"
        [[ -e "${_target}" ]] && _target="${_target}_$(date +%m%d_%H%M%S)"
        echo "[master] archiving: mv '${RUN_DIR}' '${_target}'"
        mv "${RUN_DIR}" "${_target}"
        echo "[master] archive done: ${_target}"
      fi
    elif [[ "${DO_ARCHIVE}" == "true" ]]; then
      echo "[master] archive skipped (EVAL_RC=${EVAL_RC}). To archive manually later: mv '${RUN_DIR}' '${ARCHIVE_OUTPUT_ROOT:-<target>}/'."
    fi

    exit "${EVAL_RC}"
    ;;

  worker)
    # Worker pod. Sit in the posteval watcher loop and fulfill requests
    # that master writes into RUN_DIR/dlc_rendezvous/post_eval/.
    export POSTEVAL_WORKER_CUDA_VISIBLE_DEVICES="${POSTEVAL_WORKER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    export CUDA_VISIBLE_DEVICES="${POSTEVAL_WORKER_CUDA_VISIBLE_DEVICES}"
    export MODEL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    IFS=',' read -r -a _WORKER_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    export VLLM_TP_SIZE="${POSTEVAL_WORKER_VLLM_TP_SIZE:-${#_WORKER_GPUS[@]}}"

    # NCCL safety nets (also needed here -- _vllm_runtime.sh would set
    # them, but we re-set here before the source so pre-flight dump
    # reflects the same values).
    export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
    [[ "${NCCL_P2P_DISABLE:-}" == "1" ]] && unset NCCL_P2P_DISABLE
    export NCCL_NET_GDR_DISABLE="${NCCL_NET_GDR_DISABLE:-1}"

    # Load vLLM runtime helpers (retry / orphan-killer / drain) and the
    # rendezvous protocol in this shell, then block in watcher.
    _rdv_helper="${REPO_ROOT}/scripts/supplement_2rounds/_rendezvous_dlc.sh"
    _vllm_runtime="${REPO_ROOT}/scripts/supplement_2rounds/_vllm_runtime.sh"
    for _f in "${_vllm_runtime}" "${_rdv_helper}"; do
      [[ -f "${_f}" ]] || { echo "[worker] ERROR: required helper missing: ${_f}"; exit 2; }
    done

    echo "[worker rank=${DLC_NODE_RANK:-?}] sourcing vLLM runtime + rendezvous helpers"
    # shellcheck disable=SC1090
    source "${_vllm_runtime}"
    # shellcheck disable=SC1090
    source "${_rdv_helper}"
    rdv_init_root "${RUN_DIR}"
    echo "[worker] entering posteval watcher loop; will exit when master writes all_done.marker or master_abort.marker"
    set +e
    posteval_worker_watch
    _rc=$?
    set -e
    echo "[worker] posteval_worker_watch exited rc=${_rc}"
    exit "${_rc}"
    ;;

  *)
    echo "[rerun_posteval_2nodes] unknown role '${ROLE}'"; exit 3 ;;
esac
