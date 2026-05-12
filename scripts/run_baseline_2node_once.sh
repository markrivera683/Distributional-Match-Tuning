#!/usr/bin/env bash
# Two-node launcher for the baseline two-round vLLM eval.
#
# WHAT THIS SCRIPT DOES:
#   - No training, no teacher, no Ray. Pure inference-only baseline number,
#     same protocol as run_baseline.sh but sharded across two nodes for
#     ~2x wall-time speedup on the 5328-prompt 16k/32k two-round eval.
#   - DLC / DSW / single-node mode autodetection (mirrors
#     run_G3_rebase_2node_once.sh).
#   - In single-node mode falls back to scripts/run_baseline.sh.
#
# DEFAULT MODEL:
#   /mnt/data/models/Qwen3.5-4B/  (override via MODEL_PATH=...)
#
# DEPLOYMENT MODES:
#   - DSW 2-node SSH:  user explicitly sets HEAD_NODE / WORKER_NODE,
#                       head pod uses ssh to bring up worker-side eval shard.
#   - DLC multi-pod:   PAI DLC starts master + worker pod with the same
#                       startup command and injects RANK / WORLD_SIZE /
#                       MASTER_ADDR per pod. Both pods run THIS launcher;
#                       RANK=0 is master (writes rendezvous request),
#                       RANK>0 is worker (parks in posteval watcher).
#                       SSH is NOT used.
#   - Single-node:     no HEAD_NODE / WORKER_NODE, no DLC env vars.
#                       Delegates to scripts/run_baseline.sh.
#
# NOTABLE CHOICES vs run_G3_rebase_2node_once.sh:
#   - No teacher launch, no ray cluster, no ray job submit. Worker pod's
#     bootstrap immediately enters the post-eval rendezvous watcher.
#   - dispatch decision is persisted to ${RUN_DIR}/dlc_dispatch.env early,
#     so an AIMaster pod restart (which can drop DLC env vars in the
#     newly-spawned launcher process) doesn't make the rerun fall back
#     to ssh dispatch (the bug that broke g3_dlci89a8a69v5nhm post-eval).
#
# Usage:
#   bash scripts/run_baseline_2node_once.sh                              # DLC autodetect
#   HEAD_NODE=node0 WORKER_NODE=node1 bash scripts/run_baseline_2node_once.sh
#   MODEL_PATH=/mnt/data/.../outputs/<run>/model bash scripts/run_baseline_2node_once.sh   # eval a trained ckpt

set -euo pipefail

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "$csv" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"${csv}"
}

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
      echo "[ERROR] failed to resolve IPv4 for host: ${host}" >&2
      exit 1
    fi
    sleep "${resolve_retry_seconds}"
    waited=$((waited + resolve_retry_seconds))
  done
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[ERROR] required command not found: ${cmd}"
    exit 1
  fi
}

# =====================================================================
# 1) DEPLOYMENT MODE DETECTION
# =====================================================================
HEAD_NODE="${HEAD_NODE:-}"
WORKER_NODE="${WORKER_NODE:-}"
HEAD_NODE_IP="${HEAD_NODE_IP:-}"
WORKER_NODE_IP="${WORKER_NODE_IP:-}"
WORKER_SSH_HOST="${WORKER_SSH_HOST:-}"
SSH_USER="${SSH_USER:-}"
SSH_OPTS="${SSH_OPTS:-}"

SINGLE_NODE_MODE="false"
DLC_MODE="false"
DLC_NODE_RANK=""
DLC_MASTER_ADDR=""
DLC_WORLD_SIZE="${WORLD_SIZE:-${PET_WORLD_SIZE:-1}}"

if [[ -n "${PET_NODE_RANK:-}" ]]; then
  DLC_MODE="true"
  DLC_NODE_RANK="${PET_NODE_RANK}"
  DLC_MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
elif [[ -n "${RANK:-}" && -n "${MASTER_ADDR:-}" && "${DLC_WORLD_SIZE:-1}" -gt 1 ]]; then
  DLC_MODE="true"
  DLC_NODE_RANK="${RANK}"
  DLC_MASTER_ADDR="${MASTER_ADDR}"
fi

if [[ "${DLC_MODE}" == "true" ]]; then
  if [[ -z "${DLC_MASTER_ADDR}" ]]; then
    echo "[ERROR] DLC mode detected (RANK=${DLC_NODE_RANK} WORLD_SIZE=${DLC_WORLD_SIZE})"
    echo "        but MASTER_ADDR is empty. Cannot route rendezvous target."
    exit 1
  fi
  if [[ -z "${HEAD_NODE}" && -z "${WORKER_NODE}" ]]; then
    HEAD_NODE="${DLC_MASTER_ADDR}"
    WORKER_NODE="dlc-rank-${DLC_NODE_RANK}-pod"   # symbolic; SSH path is never used
  fi
  echo "[INFO] DLC multi-pod mode: rank=${DLC_NODE_RANK} world_size=${DLC_WORLD_SIZE} master=${DLC_MASTER_ADDR}"
elif [[ -z "${HEAD_NODE}" && -z "${WORKER_NODE}" ]]; then
  SINGLE_NODE_MODE="true"
  HEAD_NODE="$(hostname)"
  WORKER_NODE="${HEAD_NODE}"
  echo "[INFO] single-node mode: HEAD_NODE=WORKER_NODE=${HEAD_NODE}"
elif [[ -z "${HEAD_NODE}" || -z "${WORKER_NODE}" ]]; then
  echo "[ERROR] HEAD_NODE / WORKER_NODE must both be set (DSW 2-node ssh)"
  echo "        or both be unset (single-node / DLC autodetect)."
  exit 1
fi

SKIP_SSH_BOOTSTRAP="false"
if [[ "${SINGLE_NODE_MODE}" == "true" || "${DLC_MODE}" == "true" ]]; then
  SKIP_SSH_BOOTSTRAP="true"
fi

# =====================================================================
# 2) PATHS / VENV / MODEL / DATA
# =====================================================================
REPO_ROOT="${REPO_ROOT:-/mnt/data/ebft-distribution-new/code}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-4B/}"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"

# Venvs live on local ext4 (ossfs2 can't host venv symlinks). See
# scripts/setup_env.sh for the bootstrap that creates and snapshots them.
TEACHER_VENV="${TEACHER_VENV:-/mnt/workspace/venvs/.teacherVenv}"
STUDENT_VENV="${STUDENT_VENV:-/mnt/workspace/venvs/.venv}"
ANALYSIS_VENV="${ANALYSIS_VENV:-${STUDENT_VENV}}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED=1

# NCCL safety nets (mirrors run_G3_rebase_2node_once.sh; both training and
# eval need NVLink intra-node + host-staged RoCE inter-node to avoid the
# mlx5 QP fatal we saw on PAI-DLC).
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
[[ "${NCCL_P2P_DISABLE:-}" == "1" ]] && unset NCCL_P2P_DISABLE
export NCCL_NET_GDR_DISABLE="${NCCL_NET_GDR_DISABLE:-1}"

# Make sure the student venv's binaries (notably python/pip) are first on
# PATH. Without this the launcher fails on a fresh DLC pod (no setup_env
# auto-activate).
if [[ -d "${STUDENT_VENV}/bin" ]]; then
  export PATH="${STUDENT_VENV}/bin:${PATH}"
fi

# =====================================================================
# 3) IP RESOLUTION (DLC: worker IP comes via rendezvous file; placeholder)
# =====================================================================
HEAD_NODE_IP="${HEAD_NODE_IP:-$(resolve_host_ip "${HEAD_NODE}")}"
if [[ "${DLC_MODE}" == "true" ]]; then
  WORKER_NODE_IP="${WORKER_NODE_IP:-${HEAD_NODE_IP}}"   # placeholder
else
  WORKER_NODE_IP="${WORKER_NODE_IP:-$(resolve_host_ip "${WORKER_NODE}")}"
fi
WORKER_SSH_HOST="${WORKER_SSH_HOST:-${WORKER_NODE_IP}}"

if [[ -n "${SSH_USER}" ]]; then
  WORKER_SSH_TARGET="${SSH_USER}@${WORKER_SSH_HOST}"
else
  WORKER_SSH_TARGET="${WORKER_SSH_HOST}"
fi

CURRENT_HOSTNAME="$(hostname)"
CURRENT_HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
CURRENT_HOST_IPS="$(hostname -I 2>/dev/null || true)"

# Head-only check: only enforce in DSW ssh mode. In DLC mode RANK>0 pods
# legitimately run this script (they take the worker bootstrap branch).
if [[ "${DLC_MODE}" != "true" && "${SINGLE_NODE_MODE}" != "true" ]]; then
  if [[ "${CURRENT_HOSTNAME}" != "${HEAD_NODE}" && "${CURRENT_HOSTNAME_SHORT}" != "${HEAD_NODE}" ]]; then
    case " ${CURRENT_HOST_IPS} " in
      *" ${HEAD_NODE_IP} "*) ;;
      *)
        echo "[ERROR] this launcher must be executed only on the head node."
        echo "        current host: ${CURRENT_HOSTNAME}"
        echo "        expected head: ${HEAD_NODE} (${HEAD_NODE_IP})"
        exit 1
        ;;
    esac
  fi
fi

# =====================================================================
# 4) GPU / VLLM (each node uses all 8 GPUs at TP=8 for its dataset shard)
# =====================================================================
HEAD_CUDA_VISIBLE_DEVICES="${HEAD_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
WORKER_CUDA_VISIBLE_DEVICES="${WORKER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
HEAD_VLLM_TP_SIZE="${HEAD_VLLM_TP_SIZE:-$(count_csv_items "${HEAD_CUDA_VISIBLE_DEVICES}")}"
WORKER_VLLM_TP_SIZE="${WORKER_VLLM_TP_SIZE:-$(count_csv_items "${WORKER_CUDA_VISIBLE_DEVICES}")}"
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-${HEAD_CUDA_VISIBLE_DEVICES}}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-${HEAD_VLLM_TP_SIZE}}"

POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
FIRST_PASS_MAX_NEW_TOKENS="${FIRST_PASS_MAX_NEW_TOKENS:-16384}"
SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS:-32768}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY:-1.0}"
POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N:-1}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_PROGRESS_BATCH_SIZE="${VLLM_PROGRESS_BATCH_SIZE:-256}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
VLLM_SEED="${VLLM_SEED:-1234}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-}"

# 8-hour worker watcher timeout (safe upper bound for 5328 prompts at
# 16k+32k two-round eval; default 3h was empirically too tight).
POSTEVAL_RDV_WORKER_TIMEOUT="${POSTEVAL_RDV_WORKER_TIMEOUT:-28800}"
POSTEVAL_RDV_MASTER_TIMEOUT="${POSTEVAL_RDV_MASTER_TIMEOUT:-28800}"

POST_EVAL_SCRIPT="${POST_EVAL_SCRIPT:-${REPO_ROOT}/scripts/supplement_2rounds/baseline_2node.sh}"
POST_EVAL_TAG="${POST_EVAL_TAG:-2rounds_vllm}"

# =====================================================================
# 5) RUN_NAME / RUN_DIR  (DLC: derive jobid so master/worker agree on dir)
# =====================================================================
if [[ "${DLC_MODE}" == "true" && -z "${RUN_NAME:-}" ]]; then
  _dlc_job_id="$(hostname | sed -E 's/^(dlc[a-z0-9]+)-(master|worker)-[0-9]+$/\1/' || true)"
  if [[ -n "${_dlc_job_id}" && "${_dlc_job_id}" != "$(hostname)" ]]; then
    RUN_NAME="baseline_${_dlc_job_id}"
  fi
fi
RUN_NAME="${RUN_NAME:-baseline_2node_$(date +%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
POST_EVAL_LOG_DIR="${POST_EVAL_LOG_DIR:-${RUN_DIR}/supplement_logs}"
SCRIPT_SOURCE_PATH="${BASH_SOURCE[0]:-$0}"
TRAIN_CONFIG_SNAPSHOT_PATH="${RUN_DIR}/run_context.env"
TRAIN_CONFIG_SUMMARY_PATH="${RUN_DIR}/run_summary.txt"
LAUNCHER_SNAPSHOT_PATH="${RUN_DIR}/launcher_snapshot.sh"
DLC_DISPATCH_ENV_PATH="${RUN_DIR}/dlc_dispatch.env"
mkdir -p "${RUN_DIR}" "${POST_EVAL_LOG_DIR}"

# =====================================================================
# 6) ARCHIVE knobs
# =====================================================================
ARCHIVE_OUTPUTS_AFTER_RUN="${ARCHIVE_OUTPUTS_AFTER_RUN:-true}"
ARCHIVE_OUTPUT_ROOT="${ARCHIVE_OUTPUT_ROOT:-/mnt/data/ebft-teacher-distribution/outputs_baseline_2node}"

# =====================================================================
# 7) DISPATCH DECISION (persisted to OSS so AIMaster restarts can recover)
# =====================================================================
# Why we persist: in run_G3_rebase_2node_once.sh the dispatch decision
# was made AFTER a 31h training stage. If AIMaster restarted the launcher
# pod mid-run, the new launcher process started without DLC env vars
# (PAI does not always re-inject WORLD_SIZE / RANK on restart), so
# `${DLC_MODE}` re-detected as false, dispatch silently fell back to ssh,
# and post-eval failed with `ssh: connection refused` against the worker
# pod (which has no sshd). Persisting the decision to an OSS-shared file
# lets the recovery path read it back instead of relying on shell vars.
write_dlc_dispatch_env() {
  local dispatch="$1"
  local target="$2"
  local mode="$3"
  # ossfs2 rejects O_TRUNC of an existing file (EINVAL, fuse mis-reports
  # as ENOSPC); pre-delete to force the open() to take the O_CREAT path.
  rm -f "${DLC_DISPATCH_ENV_PATH}" 2>/dev/null || true
  cat > "${DLC_DISPATCH_ENV_PATH}" <<EOF
# Auto-generated by run_baseline_2node_once.sh
# UTC: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
DEPLOY_MODE=${mode}
POSTEVAL_WORKER_DISPATCH=${dispatch}
WORKER_SSH_TARGET=${target}
EOF
}

if [[ "${DLC_MODE}" == "true" ]]; then
  write_dlc_dispatch_env "rendezvous" "" "dlc"
elif [[ "${SINGLE_NODE_MODE}" == "true" ]]; then
  write_dlc_dispatch_env "ssh" "" "single"
else
  write_dlc_dispatch_env "ssh" "${WORKER_SSH_TARGET}" "dsw"
fi

# =====================================================================
# 8) PRE-FLIGHT CHECKS
# =====================================================================
require_cmd curl
if [[ "${SKIP_SSH_BOOTSTRAP}" != "true" ]]; then
  require_cmd ssh
fi
[[ -d "${REPO_ROOT}" ]] || { echo "[ERROR] REPO_ROOT not found: ${REPO_ROOT}"; exit 1; }
[[ -e "${MODEL_PATH}" ]] || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 1; }
[[ -e "${EVAL_DATA}" ]] || { echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"; exit 1; }
[[ -x "${TEACHER_PYTHON_BIN}" ]] || { echo "[ERROR] TEACHER_PYTHON_BIN not executable: ${TEACHER_PYTHON_BIN}"; exit 1; }
[[ -x "${ANALYSIS_PYTHON_BIN}" ]] || { echo "[ERROR] ANALYSIS_PYTHON_BIN not executable: ${ANALYSIS_PYTHON_BIN}"; exit 1; }
[[ -f "${POST_EVAL_SCRIPT}" ]] || { echo "[ERROR] POST_EVAL_SCRIPT not found: ${POST_EVAL_SCRIPT}"; exit 1; }

# =====================================================================
# 9) METADATA SNAPSHOT
# =====================================================================
write_run_metadata() {
  local vars=(
    RUN_NAME OUTPUT_ROOT RUN_DIR POST_EVAL_LOG_DIR
    HEAD_NODE HEAD_NODE_IP WORKER_NODE WORKER_NODE_IP WORKER_SSH_HOST SSH_USER SSH_OPTS
    DLC_MODE DLC_NODE_RANK DLC_MASTER_ADDR DLC_WORLD_SIZE SINGLE_NODE_MODE SKIP_SSH_BOOTSTRAP
    HEAD_CUDA_VISIBLE_DEVICES WORKER_CUDA_VISIBLE_DEVICES MODEL_CUDA_VISIBLE_DEVICES
    HEAD_VLLM_TP_SIZE WORKER_VLLM_TP_SIZE VLLM_TP_SIZE
    REPO_ROOT MODEL_PATH EVAL_DATA TEACHER_VENV STUDENT_VENV ANALYSIS_VENV
    TEACHER_PYTHON_BIN ANALYSIS_PYTHON_BIN
    POST_EVAL_SCRIPT POST_EVAL_TAG
    POST_EVAL_MAX_SAMPLES POST_EVAL_PROMPT_MAX_LEN
    FIRST_PASS_MAX_NEW_TOKENS SECOND_PASS_MAX_NEW_TOKENS
    POST_EVAL_TEMPERATURE POST_EVAL_TOP_P POST_EVAL_REPETITION_PENALTY POST_EVAL_BEST_OF_N
    VLLM_MAX_NUM_SEQS VLLM_PROGRESS_BATCH_SIZE VLLM_ENABLE_PREFIX_CACHING
    VLLM_GPU_MEMORY_UTILIZATION VLLM_SEED INPUT_TEMPLATE
    POSTEVAL_RDV_WORKER_TIMEOUT POSTEVAL_RDV_MASTER_TIMEOUT
    NCCL_P2P_LEVEL NCCL_NET_GDR_DISABLE
    ARCHIVE_OUTPUTS_AFTER_RUN ARCHIVE_OUTPUT_ROOT
  )

  cp -f "${SCRIPT_SOURCE_PATH}" "${LAUNCHER_SNAPSHOT_PATH}" 2>/dev/null || true
  rm -f "${TRAIN_CONFIG_SNAPSHOT_PATH}" 2>/dev/null || true
  {
    echo "# Auto-generated run context snapshot (baseline 2-node)"
    echo "# UTC: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    for name in "${vars[@]}"; do
      printf "%s=%q\n" "${name}" "${!name-}"
    done
  } > "${TRAIN_CONFIG_SNAPSHOT_PATH}"

  rm -f "${TRAIN_CONFIG_SUMMARY_PATH}" 2>/dev/null || true
  {
    echo "run_name: ${RUN_NAME}"
    echo "run_dir: ${RUN_DIR}"
    echo "model_path: ${MODEL_PATH}"
    echo "eval_data: ${EVAL_DATA}"
    echo "post_eval_script: ${POST_EVAL_SCRIPT}"
    echo "post_eval_max_samples: ${POST_EVAL_MAX_SAMPLES}"
    echo "first_pass_max_new_tokens: ${FIRST_PASS_MAX_NEW_TOKENS}"
    echo "second_pass_max_new_tokens: ${SECOND_PASS_MAX_NEW_TOKENS}"
    echo "deploy_mode: $(grep -E '^DEPLOY_MODE=' "${DLC_DISPATCH_ENV_PATH}" 2>/dev/null | cut -d= -f2 || echo unknown)"
    echo "archive_output_root: ${ARCHIVE_OUTPUT_ROOT}"
    echo "launcher_snapshot: ${LAUNCHER_SNAPSHOT_PATH}"
  } > "${TRAIN_CONFIG_SUMMARY_PATH}"
}

archive_run_outputs() {
  local target_root="$1"
  local target_dir
  local old_run_dir="${RUN_DIR}"

  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[archive] skip: RUN_DIR not found: ${RUN_DIR}"
    return 0
  fi

  mkdir -p "${target_root}"
  target_dir="${target_root}/$(basename "${RUN_DIR}")"
  if [[ -e "${target_dir}" ]]; then
    target_dir="${target_dir}_$(date +%m%d_%H%M%S)"
  fi

  echo "[archive] moving run outputs to: ${target_dir}"
  mv "${RUN_DIR}" "${target_dir}"
  RUN_DIR="${target_dir}"
  POST_EVAL_LOG_DIR="${RUN_DIR}/supplement_logs"
  TRAIN_CONFIG_SNAPSHOT_PATH="${RUN_DIR}/run_context.env"
  TRAIN_CONFIG_SUMMARY_PATH="${RUN_DIR}/run_summary.txt"
  LAUNCHER_SNAPSHOT_PATH="${RUN_DIR}/launcher_snapshot.sh"
  DLC_DISPATCH_ENV_PATH="${RUN_DIR}/dlc_dispatch.env"
  write_run_metadata
}

write_final_status() {
  rm -f "${RUN_DIR}/final_status.env" 2>/dev/null || true
  {
    echo "# Auto-generated final status"
    echo "# UTC: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    printf "EVAL_RC=%q\n" "${EVAL_RC:-0}"
    printf "ARCHIVE_RC=%q\n" "${ARCHIVE_RC:-0}"
    printf "FINAL_RC=%q\n" "${FINAL_RC:-0}"
    printf "RUN_DIR=%q\n" "${RUN_DIR:-}"
    printf "POST_EVAL_LOG_DIR=%q\n" "${POST_EVAL_LOG_DIR:-}"
  } > "${RUN_DIR}/final_status.env"
}

write_run_metadata

# =====================================================================
# 10) SINGLE-NODE FALLBACK -> delegate to scripts/run_baseline.sh
# =====================================================================
if [[ "${SINGLE_NODE_MODE}" == "true" ]]; then
  echo "[INFO] single-node mode -> delegating to scripts/run_baseline.sh"
  export REPO_ROOT MODEL_PATH EVAL_DATA
  export TEACHER_VENV STUDENT_VENV ANALYSIS_VENV
  export TEACHER_PYTHON_BIN ANALYSIS_PYTHON_BIN
  export MODEL_CUDA_VISIBLE_DEVICES VLLM_TP_SIZE VLLM_GPU_MEMORY_UTILIZATION
  export POST_EVAL_MAX_SAMPLES POST_EVAL_PROMPT_MAX_LEN
  export FIRST_PASS_MAX_NEW_TOKENS SECOND_PASS_MAX_NEW_TOKENS
  export POST_EVAL_TEMPERATURE POST_EVAL_TOP_P
  export POST_EVAL_REPETITION_PENALTY POST_EVAL_BEST_OF_N
  export VLLM_MAX_NUM_SEQS VLLM_PROGRESS_BATCH_SIZE VLLM_ENABLE_PREFIX_CACHING VLLM_SEED
  export INPUT_TEMPLATE RUN_NAME RUN_DIR OUTPUT_ROOT
  exec bash "${REPO_ROOT}/scripts/run_baseline.sh"
fi

# =====================================================================
# 11) DLC WORKER POD ENTRY POINT (rank > 0)
# =====================================================================
# In DLC multi-pod mode the worker pod's only job is to park in the
# post-eval rendezvous watcher. No teacher, no ray, no training -- much
# simpler than the G3_2node worker bootstrap (which had to launch 6
# teacher vLLM workers first and then ray-join, and only entered the
# watcher post-training, exposing the env-loss bug).
dlc_worker_bootstrap_eval_only() {
  echo "================================================================"
  echo "[DLC worker rank=${DLC_NODE_RANK}] starting on $(hostname)"
  echo "[DLC worker rank=${DLC_NODE_RANK}] entering post-eval rendezvous watcher"

  # Pick our routable IP and write to rendezvous file (master uses this
  # to know we exist; the dispatch path itself does not need the IP).
  local my_ip
  my_ip="$(hostname -I 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i !~ /^127\./ && $i !~ /:/) {print $i; exit}}')"
  if [[ -z "${my_ip}" ]]; then
    echo "[DLC worker] ERROR: could not determine own IP via 'hostname -I'"
    exit 1
  fi
  echo "[DLC worker] my IP: ${my_ip}"

  local rdv_dir="${RUN_DIR}/dlc_rendezvous"
  mkdir -p "${rdv_dir}"
  local ip_file="${rdv_dir}/worker_${DLC_NODE_RANK}_ip.txt"
  rm -f "${ip_file}" 2>/dev/null || true
  echo "${my_ip}" > "${ip_file}"
  echo "[DLC worker] wrote IP to ${ip_file}"

  # IP keepalive: re-write IP every 5s. See run_G3_rebase_2node_once.sh
  # comments for why this is needed (master may rm -f stale IP files at
  # startup, and the worker pod can boot before the master).
  (
    while :; do
      printf '%s\n' "${my_ip}" > "${ip_file}" 2>/dev/null || true
      sleep 5
    done
  ) &
  local _ip_keepalive_pid=$!
  echo "[DLC worker] IP keepalive started (pid=${_ip_keepalive_pid}, interval=5s)"

  # Cleanup trap.
  _dlc_worker_eval_cleanup() {
    if [[ -n "${_ip_keepalive_pid:-}" ]]; then
      kill "${_ip_keepalive_pid}" 2>/dev/null || true
    fi
  }
  trap _dlc_worker_eval_cleanup EXIT INT TERM

  # Force all 8 of this pod's GPUs into vLLM's CUDA_VISIBLE_DEVICES.
  export CUDA_VISIBLE_DEVICES="${POSTEVAL_WORKER_CUDA_VISIBLE_DEVICES:-${WORKER_CUDA_VISIBLE_DEVICES}}"
  export MODEL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  local _visible_count
  _visible_count="$(count_csv_items "${CUDA_VISIBLE_DEVICES}")"
  export VLLM_TP_SIZE="${POSTEVAL_WORKER_VLLM_TP_SIZE:-${_visible_count}}"
  export REPO_ROOT TEACHER_VENV
  export TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
  export PROGRESS_HELPER="${PROGRESS_HELPER:-${REPO_ROOT}/scripts/supplement/vllm_generate_progress.py}"
  export POSTEVAL_RDV_WORKER_TIMEOUT POSTEVAL_RDV_MASTER_TIMEOUT
  export MODEL_PATH

  local _vllm_runtime_path="${REPO_ROOT}/scripts/supplement_2rounds/_vllm_runtime.sh"
  local _rdv_helper_path="${REPO_ROOT}/scripts/supplement_2rounds/_rendezvous_dlc.sh"
  if [[ ! -f "${_rdv_helper_path}" ]]; then
    echo "[DLC worker rank=${DLC_NODE_RANK}] post-eval rendezvous helper missing: ${_rdv_helper_path}"
    exit 1
  fi
  echo "[DLC worker rank=${DLC_NODE_RANK}] sourcing vLLM runtime helpers"
  # shellcheck disable=SC1090
  source "${_vllm_runtime_path}"
  # shellcheck disable=SC1090
  source "${_rdv_helper_path}"
  rdv_init_root "${RUN_DIR}"
  echo "[DLC worker rank=${DLC_NODE_RANK}] entering posteval_worker_watch (timeout ${POSTEVAL_RDV_WORKER_TIMEOUT}s)"
  set +e
  posteval_worker_watch
  local _rdv_rc=$?
  set -e
  echo "[DLC worker rank=${DLC_NODE_RANK}] watcher exited rc=${_rdv_rc}"
  exit "${_rdv_rc}"
}

if [[ "${DLC_MODE}" == "true" && "${DLC_NODE_RANK}" -gt 0 ]]; then
  dlc_worker_bootstrap_eval_only
fi

# =====================================================================
# 12) DLC MASTER: wait for worker IP file (informational only;
#     dispatch goes through rendezvous, not ssh, so we don't strictly
#     need the IP. But we still wait so launcher logs show worker is up.)
# =====================================================================
if [[ "${DLC_MODE}" == "true" ]]; then
  rdv_dir="${RUN_DIR}/dlc_rendezvous"
  mkdir -p "${rdv_dir}"
  rm -f "${rdv_dir}"/worker_*_ip.txt 2>/dev/null || true
  echo "[DLC master] waiting for worker pod IP at ${rdv_dir}/worker_1_ip.txt..."
  waited=0
  worker_ip_wait_seconds="${DLC_WORKER_IP_WAIT_SECONDS:-480}"
  while [[ ! -s "${rdv_dir}/worker_1_ip.txt" ]]; do
    sleep 3
    waited=$((waited + 3))
    if (( waited >= worker_ip_wait_seconds )); then
      echo "[DLC master] ERROR: worker pod IP not seen in ${worker_ip_wait_seconds}s"
      exit 1
    fi
  done
  WORKER_NODE_IP="$(tr -d '[:space:]' < "${rdv_dir}/worker_1_ip.txt")"
  echo "[DLC master] worker IP: ${WORKER_NODE_IP} (after ${waited}s)"
fi

# =====================================================================
# 13) DSW MASTER: ssh connectivity check
# =====================================================================
if [[ "${SKIP_SSH_BOOTSTRAP}" != "true" ]]; then
  echo "[1/2] connectivity check to worker..."
  ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -lc 'hostname'" >/dev/null
fi

# =====================================================================
# 14) BANNER
# =====================================================================
echo "========== Baseline 2-node post-eval launcher =========="
echo "RUN_DIR:                    ${RUN_DIR}"
echo "MODEL_PATH:                 ${MODEL_PATH}"
echo "EVAL_DATA:                  ${EVAL_DATA}"
echo "HEAD_NODE / IP:             ${HEAD_NODE} / ${HEAD_NODE_IP}"
echo "WORKER_NODE / IP:           ${WORKER_NODE} / ${WORKER_NODE_IP}"
echo "Head GPUs / TP:             ${HEAD_CUDA_VISIBLE_DEVICES} / ${HEAD_VLLM_TP_SIZE}"
echo "Worker GPUs / TP:           ${WORKER_CUDA_VISIBLE_DEVICES} / ${WORKER_VLLM_TP_SIZE}"
echo "Deploy mode:                $(grep -E '^DEPLOY_MODE=' "${DLC_DISPATCH_ENV_PATH}" | cut -d= -f2)"
echo "Dispatch:                   $(grep -E '^POSTEVAL_WORKER_DISPATCH=' "${DLC_DISPATCH_ENV_PATH}" | cut -d= -f2)"
echo "Post-eval script:           ${POST_EVAL_SCRIPT}"
echo "Post-eval first/second:     ${FIRST_PASS_MAX_NEW_TOKENS}/${SECOND_PASS_MAX_NEW_TOKENS} tokens"
echo "Worker watcher timeout:     ${POSTEVAL_RDV_WORKER_TIMEOUT}s"
echo "Master rendezvous timeout:  ${POSTEVAL_RDV_MASTER_TIMEOUT}s"
echo "Archive after run:          ${ARCHIVE_OUTPUTS_AFTER_RUN} -> ${ARCHIVE_OUTPUT_ROOT}"
echo "========================================================"

# =====================================================================
# 15) RUN POST-EVAL  (re-load dispatch decision from OSS-shared file
#                     to recover from any AIMaster-induced env loss)
# =====================================================================
EVAL_RC=0
ARCHIVE_RC=0

# shellcheck disable=SC1090
source "${DLC_DISPATCH_ENV_PATH}"
echo "[dispatch] loaded from ${DLC_DISPATCH_ENV_PATH}: mode=${DEPLOY_MODE} dispatch=${POSTEVAL_WORKER_DISPATCH}"

# Export everything baseline_2node.sh expects.
export RUN_DIR MODEL_PATH EVAL_DATA REPO_ROOT
export TEACHER_VENV ANALYSIS_VENV
export TEACHER_PYTHON_BIN ANALYSIS_PYTHON_BIN
export MODEL_CUDA_VISIBLE_DEVICES VLLM_TP_SIZE VLLM_GPU_MEMORY_UTILIZATION
export HEAD_CUDA_VISIBLE_DEVICES WORKER_CUDA_VISIBLE_DEVICES
export HEAD_VLLM_TP_SIZE WORKER_VLLM_TP_SIZE
export POST_EVAL_MAX_SAMPLES POST_EVAL_PROMPT_MAX_LEN
export FIRST_PASS_MAX_NEW_TOKENS SECOND_PASS_MAX_NEW_TOKENS
export POST_EVAL_TEMPERATURE POST_EVAL_TOP_P
export POST_EVAL_REPETITION_PENALTY POST_EVAL_BEST_OF_N
export VLLM_MAX_NUM_SEQS VLLM_PROGRESS_BATCH_SIZE VLLM_ENABLE_PREFIX_CACHING VLLM_SEED
export INPUT_TEMPLATE
export LOG_DIR="${POST_EVAL_LOG_DIR}"
export EVAL_TAG="${POST_EVAL_TAG}"
export POSTEVAL_WORKER_DISPATCH WORKER_SSH_TARGET SSH_OPTS
export POSTEVAL_RDV_WORKER_TIMEOUT POSTEVAL_RDV_MASTER_TIMEOUT
export NCCL_P2P_LEVEL NCCL_NET_GDR_DISABLE

set +e
bash "${POST_EVAL_SCRIPT}" "${RUN_DIR}"
EVAL_RC=$?
set -e
if (( EVAL_RC != 0 )); then
  echo "[ERROR] post-eval failed with exit code ${EVAL_RC}; run outputs will still be archived."
fi

# =====================================================================
# 16) ARCHIVE
# =====================================================================
if [[ "${ARCHIVE_OUTPUTS_AFTER_RUN}" == "true" ]]; then
  set +e
  archive_run_outputs "${ARCHIVE_OUTPUT_ROOT}"
  ARCHIVE_RC=$?
  set -e
  if (( ARCHIVE_RC != 0 )); then
    echo "[ERROR] archiving run outputs failed with exit code ${ARCHIVE_RC}"
  fi
fi

# =====================================================================
# 17) FINAL STATUS
# =====================================================================
FINAL_RC=0
if (( EVAL_RC != 0 )); then
  FINAL_RC=${EVAL_RC}
elif (( ARCHIVE_RC != 0 )); then
  FINAL_RC=${ARCHIVE_RC}
fi

write_final_status

echo "[done] logs: ${RUN_DIR}"
exit "${FINAL_RC}"
