#!/usr/bin/env bash
# Two-node launcher for the G1 rebase training recipe.
#
# CONTROLLED VARIABLES vs run_G1_rebase.sh (single node):
#   Same RL algorithm (pointwise reward + cf_target_mode=single + frozen
#   critic + no teacher), same batch / lr / kl / etc. The only diff is:
#     - actor (+ ref colocate) on head node, 8 GPUs
#     - critic (+ reward colocate) on worker node, 8 GPUs
#     - effective student world = 16 GPUs across 2 nodes
#
# CONTROLLED VARIABLES vs run_G3_rebase_2node_once.sh:
#   G3 had teacher (qwen3.5-27b) launched as 6+6 vLLM workers per node,
#   plus G3 actor=2 / critic=2 student GPUs. G1 has NO teacher, so:
#     - no teacher pod launches
#     - no teacher api base / cache / prefetch
#     - all 8 GPUs per node go to student (actor on head, critic on worker)
#     - single Ray cluster across 2 nodes; only the student trains
#
# DEFAULT MODEL:
#   /mnt/data/models/Qwen3.5-4B/  (override via MODEL_PATH=...)
#
# DEPLOYMENT MODES (autodetected, mirrors run_G3_rebase_2node_once.sh):
#   - DSW 2-node SSH: HEAD_NODE / WORKER_NODE explicitly set; ssh used.
#   - DLC multi-pod:  PAI injects RANK / WORLD_SIZE / MASTER_ADDR; both
#                     pods run THIS launcher; rank=0 master, rank>0 worker.
#                     SSH NOT used; rendezvous + ray join only.
#   - Single-node:    no HEAD_NODE / WORKER_NODE / DLC env; delegates to
#                     scripts/run_G1_rebase.sh.
#
# DLC POST-EVAL DISPATCH BUG FIX:
#   In g3_dlci89a8a69v5nhm, post-eval went through ssh dispatch instead of
#   rendezvous because the DLC env vars (RANK / WORLD_SIZE) were dropped
#   when AIMaster restarted the launcher pod mid-run, so the train-job
#   completion path re-detected DLC_MODE=false. We now persist the
#   dispatch decision to OSS-shared ${RUN_DIR}/dlc_dispatch.env at the
#   very start, and re-source it after training so it survives an
#   AIMaster pod restart.

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
    echo "        but MASTER_ADDR is empty. Cannot route ray join target."
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
# 2) IP RESOLUTION (DLC: worker IP comes via rendezvous file later)
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
        echo "        worker node will be started remotely through ssh and must not run this script directly."
        exit 1
        ;;
    esac
  fi
fi

# =====================================================================
# 3) GPU LAYOUT (no teacher; 8 student GPUs per node)
# =====================================================================
HEAD_STUDENT_CUDA_VISIBLE_DEVICES="${HEAD_STUDENT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ "${SINGLE_NODE_MODE}" == "true" ]]; then
  WORKER_STUDENT_CUDA_VISIBLE_DEVICES="${WORKER_STUDENT_CUDA_VISIBLE_DEVICES:-}"
else
  WORKER_STUDENT_CUDA_VISIBLE_DEVICES="${WORKER_STUDENT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
fi

# 2-node layout: actor (+ ref colocate) on head 8 GPUs, critic (+ reward
# colocate) on worker 8 GPUs. Same train_batch_size as single-node G1
# (128) so optimization signal volume is identical and runs are
# directly comparable.
if [[ "${SINGLE_NODE_MODE}" == "true" ]]; then
  ACTOR_GPUS="${ACTOR_GPUS:-4}"
  CRITIC_GPUS="${CRITIC_GPUS:-4}"
else
  ACTOR_GPUS="${ACTOR_GPUS:-8}"
  CRITIC_GPUS="${CRITIC_GPUS:-8}"
fi
REF_GPUS="${REF_GPUS:-${ACTOR_GPUS}}"
REWARD_GPUS="${REWARD_GPUS:-${CRITIC_GPUS}}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
CRITIC_NUM_NODES="${CRITIC_NUM_NODES:-1}"
REF_NUM_NODES="${REF_NUM_NODES:-1}"
REWARD_NUM_NODES="${REWARD_NUM_NODES:-1}"

# =====================================================================
# 4) PATHS / VENV / MODEL / DATA
# =====================================================================
REPO_ROOT="${REPO_ROOT:-/mnt/data/ebft-distribution-new/code}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-4B/}"
DEFAULT_TRAIN_DATA="/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
TRAIN_DATA="${TRAIN_DATA:-${DEFAULT_TRAIN_DATA}}"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"
PROMPT_SPLIT="${PROMPT_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"

# Venvs live on local ext4 (ossfs2 can't host venv symlinks).
TEACHER_VENV="${TEACHER_VENV:-/mnt/workspace/venvs/.teacherVenv}"
STUDENT_VENV="${STUDENT_VENV:-/mnt/workspace/venvs/.venv}"
ANALYSIS_VENV="${ANALYSIS_VENV:-${STUDENT_VENV}}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"

# HF blobs go on persistent OSS (model weights survive container restart;
# downloads are tmp+rename, OSS-safe).
export HF_HOME="${HF_HOME:-/mnt/data/ebft-distribution-new/caches/hf}"
# Compile caches MUST be on local ext4: ossfs2 rejects "seek + write into
# existing file" with EINVAL.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/workspace/.torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/workspace/.triton_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED=1

# NCCL safety nets (mirrors run_G3_rebase_2node_once.sh):
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
[[ "${NCCL_P2P_DISABLE:-}" == "1" ]] && unset NCCL_P2P_DISABLE
export NCCL_NET_GDR_DISABLE="${NCCL_NET_GDR_DISABLE:-1}"

if [[ -d "${STUDENT_VENV}/bin" ]]; then
  export PATH="${STUDENT_VENV}/bin:${PATH}"
fi

# =====================================================================
# 5) TRAINING KNOBS — IDENTICAL to run_G1_rebase.sh single-node defaults
# =====================================================================
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE))}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-4}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-4}"
MICRO_REWARD_BATCH_SIZE="${MICRO_REWARD_BATCH_SIZE:-4}"

PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-384}"
CONTEXT_MAX_LEN="${CONTEXT_MAX_LEN:-8}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-8}"
STRIDE="${STRIDE:-8}"

NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
TARGET_STEPS="${TARGET_STEPS:-500}"
DEFAULT_MAX_SAMPLES="$((TARGET_STEPS * TRAIN_BATCH_SIZE / N_SAMPLES_PER_PROMPT / NUM_EPISODES / MAX_EPOCHS))"
MAX_SAMPLES="${MAX_SAMPLES:-${DEFAULT_MAX_SAMPLES}}"

ACTOR_LR="${ACTOR_LR:-1e-6}"
CRITIC_LR="${CRITIC_LR:-0.0}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-0.0}"

# =====================================================================
# 6) EVAL / CHECKPOINT
# =====================================================================
ONLINE_EVAL="${ONLINE_EVAL:-false}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-1}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-${GENERATE_MAX_LEN}}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_EVEN_COUNT="${SAVE_EVEN_COUNT:-10}"

# =====================================================================
# 7) POST-TRAINING TWO-ROUND EVAL
# =====================================================================
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
RUN_TWO_ROUND_EVAL="${RUN_TWO_ROUND_EVAL:-${EVAL_AFTER_TRAIN}}"
POST_EVAL_SCRIPT="${POST_EVAL_SCRIPT:-${REPO_ROOT}/scripts/supplement_2rounds/G1_2node.sh}"
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
VLLM_SEED="${VLLM_SEED:-1234}"
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-$(count_csv_items "${MODEL_CUDA_VISIBLE_DEVICES}")}"
POST_EVAL_TAG="${POST_EVAL_TAG:-post_train}"
POST_EVAL_LOG_DIR="${POST_EVAL_LOG_DIR:-}"

# 8-hour worker watcher timeout (override of upstream 3h default).
POSTEVAL_RDV_WORKER_TIMEOUT="${POSTEVAL_RDV_WORKER_TIMEOUT:-28800}"
POSTEVAL_RDV_MASTER_TIMEOUT="${POSTEVAL_RDV_MASTER_TIMEOUT:-28800}"

# =====================================================================
# 8) ARCHIVE
# =====================================================================
ARCHIVE_OUTPUTS_AFTER_RUN="${ARCHIVE_OUTPUTS_AFTER_RUN:-true}"
ARCHIVE_OUTPUT_ROOT="${ARCHIVE_OUTPUT_ROOT:-/mnt/data/ebft-teacher-distribution/outputs_g1_2node}"

# =====================================================================
# 9) RAY KNOBS
# =====================================================================
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_WAIT_SECONDS="${RAY_WAIT_SECONDS:-120}"

# =====================================================================
# 10) RUN_NAME / RUN_DIR
# =====================================================================
if [[ "${DLC_MODE}" == "true" && -z "${RUN_NAME:-}" ]]; then
  _dlc_job_id="$(hostname | sed -E 's/^(dlc[a-z0-9]+)-(master|worker)-[0-9]+$/\1/' || true)"
  if [[ -n "${_dlc_job_id}" && "${_dlc_job_id}" != "$(hostname)" ]]; then
    RUN_NAME="g1_${_dlc_job_id}"
  fi
fi
RUN_NAME="${RUN_NAME:-g1_rebase_2node_once_$(date +%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SAVE_PATH="${RUN_DIR}/model"
TB_DIR="${RUN_DIR}/tensorboard"
RAY_LOG_DIR="${RUN_DIR}/ray_logs"
PID_DIR="${RUN_DIR}/pids"
JOB_SCRIPT="${RUN_DIR}/run_train_once.sh"
JOB_LOG="${RUN_DIR}/ray_job.log"
POST_EVAL_LOG_DIR="${POST_EVAL_LOG_DIR:-${RUN_DIR}/supplement_logs}"
SCRIPT_SOURCE_PATH="${BASH_SOURCE[0]:-$0}"
TRAIN_CONFIG_SNAPSHOT_PATH="${RUN_DIR}/run_context.env"
TRAIN_CONFIG_SUMMARY_PATH="${RUN_DIR}/run_summary.txt"
LAUNCHER_SNAPSHOT_PATH="${RUN_DIR}/launcher_snapshot.sh"
DLC_DISPATCH_ENV_PATH="${RUN_DIR}/dlc_dispatch.env"
mkdir -p "${RUN_DIR}" "${SAVE_PATH}" "${TB_DIR}" "${RAY_LOG_DIR}" "${PID_DIR}" "${POST_EVAL_LOG_DIR}"

# =====================================================================
# 11) DISPATCH DECISION (persisted; recovery point for AIMaster restart)
# =====================================================================
write_dlc_dispatch_env() {
  local dispatch="$1"
  local target="$2"
  local mode="$3"
  rm -f "${DLC_DISPATCH_ENV_PATH}" 2>/dev/null || true
  cat > "${DLC_DISPATCH_ENV_PATH}" <<EOF
# Auto-generated by run_G1_rebase_2node_once.sh
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
# 12) PRE-FLIGHT CHECKS
# =====================================================================
require_cmd curl
require_cmd ray
if [[ "${SKIP_SSH_BOOTSTRAP}" != "true" ]]; then
  require_cmd ssh
fi
[[ -d "${REPO_ROOT}" ]] || { echo "[ERROR] REPO_ROOT not found: ${REPO_ROOT}"; exit 1; }
[[ -e "${MODEL_PATH}" ]] || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 1; }
[[ -e "${TRAIN_DATA}" ]] || { echo "[ERROR] TRAIN_DATA not found: ${TRAIN_DATA}"; exit 1; }
[[ -x "${STUDENT_PYTHON_BIN}" ]] || { echo "[ERROR] STUDENT_PYTHON_BIN not executable: ${STUDENT_PYTHON_BIN}"; exit 1; }

if [[ "${RUN_TWO_ROUND_EVAL}" == "true" ]]; then
  [[ -e "${EVAL_DATA}" ]] || { echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"; exit 1; }
  [[ -x "${TEACHER_PYTHON_BIN}" ]] || { echo "[ERROR] TEACHER_PYTHON_BIN not executable: ${TEACHER_PYTHON_BIN}"; exit 1; }
  [[ -x "${ANALYSIS_PYTHON_BIN}" ]] || { echo "[ERROR] ANALYSIS_PYTHON_BIN not executable: ${ANALYSIS_PYTHON_BIN}"; exit 1; }
  [[ -f "${POST_EVAL_SCRIPT}" ]] || { echo "[ERROR] POST_EVAL_SCRIPT not found: ${POST_EVAL_SCRIPT}"; exit 1; }
fi

# =====================================================================
# 13) BATCH SIZE VALIDATION
# =====================================================================
if (( TRAIN_BATCH_SIZE != N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE )); then
  echo "[ERROR] TRAIN_BATCH_SIZE must equal N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE"
  echo "        got ${TRAIN_BATCH_SIZE} vs ${N_SAMPLES_PER_PROMPT} * ${ROLLOUT_BATCH_SIZE}"
  exit 1
fi
if (( TRAIN_BATCH_SIZE % (MICRO_TRAIN_BATCH_SIZE * ACTOR_GPUS) != 0 )); then
  echo "[ERROR] train_batch_size % (micro_train_batch_size * actor_gpus) != 0"
  echo "        ${TRAIN_BATCH_SIZE} % (${MICRO_TRAIN_BATCH_SIZE} * ${ACTOR_GPUS}) != 0"
  exit 1
fi
if (( MICRO_TRAIN_BATCH_SIZE < N_SAMPLES_PER_PROMPT || MICRO_TRAIN_BATCH_SIZE % N_SAMPLES_PER_PROMPT != 0 )); then
  echo "[ERROR] MICRO_TRAIN_BATCH_SIZE must be >= N_SAMPLES_PER_PROMPT and divisible by it"
  exit 1
fi
if (( MICRO_ROLLOUT_BATCH_SIZE % N_SAMPLES_PER_PROMPT != 0 )); then
  echo "[ERROR] MICRO_ROLLOUT_BATCH_SIZE must be divisible by N_SAMPLES_PER_PROMPT"
  exit 1
fi

head_student_gpu_count="$(count_csv_items "${HEAD_STUDENT_CUDA_VISIBLE_DEVICES}")"
worker_student_gpu_count="$(count_csv_items "${WORKER_STUDENT_CUDA_VISIBLE_DEVICES}")"

if [[ "${SINGLE_NODE_MODE}" == "true" ]]; then
  ACTOR_NUM_NODES=1
  CRITIC_NUM_NODES=1
  REF_NUM_NODES=1
  REWARD_NUM_NODES=1
  if (( head_student_gpu_count != ACTOR_GPUS + CRITIC_GPUS )); then
    echo "[ERROR] (single-node) head student gpu count (${head_student_gpu_count})"
    echo "        must equal ACTOR_GPUS(${ACTOR_GPUS}) + CRITIC_GPUS(${CRITIC_GPUS})"
    exit 1
  fi
else
  if (( head_student_gpu_count != ACTOR_GPUS )); then
    echo "[ERROR] head student gpu count (${head_student_gpu_count}) must equal ACTOR_GPUS(${ACTOR_GPUS})"
    exit 1
  fi
  if (( worker_student_gpu_count != CRITIC_GPUS )); then
    echo "[ERROR] worker student gpu count (${worker_student_gpu_count}) must equal CRITIC_GPUS(${CRITIC_GPUS})"
    exit 1
  fi
fi

# =====================================================================
# 14) METADATA SNAPSHOT
# =====================================================================
write_run_metadata() {
  local vars=(
    RUN_NAME OUTPUT_ROOT RUN_DIR SAVE_PATH TB_DIR RAY_LOG_DIR PID_DIR JOB_SCRIPT JOB_LOG
    HEAD_NODE HEAD_NODE_IP WORKER_NODE WORKER_NODE_IP WORKER_SSH_HOST SSH_USER SSH_OPTS
    DLC_MODE DLC_NODE_RANK DLC_MASTER_ADDR DLC_WORLD_SIZE SINGLE_NODE_MODE SKIP_SSH_BOOTSTRAP
    HEAD_STUDENT_CUDA_VISIBLE_DEVICES WORKER_STUDENT_CUDA_VISIBLE_DEVICES
    ACTOR_GPUS CRITIC_GPUS REF_GPUS REWARD_GPUS
    ACTOR_NUM_NODES CRITIC_NUM_NODES REF_NUM_NODES REWARD_NUM_NODES
    REPO_ROOT MODEL_PATH TRAIN_DATA EVAL_DATA TEACHER_VENV STUDENT_VENV ANALYSIS_VENV
    N_SAMPLES_PER_PROMPT ROLLOUT_BATCH_SIZE TRAIN_BATCH_SIZE MICRO_TRAIN_BATCH_SIZE MICRO_ROLLOUT_BATCH_SIZE MICRO_REWARD_BATCH_SIZE
    PROMPT_MAX_LEN CONTEXT_MAX_LEN GENERATE_MAX_LEN STRIDE NUM_EPISODES MAX_EPOCHS TARGET_STEPS MAX_SAMPLES
    ACTOR_LR CRITIC_LR CRITIC_LR_HEAD
    EVAL_STEPS EVAL_MAX_SAMPLES EVAL_GENERATE_MAX_LEN SAVE_STEPS SAVE_EVEN_COUNT ONLINE_EVAL
    EVAL_AFTER_TRAIN RUN_TWO_ROUND_EVAL POST_EVAL_SCRIPT
    MODEL_CUDA_VISIBLE_DEVICES VLLM_TP_SIZE
    POST_EVAL_MAX_SAMPLES POST_EVAL_PROMPT_MAX_LEN
    FIRST_PASS_MAX_NEW_TOKENS SECOND_PASS_MAX_NEW_TOKENS
    POST_EVAL_TEMPERATURE POST_EVAL_TOP_P POST_EVAL_REPETITION_PENALTY POST_EVAL_BEST_OF_N
    VLLM_MAX_NUM_SEQS VLLM_PROGRESS_BATCH_SIZE VLLM_ENABLE_PREFIX_CACHING VLLM_SEED
    POSTEVAL_RDV_WORKER_TIMEOUT POSTEVAL_RDV_MASTER_TIMEOUT
    POST_EVAL_TAG POST_EVAL_LOG_DIR
    NCCL_P2P_LEVEL NCCL_NET_GDR_DISABLE
    ARCHIVE_OUTPUTS_AFTER_RUN ARCHIVE_OUTPUT_ROOT
  )

  cp -f "${SCRIPT_SOURCE_PATH}" "${LAUNCHER_SNAPSHOT_PATH}" 2>/dev/null || true
  rm -f "${TRAIN_CONFIG_SNAPSHOT_PATH}" 2>/dev/null || true
  {
    echo "# Auto-generated run context snapshot (G1 2-node)"
    echo "# UTC: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    for name in "${vars[@]}"; do
      printf "%s=%q\n" "${name}" "${!name-}"
    done
  } > "${TRAIN_CONFIG_SNAPSHOT_PATH}"

  rm -f "${TRAIN_CONFIG_SUMMARY_PATH}" 2>/dev/null || true
  {
    echo "run_name: ${RUN_NAME}"
    echo "run_dir: ${RUN_DIR}"
    echo "save_path: ${SAVE_PATH}"
    echo "model_path: ${MODEL_PATH}"
    echo "train_data: ${TRAIN_DATA}"
    echo "eval_data: ${EVAL_DATA}"
    echo "train_batch_size: ${TRAIN_BATCH_SIZE}"
    echo "actor_gpus: ${ACTOR_GPUS}"
    echo "critic_gpus: ${CRITIC_GPUS}"
    echo "target_steps: ${TARGET_STEPS}"
    echo "max_samples: ${MAX_SAMPLES}"
    echo "post_eval_script: ${POST_EVAL_SCRIPT}"
    echo "post_eval_max_samples: ${POST_EVAL_MAX_SAMPLES}"
    echo "deploy_mode: $(grep -E '^DEPLOY_MODE=' "${DLC_DISPATCH_ENV_PATH}" 2>/dev/null | cut -d= -f2 || echo unknown)"
    echo "archive_output_root: ${ARCHIVE_OUTPUT_ROOT}"
    echo "launcher_snapshot: ${LAUNCHER_SNAPSHOT_PATH}"
    echo "job_script: ${JOB_SCRIPT}"
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
  SAVE_PATH="${RUN_DIR}/model"
  TB_DIR="${RUN_DIR}/tensorboard"
  RAY_LOG_DIR="${RUN_DIR}/ray_logs"
  PID_DIR="${RUN_DIR}/pids"
  JOB_SCRIPT="${RUN_DIR}/run_train_once.sh"
  JOB_LOG="${RUN_DIR}/ray_job.log"
  if [[ -n "${POST_EVAL_LOG_DIR}" && "${POST_EVAL_LOG_DIR}" == "${old_run_dir}"* ]]; then
    POST_EVAL_LOG_DIR="${RUN_DIR}${POST_EVAL_LOG_DIR#${old_run_dir}}"
  fi
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
    printf "TRAIN_RC=%q\n" "${TRAIN_RC:-0}"
    printf "EVAL_RC=%q\n" "${EVAL_RC:-0}"
    printf "ARCHIVE_RC=%q\n" "${ARCHIVE_RC:-0}"
    printf "FINAL_RC=%q\n" "${FINAL_RC:-0}"
    printf "RUN_DIR=%q\n" "${RUN_DIR:-}"
    printf "SAVE_PATH=%q\n" "${SAVE_PATH:-}"
    printf "POST_EVAL_LOG_DIR=%q\n" "${POST_EVAL_LOG_DIR:-}"
  } > "${RUN_DIR}/final_status.env"
}

write_run_metadata

# =====================================================================
# 15) SINGLE-NODE FALLBACK -> delegate to scripts/run_G1_rebase.sh
# =====================================================================
if [[ "${SINGLE_NODE_MODE}" == "true" ]]; then
  echo "[INFO] single-node mode -> delegating to scripts/run_G1_rebase.sh"
  export REPO_ROOT MODEL_PATH TRAIN_DATA EVAL_DATA
  export TEACHER_VENV STUDENT_VENV ANALYSIS_VENV
  export TEACHER_PYTHON_BIN STUDENT_PYTHON_BIN ANALYSIS_PYTHON_BIN
  export ACTOR_GPUS CRITIC_GPUS
  export N_SAMPLES_PER_PROMPT ROLLOUT_BATCH_SIZE TRAIN_BATCH_SIZE
  export MICRO_TRAIN_BATCH_SIZE MICRO_ROLLOUT_BATCH_SIZE MICRO_REWARD_BATCH_SIZE
  export PROMPT_MAX_LEN CONTEXT_MAX_LEN GENERATE_MAX_LEN STRIDE
  export NUM_EPISODES MAX_EPOCHS TARGET_STEPS MAX_SAMPLES
  export ACTOR_LR CRITIC_LR CRITIC_LR_HEAD
  export ONLINE_EVAL EVAL_STEPS EVAL_MAX_SAMPLES EVAL_GENERATE_MAX_LEN SAVE_STEPS SAVE_EVEN_COUNT
  export EVAL_AFTER_TRAIN RUN_TWO_ROUND_EVAL
  export POST_EVAL_MAX_SAMPLES POST_EVAL_PROMPT_MAX_LEN
  export FIRST_PASS_MAX_NEW_TOKENS SECOND_PASS_MAX_NEW_TOKENS
  export POST_EVAL_TEMPERATURE POST_EVAL_TOP_P POST_EVAL_REPETITION_PENALTY POST_EVAL_BEST_OF_N
  export VLLM_MAX_NUM_SEQS VLLM_PROGRESS_BATCH_SIZE VLLM_ENABLE_PREFIX_CACHING VLLM_SEED
  export MODEL_CUDA_VISIBLE_DEVICES VLLM_TP_SIZE
  export RUN_NAME OUTPUT_ROOT
  exec bash "${REPO_ROOT}/scripts/run_G1_rebase.sh"
fi

# =====================================================================
# 16) RUNTIME PROCESS BOOK-KEEPING + CLEANUP TRAP
# =====================================================================
RAY_HEAD_PID=""
RUNTIME_STOPPED=0

stop_runtime_processes() {
  if [[ "${RUNTIME_STOPPED}" == "1" ]]; then
    return 0
  fi

  echo "[cleanup] stopping local ray..."
  if [[ -n "${RAY_HEAD_PID}" ]] && kill -0 "${RAY_HEAD_PID}" 2>/dev/null; then
    kill "${RAY_HEAD_PID}" 2>/dev/null || true
    wait "${RAY_HEAD_PID}" 2>/dev/null || true
  fi
  ray stop --force >/dev/null 2>&1 || true

  if [[ "${SKIP_SSH_BOOTSTRAP}" == "true" ]]; then
    # Single-node mode: no worker pod (we never reach here in that case
    # because single-node fallback exec's earlier).
    # DLC mode: worker pod has its own EXIT trap that calls ray stop on
    # the worker pod when ray cluster is shut down.
    :
  else
    echo "[cleanup] stopping worker-side ray..."
    ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -s -- '${PID_DIR}'" <<'EOF' >/dev/null 2>&1 || true
set +e
PID_DIR="$1"
shopt -s nullglob
for pid_file in "${PID_DIR}"/ray_worker.pid; do
  if [[ -f "${pid_file}" ]]; then
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  fi
done
ray stop --force >/dev/null 2>&1 || true
EOF
  fi
  RUNTIME_STOPPED=1
}

cleanup() {
  stop_runtime_processes || true
}
trap cleanup EXIT INT TERM

# =====================================================================
# 17) DLC WORKER POD ENTRY POINT (rank > 0)
# =====================================================================
# Worker pod responsibilities (much simpler than G3 because no teacher):
#   1) write own IP to OSS rendezvous file (+ keepalive)
#   2) wait for master ray head
#   3) ray join the cluster as critic worker (8 GPUs)
#   4) when training ray cluster shuts down, enter post-eval rendezvous
#      watcher (parked there until master writes all_done.marker)
dlc_worker_bootstrap() {
  echo "================================================================"
  echo "[DLC worker rank=${DLC_NODE_RANK}] starting on $(hostname)"
  echo "[DLC worker rank=${DLC_NODE_RANK}] master address: ${DLC_MASTER_ADDR}:${RAY_PORT}"

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

  # IP keepalive (5s loop) - protects against master rm-on-restart races.
  (
    while :; do
      printf '%s\n' "${my_ip}" > "${ip_file}" 2>/dev/null || true
      sleep 5
    done
  ) &
  local _ip_keepalive_pid=$!
  echo "[DLC worker] IP keepalive started (pid=${_ip_keepalive_pid}, interval=5s)"

  _dlc_worker_cleanup() {
    echo "[DLC worker] cleaning up local ray..."
    if [[ -n "${_ip_keepalive_pid:-}" ]]; then
      kill "${_ip_keepalive_pid}" 2>/dev/null || true
    fi
    ray stop --force >/dev/null 2>&1 || true
  }
  trap _dlc_worker_cleanup EXIT INT TERM

  # Wait for master ray head.
  local _ray_head_wait="${DLC_WORKER_RAY_HEAD_WAIT_SECONDS:-1800}"
  if (( _ray_head_wait < RAY_WAIT_SECONDS * 4 )); then
    _ray_head_wait=$(( RAY_WAIT_SECONDS * 4 ))
  fi
  echo "[DLC worker] waiting for master ray head at ${DLC_MASTER_ADDR}:${RAY_PORT} (timeout ${_ray_head_wait}s)..."
  local waited=0
  until ray status --address "${DLC_MASTER_ADDR}:${RAY_PORT}" >/dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    if (( waited >= _ray_head_wait )); then
      echo "[DLC worker] ERROR: master ray head didn't come up in ${_ray_head_wait}s"
      exit 1
    fi
  done
  echo "[DLC worker] master ray head reachable after ${waited}s"

  # Join ray cluster (critic worker side, 8 GPUs).
  echo "[DLC worker] joining ray cluster via 'ray start --address=${DLC_MASTER_ADDR}:${RAY_PORT}'..."
  CUDA_VISIBLE_DEVICES="${WORKER_STUDENT_CUDA_VISIBLE_DEVICES}" \
    ray start --address "${DLC_MASTER_ADDR}:${RAY_PORT}" \
              --num-gpus "${worker_student_gpu_count}" \
              --block &
  local ray_pid=$!
  echo "[DLC worker] ray join started (pid=${ray_pid}); waiting until cluster shuts down..."
  echo "================================================================"
  wait "${ray_pid}"
}

if [[ "${DLC_MODE}" == "true" && "${DLC_NODE_RANK}" -gt 0 ]]; then
  dlc_worker_bootstrap
  echo "[DLC worker rank=${DLC_NODE_RANK}] ray cluster shut down."

  # Stay alive for the post-eval rendezvous watcher.
  if [[ "${RUN_TWO_ROUND_EVAL:-true}" == "true" ]]; then
    _rdv_helper_path="${REPO_ROOT}/scripts/supplement_2rounds/_rendezvous_dlc.sh"
    _vllm_runtime_path="${REPO_ROOT}/scripts/supplement_2rounds/_vllm_runtime.sh"
    if [[ ! -f "${_rdv_helper_path}" ]]; then
      echo "[DLC worker rank=${DLC_NODE_RANK}] post-eval rendezvous helper missing: ${_rdv_helper_path}"
      echo "                                    falling back to exit 0 (single-node eval only)."
      exit 0
    fi
    export CUDA_VISIBLE_DEVICES="${POSTEVAL_WORKER_CUDA_VISIBLE_DEVICES:-${WORKER_STUDENT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
    export MODEL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    _VISIBLE_COUNT="$(count_csv_items "${CUDA_VISIBLE_DEVICES}")"
    export VLLM_TP_SIZE="${POSTEVAL_WORKER_VLLM_TP_SIZE:-${_VISIBLE_COUNT}}"
    export REPO_ROOT TEACHER_VENV
    export TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
    export PROGRESS_HELPER="${PROGRESS_HELPER:-${REPO_ROOT}/scripts/supplement/vllm_generate_progress.py}"
    export POSTEVAL_RDV_WORKER_TIMEOUT POSTEVAL_RDV_MASTER_TIMEOUT
    # MODEL_PATH for post-eval is the trained ckpt under SAVE_PATH.
    export MODEL_PATH="${SAVE_PATH}"
    echo "[DLC worker rank=${DLC_NODE_RANK}] sourcing vLLM runtime helpers for post-eval"
    # shellcheck disable=SC1090
    source "${_vllm_runtime_path}"
    # shellcheck disable=SC1090
    source "${_rdv_helper_path}"
    rdv_init_root "${RUN_DIR}"
    echo "[DLC worker rank=${DLC_NODE_RANK}] entering post-eval rendezvous watch loop"
    set +e
    posteval_worker_watch
    _rdv_rc=$?
    set -e
    echo "[DLC worker rank=${DLC_NODE_RANK}] post-eval watcher exited rc=${_rdv_rc}; shutting down"
    exit "${_rdv_rc}"
  fi

  echo "[DLC worker rank=${DLC_NODE_RANK}] exiting"
  exit 0
fi

# =====================================================================
# 18) DLC MASTER: wait for worker pod IP
# =====================================================================
if [[ "${DLC_MODE}" == "true" ]]; then
  rdv_dir="${RUN_DIR}/dlc_rendezvous"
  mkdir -p "${rdv_dir}"
  rm -f "${rdv_dir}"/worker_*_ip.txt 2>/dev/null || true
  echo "[DLC master] waiting for worker pod IP at ${rdv_dir}/worker_1_ip.txt..."
  waited=0
  while [[ ! -s "${rdv_dir}/worker_1_ip.txt" ]]; do
    sleep 3
    waited=$((waited + 3))
    if (( waited >= RAY_WAIT_SECONDS * 4 )); then
      echo "[DLC master] ERROR: worker pod IP not seen in $((RAY_WAIT_SECONDS * 4))s"
      exit 1
    fi
  done
  WORKER_NODE_IP="$(tr -d '[:space:]' < "${rdv_dir}/worker_1_ip.txt")"
  echo "[DLC master] worker IP: ${WORKER_NODE_IP} (after ${waited}s)"
fi

# =====================================================================
# 19) DSW MASTER: ssh check
# =====================================================================
if [[ "${SKIP_SSH_BOOTSTRAP}" != "true" ]]; then
  echo "[1/3] connectivity check to worker..."
  ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -lc 'hostname'" >/dev/null
fi

# =====================================================================
# 20) START RAY CLUSTER
# =====================================================================
if [[ "${SKIP_SSH_BOOTSTRAP}" == "true" ]]; then
  if [[ "${DLC_MODE}" == "true" ]]; then
    echo "[1/3] DLC mode: worker pod connects in via ray start; skipping ssh check"
  fi
fi

echo "[2/3] starting ray cluster (head)..."
ray stop --force >/dev/null 2>&1 || true
rm -f "${RAY_LOG_DIR}/head.log" 2>/dev/null || true
CUDA_VISIBLE_DEVICES="${HEAD_STUDENT_CUDA_VISIBLE_DEVICES}" \
ray start --head \
  --node-ip-address "${HEAD_NODE_IP}" \
  --port "${RAY_PORT}" \
  --dashboard-host 0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}" \
  --num-gpus "${head_student_gpu_count}" \
  --block \
  > "${RAY_LOG_DIR}/head.log" 2>&1 &
RAY_HEAD_PID=$!

sleep 5

if [[ "${SKIP_SSH_BOOTSTRAP}" == "true" ]]; then
  if [[ "${DLC_MODE}" == "true" ]]; then
    echo "[ray] DLC mode: worker pod will join via 'ray start --address=${HEAD_NODE_IP}:${RAY_PORT}' on its own"
  fi
else
  ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -lc '
    set -euo pipefail
    ray stop --force >/dev/null 2>&1 || true
    mkdir -p ${PID_DIR}
    nohup bash -lc \"CUDA_VISIBLE_DEVICES=${WORKER_STUDENT_CUDA_VISIBLE_DEVICES} ray start --address ${HEAD_NODE_IP}:${RAY_PORT} --num-gpus ${worker_student_gpu_count} --block\" > ${RAY_LOG_DIR}/worker.log 2>&1 &
    echo \$! > ${PID_DIR}/ray_worker.pid
  '"
fi

waited=0
until ray status --address "${HEAD_NODE_IP}:${RAY_PORT}" >/dev/null 2>&1; do
  sleep 2
  waited=$((waited + 2))
  if (( waited >= RAY_WAIT_SECONDS )); then
    echo "[ERROR] ray cluster did not become ready in time."
    exit 1
  fi
done

# =====================================================================
# 21) GENERATE TRAIN JOB SCRIPT
# =====================================================================
echo "[3/3] generating train job script..."

# Build ONLINE_EVAL_ARGS based on the switch.
if [[ "${ONLINE_EVAL}" == "true" ]]; then
  ONLINE_EVAL_INLINE="--eval_dataset '${EVAL_DATA}' --eval_split '${EVAL_SPLIT}' --eval_steps '${EVAL_STEPS}' --eval_max_samples '${EVAL_MAX_SAMPLES}' --eval_generate_max_len '${EVAL_GENERATE_MAX_LEN}'"
else
  ONLINE_EVAL_INLINE="--eval_steps -1 --eval_down_steps -1"
fi

cat > "${JOB_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export HF_HOME='${HF_HOME}'
export HF_HUB_OFFLINE='${HF_HUB_OFFLINE}'
export HF_DATASETS_OFFLINE='${HF_DATASETS_OFFLINE}'
export HF_HUB_DISABLE_XET='${HF_HUB_DISABLE_XET}'
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_WORKER_MULTIPROC_METHOD='${VLLM_WORKER_MULTIPROC_METHOD}'
export PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF}'
export PYTHONUNBUFFERED=1
export RAY_ADDRESS=auto
export NCCL_P2P_LEVEL='${NCCL_P2P_LEVEL}'
export NCCL_NET_GDR_DISABLE='${NCCL_NET_GDR_DISABLE}'

cd '${REPO_ROOT}'

'${STUDENT_PYTHON_BIN}' -m openrlhf.cli.train_ebft_ray \\
  --bf16 --flash_attn --pretrain_mode --no_chat_template \\
  --disable_ds_ckpt --colocate_actor_ref --colocate_critic_reward \\
  --use_kl_loss --use_whitening \\
  --distribution_reward_type pointwise \\
  --feature_map_type identity --rff_num_features 128 --rff_sigma 1.0 --rff_seed 43 \\
  --cf_num_freqs 128 --cf_sigma 1.0 --cf_seed 43 --cf_alpha 0.5 --cf_beta 0.5 --cf_reward_scale 1.0 \\
  --cf_target_mode single --cf_teacher_lambda 0.0 --cf_teacher_n_samples '${N_SAMPLES_PER_PROMPT}' \\
  --embed_method last_token --critic_sequence_level last_token \\
  --critic_learning_rate '${CRITIC_LR}' --critic_lr_head '${CRITIC_LR_HEAD}' \\
  --pretrain '${MODEL_PATH}' --critic_pretrain '${MODEL_PATH}' \\
  --prompt_data '${TRAIN_DATA}' \\
  --input_key question --label_key answer --output_key answer \\
  --prompt_split '${PROMPT_SPLIT}' \\
  --prompt_max_len '${PROMPT_MAX_LEN}' \\
  --context_max_len '${CONTEXT_MAX_LEN}' \\
  --generate_max_len '${GENERATE_MAX_LEN}' \\
  --stride '${STRIDE}' \\
  --n_samples_per_prompt '${N_SAMPLES_PER_PROMPT}' \\
  --rollout_batch_size '${ROLLOUT_BATCH_SIZE}' \\
  --train_batch_size '${TRAIN_BATCH_SIZE}' \\
  --micro_train_batch_size '${MICRO_TRAIN_BATCH_SIZE}' \\
  --micro_rollout_batch_size '${MICRO_ROLLOUT_BATCH_SIZE}' \\
  --micro_reward_batch_size '${MICRO_REWARD_BATCH_SIZE}' \\
  --max_samples '${MAX_SAMPLES}' \\
  --num_episodes '${NUM_EPISODES}' \\
  --max_epochs '${MAX_EPOCHS}' \\
  --actor_num_nodes '${ACTOR_NUM_NODES}' --actor_num_gpus_per_node '${ACTOR_GPUS}' \\
  --critic_num_nodes '${CRITIC_NUM_NODES}' --critic_num_gpus_per_node '${CRITIC_GPUS}' \\
  --ref_num_nodes '${REF_NUM_NODES}' --ref_num_gpus_per_node '${REF_GPUS}' \\
  --reward_num_nodes '${REWARD_NUM_NODES}' --reward_num_gpus_per_node '${REWARD_GPUS}' \\
  --advantage_estimator rloo --init_kl_coef 0.0 --kl_estimator k2 \\
  --temperature 0.6 --top_p 1.0 --actor_learning_rate '${ACTOR_LR}' \\
  --zero_stage 2 --adam_offload --ref_reward_offload --lr_warmup_ratio 0.03 --critic_lr_warmup_ratio 0.0 \\
  --seed 43 \\
  ${ONLINE_EVAL_INLINE} \\
  --logging_steps 10 \\
  --save_steps '${SAVE_STEPS}' --save_even_count '${SAVE_EVEN_COUNT}' --save_hf_ckpt \\
  --use_tensorboard '${TB_DIR}' \\
  --save_path '${SAVE_PATH}' --ckpt_path '${SAVE_PATH}/ckpt' \\
  --wandb_run_name '${RUN_NAME}' \\
  2>&1 | tee '${RUN_DIR}/train.log'
EOF

chmod +x "${JOB_SCRIPT}"

# =====================================================================
# 22) BANNER
# =====================================================================
echo "========== G1 2-node once launcher =========="
echo "RUN_DIR:                    ${RUN_DIR}"
echo "MODEL_PATH:                 ${MODEL_PATH}"
echo "HEAD_NODE / IP:             ${HEAD_NODE} / ${HEAD_NODE_IP}"
echo "WORKER_NODE / IP:           ${WORKER_NODE} / ${WORKER_NODE_IP}"
echo "Head student GPUs:          ${HEAD_STUDENT_CUDA_VISIBLE_DEVICES} (count=${head_student_gpu_count})"
echo "Worker student GPUs:        ${WORKER_STUDENT_CUDA_VISIBLE_DEVICES} (count=${worker_student_gpu_count})"
echo "Actor/Critic GPUs:          ${ACTOR_GPUS}/${CRITIC_GPUS}"
echo "Train batch / micro:        ${TRAIN_BATCH_SIZE} / ${MICRO_TRAIN_BATCH_SIZE}"
echo "Target steps / max_samples: ${TARGET_STEPS} / ${MAX_SAMPLES}"
echo "Reward type:                pointwise (cf_target_mode=single, no teacher)"
echo "Deploy mode:                $(grep -E '^DEPLOY_MODE=' "${DLC_DISPATCH_ENV_PATH}" | cut -d= -f2)"
echo "Dispatch:                   $(grep -E '^POSTEVAL_WORKER_DISPATCH=' "${DLC_DISPATCH_ENV_PATH}" | cut -d= -f2)"
echo "Post-train eval:            ${EVAL_AFTER_TRAIN}"
echo "Post-eval script:           ${POST_EVAL_SCRIPT}"
echo "Post-eval first/second:     ${FIRST_PASS_MAX_NEW_TOKENS}/${SECOND_PASS_MAX_NEW_TOKENS} tokens"
echo "Worker watcher timeout:     ${POSTEVAL_RDV_WORKER_TIMEOUT}s"
echo "Archive after run:          ${ARCHIVE_OUTPUTS_AFTER_RUN} -> ${ARCHIVE_OUTPUT_ROOT}"
echo "============================================="

# =====================================================================
# 23) SUBMIT TRAIN JOB
# =====================================================================
echo "[train] submitting one Ray job..."
TRAIN_RC=0
EVAL_RC=0
ARCHIVE_RC=0

set +e
ray job submit \
  --address "http://${HEAD_NODE_IP}:${RAY_DASHBOARD_PORT}" \
  -- bash "${JOB_SCRIPT}" | tee "${JOB_LOG}"
TRAIN_RC=$?
set -e

if (( TRAIN_RC != 0 )); then
  echo "[ERROR] training failed with exit code ${TRAIN_RC}"
fi

echo "[post-run] stopping ray processes before eval/archive ..."
stop_runtime_processes

# =====================================================================
# 24) POST-TRAINING TWO-ROUND EVAL
#     Re-load dispatch decision from OSS-shared file to recover from
#     any AIMaster-induced env loss during training.
# =====================================================================
if [[ "${RUN_TWO_ROUND_EVAL}" == "true" ]]; then
  if (( TRAIN_RC == 0 )); then
    echo ""
    echo "===== Running two-round 16k/32k completion eval (2-node parallel) ====="
    echo "[post-eval] running eval from checkpoint: ${SAVE_PATH}"

    # Recover dispatch decision (resilient to AIMaster restart that may
    # have dropped DLC env vars during training).
    if [[ -f "${DLC_DISPATCH_ENV_PATH}" ]]; then
      # shellcheck disable=SC1090
      source "${DLC_DISPATCH_ENV_PATH}"
      echo "[post-eval] dispatch loaded from ${DLC_DISPATCH_ENV_PATH}: mode=${DEPLOY_MODE} dispatch=${POSTEVAL_WORKER_DISPATCH}"
    else
      echo "[post-eval] WARN: ${DLC_DISPATCH_ENV_PATH} missing; falling back to in-memory shell vars"
    fi

    set +e
    export RUN_DIR MODEL_PATH="${SAVE_PATH}"
    export REPO_ROOT
    export TEACHER_VENV ANALYSIS_VENV
    export TEACHER_PYTHON_BIN ANALYSIS_PYTHON_BIN
    export MODEL_CUDA_VISIBLE_DEVICES VLLM_TP_SIZE
    export POST_EVAL_MAX_SAMPLES POST_EVAL_PROMPT_MAX_LEN
    export FIRST_PASS_MAX_NEW_TOKENS SECOND_PASS_MAX_NEW_TOKENS
    export POST_EVAL_TEMPERATURE POST_EVAL_TOP_P
    export POST_EVAL_REPETITION_PENALTY POST_EVAL_BEST_OF_N
    export VLLM_MAX_NUM_SEQS VLLM_PROGRESS_BATCH_SIZE VLLM_ENABLE_PREFIX_CACHING VLLM_SEED
    export EVAL_DATA
    export LOG_DIR="${POST_EVAL_LOG_DIR}"
    export EVAL_TAG="${POST_EVAL_TAG}"
    export POSTEVAL_WORKER_DISPATCH WORKER_SSH_TARGET SSH_OPTS
    export POSTEVAL_RDV_WORKER_TIMEOUT POSTEVAL_RDV_MASTER_TIMEOUT
    export NCCL_P2P_LEVEL NCCL_NET_GDR_DISABLE
    bash "${POST_EVAL_SCRIPT}" "${RUN_DIR}"
    EVAL_RC=$?
    set -e
    if (( EVAL_RC != 0 )); then
      echo "[ERROR] post-eval failed with exit code ${EVAL_RC}; run outputs will still be archived."
    fi
  else
    echo "[post-eval] skipped because training did not finish successfully."
  fi
fi

# =====================================================================
# 25) ARCHIVE
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
# 26) FINAL STATUS
# =====================================================================
FINAL_RC=0
if (( TRAIN_RC != 0 )); then
  FINAL_RC=${TRAIN_RC}
elif (( EVAL_RC != 0 )); then
  FINAL_RC=${EVAL_RC}
elif (( ARCHIVE_RC != 0 )); then
  FINAL_RC=${ARCHIVE_RC}
fi

write_final_status

echo "[done] logs: ${RUN_DIR}"
exit "${FINAL_RC}"
