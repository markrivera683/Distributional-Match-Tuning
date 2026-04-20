#!/usr/bin/env bash
# Two-node launcher for the G3 rebase training recipe.
# Assumptions:
# 1) Run this script on the head node.
# 2) Passwordless SSH from head -> worker is available.
# 3) Each node has 8 GPUs.
# 4) Teacher uses GPUs 0-5 on both nodes (12 GPUs total).
# 5) Student uses GPUs 6-7 on both nodes (4 GPUs total):
#      - actor/ref world:   1 node x 2 GPUs
#      - critic/reward:     1 node x 2 GPUs
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

build_teacher_urls() {
  local host_ip="$1"
  local base_port="$2"
  local worker_count="$3"
  local urls=""
  local port
  local i
  for (( i=0; i<worker_count; i++ )); do
    port=$((base_port + i))
    [[ -n "${urls}" ]] && urls+=","
    urls+="http://${host_ip}:${port}/v1"
  done
  echo "${urls}"
}

HEAD_NODE="${HEAD_NODE:-}"
WORKER_NODE="${WORKER_NODE:-}"
HEAD_NODE_IP="${HEAD_NODE_IP:-}"
WORKER_NODE_IP="${WORKER_NODE_IP:-}"
WORKER_SSH_HOST="${WORKER_SSH_HOST:-}"
SSH_USER="${SSH_USER:-}"
SSH_OPTS="${SSH_OPTS:-}"

if [[ -z "${HEAD_NODE}" || -z "${WORKER_NODE}" ]]; then
  echo "[ERROR] please set HEAD_NODE and WORKER_NODE."
  echo "Example:"
  echo "  HEAD_NODE=node0 WORKER_NODE=node1 bash scripts/run_G3_rebase_2node_once.sh"
  exit 1
fi

HEAD_NODE_IP="${HEAD_NODE_IP:-$(resolve_host_ip "${HEAD_NODE}")}"
WORKER_NODE_IP="${WORKER_NODE_IP:-$(resolve_host_ip "${WORKER_NODE}")}"
WORKER_SSH_HOST="${WORKER_SSH_HOST:-${WORKER_NODE_IP}}"

if [[ -n "${SSH_USER}" ]]; then
  WORKER_SSH_TARGET="${SSH_USER}@${WORKER_SSH_HOST}"
else
  WORKER_SSH_TARGET="${WORKER_SSH_HOST}"
fi

CURRENT_HOSTNAME="$(hostname)"
CURRENT_HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
CURRENT_HOST_IPS="$(hostname -I 2>/dev/null || true)"

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

HEAD_TEACHER_CUDA_VISIBLE_DEVICES="${HEAD_TEACHER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
WORKER_TEACHER_CUDA_VISIBLE_DEVICES="${WORKER_TEACHER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
HEAD_STUDENT_CUDA_VISIBLE_DEVICES="${HEAD_STUDENT_CUDA_VISIBLE_DEVICES:-6,7}"
WORKER_STUDENT_CUDA_VISIBLE_DEVICES="${WORKER_STUDENT_CUDA_VISIBLE_DEVICES:-6,7}"

ACTOR_GPUS="${ACTOR_GPUS:-2}"
CRITIC_GPUS="${CRITIC_GPUS:-2}"
REF_GPUS="${REF_GPUS:-${ACTOR_GPUS}}"
REWARD_GPUS="${REWARD_GPUS:-${CRITIC_GPUS}}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
CRITIC_NUM_NODES="${CRITIC_NUM_NODES:-1}"
REF_NUM_NODES="${REF_NUM_NODES:-1}"
REWARD_NUM_NODES="${REWARD_NUM_NODES:-1}"

TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/mnt/data/models/qwen3.5-27b}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-qwen3.5-27b}"
TEACHER_BASE_PORT="${TEACHER_BASE_PORT:-8004}"
TEACHER_API_KEY="${TEACHER_API_KEY:-teacher-local}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
TEACHER_DTYPE="${TEACHER_DTYPE:-bfloat16}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-2048}"
TEACHER_MAX_NUM_SEQS="${TEACHER_MAX_NUM_SEQS:-32}"
TEACHER_MAX_BATCHED_TOKENS="${TEACHER_MAX_BATCHED_TOKENS:-16384}"
TEACHER_GPU_MEMORY_UTIL="${TEACHER_GPU_MEMORY_UTIL:-0.96}"
TEACHER_WAIT_SECONDS="${TEACHER_WAIT_SECONDS:-3600}"

REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Match-Tuning}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/teacher_model/models/Qwen3.5-0.8B}"
DEFAULT_TRAIN_DATA="/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
TRAIN_DATA="${TRAIN_DATA:-${DEFAULT_TRAIN_DATA}}"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"
TEACHER_VENV="${TEACHER_VENV:-${REPO_ROOT}/.teacherVenv}"
STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"
TEACHER_VLLM_BIN="${TEACHER_VLLM_BIN:-${TEACHER_VENV}/bin/vllm}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"

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

CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.6}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-8}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-768}"
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-200}"
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-3}"
TEACHER_REMOTE_BATCH_SIZE="${TEACHER_REMOTE_BATCH_SIZE:-48}"
TEACHER_SYSTEM_PROMPT_TEXT="${TEACHER_SYSTEM_PROMPT_TEXT:-You are a precise assistant. produce a correct and well-reasoned answer. Step by step when necessary. Keep reasoning sufficient. Final answer is clearly stated.}"
TEACHER_SYSTEM_PROMPT_ID="${TEACHER_SYSTEM_PROMPT_ID:-v1-balanced}"
TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-/root/outputs/teacher_cache_shared}"

ENABLE_TEACHER_PREFETCH="${ENABLE_TEACHER_PREFETCH:-true}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-2}"
PREFETCH_MAX_WORKERS="${PREFETCH_MAX_WORKERS:-6}"

EVAL_STEPS="${EVAL_STEPS:-200}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-50}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-${GENERATE_MAX_LEN}}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_EVEN_COUNT="${SAVE_EVEN_COUNT:-10}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
POST_EVAL_SCRIPT="${POST_EVAL_SCRIPT:-${REPO_ROOT}/scripts/supplement/G3_eval.sh}"
POST_EVAL_STUDENT_CUDA_VISIBLE_DEVICES="${POST_EVAL_STUDENT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
POST_EVAL_NNODES="${POST_EVAL_NNODES:-2}"
POST_EVAL_NPROC="${POST_EVAL_NPROC:-16}"
POST_EVAL_NPROC_PER_NODE="${POST_EVAL_NPROC_PER_NODE:-8}"
POST_EVAL_HEAD_CUDA_VISIBLE_DEVICES="${POST_EVAL_HEAD_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
POST_EVAL_WORKER_CUDA_VISIBLE_DEVICES="${POST_EVAL_WORKER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
POST_EVAL_MAX_NEW_TOKENS="${POST_EVAL_MAX_NEW_TOKENS:-8192}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_MICRO_BATCH_SIZE="${POST_EVAL_MICRO_BATCH_SIZE:-64}"
POST_EVAL_MASTER_PORT="${POST_EVAL_MASTER_PORT:-29501}"
POST_EVAL_TAG="${POST_EVAL_TAG:-post_train}"
POST_EVAL_LOG_DIR="${POST_EVAL_LOG_DIR:-}"
ARCHIVE_OUTPUTS_AFTER_RUN="${ARCHIVE_OUTPUTS_AFTER_RUN:-true}"
ARCHIVE_OUTPUT_ROOT="${ARCHIVE_OUTPUT_ROOT:-/mnt/data/ebft-teacher-distribution/outputs_g3_0.99}"
ARCHIVE_SHARED_TEACHER_CACHE_MODE="${ARCHIVE_SHARED_TEACHER_CACHE_MODE:-copy}"
ARCHIVE_SHARED_TEACHER_CACHE_DIR="${ARCHIVE_SHARED_TEACHER_CACHE_DIR:-${TEACHER_CACHE_DIR}}"

FEATURE_ADAPTER_RANK="${FEATURE_ADAPTER_RANK:-64}"
FEATURE_ADAPTER_DROPOUT="${FEATURE_ADAPTER_DROPOUT:-0.0}"
UNFREEZE_LAYERS="${UNFREEZE_LAYERS:-0}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
CE_LOSS_COEF="${CE_LOSS_COEF:-0.03}"
EMA_BETA="${EMA_BETA:-0.99}"
CRITIC_LR="${CRITIC_LR:-0}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-5e-5}"
CRITIC_CLASSIFIER_LOSS_COEF="${CRITIC_CLASSIFIER_LOSS_COEF:-0.0}"
CRITIC_DIRECT_DISCREPANCY_COEF="${CRITIC_DIRECT_DISCREPANCY_COEF:-0.1}"
CRITIC_DIRECT_DISCREPANCY_TARGET="${CRITIC_DIRECT_DISCREPANCY_TARGET:-ema_gt}"
DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-0.5}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_WAIT_SECONDS="${RAY_WAIT_SECONDS:-120}"

RUN_NAME="${RUN_NAME:-g3_rebase_2node_once_$(date +%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SAVE_PATH="${RUN_DIR}/model"
TB_DIR="${RUN_DIR}/tensorboard"
TEACHER_LOG_DIR="${RUN_DIR}/teacher_logs"
RAY_LOG_DIR="${RUN_DIR}/ray_logs"
PID_DIR="${RUN_DIR}/pids"
JOB_SCRIPT="${RUN_DIR}/run_train_once.sh"
JOB_LOG="${RUN_DIR}/ray_job.log"
POST_EVAL_LOG_DIR="${POST_EVAL_LOG_DIR:-${RUN_DIR}/supplement_logs}"
SCRIPT_SOURCE_PATH="${BASH_SOURCE[0]:-$0}"
TRAIN_CONFIG_SNAPSHOT_PATH="${RUN_DIR}/run_context.env"
TRAIN_CONFIG_SUMMARY_PATH="${RUN_DIR}/run_summary.txt"
LAUNCHER_SNAPSHOT_PATH="${RUN_DIR}/launcher_snapshot.sh"
mkdir -p "${RUN_DIR}" "${SAVE_PATH}" "${TB_DIR}" "${TEACHER_LOG_DIR}" "${RAY_LOG_DIR}" "${PID_DIR}" "${TEACHER_CACHE_DIR}"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

require_cmd curl
require_cmd ray
require_cmd ssh

if [[ ! -x "${TEACHER_VLLM_BIN}" ]]; then
  echo "[ERROR] TEACHER_VLLM_BIN not executable: ${TEACHER_VLLM_BIN}"
  exit 1
fi
if [[ ! -x "${STUDENT_PYTHON_BIN}" ]]; then
  echo "[ERROR] STUDENT_PYTHON_BIN not executable: ${STUDENT_PYTHON_BIN}"
  exit 1
fi
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "[ERROR] REPO_ROOT not found: ${REPO_ROOT}"
  exit 1
fi
if [[ ! -e "${TRAIN_DATA}" ]]; then
  echo "[ERROR] TRAIN_DATA not found: ${TRAIN_DATA}"
  exit 1
fi
if [[ ! -e "${EVAL_DATA}" ]]; then
  echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"
  exit 1
fi

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
  TEACHER_LOG_DIR="${RUN_DIR}/teacher_logs"
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
  write_run_metadata
}

archive_shared_teacher_cache() {
  local mode="$1"
  local source_dir="$2"
  local dest_dir

  case "${mode}" in
    skip|"")
      echo "[archive] shared teacher cache archive skipped."
      return 0
      ;;
    copy|move)
      ;;
    *)
      echo "[ERROR] invalid ARCHIVE_SHARED_TEACHER_CACHE_MODE=${mode}. Use skip, copy, or move."
      return 1
      ;;
  esac

  if [[ ! -d "${source_dir}" ]]; then
    echo "[archive] shared teacher cache not found, skip: ${source_dir}"
    return 0
  fi

  dest_dir="${RUN_DIR}/$(basename "${source_dir}")"
  if [[ -e "${dest_dir}" ]]; then
    dest_dir="${dest_dir}_$(date +%m%d_%H%M%S)"
  fi

  echo "[archive] ${mode} shared teacher cache to: ${dest_dir}"
  if [[ "${mode}" == "copy" ]]; then
    cp -a "${source_dir}" "${dest_dir}"
  else
    mv "${source_dir}" "${dest_dir}"
  fi
}

write_run_metadata() {
  local vars=(
    RUN_NAME OUTPUT_ROOT RUN_DIR SAVE_PATH TB_DIR TEACHER_LOG_DIR RAY_LOG_DIR PID_DIR JOB_SCRIPT JOB_LOG
    HEAD_NODE HEAD_NODE_IP WORKER_NODE WORKER_NODE_IP WORKER_SSH_HOST SSH_USER SSH_OPTS
    HEAD_TEACHER_CUDA_VISIBLE_DEVICES WORKER_TEACHER_CUDA_VISIBLE_DEVICES
    HEAD_STUDENT_CUDA_VISIBLE_DEVICES WORKER_STUDENT_CUDA_VISIBLE_DEVICES
    ACTOR_GPUS CRITIC_GPUS REF_GPUS REWARD_GPUS
    ACTOR_NUM_NODES CRITIC_NUM_NODES REF_NUM_NODES REWARD_NUM_NODES
    TEACHER_MODEL_PATH TEACHER_MODEL_NAME TEACHER_BASE_PORT TEACHER_API_KEY TEACHER_TP_SIZE TEACHER_DTYPE
    TEACHER_MAX_MODEL_LEN TEACHER_MAX_NUM_SEQS TEACHER_MAX_BATCHED_TOKENS TEACHER_GPU_MEMORY_UTIL
    TEACHER_WAIT_SECONDS TEACHER_API_BASE TEACHER_CACHE_DIR ENABLE_TEACHER_PREFETCH PREFETCH_DEPTH PREFETCH_MAX_WORKERS
    REPO_ROOT MODEL_PATH TRAIN_DATA EVAL_DATA
    N_SAMPLES_PER_PROMPT ROLLOUT_BATCH_SIZE TRAIN_BATCH_SIZE MICRO_TRAIN_BATCH_SIZE MICRO_ROLLOUT_BATCH_SIZE MICRO_REWARD_BATCH_SIZE
    PROMPT_MAX_LEN CONTEXT_MAX_LEN GENERATE_MAX_LEN STRIDE NUM_EPISODES MAX_EPOCHS TARGET_STEPS MAX_SAMPLES
    CF_TEACHER_LAMBDA CF_TEACHER_N_SAMPLES TEACHER_TEMPERATURE TEACHER_TOP_P TEACHER_MAX_NEW_TOKENS
    TEACHER_TIMEOUT TEACHER_MAX_RETRIES TEACHER_REMOTE_BATCH_SIZE TEACHER_SYSTEM_PROMPT_ID
    FEATURE_ADAPTER_RANK FEATURE_ADAPTER_DROPOUT UNFREEZE_LAYERS ACTOR_LR CE_LOSS_COEF EMA_BETA
    CRITIC_LR CRITIC_LR_HEAD CRITIC_CLASSIFIER_LOSS_COEF CRITIC_DIRECT_DISCREPANCY_COEF CRITIC_DIRECT_DISCREPANCY_TARGET
    DIVERSITY_REW_COEF ALIGNMENT_REW_COEF
    EVAL_STEPS EVAL_MAX_SAMPLES EVAL_GENERATE_MAX_LEN SAVE_STEPS SAVE_EVEN_COUNT
    EVAL_AFTER_TRAIN POST_EVAL_SCRIPT POST_EVAL_NNODES POST_EVAL_NPROC POST_EVAL_NPROC_PER_NODE
    POST_EVAL_HEAD_CUDA_VISIBLE_DEVICES POST_EVAL_WORKER_CUDA_VISIBLE_DEVICES
    POST_EVAL_MAX_SAMPLES POST_EVAL_PROMPT_MAX_LEN POST_EVAL_MAX_NEW_TOKENS POST_EVAL_TEMPERATURE
    POST_EVAL_TOP_P POST_EVAL_MICRO_BATCH_SIZE POST_EVAL_MASTER_PORT POST_EVAL_TAG POST_EVAL_LOG_DIR
    ARCHIVE_OUTPUTS_AFTER_RUN ARCHIVE_OUTPUT_ROOT ARCHIVE_SHARED_TEACHER_CACHE_MODE ARCHIVE_SHARED_TEACHER_CACHE_DIR
  )

  mkdir -p "${RUN_DIR}"
  cp -f "${SCRIPT_SOURCE_PATH}" "${LAUNCHER_SNAPSHOT_PATH}" 2>/dev/null || true

  {
    echo "# Auto-generated run context snapshot"
    echo "# UTC: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    for name in "${vars[@]}"; do
      printf "%s=%q\n" "${name}" "${!name-}"
    done
  } > "${TRAIN_CONFIG_SNAPSHOT_PATH}"

  {
    echo "run_name: ${RUN_NAME}"
    echo "run_dir: ${RUN_DIR}"
    echo "save_path: ${SAVE_PATH}"
    echo "teacher_api_base: ${TEACHER_API_BASE}"
    echo "teacher_cache_dir: ${TEACHER_CACHE_DIR}"
    echo "train_data: ${TRAIN_DATA}"
    echo "eval_data: ${EVAL_DATA}"
    echo "train_batch_size: ${TRAIN_BATCH_SIZE}"
    echo "target_steps: ${TARGET_STEPS}"
    echo "max_samples: ${MAX_SAMPLES}"
    echo "post_eval_nnodes: ${POST_EVAL_NNODES}"
    echo "post_eval_nproc_per_node: ${POST_EVAL_NPROC_PER_NODE}"
    echo "archive_output_root: ${ARCHIVE_OUTPUT_ROOT}"
    echo "archive_shared_teacher_cache_mode: ${ARCHIVE_SHARED_TEACHER_CACHE_MODE}"
    echo "launcher_snapshot: ${LAUNCHER_SNAPSHOT_PATH}"
    echo "job_script: ${JOB_SCRIPT}"
  } > "${TRAIN_CONFIG_SUMMARY_PATH}"
}

write_final_status() {
  {
    echo "# Auto-generated final status"
    echo "# UTC: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    printf "TRAIN_RC=%q\n" "${TRAIN_RC:-0}"
    printf "EVAL_RC=%q\n" "${EVAL_RC:-0}"
    printf "ARCHIVE_RC=%q\n" "${ARCHIVE_RC:-0}"
    printf "SHARED_CACHE_ARCHIVE_RC=%q\n" "${SHARED_CACHE_ARCHIVE_RC:-0}"
    printf "FINAL_RC=%q\n" "${FINAL_RC:-0}"
    printf "RUN_DIR=%q\n" "${RUN_DIR:-}"
    printf "SAVE_PATH=%q\n" "${SAVE_PATH:-}"
    printf "POST_EVAL_LOG_DIR=%q\n" "${POST_EVAL_LOG_DIR:-}"
    printf "TRAIN_CONFIG_SNAPSHOT_PATH=%q\n" "${TRAIN_CONFIG_SNAPSHOT_PATH:-}"
    printf "TRAIN_CONFIG_SUMMARY_PATH=%q\n" "${TRAIN_CONFIG_SUMMARY_PATH:-}"
  } > "${RUN_DIR}/final_status.env"
}

head_teacher_gpu_count="$(count_csv_items "${HEAD_TEACHER_CUDA_VISIBLE_DEVICES}")"
worker_teacher_gpu_count="$(count_csv_items "${WORKER_TEACHER_CUDA_VISIBLE_DEVICES}")"
head_student_gpu_count="$(count_csv_items "${HEAD_STUDENT_CUDA_VISIBLE_DEVICES}")"
worker_student_gpu_count="$(count_csv_items "${WORKER_STUDENT_CUDA_VISIBLE_DEVICES}")"

if (( head_teacher_gpu_count % TEACHER_TP_SIZE != 0 )); then
  echo "[ERROR] head teacher gpu count must be divisible by TEACHER_TP_SIZE"
  exit 1
fi
if (( worker_teacher_gpu_count % TEACHER_TP_SIZE != 0 )); then
  echo "[ERROR] worker teacher gpu count must be divisible by TEACHER_TP_SIZE"
  exit 1
fi
if (( head_student_gpu_count != ACTOR_GPUS )); then
  echo "[ERROR] head student gpu count must equal ACTOR_GPUS for this 2-node layout"
  exit 1
fi
if (( worker_student_gpu_count != CRITIC_GPUS )); then
  echo "[ERROR] worker student gpu count must equal CRITIC_GPUS for this 2-node layout"
  exit 1
fi
if (( TRAIN_BATCH_SIZE != N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE )); then
  echo "[ERROR] TRAIN_BATCH_SIZE must equal N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE"
  exit 1
fi
if (( TRAIN_BATCH_SIZE % (MICRO_TRAIN_BATCH_SIZE * ACTOR_GPUS) != 0 )); then
  echo "[ERROR] train_batch_size % (micro_train_batch_size * actor_gpus) != 0"
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

HEAD_TEACHER_WORKER_COUNT="$((head_teacher_gpu_count / TEACHER_TP_SIZE))"
WORKER_TEACHER_WORKER_COUNT="$((worker_teacher_gpu_count / TEACHER_TP_SIZE))"
HEAD_TEACHER_API_BASE="$(build_teacher_urls "${HEAD_NODE_IP}" "${TEACHER_BASE_PORT}" "${HEAD_TEACHER_WORKER_COUNT}")"
WORKER_TEACHER_API_BASE="$(build_teacher_urls "${WORKER_NODE_IP}" "${TEACHER_BASE_PORT}" "${WORKER_TEACHER_WORKER_COUNT}")"
TEACHER_API_BASE="${HEAD_TEACHER_API_BASE},${WORKER_TEACHER_API_BASE}"

echo "========== G3 2-node once launcher =========="
echo "RUN_DIR:                    ${RUN_DIR}"
echo "HEAD_NODE / IP:             ${HEAD_NODE} / ${HEAD_NODE_IP}"
echo "WORKER_NODE / IP:           ${WORKER_NODE} / ${WORKER_NODE_IP}"
echo "Head teacher GPUs:          ${HEAD_TEACHER_CUDA_VISIBLE_DEVICES}"
echo "Worker teacher GPUs:        ${WORKER_TEACHER_CUDA_VISIBLE_DEVICES}"
echo "Head student GPUs:          ${HEAD_STUDENT_CUDA_VISIBLE_DEVICES}"
echo "Worker student GPUs:        ${WORKER_STUDENT_CUDA_VISIBLE_DEVICES}"
echo "Teacher worker count:       ${HEAD_TEACHER_WORKER_COUNT} + ${WORKER_TEACHER_WORKER_COUNT}"
echo "Teacher API:                ${TEACHER_API_BASE}"
echo "Actor/Critic GPUs:          ${ACTOR_GPUS}/${CRITIC_GPUS}"
echo "Actor/Critic nodes:         ${ACTOR_NUM_NODES}/${CRITIC_NUM_NODES}"
echo "Target steps:               ${TARGET_STEPS}"
echo "Max samples:                ${MAX_SAMPLES}"
echo "Post-train eval:            ${EVAL_AFTER_TRAIN}"
echo "Post-eval nnodes:           ${POST_EVAL_NNODES}"
echo "Post-eval nproc/node:       ${POST_EVAL_NPROC_PER_NODE}"
echo "Single training submitter:  ${HEAD_NODE}"
echo "============================================="

write_run_metadata

LOCAL_TEACHER_PIDS=()
RAY_HEAD_PID=""
RUNTIME_STOPPED=0

stop_runtime_processes() {
  local pid
  if [[ "${RUNTIME_STOPPED}" == "1" ]]; then
    return 0
  fi

  echo "[cleanup] stopping local teacher workers..."
  for pid in "${LOCAL_TEACHER_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${LOCAL_TEACHER_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
  done

  echo "[cleanup] stopping local ray..."
  if [[ -n "${RAY_HEAD_PID}" ]] && kill -0 "${RAY_HEAD_PID}" 2>/dev/null; then
    kill "${RAY_HEAD_PID}" 2>/dev/null || true
    wait "${RAY_HEAD_PID}" 2>/dev/null || true
  fi
  ray stop --force >/dev/null 2>&1 || true

  echo "[cleanup] stopping worker-side teacher/ray..."
  ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -s -- '${PID_DIR}'" <<'EOF' >/dev/null 2>&1 || true
set +e
PID_DIR="$1"
shopt -s nullglob
for pid_file in "${PID_DIR}"/teacher_worker_*.pid "${PID_DIR}"/ray_worker.pid; do
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
  RUNTIME_STOPPED=1
}

cleanup() {
  stop_runtime_processes || true
}
trap cleanup EXIT INT TERM

wait_for_http_health() {
  local url="$1"
  local waited=0
  until curl -sf "${url%/v1}/health" >/dev/null; do
    sleep 3
    waited=$((waited + 3))
    if (( waited >= TEACHER_WAIT_SECONDS )); then
      echo "[ERROR] health timeout: ${url}"
      return 1
    fi
  done
}

echo "[1/5] connectivity check to worker..."
ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -lc 'hostname'" >/dev/null

echo "[2/5] launching worker-side teacher services..."
ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -s -- \
  '${RUN_DIR}' \
  '${TEACHER_VLLM_BIN}' \
  '${TEACHER_MODEL_PATH}' \
  '${TEACHER_MODEL_NAME}' \
  '${WORKER_TEACHER_CUDA_VISIBLE_DEVICES}' \
  '${TEACHER_BASE_PORT}' \
  '${TEACHER_TP_SIZE}' \
  '${TEACHER_DTYPE}' \
  '${TEACHER_API_KEY}' \
  '${TEACHER_MAX_MODEL_LEN}' \
  '${TEACHER_MAX_NUM_SEQS}' \
  '${TEACHER_MAX_BATCHED_TOKENS}' \
  '${TEACHER_GPU_MEMORY_UTIL}'" <<'EOF'
set -euo pipefail
RUN_DIR="$1"
TEACHER_VLLM_BIN="$2"
TEACHER_MODEL_PATH="$3"
TEACHER_MODEL_NAME="$4"
GPU_CSV="$5"
BASE_PORT="$6"
TP_SIZE="$7"
DTYPE="$8"
API_KEY="$9"
MAX_MODEL_LEN="${10}"
MAX_NUM_SEQS="${11}"
MAX_BATCHED_TOKENS="${12}"
GPU_MEMORY_UTIL="${13}"

mkdir -p "${RUN_DIR}/teacher_logs"
mkdir -p "${RUN_DIR}/pids"
IFS=',' read -r -a GPU_IDS <<< "${GPU_CSV}"
WORKER_COUNT=$(( ${#GPU_IDS[@]} / TP_SIZE ))
for (( w=0; w<WORKER_COUNT; w++ )); do
  port=$(( BASE_PORT + w ))
  gpu_start=$(( w * TP_SIZE ))
  worker_gpus=""
  for (( g=gpu_start; g<gpu_start+TP_SIZE; g++ )); do
    [[ -n "${worker_gpus}" ]] && worker_gpus+=","
    worker_gpus+="${GPU_IDS[$g]}"
  done
  log_file="${RUN_DIR}/teacher_logs/worker_${HOSTNAME}_${w}.log"
  nohup bash -lc "
    CUDA_VISIBLE_DEVICES='${worker_gpus}' \
    '${TEACHER_VLLM_BIN}' serve '${TEACHER_MODEL_PATH}' \
      --served-model-name '${TEACHER_MODEL_NAME}' \
      --host 0.0.0.0 \
      --port '${port}' \
      --tensor-parallel-size '${TP_SIZE}' \
      --dtype '${DTYPE}' \
      --api-key '${API_KEY}' \
      --generation-config vllm \
      --max-model-len '${MAX_MODEL_LEN}' \
      --max-num-seqs '${MAX_NUM_SEQS}' \
      --max-num-batched-tokens '${MAX_BATCHED_TOKENS}' \
      --gpu-memory-utilization '${GPU_MEMORY_UTIL}' \
      --limit-mm-per-prompt '{\"image\":0,\"video\":0,\"audio\":0}' \
      --enable-chunked-prefill
  " > "${log_file}" 2>&1 &
  echo $! > "${RUN_DIR}/pids/teacher_worker_${w}.pid"
done
EOF

echo "[3/5] launching head-side teacher services..."
IFS=',' read -r -a _HEAD_GPU_IDS <<< "${HEAD_TEACHER_CUDA_VISIBLE_DEVICES}"
for (( _w=0; _w<HEAD_TEACHER_WORKER_COUNT; _w++ )); do
  _port=$(( TEACHER_BASE_PORT + _w ))
  _gpu_start=$(( _w * TEACHER_TP_SIZE ))
  _worker_gpus=""
  for (( _g=_gpu_start; _g<_gpu_start+TEACHER_TP_SIZE; _g++ )); do
    [[ -n "${_worker_gpus}" ]] && _worker_gpus+=","
    _worker_gpus+="${_HEAD_GPU_IDS[$_g]}"
  done
  _log="${TEACHER_LOG_DIR}/worker_${HEAD_NODE}_${_w}.log"
  CUDA_VISIBLE_DEVICES="${_worker_gpus}" \
  "${TEACHER_VLLM_BIN}" serve "${TEACHER_MODEL_PATH}" \
    --served-model-name "${TEACHER_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${_port}" \
    --tensor-parallel-size "${TEACHER_TP_SIZE}" \
    --dtype "${TEACHER_DTYPE}" \
    --api-key "${TEACHER_API_KEY}" \
    --generation-config vllm \
    --max-model-len "${TEACHER_MAX_MODEL_LEN}" \
    --max-num-seqs "${TEACHER_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${TEACHER_MAX_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${TEACHER_GPU_MEMORY_UTIL}" \
    --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
    --enable-chunked-prefill \
    > "${_log}" 2>&1 &
  LOCAL_TEACHER_PIDS+=("$!")
done

echo "[4/5] waiting for teacher health checks..."
IFS=',' read -r -a _ALL_TEACHER_URLS <<< "${TEACHER_API_BASE}"
for _url in "${_ALL_TEACHER_URLS[@]}"; do
  wait_for_http_health "${_url}"
  echo "  [teacher] healthy: ${_url}"
done

echo "[5/5] starting ray cluster..."
ray stop --force >/dev/null 2>&1 || true
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

ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -lc '
  set -euo pipefail
  ray stop --force >/dev/null 2>&1 || true
  mkdir -p ${PID_DIR}
  nohup bash -lc \"CUDA_VISIBLE_DEVICES=${WORKER_STUDENT_CUDA_VISIBLE_DEVICES} ray start --address ${HEAD_NODE_IP}:${RAY_PORT} --num-gpus ${worker_student_gpu_count} --block\" > ${RAY_LOG_DIR}/worker.log 2>&1 &
  echo \$! > ${PID_DIR}/ray_worker.pid
'"

waited=0
until ray status --address "${HEAD_NODE_IP}:${RAY_PORT}" >/dev/null 2>&1; do
  sleep 2
  waited=$((waited + 2))
  if (( waited >= RAY_WAIT_SECONDS )); then
    echo "[ERROR] ray cluster did not become ready in time."
    exit 1
  fi
done

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

cd '${REPO_ROOT}'

PREFETCH_FLAGS=()
if [[ '${ENABLE_TEACHER_PREFETCH}' == 'true' ]]; then
  PREFETCH_FLAGS=(
    --enable_teacher_prefetch
    --prefetch_depth '${PREFETCH_DEPTH}'
    --prefetch_max_workers '${PREFETCH_MAX_WORKERS}'
  )
fi

'${STUDENT_PYTHON_BIN}' -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --pretrain_mode --no_chat_template \
  --disable_ds_ckpt --colocate_actor_ref --colocate_critic_reward \
  --gradient_checkpointing --use_kl_loss --use_whitening --enable_ema \
  --feature_adapter_enable \
  --feature_adapter_type residual_bottleneck \
  --feature_adapter_rank '${FEATURE_ADAPTER_RANK}' \
  --feature_adapter_dropout '${FEATURE_ADAPTER_DROPOUT}' \
  --feature_adapter_unfreeze_layers '${UNFREEZE_LAYERS}' \
  --distribution_reward_type cf_l1oo \
  --feature_map_type identity --rff_num_features 128 --rff_sigma 1.0 --rff_seed 43 \
  --cf_num_freqs 128 --cf_sigma 1.0 --cf_seed 43 --cf_alpha 0.5 --cf_beta 0.5 --cf_reward_scale 1.0 \
  --cf_target_mode teacher --cf_teacher_lambda '${CF_TEACHER_LAMBDA}' --cf_teacher_n_samples '${CF_TEACHER_N_SAMPLES}' \
  --teacher_backend remote \
  --teacher_api_base '${TEACHER_API_BASE}' \
  --teacher_api_key '${TEACHER_API_KEY}' \
  --teacher_api_style completions \
  --teacher_model_name '${TEACHER_MODEL_NAME}' \
  --teacher_timeout '${TEACHER_TIMEOUT}' \
  --teacher_max_retries '${TEACHER_MAX_RETRIES}' \
  --teacher_remote_batch_size '${TEACHER_REMOTE_BATCH_SIZE}' \
  --teacher_temperature '${TEACHER_TEMPERATURE}' \
  --teacher_top_p '${TEACHER_TOP_P}' \
  --teacher_max_new_tokens '${TEACHER_MAX_NEW_TOKENS}' \
  --teacher_system_prompt_text '${TEACHER_SYSTEM_PROMPT_TEXT}' \
  --teacher_system_prompt_id '${TEACHER_SYSTEM_PROMPT_ID}' \
  --teacher_cache_enable --teacher_cache_dir '${TEACHER_CACHE_DIR}' \
  "\${PREFETCH_FLAGS[@]}" \
  --embed_method last_token --critic_sequence_level last_token \
  --critic_learning_rate '${CRITIC_LR}' \
  --critic_lr_head '${CRITIC_LR_HEAD}' \
  --critic_classifier_loss_coef '${CRITIC_CLASSIFIER_LOSS_COEF}' \
  --critic_direct_discrepancy_coef '${CRITIC_DIRECT_DISCREPANCY_COEF}' \
  --critic_direct_discrepancy_target '${CRITIC_DIRECT_DISCREPANCY_TARGET}' \
  --ema_beta '${EMA_BETA}' \
  --ce_loss_coef '${CE_LOSS_COEF}' \
  --diversity_rew_coef '${DIVERSITY_REW_COEF}' \
  --alignment_rew_coef '${ALIGNMENT_REW_COEF}' \
  --actor_learning_rate '${ACTOR_LR}' \
  --pretrain '${MODEL_PATH}' --critic_pretrain '${MODEL_PATH}' \
  --prompt_data '${TRAIN_DATA}' --eval_dataset '${EVAL_DATA}' \
  --input_key question --label_key answer --output_key answer \
  --prompt_split train --eval_split test \
  --prompt_max_len '${PROMPT_MAX_LEN}' \
  --context_max_len '${CONTEXT_MAX_LEN}' \
  --generate_max_len '${GENERATE_MAX_LEN}' \
  --stride '${STRIDE}' \
  --n_samples_per_prompt '${N_SAMPLES_PER_PROMPT}' \
  --rollout_batch_size '${ROLLOUT_BATCH_SIZE}' \
  --train_batch_size '${TRAIN_BATCH_SIZE}' \
  --micro_train_batch_size '${MICRO_TRAIN_BATCH_SIZE}' \
  --micro_rollout_batch_size '${MICRO_ROLLOUT_BATCH_SIZE}' \
  --micro_reward_batch_size '${MICRO_REWARD_BATCH_SIZE}' \
  --max_samples '${MAX_SAMPLES}' \
  --num_episodes '${NUM_EPISODES}' \
  --max_epochs '${MAX_EPOCHS}' \
  --actor_num_nodes '${ACTOR_NUM_NODES}' --actor_num_gpus_per_node '${ACTOR_GPUS}' \
  --critic_num_nodes '${CRITIC_NUM_NODES}' --critic_num_gpus_per_node '${CRITIC_GPUS}' \
  --ref_num_nodes '${REF_NUM_NODES}' --ref_num_gpus_per_node '${REF_GPUS}' \
  --reward_num_nodes '${REWARD_NUM_NODES}' --reward_num_gpus_per_node '${REWARD_GPUS}' \
  --advantage_estimator rloo --init_kl_coef 0.0 --kl_estimator k2 \
  --temperature 0.6 --top_p 1.0 \
  --zero_stage 2 --lr_warmup_ratio 0.03 --critic_lr_warmup_ratio 0.0 \
  --seed 43 \
  --eval_steps '${EVAL_STEPS}' \
  --eval_max_samples '${EVAL_MAX_SAMPLES}' \
  --eval_generate_max_len '${EVAL_GENERATE_MAX_LEN}' \
  --logging_steps 10 \
  --save_steps '${SAVE_STEPS}' --save_even_count '${SAVE_EVEN_COUNT}' --save_hf_ckpt \
  --use_tensorboard '${TB_DIR}' \
  --save_path '${SAVE_PATH}' --ckpt_path '${SAVE_PATH}/ckpt' \
  --wandb_run_name '${RUN_NAME}' \
  2>&1 | tee '${RUN_DIR}/train.log'
EOF

chmod +x "${JOB_SCRIPT}"

echo "[train] submitting one Ray job..."
echo "[train] the worker node does not run train_ebft_ray directly; it only hosts teacher/ray worker processes."
TRAIN_RC=0
EVAL_RC=0
ARCHIVE_RC=0
SHARED_CACHE_ARCHIVE_RC=0

set +e
ray job submit \
  --address "http://${HEAD_NODE_IP}:${RAY_DASHBOARD_PORT}" \
  -- bash "${JOB_SCRIPT}" | tee "${JOB_LOG}"
TRAIN_RC=$?
set -e

if (( TRAIN_RC != 0 )); then
  echo "[ERROR] training failed with exit code ${TRAIN_RC}"
fi

echo "[post-run] stopping teacher/ray processes before eval/archive ..."
stop_runtime_processes

if [[ "${EVAL_AFTER_TRAIN}" == "true" ]]; then
  if (( TRAIN_RC == 0 )); then
    echo "[post-eval] running eval from checkpoint: ${SAVE_PATH}"
    set +e
    EVAL_DATA="${EVAL_DATA}" \
    REPO_ROOT="${REPO_ROOT}" \
    STUDENT_VENV="${STUDENT_VENV}" \
    STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN}" \
    STUDENT_CUDA_VISIBLE_DEVICES="${POST_EVAL_HEAD_CUDA_VISIBLE_DEVICES}" \
    HEAD_NODE="${HEAD_NODE}" \
    HEAD_NODE_IP="${HEAD_NODE_IP}" \
    WORKER_NODE="${WORKER_NODE}" \
    WORKER_NODE_IP="${WORKER_NODE_IP}" \
    SSH_USER="${SSH_USER}" \
    SSH_OPTS="${SSH_OPTS}" \
    POST_EVAL_NNODES="${POST_EVAL_NNODES}" \
    POST_EVAL_NPROC="${POST_EVAL_NPROC}" \
    POST_EVAL_NPROC_PER_NODE="${POST_EVAL_NPROC_PER_NODE}" \
    POST_EVAL_HEAD_CUDA_VISIBLE_DEVICES="${POST_EVAL_HEAD_CUDA_VISIBLE_DEVICES}" \
    POST_EVAL_WORKER_CUDA_VISIBLE_DEVICES="${POST_EVAL_WORKER_CUDA_VISIBLE_DEVICES}" \
    POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES}" \
    POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN}" \
    POST_EVAL_MAX_NEW_TOKENS="${POST_EVAL_MAX_NEW_TOKENS}" \
    POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE}" \
    POST_EVAL_TOP_P="${POST_EVAL_TOP_P}" \
    POST_EVAL_MICRO_BATCH_SIZE="${POST_EVAL_MICRO_BATCH_SIZE}" \
    POST_EVAL_MASTER_PORT="${POST_EVAL_MASTER_PORT}" \
    LOG_DIR="${POST_EVAL_LOG_DIR}" \
    EVAL_TAG="${POST_EVAL_TAG}" \
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

if [[ "${ARCHIVE_OUTPUTS_AFTER_RUN}" == "true" ]]; then
  set +e
  archive_run_outputs "${ARCHIVE_OUTPUT_ROOT}"
  ARCHIVE_RC=$?
  if (( ARCHIVE_RC == 0 )); then
    archive_shared_teacher_cache "${ARCHIVE_SHARED_TEACHER_CACHE_MODE}" "${ARCHIVE_SHARED_TEACHER_CACHE_DIR}"
    SHARED_CACHE_ARCHIVE_RC=$?
  else
    SHARED_CACHE_ARCHIVE_RC=0
  fi
  set -e
  if (( ARCHIVE_RC != 0 )); then
    echo "[ERROR] archiving run outputs failed with exit code ${ARCHIVE_RC}"
  fi
  if (( SHARED_CACHE_ARCHIVE_RC != 0 )); then
    echo "[ERROR] archiving shared teacher cache failed with exit code ${SHARED_CACHE_ARCHIVE_RC}"
  fi
fi

FINAL_RC=0
if (( TRAIN_RC != 0 )); then
  FINAL_RC=${TRAIN_RC}
elif (( EVAL_RC != 0 )); then
  FINAL_RC=${EVAL_RC}
elif (( ARCHIVE_RC != 0 )); then
  FINAL_RC=${ARCHIVE_RC}
elif (( SHARED_CACHE_ARCHIVE_RC != 0 )); then
  FINAL_RC=${SHARED_CACHE_ARCHIVE_RC}
fi

write_final_status

echo "[done] logs: ${RUN_DIR}"
exit "${FINAL_RC}"
