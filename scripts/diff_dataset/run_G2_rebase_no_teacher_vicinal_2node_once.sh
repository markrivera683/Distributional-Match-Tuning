#!/usr/bin/env bash
# Standalone diff-dataset G2 no-teacher VICINAL 2-node launcher.
#
# Dataset: OpenCodeInstruct train pool, MBPP + HumanEval post-eval.
# Model:   Qwen3.5-4B student by default.
# Reward:  cf_l1oo with cf_target_mode=vicinal. No teacher process, no
#          teacher backend, no teacher cache, no teacher prefetch.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/diff_dataset/_common.sh
source "${SCRIPT_DIR}/_common.sh"

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "${csv}" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"${csv}"
}

resolve_host_ip() {
  local host="$1"
  local ip=""
  local waited=0
  local wait_seconds="${HOST_RESOLVE_WAIT_SECONDS:-60}"
  local retry_seconds="${HOST_RESOLVE_RETRY_SECONDS:-2}"

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
    if (( waited >= wait_seconds )); then
      echo "[ERROR] failed to resolve IPv4 for host: ${host}" >&2
      exit 1
    fi
    sleep "${retry_seconds}"
    waited=$((waited + retry_seconds))
  done
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[ERROR] required command not found: ${cmd}"
    exit 1
  fi
}

prepare_diff_datasets

# ---------------------------------------------------------------------------
# 1) Deployment mode
# ---------------------------------------------------------------------------
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
elif [[ -n "${RANK:-}" && -n "${MASTER_ADDR:-}" && "${DLC_WORLD_SIZE}" -gt 1 ]]; then
  DLC_MODE="true"
  DLC_NODE_RANK="${RANK}"
  DLC_MASTER_ADDR="${MASTER_ADDR}"
fi

if [[ "${DLC_MODE}" == "true" ]]; then
  [[ -n "${DLC_MASTER_ADDR}" ]] || { echo "[ERROR] DLC mode detected but MASTER_ADDR is empty"; exit 1; }
  HEAD_NODE="${HEAD_NODE:-${DLC_MASTER_ADDR}}"
  WORKER_NODE="${WORKER_NODE:-dlc-rank-${DLC_NODE_RANK}-pod}"
  echo "[INFO] DLC mode: rank=${DLC_NODE_RANK} world_size=${DLC_WORLD_SIZE} master=${DLC_MASTER_ADDR}"
elif [[ -z "${HEAD_NODE}" && -z "${WORKER_NODE}" ]]; then
  SINGLE_NODE_MODE="true"
  HEAD_NODE="$(hostname)"
  WORKER_NODE="${HEAD_NODE}"
  echo "[INFO] single-node fallback: HEAD_NODE=WORKER_NODE=${HEAD_NODE}"
elif [[ -z "${HEAD_NODE}" || -z "${WORKER_NODE}" ]]; then
  echo "[ERROR] HEAD_NODE / WORKER_NODE must both be set, or both be unset."
  exit 1
fi

SKIP_SSH_BOOTSTRAP="false"
if [[ "${SINGLE_NODE_MODE}" == "true" || "${DLC_MODE}" == "true" ]]; then
  SKIP_SSH_BOOTSTRAP="true"
fi

HEAD_NODE_IP="${HEAD_NODE_IP:-$(resolve_host_ip "${HEAD_NODE}")}"
if [[ "${DLC_MODE}" == "true" ]]; then
  WORKER_NODE_IP="${WORKER_NODE_IP:-${HEAD_NODE_IP}}"
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
if [[ "${DLC_MODE}" != "true" && "${SINGLE_NODE_MODE}" != "true" ]]; then
  if [[ "${CURRENT_HOSTNAME}" != "${HEAD_NODE}" && "${CURRENT_HOSTNAME_SHORT}" != "${HEAD_NODE}" ]]; then
    case " ${CURRENT_HOST_IPS} " in
      *" ${HEAD_NODE_IP} "*) ;;
      *)
        echo "[ERROR] run this script only on the head node."
        echo "        current: ${CURRENT_HOSTNAME}"
        echo "        head:    ${HEAD_NODE} (${HEAD_NODE_IP})"
        exit 1
        ;;
    esac
  fi
fi

# ---------------------------------------------------------------------------
# 2) Data / model / env
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-4B}"
TRAIN_DATA="${TRAIN_DATA:-${PREPARED_DATA_DIR}/opencodeinstruct_qa_100k.jsonl}"
EVAL_DATA="${EVAL_DATA:-${MBPP_EVAL_DATA}}"
POST_EVAL_SCRIPT="${POST_EVAL_SCRIPT:-${SCRIPT_DIR}/posteval_code_2node.sh}"
CODE_POST_EVAL_WORKER="${CODE_POST_EVAL_WORKER:-${SCRIPT_DIR}/posteval_code_pass1.sh}"

STUDENT_VENV="${STUDENT_VENV:-/mnt/workspace/venvs/.venv}"
TEACHER_VENV="${TEACHER_VENV:-/mnt/workspace/venvs/.teacherVenv}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
ANALYSIS_VENV="${ANALYSIS_VENV:-${STUDENT_VENV}}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"

export HF_HOME="${HF_HOME:-/mnt/data/ebft-distribution-new/caches/hf}"
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
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
[[ "${NCCL_P2P_DISABLE:-}" == "1" ]] && unset NCCL_P2P_DISABLE
export NCCL_NET_GDR_DISABLE="${NCCL_NET_GDR_DISABLE:-1}"

if [[ -d "${STUDENT_VENV}/bin" ]]; then
  export PATH="${STUDENT_VENV}/bin:${PATH}"
fi

# ---------------------------------------------------------------------------
# 3) Training knobs
# ---------------------------------------------------------------------------
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
DEFAULT_MAX_SAMPLES="$((TARGET_STEPS * TRAIN_BATCH_SIZE / N_SAMPLES_PER_PROMPT / NUM_EPISODES / MAX_EPOCHS))"
MAX_SAMPLES="${MAX_SAMPLES:-${DEFAULT_MAX_SAMPLES}}"

FEATURE_MAP_TYPE="${FEATURE_MAP_TYPE:-identity}"
RFF_NUM_FEATURES="${RFF_NUM_FEATURES:-128}"
RFF_SIGMA="${RFF_SIGMA:-1.0}"
RFF_SEED="${RFF_SEED:-43}"
CF_NUM_FREQS="${CF_NUM_FREQS:-128}"
CF_SIGMA="${CF_SIGMA:-1.0}"
CF_SEED="${CF_SEED:-43}"
CF_ALPHA="${CF_ALPHA:-0.5}"
CF_BETA="${CF_BETA:-0.5}"
CF_REWARD_SCALE="${CF_REWARD_SCALE:-1.0}"
CF_TARGET_MODE="${CF_TARGET_MODE:-vicinal}"
CF_TARGET_NUM_REFS="${CF_TARGET_NUM_REFS:-8}"
CF_TARGET_STD="${CF_TARGET_STD:-0.05}"
CF_TARGET_SEED="${CF_TARGET_SEED:-43}"
CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.0}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-${N_SAMPLES_PER_PROMPT}}"

ACTOR_LR="${ACTOR_LR:-1e-6}"
CRITIC_LR="${CRITIC_LR:-0}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-0}"
CE_LOSS_COEF="${CE_LOSS_COEF:-0.03}"
ENABLE_EMA="${ENABLE_EMA:-false}"
EMA_BETA="${EMA_BETA:-0.99}"
FEATURE_ADAPTER_ENABLE="${FEATURE_ADAPTER_ENABLE:-false}"
FEATURE_ADAPTER_RANK="${FEATURE_ADAPTER_RANK:-64}"
FEATURE_ADAPTER_DROPOUT="${FEATURE_ADAPTER_DROPOUT:-0.0}"
UNFREEZE_LAYERS="${UNFREEZE_LAYERS:-0}"
CRITIC_CLASSIFIER_LOSS_COEF="${CRITIC_CLASSIFIER_LOSS_COEF:-0.0}"
CRITIC_DIRECT_DISCREPANCY_COEF="${CRITIC_DIRECT_DISCREPANCY_COEF:-0.0}"
CRITIC_DIRECT_DISCREPANCY_TARGET="${CRITIC_DIRECT_DISCREPANCY_TARGET:-ema_gt}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"
DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-1.0}"

ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-rloo}"
INIT_KL_COEF="${INIT_KL_COEF:-0.0}"
KL_ESTIMATOR="${KL_ESTIMATOR:-k2}"
ZERO_STAGE="${ZERO_STAGE:-3}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.03}"
CRITIC_LR_WARMUP_RATIO="${CRITIC_LR_WARMUP_RATIO:-0.0}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
GLOBAL_SEED="${GLOBAL_SEED:-43}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

EVAL_STEPS="${EVAL_STEPS:--1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-50}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-${GENERATE_MAX_LEN}}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_EVEN_COUNT="${SAVE_EVEN_COUNT:-10}"

# 2-node no-teacher: all 8 GPUs are available for student on each node.
HEAD_STUDENT_CUDA_VISIBLE_DEVICES="${HEAD_STUDENT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ "${SINGLE_NODE_MODE}" == "true" ]]; then
  WORKER_STUDENT_CUDA_VISIBLE_DEVICES="${WORKER_STUDENT_CUDA_VISIBLE_DEVICES:-}"
  ACTOR_GPUS="${ACTOR_GPUS:-4}"
  CRITIC_GPUS="${CRITIC_GPUS:-4}"
else
  WORKER_STUDENT_CUDA_VISIBLE_DEVICES="${WORKER_STUDENT_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  ACTOR_GPUS="${ACTOR_GPUS:-8}"
  CRITIC_GPUS="${CRITIC_GPUS:-8}"
fi
REF_GPUS="${REF_GPUS:-${ACTOR_GPUS}}"
REWARD_GPUS="${REWARD_GPUS:-${CRITIC_GPUS}}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
CRITIC_NUM_NODES="${CRITIC_NUM_NODES:-1}"
REF_NUM_NODES="${REF_NUM_NODES:-1}"
REWARD_NUM_NODES="${REWARD_NUM_NODES:-1}"

# ---------------------------------------------------------------------------
# 4) Post-eval / output
# ---------------------------------------------------------------------------
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
RUN_TWO_ROUND_EVAL="${RUN_TWO_ROUND_EVAL:-${EVAL_AFTER_TRAIN}}"
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-$(count_csv_items "${MODEL_CUDA_VISIBLE_DEVICES}")}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_PROGRESS_BATCH_SIZE="${VLLM_PROGRESS_BATCH_SIZE:-128}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_SEED="${VLLM_SEED:-1234}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
POST_EVAL_TAG="${POST_EVAL_TAG:-post_train}"

ARCHIVE_OUTPUTS_AFTER_RUN="${ARCHIVE_OUTPUTS_AFTER_RUN:-true}"
ARCHIVE_OUTPUT_ROOT="${ARCHIVE_OUTPUT_ROOT:-/mnt/data/ebft-teacher-distribution/outputs_diff_dataset_g2_no_teacher_vicinal}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_WAIT_SECONDS="${RAY_WAIT_SECONDS:-120}"

if [[ "${DLC_MODE}" == "true" && -z "${RUN_NAME:-}" ]]; then
  _dlc_job_id="$(hostname | sed -E 's/^(dlc[a-z0-9]+)-(master|worker)-[0-9]+$/\1/' || true)"
  if [[ -n "${_dlc_job_id}" && "${_dlc_job_id}" != "$(hostname)" ]]; then
    RUN_NAME="diff_g2_no_teacher_vicinal_${_dlc_job_id}"
  fi
fi
if [[ -z "${RUN_NAME:-}" && "${DLC_MODE}" != "true" ]]; then
  RUN_NAME="diff_g2_no_teacher_vicinal_qwen35_4b_2node_$(date +%m%d_%H%M)"
fi
RUN_NAME="${RUN_NAME:-diff_g2_no_teacher_vicinal_qwen35_4b_2node_$(date +%m%d_%H%M)}"
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
SCRIPT_LOG_PATH="${SCRIPT_LOG_PATH:-${RUN_DIR}/$(basename "$0" .sh).log}"
mkdir -p "${RUN_DIR}" "${SAVE_PATH}" "${TB_DIR}" "${RAY_LOG_DIR}" "${PID_DIR}"
exec > >(tee -a "${SCRIPT_LOG_PATH}") 2>&1

# ---------------------------------------------------------------------------
# 5) Validation / metadata
# ---------------------------------------------------------------------------
require_cmd ray
if [[ "${SKIP_SSH_BOOTSTRAP}" != "true" ]]; then
  require_cmd ssh
fi

[[ -x "${STUDENT_PYTHON_BIN}" ]] || { echo "[ERROR] STUDENT_PYTHON_BIN not executable: ${STUDENT_PYTHON_BIN}"; exit 1; }
[[ -d "${REPO_ROOT}" ]] || { echo "[ERROR] REPO_ROOT not found: ${REPO_ROOT}"; exit 1; }
[[ -e "${MODEL_PATH}" ]] || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 1; }
[[ -e "${TRAIN_DATA}" ]] || { echo "[ERROR] TRAIN_DATA not found: ${TRAIN_DATA}"; exit 1; }
[[ -e "${EVAL_DATA}" ]] || { echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"; exit 1; }

if [[ "${RUN_TWO_ROUND_EVAL}" == "true" ]]; then
  [[ -x "${TEACHER_PYTHON_BIN}" ]] || { echo "[ERROR] TEACHER_PYTHON_BIN not executable: ${TEACHER_PYTHON_BIN}"; exit 1; }
  [[ -x "${ANALYSIS_PYTHON_BIN}" ]] || { echo "[ERROR] ANALYSIS_PYTHON_BIN not executable: ${ANALYSIS_PYTHON_BIN}"; exit 1; }
  [[ -f "${POST_EVAL_SCRIPT}" ]] || { echo "[ERROR] POST_EVAL_SCRIPT not found: ${POST_EVAL_SCRIPT}"; exit 1; }
  [[ -f "${CODE_POST_EVAL_WORKER}" ]] || { echo "[ERROR] CODE_POST_EVAL_WORKER not found: ${CODE_POST_EVAL_WORKER}"; exit 1; }
fi

head_student_gpu_count="$(count_csv_items "${HEAD_STUDENT_CUDA_VISIBLE_DEVICES}")"
worker_student_gpu_count="$(count_csv_items "${WORKER_STUDENT_CUDA_VISIBLE_DEVICES}")"

if [[ "${SINGLE_NODE_MODE}" == "true" ]]; then
  ACTOR_NUM_NODES=1
  CRITIC_NUM_NODES=1
  REF_NUM_NODES=1
  REWARD_NUM_NODES=1
  if (( head_student_gpu_count != ACTOR_GPUS + CRITIC_GPUS )); then
    echo "[ERROR] single-node visible GPUs (${head_student_gpu_count}) must equal ACTOR_GPUS + CRITIC_GPUS ($((ACTOR_GPUS + CRITIC_GPUS)))"
    exit 1
  fi
else
  if (( head_student_gpu_count != ACTOR_GPUS )); then
    echo "[ERROR] head student GPU count (${head_student_gpu_count}) must equal ACTOR_GPUS (${ACTOR_GPUS})"
    exit 1
  fi
  if (( worker_student_gpu_count != CRITIC_GPUS )); then
    echo "[ERROR] worker student GPU count (${worker_student_gpu_count}) must equal CRITIC_GPUS (${CRITIC_GPUS})"
    exit 1
  fi
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

write_run_metadata() {
  local vars=(
    RUN_NAME OUTPUT_ROOT RUN_DIR SAVE_PATH TB_DIR RAY_LOG_DIR PID_DIR JOB_SCRIPT JOB_LOG
    HEAD_NODE HEAD_NODE_IP WORKER_NODE WORKER_NODE_IP WORKER_SSH_TARGET SSH_OPTS
    HEAD_STUDENT_CUDA_VISIBLE_DEVICES WORKER_STUDENT_CUDA_VISIBLE_DEVICES
    ACTOR_GPUS CRITIC_GPUS REF_GPUS REWARD_GPUS ACTOR_NUM_NODES CRITIC_NUM_NODES REF_NUM_NODES REWARD_NUM_NODES
    MODEL_PATH TRAIN_DATA EVAL_DATA MBPP_EVAL_DATA HUMANEVAL_EVAL_DATA POST_EVAL_DATASETS
    N_SAMPLES_PER_PROMPT ROLLOUT_BATCH_SIZE TRAIN_BATCH_SIZE MICRO_TRAIN_BATCH_SIZE MICRO_ROLLOUT_BATCH_SIZE MICRO_REWARD_BATCH_SIZE
    PROMPT_MAX_LEN CONTEXT_MAX_LEN GENERATE_MAX_LEN STRIDE TARGET_STEPS MAX_SAMPLES
    FEATURE_MAP_TYPE RFF_NUM_FEATURES RFF_SIGMA RFF_SEED CF_NUM_FREQS CF_SIGMA CF_SEED CF_ALPHA CF_BETA CF_REWARD_SCALE
    CF_TARGET_MODE CF_TARGET_NUM_REFS CF_TARGET_STD CF_TARGET_SEED CF_TEACHER_LAMBDA CF_TEACHER_N_SAMPLES
    ACTOR_LR CRITIC_LR CRITIC_LR_HEAD CE_LOSS_COEF ENABLE_EMA EMA_BETA
    FEATURE_ADAPTER_ENABLE FEATURE_ADAPTER_RANK FEATURE_ADAPTER_DROPOUT UNFREEZE_LAYERS
    CRITIC_CLASSIFIER_LOSS_COEF CRITIC_DIRECT_DISCREPANCY_COEF CRITIC_DIRECT_DISCREPANCY_TARGET
    ALIGNMENT_REW_COEF DIVERSITY_REW_COEF
    ADVANTAGE_ESTIMATOR INIT_KL_COEF KL_ESTIMATOR ZERO_STAGE LR_WARMUP_RATIO CRITIC_LR_WARMUP_RATIO
    TEMPERATURE TOP_P GLOBAL_SEED LOGGING_STEPS
    EVAL_STEPS EVAL_MAX_SAMPLES EVAL_GENERATE_MAX_LEN SAVE_STEPS SAVE_EVEN_COUNT
    RUN_TWO_ROUND_EVAL POST_EVAL_SCRIPT CODE_POST_EVAL_WORKER POST_EVAL_MAX_SAMPLES
    MODEL_CUDA_VISIBLE_DEVICES VLLM_TP_SIZE VLLM_MAX_NUM_SEQS VLLM_PROGRESS_BATCH_SIZE VLLM_ENABLE_PREFIX_CACHING VLLM_SEED
    CODE_EVAL_MAX_NEW_TOKENS CODE_EVAL_TEMPERATURE CODE_EVAL_TOP_P CODE_EVAL_REPETITION_PENALTY CODE_EVAL_TIMEOUT_SECONDS
    ARCHIVE_OUTPUTS_AFTER_RUN ARCHIVE_OUTPUT_ROOT
  )
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
    echo "model_path: ${MODEL_PATH}"
    echo "train_data: ${TRAIN_DATA}"
    echo "eval_data: ${EVAL_DATA}"
    echo "distribution_reward_type: cf_l1oo"
    echo "cf_target_mode: ${CF_TARGET_MODE}"
    echo "teacher_in_reward: false"
    echo "enable_ema: ${ENABLE_EMA}"
    echo "ema_beta: ${EMA_BETA}"
    echo "feature_adapter_enable: ${FEATURE_ADAPTER_ENABLE}"
    echo "paper_gamma_ce_loss_coef: ${CE_LOSS_COEF}"
    echo "paper_alpha_diversity_over_alignment: ${DIVERSITY_REW_COEF}/${ALIGNMENT_REW_COEF}"
    echo "post_eval_script: ${POST_EVAL_SCRIPT}"
    echo "post_eval_datasets: ${POST_EVAL_DATASETS}"
  } > "${TRAIN_CONFIG_SUMMARY_PATH}"
}

write_final_status() {
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

archive_run_outputs() {
  local target_root="$1"
  local target_dir
  local old_run_dir="${RUN_DIR}"
  [[ -d "${RUN_DIR}" ]] || return 0
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
  write_run_metadata
}

write_run_metadata

echo "========== Diff-Dataset no-teacher ${CF_TARGET_MODE} 2-node =========="
echo "RUN_DIR:                  ${RUN_DIR}"
echo "HEAD_NODE / IP:           ${HEAD_NODE} / ${HEAD_NODE_IP}"
echo "WORKER_NODE / IP:         ${WORKER_NODE} / ${WORKER_NODE_IP}"
echo "Head student GPUs:        ${HEAD_STUDENT_CUDA_VISIBLE_DEVICES}"
echo "Worker student GPUs:      ${WORKER_STUDENT_CUDA_VISIBLE_DEVICES:-<none>}"
echo "Actor/Critic GPUs:        ${ACTOR_GPUS}/${CRITIC_GPUS}"
echo "MODEL_PATH:               ${MODEL_PATH}"
echo "TRAIN_DATA:               ${TRAIN_DATA}"
echo "EVAL_DATA:                ${EVAL_DATA}"
echo "distribution_reward_type: cf_l1oo"
echo "cf_target_mode:           ${CF_TARGET_MODE}"
echo "cf_target_num_refs:       ${CF_TARGET_NUM_REFS}"
echo "cf_teacher_lambda:        ${CF_TEACHER_LAMBDA}"
echo "enable_ema / beta:        ${ENABLE_EMA} / ${EMA_BETA}"
echo "feature_adapter:          ${FEATURE_ADAPTER_ENABLE}, rank=${FEATURE_ADAPTER_RANK}, unfreeze=${UNFREEZE_LAYERS}"
echo "critic lr/head:           ${CRITIC_LR}/${CRITIC_LR_HEAD}"
echo "critic aux losses:        classifier=${CRITIC_CLASSIFIER_LOSS_COEF}, direct=${CRITIC_DIRECT_DISCREPANCY_COEF}, target=${CRITIC_DIRECT_DISCREPANCY_TARGET}"
echo "paper gamma / CE coef:    ${CE_LOSS_COEF}"
echo "paper alpha proxy:        ${DIVERSITY_REW_COEF}/${ALIGNMENT_REW_COEF}"
echo "advantage / KL:           ${ADVANTAGE_ESTIMATOR}, init_kl=${INIT_KL_COEF}, estimator=${KL_ESTIMATOR}"
echo "zero/warmup:              ZeRO-${ZERO_STAGE}, lr=${LR_WARMUP_RATIO}, critic_lr=${CRITIC_LR_WARMUP_RATIO}"
echo "POST_EVAL_SCRIPT:         ${POST_EVAL_SCRIPT}"
echo "POST_EVAL_DATASETS:       ${POST_EVAL_DATASETS}"
echo "==============================================================="

# ---------------------------------------------------------------------------
# 6) Ray lifecycle
# ---------------------------------------------------------------------------
RAY_HEAD_PID=""
RUNTIME_STOPPED=0

stop_runtime_processes() {
  if [[ "${RUNTIME_STOPPED}" == "1" ]]; then
    return 0
  fi
  echo "[cleanup] stopping ray..."
  if [[ -n "${RAY_HEAD_PID}" ]] && kill -0 "${RAY_HEAD_PID}" 2>/dev/null; then
    kill "${RAY_HEAD_PID}" 2>/dev/null || true
    wait "${RAY_HEAD_PID}" 2>/dev/null || true
  fi
  ray stop --force >/dev/null 2>&1 || true
  if [[ "${SKIP_SSH_BOOTSTRAP}" != "true" ]]; then
    ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -lc 'ray stop --force >/dev/null 2>&1 || true'" >/dev/null 2>&1 || true
  fi
  RUNTIME_STOPPED=1
}
trap 'stop_runtime_processes || true' EXIT INT TERM

dlc_worker_bootstrap() {
  echo "[DLC worker rank=${DLC_NODE_RANK}] waiting for master ray at ${DLC_MASTER_ADDR}:${RAY_PORT}"
  local waited=0
  local wait_seconds="${DLC_WORKER_RAY_HEAD_WAIT_SECONDS:-480}"
  until ray status --address "${DLC_MASTER_ADDR}:${RAY_PORT}" >/dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    if (( waited >= wait_seconds )); then
      echo "[DLC worker] ERROR: master ray head did not appear in ${wait_seconds}s"
      exit 1
    fi
  done

  echo "[DLC worker] joining ray cluster with GPUs ${WORKER_STUDENT_CUDA_VISIBLE_DEVICES}"
  CUDA_VISIBLE_DEVICES="${WORKER_STUDENT_CUDA_VISIBLE_DEVICES}" \
    ray start --address "${DLC_MASTER_ADDR}:${RAY_PORT}" \
      --num-gpus "${worker_student_gpu_count}" \
      --block &
  local ray_pid=$!
  wait "${ray_pid}" || true

  if [[ "${RUN_TWO_ROUND_EVAL}" == "true" ]]; then
    local rdv_helper="${REPO_ROOT}/scripts/supplement_2rounds/_rendezvous_dlc.sh"
    local vllm_runtime="${REPO_ROOT}/scripts/supplement_2rounds/_vllm_runtime.sh"
    if [[ -f "${rdv_helper}" && -f "${vllm_runtime}" ]]; then
      export CUDA_VISIBLE_DEVICES="${POSTEVAL_WORKER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
      export MODEL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
      export VLLM_TP_SIZE="${POSTEVAL_WORKER_VLLM_TP_SIZE:-$(count_csv_items "${CUDA_VISIBLE_DEVICES}")}"
      export REPO_ROOT TEACHER_VENV TEACHER_PYTHON_BIN
      export PROGRESS_HELPER="${PROGRESS_HELPER:-${REPO_ROOT}/scripts/supplement/vllm_generate_progress.py}"
      # shellcheck disable=SC1090
      source "${vllm_runtime}"
      # shellcheck disable=SC1090
      source "${rdv_helper}"
      rdv_init_root "${RUN_DIR}"
      echo "[DLC worker] entering post-eval rendezvous watch loop"
      posteval_worker_watch
    fi
  fi
}

if [[ "${DLC_MODE}" == "true" && "${DLC_NODE_RANK}" -gt 0 ]]; then
  dlc_worker_bootstrap
  exit 0
fi

if [[ "${SKIP_SSH_BOOTSTRAP}" == "true" ]]; then
  echo "[1/3] no ssh worker bootstrap needed"
else
  echo "[1/3] connectivity check to worker..."
  ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -lc 'hostname'" >/dev/null
fi

echo "[2/3] starting ray cluster..."
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
if [[ "${SKIP_SSH_BOOTSTRAP}" != "true" ]]; then
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

# ---------------------------------------------------------------------------
# 7) Training job
# ---------------------------------------------------------------------------
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

EXTRA_TRAIN_FLAGS=()
if [[ '${ENABLE_EMA}' == 'true' ]]; then
  EXTRA_TRAIN_FLAGS+=(--enable_ema --ema_beta '${EMA_BETA}')
fi
if [[ '${FEATURE_ADAPTER_ENABLE}' == 'true' ]]; then
  EXTRA_TRAIN_FLAGS+=(
    --feature_adapter_enable
    --feature_adapter_type residual_bottleneck
    --feature_adapter_rank '${FEATURE_ADAPTER_RANK}'
    --feature_adapter_dropout '${FEATURE_ADAPTER_DROPOUT}'
    --feature_adapter_unfreeze_layers '${UNFREEZE_LAYERS}'
    --critic_classifier_loss_coef '${CRITIC_CLASSIFIER_LOSS_COEF}'
    --critic_direct_discrepancy_coef '${CRITIC_DIRECT_DISCREPANCY_COEF}'
    --critic_direct_discrepancy_target '${CRITIC_DIRECT_DISCREPANCY_TARGET}'
  )
fi

'${STUDENT_PYTHON_BIN}' -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --pretrain_mode --no_chat_template \
  --disable_ds_ckpt --colocate_actor_ref --colocate_critic_reward \
  --gradient_checkpointing --gradient_checkpointing_use_reentrant --use_kl_loss --use_whitening \
  "\${EXTRA_TRAIN_FLAGS[@]}" \
  --distribution_reward_type cf_l1oo \
  --feature_map_type '${FEATURE_MAP_TYPE}' \
  --rff_num_features '${RFF_NUM_FEATURES}' --rff_sigma '${RFF_SIGMA}' --rff_seed '${RFF_SEED}' \
  --cf_num_freqs '${CF_NUM_FREQS}' --cf_sigma '${CF_SIGMA}' --cf_seed '${CF_SEED}' \
  --cf_alpha '${CF_ALPHA}' --cf_beta '${CF_BETA}' --cf_reward_scale '${CF_REWARD_SCALE}' \
  --cf_target_mode '${CF_TARGET_MODE}' \
  --cf_target_num_refs '${CF_TARGET_NUM_REFS}' \
  --cf_target_std '${CF_TARGET_STD}' \
  --cf_target_seed '${CF_TARGET_SEED}' \
  --cf_teacher_lambda '${CF_TEACHER_LAMBDA}' \
  --cf_teacher_n_samples '${CF_TEACHER_N_SAMPLES}' \
  --embed_method last_token --critic_sequence_level last_token \
  --critic_learning_rate '${CRITIC_LR}' \
  --critic_lr_head '${CRITIC_LR_HEAD}' \
  --ce_loss_coef '${CE_LOSS_COEF}' \
  --alignment_rew_coef '${ALIGNMENT_REW_COEF}' \
  --diversity_rew_coef '${DIVERSITY_REW_COEF}' \
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
  --advantage_estimator '${ADVANTAGE_ESTIMATOR}' --init_kl_coef '${INIT_KL_COEF}' --kl_estimator '${KL_ESTIMATOR}' \
  --temperature '${TEMPERATURE}' --top_p '${TOP_P}' \
  --zero_stage '${ZERO_STAGE}' --lr_warmup_ratio '${LR_WARMUP_RATIO}' --critic_lr_warmup_ratio '${CRITIC_LR_WARMUP_RATIO}' \
  --seed '${GLOBAL_SEED}' \
  --eval_steps '${EVAL_STEPS}' \
  --eval_max_samples '${EVAL_MAX_SAMPLES}' \
  --eval_generate_max_len '${EVAL_GENERATE_MAX_LEN}' \
  --logging_steps '${LOGGING_STEPS}' \
  --save_steps '${SAVE_STEPS}' --save_even_count '${SAVE_EVEN_COUNT}' --save_hf_ckpt \
  --use_tensorboard '${TB_DIR}' \
  --save_path '${SAVE_PATH}' --ckpt_path '${SAVE_PATH}/ckpt' \
  --wandb_run_name '${RUN_NAME}' \
  2>&1 | tee '${RUN_DIR}/train.log'
EOF
chmod +x "${JOB_SCRIPT}"

echo "[3/3] submitting Ray job..."
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

echo "[post-run] stopping ray before eval/archive ..."
stop_runtime_processes

# ---------------------------------------------------------------------------
# 8) Code post-eval
# ---------------------------------------------------------------------------
if [[ "${RUN_TWO_ROUND_EVAL}" == "true" ]]; then
  if (( TRAIN_RC == 0 )); then
    echo "===== Running MBPP + HumanEval code post-eval ====="
    set +e
    export RUN_DIR MODEL_PATH="${SAVE_PATH}"
    export REPO_ROOT TEACHER_VENV ANALYSIS_VENV TEACHER_PYTHON_BIN ANALYSIS_PYTHON_BIN
    export MODEL_CUDA_VISIBLE_DEVICES VLLM_TP_SIZE VLLM_MAX_NUM_SEQS VLLM_PROGRESS_BATCH_SIZE VLLM_ENABLE_PREFIX_CACHING VLLM_SEED
    export POST_EVAL_SCRIPT CODE_POST_EVAL_WORKER POST_EVAL_DATASETS POST_EVAL_MAX_SAMPLES POST_EVAL_PROMPT_MAX_LEN
    export CODE_EVAL_MAX_NEW_TOKENS CODE_EVAL_TEMPERATURE CODE_EVAL_TOP_P CODE_EVAL_REPETITION_PENALTY CODE_EVAL_TIMEOUT_SECONDS
    POSTEVAL_WORKER_DISPATCH="${POSTEVAL_WORKER_DISPATCH:-ssh}"
    if [[ "${DLC_MODE}" == "true" ]]; then
      POSTEVAL_WORKER_DISPATCH="rendezvous"
      WORKER_SSH_TARGET=""
    elif [[ "${SINGLE_NODE_MODE}" == "true" ]]; then
      WORKER_SSH_TARGET=""
    fi
    export POSTEVAL_WORKER_DISPATCH WORKER_SSH_TARGET SSH_OPTS
    export NCCL_P2P_LEVEL NCCL_NET_GDR_DISABLE
    bash "${POST_EVAL_SCRIPT}" "${RUN_DIR}"
    EVAL_RC=$?
    set -e
    if (( EVAL_RC != 0 )); then
      echo "[ERROR] post-eval failed with exit code ${EVAL_RC}; run outputs will still be archived."
    fi
  else
    echo "[post-eval] skipped because training failed."
  fi
fi

# ---------------------------------------------------------------------------
# 9) Archive / final status
# ---------------------------------------------------------------------------
if [[ "${ARCHIVE_OUTPUTS_AFTER_RUN}" == "true" ]]; then
  set +e
  archive_run_outputs "${ARCHIVE_OUTPUT_ROOT}"
  ARCHIVE_RC=$?
  set -e
  if (( ARCHIVE_RC != 0 )); then
    echo "[ERROR] archiving run outputs failed with exit code ${ARCHIVE_RC}"
  fi
fi

FINAL_RC=0
if (( TRAIN_RC != 0 )); then
  FINAL_RC=${TRAIN_RC}
elif (( EVAL_RC != 0 )); then
  FINAL_RC=${EVAL_RC}
elif (( ARCHIVE_RC != 0 )); then
  FINAL_RC=${ARCHIVE_RC}
fi

write_final_status
echo "[done] RUN_DIR=${RUN_DIR} FINAL_RC=${FINAL_RC}"
exit "${FINAL_RC}"
