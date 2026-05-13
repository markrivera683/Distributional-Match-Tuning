#!/usr/bin/env bash
# Standalone SFT baseline for diff-dataset code generation comparison.
# All important train/eval parameters are declared in this file.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "${csv}" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"${csv}"
}

bool_flag() {
  local enabled="$1"
  local flag="$2"
  if [[ "${enabled}" == "true" ]]; then
    printf '%s\n' "${flag}"
  fi
}

prepare_diff_datasets() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_code_datasets.py" \
    --output-dir "${PREPARED_DATA_DIR}" \
    --train-samples "${TRAIN_SAMPLE_POOL}" \
    ${PREPARE_DIFF_DATASETS_FORCE:+--force}
}

ensure_envs() {
  if [[ "${RUN_SETUP_ENV:-true}" != "true" ]]; then
    return 0
  fi

  if [[ ! -x "${STUDENT_PYTHON_BIN}" || ! -x "${TEACHER_PYTHON_BIN}" ]]; then
    echo "[env] missing venv under ${LOCAL_ROOT}; running setup_env.sh"
    LOCAL_ROOT="${LOCAL_ROOT}" bash "${REPO_ROOT}/scripts/setup_env.sh"
  else
    echo "[env] using existing venvs under ${LOCAL_ROOT}"
  fi

  if ! "${STUDENT_PYTHON_BIN}" -c "import deepspeed, ray, torch, transformers" >/dev/null 2>&1; then
    echo "[env] student env is incomplete; rebuilding ${STUDENT_VENV}"
    LOCAL_ROOT="${LOCAL_ROOT}" EBFT_REBUILD_VENV=1 SKIP_TEACHER=1 bash "${REPO_ROOT}/scripts/setup_env.sh"
  fi

  if ! "${CODE_BENCHMARK_PYTHON_BIN}" -c "import datasets, torch, vllm" >/dev/null 2>&1; then
    echo "[env] installing benchmark deps into ${CODE_BENCHMARK_PYTHON_BIN}"
    "${CODE_BENCHMARK_PYTHON_BIN}" -m pip install "datasets==4.8.4"
  fi

  if ! "${CODE_BENCHMARK_PYTHON_BIN}" -c "import datasets, torch, vllm" >/dev/null 2>&1; then
    echo "[env] teacher benchmark env is incomplete; rebuilding ${TEACHER_VENV}"
    LOCAL_ROOT="${LOCAL_ROOT}" EBFT_REBUILD_VENV=1 SKIP_STUDENT=1 bash "${REPO_ROOT}/scripts/setup_env.sh"
  fi
}

run_code_benchmark() {
  local eval_name="$1"
  local output_dir="$2"
  local log_path="$3"
  local greedy_temperature="$4"
  local sample_temperature="$5"
  local n_samples="$6"
  local passk_list="$7"
  local prefix_cache_args=()
  local benchmark_extra_args=()

  mkdir -p "${output_dir}" "$(dirname "${log_path}")"
  [[ "${VLLM_ENABLE_PREFIX_CACHING}" == "true" ]] && prefix_cache_args+=(--enable_prefix_caching)
  [[ "${CODE_EVAL_SKIP_MISSING_TOOLCHAINS}" == "true" ]] && benchmark_extra_args+=(--skip_missing_toolchains)

  echo ""
  echo "===== code post-eval: ${eval_name} ====="
  echo "output_dir:          ${output_dir}"
  echo "greedy_temperature:  ${greedy_temperature}"
  echo "sample_temperature:  ${sample_temperature}"
  echo "n_samples:           ${n_samples}"
  echo "passk_list:          ${passk_list}"

  CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES}" \
  "${CODE_BENCHMARK_PYTHON_BIN}" "${CODE_BENCHMARK_SCRIPT}" \
    --model_path "${SAVE_PATH}" \
    --output_dir "${output_dir}" \
    --benchmarks "${CODE_BENCHMARKS}" \
    --backend "${CODE_BENCHMARK_BACKEND}" \
    --humaneval_dataset "${HUMANEVAL_EVAL_DATA}" \
    --humaneval_split "${HUMANEVAL_EVAL_SPLIT}" \
    --mbpp_dataset "${MBPP_EVAL_DATA}" \
    --mbpp_config "${MBPP_EVAL_CONFIG}" \
    --mbpp_split "${MBPP_EVAL_SPLIT}" \
    --prompt_max_len "${POST_EVAL_PROMPT_MAX_LEN}" \
    --max_new_tokens "${CODE_EVAL_MAX_NEW_TOKENS}" \
    --greedy_temperature "${greedy_temperature}" \
    --sample_temperature "${sample_temperature}" \
    --top_p "${CODE_EVAL_TOP_P}" \
    --repetition_penalty "${CODE_EVAL_REPETITION_PENALTY}" \
    --n_samples "${n_samples}" \
    --passk_list "${passk_list}" \
    --max_samples_per_benchmark "${POST_EVAL_MAX_SAMPLES}" \
    --timeout_seconds "${CODE_EVAL_TIMEOUT_SECONDS}" \
    --tp_size "${VLLM_TP_SIZE}" \
    --max_num_seqs "${VLLM_MAX_NUM_SEQS}" \
    --seed "${VLLM_SEED}" \
    --greedy_batch_size "${CODE_EVAL_GREEDY_BATCH_SIZE}" \
    --sample_batch_size "${CODE_EVAL_SAMPLE_BATCH_SIZE}" \
    --detail_preview_chars "${CODE_EVAL_DETAIL_PREVIEW_CHARS}" \
    "${prefix_cache_args[@]}" \
    "${benchmark_extra_args[@]}" \
    2>&1 | tee "${log_path}"
}

# ---------------------------------------------------------------------------
# 1) Explicit paths / envs
# ---------------------------------------------------------------------------
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/workspace}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-4B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs/diff_dataset}"
PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-/mnt/data/ebft-distribution-new/outputs/diff_dataset_prepared}"
TRAIN_SAMPLE_POOL="${TRAIN_SAMPLE_POOL:-100000}"
TRAIN_DATA="${TRAIN_DATA:-${PREPARED_DATA_DIR}/opencodeinstruct_qa_100k.jsonl}"
EVAL_DATA="${EVAL_DATA:-${PREPARED_DATA_DIR}/mbpp_eval_qa.jsonl}"
MBPP_EVAL_DATA="${MBPP_EVAL_DATA:-${PREPARED_DATA_DIR}/mbpp_eval_qa.jsonl}"
HUMANEVAL_EVAL_DATA="${HUMANEVAL_EVAL_DATA:-${PREPARED_DATA_DIR}/humaneval_eval_qa.jsonl}"
HUMANEVAL_EVAL_SPLIT="${HUMANEVAL_EVAL_SPLIT:-test}"
MBPP_EVAL_CONFIG="${MBPP_EVAL_CONFIG:-}"
MBPP_EVAL_SPLIT="${MBPP_EVAL_SPLIT:-test}"
PREPARE_DIFF_DATASETS_FORCE="${PREPARE_DIFF_DATASETS_FORCE:-}"

STUDENT_VENV="${STUDENT_VENV:-${LOCAL_ROOT}/venvs/.venv}"
TEACHER_VENV="${TEACHER_VENV:-${LOCAL_ROOT}/venvs/.teacherVenv}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
CODE_BENCHMARK_PYTHON_BIN="${CODE_BENCHMARK_PYTHON_BIN:-${TEACHER_PYTHON_BIN}}"
DEEPSPEED_LAUNCHER_MODULE="${DEEPSPEED_LAUNCHER_MODULE:-deepspeed.launcher.runner}"

# ---------------------------------------------------------------------------
# 2) Process environment
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/mnt/data/ebft-distribution-new/caches/hf}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${LOCAL_ROOT}/.torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_ROOT}/.triton_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAY_DISABLE_DOCKER_CPU_WARNING="${RAY_DISABLE_DOCKER_CPU_WARNING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
[[ "${NCCL_P2P_DISABLE:-}" == "1" ]] && unset NCCL_P2P_DISABLE
export NCCL_NET_GDR_DISABLE="${NCCL_NET_GDR_DISABLE:-1}"

# ---------------------------------------------------------------------------
# 3) SFT training parameters
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GPU_COUNT="$(count_csv_items "${CUDA_VISIBLE_DEVICES}")"

SFT_BF16="${SFT_BF16:-true}"
SFT_FLASH_ATTN="${SFT_FLASH_ATTN:-true}"
SFT_GRADIENT_CHECKPOINTING="${SFT_GRADIENT_CHECKPOINTING:-true}"
SFT_GRADIENT_CHECKPOINTING_USE_REENTRANT="${SFT_GRADIENT_CHECKPOINTING_USE_REENTRANT:-false}"
SFT_FULL_DETERMINISM="${SFT_FULL_DETERMINISM:-false}"
SFT_DISABLE_DS_CKPT="${SFT_DISABLE_DS_CKPT:-true}"
SFT_SAVE_HF_CKPT="${SFT_SAVE_HF_CKPT:-true}"
SFT_LOAD_CHECKPOINT="${SFT_LOAD_CHECKPOINT:-false}"
SFT_USE_DS_UNIVERSAL_CKPT="${SFT_USE_DS_UNIVERSAL_CKPT:-false}"
SFT_ADAM_OFFLOAD="${SFT_ADAM_OFFLOAD:-false}"
SFT_OVERLAP_COMM="${SFT_OVERLAP_COMM:-false}"
SFT_DEEPCOMPILE="${SFT_DEEPCOMPILE:-false}"
SFT_DISABLE_FAST_TOKENIZER="${SFT_DISABLE_FAST_TOKENIZER:-false}"
SFT_USE_LIGER_KERNEL="${SFT_USE_LIGER_KERNEL:-false}"
SFT_LOAD_IN_4BIT="${SFT_LOAD_IN_4BIT:-false}"
SFT_PACKING_SAMPLES="${SFT_PACKING_SAMPLES:-false}"
SFT_PRETRAIN_MODE="${SFT_PRETRAIN_MODE:-false}"
SFT_MULTITURN="${SFT_MULTITURN:-false}"
SFT_APPLY_CHAT_TEMPLATE="${SFT_APPLY_CHAT_TEMPLATE:-false}"
SFT_USE_MS="${SFT_USE_MS:-false}"

SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-128}"
SFT_MICRO_TRAIN_BATCH_SIZE="${SFT_MICRO_TRAIN_BATCH_SIZE:-4}"
SFT_EVAL_BATCH_SIZE="${SFT_EVAL_BATCH_SIZE:-64}"
SFT_EVAL_DOWN_BATCH_SIZE="${SFT_EVAL_DOWN_BATCH_SIZE:-128}"
SFT_MAX_LEN="${SFT_MAX_LEN:-2048}"
SFT_PROMPT_MAX_LEN="${SFT_PROMPT_MAX_LEN:-1024}"
SFT_GENERATE_MAX_LEN="${SFT_GENERATE_MAX_LEN:-1024}"
SFT_MAX_EPOCHS="${SFT_MAX_EPOCHS:-1}"
SFT_MAX_SAMPLES="${SFT_MAX_SAMPLES:-16000}"
SFT_EVAL_MAX_SAMPLES="${SFT_EVAL_MAX_SAMPLES:-1000}"
SFT_EVAL_DOWN_MAX_SAMPLES="${SFT_EVAL_DOWN_MAX_SAMPLES:-0}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1e-5}"
SFT_LR_WARMUP_RATIO="${SFT_LR_WARMUP_RATIO:-0.03}"
SFT_LR_SCHEDULER="${SFT_LR_SCHEDULER:-cosine_with_min_lr}"
SFT_L2="${SFT_L2:-0}"
SFT_ADAM_BETA1="${SFT_ADAM_BETA1:-0.9}"
SFT_ADAM_BETA2="${SFT_ADAM_BETA2:-0.95}"
SFT_MAX_NORM="${SFT_MAX_NORM:-1.0}"
SFT_ZERO_STAGE="${SFT_ZERO_STAGE:-2}"
SFT_ZPG="${SFT_ZPG:-1}"
SFT_GRAD_ACCUM_DTYPE="${SFT_GRAD_ACCUM_DTYPE:-}"
SFT_DS_TENSOR_PARALLEL_SIZE="${SFT_DS_TENSOR_PARALLEL_SIZE:-1}"
SFT_AUX_LOSS_COEF="${SFT_AUX_LOSS_COEF:-0}"
SFT_RING_ATTN_SIZE="${SFT_RING_ATTN_SIZE:-1}"
SFT_RING_HEAD_STRIDE="${SFT_RING_HEAD_STRIDE:-1}"
SFT_SEED="${SFT_SEED:-43}"

SFT_VLLM_TENSOR_PARALLEL_SIZE="${SFT_VLLM_TENSOR_PARALLEL_SIZE:-1}"
SFT_VLLM_GPU_MEMORY_UTILIZATION="${SFT_VLLM_GPU_MEMORY_UTILIZATION:-0.7}"
SFT_MAX_NEW_TOKENS="${SFT_MAX_NEW_TOKENS:-512}"
SFT_TEMPERATURE="${SFT_TEMPERATURE:-0.6}"
SFT_EVAL_N_SAMPLES_PER_PROMPT="${SFT_EVAL_N_SAMPLES_PER_PROMPT:-1}"
SFT_TOP_P="${SFT_TOP_P:-0.95}"
SFT_MAX_TOKENS="${SFT_MAX_TOKENS:-2048}"

SFT_LORA_RANK="${SFT_LORA_RANK:-0}"
SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-16}"
SFT_TARGET_MODULES="${SFT_TARGET_MODULES:-all-linear}"
SFT_LORA_DROPOUT="${SFT_LORA_DROPOUT:-0}"

SFT_DATASET_PROBS="${SFT_DATASET_PROBS:-}"
SFT_DATASET_SPLIT="${SFT_DATASET_SPLIT:-train}"
SFT_TRAIN_SPLIT="${SFT_TRAIN_SPLIT:-train}"
SFT_EVAL_SPLIT="${SFT_EVAL_SPLIT:-train}"
SFT_INPUT_KEY="${SFT_INPUT_KEY:-question}"
SFT_OUTPUT_KEY="${SFT_OUTPUT_KEY:-answer}"
SFT_LABEL_KEY="${SFT_LABEL_KEY:-answer}"
SFT_INPUT_TEMPLATE="${SFT_INPUT_TEMPLATE:-}"
SFT_TOKENIZER_CHAT_TEMPLATE="${SFT_TOKENIZER_CHAT_TEMPLATE:-}"

SFT_HUMANEVAL_CONFIG="${SFT_HUMANEVAL_CONFIG:-}"
SFT_HUMANEVAL_SPLIT="${SFT_HUMANEVAL_SPLIT:-test}"
SFT_MBPP_CONFIG="${SFT_MBPP_CONFIG:-}"
SFT_MBPP_SPLIT="${SFT_MBPP_SPLIT:-test}"
SFT_MULTIPL_DATASET="${SFT_MULTIPL_DATASET:-}"
SFT_MULTIPL_CONFIG="${SFT_MULTIPL_CONFIG:-humaneval-py}"
SFT_MULTIPL_SPLIT="${SFT_MULTIPL_SPLIT:-test}"

SFT_USE_WANDB="${SFT_USE_WANDB:-}"
SFT_WANDB_ORG="${SFT_WANDB_ORG:-}"
SFT_WANDB_GROUP="${SFT_WANDB_GROUP:-}"
SFT_WANDB_PROJECT="${SFT_WANDB_PROJECT:-openrlhf_train_debug}"

SFT_LOGGING_STEPS="${SFT_LOGGING_STEPS:-10}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:--1}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:--1}"
SFT_MAX_CKPT_NUM="${SFT_MAX_CKPT_NUM:-3}"
SFT_MAX_CKPT_MEM="${SFT_MAX_CKPT_MEM:-100000000}"

# ---------------------------------------------------------------------------
# 4) Code post-eval parameters
# ---------------------------------------------------------------------------
RUN_CODE_POST_EVAL="${RUN_CODE_POST_EVAL:-true}"
CODE_BENCHMARK_SCRIPT="${CODE_BENCHMARK_SCRIPT:-${REPO_ROOT}/scripts/benchmarks/run_code_generation_benchmarks.py}"
CODE_BENCHMARKS="${CODE_BENCHMARKS:-humaneval,mbpp}"
CODE_BENCHMARK_BACKEND="${CODE_BENCHMARK_BACKEND:-vllm}"
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-$(count_csv_items "${MODEL_CUDA_VISIBLE_DEVICES}")}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_SEED="${VLLM_SEED:-1234}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
CODE_EVAL_MAX_NEW_TOKENS="${CODE_EVAL_MAX_NEW_TOKENS:-1024}"
CODE_EVAL_TOP_P="${CODE_EVAL_TOP_P:-1.0}"
CODE_EVAL_REPETITION_PENALTY="${CODE_EVAL_REPETITION_PENALTY:-1.0}"
CODE_EVAL_TIMEOUT_SECONDS="${CODE_EVAL_TIMEOUT_SECONDS:-10}"
CODE_EVAL_GREEDY_BATCH_SIZE="${CODE_EVAL_GREEDY_BATCH_SIZE:-16}"
CODE_EVAL_SAMPLE_BATCH_SIZE="${CODE_EVAL_SAMPLE_BATCH_SIZE:-4}"
CODE_EVAL_DETAIL_PREVIEW_CHARS="${CODE_EVAL_DETAIL_PREVIEW_CHARS:-4096}"
CODE_EVAL_SKIP_MISSING_TOOLCHAINS="${CODE_EVAL_SKIP_MISSING_TOOLCHAINS:-false}"
SAMPLE_PASS1_TEMPERATURE="${SAMPLE_PASS1_TEMPERATURE:-0.6}"
SAMPLE_PASS1_N_SAMPLES="${SAMPLE_PASS1_N_SAMPLES:-1}"
SAMPLE_PASS1_PASSK_LIST="${SAMPLE_PASS1_PASSK_LIST:-1}"
SAMPLE_PASS16_TEMPERATURE="${SAMPLE_PASS16_TEMPERATURE:-0.6}"
SAMPLE_PASS16_N_SAMPLES="${SAMPLE_PASS16_N_SAMPLES:-16}"
SAMPLE_PASS16_PASSK_LIST="${SAMPLE_PASS16_PASSK_LIST:-16}"

# ---------------------------------------------------------------------------
# 5) Output
# ---------------------------------------------------------------------------
RUN_NAME="${RUN_NAME:-diff_sft_qwen35_4b_1node_$(date +%m%d_%H%M)}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
SAVE_PATH="${SAVE_PATH:-${RUN_DIR}/model}"
CKPT_PATH="${CKPT_PATH:-${RUN_DIR}/ckpt}"
TB_DIR="${TB_DIR:-${RUN_DIR}/tensorboard}"
SCRIPT_LOG_PATH="${SCRIPT_LOG_PATH:-${RUN_DIR}/$(basename "$0" .sh).log}"
RUN_CONTEXT_PATH="${RUN_CONTEXT_PATH:-${RUN_DIR}/run_context.env}"
RUN_SUMMARY_PATH="${RUN_SUMMARY_PATH:-${RUN_DIR}/run_summary.txt}"

mkdir -p "${RUN_DIR}" "${SAVE_PATH}" "${CKPT_PATH}" "${TB_DIR}"
exec > >(tee -a "${SCRIPT_LOG_PATH}") 2>&1

ensure_envs
prepare_diff_datasets

[[ -x "${STUDENT_PYTHON_BIN}" ]] || { echo "[ERROR] STUDENT_PYTHON_BIN not executable: ${STUDENT_PYTHON_BIN}"; exit 1; }
[[ -x "${CODE_BENCHMARK_PYTHON_BIN}" ]] || { echo "[ERROR] CODE_BENCHMARK_PYTHON_BIN not executable: ${CODE_BENCHMARK_PYTHON_BIN}"; exit 1; }
[[ -e "${MODEL_PATH}" ]] || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 1; }
[[ -e "${TRAIN_DATA}" ]] || { echo "[ERROR] TRAIN_DATA not found: ${TRAIN_DATA}"; exit 1; }
[[ -e "${EVAL_DATA}" ]] || { echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"; exit 1; }
[[ -e "${MBPP_EVAL_DATA}" ]] || { echo "[ERROR] MBPP_EVAL_DATA not found: ${MBPP_EVAL_DATA}"; exit 1; }
[[ -e "${HUMANEVAL_EVAL_DATA}" ]] || { echo "[ERROR] HUMANEVAL_EVAL_DATA not found: ${HUMANEVAL_EVAL_DATA}"; exit 1; }
[[ -f "${CODE_BENCHMARK_SCRIPT}" ]] || { echo "[ERROR] CODE_BENCHMARK_SCRIPT not found: ${CODE_BENCHMARK_SCRIPT}"; exit 1; }

if (( SFT_TRAIN_BATCH_SIZE % (SFT_MICRO_TRAIN_BATCH_SIZE * GPU_COUNT) != 0 )); then
  echo "[ERROR] SFT_TRAIN_BATCH_SIZE must be divisible by SFT_MICRO_TRAIN_BATCH_SIZE * GPU_COUNT"
  echo "        ${SFT_TRAIN_BATCH_SIZE} % (${SFT_MICRO_TRAIN_BATCH_SIZE} * ${GPU_COUNT}) != 0"
  exit 1
fi

{
  echo "# Auto-generated SFT run context"
  echo "# UTC: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  for name in \
    REPO_ROOT LOCAL_ROOT STUDENT_VENV TEACHER_VENV STUDENT_PYTHON_BIN CODE_BENCHMARK_PYTHON_BIN \
    MODEL_PATH OUTPUT_ROOT RUN_NAME RUN_DIR SAVE_PATH CKPT_PATH TB_DIR \
    PREPARED_DATA_DIR TRAIN_SAMPLE_POOL TRAIN_DATA EVAL_DATA MBPP_EVAL_DATA HUMANEVAL_EVAL_DATA \
    HUMANEVAL_EVAL_SPLIT MBPP_EVAL_CONFIG MBPP_EVAL_SPLIT \
    CUDA_VISIBLE_DEVICES GPU_COUNT \
    SFT_BF16 SFT_FLASH_ATTN SFT_GRADIENT_CHECKPOINTING SFT_GRADIENT_CHECKPOINTING_USE_REENTRANT \
    SFT_FULL_DETERMINISM \
    SFT_DISABLE_DS_CKPT SFT_SAVE_HF_CKPT SFT_LOAD_CHECKPOINT SFT_USE_DS_UNIVERSAL_CKPT \
    SFT_ADAM_OFFLOAD SFT_OVERLAP_COMM SFT_DEEPCOMPILE SFT_DISABLE_FAST_TOKENIZER SFT_USE_LIGER_KERNEL \
    SFT_LOAD_IN_4BIT SFT_PACKING_SAMPLES SFT_PRETRAIN_MODE SFT_MULTITURN SFT_APPLY_CHAT_TEMPLATE SFT_USE_MS \
    SFT_TRAIN_BATCH_SIZE SFT_MICRO_TRAIN_BATCH_SIZE SFT_EVAL_BATCH_SIZE SFT_EVAL_DOWN_BATCH_SIZE \
    SFT_MAX_LEN SFT_PROMPT_MAX_LEN SFT_GENERATE_MAX_LEN SFT_MAX_EPOCHS SFT_MAX_SAMPLES \
    SFT_EVAL_MAX_SAMPLES SFT_EVAL_DOWN_MAX_SAMPLES SFT_LEARNING_RATE SFT_LR_WARMUP_RATIO SFT_LR_SCHEDULER \
    SFT_L2 SFT_ADAM_BETA1 SFT_ADAM_BETA2 SFT_MAX_NORM SFT_ZERO_STAGE SFT_ZPG SFT_GRAD_ACCUM_DTYPE \
    SFT_DS_TENSOR_PARALLEL_SIZE SFT_AUX_LOSS_COEF SFT_RING_ATTN_SIZE SFT_RING_HEAD_STRIDE SFT_SEED \
    SFT_VLLM_TENSOR_PARALLEL_SIZE SFT_VLLM_GPU_MEMORY_UTILIZATION SFT_MAX_NEW_TOKENS SFT_TEMPERATURE \
    SFT_EVAL_N_SAMPLES_PER_PROMPT SFT_TOP_P SFT_MAX_TOKENS \
    SFT_LORA_RANK SFT_LORA_ALPHA SFT_TARGET_MODULES SFT_LORA_DROPOUT \
    SFT_DATASET_PROBS SFT_DATASET_SPLIT SFT_TRAIN_SPLIT SFT_EVAL_SPLIT SFT_INPUT_KEY SFT_OUTPUT_KEY SFT_LABEL_KEY \
    SFT_INPUT_TEMPLATE SFT_TOKENIZER_CHAT_TEMPLATE SFT_HUMANEVAL_CONFIG SFT_HUMANEVAL_SPLIT SFT_MBPP_CONFIG SFT_MBPP_SPLIT \
    SFT_MULTIPL_DATASET SFT_MULTIPL_CONFIG SFT_MULTIPL_SPLIT SFT_USE_WANDB SFT_WANDB_ORG SFT_WANDB_GROUP SFT_WANDB_PROJECT \
    SFT_LOGGING_STEPS SFT_EVAL_STEPS SFT_SAVE_STEPS SFT_MAX_CKPT_NUM SFT_MAX_CKPT_MEM \
    RUN_CODE_POST_EVAL CODE_BENCHMARK_SCRIPT CODE_BENCHMARKS CODE_BENCHMARK_BACKEND MODEL_CUDA_VISIBLE_DEVICES \
    VLLM_TP_SIZE VLLM_MAX_NUM_SEQS VLLM_ENABLE_PREFIX_CACHING VLLM_SEED POST_EVAL_MAX_SAMPLES POST_EVAL_PROMPT_MAX_LEN \
    CODE_EVAL_MAX_NEW_TOKENS CODE_EVAL_TOP_P CODE_EVAL_REPETITION_PENALTY CODE_EVAL_TIMEOUT_SECONDS \
    CODE_EVAL_GREEDY_BATCH_SIZE CODE_EVAL_SAMPLE_BATCH_SIZE CODE_EVAL_DETAIL_PREVIEW_CHARS CODE_EVAL_SKIP_MISSING_TOOLCHAINS \
    SAMPLE_PASS1_TEMPERATURE SAMPLE_PASS1_N_SAMPLES \
    SAMPLE_PASS1_PASSK_LIST SAMPLE_PASS16_TEMPERATURE SAMPLE_PASS16_N_SAMPLES SAMPLE_PASS16_PASSK_LIST; do
    printf "%s=%q\n" "${name}" "${!name-}"
  done
} > "${RUN_CONTEXT_PATH}"

{
  echo "run_name: ${RUN_NAME}"
  echo "run_dir: ${RUN_DIR}"
  echo "model_path: ${MODEL_PATH}"
  echo "train_data: ${TRAIN_DATA}"
  echo "sft_lr: ${SFT_LEARNING_RATE}"
  echo "sft_train_batch_size: ${SFT_TRAIN_BATCH_SIZE}"
  echo "sft_micro_train_batch_size: ${SFT_MICRO_TRAIN_BATCH_SIZE}"
  echo "sft_max_len: ${SFT_MAX_LEN}"
  echo "sft_max_epochs: ${SFT_MAX_EPOCHS}"
  echo "sft_max_samples: ${SFT_MAX_SAMPLES}"
  echo "post_eval_sample_pass1: temp=${SAMPLE_PASS1_TEMPERATURE}, n=${SAMPLE_PASS1_N_SAMPLES}, passk=${SAMPLE_PASS1_PASSK_LIST}"
  echo "post_eval_sample_pass16: temp=${SAMPLE_PASS16_TEMPERATURE}, n=${SAMPLE_PASS16_N_SAMPLES}, passk=${SAMPLE_PASS16_PASSK_LIST}"
} > "${RUN_SUMMARY_PATH}"

echo "========== Diff-Dataset SFT baseline 1-node =========="
echo "RUN_DIR:                    ${RUN_DIR}"
echo "MODEL_PATH:                 ${MODEL_PATH}"
echo "TRAIN_DATA:                 ${TRAIN_DATA}"
echo "CUDA_VISIBLE_DEVICES:       ${CUDA_VISIBLE_DEVICES} (count=${GPU_COUNT})"
echo "SFT max_samples/epochs:     ${SFT_MAX_SAMPLES}/${SFT_MAX_EPOCHS}"
echo "SFT batch/micro:            ${SFT_TRAIN_BATCH_SIZE}/${SFT_MICRO_TRAIN_BATCH_SIZE}"
echo "SFT max_len:                ${SFT_MAX_LEN}"
echo "SFT lr/scheduler/warmup:    ${SFT_LEARNING_RATE}/${SFT_LR_SCHEDULER}/${SFT_LR_WARMUP_RATIO}"
echo "SFT zero_stage:             ${SFT_ZERO_STAGE}"
echo "SFT bf16/flash/checkpoint:  ${SFT_BF16}/${SFT_FLASH_ATTN}/${SFT_GRADIENT_CHECKPOINTING}"
echo "post-eval sample pass@1:    temp=${SAMPLE_PASS1_TEMPERATURE}, n=${SAMPLE_PASS1_N_SAMPLES}, passk=${SAMPLE_PASS1_PASSK_LIST}"
echo "post-eval sample pass@16:   temp=${SAMPLE_PASS16_TEMPERATURE}, n=${SAMPLE_PASS16_N_SAMPLES}, passk=${SAMPLE_PASS16_PASSK_LIST}"
echo "benchmark python:           ${CODE_BENCHMARK_PYTHON_BIN}"
echo "======================================================"

cd "${REPO_ROOT}"

SFT_FLAGS=()
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_BF16}" "--bf16")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_FLASH_ATTN}" "--flash_attn")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_GRADIENT_CHECKPOINTING}" "--gradient_checkpointing")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_GRADIENT_CHECKPOINTING_USE_REENTRANT}" "--gradient_checkpointing_use_reentrant")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_FULL_DETERMINISM}" "--full_determinism")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_DISABLE_DS_CKPT}" "--disable_ds_ckpt")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_SAVE_HF_CKPT}" "--save_hf_ckpt")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_LOAD_CHECKPOINT}" "--load_checkpoint")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_USE_DS_UNIVERSAL_CKPT}" "--use_ds_universal_ckpt")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_ADAM_OFFLOAD}" "--adam_offload")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_OVERLAP_COMM}" "--overlap_comm")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_DEEPCOMPILE}" "--deepcompile")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_DISABLE_FAST_TOKENIZER}" "--disable_fast_tokenizer")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_USE_LIGER_KERNEL}" "--use_liger_kernel")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_LOAD_IN_4BIT}" "--load_in_4bit")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_PACKING_SAMPLES}" "--packing_samples")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_PRETRAIN_MODE}" "--pretrain_mode")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_MULTITURN}" "--multiturn")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_APPLY_CHAT_TEMPLATE}" "--apply_chat_template")
while IFS= read -r flag; do [[ -n "${flag}" ]] && SFT_FLAGS+=("${flag}"); done < <(bool_flag "${SFT_USE_MS}" "--use_ms")

OPTIONAL_ARGS=()
[[ -n "${SFT_DATASET_PROBS}" ]] && OPTIONAL_ARGS+=(--dataset_probs "${SFT_DATASET_PROBS}")
[[ -n "${SFT_INPUT_TEMPLATE}" ]] && OPTIONAL_ARGS+=(--input_template "${SFT_INPUT_TEMPLATE}")
[[ -n "${SFT_TOKENIZER_CHAT_TEMPLATE}" ]] && OPTIONAL_ARGS+=(--tokenizer_chat_template "${SFT_TOKENIZER_CHAT_TEMPLATE}")
[[ -n "${SFT_GRAD_ACCUM_DTYPE}" ]] && OPTIONAL_ARGS+=(--grad_accum_dtype "${SFT_GRAD_ACCUM_DTYPE}")
[[ -n "${SFT_USE_WANDB}" ]] && OPTIONAL_ARGS+=(--use_wandb "${SFT_USE_WANDB}")
[[ -n "${SFT_WANDB_ORG}" ]] && OPTIONAL_ARGS+=(--wandb_org "${SFT_WANDB_ORG}")
[[ -n "${SFT_WANDB_GROUP}" ]] && OPTIONAL_ARGS+=(--wandb_group "${SFT_WANDB_GROUP}")
[[ -n "${SFT_MULTIPL_DATASET}" ]] && OPTIONAL_ARGS+=(--multipl_dataset "${SFT_MULTIPL_DATASET}")

TRAIN_RC=0
EVAL_RC=0

set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${STUDENT_PYTHON_BIN}" -m "${DEEPSPEED_LAUNCHER_MODULE}" --module openrlhf.cli.train_sft \
  "${SFT_FLAGS[@]}" \
  --save_path "${SAVE_PATH}" \
  --save_steps "${SFT_SAVE_STEPS}" \
  --logging_steps "${SFT_LOGGING_STEPS}" \
  --eval_steps "${SFT_EVAL_STEPS}" \
  --ckpt_path "${CKPT_PATH}" \
  --max_ckpt_num "${SFT_MAX_CKPT_NUM}" \
  --max_ckpt_mem "${SFT_MAX_CKPT_MEM}" \
  --micro_train_batch_size "${SFT_MICRO_TRAIN_BATCH_SIZE}" \
  --train_batch_size "${SFT_TRAIN_BATCH_SIZE}" \
  --eval_batch_size "${SFT_EVAL_BATCH_SIZE}" \
  --eval_down_batch_size "${SFT_EVAL_DOWN_BATCH_SIZE}" \
  --max_norm "${SFT_MAX_NORM}" \
  --seed "${SFT_SEED}" \
  --zero_stage "${SFT_ZERO_STAGE}" \
  --zpg "${SFT_ZPG}" \
  --ds_tensor_parallel_size "${SFT_DS_TENSOR_PARALLEL_SIZE}" \
  --max_epochs "${SFT_MAX_EPOCHS}" \
  --aux_loss_coef "${SFT_AUX_LOSS_COEF}" \
  --pretrain "${MODEL_PATH}" \
  --learning_rate "${SFT_LEARNING_RATE}" \
  --lr_warmup_ratio "${SFT_LR_WARMUP_RATIO}" \
  --lr_scheduler "${SFT_LR_SCHEDULER}" \
  --l2 "${SFT_L2}" \
  --adam_betas "${SFT_ADAM_BETA1}" "${SFT_ADAM_BETA2}" \
  --ring_attn_size "${SFT_RING_ATTN_SIZE}" \
  --ring_head_stride "${SFT_RING_HEAD_STRIDE}" \
  --vllm_tensor_parallel_size "${SFT_VLLM_TENSOR_PARALLEL_SIZE}" \
  --vllm_gpu_memory_utilization "${SFT_VLLM_GPU_MEMORY_UTILIZATION}" \
  --max_new_tokens "${SFT_MAX_NEW_TOKENS}" \
  --temperature "${SFT_TEMPERATURE}" \
  --eval_n_samples_per_prompt "${SFT_EVAL_N_SAMPLES_PER_PROMPT}" \
  --top_p "${SFT_TOP_P}" \
  --max_tokens "${SFT_MAX_TOKENS}" \
  --lora_rank "${SFT_LORA_RANK}" \
  --lora_alpha "${SFT_LORA_ALPHA}" \
  --target_modules "${SFT_TARGET_MODULES}" \
  --lora_dropout "${SFT_LORA_DROPOUT}" \
  --dataset "${TRAIN_DATA}" \
  --eval_dataset "${EVAL_DATA}" \
  --dataset_split "${SFT_DATASET_SPLIT}" \
  --train_split "${SFT_TRAIN_SPLIT}" \
  --eval_split "${SFT_EVAL_SPLIT}" \
  --max_samples "${SFT_MAX_SAMPLES}" \
  --eval_max_samples "${SFT_EVAL_MAX_SAMPLES}" \
  --eval_down_max_samples "${SFT_EVAL_DOWN_MAX_SAMPLES}" \
  --humaneval_dataset "${HUMANEVAL_EVAL_DATA}" \
  --humaneval_config "${SFT_HUMANEVAL_CONFIG}" \
  --humaneval_split "${SFT_HUMANEVAL_SPLIT}" \
  --mbpp_dataset "${MBPP_EVAL_DATA}" \
  --mbpp_config "${SFT_MBPP_CONFIG}" \
  --mbpp_split "${SFT_MBPP_SPLIT}" \
  --multipl_config "${SFT_MULTIPL_CONFIG}" \
  --multipl_split "${SFT_MULTIPL_SPLIT}" \
  --input_key "${SFT_INPUT_KEY}" \
  --output_key "${SFT_OUTPUT_KEY}" \
  --label_key "${SFT_LABEL_KEY}" \
  --prompt_max_len "${SFT_PROMPT_MAX_LEN}" \
  --generate_max_len "${SFT_GENERATE_MAX_LEN}" \
  --wandb_project "${SFT_WANDB_PROJECT}" \
  --wandb_run_name "${RUN_NAME}" \
  --use_tensorboard "${TB_DIR}" \
  "${OPTIONAL_ARGS[@]}" \
  2>&1 | tee "${RUN_DIR}/train_sft.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e

if (( TRAIN_RC != 0 )); then
  echo "[ERROR] SFT training failed with exit code ${TRAIN_RC}"
fi

if [[ "${RUN_CODE_POST_EVAL}" == "true" && "${TRAIN_RC}" -eq 0 ]]; then
  set +e
  run_code_benchmark \
    "sample_pass1_temp06" \
    "${RUN_DIR}/code_benchmarks/sample_pass1_temp06" \
    "${RUN_DIR}/supplement_logs/code_benchmarks/sample_pass1_temp06.log" \
    "0.0" \
    "${SAMPLE_PASS1_TEMPERATURE}" \
    "${SAMPLE_PASS1_N_SAMPLES}" \
    "${SAMPLE_PASS1_PASSK_LIST}"
  PASS1_RC=$?
  run_code_benchmark \
    "sample_pass16_temp06" \
    "${RUN_DIR}/code_benchmarks/sample_pass16_temp06" \
    "${RUN_DIR}/supplement_logs/code_benchmarks/sample_pass16_temp06.log" \
    "0.0" \
    "${SAMPLE_PASS16_TEMPERATURE}" \
    "${SAMPLE_PASS16_N_SAMPLES}" \
    "${SAMPLE_PASS16_PASSK_LIST}"
  PASS16_RC=$?
  set -e
  if (( PASS1_RC != 0 )); then
    EVAL_RC=${PASS1_RC}
  elif (( PASS16_RC != 0 )); then
    EVAL_RC=${PASS16_RC}
  fi
elif [[ "${RUN_CODE_POST_EVAL}" == "true" ]]; then
  echo "[post-eval] skipped because SFT training failed."
fi

FINAL_RC=0
if (( TRAIN_RC != 0 )); then
  FINAL_RC=${TRAIN_RC}
elif (( EVAL_RC != 0 )); then
  FINAL_RC=${EVAL_RC}
fi

{
  echo "# Auto-generated final status"
  echo "# UTC: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  printf "TRAIN_RC=%q\n" "${TRAIN_RC}"
  printf "EVAL_RC=%q\n" "${EVAL_RC}"
  printf "FINAL_RC=%q\n" "${FINAL_RC}"
  printf "RUN_DIR=%q\n" "${RUN_DIR}"
  printf "SAVE_PATH=%q\n" "${SAVE_PATH}"
} > "${RUN_DIR}/final_status.env"

echo "[done] RUN_DIR=${RUN_DIR} FINAL_RC=${FINAL_RC}"
exit "${FINAL_RC}"
