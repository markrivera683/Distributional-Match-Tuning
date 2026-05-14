#!/usr/bin/env bash
set -euo pipefail

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "${csv}" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"${csv}"
}

compute_equiv_samples() {
  python - "$1" "$2" <<'PY'
import math
import sys

base = int(sys.argv[1])
ratio = float(sys.argv[2])
if not (0.10 <= ratio <= 0.25):
    raise SystemExit(f"EBFT_EQUIV_EPOCH must be in [0.10, 0.25], got {ratio}")
print(max(1, math.floor(base * ratio)))
PY
}

has_model_weights() {
  local model_dir="$1"
  local candidate

  [[ -d "${model_dir}" ]] || return 1

  for candidate in "${model_dir}"/*.safetensors "${model_dir}"/*.bin "${model_dir}"/*.pt; do
    if [[ -e "${candidate}" ]]; then
      return 0
    fi
  done

  return 1
}

is_local_resource() {
  local value="$1"
  [[ "${value}" == /* || "${value}" == ./* || "${value}" == ../* ]]
}

path_spec_exists() {
  local value="$1"

  if [[ "${value}" == *"*"* || "${value}" == *"?"* || "${value}" == *"["* ]]; then
    compgen -G "${value}" >/dev/null
    return
  fi

  [[ -e "${value}" ]]
}

resolve_dataset_spec() {
  local value="$1"
  local parquet_glob

  if [[ -d "${value}" ]]; then
    parquet_glob="${value}/data/*.parquet"
    if compgen -G "${parquet_glob}" >/dev/null; then
      echo "${parquet_glob}"
      return
    fi
  fi

  echo "${value}"
}

# --------------------------------------------------------------------
# 0) RUNTIME / DEVICES
#    Default to an 8x A100-style layout while keeping the
#    paper-facing global batch settings unchanged.
# --------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
CRITIC_GPUS="${CRITIC_GPUS:-4}"
REF_GPUS="${REF_GPUS:-${ACTOR_GPUS}}"
REWARD_GPUS="${REWARD_GPUS:-${CRITIC_GPUS}}"

# --------------------------------------------------------------------
# 1) PATHS / DATA
# --------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Match-Tuning}"
MODEL_PATH="${MODEL_PATH:-/root/model}"
TRAIN_DATA="${TRAIN_DATA:-/root/OpenCode}"
VALIDATION_DATA="${VALIDATION_DATA:-${TRAIN_DATA}}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train[1000:]}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-train[:1000]}"
INPUT_KEY="${INPUT_KEY:-input}"
OUTPUT_KEY="${OUTPUT_KEY:-output}"
LABEL_KEY="${LABEL_KEY:-output}"
# Downstream validation sets (paper protocol)
DOWNSTREAM_HUMANEVAL_DATASET="${DOWNSTREAM_HUMANEVAL_DATASET:-openai/openai_humaneval}"
DOWNSTREAM_HUMANEVAL_SPLIT="${DOWNSTREAM_HUMANEVAL_SPLIT:-test}"
DOWNSTREAM_MBPP_DATASET="${DOWNSTREAM_MBPP_DATASET:-google-research-datasets/mbpp}"
DOWNSTREAM_MBPP_CONFIG="${DOWNSTREAM_MBPP_CONFIG:-sanitized}"
DOWNSTREAM_MBPP_SPLIT="${DOWNSTREAM_MBPP_SPLIT:-test}"
DOWNSTREAM_MULTIPLE_DATASET="${DOWNSTREAM_MULTIPLE_DATASET:-nuprl/MultiPL-E}"
# The current in-training hook keeps a single Python-friendly config here,
# while the full target protocol is listed below for reproducibility.
DOWNSTREAM_MULTIPLE_CONFIG="${DOWNSTREAM_MULTIPLE_CONFIG:-humaneval-cpp}"
DOWNSTREAM_MULTIPLE_SPLIT="${DOWNSTREAM_MULTIPLE_SPLIT:-test}"
DOWNSTREAM_BENCHMARKS="${DOWNSTREAM_BENCHMARKS:-HumanEval,MBPP,MultiPL-E}"
DOWNSTREAM_MULTIPLE_LANGUAGES="${DOWNSTREAM_MULTIPLE_LANGUAGES:-cpp,js,ts,rs,cs,go,php,java}"
DOWNSTREAM_MULTIPLE_TARGET_CONFIGS="${DOWNSTREAM_MULTIPLE_TARGET_CONFIGS:-humaneval-cpp,humaneval-js,humaneval-ts,humaneval-rs,humaneval-cs,humaneval-go,humaneval-php,humaneval-java}"
DOWNSTREAM_METRICS="${DOWNSTREAM_METRICS:-greedy_accuracy,pass@1,pass@4,pass@16}"
DOWNSTREAM_GREEDY_TEMPERATURE="${DOWNSTREAM_GREEDY_TEMPERATURE:-0.0}"
DOWNSTREAM_PASSK_TEMPERATURE="${DOWNSTREAM_PASSK_TEMPERATURE:-0.6}"
DOWNSTREAM_PASSK_LIST="${DOWNSTREAM_PASSK_LIST:-1,4,16}"
STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"
DEEPSPEED_LAUNCHER_MODULE="${DEEPSPEED_LAUNCHER_MODULE:-deepspeed.launcher.runner}"

TRAIN_DATA="$(resolve_dataset_spec "${TRAIN_DATA}")"
VALIDATION_DATA="$(resolve_dataset_spec "${VALIDATION_DATA}")"

# --------------------------------------------------------------------
# 2) STAGE 1 — SFT WARM-START
#     Validation hyperparameters from the paper:
#       Training Batch Size = 64
#       Epochs = 1
#       Max Length = 2048
#       Learning Rate = 1e-5
#       Scheduler = Warmup + Cosine Decay to 0.1 x lr
#       Warmup = 0.03
# --------------------------------------------------------------------
GLOBAL_SEED="${GLOBAL_SEED:-43}"

SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-64}"
SFT_MICRO_TRAIN_BATCH_SIZE="${SFT_MICRO_TRAIN_BATCH_SIZE:-8}"
SFT_EVAL_BATCH_SIZE="${SFT_EVAL_BATCH_SIZE:-64}"
SFT_MAX_LEN="${SFT_MAX_LEN:-2048}"
SFT_MAX_EPOCHS="${SFT_MAX_EPOCHS:-1}"
SFT_LR="${SFT_LR:-1e-5}"
SFT_WARMUP_RATIO="${SFT_WARMUP_RATIO:-0.03}"
SFT_LR_SCHEDULER="${SFT_LR_SCHEDULER:-cosine_with_min_lr}"
SFT_MAX_SAMPLES="${SFT_MAX_SAMPLES:-100000}"
SFT_EVAL_MAX_SAMPLES="${SFT_EVAL_MAX_SAMPLES:-1000}"
SFT_EVAL_DOWN_MAX_SAMPLES="${SFT_EVAL_DOWN_MAX_SAMPLES:-1000}"
SFT_EVAL_DOWN_BATCH_SIZE="${SFT_EVAL_DOWN_BATCH_SIZE:-128}"

# --------------------------------------------------------------------
# 3) STAGE 2 — EBFT
#     Validation hyperparameters from the paper (single EBFT stage):
#       Rollout Batch Size = 16
#       Sequence Length = 1024
#       Completion Length = 8
#       Stride = 8
#       Actor Learning Rate = 1e-6
#       Temperature = 0.6
#       KL Coefficient = 0
#       Samples per Prompt = 4
#       Training Batch Size = 64
#       Warmup = 0.03
#       Adam Betas = (0.9, 0.95)
#       Num Epochs = 1
#
#     This script still allows the previously requested max_samples-based
#     EBFT budget override via EBFT_EQUIV_EPOCH.
# --------------------------------------------------------------------
EBFT_ROLLOUT_BATCH_SIZE="${EBFT_ROLLOUT_BATCH_SIZE:-16}"
EBFT_TRAIN_BATCH_SIZE="${EBFT_TRAIN_BATCH_SIZE:-64}"
EBFT_MICRO_TRAIN_BATCH_SIZE="${EBFT_MICRO_TRAIN_BATCH_SIZE:-8}"
EBFT_MICRO_ROLLOUT_BATCH_SIZE="${EBFT_MICRO_ROLLOUT_BATCH_SIZE:-8}"
EBFT_MICRO_REWARD_BATCH_SIZE="${EBFT_MICRO_REWARD_BATCH_SIZE:-8}"

EBFT_PROMPT_MAX_LEN="${EBFT_PROMPT_MAX_LEN:-1024}"
EBFT_CONTEXT_MAX_LEN="${EBFT_CONTEXT_MAX_LEN:-8}"
EBFT_GENERATE_MAX_LEN="${EBFT_GENERATE_MAX_LEN:-8}"
EBFT_STRIDE="${EBFT_STRIDE:-8}"

EBFT_N_SAMPLES_PER_PROMPT="${EBFT_N_SAMPLES_PER_PROMPT:-4}"
EBFT_ACTOR_LR="${EBFT_ACTOR_LR:-1e-6}"
EBFT_TEMPERATURE="${EBFT_TEMPERATURE:-0.6}"
EBFT_TOP_P="${EBFT_TOP_P:-1.0}"
EBFT_INIT_KL_COEF="${EBFT_INIT_KL_COEF:-0.0}"
EBFT_WARMUP_RATIO="${EBFT_WARMUP_RATIO:-0.03}"
EBFT_MAX_EPOCHS="${EBFT_MAX_EPOCHS:-1}"
EBFT_NUM_EPISODES="${EBFT_NUM_EPISODES:-1}"

EBFT_EQUIV_EPOCH="${EBFT_EQUIV_EPOCH:-0.25}"
EBFT_BASE_SAMPLE_COUNT="${EBFT_BASE_SAMPLE_COUNT:-100000}"
EBFT_MAX_SAMPLES="${EBFT_MAX_SAMPLES:-$(compute_equiv_samples "${EBFT_BASE_SAMPLE_COUNT}" "${EBFT_EQUIV_EPOCH}")}"

# Paper-shaped implementation defaults needed for a concrete warm-start run.
FEATURE_MAP_TYPE="${FEATURE_MAP_TYPE:-identity}"
DISTRIBUTION_REWARD_TYPE="${DISTRIBUTION_REWARD_TYPE:-pointwise}"
CF_TARGET_MODE="${CF_TARGET_MODE:-single}"
CE_LOSS_COEF="${CE_LOSS_COEF:-0.03}"
DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-0.5}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"
EMA_BETA="${EMA_BETA:-0.9}"
CRITIC_CLASSIFIER_LOSS_COEF="${CRITIC_CLASSIFIER_LOSS_COEF:-0.0}"

# --------------------------------------------------------------------
# 4) EVAL / LOGGING
#     Downstream validation protocol to keep documented in-script:
#       Eval sets = HumanEval, MBPP, MultiPL-E
#       MultiPL-E languages = C++, JavaScript, TypeScript, Rust, C#, Go, PHP, Java
#       Metrics = greedy accuracy, pass@1, pass@4, pass@16
#       Temperatures = greedy: 0.0, pass@k: 0.6
# --------------------------------------------------------------------
SFT_LOGGING_STEPS="${SFT_LOGGING_STEPS:-10}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:--1}"

EBFT_LOGGING_STEPS="${EBFT_LOGGING_STEPS:-10}"
EBFT_EVAL_STEPS="${EBFT_EVAL_STEPS:-100}"
EBFT_EVAL_MAX_SAMPLES="${EBFT_EVAL_MAX_SAMPLES:-1000}"
EBFT_EVAL_DOWN_MAX_SAMPLES="${EBFT_EVAL_DOWN_MAX_SAMPLES:-128}"
EBFT_EVAL_BATCH_SIZE="${EBFT_EVAL_BATCH_SIZE:-16}"
EBFT_EVAL_DOWN_BATCH_SIZE="${EBFT_EVAL_DOWN_BATCH_SIZE:-128}"
EBFT_EVAL_GENERATE_MAX_LEN="${EBFT_EVAL_GENERATE_MAX_LEN:-512}"
EBFT_EVAL_N_SAMPLES_PER_PROMPT="${EBFT_EVAL_N_SAMPLES_PER_PROMPT:-4}"
EBFT_EVAL_N_SAMPLES_PER_PROMPT_DOWN="${EBFT_EVAL_N_SAMPLES_PER_PROMPT_DOWN:-4}"
EBFT_SAVE_STEPS="${EBFT_SAVE_STEPS:--1}"

# Post-training benchmark harness
RUN_POST_STAGE1_BENCHMARKS="${RUN_POST_STAGE1_BENCHMARKS:-true}"
RUN_POST_STAGE2_BENCHMARKS="${RUN_POST_STAGE2_BENCHMARKS:-true}"
CODE_BENCHMARK_SCRIPT="${CODE_BENCHMARK_SCRIPT:-${REPO_ROOT}/scripts/benchmarks/run_code_generation_benchmarks.py}"
CODE_BENCHMARK_BACKEND="${CODE_BENCHMARK_BACKEND:-auto}"
CODE_BENCHMARKS_TO_RUN="${CODE_BENCHMARKS_TO_RUN:-humaneval,mbpp,multipl}"
CODE_BENCHMARK_PROMPT_MAX_LEN="${CODE_BENCHMARK_PROMPT_MAX_LEN:-1024}"
CODE_BENCHMARK_MAX_NEW_TOKENS="${CODE_BENCHMARK_MAX_NEW_TOKENS:-512}"
CODE_BENCHMARK_TOP_P="${CODE_BENCHMARK_TOP_P:-1.0}"
CODE_BENCHMARK_N_SAMPLES="${CODE_BENCHMARK_N_SAMPLES:-16}"
CODE_BENCHMARK_GREEDY_BATCH_SIZE="${CODE_BENCHMARK_GREEDY_BATCH_SIZE:-16}"
CODE_BENCHMARK_SAMPLE_BATCH_SIZE="${CODE_BENCHMARK_SAMPLE_BATCH_SIZE:-4}"
CODE_BENCHMARK_MAX_NUM_SEQS="${CODE_BENCHMARK_MAX_NUM_SEQS:-128}"
CODE_BENCHMARK_TP_SIZE="${CODE_BENCHMARK_TP_SIZE:-4}"
CODE_BENCHMARK_TIMEOUT_SECONDS="${CODE_BENCHMARK_TIMEOUT_SECONDS:-10}"
CODE_BENCHMARK_MAX_SAMPLES_PER_BENCHMARK="${CODE_BENCHMARK_MAX_SAMPLES_PER_BENCHMARK:-0}"
CODE_BENCHMARK_ENABLE_PREFIX_CACHING="${CODE_BENCHMARK_ENABLE_PREFIX_CACHING:-false}"

# --------------------------------------------------------------------
# 5) ENV / RUN DIR
# --------------------------------------------------------------------
DEFAULT_HF_HUB_OFFLINE="1"
DEFAULT_HF_DATASETS_OFFLINE="1"

if ! is_local_resource "${MODEL_PATH}"; then
  DEFAULT_HF_HUB_OFFLINE="0"
fi

if ! is_local_resource "${TRAIN_DATA}" || ! is_local_resource "${VALIDATION_DATA}"; then
  DEFAULT_HF_DATASETS_OFFLINE="0"
fi

# Benchmark datasets (HumanEval / MBPP / MultiPL-E) are expected to be
# pre-cached under HF_HOME.  Offline mode is fine as long as the cache is
# populated (see verification at the bottom of this section).
# If you ever need to download them for the first time, temporarily run:
#   HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 bash scripts/reproduction_sftEbft.sh

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-${DEFAULT_HF_HUB_OFFLINE}}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-${DEFAULT_HF_DATASETS_OFFLINE}}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

RUN_NAME="${RUN_NAME:-repro_sft_ebft_qwen25_1p5b_$(date +%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

SFT_SAVE_PATH="${SFT_SAVE_PATH:-${RUN_DIR}/stage1_sft_model}"
SFT_CKPT_PATH="${SFT_CKPT_PATH:-${RUN_DIR}/stage1_sft_ckpt}"
SFT_TB_DIR="${SFT_TB_DIR:-${RUN_DIR}/tb_stage1_sft}"

EBFT_SAVE_PATH="${EBFT_SAVE_PATH:-${RUN_DIR}/stage2_ebft_model}"
EBFT_CKPT_PATH="${EBFT_CKPT_PATH:-${RUN_DIR}/stage2_ebft_ckpt}"
EBFT_TB_DIR="${EBFT_TB_DIR:-${RUN_DIR}/tb_stage2_ebft}"

SCRIPT_NAME="$(basename "$0" .sh)"
SCRIPT_LOG_PATH="${RUN_DIR}/${SCRIPT_NAME}.log"

mkdir -p "${RUN_DIR}" "${SFT_SAVE_PATH}" "${SFT_CKPT_PATH}" "${SFT_TB_DIR}" "${EBFT_SAVE_PATH}" "${EBFT_CKPT_PATH}" "${EBFT_TB_DIR}"
exec > >(tee -a "${SCRIPT_LOG_PATH}") 2>&1

# --------------------------------------------------------------------
# 6) SANITY CHECKS
# --------------------------------------------------------------------
gpu_count="$(count_csv_items "${CUDA_VISIBLE_DEVICES}")"

if [[ ! -x "${STUDENT_PYTHON_BIN}" ]]; then
  echo "[ERROR] STUDENT_PYTHON_BIN not executable: ${STUDENT_PYTHON_BIN}"
  exit 1
fi

if ! "${STUDENT_PYTHON_BIN}" -c "import importlib; importlib.import_module('${DEEPSPEED_LAUNCHER_MODULE}')" >/dev/null 2>&1; then
  echo "[ERROR] DeepSpeed launcher module is not importable from ${STUDENT_PYTHON_BIN}"
  exit 1
fi

if (( gpu_count == 0 )); then
  echo "[ERROR] No visible GPU found in CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  exit 1
fi

if [[ "${MODEL_PATH}" == /* && ! -e "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

if [[ "${MODEL_PATH}" == /* ]] && ! has_model_weights "${MODEL_PATH}"; then
  echo "[ERROR] MODEL_PATH exists but no model weights were found: ${MODEL_PATH}"
  exit 1
fi

if [[ "${HF_HUB_OFFLINE}" == "1" ]] && ! is_local_resource "${MODEL_PATH}"; then
  echo "[ERROR] HF_HUB_OFFLINE=1 but MODEL_PATH is a remote repo id: ${MODEL_PATH}"
  echo "        Use a local model directory or set HF_HUB_OFFLINE=0."
  exit 1
fi

if is_local_resource "${TRAIN_DATA}" && ! path_spec_exists "${TRAIN_DATA}"; then
  echo "[ERROR] TRAIN_DATA not found: ${TRAIN_DATA}"
  exit 1
fi

if is_local_resource "${VALIDATION_DATA}" && ! path_spec_exists "${VALIDATION_DATA}"; then
  echo "[ERROR] VALIDATION_DATA not found: ${VALIDATION_DATA}"
  exit 1
fi

if is_local_resource "${DOWNSTREAM_HUMANEVAL_DATASET}" && ! path_spec_exists "${DOWNSTREAM_HUMANEVAL_DATASET}"; then
  echo "[ERROR] DOWNSTREAM_HUMANEVAL_DATASET not found: ${DOWNSTREAM_HUMANEVAL_DATASET}"
  exit 1
fi

if is_local_resource "${DOWNSTREAM_MBPP_DATASET}" && ! path_spec_exists "${DOWNSTREAM_MBPP_DATASET}"; then
  echo "[ERROR] DOWNSTREAM_MBPP_DATASET not found: ${DOWNSTREAM_MBPP_DATASET}"
  exit 1
fi

if is_local_resource "${DOWNSTREAM_MULTIPLE_DATASET}" && ! path_spec_exists "${DOWNSTREAM_MULTIPLE_DATASET}"; then
  echo "[ERROR] DOWNSTREAM_MULTIPLE_DATASET not found: ${DOWNSTREAM_MULTIPLE_DATASET}"
  exit 1
fi

if [[ "${HF_DATASETS_OFFLINE}" == "1" ]] && (! is_local_resource "${TRAIN_DATA}" || ! is_local_resource "${VALIDATION_DATA}"); then
  echo "[ERROR] HF_DATASETS_OFFLINE=1 but TRAIN_DATA/VALIDATION_DATA use remote dataset ids."
  echo "        Use local dataset paths or set HF_DATASETS_OFFLINE=0."
  exit 1
fi

if [[ "${HF_DATASETS_OFFLINE}" == "1" ]] && \
   (! is_local_resource "${DOWNSTREAM_HUMANEVAL_DATASET}" || \
    ! is_local_resource "${DOWNSTREAM_MBPP_DATASET}" || \
    ! is_local_resource "${DOWNSTREAM_MULTIPLE_DATASET}"); then
  echo "[WARN] HF_DATASETS_OFFLINE=1 and downstream benchmarks use remote HF ids."
  echo "       This is fine if they are already cached under ${HF_HOME}."
  echo "       If loading fails, run once with HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 to populate the cache."
fi

if [[ "${RUN_POST_STAGE1_BENCHMARKS}" == "true" || "${RUN_POST_STAGE2_BENCHMARKS}" == "true" ]]; then
  if [[ ! -f "${CODE_BENCHMARK_SCRIPT}" ]]; then
    echo "[ERROR] CODE_BENCHMARK_SCRIPT not found: ${CODE_BENCHMARK_SCRIPT}"
    exit 1
  fi
fi

if (( SFT_TRAIN_BATCH_SIZE % (SFT_MICRO_TRAIN_BATCH_SIZE * gpu_count) != 0 )); then
  echo "[ERROR] SFT train_batch_size must be divisible by micro_train_batch_size * gpu_count"
  echo "        ${SFT_TRAIN_BATCH_SIZE} % (${SFT_MICRO_TRAIN_BATCH_SIZE} * ${gpu_count}) != 0"
  exit 1
fi

if (( ACTOR_GPUS + CRITIC_GPUS > gpu_count )); then
  echo "[ERROR] ACTOR_GPUS(${ACTOR_GPUS}) + CRITIC_GPUS(${CRITIC_GPUS}) > visible GPU count(${gpu_count})"
  exit 1
fi

if (( EBFT_TRAIN_BATCH_SIZE != EBFT_N_SAMPLES_PER_PROMPT * EBFT_ROLLOUT_BATCH_SIZE )); then
  echo "[ERROR] EBFT train_batch_size must equal rollout_batch_size * n_samples_per_prompt"
  echo "        got ${EBFT_TRAIN_BATCH_SIZE} vs ${EBFT_ROLLOUT_BATCH_SIZE} * ${EBFT_N_SAMPLES_PER_PROMPT}"
  exit 1
fi

if (( EBFT_MICRO_TRAIN_BATCH_SIZE < EBFT_N_SAMPLES_PER_PROMPT || EBFT_MICRO_TRAIN_BATCH_SIZE % EBFT_N_SAMPLES_PER_PROMPT != 0 )); then
  echo "[ERROR] EBFT micro_train_batch_size must be >= n_samples_per_prompt and divisible by it"
  exit 1
fi

if (( EBFT_MICRO_ROLLOUT_BATCH_SIZE % EBFT_N_SAMPLES_PER_PROMPT != 0 )); then
  echo "[ERROR] EBFT micro_rollout_batch_size must be divisible by n_samples_per_prompt"
  exit 1
fi

if (( EBFT_PROMPT_MAX_LEN < EBFT_GENERATE_MAX_LEN + EBFT_CONTEXT_MAX_LEN )); then
  echo "[ERROR] EBFT prompt_max_len must be >= generate_max_len + context_max_len"
  exit 1
fi

if (( (EBFT_PROMPT_MAX_LEN - EBFT_GENERATE_MAX_LEN - EBFT_CONTEXT_MAX_LEN) % EBFT_STRIDE != 0 )); then
  echo "[ERROR] (prompt_max_len - generate_max_len - context_max_len) must be divisible by stride"
  exit 1
fi

# Re-run the range check even when EBFT_MAX_SAMPLES is manually overridden.
compute_equiv_samples "${EBFT_BASE_SAMPLE_COUNT}" "${EBFT_EQUIV_EPOCH}" >/dev/null

echo "========== Reproduction: SFT -> EBFT =========="
echo "RUN_DIR:                     ${RUN_DIR}"
echo "CUDA_VISIBLE_DEVICES:        ${CUDA_VISIBLE_DEVICES} (count=${gpu_count})"
echo "Actor/Critic GPUs:           ${ACTOR_GPUS}/${CRITIC_GPUS}"
echo "Model:                       ${MODEL_PATH}"
echo "Train data:                  ${TRAIN_DATA}"
echo "Validation data:             ${VALIDATION_DATA}"
echo "Train split:                 ${TRAIN_SPLIT}"
echo "Validation split:            ${VALIDATION_SPLIT}"
echo "Input/Output/Label keys:     ${INPUT_KEY}/${OUTPUT_KEY}/${LABEL_KEY}"
echo "Downstream benchmarks:       ${DOWNSTREAM_BENCHMARKS}"
echo "Downstream HumanEval:        ${DOWNSTREAM_HUMANEVAL_DATASET} [${DOWNSTREAM_HUMANEVAL_SPLIT}]"
echo "Downstream MBPP:             ${DOWNSTREAM_MBPP_DATASET} (${DOWNSTREAM_MBPP_CONFIG}) [${DOWNSTREAM_MBPP_SPLIT}]"
echo "Downstream MultiPL-E hook:   ${DOWNSTREAM_MULTIPLE_DATASET} (${DOWNSTREAM_MULTIPLE_CONFIG}) [${DOWNSTREAM_MULTIPLE_SPLIT}]"
echo "MultiPL-E target languages:  ${DOWNSTREAM_MULTIPLE_LANGUAGES}"
echo "MultiPL-E target configs:    ${DOWNSTREAM_MULTIPLE_TARGET_CONFIGS}"
echo "Downstream metrics:          ${DOWNSTREAM_METRICS}"
echo "Downstream temps:            greedy=${DOWNSTREAM_GREEDY_TEMPERATURE}, pass@k=${DOWNSTREAM_PASSK_TEMPERATURE}"
echo "Downstream pass@k list:      ${DOWNSTREAM_PASSK_LIST}"
echo "Post Stage1 benchmarks:      ${RUN_POST_STAGE1_BENCHMARKS}"
echo "Post Stage2 benchmarks:      ${RUN_POST_STAGE2_BENCHMARKS}"
echo "Benchmark backend:           ${CODE_BENCHMARK_BACKEND}"
echo "Benchmarks to run:           ${CODE_BENCHMARKS_TO_RUN}"
echo "Benchmark samples/prompt:    ${CODE_BENCHMARK_N_SAMPLES}"
echo "HF_HUB_OFFLINE:              ${HF_HUB_OFFLINE}"
echo "HF_DATASETS_OFFLINE:         ${HF_DATASETS_OFFLINE}"
echo ""
echo "[Stage 1] SFT warm-start"
echo "  train_batch_size:          ${SFT_TRAIN_BATCH_SIZE}"
echo "  max_epochs:                ${SFT_MAX_EPOCHS}"
echo "  max_len:                   ${SFT_MAX_LEN}"
echo "  learning_rate:             ${SFT_LR}"
echo "  lr_scheduler:              ${SFT_LR_SCHEDULER}"
echo "  warmup_ratio:              ${SFT_WARMUP_RATIO}"
echo ""
echo "[Stage 2] EBFT"
echo "  rollout_batch_size:        ${EBFT_ROLLOUT_BATCH_SIZE}"
echo "  train_batch_size:          ${EBFT_TRAIN_BATCH_SIZE}"
echo "  prompt_max_len:            ${EBFT_PROMPT_MAX_LEN}"
echo "  context_max_len:           ${EBFT_CONTEXT_MAX_LEN}"
echo "  generate_max_len:          ${EBFT_GENERATE_MAX_LEN}"
echo "  stride:                    ${EBFT_STRIDE}"
echo "  n_samples_per_prompt:      ${EBFT_N_SAMPLES_PER_PROMPT}"
echo "  actor_learning_rate:       ${EBFT_ACTOR_LR}"
echo "  temperature:               ${EBFT_TEMPERATURE}"
echo "  distribution_reward:       ${DISTRIBUTION_REWARD_TYPE}"
echo "  cf_target_mode:            ${CF_TARGET_MODE}"
echo "  init_kl_coef:              ${EBFT_INIT_KL_COEF}"
echo "  warmup_ratio:              ${EBFT_WARMUP_RATIO}"
echo "  adam_betas:                (0.9, 0.95)"
echo "  critic_classifier_loss:    ${CRITIC_CLASSIFIER_LOSS_COEF}"
echo "  max_epochs:                ${EBFT_MAX_EPOCHS}"
echo "  equiv_epoch_budget:        ${EBFT_EQUIV_EPOCH}"
echo "  ebft_max_samples:          ${EBFT_MAX_SAMPLES}"
echo "==============================================="

run_code_benchmarks() {
  local stage_name="$1"
  local model_path="$2"
  local bench_dir="$3"

  mkdir -p "${bench_dir}"
  echo ""
  echo "===== ${stage_name}: downstream code benchmarks ====="

  local benchmark_cmd=(
    "${STUDENT_PYTHON_BIN}"
    "${CODE_BENCHMARK_SCRIPT}"
    --model_path "${model_path}"
    --output_dir "${bench_dir}"
    --backend "${CODE_BENCHMARK_BACKEND}"
    --benchmarks "${CODE_BENCHMARKS_TO_RUN}"
    --prompt_max_len "${CODE_BENCHMARK_PROMPT_MAX_LEN}"
    --max_new_tokens "${CODE_BENCHMARK_MAX_NEW_TOKENS}"
    --top_p "${CODE_BENCHMARK_TOP_P}"
    --greedy_temperature "${DOWNSTREAM_GREEDY_TEMPERATURE}"
    --sample_temperature "${DOWNSTREAM_PASSK_TEMPERATURE}"
    --passk_list "${DOWNSTREAM_PASSK_LIST}"
    --n_samples "${CODE_BENCHMARK_N_SAMPLES}"
    --seed "${GLOBAL_SEED}"
    --greedy_batch_size "${CODE_BENCHMARK_GREEDY_BATCH_SIZE}"
    --sample_batch_size "${CODE_BENCHMARK_SAMPLE_BATCH_SIZE}"
    --max_num_seqs "${CODE_BENCHMARK_MAX_NUM_SEQS}"
    --tp_size "${CODE_BENCHMARK_TP_SIZE}"
    --timeout_seconds "${CODE_BENCHMARK_TIMEOUT_SECONDS}"
    --max_samples_per_benchmark "${CODE_BENCHMARK_MAX_SAMPLES_PER_BENCHMARK}"
    --skip_missing_toolchains
    --humaneval_dataset "${DOWNSTREAM_HUMANEVAL_DATASET}"
    --humaneval_split "${DOWNSTREAM_HUMANEVAL_SPLIT}"
    --mbpp_dataset "${DOWNSTREAM_MBPP_DATASET}"
    --mbpp_config "${DOWNSTREAM_MBPP_CONFIG}"
    --mbpp_split "${DOWNSTREAM_MBPP_SPLIT}"
    --multipl_dataset "${DOWNSTREAM_MULTIPLE_DATASET}"
    --multipl_configs "${DOWNSTREAM_MULTIPLE_TARGET_CONFIGS}"
    --multipl_split "${DOWNSTREAM_MULTIPLE_SPLIT}"
  )

  if [[ "${CODE_BENCHMARK_ENABLE_PREFIX_CACHING}" == "true" ]]; then
    benchmark_cmd+=(--enable_prefix_caching)
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${benchmark_cmd[@]}" 2>&1 | tee "${bench_dir}/benchmark.log"
}

ray stop --force 2>/dev/null || true
sleep 2
cd "${REPO_ROOT}"

# --------------------------------------------------------------------
# 7) STAGE 1 — SFT WARM-START
# --------------------------------------------------------------------
echo ""
echo "===== Stage 1 / 2: SFT warm-start ====="

sft_cmd=(
  "${STUDENT_PYTHON_BIN}"
  -m
  "${DEEPSPEED_LAUNCHER_MODULE}"
  --module
  openrlhf.cli.train_sft
  --bf16
  --flash_attn
  --gradient_checkpointing
  --disable_ds_ckpt
  --save_hf_ckpt
  --pretrain "${MODEL_PATH}"
  --dataset "${TRAIN_DATA}"
  --eval_dataset "${VALIDATION_DATA}"
  --dataset_split "${TRAIN_SPLIT}"
  --train_split "${TRAIN_SPLIT}"
  --eval_split "${VALIDATION_SPLIT}"
  --input_key "${INPUT_KEY}"
  --output_key "${OUTPUT_KEY}"
  --label_key "${LABEL_KEY}"
  --max_len "${SFT_MAX_LEN}"
  --train_batch_size "${SFT_TRAIN_BATCH_SIZE}"
  --micro_train_batch_size "${SFT_MICRO_TRAIN_BATCH_SIZE}"
  --eval_batch_size "${SFT_EVAL_BATCH_SIZE}"
  --eval_down_batch_size "${SFT_EVAL_DOWN_BATCH_SIZE}"
  --max_samples "${SFT_MAX_SAMPLES}"
  --eval_max_samples "${SFT_EVAL_MAX_SAMPLES}"
  --eval_down_max_samples "${SFT_EVAL_DOWN_MAX_SAMPLES}"
  --max_epochs "${SFT_MAX_EPOCHS}"
  --learning_rate "${SFT_LR}"
  --lr_warmup_ratio "${SFT_WARMUP_RATIO}"
  --lr_scheduler "${SFT_LR_SCHEDULER}"
  --adam_betas 0.9 0.95
  --zero_stage 2
  --seed "${GLOBAL_SEED}"
  --logging_steps "${SFT_LOGGING_STEPS}"
  --save_steps "${SFT_SAVE_STEPS}"
  --eval_steps "${SFT_EVAL_STEPS}"
  --save_path "${SFT_SAVE_PATH}"
  --ckpt_path "${SFT_CKPT_PATH}"
  --use_tensorboard "${SFT_TB_DIR}"
  --wandb_run_name "${RUN_NAME}_stage1_sft"
  --humaneval_dataset "${DOWNSTREAM_HUMANEVAL_DATASET}"
  --humaneval_split "${DOWNSTREAM_HUMANEVAL_SPLIT}"
  --mbpp_dataset "${DOWNSTREAM_MBPP_DATASET}"
  --mbpp_config "${DOWNSTREAM_MBPP_CONFIG}"
  --mbpp_split "${DOWNSTREAM_MBPP_SPLIT}"
  --multipl_dataset "${DOWNSTREAM_MULTIPLE_DATASET}"
  --multipl_config "${DOWNSTREAM_MULTIPLE_CONFIG}"
  --multipl_split "${DOWNSTREAM_MULTIPLE_SPLIT}"
)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${sft_cmd[@]}" 2>&1 | tee "${RUN_DIR}/stage1_sft.log"

if ! has_model_weights "${SFT_SAVE_PATH}"; then
  echo "[ERROR] Stage 1 finished but no warm-start weights were found in ${SFT_SAVE_PATH}"
  exit 1
fi

echo "[Stage 1] Warm-start checkpoint ready at ${SFT_SAVE_PATH}"

ray stop --force 2>/dev/null || true
sleep 2

if [[ "${RUN_POST_STAGE1_BENCHMARKS}" == "true" ]]; then
  run_code_benchmarks "Stage 1" "${SFT_SAVE_PATH}" "${RUN_DIR}/benchmarks_stage1"
  ray stop --force 2>/dev/null || true
  sleep 2
fi

# --------------------------------------------------------------------
# 8) STAGE 2 — EBFT
# --------------------------------------------------------------------
echo ""
echo "===== Stage 2 / 2: EBFT ====="

ebft_cmd=(
  "${STUDENT_PYTHON_BIN}" -m openrlhf.cli.train_ebft_ray
  --bf16
  --flash_attn
  --gradient_checkpointing
  --pretrain_mode
  --no_chat_template
  --disable_ds_ckpt
  --colocate_actor_ref
  --colocate_critic_reward
  --use_kl_loss
  --use_whitening
  --enable_ema
  --distribution_reward_type "${DISTRIBUTION_REWARD_TYPE}"
  --cf_target_mode "${CF_TARGET_MODE}"
  --feature_map_type "${FEATURE_MAP_TYPE}"
  --pretrain "${SFT_SAVE_PATH}"
  --critic_pretrain "${MODEL_PATH}"
  --prompt_data "${TRAIN_DATA}"
  --eval_dataset "${VALIDATION_DATA}"
  --input_key "${INPUT_KEY}"
  --label_key "${LABEL_KEY}"
  --output_key "${OUTPUT_KEY}"
  --prompt_split "${TRAIN_SPLIT}"
  --eval_split "${VALIDATION_SPLIT}"
  --prompt_max_len "${EBFT_PROMPT_MAX_LEN}"
  --context_max_len "${EBFT_CONTEXT_MAX_LEN}"
  --generate_max_len "${EBFT_GENERATE_MAX_LEN}"
  --stride "${EBFT_STRIDE}"
  --n_samples_per_prompt "${EBFT_N_SAMPLES_PER_PROMPT}"
  --rollout_batch_size "${EBFT_ROLLOUT_BATCH_SIZE}"
  --train_batch_size "${EBFT_TRAIN_BATCH_SIZE}"
  --micro_train_batch_size "${EBFT_MICRO_TRAIN_BATCH_SIZE}"
  --micro_rollout_batch_size "${EBFT_MICRO_ROLLOUT_BATCH_SIZE}"
  --micro_reward_batch_size "${EBFT_MICRO_REWARD_BATCH_SIZE}"
  --max_samples "${EBFT_MAX_SAMPLES}"
  --num_episodes "${EBFT_NUM_EPISODES}"
  --max_epochs "${EBFT_MAX_EPOCHS}"
  --actor_num_nodes 1
  --actor_num_gpus_per_node "${ACTOR_GPUS}"
  --critic_num_nodes 1
  --critic_num_gpus_per_node "${CRITIC_GPUS}"
  --ref_num_nodes 1
  --ref_num_gpus_per_node "${REF_GPUS}"
  --reward_num_nodes 1
  --reward_num_gpus_per_node "${REWARD_GPUS}"
  --advantage_estimator rloo
  --init_kl_coef "${EBFT_INIT_KL_COEF}"
  --kl_estimator k2
  --temperature "${EBFT_TEMPERATURE}"
  --top_p "${EBFT_TOP_P}"
  --actor_learning_rate "${EBFT_ACTOR_LR}"
  --critic_learning_rate 0.0
  --critic_lr_head 0.0
  --critic_classifier_loss_coef "${CRITIC_CLASSIFIER_LOSS_COEF}"
  --lr_warmup_ratio "${EBFT_WARMUP_RATIO}"
  --lr_scheduler cosine_with_min_lr
  --adam_betas 0.9 0.95
  --zero_stage 2
  --seed "${GLOBAL_SEED}"
  --ema_beta "${EMA_BETA}"
  --hidden_state_method concat
  --embed_method last_token
  --critic_sequence_level last_token
  --classifier_sequence_selection closest
  --ce_loss_coef "${CE_LOSS_COEF}"
  --rl_loss_coef 1.0
  --diversity_rew_coef "${DIVERSITY_REW_COEF}"
  --alignment_rew_coef "${ALIGNMENT_REW_COEF}"
  --eval_steps "${EBFT_EVAL_STEPS}"
  --eval_max_samples "${EBFT_EVAL_MAX_SAMPLES}"
  --eval_down_max_samples "${EBFT_EVAL_DOWN_MAX_SAMPLES}"
  --eval_batch_size "${EBFT_EVAL_BATCH_SIZE}"
  --eval_down_batch_size "${EBFT_EVAL_DOWN_BATCH_SIZE}"
  --eval_generate_max_len "${EBFT_EVAL_GENERATE_MAX_LEN}"
  --eval_n_samples_per_prompt "${EBFT_EVAL_N_SAMPLES_PER_PROMPT}"
  --eval_n_samples_per_prompt_down "${EBFT_EVAL_N_SAMPLES_PER_PROMPT_DOWN}"
  --eval_temperature 0.6
  --eval_temperature_down 0.6
  --logging_steps "${EBFT_LOGGING_STEPS}"
  --save_steps "${EBFT_SAVE_STEPS}"
  --save_hf_ckpt
  --save_path "${EBFT_SAVE_PATH}"
  --ckpt_path "${EBFT_CKPT_PATH}"
  --use_tensorboard "${EBFT_TB_DIR}"
  --wandb_run_name "${RUN_NAME}_stage2_ebft"
)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${ebft_cmd[@]}" 2>&1 | tee "${RUN_DIR}/stage2_ebft.log"

ray stop --force 2>/dev/null || true
sleep 2

if [[ "${RUN_POST_STAGE2_BENCHMARKS}" == "true" ]]; then
  run_code_benchmarks "Stage 2" "${EBFT_SAVE_PATH}" "${RUN_DIR}/benchmarks_stage2"
fi

echo ""
echo "===== Finished ====="
echo "Run dir:        ${RUN_DIR}"
echo "Stage 1 model:  ${SFT_SAVE_PATH}"
echo "Stage 2 model:  ${EBFT_SAVE_PATH}"
