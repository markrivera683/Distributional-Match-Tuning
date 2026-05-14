#!/usr/bin/env bash
# Standalone diff-dataset G1 launcher.
#
# Dataset: OpenCodeInstruct train pool, MBPP + HumanEval post-eval.
# Model:   Qwen3.5-2B student by default.
# Reward:  pointwise, cf_target_mode=single, no teacher.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "${csv}" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"${csv}"
}

write_jsonl_datasets() {
  local node_rank="${PET_NODE_RANK:-${RANK:-0}}"
  local world_size="${PET_WORLD_SIZE:-${WORLD_SIZE:-1}}"
  local manifest="${PREPARED_DATA_DIR}/manifest.env"

  if (( world_size > 1 && node_rank > 0 )); then
    echo "[prepare] worker rank=${node_rank}: waiting for prepared datasets at ${PREPARED_DATA_DIR}"
    local waited=0
    while [[ ! -s "${manifest}" || ! -s "${TRAIN_DATA}" || ! -s "${MBPP_EVAL_DATA}" || ! -s "${HUMANEVAL_EVAL_DATA}" ]]; do
      sleep 5
      waited=$((waited + 5))
      if (( waited >= PREPARE_DIFF_DATASETS_WAIT_SECONDS )); then
        echo "[ERROR] prepared datasets not ready after ${PREPARE_DIFF_DATASETS_WAIT_SECONDS}s"
        exit 1
      fi
    done
    return 0
  fi

  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_code_datasets.py" \
    --output-dir "${PREPARED_DATA_DIR}" \
    --train-samples "${TRAIN_SAMPLE_POOL}" \
    ${PREPARE_DIFF_DATASETS_FORCE:+--force}
}

run_code_benchmark_posteval() {
  local output_root="${RUN_DIR}/code_benchmarks"
  local log_dir="${RUN_DIR}/supplement_logs/code_benchmarks"
  local model_label="${POST_EVAL_MODEL_LABEL:-g1_${RUN_NAME}}"
  local only_model_specs="${model_label}|${SAVE_PATH}"
  local rc=0
  local script_rc=0
  mkdir -p "${output_root}" "${log_dir}"

  echo ""
  echo "===== code post-eval via diff_dataset code-eval launchers ====="
  echo "[post-eval] only_model_specs=${only_model_specs}"

  RUN_NAME="${RUN_NAME}_code_eval_repeats" \
  OUTPUT_ROOT="${output_root}" \
  ONLY_MODEL_SPECS="${only_model_specs}" \
  CODE_BENCHMARK_PYTHON_BIN="${CODE_BENCHMARK_PYTHON_BIN}" \
  CODE_BENCHMARK_SCRIPT="${CODE_BENCHMARK_SCRIPT}" \
  PREPARED_DATA_DIR="${PREPARED_DATA_DIR}" \
  HUMANEVAL_EVAL_DATA="${HUMANEVAL_EVAL_DATA}" \
  MBPP_EVAL_DATA="${MBPP_EVAL_DATA}" \
  MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES}" \
  VLLM_TP_SIZE="${VLLM_TP_SIZE}" \
  VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
  BASE_SEED="${VLLM_SEED}" \
  PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN}" \
  MAX_NEW_TOKENS="${CODE_EVAL_MAX_NEW_TOKENS}" \
  TOP_P="${CODE_EVAL_TOP_P}" \
  REPETITION_PENALTY="${CODE_EVAL_REPETITION_PENALTY}" \
  GREEDY_BATCH_SIZE="${CODE_EVAL_GREEDY_BATCH_SIZE}" \
  SAMPLE_BATCH_SIZE="${CODE_EVAL_SAMPLE_BATCH_SIZE}" \
  MAX_SAMPLES_PER_BENCHMARK="${POST_EVAL_MAX_SAMPLES}" \
  TIMEOUT_SECONDS="${CODE_EVAL_TIMEOUT_SECONDS}" \
  bash "${SCRIPT_DIR}/run_code_eval_repeats_baseline_g1_g2_g3.sh" \
    2>&1 | tee "${log_dir}/run_code_eval_repeats.log"
  script_rc=${PIPESTATUS[0]}
  (( script_rc != 0 )) && rc=${script_rc}

  RUN_NAME="${RUN_NAME}_code_eval_pass16_once" \
  OUTPUT_ROOT="${output_root}" \
  ONLY_MODEL_SPECS="${only_model_specs}" \
  CODE_BENCHMARK_PYTHON_BIN="${CODE_BENCHMARK_PYTHON_BIN}" \
  CODE_BENCHMARK_SCRIPT="${CODE_BENCHMARK_SCRIPT}" \
  PREPARED_DATA_DIR="${PREPARED_DATA_DIR}" \
  HUMANEVAL_EVAL_DATA="${HUMANEVAL_EVAL_DATA}" \
  MBPP_EVAL_DATA="${MBPP_EVAL_DATA}" \
  MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES}" \
  VLLM_TP_SIZE="${VLLM_TP_SIZE}" \
  VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
  BASE_SEED="${VLLM_SEED}" \
  PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN}" \
  MAX_NEW_TOKENS="${CODE_EVAL_MAX_NEW_TOKENS}" \
  TOP_P="${CODE_EVAL_TOP_P}" \
  REPETITION_PENALTY="${CODE_EVAL_REPETITION_PENALTY}" \
  GREEDY_BATCH_SIZE="${CODE_EVAL_GREEDY_BATCH_SIZE}" \
  SAMPLE_BATCH_SIZE="${CODE_EVAL_SAMPLE_BATCH_SIZE}" \
  MAX_SAMPLES_PER_BENCHMARK="${POST_EVAL_MAX_SAMPLES}" \
  TIMEOUT_SECONDS="${CODE_EVAL_TIMEOUT_SECONDS}" \
  bash "${SCRIPT_DIR}/run_code_eval_pass16_once_baseline_g1_g2_g3.sh" \
    2>&1 | tee "${log_dir}/run_code_eval_pass16_once.log"
  script_rc=${PIPESTATUS[0]}
  (( rc == 0 && script_rc != 0 )) && rc=${script_rc}

  return "${rc}"
}

# ---------------------------------------------------------------------------
# 1) Explicit paths / data / model
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/mnt/data/ebft-distribution-new/code}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-2B}"
PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-/mnt/data/ebft-distribution-new/outputs/diff_dataset_prepared}"
TRAIN_SAMPLE_POOL="${TRAIN_SAMPLE_POOL:-100000}"
TRAIN_DATA="${TRAIN_DATA:-${PREPARED_DATA_DIR}/opencodeinstruct_qa_100k.jsonl}"
MBPP_EVAL_DATA="${MBPP_EVAL_DATA:-${PREPARED_DATA_DIR}/mbpp_eval_qa.jsonl}"
HUMANEVAL_EVAL_DATA="${HUMANEVAL_EVAL_DATA:-${PREPARED_DATA_DIR}/humaneval_eval_qa.jsonl}"
POST_EVAL_DATASETS="${POST_EVAL_DATASETS:-mbpp:${MBPP_EVAL_DATA},humaneval:${HUMANEVAL_EVAL_DATA}}"
EVAL_DATA="${EVAL_DATA:-${MBPP_EVAL_DATA}}"
PROMPT_SPLIT="${PROMPT_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PREPARE_DIFF_DATASETS_WAIT_SECONDS="${PREPARE_DIFF_DATASETS_WAIT_SECONDS:-1800}"

STUDENT_VENV="${STUDENT_VENV:-/mnt/workspace/venvs/.venv}"
TEACHER_VENV="${TEACHER_VENV:-/mnt/workspace/venvs/.teacherVenv}"
ANALYSIS_VENV="${ANALYSIS_VENV:-${STUDENT_VENV}}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"

# ---------------------------------------------------------------------------
# 2) Environment
# ---------------------------------------------------------------------------
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
export RAY_TMPDIR="${RAY_TMPDIR:-/root/ray_tmp}"
export TMPDIR="${TMPDIR:-${RAY_TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_LAUNCH_MODE="${NCCL_LAUNCH_MODE:-GROUP}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

if [[ -d "${STUDENT_VENV}/bin" ]]; then
  export PATH="${STUDENT_VENV}/bin:${PATH}"
fi
mkdir -p "${RAY_TMPDIR}"

# ---------------------------------------------------------------------------
# 3) GPU layout
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
CRITIC_GPUS="${CRITIC_GPUS:-4}"
REF_GPUS="${REF_GPUS:-${ACTOR_GPUS}}"
REWARD_GPUS="${REWARD_GPUS:-${CRITIC_GPUS}}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
CRITIC_NUM_NODES="${CRITIC_NUM_NODES:-1}"
REF_NUM_NODES="${REF_NUM_NODES:-1}"
REWARD_NUM_NODES="${REWARD_NUM_NODES:-1}"

# ---------------------------------------------------------------------------
# 4) Training hyperparameters, explicitly listed
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
MAX_SAMPLES="${MAX_SAMPLES:--1}"

DISTRIBUTION_REWARD_TYPE="${DISTRIBUTION_REWARD_TYPE:-pointwise}"
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
CF_TARGET_MODE="${CF_TARGET_MODE:-single}"
CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.0}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-${N_SAMPLES_PER_PROMPT}}"

# Paper knobs, explicit:
#   gamma = CE_LOSS_COEF
#   alpha ~= DIVERSITY_REW_COEF / ALIGNMENT_REW_COEF
CE_LOSS_COEF="${CE_LOSS_COEF:-0.03}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"
DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-1.0}"

ACTOR_LR="${ACTOR_LR:-1e-6}"
CRITIC_LR="${CRITIC_LR:-0.0}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-0.0}"
ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-rloo}"
INIT_KL_COEF="${INIT_KL_COEF:-0.0}"
KL_ESTIMATOR="${KL_ESTIMATOR:-k2}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
ZERO_STAGE="${ZERO_STAGE:-2}"
ADAM_OFFLOAD="${ADAM_OFFLOAD:-true}"
REF_REWARD_OFFLOAD="${REF_REWARD_OFFLOAD:-true}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.03}"
CRITIC_LR_WARMUP_RATIO="${CRITIC_LR_WARMUP_RATIO:-0.0}"
GLOBAL_SEED="${GLOBAL_SEED:-43}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

ONLINE_EVAL="${ONLINE_EVAL:-false}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-1}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-${GENERATE_MAX_LEN}}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_EVEN_COUNT="${SAVE_EVEN_COUNT:-10}"

# ---------------------------------------------------------------------------
# 5) Code post-eval hyperparameters, explicitly listed
# ---------------------------------------------------------------------------
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
RUN_CODE_POST_EVAL="${RUN_CODE_POST_EVAL:-${EVAL_AFTER_TRAIN}}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-$(count_csv_items "${MODEL_CUDA_VISIBLE_DEVICES}")}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_PROGRESS_BATCH_SIZE="${VLLM_PROGRESS_BATCH_SIZE:-128}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_SEED="${VLLM_SEED:-1234}"
CODE_BENCHMARK_SCRIPT="${CODE_BENCHMARK_SCRIPT:-${REPO_ROOT}/scripts/benchmarks/run_code_generation_benchmarks.py}"
CODE_BENCHMARK_PYTHON_BIN="${CODE_BENCHMARK_PYTHON_BIN:-${TEACHER_PYTHON_BIN}}"
CODE_BENCHMARKS="${CODE_BENCHMARKS:-humaneval}"
CODE_BENCHMARK_BACKEND="${CODE_BENCHMARK_BACKEND:-vllm}"
CODE_EVAL_MAX_NEW_TOKENS="${CODE_EVAL_MAX_NEW_TOKENS:-1024}"
CODE_EVAL_TEMPERATURE="${CODE_EVAL_TEMPERATURE:-0.0}"
CODE_EVAL_TOP_P="${CODE_EVAL_TOP_P:-1.0}"
CODE_EVAL_REPETITION_PENALTY="${CODE_EVAL_REPETITION_PENALTY:-1.0}"
CODE_EVAL_TIMEOUT_SECONDS="${CODE_EVAL_TIMEOUT_SECONDS:-10}"
CODE_EVAL_N_SAMPLES="${CODE_EVAL_N_SAMPLES:-1}"
CODE_EVAL_PASSK_LIST="${CODE_EVAL_PASSK_LIST:-1}"
CODE_EVAL_GREEDY_BATCH_SIZE="${CODE_EVAL_GREEDY_BATCH_SIZE:-16}"
CODE_EVAL_SAMPLE_BATCH_SIZE="${CODE_EVAL_SAMPLE_BATCH_SIZE:-4}"
HUMANEVAL_EVAL_SPLIT="${HUMANEVAL_EVAL_SPLIT:-test}"
MBPP_EVAL_CONFIG="${MBPP_EVAL_CONFIG:-}"
MBPP_EVAL_SPLIT="${MBPP_EVAL_SPLIT:-test}"

# ---------------------------------------------------------------------------
# 6) Output
# ---------------------------------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs/diff_dataset}"
RUN_NAME="${RUN_NAME:-diff_g1_qwen35_2b_$(date +%m%d_%H%M)}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
SAVE_PATH="${RUN_DIR}/model"
TB_DIR="${RUN_DIR}/tensorboard"
SCRIPT_LOG_PATH="${SCRIPT_LOG_PATH:-${RUN_DIR}/$(basename "$0" .sh).log}"
RUN_CONTEXT_PATH="${RUN_DIR}/run_context.env"
RUN_SUMMARY_PATH="${RUN_DIR}/run_summary.txt"
mkdir -p "${RUN_DIR}" "${SAVE_PATH}" "${TB_DIR}"
exec > >(tee -a "${SCRIPT_LOG_PATH}") 2>&1

# ---------------------------------------------------------------------------
# 7) Prepare / validate
# ---------------------------------------------------------------------------
write_jsonl_datasets

gpu_count="$(count_csv_items "${CUDA_VISIBLE_DEVICES}")"
if (( ACTOR_GPUS + CRITIC_GPUS > gpu_count )); then
  echo "[ERROR] ACTOR_GPUS(${ACTOR_GPUS}) + CRITIC_GPUS(${CRITIC_GPUS}) > GPU count(${gpu_count})"
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

[[ -x "${STUDENT_PYTHON_BIN}" ]] || { echo "[ERROR] STUDENT_PYTHON_BIN not executable: ${STUDENT_PYTHON_BIN}"; exit 1; }
[[ -d "${REPO_ROOT}" ]] || { echo "[ERROR] REPO_ROOT not found: ${REPO_ROOT}"; exit 1; }
[[ -e "${MODEL_PATH}" ]] || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 1; }
[[ -e "${TRAIN_DATA}" ]] || { echo "[ERROR] TRAIN_DATA not found: ${TRAIN_DATA}"; exit 1; }
if [[ "${RUN_CODE_POST_EVAL}" == "true" ]]; then
  [[ -x "${CODE_BENCHMARK_PYTHON_BIN}" ]] || { echo "[ERROR] CODE_BENCHMARK_PYTHON_BIN not executable: ${CODE_BENCHMARK_PYTHON_BIN}"; exit 1; }
  [[ -f "${CODE_BENCHMARK_SCRIPT}" ]] || { echo "[ERROR] CODE_BENCHMARK_SCRIPT not found: ${CODE_BENCHMARK_SCRIPT}"; exit 1; }
  [[ -e "${MBPP_EVAL_DATA}" ]] || { echo "[ERROR] MBPP_EVAL_DATA not found: ${MBPP_EVAL_DATA}"; exit 1; }
  [[ -e "${HUMANEVAL_EVAL_DATA}" ]] || { echo "[ERROR] HUMANEVAL_EVAL_DATA not found: ${HUMANEVAL_EVAL_DATA}"; exit 1; }
fi

ONLINE_EVAL_ARGS=()
if [[ "${ONLINE_EVAL}" == "true" ]]; then
  [[ -e "${EVAL_DATA}" ]] || { echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"; exit 1; }
  ONLINE_EVAL_ARGS=(
    --eval_dataset "${EVAL_DATA}"
    --eval_split "${EVAL_SPLIT}"
    --eval_steps "${EVAL_STEPS}"
    --eval_max_samples "${EVAL_MAX_SAMPLES}"
    --eval_generate_max_len "${EVAL_GENERATE_MAX_LEN}"
  )
else
  ONLINE_EVAL_ARGS=(--eval_steps -1 --eval_down_steps -1)
fi

ADAM_OFFLOAD_ARGS=()
[[ "${ADAM_OFFLOAD}" == "true" ]] && ADAM_OFFLOAD_ARGS+=(--adam_offload)
[[ "${REF_REWARD_OFFLOAD}" == "true" ]] && ADAM_OFFLOAD_ARGS+=(--ref_reward_offload)

{
  echo "# Auto-generated run context snapshot"
  echo "# UTC: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  for name in \
    RUN_NAME RUN_DIR SAVE_PATH MODEL_PATH TRAIN_DATA POST_EVAL_DATASETS \
    RAY_TMPDIR TMPDIR \
    CUDA_VISIBLE_DEVICES ACTOR_GPUS CRITIC_GPUS REF_GPUS REWARD_GPUS \
    N_SAMPLES_PER_PROMPT ROLLOUT_BATCH_SIZE TRAIN_BATCH_SIZE MICRO_TRAIN_BATCH_SIZE MICRO_ROLLOUT_BATCH_SIZE MICRO_REWARD_BATCH_SIZE \
    PROMPT_MAX_LEN CONTEXT_MAX_LEN GENERATE_MAX_LEN STRIDE NUM_EPISODES MAX_EPOCHS MAX_SAMPLES \
    DISTRIBUTION_REWARD_TYPE CF_TARGET_MODE CF_TEACHER_LAMBDA CF_TEACHER_N_SAMPLES \
    CE_LOSS_COEF ALIGNMENT_REW_COEF DIVERSITY_REW_COEF \
    ACTOR_LR CRITIC_LR CRITIC_LR_HEAD ADVANTAGE_ESTIMATOR INIT_KL_COEF KL_ESTIMATOR TEMPERATURE TOP_P ZERO_STAGE \
    ADAM_OFFLOAD REF_REWARD_OFFLOAD LR_WARMUP_RATIO CRITIC_LR_WARMUP_RATIO GLOBAL_SEED LOGGING_STEPS \
    ONLINE_EVAL EVAL_STEPS EVAL_MAX_SAMPLES SAVE_STEPS SAVE_EVEN_COUNT RUN_CODE_POST_EVAL \
    CODE_BENCHMARK_SCRIPT CODE_BENCHMARK_PYTHON_BIN CODE_BENCHMARKS CODE_BENCHMARK_BACKEND \
    CODE_EVAL_MAX_NEW_TOKENS CODE_EVAL_TEMPERATURE CODE_EVAL_TOP_P CODE_EVAL_TIMEOUT_SECONDS CODE_EVAL_N_SAMPLES CODE_EVAL_PASSK_LIST; do
    printf "%s=%q\n" "${name}" "${!name-}"
  done
} > "${RUN_CONTEXT_PATH}"

{
  echo "run_name: ${RUN_NAME}"
  echo "run_dir: ${RUN_DIR}"
  echo "model_path: ${MODEL_PATH}"
  echo "train_data: ${TRAIN_DATA}"
  echo "distribution_reward_type: ${DISTRIBUTION_REWARD_TYPE}"
  echo "cf_target_mode: ${CF_TARGET_MODE}"
  echo "teacher_in_reward: false"
  echo "gamma_ce_loss_coef: ${CE_LOSS_COEF}"
  echo "alpha_diversity_over_alignment: ${DIVERSITY_REW_COEF}/${ALIGNMENT_REW_COEF}"
  echo "post_eval_datasets: ${POST_EVAL_DATASETS}"
} > "${RUN_SUMMARY_PATH}"

echo "========== Diff-Dataset G1 standalone =========="
echo "RUN_DIR:                  ${RUN_DIR}"
echo "MODEL_PATH:               ${MODEL_PATH}"
echo "TRAIN_DATA:               ${TRAIN_DATA}"
echo "GPUs:                     ${CUDA_VISIBLE_DEVICES} (count=${gpu_count})"
echo "Actor/Critic GPUs:        ${ACTOR_GPUS}/${CRITIC_GPUS}"
echo "distribution_reward_type: ${DISTRIBUTION_REWARD_TYPE}"
echo "cf_target_mode:           ${CF_TARGET_MODE}"
echo "teacher_in_reward:        false"
echo "gamma / CE coef:          ${CE_LOSS_COEF}"
echo "alpha proxy:              ${DIVERSITY_REW_COEF}/${ALIGNMENT_REW_COEF}"
echo "num_episodes/max_epochs:  ${NUM_EPISODES}/${MAX_EPOCHS}"
echo "max_samples:             ${MAX_SAMPLES} (-1 means full train split)"
echo "online_eval:              ${ONLINE_EVAL}"
echo "code_post_eval:           ${RUN_CODE_POST_EVAL}"
echo "code_benchmark_script:    ${CODE_BENCHMARK_SCRIPT}"
echo "code_benchmarks:          ${CODE_BENCHMARKS}"
echo "==============================================="

# ---------------------------------------------------------------------------
# 8) Train
# ---------------------------------------------------------------------------
ray stop --force 2>/dev/null || true
sleep 2
cd "${REPO_ROOT}"

TRAIN_RC=0
EVAL_RC=0
set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${STUDENT_PYTHON_BIN}" -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --pretrain_mode --no_chat_template \
  --disable_ds_ckpt --colocate_actor_ref --colocate_critic_reward \
  --use_kl_loss --use_whitening \
  --distribution_reward_type "${DISTRIBUTION_REWARD_TYPE}" \
  --feature_map_type "${FEATURE_MAP_TYPE}" \
  --rff_num_features "${RFF_NUM_FEATURES}" \
  --rff_sigma "${RFF_SIGMA}" \
  --rff_seed "${RFF_SEED}" \
  --cf_num_freqs "${CF_NUM_FREQS}" \
  --cf_sigma "${CF_SIGMA}" \
  --cf_seed "${CF_SEED}" \
  --cf_alpha "${CF_ALPHA}" \
  --cf_beta "${CF_BETA}" \
  --cf_reward_scale "${CF_REWARD_SCALE}" \
  --cf_target_mode "${CF_TARGET_MODE}" \
  --cf_teacher_lambda "${CF_TEACHER_LAMBDA}" \
  --cf_teacher_n_samples "${CF_TEACHER_N_SAMPLES}" \
  --ce_loss_coef "${CE_LOSS_COEF}" \
  --alignment_rew_coef "${ALIGNMENT_REW_COEF}" \
  --diversity_rew_coef "${DIVERSITY_REW_COEF}" \
  --embed_method last_token \
  --critic_sequence_level last_token \
  --critic_learning_rate "${CRITIC_LR}" \
  --critic_lr_head "${CRITIC_LR_HEAD}" \
  --actor_learning_rate "${ACTOR_LR}" \
  --pretrain "${MODEL_PATH}" \
  --critic_pretrain "${MODEL_PATH}" \
  --prompt_data "${TRAIN_DATA}" \
  --input_key question \
  --label_key answer \
  --output_key answer \
  --prompt_split "${PROMPT_SPLIT}" \
  --prompt_max_len "${PROMPT_MAX_LEN}" \
  --context_max_len "${CONTEXT_MAX_LEN}" \
  --generate_max_len "${GENERATE_MAX_LEN}" \
  --stride "${STRIDE}" \
  --n_samples_per_prompt "${N_SAMPLES_PER_PROMPT}" \
  --rollout_batch_size "${ROLLOUT_BATCH_SIZE}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --micro_train_batch_size "${MICRO_TRAIN_BATCH_SIZE}" \
  --micro_rollout_batch_size "${MICRO_ROLLOUT_BATCH_SIZE}" \
  --micro_reward_batch_size "${MICRO_REWARD_BATCH_SIZE}" \
  --max_samples "${MAX_SAMPLES}" \
  --num_episodes "${NUM_EPISODES}" \
  --max_epochs "${MAX_EPOCHS}" \
  --actor_num_nodes "${ACTOR_NUM_NODES}" \
  --actor_num_gpus_per_node "${ACTOR_GPUS}" \
  --critic_num_nodes "${CRITIC_NUM_NODES}" \
  --critic_num_gpus_per_node "${CRITIC_GPUS}" \
  --ref_num_nodes "${REF_NUM_NODES}" \
  --ref_num_gpus_per_node "${REF_GPUS}" \
  --reward_num_nodes "${REWARD_NUM_NODES}" \
  --reward_num_gpus_per_node "${REWARD_GPUS}" \
  --advantage_estimator "${ADVANTAGE_ESTIMATOR}" \
  --init_kl_coef "${INIT_KL_COEF}" \
  --kl_estimator "${KL_ESTIMATOR}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --zero_stage "${ZERO_STAGE}" \
  "${ADAM_OFFLOAD_ARGS[@]}" \
  --lr_warmup_ratio "${LR_WARMUP_RATIO}" \
  --critic_lr_warmup_ratio "${CRITIC_LR_WARMUP_RATIO}" \
  --seed "${GLOBAL_SEED}" \
  "${ONLINE_EVAL_ARGS[@]}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_even_count "${SAVE_EVEN_COUNT}" \
  --save_hf_ckpt \
  --use_tensorboard "${TB_DIR}" \
  --save_path "${SAVE_PATH}" \
  --ckpt_path "${SAVE_PATH}/ckpt" \
  --wandb_run_name "${RUN_NAME}" \
  2>&1 | tee "${RUN_DIR}/train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e

ray stop --force 2>/dev/null || true

if (( TRAIN_RC != 0 )); then
  echo "[ERROR] training failed with exit code ${TRAIN_RC}"
fi

# ---------------------------------------------------------------------------
# 9) Code post-eval
# ---------------------------------------------------------------------------
if [[ "${RUN_CODE_POST_EVAL}" == "true" && "${TRAIN_RC}" -eq 0 ]]; then
  set +e
  run_code_benchmark_posteval
  EVAL_RC=$?
  set -e
elif [[ "${RUN_CODE_POST_EVAL}" == "true" ]]; then
  echo "[post-eval] skipped because training failed."
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

echo "Diff-dataset G1 completed at $(date) rc=${FINAL_RC}" > "${RUN_DIR}/status.txt"
echo "[done] logs: ${RUN_DIR}"
exit "${FINAL_RC}"
