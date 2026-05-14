#!/usr/bin/env bash
# 1-GPU smoke test for G1 rebase (run_G1_rebase.sh): pointwise / single GT, no teacher.
# Tiny batches + --colocate_all_models. Mirrors G2/G3 smoketests except no teacher path.
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ACTOR_GPUS="${ACTOR_GPUS:-1}"
CRITIC_GPUS="${CRITIC_GPUS:-1}"
REF_GPUS="${REF_GPUS:-1}"
REWARD_GPUS="${REWARD_GPUS:-1}"

REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Match-Tuning}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/teacher_model/models/Qwen3.5-0.8B}"
DEFAULT_TRAIN_DATA="/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
TRAIN_DATA="${TRAIN_DATA:-${DEFAULT_TRAIN_DATA}}"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"
STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"

N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE))}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-${N_SAMPLES_PER_PROMPT}}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-2}"
MICRO_REWARD_BATCH_SIZE="${MICRO_REWARD_BATCH_SIZE:-1}"

PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-64}"
CONTEXT_MAX_LEN="${CONTEXT_MAX_LEN:-8}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-8}"
STRIDE="${STRIDE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"

# Align with run_G1_rebase.sh (scaled down for smoke)
EVAL_STEPS="${EVAL_STEPS:-1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-2}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-${GENERATE_MAX_LEN}}"
SAVE_STEPS="${SAVE_STEPS:-1}"
SAVE_EVEN_COUNT="${SAVE_EVEN_COUNT:-2}"

EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
POST_EVAL_NPROC="${POST_EVAL_NPROC:-1}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-4}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-64}"
POST_EVAL_MAX_NEW_TOKENS="${POST_EVAL_MAX_NEW_TOKENS:-256}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_MICRO_BATCH_SIZE="${POST_EVAL_MICRO_BATCH_SIZE:-2}"
POST_EVAL_MASTER_PORT="${POST_EVAL_MASTER_PORT:-29511}"

ENABLE_FLASH_ATTN="${ENABLE_FLASH_ATTN:-false}"

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

STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"
if [[ ! -x "${STUDENT_PYTHON_BIN}" ]]; then
  echo "[ERROR] STUDENT_PYTHON_BIN not executable: ${STUDENT_PYTHON_BIN}"
  echo "        expected student env: ${STUDENT_VENV}"
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi
if [[ "${TRAIN_DATA}" == "${DEFAULT_TRAIN_DATA}" && ! -e "${TRAIN_DATA}" && -f "${FALLBACK_LOCAL_DATA}" ]]; then
  echo "[WARN] TRAIN_DATA default not found, fallback to ${FALLBACK_LOCAL_DATA}"
  TRAIN_DATA="${FALLBACK_LOCAL_DATA}"
fi
if [[ "${EVAL_DATA}" == "${DEFAULT_EVAL_DATA}" && ! -e "${EVAL_DATA}" && -f "${FALLBACK_LOCAL_DATA}" ]]; then
  echo "[WARN] EVAL_DATA default not found, fallback to ${FALLBACK_LOCAL_DATA}"
  EVAL_DATA="${FALLBACK_LOCAL_DATA}"
fi
if [[ "${TRAIN_DATA}" == /* && ! -e "${TRAIN_DATA}" ]]; then
  echo "[ERROR] TRAIN_DATA not found: ${TRAIN_DATA}"
  exit 1
fi
if [[ "${EVAL_DATA}" == /* && ! -e "${EVAL_DATA}" ]]; then
  echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"
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

RUN_NAME="${RUN_NAME:-smoke_g1_rebase_1gpu_$(date +%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs/smoketest_1gpu}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SAVE_PATH="${RUN_DIR}/model"
TB_DIR="${RUN_DIR}/tensorboard"
POST_EVAL_OUTPUT_PATH="${POST_EVAL_OUTPUT_PATH:-${RUN_DIR}/eval_results.jsonl}"
POST_EVAL_LOG_PATH="${POST_EVAL_LOG_PATH:-${RUN_DIR}/eval.log}"
mkdir -p "${RUN_DIR}" "${SAVE_PATH}" "${TB_DIR}"

echo "========== G1 rebase 1GPU SMOKETEST =========="
echo "RUN_DIR:         ${RUN_DIR}"
echo "GPU(s):          ${CUDA_VISIBLE_DEVICES}"
echo "Model path:      ${MODEL_PATH}"
echo "Train data:      ${TRAIN_DATA}"
echo "Eval data:       ${EVAL_DATA}"
echo "Student python:  ${STUDENT_PYTHON_BIN}"
echo "reward:          pointwise / single (no teacher)"
echo "FlashAttention:  ${ENABLE_FLASH_ATTN}"
echo "============================================="

FLASH_ATTN_FLAGS=()
if [[ "${ENABLE_FLASH_ATTN}" == "true" ]]; then
  FLASH_ATTN_FLAGS=(--flash_attn)
fi

ray stop --force 2>/dev/null || true
sleep 2
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${STUDENT_PYTHON_BIN}" -m openrlhf.cli.train_ebft_ray \
  --bf16 "${FLASH_ATTN_FLAGS[@]}" --pretrain_mode --no_chat_template \
  --disable_ds_ckpt --colocate_all_models \
  --gradient_checkpointing --use_kl_loss --use_whitening \
  --distribution_reward_type pointwise \
  --feature_map_type identity --rff_num_features 64 --rff_sigma 1.0 --rff_seed 43 \
  --cf_num_freqs 32 --cf_sigma 1.0 --cf_seed 43 --cf_alpha 0.5 --cf_beta 0.5 --cf_reward_scale 1.0 \
  --cf_target_mode single --cf_teacher_lambda 0.0 --cf_teacher_n_samples "${N_SAMPLES_PER_PROMPT}" \
  \
  --embed_method last_token --critic_sequence_level last_token \
  --critic_learning_rate 0.0 --critic_lr_head 0.0 \
  --pretrain "${MODEL_PATH}" --critic_pretrain "${MODEL_PATH}" \
  --prompt_data "${TRAIN_DATA}" --eval_dataset "${EVAL_DATA}" \
  --input_key question --label_key answer --output_key answer \
  --prompt_split train --eval_split test \
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
  \
  --actor_num_nodes 1 --actor_num_gpus_per_node "${ACTOR_GPUS}" \
  --critic_num_nodes 1 --critic_num_gpus_per_node "${CRITIC_GPUS}" \
  --ref_num_nodes 1 --ref_num_gpus_per_node "${REF_GPUS}" \
  --reward_num_nodes 1 --reward_num_gpus_per_node "${REWARD_GPUS}" \
  \
  --advantage_estimator rloo --init_kl_coef 0.0 --kl_estimator k2 \
  --temperature 0.6 --top_p 1.0 --actor_learning_rate 1e-6 \
  --zero_stage 2 --lr_warmup_ratio 0.03 --critic_lr_warmup_ratio 0.0 \
  --seed 43 \
  --eval_steps "${EVAL_STEPS}" \
  --eval_max_samples "${EVAL_MAX_SAMPLES}" \
  --eval_generate_max_len "${EVAL_GENERATE_MAX_LEN}" \
  --logging_steps 1 \
  --save_steps "${SAVE_STEPS}" --save_even_count "${SAVE_EVEN_COUNT}" --save_hf_ckpt \
  --use_tensorboard "${TB_DIR}" \
  --save_path "${SAVE_PATH}" --ckpt_path "${SAVE_PATH}/ckpt" \
  --wandb_run_name "${RUN_NAME}" \
  2>&1 | tee "${RUN_DIR}/train.log"

ray stop --force 2>/dev/null || true

echo ""
echo "──────────────────────────────────────────────────"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')  TRAINING FINISHED"
echo "  Logs:        ${RUN_DIR}/train.log"
echo "  Checkpoints: ${SAVE_PATH}"

if [[ "${EVAL_AFTER_TRAIN}" == "true" ]]; then
  echo ""
  echo "[post-eval] Running generation eval on ${EVAL_DATA} ..."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${STUDENT_PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node "${POST_EVAL_NPROC}" --master_port "${POST_EVAL_MASTER_PORT}" \
    -m openrlhf.cli.batch_inference \
    --eval_task generate \
    --pretrain "${SAVE_PATH}" \
    --dataset "${EVAL_DATA}" \
    --input_key question \
    --output_path "${POST_EVAL_OUTPUT_PATH}" \
    --prompt_max_len "${POST_EVAL_PROMPT_MAX_LEN}" \
    --max_new_tokens "${POST_EVAL_MAX_NEW_TOKENS}" \
    --temperature "${POST_EVAL_TEMPERATURE}" \
    --top_p "${POST_EVAL_TOP_P}" \
    --max_samples "${POST_EVAL_MAX_SAMPLES}" \
    --micro_batch_size "${POST_EVAL_MICRO_BATCH_SIZE}" \
    --bf16 \
    2>&1 | tee "${POST_EVAL_LOG_PATH}"
  echo "[post-eval] Saved: ${POST_EVAL_OUTPUT_PATH}"
  echo "[post-eval] Log:   ${POST_EVAL_LOG_PATH}"

  echo ""
  echo "[analysis] Running eval analysis ..."
  "${STUDENT_PYTHON_BIN}" "${REPO_ROOT}/scripts/analyze_eval_results.py" \
    --eval_results "${POST_EVAL_OUTPUT_PATH}" \
    --eval_dataset "${EVAL_DATA}" \
    --input_key question --label_key answer \
    --report_path "${RUN_DIR}/eval_analysis.json" \
    2>&1 | tee "${RUN_DIR}/eval_analysis.log"
fi

echo "──────────────────────────────────────────────────"
echo "G1 rebase 1GPU smoke test completed at $(date)" > "${RUN_DIR}/status.txt"
echo "[done] logs: ${RUN_DIR}"
