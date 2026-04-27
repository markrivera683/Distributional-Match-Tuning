#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  G1 Rebase — strict ablation baseline (degraded G2)            ║
# ║  pointwise reward · frozen critic · NO teacher · 16k/32k eval  ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# CONTROLLED VARIABLES vs G2 (run_G2_rebase.sh):
#   G1: distribution_reward_type = pointwise,  cf_target_mode = single,  no teacher
#   G2: distribution_reward_type = cf_l1oo,    cf_target_mode = teacher, online teacher
#
# CONTROLLED VARIABLES vs G3 (run_G3_rebase.sh):
#   G3 adds:  enable_ema, feature_adapter, trainable critic head, direct discrepancy
#
# Everything else (model, data, batch, GPU count, optimizer, seed) is IDENTICAL to G2.
#
# Usage:  bash scripts/run_G1_rebase.sh
# Override any variable via env, e.g.:
#   TARGET_STEPS=500 bash scripts/run_G1_rebase.sh
#   MAX_SAMPLES=10000 bash scripts/run_G1_rebase.sh
set -euo pipefail

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "$csv" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"$csv"
}

# ====================================================================
# 0) GPU ASSIGNMENT — same actor/critic count as G2/G3 student side
#    G1 has no teacher, but we use 4 GPUs to match G2/G3 student budget
# ====================================================================
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
CRITIC_GPUS="${CRITIC_GPUS:-4}"
REF_GPUS="${ACTOR_GPUS}"
REWARD_GPUS="${CRITIC_GPUS}"

# ====================================================================
# 1) TRAINING DATA / MODEL PATHS
# ====================================================================
REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Match-Tuning}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/gemma-4-E4B/}"
DEFAULT_TRAIN_DATA="/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
FALLBACK_LOCAL_DATA="${FALLBACK_LOCAL_DATA:-}"
TRAIN_DATA="${TRAIN_DATA:-${DEFAULT_TRAIN_DATA}}"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"
PROMPT_SPLIT="${PROMPT_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"
STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"

# ====================================================================
# 2) TRAINING KNOBS — IDENTICAL to G2
# ====================================================================
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE))}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-4}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-4}"
MICRO_REWARD_BATCH_SIZE="${MICRO_REWARD_BATCH_SIZE:-4}"

PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-512}"
CONTEXT_MAX_LEN="${CONTEXT_MAX_LEN:-8}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-8}"
STRIDE="${STRIDE:-8}"

NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
TARGET_STEPS="${TARGET_STEPS:-500}"
# Trainer computes:
#   max_steps = len(prompts_dataset) * n_samples_per_prompt / train_batch_size * num_episodes * max_epochs
# So we back-solve max_samples to hit TARGET_STEPS by default.
DEFAULT_MAX_SAMPLES="$((TARGET_STEPS * TRAIN_BATCH_SIZE / N_SAMPLES_PER_PROMPT / NUM_EPISODES / MAX_EPOCHS))"
MAX_SAMPLES="${MAX_SAMPLES:-${DEFAULT_MAX_SAMPLES}}"

# ====================================================================
# 3) EVAL / CHECKPOINT
# ====================================================================
# Online (in-training) eval — runs at step 0 + every EVAL_STEPS during training.
# Default OFF: in-training eval shares the critic actor group, and the launcher
# requires len(eval_micro_batches) >= effective_critic_actors. Easy to misconfigure
# and most users only care about the post-training two-round eval. To enable,
# pass: ONLINE_EVAL=true bash scripts/run_G1_rebase.sh
ONLINE_EVAL="${ONLINE_EVAL:-false}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-1}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-${GENERATE_MAX_LEN}}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_EVEN_COUNT="${SAVE_EVEN_COUNT:-10}"

# Post-training offline two-round eval (16k → 32k). Independent from ONLINE_EVAL.
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
RUN_TWO_ROUND_EVAL="${RUN_TWO_ROUND_EVAL:-${EVAL_AFTER_TRAIN}}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
FIRST_PASS_MAX_NEW_TOKENS="${FIRST_PASS_MAX_NEW_TOKENS:-16384}"
SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS:-32768}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY:-1.0}"
POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N:-1}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_SEED="${VLLM_SEED:-1234}"
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"

# ====================================================================
# 4) ENV / RUN DIR
# ====================================================================
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.6,roundup_power2_divisions:8}"
export PYTHONUNBUFFERED=1


# temp
export CUDA_LAUNCH_BLOCKING=1          # 让 cuda 错就地报告，不延迟到下一次同步
export TORCH_USE_CUDA_DSA=1            # 装备 device-side assertion
export NCCL_DEBUG=WARN                 # 暴露 NCCL silent fail
export NCCL_LAUNCH_MODE=GROUP          # NCCL collective 顺序更稳
export RAY_DEDUP_LOGS=0                # 拿到每个 worker 完整 stderr
export CUDA_DEVICE_MAX_CONNECTIONS=1   # 减少 stream 数，降 driver 压力（R470 兼容性）


RUN_NAME="${RUN_NAME:-g1_rebase_$(date +%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SAVE_PATH="${RUN_DIR}/model"
TB_DIR="${RUN_DIR}/tensorboard"
mkdir -p "$RUN_DIR" "$SAVE_PATH" "$TB_DIR"

# ====================================================================
# 5) SANITY CHECK
# ====================================================================
gpu_count="$(count_csv_items "$CUDA_VISIBLE_DEVICES")"
vllm_gpu_count="$(count_csv_items "${MODEL_CUDA_VISIBLE_DEVICES}")"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-${vllm_gpu_count}}"

STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"
if [[ ! -x "${STUDENT_PYTHON_BIN}" ]]; then
  echo "[ERROR] STUDENT_PYTHON_BIN not executable: ${STUDENT_PYTHON_BIN}"
  echo "        expected student env: ${STUDENT_VENV}"
  exit 1
fi

if (( ACTOR_GPUS + CRITIC_GPUS > gpu_count )); then
  echo "[ERROR] ACTOR_GPUS(${ACTOR_GPUS}) + CRITIC_GPUS(${CRITIC_GPUS}) > GPU count(${gpu_count})"
  exit 1
fi

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
  echo "        got MICRO_TRAIN_BATCH_SIZE=${MICRO_TRAIN_BATCH_SIZE}, N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}"
  exit 1
fi

if (( MICRO_ROLLOUT_BATCH_SIZE % N_SAMPLES_PER_PROMPT != 0 )); then
  echo "[ERROR] MICRO_ROLLOUT_BATCH_SIZE must be divisible by N_SAMPLES_PER_PROMPT"
  exit 1
fi

if [[ "${TRAIN_DATA}" == "${DEFAULT_TRAIN_DATA}" && -n "${FALLBACK_LOCAL_DATA}" && ! -e "${TRAIN_DATA}" && -f "${FALLBACK_LOCAL_DATA}" ]]; then
  echo "[WARN] TRAIN_DATA default not found, fallback to ${FALLBACK_LOCAL_DATA}"
  TRAIN_DATA="${FALLBACK_LOCAL_DATA}"
fi
if [[ "${EVAL_DATA}" == "${DEFAULT_EVAL_DATA}" && -n "${FALLBACK_LOCAL_DATA}" && ! -e "${EVAL_DATA}" && -f "${FALLBACK_LOCAL_DATA}" ]]; then
  echo "[WARN] EVAL_DATA default not found, fallback to ${FALLBACK_LOCAL_DATA}"
  EVAL_DATA="${FALLBACK_LOCAL_DATA}"
fi
if [[ "${TRAIN_DATA}" == /* && ! -e "${TRAIN_DATA}" ]]; then
  echo "[ERROR] TRAIN_DATA not found: ${TRAIN_DATA}"
  exit 1
fi
# EVAL_DATA only required when online eval is on (post-training two-round eval
# uses its own dataset path inside scripts/supplement_2rounds/G1.sh).
if [[ "${ONLINE_EVAL}" == "true" && "${EVAL_DATA}" == /* && ! -e "${EVAL_DATA}" ]]; then
  echo "[ERROR] EVAL_DATA not found (required when ONLINE_EVAL=true): ${EVAL_DATA}"
  exit 1
fi

# --------------------------------------------------------------------
# Build ONLINE_EVAL_ARGS based on the switch.
# When ONLINE_EVAL=false we tell the trainer to skip both the step-0 initial
# eval and the periodic eval by passing eval_steps=-1 + eval_down_steps=-1.
# We also drop --eval_dataset so the trainer doesn't waste a GPU on an unused
# eval dataloader (see openrlhf/cli/train_ebft_ray.py: trainer_needs_gpu).
# --------------------------------------------------------------------
if [[ "${ONLINE_EVAL}" == "true" ]]; then
  # Sanity: critic dispatcher requires len(eval_micro_batches) >= effective_critic_actors.
  # eval_micro_batches = ceil(EVAL_MAX_SAMPLES * N_SAMPLES_PER_PROMPT / MICRO_ROLLOUT_BATCH_SIZE)
  EVAL_TOTAL_SAMPLES=$(( EVAL_MAX_SAMPLES * N_SAMPLES_PER_PROMPT ))
  EVAL_MICRO_BATCHES=$(( (EVAL_TOTAL_SAMPLES + MICRO_ROLLOUT_BATCH_SIZE - 1) / MICRO_ROLLOUT_BATCH_SIZE ))
  if (( EVAL_MICRO_BATCHES < CRITIC_GPUS )); then
    MIN_EVAL_MAX_SAMPLES=$(( (CRITIC_GPUS * MICRO_ROLLOUT_BATCH_SIZE + N_SAMPLES_PER_PROMPT - 1) / N_SAMPLES_PER_PROMPT ))
    echo "[ERROR] ONLINE_EVAL=true but EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES} produces only"
    echo "        ${EVAL_MICRO_BATCHES} eval micro-batch(es), need >= ${CRITIC_GPUS} (CRITIC_GPUS)."
    echo "        Set EVAL_MAX_SAMPLES >= ${MIN_EVAL_MAX_SAMPLES}, or decrease CRITIC_GPUS,"
    echo "        or run with ONLINE_EVAL=false (default)."
    exit 1
  fi
  ONLINE_EVAL_ARGS=(
    --eval_dataset "${EVAL_DATA}"
    --eval_split "${EVAL_SPLIT}"
    --eval_steps "${EVAL_STEPS}"
    --eval_max_samples "${EVAL_MAX_SAMPLES}"
    --eval_generate_max_len "${EVAL_GENERATE_MAX_LEN}"
  )
else
  ONLINE_EVAL_ARGS=(
    --eval_steps -1
    --eval_down_steps -1
  )
fi

echo "========== G1 Rebase (no teacher, pointwise) =========="
echo "RUN_DIR:              ${RUN_DIR}"
echo "GPUs:                 ${CUDA_VISIBLE_DEVICES} (count=${gpu_count})"
echo "Actor/Critic GPUs:    ${ACTOR_GPUS}/${CRITIC_GPUS}"
echo "Model:                ${MODEL_PATH}"
echo "Train data:           ${TRAIN_DATA}"
if [[ "${ONLINE_EVAL}" == "true" ]]; then
  echo "Eval data:            ${EVAL_DATA}"
  echo "Prompt/Eval split:    ${PROMPT_SPLIT}/${EVAL_SPLIT}"
else
  echo "Eval data:            (online eval disabled)"
  echo "Prompt split:         ${PROMPT_SPLIT}"
fi
echo "Student python:       ${STUDENT_PYTHON_BIN}"
echo "distribution_reward:  pointwise"
echo "cf_target_mode:       single"
echo "target_steps:         ${TARGET_STEPS}"
echo "max_samples:          ${MAX_SAMPLES}"
if [[ "${ONLINE_EVAL}" == "true" ]]; then
  echo "online_eval:          true (every ${EVAL_STEPS} steps, max_samples=${EVAL_MAX_SAMPLES})"
else
  echo "online_eval:          false  (in-training eval disabled)"
fi
echo "save_steps:           ${SAVE_STEPS}"
echo "run_two_round_eval:   ${RUN_TWO_ROUND_EVAL}"
echo "======================================================="

# ====================================================================
# 6) TRAIN
# ====================================================================
ray stop --force 2>/dev/null || true
sleep 2
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${STUDENT_PYTHON_BIN}" -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --pretrain_mode --no_chat_template \
  --disable_ds_ckpt --colocate_actor_ref --colocate_critic_reward \
  --use_kl_loss --use_whitening \
  --distribution_reward_type pointwise \
  --feature_map_type identity --rff_num_features 128 --rff_sigma 1.0 --rff_seed 43 \
  --cf_num_freqs 128 --cf_sigma 1.0 --cf_seed 43 --cf_alpha 0.5 --cf_beta 0.5 --cf_reward_scale 1.0 \
  --cf_target_mode single --cf_teacher_lambda 0.0 --cf_teacher_n_samples "${N_SAMPLES_PER_PROMPT}" \
  \
  --embed_method last_token --critic_sequence_level last_token \
  --critic_learning_rate 0.0 --critic_lr_head 0.0 \
  --pretrain "${MODEL_PATH}" --critic_pretrain "${MODEL_PATH}" \
  --prompt_data "${TRAIN_DATA}" \
  --input_key question --label_key answer --output_key answer \
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
  \
  --actor_num_nodes 1 --actor_num_gpus_per_node "${ACTOR_GPUS}" \
  --critic_num_nodes 1 --critic_num_gpus_per_node "${CRITIC_GPUS}" \
  --ref_num_nodes 1 --ref_num_gpus_per_node "${REF_GPUS}" \
  --reward_num_nodes 1 --reward_num_gpus_per_node "${REWARD_GPUS}" \
  \
  --advantage_estimator rloo --init_kl_coef 0.0 --kl_estimator k2 \
  --temperature 0.6 --top_p 1.0 --actor_learning_rate 1e-6 \
  --zero_stage 1 --lr_warmup_ratio 0.03 --critic_lr_warmup_ratio 0.0 \
  --seed 43 \
  "${ONLINE_EVAL_ARGS[@]}" \
  --logging_steps 10 \
  --save_steps "${SAVE_STEPS}" --save_even_count "${SAVE_EVEN_COUNT}" --save_hf_ckpt \
  --use_tensorboard "${TB_DIR}" \
  --save_path "${SAVE_PATH}" --ckpt_path "${SAVE_PATH}/ckpt" \
  --wandb_run_name "${RUN_NAME}" \
  2>&1 | tee "${RUN_DIR}/train.log"

# ====================================================================
# 7) TWO-ROUND COMPLETION EVAL
# ====================================================================
ray stop --force 2>/dev/null || true

echo ""
echo "──────────────────────────────────────────────────"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')  TRAINING FINISHED"
echo "  Logs:        ${RUN_DIR}/train.log"
echo "  TensorBoard: ${TB_DIR}"
echo "  Checkpoints: ${SAVE_PATH}"

if [[ "${RUN_TWO_ROUND_EVAL}" == "true" ]]; then
  echo ""
  echo "===== Running two-round 16k/32k completion eval ====="
  RUN_DIR="${RUN_DIR}" \
  MODEL_PATH="${SAVE_PATH}" \
  MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES}" \
  VLLM_TP_SIZE="${VLLM_TP_SIZE}" \
  POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES}" \
  POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN}" \
  FIRST_PASS_MAX_NEW_TOKENS="${FIRST_PASS_MAX_NEW_TOKENS}" \
  SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS}" \
  POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE}" \
  POST_EVAL_TOP_P="${POST_EVAL_TOP_P}" \
  POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY}" \
  POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N}" \
  VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
  VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING}" \
  VLLM_SEED="${VLLM_SEED}" \
  bash "${REPO_ROOT}/scripts/supplement_2rounds/G1.sh"
fi

echo "──────────────────────────────────────────────────"
echo "G1 rebase run completed at $(date)" > "${RUN_DIR}/status.txt"
echo "[done] logs: ${RUN_DIR}"
