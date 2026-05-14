#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  G3 Rebase — no-teacher vicinal distribution                    ║
# ║  cf_l1oo reward · EMA/adapter kept · reuse all 8 GPUs         ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# CONTROLLED VARIABLES vs scripts/run_G3_rebase.sh:
#   - keep G3 EMA / feature-adapter / critic-head training stack
#   - switch cf_target_mode: teacher -> vicinal
#   - remove all teacher serving / remote target dependencies
#   - reuse all visible GPUs for student training by default
#
# CONTROLLED VARIABLES vs scripts/run_G2_rebase_no_teacher_vicinal.sh:
#   - add G3's trainable critic-head / direct discrepancy / EMA path
#   - keep the same no-teacher vicinal target
#
# Usage:
#   bash scripts/run_G3_rebase_no_teacher_vicinal.sh
# Override any variable via env, e.g.:
#   TARGET_STEPS=500 bash scripts/run_G3_rebase_no_teacher_vicinal.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ACTOR_GPUS=2 CRITIC_GPUS=2 \
#     bash scripts/run_G3_rebase_no_teacher_vicinal.sh
set -euo pipefail

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "${csv}" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"${csv}"
}

# ====================================================================
# 0) GPU ASSIGNMENT — teacher removed, reuse all visible GPUs
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
MODEL_PATH="${MODEL_PATH:-/mnt/data/teacher_model/models/Qwen3.5-0.8B}"
DEFAULT_TRAIN_DATA="/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
FALLBACK_LOCAL_DATA="${FALLBACK_LOCAL_DATA:-}"
TRAIN_DATA="${TRAIN_DATA:-${DEFAULT_TRAIN_DATA}}"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"
STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"

# ====================================================================
# 2) TRAINING KNOBS — G3 stack, no-teacher vicinal target
# ====================================================================
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
CF_TARGET_NUM_REFS="${CF_TARGET_NUM_REFS:-8}"
CF_TARGET_STD="${CF_TARGET_STD:-0.05}"
CF_TARGET_SEED="${CF_TARGET_SEED:-43}"
CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.0}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-${N_SAMPLES_PER_PROMPT}}"

# ── G3-specific: feature adapter + EMA ──────────────────────────────
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
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
GLOBAL_SEED="${GLOBAL_SEED:-43}"

# ====================================================================
# 3) EVAL / CHECKPOINT
# ====================================================================
EVAL_STEPS="${EVAL_STEPS:-100}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-50}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-${GENERATE_MAX_LEN}}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_EVEN_COUNT="${SAVE_EVEN_COUNT:-10}"

EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
POST_EVAL_MAX_NEW_TOKENS="${POST_EVAL_MAX_NEW_TOKENS:-8192}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_MICRO_BATCH_SIZE="${POST_EVAL_MICRO_BATCH_SIZE:-64}"
POST_EVAL_MASTER_PORT="${POST_EVAL_MASTER_PORT:-29501}"

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

RUN_NAME="${RUN_NAME:-g3_no_teacher_vicinal_8gpu_$(date +%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SAVE_PATH="${RUN_DIR}/model"
TB_DIR="${RUN_DIR}/tensorboard"
POST_EVAL_OUTPUT_PATH="${POST_EVAL_OUTPUT_PATH:-${RUN_DIR}/eval_results.jsonl}"
POST_EVAL_LOG_PATH="${POST_EVAL_LOG_PATH:-${RUN_DIR}/eval.log}"
mkdir -p "${RUN_DIR}" "${SAVE_PATH}" "${TB_DIR}"

# ====================================================================
# 5) SANITY CHECK
# ====================================================================
gpu_count="$(count_csv_items "${CUDA_VISIBLE_DEVICES}")"
POST_EVAL_NPROC="${POST_EVAL_NPROC:-${gpu_count}}"

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

if (( POST_EVAL_NPROC > gpu_count )); then
  echo "[ERROR] POST_EVAL_NPROC(${POST_EVAL_NPROC}) > GPU count(${gpu_count})"
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
if [[ "${EVAL_DATA}" == /* && ! -e "${EVAL_DATA}" ]]; then
  echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"
  exit 1
fi

echo "========== AOPS G3 no-teacher vicinal run =========="
echo "RUN_DIR:                    ${RUN_DIR}"
echo "GPUs:                       ${CUDA_VISIBLE_DEVICES} (count=${gpu_count})"
echo "Actor/Critic GPUs:          ${ACTOR_GPUS}/${CRITIC_GPUS}"
echo "Ref/Reward GPUs (colocate): ${REF_GPUS}/${REWARD_GPUS}"
echo "Model:                      ${MODEL_PATH}"
echo "Train data:                 ${TRAIN_DATA}"
echo "Eval data:                  ${EVAL_DATA}"
echo "Student python:             ${STUDENT_PYTHON_BIN}"
echo "distribution_reward:        cf_l1oo"
echo "cf_target_mode:             vicinal"
echo "cf_target_num_refs:         ${CF_TARGET_NUM_REFS}"
echo "cf_target_std:              ${CF_TARGET_STD}"
echo "cf_target_seed:             ${CF_TARGET_SEED}"
echo "teacher_in_reward:          false"
echo ""
echo "[G3 specific]"
echo "  enable_ema:               true"
echo "  ema_beta:                 ${EMA_BETA}"
echo "  feature_adapter:          rank=${FEATURE_ADAPTER_RANK} dropout=${FEATURE_ADAPTER_DROPOUT}"
echo "  unfreeze_layers:          ${UNFREEZE_LAYERS}"
echo "  critic_lr (backbone):     ${CRITIC_LR}"
echo "  critic_lr_head:           ${CRITIC_LR_HEAD}"
echo "  discrepancy_coef:         ${CRITIC_DIRECT_DISCREPANCY_COEF}"
echo "  discrepancy_target:       ${CRITIC_DIRECT_DISCREPANCY_TARGET}"
echo "  target_steps:             ${TARGET_STEPS}"
echo "  max_samples:              ${MAX_SAMPLES}"
echo "  post_eval_nproc:          ${POST_EVAL_NPROC}"
echo "===================================================="

# ====================================================================
# 6) TRAIN
# ====================================================================
ray stop --force 2>/dev/null || true
sleep 2
cd "${REPO_ROOT}"

train_cmd=(
  "${STUDENT_PYTHON_BIN}" -m openrlhf.cli.train_ebft_ray
  --bf16
  --flash_attn
  --pretrain_mode
  --no_chat_template
  --disable_ds_ckpt
  --colocate_actor_ref
  --colocate_critic_reward
  --gradient_checkpointing
  --use_kl_loss
  --use_whitening
  --enable_ema
  --feature_adapter_enable
  --feature_adapter_type residual_bottleneck
  --feature_adapter_rank "${FEATURE_ADAPTER_RANK}"
  --feature_adapter_dropout "${FEATURE_ADAPTER_DROPOUT}"
  --feature_adapter_unfreeze_layers "${UNFREEZE_LAYERS}"
  --distribution_reward_type cf_l1oo
  --feature_map_type "${FEATURE_MAP_TYPE}"
  --rff_num_features "${RFF_NUM_FEATURES}"
  --rff_sigma "${RFF_SIGMA}"
  --rff_seed "${RFF_SEED}"
  --cf_num_freqs "${CF_NUM_FREQS}"
  --cf_sigma "${CF_SIGMA}"
  --cf_seed "${CF_SEED}"
  --cf_alpha "${CF_ALPHA}"
  --cf_beta "${CF_BETA}"
  --cf_reward_scale "${CF_REWARD_SCALE}"
  --cf_target_mode vicinal
  --cf_target_num_refs "${CF_TARGET_NUM_REFS}"
  --cf_target_std "${CF_TARGET_STD}"
  --cf_target_seed "${CF_TARGET_SEED}"
  --cf_teacher_lambda "${CF_TEACHER_LAMBDA}"
  --cf_teacher_n_samples "${CF_TEACHER_N_SAMPLES}"
  --embed_method last_token
  --critic_sequence_level last_token
  --critic_learning_rate "${CRITIC_LR}"
  --critic_lr_head "${CRITIC_LR_HEAD}"
  --critic_classifier_loss_coef "${CRITIC_CLASSIFIER_LOSS_COEF}"
  --critic_direct_discrepancy_coef "${CRITIC_DIRECT_DISCREPANCY_COEF}"
  --critic_direct_discrepancy_target "${CRITIC_DIRECT_DISCREPANCY_TARGET}"
  --ema_beta "${EMA_BETA}"
  --ce_loss_coef "${CE_LOSS_COEF}"
  --diversity_rew_coef "${DIVERSITY_REW_COEF}"
  --alignment_rew_coef "${ALIGNMENT_REW_COEF}"
  --actor_learning_rate "${ACTOR_LR}"
  --pretrain "${MODEL_PATH}"
  --critic_pretrain "${MODEL_PATH}"
  --prompt_data "${TRAIN_DATA}"
  --eval_dataset "${EVAL_DATA}"
  --input_key question
  --label_key answer
  --output_key answer
  --prompt_split train
  --eval_split test
  --prompt_max_len "${PROMPT_MAX_LEN}"
  --context_max_len "${CONTEXT_MAX_LEN}"
  --generate_max_len "${GENERATE_MAX_LEN}"
  --stride "${STRIDE}"
  --n_samples_per_prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout_batch_size "${ROLLOUT_BATCH_SIZE}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --micro_train_batch_size "${MICRO_TRAIN_BATCH_SIZE}"
  --micro_rollout_batch_size "${MICRO_ROLLOUT_BATCH_SIZE}"
  --micro_reward_batch_size "${MICRO_REWARD_BATCH_SIZE}"
  --max_samples "${MAX_SAMPLES}"
  --num_episodes "${NUM_EPISODES}"
  --max_epochs "${MAX_EPOCHS}"
  --actor_num_nodes 1
  --actor_num_gpus_per_node "${ACTOR_GPUS}"
  --critic_num_nodes 1
  --critic_num_gpus_per_node "${CRITIC_GPUS}"
  --ref_num_nodes 1
  --ref_num_gpus_per_node "${REF_GPUS}"
  --reward_num_nodes 1
  --reward_num_gpus_per_node "${REWARD_GPUS}"
  --advantage_estimator rloo
  --init_kl_coef 0.0
  --kl_estimator k2
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --zero_stage 2
  --lr_warmup_ratio 0.03
  --critic_lr_warmup_ratio 0.0
  --seed "${GLOBAL_SEED}"
  --eval_steps "${EVAL_STEPS}"
  --eval_max_samples "${EVAL_MAX_SAMPLES}"
  --eval_generate_max_len "${EVAL_GENERATE_MAX_LEN}"
  --logging_steps 10
  --save_steps "${SAVE_STEPS}"
  --save_even_count "${SAVE_EVEN_COUNT}"
  --save_hf_ckpt
  --use_tensorboard "${TB_DIR}"
  --save_path "${SAVE_PATH}"
  --ckpt_path "${SAVE_PATH}/ckpt"
  --wandb_run_name "${RUN_NAME}"
)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${train_cmd[@]}" 2>&1 | tee "${RUN_DIR}/train.log"

# ====================================================================
# 7) POST-TRAINING EVAL
# ====================================================================
ray stop --force 2>/dev/null || true
source "${STUDENT_VENV}/bin/activate"
STUDENT_PYTHON_BIN="${STUDENT_VENV}/bin/python"

echo ""
echo "──────────────────────────────────────────────────"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')  TRAINING FINISHED"
echo "  Logs:        ${RUN_DIR}/train.log"
echo "  TensorBoard: ${TB_DIR}"
echo "  Checkpoints: ${SAVE_PATH}"

if [[ "${EVAL_AFTER_TRAIN}" == "true" ]]; then
  echo ""
  echo "[post-eval] Running generation eval on ${EVAL_DATA} ..."

  post_eval_cmd=(
    "${STUDENT_PYTHON_BIN}" -m torch.distributed.run
    --nproc_per_node "${POST_EVAL_NPROC}"
    --master_port "${POST_EVAL_MASTER_PORT}"
    -m openrlhf.cli.batch_inference
    --eval_task generate
    --pretrain "${SAVE_PATH}"
    --dataset "${EVAL_DATA}"
    --input_key question
    --output_path "${POST_EVAL_OUTPUT_PATH}"
    --prompt_max_len "${POST_EVAL_PROMPT_MAX_LEN}"
    --max_new_tokens "${POST_EVAL_MAX_NEW_TOKENS}"
    --temperature "${POST_EVAL_TEMPERATURE}"
    --top_p "${POST_EVAL_TOP_P}"
    --max_samples "${POST_EVAL_MAX_SAMPLES}"
    --micro_batch_size "${POST_EVAL_MICRO_BATCH_SIZE}"
    --bf16
  )

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${post_eval_cmd[@]}" 2>&1 | tee "${POST_EVAL_LOG_PATH}"

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
echo "G3 no-teacher vicinal run completed at $(date)" > "${RUN_DIR}/status.txt"
echo "[done] logs: ${RUN_DIR}"
