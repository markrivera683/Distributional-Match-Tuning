#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  G2 Rebase — no-teacher single-target + two-round completion    ║
# ║  cf_l1oo reward · single GT target · 16k/32k post-eval         ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Controlled variables vs scripts/run_G2_rebase_no_teacher_vicinal.sh:
#   - keep distribution_reward_type = cf_l1oo
#   - switch cf_target_mode: vicinal -> single
#   - switch cf_target_num_refs: 8 -> 1
#   - after training, run scripts/supplement_2rounds/G2.sh for
#     16k first pass + 32k retry completion eval
#
# Usage:
#   bash scripts/run_G2_rebase_no_teacher_single_2rounds.sh
# Override any variable via env, e.g.:
#   TARGET_STEPS=500 bash scripts/run_G2_rebase_no_teacher_single_2rounds.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ACTOR_GPUS=2 CRITIC_GPUS=2 \
#     bash scripts/run_G2_rebase_no_teacher_single_2rounds.sh
set -euo pipefail

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "${csv}" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"${csv}"
}

# --------------------------------------------------------------------
# 0) GPU ASSIGNMENT — teacher removed, reuse all visible GPUs
# --------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
CRITIC_GPUS="${CRITIC_GPUS:-4}"
REF_GPUS="${REF_GPUS:-${ACTOR_GPUS}}"
REWARD_GPUS="${REWARD_GPUS:-${CRITIC_GPUS}}"

# --------------------------------------------------------------------
# 1) TRAINING DATA / MODEL PATHS
# --------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/mnt/data/ebft-distribution-new/code}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/teacher_model/models/Qwen3.5-0.8B}"
DEFAULT_TRAIN_DATA="/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
FALLBACK_LOCAL_DATA="${FALLBACK_LOCAL_DATA:-}"
TRAIN_DATA="${TRAIN_DATA:-${DEFAULT_TRAIN_DATA}}"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"

# Venvs live on local ext4 (ossfs2 can't host venv symlinks). See
# scripts/setup_env.sh for the bootstrap that creates and snapshots them.
STUDENT_VENV="${STUDENT_VENV:-/mnt/workspace/venvs/.venv}"

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

# Reduce CUDA OOM under tight memory budgets. RLHF batches reshape every
# PPO step (rollout vs train, variable seq lens), so PyTorch's default
# fixed-size segments fragment fast. expandable_segments lets the
# allocator grow segments on demand and typically frees 1-2 GiB of
# headroom on an 80GB A100. PyTorch suggests this in the OOM message.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --------------------------------------------------------------------
# 2) TRAINING KNOBS — G2 budget, no-teacher single target
# --------------------------------------------------------------------
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
CF_TARGET_NUM_REFS="${CF_TARGET_NUM_REFS:-1}"
CF_TARGET_STD="${CF_TARGET_STD:-0.05}"
CF_TARGET_SEED="${CF_TARGET_SEED:-43}"
CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.0}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-${N_SAMPLES_PER_PROMPT}}"

ACTOR_LR="${ACTOR_LR:-1e-6}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
GLOBAL_SEED="${GLOBAL_SEED:-43}"

# --------------------------------------------------------------------
# 3) EVAL / CHECKPOINT
# --------------------------------------------------------------------
EVAL_STEPS="${EVAL_STEPS:-1000}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-50}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-${GENERATE_MAX_LEN}}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_EVEN_COUNT="${SAVE_EVEN_COUNT:-10}"

RUN_TWO_ROUND_EVAL="${RUN_TWO_ROUND_EVAL:-true}"
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

# --------------------------------------------------------------------
# 4) ENV / RUN DIR
# --------------------------------------------------------------------
# HF_HOME and PYTORCH_CUDA_ALLOC_CONF are exported above (section 1) with
# DSW-specific defaults; do not redeclare here or the upper values would be
# silently shadowed if a user pre-exported only one of the two.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED=1

RUN_NAME="${RUN_NAME:-g2_no_teacher_single_8gpu_$(date +%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SAVE_PATH="${RUN_DIR}/model"
TB_DIR="${RUN_DIR}/tensorboard"
SCRIPT_NAME="$(basename "$0" .sh)"
SCRIPT_LOG_PATH="${RUN_DIR}/${SCRIPT_NAME}.log"
mkdir -p "${RUN_DIR}" "${SAVE_PATH}" "${TB_DIR}"
exec > >(tee -a "${SCRIPT_LOG_PATH}") 2>&1

# --------------------------------------------------------------------
# 5) SANITY CHECK
# --------------------------------------------------------------------
gpu_count="$(count_csv_items "${CUDA_VISIBLE_DEVICES}")"
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
if [[ "${EVAL_DATA}" == /* && ! -e "${EVAL_DATA}" ]]; then
  echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"
  exit 1
fi
if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

echo "========== AOPS G2 no-teacher single run =========="
echo "RUN_DIR:                    ${RUN_DIR}"
echo "GPUs:                       ${CUDA_VISIBLE_DEVICES} (count=${gpu_count})"
echo "Actor/Critic GPUs:          ${ACTOR_GPUS}/${CRITIC_GPUS}"
echo "Ref/Reward GPUs (colocate): ${REF_GPUS}/${REWARD_GPUS}"
echo "Model:                      ${MODEL_PATH}"
echo "Train data:                 ${TRAIN_DATA}"
echo "Eval data:                  ${EVAL_DATA}"
echo "Student python:             ${STUDENT_PYTHON_BIN}"
echo "distribution_reward:        cf_l1oo"
echo "cf_target_mode:             single"
echo "cf_target_num_refs:         ${CF_TARGET_NUM_REFS}"
echo "teacher_in_reward:          false"
echo "target_steps:               ${TARGET_STEPS}"
echo "max_samples:                ${MAX_SAMPLES}"
echo "eval_steps:                 ${EVAL_STEPS}"
echo "save_steps:                 ${SAVE_STEPS}"
echo "run_two_round_eval:         ${RUN_TWO_ROUND_EVAL}"
echo "==================================================="

# --------------------------------------------------------------------
# 6) TRAIN
# --------------------------------------------------------------------
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
  --cf_target_mode single
  --cf_target_num_refs "${CF_TARGET_NUM_REFS}"
  --cf_target_std "${CF_TARGET_STD}"
  --cf_target_seed "${CF_TARGET_SEED}"
  --cf_teacher_lambda "${CF_TEACHER_LAMBDA}"
  --cf_teacher_n_samples "${CF_TEACHER_N_SAMPLES}"
  --embed_method last_token
  --critic_sequence_level last_token
  --critic_learning_rate 0.0
  --critic_lr_head 0.0
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
  --actor_learning_rate "${ACTOR_LR}"
  --zero_stage 3
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

# --------------------------------------------------------------------
# 7) TWO-ROUND COMPLETION EVAL
# --------------------------------------------------------------------
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
  bash "${REPO_ROOT}/scripts/supplement_2rounds/G2.sh"
fi

echo "──────────────────────────────────────────────────"
echo "G2 no-teacher single run completed at $(date)" > "${RUN_DIR}/status.txt"
echo "[done] logs: ${RUN_DIR}"
