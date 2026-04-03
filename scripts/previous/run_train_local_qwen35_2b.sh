#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------
# Local single-machine EBFT launch script (distribution matching mainline)
# ---------------------------------------------------------------------
# Usage:
#   bash scripts/run_train_local_qwen35_2b.sh
#
# Optional overrides:
#   REPO_DIR=/root/code/Distributional-Match-Tuning \
#   MODEL_PATH=/mnt/data/Qwen3.5-2B \
#   PROMPT_DATA=openai/gsm8k \
#   EVAL_DATASET=openai/gsm8k \
#   MAX_SAMPLES=64 \
#   NUM_EPISODES=1 \
#   CUDA_VISIBLE_DEVICES=0 \
#   bash scripts/run_train_local_qwen35_2b.sh
#
# Notes:
# - This first version intentionally does NOT enable teacher path.
# - It prefers cf_l1oo (distribution matching) with single target.
# - Ray is auto-started by train_ebft_ray.py (no manual ray start --head needed).
# ---------------------------------------------------------------------

REPO_DIR="${REPO_DIR:-/root/code/Distributional-Match-Tuning}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3.5-2B}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/outputs/ebft_local_qwen35_2b}"
RUN_NAME="${RUN_NAME:-local_qwen35_2b_cf_l1oo_$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
TB_DIR="${RUN_DIR}/tensorboard"
SAVE_DIR="${RUN_DIR}/model"

# Dataset defaults (follow existing repo scripts)
PROMPT_DATA="${PROMPT_DATA:-openai/gsm8k}"
EVAL_DATASET="${EVAL_DATASET:-openai/gsm8k}"
INPUT_KEY="${INPUT_KEY:-question}"
LABEL_KEY="${LABEL_KEY:-answer}"
OUTPUT_KEY="${OUTPUT_KEY:-answer}"
PROMPT_SPLIT="${PROMPT_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

# Small but real training defaults
MAX_SAMPLES="${MAX_SAMPLES:-64}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-16}"
EVAL_DOWN_MAX_SAMPLES="${EVAL_DOWN_MAX_SAMPLES:-16}"
NUM_EPISODES="${NUM_EPISODES:-1}"
SEED="${SEED:-43}"

# Optional conda activation.
# TODO: if your env name is not "openrlhf", set CONDA_ENV before running.
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  CONDA_ENV="${CONDA_ENV:-openrlhf}"
  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda activate "${CONDA_ENV}"
  else
    echo "[WARN] Conda env '${CONDA_ENV}' not found; running in current shell env."
  fi
fi

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[ERROR] REPO_DIR does not exist: ${REPO_DIR}"
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}"
  echo "Please check local model path first."
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENRLHF_RAY_OBJECT_STORE_MEMORY_BYTES="${OPENRLHF_RAY_OBJECT_STORE_MEMORY_BYTES:-4294967296}"
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_DIR}" "${TB_DIR}" "${SAVE_DIR}"

exec > >(tee -a "${RUN_DIR}/train.log") 2>&1

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Starting local EBFT run"
echo "REPO_DIR=${REPO_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "RUN_DIR=${RUN_DIR}"
echo "PROMPT_DATA=${PROMPT_DATA}"
echo "EVAL_DATASET=${EVAL_DATASET}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

cd "${REPO_DIR}"

# Clean stale Ray local processes if any; train_ebft_ray.py will init Ray itself.
ray stop --force >/dev/null 2>&1 || true

python -m openrlhf.cli.train_ebft_ray \
  --bf16 \
  --adam_offload \
  --pretrain_mode \
  --no_chat_template \
  --disable_ds_ckpt \
  --colocate_all_models \
  --use_kl_loss \
  --use_whitening \
  --enable_ema \
  --pretrain "${MODEL_PATH}" \
  --critic_pretrain "${MODEL_PATH}" \
  --prompt_data "${PROMPT_DATA}" \
  --eval_dataset "${EVAL_DATASET}" \
  --input_key "${INPUT_KEY}" \
  --label_key "${LABEL_KEY}" \
  --output_key "${OUTPUT_KEY}" \
  --prompt_split "${PROMPT_SPLIT}" \
  --eval_split "${EVAL_SPLIT}" \
  --distribution_reward_type cf_l1oo \
  --cf_target_mode single \
  --cf_num_freqs 128 \
  --cf_sigma 1.0 \
  --cf_seed 43 \
  --cf_alpha 0.5 \
  --cf_beta 0.5 \
  --cf_reward_scale 1.0 \
  --prompt_max_len 128 \
  --context_max_len 8 \
  --generate_max_len 8 \
  --stride 8 \
  --n_samples_per_prompt 4 \
  --rollout_batch_size 1 \
  --train_batch_size 4 \
  --micro_train_batch_size 4 \
  --micro_rollout_batch_size 4 \
  --micro_reward_batch_size 2 \
  --max_samples "${MAX_SAMPLES}" \
  --eval_max_samples "${EVAL_MAX_SAMPLES}" \
  --eval_down_max_samples "${EVAL_DOWN_MAX_SAMPLES}" \
  --max_epochs 1 \
  --num_episodes "${NUM_EPISODES}" \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 1 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 1 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 1 \
  --reward_num_nodes 1 \
  --reward_num_gpus_per_node 1 \
  --advantage_estimator rloo \
  --init_kl_coef 0.0 \
  --kl_estimator k2 \
  --temperature 0.6 \
  --top_p 1.0 \
  --ce_loss_coef 0.03 \
  --rl_loss_coef 1.0 \
  --diversity_rew_coef 0.0 \
  --alignment_rew_coef 1.0 \
  --critic_learning_rate 0 \
  --critic_lr_head 0 \
  --critic_classifier_loss_coef 0.0 \
  --actor_learning_rate 1e-6 \
  --zero_stage 2 \
  --lr_warmup_ratio 0.03 \
  --lr_scheduler constant_with_warmup \
  --critic_lr_scheduler constant_with_warmup \
  --seed "${SEED}" \
  --ema_beta 0.9 \
  --hidden_state_method concat \
  --embed_method last_token \
  --critic_sequence_level last_token \
  --classifier_sequence_selection closest \
  --eval_steps -1 \
  --eval_down_steps -1 \
  --save_steps -1 \
  --save_log_scale_count -1 \
  --save_even_count 0 \
  --logging_steps 1 \
  --use_tensorboard "${TB_DIR}" \
  --save_path "${SAVE_DIR}" \
  --ckpt_path "${SAVE_DIR}/ckpt" \
  --wandb_run_name "${RUN_NAME}"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Training finished"
echo "Logs: ${RUN_DIR}/train.log"
echo "TensorBoard: ${TB_DIR}"
echo "Checkpoints: ${SAVE_DIR}"
