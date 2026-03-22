#!/usr/bin/env bash
# ============================================================================
#  Teacher verification smoke test: A/B comparison
#
#  Mode A (TEACHER_ENABLE=0): cf_l1oo with single target (no teacher)
#  Mode B (TEACHER_ENABLE=1): cf_l1oo with teacher target
#
#  Usage:
#    # Mode A: baseline (no teacher)
#    TEACHER_ENABLE=0 MODEL_PATH=/root/code/Qwen3.5-2B \
#      bash scripts/run_teacher_verify.sh
#
#    # Mode B: teacher enabled
#    TEACHER_ENABLE=1 MODEL_PATH=/root/code/Qwen3.5-2B \
#      bash scripts/run_teacher_verify.sh
# ============================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/.cache/huggingface/hub}"
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.995}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENRLHF_RAY_OBJECT_STORE_MEMORY_BYTES="${OPENRLHF_RAY_OBJECT_STORE_MEMORY_BYTES:-8589934592}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Core knobs ─────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/root/code/Qwen3.5-2B}"
TEACHER_ENABLE="${TEACHER_ENABLE:-1}"

# ── Derived config based on teacher mode ───────────────────────────────────
if [[ "${TEACHER_ENABLE}" == "1" ]]; then
  MODE_TAG="teacher_ON"
  CF_TARGET_MODE="teacher"
  TEACHER_PRETRAIN="${MODEL_PATH}"
  CF_TEACHER_LAMBDA="0.5"
  CF_TEACHER_N_SAMPLES="2"
else
  MODE_TAG="teacher_OFF"
  CF_TARGET_MODE="single"
  TEACHER_PRETRAIN=""
  CF_TEACHER_LAMBDA="0.0"
  CF_TEACHER_N_SAMPLES="4"
fi

RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/teacher_verify_${MODE_TAG}}"
SAVE_PATH="${RUN_ROOT}/model"
TB_ROOT="${RUN_ROOT}/tensorboard"
RUN_NAME="teacher_verify_${MODE_TAG}_$(date +%m%d_%H%M)"

mkdir -p "${RUN_ROOT}" "${SAVE_PATH}" "${TB_ROOT}"
exec > >(tee -a "${RUN_ROOT}/train.log") 2>&1

# ── Log header ─────────────────────────────────────────────────────────────
echo "================================================================="
echo " run_teacher_verify.sh   mode=${MODE_TAG}"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================="
echo " REPO_ROOT       = ${REPO_ROOT}"
echo " MODEL_PATH      = ${MODEL_PATH}"
echo " TEACHER_ENABLE  = ${TEACHER_ENABLE}"
echo " CF_TARGET_MODE  = ${CF_TARGET_MODE}"
echo " TEACHER_PRETRAIN= ${TEACHER_PRETRAIN:-<none>}"
echo " CF_TEACHER_LAMBDA= ${CF_TEACHER_LAMBDA}"
echo " CF_TEACHER_N_SAMPLES= ${CF_TEACHER_N_SAMPLES}"
echo " RUN_ROOT        = ${RUN_ROOT}"
echo "================================================================="

cd "${REPO_ROOT}"

# ── Build teacher args conditionally ───────────────────────────────────────
TEACHER_ARGS=()
if [[ -n "${TEACHER_PRETRAIN}" ]]; then
  TEACHER_ARGS+=(--teacher_pretrain "${TEACHER_PRETRAIN}")
fi

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
  "${TEACHER_ARGS[@]}" \
  --prompt_data openai/gsm8k \
  --input_key question \
  --label_key answer \
  --prompt_split train \
  --eval_split test \
  --distribution_reward_type cf_l1oo \
  --cf_target_mode "${CF_TARGET_MODE}" \
  --cf_teacher_lambda "${CF_TEACHER_LAMBDA}" \
  --cf_teacher_n_samples "${CF_TEACHER_N_SAMPLES}" \
  --cf_num_freqs 64 \
  --cf_sigma 1.0 \
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
  --max_samples 4 \
  --eval_max_samples 4 \
  --eval_down_max_samples 4 \
  --max_epochs 1 \
  --num_episodes 1 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 1 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 1 \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 1 \
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
  --seed 43 \
  --ema_beta 0.9 \
  --hidden_state_method concat \
  --embed_method last_token \
  --critic_sequence_level last_token \
  --classifier_sequence_selection closest \
  --eval_generate_max_len 64 \
  --eval_n_samples_per_prompt 4 \
  --eval_n_samples_per_prompt_down 4 \
  --eval_steps -1 \
  --eval_down_steps -1 \
  --save_steps -1 \
  --save_log_scale_count -1 \
  --save_even_count 0 \
  --logging_steps 1 \
  --use_tensorboard "${TB_ROOT}" \
  --save_path "${SAVE_PATH}" \
  --ckpt_path "${SAVE_PATH}/ckpt" \
  --wandb_run_name "${RUN_NAME}"

echo ""
echo "================================================================="
echo " DONE  mode=${MODE_TAG}  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================="
