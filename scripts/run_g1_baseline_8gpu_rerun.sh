#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# G1 Baseline Rerun — 8×A100, pointwise EBFT, ~500 global steps
# Matches the original G1 config from run_aops_group1_baseline.sh
# Adapted for current machine paths
# ============================================================

# model:Qwen3.5-2B
# train data:aops_qa_hf
# eval data:aops_qa_hf
# gpu: 8xA100
# target completion: GT answer
# test data: test_qa.jsonl

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/.cache/huggingface/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.995}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENRLHF_RAY_OBJECT_STORE_MEMORY_BYTES="${OPENRLHF_RAY_OBJECT_STORE_MEMORY_BYTES:-8589934592}"
export PYTHONUNBUFFERED=1

# ---- Paths (adapted for current machine) ----
REPO_ROOT="/root/code/data/Distributional-Match-Tuning"
MODEL_PATH="/mnt/data/code/Qwen3.5-2B"
TRAIN_DATA="/mnt/data/data/code/aops/aops_qa_hf_dict"
EVAL_DATA="/mnt/data/data/code/aops/test_qa.jsonl"

# ---- Remote teacher (same endpoint as G2 script) ----
TEACHER_API_BASE="http://172.17.0.26:8000/v1"
TEACHER_MODEL="qwen-122b"
TEACHER_API_KEY="teacher-local"
TEACHER_API_STYLE="chat_completions"
TEACHER_BACKEND="remote"
TEACHER_TIMEOUT=180
TEACHER_MAX_RETRIES=3
TEACHER_REMOTE_BATCH_SIZE=4
TEACHER_TEMPERATURE=0.7
TEACHER_TOP_P=0.95
CF_TEACHER_LAMBDA=0.5
CF_TEACHER_N_SAMPLES=2

# ---- G1 Core: pointwise reward, frozen critic, no adapter ----
DISTRIBUTION_REWARD_TYPE="pointwise"
CF_TARGET_MODE="teacher"
EMBED_METHOD="last_token"
CRITIC_SEQUENCE_LEVEL="last_token"
CRITIC_LEARNING_RATE=0
CRITIC_LR_HEAD=0
CRITIC_CLASSIFIER_LOSS_COEF=0.0
CRITIC_DIRECT_DISCREPANCY_COEF=0.0

# ---- Feature map / CF params (identity baseline) ----
FEATURE_MAP_TYPE="identity"
RFF_NUM_FEATURES=128
RFF_SIGMA=1.0
RFF_SEED=43
CF_NUM_FREQS=128
CF_SIGMA=1.0
CF_SEED=43
CF_ALPHA=0.5
CF_BETA=0.5
CF_REWARD_SCALE=1.0
CF_TARGET_NUM_REFS=1
CF_TARGET_STD=0.05
CF_TARGET_SEED=43

# ---- Training budget: target ~500 global steps ----
# rollout_batch_size=64, so 500 * 64 = 32000 prompts needed
MAX_SAMPLES="${MAX_SAMPLES:-46000}"
NUM_EPISODES="${NUM_EPISODES:-1}"

# ---- Loss / reward coefficients ----
CE_LOSS_COEF=0.03
DIVERSITY_REW_COEF=0.5
ALIGNMENT_REW_COEF=1.0
EMA_BETA=0.9

# ---- Eval ----
# Eval disabled: GatedDeltaNet fallback OOMs on long generation
# Re-enable after installing compatible flash-linear-attention
EVAL_STEPS="${EVAL_STEPS:--1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-50}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-8}"

# ---- GPU allocation: actor=4 (+ ref colocated), critic=4 (+ reward colocated) ----
ACTOR_GPUS=4
CRITIC_GPUS=4
REF_GPUS=4
REWARD_GPUS=4

# ---- Output ----
GLOBAL_SEED=43
RUN_TAG="g1_baseline_8gpu_rerun_$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
SAVE_PATH="${RUN_ROOT}/model"
TB_ROOT="${RUN_ROOT}/tensorboard"
CACHE_DIR="${RUN_ROOT}/teacher_cache"

mkdir -p "${RUN_ROOT}" "${SAVE_PATH}" "${TB_ROOT}" "${CACHE_DIR}"

echo "============================================================"
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] G1 Baseline Rerun"
echo "============================================================"
echo "  Model:           ${MODEL_PATH}"
echo "  Train data:      ${TRAIN_DATA}"
echo "  Eval data:       ${EVAL_DATA}"
echo "  Reward type:     ${DISTRIBUTION_REWARD_TYPE}"
echo "  Target mode:     ${CF_TARGET_MODE}"
echo "  Max samples:     ${MAX_SAMPLES}"
echo "  Num episodes:    ${NUM_EPISODES}"
echo "  Teacher API:     ${TEACHER_API_BASE}"
echo "  Teacher model:   ${TEACHER_MODEL}"
echo "  Teacher lambda:  ${CF_TEACHER_LAMBDA}"
echo "  Teacher samples: ${CF_TEACHER_N_SAMPLES}"
echo "  GPU allocation:  actor=${ACTOR_GPUS} critic=${CRITIC_GPUS} ref=${REF_GPUS} reward=${REWARD_GPUS}"
echo "  Output:          ${RUN_ROOT}"
echo "============================================================"

ray stop --force 2>/dev/null || true
sleep 2

cd "${REPO_ROOT}"

python -m openrlhf.cli.train_ebft_ray \
  --bf16 --adam_offload --pretrain_mode --no_chat_template \
  --disable_ds_ckpt --colocate_actor_ref --colocate_critic_reward \
  --use_kl_loss --use_whitening --enable_ema \
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
  --cf_target_num_refs "${CF_TARGET_NUM_REFS}" \
  --cf_target_std "${CF_TARGET_STD}" \
  --cf_target_seed "${CF_TARGET_SEED}" \
  --cf_teacher_lambda "${CF_TEACHER_LAMBDA}" \
  --cf_teacher_n_samples "${CF_TEACHER_N_SAMPLES}" \
  --teacher_backend "${TEACHER_BACKEND}" \
  --teacher_api_base "${TEACHER_API_BASE}" \
  --teacher_api_key "${TEACHER_API_KEY}" \
  --teacher_api_style "${TEACHER_API_STYLE}" \
  --teacher_model_name "${TEACHER_MODEL}" \
  --teacher_timeout "${TEACHER_TIMEOUT}" \
  --teacher_max_retries "${TEACHER_MAX_RETRIES}" \
  --teacher_remote_batch_size "${TEACHER_REMOTE_BATCH_SIZE}" \
  --teacher_temperature "${TEACHER_TEMPERATURE}" \
  --teacher_top_p "${TEACHER_TOP_P}" \
  --teacher_cache_enable \
  --teacher_cache_dir "${CACHE_DIR}" \
  --embed_method "${EMBED_METHOD}" \
  --critic_sequence_level "${CRITIC_SEQUENCE_LEVEL}" \
  --critic_learning_rate "${CRITIC_LEARNING_RATE}" \
  --critic_lr_head "${CRITIC_LR_HEAD}" \
  --critic_classifier_loss_coef "${CRITIC_CLASSIFIER_LOSS_COEF}" \
  --critic_direct_discrepancy_coef "${CRITIC_DIRECT_DISCREPANCY_COEF}" \
  --ema_beta "${EMA_BETA}" \
  --ce_loss_coef "${CE_LOSS_COEF}" \
  --diversity_rew_coef "${DIVERSITY_REW_COEF}" \
  --alignment_rew_coef "${ALIGNMENT_REW_COEF}" \
  --pretrain "${MODEL_PATH}" \
  --critic_pretrain "${MODEL_PATH}" \
  --prompt_data "${TRAIN_DATA}" \
  --eval_dataset "${EVAL_DATA}" \
  --input_key question --label_key answer --output_key answer \
  --prompt_split train --eval_split test \
  --prompt_max_len 256 --context_max_len 8 --generate_max_len 8 --stride 8 \
  --n_samples_per_prompt 4 --rollout_batch_size 64 --train_batch_size 256 \
  --micro_train_batch_size 4 --micro_rollout_batch_size 4 --micro_reward_batch_size 4 \
  --max_samples "${MAX_SAMPLES}" --num_episodes "${NUM_EPISODES}" --max_epochs 1 \
  --actor_num_nodes 1 --actor_num_gpus_per_node "${ACTOR_GPUS}" \
  --critic_num_nodes 1 --critic_num_gpus_per_node "${CRITIC_GPUS}" \
  --ref_num_nodes 1 --ref_num_gpus_per_node "${REF_GPUS}" \
  --reward_num_nodes 1 --reward_num_gpus_per_node "${REWARD_GPUS}" \
  --advantage_estimator rloo --init_kl_coef 0.0 --kl_estimator k2 \
  --temperature 0.6 --top_p 1.0 --actor_learning_rate 1e-6 \
  --zero_stage 2 --lr_warmup_ratio 0.03 --lr_scheduler constant_with_warmup \
  --seed "${GLOBAL_SEED}" --eval_steps "${EVAL_STEPS}" \
  --eval_max_samples "${EVAL_MAX_SAMPLES}" --eval_generate_max_len "${EVAL_GENERATE_MAX_LEN}" \
  --save_steps -1 --logging_steps 1 --use_tensorboard "${TB_ROOT}" \
  --save_path "${SAVE_PATH}" --ckpt_path "${SAVE_PATH}/ckpt" \
  --wandb_run_name "${RUN_TAG}" \
  2>&1 | tee "${RUN_ROOT}/train.log"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] G1 Baseline Rerun finished"
echo "Logs:        ${RUN_ROOT}/train.log"
echo "TensorBoard: ${TB_ROOT}"
echo "Checkpoints: ${SAVE_PATH}"
