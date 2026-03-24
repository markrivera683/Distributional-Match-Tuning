#!/usr/bin/env bash
set -euo pipefail

# =====================================================================
# G2 Remote Teacher — SMOKE TEST
# =====================================================================
#
# Minimal-budget run to verify the full pipeline:
#   actor rollout → remote teacher sampling → teacher feature extraction
#   → target measure construction → cf_l1oo reward → LOO per-sample reward
#
# Expected runtime: < 5 minutes (single GPU).
#
# What to check in the logs:
#   [RemoteTeacher] Init: ...              ← provider initialised
#   [RemoteTeacher] Requesting ...         ← API calls firing
#   [RemoteTeacher] Done: ...              ← completions received
#   [TEACHER-TARGET] MIXED target built    ← target measure constructed
#   [TEACHER-DIAG] === Final reward stats  ← rewards are non-zero
#
# Usage:
#   bash scripts/run_g2_remote_teacher_smoke.sh
#
# ==================== EDIT THESE FIRST ====================
# [MUST] MODEL / DATA
#   MODEL_PATH=/mnt/data/Qwen3.5-2B
#   PROMPT_DATA=/mnt/data/data/aops/aops_qa_hf
#   EVAL_DATASET=/mnt/data/data/aops/test_qa.jsonl
#
# [MUST] REMOTE TEACHER ENDPOINT
#   TEACHER_API_BASE=http://host:port/v1
#   TEACHER_MODEL=your-teacher-model-name
#   TEACHER_API_KEY=your-api-key
#
# [CORE] TARGET MEASURE MIXING
#   CF_TEACHER_LAMBDA=0.5
#   CF_TEACHER_N_SAMPLES=2
#
# [CORE] GPU ASSIGNMENT (smoke defaults)
#   CUDA_VISIBLE_DEVICES=0
#   ACTOR_NUM_GPUS_PER_NODE=1
#   CRITIC_NUM_GPUS_PER_NODE=1
#   REF_NUM_GPUS_PER_NODE=1
#   REWARD_NUM_GPUS_PER_NODE=1
# ==========================================================
# =====================================================================

REPO_DIR="${REPO_DIR:-/root/code/data/Distributional-Match-Tuning}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/Qwen3.5-2B}"

PROMPT_DATA="${PROMPT_DATA:-/mnt/data/data/aops/aops_qa_hf}"
EVAL_DATASET="${EVAL_DATASET:-/mnt/data/data/aops/test_qa.jsonl}"
INPUT_KEY="${INPUT_KEY:-question}"
LABEL_KEY="${LABEL_KEY:-answer}"
OUTPUT_KEY="${OUTPUT_KEY:-answer}"
PROMPT_SPLIT="${PROMPT_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

TEACHER_API_BASE="${TEACHER_API_BASE:-http://172.17.0.26:8000/v1}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen-122b}"
TEACHER_API_KEY="${TEACHER_API_KEY:-teacher-local}"
TEACHER_API_STYLE="${TEACHER_API_STYLE:-chat_completions}"
CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.5}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-2}"
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-180}"
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-3}"
TEACHER_REMOTE_BATCH_SIZE="${TEACHER_REMOTE_BATCH_SIZE:-4}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"

# GPU assignment (keep near top for quick editing)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-1}"
CRITIC_NUM_GPUS_PER_NODE="${CRITIC_NUM_GPUS_PER_NODE:-1}"
REF_NUM_GPUS_PER_NODE="${REF_NUM_GPUS_PER_NODE:-1}"
REWARD_NUM_GPUS_PER_NODE="${REWARD_NUM_GPUS_PER_NODE:-1}"

RUN_TAG="g2_smoke_$(date -u +%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/outputs/${RUN_TAG}}"
CACHE_DIR="${OUTPUT_ROOT}/teacher_cache"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}"; exit 1
fi
if [[ ! -e "${PROMPT_DATA}" ]]; then
  echo "[ERROR] PROMPT_DATA does not exist: ${PROMPT_DATA}"; exit 1
fi

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_memory_usage_threshold=0.99
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENRLHF_RAY_OBJECT_STORE_MEMORY_BYTES=4294967296
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_ROOT}" "${CACHE_DIR}"
exec > >(tee -a "${OUTPUT_ROOT}/smoke.log") 2>&1

echo "================================================================="
echo " G2 SMOKE TEST — cf_l1oo + Remote Teacher"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================="
echo "TEACHER_API_BASE = ${TEACHER_API_BASE}"
echo "TEACHER_MODEL    = ${TEACHER_MODEL}"
echo "CF_TEACHER_LAMBDA= ${CF_TEACHER_LAMBDA}"
echo "CF_TEACHER_N_SAMPLES= ${CF_TEACHER_N_SAMPLES}"
echo "TEACHER_TEMPERATURE= ${TEACHER_TEMPERATURE}"
echo "TEACHER_TOP_P= ${TEACHER_TOP_P}"
echo "CUDA_VISIBLE_DEVICES= ${CUDA_VISIBLE_DEVICES}"
echo "ACTOR_NUM_GPUS_PER_NODE  = ${ACTOR_NUM_GPUS_PER_NODE}"
echo "CRITIC_NUM_GPUS_PER_NODE = ${CRITIC_NUM_GPUS_PER_NODE}"
echo "REF_NUM_GPUS_PER_NODE    = ${REF_NUM_GPUS_PER_NODE}"
echo "REWARD_NUM_GPUS_PER_NODE = ${REWARD_NUM_GPUS_PER_NODE}"

cd "${REPO_DIR}"
ray stop --force >/dev/null 2>&1 || true

curl -s -m 10 -H "Authorization: Bearer ${TEACHER_API_KEY}" "${TEACHER_API_BASE}/models" >/dev/null \
  && echo "[Teacher Check] API reachable" \
  || echo "[Teacher Check] WARNING: API not reachable — smoke test will fail on teacher calls"

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
  --cf_target_mode teacher \
  --cf_teacher_lambda "${CF_TEACHER_LAMBDA}" \
  --cf_teacher_n_samples "${CF_TEACHER_N_SAMPLES}" \
  --cf_num_freqs 64 \
  --cf_sigma 1.0 \
  --cf_seed 43 \
  --cf_alpha 0.5 \
  --cf_beta 0.5 \
  --cf_reward_scale 1.0 \
  --teacher_backend remote \
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
  --max_epochs 1 \
  --num_episodes 1 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node "${ACTOR_NUM_GPUS_PER_NODE}" \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node "${CRITIC_NUM_GPUS_PER_NODE}" \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node "${REF_NUM_GPUS_PER_NODE}" \
  --reward_num_nodes 1 \
  --reward_num_gpus_per_node "${REWARD_NUM_GPUS_PER_NODE}" \
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
  --eval_steps -1 \
  --eval_down_steps -1 \
  --save_steps -1 \
  --logging_steps 1 \
  --use_tensorboard "${OUTPUT_ROOT}/tensorboard" \
  --save_path "${OUTPUT_ROOT}/model" \
  --ckpt_path "${OUTPUT_ROOT}/model/ckpt" \
  --wandb_run_name "${RUN_TAG}"

echo ""
echo "================================================================="
echo " SMOKE TEST COMPLETE — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo " Log: ${OUTPUT_ROOT}/smoke.log"
echo "================================================================="
echo ""
echo "Verify these appeared in the log:"
echo "  [RemoteTeacher] Init: ...          — provider initialized"
echo "  [RemoteTeacher] Requesting ...     — API calls fired"
echo "  [TEACHER-TARGET] MIXED target ...  — target measure built"
echo "  gt_rewards ... mean=...            — rewards are non-zero"
