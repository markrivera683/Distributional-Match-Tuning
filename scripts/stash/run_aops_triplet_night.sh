#!/usr/bin/env bash
# 三组实验夜间挂机脚本
# 串行执行: G1 -> G2 -> G3

set -euo pipefail

REPO_ROOT=/mnt/workspace/EBFT/ebft_only_bundle_march19/repo/ebft_openrlhf_u1_ncfm
MODEL_PATH=/mnt/workspace/EBFT/ebft_only_bundle_march19/model/Qwen2___5-1___5B
TRAIN_DATA=/mnt/aops_train_processed
EVAL_DATA=/mnt/workspace/data/AoPS-Instruct-test/LiveAoPSBench-2024.jsonl
RUN_ROOT=/mnt/workspace/runs

mkdir -p "$RUN_ROOT"

# 环境变量
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 公共参数
MAX_SAMPLES=-1
NUM_EPISODES=1
PROMPT_MAX_LEN=1024
CONTEXT_MAX_LEN=2048
ROLLOUT_BATCH_SIZE=256
TRAIN_BATCH_SIZE=1024
MICRO_BATCH=64

#######################################
# G1: Baseline (pointwise)
#######################################
echo "========================================"
echo "Starting G1: Baseline (pointwise)"
echo "========================================"

G1_RUN="aops_g1_baseline_seed43_$(date +%m%d_%H%M)"
G1_DIR="$RUN_ROOT/$G1_RUN"
mkdir -p "$G1_DIR"

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --adam_offload --pretrain_mode --no_chat_template \
  --colocate_all_models --use_kl_loss --use_whitening \
  --distribution_reward_type pointwise \
  --feature_map_type identity --rff_num_features 128 --rff_sigma 1.0 --rff_seed 43 \
  --embed_method last_token \
  --pretrain "$MODEL_PATH" \
  --prompt_data "$TRAIN_DATA" --input_key question --label_key answer \
  --prompt_split train --prompt_max_len $PROMPT_MAX_LEN --context_max_len $CONTEXT_MAX_LEN --generate_max_len 1024 \
  --stride 512 --n_samples_per_prompt 4 \
  --rollout_batch_size $ROLLOUT_BATCH_SIZE --train_batch_size $TRAIN_BATCH_SIZE \
  --micro_train_batch_size $MICRO_BATCH --micro_rollout_batch_size $MICRO_BATCH \
  --max_samples $MAX_SAMPLES --num_episodes $NUM_EPISODES --max_epochs 1 \
  --actor_num_nodes 1 --actor_num_gpus_per_node 2 \
  --critic_num_nodes 1 --critic_num_gpus_per_node 2 \
  --ref_num_nodes 1 --ref_num_gpus_per_node 2 \
  --reward_num_nodes 1 --reward_num_gpus_per_node 2 \
  --advantage_estimator rloo --init_kl_coef 0.0 \
  --temperature 0.6 --top_p 1.0 --actor_learning_rate 1e-6 \
  --zero_stage 2 --lr_warmup_ratio 0.03 \
  --seed 43 --eval_steps -1 --logging_steps 10 \
  --save_steps 100 --ckpt_path "$G1_DIR" \
  2>&1 | tee "$G1_DIR/train.log"

echo "G1 completed at $(date)" | tee -a "$G1_DIR/status.txt"

#######################################
# G2: Dist-only (cf_l1oo, no adapter)
#######################################
echo ""
echo "========================================"
echo "Starting G2: Dist-only (cf_l1oo)"
echo "========================================"

G2_RUN="aops_g2_distonly_seed43_$(date +%m%d_%H%M)"
G2_DIR="$RUN_ROOT/$G2_RUN"
mkdir -p "$G2_DIR"

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --adam_offload --pretrain_mode --no_chat_template \
  --colocate_all_models --use_kl_loss --use_whitening \
  --distribution_reward_type cf_l1oo \
  --feature_map_type identity --rff_num_features 128 --rff_sigma 1.0 --rff_seed 43 \
  --cf_num_freqs 128 --cf_sigma 1.0 --cf_seed 43 --cf_alpha 0.5 --cf_beta 0.5 \
  --embed_method last_token --critic_sequence_level last_token \
  --pretrain "$MODEL_PATH" --critic_pretrain "$MODEL_PATH" \
  --prompt_data "$TRAIN_DATA" --input_key question --label_key answer \
  --prompt_split train --prompt_max_len $PROMPT_MAX_LEN --context_max_len $CONTEXT_MAX_LEN --generate_max_len 1024 \
  --stride 512 --n_samples_per_prompt 4 \
  --rollout_batch_size $ROLLOUT_BATCH_SIZE --train_batch_size $TRAIN_BATCH_SIZE \
  --micro_train_batch_size $MICRO_BATCH --micro_rollout_batch_size $MICRO_BATCH \
  --max_samples $MAX_SAMPLES --num_episodes $NUM_EPISODES --max_epochs 1 \
  --actor_num_nodes 1 --actor_num_gpus_per_node 2 \
  --critic_num_nodes 1 --critic_num_gpus_per_node 2 \
  --ref_num_nodes 1 --ref_num_gpus_per_node 2 \
  --reward_num_nodes 1 --reward_num_gpus_per_node 2 \
  --advantage_estimator rloo --init_kl_coef 0.0 \
  --temperature 0.6 --top_p 1.0 --actor_learning_rate 1e-6 \
  --zero_stage 2 --lr_warmup_ratio 0.03 \
  --critic_lr_warmup_ratio 0.0 --critic_learning_rate 0.0 \
  --seed 43 --eval_steps -1 --logging_steps 10 \
  --save_steps 100 --ckpt_path "$G2_DIR" \
  2>&1 | tee "$G2_DIR/train.log"

echo "G2 completed at $(date)" | tee -a "$G2_DIR/status.txt"

#######################################
# G3: Dist+EMA (cf_l1oo + adapter + EMA)
#######################################
echo ""
echo "========================================"
echo "Starting G3: Dist+EMA"
echo "========================================"

G3_RUN="aops_g3_distema_seed43_$(date +%m%d_%H%M)"
G3_DIR="$RUN_ROOT/$G3_RUN"
mkdir -p "$G3_DIR"

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --adam_offload --pretrain_mode --no_chat_template \
  --colocate_all_models --use_kl_loss --use_whitening --enable_ema \
  --distribution_reward_type cf_l1oo \
  --feature_map_type identity --rff_num_features 128 --rff_sigma 1.0 --rff_seed 43 \
  --cf_num_freqs 128 --cf_sigma 1.0 --cf_seed 43 --cf_alpha 0.5 --cf_beta 0.5 \
  --embed_method last_token --critic_sequence_level last_token \
  --critic_learning_rate 9e-6 --critic_lr_head 1e-5 \
  --critic_classifier_loss_coef 1.0 --critic_direct_discrepancy_coef 0.0 \
  --feature_adapter_type residual_bottleneck --feature_adapter_rank 64 \
  --ema_beta 0.99 --ce_loss_coef 0.03 \
  --pretrain "$MODEL_PATH" --critic_pretrain "$MODEL_PATH" \
  --prompt_data "$TRAIN_DATA" --input_key question --label_key answer \
  --prompt_split train --prompt_max_len $PROMPT_MAX_LEN --context_max_len $CONTEXT_MAX_LEN --generate_max_len 1024 \
  --stride 512 --n_samples_per_prompt 4 \
  --rollout_batch_size $ROLLOUT_BATCH_SIZE --train_batch_size $TRAIN_BATCH_SIZE \
  --micro_train_batch_size $MICRO_BATCH --micro_rollout_batch_size $MICRO_BATCH \
  --max_samples $MAX_SAMPLES --num_episodes $NUM_EPISODES --max_epochs 1 \
  --actor_num_nodes 1 --actor_num_gpus_per_node 2 \
  --critic_num_nodes 1 --critic_num_gpus_per_node 4 \
  --ref_num_nodes 1 --ref_num_gpus_per_node 1 \
  --reward_num_nodes 1 --reward_num_gpus_per_node 1 \
  --advantage_estimator rloo --init_kl_coef 0.0 \
  --temperature 0.6 --top_p 1.0 --actor_learning_rate 1e-6 \
  --zero_stage 2 --lr_warmup_ratio 0.03 \
  --critic_lr_warmup_ratio 0.0 \
  --seed 43 --eval_steps -1 --logging_steps 10 \
  --save_steps 100 --ckpt_path "$G3_DIR" \
  2>&1 | tee "$G3_DIR/train.log"

echo "G3 completed at $(date)" | tee -a "$G3_DIR/status.txt"

echo ""
echo "========================================"
echo "All three experiments completed!"
echo "========================================"
echo "G1: $G1_DIR"
echo "G2: $G2_DIR"
echo "G3: $G3_DIR"
