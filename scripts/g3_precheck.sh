#!/usr/bin/env bash
set -euo pipefail

# G3 Precheck: 最快速度验证 feature adapter 真学习
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1

REPO_ROOT=/mnt/workspace/EBFT/ebft_only_bundle_march19/repo/ebft_openrlhf_u1_ncfm
MODEL_PATH=/mnt/workspace/EBFT/ebft_only_bundle_march19/model/Qwen2___5-1___5B
TRAIN_DATA=/mnt/aops_train_processed
EVAL_DATA=/mnt/workspace/data/AoPS-Instruct-test/LiveAoPSBench-2024.jsonl

RUN_NAME="aops_g3_precheck_$(date +%m%d_%H%M)"
RUN_DIR="/mnt/workspace/runs/${RUN_NAME}"
mkdir -p "$RUN_DIR"

cd "$REPO_ROOT"

python -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --adam_offload --pretrain_mode --no_chat_template \
  --colocate_all_models \
  --use_kl_loss --use_whitening --enable_ema \
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
  --prompt_split train --prompt_max_len 256 --context_max_len 8 --generate_max_len 8 \
  --stride 8 --n_samples_per_prompt 2 --rollout_batch_size 16 --train_batch_size 32 \
  --micro_train_batch_size 8 --micro_rollout_batch_size 8 --micro_reward_batch_size 8 \
  --max_samples 64 --num_episodes 1 --max_epochs 1 \
  --actor_num_nodes 1 --actor_num_gpus_per_node 1 \
  --critic_num_nodes 1 --critic_num_gpus_per_node 1 \
  --ref_num_nodes 1 --ref_num_gpus_per_node 1 \
  --reward_num_nodes 1 --reward_num_gpus_per_node 1 \
  --advantage_estimator rloo --init_kl_coef 0.0 --kl_estimator k2 \
  --temperature 0.6 --top_p 1.0 --actor_learning_rate 1e-6 \
  --zero_stage 2 --lr_warmup_ratio 0.03 --lr_scheduler constant_with_warmup \
  --critic_lr_warmup_ratio 0.0 --seed 43 \
  --eval_steps -1 --logging_steps 1 \
  --save_steps -1 --ckpt_path "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/train.log"

echo "Precheck complete. Log: $RUN_DIR/train.log"
