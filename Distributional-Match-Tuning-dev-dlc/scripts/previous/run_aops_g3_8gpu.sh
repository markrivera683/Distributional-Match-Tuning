#!/usr/bin/env bash
# G3 Dist+EMA - 8 GPU, maximize VRAM usage, Critic gets 4 GPUs
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1

REPO_ROOT=/mnt/workspace/EBFT/ebft_only_bundle_march19/repo/ebft_openrlhf_u1_ncfm
MODEL_PATH=/mnt/workspace/EBFT/ebft_only_bundle_march19/model/Qwen2___5-1___5B
TRAIN_DATA=/mnt/aops_train_processed

RUN_NAME="aops_g3_distema_8gpu_$(date +%m%d_%H%M)"
RUN_DIR="/mnt/workspace/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"

cd "$REPO_ROOT"

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
  --prompt_split train --prompt_max_len 512 --context_max_len 2048 --generate_max_len 512 \
  --stride 256 --n_samples_per_prompt 4 \
  --rollout_batch_size 512 --train_batch_size 2048 \
  --micro_train_batch_size 128 --micro_rollout_batch_size 128 \
  --num_episodes 1 --max_epochs 1 \
  --actor_num_nodes 1 --actor_num_gpus_per_node 2 \
  --critic_num_nodes 1 --critic_num_gpus_per_node 4 \
  --ref_num_nodes 1 --ref_num_gpus_per_node 1 \
  --reward_num_nodes 1 --reward_num_gpus_per_node 1 \
  --advantage_estimator rloo --init_kl_coef 0.0 \
  --temperature 0.6 --top_p 1.0 --actor_learning_rate 1e-6 \
  --zero_stage 2 --lr_warmup_ratio 0.03 \
  --critic_lr_warmup_ratio 0.0 \
  --seed 43 --eval_steps -1 --logging_steps 10 \
  --save_steps 50 --ckpt_path "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/train.log"

echo "G3 completed at $(date)" > "$RUN_DIR/status.txt"
