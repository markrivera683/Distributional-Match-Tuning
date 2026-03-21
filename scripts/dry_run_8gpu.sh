#!/usr/bin/env bash
# 8卡 Dry-run: 验证模型真上 GPU
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_memory_usage_threshold=0.99

REPO_ROOT=/mnt/workspace/EBFT/ebft_only_bundle_march19/repo/ebft_openrlhf_u1_ncfm
MODEL_PATH=/mnt/workspace/EBFT/ebft_only_bundle_march19/model/Qwen2___5-1___5B
TRAIN_DATA=/mnt/aops_train_processed

RUN_DIR="/mnt/workspace/runs/dry_run_8gpu_v2_$(date +%m%d_%H%M)"
mkdir -p "$RUN_DIR"

cd "$REPO_ROOT"

echo "=== Dry-run 8GPU started at $(date) ===" | tee "$RUN_DIR/dryrun.log"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$RUN_DIR/dryrun.log"

# 极简配置: n_samples_per_prompt=2, rollout=128, train=256 (256=2*128)
timeout 180 python -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --adam_offload --pretrain_mode --no_chat_template \
  --colocate_all_models --use_kl_loss --use_whitening \
  --distribution_reward_type pointwise \
  --feature_map_type identity --rff_num_features 128 --rff_sigma 1.0 --rff_seed 43 \
  --embed_method last_token \
  --pretrain "$MODEL_PATH" \
  --prompt_data "$TRAIN_DATA" --input_key question --label_key answer \
  --prompt_split train --prompt_max_len 256 --context_max_len 512 --generate_max_len 128 \
  --stride 128 --n_samples_per_prompt 2 \
  --rollout_batch_size 128 --train_batch_size 256 \
  --micro_train_batch_size 32 --micro_rollout_batch_size 32 \
  --max_samples 8 --num_episodes 1 --max_epochs 1 \
  --actor_num_nodes 1 --actor_num_gpus_per_node 2 \
  --critic_num_nodes 1 --critic_num_gpus_per_node 2 \
  --ref_num_nodes 1 --ref_num_gpus_per_node 2 \
  --reward_num_nodes 1 --reward_num_gpus_per_node 2 \
  --advantage_estimator rloo --init_kl_coef 0.0 \
  --temperature 0.6 --top_p 1.0 --actor_learning_rate 1e-6 \
  --zero_stage 2 --lr_warmup_ratio 0.03 \
  --seed 43 --eval_steps -1 --logging_steps 1 \
  --save_steps -1 --ckpt_path "$RUN_DIR" \
  2>&1 | tee -a "$RUN_DIR/dryrun.log" &

PID=$!
echo "PID: $PID"

# 等待模型加载
sleep 70

echo "" | tee -a "$RUN_DIR/dryrun.log"
echo "=== GPU Status after 70s ===" | tee -a "$RUN_DIR/dryrun.log"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$RUN_DIR/dryrun.log"

echo "" | tee -a "$RUN_DIR/dryrun.log"
echo "=== Process Status ===" | tee -a "$RUN_DIR/dryrun.log"
ps aux | grep "train_ebft" | grep -v grep | wc -l | tee -a "$RUN_DIR/dryrun.log"

# 终止进程
kill $PID 2>/dev/null || true
pkill -f "train_ebft" 2>/dev/null || true
ray stop -f 2>/dev/null || true

echo "" | tee -a "$RUN_DIR/dryrun.log"
echo "=== Dry-run completed at $(date) ===" | tee -a "$RUN_DIR/dryrun.log"
