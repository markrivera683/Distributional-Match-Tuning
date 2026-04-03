#!/usr/bin/env bash
# 真正的 8 卡 Dry-run: 4+4 拓扑，去掉 colocate_all_models
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_memory_usage_threshold=0.99

REPO_ROOT=/mnt/workspace/EBFT/ebft_only_bundle_march19/repo/ebft_openrlhf_u1_ncfm
MODEL_PATH=/mnt/workspace/EBFT/ebft_only_bundle_march19/model/Qwen2___5-1___5B
# 使用正确的数据路径
TRAIN_DATA=/mnt/workspace/data/AoPS-Instruct/data

RUN_NAME="dry_run_8gpu_real_$(date +%m%d_%H%M)"
RUN_DIR="/mnt/workspace/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"

cd "$REPO_ROOT"

echo "=== 8GPU Dry-run (4+4 topology) started at $(date) ===" | tee "$RUN_DIR/dryrun.log"
echo "Python: $(which python3)" | tee -a "$RUN_DIR/dryrun.log"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$RUN_DIR/dryrun.log"
echo "Config: actor=4, critic=4, NO colocate_all_models" | tee -a "$RUN_DIR/dryrun.log"

# 8卡 dry-run: 4+4 拓扑，保守参数
python3 -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --adam_offload --pretrain_mode --no_chat_template \
  --use_kl_loss --use_whitening \
  --distribution_reward_type pointwise \
  --feature_map_type identity --rff_num_features 128 --rff_sigma 1.0 --rff_seed 43 \
  --embed_method last_token \
  --hidden_state_method last_only \
  --pretrain "$MODEL_PATH" \
  --critic_pretrain "$MODEL_PATH" \
  --prompt_data "$TRAIN_DATA" --input_key question --label_key answer \
  --prompt_max_len 256 --context_max_len 512 --generate_max_len 128 \
  --stride 128 --n_samples_per_prompt 2 \
  --rollout_batch_size 128 --train_batch_size 256 \
  --micro_train_batch_size 32 --micro_rollout_batch_size 32 \
  --max_samples 8 --num_episodes 1 --max_epochs 1 \
  --actor_num_nodes 1 --actor_num_gpus_per_node 4 \
  --critic_num_nodes 1 --critic_num_gpus_per_node 4 \
  --init_kl_coef 0.0 \
  --temperature 0.6 --top_p 1.0 --actor_learning_rate 1e-6 \
  --zero_stage 2 --lr_warmup_ratio 0.03 \
  --seed 43 --eval_steps -1 --logging_steps 1 \
  --save_steps -1 --ckpt_path "$RUN_DIR" \
  2>&1 | tee -a "$RUN_DIR/dryrun.log" &

PID=$!
echo "PID: $PID" | tee -a "$RUN_DIR/dryrun.log"

# 等待模型加载和初始化
echo "Waiting 90s for model loading..." | tee -a "$RUN_DIR/dryrun.log"
sleep 90

echo "" | tee -a "$RUN_DIR/dryrun.log"
echo "=== GPU Status after 90s ===" | tee -a "$RUN_DIR/dryrun.log"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$RUN_DIR/dryrun.log"

echo "" | tee -a "$RUN_DIR/dryrun.log"
echo "=== Ray Status ===" | tee -a "$RUN_DIR/dryrun.log"
ray status 2>/dev/null | head -50 | tee -a "$RUN_DIR/dryrun.log" || echo "Ray status unavailable" | tee -a "$RUN_DIR/dryrun.log"

echo "" | tee -a "$RUN_DIR/dryrun.log"
echo "=== Training Progress (last 20 lines) ===" | tee -a "$RUN_DIR/dryrun.log"
tail -20 "$RUN_DIR/dryrun.log" | tee -a "$RUN_DIR/dryrun.log"

# 等待训练完成或超时
echo "" | tee -a "$RUN_DIR/dryrun.log"
echo "Waiting for training to complete (max 300s)..." | tee -a "$RUN_DIR/dryrun.log"
wait $PID || true

echo "" | tee -a "$RUN_DIR/dryrun.log"
echo "=== Final GPU Status ===" | tee -a "$RUN_DIR/dryrun.log"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$RUN_DIR/dryrun.log"

echo "" | tee -a "$RUN_DIR/dryrun.log"
echo "=== Dry-run completed at $(date) ===" | tee -a "$RUN_DIR/dryrun.log"

# 清理
ray stop -f 2>/dev/null || true

