#!/usr/bin/env bash
# G1 Formal - 真正的 8 卡训练: actor=4, critic=4, 关闭 eval 避免 trainer 占 GPU
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_memory_usage_threshold=0.99
export PYTHONUNBUFFERED=1

REPO_ROOT=/mnt/workspace/EBFT/ebft_only_bundle_march19/repo/ebft_openrlhf_u1_ncfm
MODEL_PATH=/mnt/workspace/EBFT/ebft_only_bundle_march19/model/Qwen2___5-1___5B
TRAIN_DATA=/mnt/aops_train_processed

RUN_NAME="g1_formal_8gpu_restart_$(date +%m%d_%H%M)"
RUN_DIR="/mnt/workspace/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"

cd "$REPO_ROOT"

echo "=== G1 Formal 8GPU (4+4) RESTARTED at $(date) ===" | tee "$RUN_DIR/train.log"
echo "Run dir: $RUN_DIR" | tee -a "$RUN_DIR/train.log"
echo "Config: actor=4, critic=4, NO eval (trainer no GPU)" | tee -a "$RUN_DIR/train.log"

# G1 Formal: pointwise, 4+4 拓扑，关闭 eval 避免 trainer 申请 GPU
python3 -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --adam_offload --pretrain_mode --no_chat_template \
  --use_kl_loss --use_whitening \
  --distribution_reward_type pointwise \
  --embed_method concat \
  --hidden_state_method last_only \
  --pretrain "$MODEL_PATH" \
  --critic_pretrain "$MODEL_PATH" \
  --critic_learning_rate 0.0 \
  --prompt_data "$TRAIN_DATA" \
  --input_key question --label_key answer \
  --prompt_split train \
  --prompt_max_len 512 \
  --context_max_len 256 \
  --generate_max_len 128 \
  --stride 64 \
  --n_samples_per_prompt 4 \
  --rollout_batch_size 128 \
  --train_batch_size 512 \
  --micro_train_batch_size 64 \
  --micro_rollout_batch_size 64 \
  --max_samples 256 \
  --num_episodes 1 \
  --max_epochs 1 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 4 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 4 \
  --init_kl_coef 0.0 \
  --temperature 0.6 \
  --top_p 1.0 \
  --actor_learning_rate 1e-6 \
  --zero_stage 2 \
  --lr_warmup_ratio 0.03 \
  --seed 43 \
  --eval_steps -1 \
  --logging_steps 1 \
  --save_steps 50 \
  --ckpt_path "$RUN_DIR" \
  2>&1 | tee -a "$RUN_DIR/train.log"

echo "=== G1 completed at $(date) ===" | tee -a "$RUN_DIR/train.log"
