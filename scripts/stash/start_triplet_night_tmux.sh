#!/usr/bin/env bash
# 在 tmux session 中启动三组实验

SESSION="aops_triplet"
SCRIPT="/mnt/workspace/EBFT/ebft_only_bundle_march19/repo/ebft_openrlhf_u1_ncfm/scripts/run_aops_triplet_night.sh"
LOG="/mnt/workspace/runs/triplet_night_$(date +%m%d_%H%M).log"

# 杀掉已存在的 session
tmux kill-session -t "$SESSION" 2>/dev/null || true

# 创建新的 tmux session 并运行
tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "cd /mnt/workspace/EBFT/ebft_only_bundle_march19/repo/ebft_openrlhf_u1_ncfm && bash $SCRIPT 2>&1 | tee $LOG" C-m

echo "Tmux session '$SESSION' started"
echo "Attach with: tmux attach -t $SESSION"
echo "Main log: $LOG"

# 显示状态
sleep 2
tmux ls
