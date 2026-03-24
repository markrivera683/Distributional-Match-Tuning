# 训练结束后，用 batch_inference 评测
python -m openrlhf.cli.batch_inference \
  --pretrain /root/outputs/g1_baseline_8gpu_rerun_20260324_100000/model \
  --dataset /mnt/data/data/aops/aops_qa_hf \
  --split test \
  --generate_max_len 512 \
  --output_file /root/outputs/g1_baseline_8gpu_rerun_20260324_100000/model/eval_results.jsonl