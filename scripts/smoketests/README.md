# 1 GPU Smoke Tests (G2 / G3)

This folder contains **quick smoke tests** corresponding to:

- `scripts/run_G2_rebase.sh`
- `scripts/run_G3_rebase.sh`

## Purpose

- Validate end-to-end pipeline on **one visible GPU**.
- Verify local online teacher startup + API + tiny training loop.
- Not for quality benchmarking.

## Key differences from rebase scripts

- Uses **single GPU** with `--colocate_all_models` so actor/critic/trainer can share it.
- Uses a **small teacher/student checkpoint** by default:
  - `/mnt/data/teacher_model/models/Qwen3.5-0.8B`
- Uses tiny batches and short lengths for fast checks.

## Run

```bash
bash scripts/smoketest_1gpu/run_G2_rebase_smoketest_1gpu.sh
bash scripts/smoketest_1gpu/run_G3_rebase_smoketest_1gpu.sh
```

## Override examples

```bash
TEACHER_PORT=18080 STUDENT_CUDA_VISIBLE_DEVICES=0 \
bash scripts/smoketest_1gpu/run_G2_rebase_smoketest_1gpu.sh
```

```bash
CRITIC_LR_HEAD=1e-4 CRITIC_DIRECT_DISCREPANCY_COEF=0.05 \
bash scripts/smoketest_1gpu/run_G3_rebase_smoketest_1gpu.sh
```
