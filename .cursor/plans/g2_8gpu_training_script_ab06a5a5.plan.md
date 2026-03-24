---
name: G2 8GPU Training Script
overview: Create a clean, well-organized 8xA100 training script for G2 remote-teacher distributional matching, with all critical hyperparameters clearly grouped and commented at the top.
todos:
  - id: create-script
    content: Create scripts/run_g2_8gpu_remote_teacher.sh with all critical parameters clearly documented at the top
    status: completed
isProject: false
---

# G2 8-GPU Remote Teacher Training Script

## What to Create

A single new script `scripts/run_g2_8gpu_remote_teacher.sh` based on the existing `[scripts/run_g2_baseline_8gpu_rerun.sh](scripts/run_g2_baseline_8gpu_rerun.sh)`, reorganized with all critical parameters clearly grouped at the very top with explanatory comments.

## Key Changes from Existing Script

- Reorganize the top section into clearly labeled blocks with inline comments explaining what each parameter does and how to adjust it
- Group the parameters in this order:
  1. GPU allocation (CUDA_VISIBLE_DEVICES, actor/critic/ref/reward GPU counts)
  2. Remote teacher endpoint (API base, model name, API key, API style)
  3. Teacher target distribution (lambda, n_samples, temperature, top_p, max_new_tokens, cache dir)
  4. Reward function (distribution_reward_type, cf_target_mode, cf_num_freqs, cf_sigma, cf_alpha, cf_beta, cf_reward_scale, feature_map_type)
  5. Model and data paths
  6. Training budget (max_samples, num_episodes, batch sizes)
  7. Loss coefficients (ce_loss_coef, diversity_rew_coef, alignment_rew_coef)
  8. Output directory
- No logic changes to the actual `python -m openrlhf.cli.train_ebft_ray` invocation; only reorganization and documentation of the shell variables

## Batch Size Constraints (verified)

- `train_batch_size=256 == n_samples_per_prompt(4) * rollout_batch_size(64)` -- ED constraint satisfied
- `train_batch_size=256 % (micro_train_batch_size(4) * actor_world_size(4)) == 0` -- DeepSpeed constraint satisfied

