# Paper QA Baseline Review

## Goal

- Reproduce the EBFT paper-shaped `Q&A coding` baseline first.
- Do not start from AoPS or long-completion math settings.
- Keep the run comparable to the paper before changing model scale or `n`.

## Entry Script

- `scripts/run_step1_paper_qa_feature_map.sh`
- Default `FEATURE_MAP_TYPE=identity`
- Target model family: `Qwen2.5-1.5B`

## Locked Paper-Shaped Values

- `prompt_data=sjelassi/opencode-instruct_100k_200tok`
- `prompt_max_len=1024`
- `context_max_len=8`
- `generate_max_len=8`
- `n_samples_per_prompt=4`
- `rollout_batch_size=16`
- `train_batch_size=64`
- `micro_train_batch_size=8`
- `micro_rollout_batch_size=8`
- `micro_reward_batch_size=8`
- `temperature=0.6`
- `actor_learning_rate=1e-6`
- `advantage_estimator=rloo`
- `init_kl_coef=0.0`
- `use_whitening=true`
- `enable_ema=true`
- `ema_beta=0.9`
- `hidden_state_method=concat`
- `embed_method=last_token`
- `critic_sequence_level=last_token`
- `critic_learning_rate=0`
- `critic_lr_head=0`

## Suggested First Run

Use the script defaults as much as possible. Only point it at the local repo and model:

```bash
REPO_ROOT=/root/code/Distributional-Match-Tuning \
MODEL_PATH=/path/to/local/Qwen2.5-1.5B \
RUN_TAG=paperqa_identity_check \
MAX_SAMPLES=100000 \
NUM_EPISODES=2 \
bash /root/code/Distributional-Match-Tuning/scripts/run_step1_paper_qa_feature_map.sh
```

## Review Points

- `MODEL_PATH` should be a complete local checkpoint, not tokenizer-only files.
- First run should stay at `FEATURE_MAP_TYPE=identity`.
- Do not change `n_samples_per_prompt` from `4` to `16`.
- Do not replace this with `scripts/run_G2_rebase_no_teacher_single_2rounds.sh`.
- `rollout_batch_size=16` and `n_samples_per_prompt=4` are different knobs and should not be merged conceptually.

## Optional Quick Sanity Run

If you only want a cheap environment check before the full reproduction, reduce budget only:

```bash
REPO_ROOT=/root/code/Distributional-Match-Tuning \
MODEL_PATH=/path/to/local/Qwen2.5-1.5B \
RUN_TAG=paperqa_identity_smoke \
MAX_SAMPLES=4096 \
NUM_EPISODES=1 \
bash /root/code/Distributional-Match-Tuning/scripts/run_step1_paper_qa_feature_map.sh
```

This smoke run is only for pipeline validation. It should not be used as the final paper-comparable result.
