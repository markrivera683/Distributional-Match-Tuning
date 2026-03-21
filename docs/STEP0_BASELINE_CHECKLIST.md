# Step 0 Baseline Checklist

This note defines the minimum baseline setup for the stepwise worktree.

The purpose of Step 0 is simple:

- get a stable and inspectable vanilla EBFT baseline;
- make sure train-time and eval-time metrics are both visible;
- do not introduce any new method changes yet.

## Scope

Step 0 is only about the current EBFT pipeline.

It is not yet about:

- fixed RFF discrepancy;
- projected CF discrepancy;
- teacher mixtures;
- adaptive feature geometry.

## Important caveat

This worktree inherits the current local code state from the original working
repo. That means it is a practical local baseline, not yet a pristine
"paper-original, zero-local-patch" reproduction.

Current local compatibility patches exist in at least:

- `openrlhf/models/actor.py`
- `openrlhf/models/critic.py`
- `openrlhf/models/ring_attn_utils.py`
- `openrlhf/trainer/ray/ebft_actor.py`
- `openrlhf/trainer/ray/launcher.py`
- `openrlhf/utils/deepspeed/deepspeed.py`
- `openrlhf/utils/deepspeed/deepspeed_utils.py`

So Step 0 should be described as:

- `current local EBFT baseline`

not:

- `fully pristine vanilla paper reproduction`

## Ground truth training entry

Primary training entry:

- `openrlhf/cli/train_ebft_ray.py`

Main trainer loop:

- `openrlhf/trainer/ebft_trainer.py`

Main reward construction:

- `openrlhf/trainer/ppo_utils/ebft_experience_maker.py`

Task-specific downstream eval:

- `openrlhf/trainer/ebft_eval_mixin.py`

## Metrics that must exist before any upgrade work

### Train-side

These come from the `status` dict logged at:

- `ebft_trainer.py`, `Global step ...`

Must capture:

- `reward`
- `effective_reward`
- `diversity_reward`
- `gt_reward`
- `feature_map_reward`
- `std_reward`
- `zero_std_reward`
- `return`
- PPO / actor-side optimization stats
- critic-side stats when present

### Eval-side

These are logged by `EBFTTrainer.evaluate(...)`.

Must capture:

- `eval_critic_loss`
- `reward_passk`
- `reward_pass1`
- `reward_passk_effective`
- `reward_pass1_effective`
- `reward_passk_diversity`
- `reward_pass1_diversity`
- `reward_passk_gt`
- `reward_pass1_gt`
- `full_ce_loss`
- `full_perplexity`
- `mse`

### Downstream eval-side

For GSM8K / MATH, must capture:

- `reward_down_passk`
- `reward_down_pass1`
- `response_type_llm_code_pct`
- `response_type_tinygsm_code_pct`
- `response_type_text_pct`

## Logging requirement

At least one of these must be enabled:

- TensorBoard
- WandB

For Step 0, TensorBoard is preferred because it is simpler and does not require
external state.

The run should also save:

- a plain text training log;
- the exact shell command or script used;
- the output checkpoint path.

## Minimum acceptance criteria for Step 0

We should not move to Upgrade 1 until the baseline satisfies all of the
following:

1. The run completes end-to-end without crashing.
2. Train-side global-step logs are present and readable.
3. Eval-side metrics are emitted at least once.
4. Downstream eval metrics are emitted at least once.
5. TensorBoard event files or equivalent logs are created.
6. We can point to one exact script as the reproducible baseline entry.

## Step 0 default run script

Use:

- `scripts/run_step0_vanilla_qwen3_smoke.sh`

This script is intentionally small-budget. Its job is not to prove method
quality. Its job is to prove the baseline path is runnable and diagnosable.

## What to do immediately after the first successful Step 0 run

Record:

- output directory;
- model path used;
- log path;
- TensorBoard path;
- key train metrics at the last step;
- key eval metrics at step 0 and final step;
- any instability symptoms.

Only after that should we begin U1-A.
