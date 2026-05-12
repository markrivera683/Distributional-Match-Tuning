# Step 4: Feature Adapter 2-lite

This is the first implementation of the second major idea:

- frozen critic backbone
- small residual bottleneck adapter on the critic feature stream
- existing critic EMA reused as the target geometry
- no reward-formula rewrite

## New switches

- `--feature_adapter_enable`
- `--feature_adapter_type residual_bottleneck`
- `--feature_adapter_rank 64`
- `--feature_adapter_dropout 0.0`

## Current default training recipe

For the first stable version:

- `critic_learning_rate = 0`
- `critic_lr_head = 1e-5`
- `critic_classifier_loss_coef = 1.0`
- `enable_ema = true`
- `ema_beta = 0.99`

The optimizer head group now includes:

- `classifier_head`
- `feature_adapter`

The backbone remains frozen.

## Recommended first smoke

Use:

- [run_step2lite_cf_adapter_smoke.sh](/root/autodl-tmp/Energy/ebft_openrlhf_u1_ncfm/scripts/run_step2lite_cf_adapter_smoke.sh)

This runs the current mainline combination:

- first-point branch: `cf_l1oo`
- second-point branch: `feature adapter 2-lite`

## Safety properties

- The adapter is residual and zero-initialized on the up projection.
- At initialization it is exactly identity on the feature stream.
- EMA remains the reward/eval target when enabled.
- Reward branches stay pluggable:
  - `pointwise`
  - `cf_l1oo`
  - `cf_tokencloud_l1oo`
