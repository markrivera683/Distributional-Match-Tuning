# Step 1 Paper-Aligned Q&A Alignment

This note defines the first rigorous comparison setting for `Upgrade 1`.

The immediate goal is not novelty. The immediate goal is:

- reproduce the **Q&A code** EBFT setting as closely as possible;
- hold the training recipe fixed;
- compare `identity` versus `rff` under the same paper-shaped setup.

This follows the experimental discipline used in strong empirical work:

- lock the baseline before introducing a new factor;
- isolate one axis at a time;
- only compare variants under the same budget and evaluation protocol.

For this step, the relevant reference is the EBFT paper itself:

- Table 2 in `2603.12248v1.pdf`
- `configs/qa_code.yaml`
- the frozen-critic note in `README.md`

## Paper target setting

The paper-aligned target for the Q&A code task is:

- model: `Qwen/Qwen2.5-1.5B`
- dataset: `sjelassi/opencode-instruct_100k_200tok`
- prompt max length: `1024`
- stride: `8`
- context max length: `8`
- generate max length: `8`
- samples per prompt: `4`
- rollout batch size: `16`
- training batch size: `64`
- micro train batch size: `8`
- micro rollout batch size: `8`
- micro reward batch size: `8`
- actor learning rate: `1e-6`
- temperature: `0.6`
- `advantage_estimator = rloo`
- `init_kl_coef = 0.0`
- `use_whitening = true`
- `enable_ema = true`
- `ema_beta = 0.9`
- `hidden_state_method = concat`
- `embed_method = last_token`
- `critic_sequence_level = last_token`
- frozen critic: `critic_learning_rate = 0`, `critic_lr_head = 0`
- base-run episodes: `2`
- CE / diversity sweep in the paper: `ce_loss_coef in {0.0, 0.03, 0.1}` and
  `diversity_rew_coef in {0.0, 0.5, 1.0}`

## What we fix first

Before any sweep, we fix one paper-valid point:

- `ce_loss_coef = 0.03`
- `diversity_rew_coef = 0.5`
- `alignment_rew_coef = 1.0`

Reason:

- this is a paper-valid interior point;
- it avoids using an edge case such as pure FM or zero diversity penalty;
- it gives a controlled baseline for `identity` versus `rff`.

## Allowed local compatibility deviations

These deviations are allowed because we are on a single-GPU local setup rather
than the paper's original training environment:

- `colocate_all_models = true`
- `adam_offload = true`
- `flash_attn = true`
- TensorBoard logging is used instead of external WandB dependence

These are **environmental** deviations, not method deviations.

They should remain fixed across all vanilla and `U1-A` comparisons.

## What must stay locked for the first paper-aligned comparison

Locked:

- dataset
- model family
- rollout structure
- whitening on
- frozen critic
- CE / diversity coefficients
- evaluation cadence
- seed
- training budget

Only changed:

- `feature_map_type`
- if `feature_map_type = rff`, then `rff_num_features`, `rff_sigma`, and
  `rff_seed`

## Run order

### First run

Run the paper-aligned baseline with:

- `feature_map_type = identity`

This establishes the local paper-shaped vanilla anchor.

### Second run

Run the same script with:

- `feature_map_type = rff`

No other variable should change.

### Third run

Only if both runs finish cleanly and the metrics remain interpretable:

- run one repeat seed for the better variant

## Entry script

Use:

- `scripts/run_step1_paper_qa_feature_map.sh`

Default behavior:

- paper-aligned Q&A code settings
- `feature_map_type = identity`
- `ce_loss_coef = 0.03`
- `diversity_rew_coef = 0.5`

To launch `U1-A`, override only:

- `FEATURE_MAP_TYPE=rff`

## Model availability rule

The script intentionally checks for actual model weight files.

We should **not** start the paper-aligned run with a partial cache that only
contains config and tokenizer files. That would create a false start and muddy
the experiment log.

## Go / no-go rule

We move past this step only if:

1. the `identity` paper-aligned run finishes end-to-end;
2. the `rff` paper-aligned run finishes end-to-end;
3. both runs emit the same train/eval/downstream metric families;
4. no new instability is introduced by the `rff` feature map.

At this stage, the question is still:

- is the discrepancy-axis change stable under paper-shaped conditions?

not:

- does it already beat the paper on every downstream metric?
