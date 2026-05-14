# Step 1 U1 Execution Plan

This note defines the execution ladder for the first major upgrade axis:

- richer discrepancy / feature-distribution matching

At this stage we do **not** touch:

- teacher targets;
- adaptive feature geometry;
- learned spectral samplers;
- black-box distillation interfaces.

## Goal

Establish a stable and well-instrumented path for `Upgrade 1` before any more
ambitious change.

The purpose of Step 1 is:

- verify that stronger fixed discrepancy features are trainable in the current
  EBFT pipeline;
- identify a strong and stable fixed-feature baseline;
- avoid mixing in any other source of complexity.

## Allowed variables

In Step 1, we only change variables that belong to the discrepancy / feature
axis.

Allowed:

- `feature_map_type`
- `rff_num_features`
- `rff_sigma`
- training budget for stronger verification
- random seed for stability checks

Locked:

- model family
- dataset
- target measure
- whitening on/off choice
- EBFT rollout structure
- actor / critic architecture
- feature network freeze status

## Recommended execution ladder

### Stage A: longer-budget pairwise comparison

Run two experiments with identical settings except `feature_map_type`:

1. `identity` long-budget baseline
2. `rff` long-budget baseline

This establishes whether fixed RFF remains stable beyond the smoke budget.

### Stage B: narrow RFF tuning

Only if Stage A is stable, tune:

- `rff_sigma` in a small grid, e.g. `0.5, 1.0, 2.0`
- `rff_num_features` in a small grid, e.g. `64, 128, 256`

Important:

- do not combine large `sigma` and `num_features` sweeps at once;
- vary one factor while keeping the other fixed.

### Stage C: repeatability check

Take the best Stage B candidate and repeat with one or two new seeds.

This stage is required before moving on to more complex discrepancy designs.

## Default model choice

For Step 1 tuning, default to:

- `/root/autodl-tmp/RWICL/models/Qwen3-1.7B`

Reason:

- fast enough for iteration;
- already known to run in the current local EBFT setup;
- reduces confounding while we are still debugging discrepancy behavior.

Only after identifying a stable best fixed-discrepancy setting should we
consider confirming on a larger local model such as `Qwen3-4B-modelscope`.

## Default stronger verification budget

Compared with the smoke scripts, the first stronger verification run should:

- increase `max_samples`;
- keep the same core training recipe;
- reduce eval frequency enough to avoid dominating run time.

Suggested first stronger budget:

- `max_samples = 32`
- `eval_max_samples = 8`
- `eval_down_max_samples = 8`
- `eval_steps = 4`
- `eval_down_steps = 4`

This is still small enough to iterate quickly, but large enough to tell whether
`U1-A` survives beyond a four-step smoke run.

## Metrics that matter for Step 1

Primary:

- downstream metrics
- `full_ce_loss`
- `full_perplexity`
- held-out reward metrics
- `mse`

Secondary:

- reward variance proxies
- `feature_map_reward`
- `std_reward`
- `zero_std_reward`
- `actor_grad_norm`

## Go / no-go rule for Step 1

We continue with fixed discrepancy development only if all of the following are
true:

1. the run completes end-to-end without new instability;
2. evaluation remains available throughout the run;
3. metrics are at least comparable in scale and usability to the current local
   baseline;
4. the candidate does not obviously degrade core validation behavior.

We do **not** require a large downstream gain at this stage. Step 1 first needs
to prove:

- trainability;
- stability;
- diagnosability;
- tunability.

## What comes after Step 1

Only after a best fixed-RFF version is identified should we consider:

- projected / sliced CF baselines;
- richer target measures;
- teacher-assisted targets;
- adaptive geometry scout runs.
