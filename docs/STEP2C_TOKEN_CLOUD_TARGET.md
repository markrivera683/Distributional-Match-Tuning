# Step 2C: Token-Cloud Empirical Target

This is the next version that most directly tries to capture the practical
essence of NCFM without introducing teacher models.

## Core idea

Instead of comparing:

- generated sample embeddings vs one GT embedding

or:

- generated sample embeddings vs noisy perturbations around one GT embedding

we compare:

- the generated **cloud of token features** in a block
- against the GT **cloud of token features** in that same block

This gives us a target empirical measure that is richer than a Dirac point
without requiring multiple references or a teacher.

## Why this is closer to NCFM

NCFM's target side is powerful because it is an empirical distribution over
multiple real features. In our language setting, the cleanest no-teacher
analogue is:

- use all GT token features in the block as the target empirical set

So for each block:

- target set size = `T` token features
- generated set size = `N * T` token features, where `N` is the number of
  sampled completions

This is much closer to "real empirical distribution vs generated empirical
distribution" than the previous single-target or vicinal-target variants.

## Reward definition

For each `(b, g, k)` block:

- generated cloud:
  `X = {x_{j,t}}` for samples `j=1..N`, tokens `t=1..T`
- target cloud:
  `Y = {y_t}` for GT tokens `t=1..T`

We compute a fixed-frequency empirical CF discrepancy between `X` and `Y`, then
use leave-one-out marginal attribution at the **sample level**:

`r_j = D(X without sample j's token cloud, Y) - D(X, Y)`

So removing sample `j` means removing all of its token features in that block.

The result is still a reward tensor over `(sample, block)`, which lets us keep
the current PPO/replay-buffer/trainer semantics unchanged.

## Why this version is a good next step

- closer to NCFM target-side spirit
- no teacher dependency
- no trainer rewrite
- no token-level PPO rewrite
- local code change only in reward construction

## Current implementation choice

Reward mode:

- `distribution_reward_type=cf_tokencloud_l1oo`

Expected settings:

- `embed_method=token`
- `critic_sequence_level=last_token` can remain unchanged for now, because the
  token cloud is built from hidden states before the reward reduction step

## Scope boundary

This version is still intentionally conservative:

- fixed frequencies only
- no learned spectral sampler
- no teacher-augmented target
- no token-dependent weighting
- no temporal smoothing across positions

If this version works, then the next high-value upgrade is likely **teacher or
multi-reference target measures**, not more feature-space noise.

