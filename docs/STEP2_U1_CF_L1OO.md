# Step 2: Minimal NCFM-to-EBFT Distribution Matching

This branch is the first minimal distribution-matching replacement for vanilla EBFT.
It intentionally avoids learned spectral samplers, teacher targets, or adaptive feature geometry.

## Goal

Replace vanilla EBFT's pointwise reward construction

- alignment reward: generated feature vs GT feature
- diversity reward: generated feature vs other generated features

with a single group-level distribution discrepancy based on the empirical characteristic function (CF),
then convert that discrepancy into per-sample rewards with a leave-one-out marginal contribution rule.

## Implemented Variant

`distribution_reward_type=cf_l1oo`

Reward is built from:

1. A fixed random frequency bank `t ~ N(0, sigma^-2 I)`.
2. An empirical CF discrepancy between the generated sample set `X = {x_j}` and the target empirical set `Y`.
3. A leave-one-out marginal reward:

`r_j = D(X without j, Y) - D(X, Y)`

where lower `D` is better. Positive reward means sample `j` helps reduce the group discrepancy.

## Relation to NCFM

This variant borrows the core transferable idea from NCFM:

- compare distributions through characteristic-function statistics
- include both amplitude and phase terms
- keep the frequency bank fixed in the first version

It does **not** yet include:

- learned spectral sampling (`SampleNet`)
- minmax discrepancy learning
- multi-layer feature matching

## Target-Side Upgrade

The original minimal version uses a single-reference target measure.
This branch now also supports a more distributional target side:

- `CF_TARGET_MODE=single`: keep the original single GT feature
- `CF_TARGET_MODE=vicinal`: build a small local target distribution around the GT feature

The `vicinal` mode is a conservative kernel-smoothed empirical target:

- keep the exact GT feature as one target sample
- add `CF_TARGET_NUM_REFS - 1` deterministic Gaussian perturbations
- scale perturbations by the local feature RMS

This is not a teacher-based target and does not change the feature network.
It is the first minimal way to move the target side away from a Dirac measure.

## Code Paths

- CLI args: `openrlhf/cli/train_ebft_ray.py`
- CF reward utility: `openrlhf/utils/embedding_utils.py`
- Reward integration: `openrlhf/trainer/ppo_utils/ebft_experience_maker.py`
- Run script: `scripts/run_step1_paper_qa_feature_map.sh`

## First Comparison Protocol

Keep everything paper-aligned and identical to the vanilla baseline except:

- `DISTRIBUTION_REWARD_TYPE=cf_l1oo`

Optional first-pass knobs:

- `CF_NUM_FREQS=128`
- `CF_SIGMA=1.0`
- `CF_ALPHA=0.5`
- `CF_BETA=0.5`
- `CF_REWARD_SCALE=1.0`
- `CF_TARGET_MODE=vicinal`
- `CF_TARGET_NUM_REFS=4`
- `CF_TARGET_STD=0.05`

## Important Scope Boundary

This is the first simple replacement only.
If it shows any signal, later upgrades can add:

- better target distributions
- learned spectral samplers
- stronger discrepancy calibration
