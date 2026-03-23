# Step 3B: From GT Target to Distribution Target

This note formalizes the target-side transition ladder in the current EBFT
distribution-matching branch.

## 1. Current GT Usage

Baseline target-side object in Step 2 mainline:

- `nu_c = delta_{phi(c, y_gt)}`

In code terms:

- target is built from the first GT embedding sample axis (`:1`);
- this is the `cf_target_mode=single` behavior.

This gives strong anchoring and low complexity, but target support is a single
point.

## 2. Vicinal Target

Vicinal mode upgrades target support locally:

- keep exact GT point;
- add deterministic Gaussian perturbations in feature space;
- scale perturbation by local feature RMS.

Form:

- `nu_c^vicinal = (1/m) * sum_i delta_{z_i}`
- `z_1 = phi(c, y_gt)` and other `z_i` are local perturbations.

This is a smoothing baseline, not a true conditional completion distribution.

## 3. Teacher Target

Teacher mode upgrades target from GT-local perturbation to empirical sampled
completions:

- `nu_c^T = (1/m) * sum_i delta_{phi(c, y_i^T)}`
- optional mixture with GT:
  - `nu_c = (1-lambda) * delta_{phi(c, y_gt)} + lambda * nu_c^T`

In current implementation:

- teacher samples are converted to embeddings through the same local critic
  stack;
- target tensor is built by GT repetition + teacher concatenation.

## 4. Unification

A single target-ladder view:

- `Dirac`  ->  `Vicinal`  ->  `Empirical Measure`

Concretely in this repo:

- `single`  ->  `vicinal`  ->  `teacher` (or token-cloud empirical target on GT tokens).

This ladder is useful because each step increases target support while keeping
the reward shell (`cf_l1oo` family) mostly unchanged.

## 5. Impact on Training

### Variance

- Dirac target: lowest target variance, highest bias toward one reference;
- Vicinal target: moderate variance, local smoothness;
- Teacher empirical target: higher variance but potentially better conditional
  support.

### Stability

- Dirac is easiest to stabilize;
- vicinal is usually stable if perturbation scale is controlled;
- teacher target stability depends on sample quality, sampling policy, and API
  reliability (for remote backends).

### Exploration

- Dirac target tends to tighten around one mode;
- vicinal adds local shape but not true semantic alternatives;
- teacher empirical targets can introduce multi-modal target structure and
  improve exploration pressure at distribution level.

## 6. Current Code Status

Status by mode:

- `single` target: **implemented and mainline-used** ✔
- `vicinal` target: **implemented** ✔
- `teacher` target in `cf_l1oo`: **implemented but not yet finalized as
  project mainline** ◐
- token-cloud empirical GT target (`cf_tokencloud_l1oo`): **implemented** ✔
- teacher-enhanced token-cloud target: **not implemented** ✗

Is teacher truly part of this target-transition layer?

- yes, structurally:
  - it changes the target measure sample axis in `_build_cf_target_embedding`;
- but only for the `cf_l1oo` non-token branch at present.

So teacher currently belongs to:

- **implemented target-side transition mechanism**,
- **partial method coverage across reward variants**.
