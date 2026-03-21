# Step 2B: Target Measure Design for Distribution Matching in EBFT

This note clarifies what is missing from the current `cf_l1oo` variants and
what a better target-side design should look like for language tasks.

The main point is simple:

- `single-target cf_l1oo` gives us a better **discrepancy**
- but not yet a better **target distribution**

That is why it is a useful first step, but not yet a full NCFM-style
distributional upgrade.

## 1. What NCFM is really doing on the target side

NCFM does **not** compare a synthetic sample to a single reference sample.
Its target side is already an empirical distribution.

In the image setting, for each class it uses:

- a batch of real examples
- a batch of synthetic examples
- a CF-based discrepancy between the two feature distributions

So the target object in NCFM is naturally:

`nu = 1/m sum_i delta_{phi(x_i^real)}`

not:

`delta_{phi(x_gt)}`

This matters because the main strength of CF/MMD-style distribution matching is
not just the statistic itself. It is the combination of:

- a whole-distribution discrepancy
- applied to a target that is itself a distribution

## 2. What our current EBFT variants are doing

### 2.1 `single-target cf_l1oo`

Current target:

`nu_c = delta_{phi(c, y_gt)}`

Generated side:

`mu_{theta,c} = 1/n sum_j delta_{phi(c, y_hat_j)}`

So this is:

- generated distribution vs degenerate target measure

This is not wrong. It is already more distributional than vanilla EBFT because
the generated side is treated as a set/distribution and the reward is derived
from a group-level discrepancy.

But it is still not the full NCFM analogue.

### 2.2 `vicinal-target cf_l1oo`

Current target:

- exact GT feature
- plus Gaussian perturbations in feature space

So this becomes a small kernel-smoothed local target measure:

`nu_c^vicinal = 1/m sum_i delta_{z_i}`

with `z_1 = phi(c, y_gt)` and the others produced by local noise.

This is a reasonable engineering experiment, but it is still weaker than
NCFM-style real-data empirical targets because the extra samples are not real
conditional outputs. They are only local perturbations.

That is why it is plausible for this variant to be stable yet not clearly
better.

## 3. What a better language-task target measure should look like

For language generation, the best target-side object is not "arbitrary feature
noise around GT". It should be an empirical approximation to the conditional
output distribution under the task.

The ideal target is:

`q_c(y) = target conditional output distribution given context c`

and in feature space:

`nu_c = (phi(c, ·))_# q_c`

The practical question is how to approximate `q_c` without introducing too much
engineering or too many new assumptions.

## 4. Candidate target measures for language tasks

### Option A: Single reference target

Definition:

- one GT completion only

Pros:

- simplest
- cheapest
- stable
- good first ablation

Cons:

- target is still a Dirac point
- does not represent multi-modality
- weak approximation to NCFM's target side

Status:

- already implemented

### Option B: Vicinal feature-space target

Definition:

- GT feature plus local feature perturbations

Pros:

- easy to implement
- no extra model or data dependency
- useful as a local smoothing baseline

Cons:

- perturbations are not guaranteed to correspond to valid conditional outputs
- may blur semantics
- only a surrogate for richer targets

Status:

- already implemented

### Option C: Multi-reference data target

Definition:

- for each context, use multiple human-valid references when available

Pros:

- most direct NCFM analogue on supervised data
- no teacher dependency
- cleanest semantics

Cons:

- many datasets do not have multiple references
- often unavailable for code and open-ended tasks

Status:

- theoretically clean
- practically limited by datasets

### Option D: Output-augmentation target

Definition:

- generate multiple target variants from the same GT by safe semantics-preserving
  transformations

Examples:

- code formatting / identifier renaming where semantics are unchanged
- translation paraphrases from deterministic rewrite rules
- answer verbalization templates for narrow QA

Pros:

- cheaper than teacher models
- closer to "real target distribution" than Gaussian feature noise

Cons:

- task-specific
- augmentation quality is hard to guarantee
- harder to maintain across domains

Status:

- promising for code
- probably not the first generic implementation

### Option E: Teacher-augmented target measure

Definition:

- use a stronger model to sample multiple outputs conditioned on the same prompt
- combine them with GT into an empirical target measure

Example:

`nu_c = (1-lambda) delta_{phi(c, y_gt)} + lambda * 1/m sum_i delta_{phi(c, y_i^teacher)}`

Pros:

- closest practical analogue to NCFM's rich target side
- naturally captures multi-modality
- most likely to materially improve target-side distribution matching

Cons:

- adds model dependency
- quality depends on teacher reliability
- changes the method family toward teacher-augmented distillation

Status:

- probably the strongest next target-side upgrade
- but should come after the current no-teacher line is fully diagnosed

## 5. Recommended priority order

Based on current evidence, literature, and engineering practicality:

1. Keep `single-target cf_l1oo` as the minimal control.
2. Keep `vicinal-target cf_l1oo` only as a smoothing baseline.
3. Do not over-invest in better noise schedules for vicinal targets.
4. If target-side upgrade is pursued seriously, move to a **structured**
   empirical target measure rather than more sophisticated feature noise.

In other words:

- the current `vicinal` path is useful for diagnosis
- but it is unlikely to be the final target-side answer

## 6. What top literature suggests

The relevant lesson from top distillation and post-training papers is:

- richer targets help when they reflect genuine output uncertainty or
  multi-modality
- richer targets help much less when they are just synthetic noise around a
  single point

This pattern is consistent with:

- MiniLLM: on-policy generative distillation benefits from matching the teacher
  on student trajectories rather than reducing to a single reference
- OPCD / related on-policy distillation work: conditional target distributions
  matter more than offline one-shot imitation
- Entropy-aware on-policy distillation: uncertainty-aware targets are important
- Step-by-step / reasoning distillation work: multiple valid trajectories often
  matter, not just one final answer string

This strongly supports the view that:

- the target-side upgrade should eventually become a better empirical
  conditional measure
- not just a better local smoothing kernel

## 7. Engineering recommendation

If we stay within the current no-teacher phase, the right stance is:

- keep `single` as the main reference implementation
- keep `vicinal` as a low-cost auxiliary baseline
- do not treat `vicinal` as the likely final target-side design

If and when we do a serious target-side upgrade, the first strong version should
be one of:

- multi-reference empirical targets
- teacher-augmented empirical targets
- task-structured output augmentations

not "more complicated Gaussian perturbation logic"

## 8. Current practical decision

For the next stage of experiments:

- **mainline**: `single-target cf_l1oo`
- **auxiliary baseline**: `vicinal-target cf_l1oo`
- **future target-side upgrade to design seriously**:
  `teacher-augmented or multi-reference empirical target measure`

This keeps the project aligned with both:

- NCFM's true target-side insight
- good language-model engineering practice

