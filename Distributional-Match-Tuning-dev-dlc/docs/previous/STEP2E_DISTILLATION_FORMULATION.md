# Step 2E: Distribution Matching as Distillation

This note reformulates the current Step 2 teacher-target extension in explicit
distillation language, without changing the existing EBFT training scaffold.

## 1. Motivation

Current progression in this repo is:

- GT point target (vanilla / single-target);
- richer discrepancy (`cf_l1oo`);
- richer target candidates (`vicinal`, token-cloud, teacher).

Once teacher samples are introduced, the method should be read as:

- not only "GT-supervised RL-style post-training";
- but "distribution matching against a stronger target policy."

This is why Step 2E is needed: to make the distillation interpretation explicit
and operationally aligned with current code.

## 2. Objective Reformulation

For each context `c`:

- student rollout empirical measure:
  - `mu_{theta,c} = (1/n) * sum_{j=1..n} delta_{phi(c, y_j^S)}`
- teacher empirical measure:
  - `nu_c^T = (1/m) * sum_{i=1..m} delta_{phi(c, y_i^T)}`

Distillation-oriented target:

- match `mu_{theta,c}` to `nu_c^T` (or GT + teacher mixture when GT anchor is
  retained).

Current implementation-compatible form:

- `nu_c = (1-lambda) * delta_{phi(c, y_gt)} + lambda * nu_c^T`
- optimize CF discrepancy-derived rewards between generated measure and target
  measure.

## 3. Relation to Existing Methods

### SFT (CE)

- target: one token-level reference trajectory;
- objective: per-token likelihood matching.

### KD (KL)

- target: teacher token distribution (typically logits/probabilities);
- objective: divergence minimization in token space.

### RLHF (reward-model centric)

- target: scalar preference/reward signal;
- objective: policy improvement under reward model guidance.

### Vanilla EBFT

- target: GT-centered feature alignment + explicit diversity term;
- objective: feature-level alignment, not explicit teacher distribution match.

### Current method (Step 2 + teacher mode)

- target: feature-space empirical measure (GT and optional teacher samples);
- objective: distribution-level discrepancy reward (`cf_l1oo`) with leave-one-out
  sample attribution.

This places current method between classic RLHF and explicit generative
distillation:

- same policy-gradient training shell as EBFT;
- stronger distillation flavor at the target-measure level.

## 4. Reward Interpretation

In `cf_l1oo` mode:

- reward is not "does sample j match one label";
- reward is "how much sample j reduces distribution discrepancy."

Form:

- `r_j = D(X without j, Y) - D(X, Y)`

where `X` is student empirical set and `Y` is target empirical set.

Interpretation:

- CF discrepancy acts as a distribution-level reward functional;
- leave-one-out turns this functional into per-sample credit assignment;
- this is compatible with reward-weighted policy optimization while preserving a
  distillation-style target geometry.

## 5. Implementation Mapping

### `samples_list`

- contains student rollouts grouped by prompt;
- provides generated sequences and block structure.

### `embedding`

- critic hidden states are reshaped into `(B, G, N, K, D)` style tensors;
- teacher branch builds `(B, G, M, K, D)` through the same critic + feature-map
  stack.

### `reward computation`

- `get_cf_l1oo_rewards(...)` receives:
  - generated embeddings,
  - target embeddings from `_build_cf_target_embedding(...)` (single/vicinal/teacher).
- teacher mode enters through:
  - `teacher_embedding`,
  - `cf_teacher_lambda`,
  - target-side mixture construction in `embedding_utils.py`.

This mapping is already local to reward/experience maker modules; trainer and
optimizer contracts remain unchanged.

## 6. Current Gap

Current code status from a distillation viewpoint:

- teacher path exists and can enter reward construction;
- both local and remote teacher completion sources are wired;
- distribution-level reward with teacher target is functional in `cf_l1oo`.

What is still missing for a full distillation claim:

- a complete and stable distillation experiment package (budgeted comparisons,
  repeats, and documented decision criteria);
- broader objective variants (for example teacher-only target without GT anchor
  under controlled evaluation doctrine);
- token-space teacher signal integration (logprobs/logits) when needed.

So at this stage, teacher distillation is:

- **formally enabled in objective structure**,
- **not yet completed as a validated, finalized mainline experimental result**.
