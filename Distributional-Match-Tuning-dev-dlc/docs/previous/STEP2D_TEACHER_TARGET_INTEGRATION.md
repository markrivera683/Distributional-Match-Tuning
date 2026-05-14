# Step 2D: Teacher-Augmented Target Measure Integration

This step is the direct continuation of Step 2 (`cf_l1oo`) and Step 2B/2C
(`target measure` design and token-cloud extension).

The purpose is not to replace the training system. The purpose is to upgrade
the target-side object in the existing distribution-matching pipeline.

## 1. Background

Current Step 2 lineage already established:

- Step 2 (`STEP2_U1_CF_L1OO.md`):
  - `distribution_reward_type=cf_l1oo` is implemented;
  - generated-side reward is group-level leave-one-out CF discrepancy.
- Step 2B (`STEP2B_TARGET_MEASURE_DESIGN.md`):
  - target-side diagnosis is explicit: `single` and `vicinal` are still
    GT-centered.
- Step 2C (`STEP2C_TOKEN_CLOUD_TARGET.md`):
  - richer no-teacher target via GT token cloud is implemented for
    `cf_tokencloud_l1oo`.

What remains limited in the Step 2 mainline:

- `single`: target is one GT point in feature space;
- `vicinal`: target is GT plus local perturbations;
- neither directly approximates a teacher conditional completion distribution.

## 2. Problem Statement

Even after `cf_l1oo`, the target-side object can still be:

- a Dirac point (`single`), or
- a local smoothed cloud around one GT point (`vicinal`).

This means we still do not model a richer conditional completion distribution
for the target side. The discrepancy is stronger, but the target measure can
remain sparse.

## 3. Teacher Target Formulation

Define teacher empirical target for context `c`:

- `nu_c^T = (1/m) * sum_{i=1..m} delta_{phi(c, y_i^T)}`

where `y_i^T` are teacher completions and `phi` is the existing feature
pipeline.

Use GT + teacher mixture:

- `nu_c = (1-lambda) * delta_{phi(c, y_gt)} + lambda * nu_c^T`

In implementation, we represent this mixture as repeated samples:

- repeat GT feature `r` times;
- concatenate `m` teacher features;
- effective weight `lambda_eff = m / (r + m)`.

Current code chooses `r` from `(m, lambda)` so `lambda_eff` approximates the
requested `lambda`.

Why `r` and `m` matter:

- `m` controls teacher-side empirical support size;
- `r` keeps GT anchor strength explicit;
- the ratio `m:(r+m)` encodes the practical target mixture.

## 4. Tensor Design

Following Step 2B conventions, target-side tensor becomes:

- `target shape = (B, G, r+m, K, D)`

where:

- `B`: micro-batch group count (current code often uses `num_micro_batches`);
- `G`: prompt groups per micro-batch;
- `r+m`: target empirical sample axis (GT repeats + teacher samples);
- `K`: block index;
- `D`: feature dimension after embed method + optional feature map.

Construction rule:

1. build GT canonical target `(B, G, 1, K, D)`;
2. repeat GT to `(B, G, r, K, D)`;
3. concatenate teacher embedding `(B, G, m, K, D)` on sample axis;
4. pass resulting `(B, G, r+m, K, D)` into existing CF reward.

Why downstream does not need a structural rewrite:

- `get_cf_l1oo_rewards(...)` already consumes target sample axis generically;
- CF statistics average over target sample dimension internally;
- no trainer-side optimizer or replay-buffer shape contract changes are needed.

## 5. Code Integration Plan

Primary integration points (current code-aligned):

- `openrlhf/cli/train_ebft_ray.py`
  - define teacher arguments:
    - `teacher_pretrain`, `cf_teacher_lambda`, `cf_teacher_n_samples`;
    - backend controls (`teacher_backend`, remote API knobs).
  - create `teacher_model_group` for local teacher backend.

- `openrlhf/trainer/ebft_trainer.py`
  - construct teacher sampling entry:
    - local path: `teacher_generator = SamplesGenerator(teacher_model_group, ...)`;
    - remote path: `teacher_provider = build_teacher_provider(args)`;
  - inject both into `RemoteExperienceMaker`.

- `openrlhf/trainer/ppo_utils/ebft_experience_maker.py`
  - build teacher samples:
    - local: teacher actor rollout;
    - remote: HTTP provider samples (block-context conditioned);
  - run teacher sequences through critic and embedding pipeline;
  - produce `teacher_embedding` with shape `(B, G, M, K, D)`;
  - pass into CF reward path.

- `openrlhf/utils/embedding_utils.py`
  - `_build_cf_target_embedding(...)`:
    - `cf_target_mode="teacher"` path;
    - GT repeat + teacher concat;
    - final mixed target measure returned to `get_cf_l1oo_rewards(...)`.

## 6. Current Status

Status snapshot against Step 2D scope:

- `teacher_model_group` construction: **implemented** (`local` backend) ✔
- remote teacher provider path: **implemented** (`remote` backend, text API) ✔
- teacher samples into target builder: **partially implemented** (only
  `cf_l1oo` teacher path) ◐
- teacher target into reward: **partially implemented** (active in
  `distribution_reward_type=cf_l1oo`; not used in `pointwise` and not wired
  into `cf_tokencloud_l1oo`) ◐
- full distillation experiment ladder (fair baselines + repeats + report):
  **not yet completed as a stable mainline package** ✗

Boundary note:

- teacher mode is available in code and can run end-to-end;
- project mainline is still the no-teacher distribution-matching baseline.

## 7. Minimal Working Version

Minimal Step 2D working recipe (without API teacher):

- backend: local open-weight teacher only;
- `cf_target_mode=teacher`;
- `distribution_reward_type=cf_l1oo`;
- `cf_teacher_n_samples = 2~4`;
- `cf_teacher_lambda` in a small grid (for example `0.25, 0.5, 0.75`);
- keep trainer/replay/optimizer logic unchanged.

What this MWV intentionally excludes:

- API teacher reliability engineering;
- teacher logits/hidden-state distillation;
- token-cloud teacher integration;
- feature-network unfreeze.

This keeps Step 2D tightly aligned with current Step 2 engineering doctrine:
upgrade one axis at a time, preserve ablation clarity, and avoid trainer
rewrites.
