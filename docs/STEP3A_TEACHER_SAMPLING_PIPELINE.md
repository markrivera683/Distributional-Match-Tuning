# Step 3A: Teacher Sampling & Embedding Pipeline

This step documents the concrete teacher data path that feeds target-side
distribution matching, before any broader token-level or feature-unfreeze work.

## 1. Goal

Build and stabilize the end-to-end path:

- prompt
- teacher rollout/sampling
- teacher completions `y_i^T`
- local critic feature extraction
- teacher embedding tensor
- target measure construction for reward.

This is a pipeline hardening step, not a trainer redesign step.

## 2. Pipeline

Canonical flow in current code:

- `prompt` / block context
- `teacher sampling` (local teacher actor or remote API provider)
- `teacher completion text/token ids`
- `full_sequences` assembly in strided/interleaved layout
- critic forward pass
- embed method reduction (`last_token`, `mean_pooling`, `concat`, `token`)
- optional feature map (`identity`/`rff`)
- `teacher_embedding` for target builder.

In symbols:

- `c -> y_i^T -> h(c, y_i^T) -> phi(c, y_i^T)`

with `phi` implemented by existing critic + embedding stack.

## 3. Key Components

### `teacher_model_group` (local backend)

- created in `train_ebft_ray.py` when `teacher_backend=local` and
  `teacher_pretrain` is provided;
- uses policy actor type for generation;
- initialized from `teacher_pretrain`.

### `SamplesGenerator`

- shared generator utility used for actor rollout and local teacher rollout;
- supports strided generation format expected by critic path.

### `RemoteExperienceMaker`

- central integration point for reward-side construction;
- owns:
  - `teacher_samples_generator` (local path),
  - `teacher_provider` (remote path),
  - `_build_teacher_embedding(...)` and target reward wiring.

## 4. Data Flow

Current teacher branch data flow in `RemoteExperienceMaker`:

1. deduplicate prompts by grouping actor samples with `n_samples_per_prompt`;
2. collect unique prompt tensors + `doc_ids` + `qa_masks`;
3. sample teacher outputs:
   - local: teacher actor rollout;
   - remote: API completion per block-context;
4. build teacher `Experience` batches in the same sequence layout expected by
   critic forward;
5. run critic forward on teacher sequences;
6. reshape and align to actor micro-batch/group layout.

Alignment target:

- final teacher tensor shape:
  - `(num_actor_mb, num_groups_per_mb, M, num_blocks, D)`

so downstream target builder can consume it without trainer changes.

## 5. Embedding Construction

Teacher embedding currently uses the same feature pipeline as student/GT:

- same critic network;
- same hidden-state extraction configuration;
- same `embed_method`;
- same optional whitening and feature map.

So feature encoder sharing is:

- **shared** (teacher, GT, and rollout are all mapped by the same local critic
  feature stack in current code).

This keeps target/rollout geometry comparable and avoids dual-encoder drift.

## 6. Current Status

### Implemented

- local teacher sampling path;
- remote teacher sampling path (text API);
- per-block remote context sampling path;
- teacher embedding construction and shape alignment;
- teacher target integration in `cf_l1oo` reward branch.

### Partially implemented

- teacher integration across all reward types:
  - active in `cf_l1oo`,
  - not active in `pointwise`,
  - not yet wired as teacher mode for `cf_tokencloud_l1oo`.
- experiment automation and robust benchmark protocol:
  - scripts exist,
  - still needs broader repeated evaluation package to finalize conclusions.

### Interface-level only / not implemented

- remote teacher logits/hidden-states ingestion;
- teacher-side uncertainty-aware weighting;
- cross-teacher ensemble path.

## 7. Failure Modes

### Teacher collapse / low-diversity target

- symptom: teacher samples become near-identical, reducing target support;
- impact: weak distributional signal, unstable attribution gains.

### Embedding mismatch

- symptom: teacher completions mapped to incompatible feature scale or layout;
- impact: misleading discrepancy values and noisy reward.

### Remote latency / timeout bursts

- symptom: per-step teacher sampling latency dominates training;
- impact: throughput collapse, retry overhead, poor reproducibility.

### Cache contamination

- symptom: stale or incorrect cache keying across contexts/configs;
- impact: wrong teacher completions reused, silently corrupting target measure.

### Block-structure mismatch

- symptom: completion placement not aligned to strided interleaving;
- impact: critic attends to wrong positions, invalid teacher features.

This Step 3A is considered complete only when these failure modes are
instrumented and can be diagnosed from logs without trainer code surgery.
