## Step 3 Token-Level Prep

This note records the minimal, implementation-oriented plan for a token-level
version of the current `cf_l1oo` reward. The goal is not to introduce a new
research direction yet, but to prepare a safe next step if the current
sequence/block-level CF reward shows promise.

### 1. What the current code already supports

The existing EBFT code path is much closer to token-level support than it may
look at first sight.

- In `openrlhf/trainer/ppo_utils/ebft_experience_maker.py`, the critic hidden
  states are already reshaped into:
  `(num_micro_batches, num_groups, n_samples, num_blocks, generate_length, num_feat, hidden_size)`.
- `embed_method == "token"` is already recognized, and in that case the code
  intentionally keeps the token dimension instead of reducing it via
  `last_token`, `mean_pooling`, or `concat`.
- `get_alignment_rewards(...)` and `get_diversity_rewards(..., per_token=True)`
  already support token-level reward tensors.

So the main blocker is not the upstream hidden-state extraction pipeline. The
current blocker is that `cf_l1oo` explicitly disables token mode:

- `cf_l1oo currently supports non-token embeddings only`

### 2. What the current token-level gap actually is

The real gap is in the reward definition, not in the data plumbing.

Right now `get_cf_l1oo_rewards(...)` expects:

- `gen_embedding: (B, G, N, K, D)`
- `gt_embedding:  (B, G, N, K, D)`

For token-level, the natural input would instead be:

- `gen_embedding: (B, G, N, K, T, D)`
- `gt_embedding:  (B, G, N, K, T, D)`

So the missing part is a token-level CF discrepancy and its sample-wise reward
attribution.

### 3. Minimal token-level version we should build

The safest first version is:

- keep the current fixed-frequency CF machinery
- keep the current leave-one-out marginal attribution
- do **not** introduce token-token coupling or sequence-level reweighting yet
- compute the CF discrepancy independently at each token position

Concretely:

- for each `(b, g, k, t)`, compare the empirical CF of the `N` generated token
  features against the target token feature measure at that same `(b, g, k, t)`
- get a per-token leave-one-out reward
- return a tensor of shape `(B, G, N, K, T)`

This is the cleanest extension because it preserves the semantics of the
current reward and matches the existing PPO/token-advantage pathway.

### 4. Why this is the right first token-level version

This design is supported by both code structure and literature:

- The current EBFT code already has a token-level reward path.
- NCFM itself is not token-level; its transferable insight is CF-based
  distribution matching, not any specific spatial granularity.
- In language-model distillation, recent on-policy work emphasizes dense
  token-level supervision on the student's own trajectories rather than only
  coarse sequence-level signals. This supports trying token-level rewards once
  the sequence/block-level version is stable.

Relevant literature:

- MiniLLM (on-policy distillation via student trajectories)
- Self-Distilled Reasoner / OPSD (dense token-level on-policy supervision)
- Entropy-Aware On-Policy Distillation (token-level uncertainty matters)
- ToDi (token-wise divergence control matters; not all token positions should
  be treated identically)

These works do not imply that our token-level CF reward should be teacher-based
yet. They mainly support the idea that a denser token-level signal can improve
credit assignment once the training pipeline is stable enough.

### 5. What we should NOT do in the first token-level attempt

Do not combine the following into the first token-level patch:

- learned spectral sampler
- teacher target distribution
- token-dependent frequency sampling
- adaptive token weights based on entropy
- temporal smoothing across token positions
- sequence-token hybrid loss mixing beyond the existing PPO path

Those can come later if the minimal token-level version works.

### 6. Recommended implementation order

1. Add a token-capable CF reward function:
   - `get_cf_l1oo_rewards_token(...)`
   - or extend `get_cf_l1oo_rewards(...)` to accept 6D tensors

2. Keep the exact same CF ingredients:
   - fixed frequencies
   - same `alpha`, `beta`, `sigma`
   - same leave-one-out marginal reward

3. Keep target-side modes aligned with the current non-token path:
   - `single`
   - `vicinal`

4. Reuse the existing token-level PPO reward path without changing trainer
   semantics.

5. First compare against:
   - current `cf_l1oo` block/sequence-level version
   - not against a newly changed trainer

### 7. Success criteria for token-level prep

We should consider the token-level patch ready to test only if:

- it is a local reward-definition change, not a trainer rewrite
- it preserves output shapes expected by the existing PPO path
- it runs on the same smoke configuration as the current `cf_l1oo`
- it does not introduce new memory explosions relative to the current setup

### 8. Current recommendation

Do not implement token-level next. The priority order remains:

1. make the current `cf_l1oo` + `vicinal` target version stable and measurable
2. run fair small-budget comparisons against pointwise / single-target
3. only then promote token-level CF reward to the next concrete patch

