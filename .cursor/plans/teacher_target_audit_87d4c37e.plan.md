---
name: Teacher Target Audit
overview: Rigorous code-level and data-level audit of whether the remote teacher truly provides a target distribution in the current training pipeline.
todos: []
isProject: false
---

# Teacher Target Audit Report

## 1. Conclusion Summary

**Is the target truly from the remote teacher model?** YES — after the full-question fix, teacher completions are genuine full math answers from `qwen-122b`, tokenized, block-aligned, fed through the critic, and used in reward computation. The evidence chain is complete and verified.

**Is the target a "target distribution"?** PARTIALLY — it is a **multi-sample teacher empirical measure** with M=2 distinct samples per question, mixed with GT via lambda=0.5. This is a valid empirical target distribution in the NCFM/CF-L1OO sense, but with a significant caveat: the **cache freezes** the M samples, so the same question always produces the same 2 teacher completions across all training steps. The "distribution" is static, not re-sampled.

## 2. Complete Evidence Chain

### 2.1 From HTTP Response to Local Variable

```
teacher_provider.sample_targets(prompts, n_samples=2, ...)
  → _request_single(prompt, n_samples=2, ...)
    → POST http://172.17.0.26:8000/v1/chat/completions  {n: 2, ...}
    → resp.json()["choices"][0..1]["message"]["content"]
    → completions: List[str]  (2 text strings)
    → cache.put(prompt, ..., completions)   # stored as JSON list
    → return completions
  → results[idx] = fut.result()   # List[List[str]], shape [num_prompts][M]
```

Variable: `all_completions` in `_get_remote_teacher_samples()` (line 741-743 of `ebft_experience_maker.py`).

### 2.2 From Text to Token Replacement

```python
# Line 746-753: tokenize each completion
teacher_answers[did] = [
    tokenizer.encode(all_completions[q_idx][m_idx])   # List[int]
    for m_idx in range(M)
]

# Line 772-776: _build_teacher_prompt replaces answer region
# For each doc_id in chunk, for each of M samples:
result[m_idx, ans_start:ans_start+fill_len] = teacher_answer_tokens[:fill_len]
```

Variable: `teacher_prompt` — shape `(M, prompt_length)`, a tensor where answer tokens are replaced by teacher tokens.

### 2.3 From Teacher Prompt to Block Tokens to full_sequences

```python
# Line 782-800: extract block tokens from teacher prompt
for block_idx in range(K):
    start = block_idx * stride + context_length
    block_tok = teacher_prompt[m_idx][start:start+G]
# Interleave and concatenate with original prompt_ids
full_seqs.append(torch.cat([prompt_ids, gen_region]))
```

Variable: `batch_full_seqs` — shape `(M, prompt_length + G*K)`. This is an `Experience.full_sequences` tensor, identical in format to the actor's.

### 2.4 From Experience to Critic Hidden States to teacher_embedding

```python
# Line 996-1024: critic forward on teacher sequences
critic_ref = self.critic_model_group.async_run_method_batch(
    method_name="forward", sequences=t_full_seqs, ...)
hs_tensor = torch.stack(...)  # (P, M, full_seq_len, NF, H)

# Line 1029-1098: extract gen region, reshape into blocks, apply embed_method, feature_map
gen_emb = hs_tensor[:, :, prompt_length:, :, :]  # generation region only
# reshape → (P, 1, M, num_blocks, D)  →  (num_actor_mb, NG, M, K, D)
```

Variable: `teacher_embedding` returned from `_build_teacher_embedding()`.

### 2.5 From teacher_embedding to Target Measure to Reward

```python
# Line 1285 in make_experience():
teacher_embedding = self._build_teacher_embedding(...)

# Line 1331-1346: passed to get_cf_l1oo_rewards
gt_rewards_tensor = get_cf_l1oo_rewards(
    gen_embedding, gt_embedding,
    teacher_embedding=teacher_embedding,       # <-- HERE
    cf_teacher_lambda=0.5,
)

# In embedding_utils.py line 475-483:
target_embedding = _build_cf_target_embedding(
    gt_embedding, teacher_embedding=teacher_embedding, cf_teacher_lambda=0.5)

# Line 267-283: mixed target construction
r = round(m * (1.0 - lam) / lam)  # r=2 when lam=0.5, m=2
gt_repeated = target_embedding.expand(-1, -1, r, -1, -1)   # (B,G,2,K,D)
mixed = torch.cat([gt_repeated, teacher_float], dim=2)       # (B,G,4,K,D)
# → 4 points in target: 2 GT copies + 2 teacher samples

# Line 496-523: CF discrepancy with LOO
target_proj = torch.einsum("fd,bnd->bfn", freqs, target_flat)  # target_flat contains the 4 points
target_real = cos(target_proj).mean(dim=-1)  # empirical CF of 4-point target
# loo_loss - full_loss → per-sample reward
```

**The teacher embedding enters the `target_flat` variable in `get_cf_l1oo_rewards`, which directly controls the CF discrepancy target against which actor samples are scored.**

## 3. Target Type Judgment

**Classification: Multi-sample teacher-augmented empirical target measure**

Justification:

- The target is `nu = {delta(gt), delta(gt), delta(teacher_1), delta(teacher_2)}` — a 4-point empirical measure in feature space
- `teacher_1` and `teacher_2` are genuinely different completions from the remote teacher (verified from cache: different lengths, different content)
- The CF discrepancy `D(mu_gen, nu)` is computed between the actor's empirical distribution (n_samples=4 actor rollouts) and this 4-point target
- LOO attribution gives per-sample rewards based on marginal contribution to reducing `D`

This **does** constitute a legitimate distributional target. It is not a single-point target, not a pseudo-target, and not unused.

## 4. Caveats and Limitations

### 4.1 Cache Freezes the Distribution

The SQLite cache key includes `(prompt, model, n_samples, temperature, top_p, max_tokens)`. Once a question is cached, subsequent calls to `sample_targets` with the same parameters return the **same M completions forever**. This means:

- Step 1: question Q gets teacher samples `[T1, T2]` from API (cache miss)
- Step 2+: question Q gets the exact same `[T1, T2]` from cache (cache hit)
- The target distribution for Q is frozen across all training steps

**Impact:** The teacher contribution to the target measure is static. It does not evolve or re-sample during training. With M=2, this is a 2-point empirical measure from the teacher, fixed for the entire run.

**Is this a "distribution"?** Technically yes — it is a finite empirical distribution with 2 support points from the teacher + 2 copies of GT = 4 total points. But it is a frozen, low-support distribution.

### 4.2 M=2 Is Low but Non-trivial

With `cf_teacher_n_samples=2`, the target has 4 points total (2 GT + 2 teacher). This gives a non-degenerate empirical measure, but the distributional signal is weak. Increasing M to 4-8 would give a richer target.

### 4.3 Pure-Answer Chunks Have Weaker Teacher Signal

For packed chunks that contain only answer tokens (no question tokens), the "question" sent to the teacher is decoded from the answer portion. The teacher's response to these is less semantically appropriate than for chunks with genuine question text.

## 5. What Would Make It a Stronger Target Distribution

In order of priority:

- **Disable cache or use per-step sampling:** Remove or bypass the cache so each training step draws fresh M samples from the teacher. This makes the target distribution stochastic rather than frozen. (Easiest: add a `--teacher_cache_enable false` flag, already supported.)
- **Increase M:** Set `cf_teacher_n_samples=4` or `8` for a richer empirical measure. Each additional sample adds a support point to the target. Cost: more API calls but cached after first pass.
- **Set lambda=1.0 for pure-teacher target:** Currently `cf_teacher_lambda=0.5` mixes GT and teacher equally. Setting `lambda=1.0` makes the target purely teacher-derived. This is a config-only change.
- **Per-epoch re-sampling:** Clear the cache between episodes so the teacher's M samples are refreshed each epoch. Adds diversity without per-step API cost.

