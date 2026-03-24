---
name: Cold-Start Teacher Verification
overview: "Execute a rigorous cold-start verification protocol: clear all teacher caches, run a fresh smoke test, confirm cache-miss → remote-fetch → target-construction → reward, then run a fake-teacher ablation to prove the output materially affects the target."
todos:
  - id: exp1-cold
    content: Run cold-start smoke test with cache DISABLED, verify all requests are fresh API calls with real latency
    status: completed
  - id: exp2-cache
    content: Re-run with cache enabled, verify first-call miss then hit pattern, compare teacher_embedding
    status: completed
  - id: exp3-fake
    content: Run fake-teacher ablation, compare teacher_mean and reward stats against Exp 1 to prove output matters
    status: completed
  - id: final-verdict
    content: Consolidate evidence from all 3 experiments and issue final verdict on target distribution status
    status: completed
isProject: false
---

# Cold-Start Teacher Target Verification

## Cache Audit (Read-Only Findings)

**Definition:** `TeacherCache` in `[openrlhf/utils/teacher_provider.py](openrlhf/utils/teacher_provider.py)` lines 57-132.

- Storage: SQLite DB at `{cache_dir}/teacher_cache.db`, one row per `(prompt, model, n, temp, top_p, max_tokens)` SHA256 key.
- No in-memory cache layer — only SQLite on disk.
- Each smoke run creates a unique `OUTPUT_ROOT`, so cache dirs are per-run (`$OUTPUT_ROOT/teacher_cache/`).
- The provider is constructed in `build_teacher_provider()` line 361-378. Cache is only created when `--teacher_cache_enable` is true.

**Current caches found:**

- `g2_smoke_fullq_0324_133758/teacher_cache/teacher_cache.db` — 4 entries (the most recent smoke)
- Plus 4 older caches from previous runs (irrelevant since each run has its own dir)

## Verification Protocol

### Experiment 1: Cold-Start (Cache Disabled)

1. Run the smoke test with `--teacher_cache_enable` removed (disabled), to a fresh output dir
2. Every teacher request MUST be a network fetch (zero cache hits possible since cache is disabled)
3. Add temporary diagnostic logging to `_get_remote_teacher_samples` and `_build_teacher_prompt` that prints:
  - Per-question: doc_id, question length, teacher response length, first 200 chars of response
  - Per-chunk: number of answer positions replaced by teacher tokens, token diff count before/after
  - The `teacher_embedding` tensor hash (to compare across runs)
4. Verify in logs: `cache_hits=0` for every `[RemoteTeacher] Done` line, plus real API latency (>1s per request, not <0.01s which would indicate cache)

### Experiment 2: Cache-Enabled Re-Run (Same Questions)

1. Re-run with `--teacher_cache_enable` on the same output dir from Experiment 1
2. Verify: first call is cache miss (fresh dir), subsequent runs would be hits
3. Compare: teacher_embedding tensor hash must match Experiment 1 (deterministic for same completions)

### Experiment 3: Fake-Teacher Ablation

1. Temporarily patch `_get_remote_teacher_samples` to replace ALL teacher completions with the string `"FAKE FAKE FAKE"` (a fixed nonsense string)
2. Run smoke test
3. Compare `teacher_embedding` tensor mean/std against Experiment 1 — must be significantly different
4. Compare `gt_rewards` mean against Experiment 1 — should differ
5. Revert the patch after the test

### Key Log Lines to Check

For each experiment, grep for:

- `[RemoteTeacher] Requesting` — confirms API call was made
- `[RemoteTeacher] Done: ... cache_hits=N` — N must be 0 in Exp 1 (no cache)
- `[Teacher-FullQ] ... example_answer` — shows actual teacher content
- `[TEACHER-TARGET] MIXED target built: ... teacher_mean=X` — X must differ between Exp 1 and Exp 3
- `[TEACHER-DIAG] === Final reward stats === gt_rewards mean=Y` — Y should differ between experiments

### Files to Temporarily Modify

- `[scripts/run_g2_remote_teacher_smoke.sh](scripts/run_g2_remote_teacher_smoke.sh)`: remove `--teacher_cache_enable` for Experiment 1
- `[openrlhf/trainer/ppo_utils/ebft_experience_maker.py](openrlhf/trainer/ppo_utils/ebft_experience_maker.py)` `_build_teacher_prompt`: add 5-line diagnostic log showing token replacement stats
- Same file, `_get_remote_teacher_samples`: for Experiment 3, add a 2-line override to replace completions with "FAKE"
- All temporary changes reverted after verification

