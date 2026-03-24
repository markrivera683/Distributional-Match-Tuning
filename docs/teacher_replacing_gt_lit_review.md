# Replacing Ground Truth with Teacher Completions in EBFT

This note is exploratory. It does not record a confirmed method change. It
records the current literature-backed view on replacing ground-truth completion
targets with teacher model outputs in EBFT.

## Question

EBFT currently depends on a ground-truth completion `y`.

Can we replace `y` with outputs from:

- a closed-source frontier teacher;
- or a very large open(-weight) teacher;

so that the method becomes not only fine-tuning, but also a stronger form of
teacher-to-student distillation?

## Short Answer

Yes, this direction is strongly supported by the literature.

But the literature also makes one point very clear:

- the simple version, `SFT on one teacher answer`, works surprisingly well;
- the stronger version is to match the teacher's distribution, not just one
sampled completion;
- on-policy teacher-student interaction matters a lot when generation-time
mismatch is important.

For EBFT specifically, this means:

- replacing ground truth with a single teacher completion is feasible, but it is
the weakest form of this idea;
- the more principled version is to replace the empirical point target with a
teacher conditional distribution target.

This aligns naturally with our first planned upgrade from point-centered feature
alignment to distribution-level feature matching.

## What the Literature Says

### 1. Teacher-generated supervision is already a strong paradigm

There is broad evidence that large-teacher-generated responses can effectively
train smaller or weaker students.

- `InstructGPT` shows that high-quality demonstrations can substantially improve
a much smaller aligned model.
- `Llama 2` uses SFT and RLHF, and also reports practical use of context
distillation in safety pipelines.
- `Distilling Step-by-Step` shows that teacher-generated rationales can improve
sample efficiency and student performance beyond standard teacher-label
distillation.

Implication for EBFT:

- replacing human ground truth with stronger teacher outputs is not unusual;
- it can be a real capability upgrade, not just data augmentation.

### 2. Naive sequence-level distillation is useful but limited

Recent distillation literature argues that plain SFT on teacher-generated
responses underuses the teacher.

- `MiniLLM` argues that generative LLM distillation benefits from on-policy
reverse-KL rather than standard teacher-forced distillation alone.
- `Distribution-Aligned Sequence Distillation (DASD)` explicitly critiques the
common practice of sequence-level distillation for failing to capture the full
teacher output distribution and for suffering from exposure bias.
- `DLCoT` shows that reasoning distillation quality depends heavily on the
structure and transferability of teacher traces.

Implication for EBFT:

- simply swapping `y` for one teacher answer is a valid baseline, but it is not
the strongest version of the idea.

### 3. On-policy distillation is especially relevant to EBFT

EBFT is already trying to address rollout mismatch. This makes on-policy
distillation literature highly relevant.

- `MiniLLM` is a direct precedent for on-policy distillation in generative LMs.
- `On-Policy Context Distillation (OPCD)` is even closer: the student learns on
its own trajectories while matching a stronger context-conditioned teacher.
- `Black-Box On-Policy Distillation` shows that this remains possible even when
the teacher is proprietary and only text outputs are accessible.

Implication for EBFT:

- if we move to teacher-based targets, the best formulation is probably not
offline teacher answer replacement;
- the more natural fit is a student-rollout-conditioned distillation objective.

### 4. Teacher choice matters a lot

Recent reasoning-distillation work makes it clear that "a correct answer" is not
the full story.

- `Not All Correct Answers Are Equal` shows that different teacher sources yield
meaningfully different distilled students.
- `Merge-of-Thought Distillation` shows that multiple teachers can raise the
ceiling beyond single-teacher distillation.
- `DLCoT` finds that trace transfer quality depends on architecture and data
homology.

Implication for EBFT:

- replacing ground truth with teacher outputs is not only about stronger labels;
- teacher identity and teacher distribution quality are first-order design
choices.

## What This Means for EBFT

Replacing ground-truth completion `y` with a teacher completion changes the
method identity in an important way:

- current EBFT: supervised feature-aligned fine-tuning;
- upgraded EBFT: teacher-conditioned distillation in feature space.

This is conceptually valuable because it broadens the method:

- from dataset-dependent post-training;
- toward distillation from stronger policies.

## Four Concrete Variants

### Variant A: Single teacher completion replaces ground truth

Use one teacher response `y_T` in place of `y`.

Pros:

- simplest possible implementation;
- works with both closed and open teachers;
- useful as a baseline.

Cons:

- still point-centered;
- still collapses teacher distribution to one sample;
- weakest alignment with the true spirit of distillation.

### Variant B: Multiple teacher completions approximate a teacher distribution

For each context `c`, draw multiple teacher samples and treat them as an
empirical approximation to `q_T(y | c)`.

Pros:

- directly supports distribution-level matching;
- naturally aligns with the planned NCFM-inspired upgrade;
- lets us model multi-modality, uncertainty, and teacher diversity.

Cons:

- expensive for closed-source teachers;
- introduces teacher-sampling policy design questions.

### Variant C: Open-source white-box teacher distillation

Use a large open teacher and access:

- text samples;
- token probabilities / logits;
- optionally hidden states.

Pros:

- strongest supervision;
- easier to do on-policy teacher-student interaction;
- enables hybrid feature-level + token-level distillation.

Cons:

- requires large compute;
- teacher family mismatch may still matter.

### Variant D: Closed-source black-box teacher distillation

Use a proprietary teacher with only API text access.

Pros:

- potentially strongest raw capability source;
- most practically valuable if it works well.

Cons:

- only sequence-level outputs by default;
- expensive and nondeterministic;
- harder to recover a good approximation of the teacher distribution;
- harder to debug and reproduce.

## Current Hypothesis

The strongest version of this idea is not:

- `replace ground truth with one teacher answer`.

It is:

- `replace the empirical single-reference target with a teacher conditional distribution target`.

This means the third upgrade is deeply compatible with the first upgrade:

- Upgrade 1 asks for distribution-level matching instead of point-centered
matching.
- Upgrade 3 provides a stronger target distribution than the empirical
one-sample ground-truth target.

These two upgrades reinforce each other.

## Current Recommendation

If we pursue this line, the best first serious version is likely:

- start with a very strong open teacher;
- use multiple teacher samples per prompt or white-box logits if available;
- avoid defining the method around one teacher completion only;
- treat closed-source teacher distillation as an important later extension, not
the first research prototype.

Why this order makes sense:

- open teachers give us better control, reproducibility, and richer signals;
- they make it easier to distinguish algorithmic gains from API noise;
- once the method is validated, closed-source teachers become a stronger
practical extension.

## Stronger Interpretation

The literature increasingly suggests that the interesting question is not:

- `can a stronger teacher replace the ground-truth answer?`

It is:

- `can EBFT learn from a stronger teacher-conditioned target distribution?`

This distinction matters because a single teacher answer is only a lossy sample
from a richer conditional distribution.

For our setting, the third upgrade is strongest when it is framed as:

- `distillation-target upgrade`, not only `label-source replacement`.

That makes it naturally compatible with the first upgrade:

- Upgrade 1 asks EBFT to move from point-centered alignment to
distribution-level matching.
- Upgrade 3 provides a more capable target distribution than a single
human-written completion.

Together, these two upgrades make more sense than either one in isolation.

## What the Recent Distillation Literature Adds

The newer papers sharpen three points that are especially relevant to EBFT.

### A. One-answer sequence distillation is a useful baseline, but it wastes

teacher information

`DASD` explicitly revisits the common practice of SFT on teacher-generated
responses and argues that plain sequence-level distillation is not enough for
strong long-CoT transfer.

`MiniLLM` and `OPCD` point in a similar direction:

- teacher distribution shape matters;
- on-policy matching matters;
- reverse-KL-style objectives are often a better fit than standard
teacher-forced imitation for generative models.

### B. Teacher source quality is a first-order variable

`Not All Correct Answers Are Equal` makes the point especially clearly:

- even when outputs are verified as correct, the source teacher still matters;
- different teachers induce different answer length, diversity, and difficulty
profiles;
- these differences materially change student quality.

So for EBFT, teacher choice is not a secondary implementation detail. It is a
central algorithmic component.

### C. Closed-source black-box distillation is now realistic, but it is not the

best first prototype

`Black-Box On-Policy Distillation` shows that a proprietary teacher can still be
used in an on-policy setup.

This is important because it means the third upgrade is not restricted to
open-weight teachers.

But for a first research prototype, the literature still pushes us toward open
teachers because they offer:

- logits or probabilities;
- multi-sample generation at lower marginal cost;
- better reproducibility;
- easier ablations and debugging.

## Current Model-Sourcing View

As of March 17, 2026, the cleanest split is:

- `open-weight teacher`: use a very large openly released model;
- `closed/API teacher`: use a frontier hosted model with text-only or API-only
access.

For Qwen specifically, the most clearly documented open-weight flagship is
`Qwen3-235B-A22B`, announced on April 29, 2025. The official Qwen3 release post
states that this 235B-total / 22B-activated MoE model is open-weight, alongside
smaller Qwen3 models.

Separately, `Qwen2.5-Max` was announced on January 28, 2025 as an API-available
large-scale MoE model. That makes it conceptually closer to a closed-source or
hosted black-box teacher from our perspective, even though it comes from the
same broader model family.

So if we say:

- `open-source large teacher in the Qwen family`,

the clearest current candidate is:

- `Qwen3-235B-A22B`.

If we say:

- `very strong hosted Qwen teacher`,

then `Qwen2.5-Max` is a realistic example.

## Main Risks

The literature suggests several real risks:

- `teacher over-stylization`: the student copies surface style instead of
underlying competence;
- `distribution narrowing`: one-sample distillation teaches one mode and loses
uncertainty;
- `exposure bias`: teacher-forced targets do not match student rollout dynamics;
- `teacher mismatch`: not all teachers transfer equally well to all students;
- `cost and reproducibility`: especially severe with closed-source APIs.

## Not Yet Decided

The following remains open:

- whether the student should match teacher outputs in feature space only, token
space only, or both;
- whether teacher signals should replace ground truth entirely or coexist with
it;
- whether the best teacher target is one best-of-k response, multiple sampled
responses, or student-trajectory-conditioned teacher supervision.

## Working Recommendation For Our Project

If we actually implement this line later, the most defensible order is:

1. Start from a very strong open teacher.
2. Use multiple teacher samples per context whenever feasible.
3. Keep ground-truth data as an anchor in early experiments, rather than
  deleting it immediately.
4. Only after the method is stable, test pure teacher replacement and
  black-box teachers.

Why this order is attractive:

- it separates `algorithm gain` from `teacher-source noise`;
- it gives us direct access to richer supervision than text strings alone;
- it lets us study whether EBFT should distill `one answer`, `a distribution`,
or `a hybrid of teacher and human targets`.

## References

- InstructGPT: [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
- Llama 2: [https://arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288)
- Distilling Step-by-Step: [https://arxiv.org/abs/2305.02301](https://arxiv.org/abs/2305.02301)
- MiniLLM: [https://arxiv.org/abs/2306.08543](https://arxiv.org/abs/2306.08543)
- OPCD: [https://arxiv.org/abs/2602.12275](https://arxiv.org/abs/2602.12275)
- Black-Box On-Policy Distillation: [https://arxiv.org/abs/2511.10643](https://arxiv.org/abs/2511.10643)
- DASD: [https://arxiv.org/abs/2601.09088](https://arxiv.org/abs/2601.09088)
- DLCoT: [https://arxiv.org/abs/2503.16385](https://arxiv.org/abs/2503.16385)
- Not All Correct Answers Are Equal: [https://arxiv.org/abs/2505.14464](https://arxiv.org/abs/2505.14464)
- Merge-of-Thought Distillation: [https://arxiv.org/abs/2509.08814](https://arxiv.org/abs/2509.08814)
- Qwen3 official blog: [https://qwenlm.github.io/blog/qwen3/](https://qwenlm.github.io/blog/qwen3/)
- Qwen2.5-Max official blog: [https://qwenlm.github.io/blog/qwen2.5-max/](https://qwenlm.github.io/blog/qwen2.5-max/)

