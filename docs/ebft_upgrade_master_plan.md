# EBFT Upgrade Master Plan

This note is the internal source of truth for how we should present,
prioritize, and execute our upgrade plan for EBFT.

It is not a brainstorming dump. It is a method-facing research plan designed to
keep theory, literature, and engineering pointed at the same object.

The goal is not to stack modifications. The goal is to turn EBFT into a more
principled and more general method family.

## 1. One-sentence thesis

Our upgraded EBFT should be understood as:

- `distribution-level, teacher-extendable, stability-aware feature-space policy optimization`.

There is also a more theorem-facing version of the same claim:

- `vanilla EBFT can be viewed as conditional first-moment matching in a frozen pretrained feature geometry; our project generalizes it by upgrading the discrepancy class, the target measure, and eventually the feature geometry`.

The first sentence is how we should talk when we need a compact high-level
identity. The second sentence is how we should talk when we need reviewer-proof
mathematical precision.

## 2. A unified mathematical frame

The cleanest way to organize the project is to define one family of objectives.

For each context `c`, define:

- `q_c(y)` as the target conditional measure;
- `z_eta(c, y) = phi_eta(c:y)` as the feature geometry;
- `mu_{theta,c}` as the pushforward feature distribution induced by student
  rollouts `y_hat ~ p_theta(.|c)` through `z_eta`;
- `nu_c` as the pushforward target feature distribution induced by
  `y ~ q_c` through `z_eta`.

Then the family-level objective is:

- `L(theta; eta, psi) = E_c [ D_psi(mu_{theta,c}, nu_c) ] + regularizers`.

In this frame, the three upgrade axes become cleanly orthogonal:

- Upgrade 1 changes `D_psi`, the conditional discrepancy family;
- Upgrade 3 changes `q_c`, the target conditional measure;
- Upgrade 2 changes `z_eta`, the feature geometry.

This is the most important conceptual cleanup. It turns the project from "EBFT
plus three ideas" into a single family of methods.

## 3. The central diagnosis of current EBFT

Vanilla EBFT is already a serious step beyond token-level CE:

- it trains on generated trajectories rather than only teacher-forced prefixes;
- it optimizes a feature-space signal rather than exact token imitation;
- it explicitly tries to balance correctness and diversity.

But it still has three structural limitations.

### 3.1 The discrepancy family is still weak

The most precise way to describe vanilla EBFT is not:

- `not distributional at all`.

The more precise description is:

- `conditional first-moment matching in a frozen feature geometry`.

This matters because it protects us from an easy reviewer objection. EBFT is
already a form of sequence-level feature-distribution calibration, but it lives
in a weak discrepancy family centered on first moments and a fixed witness
space.

In a useful mathematical reading, it is close to a linear-kernel or
first-moment IPM / MMD-style conditional discrepancy on the induced feature
distribution.

So our claim should not be:

- `EBFT is not distributional`.

It should be:

- `EBFT is a weak form of conditional distribution matching, and we want to
  upgrade the discrepancy family beyond first-moment matching`.

### 3.2 The target source is still too narrow

Current EBFT uses a single empirical ground-truth completion `y`.

This makes the target:

- single-reference;
- dataset-bounded;
- too sparse to support richer conditional target estimation once the
  discrepancy becomes stronger.

### 3.3 The feature geometry is still fixed

Current EBFT relies on a frozen feature network.

This is stable and clean, but it likely limits the ceiling because the reward
geometry cannot adapt to the policy's actual failure modes.

## 4. The right overall framing

The best way to present our plan is not:

- `three separate improvements to EBFT`.

The best framing is:

- `one core method upgrade`;
- `one target upgrade that naturally composes with it`;
- `one metric-learning upgrade that should be added only after the first two are
  stable`.

That gives us a much cleaner story.

## 5. The upgraded method stack

### 5.1 Core upgrade: move from first-moment matching to richer conditional
distribution discrepancies

This is the first and most important change.

The method should no longer be described as:

- `feature mean alignment plus explicit diversity regularization`.

It should be described as:

- `matching richer conditional feature distributions through a stronger
  discrepancy family`.

This is where NCFM matters. The transferable insight is not image distillation.
It is:

- characteristic functions represent whole distributions;
- discrepancy should be distribution-level by construction;
- diversity should emerge from the discrepancy itself, not be bolted on as a
  separate repulsion term;
- adaptive observation spaces can be more powerful than a fixed weak kernel.

This is the main conceptual upgrade. Everything else is downstream of it.

The safest execution order inside this upgrade is itself two-stage:

- `Stage 1`: fixed spectral or kernelized discrepancy, such as projected CF,
  sliced CF, or random-Fourier-feature MMD-style matching;
- `Stage 2`: adaptive spectral discrepancy, such as learned projection or
  learned frequency sampling in the spirit of NCFM.

This staged version matters because EBFT already has one source of variance from
discrete student rollouts and REINFORCE-style updates. A fully adaptive spectral
minmax objective is promising, but not the right first shot.

### 5.2 Target upgrade: move from single-reference supervision to
teacher-extendable target distributions

Once EBFT is distribution-level, the next question is:

- `what target distribution should it match?`

The strongest answer is not:

- `a single human completion`;
- and not even `a single teacher completion`.

The strongest answer is:

- `a conditional target distribution`, which may come from human references,
  teacher samples, white-box teacher probabilities, or a hybrid target.

This is why the third idea should be presented as a natural extension of the
first one.

Upgrade 1 changes the geometry of the objective.
Upgrade 3 strengthens the source of the target.

Together, they turn EBFT from:

- `feature-aligned fine-tuning with one reference`

into:

- `teacher-extendable distributional feature-space distillation`.

There is also a practical reason this upgrade may be more than an extension.
If the discrepancy becomes richer, a single-reference empirical target may be
too sparse and too noisy to estimate a genuinely distributional objective well.

So in practice, Upgrade 3 may act as an enabling condition for Upgrade 1 at
full strength, even if the paper story still presents it as the second stage.

### 5.3 Metric-learning upgrade: move from frozen feature geometry to
stability-aware adaptive feature geometry

This is the most delicate change and should not be sold as an immediate core
ingredient.

The right question is not:

- `should we unfreeze the feature network?`

It is:

- `what constrained form of feature adaptation improves discrepancy power
  without making the reward unreliable?`

This upgrade should be treated as:

- higher ceiling;
- higher risk;
- later in the execution order.

The cleanest internal split is:

- `2-lite`: keep the backbone frozen and learn only a small projection head,
  adapter, or LoRA module with an EMA target copy;
- `2-full`: only after 2-lite works, consider limited backbone adaptation such
  as upper-layer tuning or alternating freeze/unfreeze schedules.

## 6. The best presentation order

If we present the plan in the wrong order, it will look like a bag of tricks.

The strongest presentation order is:

1. `Fix the discrepancy geometry.`
2. `Upgrade the target measure.`
3. `Only then relax the frozen metric assumption.`

This order is best for both theory and engineering.

### 6.1 Why Upgrade 1 should come first

If we do not first make EBFT genuinely richer in its discrepancy family, then
later using a teacher still leaves us with a method that mostly imitates a
stronger point or weak-moment target.

That would miss the main opportunity.

### 6.2 Why Upgrade 3 should come before Upgrade 2 in serious method
development

Teacher-based targets make the method broader and stronger while preserving a
mostly stable discrepancy substrate.

This means they increase method power without immediately introducing moving
reward geometry.

That makes causal attribution cleaner:

- if results improve, we know it is not only because the reward geometry was
  allowed to drift.

### 6.3 Why Upgrade 2 should come last

Adaptive feature geometry is promising, but it is also the easiest way to turn
the method into an unstable moving-target system.

The literature from self-supervised learning, discrepancy learning, and RLHF all
suggests the same thing:

- learned metrics can help;
- naive jointly moving targets often break reliability.

So we should only introduce this after we already know that the richer
discrepancy and richer target logic work.

## 7. The cleanest method identity

If the project matures successfully, the strongest identity statement is
something like:

- `EBFT upgraded from conditional first-moment matching in a frozen feature geometry to teacher-extendable conditional feature distribution matching`.

Only after the adaptive-feature variant is validated should we add language
like:

- `with stability-aware learned feature geometry`.

That wording matters. It keeps the paper centered on the method's real
conceptual contribution rather than on every engineering change we tried.

## 8. What we should explicitly avoid

There are several failure modes in presentation and design.

### 8.1 Do not present Upgrade 1 as a cosmetic reward rewrite

We should not say:

- `we replaced cosine similarity with a fancier discrepancy`.

That is too shallow. The real change is:

- `from first-moment conditional matching to a richer conditional discrepancy
  family`.

### 8.2 Do not present Upgrade 3 as just better labels

We should not say:

- `we replaced ground truth with better answers`.

That misses the key idea. The real change is:

- `from empirical single-reference targets to stronger conditional target
  measures`.

### 8.3 Do not present Upgrade 2 as straightforward end-to-end unfreezing

We should not say:

- `we simply unfreeze the feature network and learn a better reward`.

That is both methodologically weak and engineering-wise dangerous.

The right framing is:

- `controlled metric adaptation under explicit stabilization`.

## 9. The recommended execution plan

### Phase A: establish the core discrepancy upgrade

Goal:

- replace first-moment EBFT reward design with a genuinely richer conditional
  discrepancy in feature space.

Recommended order inside this phase:

- start with fixed projected or sliced spectral features;
- compare against fixed random Fourier or kernel baselines;
- only after that try learned spectral samplers or adaptive discrepancy nets.

Success means:

- diversity is no longer only an external penalty;
- the objective is distributional by construction;
- the new objective can be implemented and ablated cleanly against vanilla
  EBFT.

### Phase B: add teacher-conditioned target distributions

Goal:

- replace or augment single-reference targets with stronger teacher-derived
  conditional targets.

Recommended order inside this phase:

- first use a strong open-weight teacher;
- prefer multiple teacher samples or white-box teacher probabilities when
  feasible;
- keep ground-truth supervision as an anchor in early experiments;
- only later test pure teacher replacement and black-box teachers.

Success means:

- the method now clearly spans both fine-tuning and distillation settings;
- the gains are attributable to better target distributions, not only to metric
  drift.

### Phase C: explore adaptive feature geometry

Goal:

- test whether controlled feature-network adaptation raises the ceiling further.

Recommended first variant:

- frozen backbone plus small trainable geometry head;
- target encoder for reward computation;
- EMA updates;
- anchoring to initialization;
- explicit anti-collapse regularization;
- slower update timescale than the actor.

Success means:

- gains appear without obvious reward hacking or collapse;
- reward statistics remain stable;
- the adaptive variant beats the best frozen-feature version, not only the
  vanilla baseline.

## 10. Estimator design requirements

The plan is not complete unless it also specifies the estimator layer.

At minimum, the future method note must explicitly answer:

- how many student rollouts are drawn per context;
- how many target samples are drawn per context when `q_c` is teacher-augmented;
- how many projections, directions, kernels, or frequencies define `D_psi`;
- whether the discrepancy estimate is biased, unbiased, or U-statistic-based;
- how leave-one-out baselines are defined under the new discrepancy;
- how REINFORCE variance is controlled when the discrepancy becomes richer.

Without this layer, the plan remains a good narrative but not yet a complete
method.

## 11. The engineering doctrine behind the plan

This plan is not only about theory. It also reflects what usually works in
practice.

### 11.1 Separate improvements that change different causal factors

If we change:

- discrepancy geometry;
- target source;
- and reward substrate

all at once, we will not know why the method improved or failed.

So the phases are not only cleaner for writing. They are cleaner for science.

### 11.2 Prefer the strongest stable baseline before adaptive components

In practice, the most reliable sequence is:

- first make the static-objective version strong;
- then add stronger supervision;
- only then add moving-target components.

This is consistent with both RLHF engineering and representation-learning
practice.

### 11.3 Preserve an interpretable ablation ladder

At every stage, we should be able to compare:

- vanilla EBFT;
- discrepancy-upgraded EBFT;
- discrepancy-upgraded EBFT plus teacher targets;
- discrepancy-upgraded EBFT plus teacher targets plus adaptive feature geometry.

If that ladder is broken, the final story will be weaker.

## 12. Evaluation doctrine

If the method claims to improve distribution-level matching, evaluation should
not live only on single-answer or low-entropy tasks.

The strongest evaluation doctrine is:

- keep exact-answer tasks for comparability;
- elevate translation, open-ended generation, or multi-reference settings;
- analyze gains as a function of target entropy, teacher uncertainty, or answer
  multiplicity;
- test whether diversity improvements are real rather than only stylistic.

This is especially important because recent on-policy distillation work shows
that teacher uncertainty changes what type of imitation objective is desirable.

## 13. Go / no-go criteria

The project should define failure conditions early.

### Upgrade 1

Stop or rethink if:

- the training discrepancy goes down but held-out calibration, CE, or
  downstream quality does not improve;
- diversity appears only in-sample and not on held-out prompts;
- learned spectral variants outperform fixed variants only in training metrics.

### Upgrade 3

Do not allow pure teacher replacement to become the default if:

- teacher-match improves but human-grounded or GT-grounded metrics worsen;
- gains disappear when online or diverse target sampling is removed;
- the method becomes heavily teacher-style-dependent.

### Upgrade 2

Do not let it become a mainline dependency if:

- reward ranking consistency drifts sharply;
- effective feature rank collapses;
- EMA target drift or reward-statistic instability becomes large;
- the adaptive version cannot beat the best frozen-feature teacher-target
  baseline.

## 14. Current decision status

### Confirmed

- Upgrade 1 is a committed direction.

### Strong but still exploratory

- Upgrade 3 is strongly supported and fits the main story well.

### Promising but highest-risk

- Upgrade 2 is plausible, but should remain exploratory until we have a strong
  frozen-feature distributional teacher-target variant.

## 15. The guiding filter for every future decision

When we evaluate a new idea, the first three questions should be:

1. Does it make EBFT more genuinely distribution-level?
2. Does it strengthen the target in a principled way rather than only adding
   noise or style?
3. Does it preserve reward reliability and ablation clarity?

If an idea fails these filters, it should not become part of the main method.

## 16. Literature anchors

The plan is most strongly grounded in the following lines of work:

- `EBFT` for feature-space on-policy optimization.
- `NCFM` for characteristic-function-based distribution discrepancy and adaptive
  discrepancy design.
- `MiniLLM`, `OPCD`, `DASD`, and related distillation work for teacher
  distribution matching and on-policy student training.
- `BYOL`, `MoCo`, `DINO`, `SimSiam`, `VICReg`, and related representation
  learning work for asymmetric target networks and anti-collapse stabilization.
- `InstructGPT`, `Llama 2`, `UP-RLHF`, and later RLHF robustness work for reward
  reliability under policy improvement.
