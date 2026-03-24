# Feature Network Unfreezing for EBFT

This note is exploratory. It does not record a confirmed method change. It
records the current literature-backed view on whether EBFT's frozen feature
network should be partially or fully unfrozen.

## Question

EBFT currently uses a frozen feature network to define the reward signal in
feature space. Can a suitable unfreezing strategy raise the method ceiling?

## Short Answer

Probably yes, but not through naive joint training.

The literature points in a consistent direction:

- learned discrepancy networks are often more powerful than fixed ones;
- but moving targets create instability, collapse, and reward exploitation;
- successful systems usually introduce asymmetry, target networks, slow updates,
  uncertainty control, or strong regularization.

So the right question is not:

- `should we unfreeze the feature network at all?`

It is:

- `what constrained form of unfreezing preserves reward reliability while
  increasing discrepancy power?`

## Literature Takeaways

### 1. Learned discrepancy usually increases test power

Evidence from distribution matching and two-sample testing suggests that a fixed
feature map is often suboptimal.

- `MMD-GAN` shows that adversarially learned kernels can improve discrepancy
  quality over fixed MMD kernels.
- `Learning Deep Kernels for Non-Parametric Two-Sample Tests` shows that deep,
  learned kernels can substantially improve test power.
- `NCFM` pushes this further by treating discrepancy definition itself as a
  minmax problem over characteristic-function sampling.

Implication for EBFT:

- a frozen feature network is stable, but it likely leaves performance on the
  table because the discrepancy is not allowed to adapt to the policy's failure
  modes.

### 2. Moving targets need explicit stabilization

When the representation-defining network moves during training, successful
methods almost never let both sides chase each other freely.

- `DQN` stabilizes bootstrap targets with a target network.
- `MoCo` uses a momentum encoder to keep the contrastive dictionary consistent.
- `BYOL` uses online/target asymmetry with EMA updates.
- `DINO` also relies on a momentum teacher.
- `SimSiam` shows that stop-gradient and asymmetry are critical even without a
  momentum encoder.

Implication for EBFT:

- if the feature network is unfrozen, reward computation should almost certainly
  use a slow target copy rather than the raw online encoder.

### 3. Collapse prevention cannot be left implicit

Once the feature network becomes trainable, there is a real danger that the
representation geometry becomes degenerate or overly narrow.

- `VICReg` shows that variance and covariance regularization can explicitly
  prevent collapse.
- `Barlow Twins` shows that redundancy reduction can stabilize representation
  learning without negative pairs.
- follow-up VCReg work suggests that high-variance, low-covariance
  representations improve transfer and reduce collapse-like behavior.

Implication for EBFT:

- any trainable feature network variant should likely include explicit
  anti-collapse regularization, not only task reward.

### 4. RLHF experience warns against unconstrained moving rewards

Reward-side adaptation in alignment is attractive, but non-stationary rewards
are easy for policies to exploit.

- `InstructGPT` and `Learning to Summarize from Human Feedback` use a learned
  reward model as a fixed optimization target during policy improvement.
- `Llama 2` explicitly emphasizes KL regularization and reward whitening to
  reduce reward hacking and stabilize RLHF.
- `UP-RLHF` uses reward ensembles and uncertainty penalties to combat
  overoptimization.
- `Iterative Data Smoothing` studies reward-model degradation and proposes
  iterative correction.
- `AdvPO` uses uncertainty-aware robust optimization against reward
  overoptimization.
- `Reward Model Overoptimisation in Iterated RLHF` reinforces that iterated
  reward/policy updates can help but remain delicate.

Implication for EBFT:

- if we unfreeze the feature network that defines reward, we are entering the
  same design space as moving or iterated reward models.
- this likely increases the method ceiling, but also increases the chance of
  reward hacking unless we separately monitor reward reliability.

## Current Hypothesis for EBFT

The strongest current hypothesis is:

- full end-to-end unfreezing is too risky as a first move;
- a constrained, asymmetric, slow-moving feature-learning setup is much more
  plausible.

## Most Plausible Design Directions

### Direction A: Online feature net + EMA target feature net

Use two feature networks:

- an `online` feature network that is updated;
- a `target` feature network that provides reward features and is updated by EMA.

This is the most literature-supported stabilization pattern.

### Direction B: Partial unfreezing only

Instead of full unfreezing:

- tune only upper layers;
- or tune only lightweight adapters / LoRA modules;
- or tune only a projection head on top of frozen backbone features.

This reduces drift in reward geometry while allowing some task adaptation.

### Direction C: Regularize toward the initialization

If the feature network moves, it should remain anchored.

Possible anchors:

- parameter-space L2 to initialization;
- KL / output-space consistency to the initial frozen network;
- feature-space consistency on held-out ground-truth completions.

### Direction D: Add explicit anti-collapse regularization

If the feature network is trainable, collapse-prevention should be explicit.

Candidate families:

- variance floor;
- covariance penalty / decorrelation;
- redundancy reduction;
- stop-gradient asymmetry where appropriate.

### Direction E: Separate feature learning timescale from policy learning timescale

The feature network should probably move more slowly than the actor.

Concretely, this suggests:

- alternating updates;
- lower learning rate on the feature network;
- delayed feature updates;
- or feature updates only every K policy steps.

## What Seems Wrong

The following options currently look weak:

- fully unfreezing the feature network and using it directly for reward in the
  same step;
- updating actor and feature net with identical objectives and identical
  timescales;
- removing the frozen anchor without adding a target-network mechanism;
- assuming whitening alone is enough to control representation drift.

## Current Recommendation

If we explore this direction, the first serious version should likely be:

- `partially trainable feature network`
- `EMA target copy for reward computation`
- `adapter-only or upper-layer tuning`
- `explicit anti-collapse regularization`
- `strong anchoring to the original frozen network`

This keeps the main upside of unfreezing:

- a more adaptive and potentially more discriminative feature geometry;

while respecting the main lesson from RLHF and self-supervised learning:

- moving targets need asymmetry and stabilization.

## Not Yet Decided

The following is still open:

- whether the online feature network should maximize discrepancy in an
  adversarial sense, or simply improve representation quality under a
  regularized auxiliary loss;
- whether the first trainable component should be the entire feature backbone,
  only a projection head, or only LoRA adapters;
- whether feature updates should be driven by policy rollouts, ground-truth
  completions, or both.

