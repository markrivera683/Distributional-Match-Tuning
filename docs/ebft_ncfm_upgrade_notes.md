# EBFT x NCFM Upgrade Notes

This note records only the upgrade points that we have already decided to
pursue. It is intentionally narrow. Anything that is still exploratory should
stay out of this file until we commit to it.

## Confirmed Upgrade 1

Upgrade EBFT from `point-centered feature alignment` to `conditional feature
distribution matching`.

## Why This Upgrade Is Needed

Current EBFT is already better than token-level CE, but its reward is still
organized around a point-centered target:

- generated samples are aligned to a single ground-truth feature target;
- diversity is added as an extra repulsion term between generated samples.

This means EBFT is not yet doing true distribution matching. It is closer to:

- `align to phi(y)`,
- then `penalize collapse`.

That is a strong step beyond CE, but it is still not the same as directly
matching the conditional output distribution in representation space.

NCFM gives us the key insight we want to import:

- do not treat distribution matching as mean matching plus regularization;
- define a discrepancy that is distribution-level by construction;
- optimize the model against that discrepancy itself.

## What We Are Learning From NCFM

The part of NCFM we want is not "image distillation" and not the exact code
structure around synthetic images. The transferable idea is the metric design.

What we want to carry over:

- use characteristic functions as the object that represents the whole feature
  distribution, not only its first moment;
- treat distribution matching as a discrepancy over sampled frequency
  arguments, rather than as cosine or Euclidean alignment to one target point;
- move from a fixed hand-crafted discrepancy to a learned or adaptive
  discrepancy space;
- make realism and diversity emerge from the discrepancy itself, instead of
  bolting diversity on as a separate auxiliary penalty.

In NCFM terms, the important inheritance is:

- `min over model / synthetic object`,
- `max over discrepancy-defining network or sampling strategy`,
- `CF-based distribution discrepancy in feature space`.

## What This Means For EBFT

The intended conceptual change is:

- current EBFT: `feature mean alignment + diversity penalty`
- target upgrade: `conditional feature distribution alignment`

So the new training signal should no longer be framed as:

- "sample j should be close to the single target feature";

it should be framed as:

- "the conditional distribution of generated features under context c should
  match the target conditional feature distribution under the same context".

This is the main method change. It is not a cosmetic reward rewrite.

## What We Explicitly Do Not Want

We do not want to make a shallow variant that only:

- replaces cosine similarity with a more complicated pairwise formula;
- keeps the same point-target structure and simply renames it
  distribution matching;
- adds another diversity term on top of EBFT and calls that the NCFM idea.

Those would miss the point of why we brought in NCFM.

## Current Working Interpretation

Our first upgrade should be understood as:

- replace EBFT's point-centered feature reward with a characteristic-function-
  based conditional distribution discrepancy in feature space;
- let the training objective operate on generated feature distributions rather
  than on one-sample-to-one-target alignment;
- use the NCFM perspective to make diversity an intrinsic property of the
  discrepancy, not only an external penalty.

## Immediate Design Consequence

When we discuss future EBFT changes, the first filter should be:

- does this make the method more genuinely distribution-level?

If the answer is no, it is not part of this first upgrade.

