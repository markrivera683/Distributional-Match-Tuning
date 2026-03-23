# Step X: Progress Checkpoint (EBFT -> Distribution Matching -> Distillation)

This checkpoint is a stepwise execution control note. It is not a horizontal
project summary.

## 1. Completed Status by Step

### Step 0

- `STEP0_BASELINE_CHECKLIST` exists and baseline entry path is defined;
- status: **completed for baseline-definition layer** ✔

### Step 1

- paper-aligned and execution-plan docs are in place;
- discrepancy-axis controlled runs are scripted;
- status: **completed for Step 1 planning + execution scaffold** ✔

### Step 2

- `cf_l1oo` reward path is integrated and runnable;
- status: **completed** ✔

### Step 2B

- target-measure design is formalized and mapped to implementation;
- status: **completed as design-to-code bridge** ✔

### Step 2C

- `cf_tokencloud_l1oo` implementation exists in reward utilities and pipeline;
- status: **code path completed, broader stabilization/evidence still ongoing** ◐

### Teacher extension

- local + remote teacher paths exist;
- teacher target enters `cf_l1oo` reward path;
- status: **partially completed (not yet final mainline)** ◐

## 2. Current Mainline

Current method mainline is:

- distribution matching under Step 2 (`cf_l1oo`) with no-teacher baseline as the
  primary comparison anchor.

Teacher path is currently:

- available and testable,
- but still treated as extension branch rather than default mainline.

## 3. Not Completed but Designed

Designed and partially wired, not yet fully closed:

- teacher-augmented target measure as a stable experiment package;
- distillation-oriented objective interpretation with complete empirical
  validation;
- feature-network unfreeze / adaptive geometry path (2-full style).

## 4. What We Should Not Do Now

Before teacher-target line is fully benchmarked, avoid:

- early feature-network unfreeze as mainline dependency;
- mixing token-level objective redesign and teacher-target redesign in one patch;
- simultaneous large-axis changes that break ablation clarity.

## 5. Next Core Actions

1. finalize local open-weight teacher MWV (`m=2~4`, small lambda sweep) under
   fixed Step 2 settings;
2. run fair baseline ladder: `single` vs `vicinal` vs `teacher` with matched
   budget and seeds;
3. lock go/no-go criteria for promoting teacher mode from extension to
   mainline candidate;
4. keep remote/API teacher as extension validation path after local teacher
   behavior is stable;
5. postpone feature-unfreeze experiments until target-side distillation gains
   are reproducible.
