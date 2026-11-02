---
id: "00496"
slug: simplify-demand-network-input-and-output-pruning
title: >-
  Can the demand neural network drop its ensemble-spread inputs, merge its holiday columns, drop its
  weekly lag, or use one Gaussian instead of three without losing forecast skill?
theme: production-reference-and-best-guess
also_relevant_to:
  - demand-inputs
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: the new networks and forecasts under this experiment's private data folder only;
  every earlier experiment is read and never written. Serving path: not touched (this repository
  holds none). Degradation rule: not touched. Designs: more than one defensible (how to remove an
  input, and which weekly or daily lag to drop), so the plan fixes both before any score is read.
  Callers: this folder's scripts import experiments 397, 425, 470, and 494. One Sonnet plan review,
  one Sonnet code review before the first fit, one Sonnet results review, and two Sonnet write-up
  reviews. Part of the neural simplification series (step 6 of 7).
hypothesis: >-
  Four inputs or outputs of the demand neural network can be removed one at a time, each within the
  equivalence limit: (a) the seven ensemble-spread inputs (the mean forecast stays), (c) the four
  holiday columns that describe the Christmas period and bank holidays, merged into one day-type
  column, (d) the three inputs that describe the weekly lag (its value, its missing flag, and its
  age), and (b) the second and third Gaussian of the three-Gaussian output. Serving then needs fewer
  inputs and a smaller output layer.
prediction: >-
  Written before any score is read. P1: (a) is within the limit at the eight and the six (the
  spread inputs add little when the ensemble mean is present). P2: (c) is within the limit at both
  (the nearest-holiday columns stay). P3: (d) loses 0.00 to 0.08 points at the six and is within the
  limit at the eight (the weekly lag matters at short lead times only). P4: (b) loses 0.00 to 0.10
  points and its q99 coverage falls by up to 0.02, because one Gaussian cannot carry a heavy tail.
  P5:
  the combined network of the arms that pass scores within the limit of PretrainOnce.
decision_rule: >-
  An arm is equivalent to PretrainOnce when the upper Bonferroni bound of its CRPS difference to
  PretrainOnce (origin-day block bootstrap within months, 100,000 resamples, since a Bonferroni
  level of 0.0025 leaves only 50 tail draws at 20,000) as a one-sided upper bound is at most the
  limit at
  the eight and at the six ("the six" is the `six_clean` slice), and the upper Bonferroni bound of
  its q99 pinball-loss difference is at most +0.02 points. The limit is the printed two-seed minimum
  detectable effect (MDE, experiment 428's definition with k = 2 seeds, restricted to the folds
  scored): 0.086 points at the eight and 0.170 at the six, which sit at or above the briefs' caps of
  0.03 (a), 0.02 (c), 0.05 (d), and 0.05 (b). An interval that reaches past the limit is
  inconclusive, not equivalent, and an equivalence claim at the MDE cannot show a loss smaller than
  the MDE. Arm (b) also needs q99 coverage within 0.01 of PretrainOnce. Bonferroni count 20 (5
  arms x 2 groups x [CRPS, q99 pinball loss]), fixed up front; it is 16 if the combined arm does
  not run. The q99 coverage check for (b) is a point estimate and is not counted. The page prints
  the q99 pinball MDE before any contrast. It also prints, for each arm, whether the interval met
  the brief's original cap (0.03, 0.02, 0.05, 0.05), because the limits above are 1.7 to 8.5 times
  those caps and the serving table would otherwise hide how loose "equivalent" is. Arm (b) is
  expected to fail its coverage check (P4), and the 1.15 widening is applied to it unchanged,
  which is a caveat on its q99 result. Descriptive and not counted: the fresh-pretraining check
  (below), a Christmas-week row for the 2025-12 fold (one Christmas, so inconclusive by
  construction, and the only place (c) can hurt), a month-by-half-hour climatology row beside K2,
  the interval across folds, cold nights, the 12 to 36 hour band, tails, and curtailment cost. A
  combined arm joins the arms that are equivalent at both groups; if none is, it is not run.
  Recommend each equivalent arm as a removable input or output; recommend the combined network if it
  is equivalent too.
benchmark: >-
  PretrainOnce, experiment 494's arm (pretrained once at the 2025-07 cut and fine-tuned each month,
  two seeds, stored), which shares every arm's pretraining cut and fine-tuning recipe. K2
  (experiment 425's arm A, fresh pretraining, seeds 0 and 1) is reported beside it.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity, q01, q05, q95, and q99 coverage and pinball loss,
  avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the signed-MW six.
---

## Plan

### Question

The demand neural network has 46 input columns, a weekly lag, and a three-Gaussian output. Each
removable input is a column production must compute and store, and each removable output shrinks the
network. This experiment removes them one at a time from experiment 494's PretrainOnce recipe, which
is equivalent to the fresh refit (experiment 494) and saves the pretraining stage.

### Design

- **Rows and folds.** The 5 folds of experiment 449 (2025-10, 2025-12, 2026-01, 2026-03, 2026-06) on
  the rows of experiment 470. Every arm trains on exactly the rows of experiment 425's arm A: the
  D15 and D18 start cut and the flagged exact-zero drop apply, and the row set does not depend on
  the inputs. Cleaned data, monthly refits, seeds 0 and 1, widening 1.15.
- **Removing an input.** Each arm trains the experiment 425 network (pooled plain network, Primary
  embedding, `g09`, learning rate 2e-4 in fine-tuning) with a fixed input transform in front of the
  first layer: the removed columns are set to zero, and the merged column is computed from the
  columns it replaces. For (a) and (d), a zeroed input carries no information and receives no
  gradient, so the network computes the same function as one built without those columns (the unused
  first-layer weights start at random values, so initialisation differs slightly). Arm (c) is not of
  that kind: the merged ordinal is a design choice, and `bank_holiday` is also recoverable from
  `nearest_days`. The serving table counts the inputs actually needed.
    - **(a) No ensemble spread.** The 7 `std_*` columns are zeroed. The ensemble mean stays.
    - **(c) One day-type column.** `christmas_inside`, `christmas_days`, `christmas_tent`, and
      `bank_holiday` are replaced by one column `bank_holiday + 2 * christmas_inside` (0 for an
      ordinary day, 1 for a bank holiday, 2 inside the Christmas period, 3 for both). This removes
      the graded position within the Christmas period. `nearest_days` and `nearest_inside` stay.
    - **(d) Two origin lags.** `lag_week`, `lag_week_missing`, and `lag_week_age` are zeroed. The
      newest-reading lag and the same-time-yesterday lag stay. Training rows do not change.
    - **(b) One Gaussian.** `components = 1`, so the output layer is 3 values, not 9.
    - **Combined.** The arms that pass, together.
- **Where the transform sits.** The input mask or merge is a wrapper module created inside
  `fit397.make_model`, driven by an arm field recorded in `meta.json`. Training, pretraining,
  forecasting, and reloading a saved network all pass through it. It acts after the standardiser's
  clipping and NaN fill, and after `prep425.fold_pretrain` overwrites the `std_*` columns with the
  lead-band median table, so pretraining rows carry no residual spread values. Pretraining uses the
  2025-07 fold's standardiser and spread table, as the stored PretrainOnce does, and fine-tuning
  uses each month's own. **Gate before any arm is read:** an identity-transform arm must reproduce
  the stored PretrainOnce forecasts, up to GPU nondeterminism.
- **Pretrain once, with a fresh check.** Each arm pretrains once, at the 2025-07 cut (seeds 0 and 1,
  ERA5 and CAMS rows, the arm's own transform), and fine-tunes at each of the 5 folds from those
  weights. This is experiment 494's PretrainOnce recipe, so the baseline is PretrainOnce. To check
  that the pretrain-once shortcut does not change a verdict, arm (d) is also run with fresh
  pretraining at every fold (80 GPU seconds per seed-fold) and compared with K2, with the same
  limits and group split. Arm (d) is chosen because the lags are the inputs most tied to
  pretraining. The check passes if its equivalence verdict agrees with (d) against PretrainOnce. If
  the verdicts disagree, the page reports both and (d) stays out of the combined arm.
- **Not run.** Experiment 494 measured pretrain-once, so no arm tests it again. Arms that change
  hidden width, depth, or the training rows are other experiments.

### Scoring and stop rules

- The experiment 470 scorer as in experiment 494: each arm against PretrainOnce, all on the same
  rows, with the printed two-seed MDE before any contrast. The scorer reports the eight, the six,
  cold winter nights, the 12 to 36 hour band, q01, q05, q95, q99 coverage and pinball loss, and
  avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the six. D15 and D18 and Christmas week
  are report-only.
- **Stop.** An arm that is not equivalent is not combined.
- **Serving table.** Inputs, output values, networks per pass, refits per year, and the change in
  CRPS against PretrainOnce, per arm.

### Compute

- Pretraining: 4 arms x 2 seeds x about 46 s. Fine-tuning: 4 arms x 5 folds x 2 seeds x about 34 s.
  The fresh-pretraining check: 5 folds x 2 seeds x 80 s. The combined arm: about 7 minutes. Roughly
  50 GPU minutes in all (GPU only; per-fold data preparation and forecasting bring the wall clock to
  about 1.5 hours), one GPU process under 8 GB, one feeder core. The briefs ask for arms a, c, and d
  first and b only if budget allows; at this cost all four run.

### Standing practices followed

Monthly refits; cleaned data; D15 and D18 cut; saved networks; stored forecasts reused
(PretrainOnce and K2); tails and cost; example forecasts and the 14-day plot; the process order of
the ml-experiment skill. Not applicable: load-shift event plots.

### Baseline choice

The brief names experiment 425's arm A and K2 as the baseline. This plan uses PretrainOnce as the
primary baseline, because every arm shares its pretraining cut and fine-tuning recipe, and
experiment 494 found it equivalent to K2. K2 stays in the serving table. Step 1 of the series (the
calibration rule) does not change the demand networks, so it leaves this baseline unchanged.

### Series

Step 6 of the neural simplification series (brief of 2026-11-01), after experiments 494 (refit
cadence and pretrain once) and steps 1 and 4 in other pull requests.
