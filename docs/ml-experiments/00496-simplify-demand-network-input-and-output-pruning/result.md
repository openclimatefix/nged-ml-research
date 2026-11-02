---
id: "00496"
slug: simplify-demand-network-input-and-output-pruning
title: >-
  Can the demand neural network drop its ensemble-spread inputs, merge its holiday columns, drop its
  weekly lag, or use one Gaussian instead of three without losing forecast skill?
theme: production-reference-and-best-guess
also_relevant_to:
  - demand-inputs
status: finished
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
verdict: partly supported
best_arm: >-
  No spread inputs (arm a): CRPS 4.702 points at the eight MVA-only Primaries and 3.441 at the six
  signed-MW Primaries, against 4.690 and 3.484 for PretrainOnce (PO).
headline: >-
  Of four removals, only the 7 ensemble-spread inputs came out equivalent to PretrainOnce at both
  groups (+0.012 points at the eight, Bonferroni upper bound +0.042 against a limit of 0.086; -0.043
  at the six, upper bound -0.004 against 0.170). Merging the 4 holiday columns into one cost +0.095
  at the eight and +0.229 at the six. Dropping the 3 weekly-lag inputs cost +0.079 at the eight (not
  shown equivalent) and +0.050 at the six (equivalent). One Gaussian instead of three cost +0.152 at
  the eight and failed the q99 pinball limit.
lessons: >-
  The fresh-pretraining check on arm d gave the same verdicts as arm d against PretrainOnce, so
  pretrain-once did not change a verdict. Arm a is worse on cold winter nights at the six (+0.187,
  Bonferroni interval +0.014 to +0.312), so an equivalence claim on the whole sample hides a loss
  on the nights NGED cares about most, and its gain at the six comes mostly from one fold
  (-0.268 in 2026-01).
take_home_message: >-
  The demand neural network can drop its 7 ensemble-spread inputs (46 inputs down to 39) with no
  measurable loss in five scored months: the CRPS change was +0.012 points at the 8 MVA-only
  Primaries and -0.043 at the 6 signed-MW Primaries, both inside the limit, although the network
  scored 0.187 points worse at the six on cold winter nights. The other three simplifications cost
  more than the limit allows: merging the holiday columns, dropping the weekly lag (at the eight),
  and using one Gaussian instead of three (at the eight).
advice:
  - >-
    Consider dropping the 7 ensemble-spread inputs from the demand network, after checking the cold
    winter nights at the six on more folds; it matched PretrainOnce at both groups (Results table,
    figure 1).
  - >-
    Keep the four holiday columns, the three weekly-lag inputs, and the three-Gaussian output; each
    removal scored worse than PretrainOnce by more than the limit at the eight or the six (Results
    table, figure 1).
  - >-
    Check every equivalence claim on cold winter nights as well as on the whole sample, because the
    spread-input removal passed the whole sample and lost 0.187 points at the six on cold nights
    (Results table).
production_status: candidate
adopted_in: none
---

**The demand neural network can drop its 7 ensemble-spread inputs (46 inputs down to 39) with no
loss that five scored months can detect: its CRPS changed by +0.012 points at the 8 MVA-only
Primaries and by -0.043 at the 6 signed-MW Primaries (a lower score is better), both inside the
equivalence limit. The other three simplifications scored worse than the limit allows. On cold
winter nights at the six, the network without spread inputs scored 0.187 points worse, which
the equivalence claim on the whole sample does not show.**

We asked whether four parts of the demand network could be removed one at a time. Each arm starts
from PretrainOnce (experiment 494: pretrained once at the 2025-07 cut, fine-tuned each month, two
seeds), which was equivalent to a fresh refit. The CRPS is in points of effective capacity. "The
eight" are the 8 Primaries that report only apparent power (MVA); "the six" are the 6 that report
signed megawatts, scored on rows with no labelled event. The limits are the printed two-seed minimum
detectable effect: 0.086 points at the eight and 0.170 at the six.

| Arm | What is removed | Inputs | Output values |
|---|---|---|---|
| PO | nothing (PretrainOnce) | 46 | 9 |
| a | the 7 ensemble-spread (`std_*`) inputs; the ensemble mean stays | 39 | 9 |
| c | the 4 Christmas and bank-holiday columns, replaced by one day-type column | 43 | 9 |
| d | the 3 weekly-lag inputs (value, missing flag, age) | 43 | 9 |
| b | two of the three Gaussians of the output | 46 | 3 |
| dF | arm d, with fresh pretraining at every fold (a check) | 43 | 9 |

Verdict: partly supported. Arm a is equivalent to PO at both groups. Arm c is not shown equivalent
at either group (upper bounds +0.143 and +0.332). Arm d is not shown equivalent at the eight
(+0.124)
and is equivalent at the six (+0.089). Arm b is not shown equivalent at the eight (+0.201) and fails
the q99 pinball limit there (+0.029 against 0.02). No combined network was run, because arm a is
the only arm equivalent at both groups.

## Figures

![Each arm's CRPS difference to PretrainOnce at the eight and the six, with intervals and
limits](figure_1_headline.svg)

![The same difference in each of the five folds](figure_2_per_fold.svg)

![Tail calibration of PretrainOnce and the four arms](figure_3_tail_calibration.svg)

![A day-ahead forecast of a signed-MW Primary](figure_4_example_D10_day_ahead.svg)

![A 14-day forecast of a signed-MW Primary](figure_4_example_D10_14_days.svg)

![A day-ahead forecast of an MVA-only Primary](figure_4_example_D4_day_ahead.svg)

![A 14-day forecast of an MVA-only Primary](figure_4_example_D4_14_days.svg)

## Results

- **Setup.** Each arm is experiment 425's arm A (pooled plain network, Primary embedding, `g09`)
  behind a fixed input transform, pretrained once at the 2025-07 cut and fine-tuned at each of the 5
  folds (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), seeds 0 and 1, widening 1.15, rows of
  experiment 470 (`experiment_470_intersection_v1`), cleaned data with 1,179 scoring actuals dropped
  for a flagged exact zero, the D15 and D18 start cut applied. The removed inputs are multiplied by
  zero inside the network, after the standardiser and after the pretraining spread table, so
  training, forecasting, and reloading share one code path. A gate before any arm: an identity
  transform started from experiment 425's stored 2025-07 weights reproduced PretrainOnce's stored
  forecasts to within 5e-6.
- **Minimum detectable effect (two seeds, five folds).** 0.086 points at the eight and 0.170 at the
  six. The briefs' caps (0.03 for a, 0.02 for c, 0.05 for d and b) lie below those limits, so an
  equivalence claim at the limit cannot show a loss smaller than the minimum detectable effect, and
  an interval that reaches past the limit is inconclusive. Bonferroni count 20 (5 arms x 2 groups x
  CRPS and q99 pinball loss), with 100,000 origin-day block resamples. The scorer's Bonferroni
  interval is two-sided, and the decision rule reads its upper end.
- **CRPS, difference to PO (points).** The first interval is the Bonferroni interval over origin
  days; the second is the 95% interval over folds.

| Arm | Eight | Six | Verdict |
|---|---|---|---|
| a | +0.012 [-0.020, +0.042]; folds [-0.034, +0.059] | -0.043 [-0.084, -0.004]; folds [-0.215, +0.125] | equivalent at both |
| c | +0.095 [+0.052, +0.143]; folds [-0.029, +0.211] | +0.229 [+0.140, +0.332]; folds [-0.143, +0.588] | not shown equivalent at both |
| d | +0.079 [+0.038, +0.124]; folds [-0.016, +0.174] | +0.050 [+0.013, +0.089]; folds [-0.070, +0.169] | not shown equivalent at the eight; equivalent at the six |
| b | +0.152 [+0.103, +0.201]; folds [+0.055, +0.242] | +0.082 [+0.043, +0.129]; folds [-0.034, +0.199] | not shown equivalent at the eight; equivalent at the six |

- **q99 pinball loss, difference to PO (points).** The upper Bonferroni bounds are all inside the
  0.02 limit except arm b at the eight: a -0.003 (bound +0.001) and -0.008 (-0.003); c +0.003
  (+0.007) and +0.013 (+0.018); d +0.005 (+0.008) and +0.005 (+0.009); b +0.016 (+0.029) and -0.003
  (+0.003), the first number at the eight and the second at the six.
- **Absolute CRPS (points).** At the eight: K2 4.720, PO 4.690, a 4.702, c 4.785, d 4.769, b 4.842.
  At the six: K2 3.486, PO 3.484, a 3.441, c 3.712, d 3.534, b 3.566.
- **Against K2 (descriptive).** Arm a is also equivalent to K2: -0.018 [-0.049, +0.011] at the eight
  and -0.045 [-0.077, -0.014] at the six. Arms c, d, and b are not shown equivalent to K2 at the
  eight (+0.065, +0.049, +0.122).
- **Fold by fold.** Arm a's gain at the six comes mostly from 2026-01 (-0.268; the other four folds
  range from -0.059 to +0.083), so the gain is not established. Leaving one fold out moves arm a's
  difference at the eight between 0.000 and +0.023, which stays inside the limit.
- **Fresh-pretraining check on arm d (dF).** dF against K2 (both pretrained fresh): +0.051
  [+0.007, +0.096] at the eight (not shown equivalent) and +0.062 [+0.031, +0.092] at the six
  (equivalent). dF against PO gives +0.080 and +0.064. The verdicts equal arm d's, so the
  pretrain-once shortcut did not change a verdict.
- **Cold winter nights (CRPS difference to PO, Bonferroni interval; descriptive).** At the eight: a
  +0.051 [-0.041, +0.172], c +0.053 [-0.037, +0.203], d +0.225 [+0.042, +0.445], b +0.273 [+0.155,
  +0.405]. At the six: a +0.187 [+0.014, +0.312], c +0.745 [+0.353, +1.109], d +0.167 [+0.051,
  +0.261], b +0.288 [+0.042, +0.475]. Every arm is worse on cold nights at the six, and
  arm a's loss there (+0.187 against -0.043 on the whole sample) is the largest sign that
  the whole-sample equivalence overstates the case for removing the spread inputs.
- **12 to 36 hours (CRPS difference to PO).** At the eight: a -0.027, c +0.024, d +0.025, b +0.116.
  At the six: a +0.033, c +0.265, d +0.083, b +0.114.
- **Christmas week (descriptive, one Christmas).** At the eight a is -0.320 [-0.411, -0.206] better
  than PO; c -0.017, d +0.010, b -0.162. At the six: a +0.046, c +0.204, d +0.228, b +0.200. Arm c
  does not lose at the eight in Christmas week, so its loss comes from other days, mostly at the
  six (12 to 36 hours +0.265).
- **Tails (widened 1.15, all folds, rows with no labelled event).** q99 coverage at the eight: PO
  0.987, a 0.986, c 0.984, d 0.984, b 0.978. At the six: PO 0.979, a 0.983, c 0.974, d 0.977,
  b 0.974. Arm b's q99 coverage falls by 0.009 at the eight, inside the 0.01 limit. q95 coverage is
  0.934 (PO) and 0.923 (b) at the eight, 0.907 (PO) and 0.895 (c) at the six.
- **Avoidable-curtailment cost at the six, difference to PO (experiment 470's cost measure).** At
  f = 0.9: a -0.461 [-1.370, +0.039], c +0.193 [-0.009, +0.541], d -0.119 [-0.265, -0.004], b -0.307
  [-0.917, +0.027]. At f = 0.5: a -1.234 [-4.226, +0.888], c +2.539 [+0.773, +4.654], d +0.917
  [-0.219, +2.230], b +0.195 [-1.315, +1.698]. Only arm c's cost differs from zero at f = 0.5 (a
  higher cost).
- **D15 and D18 (report only).** CRPS: PO 11.588, a 10.566, c 10.868, d 11.523, b 11.068, K2 10.956.
- **Serving table.** Every arm keeps 2 networks per pass, 12 monthly fine-tunes, and 1 pretraining
  per year (about 38 GPU seconds per seed-refit, experiment 494). Arm a needs 39 inputs instead of
  46 and no ensemble-spread values at serving; change in CRPS against PO +0.012 at the eight and
  -0.043 at the six. Arm c: 43 inputs, +0.095 and +0.229. Arm d: 43 inputs and no read of the
  load one week earlier, +0.079 and +0.050. Arm b: a 3-value output instead of 9, +0.152 and +0.082.

## Checks

- **Median minus mean forecast per Primary.** PO has a median below the mean at 16 of 16 Primaries
  (largest gap -2.6 points of capacity), K2 at 15 of 16 (-2.4), and arm a at 15 of 16 (-1.7). Arm
  b's
  gap is zero by construction, because one Gaussian is symmetric.
- **Error of the median beside the CRPS contrast.** Mean absolute error of the median forecast at
  the eight: PO 6.724, a 6.734, c 6.854, d 6.840, b 6.970 points; at the six: PO 5.662, a 5.600, c
  5.979, d 5.700, b 5.749. The directions agree with the CRPS contrasts for every arm (a at the six
  improves in both), so trigger 3 did not fire.
- **Triggers.** No trigger fired and no audit was run. A leave-one-fold-out check on arm a at the
  eight stays inside the limit (0.000 to +0.023). One Sonnet code review before the first fit found
  a bug that would have made the fresh-pretraining arm silently reuse pretrain-once weights if it
  ran after another arm in one process; the fix was made before any fit.

## Caveats

- Five folds, each from one pair of networks, give wide intervals across folds; the origin-day
  interval treats each fitted network as fixed, so that interval is optimistic. The interval across
  folds for arm a at the six reaches +0.125 and for arm c reaches +0.588.
- Each removal is tested alone; the removals might interact, and no combined network was run.
- Arm c is a design choice: a single ordinal (0 to 3) in place of four columns. A different merge
  (for example a one-hot day type) might lose less; this experiment does not test that.
- Zeroing an input gives the same function as removing it, but the unused first-layer weights start
  at random values, so initialisation differs slightly from a network built without them.
- Arm b keeps the 1.15 widening used for the three-Gaussian arms; a different widening might change
  its q99 result.
- NGED's event labels are incomplete: "rows with no labelled event" may hold unlabelled events.
  Every score is selection-biased, and July 2026 is unused.
- Reviews: one Sonnet plan review and one Sonnet code review before the first fit; the results
  review and two write-up reviews follow in this pull request.

## What to try next

- Re-test the removal of the spread inputs on more folds, scoring cold winter nights at the six
  separately before any change to serving.
- Test other merges of the holiday columns (a one-hot day type, or keeping the Christmas position
  inside the merged column) before dropping the idea.

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
