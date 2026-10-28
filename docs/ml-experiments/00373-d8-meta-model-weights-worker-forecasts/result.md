---
id: '00373'
slug: d8-meta-model-weights-worker-forecasts
title: >-
  Does an XGBoost meta-model that weights several worker forecasts by context beat the best single
  family at D8?
theme: calibration-and-uncertainty
also_relevant_to:
  - production-reference-and-best-guess
status: done
kind: other
size: >-
  medium. Stored: private files under data/ml-experiments/00373-d8-meta-model-weights-worker-forecasts/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is defensible
  for the meta-model's target (the actual, or the actual minus a base median), so the plan fixes the
  second. Callers: the scripts read the stored D8 forecasts of experiments 355 and 360 and call
  experiment 354's scoring helpers. Asset graph: not touched.
hypothesis: >-
  At D8 (Marsh Lane), an XGBoost meta-model that sees the worker forecasts' quantiles, recent power,
  lead time, calendar, and each worker's recent error can weight the workers by context, and so
  scores better than the best single family chosen out of sample.
prediction: >-
  Written before any score is read. The meta-model scores at least 0.2 points of quantile score below
  the best single family chosen out of sample at D8, and at least 0.1 points below the fixed 50/50
  blend of the GAM stack and BG2. It fails if the meta-model is within 0.1 points of the 50/50 blend.
decision_rule: >-
  Paired contrasts, whole valid months resampled (20,000 resamples, seed 0), 95% intervals, on the
  scored months October 2025 to June 2026 (the first three valid months only train the meta-model).
  C1 meta-model minus the best single family; C2 meta-model minus the 50/50 blend; C3 meta-model minus
  BG2; C4 meta-model minus the meta-model without lagged power; C5 meta-model minus the meta-model
  without calendar context. `supported` when C1 is at most -0.2 with an interval wholly below zero and
  C2 is at most -0.1. `small gain` when C2 is wholly below zero but above -0.1. `not supported`
  otherwise. Stop rule: if C2 is not at most -0.1 at D8, stop and run nothing at D2, D3, or D4. The
  threshold is 0.1 because the meta-model needs every family to run for every substation in
  production, and the 0.2 for C1 pays for that serving cost. Gate: the script reproduces 8.338 (GAM
  stack) and 7.593 (BG2) on the 96,986 rows of experiment 354 before any meta-model is fitted.
benchmark: >-
  BG2, the GAM stack (experiment 355 arm W), each of BG2's five members alone (experiment 360),
  climatology, and manual_heuristic, on the same rows
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of the series' effective
  capacity (1 point is about 0.1 MW at D8); median error in MW; q01, q05, q95, q99 coverage and
  pinball loss
code: ml-experiments/00373-d8-meta-model-weights-worker-forecasts/
verdict: not supported
best_arm: >-
  None as a change. The meta-model scores 7.406 against 7.434 for the 50/50 blend of the GAM stack
  and BG2 (-0.028, 95% interval -0.145 to +0.094) and 7.405 for the best single family (0.000,
  -0.451 to +0.354), 73,331 rows at D8.
headline: >-
  An XGBoost meta-model over seven stored worker forecasts does not beat a fixed 50/50 blend or the
  best single family at D8. It is 0.028 points below the 50/50 blend (-0.145 to +0.094), which
  triggers the stop rule, and level with the best single family (the analogue member, chosen out of
  sample in every month). On bank holidays and 24 December to 2 January it scores 0.471 points below
  the blend (-0.759 to -0.211) and 1.119 below the best single family (-1.651 to 0.000), and it is
  level with both on other days.
lessons:
  - >-
    The workers' errors are too alike for a meta-model to find a context in which one beats the
    rest: the meta-model's median error correlates 0.956 with the blend's.
  - >-
    The one context where the meta-model differs is the calendar: on 3,765 rows from bank holidays and
    24 December to 2 January it scores 0.47 points below the 50/50 blend, with no loss elsewhere.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  A second-level XGBoost model that weights BG2's five members and the GAM stack by context scored
  no better than a fixed 50/50 blend at substation D8 (0.03 points of capacity lower, an interval
  that includes zero), so serving every family for every substation is not justified on this
  evidence. The only context where it helped was holidays and the Christmas period (0.47 points
  lower, on 3,765 rows), which a single rule might capture more cheaply. This is one substation and
  nine scored months.
advice:
  - >-
    Do not build a meta-model over worker forecasts for D8's uncertainty; it is 0.028 points below
    the 50/50 blend with an interval including zero (C2, Figure 1).
  - >-
    If a worker blend is wanted, test a holiday-and-Christmas weighting rule against the 50/50
    blend, because the meta-model's only gain is on those days (event-day row in Figure 1).
  - >-
    Store rolling-origin forecasts from earlier years before testing any learned weighting again,
    because one scored year leaves nine months to score and three to train on (scope section).
date: 2026-10-28
---

**Take-home message: weighting the workers by context does not help at D8, except possibly around
holidays.** We asked whether a second XGBoost model, given the forecasts of seven existing
forecasters plus recent power and calendar context, could choose which forecaster to trust at each
time, and so beat the best one. We trained it on earlier months and scored it on the next month,
from October 2025 to June 2026. It scored 7.406 points of quantile score, against 7.434 for an even
blend of the GAM stack and BG2 and 7.405 for the best single family. The verdict is `not supported`,
and the stop rule ends the work before D2, D3, and D4.

**What we did.** The seven workers are BG2's five members (weekly XGBoost, monthly XGBoost,
analogue, pooled by substation type, XGBoost stack), the GAM stack of
[experiment 355](../00355-d8-weekly-refit-and-median-band-swap/result.md), and BG2 itself
([experiment 87](../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md); the members
come from [experiment 360](../00360-bg2-member-drop-one-at-d8/result.md)). We stored no new
forecasts. Scores are the quantile score in points of D8's 10.38 MW capacity (twice the mean pinball
loss over 11 quantile levels; lower is better). The 50/50 blend is the one of
[experiment 359](../00359-d8-gam-bg2-quantile-blend/result.md).

**Caveat.** Interim, one substation, one scored year, 0 to 168 hours; it is unproven whether the
generator explains the steps and the gap.

![Contrasts](figure_1_contrasts.svg)

![Monthly score](figure_2_monthly.svg)

![Absolute score](figure_3_absolute.svg)

## Results

- **Arms.** Meta-model; meta-model without lagged power; meta-model without calendar context; best
  single family out of sample; 50/50 blend of the GAM stack and BG2; the GAM stack; BG2; each BG2
  member alone; climatology; manual heuristic. Code is in `ml-experiments/00373-*/`
  (`e00373_meta.py`, `e00373_score.py`, `e00373_charts.py`) and reuses the scoring helpers of
  experiments 252, 264, and 354.
- **Rows.** 73,331 half-hourly rows, October 2025 to June 2026 (the first three months only train
  the meta-model), 0 to 168 hours, `fold_a_daily_gap_v1`. Intervals resample whole valid months
  (20,000 resamples, seed 0). The 99% intervals are in `scores.json`.
- **Gate.** The script reproduces the GAM stack (8.338), BG2 (7.593), and each member alone (7.560,
  7.694, 7.472, 7.633, 8.562) on the 96,986 rows before any meta-model is fitted.
- **Absolute scores** (points, scored months): meta-model 7.406; best single family 7.405; 50/50
  blend 7.434; BG2 7.570; GAM stack 7.983; manual heuristic 7.963; climatology 8.836.
- **Contrasts (95%).** C1 meta-model minus best single family 0.000 (-0.451 to +0.354). C2 minus the
  50/50 blend -0.028 (-0.145 to +0.094). C3 minus BG2 -0.165 (-0.602 to +0.139). C4 minus the
  meta-model without lagged power -0.008 (-0.017 to -0.001). C5 minus the meta-model without
  calendar +0.018 (-0.031 to +0.088).
- **Holiday and event days.** On 3,765 rows from bank holidays and 24 December to 2 January, the
  meta-model scores 6.162, against 6.633 for the 50/50 blend (-0.471, -0.759 to -0.211) and 7.282 for
  the best single family (-1.119, -1.651 to 0.000). On all other days it scores 7.473, against 7.477
  and 7.412 (-0.004, -0.115 to +0.119; +0.061, -0.327 to +0.380), so non-inferiority holds
  against the blend and holds within noise against the best single family.
- **Tails** (share of readings below q01 and above q99, nominal 0.01; q05 to q95 coverage, nominal
  0.90): meta-model 0.005, 0.013, 0.897; best single family 0.003, 0.001, 0.925; 50/50 blend 0.002,
  0.006, 0.926; BG2 0.001, 0.003, 0.890. Pinball loss at q01 / q05 / q95 / q99: meta-model 0.413 /
  1.597 / 1.810 / 0.508; best single family 0.366 / 1.845 / 1.714 / 0.453; 50/50 blend 0.410 /
  1.702 / 1.732 / 0.444.
- **Serving cost.** Serving the meta-model needs every worker to run for every substation, plus one
  XGBoost fit per refit, for a gain whose interval includes zero.
- **Post hoc.** The first run chose BG2 as the best single family every month. BG2 is itself a blend,
  so the final choice excludes it and picks among the six other forecasters; the analogue member won
  every month. This changed no planned contrast's definition apart from C1's comparator.

## Checks

- **Median minus the 50/50 blend's median**, mean over rows (points): meta-model +0.02, GAM stack
  -1.07, BG2 +1.07, best single family (analogue) +1.63. The stored member forecasts hold no mean, so
  the median stands in for the mean forecast.
- **Median error beside the score contrasts.** Median absolute error (points): meta-model 13.476,
  best single family 13.418, 50/50 blend 13.631, BG2 13.821. The meta-model beats the blend on both
  score and median error, and is level with the best single family on score and 0.058 worse on median
  error, so the two metrics do not disagree about direction at any contrast whose interval excludes
  zero. The meta-model's median error correlates 0.956 with the blend's.
- **Triggers.** Trigger 2 fired in letter: the result contradicts its prediction. The explanation
  recorded here (worker errors are highly correlated, so context has little to select) was not tested
  by a further experiment. No other trigger fired: C5's nested control moves the score by 0.018,
  inside its interval half-width of 0.06; no sister experiment exists to compare interval
  widths; and no later design depends on this result. No audit was run, because the verdict
  is negative and the stop rule ends the work.

## Scope and what to try next

- **Not run.** D2, D3, D4 (stop rule), the six decision Primaries, and the replicate year 2024-25
  (rolling-origin forecasts of every worker are not stored for either).
- **Leads.** Weather regime is not an input: the stored frames hold no forecast weather, and their
  sunny and cold flags are computed from observed weather. A later test with ensemble-driven regime
  flags is the open version of that idea.
- **What the experiment narrows.** A learned weighting adds nothing over a fixed blend on ordinary
  days at D8. A rule that changes the weights on holidays and 24 December to 2 January is the one
  version worth testing, on more than 3,765 rows, which needs a longer history of worker forecasts.
