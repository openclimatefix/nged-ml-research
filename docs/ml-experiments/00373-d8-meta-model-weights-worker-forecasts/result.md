---
id: '00373'
slug: d8-meta-model-weights-worker-forecasts
title: >-
  Does an XGBoost meta-model that weights several worker forecasts by context beat the best single
  family at D8?
theme: calibration-and-uncertainty
also_relevant_to:
  - production-reference-and-best-guess
status: planned
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
date: 2026-10-28
---

**Plan only; no score has been read.** This page is committed before any code runs.

## The five plan items

- **Workers.** Seven stored forecasts at D8, each with 11 quantiles on the same 96,986 rows: BG2's five
  members (weekly XGBoost, monthly XGBoost, analogue, pooled by type, XGBoost stack; experiment 360's
  `only_*` columns), the GAM stack (experiment 355 arm W, weekly refit), and BG2 itself.
- **Strictly out of sample.** Every stored worker forecast was made by a fit that ended before its
  run, so the workers are out of sample by construction. The meta-model trains on an expanding window:
  for scored month m it trains only on rows whose valid time precedes the first forecast issue time
  of month m, so every training target was observed. The stored forecasts begin in July 2025, so
  July to September 2025 only train, and October 2025 to June 2026 is scored. Every arm, including
  BG2, is also scored on these nine months so contrasts compare like with like; the whole-year gate
  rows are reproduced first.
- **Meta-model.** One XGBoost model with the quantile-error loss at the 11 levels (`n_jobs` 12). The
  target is the actual minus the median of the 50/50 blend, so the model learns a correction and its
  quantiles are sorted to avoid crossing. Inputs: each worker's q05, q50, and q95; the spread among
  worker medians; the lead time; actual power at the last half-hour at or before the run's issue time
  (lagged power); hour, month, day of week, and an England bank-holiday flag; and each worker's mean
  absolute median error over the seven days before the run (workers' recent out-of-sample error).
  The brief's weather regime is left out: the stored frames hold no weather, and the sunny and cold
  flags in them are computed from observed weather, which a forecast could not read.
- **Arms.** Best single family chosen out of sample (per scored month, the worker with the lowest
  training score); fixed 50/50 blend of the GAM stack and BG2 (experiment 359); GAM stack; BG2; meta-model;
  meta-model without lagged power; meta-model without calendar context; plus climatology and
  manual_heuristic as references.
- **Scoring.** Quantile score, median MAE, correlation of the median with the actual, and tail
  coverage and pinball loss at q01, q05, q95, q99, by lead band (0 to 24, 24 to 72, 72 to 168 hours),
  month-resampled 95% intervals. Holiday and event days (bank holidays, 24 December to 2 January)
  are scored apart from the rest, and the rest checks non-inferiority.

## Scope limit

- **Reachable now.** D8 first. If the stop rule passes, D2 and D4 from the same stored frames.
- **Not reachable without new fits.** The six decision Primaries and the replicate year 2024-25 need
  rolling-origin refits of every worker, which are not stored. They are left to a follow-up the manager
  decides on, and the result page says so.
- **Serving cost.** Every worker runs for every substation and the meta-model adds one fit per refit.

## Advice followed

- Experiment 359's advice to choose a weight on one period and score it on the other is followed by
  the expanding window.
- Experiment 360's finding that no single member explains BG2's lead is why the best single family
  is chosen out of sample instead of being fixed in advance.

**Caveat on every headline number.** Interim, one substation, one scored year, 0 to 168 hours; it is
unproven whether the generator explains the steps and the gap.
