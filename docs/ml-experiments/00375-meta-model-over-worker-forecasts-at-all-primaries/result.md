---
id: '00375'
slug: meta-model-over-worker-forecasts-at-all-primaries
title: >-
  Does the meta-model over worker forecasts beat a 50/50 blend across the 16 Primaries, not only at
  D8?
theme: calibration-and-uncertainty
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
size: >-
  medium. Stored: private files under
  data/ml-experiments/00375-meta-model-over-worker-forecasts-at-all-primaries/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: more than one is defensible (one
  meta-model per Primary, or one pooled over Primaries), so the plan fixes experiment 373's
  per-Primary design with no new tuning. Callers: the scripts read experiment 87's stored member
  forecasts through experiment 360's rebuild, the stored GAM-stack forecasts of experiments 155,
  252, and 355, and call experiment 373's meta-model and experiment 252's scoring helpers. Asset
  graph: not touched.
hypothesis: >-
  The XGBoost meta-model of experiment 373, which weights BG2's five members, the GAM stack, and BG2
  by context, scores better than the fixed 50/50 blend of the GAM stack and BG2 across the 16
  Primaries, so one meta-model architecture works for any substation.
prediction: >-
  Written before any score is read. Pooled over the Primaries that have a stored GAM stack, the
  meta-model scores at least 0.1 points of quantile score below the 50/50 blend, with a 95% interval
  wholly below zero, and scores below the blend at a majority of those Primaries. Experiment 373
  found -0.028 at D8 alone, so this prediction can fail, and the plan expects it to.
decision_rule: >-
  Paired contrasts, whole valid months resampled (20,000 resamples, seed 0), 95% intervals, on the
  scored months October 2025 to June 2026 at each Primary (the first 3 valid months only train the
  meta-model, as in experiment 373). C1 meta-model minus the 50/50 blend, pooled as the mean over
  the Primaries with a stored stack, and per Primary. C2 meta-model minus the best single worker
  chosen out of sample (BG2 excluded). C3 meta-model minus BG2. `supported` when C1 pooled is at
  most -0.1 with an interval wholly below zero and C1 is below zero at more than half of the
  Primaries. `small gain` when C1 pooled is wholly below zero but above -0.1. `not supported`
  otherwise. The threshold is 0.1 because the meta-model needs every worker to run for every
  substation in production. Primaries with no stored stack (D15, D18) take BG2 as the blend
  baseline, are reported on their own, and are left out of the pooled figure. Cold winter nights
  and sunny April-to-June half-hours are reported as numbers, not as decision-rule contrasts.
  Gate: the rebuilt BG2 reproduces experiment 360's D2, D3, D4, and D8 scores, and the stored stack
  scores match their source experiments, before any meta-model is fitted.
benchmark: >-
  The 50/50 blend of the GAM stack and BG2, each worker alone, BG2, the GAM stack, climatology, and
  manual_heuristic, on the same rows at each Primary
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of the series' effective
  capacity; median absolute error; q01, q05, q95, q99 coverage
code: ml-experiments/00375-meta-model-over-worker-forecasts-at-all-primaries/
date: 2026-10-28
---

**Plan.** Experiment 373 found no gain from a meta-model at D8 (-0.028 points against the 50/50
blend, 95% -0.145 to +0.094). The maintainer wants the same test at all 16 Primaries, because the
aim is one architecture that works for any substation.

- **Workers.** BG2's five members and BG2, rebuilt for all 16 Primaries from experiment 87's stored
  refit predictions with experiment 360's code (no fits). The GAM stack is read from experiment
  252 (D2, D3, D4, D8 as in 355, D10, D11, D12, D19, D21, D22) and experiment 155's arm D (D5, D9,
  D14, D16). D15 and D18 have no stored stack.
- **Meta-model.** Experiment 373's, unchanged: one XGBoost quantile model per Primary, trained on an
  expanding window of earlier valid months, target the actual minus the 50/50 blend's median. At
  D15 and D18 the base is BG2 and the stack columns are absent.
- **Rows.** The scored year (2025-07-02 to 2026-06-30), leads up to 168 hours, outside the
  hand-filtered load-shift events, with a reading.
- **Prior advice followed.** Experiment 359: test a blend at Primaries where the stack beats BG2
  (this experiment does, by construction). Experiment 373: store rolling-origin forecasts (the new
  work folder holds the joined worker frames of every Primary).
- **Caveat on D8 numbers.** It is unproven whether the generator explains the steps and the gap.
  Lead range 0 to 168 hours, shorter than NGED's 1 to 10 days.
