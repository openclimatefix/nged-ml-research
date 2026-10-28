---
id: '00368'
slug: d8-conditional-band-head-on-gam-stack-residuals
title: >-
  Does one forecast-band head, fitted on the GAM stack's own residuals and pooled across Primaries,
  improve D8's uncertainty without hurting other Primaries?
theme: calibration-and-uncertainty
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
size: >-
  medium. Stored: private files under data/ml-experiments/00368-*/ only. Serving path: not touched.
  Degradation rule: not touched (a band head is an R&D device here). Designs: more than one is
  defensible (XGBoost quantile offsets, a mixture density network), so the plan fixes both as arms.
  Callers: the scripts import experiments 252, 264, and 354 and call their scoring and as-of code.
  Asset graph: not touched.
hypothesis: >-
  One band head fitted on every Primary's as-of residuals of the GAM stack, and conditioned on lead,
  hour, weather, the site, and the stack's recent residuals, closes at least half of D8's gap to BG2
  in quantile score, because it widens D8's band where D8's errors are large and does not shrink the
  other Primaries' bands at long lead the way experiment 354's D8-only band did.
prediction: >-
  Written before any 368 score is read. Control is experiment 354's best arm (fit b monthly, arm 3
  band), scoring 8.382 on 96,986 D8 rows. The better of the two heads scores 7.97 or less at D8
  (closing at least half of the 0.789-point gap between 8.382 and BG2's 7.593). At D2 and D4 the
  head scores no more than 3% above the stored donated-spread forecast. The pooled score over the
  seven other Primaries is no more than 0.02 points above the stored donated-spread forecast. The
  share of D8 readings below q01 falls from 4.3% (arm 3) to 1.5% or less.
decision_rule: >-
  Contrasts (paired, month-resampled 99% intervals, Bonferroni for the four planned contrasts): C1,
  best head minus control at D8; C2, best head minus stored forecast at D2; C3, the same at D4; C4,
  the same pooled over the seven other Primaries. Supported when C1 is at or below -0.41 (half the
  gap to BG2 closes: D8 at 7.97 or lower), C2 and C3 are no more than +3% of the stored score, and C4
  is no more than +0.02 points. The threshold for C1 is set by the cost of a new pipeline step that
  every Primary would carry. Stop rule from the brief: if the best head closes under 25% of the gap
  (D8 above 8.18), or a placebo fails, stop band work. Placebo: the same head fitted on
  features permuted across rows within each site and month. The placebo fails if it scores within
  0.05 points of, or better than, the real head at D8. Control gate: the script reproduces 8.382 at
  D8 for experiment 354's arm 3 before any head is read.
benchmark: >-
  Experiment 354's best arm at D8 (fit b monthly, arm 3 band, 8.382; weekly 8.338 from experiment
  355 as a reference), the stored donated-spread forecast at the other Primaries, BG2, climatology,
  and the manual heuristic from experiment 87
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of the series' effective
  capacity; share of readings below q01 and q05 and above q95 and q99; coverage by lead band
code: ml-experiments/00368-d8-conditional-band-head-on-gam-stack-residuals/
date: 2026-10-28
---

**Plan only: no score has been read.**

D8 is a primary substation whose forecast band, in experiment 354, was rebuilt from D8's own past
errors. That cut D8's score by 2.27 points but raised D2's by 5.5% and D4's by 6.1%, because the
band did not widen with lead time. This experiment fits one band head on the residuals of the
stack's stored median at eight Primaries and conditions it on lead, hour of day, weather, the site,
and the stack's recent residuals.

## The five plan items

- **Arms.** Control is experiment 354's arm 3 at D8, and the stored donated-spread forecast at the
  seven other Primaries (D2, D3, D4, D12, D19, D21, D22). Arm X is XGBoost quantile offsets around
  the stored median (one `reg:quantileerror` model for all 11 quantiles, median offset fixed at 0).
  Arm M is a three-component Gaussian mixture density network on the residual. Arm X-shuffled is
  arm X fitted on permuted features, as the placebo. Experiment 354's arm 2 is the comparison at
  every Primary.
- **Features.** Lead hours, the stored median, the stored band width (q95 minus q05), local hour,
  day of week, temperature, the unit solar shape, solar elevation, a site code, and the as-of mean
  absolute residual of the last 24 hours and of the last 7 days. No month or day-of-year column, so
  the head cannot memorise the calendar.
- **Fit and rows.** Refit at the start of each calendar month on residuals whose reading arrived
  (valid time plus 6 hours) before that month, so nothing from the future enters. Rows before the
  first fit with 28 days of resolved data keep the stored band, as in experiment 354. Scored rows
  are the scored year 2025-07-02 to 2026-06-30.
- **Scoring.** The quantile score, tail shares, and coverage by lead band, at D8, D2, D4, and the
  pooled seven, with month-resampled 95% intervals from `e00252_score`.
- **Checks.** Gate, the median-minus-mean and mean-forecast error per arm, a leave-one-site-out
  check of C4, and the audit triggers.

## Not part of this plan

- The "six decision Primaries" of the brief are not stored in experiment 252's forecasts, so the
  pooled set is the seven other Primaries that are. This is a stated deviation.
- The curtailment-cost metric needs a node and its generators, so it is not run here.
- Lab shortcuts: stored forecasts, hindsight-free as-of residuals with a 6-hour reporting lag,
  horizon 0.5 to 168 hours (NGED needs 1 to 10 days).

Caveat for every headline number: interim, one substation, one scored year; it is unproven whether
the generator explains the steps and the gap.
