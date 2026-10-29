---
id: "00390"
slug: xgboost-d11-d14-gap-and-load-shift-event-lags
theme: production-reference-and-best-guess
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v1
size: >-
  simple. Stored: result tables and figures in this experiment's private folder only; experiments
  377, 379, 380, 381, and 383 are read and never written. Designs: one (re-score stored forecasts;
  no refit). Callers: the scoring script imports experiment 377's folder and
  `studies.eval_protocol`. Production serving path and degradation rule: not touched.
hypothesis: >-
  The weekly XGBoost bundle of experiment 381 trails the GAM stack at D11 and D14 because it does
  not track the recent level of demand, as the GAM stack's Kalman level filter does, and not
  because a load-shift event lies in its power lags. When a lag lies inside a load-shift event, the
  weekly bundle's upper tail fails and its median is pulled towards the lag.
prediction: >-
  Written before any score is read. P1. A causal offset equal to the median of the weekly bundle's
  own residuals over the previous 7 days closes at least half of the bundle's CRPS gap to the GAM
  stack at D11 and at D14. P2. At D11 and D14 the gap is the same in size at days 1 to 3 and at
  days 8 to 10 (experiment 383's interim re-score found +0.36 and +0.37 at D14). P3. On rows with a
  lag inside a labelled event, q95 coverage falls below 0.85 and the 90% band is at most 10% wider
  than on rows with clean lags. P4. The signed median error on those rows has the sign of the
  event (the median sits at the lagged level), and its size grows with the event's depth.
decision_rule: >-
  Descriptive re-scoring of stored forecasts under `demand-v1`: nothing is refitted, so no
  Bonferroni family is set and no adoption claim is made. Part A reads "level tracking is the cause"
  as supported when the 7-day offset closes at least half of the gap at both D11 and D14, with a
  95% month-resampled interval for the offset's own change excluding zero. Part B resamples whole
  events; a slice with fewer than 5 events is descriptive only. The threshold of zero applies to
  anything that costs nothing to port, such as a different column.
benchmark: >-
  The GAM stack of experiment 377 (stack without S below 168 h) and BG2, on `fold_a_daily_gap_v1`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles, mean absolute error of the
  median, pinball loss and coverage at q01, q05, q95, and q99, in points of effective capacity,
  from `studies.eval_protocol`; leads 0.5 to 336 hours; headline band days 1 to 10 and, for Part
  A, days 11 and 14.
code: ml-experiments/00390-xgboost-d11-d14-gap-and-load-shift-event-lags/
date: 2026-10-29
---

# Why the weekly XGBoost bundle trails the GAM stack at D11 and D14, and what a load-shift event in the lags does to it

Planned. The full design is in the private brief
`briefs/xgboost_d11_d14_gap_and_event_lags_analysis_brief.md`. Experiment 383's figures are cited
as interim until its pull request merges. The maintainer's rule of monthly refits does not apply,
because nothing is refitted.
