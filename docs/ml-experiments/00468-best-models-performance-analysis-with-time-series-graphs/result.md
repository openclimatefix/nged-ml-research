---
id: "00468"
slug: best-models-performance-analysis-with-time-series-graphs
title: >-
  How do the best wind, solar, and demand forecasters behave hour by hour and event by event, and
  where do they fail?
theme: production-reference-and-best-guess
also_relevant_to:
  - generator-forecasting
  - lagged-power-and-error-correction
status: planned
kind: other
protocol: none
size: >-
  medium. Stored: tables and figures only, nothing fitted (private folder
  data/ml-experiments/00468-best-models-performance-analysis-with-time-series-graphs/). Several
  designs would show the same behaviour (which weeks, which events, which scenarios), so the plan
  fixes the selection rules before any chart is read. Callers: the scripts import the folders of
  experiments 404, 428, 454, 456, and 461. Production serving path and degradation rule: not
  touched. One Sonnet code review before the first scoring run, one Sonnet results review, one
  Sonnet prose review.
hypothesis: >-
  The best forecasters' failures are concentrated in a few identifiable situations (a stormy week
  for wind, a cloudy week for solar, and the first days of a load shift that the lagged power
  does not reflect), and a set of time-series graphs makes those situations visible.
prediction: >-
  Written before any chart is read. P1. Wind quantile bands cover q05 and q95 within 3 points in
  each month, and the share above q99 stays under 1.5% in every month. P2. The worst 20
  half-hours by q99 pinball loss at each farm sit in a few storm days and not in scattered
  half-hours. P3. In real load-shift events, the forecasts that use the lagged power follow the
  shifted power with a delay of at most 1 day while the lagless forecast stays near the pre-event
  level, and the lagged forecasts' errors during the event are larger than the lagless forecast's
  in at least half the labelled events.
decision_rule: >-
  This is a descriptive analysis, so no arm is adopted. A prediction counts as holding when the
  stated number holds on the rows shown. Contrasts are reported with month-block bootstrap
  intervals at a Bonferroni level set by the count of contrasts, and the minimum detectable effect
  beside each. Interim numbers are labelled interim and unaudited.
benchmark: >-
  Wind: research default 1, research default 2, and climatology. Solar: experiment 310's blend,
  the XGBoost model, climatology, and the manual heuristic. Demand: BG2 (to 168 hours only), the
  manual heuristic, climatology, and the monthly GAM as a point forecast. No wind or solar GAM,
  BG2, or manual-heuristic quantile forecast exists, so none is invented.
row_set_id: none
metric: >-
  Quantile score in points of effective capacity and in MW (private store only), coverage and
  pinball loss at q01, q05, q95, and q99 per month, error by lead time, and the avoidable-curtailment
  cost at f = 0.9, 0.75, and 0.5 for the demand scenarios.
code: ml-experiments/00468-best-models-performance-analysis-with-time-series-graphs/
---

# Experiment 468: how the best forecasters behave, hour by hour and event by event

Planned. Written 2026-11-01. The brief is `briefs/best_models_performance_analysis_brief.md` in the
private store, with its edits of 2026-11-01.

## Plan

- **Part A, wind and solar.** Farm A, Farm B, Farm C, Solar A, and Solar B only. Best arms: wind
  `member_plain` at p99.9 capacity; solar `a2_member` and `g0_onepass` at the old capacity rule
  (experiment 461 did not adopt p99.9 for Solar A). Experiment 466 left these arms unchanged.
  Rows are experiment 461's stored rows; references are cut to the same rows.
- **Part A graphs.** Day-ahead against actual for chosen weeks; a 14-day wind forecast; quantile
  fans; rolling coverage by month; error by lead time; the worst 20 half-hours at q99 and q01; the
  effect of table P and the capacity rule. Experiment 461's private fans and worst-20 file are
  reused, not redrawn.
- **Part B, real events first.** The labelled events of experiment 404 and the scan events of
  experiment 456, shown for S1, B-mix, and K2-S1 with the lagless forecast. The injected-shift
  scenarios (about 8) wait for the manager's go.
- **Standing practices followed:** cleaned data, D15 and D18 cut before 2025-03-01, stored
  predictions reused, no fits, draft PR, generators anonymised (no dates on generator axes).
