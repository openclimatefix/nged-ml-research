---
id: "00576"
slug: lag-gap-by-half-hour-type-eight-mva-and-six-mw-primaries
title: >-
  Why do lags beat the lag-free output offset at the eight MVA Primaries but not at the six MW
  Primaries: sunny half-hours or one Primary?
theme: lagged-power-and-error-correction
also_relevant_to: []
status: planned
kind: other
protocol: demand-v2
size: >-
  simple. Stored: one small score table in the private data folder, nothing else. One defensible
  design: a re-score of experiment 572's stored forecasts, with no new fit and no new data. Callers:
  the code imports experiment 572's scoring module and experiment 565's helpers (named by
  search). Serving path and degradation rule: not touched (this repository holds none).
hypothesis: >-
  (a) Lags correct the lag-free network's poor modelling of reverse flow at sunny half-hours at the
  eight MVA Primaries, so the CRPS gap (E2 minus v3, and E2 minus v3+D5) concentrates in sunny
  half-hours. (b) Much of the gap at the eight MVA Primaries comes from one Primary, D8, which
  has a gas generator.
prediction: >-
  Written before any scoring. Pa: at the eight MVA Primaries at 24-120 h, the gap per half-hour is
  at least twice as large in sunny daytime half-hours as in cloudy daytime half-hours, and the
  three largest half-hour types hold more than half of the total gap (fails if the ratio is below
  1.5). Pb: leaving D8 out removes at least 40% of the E2 minus v3+D5 gap at the eight MVA Primaries
  (fails if it removes less than 20%). At the six MW Primaries the gap is near zero in every
  half-hour type.
decision_rule: >-
  A gap counts when its Bonferroni month-block interval (5 blocks, near the minimum to maximum of
  the five folds) excludes zero. Bonferroni count 12: 2 contrasts (E2 minus v3, E2 minus v3+D5) x 2
  groups x 3 lead bands, on all rows. Every table by half-hour type, Primary, hour of day, or season
  is descriptive, with its own uncorrected interval and the minimum detectable effect beside it.
  This is an explanation of an existing result, so no porting threshold applies.
benchmark: >-
  E2 (experiment 565's D5, stored), v3 and v3+D5 (experiment 572, stored), M0 report-only.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity; gap = first arm minus second, positive means the first arm
  is worse.
---

Plan. Re-score the stored forecasts of experiment 572 by half-hour type; no fit runs.
