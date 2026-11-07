---
id: "00619"
slug: bsp-gsp-lag-enabled-network-for-the-bake-off
title: >-
  Does a lag-enabled demand network forecast the 2 bulk supply points and the 2 grid supply points
  better than the lag-free network at days 1 to 3?
theme: hierarchy-and-pooling
also_relevant_to: []
status: planned
kind: other
date: 2026-11-07
size: >-
  Medium, by the five triggers. (1) It stores fitted weights, forecasts, and a Parquet export under
  this experiment's private data folder. (2) One design is defensible: the maintainer named it (shared
  lag encoder with lag age, lags at least 24 h old, the D5 offset, no gate). (3) It spans code whose
  callers cannot be named without searching: the scripts import experiments 87, 397, 415, 428, 448,
  510, 511, 559, 560, 572, 574, 577, 591, and 611. (4) It touches no serving path and no degradation
  rule. (5) It changes no contract. One Sonnet code review before the first fit, one Sonnet diff
  review before merge.
hypothesis: >-
  At the 2 bulk supply points (BSPs) and the 2 grid supply points (GSPs), a demand network that
  also reads the power of one day and one week earlier forecasts better than the lag-free network
  at days 1 to 3, and no better after day 4.
prediction: >-
  Written before any score is read. The lag-enabled network beats the lag-free network of
  experiment 611 (Design 1) at days 1 to 3 (leads of 0 to 72 h) by at least the minimum detectable
  effect, in both groups, and ties it at days 4 to 10 (leads of 72 to 240 h), meaning the
  contrast interval includes zero. The q05 to q95 band keeps covering fewer than 0.90 of rows.
decision_rule: >-
  Descriptive, because the 5 development months give wide intervals. The primary contrast is
  lag-enabled minus Design 1 in pooled CRPS, per group, at days 1 to 3 and at days 4 to 10. It
  counts as a gain when its Bonferroni month-block interval (5 blocks) excludes zero and the point
  estimate is beyond the minimum detectable effect. The smallest gain worth porting is the minimum
  detectable effect, because the network adds a lag input pipeline to the serving path.
benchmark: >-
  Experiment 611's Design 1 (lag-free network with the O2 offset) and Design 2 (experiment 415's
  lag network), BG2 (weekly XGBoost model), the manual heuristic, and the climatology, all stored,
  scored on the rows shared with the lag-enabled network.
row_set_id: experiment_611_nodes_v1
metric: >-
  CRPS and the mean absolute error of the median, in points of effective capacity, and q05 to q95
  coverage, by day of lead time (1 to 10) and by group (BSPs, GSPs), with the GSP that mostly exports
  power reported separately.
code: ml-experiments/00619-bsp-gsp-lag-enabled-network-for-the-bake-off/
production_status: not-for-production
adopted_in: none
---

Planned. The plan is in the pull request description; the write-up replaces this page when the
fits finish.
