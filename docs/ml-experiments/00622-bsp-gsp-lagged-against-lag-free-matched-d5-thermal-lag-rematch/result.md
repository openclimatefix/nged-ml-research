---
id: "00622"
slug: bsp-gsp-lagged-against-lag-free-matched-d5-thermal-lag-rematch
title: >-
  Does a lag-enabled demand network beat a lag-free network at days 1 to 3 at the 2 bulk supply
  points and the 2 grid supply points, when both use the D5 offset and the thermal-lag columns?
theme: hierarchy-and-pooling
also_relevant_to: []
status: planned
kind: other
date: 2026-11-07
size: >-
  Medium, by the five triggers. (1) It stores fitted weights, forecasts, thermal-lag tables, and a
  Parquet export under this experiment's private data folder. (2) One design is defensible: the
  maintainer named it (D5 offset and thermal-lag columns in both arms, only the lags differ). (3) It
  spans code whose callers cannot be named without searching: the scripts import experiments 87, 397,
  415, 428, 437, 448, 510, 511, 559, 560, 562, 569, 572, 574, 577, 591, 611, and 619. (4) It touches
  no serving path and no degradation rule. (5) It changes no contract. One Sonnet code review before
  the first fit, one Sonnet diff review before merge.
hypothesis: >-
  At the 2 bulk supply points (BSPs) and the 2 grid supply points (GSPs), a demand network that
  reads the power of one day and one week earlier forecasts better than a lag-free network at days 1
  to 3, once both arms use the D5 offset and the thermal-lag and solar-gain columns, and the two arms
  tie from day 4.
prediction: >-
  Written before any score is read. The lag-enabled network beats the matched lag-free network at
  days 1 to 3 (leads of 0 to 72 h) by at least the minimum detectable effect, in both groups, and
  ties it at days 4 to 10 (leads of 72 to 240 h), meaning the contrast interval includes zero. This
  can fail: experiment 619 found a gain of 0.12 points (BSPs) and 1.28 points (GSPs) against
  minimum detectable effects of 0.49 and 3.64, with an unmatched comparator.
decision_rule: >-
  The primary contrast is lag-enabled minus lag-free, both with D5 and thermal-lag columns, in
  pooled CRPS per group at days 1 to 3 and at days 4 to 10 (family of 4, Bonferroni count 4,
  month-block interval over the 5 development months). It counts as a gain when the interval excludes
  zero and the point estimate is beyond the minimum detectable effect, because the lag network adds a
  lag input pipeline to the serving path. Secondary contrasts (family of 24 with the primary ones,
  Bonferroni count 24): the lag effect without thermal-lag columns, the thermal-lag effect inside each
  arm, and each arm against BG2 and Design 1 of experiment 611.
benchmark: >-
  Experiment 619's stored lag-enabled forecast (D5, no thermal-lag columns), experiment 611's
  Design 1, BG2, the manual heuristic, and the climatology, all stored.
row_set_id: experiment_611_nodes_v1
metric: >-
  CRPS and the mean absolute error of the median, in points of effective capacity, and q05 to q95
  coverage, by day of lead time (1 to 10) and by group (BSPs, GSPs), with tail calibration at q01,
  q05, q95, and q99.
code: ml-experiments/00622-bsp-gsp-lagged-against-lag-free-matched-d5-thermal-lag-rematch/
---

**Planned: the maintainer asked for experiment 619's comparison to be repeated with the confounds
removed.** Experiment 619 compared a lag-enabled network (D5 offset, no thermal-lag columns) with
experiment 611's lag-free network (O2 offset). The two differed in the offset as well as the lags.
This experiment refits both arms with the D5 offset and with the thermal-lag and solar-gain columns
of experiment 569, on the same folds, months, seeds, and recipe, so the lags are the only
difference. The lag-free arm is the lag-enabled network with the day-lag and week-lag columns set to
"absent" in every row, in pre-training, fine-tuning, and scoring.

Arms (each with seeds 0 and 1, five development months, no gate, no load-shift handling, no
neighbour inputs, July 2026 untouched):

- **Lag-enabled with thermal-lag columns** and **lag-free with thermal-lag columns**: the primary
  pair.
- **Lag-free without thermal-lag columns**: completes the two-by-two with experiment 619's stored
  lag-enabled forecast, so the thermal-lag effect can be read inside each arm.
