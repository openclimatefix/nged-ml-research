---
id: "00541"
slug: k2-s1-rebuilt-on-origin-time-weather
title: >-
  How much of K2-S1's score survives when its lag-free heads use only weather from the run issued at
  or before the forecast origin?
theme: lagged-power-and-error-correction
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: three cells of quantile forecasts per scored month, scores, and a correction table
  in this experiment's private data folder; no model weights are kept. Designs: one defensible (keep
  experiment 454's S1 heads, which already read origin-time weather, and replace the lag-free heads
  by refits fed origin-time weather). Callers: the scripts import experiments 404, 454, 521, and
  531's code. Serving path: not touched (this repository holds none). Degradation rule: not touched.
  Cost: about 15 small GPU fits per month for five months, under 1 GPU-hour in total.
hypothesis: >-
  Experiment 404's lag-free heads read, for each target, the run published 9.5 to 33 hours before
  the target's valid time, whatever the forecast origin. That is a look-ahead at long lead, and it
  inflates K2-S1 (two S1 heads plus two lag-free heads) at lead times beyond about 24 hours.
  Rebuilding K2-S1 with lag-free heads that read the run issued at or before the forecast origin
  removes the look-ahead.
prediction: >-
  Written before any score is read. P1: the rebuilt K2-S1 matches the old K2-S1 at 0 to 24 h, within
  the minimum detectable effect (MDE), because the two weather feeds are identical there. P2: the
  rebuilt K2-S1 is worse than the old one at 120 to 240 h by about 0.17 points of CRPS at the
  MVA-only eight and about 0.20 at the signed-MW six, which are the gaps that experiment 531
  measured between L-1d and the old K2-S1 at those leads. Both predictions can fail.
decision_rule: >-
  Contrasts: rebuilt minus old K2-S1, per lead band, at the eight and the six, on rows with no
  labelled event. P1 holds if every band below 24 h has a contrast inside plus or minus its MDE.
  P2 holds if the 120 to 240 h contrast lies within 0.05 of the predicted value and its
  month-block interval excludes zero. The smallest effect worth acting on is any whose interval
  excludes zero, because the correction changes only a reference score, at no cost in production.
  Intervals are month-block bootstraps with 5 blocks (near the minimum to maximum of the folds),
  with per-fold signs. The Bonferroni count is 28 (7 lead bands, 2 row sets, 2 contrasts against
  the old K2-S1 and against L-1d). Other contrasts are report-only.
benchmark: >-
  The old K2-S1 (experiment 454, stored cell c0 scores in experiment 511's rows), L-1d (511), A-pool
  (clean), A-pool minus lags (537), and R-DW (539).
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity by lead band (0 to 6 h, 6 to 12 h, 12 to 24 h, 24 to 48 h,
  48 to 72 h, 72 to 120 h, 120 to 240 h) at the eight and the six, as in experiments 521 and 531.
---

# K2-S1 rebuilt on origin-time weather

Planned. The result is written after the scores exist.
