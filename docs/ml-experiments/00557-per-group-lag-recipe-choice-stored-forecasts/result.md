---
id: "00557"
slug: per-group-lag-recipe-choice-stored-forecasts
title: >-
  Does one demand network for all 14 Primaries lose accuracy against using the lag-light network at
  the eight and the lag-free network at the six?
theme: lagged-power-and-error-correction
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  simple. Stored: only a private scores folder; no fit, no new data, no new method. One defensible
  design: the per-row scores of experiments 548 (arm E) and 553 (arm RDWT-D15) are re-scored, and
  the per-group option is a row selection, so it needs no forecast of its own. Callers: the script
  imports experiments 521, 553, and 87. Serving path: not touched (this repository holds none).
  Degradation rule: not touched. Cost: CPU only, 2 cores, minutes.
hypothesis: >-
  Using the lag-light network (15% lag dropout, readings present) at the eight MVA-only Primaries
  and the lag-free recipe at the six signed-MW Primaries scores differently from the lag-light
  network at all 14, because at the six the lag-free recipe already matches the lag-enabled network
  A-pool.
prediction: >-
  Written before any score is read. Sign: lag-light minus lag-free, negative favours lag-light.
  P1: at the six, over 24-120 h, lag-light minus lag-free lies between -0.10 and 0.00 (lag-light is
  better), which fails the 0.03 test from the better side; the per-group choice then gives up
  accuracy at the six. P2: at the eight, option (a) equals option (c) by construction, and lag-free
  everywhere (b) trails it by 0.10 to 0.14 over 24-120 h (experiments 548 and 553 report 4.555
  against 4.438). P3: in Christmas week at the eight, lag-light is no worse than lag-free by more
  than 0.10. P4: the injected-shift leak of lag-light (5.33 points of capacity) is above that of the
  lag-free recipe (4.55) and under the 6.0 limit, as stored.
decision_rule: >-
  Fixed before scoring. Option (a) passes if its CRPS is within 0.03 points of option (c) at the six
  and at the eight, in the headline band 24-120 h and over all leads; it fails otherwise, and a
  failure at the six with (a) worse than (c) means one network for all 14 costs accuracy at the
  six, while a failure with (a) better means the per-group choice gives up accuracy. At the eight
  (a) and (c) are the same network, so the eight half of the test holds by construction. Counted
  contrasts for the Bonferroni correction: lag-light minus lag-free, per lead band (7), at the eight
  and at the six (2), so 14 tests. All other contrasts are report-only. Intervals are month-block
  bootstraps with 5 blocks (near the minimum to maximum of the folds), with per-fold signs and the
  minimum detectable effect (from the two seeds of L-1d, experiment 511) beside each contrast. Wording
  is "interval excludes zero" or "not shown". The threshold for acting is zero, because the options
  differ only in which stored recipe runs at which Primary; the operational cost of each option is
  stated in one sentence and nothing is adopted.
benchmark: >-
  Lag-light RDWT-D15 (experiment 553), lag-free E (experiment 548), A-pool (experiment 425), L-1d
  (experiment 511), and `manual_heuristic` for scale, all read from stored per-row scores on
  experiment 533's rows.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at the eight MVA-only and the six signed-MW Primaries by
  seven lead bands; D15 and D18 report-only; July 2026 unused; slices from experiment 551.
---

Plan only. The write-up replaces this page when the scores are read.
