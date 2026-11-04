---
id: "00550"
slug: lag-light-network-trailing-temperature-windows-with-placebo
title: >-
  Do trailing temperature windows still help the lag-light demand network once the day-ago and
  week-ago power lags are in?
theme: lagged-power-and-error-correction
also_relevant_to:
  - demand-inputs
status: planned
kind: other
size: >-
  Medium. Stored: GPU fits of two arms, 2 seeds, five folds of daily refits, written to the private
  data folder. One design is defensible (experiment 539's harness plus experiment 548's window
  columns). The script imports the harnesses of experiments 511, 515, 526, 539, and 548. No serving
  path or degradation rule is touched.
hypothesis: >-
  The three trailing temperature windows (means of the ensemble-mean 2 m temperature over the 6,
  24, and 72 hours ending at the valid time) still lower the error on cold winter nights when the
  network also reads the day-ago and week-ago power lags (R-DW of experiment 539), because the
  windows describe the weather of the last three days and a lag of one day or one week carries only
  one reading of it.
prediction: >-
  Written 2026-11-04 before any arm was scored. On cold nights (below 5 degrees C, 18:00 to 08:00)
  at the eight MVA-only Primaries, (R-DW + W-T) minus (R-DW + W-Tnull) is at most -0.05 points of
  capacity (CRPS). Experiment 526 found -0.093 without the lags.
decision_rule: >-
  Supported if that contrast is at most -0.05 and its month-block interval excludes zero. Not
  supported if the contrast is above -0.02 (the lags already carry the information). Between the
  two, partly supported. The 0.05 is set by the cost of serving three more input columns and a
  weather-window table. Intervals are month-block bootstrap with 5 blocks (near the min-to-max
  range), per-fold signs, the minimum detectable effect beside every contrast, and the Bonferroni
  count stated as a number in the write-up.
benchmark: >-
  R-DW (experiment 539), L-1d plus W-T (experiment 548, arm E), A-pool (experiment 425 arm A);
  manual_heuristic as a reference.
row_set_id: >-
  main16 of experiment 533 (the five folds 2025-10, 2025-12, 2026-01, 2026-03, and 2026-06). July
  2026 is unused.
metric: >-
  CRPS in points of capacity at the eight MVA-only and the six signed-MW Primaries (rows with no
  labelled event), by the seven lead bands. D15 and D18 are report-only.
production_status: not-for-production
adopted_in: none
---
**Planned.** Arms: R-DW + W-T and R-DW + W-Tnull (the matched placebo of experiment 526: the same
three columns taken from a shuffled other day). The results are written once the fits finish.
