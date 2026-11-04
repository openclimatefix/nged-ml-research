---
id: "00548"
slug: lag-free-release-candidate-assembly-with-trailing-temperature-windows
title: >-
  Does the daily-refit lag-free demand network with trailing temperature windows get within reach of
  the lag-enabled network, once weather is limited to the run issued at the forecast origin?
theme: lagged-power-and-error-correction
also_relevant_to:
  - demand-inputs
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: pretrained weights, fitted networks, and raw quantile forecasts in this
  experiment's private data folder only; every earlier experiment is read and never written. One
  defensible design: experiment 511's L-1d recipe (ERA5 and CAMS pretraining, daily refit,
  90-day half-life, flat 1.15 widening) plus the three trailing temperature window columns of
  experiments 515 and 526. Callers: the scripts import experiments 397, 428, 510, 511, 515, 521, and
  539. Serving path: not touched (this repository holds none). Degradation rule: not touched.
  Cost: about 2 GPU-hours or less (two pretraining folds and about 300 daily fits per seed).
hypothesis: >-
  The best lag-free recipe assembled from parts that already have evidence (ERA5 and CAMS
  pretraining, daily refit with a 90-day half-life, trailing temperature windows, flat 1.15
  widening) reaches the score of the lag-enabled network A-pool at 120 to 240 h of lead but not at 24
  to 48 h, where the gap is mostly the recent-power inputs.
prediction: >-
  Written before any score is read. Contrast sign: first arm minus second arm in points of CRPS,
  negative favours the first. P1: at the eight, the assembled network E minus A-pool lies inside
  the minimum detectable effect (MDE) at 120 to 240 h (L-1d is already at +0.082 against an MDE of
  0.16) and lies above the MDE at 24 to 48 h, where L-1d minus A-pool is +0.288; the expected E minus
  A-pool there is about +0.23 to +0.27, because experiment 526 found a gain of about 0.06 points from
  the windows on all rows at the eight, mostly from one fold. P2: E minus L-1d at the eight is
  between -0.08 and 0.00 in every band from 24 h on, and its interval excludes zero in at most the
  bands from 24 to 120 h. P3: E minus the retrained K2-S1 is not shown to differ from zero at 120
  to 240 h. P4: E leaks an injected load-shift no more than L-1d does (4.49 points of capacity, plus
  0.5). Any miss is stated plainly in the write-up.
decision_rule: >-
  Contrasts counted for the Bonferroni correction: E minus A-pool, E minus L-1d, and E minus
  R-DW (experiment 539), per lead band (7), at the eight and at the six (2), so 42 tests. All
  other contrasts are report-only. Intervals are month-block bootstraps with 5 blocks (near the
  minimum to maximum of the folds), with per-fold signs and the MDE beside each contrast. A result
  is worded "interval excludes zero" or "not shown". The smallest gain worth acting on is any whose
  interval excludes zero, because the windows add three input columns and no other cost in
  production. E is a release candidate only if E minus L-1d excludes zero in the predicted
  direction at the eight without a loss at the six larger than the MDE.
benchmark: >-
  L-1d (experiment 511, two seeds), A-pool (experiments 425 and 533), R-DW (experiment 539), and the
  K2-S1 rebuilt on origin-time weather (experiment 541), all on experiment 533's rows.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity by lead band (0 to 6 h, 6 to 12 h, 12 to 24 h, 24 to 48 h,
  48 to 72 h, 72 to 120 h, 120 to 240 h) at the eight MVA-only and the six signed-MW Primaries,
  on rows with no labelled event, with a cold-night slice, the q95 and q99 pinball loss, and the
  injected-shift leak in points of capacity.
---

**Plan only.** Folds: the five folds of experiment 511 (2025-10, 2025-12, 2026-01, 2026-03,
2026-06), two seeds, July 2026 neither scored nor tuned on. Experiment 526's ten winter folds hold
no L-1d, A-pool, or R-DW scores, so they cannot be compared with the lag-enabled arms and are used
only through experiment 526's own W-T result. Dropped from the assembly: the 4-head mixture (experiment 527
found head count does not explain the gap), per-lead widening (experiment 523 found at most 0.024
points from it), and a two-stage ensemble calibrator (no stored follow-up of experiment 524 supports
it).
