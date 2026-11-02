---
id: "00502"
slug: leave-one-lag-out-power-lag-ablation-xgboost-demand
title: >-
  Does dropping only the week lag, only the day lag, or only the origin lag cut the leak of a step in
  the power lags into the XGBoost demand median at a smaller score loss than dropping all three?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
date: 2026-11-02
kind: other
protocol: demand-v2
size: >-
  medium. Stored: 72 boosters and 72 forecast tables (A-nowk, A-noday, and a CUDA repeat of arm (a),
  "a-rep") in this experiment's private data folder; sweeps write small summary tables. More than one
  design: no (the brief fixes the leave-one-lag-out arms). Callers: the scripts import experiment
  87's, 383's, 392's, 402's, and 465's modules. Production serving path and degradation rule: not
  touched (this repository holds none).
hypothesis: >-
  Dropping the week lag alone removes about 20 points of the 38 to 41% leak that experiment 392
  measured with all three lags affected, at a score loss on rows with no labelled event inside the
  equivalence limit of 0.11 points of effective capacity.
prediction: >-
  Written before any score is read. P1. A-nowk, with a step in both lags it keeps, leaks at most 0.20
  of a 5% step at 24 to 72 h leads, and its CRPS loss against arm (a) on rows with no labelled event
  is at most +0.11; if the loss is larger, it falls between (a) and (b). P2. A-noorigin leaks the
  same as (a) within 0.05 with all its lags affected (the origin lag carries about 1%). P3. A-noday,
  with both its lags affected, leaks 0.20 to 0.30. P4. No new arm beats the fixed 50/50 mixture (f)
  on rows with no labelled event.
decision_rule: >-
  Dropping a lag removes a column and costs nothing at serving, so the smallest gain worth porting
  is zero on the sweep share, at a CRPS loss on rows with no labelled event whose whole 95%
  Bonferroni interval lies inside +-0.11 ("equivalent") and a q99 pinball loss within +0.02 of arm
  (a). Harm rows are descriptive: their intervals span about 1 point, so they cannot show
  equivalence. Stop rule: if no single-lag removal cuts the 3-lag-equivalent leak below 0.25 without
  a loss above +0.30, close the line and say the leak is spread across the lags; the MVA-only eight
  are not run unless one arm passes. A drop that is more robust but less accurate is reported as
  "more robust, not more accurate" and the manager decides. Contrasts: A-nowk, A-noday, and the
  re-scored A-noorigin, each minus (a) and minus (b), on rows with no labelled event and on harm rows
  at the six: 3 x 2 x 2 = 12, Bonferroni k = 12 at a 95% family level.
benchmark: >-
  Arm (a), experiment 402's stored R0 lagged forecast; arm (b), experiment 392's stored R0 lagless
  forecast; arm (f), their fixed 50/50 mixture of probability distributions; arm (a-rep), this
  experiment's CUDA refit of arm (a); A-noorigin is experiment 388's arm E, re-scored.
row_set_id: experiment_392_monthly_four_folds
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of effective capacity from the 11 quantiles of
  the 51-member mixture, on the intersection of finite actuals and all arms' forecasts, valid time
  before 2026-07-01.
---

**Which power lag carries the information and which carries the leak? Status: planned; no score has been read.** The plan follows the brief of the load-shift Discuss
session (2026-11-02) and its queue-order edits.

## Plan

- **Recipe.** Experiment 392's R0 arm, as copied by experiment 402's refit script: monthly refits at
  2025-10-01, 2026-01-01, 2026-03-01, and 2026-06-01, 500 rounds, `eta` 0.05, 11 quantiles, the ENS
  control member for training, 51 members pooled at scoring, CUDA. Flagged `substation_zero`
  half-hours are dropped from fits and scoring actuals. The noisy start of D15 and D18 is not
  relevant, because only the six signed-MW Primaries (D5, D9, D10, D11, D14, D16) are fitted.
- **Retraining cadence.** Monthly, because this is a screen-level contrast of a column set and not a
  cadence question. A candidate that passes is re-checked weekly before any release claim.
- **New fits (72).** A-nowk (no `lag_same_time_latest_week`), A-noday (no `lag_same_time_latest_day`),
  and a-rep (the three original lags, refitted on CUDA once more), each for 6 Primaries and 4
  refits. A-noorigin is not fitted: its score is experiment 388's arm E re-scored on this
  experiment's equal rows (192 stored forecasts), and it is fitted (24 fits) only if its sweep share
  is wanted after A-nowk and A-noday report.
- **Reused.** Arm (a) from experiment 402 and arm (b) from experiment 392 (stored forecasts);
  experiment 465's `sweep.parquet` for arms (a) and (b). Checked before use: same series, init
  times, leads, and scoring rows (equal-row intersection), and the 1.15 widening applied once.
- **MDE.** Printed first, from arm (a) minus a-rep on the four dates. Experiment 465's
  `a_cpu_control` covers one refit date and compares CPU with GPU, so it is not the MDE here. If
  (a) minus a-rep is under 0.02, the device and seed noise is small and the equivalence limit stays
  0.11.
- **Scoring.** Experiment 392's and 465's scorers on equal rows: rows with no labelled event, harm
  rows, rows inside events (normal-running-arrangement corrected actuals, depth 1.0, with 0.75 and
  1.25 as sensitivities), rows within 7 days after events, and cold nights. Month-block bootstrap
  (4 blocks) with the sign count beside each interval; event slices use whole-event intervals. Tails
  (q01, q05, q95, q99 coverage; q95 and q99 pinball) raw and with 1.15 widening; avoidable
  curtailment at f = 0.9, 0.75, and 0.5 as differences against (a), (b), (f), BG2, S1, and
  manual_heuristic. The 12 to 36 h band is reported on its own.
- **Sweeps.** Experiment 392's column-level sweep and experiment 465's readings-level sweep and
  bad-data injections on the new arms, first on 2 Primaries at 2 refit dates, extended only if a
  result is worth it. Every scorer, sweep, and injection filters `valid_time` < 2026-07-01
  explicitly, because stored forecasts run to 2026-07-15.
- **Unlogged-event caveat.** NGED's event labels are incomplete, so rows with no labelled event may
  hold events, the harm-row table is optimistic, and real-event slices undercount.
- **Advice followed.** Report absolute scores beside every contrast; equal rows; MDE before
  contrasts; Bonferroni k = 12; the six and the eight reported apart.
