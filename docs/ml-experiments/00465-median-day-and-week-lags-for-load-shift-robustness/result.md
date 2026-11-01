---
id: "00465"
slug: median-day-and-week-lags-for-load-shift-robustness
title: >-
  Do the median of the last 7 same-time-of-day readings and of the last 3 same-time-of-week readings,
  in place of the day and week power lags, make the XGBoost demand forecast far less sensitive to a
  load-shift event that lasts under 7 days?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
date: 2026-11-01
kind: other
protocol: demand-v2
size: >-
  medium. Stored: 64 boosters and 64 forecast tables of arm (c) in this experiment's private data
  folder; the injection sweeps write small summary tables. Production serving path: not touched (this
  repository holds none). Degradation rule: not touched here, but the write-up states what a served
  model with median lags would do with a missing or stale day. Designs: the injection into the
  readings (not into feature columns) is fixed by the brief. Callers: the scripts import experiment
  87's, 383's, 392's, and 402's modules.
hypothesis: >-
  Replacing the day lag and the week lag by the median of the 7 most recent same-time-of-day readings
  and of the 3 most recent same-time-of-week readings makes the median of an XGBoost demand forecast
  move by under 15% of a step added to the readings of the last 48 hours (against 38% to 41% for the
  single-reading lags of experiment 392), at a loss of under 0.05 points of capacity on rows with no
  labelled event.
prediction: >-
  Written before any score is read. Scores are CRPS in points of effective capacity. P1. Arm (c)'s
  sweep share for event lengths up to 48 hours is under 0.15 at 24 to 72 hour leads, against 0.38 to
  0.41 for arm (a), and rises towards arm (a)'s share at 7 days. P2. Arm (c)'s CRPS on rows with no
  labelled event at the signed-MW six is within +0.05 of arm (a)'s (experiment 388's week-median
  columns lost 0.049, so this is at risk). P3. Arm (c) beats arm (a) on harm rows by at least 0.1.
  P4. Arm (c) with the origin lag alone moved at k = 1 still leaks 0.01 to 0.05, as in experiment 392.
decision_rule: >-
  Robust lags change the feature pipeline only, so the smallest gain worth porting is zero on harm
  rows with a 95% interval that excludes zero, plus a loss under 0.05 on rows with no labelled event.
  Stop after arm (c) if its sweep share for event lengths up to 48 hours exceeds 0.25, or its CRPS on
  rows with no labelled event is worse than arm (a)'s by 0.10 or more. Arm (d), the robust lags plus
  the original lags, is demoted and runs only if arm (c) shields the bad-data injections but loses
  clean rows. Contrasts: arms (c) and (e) each minus (a), (b), and (f) on clean rows, harm rows, and
  real-event rows at the signed-MW six: 2 x 3 x 3 = 18 (arm (d) adds 9 if it runs, giving 27), with a
  Bonferroni family level of 95%. If arm (c) shields the bad-data injections but does not beat arm (a)
  on harm rows, the verdict is "more robust, not more accurate" and the manager decides.
benchmark: >-
  Arm (a), experiment 402's stored R0 lagged forecast (`work/arm_preds/r0_lagged`, 64 files); arm
  (b), experiment 392's stored R0 lagless forecast; arm (f), their fixed 50/50 mixture of
  probability distributions, on identical rows.
row_set_id: experiment_392_monthly_four_folds
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of effective capacity from the 11 quantiles of
  the 51-member mixture, on the intersection of finite actuals and all arms' forecasts.
---

**Plan only: no score has been read.** The question is whether a forecast that reads the median of
recent days, not the single reading one day or one week before the valid time, ignores a short
load-shift event in its recent power. Arm (c) is refitted at the same four monthly refits as
experiments 392 and 402 (2025-10-01, 2026-01-01, 2026-03-01, 2026-06-01; 16 Primaries; 64 fits).

## Plan

- **Recipe.** Experiment 87's `n1h` XGBoost member as run by experiment 392 (copied through
  experiment 402's `e00402_refit.py`): monthly refits, 500 rounds, `eta` 0.05, 11 quantiles, the
  ECMWF ENS control member for training, 51 members pooled as a mixture at scoring, `colsample` 1.0,
  plain targets (arm `r0`), flagged `substation_zero` half-hours dropped, D15 and D18 cut before
  2025-03-01 (the same cleaned readings experiment 87 holds). The neural loops of experiments 397 and
  428 and the cross-fitted tables of experiments 404 and 449 do not apply.
- **Lags of arm (c).** `lag_origin` as before; `lag_day_med7`, the median of the 7 readings at the
  valid time's time of day on the 7 most recent days ending at the newest usable reading;
  `lag_week_med3`, the median of the 3 readings at its time of day and weekday on the 3 most recent
  weeks. Missing readings are skipped. The leakage rule of `origin_lags` is asserted again.
- **Reused.** Arm (a) is experiment 402's `r0_lagged` (experiment 392's own `r0_lagged` folder is
  incomplete); arm (b) is experiment 392's `r0_lagless`. Arms (e) and (f) need no fit: they are
  mixtures of probability distributions (not quantile averages) of stored quantiles, as in
  experiment 400's `arm_mix_fixed`.
- **Compute.** 64 CPU fits with at most 6 cores, no GPU, memory under 10 GB. Experiment 402's 64 CPU
  fits took 15,217 s. The first fit is timed, including the median feature build, and reported to
  the manager before the rest start.
- **Noise floor.** The minimum detectable effect comes from the reproduction of arm (a) (experiment
  402's stored forecast against a second run of the same recipe on one refit month, at most 16
  extra fits), stated before any contrast is read.
- **Readings-level injection sweep.** Add a step of 5% and of 2.5% of capacity, both signs, to the
  half-hourly readings in [origin − L, origin] for L = 6 h, 24 h, 48 h, 3 days, 5 days, and 7 days,
  rebuild every lag from the modified readings, and predict with the stored boosters. Report the
  median shift as a share of the step by L and lead band for arms (a), (c), (e), (f), and the lagless
  control (which must follow exactly 0), the band-width ratio, and coverage. Repeat the old
  column-level sweep on arm (c)'s `lag_origin` alone as a check. Widening 1.0, 1.05, 1.10, and 1.15
  on arm (c).
- **Bad data and level change.** A stuck reading (24 h constant), a dropout to zero (6 h), and a
  spike (4 to 20 times the stage-1 residual median absolute deviation) injected into the Primary's own
  readings; and a persistent step of 5% of capacity lasting 30 days, reporting the days arm (c) takes
  to reach 90% of it.
- **Scoring.** Absolute CRPS beside every contrast, a 12 to 36 hour lead band on its own, tails
  (q01, q05, q95, q99 coverage and pinball loss), avoidable curtailment at f = 0.9, 0.75, and 0.5,
  cold nights, month-block intervals with the sign count over the 4 months, and whole-event
  intervals. NGED's event labels are incomplete, so the clean-row harm table is optimistic and the
  real-event slices undercount.
- **Caveats to state.** Events longer than 7 days defeat any 7-day median; the median follows a
  genuine level change late; the day median needs 7 days of power history and the week median 3
  weeks at the serving origin.
- **Prior evidence followed.** Experiment 388 (monthly arm B: the mean and median of six weekly
  analogues added to the three lags scored +0.049 worse at the signed-MW six); experiment 392 (a
  step of 5% in all three lags moves the median 38% to 41% of the step); experiment 458 (keep the
  fixed 50/50 mixture).
