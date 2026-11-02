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
status: finished
verdict: partly supported
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
best_arm: >-
  No single arm. Dropping the day lag shields events up to 3 days and bad data (a stuck meter, a
  dropout) at +0.049 CRPS; dropping the week lag cuts the 7-day leak at +0.049 CRPS but leaks more
  for short events. Dropping the origin lag scores -0.019 CRPS (experiment 388's arm E, re-scored).
headline: >-
  On rows with no labelled event at the six Primaries metered in signed megawatts (leads to 10
  days), CRPS against all three lags is +0.049 for no week lag [-0.034, +0.102], +0.049 for no day
  lag [+0.015, +0.113], and -0.019 for no origin lag [-0.034, -0.011]; the noise floor (a repeat
  fit) is -0.0002. Interim and unaudited.
lessons: >-
  The leak is not carried by one lag. The day lag carries the short-event leak and the sensitivity
  to bad data, and the week lag carries part of the 5 to 7 day leak. Removing either costs about
  0.05 CRPS, and removing the week lag makes short events and bad data worse.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  Dropping a single power lag does not make the XGBoost demand forecast robust to a load-shift
  event at every length. Without the day lag, a step of 5% of capacity in the last 3 days does not
  move the median forecast (0.00 of the step, against 0.06 with all three lags), but a
  7-day step still moves it by 0.29, and the score loss is +0.049 points of capacity. Without the
  week lag, the 7-day leak falls from 0.33 to 0.22, but a step of 24 hours to 5 days leaks 0.20 to
  0.22 and a stuck meter moves the median by 2.2 points of capacity (0.8 with all three lags). Interim and
  unaudited, two Primaries and two refit dates for the leak figures.
advice:
  - >-
    Do not drop the week lag to reduce the leak of short events or bad data: it raises both (24
    hour step 0.20 of the step against 0.06, stuck meter 2.2 against 0.8 points of capacity), at
    +0.049 CRPS (experiment 502, figure 1).
  - >-
    If a single lag is dropped for robustness, drop the day lag: it shields steps up to 3 days and
    a stuck meter or dropout, at +0.049 CRPS [+0.015, +0.113] on rows with no labelled event, but
    the interval just exceeds the 0.11 equivalence limit (experiment 502, figure 2).
  - >-
    Score the origin lag removal (arm E of experiment 388, -0.019 [-0.034, -0.011]) before
    spending fits on it, and sweep arm E once to see whether the origin lag leaks (experiment 502).
code: >-
  ml-experiments/00502-leave-one-lag-out-power-lag-ablation-xgboost-demand/
---

**Dropping one power lag does not remove the leak at every event length, and every single-lag
drop costs about 0.05 points of capacity (CRPS) on rows with no labelled event.** The day lag
carries the leak of events up to 3 days and the sensitivity to a stuck meter or a dropout. The
week lag carries part of the leak of events of 5 to 7 days. Dropping the week lag makes short
events and bad data worse. The three lags are power readings 1 day, 2 days, and 7 days (the origin,
day, and week lags) before the forecast time. NGED's event labels are incomplete, so "rows with no
labelled event" may hold unlogged events, the harm rows are optimistic, and the real-event slices
undercount. Every number is interim and unaudited.

![Share of a step that reaches the median forecast, by event length](figure_1_sweep_share.svg)

![Change in CRPS against all three lags](figure_2_crps_contrasts.svg)

## Setup

- **Arms.** All fitted like experiment 402's arm (a): monthly refits (2025-10-01, 2026-01-01,
  2026-03-01, 2026-06-01), 500 rounds, `eta` 0.05, 11 quantiles, CUDA, six Primaries metered in
  signed megawatts (D5, D9, D10, D11, D14, D16). "No week lag" and "no day lag" are 24 fits each.
  "a-rep" is a second CUDA fit of the three original lags (24 fits). "No origin lag" is experiment
  388's arm E, re-scored on the same rows with no fits; experiment 388 fitted it separately, so its
  device and code path may differ from arm (a)'s.
- **Cadence.** Monthly refits, as the standing rule says for research. The contrasts could depend
  on the schedule; no weekly check was run.
- **Reused.** Arm (a) from experiment 402 (`r0_lagged`), arm (b) (no lags) from experiment 392
  (`r0_lagless`), the 50/50 mixture of (a) and (b) (f), and experiment 465's `sweep.parquet` for
  arms (a) and (b). Code reused: experiments 87, 383, 392, 402, and 465.
- **Rows.** The intersection of finite actuals and every arm's forecast, valid time before
  2026-07-01, leads to 240 hours, no flagged zeros dropped at these six Primaries (0 in every fold).
  Rows with no labelled event: 326,964; harm rows (a lag lies in an event and the target does not):
  5,449; rows inside events: 8,740; rows within 7 days after an event: 14,441. The 12 to 36 hour
  band has 34,214 rows with no labelled event.
- **Noise floor.** Arm (a) minus a-rep is -0.0002 CRPS [-0.0012, +0.0006], so the device and seed
  noise is below 0.02 and the equivalence limit stays 0.11 (experiment 465's range).

## Results

CRPS in points of effective capacity, six Primaries, leads to 240 hours, no widening, intervals are
month-block bootstraps at the 95% Bonferroni level (k = 12). With 4 months (harm rows 3), each
interval is the range of the monthly differences, so the equivalence statements are screen-level.
Smaller is better.

- **Absolute scores, rows with no labelled event:** all three lags 3.096; no lags 3.247; 50/50
  mixture 3.064; no week lag 3.145; no day lag 3.145; no origin lag 3.077.
- **Contrast against all three lags, rows with no labelled event:** no week lag +0.049
  [-0.034, +0.102] (inside the 0.11 limit, so equivalent); no day lag +0.049 [+0.015, +0.113]
  (the upper end is just outside the limit); no origin lag -0.019 [-0.034, -0.011]. The sign count
  over the four months: no week lag is worse in 3 of 4 months and no day lag in 4 of 4.
- **Contrast against no lags, rows with no labelled event:** no week lag -0.102 [-0.268, +0.038];
  no day lag -0.102 [-0.257, +0.115]; no origin lag -0.170 [-0.404, +0.060].
- **12 to 36 hour band, rows with no labelled event:** no week lag +0.038 [-0.034, +0.110]; no day
  lag +0.110 [+0.069, +0.175]; no origin lag -0.016 [-0.061, +0.001].
- **Within 7 days after an event (absolute):** all three lags 4.950; no lags 4.184; mixture 4.293;
  no week lag 4.415; no day lag 5.146; no origin lag 4.944. Dropping the week lag recovers about
  two thirds of the gain from dropping all lags here.
- **Harm rows (descriptive; the intervals span about 0.5 to 1 point and cannot show equivalence):**
  all three lags 3.897; no lags 3.686; mixture 3.570; no week lag 3.718 (-0.179 against all three
  lags, interval -0.271 to +0.383); no day lag 4.085 (+0.188 [-0.055, +0.249]); no origin lag 3.949.
- **Tails, rows with no labelled event (no widening; with the 1.15 widening no interval changes
  sign):**
  q99 pinball loss 0.341 (all three lags), 0.345 (no week lag), 0.343 (no day lag), 0.338 (no
  origin lag), all within the +0.02 limit. q95 coverage 0.902, 0.895, 0.898, and 0.905.
- **Leak of a step of 5% of capacity (readings-level sweep, 24 to 72 hour leads, pooled over two
  Primaries (D5, D16) and two refit dates (2026-01-01, 2026-06-01); median shift as a share of the
  step):**

| Step length | All three lags | No week lag | No day lag | No lags |
|---|---|---|---|---|
| 24 hours | 0.06 | 0.20 | -0.01 | 0 |
| 3 days | 0.06 | 0.22 | -0.01 | 0 |
| 5 days | 0.19 | 0.22 | 0.13 | 0 |
| 7 days | 0.33 | 0.22 | 0.29 | 0 |

- **Bad data (median shift in points of capacity, 24 to 72 hour leads; the injection is
  experiment 465's):** a stuck meter for 24 hours: -0.82 (all three lags), -2.20 (no week lag),
  -0.02 (no day lag). A dropout to zero for 6 hours: -0.19, -1.35, -0.09. A spike of 20 times the
  week-on-week median change: +3.79, +5.32, +5.50. The band width ratio stays between 0.98 and 1.19.
- **Predictions.** P1 partly: no week lag costs +0.049 (inside the limit), but the readings-level
  leak is 0.20 to 0.22, not at most 0.20, and the column-level sweep was not run. P2 not tested (the
  sweep of the origin lag was not run). P3 not supported for short events: the day lag arm leaks
  -0.01 up to 3 days and 0.13 at 5 days, and 0.29 at 7 days is inside 0.20 to 0.30. P4 holds: the
  50/50 mixture (3.064) beats every new arm on rows with no labelled event.
- **Stop rule.** Not triggered, because the week lag arm cuts the 7-day leak below 0.25 at a loss
  under +0.30. The MVA-only eight were not fitted, because no arm cuts the leak at every length
  without raising it at another.

## Checks and caveats

- **Not run.** Avoidable curtailment, cold nights, the whole-event intervals for event slices, the
  column-level sweep of every lag subset, example forecast time series, and load-shift event plots.
  The leak figures use two Primaries and two refit dates because of the brief's cost limit, so they
  are a screen.
- **Audit triggers.** Two fired: every interval rests on 4 months (3 for harm rows), and the leak
  figures rest on 4 site-refit cells whose spread is wide (for example 0.16 to 0.37 for the no week
  lag arm), so both are screens. The mean-forecast check could not be run because the stored
  forecasts hold quantiles only.
- **Unlogged events.** NGED says load-shift events may be missing from its logs, so clean rows can
  hold events and the harm-row table is optimistic.
- **Device.** The arms of experiment 502 and arm (a) are CUDA fits. Arm E of experiment 388 may be a
  CPU fit; its contrast against arm (a) can carry up to 0.06 to 0.11 of device difference
  (experiment 465).

## What the next experiment should answer

- Does a day-lag-free forecast mixed 50/50 with the no-lag forecast match the mixture of all
  three lags (3.064) on rows with no labelled event and after events?
- Does the readings-level sweep of the origin lag arm show a leak above 0.06?

Scripts: `e00502_fit.py`, `e00502_score.py`, `e00502_sweep.py`, `e00502_figures.py`. Predictions:
`work/arm_preds/{a_rep,nowk,noday}` (private data folder); boosters in `work/boosters/`.
