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
status: finished
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
verdict: partly supported
best_arm: >-
  Arm (c), the median lags, cuts the share of a short step that leaks into the median forecast but
  does not beat arm (a) on harm rows. At the signed-MW six, the share for a step lasting 48 hours
  is 0.101 against 0.217 for arm (a) (lead 0-240 h, a step of 5% of capacity).
headline: >-
  Replacing the day and week power lags by medians cuts the leak of a short step in recent power
  roughly in half (0.07 to 0.15 of the step for events up to 3 days, against 0.17 to 0.25), but the
  gain disappears at 7 days (0.37 to 0.40 against 0.39 to 0.43). Clean-row CRPS is +0.050 at the
  six, inside the noise floor of a repeated fit (0.06), and harm rows show no gain. Interim and
  unaudited.
lessons: >-
  A median of recent same-time readings shields the forecast from a stuck meter and from events
  shorter than the median window, and nothing else. A dropout to zero and a spike still move it, and
  the origin lag, which the median leaves single, leaks more than in arm (a).
production_status: not-for-production
adopted_in: none
take_home_message: >-
  Median day and week lags make the forecast more robust to short events and stuck readings, not
  more accurate, and the manager decides whether that is worth porting. A step of 5% of capacity in
  the last 48 hours moves the median forecast by 0.10 of the step at the six Primaries metered in
  signed megawatts, against 0.22 for the original lags. The shielding fades at 7 days, a dropout and
  a spike are not shielded, and the loss on rows with no labelled event (+0.050) is within the
  repeat-fit noise (0.06). Interim and unaudited, four scored months.
advice:
  - >-
    Treat median lags as a robustness option, not an accuracy gain. They halve the leak of events
    up to 3 days and fully shield a stuck meter, at a clean-row loss of +0.050 that four scored
    months cannot separate from zero (experiment 465).
  - >-
    Keep the 50/50 mixture with the lagless forecast as the main protection. The mixture of arm (c)
    and the lagless forecast does not beat the mixture of arm (a) and the lagless forecast on harm
    rows (experiment 465).
  - >-
    If median lags are ported, clean dropouts and spikes upstream, since the median does not
    shield them, and decide how the served model treats a missing or stale day (experiment 465).
code: >-
  ml-experiments/00465-median-day-and-week-lags-for-load-shift-robustness/
---

**Median day and week lags make the median forecast about half as sensitive to a step in the last
2 to 3 days of power, but not more accurate.** The shielding shrinks as the event lengthens and is
gone at 7 days. A stuck meter is shielded almost completely, but a dropout to zero and a spike are
not. The clean-row loss against the original lags is +0.050 points of capacity at the six Primaries
metered in signed megawatts, which is inside the 0.06 difference between two fits of the same
recipe. NGED's event labels are incomplete, so "rows with no labelled event" may hold unlogged
events, and every real-event slice undercounts. Every number is interim and unaudited.

![Each arm minus each control, by row set](figure_1_contrasts.svg)

![Share of a step that reaches the median forecast, by event length](figure_2_sweep_share.svg)

![Response to a 30-day level change](figure_3_level_change.svg)

![Day-ahead forecast at D5, winter](figure_4_day_ahead_winter.svg)

![Fourteen-day forecast, summer](figure_5_fourteen_day_summer.svg)

![Forecast across a labelled event at D14](figure_6_event_D14.svg)

![Forecast across a labelled event at D3](figure_7_event_D3.svg)

![Forecast across a labelled event at D12](figure_8_event_D12.svg)

![Forecast across a labelled event at D19](figure_9_event_D19.svg)

## Results

Scores are CRPS-like pinball loss in points of effective capacity; a contrast is the first arm minus
the second, so negative favours the first. Arms: (a) the original lags (experiment 402), (b) the
lagless forecast (experiment 392), (c) the median lags, (e) the 50/50 mixture of the probability
distributions of (c) and (b), (f) the 50/50 mixture of (a) and (b). Months are the resampling unit
(4 scored months, 3 with labelled events). Intervals are 95% month-block intervals; the 99.72%
Bonferroni intervals for the 18 pre-registered contrasts are in `contrasts.csv`.

- **Speed.** The 64 fits of arm (c) took 6,442 s on 6 CPU cores, including the median feature build.
- **Sweep share, signed-MW six, lead 0-240 h, step 5% of capacity.** For an event of 24, 48, and
  72 hours, arm (c) moves the median forecast by 0.068, 0.101, and 0.147 of the step, against 0.172,
  0.217, and 0.241 for (a). At 5 and 7 days the shares are 0.276 and 0.366 for (c), 0.311 and 0.408
  for (a). At the MVA-only eight the shares are 0.097, 0.131, and 0.174 for (c) against 0.201, 0.240,
  and 0.259. The lagless control moves by exactly 0. A step of 2.5% gives the same shares to within
  0.02.
- **Short events leak slightly more.** At an event of 6 hours, (c) leaks 0.042 against 0.021 for (a)
  at the six, because the origin lag, which the median leaves single, carries more weight in (c).
  Moving the origin lag alone by 5% moves the median by 0.038 of the step in (c) against -0.013 in
  (a) at the six, and 0.068 against 0.023 at the MVA-only eight.
- **Band width and coverage.** The 5% to 95% band width does not change with the injected step in
  any arm (ratio 0.996 to 1.003). Clean-row coverage of the 5-95 band is 0.853 for (c) and 0.863 for
  (a) at the six for an event of 48 hours, and 0.898 and 0.906 at a widening of 1.15. Event-row
  coverage is 0.722 for (c) and 0.764 for (a) at 48 hours, and 0.785 and 0.825 at 1.15.
- **Bad data, mean shift of the median, signed-MW six, lead 24-72 h (points of capacity).** Stuck
  24 h: -0.036 for (c), -2.308 for (a). Dropout to zero for 6 h: +1.141 for (c), +0.926 for (a).
  Spike x20: +4.340 and +3.192. Spike x4: +2.706 and +1.616. The median shields a stuck meter but
  not a dropout or a spike, because a single bad reading at the origin lag passes straight through.
- **Level change (5% of capacity for 30 days).** The median forecast follows the new level slowly at
  first and further later: 0.310 of the step by day 6, 0.511 by day 24, 0.562 by day 30 for (c) at
  the six, against 0.373, 0.456, and 0.490 for (a).
- **Clean rows, signed-MW six (rows with no labelled event, lead 0-240 h).** (c) minus (a) is
  +0.050 [-0.004, +0.087], with 3 of 4 months worse. (c) minus (b) is -0.101 [-0.285, +0.159]. (c)
  minus (f) is +0.082 [-0.002, +0.220]. (e) minus (f) is +0.014 [-0.008, +0.045]. The early stop
  rule (clean-row loss of 0.10 or more) does not fire.
- **Noise floor.** Refitting arm (a) on CPU with this experiment's settings and scoring it against
  experiment 402's stored forecast on the 2026-01 refit month gives a difference of -0.060 at the
  six and +0.002 at the MVA-only eight. The +0.050 clean-row loss of (c) is therefore inside the
  noise of a single refit at the six.
- **Harm rows (during a labelled event), signed-MW six.** (c) minus (a) is -0.207 [-0.219, +0.827],
  1 of 3 months worse. (c) minus (f) is -0.769 [-1.001, +0.304]. (e) minus (a) is +0.588 [+0.575,
  +0.971], 3 of 3 months worse. No arm beats (a) with an interval that excludes zero (the
  prediction P3 required -0.1 with an interval excluding zero).
- **Harm rows, MVA-only eight.** (c) minus (a) is +0.690 [+0.050, +0.970], 3 of 3 months worse, and
  (e) minus (a) is -1.423 [-1.911, -0.248], 0 of 3 months worse. The groups disagree in sign.
- **After a labelled event ends (days 7 on), signed-MW six.** (c) minus (a) is +0.194 [-0.188,
  +0.345]. (c) minus (f) is +0.852 [+0.111, +1.232], 3 of 3 months worse. At the MVA-only eight, (c)
  minus (a) is -0.123 [-0.358, -0.007].
- **Avoidable curtailment cost (widening 1.15, units of the cost report).** On the three Primaries
  with event rows, (a) 3.394e6, (b) 2.152e6, (c) 3.422e6, (e) 2.374e6, (f) 2.218e6. Over all rows at
  the six, (a) 4.507e7, (b) 5.615e7, (c) 4.660e7, (e) 4.602e7, (f) 4.526e7. Arm (c) costs slightly
  more than (a) in both.
- **12-36 h lead band, (c) minus (a).** Signed-MW six: clean rows +0.030 [-0.006, +0.064],
  harm rows -0.188 [-0.221, +0.814], after the event +0.371 [+0.019, +0.713]. MVA-only eight: clean
  rows +0.015 [-0.040, +0.063], harm rows +0.789 [-0.039, +1.114]. The signs match the 0-240 h
  band.
- **Predictions.** P1 holds: the share for events up to 48 hours is 0.068 to 0.101 at 24 to 72 hour
  leads (under 0.15), and rises towards (a)'s share at 7 days (0.366 against 0.408). P2 holds
  narrowly: +0.0498 against the limit of +0.05, inside the noise. P3 fails: harm rows show no gain
  with an interval excluding zero at the six, and (c) is worse at the MVA-only eight. P4 fails: the
  origin lag alone moves arm (c) by 0.054 at 24 to 72 h at the six (predicted 0.01 to 0.05) and 0.087
  at the MVA-only eight.
- **Checks.** Median minus mean forecast per site was not computed, because the stored tables hold
  quantiles and no mean forecast. The robust lags assert the leakage rule of `origin_lags`. The
  noise-floor difference (-0.060 at the six) exceeds the half-width of the clean-row interval, so
  the clean-row loss is reported as inside the noise. The harm-row contrast changes sign between
  the two groups, so no pooled claim is made. The leave-one-Primary-out check was not run.

## Caveats

- **Events longer than 7 days defeat any 7-day median.** At 7 days the share is 0.37 to 0.40.
- **The median follows a genuine level change more slowly early.** It reaches 0.31 of a persistent
  step by day 6 against 0.37 for (a), then overtakes it.
- **Unlogged events.** NGED's event labels are incomplete, so rows with no labelled event may hold
  events, and harm-row counts are lower bounds.
- **CPU against GPU.** The fits ran on CPU and experiment 402's on CUDA. The noise-floor control
  bounds the difference at 0.06 at the six, one month only.
- **Three event months and 18 contrasts.** The Bonferroni family is 18 contrasts. The (c) minus (e)
  contrasts are descriptive.
- **Not run.** Arm (d) (robust plus original lags, demoted by the brief), the cold-night rows, the
  NRA-corrected actuals of experiment 455, and the CRPS at widenings 1.05 and 1.10.
- **Production options for a missing or stale day.** If a day of power is missing, the 7-reading
  median can use the remaining readings, as the fit does, or fall back to the lagless forecast.
  This page describes the options and leaves the choice to the maintainer.
