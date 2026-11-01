---
id: "00464"
slug: solar-efficiency-history-stage-0-post-hoc-scaling
title: >-
  Does multiplying the stored solar forecasts by each farm's recent efficiency, the ratio of actual
  power to the power the observed weather should have produced, lower the quantile score?
theme: metered-solar-farms
also_relevant_to:
  - generator-forecasting
  - calibration-and-uncertainty
status: planned
date: 2026-11-01
kind: other
size: >-
  medium. Stored: ratio tables, scaled-forecast scores and figures in this experiment's private
  folder only; the stored forecasts of experiment 430 are read and never changed. Designs: more than
  one defensible design exists (two definitions of expected power, raw or normalised multiplier), so
  the plan fixes both and scores every combination. Callers: this folder's scripts, which import
  experiment 310's scoring module and experiment 430's common module. Serving path and degradation
  rule: not touched (this repository holds none). Asset graph: not touched. Sonnet writes and
  reviews; no Opus.
hypothesis: >-
  Part of the solar forecast error is a slow change in each farm's efficiency (soiling, degradation,
  a level shift). The ratio of actual power to the power the observed weather should have produced,
  taken as a trailing median over weeks to months, estimates that efficiency, so multiplying the
  stored forecast quantiles by it lowers the quantile score at the farms whose output level changed
  (Solar A, Solar B, and Solar D, from experiment 453) and leaves the other farms unchanged.
prediction: >-
  Written before any score is read. Scores are quantile scores in points of capacity (lower is
  better) on the farm-group mean of the scored year, days 1 to 10. At least one scaling method
  (window, exponential level, or Kalman filter, with the normalised multiplier) lowers the mean
  score of Solar A, B, and D by more than 1% against the unscaled forecast of the same network, and
  changes the mean score of Solar C, E, and F by less than 1% in either direction. This can fail if
  the remaining error is weather error and not efficiency drift.
decision_rule: >-
  Primary arms are the 18 normalised-multiplier arms on the stored `a2_member` forecast (9 methods x
  the 2 causal sources of expected power, (b) and (c) below), with a 2-day delivery lag. Primary
  contrasts are the percentage change of each farm's mean score against the unscaled `a2_member`:
  108 contrasts (18 arms x 6 farms), month-block resampling of the 12 scored months, 200,000 draws,
  and intervals at the Bonferroni level 1 - 0.05/108 (99.954%). A method gains at a farm if the
  interval excludes zero on the negative side and the point estimate is below -1%. Group means of
  {Solar A, B, D} and {Solar C, E, F} are descriptive. The minimum detectable effect is the
  half-width of these intervals, reported first; where it exceeds 1% the arm cannot support a gain.
  The 1% margin is the maintainer's solar margin (the brief of 2026-11-01); it sits above zero
  because the serving path would need 112 days of delivered power and a stored expected power. The
  whole line stops if no primary arm gains at any single farm. If Solar C, E, or F also moves by
  more than 1% (point estimate), the part of the prediction that the other farms are unchanged
  fails, and the page says the drift is not specific to A, B, and D. The tails count as harmed at a
  farm if its 0.01 or 0.99 pinball loss rises by more than 0.02 points (the unscaled loss is
  reported beside it).
benchmark: >-
  The unscaled `a2_member` and `g0_onepass` forecasts of experiment 430 (seeds 0 to 2, averaged),
  same rows.
row_set_id: >-
  Experiment 310's rows as scored by experiment 430: Solar A to Solar F, scored year 2025-09 to
  2026-08, day bands d1 to d8_10, quarterly refits, rows with finite quantiles in all arms.
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), pinball loss
  and coverage at 0.01, 0.05, 0.95, and 0.99, median minus mean forecast
code: ml-experiments/00464-solar-efficiency-history-stage-0-post-hoc-scaling/
---

**Stage 0 of a four-stage line, run on CPU with no neural network fits.** The question for the whole
line is whether a long history of each farm's efficiency improves the solar forecast on days 1
to 10. Stage 0 asks it the cheapest way: take the stored forecasts of experiment 430 and multiply
their quantiles by a trailing estimate of efficiency. If no estimate helps any farm, the later
stages (ratio inputs to the network, a learned embedding, a state inside the network) stop.

## Plan

- **Standing practices followed.** Cleaned data: the validity rules below drop stuck-run,
  daylight-zero, and spike half-hours from the ratio (the flagged count is reported per farm).
  Stored predictions reused: experiment 430's `a2_member` and `g0_onepass` (seeds 0 to 2); nothing
  neural is refitted. No refits, so monthly cadence is moot. No calendar columns enter. The D15 and
  D18 cut is demand-only and not applicable. No fitted booster is saved because the expected-power
  boosters are small CPU models rebuilt in under a minute (state file written anyway).
- **Earlier advice followed.** Solar margin 1% (brief); farms labelled Solar A to Solar F;
  experiment 453's finding that Solar A, B, and D produced more in 2021 to 2023 than now motivates
  the group split; experiment 430's finding that `a2_member` and `g0_onepass` are within seed noise
  of each other, so both are scaled and the first is primary.
- **Data.** Power over effective capacity, CAMS clear-sky and observed irradiance, ERA5 temperature,
  and sun geometry from experiment 218's `farm_rows_cams.parquet` (2019-09 to the scored year). The
  observed weather exists only after about 5 days, and the ratio uses days that ended at least 6
  hours before the forecast origin, so every input is available at forecast time (the newest day
  used is two calendar days before the origin's date). The CAMS irradiance and ERA5 temperature
  delivery delays are not all known to be that short (ERA5 arrives about 5 days late), so the 18
  primary arms are also scored with a 6-day lag as a secondary check. The capacity rule is the one
  the stored forecasts use (experiment 453's decision has not merged); results are in points of
  capacity, and MW per farm waits for the capacity rule.
- **Expected power, three versions.**
    - **(a) Observed-weather model, leave-one-quarter-out.** One small XGBoost regressor per farm
      (CPU, 1 thread, depth 5, 200 rounds) predicts power over capacity from the observed GHI, DNI,
      DHI, clear-sky GHI, clear-sky index, temperature, sun elevation, and sun azimuth. It is
      cross-fitted by leaving out one year-quarter (for example January to March 2024) at a time.
      Caveat: the other quarters include later years, so the regressor knows the scored period, and
      absorbs part of the drift the ratio should show. (a) is not causal and is a secondary,
      diagnostic arm only.
    - **(c) Observed-weather model, forward-only.** The same regressor, but each year-quarter is
      predicted by a regressor trained on earlier year-quarters only. The first four year-quarters of
      a farm, which have no earlier data, use leave-one-quarter-out within that first year. (c) is
      causal apart from those four.
    - **(b) Clear-sky proxy.** Expected power is the farm's constant `k` times the clear-sky GHI, with
      `k` the median of power over clear-sky GHI over the farm's first 365 days of valid near-clear
      half-hours (observed GHI at least 0.9 times clear-sky GHI, clear-sky GHI above 300 W/m2). The
      ratio uses only near-clear half-hours, because over all half-hours cloud, not efficiency,
      dominates the ratio of power to clear-sky power. (b) is causal. The restriction to near-clear
      half-hours is a deviation from the brief's wording, chosen before any score is read.
- **Valid half-hours.** Drop null power, runs of 4 or more consecutive readings equal within 1e-4 at
  power above 0.02, exact-zero power while clear-sky GHI is above 200 W/m2, power above 1.5, and
  half-hours of a day that has a run of 4 or more equal readings at its maximum (the curtailment
  proxy; a flat top is the usual signature of an export limit).
- **Daily ratio.** Per farm and UTC day, the sum of power over the sum of expected power across
  valid half-hours whose expected power exceeds 0.10 (of capacity); source (b) also needs near-clear
  half-hours. The count of those half-hours as weight. A day with fewer than 4 such half-hours is
  missing.
- **Methods (9), each yielding a value at the end of every day.** Weighted median of the daily ratio
  over the last 7, 14, 28, 56, and 112 days (missing where fewer than 40% of the days are valid);
  exponentially weighted mean of the log daily ratio with half-lives of 14, 28, and 56 days (weights
  times the day weight); a local-linear-trend Kalman filter of the log daily ratio with observation
  variance set per farm from the robust spread (1.4826 x median absolute deviation) of the log daily
  ratio around its 14-day median over the history before the refit, level variance 0.01 times and
  trend variance 0.0001 times the observation variance. One setting, fixed now, not tuned. The
  multiplier is the filtered level on the last allowed day, with no trend extrapolation to the valid
  time.
- **Multiplier at an origin.** The method's value from the newest day at least two calendar days
  before the origin's date. Raw multiplier: that value. Normalised multiplier: that value divided by
  the weighted median of all daily ratios before the refit date of the network that produced the
  forecast, because the network already learned the average efficiency of its training years. A
  missing value gives multiplier 1. The multiplier is clipped to 0.5 to 1.5, and a scaled quantile
  is clipped to the largest value the unscaled network gave for the farm in the refit. Quantiles
  keep their order because the multiplier is positive.
- **Arms.** 18 primary arms (9 methods, normalised, on `a2_member`, for sources (b) and (c));
  secondary arms: the same 18 on `g0_onepass` (replication), the 9 source-(a) arms, the 9
  raw-multiplier arms of source (c), and the 6-day-lag versions of the 18 primary arms. The
  reference of every contrast is the unscaled forecast of the same network.
- **Scoring.** The experiment 430 scoring path (`e00310_score.row_losses` on sorted quantiles),
  seeds averaged quantile by quantile, same rows in every arm. Report the farm-group means ({A, B,
  D}, {C, E, F}), each farm, days 1 to 10 pooled and band d1 (12 to 36 hours) separately, tail
  calibration (0.01, 0.05, 0.95, and 0.99 pinball loss and coverage), median minus mean forecast,
  the mean-forecast error beside each quantile-score contrast, the absolute score of every arm and
  of the unscaled network beside each contrast, the seed count (3), the share of origins with a
  missing multiplier per farm and month, the share of scaled quantiles clipped, and the Bonferroni
  count (108). "Band d1" is the 12 to 36 hour lead band, the shortest the stored forecasts hold
  (their runs start at 00 UTC), and days 1 to 10 pool all stored bands. Production cost: the serving
  path would need 112 days of delivered power and a stored expected power per farm instead of
  weather only. MW per farm waits for the capacity rule.
- **Figures.** Daily ratio and its 56-day trailing median per farm over all years, for both sources;
  one explainability chart per farm of the trailing efficiency over the years beside the stored
  forecast's monthly median error in the scored year; the group score change of each method; the
  example day-ahead and 14-day forecasts, scaled and unscaled, for one farm per group; the
  multiplier over the scored year per farm.
- **Compute.** CPU only, about 1 to 3 hours; 1 to 2 cores, `OMP_NUM_THREADS=1`; scoring held to 16
  GB.
- **What this stage cannot say.** The stored forecasts cover only the scored year, so the
  explainability chart shows forecast error for that year only. Source (b) takes its ratio from
  near-clear half-hours and applies it to all-sky forecasts, which is untested. Source (b) is often
  missing in winter, which biases it toward no effect (the missing share is reported). A gain here
  is a gain from a post-hoc multiplier on a network trained without the ratio, so it bounds what
  stage 1 (ratio as a network input) can add only loosely. Source (a) leaks the whole record's
  average efficiency into the expected-power model.
