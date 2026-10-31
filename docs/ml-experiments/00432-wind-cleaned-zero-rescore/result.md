---
id: "00432"
slug: wind-cleaned-zero-rescore
title: >-
  Does scoring the stored wind forecasts against a cleaned-zero target, where the idle half-hours
  count as zero output, change which forecast ranks best?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
date: 2026-10-31
size: >-
  medium. Stored: tables and figures under data/ml-experiments/00432-wind-cleaned-zero-rescore/
  only; no forecast is written. Serving path: not touched. Degradation rule: not touched. Designs:
  more than one row-set definition is defensible (which of the experiment 210 flags count as
  idle), so the plan fixes it before any score is read. Callers: this folder's script only, which
  reads stored forecasts of experiments 336, 339, 343, and 347 and the experiment 210 mask, and
  calls no reviewed module beyond `studies.*`. One Sonnet plan review, one Sonnet code review
  before the first run, one Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  The maintainer wants a wind forecast to say zero where the farm should output zero, and an
  evaluation that never asks the forecast to predict the idle noise. If the half-hours that the
  experiment 210 mask flags as idle are scored against a target of 0 (the cleaned-zero row set),
  then a forecast that puts its lower quantiles at zero below the turbines' cut-in speed is no
  longer penalised for the farms' idle floor, so experiment 336's zeroing arm x1, which lost on
  the raw readings, wins or ties on the cleaned-zero row set.
prediction: >-
  Written before any cleaned-zero score is read. P1: on the cleaned-zero row set, arm x1 of
  experiment 336 minus Wind farms research default 1 is negative (x1 wins) pooled over the three
  farms at true days 1 to 10, by at least 0.05 points of capacity, with a month-block 95% interval
  that excludes zero; on the raw row set the difference stays at about +0.00 to +0.02 as stored.
  P2: the ranking of the seven forecasts is the same on the masked and the cleaned-zero row sets
  (a prediction that could well fail, see below). P3: the experiment 210 mask beats a calendar
  placebo (G3): the drop in the default's score from the raw to the masked row set is larger than
  the 95th percentile of 200 random drops with the same farm, month, and hour profile. A second
  placebo that also matches the wind-speed bin is expected to reproduce most of the effect,
  because the mask is an ERA5 wind rule; it shows how much of the effect is the wind itself. P4:
  on the conditioned low-wind slice (a stress test, because it keeps only half-hours where the
  farm generated) arm x1 loses to the default, and on the outcome-free slice (ERA5 speed from
  cut-in to 1.5 times cut-in, any power) no forecast differs from the default by more than 0.05
  points. P2 is expected to fail for the arms with mass at zero (arm x1 and the head arms with the
  idle-floor rule): they should move up the ranking on the cleaned-zero row sets.
decision_rule: >-
  Definitions fixed before any score is read. Row sets: raw (all half-hours); masked (every
  half-hour in the experiment 210 mask dropped); cleaned zero (a 210 `low_wind_idle` half-hour
  whose observed power is at or below the farm's idle-band 95% value gets the target 0, any other
  `low_wind_idle` half-hour keeps its reading because the farm generated, and 210
  `outage_suspected` half-hours are dropped, because a forecast cannot know of an unannounced
  outage and scoring an outage as zero would reward random outage guesses); cleaned zero ERA5, a
  sensitivity row set in which every `low_wind_idle` half-hour gets the target 0. The
  `low_wind_idle` flag is an ERA5 wind rule alone (wind below the fitted threshold and not in an
  outage run), so the plan reports, per farm, how many flagged half-hours had power above the idle
  band. Score: experiment 33's quantile score (twice the mass-weighted pinball loss of nine of the
  11 stored levels, as in the stored per-row scores; it approximates the continuous ranked
  probability score), in points of capacity, seed-averaged per row as the stored scores are, with
  pinball loss at q05, q50, q95, and q99 beside it. Bands: true days 1 to 10 (24 to 240 hours) and
  12 to 36 hours. Intervals: 95% month-block bootstrap (12 test months, 2,000 draws) of paired
  differences against the default, for each farm and pooled. The experiment 336 check: x1 minus
  default on the cleaned-zero row set; "x1 still loses" if the interval is above zero, "x1 wins"
  if below, and "cannot tell" if it contains zero. A win is expected by construction, because x1
  forecasts exactly zero below cut-in and the cleaned-zero target is zero on 99.9% of those
  half-hours, so a win is not support for hypothesis H-P (a wind network with a physics layer is
  more robust to the idle noise); only "still loses" weakens it. The cost of zeroing wrongly is
  read on the outcome-free slice and on the `low_wind_idle` half-hours where the farm generated,
  and the overlap between x1's zeroed half-hours and the mask is reported. No margin applies,
  because this is a re-score; the 0.05-point size in P1 is the smallest difference the plan calls
  a clear win. The cleaned-zero row set is used for evaluation only if the experiment 210 mask
  beats the calendar placebo (P3); otherwise the mask removes hard weather and experiment 429
  keeps the raw and masked row sets. Experiment 429's final quantiles do not exist yet, so its
  question is answered conditionally: it may adopt the cleaned-zero row set as it stands if P3
  holds and its arms keep the same order on the masked and cleaned-zero row sets (G5), checked
  when its quantiles exist. The experiment 210 threshold was fitted on all years, including the
  scored year; that is acceptable for a mask and is stated. Residual idle noise at high wind
  outside runs of 12 hours or more stays in every row set. No gain threshold applies, because this
  is a re-score and changes nothing in production.
benchmark: >-
  Wind farms research default 1 (the stored `default` arm of experiments 336, 339, and 343, which
  are the same forecast).
row_set_id: >-
  The 107,121 half-hours of W1 to W3 scored by experiments 336, 339, 343, and 347 (53 Tuesday runs
  of the scored year 2025-07-01 to 2026-06-30, leads 0.5 hours to 15 days); the same keys in every
  arm, checked before scoring. Not comparable with the leaderboard.
metric: >-
  Quantile score (points of capacity) and pinball loss at q05, q50, q95, and q99 on three row
  sets, per farm and pooled, at true days 1 to 10 and at 12 to 36 hours.
code: ml-experiments/00432-wind-cleaned-zero-rescore/
take_home_message: >-
  Planned; the take-home message is written after the scores are read.
---

**Status: planned. No cleaned-zero score has been read.** Nothing is fitted. The script reads the
stored quantile forecasts of experiments 336, 339, 343, and 347 and the experiment 210 mask, and
scores each forecast on three row sets.

## Plan

- **Forecasts re-scored (seven, all on the same 107,121 rows):**
    - the default: Wind farms research default 1 (`default.npy` of experiments 336, 339, and 343,
      the same forecast);
    - experiment 336, arm x1: the default with all 11 quantiles set to zero below cut-in;
    - experiment 339, arms x1, x2, and x3: the mixture head, the head plus the idle-floor rule, and
      the head trained on all half-hours;
    - experiment 343, arm M: the XGBoost model trained on the ensemble mean;
    - experiment 347, arm y2: the physics plant plus the mixture head.
- **Missing:** experiment 429's forecasts are not yet stored as final quantiles (only `parts/` and
  `models/` exist), so they are not scored here. The script takes a forecast table as input, so the
  same code scores them once they exist.
- **Idle and outage in the 210 mask:** `low_wind_idle` rows are the idle half-hours (class I1 of the
  design note), because the flag is set only where ERA5 100 m wind is below cut-in and power lies in
  the idle band. `outage_suspected` rows (at least 12 hours in the idle band at ERA5 wind above 1.5
  times cut-in) are outages (class I2) and are dropped from the cleaned-zero row set.
- **Guards:** G1, the mask file is hashed and its hash is printed before any score. G2, dropped
  share by month, 1 m/s wind bin, hour of day, and weekday, as a figure of shares. G3, the placebos:
  random drops with the mask's count in each (farm, month, hour) stratum, and in each (farm, month,
  hour, wind bin) stratum, 200 draws each. G4,
  the low-wind slices: ERA5 speed between cut-in and 1.5 times cut-in, with observed power above
  the farm's idle-band 95% value (chosen on the outcome, as the brief states, so a stress test and
  not a pass/fail gate) and without that condition. G5, the
  rank agreement of the arms on the masked and the cleaned-zero row sets.
- **Zero-mass report:** for each arm, the share of half-hours whose q05 and whose q50 sit at or below
  0.5% of capacity, on idle-flagged and on kept half-hours. A forecast whose 11 quantiles are all 0
  scores 0 on a cleaned-zero row, because pinball loss against a zero actual is (1 − τ) times the
  quantile; against a raw idle floor of 0.3 MVA the same forecast scores τ times 0.3.
- **Standing practices:** no model is fitted, so monthly refits, saved boosters, and the GPU do not
  apply. Cleaned data and the D15 and D18 cut are Primary-substation rules and do not apply to wind
  farms. Tail pinball loss and coverage at q05, q95, and q99 are reported. Avoidable-curtailment
  cost is not scored, because no load-shift events are involved. No farm time series is plotted,
  because the brief allows normalised, aggregated charts only, so the example-forecast and
  event-plot practices are not followed.
- **Earlier advice followed:** reuse stored predictions rather than refit (experiment 336 and 339
  rows); score on the same rows in every arm; say plainly that 210's mask uses ERA5 and observed
  power, so it serves evaluation and training only.

## Results

To be written after the scores are read.
