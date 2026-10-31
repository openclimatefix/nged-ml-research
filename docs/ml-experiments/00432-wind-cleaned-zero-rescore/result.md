---
id: "00432"
slug: wind-cleaned-zero-rescore
title: >-
  Does scoring the stored wind forecasts against a cleaned-zero target, where the idle half-hours
  count as zero output, change which forecast ranks best?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: finished
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
verdict: partly supported
best_arm: >-
  none. On the cleaned-zero row set at true days 1 to 10 the best forecast is the plant plus
  mixture head (10.265 points, 0.018 below the default, 95% interval -0.262 to +0.219); no arm
  differs from the default by an amount the data can tell from zero.
headline: >-
  Scoring the idle half-hours as zero output changes no forecast by more than about 0.03 points
  against the default, and zeroing the forecast below cut-in (experiment 336, x1) ties the default
  on cleaned zeros (-0.002 points, interval -0.008 to +0.004) after losing 0.015 on raw readings.
  The mixture-head arms lose almost all of their 0.22 to 0.23 point lead on the masked rows once
  the idle half-hours return with a target of zero.
lessons: >-
  The default already puts its 0.05 quantile at or near zero on 97% of the idle half-hours, so a
  target of zero leaves nothing for zeroing to fix. The mixture heads put theirs there on 38%, so
  they lose on the idle rows and keep a gain only on rows the mask drops. The experiment 210 mask
  is an ERA5 wind rule, and dropping its rows raises the default's score by 0.52 points because
  the dropped half-hours are easy.
take_home_message: >-
  Scoring idle half-hours as zero output does not change which wind forecast is best, and it does
  not help the idea of forcing the forecast to zero below cut-in: that arm ties the default
  (-0.002 points, interval -0.008 to +0.004), because the default already forecasts near zero
  there. It does remove most of the mixture-head arms' lead (about 0.23 points on the masked rows,
  0.02 on cleaned-zero rows), because those heads rarely forecast zero on idle half-hours. Cleaned
  zeros are a reasonable headline for experiment 429, with raw scores beside them.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Report wind scores on the raw and cleaned-zero row sets together and treat the masked row
    set as a diagnostic, because dropping the mask's rows raises every arm's score by about
    0.5 points and removes the idle half-hours where the mixture heads lose (headline figure).
  - >-
    Do not add a below-cut-in zeroing rule to a wind forecast that already clips its low
    quantiles: against cleaned zeros it ties the default (x1 minus default -0.002 points)
    (headline figure).
  - >-
    Give any wind head a probability mass at zero before scoring it on cleaned zeros, because
    the current mixture heads put their 0.05 quantile at or near zero on 38% of idle
    half-hours against the default's 97% (zero-mass bullets).
  - >-
    Compare any new wind mask with a placebo that matches the wind-speed bin as well as the
    calendar, because a placebo matching only farm, month, and hour is beaten by any ERA5 wind
    rule (placebo figure).
---

**Take-home: scoring the idle half-hours as zero output does not change which wind forecast is best,
and it does not reward forcing the forecast to zero below cut-in.** Zeroing the forecast below
cut-in (experiment 336, arm x1) lost 0.015 points against raw readings (interval +0.002 to +0.037).
Against the cleaned-zero target it ties Wind farms research default 1: x1 minus default is -0.002
points (interval -0.008 to +0.004), because the default already puts its 0.05 quantile at or near
zero on 97% of the idle half-hours. The mixture-head arms lead the default by about 0.23 points on
the masked rows, and by 0.02 on cleaned-zero rows with intervals that include zero, because the heads
rarely forecast zero when the farm is idle. The verdict is partly supported: the tie that the
hypothesis allowed holds, and the win of at least 0.05 points that the prediction asked for does not.

The question was whether the maintainer's rule, that the forecast says zero where the farm should
output zero and the evaluation never asks for the idle noise, changes the ranking of the stored
forecasts. We re-scored seven stored forecasts on three row sets of the same 107,121 half-hours. The
answer is that the ranking hardly moves, and the idle half-hours explain why the heads' lead shrinks.
It narrows the search: a rule that zeroes the forecast is not worth porting, and a head that cannot
put mass at zero is the weakness to fix. Hypothesis H-P, that a wind network with a physics layer
is more robust to idle noise, is not weakened by this check, and the check also cannot support it.

![Each forecast's quantile score minus the default's on three row sets](headline.svg)

![Share of half-hours the mask drops by wind speed, hour, weekday, and month](dropped_share.svg)

![The mask's effect on the default's score against random drops](placebo.svg)

## Results

- **Row sets (true days 1 to 10, 3 farms pooled).** Raw: 66,269 half-hours. Masked: 59,861. Cleaned
  zero: 65,740, which is raw minus the 529 half-hours inside `outage_suspected` runs.
  Of the 9,787 `low_wind_idle` half-hours in the scored year, 914 (9.3%) had observed power above the
  farm's idle band (172, 303, and 439 at the three farms), so they keep their reading in the
  cleaned-zero set. The ERA5-only variant sets all 9,787 to zero.
- **Absolute scores, default, points of capacity.** Raw 10.237, masked 10.752, cleaned zero 10.282,
  cleaned zero ERA5 10.294. Dropping the mask's rows raises the default's score by 0.515 points
  because the dropped half-hours (idle at low wind) are easy.
- **Arm scores on cleaned zeros (minus default, 95% interval from 12 month blocks).**
    - Plant plus head (347, y2): 10.265 (-0.018, -0.262 to +0.219).
    - Head (339, x1): 10.265 (-0.017, -0.296 to +0.230). Head plus rule (x2): 10.266 (-0.016).
    - Zero below cut-in (336, x1): 10.280 (-0.002, -0.008 to +0.004).
    - Head on all half-hours (339, x3): 10.310 (+0.028, -0.163 to +0.242).
    - XGBoost on ensemble mean (343, M): 10.363 (+0.080, -0.148 to +0.322).
- **Same arms on the masked rows:** y2 -0.234, head x1 -0.229, head x2 -0.224, x1 +0.006 (0.000 to
  0.015), head x3 +0.012, M +0.081. On raw rows x1 is +0.015 (+0.002 to +0.037).
- **Experiment 336 check.** x1 minus default is -0.002 on cleaned zeros and -0.008 (-0.016 to
  0.000) on the ERA5-only variant. Per farm on cleaned zeros: W1 -0.003, W2 -0.009 (-0.017 to
  -0.001), W3 +0.006 (-0.005 to +0.019). The verdict of the pre-registered rule is "cannot tell",
  so x1 does not "still lose" and H-P is not weakened.
- **Zero mass.** The share of half-hours whose 0.05 quantile is at or below 0.5% of capacity, on
  idle-flagged half-hours: default 0.97, x1 0.97, M 0.95, head x3 0.77, head x1 0.38, head x2
  0.35, y2 0.38. The head arms' actual falls below their q05 on about 10% of cleaned-zero rows
  against 2.1% for the default. The 0.5 quantile is at or near zero on at most 7% of idle-flagged
  half-hours for every arm, so no forecast is a point forecast of zero there.
- **Ranking (G5).** Masked order: y2, head x1, head x2, default, x1, head x3, M. Cleaned-zero order:
  y2, head x1, head x2, x1, default, head x3, M. One adjacent pair swaps (default and x1, a gap of
  0.002 to 0.006 points), so the ranking is the same within noise, and P2 holds only in that sense.
- **Placebo (G3).** The mask's effect on the default (kept minus all) is +0.515. Random drops
  matching farm, month, and hour give +0.12 (5th to 95th percentile 0.10 to 0.14), so the mask is far
  outside them. Random drops that also match the wind-speed bin give +0.54 (0.53 to 0.54), which is
  at least as large as the mask's. The mask therefore removes easy half-hours and is a wind rule; it
  does not remove hard weather. The sign matters: P3 as written spoke of a drop in the score, but
  removing easy half-hours raises it.
- **Low-wind slices (G4), minus default.** Conditioned on power above the idle band (810
  half-hours, a stress test): x1 +0.264 (0.000 to +0.568), head x2 +0.412 (+0.023 to +0.723), head
  x3 -0.226 (-0.465 to +0.098). Outcome-free (4,879 half-hours): x1 +0.085 (+0.004 to +0.193), heads
  x1, x2, and y2 about +1.3 (every interval above +1.0), head x3 -0.080 (-0.296 to +0.157), M +0.110.
- **Tails, days 1 to 10 pooled on cleaned zeros.** Share of actuals above q99: default 1.5%, x1
  1.8%, head x1 2.0%, head x2 2.3%, M 1.7%; above q95: 6.1%, 6.3%, 6.2%, 6.4%, 5.6%. Every arm has
  more than 1% above q99 and none is calibrated in the upper tail. Pinball at q99: default 0.595, x1
  0.616, head x1 0.625.
- **12 to 36 hours (cleaned zero).** Head x3 -0.233, head x2 -0.218, head x1 -0.216, y2 -0.167, M
  -0.064, x1 -0.005; every interval except x1's includes zero.
- **Pre-registered predictions.** P1 (x1 wins by 0.05 or more): not met, x1 ties. P2 (same ranking):
  holds up to one noise-level swap. P3 (mask beats the calendar placebo): holds in magnitude, but the
  sign was misworded and the wind-matched placebo equals it. P4: the conditioned slice shows x1
  losing, as written, and the outcome-free part fails because the head arms are 1.3 points worse.
- **Experiment 429.** Its final quantiles are not stored, so it is not scored. It can adopt the
  cleaned-zero row set as a headline beside the raw set: the set changes no ranking here, the mask is
  a wind rule rather than a source of hidden bias, and the check to repeat once its quantiles exist
  is the rank agreement of its arms on the masked and cleaned-zero sets.

## Checks

- **Reproduction gate:** every stored per-row score reproduced to 1e-6 (the default, x1 of 336, x1,
  x2, and x3 of 339, y2 of 347, and M against 343's stored mean). The experiment 347 arm uses its
  five stored seeds, not the five-seed mean of quantiles in `predictions_y2.parquet`.
- **Mask hash (G1):** sha256 `7ab3fe48...5368e4`, printed before scoring.
- **Median minus mean forecast per site and the mean-forecast error** are not recomputed: the
  stored forecasts are re-scored on a different target only, and the arms' mean-forecast errors are
  in the earlier experiments' pages.
- **Audit triggers:** none fires that this re-score could cause. The cleaned-zero target uses
  observed power in the row-set definition (a label, as the design note allows), which is why the
  ERA5-only variant is reported beside it; the two agree to 0.012 points for every arm.
- **Not applicable:** monthly refits, saved models, the GPU, avoidable-curtailment cost, and example
  forecast plots (no farm time series is drawn, by the brief).

## Caveats

- **Three farms, two weather cells, one scored year, 12 month blocks.** Intervals are wide: only
  the x1 contrast, which differs from the default on few rows, is tight.
- **The experiment 210 mask was fitted on all years, including the scored year.** That suits a mask
  for scoring and would be leakage for a feature.
- **Idle noise at high wind outside runs of 12 hours or more stays in every row set.**
- **The scored lead range is 0.5 hours to 15 days; the headline band is days 1 to 10.** The
  head arms' gain at 12 to 36 hours is larger than at days 1 to 10.
- **The conditioned slice selects on the outcome.** It is a stress test of zeroing, not a gate.

## What to try next

- Give the mixture head a point mass at zero (experiment 429's censored likelihood) and re-score on
  cleaned zeros: the head arms lose their lead here only because they cannot forecast zero.
- Replace the ERA5 wind rule with a detector of idle readings that does not depend on wind (the
  design note's texture rules) and repeat the placebo with the wind-bin match.

## Plan (as committed before any score)

Nothing is fitted. The script reads the
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
