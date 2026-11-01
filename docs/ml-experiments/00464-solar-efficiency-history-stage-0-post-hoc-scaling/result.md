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
status: finished
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
verdict: not supported
best_arm: >-
  `a2_member|c|med14|norm|lag2` (14-day trailing median of the forward-only ratio, normalised, on
  the stored `a2_member` forecast): the only change below zero is Solar C at -0.46% (99.95% interval
  -4.26% to +1.64%); at Solar A, B, and D it is +0.34%, +2.04%, and +0.63%. Unscaled `a2_member`:
  6.68, 6.62, and 7.53 points of capacity at Solar A, B, and D, 6.19, 7.49, and 6.35 at Solar C, E,
  and F (days 1 to 10, three seeds).
headline: >-
  Multiplying the stored solar forecasts by each farm's recent efficiency did not lower the score at
  any farm: the best of 18 methods moved Solar C by -0.46% and made Solar A, B, and D worse.
lessons: >-
  The slow efficiency changes visible in the ratio are not what limits the solar forecast. The
  remaining error is weather error, so the later stages of the line (ratio as a network input, an
  embedding, a state inside the network) have no stage-0 evidence behind them.
take_home_message: >-
  Scaling the stored solar forecasts by each farm's recent efficiency (actual power over the power
  the observed weather should have produced) did not lower the quantile score at any of the six
  farms, including the three whose output level changed between years. The best of 18 methods moved
  one farm by -0.46% and made the others worse, so slow efficiency change is not the error that
  limits the solar forecast. The later stages, which feed the ratio to the network, have no evidence
  behind them and stop.
advice:
  - >-
    Do not run stages 1 to 4 of the solar efficiency-history line, which put the trailing efficiency
    ratio into the network, because no stage-0 scaling method lowered any farm's score (figure 1).
  - >-
    Look for the solar forecast's remaining error in the weather forecast, not in farm efficiency;
    the monthly forecast bias is uncorrelated with the trailing efficiency (figure 3, +0.04).
  - >-
    Treat the 12-month block bootstrap of one farm's score as able to resolve only about 2% to 8% at
    a Bonferroni level of 108 contrasts; plan a smaller family or more months before pre-registering
    a 1% margin per farm (the table of half-widths).
production_status: not-for-production
adopted_in: none
---

**Scaling the stored solar forecasts by each farm's recent efficiency did not lower the score at any
farm, so slow efficiency change is not what limits the forecast.** The best of 18 scaling methods
lowered Solar C's mean quantile score by 0.46% and raised Solar A, B, and D by 0.3% to 2.0%. The
efficiency itself does vary: the 56-day trailing ratio at Solar A fell to about 0.8 of its earlier
average for several months and then recovered (figure 2). Those swings do not line up with the
forecast's monthly bias (correlation +0.04, figure 3). What this narrows: the remaining error is
mostly weather error, and a long history of the ratio is unlikely to add to it.

**Question and method.** National Grid Electricity Distribution (NGED) meters six solar farms,
labelled Solar A to Solar F here. Efficiency is the farm's actual power divided by the power the
observed weather should have produced. Each day gets one ratio, and a trailing median, an
exponentially weighted level, or a Kalman filter turns the days into one multiplier at the forecast
origin. The experiment multiplied the quantiles of
[experiment 430's](../00430-unified-forecaster-solar-and-pretraining-ablations/result.md) stored
`a2_member` forecast by that multiplier (no networks refitted) and compared the quantile score with
the unscaled forecast over the 12 scored months. Three sources of expected power were used: (a) a
weather model that saw the scored quarter, (b) a clear-sky proxy, and (c) a weather model trained on
earlier quarters only.

![Change in score of every primary method, per farm](figure_1_contrasts.svg)

![Daily efficiency ratio and its 56-day trailing median](figure_2_ratio_over_years.svg)

![Trailing efficiency against the monthly forecast bias](figure_3_efficiency_vs_bias.svg)

![Example forecasts, scaled and unscaled](figure_4_examples.svg)

## Results

- **Verdict: not supported.** A gain means an interval below zero and a point estimate below -1%.
  The prediction (a gain of more than 1% at Solar A, B, and D, and under 1% elsewhere) failed. By
  the stop rule the line stops: no primary arm gains at any farm.
- **Primary arms (a2_member, sources b and c, 2-day lag, normalised).** Lowest change in the mean
  quantile score per farm over days 1 to 10: Solar A +0.34%, Solar B +0.54%, Solar C -0.46%, Solar D
  +0.08%, Solar E -0.01%, and Solar F -0.03%. The lowest group means are {Solar A, B, D} +0.85%
  (`c|ew14`) and {Solar C, E, F} +0.28% (`c|med112`). Source (b), the clear-sky proxy, is worse at
  every farm (+2.6% to +24%).
- **Why the intervals are wide.** Two consecutive scored months at Solar A change by -18.5% and
  +18.8% when scaled, and the 12 months are resampled as blocks, so Solar A's interval swings by
  about 9 points. Solar C has one month of -8.6%. These are month-level outliers of the multiplier,
  not a code fault.
- **Minimum detectable effect.** The Bonferroni intervals (108 contrasts) are wide: half-widths of
  8.4%, 2.8%, 3.0%, 1.9%, 2.4%, and 1.4% for Solar A to F on `c|med14`. Only Solar F and Solar D can
  rule out a 1% gain; at Solar A, B, C, and E a gain of 1% is inside the interval, so the verdict
  rests on every point estimate being at or above zero, not on intervals.
- **Secondary arms.** The 6-day lag, source (a), the raw multiplier, and the `g0_onepass` network
  give the same picture: the lowest changes are -0.05%, -0.18%, -0.33%, and -0.73% (Solar C on
  `g0_onepass|c|med14`).
- **Day 1 (12 to 36 hours).** The lowest point estimates are at Solar A (-4.8% for `c|med14`), with
  an interval of -25% to +3%. Not a primary contrast and not a finding; it is the one place a later
  stage could look.
- **Tails.** On `a2_member|c|med14`, the 0.99 pinball loss rises at every farm (+0.002 to +0.127
  points; +0.127 at Solar A, +0.035 at Solar C, +0.057 at Solar D, +0.054 at Solar E, all above the
  0.02 limit), and the 0.99 coverage falls from 97.5% to 95.7% at Solar A and from 99.3% to 97.5% at
  Solar C.

## Checks

- **Missing multipliers.** Source (c) is missing at 0% to 3% of origins at Solar A to D and F and
  20.8% at Solar E. Source (b) is missing at 25% to 27% of origins at Solar A to D and F and 41% at
  Solar E, which biases it toward no effect but cannot explain +2.6% to +24%.
- **Clipping.** 1% to 11% of scaled 0.99 quantiles hit the cap (the largest q99 of that farm and
  refit).
- **Mean-forecast error beside the quantile score** (`a2_member|c|med14`, mean absolute error in
  points of capacity, unscaled to scaled): Solar A 12.61 to 12.83, B 13.22 to 13.22, C 12.17 to
  12.25, D 14.78 to 14.71, E 14.44 to 14.46, F 12.64 to 12.69.
- **Seeds.** Three seeds, averaged quantile by quantile, in every arm.
- **Cleaning.** Stuck runs, daylight zeros, spikes above 1.5 of capacity, and curtailment-flat days
  are dropped from the ratio; days whose ratio was exactly 0 are treated as missing.

## Caveats

- Source (a) saw the scored quarter's power, so it is not causal; sources (b) and (c) are.
- Source (c)'s regressor for a farm's first four year-quarters is leave-one-quarter-out within that
  year.
- Source (b) takes its ratio from near-clear half-hours and applies it to all-sky forecasts.
- The multiplier assumes observed weather and power reach the service two days after the day ends;
  ERA5 arrives about 5 days late, and the 6-day lag arms are within 0.05% of the best.
- The stored forecasts cover only the scored year, so the efficiency-versus-bias check has 70 farm
  months.

## What to try next

- Look for the forecast's remaining error in the weather forecast itself, for example at the
  irradiance forecast's spread and bias by sky type.
- If stage 1 is ever run, restrict it to Solar A on the day-ahead band, the only place a point
  estimate (-4.8%) was visibly below zero.

## Process

- Plan review: a fresh Sonnet agent found 13 points; all accepted or folded in (forward-only source
  (c), per-farm primary contrasts, Bonferroni 108, 6-day lag arms, defined Kalman multiplier).
- Code review: a fresh Sonnet agent found two critical bugs (zero daily ratios made the exponential
  and Kalman methods NaN at four farms; a duplicate `band` column in the aggregation) and five
  lower-severity points; all fixed before the scoring run.
- Results review: a fresh Sonnet agent re-derived the headline numbers from the raw forecasts with
  its own code. It averaged six seeds where this page averages three, so per-farm changes differ by
  up to 0.5 points (Solar A +0.9% against +0.3%) and the sign of Solar C (-0.6%) and Solar E (+0.9%)
  agree with the page. It found no lookahead and no clipped multiplier, and asked that "no arm
  gains" read "no arm gains significantly", which the verdict line now states.
