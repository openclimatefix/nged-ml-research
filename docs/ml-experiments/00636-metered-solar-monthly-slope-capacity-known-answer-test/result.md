---
id: "00636"
slug: metered-solar-monthly-slope-capacity-known-answer-test
title: >-
  Can a monthly regression of power on irradiance find known capacity steps at the six metered solar
  farms, including a loss of direct-current capacity hidden under inverter clipping?
theme: metered-solar-farms
also_relevant_to:
  - solar-disaggregation
status: planned
kind: other
date: 2026-11-07
size: >-
  medium. Stored: private statistics tables under data/ml-experiments/ only. Designs: more than one
  defensible estimator and injection, so the estimators, worlds, sizes, and thresholds are fixed
  below before any script ran. Callers: this folder's scripts only; it copies experiment 218's rows
  and experiment 618's envelope filter. CPU only, at most 4 cores, about 2 CPU-hours. No forecast is
  scored. Serving path and degradation rule: not touched.
hypothesis: >-
  A least-squares regression of a farm's output on a fitted irradiance expectation over all daylight
  half-hours of a month, weighted by the clear-sky index squared, finds a known capacity step of 10%
  within 3 months at 4 of the 5 testable farms (Solar A, B, C, D, and F). It finds the step whether
  the loss is in alternating-current output or in direct-current capacity under inverter clipping,
  and it raises few false steps in cloudy months and under random curtailment.
prediction: >-
  Written before any script ran. World 1 (alternating-current loss): the slope finds a 10% step
  within 3 months at least 80% of the time at Solar C and Solar F, and at most 25% of the time at
  Solar A, B, and D, as the first analysis (Opus, 2026-11-07) found. World 2 (direct-current loss
  under clipping): the slope's recall is within 10 points of world 1 at every farm, because only 1%
  to 2.5% of daylight half-hours sit at the alternating-current limit, and the sunny-row envelope
  sees under half of a 10% loss at Solar C, D, and F. World 3: the false-step rate in the three
  cloudiest months is above the rate in other months at 3 or more farms. World 4: zeroing 5% of a
  month's half-hours is detected at 1 month at Solar C and F more than 50% of the time, so the
  false-alarm rate exceeds the first analysis's. The stop rule is expected to fire, because Solar A,
  B, and D already missed in the first analysis.
decision_rule: >-
  Fixed by the brief and not tuned. A farm misses a 10% step in a world when its recall at 3 months
  is under 80% (the first analysis's pass bar). If a 10% step is missed at 3 months at 3 or more of
  the 5 farms in world 1 or in world 2, the estimator is reported as not usable at those farms, and
  the line stops with no tuning. Otherwise it passes where recall is at least 80%. The world 4
  false-alarm rate must stay at or below 5% at 1 month, the rate that the 95th-percentile threshold
  sets on the uninjected series. The minimum detectable step is the smallest injected loss with
  recall of at least 80%.
benchmark: >-
  The first analysis (Opus, 2026-11-07): slope recall of a 10% step at 3 months was 15%, 25%, 100%,
  15%, and 100% at Solar A, B, C, D, and F, and the sunny-row envelope's was 5%, 10%, 70%, 15%, and
  55%.
row_set_id: farm_rows_cams_to_2026_06_30
metric: >-
  Recall of an injected step by farm, world, size, and horizon (1, 3, 6, and 12 months), with a
  Wilson 95% interval; recovered size over injected size; false-step rate; minimum detectable step.
---

**Plan only. The results sections are written after the run.**

## Question

The first analysis found that a monthly regression of power on irradiance over all daylight
half-hours carries 3 to 11 times more information a year than experiment 618's sunny-row filter. It
tested only a loss of all output by a fixed share (a pure alternating-current loss). This experiment
adds the harder cases: a direct-current loss hidden under the inverter limit, cloudy months, and
random curtailment.

## Data and estimators

- **Rows.** Experiment 218's `farm_rows_cams.parquet` to 2026-06-30: output as a fraction of
  stored capacity, CAMS irradiance (Copernicus Atmosphere Monitoring Service), and ERA5 temperature.
  Solar E has 20 months and is left out of the injection tests. Development data only.
- **`slope`.** One robust (Huber) regression per farm of output on irradiance terms over all
  half-hours with clear-sky irradiance of at least 100 W/m2, then each month's least-squares slope of
  output on that expectation through the origin, weighted by the clear-sky index squared. Every
  injected series is refitted from scratch.
- **`envelope`.** Experiment 618's filter: sunny half-hours (clear-sky index at least 0.9,
  clear-sky irradiance at least 700 W/m2, runs of 6 or more), one median ratio to a robust
  expectation per sunny day, the month's value the mean of its days. It is the alternating-current
  limit number of the two-number estimator; `slope` is the direct-current gain number.
- **`slope_nozero` (post hoc).** `slope` with half-hours dropped when the output is zero and the
  expectation is above 0.3. It is added to see whether zero rows explain world 4, and it enters no
  decision.
- **Deseasonalising.** Each monthly value is divided by the median of the same calendar month
  across the years before the start month, so an injected step cannot move the reference. A
  calendar month with no earlier year uses all years.
- **Statistic and threshold.** For a start month and a horizon k, the mean of the k months from the
  start over the mean of the 12 months before, minus 1. The threshold per farm, estimator, and
  horizon is the 95th percentile of the absolute statistic on the uninjected series over the 20
  start months (January, April, July, and October of 2021 to 2025; the cloudy starts of world 3 are
  excluded). A step is detected when the statistic is below minus the threshold. Windows with no
  data count as missed. A window running past 2026-06-30 averages only the months that exist, in
  the thresholds and the injections alike.

## Worlds

1. **Alternating-current loss.** Multiply all output from the start month on by one minus the
   loss. Losses 3%, 5%, 7.5%, 10%, and 15%, 20 start months per farm.
2. **Direct-current loss under clipping.** The alternating-current limit is the 99.5th percentile
   of a farm's daylight output. A half-hour with output from 97% of the limit up to the limit is
   clipped. The data cannot show how much direct-current power a clipped half-hour held, so this
   stress case assumes 15% more than its output. A loss removes its share of the direct-current
   power, and the result is clipped at the limit (half-hours above the limit just lose the share),
   so a 10% loss is invisible on clipped half-hours and flat tops stay at the limit. Same losses
   and starts. Clipped half-hours are 0.4% to 1.1% of daylight half-hours, so the world differs
   little from world 1 at farms that clip rarely.
3. **Cloudy months.** The three months with the lowest mean clear-sky index relative to the median
   of the same calendar month, per farm, in 2021 to 2025. Inject losses of 5% and 10% in worlds 1
   and 2 at those starts. The false-step rate is the share of uninjected statistics below minus the
   threshold at those starts, pooled over the 5 farms (15 windows). A wider check uses the 10
   cloudiest months.
4. **Curtailment placebo.** Zero 5% of one month's daylight half-hours at random, three repeats at
   each of the 20 starts, with no capacity change. The false-alarm rate is the share of statistics at
   horizon 1 below minus the threshold.

## Reported

- **Recall and recovered size** by farm, world, loss, and horizon (1, 3, 6, and 12 months), with
  Wilson 95% intervals; recovered size is the median of the injected statistic minus the uninjected
  statistic at the same start, over the injected loss.
- **Minimum detectable step**: the smallest loss with recall of at least 80%, interpolated linearly
  between the injected losses, per farm, world, estimator, and horizon.
- **Two-number estimator**: in worlds 1 and 2, the recall and recovered size of `envelope` (the
  limit number) beside `slope` (the gain number), and the share of daylight half-hours at the limit.
- **Figures**: one panel per world (at most two panels per figure), intervals on the headline chart.

## Standing practices

Follows: cleaned data is not used because the rows are the stored farm rows; no model is fitted
beyond the regression; nothing is scored after 2026-06-30. Does not apply: refits, tails, forecast
plots. Farms appear as Solar A to F; no coordinates and no calendar dates for any farm's series; the
months of the cloudiest starts appear in the text as month numbers only if needed.
