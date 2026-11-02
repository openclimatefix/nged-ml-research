---
id: "00504"
slug: neural-demand-leak-free-recent-cams-irradiance-inputs
title: >-
  Do satellite irradiance values that a live service could have at forecast time lower the pooled
  demand neural network's error at the eight Primaries whose meters report apparent power only?
theme: demand-inputs
also_relevant_to:
  - solar-disaggregation
  - weather-products-and-blending
status: done
kind: other
protocol: demand-v2
size: >-
  medium. Stored: fine-tuned neural networks and forecasts under this experiment's private data
  folder only; experiments 425, 449, 470, 480, and 494 are read and never written. Serving path and
  degradation rule: not touched (this repository holds neither). Designs: more than one would test
  the hypothesis (which days feed the columns, and whether to re-pretrain). The plan fixes
  whole-day columns, a 36-hour availability rule, and fine-tuning from the stored pretrained
  weights before any score is read. Callers: the fit script imports experiments 397 and 425 and
  patches their input builders. The scoring script imports experiments 470 and 494.
hypothesis: >-
  Giving the pooled demand neural network (experiment 425's arm A, which already reads the ensemble
  forecast of shortwave radiation) three extra inputs built only from Copernicus Atmosphere
  Monitoring Service (CAMS) satellite irradiance that a live service would have at the forecast
  origin lowers its continuous ranked probability score (CRPS) at days 1 to 10 at the eight
  Primaries that report only apparent power in megavolt-amperes (MVA): D2, D3, D4, D8, D12, D19,
  D21, and D22. The reason is that recent observed sunshine says how much embedded solar generation
  hides in those readings, which the ensemble forecast does not.
prediction: >-
  Written before any score is read. P1. A-cams lowers pooled CRPS at days 1 to 10 at the eight by
  0.05 to 0.15 points of effective capacity against K3 (the quantile average of the stored arm A
  seeds 0 to 2). The XGBoost model of experiment 480 gained 0.355 points from realised CAMS
  columns, but it had no irradiance input of its own, so a much smaller neural gain is expected.
  P2. The gain is largest at D12, on sunny half-hours (CAMS clear-sky index at least 0.7, 10:00 to
  15:00), and at days 2 to 5, where three-day-old observed cloudiness still carries information and
  the ensemble shortwave is no longer near-perfect (a hypothesis, not a measurement). P3. The
  realised-CAMS arm (A-cams-real, an upper bound) gains more than A-cams. P4. The one-number
  cloudiness arm (A-cams-lag-only) gains between 0 and A-cams. P5. At the signed-MW six and on cold
  winter nights, A-cams loses no more than 0.02 points of CRPS, and its q99 pinball loss is within
  5% of K3's at both groups.
decision_rule: >-
  1. Cadence: monthly refits at five folds (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), seeds 0
  to 2, fine-tuned from experiment 425's stored pretrained weights of the same month and seed.
  2. A-cams and K3 are each pooled by the quantile average of their seeds 0 to 2 at each fold.
  3. The gate is A-cams minus K3, CRPS at days 1 to 10 at the eight, with a Bonferroni interval
  from an origin-day block bootstrap within months (20,000 resamples would leave about 40 draws in
  each tail at a count of 12, so the scorer is run with 100,000). The bootstrap departs from the
  brief's month-block bootstrap, which has only five blocks, and matches experiment 494.
  4. The Bonferroni count is 12 (3 arms x 2 groups x 2 metrics, CRPS at days 1 to 10 and q99
  pinball loss), as in experiment 494; the brief's count of arms x groups x lead bands becomes
  arms x groups x metrics because days 1 to 10 is the only lead band counted. All three arms get
  Bonferroni intervals, and only A-cams is gated.
  5. The minimum detectable effect (MDE, experiment 428's definition with k = 3 seeds and five
  folds, from the spread of the single-seed scores of the stored arm A) is printed before any
  contrast; experiment 449 gives about 0.08 at the eight.
  6. The leak-free columns pass when the gate interval lies wholly below zero and the point
  estimate is at least MDE below zero. The threshold is MDE, not zero, because the change adds a
  CAMS feed and a daily lookup to serving, which the live service does not have today.
  7. A gain whose interval lies wholly below zero but whose point estimate is smaller than the MDE
  is reported as a small gain.
  8. If the interval includes zero and the interval's lower end is above -0.053 points (15% of
  experiment 480's 0.355), the page reports no detectable gain at this design, with fine-tuning
  only, so the neural network may already carry most of the irradiance information the XGBoost model
  lacked. If the interval includes zero and its lower end is at or below -0.053, the result is
  inconclusive.
  9. A "no loss" claim is an equivalence claim, made for the signed-MW six (CRPS and q99 pinball
  loss) and the eight (q99 pinball loss) only when the upper Bonferroni bound is at most 0.02
  points. Where the printed MDE of the group exceeds 0.02, as the queue-order check expects at the
  six, the claim is reported as inconclusive without being tested.
  10. Descriptive and not gated: A-cams-real and A-cams-lag-only, D12 alone and the eight without
  D12, lead bands, sunny-midday rows, cold winter nights, the 12 to 36 hour band, tails,
  avoidable-curtailment cost at f = 0.9, 0.75, and 0.5, a prediction-only check with the new
  columns set to their training mean, and a prediction-only check with the columns taken from
  one more day earlier.
  11. Audit trigger 4 of the `ml-experiment` skill: if the eight without D12 flips the gate
  verdict, the audit runs before merge.
  12. Every score is selection-biased, because the five folds that choose the candidate also score
  it. Rows with a valid time on or after 2026-07-01 are not scored; July 2026 stays unused, and any
  later July scoring starts 2026-07-02.
  13. The experiment does not touch the sign problem of the MVA-only Primaries and leaves the
  decision on the neural MVA experiment to the maintainer.
benchmark: >-
  K3, the quantile average of arm A seeds 0 to 2 of experiment 425 (stored, same pretrained weights,
  same rows). Also shown: the stored five-seed pool and seed 0 of arm A, `manual_heuristic`, and the
  climatology.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at days 1 to 10; q01, q05, q95, and q99 coverage and pinball
  loss; avoidable-curtailment cost at the signed-MW six.
verdict: not supported
best_arm: >-
  None. A-cams (leak-free columns) minus K3: +0.006 points of CRPS at days 1 to 10 over the eight
  (Bonferroni interval +0.001 to +0.011), so slightly worse. The report-only upper bound
  A-cams-real (realised target-time columns) minus K3: -0.318 (interval -0.393 to -0.246).
headline: >-
  Satellite irradiance that a live service could hold (whole days at least 36 hours old) did not
  lower the demand neural network's error: CRPS rose by 0.006 points at the eight Primaries that
  report apparent power only and by 0.010 at the six that report signed megawatts. The same kind of
  column read at the target half-hour itself lowered CRPS by 0.318 at the eight and 0.467 at the
  six.
lessons: >-
  In these data, recent cloudiness carries little information about a later day (the daily
  correlation falls from 0.25 at one day to 0.16 at three days), and the fine-tuned network did not
  use it. The gain of the realised columns measures how much error the ensemble's sunshine forecast
  causes, not what a satellite feed could supply. A fresher feed was not tested.
take_home_message: >-
  Satellite sunshine measurements that a live service could have at forecast time (whole days at
  least 36 hours old) did not help the demand neural network in this fine-tuning design: its error
  rose by 0.1% at the eight Primaries that report only apparent power. The same satellite columns
  read at the forecast's own half-hour would cut the error by 6.8%, so the improvement comes from
  knowing the sunshine of the target half-hour, not from older sunshine, and better sunshine
  forecasts or a nowcast are the next place to look. The experiment did not test a fresher feed (the
  previous day).
advice:
  - >-
    Do not add satellite irradiance columns that are at least 36 hours old to the demand neural
    network: whole-day columns of that age raised CRPS by 0.006 points at the eight (Results).
  - >-
    Read the gain of an input built from realised weather at the target time as the error that
    forecast error in that weather variable causes, not as the gain of a feed that serving can
    supply (A-cams-real, Results).
  - >-
    Try a nowcast-style input at short lead before a fresher satellite feed (the previous whole
    day, available at 09:00 UTC), because the daily cloudiness correlation is 0.25 at one day and
    0.16 at three days (figure 4).
  - >-
    Score cold winter nights separately whenever a weather input is added: the realised-irradiance
    columns lost 0.098 points at the eight on those nights while gaining overall (Results).
production_status: not-for-production
adopted_in: none
---

**Satellite sunshine measurements that a live service could have at forecast time did not help the
demand neural network. The same measurements read at the forecast's own half-hour would have helped
a lot.** We scored the forecast with the continuous ranked probability score (CRPS; lower is
better), in points (1 point is 1% of a substation's capacity). At the eight Primary substations
(the main distribution substations) whose meters report apparent power in megavolt-amperes (MVA)
rather than signed megawatts, the demand neural network of experiment 425 (arm A) scored 4.699. With
three columns built only from satellite data that exists at forecast time, it scored 4.705, a rise
of 0.1%. Columns that held the satellite irradiance measured afterwards for the half-hour being
forecast lowered the score by 0.318 points (6.8%). That gain measures how much error the ensemble
weather forecast of sunshine (many runs of ECMWF's weather model) causes. None of the feeds we
tested supplies it. The next place to look is a nowcast (a forecast of the next few hours made from
the latest observations) or a better sunshine forecast at short lead, because the realised arm
gained 0.134 points at day 1 and 0.407 at days 6 to 10.

**The question was whether a satellite record of recent days improves the demand neural network.**
We asked whether the neural network that forecasts demand at 16 Primary substations forecasts
better when it can see how sunny the last few days were, as measured by the Copernicus Atmosphere
Monitoring Service (CAMS) satellite retrieval. The network already reads the ensemble weather
forecast of sunshine. We built three inputs from CAMS days that were wholly over at least 36 hours
before the forecast (a delay we assumed, because the CAMS table has no availability stamp), added
them to experiment 425's stored network (its arm A), fine-tuned from the stored weights, and
compared the result with the stored network at five scored months. The idea comes from
[experiment 480](../00480-xgboost-cams-columns-at-mva-only-primaries-lab-conditions/result.md),
where two realised satellite columns lowered the error of an XGBoost model by 0.355 points at the
same eight Primaries. The score is CRPS in points of effective capacity (1 point is 1% of a
Primary's capacity). "The eight" are the eight Primaries that report only apparent power in
megavolt-amperes, and "the six" are the six that report signed megawatts.

| Arm | What it is |
|---|---|
| K3 | The stored arm A, quantile average of seeds 0 to 2 (no fit). |
| A-cams | K3's neural network plus the three columns built only from data available at forecast time (the "leak-free" columns), fine-tuned (three seeds, pooled the same way). |
| A-cams-lag-only | K3's neural network plus the cloudiness column only. |
| A-cams-real | K3's neural network plus the realised unit PV shape and clear-sky index of the target half-hour, as in experiment 480 (report only; a serving system cannot supply them). |

Verdict: not supported. The gate needed A-cams to lower CRPS at the eight by at least the minimum
detectable effect (0.077), and A-cams raised it by 0.006. The interval excludes a gain of 0.053
(15% of experiment 480's 0.355), so at this design the neural network gained no detectable
improvement from the older satellite sunshine we gave it.

## Figures

![CRPS difference to K3 at the eight and the six, with intervals and the minimum detectable
effect](figure_1_headline.svg)

![The same difference in each of the five folds](figure_2_per_fold.svg)

![Tail calibration of K3 and the three arms](figure_3_tail_calibration.svg)

![Correlation of daily cloudiness at lags of 1 to 7 days](figure_4_autocorrelation.svg)

![A day-ahead forecast of an MVA-only Primary in winter](figure_5_example_D4_day_ahead.svg)

![A 14-day forecast of an MVA-only Primary in winter](figure_5_example_D4_14_days.svg)

![A day-ahead forecast of an MVA-only Primary in summer](figure_5_example_D12_day_ahead.svg)

![A 14-day forecast of an MVA-only Primary in summer](figure_5_example_D12_14_days.svg)

## Results

- **Design.** Five monthly folds (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), seeds 0 to 2,
  experiment 470's intersection of rows (`experiment_470_intersection_v1`), widening 1.15, the D15
  and D18 start cut of 2025-03-01, flagged exact-zero half-hours dropped (1,179 scoring actuals,
  and 2,155 to 2,925 fine-tuning rows per fold), valid times from 2026-07-01 not scored. Monthly
  refits; each network starts from the stored pretrained arm A weights of the same month and seed,
  with zero weights for the new first-layer columns, then fine-tunes at a learning rate of 2e-4.
  Code: this folder, importing experiments 397, 425, 454, 470, and 494 (the scoring pattern).
  Fitted networks: the experiment's private `work/models/` folder, never in git.
- **Columns.** Origins are 09:00 UTC. For an origin on date D0 the cutoff is 21:00 UTC on D0-2, so
  the latest wholly available day is D0-3, which ends 57 hours before the origin. The three columns
  are: (a) the CAMS irradiance at the target half-hour's time of day on that day. (b) the mean of
  that irradiance over that day and the two before, and (c) the day's summed irradiance over its
  summed clear-sky irradiance. The [lag-features study](../../studies/forecasts/lag-features.md)
  found the previous whole day available at 09:00 UTC after February 2026, so a service could be one
  to two days fresher than this design.
- **Minimum detectable effect (three seeds, five folds).** 0.077 points at the eight and 0.124 at
  the six (experiment 428's definition, from the seed spread of the stored arm A). Bonferroni
  count 12 (3 arms x 2 groups x CRPS and q99 pinball loss); 100,000 origin-day block-bootstrap
  resamples.
- **CRPS, difference to K3 (points; Bonferroni interval).**

| Arm | Eight | Six (no labelled event) |
|---|---|---|
| A-cams | +0.006 [+0.001, +0.011] | +0.010 [+0.004, +0.017] |
| A-cams-lag-only | +0.004 [-0.000, +0.009] | +0.009 [+0.003, +0.015] |
| A-cams-real (report only) | -0.318 [-0.393, -0.246] | -0.467 [-0.577, -0.359] |

- **Absolute CRPS (points).** At the eight: K3 4.699, A-cams 4.705, A-cams-lag-only 4.703, and
  A-cams-real 4.381. At the six: K3 3.463, A-cams 3.474, A-cams-lag-only 3.472, and A-cams-real
  2.997.
- **Equivalence at the 0.02 limit.** The q99 pinball loss difference of A-cams is +0.001 at the
  eight (upper bound +0.002) and -0.000 at the six (upper +0.001), so A-cams is equivalent to K3 in
  the q99 tail at both groups. CRPS at the six is not tested for equivalence, because the minimum
  detectable effect there (0.124) exceeds 0.02.
- **Tails.** q99 coverage is 0.9908 (A-cams) against 0.9911 (K3) at the eight and 0.9845 against
  0.9847 at the six; q95 coverage 0.949 against 0.950 and 0.922 against 0.924. q99 pinball loss
  0.252 (A-cams) against 0.251 (K3) at the eight. A-cams-real has q99 pinball loss 0.238 at the
  eight.
- **Where the realised arm gains.** By lead at the eight: -0.134 at day 1, -0.256 at days 2 to 5,
  -0.407 at days 6 to 10. On sunny midday rows (clear-sky index at least 0.7, 10:00 to 15:00 UTC)
  -1.110 at the eight and -1.222 at the six. At D12 alone -0.635 (K3's CRPS there is 8.589), and at
  the eight without D12 -0.273. The realised arm gains nothing at D8 (+0.005) and in the 2025-12
  fold (+0.038 at the eight), and it loses on cold winter nights: +0.098 at the eight and +0.121 at
  the six.
- **Descriptive slices for A-cams (95% intervals, not corrected).** D12 alone +0.014 [+0.002,
  +0.027]; the eight without D12 +0.005 [+0.001, +0.009]; cold winter nights +0.008 at the eight
  [-0.005, +0.022] and +0.008 at the six [-0.005, +0.021]; the 12 to 36 hour band +0.004 at the
  eight [-0.000, +0.009] and +0.013 at the six [+0.006, +0.020]; sunny midday -0.004 at the eight
  [-0.013, +0.004] and -0.015 at the six [-0.029, -0.001]. Fold by fold at the eight A-cams ranges
  from -0.003 to +0.018.
- **Avoidable-curtailment cost at the six, difference to K3 (experiment 470's measure).** At f =
  0.9: A-cams +0.016 [-0.001, +0.048], A-cams-real +0.129 [-0.012, +0.353]. At f = 0.75: +0.082
  [+0.001, +0.206] and +0.809 [-0.035, +1.997]. At f = 0.5: +0.111 [-0.031, +0.327] and +0.091
  [-1.704, +2.410]. The realised columns therefore do not lower the curtailment cost at the six,
  although they lower CRPS. All the neural networks cost far less than `manual_heuristic` at f =
  0.75 and 0.5 (K3 -5.8 and -16.1).
- **D15 and D18 (report only).** CRPS 10.83 (K3), 10.86 (A-cams), 10.50 (A-cams-real).
- **Why the leak-free columns do not help (tested).** (1) The validation loss of the fine-tuned
  A-cams neural network is higher than the stored arm A's in 14 of 15 fits (mean +0.0008), so the
  fine-tune found no use for the columns. (2) The daily correlation of cloudiness between two days
  is 0.25 at one day, 0.16 at three days, and 0.10 at seven (figure 4), so a three-day-old value
  holds little information. (3) The same fine-tune does use the realised columns (validation loss
  0.24 to 0.29 against 0.33 to 0.39), so the recipe can use informative columns. (4) Setting the
  A-cams columns to their training mean at forecast time changes CRPS by -0.006 [-0.013, +0.000]
  at the eight, and moving the day one day earlier changes it by +0.000 [-0.006, +0.006].
  A null here may still mean that the columns were not learned in the fine-tune from zero weights;
  a retrained network was not tested.
- **Serving table.** A-cams needs a CAMS feed that arrives about a day late (not ingested today),
  one stored cloudiness value per Primary per day plus three half-hourly values per target, and
  leaves the number of neural networks per pass unchanged. The fallback when the feed is late (set
  the columns to their training mean and widen the bands) changes CRPS by -0.006 points [-0.013,
  +0.000] at the eight (no loss) in the prediction-only check. Because A-cams gains nothing, adding
  the feed is not worth it. A-cams-real cannot be served at any lead, because its columns describe
  the target half-hour itself.
- **Fit times.** 26 to 31 s for A-cams-lag-only and 29 to 60 s for A-cams-real per seed; 67 to 120 s
  for A-cams, which includes two extra prediction-only forecasts. 45 fine-tunes took about 1 hour of
  one GPU process.

## Checks

- **Median minus mean.** The stored forecasts hold 11 quantiles only. The mean of the quantiles
  minus the median per site differs by under 0.03 points between K3 and A-cams (D12 +2.94 for
  both), so the new columns do not shift the forecast.
- **Median error beside CRPS.** The mean absolute error of the median forecast moves with CRPS:
  K3 and A-cams agree to within 0.02 points at every Primary, and A-cams-real lowers it at 13 of 14
  Primaries (at D8 it rises by 0.08).
- **Triggers of the audit rules.** None fired. No control failed. The result contradicts the
  prediction (P1: a gain of 0.05 to 0.15), and the explanation tests above record why. The
  quantile-score and median-error contrasts agree. Leaving out D12 leaves the sign and verdict of
  the gate unchanged (+0.005 without D12). The nested arm A-cams-lag-only beats A-cams by 0.002,
  less than the half-width (0.005) of A-cams's interval. The realised arm's interval at the eight
  is 0.55 times as wide as experiment 480's.

## Caveats

- **Only a 36-hour-old feed was tested.** A fresher feed (the previous whole day) was not fitted.
  The correlation in figure 4 suggests a small gain at most (0.25 at one day).
- **Selection bias.** The five folds that choose the candidate also score it, and July 2026 stays
  unused.
- **Fine-tune only.** The new columns start at zero weight and the network was fine-tuned for about
  10 epochs, so the experiment cannot say whether a network trained with the columns from the start
  would use them.
- **The horizon is 1 to 10 days**, the range NGED care about. The realised columns are an upper
  bound, because a service never has them.

## What to try next

- **Nowcast.** Test a satellite or station nowcast of sunshine as an input at leads under 6 hours,
  where the ensemble's forecast is least accurate.
- **A fresher feed.** A feed with the previous whole day (available at 09:00 UTC) has not been
  tested; testing it would take one arm and 15 fits, about 0.2 GPU-hours.
- **Sunshine forecast error.** The realised arm's gain per lead band bounds what a better sunshine
  forecast could add (0.318 points, 6.8% of CRPS at the eight).
