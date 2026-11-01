---
id: "00468"
slug: best-models-performance-analysis-with-time-series-graphs
title: >-
  How do the best wind, solar, and demand forecasters behave hour by hour and event by event, and
  where do they fail?
theme: production-reference-and-best-guess
also_relevant_to:
  - generator-forecasting
  - lagged-power-and-error-correction
status: finished
kind: other
protocol: none
size: >-
  medium. Stored: tables and figures only, nothing fitted (private folder
  data/ml-experiments/00468-best-models-performance-analysis-with-time-series-graphs/). Several
  designs would show the same behaviour (which weeks, which events, which scenarios), so the plan
  fixes the selection rules before any chart is read. Callers: the scripts import the folders of
  experiments 404, 428, 454, 456, and 461. Production serving path and degradation rule: not
  touched. One Sonnet code review before the first scoring run, one Sonnet results review, one
  Sonnet prose review.
hypothesis: >-
  The best forecasters' failures are concentrated in a few identifiable situations (a stormy week
  for wind, a cloudy week for solar, and the first days of a load shift that the lagged power
  does not reflect), and a set of time-series graphs makes those situations visible.
prediction: >-
  Written before any chart is read. P1. Wind quantile bands cover q05 and q95 within 3 points in
  each month, and the share above q99 stays under 1.5% in every month. P2. The worst 20
  half-hours by q99 pinball loss at each farm sit in a few storm days and not in scattered
  half-hours. P3. In real load-shift events, the forecasts that use the lagged power follow the
  shifted power with a delay of at most 1 day while the lagless forecast stays near the pre-event
  level, and the lagged forecasts' errors during the event are larger than the lagless forecast's
  in at least half the labelled events.
decision_rule: >-
  This is a descriptive analysis, so no arm is adopted. A prediction counts as holding when the
  stated number holds on the rows shown. Contrasts are reported with month-block bootstrap
  intervals at a Bonferroni level set by the count of contrasts, and the minimum detectable effect
  beside each. Interim numbers are labelled interim and unaudited.
benchmark: >-
  Wind: research default 1, research default 2, and climatology. Solar: experiment 310's blend,
  the XGBoost model, climatology, and the manual heuristic. Demand: BG2 (to 168 hours only), the
  manual heuristic, climatology, and the monthly GAM as a point forecast. No wind or solar GAM,
  BG2, or manual-heuristic quantile forecast exists, so none is invented.
row_set_id: none
metric: >-
  Quantile score in points of effective capacity and in MW (private store only), coverage and
  pinball loss at q01, q05, q95, and q99 per month, error by lead time, and the avoidable-curtailment
  cost at f = 0.9, 0.75, and 0.5 for the demand scenarios.
code: ml-experiments/00468-best-models-performance-analysis-with-time-series-graphs/
verdict: partly supported
best_arm: >-
  Not an arm contest. Wind `member_plain` (p99.9 capacity) pools to 0.95%, 0.79%, and 0.90% of
  actuals above q99 at Farm A, B, and C; solar `a2_member` is level with experiment 310's blend
  (+0.23 points of capacity, interval -0.18 to +0.74).
headline: >-
  Pooled over a year the wind and solar tails are close to nominal after table P, but single
  months are not: monthly q05 shares reach 14% to 16% for wind and the share above q99 reaches 2.1%
  to 3.1%. Demand forecasts do not react to a load shift by following the shifted power, and the
  lagless network is as good as the lagged networks in 4 of 7 real events.
lessons: >-
  Report tail calibration per month and not only pooled, because the pooled share hides bad months.
  Real events differ in size by a factor of six in error, so one pooled event mean hides them.
take_home_message: >-
  The wind and solar tails look calibrated over a whole year, but individual months miss by a
  factor of two or three, so a service that relies on one month's tail should widen its bands.
  In the 7 real load-shift or level-step events we could plot, the networks that read lagged power
  were no better than the lagless network in 4, so lagged power helps in some events and hurts in
  others. The injected-shift scenarios and the wind reference contrasts are not done yet.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Check tail calibration month by month for wind and solar, not only pooled over the year
    (figures 1 and 2).
  - >-
    Cut solar reference contrasts to the stored half-hours of experiment 310 and report the
    minimum detectable effect beside each (solar contrast table).
  - >-
    Plot real demand events with the lagless forecast beside the lagged forecasts, because the
    lagless network matched them in 4 of 7 events (figures 7 to 10).
---

**Take-home.** Over a year the best wind and solar forecasts have tails close to the nominal
shares, but single months miss badly, and the demand networks that read lagged power are not
reliably better than the lagless network during real events. This page covers Part A (wind and
solar) and the real-event half of Part B. It does not cover the injected-shift scenarios, the wind
reference contrasts, the NRA-corrected actuals, or the surprise inputs, which are still to do.

We asked where the best wind, solar, and demand forecasters fail, by plotting stored forecasts
against what happened. Nothing was fitted. Wind uses experiment 461's `member_plain` (capacity at
the 99.9th percentile of earlier power) at Farm A, B, and C. Solar uses experiment 461's
`a2_member` at the old capacity rule at Solar A and B, because experiment 461 did not adopt p99.9
for Solar A. Demand uses experiment 454's stored S1, B-mix, K2-S1, and lagless forecasts. The
time-series charts of Farm A to C and Solar A and B carry megawatts of metered generators and stay
in the private store (`data/ml-experiments/00468-.../part_a/`); this page shows only charts that
carry no generator time series.

## Part A: wind and solar

![Share of actuals beyond each tail quantile, wind, by month](figure_1_wind_tail_coverage_by_month.svg)

![Share of actuals beyond each tail quantile, solar, by month](figure_2_solar_tail_coverage_by_month.svg)

![Wind error of the median forecast by lead time](figure_3_wind_error_by_lead.svg)

![Solar error by lead time with the stored references](figure_4_solar_error_by_lead_with_references.svg)

![Effect of table P, wind](figure_5_table_p_wind.svg)

![Effect of table P, solar](figure_6_table_p_solar.svg)

- **Pooled tails (after table P).** Share above q99: Farm A 0.95%, Farm B 0.79%, Farm C 0.90%,
  Solar A 1.69%, Solar B 1.42%. Share below q01: 0.31%, 1.41%, 0.86%, 0.65%, 1.10%.
- **Single months miss.** The worst month's share below q05 is 14.1%, 10.3%, and 16.3% at the three
  farms (nominal 5%), and the worst month's share above q99 is 2.6%, 2.1%, and 3.1% (nominal 1%).
  Solar A's worst month above q99 is 12.3%. Prediction P1 (within 3 points each month) did not
  hold; only the pooled share above q99 stayed under 1.5% at the wind farms.
- **Error by lead.** The mean absolute error of the median forecast rises from about 9% of
  capacity at day 1 to 17% to 21% at days 9 to 10 for wind, and from 9% to 10% to 15% for solar.
- **Solar contrasts** (`a2_member` minus reference, points of capacity, 98.75% month-block interval,
  Bonferroni count 4, 11 scored months, 120,249 half-hours; the minimum detectable effect is in
  brackets): blend +0.23 (-0.18 to +0.74; 0.60); XGBoost alone -0.56 (-1.05 to -0.03; 0.69);
  climatology -3.71 (-4.29 to -2.94; 0.92); manual heuristic -3.32 (-4.23 to -2.34; 1.27).
  Absolute errors: `a2_member` 12.7%, blend 12.5%, XGBoost 13.3%, climatology 16.4%, manual
  heuristic 16.1%. The `a2_member` gain over XGBoost is under the minimum detectable effect.
- **Table P.** It cuts the share above q99 at the wind farms (Farm A 2.10% to 0.95%) and leaves
  the score within 0.1% (see the table P figures).
- **Worst half-hours.** The 20 worst q99 half-hours at each wind farm come from 5 to 7 different
  months, so they are scattered, not one storm. Prediction P2 did not hold at month level; the
  day-level check was not run. The CSVs are private.
- **Not done:** wind reference contrasts (research defaults 1 and 2, climatology, manual
  heuristic), because the stored wind references load through code that needs PyTorch, which this
  CPU-only session does not install; the 14-day wind chart (experiment 429 holds one); the solar
  q01 contrast against references; a curtailment week (no label exists for wind or solar).

## Part B: demand in real events

![D5, a labelled event with the largest error](figure_event_02_D5_111746.svg)

![D5, a labelled event](figure_event_03_D5_107175.svg)

![D14, a labelled event](figure_event_06_D14_114748.svg)

![D12, an event found by the scan with no NGED label](figure_event_07_D12_109920.svg)

- **Events.** 7 events whose day-ahead rows (12 to under 36 hours ahead) fit inside one stored
  month: 6 labelled events from experiment 404 and 1 scan-found event from experiment 456 (the scan
  found 6 more that did not fit). NGED's labels are incomplete, so a scan-found event may be a load
  shift, a level change, or neither.
- **Mean absolute error of the median forecast during each event** (points of capacity: lagless,
  S1, B-mix, K2-S1): D5 3.9, 5.2, 3.7, 3.7; D5 16.0, 16.1, 15.5, 15.2; D5 5.3, 4.2, 4.8, 3.9; D14
  24.8, 26.8, 26.3, 25.8; D16 4.0, 3.4, 3.7, 3.5; D14 13.6, 10.0, 12.6, 12.0; D12 (scan) 21.2,
  14.7, 17.1, 15.8.
- **Lagged against lagless.** S1's error is above the lagless error in 3 of 7 events, and K2-S1's
  is below it in 6 of 7. Prediction P3 (lagged forecasts worse in at least half the events) did
  not hold. Two events (D5 at 16.0 and D14 at 24.8) are large for every network, so the lags do not
  explain them.
- **Not done:** the NRA-corrected actual, the surprise inputs, injected shifts, and the
  avoidable-curtailment cost. With 7 events no interval is reported; these numbers are interim and
  unaudited.

## Checks

- **Median minus mean forecast:** not computed; this page scores no mean forecast.
- **Triggers:** none fired; no control or nested arm exists. Code review (one fresh Sonnet agent)
  found no numerical bug, and its row-selection and wording findings were applied.
- **Caveats:** NGED's event labels are incomplete, so rows with no labelled event may hold events.
  The wind rows cover true days 1 to 10 only. Solar `a2_member` is at the old capacity rule.
