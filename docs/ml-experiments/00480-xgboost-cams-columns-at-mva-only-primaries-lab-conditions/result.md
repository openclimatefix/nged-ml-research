---
id: '00480'
slug: xgboost-cams-columns-at-mva-only-primaries-lab-conditions
title: >-
  Do two columns from realised satellite irradiance lower the XGBoost demand forecast's error at the
  eight Primaries whose meters report apparent power only?
theme: demand-inputs
also_relevant_to:
  - solar-disaggregation
  - weather-products-and-blending
status: finished
kind: other
protocol: demand-v1
size: >-
  medium. Stored: forecasts and fitted boosters in this experiment's private data folder only;
  experiments 381, 87, and 377 are read and never written. Production serving path: not touched
  (this repository holds none). Designs: more than one would test the hypothesis (the clear-sky
  index could use the Haurwitz clear-sky model or the CAMS clear-sky field, and the unit PV shape
  could be the fleet plant or a flat cosine); the plan fixes one of each before any score is read.
  Callers: the fit script subclasses the arm of experiment 381's fit script and imports experiment
  87's folder; the scoring script imports experiment 381's scoring script and `studies.eval_protocol`.
hypothesis: >-
  Giving the monthly XGBoost demand model two extra input columns per half-hour, the unit
  photovoltaic (PV) output shape and the clear-sky index, both computed from realised Copernicus
  Atmosphere Monitoring Service (CAMS) irradiance, lowers the continuous ranked probability score
  (CRPS) at days 1 to 10 at the eight MVA-only Primaries (D2, D3, D4, D8, D12, D19, D21, D22),
  because embedded solar generation hides inside their apparent-power readings and the model has
  no input that says how much of it a half-hour had.
prediction: >-
  Written before any score is read. P1. The pooled CRPS at days 1 to 10 over the eight falls by
  0.03 to 0.10 points of effective capacity (A1 minus A0). P2. The gain is largest at D12 and D8,
  the two MVA-only Primaries where experiment 420's export gate opens, and on sunny half-hours
  (realised clear-sky index at or above 0.7), where it is 0.10 to 0.40 points. P3. The gain is
  concentrated in days 1 to 3 and smaller beyond, because the ensemble's own shortwave already
  describes the sun at long lead, and the realised CAMS columns replace a forecast with the truth
  only where the forecast is least certain. P4. At the other eight Primaries (the signed-MW six,
  D15, and D18) the columns change the score by less than 0.03 points in either direction.
decision_rule: >-
  Cadence: monthly refits (12 refits, origins 2025-07-01 to 2026-06-01), the recipe of experiment
  381's `monthly/control`, on CPU. Confirmatory contrast C1 = A1 minus A0, CRPS at days 1 to 10,
  pooled over the eight MVA-only Primaries, paired month-block bootstrap of
  `studies.eval_protocol`, 20,000 resamples, read at the 99.67% level (1 - 0.01/3: the family has
  three contrasts, C1 here and C2 and C3 of the queued signed-label arms, and C2 and C3 are not
  run). The columns are kept when the interval lies wholly below zero. Because the columns cost
  nothing to port beyond a CAMS download (so the threshold on the gain itself is zero), the port
  decision also needs no Primary worse than +0.05 points and a pooled upper-tail check (below).
  The columns use realised CAMS values, so they are available for lag times older than about 1 day
  and not for the first day of lead; a gain that exists only at days 1 to 3 is reported as a gain
  for an input that serving cannot supply, and the port claim is limited to lag-time use. No-harm
  at the other eight: A1 minus A0 pooled over the six signed-MW Primaries, and over D15 and D18,
  has an upper 99.67% bound at or below +0.05. Descriptive only: each Primary at 95%, D12 and D8
  without the rest, lead bands, sunny half-hours three ways, tails, avoidable curtailment cost.
  Not run in this stage: the known-answer test and arms A2 (see Scope).
benchmark: >-
  A0, the monthly control refitted fresh on CPU at the same Primaries (an arm with the same rows,
  recipe, and seed), and as references experiment 381's stored `monthly/control` (fitted on the
  GPU), the GAM stack of experiment 377 (stack without S below 168 h), `manual_heuristic`, and
  `climatology`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles, mean absolute error of the
  median, and twCRPS, in points of effective capacity, from `studies.eval_protocol`; leads 0.5 to
  336 hours; headline band days 1 to 10.
code: ml-experiments/00480-xgboost-cams-columns-at-mva-only-primaries-lab-conditions/
verdict: supported
best_arm: >-
  A1 (monthly XGBoost model plus the CAMS unit PV shape and clear-sky index) minus A0: -0.355
  points of CRPS at days 1 to 10 over the eight MVA-only Primaries (99.67% interval -0.497 to
  -0.228)
headline: >-
  Two columns of realised CAMS irradiance lowered the CRPS at all 16 Primaries: -0.355 over the
  eight MVA-only Primaries, -0.476 over the six signed-MW Primaries, and -0.769 at D12. The gain
  sits on sunny half-hours and grows with lead, which is what realised weather in place of a
  forecast would do.
lessons: >-
  The gain is an upper bound on what a forecast irradiance could give, because the columns hold
  realised irradiance at every lead. The columns did not lower the avoidable curtailment cost at
  the 0.9 limit at the signed-MW six, so a better mean score did not carry over to the tail cost.
take_home_message: >-
  Giving the XGBoost demand model the satellite-measured sunshine of each half-hour lowered its
  error at every one of 16 Primary substations, by 0.36 points of capacity on average at the eight
  that report apparent power and by 0.48 at the six that report real power. The gain is real only
  for inputs a live service can have: the satellite data arrives about a day late, so it can
  describe past days and not the first day ahead, and the gain is an upper bound on what the
  weather forecast could give. The next step is to measure how much of the gain a forecast
  irradiance keeps.
advice:
  - >-
    Port the CAMS unit PV shape and clear-sky index for lag-time use only (data at least a day
    old), not for the first day of lead (C1).
  - >-
    Treat the 0.355-point gain as an upper bound when planning forecast-weather experiments,
    because the columns hold realised irradiance at every lead (C1, lead bands).
  - >-
    Cap the clear-sky index at 1.5 and set it to 0 where the Haurwitz clear-sky irradiance is
    below 1 W/m2, because the ratio reaches 10.9 near sunrise and sunset (design).
  - >-
    Score a gain from irradiance columns on avoidable curtailment cost as well as CRPS, because
    the gain at the 0.9 limit was absent at the signed-MW six (curtailment table).
production_status: candidate
adopted_in: none
---

**Take-home.** Giving the monthly XGBoost demand model the satellite-measured sunshine of each
half-hour lowered its error at all 16 Primaries. The gain was 0.36 points of capacity over the eight
Primaries that report apparent power (MVA) and 0.48 over the six that report real power (MW). The
gain is an upper bound on what a live service can have: the columns hold realised irradiance at
every lead, and the satellite data arrives about a day late, so only data at least a day old can
feed the model. The next experiment should replace the realised columns with columns built from the
ensemble's forecast irradiance and measure how much of the gain remains.

## What was asked and what happened

We asked whether two extra input columns, computed from the irradiance that a satellite measured in
each half-hour, lower the error of the monthly XGBoost demand forecast. The two columns are the
output of a generic south-facing solar fleet under that irradiance, and the ratio of the irradiance
to its clear-sky value. We refitted the model monthly for 12 months with and without the columns
(arms A1 and A0), on the same rows, and compared the continuous ranked probability score (CRPS) at
days 1 to 10 ahead, in points of capacity (negative favours A1).

A1 was better at every group and every Primary, and the gain was far larger than the 0.03 to 0.10
points predicted. The gain sits on sunny half-hours and grows with lead, which is what realised
weather in place of a forecast would give. This is stage 1 of a larger brief. The known-answer test
and the sign-label arms (A2) were not started, and they do not start unless the maintainer asks,
because A1 shows the gain comes from the columns alone and needs no sign label.

## Figures

![A1 minus A0 at each group and Primary](headline.svg)

![A1 minus A0 by lead band](lead_bands.svg)

![A1 minus A0 on sunny half-hours and on other slices](slices.svg)

![Day-ahead forecasts at D12](example_day_ahead.svg)

![Full 14-day forecasts at D12](example_14_day.svg)

## Results

- **Arms.** A0 is experiment 381's monthly control (experiment 87's `n1h` recipe: 500 rounds, eta
  0.05, depth 6, 0.5-year half-life, 51 members mixed) refitted on CPU. A1 is A0 plus
  `cams_unit_pv_shape` and `cams_clear_sky_index`. Cadence: monthly refits, 12 refits from
  2025-07-01 to 2026-06-01, 384 fits in 8.6 hours on 12 threads. Code:
  `ml-experiments/00480-xgboost-cams-columns-at-mva-only-primaries-lab-conditions/`, which imports
  the folders of experiments 87, 377, and 381.
- **Weather source at lag times.** CAMS radiation service satellite retrieval at each Primary's
  coordinates, hourly. CAMS publishes with about a 1-day delay, so the columns are unavailable at
  forecast time for the first day of lead. In this experiment they hold the half-hour's own realised
  value at every lead, so every figure here is a lab-condition upper bound.
- **Row set.** `fold_a_daily_gap_v1`, 4,012,330 shared half-hours at the 16 Primaries (core set of
  A0, A1, and experiment 381's stored control). Scored half-hours with a missing actual: 0. The
  noisy start of D15 and D18 (before 2025-03-01) is cut from training rows and lags. D5, D12, D15,
  and D18 are marked in the tables.
- **Minimum detectable effect, computed before any score.** 0.032 points at the eight and 0.105
  points at D12, from experiment 381's `solar` arm minus its `control` arm. C1 is about 11 times
  that value.
- **Code check.** A0 minus experiment 381's stored GPU control: +0.001 (-0.003 to +0.005) at D5 and
  +0.006 (-0.010 to +0.036) at D12, inside the 0.05 limit.

| Contrast (A1 minus A0), CRPS days 1 to 10 | Estimate | Interval |
|---|---|---|
| C1, MVA-only eight | -0.355 | 99.67%: -0.497 to -0.228 |
| MVA-only seven (no D8) | -0.386 | 99.67%: -0.543 to -0.241 |
| Signed-MW six | -0.476 | 99.67%: -0.627 to -0.316 |
| D15 and D18 | -0.313 | 99.67%: -0.429 to -0.201 |
| D12 | -0.769 | 99.67%: -1.180 to -0.406 |
| D8 | -0.138 | 99.67%: -0.220 to -0.058 |

- **Per Primary (95%).** All 16 point estimates are negative, from -0.053 at D4 to -0.843 at D5. 15
  of 16 intervals lie wholly below zero. D15 is the exception, -0.149 (-0.305 to +0.018). No Primary
  is worse than +0.05.
- **Absolute CRPS (days 1 to 10).** A0 and A1 per Primary are in `scores/scores.parquet` in the data
  folder. At D12: A0 6.82, A1 6.05, stack 7.52, BG2 6.75, `manual_heuristic` 7.92, `climatology`
  8.83. At D5: A0 3.89, A1 3.05, stack 4.08, BG2 3.94. A1 is below `manual_heuristic` and
  `climatology` at all 14 Primaries of the signed-MW and MVA-only groups, and at D12 and D5 it is
  below BG2.
- **Lead bands.** The gain at the eight grows from -0.24 at day 1 to -0.42 at days 8 to 10 and stays
  level through days 11 to 14. At the signed-MW six it grows from -0.32 to -0.57.
- **Slices.** Sunny half-hours (unit shape): -0.97 (-1.41 to -0.61) at the eight and -1.33 (-1.77 to
  -0.96) at the six. CAMS index at least 0.7: -0.63 at the eight and -0.88 at the six. Winter
  nights, cold nights, and Economy 7 step hours move by under 0.02 points. Rows in the 7 days after
  a labelled event: -0.37 (-0.73 to -0.17) at the eight. Rows during a labelled event: -0.28 (-0.65
  to +0.05). These counts are lower bounds, because NGED's event labels are incomplete: rows with no
  labelled event may hold unlabelled events.
- **D12 bias on sunny half-hours (median minus actual, points, unit-shape slice).** A0 -4.89, A1
  +2.09 over days 1 to 10. A0 runs from -2.10 at day 1 to -6.34 at days 8 to 10, and A1 stays
  between +1.87 and +2.37. The columns remove the under-forecast and slightly overshoot. At D8 the
  bias falls from +9.97 to +7.90.

| Tails, mean over each group (days 1 to 10) | A0 MVA-only | A1 MVA-only | A0 signed-MW | A1 signed-MW |
|---|---|---|---|---|
| Coverage q01 (nominal 0.01) | 0.0126 | 0.0124 | 0.0074 | 0.0070 |
| Coverage q05 (0.05) | 0.069 | 0.072 | 0.052 | 0.055 |
| Coverage q95 (0.95) | 0.933 | 0.936 | 0.909 | 0.913 |
| Coverage q99 (0.99) | 0.991 | 0.991 | 0.980 | 0.981 |
| Pinball q01 | 0.248 | 0.242 | 0.249 | 0.240 |
| Pinball q05 | 0.883 | 0.829 | 0.674 | 0.600 |
| Pinball q95 | 0.975 | 0.903 | 0.947 | 0.844 |
| Pinball q99 | 0.332 | 0.326 | 0.385 | 0.378 |
| twCRPS | 0.201 | 0.191 | 0.386 | 0.380 |

- **Tails.** The upper tail improves most (pinball q95 -0.07 at the eight and -0.10 at the six).
  Coverage moves by under 0.005 and stays at the same side of nominal as A0.
- **Avoidable curtailment cost, signed-MW six, scored year, pounds (as
  `metrics.avoidable_curtailment_cost` defines it).** At the 0.9 limit, the headline: A1 is lower
  than A0 at D10 (28,733 against 29,036), D11 (201,909 against 213,752), and D5 (426,074 against
  427,491), and higher at D14 (2.217 million against 2.177 million), D16 (7.465 million against
  7.446 million), and D9 (169,524 against 166,786). The stack is lower than both at D10, D11, D14,
  D16, and D9. At the 0.5 limit A1 is lower than A0 at all six Primaries (for example D16, 24.3
  million against 25.2 million). The two row sets "during a labelled event" and "in the 7 days after
  one" could not be scored: the function needs a perfect-forecast arm that is empty on those rows.
  The better CRPS did not carry over to the cost at the 0.9 limit.
- **Which of the pre-registered rules held.** C1's interval lies wholly below zero and no Primary is
  worse than +0.05, so the port rule is met for the two columns, limited to lag-time use. The
  no-harm contrasts at the signed-MW six and at D15 and D18 have upper bounds at -0.316 and -0.201,
  below +0.05. The A2 stop rules were not tested.
- **Predictions.** P1 (0.03 to 0.10 points at the eight) failed on size: -0.355. P2 (largest at D12
  and D8) held for D12 only; D8 is among the smaller gains and D5 is the largest overall. P3 (gain
  concentrated in days 1 to 3) failed: the gain grows with lead. P4 (under 0.03 at the other eight)
  failed: the signed-MW six gain -0.476.

## Checks

- **Median minus mean forecast per Primary (MW, mean of the quantile-based mean estimate).** All 16
  are between -0.093 and +0.004 in A0 and between -0.089 and +0.004 in A1, so the spread does not
  shift the forecast. The gap is smaller in A1 at 13 of 16 Primaries.
- **Error of the median beside the CRPS contrasts.** The median's mean absolute error is lower in A1
  at all 16 Primaries (for example D5 6.76 to 5.11, D12 12.27 to 10.67), so audit trigger 3 did not
  fire.
- **Triggers fired.** Trigger 2 (a result contradicting its prediction) fired on size. The tested
  explanation is that the columns hold realised irradiance in place of a forecast: the gain is 96%
  within 08:00 to 16:00 UTC (peak -1.43 points at 12:00), under 0.015 points between 19:00 and
  05:00, negative in all 12 scored months, and larger in the months of strongest sun. A fresh Sonnet
  audit (report in the private `briefs/e480_audit_report.md`) rebuilt the CRPS from the stored
  quantiles with its own code, matched all 16 per-Primary contrasts to within 0.0005, and found no
  leakage in the CAMS join. The audit did not retrain, so a refit with the CAMS columns shuffled
  across days at one Primary remains the stronger test. Trigger 4 did not fire: leaving out any one
  Primary leaves the eight between -0.295 (without D12) and -0.398 (without D4), and the six between
  -0.404 (without D5) and -0.503 (without D11), all intervals below zero. Trigger 5 did not fire:
  the control A0 moved by under 0.01 against experiment 381's stored control. Trigger 7 is moot
  because the A2 arms were not started.

## Caveats and what to try next

- **Lab conditions.** The gain is an upper bound. A live service has the realised satellite value
  only for data at least about a day old.
- **Mixed result at the tail cost.** The better CRPS at the signed-MW six did not lower the
  avoidable curtailment cost at the 0.9 limit.
- **D12 overshoot.** A1 over-forecasts sunny D12 half-hours by about 2 points at all leads, so the
  correction overshoots.
- **Not tested here.** Power factor 0.97 stays a known failure of the reverse-flow rule (experiment
  366). The known-answer test and the A2 arms repeat work in experiments 366, 372, 374, 387, 393,
  420, and 424 and are not started.
- **Next.** Build the same two columns from the ensemble's forecast irradiance and from CAMS only
  for lag times older than a day, and measure how much of the 0.355-point gain remains. A
  day-shuffle refit of the CAMS columns at one Primary would test the mechanism directly.

## Problems caused by the move

None.
