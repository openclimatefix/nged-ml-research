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
status: planned
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
---

**Status: planned.** This page is the pre-registered plan; the write-up replaces it when the fits
finish. Nothing here has been scored yet.

We ask whether a monthly XGBoost demand model forecasts better when each half-hour carries two
extra columns computed from the irradiance that satellites measured, rather than what the weather
model forecast. The eight Primaries in the test report apparent power in MVA, which hides reverse
flow from embedded solar panels, so the model has no direct view of how much solar a half-hour
held. We refit the model monthly with and without the two columns on the same rows, and compare
the CRPS of the forecasts at days 1 to 10. This is "lab" conditions: the columns use realised
irradiance, which a live service cannot have for the first day of lead.

## Scope: stage 1 only

**This experiment is stage 1 of the brief `xgboost_mva_signed_label_cams_brief`: arms A0 and A1.**
The queue-order check of 2026-11-01 found that the rest of the brief repeats stored work: the
reverse-flow rule and its placebos (experiment 366), the GAM's rule-chosen fits (experiments 372
and 374), the folded XGBoost loss (experiment 387, whose sign followed the start tilt at 68% to 98%
of sunny middays), the fold's effect on the CRPS gap (experiment 393), and the neural network's
export gate at D8 and D12 (experiments 420 and 424). The one open piece is whether the two CAMS
columns alone, with no sign label, help the XGBoost model at the eight. The known-answer test (level
2.0, then 1.0) and the A2 arms (sign label with a threshold of 0.7 and 0.5, the switched-off arm,
and the shuffled arm) are queued behind this stage and are not started. If A1 shows no gain they
do not start without the maintainer's word.

## Design

- **Arms.** A0 is `monthly/control` of experiment 381 refitted fresh: the recipe of experiment 87's
  `n1h` configuration (the BG2 monthly member), 500 rounds, eta 0.05, depth 6, seed 0, built-in
  `reg:quantileerror` at 11 levels, whole-history rows with a 0.5-year half-life, 51 members mixed
  at prediction time. A1 is A0 plus two columns.
- **The two columns.** For each half-hour ending at `valid_time`: `cams_unit_pv_shape`, the unit PV
  output of the generic south-facing fleet from `studies.dp_features.unit_pv_shape`, and
  `cams_clear_sky_index`, the CAMS global horizontal irradiance of that half-hour (the hour's
  value spread over its two half-hours by the cosine of the zenith, `studies.pv_physics.half_hour_sun_and_sky`)
  divided by the Haurwitz clear-sky irradiance at the half-hour's midpoint, set to 0 where the
  clear-sky irradiance is below 1 W/m2. Both are the half-hour's own realised values and are the
  same for every ensemble member, and they enter training rows, prediction rows, and nothing else.
- **CAMS source and delay.** The CAMS radiation service's satellite retrieval at each site's
  coordinates (`CAMS_PATH` in `studies.dp_features`, hourly, 2019-09 to 2026-09). The service
  publishes with about a 1-day delay, so the columns are unavailable at forecast time for any valid
  time in the first day of lead and are available for lag times older than about 1 day.
- **Rows and series.** The 16 Primaries, `fold_a_daily_gap_v1`, scored on the shared half-hours of
  every arm (`metrics.shared_slice_table`, one row set for A0 and A1). Fitted at all 16: the eight
  MVA-only Primaries carry the confirmatory contrast, and the other eight are the no-harm group.
  D5, D12, D15, and D18 are marked in the tables (D12 is a reverse-flow Primary; D5 is the hard
  case for the reverse-flow rule; D15 and D18 are the short-history Primaries).
- **Standing practices followed.** Monthly refits, no weekly. Cleaned data: training rows and lags
  come from experiment 87's `observed.parquet` (flagged half-hours are nulls and drop out of the
  training rows), and the number of flagged half-hours in the scored set is printed. The noisy start
  of D15 and D18 is cut: the power series of both are set to missing before 2025-03-01, so training
  rows and lag inputs omit them; this differs from experiment 381, so A0 is refitted at these two
  Primaries rather than read from the store. Fitted boosters are saved as `.ubj` in the experiment's
  data folder. Reviews: a fresh Sonnet agent reviews the code before the first fit, another checks
  the results, and two review the write-up.
- **Compute.** CPU only, 3 threads, `MemoryMax=16G`, no GPU (experiment 381's stored control used the
  GPU, so A0 is refitted on CPU to give the contrast the same device on both arms). The fit count is
  384: A0 and A1 at 16 Primaries and 12 refits (192 fits each). The queue-order check priced the stage
  at about 288 fits (eight Primaries for A0 and A1, and A1 alone at the other eight); the extra 96 are
  A0 at the other eight, needed because the stored control was fitted on the GPU and without the
  D15 and D18 cut.
- **Stored forecasts reused.** Experiment 381's stored `monthly/control` is a reference and a code
  check, not a comparator for the contrast. Experiment 377's stack, `manual_heuristic`, `climatology`,
  and BG2 are read from the stores as references.

## Pre-registered checks

1. **Code check.** A0 at D5 and D12 matches experiment 381's stored `monthly/control` within 0.05
   points of CRPS at days 1 to 10. A larger gap is reported as a device or seed effect and the
   contrast is read against the fresh A0 only.
2. **Minimum detectable effect.** `e00480_mde.py` computes the half-width of the 99.67% interval of
   experiment 381's stored `solar` arm minus its `control` arm over the eight, and over D12 alone,
   before any A0 or A1 score is read; the result is written to this page in its own commit. A C1
   point estimate smaller in size than that half-width cannot be told from zero.
3. **Row hygiene.** All tables use "rows with no labelled event" for rows outside NGED's hand
   labels, because NGED's event labels are incomplete: clean rows may hold unlabelled events and
   event counts are lower bounds.
4. **Tails.** q01, q05, q95, and q99 coverage and pinball loss, per group and per Primary, beside
   CRPS and twCRPS, and the avoidable-curtailment cost in the three row sets (during a labelled
   event, in the 7 days after one, and across all rows).
5. **Sunny half-hours three ways:** realised `sunny_clear_sky_index`, the CAMS index at or above
   0.7, and `sunny_unit_shape`. D12's bias on sunny half-hours is reported by lead band for A0 and
   A1, because experiment 381's bias of -4.92 points at D12 is mostly selection on realised sunshine.
6. **Six and eight apart.** The six signed-MW Primaries and the eight MVA-only Primaries are
   reported separately, with and without D8 in the MVA-only group, and never pooled into one figure.

## Known failures and risks

- **Reactive power.** The reverse-flow rule of experiment 366 holds only for reactive power zero. At
  a power factor of 0.97 it fails (0 of 5 Primaries in experiment 366). Stage 1 uses no sign label, so
  the failure does not reach A0 or A1, and the queued A2 arms inherit it.
- **Lab conditions.** The columns hold realised irradiance. A gain is an upper bound on what the
  forecast irradiance can deliver and does not carry to a live service at the first day of lead.
- **Solar attribution.** The columns assume that irradiance explains the embedded generation at each
  Primary. Where other generators sit behind the meter, the columns carry nothing for them.

## What the next experiment should answer

To be written with the results.
