---
id: "00437"
slug: neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles
title: >-
  Do simple input bundles that did not help XGBoost (thermal lag, solar gain, look-ahead, and
  calendar, tariff, and national signals) lower the error of the pooled mixture density network?
theme: demand-inputs
also_relevant_to:
  - weather-products-and-blending
  - production-reference-and-best-guess
status: planned
kind: change
protocol: demand-v2
size: >-
  medium. Stored: forecasts and fitted networks in this experiment's private folder only; every
  earlier experiment is read and never written. Production serving path: not touched (this
  repository holds none). Degradation rule: not touched. Designs: more than one defensible (which
  windows, which pretraining form), so the arms register each. Callers: the scripts import
  experiments 11, 87, 48, 377, 381, 397, 407, 418, and 425 and the `studies` evaluation code,
  whose callers sit in those folders. No new data is downloaded.
hypothesis: >-
  The pooled plain mixture density network of experiment 425 sees one valid time per row, so it
  cannot see the hours before or after that time. Adding columns that summarise those hours
  (bundle T, thermal lag; bundle S, solar gain; bundle F, look-ahead) or that carry the calendar,
  tariff, and national signals (bundle C) lowers its continuous ranked probability score (CRPS)
  at days 1 to 10, because the network trains across all 16 Primaries (and, in the pretrained
  form, across 6 years of ERA5 and CAMS history) and may find signal that the per-Primary
  XGBoost model and the GAM stack missed.
prediction: >-
  Written before any fit and before any score is read. Most single bundles will land inside the
  minimum detectable effect, so the registered predictions are directions with sizes that could
  fail. P1. S lowers the CRPS on cold sunny days (ENS-mean temperature below 5 degrees C, clear-sky
  index above 0.6, 09:00 to 14:30 local) by at least twice as much as on cold overcast days
  (clear-sky index below 0.3), the maintainer's heating hypothesis; its overall gain is 0.01 to 0.05
  points at both groups. P2. T lowers the CRPS by 0.01 to 0.05 at both groups, and more on cold
  winter nights. P3. F lowers the CRPS by 0.01 to 0.05 at the MVA-only eight (the home-battery and
  pre-heating argument) and by less at the signed-MW six. P4. C lowers the CRPS on the Economy 7
  step half-hours by 0.02 to 0.10 and changes nothing elsewhere; its national-signal columns
  (price, Demand Flexibility Service flag, national demand, embedded generation of the previous
  day) change the CRPS by under 0.01, as experiments 10 and 185 found for other models. P5. The
  pretrained form (T, S, and F pretrained on ERA5 and CAMS history, then fine-tuned) gains at least
  as much over the pretrained control `A` as the ensemble-only form gains over `R`, because the
  pretraining rows hold 6 years of weather windows.
decision_rule: >-
  CRPS at days 1 to 10 in points of effective capacity, per group (signed-MW six; MVA-only eight;
  D15 and D18 report-only), widened by 1.15, paired by fold and seed, month-block bootstrap of
  `studies.eval_protocol` with 20,000 resamples, over the four screen folds (2025-10, 2025-12,
  2026-01, 2026-03) and three seeds (0 to 2). Seven bundle arms are tested (T, S, C, F against
  `R`; T_pre, S_pre, F_pre against `A`), so the interval level is 99.29% (Bonferroni, alpha
  0.0071). The smallest gain worth carrying into a full evaluation is 0.03 points, because each
  bundle adds a pipeline step that must be built from forecast inputs at serving time. A bundle
  advances if its interval lies wholly below -0.03 at the signed-MW six or the MVA-only eight, its
  point estimate at the other group is below +0.03, and its increase in pinball loss at q95 and
  q99 is at most 0.02 at both groups. Before any contrast is read the scorer prints the minimum
  detectable effect: the standard deviation over the four folds of the signed-MW six CRPS
  difference between the control's seeds 0 to 2 and seeds 3 and 4, divided by the square root of
  two (a lower bound; experiment 425 found 0.0368 over 12 folds, and experiment 419 about 0.074 on
  its convention). A contrast smaller than that is "inside the noise". This is a four-fold screen:
  every number is interim, no 99% claim is made, and nothing here enters a research default.
  Halves of a winning bundle and combinations are descriptive.
benchmark: >-
  `R` (experiment 425's ensemble-only pooled plain network, seeds 0 to 2, same rows) for the
  ensemble-only arms and `A` (experiment 425's pretrained network, seeds 0 to 2) for the pretrained
  arms; BG2, `manual_heuristic`, `climatology`, and the monthly XGBoost recipe `S0M` as
  absolute-skill references on the shared half-hours.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, coverage and pinball loss at q01, q05,
  q95, and q99, the avoidable-curtailment cost, and four slices (cold sunny days, cold overcast
  days, shoulder seasons, morning ramp hours) beside the protocol's slices, in points of effective
  capacity, from `studies.eval_protocol`; leads 0.5 to 336 hours; the headline band is days 1 to 10.
code: ml-experiments/00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles/
---

**This plan tests, on four folds, whether windows of weather and a few calendar and national
signals lower the error of the pooled neural network that experiments 397 to 425 built.** The
network predicts 11 quantiles of Primary substation demand from the ECMWF ensemble mean and spread
at one valid time. Experiment 419 gave it a centred daily mean of shortwave and found
-0.012 (-0.043 to +0.013), one sixth of its minimum detectable effect. Trailing and forward windows
of temperature and shortwave were never given to the neural network, although the maintainer asked
for them on 2026-10-31.

## Design

- **Reference.** `R` and `A` of [experiment 425](../00425-best-mdn-architecture-assembly-with-ablations/result.md):
  the pooled plain network with the 8-wide Primary embedding, 3 Gaussian components, setting `g09`,
  and the ensemble mean and spread inputs, trained on ensemble rows only (`R`) or pretrained on
  ERA5 and CAMS rows and fine-tuned at a learning rate of 2e-4 (`A`). Same rows, the D15 and D18
  start cut at 2025-03-01, the flagged exact-zero drop, held-out weeks, row weights, 1.15 widening,
  and monthly refits. The stored seeds 0 to 2 of both arms on the four screen folds are the controls.
  A check (`e00437_fit.py check`) proves that a fold built here without extra columns equals
  experiment 425's fold, bit for bit.
- **Arms.** Each arm appends columns to the 46 inputs and changes nothing else. Bundles:
    - **T, thermal lag (8 columns).** Trailing means of the ensemble-mean temperature over 3, 6, and
      24 hours; the temperature change over 3 and 6 hours; the minimum and maximum over the local
      day of the valid time; and the days since the trailing 24-hour mean last exceeded 5 degrees C,
      capped at 10.
    - **S, solar gain (6 columns).** Trailing means of the ensemble-mean shortwave over 3, 6, and 24
      hours; the 3-hour change in the clear-sky index (shortwave over Haurwitz clear-sky shortwave,
      held at its last daylight value overnight; the helper of experiment 396 arm W7); and the
      realised CAMS irradiance of the last complete UTC day before the origin (its mean in W/m2 and
      its ratio to CAMS clear-sky). CAMS publishes the previous day at midnight UTC, so production
      can read both.
    - **F, look-ahead (6 columns).** Forward means of ensemble-mean temperature and shortwave over
      the 6 and 24 hours after the valid time, taken from the same run only, and the share of each
      window the run covers (windows are truncated at the end of the 14-day horizon).
    - **C, calendar, tariff, and national signals (14 columns).** The Economy 7 step as a UTC
      half-hour indicator and that indicator times degrees below 15.5 C; five Lincolnshire
      school-term columns from experiment 11's calendar (term flag; signed days to the nearest term
      start and end; signed days from the year's first working day; a window flag); and seven
      signals of the local day before the origin: the mean, maximum, and 16:00 to 19:00 mean of the
      Elexon market index price, a Demand Flexibility Service flag, and the day's mean national
      demand, embedded solar, and embedded wind (NESO historic demand). All 16 Primaries lie at
      52.9 to 53.3 degrees N, 0.2 degrees W to 0.3 degrees E, inside the box that experiment 11 found
      to be Lincolnshire, so one county's calendar applies to every Primary.
- **Pretrained form.** T, S, and F are also run as `T_pre`, `S_pre`, and `F_pre`: pretrained on
  experiment 418's ERA5-and-CAMS pseudo-run rows with the same columns computed from the dense
  weather, then fine-tuned on ensemble rows at 2e-4. C has no pretraining form, because the
  pretraining rows carry no school or national columns. A pretrained network is fitted for each
  bundle, fold, and seed, and cannot be shared across folds: the pretraining rows end 3.5 hours
  before each fold's refit, so a network pretrained for a later fold has seen the earlier fold's
  test month.
- **Window construction.** A trailing window ends at the valid time. For half-hours at or before
  the forecast origin (09:00 UTC) the series holds the ensemble mean of the previous run that
  covers them (leads of about 9 to 33 hours); after the origin it holds this run's own ensemble
  mean. ERA5 is not used, because it arrives about 5 days late and a column built from it could not
  be served. `e00437_features.py check` rebuilds sample runs with every later run deleted and
  requires identical columns (it did, to 0.0). In the pretrained form the windows come from the
  dense ERA5-and-CAMS series, which is the same stand-in the pretraining rows use for the
  valid-time weather.
- **Known mismatch in the pretrained form.** `check-pretraining` printed each column's standardised
  mean and spread in the pretraining rows and the ensemble rows for the 2026-01 fold. The weather
  windows agree to within 0.25 standard deviations in the mean and 0.07 in the spread, except
  `csi_change_3h`, whose spread is 2.6 in the pretraining rows against 0.9 in the ensemble rows. CAMS
  irradiance changes more over 3 hours than the mean of 51 ensemble members does, the same smoothing
  gap that affects the shortwave input itself. The write-up reports it as a caveat for `S_pre`.
- **Dismantling.** A bundle that advances is split into halves and refitted in the same form:
  `T_means`/`T_rest`, `S_means`/`S_rest`, `F_temperature`/`F_shortwave`, and for C
  `C_tariff`/`C_school`/`C_national`. A combination of the winning bundles (`T+S+F`, with or
  without `C`) is fitted once at least two bundles advance. These are descriptive.
- **Fits and compute.** 4 bundles by 4 folds by 3 seeds is 48 ensemble-only fits (about 75 seconds
  each including the forecast) and 36 pretrained fits (about 55 seconds of pretraining and 40 of
  fine-tuning each, plus a pretraining-row build of about 2 minutes per fold and bundle). That is
  about 1.0 plus 1.8 GPU-hours before dismantling, and up to 2 more GPU-hours for the halves and
  the combination. One GPU process under 8 GB, one feeder core, 8 GB of host memory, one BLAS
  thread.
- **Standing practices followed.** Monthly refits; cleaned data (flagged exact-zero half-hours
  dropped, counts printed per fold); the D15 and D18 cut at 2025-03-01; every fitted network saved
  under `work/models/`; stored controls reused; tails, the avoidable-curtailment cost, and the
  example forecasts reported; four-fold screen with every number labelled interim.

## Results

Not yet run.
