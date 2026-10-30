---
id: "00409"
slug: mdn-solar-term-day-of-year-and-anomaly-known-answer
title: >-
  Can the explicit solar term inside the mixture density network read the sky once the network has a
  day-of-year home for the daylight profile, and does it then improve the net-demand forecast?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
  - hierarchy-and-pooling
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted networks, predictions, the learned solar contribution per row, and the
  injected copies of the data, all in this experiment's private folder. Production serving path: not
  touched (this repository holds none). Degradation rule: not touched. Designs: more than one is
  defensible (how the known-answer test is repaired, how the day-of-year term is built, how the
  placebo limit is set). Callers: the code copies experiment 405's modules and imports experiments
  87, 129, 397, and 405.
hypothesis: >-
  A pooled mixture density network that has a day-of-year by hour term in its demand module can
  attribute the monthly daylight profile of a farm to the demand module, so the explicit solar term
  learns capacity only from the day-to-day sky. Under that repair the known-answer test can be
  passed, and the solar term then lowers the net-demand CRPS at the six signed-direction Primaries
  against the same network given no solar term.
prediction: >-
  Written before any fit. K1 (anomaly variant): the learned capacity increment of C2 is 0.5 to 1.5
  of the least-squares read-out of the injected anomaly on the ensemble unit shape at 4 of the 6
  Primaries. K2 (anomaly placebo): the day-shuffled anomaly recovers at most the profile-only
  benchmark plus 0.20, where the benchmark is the same read-out on the shuffled series (expected
  near 0). K3 (profile variant): the real-farm increment recovers more than the day-shuffled farm
  by more than 2 seed standard deviations at 4 of the 6 Primaries. P1: C3 minus A0-noirr-doy is
  between -0.15 and 0.00 CRPS points at the signed-MW six. P2: C2 minus A0-noirr-doy is negative by
  0.05 to 0.30 points. P3: C3 minus C2 is between -0.10 and 0.00 points with an interval that
  includes zero. The score arms run only if K1 and K2 both hold.
decision_rule: >-
  See "Pre-registered rules". The known-answer gate is passed when K1 and K2 both hold on the
  anomaly variant. A score contrast counts as a gain only if its 95% month-block interval,
  Bonferroni-corrected across the contrasts listed, excludes zero and the point estimate exceeds the
  contrast's minimum detectable effect (2 times the standard deviation of its seed-to-seed
  difference over the 5 folds). A change that costs nothing in production would be worth porting at
  any gain with an interval excluding zero, but the solar term adds a module and a capacity input,
  so the threshold is the minimum detectable effect.
benchmark: >-
  Arm A0-noirr-doy (the pooled plain network with no solar term, no irradiance, and the day-of-year
  term), arm A0 (the pooled plain network of experiment 403, stored), and the stored forecasts of the
  monthly XGBoost (experiment 389), BG2, and the GAM stack, read and never refitted.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, in points of effective capacity, from
  `studies.eval_protocol`; leads 0.5 to 336 hours; the headline band is days 1 to 10.
date: 2026-10-30
code: ml-experiments/00409-mdn-solar-term-day-of-year-and-anomaly-known-answer/
---

**Status: planned. Nothing has been fitted.** The plan follows the maintainer's request after
experiment 405, whose placebo gate could not be met.

## What is asked

**The question is whether the explicit solar term can be shown to read the sky, and if so whether it
lowers the net-demand error.** Experiment 405's known-answer test failed its placebo gate because a
day-shuffled farm series keeps 79% of the real series' variance through its monthly daylight
profile, and the network had no day-of-year term to hold that profile. The aim behind the series is a
mixture density network that combines the best of the GAM and XGBoost: interpretable, differentiable
physics parts (here an explicit solar term) beside flexible learned parts, so that the fitted network
gives explainable outputs and the best CRPS. This plan therefore also stores and plots the learned
solar contribution, so that the solar term can be read on its own.

## Review of the plan against the latest evidence

- **Experiment 405 and its audit** (merged) set every change below: the profile-only benchmark, the
  day-of-year term, the anomaly injection, and the dropped divisor.
- **Experiment 403** (merged): pooled A0 is the reference network, CRPS 3.351 at the signed-MW six.
  Its stored forecasts are the A0 control if its rows match this experiment's rows (checked before
  scoring; otherwise A0 is refitted).
- **Standing rules added to `common.md` since the brief:** cleaned data, the D15 and D18 start cut,
  monthly refits, saved models, tails, example forecast graphs, the seven-step process. All are
  followed below. No result merged since the brief answers the question.
- **Departures from experiment 405:** the divisor is dropped (finding 4: the three other farms carry
  the same signal, so the ratio sits near 1); the increment-penalty arm is dropped (no effect);
  the placebo limit is set from a measured benchmark instead of copied from experiment 129.

## Terms

- **Net demand, signed-MW six, MVA-only eight, solar term, capacity:** as in experiment 405. The
  solar term is a capacity per Primary and season-year times a fixed physical unit shape, and the
  network subtracts it from the demand mixture.
- **Daylight profile.** The mean of a farm's output at each half-hour of the day within a calendar
  month of a year. **Day-to-day anomaly.** The farm's output minus its daylight profile.
- **Least-squares read-out.** The capacity change that best explains an injected series from the
  ensemble unit shape on the fold's training rows, `dC_LS = eff * sum(f * u) / sum(u^2)`, over
  daylight rows of the last season-year, where `f` is the injected series (a fraction of effective
  capacity `eff`) and `u` the unit shape. It is the benchmark a network that read the shape linearly
  would reach.

## Data and rows

As in experiment 405: experiment 397's raw matrices in the `summary` scheme (ECMWF ENS mean and
spread, one pass), 16 Primaries, pooled network with an 8-wide Primary embedding, 3 Gaussian
components, 2 hidden layers of 64. **Cleaned data:** `substation_zero` half-hours are dropped from
training, lags, inference, and scoring, as experiment 397's matrices do; the dropped count per fold
is reported. **Start cut:** D15 and D18 drop every row, lag input, and scoring actual before
2025-03-01 (applied; D15 and D18 stay report-only). **Cadence:** monthly refits on a 730-day window,
five screen months (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), 3 seeds, each booster saved to
the experiment's private folder. **Weather at lag times:** the lags are the Primary's own power, so
no weather is used at lag times and the ERA5 delay does not apply.

## Architecture

**The solar module is experiment 405's.** The net-demand mixture keeps the demand mixture's weights;
component `k` has mean `mu_k - m` and variance `sigma_k^2 + s^2`, where `m = C[p, y] * u * g` and
`s` follows the ensemble spread. Capacity is learned per Primary and season-year under a Gaussian
prior from the studies page (C2), with the adapter `g` (C3).

**The change is a day-of-year by hour term in the demand module.** The demand module of every C arm
and of A0-noirr-doy gets 20 extra inputs: the sine and cosine of the day of year at one and two
harmonics (4 inputs), and each of the four by the four existing hour-of-day inputs (`sin_hour`,
`cos_hour`, `sin_hour_2`, `cos_hour_2`; 16 inputs). The network can then represent the monthly
daylight shape of the Primary's net demand in the demand module. Irradiance and the three
solar-geometry columns stay removed, as in experiment 405.

## Known-answer test (runs first, gates the score arms)

**The test injects a held-out farm's output into the six signed-MW Primaries and asks whether the
learned capacity rises by the share the ensemble forecast can explain.** It runs on the 2026-06
fold only. Injection follows experiment 405 (farm C, increment 50% of the studies page's 2025/26
capacity, subtracted from the target and the three lags, dropouts removed from every condition).

| Condition | Injected series |
|---|---|
| control | none |
| held_profile | the real farm series (variant a) |
| placebo_profile | the real farm series with its days shuffled within the month (variant a) |
| held_anomaly | the farm's day-to-day anomaly, its daylight profile removed (variant b) |
| placebo_anomaly | the anomaly with its days shuffled within the month (variant b) |

The daylight profile is the farm's mean at each half-hour within each calendar month of each year,
so each month's anomaly has zero mean at every half-hour and the shuffled anomaly keeps no profile.
Arms: C2 (prior) and C4 (no prior, neutral 3 MW start), both with the day-of-year term, 3 seeds.
That is 5 conditions x 2 arms x 3 seeds = 30 fits.

**Recovery** is the seed-mean learned capacity increment (injected minus control, at the last
season-year in the window) divided by `dC_LS` of the injected series on the same rows. The
least-squares read-out is computed for every injected series, including each placebo, so the
benchmark is measured on the same series as the fit.

## Pre-registered rules

1. **Gate K1 (recovery, anomaly variant).** C2's recovery on `held_anomaly` lies in 0.5 to 1.5 at 4
   of the 6 Primaries. The band is wider than experiment 129's 0.75 to 1.25 because the anomaly is
   only about one fifth of the farm's variance and the ensemble forecast correlates with it at about
   0.37.
2. **Gate K2 (placebo, anomaly variant).** C2's recovery on `placebo_anomaly` is at most
   `dC_LS(placebo_anomaly) / dC_LS(held_anomaly) + 0.20`, in median and in maximum over the six. The
   benchmark term is expected near 0 because the shuffle destroys the anomaly's link to the forecast.
3. **Gates are on the anomaly variant, C2.** The profile variant (a) decides nothing. It reports K3
   (real minus placebo recovery above 2 seed standard deviations at 4 of 6) and the same recovery
   ratios against its own benchmark (0.84 to 0.97 by the audit of experiment 405), to show how much
   of the profile the day-of-year term absorbs. C4 is reported beside C2, not gated.
4. **If K1 or K2 fails**, the page reports the failure, with the profile variant and C4 beside it,
   and the score arms do not run.
5. **If K1 and K2 hold**, the score arms run: A0-noirr-doy, C1 (capacity pinned to the experiment
   269 estimate), C2, C3, C4, C5a (monthly capacity path, smoothness penalty), and C5b (persistence
   penalty), all with the day-of-year term, 3 seeds, 5 folds; A0 is read from experiment 403. The
   contrasts at the signed-MW six, CRPS at days 1 to 10, month-block bootstrap at 95% with 20,000
   resamples, Bonferroni across the nine: C3 minus A0-noirr-doy, C2 minus A0-noirr-doy, C2 minus A0,
   C3 minus A0, C3 minus C2, C2 minus C1, C4 minus C3, C5a minus C3, C5b minus C3.
   The MVA-only eight are reported, not contrasted.
6. **Minimum detectable effect** and the gain rule are as in the decision rule above.

## Reports

- **Tails and cost.** Coverage and pinball loss at q01, q05, q95, and q99, raw and at the 1.15
  widening; the cost at the scaled export limit (f = 0.9) with `studies.node_curtailment`; the
  sunniest third of daylight half-hours and sunny April to June.
- **The solar contribution, readable on its own.** The learned solar output `m` in MW is stored per
  scored row. The page plots, for each of the six, the mean learned output by month and hour beside
  the physical clear-sky shape times the studies-page capacity, the learned capacity path with
  intervals from seeds, and the same Primary's forecast from the network with the solar term beside
  the forecast from A0-noirr-doy (the same network with no solar term).
- **Example forecasts.** A day-ahead and a full 14-day forecast from one run, for one signed-MW and
  one MVA-only Primary, in a winter and a summer month.
- **Checks.** Median minus mean forecast per site, the mean-forecast error beside each contrast, the
  audit triggers that fired, the flagged half-hours dropped per fold, and the saved-booster path.

## Compute and fits

- **Known-answer test:** 30 fits at 50 to 200 seconds each, 0.5 to 2 hours.
- **Score arms (only if the gate holds):** 7 arms x 5 folds x 3 seeds = 105 fits, 1.5 to 6 hours.
- **Stored predictions reused:** experiment 403's A0 forecasts, the XGBoost, BG2, and GAM-stack
  references. Experiment 405's `extras` and `u99` files are reused, which saves the unit-shape pass.
- **Resources:** one GPU process under 8 GB, one feeder core, `MemoryMax=8G`, `OMP_NUM_THREADS=1`.
  No fit starts until the manager grants the fit slot (idle CPU at least 30%, available memory at
  least 15 GB).

## Earlier advice followed and departed from

- **Followed:** calibrate the placebo gate to the network; give the demand module a day-of-year by
  hour term and inject the anomaly (experiment 405 advice); use an as-of capacity; report absolute
  skill beside contrasts.
- **Departed from:** experiment 129's divisor and five-year drift gate; the former is uninformative
  (experiment 405 audit, finding 4) and the latter does not fit a two-season-year window.

## Reviews

To be filled in: design review of this plan, code review before the first fit, results review,
write-up review.
