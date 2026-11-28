---
id: "00737"
slug: bayesian-solar-capacity-part-1
title: >-
  Can a Bayesian fit read the capacity of unmetered solar panels, quarter by quarter, from the power
  of the six Primary substations that NGED meters in signed megawatts, when we add a fleet of known
  size to the real record?
theme: solar-disaggregation
also_relevant_to:
  - lsc-detectors
status: planned
date: 2026-11-28
kind: other
size: >-
  Medium. (1) Stored: private fit tables, posterior tables, and score tables under
  data/ml-experiments/00737-bayesian-solar-capacity-part-1/ only; no neural network is trained, and
  no fitted model is saved except the small coefficient files. (2) Designs: more than one design
  defensibly tests the question (fit the residual of experiment 734, fit the power with its own
  demand baseline, or fit daily sums), so the choice is fixed below. (3) Callers: this folder's
  scripts, which import the folders of experiments 39, 653, and 660 and `studies.pv_physics` and
  `studies.pv_separation` read-only; nothing imports them. (4) Serving path and degradation rule:
  not touched. (5) Contracts: not touched. CPU only, no GPU time, about 17 CPU-hours, with a stop at
  26 CPU-hours; the ML experiment manager allocates the cores. Reviews: plan, code, numbers, and
  prose.
hypothesis: >-
  A fit that knows the physics of solar panels can read, from a Primary substation's power alone,
  how many megawatts of unmetered solar panels sit behind the substation in each quarter, and can do
  so better than two simpler fits. We test the claim at the six Primaries that NGED (National Grid
  Electricity Distribution) meters in signed megawatts, by adding to each Primary's real record a
  synthetic fleet of known AC (alternating-current) capacity that grows over the 18 quarters from
  2022 to mid-2026. The fit, arm A, is a Bayesian regression of the power on a calendar-and-temperature
  baseline and a solar term, with one capacity per quarter, a prior that favours growth from one
  quarter to the next, and a fixed DC:AC ratio of 1.15 (the panels' direct-current rating is 1.15
  times the inverter's alternating-current limit). The path error is the mean absolute gap between
  the fitted and the injected capacity over the 18 quarters, as a share of the mean injected
  capacity. The claim has three parts. First, A has a path error at least 0.10 lower than arm F,
  a fit with one flat capacity (C1, a sanity gate). Second, A has a path error at least 0.05 lower
  than arm Qt, which fits each quarter separately with a flat prior (C2). Third, A passes seven
  bars (B1 to B7) on a synthetic fleet: the recovered share (the fitted increment over the injected
  capacity, summed over the quarters) lies near 1; the 80% interval covers the injected increment
  about 80% of the time; a placebo sky (the sunshine record of another year, with no fleet added)
  yields almost no capacity; a fleet whose orientation mix, tilt, and DC:AC ratio differ from the
  prior's is recovered as well as a matching fleet; a fleet built half from a real solar farm's
  output is recovered as well as a fleet built from physics alone; a load that follows the day's
  cloud moves the fitted capacity by at most 0.35 of the fleet; and a 30-day load transfer moves
  it by less than 0.25 MW. Every result is for a synthetic fleet and says nothing about the real
  capacity.
prediction: >-
  Written before any fit. P1 (bar B1): on the 162 main worlds, the median recovered share per
  Primary for the Bayesian fit A is 0.75 to 1.00, so B1 (0.85 to 1.15 at 5 of 6 Primaries, for the
  2.5 MW and 5 MW fleets) passes with probability about 50%. The range starts at the 0.77 of
  experiment 39 because A has the same kind of baseline and the same heavy-tailed weights, and
  the plan has no reason to expect more; the fleet and the fit share one forward model, so the
  share is also an upper limit for a real fleet. The oracle arm O (A given the injected mix, tilt,
  and ratio) reads 0.95 to 1.05. P2 (C1): the flat fit F has a path error of 0.53 on the ramp, 0.89
  on the step, and 0.79 on the accelerating shape if its one capacity sits at the mean, so
  C1 lies between +0.25 and +0.70 and passes at 6 of 6 Primaries with probability about 95%. P3
  (C2): the separate quarterly fits Qt are noisier than A on the ramp and the accelerating shape,
  so C2 lies between +0.02 and +0.10 there, and Qt follows a 5-quarter-wide step that A smooths,
  so C2 lies between -0.08 and 0.00 on the step shape; the pooled C2 lies between -0.02 and +0.06,
  and counts at 6 of 6 Primaries with probability about 40%. P4 (B3): the placebo sky gives a
  median capacity of 0.0 to 0.3 of the real fit's, so B3 passes at 4 to 6 Primaries. P5 (B6 and
  B7): a load that follows the day's cloud, at 25% of the fleet's capacity, raises the fitted
  capacity by +10% to +30% of the fleet's capacity (experiment 40, error bars on PV capacity,
  found about +20%), and a flat 30-day transfer of 1.5 MW moves the fitted capacity of its quarter
  by under 0.25 MW. P6: the free-ratio arm Af does not beat A by more than 0.03 on the mismatched
  worlds. The chance that each bar passes is about 50% for B1, 60% for B2, 75% for B3, 70% for B4,
  60% for B5, 80% for B6, and 85% for B7; all seven pass together with probability about 12%.
  Maintainer-side estimate of the verdicts, recomputed from P1 to P5: `supported` about 2%, `small
  gain` about 2%, `partly supported` about 16%, `not supported` about 42%, `inconclusive` about
  30%, `refuted` about 5%, `invalid` about 3%.
decision_rule: >-
  Fixed before any fit. C1 (F minus A) and C2 (Qt minus A) are each a difference in path error E on
  the main worlds, so a positive value means A is better. The unit of evidence is the Primary: each
  Primary's median difference over its worlds is one paired observation, so a contrast has six
  observations. An exact one-sided sign test at 0.025 (0.05 over two contrasts) is applied to the
  six observations. A split of 6 of 6 passes the test (p = 0.016), and a split of 5 of 6 fails it
  (p = 0.109). A contrast "counts" when the sign test passes, and "meets its threshold" when the
  median of the six per-Primary medians is at least 0.10 (C1) or 0.05 (C2). Both thresholds are
  differences in path error, which is a share of the mean injected capacity. C1 is a sanity gate:
  the flat fit F cannot have a path error below about 0.53 on any growth shape, so C1 fails only
  if A's path error exceeds about 0.43, and C1 cannot decide the verdict between A and a rival that
  follows the path. C2 is the contrast that could fail, and its threshold of 0.05 is a judgement
  that the maintainer may change before the first fit: stage 2 adds a physical forward model, two
  weather products, and a prior with a tuned value, so A has to beat the fit without a prior by a
  margin larger than the 0.03 that two seeds of one arm are expected to differ by. Seven bars on A
  (all counted on the worlds named): B1, the median recovered share per Primary within 0.85 to 1.15
  at 5 of 6 Primaries, for the 2.5 MW and 5 MW main worlds; B2, the pooled coverage of the 80%
  interval on the injected increment between 0.65 and 0.92; B3, the capacity fitted on a placebo
  sky at or below 0.25 of the real fit's capacity at 5 of 6 Primaries; B4, the median recovered
  share on the mismatched worlds at most 0.10 below the share on the matched worlds; B5, the median
  recovered share on the 12 real-farm worlds at most 0.10 below the median on the 2.5 MW ramp main
  worlds; B6, the median over the six Primaries of the cloud-load world's fitted increment, as a
  share of the fleet's capacity, at most 0.35; B7, the median over the six Primaries of the
  transfer world's shift in the fitted capacity at most 0.25 MW. The six gates G1, G1b, and G2 to
  G5 (known-answer, injection, look-ahead, timing, and convergence checks) pass first, or the
  verdict is `invalid`. The table "Verdict mapping" gives one row for every combination, read top
  to bottom, and the first row that matches applies. The Primaries share their weather, so the
  sign test's assumption of independent observations is generous; the verdict is read as
  descriptive evidence and not as a calibrated test.
benchmark: >-
  F (one flat capacity), Qt (A's likelihood and solar term, a flat wide prior on each quarter's
  capacity, fitted independently, with the baseline fitted on the whole record), and the references
  Q (the bounded least-squares fit of `studies.pv_separation` on each quarter alone), Ad (A on daily
  sums), Af (A with the DC:AC ratio fitted), O (A given the injected mix, tilt, and ratio), and R
  (the residual of experiment 734, read from stored forecasts). The recovered shares of experiment
  39 (0.77 half-hourly, 0.85 on daily sums) and of experiments 656 and 662 (0.76 to 0.82 for the
  linear probe at the signed Primaries) are read-only reference numbers.
row_set_id: >-
  experiment 653's grid: the six signed-MW Primaries (D5, D9, D10, D11, D14, D16), half-hourly,
  2022-01-01 to 2026-06-30 (18 quarters), cleaned signed power; D9 has data in about 63% of the
  half-hours. Not comparable with the leaderboard.
metric: >-
  Path error E (mean absolute gap between the fitted and the injected capacity per quarter, over
  the mean injected capacity), recovered share (sum of the fitted increments over the sum of the
  injected capacities), and the coverage of the 80% interval on the injected increment, on
  injected fleets of 1 MW, 2.5 MW, and 5 MW AC capacity.
code: ml-experiments/00737-bayesian-solar-capacity-part-1/
---

**The fit labels history after the fact, and the page is a plan, so no number below is a
result.** We ask whether a fit that knows the physics of solar panels can read, from a Primary
substation's power meter alone, how many megawatts of unmetered solar panels sit behind the
substation in each quarter of a year. A Primary is the kind of substation that steps voltage down
towards customers. NGED (National Grid Electricity Distribution) meters the power at a Primary as
demand minus the output of every solar panel connected below, and does not meter the panels. NGED
meters six of the Primaries in signed megawatts, which means the power is negative when the
substation exports power to the grid, and the six are the subject of the page.

**The fit is a Bayesian regression, and the test adds a fleet of known size to the real record.**
The regression turns satellite sunshine (the Copernicus Atmosphere Monitoring Service, CAMS), air
temperature, and wind speed (the ERA5 reanalysis, which is a record of past weather) into the
alternating-current (AC) output of one megawatt of panels. It then fits one capacity per quarter,
with a prior (an assumption stated before the data are read) that makes growth from one quarter to
the next more likely than a fall. The test adds the synthetic output of a fleet of known capacity,
growing over the 18 quarters from 2022 to mid-2026, to the real signed power of each of the six
Primaries, and asks how closely the fit recovers each quarter's capacity. The fit reads weather and
power from after each half-hour it labels, so it can label a historical record and cannot estimate
a live one.

**The page belongs to the second of four detectors that read a Primary's history.** The detector
for unmetered solar capacity is stage 2 of the pipeline in [How the history detectors fit
together](../detector-pipeline.md). Part 1 covers the six Primaries metered in signed megawatts, and
the Primaries metered in MVA (apparent power, which has no sign) wait for a later part. The
[background page on the signed-MW Primaries](../background/reverse-flow-at-signed-mw-primaries.md)
lists eight: D5, D9, D10, D11, D14, D15, D16, and D18. Part 1 uses six of the eight, for two
reasons. D15 has no stored power file and is not on the grid of experiment 653. D18 has finite
readings in about 33% of its half-hours, and experiment 653 cuts the record before 2025-03, which
leaves about 16 months. D18 could join later as an unscored reference.

**The earlier experiments leave a gap that this experiment fills, because none of them combines a
quarterly capacity path, an interval on every quarter, and a test with a known growing fleet.**
[Experiment 39](../00039-pv-disaggregation-cams-injection/result.md) added the recorded output of a
real solar farm to a Primary's power (CAMS injection). Its fit recovered 0.77 of the added farm's
capacity half-hourly and 0.85 on daily sums, where a recovered share is the fitted capacity
divided by the capacity added, and it fitted one capacity per year. [Experiment
45](../00045-pvdis-4-weekly-capacity-asymmetric-l1/result.md) (weekly capacity) found that a weekly
path was too noisy. The probes of [experiment
656](../00656-pv-probe-through-the-mva-fold-at-eight-primaries/result.md) and [experiment
662](../00662-pv-probe-through-the-mva-fold-matched-level-redesign/result.md), a redesign of
experiment 656, read a capacity through a neural network and give no interval. A probe is a linear
read-out of capacity from a trained neural network, and the experiments ran it at Primaries whose
meters report MVA. A quarter is the middle ground between a year and a week: 90 days is long
enough to average the cloud, and short enough to follow a farm that is connected part-way through
the record.

## Design

**The fit reads the signed power itself, with its own demand baseline, and does not read the
residual of experiment 734.** Three designs could supply the observation.

- **Design A (chosen): the cleaned signed power, with a calendar-and-temperature baseline fitted
  together with the solar term.** Every Primary's power is `baseline + slow level - solar`, as in
  [experiment 39](../00039-pv-disaggregation-cams-injection/result.md) and
  `studies.pv_separation`. The fit reads no lagged power, no demand forecast, and no weather
  forecast.
- **Design B (rejected): power minus the stored forecast of experiment 130, as in [experiment
  734](../00734-switching-offset-model-part-1/result.md) (switching offset model).** The stored
  forecast comes from a convex regression model of demand (experiment 130: demand fitted to the
  calendar and the weather, refitted monthly on ERA5 and CAMS) that has already seen the sunshine.
  [Experiment 657](../00657-event-table-with-explicit-solar-term-against-plain-arm/result.md)
  (explicit solar term in the event table) fitted a gain of 0.003 for a solar term on that
  residual, which means that almost no solar output is left in the residual. A fit of capacity to
  the residual would read near zero on the real record, and an injected fleet would still look
  fully recoverable, because the stored forecast never saw the injection. The reference arm R runs
  design B to show both effects.
- **Design C (rejected): daily sums.** Experiment 39 found that daily sums recovered 0.08 more
  (+0.04 to +0.10) than half-hours. The reference arm Ad runs the same Bayesian fit on daily sums,
  and the main arm keeps half-hours because stage 3 needs the half-hourly solar estimate.

**The inference method is a linear-Gaussian regression with a Student-t likelihood, solved by
expectation-maximisation, with a conditional Gaussian posterior at the converged weights.** Three
facts support the choice. First, the solar term is linear in the capacity once the sky, the
orientation mix, the tilt, and the DC:AC ratio are fixed, so the posterior of the capacities, given
the noise weights, is Gaussian. Second, [experiment
734](../00734-switching-offset-model-part-1/result.md) and [experiment
653](../00653-edge-state-event-table-penalised-segmentation/result.md) use a Student-t density
with 4 degrees of freedom for the residual, because switching steps and spikes make the tails
heavy. The fit uses the same density as a scale mixture, and expectation-maximisation updates the
mixture's weights in at most 30 iterations. Third, a CPU solve of about 420 columns on 78,816 rows
takes about 0.8 second for one weighted Gram matrix, so no sampler is needed. Markov chain Monte
Carlo would add run time and a convergence question, and no information about a Gaussian
posterior. The posterior at the converged weights is a conditional posterior, because it holds the
weights and the noise scales fixed, and the page never calls it a Laplace approximation.

**The likelihood and the priors are written out in full.**

- **Likelihood.** The power of half-hour i is `baseline_i + level_q(i) - capacity_q(i) x s_i + e_i`,
  where `s_i` is the unit solar term (below) and `e_i` follows a Student-t density with 4 degrees
  of freedom. The scale of `e_i` is `sigma_day` on daylight half-hours (CAMS irradiance above 5
  W/m2) and `sigma_night` on the others, because the variance of solar power is much larger by
  day. Each expectation-maximisation iteration sets the weight of row i to `(4 + 1) / (4 + (r_i /
  sigma)^2)`, where `r_i` is the residual and `sigma` is the scale of the row's class, and then sets
  each class's `sigma` to the weighted root mean square of the residuals in the class.
- **Why the weights are tested.** Down-weighting large residuals could remove the cloudy
  half-hours that identify the capacity, and bias the capacity down. Gate G1b tests the bias: with
  heavy-tailed noise at the daytime scale, A's recovered share is at least 0.95.
- **Baseline prior.** Every baseline coefficient has a zero-mean Gaussian prior whose precision is
  the ridge penalty of `studies.pv_separation` (`RIDGE_PER_COLUMN` = 1e-3, relative to the mean
  squared column norm). The 18 quarterly levels and the 144 day-type indicators both sum to 1 in
  every row, so the design would be rank deficient by 1. The first quarter has no level, which
  leaves 17 levels, and the ridge penalty keeps the remaining solve well conditioned.
- **Capacity prior.** The AC capacity of quarter q is the capacity of quarter q-1 plus a growth
  step that is Gaussian with a positive mean (0.10 MW a quarter) and a standard deviation tuned
  once on tuning worlds from the grid 0.15, 0.30, and 0.60 MW. The first quarter has a Gaussian
  prior with mean 2 MW and standard deviation 2 MW. Quarters in which the Primary has no usable
  daylight rows (D9's first six) take their capacity from the prior alone and are not scored.

- **Baseline.** One indicator per local half-hour of the day and day type (weekday, Saturday, and
  Sunday), as `studies.pv_separation.baseline_design` builds them; two annual harmonics (a cosine
  and a sine each) for each half-hour; the ERA5 2 m temperature as heating and cooling degree-hours
  around 15.5 degrees Celsius plus the trailing 24-hour and 72-hour means of temperature; 48
  indicators for bank holidays, one per half-hour of the day, read from the `is_holiday` column of
  experiment 6's `features.parquet` (`baseline_design` has no bank-holiday day type); and one flat
  level per quarter. The design has 423 columns: 144 day-type indicators, 192 harmonic columns, 4
  temperature columns, 48 holiday columns, 17 quarterly levels, and 18 solar columns. The level
  absorbs a demand change that is flat across the day, which is the shape of a switching step. A
  daytime-only change in demand reads as a change in solar output. The write-up reports this
  limitation. Two annual harmonics per half-hour may also absorb part of the solar seasonal
  profile, and a cooling-degree term correlates with sunny days. A reviewer estimated that the
  harmonics could cost 25% to 30% of the recovered share. Bar B1 measures the loss on the real
  records, and arm O shows the ceiling.
- **Solar term.** The AC output of one megawatt of panels of a fixed fleet, built by
  `e00737_forward.py` from `studies.pv_physics`: CAMS hourly irradiance split into half-hours, a
  plane-of-array model for east-, south-, and west-facing panels at 25 degrees from horizontal
  with the fixed shares 0.2, 0.6, and 0.2 (the fleet mix that [experiment
  73](../00073-pvdis-6-orientation-mix-and-dc-ac-ratio/result.md) (orientation mix and DC:AC ratio)
  used as its guess), the cell-temperature derate, and clipping at the AC capacity with the fixed
  DC:AC ratio of 1.15. The page does not publish the shares. Experiment 73 found that the shares are
  identified only with an error of 0.20, and that the DC:AC ratio could not be recovered once a
  farm was added to a Primary's power.
- **Cell-temperature derate (new code).** `studies.pv_physics` and `studies.pv_separation` hold
  no cell-temperature model, and `physics_model.py` and `dp_gam.py` use a fixed 25 K rise at 1,000
  W/m2. `e00737_forward.py` adds the derate in this order: plane-of-array irradiance `G`; cell
  temperature from the Faiman model, `T_cell = T_air + G / (u0 + u1 x wind)`, with the coefficients
  `u0` = 25 W/m2/K and `u1` = 6.84 W s/m3/K (the defaults of the `pvlib` library); DC power equal
  to the DC rating times `G / 1000` times `1 - 0.004 (T_cell - 25)`; and then clipping at the AC
  capacity. The DC rating is the AC capacity times the DC:AC ratio. ERA5 is hourly and the power
  is half-hourly, so the temperature and the wind speed at a half-hour's end time are linear
  interpolations between the two neighbouring hourly values. Three unit tests cover the new code.
  The Faiman test checks that `G` = 800 W/m2, 20 degrees Celsius, and 2 m/s give a cell
  temperature of 40.7 degrees. The order test checks that a DC:AC ratio of 1.5 at a cell
  temperature of 45 degrees gives exactly the AC capacity, where clipping before the derate would
  give 0.92 of it. The interpolation test checks a half-hour at the half-hour mark against the mean
  of its two neighbours.
- **Interval.** The 80% interval of each quarter's capacity is the central 80% of the conditional
  Gaussian posterior, with its variance multiplied by `(1 + r) / (1 - r)`, where `r` is the median
  lag-1 autocorrelation of the whitened daylight residuals within each day. Bar B2 tests the
  coverage.
- **Switching events.** The fit treats a switching event as step noise. The Student-t weights and
  the flat quarterly level absorb part of a step, and the page reports how far a 30-day transfer
  moves the capacity (P5, bar B7). The fit masks no event. Masking the called events is stage 4 of
  the pipeline, and the plan does not include it.

**What the fit reads, and what it must not read.** The inputs are listed here so that the review
can check each one.

- **Read:** the cleaned signed power of experiment 327's store (`power/power_<id>.npy`, in which
  the `substation_zero` rule has flagged half-hours that are dropped, and the page reports the
  count); CAMS global horizontal irradiance at the Primary's own coordinates for the hours whose
  reliability is 0.9 or more (other hours are left out of the fit, and are not set to zero); ERA5 2
  m temperature and 10 m wind speed (experiment 6's `features.parquet`, a recorded analysis with no
  forecast-origin column, read from the `temperature_2m` and `wind_speed_10m` columns); the solar
  position, which is a function of time and place; and the calendar.
- **Not read:** any past power as an input to the baseline; any demand forecast; any weather
  forecast, ensemble member, or lead time; the `temperature_mean_24h` and `temperature_mean_72h`
  columns of `features.parquet` (they are not a trailing, centred, or leading rolling mean of
  `temperature_2m`, because the mean gap is 0.9 K, so `e00737_forward.py` builds the means from
  `temperature_2m`); any switching label from NGED's logs (the logs score nothing here); and any
  row from 2026-07-01 on. The weather files run past that date (ERA5 to 2026-07-31, CAMS to
  2026-09-10, and the farm file to 2026-10), so every loader cuts at 2026-07-01 and a test asserts
  the cut.
- **Look-ahead.** The fit is a whole-record smoother, so the capacity of quarter q uses power and
  weather from later quarters. ERA5 and CAMS are published about 5 days late, so the result bears
  on labelling a historical record and not on a live estimate. A past-only sample (6 Primaries, the
  real record cut at 4 quarter ends) measures the loss from cutting the record, and the page
  reports the sample beside the whole-record fit and never in place of it. Gate G3 tests the design
  and the injection.

## The worlds and the arms

**A world is the real record of one Primary with one synthetic fleet added, and the unit of
evidence is the Primary.** The injected output is subtracted from the signed power, because
generation lowers the reading. The fleet's capacity follows one of three growth shapes from zero in
the first quarter to its final AC capacity: a ramp (linear), a step (zero for the first eight
quarters, then the final capacity), and an accelerating path (quadratic). The three final
capacities are 1 MW, 2.5 MW, and 5 MW. [Experiment
649](../00649-pv-capacity-table-season-year-from-stored-probes/result.md)'s probe reads 2.4 MW to
3.9 MW of unmetered solar behind the six Primaries in April to September (whole-summer values), so
the three sizes span from well under that fleet to above it. At the two Primaries with the lowest
midday readings, D5 and D10, the midday reading is about 1 MW to 1.5 MW, so a 5 MW fleet is
several times the reading, and the page says so. A fleet smaller than 1 MW sits below the noise of
a quarter at the smallest Primary and is not planned. The injected fleet's orientation mix, tilt
(15 to 40 degrees from horizontal), and DC:AC ratio (1.0 to 1.5) are drawn per world from a seed,
so the fleet differs from the prior in the way a real fleet will.

**The scored seeds differ from the tuning seeds.** The tuning worlds use seeds 737000 to 737009.
The main worlds use seeds 737101, 737102, and 737103, the matched worlds use 737101 and 737102,
and the cloud, transfer, placebo, and control worlds use 737101.

| World type | Primaries | Sizes (MW) | Shapes | Seeds | Worlds | Counted |
|---|---|---|---|---|---|---|
| Main, mismatched mix, tilt, and ratio | 6 | 1, 2.5, 5 | ramp, step, accelerating | 3 | 162 | C1, C2, B1, B2, B4 |
| Matched (the prior's own mix, tilt, and ratio) | 6 | 2.5, 5 | ramp | 2 | 24 | B4 only |
| Half real metered farm, half physics mixture | 6 | 2.5 | ramp | 2 farms | 12 | B5 |
| Cloud-following load added | 6 | 2.5 | ramp | 1 | 6 | B6 |
| Flat transfer of 1.5 MW for 30 days, three times | 6 | 2.5 | ramp | 1 | 6 | B7 |
| Placebo sky (CAMS of the same day one year later, no injection) | 6 | none | none | 1 | 6 | B3 |
| No injection (the real record) | 6 | none | none | 1 | 6 | Control |

**The page counts 210 injected worlds, 12 uninjected fits, and 120 tuning worlds.** The 210
injected worlds are 162 main, 24 matched, 12 real-farm, 6 cloud, and 6 transfer. The 12 uninjected
fits are 6 placebo and 6 control. The tuning worlds are 10 seeds, times 2 sizes (2 MW and 4 MW),
times 6 Primaries, which is 120 worlds, and the three growth shapes are assigned by the seed
(seed modulo 3: ramp, step, accelerating). The placebo sky uses the CAMS irradiance of the same
day 365 days later, and the last four quarters use 365 days earlier, because the loaders cut at
2026-07-01.

**The three special worlds are built as follows.**

- **Real-farm worlds.** The half-real-farm file is `cleaned_solar_farms.parquet` of experiment 179
  (six farms; columns `time_series_id`, `time`, `power` as float32, `clear_sky_ghi_w_m2`, and
  `drop_reason`; 579,023 rows). The file has no capacity column, so each farm's power is divided by
  `studies.pv_dataset.pv_sites().effective_capacity_mw` to give a per-unit output, and the loader
  cuts it at 2026-07-01. With the six farms sorted by their series identifier as farm 0 to farm
  5, and the six Primaries in ascending D-number order as Primary 0 to Primary 5, Primary k takes
  farms k and (k + 1) modulo 6, one farm for each of the two seeds. In each quarter the world's
  output is half the capacity from the farm's per-unit output and half from the physics mixture. A
  farm half-hour with no finite reading takes the physics mixture instead, and a quarter in which
  the farm has finite readings in under 60% of its daylight half-hours (for example before a
  late-starting farm's first reading) takes the physics mixture alone.
- **Cloud-following load.** Experiment 40's `e00040_arms.confound_shapes(window=...)` returns eight
  shapes, and the world uses the one named `flat`, the day's cloud fraction scaled to a mean of 1,
  at 25% of the fleet's capacity, added as demand (a positive load that rises with cloud). The
  code review fixes the window before any world is built.
- **Transfer.** Three 30-day windows, placed by the seed without overlap, each add a flat 1.5 MW
  to the Primary's signed power, as a direct array addition. A test compares the array addition
  with `apply_events` of experiment 660 on one row, and the code review checks that the two agree.

**The fitted increment of a world is its fit minus the same arm's fit of the uninjected record, and
the increment is the uncut posterior mean.** The real record already holds unmetered solar, so the
fit of the injected world holds the real capacity plus the injected capacity, and the control fit
holds the real capacity alone. With a linear fit and fixed weights the two would differ by the
injection exactly, which [experiment 39](../00039-pv-disaggregation-cams-injection/result.md)
showed makes a known-answer test blind to a demand confound. The heavy-tailed weights make the
difference inexact, and the real noise enters the difference. The cloud-following load and the
transfer worlds are the tests of the confound. The path error E, the recovered share, and the
coverage all use the uncut Gaussian posterior mean, so a fit can report a negative increment.
Arm Q is non-negative by construction and cannot report one, so the page notes Q's one-sided
errors beside its row. The cut at zero applies only to the capacities reported for the real record.

**The 80% interval of the increment has a defined variance.** The increment is the injected
world's posterior mean minus the control's posterior mean, and the two fits' errors are strongly
correlated. The interval takes its standard deviation from the injected fit alone, with the
`(1 + r) / (1 - r)` inflation of the interval's definition applied, which is conservative because
the difference of two positively correlated errors has a smaller variance than either. It is the
central 80% of a Gaussian around the uncut increment, with no truncation before the subtraction.

| Arm | What it is | Role |
|---|---|---|
| A | The Bayesian fit above | Headline |
| F | A with one flat capacity over the 18 quarters (growth step fixed at zero) | Baseline for C1, a sanity gate |
| Qt | A's likelihood and solar term, with an independent flat Gaussian prior on each quarter's capacity (mean 2 MW, standard deviation 5 MW) and the baseline fitted on the whole record | Baseline for C2 |
| Q | The bounded least-squares fit of `studies.pv_separation.separate` run on each quarter alone, with a baseline fitted on that quarter's rows | Reference (the existing method) |
| Ad | A on daily sums of the daylight half-hours and the daily baseline | Reference (experiment 39's advice) |
| Af | A with the DC:AC ratio chosen from 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 by the highest marginal likelihood of the data (the Gaussian evidence at the converged weights), not by the posterior density | Reference (is a fixed ratio worse?) |
| O | A given the injected mix, tilt, and ratio | Reference (the best any fit of this form can do) |
| R | The residual of experiment 734 (power minus the stored forecast) with a flat quarterly level, with the solar term | Reference (design B) |

**Arm Qt carries the counted contrast, and arm Q stays as a reference row, because Q differs from
A in three ways besides the prior.** Q runs `studies.pv_separation.separate`, which is a bounded
least-squares fit and not a Student-t fit, has four free orientation weights (east, south, west,
and a tracker) with a ridge penalty, and fits its baseline on one quarter's rows alone, which is
about 4,400 rows for 336 or more columns, so the annual harmonics are not identified. A contrast
of A against Q would mix the prior's value with those three differences. Qt keeps A's likelihood,
solar term, and whole-record baseline and changes only the prior, so C2 isolates the value of the
growth prior.

## Predictions, counted contrasts, and bars

**A contrast passes the sign test only if A beats the baseline at all six Primaries.** Each
Primary gives one paired observation per contrast, and the table shows what each split means.

| Primaries where A is better than the baseline | Exact one-sided p | Counts at 0.025 |
|---|---|---|
| 6 of 6 | 0.016 | Yes |
| 5 of 6 | 0.109 | No |
| 4 of 6 | 0.344 | No |
| 3 of 6 or fewer | 0.656 or more | No |

**The sign test is deliberately strict, and its power is low.** A true gain that is positive at
each Primary with probability 0.9 passes the test with probability 0.53 (0.9 to the power 6), and a
gain that is positive with probability 0.8 passes with probability 0.26. The worlds of one Primary
share that Primary's real noise, so the worlds do not add independent evidence. The six Primaries
also share their weather, so the observations are not independent either, and the effective number
of observations is below 6. The world-resampled 95% intervals that the page reports ignore the
Primary-to-Primary spread, and decide only between `not supported` and `inconclusive`. The flat fit
F should lose by far more than 0.10, so C1 is nearly certain to pass, and C2 is the contrast that
could fail. The recovered-share bar B1 uses six Primaries too, so B1 resolves a median share only
to about 0.10. The seven bars together pass with probability about 12%, which is why the plan puts
`supported` at about 2%.

**The verdict mapping has one row for every combination of the three inputs, and the first row that
matches applies.** The inputs are C1 (passes, reversed, or fails), C2 (the four states below), and
the bars (all seven pass, or at least one fails). C2 is in one of four states: M (counts and the
median is 0.05 or more), S (counts and the median is above 0 and under 0.05), N (does not count),
or R (Qt beats A at 6 of 6 Primaries). C1 passes when it counts and its median is 0.10 or more,
and is reversed when F beats A at 6 of 6 Primaries.

| Row | Verdict | Condition |
|---|---|---|
| 1 | `invalid` | Any gate G1, G1b, or G2 to G5 fails, or a look-ahead is found in the inputs. |
| 2 | `refuted` | C1 is reversed, or C2 is in state R. |
| 3 | `supported` | C1 passes, C2 is in state M, and all seven bars pass. |
| 4 | `partly supported` | C1 passes, C2 is in state M, and at least one bar fails (the fit follows the path, but its interval, placebo, or robustness is not trustworthy, so the path is usable as a point estimate only). |
| 5 | `small gain` | C1 passes, C2 is in state S, and all seven bars pass. |
| 6 | `not supported` | C1 passes and C2 is in state S with at least one bar failing; or C1 passes and C2 is in state N, with the upper end of C2's world-resampled 95% interval below 0.05. |
| 7 | `inconclusive` | All other cells: C1 fails without being reversed; C2 is in state N with the upper end of the interval at 0.05 or more; fewer than 5 Primaries with a converged fit; or any bar that cannot be read. |

**No verdict supports a claim about the real record's capacity.** Bars B6 and B7 test two
confounds, a load that follows the cloud and a switching transfer, that can move the fitted
capacity of a real Primary. A `supported` verdict says that A recovers a synthetic fleet, and that
two measured confounds move the fit by less than the stated limits. The real-record section below
reports descriptive aggregates, and the page claims no more about the real record than the
aggregates show.

**The reference arms never change the verdict, and the page reports them for the next
experiments.** O sets the ceiling, so a B1 that O also misses means the bar is unreachable, and
the page then says that the bar, and not A, failed (the lesson of [experiment
716's](../00716-msa-unmetered-pv-part-1-history-estimate/result.md) known-answer gate, history
estimate). Af says whether the free DC:AC ratio is worth fitting. Ad says whether daily sums still
recover more. R shows the effect of design B. Q shows how the existing method compares.

**The fleet and the fit share one forward model, so the counted bars B1 to B4 test the fit's
statistics and not its physics.** The fleet's output uses the same CAMS irradiance, ERA5
temperature, Erbs split of sunshine into direct and diffuse parts, plane-of-array code, and
clipping as the fit, and only the orientation mix, the tilt, and the DC:AC ratio differ. A fleet
made this way is easier to recover than a real fleet. Bar B5 therefore counts the 12 real-farm
worlds, whose output comes from a metered farm and not from the forward model, and the verdict
words say "synthetic fleet" in their first line.

## Tuning

**The growth standard deviation is tuned on the tuning worlds, and the tuning shapes are named.**
The 120 tuning worlds use fleets of 2 MW and 4 MW with the shape assigned by the seed. Each of the
three grid values (0.15, 0.30, and 0.60 MW a quarter) is fitted by arm A, and the value with the
lowest median path error over the tuning worlds is kept. A growth standard deviation of 0.60 MW a
quarter cannot follow a 5 MW step in one quarter, which is about 8 standard deviations. The prior
spreads the step over several quarters, so the step shape loses to Q and Qt, and the prediction
P3 already allows for that (C2 of -0.08 to 0.00 on the step shape). The page reports C2 for each
shape beside the pooled C2 that the decision rule counts, and tuning on a mix of shapes picks a
value that suits the ramp and the accelerating path.

## Real record

**The fit is run on the six real records, and the page publishes only aggregates.** The aggregates
are the median over the six Primaries of the ratio of the last year's mean capacity to the first
year's; the share of Primaries whose capacity rises between the first and the last year with
posterior probability 0.9 or more; the capacity fitted on the placebo sky as a share of the real
fit's (bar B3); the capacity of arm R as a share of A's; the number of the six Primaries whose
April-to-September mean capacity has an 80% interval that overlaps the season-year table of
[experiment 649](../00649-pv-capacity-table-season-year-from-stored-probes/result.md) (capacity
from stored probes), as a count k of 6; and the share of quarters whose 80% interval is wider than
half the median capacity. One figure shows the median over the six Primaries of the capacity path
as a multiple of each Primary's own mean, with the range over the six. **No per-Primary capacity
in megawatts, and no Primary's own path, appears on the page.** The six Primaries are substations,
so the rule on metered generators does not cover them, but the page follows the convention of
experiments 716 and 729 and gives estimates of unmetered solar anonymised. The private tables hold
every number.

## Gates

**Six gates must pass before any verdict counts, and a failed gate gives `invalid`.**

- **G1, the noise-free recovery.** On a synthetic series that holds only a baseline, an injected
  fleet of the prior's own mix and ratio, and Gaussian noise at the Primary's own scale, A
  recovers each quarter's capacity to within 3% in the mean over 100 draws, and its 80% interval
  covers the truth between 0.75 and 0.85 of the cells.
- **G1b, the heavy-tailed recovery.** On the same series with heavy-tailed noise (a Student-t
  density with 4 degrees of freedom) at the daytime scale, A's recovered share is at least 0.95 in
  the mean over 100 draws. The gate tests whether the Student-t weights bias the capacity down.
- **G2, the injection is exact and nonzero.** The added output equals the fleet formula to a
  relative tolerance of 1e-9. In every quarter, the injected output is nonzero in at least 99% of
  the half-hours whose CAMS irradiance exceeds 50 W/m2 and exactly zero when the sun is below 5
  degrees of elevation. The check would have caught the silent solar output of [experiment
  729](../00729-msa-d8-neural-untangler-synthetic-known-answers/result.md) (invalid), where half
  the simulated solar farms produced nothing.
- **G3, no look-ahead of the forbidden kind.** The injected output is a function of CAMS, ERA5,
  the sun's position, and the seed only, and the test builds a copy of the same Primary with
  shuffled power and checks that the injected output and the design matrix are identical. The
  temperature means are trailing means built from `temperature_2m`, the test checks each mean
  against a direct calculation from values at or before the half-hour's end, and the inputs hold no
  forecast column.
- **G4, the timing convention.** Three conventions meet, and a one-slot error is exactly 30
  minutes, which a test of the peak against solar noon cannot see (CAMS is hourly). CAMS labels
  the end of the hour. `half_hour_sun_and_sky` returns half-hour end stamps. The power index `i` of
  experiment 327's store is the half-hour starting at the origin plus 30 minutes times `i` (the
  docstring of experiment 660's generator), whereas `baseline_design` takes end times. The code
  review brief pins the mapping from index to end time. The gate runs on the real half-hourly
  series: for each Primary, it compares the mean daytime dip of real summer power minus real winter
  power (April to September against October to March, by half-hour of the day) with the model's
  unit solar curve for the same two seasons, shifted by -1, 0, and +1 half-hour, and requires the
  best shift (the one with the highest correlation over the 48 half-hours) to be 0 at each Primary
  and in the pooled mean. The gate also requires that the real nights carry no CAMS irradiance,
  and the test calls the real loader on a small fixture shaped like the real file (the second
  lesson of experiment 729).
- **G5, convergence.** The expectation-maximisation fit converges in 30 iterations or fewer in 99%
  of the fits, and no fit ends with every quarter's capacity at its bound.

## Synthetic world gallery

**The gallery shows synthetic quantities only, and the figures are drawn before the first scored
fit.** The 11 figures use an axis of hours or quarters and never a calendar date, and no figure
shows a metered series, the real power of a Primary, or a metered generator. Stage 0 draws the
forward-model figures (3) and the injected-fleet figures (3). Stage 1 draws the four posterior
figures on the tuning seeds, before the first scored world is fitted. The real-data aggregate
figure (1) comes after the real-record fits.

- **The forward model (3 figures).** One megawatt of AC capacity at a clear and a cloudy sky for
  each orientation; the clipping at the DC:AC ratios 1.0, 1.15, and 1.5; and the cell-temperature
  derate at three air temperatures.
- **The injected fleets (3 figures).** The nine growth paths (3 shapes by 3 sizes) in MW per
  quarter; the output of the 5 MW ramp fleet over one week in a bright and a dull month, drawn
  alone for the prior's fleet and for two mismatched fleets; and the capacity prior, with 20 paths
  drawn from it.
- **The posterior on injected worlds (4 figures).** Drawn from the tuning seeds only: for the
  three shapes at 2.5 MW, the injected capacity per quarter against the posterior mean and the 80%
  interval of A, F, and Qt, shown as capacity (a synthetic quantity, since the fit is of the
  increment); and one world each for the cloud-following load and the transfer.
- **Real-data aggregates (1 figure).** The median capacity path over the six Primaries as a
  multiple of each Primary's own mean, with the range, as the section above says.

**The smoke check on the data files** asserts G2 on every world file before a fit reads it, that
each of the 210 injected worlds and the 12 uninjected fits is present, and that no injected series
is all zero. The write-up repeats the gallery with the results added.

## Stages and compute

**The compute estimate is about 17 CPU-hours at one thread per worker, and the stop is at 26
CPU-hours.** The ML experiment manager allocates the cores. A timing check of one weighted Gram
matrix of 78,816 rows by 400 columns on one thread took 0.78 second, and a solve of the 400-column
system took 0.02 second. A fit of 30 expectation-maximisation iterations therefore takes about 24
seconds, and the estimate assumes all 30 iterations. Af warm-starts each of its six ratios from
A's weights and is assumed to need 10 iterations each, or about 8 seconds a fit. A world needs 6
cold fits (A, F, Qt, Q, O, and R, at 24 seconds each), the 6 warm fits of Af, and the daily-sum
fit Ad, which takes about 1 second, so a world takes about 195 seconds, and 222 worlds (210
injected and 12 uninjected) take about 12.0 CPU-hours. The other stages follow the table. The six
Primaries are independent, so up to 6 workers can run side by side, and the manager decides how
many.

| Stage | Work | Expected CPU-hours |
|---|---|---|
| 0 | Build the worlds, run G2 to G5, and draw the forward-model and fleet figures | 0.5 |
| 0b | Run G1 and G1b (100 draws each, one fit of 24 seconds per draw) | 1.3 |
| 1 | Tune the growth standard deviation (120 tuning worlds, three grid values, arm A only, 24 seconds a fit) and draw the posterior figures | 2.4 |
| 2 | Fit the 222 scored worlds and uninjected fits with all arms (195 seconds each) | 12.0 |
| 3 | Real-record fits (6 Primaries, 195 seconds each, 0.3), the past-only sample (24 fits of A, 0.2), matching to experiment 649, scores, and figures (0.2) | 0.7 |
| Total | | 16.9 |

**Peak memory is about 1 GB per worker, and the scope cap is 2.5 GB per worker.** The design
matrix of 78,816 rows by 423 columns in 64-bit floats is 267 MB, and the weighted copy and the Gram
matrix take about as much again. Each stage writes one file per world, atomically, with a
per-phase time log. The first world's measured time replaces the estimates above, and the session
asks the ML experiment manager before passing the 26 CPU-hour stop.

## Run order and reviews

**No stage runs before its review marker exists.** `e00737_phases.py` checks that the marker for
the next stage exists before it starts the stage, and `e00737_run.py` is the runner that calls it.

1. A fresh Sonnet agent that did not write the plan reviews the plan (a fresh Opus agent, if the
   manager's brief names one). A fresh Sonnet agent that did not write the code reviews the code:
   leakage, the sign of the injection, the half-hour convention of CAMS and the mapping from the
   power index to the end time, the match with the baseline of `studies.pv_separation`, the
   posterior's variance, the direct array addition of the transfer, and the 2026-07-01 cut in every
   loader. A second fresh Sonnet agent reviews the fixed code if the first review flags many
   problems.
2. A real-environment smoke test runs one world on one Primary, with a scratch output folder
   (`.claude/scratch/`) and nothing written to the experiment's data folder, through every arm and
   the scoring code.
3. The ML experiment manager receives a "fits starting" line before stage 2, with the fit count,
   the projected CPU-hours, the memory caps, and the cores. The runs happen inside `systemd-run
   --user --scope` with the memory cap above and one thread for each numerical library, after the
   manager approves the workstation run. `e00737_run.py` writes a per-phase time log with each
   job's start and end times.
4. A Sonnet agent checks the write-up against the printed report, a second Sonnet agent runs the
   prose review, and the anonymisation check below runs before the pull request leaves draft.

## Standing practices, advice, and privacy

- **Followed:** cleaned data (the dropped half-hours are counted per Primary); a per-phase time log;
  a synthetic world gallery and a smoke check; the stored forecast of experiment 327 read only for
  arm R; the real-environment smoke test; no stage past its review gate; the hash of every edited
  reviewed script recorded; a placebo beside every contrast; the tuning of one value only, on
  separate seeds, with the caveat that the tuning and scored worlds share the real history.
- **Not applicable:** the D15 and D18 cut (neither is among the six); monthly refits and model
  saving (the fit has no stored coefficients to save); training-loss plots (the fit is a
  closed-form solve with an expectation-maximisation loop of at most 30 iterations); tail coverage
  and the load-shift scenarios (no forecast is scored); the leaderboard (no forecast).
- **Advice followed:**
    - Treat injection as a test of the shape and the irradiance only, and add noise and a cloud
      load (experiments 39 and 73: the world with the real noise, and the cloud world).
    - Use CAMS at the substation and leave out the MVA Primaries (experiment 39).
    - Use one capacity per fixed period, and score against a flat path (experiment 45: the
      flat-path arm F).
    - Check the last period against the final fit (experiment 45: the past-only sample, where the
      latest week had 3.6 times the path error).
    - Keep the DC:AC ratio fixed at 1.15, and publish no orientation shares (experiment 73).
    - Inject at tilts other than the fitted one (experiment 73: tilt 15 to 40 degrees).
    - Do not add a solar term to experiment 130's residual (experiment 657: design B).
    - Smoke-check that each simulated component is nonzero, and call the real loader in a test
      (experiment 729).
    - Set the bar so that an oracle can reach it (experiment 716: arm O).
    - Say when an injected fleet is large against a Primary's midday reading (experiment 662: at D5
      and D10 the midday reading is about 1 MW to 1.5 MW, so the 5 MW fleet is several times that
      reading, and the page says so).
- **Advice departed from.** Experiment 39 advises fitting daily sums, which recovered 0.08 more.
  The main arm keeps half-hours, because stage 3 needs a half-hourly solar estimate, and Ad reports
  the daily-sum result beside the main arm. Experiment 39 also advises dividing a fitted capacity
  by its recovered share (0.77 or 0.85), and this experiment does not, because the page reports
  the recovered share and the interval instead. Experiment 45 advises a bootstrap over residual
  blocks only with a warning on coverage, and this plan uses the Gaussian posterior and tests its
  coverage. The advice of experiment 732 to test a solar estimate on real days with no solar farm
  cannot be followed, because every one of the six Primaries has solar behind it, and the placebo
  sky is the substitute.
- **Privacy.** The page and every figure hold aggregates and synthetic quantities only. No metered
  generator's name, identifier, size, dates, register values, or time series is published, and the
  half-real-farm worlds report shares recovered over the six farms with no farm labelled. No
  engine size or engine date appears on any published page. Real capacities appear only as ratios
  and counts, under the anonymised D codes. The anonymisation check runs on the page, the figures,
  the pull request body, and the take-home message before the pull request leaves draft.
- **Not planned.** The MVA-metered Primaries (stage 2 at those Primaries needs the sign neural
  network and is the next experiment), masking called switching events (stage 4), the feedback
  round, the residual report, D18 as an unscored reference, and the use of the capacity in a
  forecaster (experiments 716 and 717 found no detectable forecast gain from an unmetered-solar
  estimate).

## Details the code review checks

- **The farm file's timestamp convention.** The code review confirms how the farm file labels its
  half-hours and aligns the farms to the six Primaries' half-hours under gate G4's mapping.
- **The window of the cloud shape.** The code review fixes the window that `confound_shapes`
  takes, before any world is built.
- **The transfer.** The code review runs the comparison with `apply_events` on one row.

## Usable half-hours per quarter

**The scored quarters are those with at least 60% of the quarter's daylight half-hours usable, and
the threshold of 2,000 half-hours excluded half the record.** A usable half-hour has CAMS
reliability of 0.9 or more, irradiance above 5 W/m2, and a finite power reading. A quarter has
between 1,530 and 2,744 daylight half-hours, with the winter quarters (the fourth quarter of a
calendar year and the first) at the low end, so the 2,000 threshold excluded nine of the 18
quarters at D5, D10, D11, D14, and D16 and 13 of 18 at D9 (D9's first six quarters have no power).
The counts below read the CAMS and power files without changing them, on experiment 653's grid,
with each half-hour assigned to a calendar quarter by its end time in UTC.

| Primary | Quarters with data | Quarters scored at 60% | Winter quarters scored (of 9) | Usable half-hours in scored quarters | Quarters below 2,000 |
|---|---|---|---|---|---|
| D5 | 18 | 18 | 9 | 1,282 to 2,470 | 9 |
| D9 | 12 | 11 | 6 | 1,270 to 2,475 | 13 |
| D10 | 18 | 18 | 9 | 1,277 to 2,474 | 9 |
| D11 | 18 | 18 | 9 | 1,275 to 2,476 | 9 |
| D14 | 18 | 18 | 9 | 1,283 to 2,470 | 9 |
| D16 | 18 | 18 | 9 | 1,284 to 2,474 | 9 |

**The share of daylight half-hours that are usable is 0.82 to 0.92 in every quarter that has
data, except one quarter at D9.** D9's seventh quarter has 857 usable of 2,581 daylight half-hours
(0.33) and is left out, and D9's first six quarters have no usable half-hours. Every winter
quarter of the five full-record Primaries is scored, and D9 scores 6 of its 9 winter quarters. A
short absolute threshold of about 600 half-hours would give 18 scored quarters at each of the
five and 12 at D9, and the share rule is preferred because it also scales with the season. The
page reports the count of scored quarters per Primary, and the dropped half-hours (the
`substation_zero` rule, and the unreliable CAMS hours) per Primary.

## Method

### Architecture

- The fit is a Bayesian linear regression with a Student-t likelihood (4 degrees of freedom), whose
  weights and daylight and night scales come from at most 30 expectation-maximisation iterations,
  and a conditional Gaussian posterior at the converged weights.
- The inputs are the cleaned signed power, the CAMS irradiance (reliability 0.9 or more), the ERA5
  2 m temperature and 10 m wind speed, the solar position, and the calendar. Nothing else enters.
- The solar term is `studies.pv_physics`' plane-of-array model for three orientations at the fixed
  shares and 25 degrees from horizontal, with the Faiman cell-temperature derate (new code in
  `e00737_forward.py`, applied before the clipping) and clipping at the DC:AC ratio 1.15.
- The capacity of each of the 18 quarters follows a random walk with a positive drift (0.10 MW a
  quarter) and a tuned standard deviation, with a Gaussian prior on the first quarter. Each quarter
  after the first also has a flat level, and the baseline has a ridge prior.
- Arm F sets the growth step to zero, Qt gives each quarter an independent flat prior, Q fits each
  quarter alone with the bounded least squares of `studies.pv_separation`, Ad uses daily sums, Af
  selects the DC:AC ratio by marginal likelihood, O reads the injected fleet's parameters, and R
  replaces the power by experiment 734's residual.

### Training regime

- Nothing is trained. One value, the growth standard deviation, is chosen from three on 120 tuning
  worlds (seeds 737000 to 737009, with fleets of 2 MW and 4 MW on all six Primaries, the shape
  assigned by the seed). The tuning and the scored worlds share the real history, so the reported
  recovery is optimistic by at most the effect of one tuned value.
- The injected fleets are drawn by the experiment's own code from the seed of each world: the
  growth shape and the final capacity, the orientation mix (a Dirichlet draw around the fleet mix,
  or the prior's mix for the matched worlds), the tilt, and the DC:AC ratio.
- Compute: CPU only, about 17 CPU-hours at one thread (30 iterations assumed for each cold fit,
  10 for each warm fit of Af), a stop at 26, and 0 GPU-process-hours.

### Measurement

- The scored rows are the quarters from 2022 to mid-2026 for each Primary with at least 60% of
  the quarter's daylight half-hours usable (the table above gives each Primary's count), with the
  control fit subtracted from each injected fit.
- The metrics are the path error E, the recovered share, and the coverage of the 80% interval,
  each per world, then summarised per Primary by the median over its worlds. All three use the
  uncut posterior mean.
- The contrasts C1 and C2 use the exact one-sided sign test over the six Primaries at 0.025, with
  world-resampled 95% intervals (10,000 draws) reported beside them, and the verdict is read as
  descriptive evidence because the Primaries share their weather. Every number before the final
  scoring is interim.
- The path error divides by the mean injected capacity of the world, so the path error is a share,
  and the recovered share divides by the injected capacities' sum. A capacity is an AC capacity in
  megawatts (the fleet's 99th-percentile half-hourly AC output equals the AC capacity at a DC:AC
  ratio of 1.15).

## Files

- `ml-experiments/00737-bayesian-solar-capacity-part-1/` (to be written): `e00737_forward.py` builds
  the solar term from `studies.pv_physics`, adds the Faiman derate, and builds the temperature
  means from `temperature_2m`; `e00737_inputs.py` loads the power, CAMS, and ERA5, cuts every
  loader at 2026-07-01, and builds the baseline with `studies.pv_separation.baseline_design` and
  the holiday columns; `e00737_fit.py` holds the Student-t expectation-maximisation, the Gaussian
  posterior, and the eight arms; `e00737_worlds.py` builds the injected, placebo, cloud-load,
  real-farm, and transfer worlds and runs G2 and G3; `e00737_gates.py` runs G1, G1b, G4, and G5;
  `e00737_score.py` computes E, the recovered share, the coverage, the sign tests, and the verdict
  row; `e00737_run.py` runs the stages and writes the per-phase time log;
  `e00737_phases.py` checks that the review marker of each stage exists before the stage starts;
  `e00737_figures.py` draws the gallery and the real-record aggregate figure.
- Read only: `ml-experiments/00039-pv-disaggregation-cams-injection/` (the farm loader),
  `ml-experiments/00653-edge-state-event-table-penalised-segmentation/` (the grid and the panel),
  `ml-experiments/00660-shared-generator-of-injected-load-transfer-events/` (`apply_events`, for
  the transfer test), `ml-experiments/00040-pv-capacity-error-bars/` (the `flat` cloud shape), and
  `studies.pv_physics` and `studies.pv_separation`.
- Read: `data/ml-experiments/00327-lsc-label-free-detect-mask-train-pipeline/power/power_<id>.npy`
  (cleaned signed power) and `.../forecast/C/forecast_<id>.npy` (arm R only);
  `data/studies/downloads/reanalysis/CAMS/demand_points/cams_demand_points.parquet` (columns `site`,
  `time`, `ghi_w_m2`, `reliability`);
  `data/ml-experiments/00006-stage1a-demand-input-feature-sweep/inputs/features.parquet` (the ERA5
  `temperature_2m` and `wind_speed_10m`, and the `is_holiday` flag);
  `data/ml-experiments/00649-pv-capacity-table-season-year-from-stored-probes/work/capacity_table.parquet`
  (the reference table);
  `data/ml-experiments/00179-metered-solar-farm-exploration/cleaned_solar_farms.parquet` (the
  half-real-farm worlds).
- Written: `data/ml-experiments/00737-bayesian-solar-capacity-part-1/` (never committed): world
  files, fit and posterior tables, score tables, the time log, and the review markers.
