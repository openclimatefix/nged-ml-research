---
id: "00711"
slug: ens-total-cloud-cover-for-primary-demand-and-solar-farms
title: >-
  Does the ECMWF ensemble's forecast of total cloud cover improve the Primary demand forecast or the
  forecast of NGED's metered solar farms?
theme: weather-products-and-blending
also_relevant_to:
  - demand-inputs
  - metered-solar-farms
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores about 950 fitted neural networks and their per-row forecasts, and it builds
  new cloud summary tables; (2) more than one design defensibly tests the hypothesis (the cloud
  summary, the placebo, and the handling of a 19-month gap in the cloud history); (3) it copies
  training code from experiments 707 and 666, whose callers cannot be named without searching; (4)
  it touches no serving path or degradation rule; (5) it changes no contract. Full set of reviews:
  plan, code before each script's first run, results, write-up, prose, and the audit triggers.
hypothesis: >-
  Giving the neural network the mean and the spread, across the 51 ECMWF ENS members, of total
  cloud cover from the run issued at or before the forecast origin lowers the CRPS of (a) the
  Primary demand forecast and (b) the metered solar farm forecast by more than a placebo cloud
  series with the same missing-value pattern does.
prediction: >-
  Written before any fit, and all numbers interim. For Primary demand, the real-minus-placebo
  contrast is between -0.03 and +0.01 points of CRPS at both groups of Primaries, so the 0.02
  threshold is not cleared; experiment 559's arm A11c found +0.014 (-0.023 to +0.050) at the eight
  MVA-only Primaries and +0.033 (-0.025 to +0.099) at the six signed-MW Primaries. For the solar
  farms, the contrast is between -0.03 and 0 points, about 1% of the 2.91 points of daylight CRPS
  of experiment 666's PLP network, because ENS cloud comes from the same weather model as the ENS
  shortwave radiation that PLP already reads.
decision_rule: >-
  Each planned reading (C1, C2, C3, S1, S2, S3) is read at 1 - 0.05/9 = 99.44% on its interval
  [lower, upper], and the decisive interval is the wider of the month-block bootstrap interval and
  the t interval on the monthly differences. The first test that holds decides. Lower above 0:
  refuted. Upper below -0.02: supported. Upper below 0: small gain. Lower at or above -0.02 and
  upper at or above 0: not supported (a gain of 0.02 is excluded). Otherwise inconclusive. The
  threshold of 0.02 points is the minimum gain that both protocols set for a new input, because
  the column adds one variable to ingest, validate, and serve, plus a missing-value rule for the
  19-month gap. A target reads as the better of its two summary readings, ranked supported, small
  gain, inconclusive, not supported, refuted. Partly supported: the two summary readings at a
  group are inconclusive or not supported and C3 is supported there. A failed positive control
  makes both Primary demand groups inconclusive.
benchmark: >-
  The matching placebo arm of each real arm; stored `R_F` and `A_F` of experiment 707, stored PLP of
  experiment 666, climatology, the manual heuristic, and the stored XGBoost-based forecasts BG2
  (experiment 87) and S0M (experiment 389) as reference rows only.
row_set_id: >-
  fold_a_daily_gap_v1 (Primary demand); backfill_daily_2023_04_to_2024_03_v1 (Primary demand
  replicate); solar_daily_1200_2025_12_to_2026_06_v1 (solar farms)
metric: >-
  CRPS (twice the mean pinball loss over 11 quantile levels) in points of effective capacity at lead
  days 1 to 10, with the normalised MAE of the median beside it; q01, q05, q95, and q99 coverage and
  pinball loss; cold winter nights; avoidable-curtailment cost.
production_status: not-for-production
adopted_in: none
---
**Status: planned. The sections below are the plan; the finished page replaces them. Every number
on this page is interim: each is quoted from an earlier experiment, read from a log, or estimated,
and none is a result of this experiment.**

This experiment asks whether a forecast of cloud cover helps two forecasts that already use the
weather. The first is the forecast of the power that 16 Primary substations draw. The second is the
forecast of the output of six metered solar farms. The weather forecast comes from the European
Centre for Medium-Range Weather Forecasts (ECMWF), which runs 51 versions of each forecast (an
ensemble) with slightly different starting conditions. We give each neural network the mean and the
spread of the 51 cloud-cover forecasts. We then compare it with a twin network that receives the
cloud forecast of a different day, with exactly the same gaps. A gain over the twin is a gain from
the cloud forecast itself.

**The maintainer asked for this experiment by name on 2026-11-19.** The request was: "when the ENS
total cloud cover data lands, pls run an experiment to see if it helps with primary demand
forecasting and/or NGED solar farm forecasting". That request is the approval that the stop rule
below requires.

## Summary of arms and contrasts

**The cloud variable is `total_cloud_cover_atmosphere`, from 0% to 100%.** The experiment adds it to
two neural networks. The first is the pooled mixture density network of experiment 707 for Primary
demand, in the arm that experiment calls `R_F`. The second is the solar network PLP of experiments
659 and 666 for the six metered solar farms. The deciding contrast subtracts a placebo arm that
holds another day's cloud with the same missing-value pattern, so a gain cannot come from the
missing-value flag acting as a date.

**The 19-month gap in the cloud history gets three answers, and each answer has its own arm.** The
cloud history has a gap from 2024-04-01 to 2025-11-20. The headline arms keep every training row
and mask the gap with a flag. A same-row pair drops the gap rows from both the cloud arm and its
reference. A gap-free replicate for Primary demand scores 2023-04 to 2024-03, when every training
and scoring run carries cloud.

**The experiment is `kind: other` and adopts nothing.** Neither base network is a logged research
default. The Primary demand research default 3 is the GAM stack, and Metered solar farms research
default 1 is a three-way blend. A supported result is a candidate for a later `change` experiment
on the port target (the Candidate `P7+D50` for Primary demand, and PLP for the solar farms).

**No new XGBoost model is fitted.** The stored XGBoost-based forecasts BG2 and S0M appear as
reference rows only, as the maintainer's decision of 2026-11-02 requires.

## 1. Hypothesis, prediction, and the smallest gain worth porting

**The hypothesis and the prediction are in the front matter and are written before any fit.** The
prediction rests on three earlier results. Experiment 22 found that replacing every ENS weather
input with ERA5 at serving gained only 0.10 points. Experiment 559's arm A11c found small positive
gains with intervals that contain zero. For the solar farms, UKV irradiance and cloud, from a
different weather model, gained 1.1% in experiment 248.

**The smallest gain worth porting is 0.02 points of CRPS for both targets.** Both evaluation
protocols set 0.02 points as the minimum gain for a new input. A threshold of zero would be too
low, because the variable costs one more column to ingest, validate, and serve.

**The cloud-scored folds cannot be expected to resolve 0.02 points at the six signed-MW Primaries.**
The minimum detectable effect, estimated in section 7, is larger than 0.02 there. A reading at the
six is therefore expected to come out `inconclusive` or `small gain`, which is why the gap-free
replicate contrast C3 is planned.

## 2. Stop rule and the maintainer's approval

**The rule says the fourth experiment on one idea after three `not supported` or `refuted` verdicts
needs the maintainer's approval, and is scored at every Primary with no screening arm.** The text is
in `evaluation-protocol-demand.md` ("Stop rule for blind alleys"). A `supported` or `small gain`
verdict on the idea resets the count. The dates below are the commit dates of each `result.md`.

**Count on ENS total cloud cover itself: 0.** The direct cloud tests are experiment 559
(`partly supported`), experiment 57 (`inconclusive`), and experiments 229, 246, and 248
(`small gain`, UKV cloud). Experiment 568 is `not supported`, but its verdict is on the attention
mixer; if the manager counts experiment 568, the count is 1.

**Count on irradiance-like inputs to the Primary demand forecast: 3 after the reset.** The
experiments 187, 193, and 237 were `supported` on 2026-10-22 and reset the count to 0. Three
experiments then returned `not supported`.

| Order | Experiment | Theme | Verdict | Date | Running count |
|---|---|---|---|---|---|
| 1 | 9 | weather-products-and-blending | `refuted` | 2026-10-14 | 1 |
| 2 | 22 | production-reference-and-best-guess | `refuted` | 2026-10-15 | 2 |
| 3 | 50 | weather-products-and-blending | `refuted` | 2026-10-16 | 3 |
| 4 | 187, 193, 237 | differentiable-physics themes | `supported` | 2026-10-22 | reset to 0 |
| 5 | 419 | production-reference-and-best-guess | `not supported` | 2026-10-31 | 1 |
| 6 | 437 | demand-inputs | `not supported` | 2026-10-31 | 2 |
| 7 | 569 | demand-inputs | `not supported` | 2026-11-05 | 3 |

**Experiments 703 and 704 do not reset the count.** Both were `supported` on 2026-11-18, and each
supports removing two irradiance columns, not adding one. If the manager rules that the GAM stack's
solar summary (experiments 187, 193, and 237) is a different idea from an input to a neural network,
no reset applies, and the count is 3 or more on either reading.

**Count on irradiance-like inputs to the solar farm forecast: 2 strict (experiments 272 and 292),
5 broad.** The `small gain` verdicts of experiments 229, 246, and 248 reset the count on 2026-10-22.
Experiment 272 (`not supported`, extra ensemble variables) and experiment 292 (`not supported`,
neighbour-cell irradiance) followed on 2026-10-23. The broad reading adds experiments 275, 430, and
443, which change the training data or the design and add no input.

**Consequence.** The rule applies to Primary demand, and the plan applies it to the solar farms too.
Every Primary and every farm is scored on every fold from the first run, and no four-month screen
is run. The standing practice "screen on a few folds first" is therefore not followed.

**Stop conditions inside the experiment.** Stop and report to the manager if the gate fails, if the
era check fails and the manager does not approve an era indicator, if any origin assertion fails,
or if the first fit's memory exceeds the cap.

## 3. Base networks, rows, and folds

### Primary demand

- **Network.** Experiment 707's arm `R_F`: the pooled mixture density network of experiment 425,
  widened by 1.15, with no ERA5 and CAMS pre-training, an `S_ens` stage on every ENS row from
  2021-07-20 to the fold's cut, and a fine-tune on the last 730 days.
- **Why `R_F`.** The arm scores 2.821 points at the six, against 2.820 for `R_F+flags` and 2.840 for
  the pre-trained `A_F`, and is 0.041 points behind `A_F` at the eight. Choosing `R_F` keeps ERA5
  cloud, a fraction from a different product, out of the experiment.
- **Rows.** Row set `fold_a_daily_gap_v1`: every daily 00 UTC run, origin 09:00 UTC, lead days 1 to
  10, all 16 Primaries. The six, the eight (with and without D8), and D15 and D18 (report-only) are
  scored separately and never pooled.
- **Folds.** The 12 monthly folds 2025-07 to 2026-06 of experiment 707. Fold M trains on rows known
  3.5 hours before the first origin of month M and scores the runs of month M. No valid time on or
  after 2026-07-01 is scored, because `experiment_store.write_predictions` refuses it.
- **Cloud-scored folds.** The seven folds 2025-12 to 2026-06, whose scored runs all carry cloud once
  the re-write of the Delta table lands. The planned contrasts C1 and C2 read these seven folds only.
- **Gap-scored folds.** The four folds 2025-07 to 2025-10, whose scored runs carry no cloud. The
  cloud arms see the flag set on every scored row there, so real minus placebo should be zero, and
  the folds are a built-in negative control. Fold 2025-11 mixes both kinds of run (cloud from
  2025-11-21), is reported on its own, and enters no contrast.
- **Gap-free replicate.** A second row set, `backfill_daily_2023_04_to_2024_03_v1`: the same
  recipe, 12 monthly folds 2023-04 to 2024-03, training from 2021-07-20. Every training and scoring
  run carries cloud, so the replicate needs no flag and no mask. D9's first reading in the cleaned
  power table is 2023-11-21, so the replicate scores the six without D9 and labels the group "five
  signed-MW Primaries". D15 and D18 have no replicate.

### Solar farms

- **Network.** PLP of experiments 659 and 666: 18 inputs, two hidden layers of 64 units, a
  three-component Gaussian mixture centred on the mean physics-lite power, 20 epochs, no farm
  column.
- **Two departures from experiment 666, applied to every arm.** The refit is monthly, as the
  standing practice asks, and the training window starts at the first back-filled run (2020-10-15)
  instead of 2024-04-01. The stored quarterly PLP forecasts of experiment 666 stay as a reference
  row.
- **Rows.** A new row set, `solar_daily_1200_2025_12_to_2026_06_v1`: every daily 00 UTC run from
  2025-12-01, origin 12:00 UTC, daylight half-hours, leads 12 hours to under 348 hours from the run,
  valid times before 2026-07-01.
- **Headline band.** The headline covers days 1 to 10 (12 to under 252 hours). The solar protocol
  sets 12 to 36 hours as its headline band, so the plan departs from the protocol, because NGED
  care about days 1 to 10. The 12-to-36-hour reading (S3) is reported beside S1 in the headline
  caveat.
- **Folds.** Seven monthly folds 2025-12 to 2026-06. A refit at the first of month M trains on
  daylight half-hours whose run is before the refit and whose valid time is at least 1 day before
  it, and forecasts the origins of month M.
- **Farms with short histories.** The farms have readings from different dates. The training window
  of each farm starts at its first usable run, and the page publishes no per-label start date,
  first-data date, or commissioning date.

**Every arm of a contrast scores exactly the same rows.** Rows are dropped only by rules that read
the target or the calendar: a flagged `substation_zero` half-hour in the target or a lag, a D15 or
D18 row before 2025-03-01, a Primary's first usable date, and the run-date windows of section 4. No
rule reads a cloud value. The scorer asserts equal row keys across arms before any contrast.

## 4. Cloud summaries and the handling of the gap

**Three summaries of the 51 members are built per row, all from the run named in section 7.**

- **T1, mean and spread:** the ensemble mean and the population standard deviation of total cloud
  cover, divided by 100 (two columns).
- **T2, clear and overcast shares:** T1 plus the share of members at or below 20% cloud and the
  share at or above 80% (four columns). Cloud cover piles up near 0% and 100%, so a mean of 50% can
  describe 51 half-cloudy members or a split between clear and overcast members, and the two shares
  separate those cases.
- **T3, daylight mean (exploratory, Primary demand only):** T1 plus the mean of the ensemble-mean
  cloud over the daylight half-hours of the row's local valid day (three columns). T3 is in no
  planned contrast.

**Headline handling: train on every row, and mask the gap.** A row whose run falls in 2024-04-01 to
2025-11-20, or is the upstream gap of 2024-01-03, gets 0 in every standardised cloud column and 1 in
one `cloud_missing` column. Standardisation uses the fold's training rows with `cloud_missing = 0`.
The rule reads the run date, never the cloud value.

**The flag-only arm measures what the flag alone does.** The arm holds `cloud_missing` and no cloud
column. Flag-only minus the no-cloud reference is the date-proxy effect that experiment 568 warned
about, and is reported with its interval.

**Same-row sensitivity: drop the gap from both arms.** A cloud arm and its reference train only on
rows whose run is outside 2024-04-01 to 2025-11-20, with no flag. Both arms lose 19 months of recent
power history, so only the pair's difference is read, never its absolute scores.

**An era check runs before any fit.** The back-filled runs and the recent runs come from two
ingests, and ECMWF changed the ENS resolution on 2023-06-27. For each of the 16 Primaries and 6 farm
cells, compute the monthly mean of ENS cloud minus ERA5 `total_cloud_cover` (times 100) at leads 12
to 36 hours, and the monthly ENS spread. The ERA5 source is the column `total_cloud_cover` of
`ml-experiments/00607-bakeoff-extend-cached-inputs-to-2026-10-31/work/era5_wind_components.parquet`,
a fraction for 22 series from 2019-09-01 to 2026-10-31. Compare the same calendar months across
eras (December to March of 2021-22 to 2023-24 against December 2025 to March 2026), because cloud
bias is seasonal. A step at 2023-06-27, or between those two groups of months, larger than 3
standard deviations of the adjacent 12 monthly values fails the check. On a failure, every arm gains
an era indicator if the manager approves, and the page says so.

## 5. Arms

**Primary demand.** "Stored" means no fit. The seed count is per fold.

| Arm | Inputs beyond `R_F` | Folds | Seeds |
|---|---|---|---|
| `D0` (stored `R_F`) | none | 12 | 0 to 4, stored |
| `D0n` (gate) | none, refit by the new code | 2026-01, 2026-06 | 3 |
| `DF` | `cloud_missing` only | 12 | 6 |
| `D1` | T1 real, masked | 12 | 6 |
| `D1p` | T1 shuffled-run placebo, same mask | 12 | 6 |
| `D1t` | T1 time-shifted placebo, same mask | 12 | 3 |
| `D2` | T2 real, masked | 12 | 6 |
| `D2p` | T2 shuffled-run placebo, same mask | 12 | 6 |
| `D3` (exploratory) | T3 real, masked | 7 cloud-scored | 3 |
| `D0g`, `D1g` | none; T1 real. Gap rows dropped | 7 cloud-scored | 3 each |
| `K1`, `K1p` (positive control) | section 6 | 7 cloud-scored | 3 each |
| `R0`, `R1`, `R1p` (replicate) | none; T1 real; T1 placebo. No flag | 12 replicate | 3, 6, 6 |

**Solar farms.**

| Arm | Inputs beyond PLP's 18 | Folds | Seeds |
|---|---|---|---|
| `P0` (stored PLP, experiment 666) | none; quarterly refit, window from 2024-04-01 | overlap only | 0 to 2, stored |
| `P0L` | none; monthly refit, window from 2020-10-15 | 7 | 3 |
| `PF` | `cloud_missing` only | 7 | 5 |
| `P1`, `P1p` | T1 real; T1 shuffled-run placebo | 7 | 5 each |
| `P1t` | T1 time-shifted placebo | 7 | 3 |
| `P2`, `P2p` | T2 real; T2 shuffled-run placebo | 7 | 5 each |
| `P0g`, `P1g` | none; T1 real. Gap rows dropped | 7 | 3 each |

**Seed sets are pre-registered for every contrast, because `losses_for_bootstrap` raises when two
arms hold different seeds.** Each contrast is read on the seeds that the two arms share. `D1t` minus
`DF` is read on seeds 0 to 2. `P1t` minus `PF` is read on seeds 0 to 2. `D1` minus `D0` is read on
seeds 0 to 4. `P1` minus `P0L` is read on seeds 0 to 2. In the minimum-detectable-effect null, the
second half of the seeds is relabelled before pairing.

**Reference rows come from stored forecasts or seedless recipes only:** `climatology` (the
month-by-half-hour `ClimatologyForecaster`, trained on each fold's window and repeated once per
seed), `manual_heuristic`, BG2 and S0M for Primary demand, and stored `R_F` and `A_F` of experiment
707. No reference row comes from experiment 404's lag-free head or from any forecast that the memory
file `weather-look-ahead-404.md` lists as voided.

**The scorer prints every arm's input column list**, so a reviewer can check each arm against these
tables. A real arm and its placebo have the same columns in the same order.

## 6. Placebos and controls

**Both placebos copy the missing pattern exactly.** A placebo row is missing if and only if the real
row is missing, and the code asserts that the two `cloud_missing` columns are equal on every row
before any fit. With a different pattern, the placebo and the real arm would differ in what the
flag says about the date.

- **Shuffled-run placebo (in the planned contrasts).** For each series, the cloud columns of run
  `r`, at every lead, are replaced by the cloud columns of a donor run at the same leads. The donor
  is drawn among the runs that carry cloud, lie in another calendar year, fall within 15 days of
  `r`'s day of the year, and were published before the fold's cut. The donor map is seeded by
  series, fold, and fitting seed, and is stored under `work/placebo/` so an audit can read it.
  Swapping a whole run keeps the daily cycle, the growth of spread with lead, and the season. The
  rule is experiment 707's donor rule applied to forecast runs. A recent run (2025-11-21 on) takes
  its donor from the back-filled years, and the page states that the donor then comes from the
  earlier ingest.
- **Time-shifted placebo (second control).** The cloud columns of run `r` are replaced by those of
  the run issued 14 days earlier, at the same leads. Where that run does not exist (the first 14
  runs of a segment, or the upstream gap of 2024-01-03), the run issued 21 days earlier is used, and
  then the run 7 days earlier. Every donor is issued before the row's own run.

**The checks use a common correction.** Five checks read at 95% (four placebo-carries-nothing
contrasts and the gap-scored contrast) are corrected to 99%. One failing check at 99% triggers an
audit and changes no verdict.

- **Placebo carries nothing:** `D1p` minus `DF`, `D1t` minus `DF`, `P1p` minus `PF`, and `P1t` minus
  `PF` each have an interval that contains zero. A failure fires audit trigger 5.
- **Gap-scored folds:** `D1` minus `D1p` on folds 2025-07 to 2025-10 has an interval that contains
  zero. The interval is a t interval over the four folds and the seeds, because four months give
  only 35 distinct bootstrap resamples. A failure means that the training-time difference between
  real and placebo cloud moves forecasts that receive no cloud, and fires audit trigger 5.
- **Positive control (Primary demand):** temperature goes through the cloud's exact handling. `K1`
  removes the mean and spread of 2 m temperature and of 2 m dew point from the base inputs and adds
  them back as four masked columns with the cloud's missing pattern and the flag. `K1p` is its
  shuffled-run placebo. The control passes if `K1` minus `K1p` is wholly below zero at both groups.
  A failure means the masked-gap handling cannot pass a known signal across the gap, and the Primary
  demand reading of C1 and C2 is then `inconclusive` whatever the intervals say.
- **Solar farms have no positive control of this kind.** Every strong PLP input feeds the
  physics-lite power at the centre of the mixture, so masking one changes the network. A solar null
  is read only through an interval that bounds the effect, and the same-row pair checks the gap
  handling.
- **Gate.** `D0n` matches stored `R_F` on folds 2026-01 and 2026-06 within 2 standard errors of the
  seed-to-seed spread, and a prediction-only check reloads stored `R_F` states and reproduces the
  stored forecasts to 1e-4. `P0L` is compared with stored `P0` descriptively, because the refit
  schedule and the window both differ.

## 7. Weather at the forecast origin

**Every row takes all of its weather, cloud included, from one run: the 00 UTC run issued at or
before the forecast origin and published by then.** A run counts as published 9 hours after its
initialisation. For Primary demand the origin is the run time plus 9 hours (09:00 UTC). For the
solar farms the origin is 12:00 UTC on the run's day. No weather value, cloud included, comes from a
run issued after the origin.

**The code asserts the rule on every row of every arm, training and scoring, real and placebo.** The
Primary demand code calls experiment 707's `assert_origin_after_run` (`nwp_init_time + 9 h <=
origin`) on the cloud summaries as well as on the seven existing variables. The solar input builder
has no row-level assertion, so the solar code adds one: `init_time + 9 h <= origin`, and `init_time`
equal to 00 UTC of the origin's date. A failed assertion makes the experiment `invalid`.

**The leakage condition for a shuffled-run donor is its publication before the fold's cut.** Every
donor run satisfies `donor_init + 9 h <= cut`. The donor need not precede the row's own run, because
a donor issued before the cut has no bearing on the row's origin. The time-shifted donor precedes
the row's run by construction.

**The code prints one example row per target.** For Primary demand: run 2026-01-14 00:00 UTC, origin
2026-01-14 09:00 UTC, target 2026-01-19 17:30 UTC (lead 128.5 hours), with cloud mean and spread
from the run of 2026-01-14 and never from the run of 2026-01-19. For the solar farms: run
2026-04-10 00:00 UTC, origin 12:00 UTC, target 2026-04-13 11:30 UTC, with cloud from the run of
2026-04-10.

**No score above 24 hours comes from experiment 404's lag-free head.** Neither network is that
head: `R_F` reads three origin lags of power at least 3.5 hours old, and PLP reads no power.
Experiment 404's head took the run nearest each valid time, which is weather issued after the
origin. No arm, reference row, or comparison here reads a forecast or a score from that lineage.

**Cloud feeds no lag.** T3's daylight mean covers the valid day's forecast cloud from the row's own
run, never a later run's.

## 8. Planned contrasts, scoring, and the decision rule

**Nine planned readings, from six contrasts, are each read at 1 - 0.05/9 = 99.44%.** Negative favours
real cloud. Every other number is exploratory and labelled so.

| Contrast | Definition | Rows | Readings |
|---|---|---|---|
| C1 | `D1` minus `D1p` | cloud-scored folds, days 1 to 10 | the six; the eight |
| C2 | `D2` minus `D2p` | cloud-scored folds, days 1 to 10 | the six; the eight |
| C3 | `R1` minus `R1p` | replicate folds, days 1 to 10 | the five; the eight |
| S1 | `P1` minus `P1p` | daylight, days 1 to 10, six farms pooled | one |
| S2 | `P2` minus `P2p` | daylight, days 1 to 10, six farms pooled | one |
| S3 | `P1` minus `P1p` | daylight, 12 to 36 hours (the solar protocol's band) | one |

**Metric.** CRPS as twice the mean pinball loss over the 11 quantile levels, in points of effective
capacity, with the normalised MAE of the median beside every CRPS number. Each farm's error is
divided by its own capacity before any mean.

**Interval.** Per-row losses of each seed go through `experiment_store.losses_for_bootstrap` and
`studies.bootstrap.bootstrap_difference_at_level`, which resamples whole calendar months paired
across arms and draws a fitting seed, with 50,000 resamples (about 140 draws per tail at 99.44%).
Seed noise is therefore inside the interval. The decisive interval is the wider of this interval
and a t interval on the monthly differences, because a month bootstrap on 7 months covers less often
than its nominal level. The contrast on seed-averaged quantiles (experiment 707's convention) is
printed beside both as a descriptive reading.

**Minimum detectable effect.** The scorer prints it before any contrast is read. The minimum
detectable effect is the half-width, at 99.44%, of a null contrast between two halves of the
placebo arm's seeds (seeds 0 to 2 against 3 to 5 for Primary demand; 0 and 1 against 2 and 3 for
solar). Because the bootstrap draws one fitting seed per resample, a half-split has the same noise
structure as the real contrast. Experiment 707's intervals left seed noise out, and its caveat says
the seed noise was about as large as its C1 at the six. The minimum detectable effect is therefore
estimated (interim, not measured) at about 0.05 points at the six and 0.03 at the eight on the
cloud-scored folds, and about 0.03 and 0.02 on the replicate. For the solar farms the only guide is
experiment 292's half-width of 0.33% at 12 to 36 hours, about 0.02 points. **C1 and C2 at the six
are not expected to resolve a gain of 0.02 points.**

**Verdict per target.** The target's verdict is the better of its C1 and C2 readings (S1 and S2 for
the solar farms), per group, ranked `supported`, `small gain`, `inconclusive`, `not supported`,
`refuted`, with two modifiers for Primary demand. If C1 and C2 are `inconclusive` or `not
supported` at a group and C3 is `supported` there, the group reads `partly supported`: cloud
carries information that did not cross the gap. If C1 or C2 is `supported` and C3's lower limit is
above 0, audit trigger 2 fires and the group reads `inconclusive` until the audit reports.

**Front-matter verdict.** `supported` if the six, the eight, and the solar farms are all
`supported`. `partly supported` if any is `supported`, `partly supported`, or `small gain`.
`refuted` if any is `refuted` and none is `supported`. `not supported` if all three are
`not supported`. Otherwise `inconclusive`. The first paragraph of the finished page states each
target's verdict separately, because the question has two parts.

**Slices, all exploratory, each with its half-hour count and a 95% interval:**

- **By lead time.** Primary demand: day 1, days 2 to 3, days 4 to 6, days 7 to 10. Solar farms: the
  same bands plus days 11 to 14. ENS stamps are 6-hourly after 144 hours, so cloud past day 6 is a
  linear interpolation between 6-hourly values, and the page says so.
- **Sunny and overcast half-hours.** The slices are defined from what happened, never from an arm's
  input: the realised clear-sky index (CAMS global horizontal irradiance over the Haurwitz clear-sky
  irradiance) at the valid time, on half-hours with the sun at least 10 degrees above the horizon.
  Sunny is an index of 0.8 or more, overcast 0.4 or less (`studies.era5_ladder.CLEAR_SKY_INDEX_THRESHOLDS`),
  and the half-hours between are a third row. Night half-hours are a fourth row for Primary demand.
- **Overall**, per Primary, per farm label (A to F), per fold, and against the number of months of
  recent cloud in the training window (1 to 7), which shows whether a gain grows as cloud history
  returns.
- **Protocol reports:** the eight with and without D8; q01, q05, q95, and q99 coverage and pinball
  loss; cold winter nights; sunny April to June; the avoidable-curtailment cost; a day-ahead and a
  14-day example forecast per target.
- **Real minus reference** (`D1` minus `D0`, `P1` minus `P0L`) and **flag-only minus reference** are
  reported with intervals as the total effect and the date-proxy effect.

**Checks list on the finished page:** the median minus the mean forecast per site; the MAE contrast
beside each CRPS contrast; which audit triggers fired. A leave-one-Primary-out and a
leave-one-farm-out reading of each planned contrast is printed, because experiment 292's solar gain
vanished when one farm was left out.

**The four load-shift scenarios do not apply**, because the experiment does not concern load-shift
events.

## 9. Rebuilding the ENS summary caches

**The rebuild writes only inside this experiment's data folder and needs no change outside the
experiment's code folder.** `pyproject.toml` pins production revision `f9dcd22`, whose `Nwp` schema
holds the column, and the production half-hourly resampler treats cloud as a continuous variable.
Nothing under `src/`, `packages/`, or a contract changes. The main research checkout pins an older
revision with no cloud field, so the session runs `uv sync --group gpu` in this worktree, and every
build script asserts `"total_cloud_cover_atmosphere" in Nwp.model_fields` at its start.

**Primary demand.** A new script writes `work/ens_cloud/init=YYYY-MM-DD.parquet`, one file per run,
keyed like the existing summaries (`time_series_id`, `power_fcst_init_time`, `valid_time`). The
script reads `Nwp.scan_delta` for one run and one column, resamples the 51 members with
`ml_core.features._nwp._upsample_nwp_to_half_hourly`, and writes the T1 and T2 columns. The existing
seven-variable summaries of experiments 397, 607, and 707 are read as they are and joined on the
three keys. A second script builds the raw matrices with the cloud columns appended. The runs needed
are 2021-07-20 to 2024-03-31 (985 runs, because 2024-01-03 is absent) and 2025-11-21 to 2026-06-30
(222 runs). No file is written for a gap run.

**Solar farms.** A copy of `e00666_solar_inputs.py` in this experiment's folder reads the six farm
cells from 2020-10-15, adds the 51-member cloud to `read_members`, and writes its tables under this
experiment's `solar_inputs/`. The copy reads through `Nwp.scan_delta`, or first checks on one
partition from 2023 and one from 2025 that `pl.scan_delta` returns the cloud column (values for the
2023 run, nulls for the 2025 run) rather than raising. Experiment 666's tables are not overwritten.

**Experiment 561's 222 cloud files are a cross-check, not an input.** They were built from the
Dynamical.org store with a different area weighting. The build prints the correlation and the mean
absolute difference between the two builds for the overlapping runs.

**Three conditions come before any fit or any build of a recent run, in order.**

1. The manager's message that the re-write worker has reported its validation. No fit and no build
   of a recent run starts before that message. The back-filled runs (to 2024-03-31) are not being
   re-written, and their cloud summaries can be built first if the manager agrees that reading the
   table during the re-write is safe.
2. The experiment's own check of the table at one pinned Delta version, recorded in the lineage note
   and passed to every later read: every run from 2025-11-21 to 2026-06-30 holds 51 members,
   7,243,785 rows, and no null cloud; cloud lies in 0 to 100; recomputed means and spreads of the
   seven older variables match the stored summary files for 20 sampled runs to float tolerance; and
   the era check of section 4 passes.
3. The `data-validation` checklist on the cloud summaries, and a fresh code review of the build
   script before its first run.

**The build is CPU work of unmeasured length.** Time one run first, then run 4 single-threaded
workers, each writing one run's file atomically and skipping a file that exists.

## 10. Compute

**The experiment runs on 2 concurrent fit streams.** The manager's instruction is that the caps of
concurrent `systemd-run` scopes must sum below the free memory, which is about 43 GB available now.
Each stream's cap is set from a peak that one measuring fit has measured, and the two caps together
stay below the free memory at launch. The GPU share is for the manager to give when the first fit is
requested.

**No fit starts until the manager confirms that the ENS re-run is validated.** Before the first fit
(CPU only), the experiment runs the cloud builds (4 workers, 1 thread each), the era check, the
placebo donor maps, the raw matrices, and a code review of every script.

**Launch rules.** Before each launch, `free -g` shows memory available for the sum of the caps and
`uptime` shows a load below 24. Every Python command runs with `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`, under `systemd-run --user --scope -p
MemorySwapMax=0 -p OOMPolicy=continue` with a `MemoryMax` set as above (16G for the measuring fit),
and with `uv run --no-sync` after `uv sync --group gpu`. Each stream runs one (arm, fold, seed) job
at a time, with a fixed seed, its own output file, and an atomic write, so a crashed job reruns
without touching the others. The fourth core is for loading and scoring.

**Job count (estimate).**

| Target | Fits | Basis |
|---|---|---|
| Primary demand, 12 recent folds | 423 | `DF`, `D1`, `D1p`, `D2`, and `D2p` at 72 each; `D1t` 36; `D0n` 6; `D3` 21; no fit for `D0` |
| Primary demand, 7-fold pairs | 84 | `D0g`, `D1g`, `K1`, and `K1p` at 21 each |
| Primary demand, replicate | 180 | `R0` 36; `R1` and `R1p` at 72 each |
| Solar farms | 259 | `PF`, `P1`, `P1p`, `P2`, and `P2p` at 35 each; `P0L`, `P1t`, `P0g`, and `P1g` at 21 each |
| Total | 946 | |

**Time (estimate).** A Primary demand fit took 62 s of GPU time in experiment 707's logs (a mean of
38 s for `S_ens` on 10.3 to 13.0 million rows, and 24 s for the fine-tune and forecast), so 687 fits
take about 12 GPU-hours, and up to twice that with data loading. The solar fit time rests on
experiment 666's log, which prints the height of the forecast table and not the training rows, so
the training-row count is unknown. The window from 2020-10-15 is about 3 times longer than the
window of experiment 666, so 259 fits are estimated at about 9 GPU-hours. The total is about 21 to
42 stream-hours, so on 2 streams the run takes about 10 to 21 hours of wall time. The first fit
measures wall time, peak resident memory (the scope's `MemoryPeak`), and peak GPU memory
(`torch.cuda.max_memory_allocated`), and the page reports the corrected figures.

**First-fit request to the manager.** One stream runs one fit of `D1` on fold 2026-06, seed 0. The
request asks for a GPU share for 2 streams (experiment 707 planned 5 GB per stream and never
verified it; the cloud columns add under 5% to the input matrix) and for confirmation that the
memory cap is per command and not for the slot. The experiment reports one line when the fits start
and one when the last fit ends and the slot is free.

## 11. Data hygiene, anonymisation, and standing practices

**Farms are labelled A to F only, and no farm name, identifier, coordinate, or megawatt series
appears in a chart, a report, a commit message, a brief, or a pull-request body.** The labels come
from `studies.anonymise.site_labels_for`. The page publishes no start date, first-data date, or
commissioning date for any label, because such a date would let a reader link a label to a public
record. The solar example forecast uses a day-count axis with no calendar dates, states only the
month and year in the text, and sets `aria=False` on the mark.

**Standing practices followed:** monthly refits; cleaned data and the D15 and D18 cut; every fitted
state saved under this experiment's data folder; stored predictions reused as references; tails and
curtailment cost reported; example forecasts plotted; a draft pull request. **Departed from:** the
four-month screen (section 2).

**Earlier advice followed:** a placebo with the same missing pattern as the real arm (experiments
559 and 568), a cloud-missing flag tested on its own (experiment 568), and a paired contrast on the
same rows with the seed noise inside the interval (experiment 707).

## 12. Review points to settle at implementation

**The design review raised six points that this plan does not yet settle.** Each is settled with the
manager, or by a plan review, before the first fit.

- **Fewer seeds.** The bootstrap draws one fitting seed per resample, so the interval does not
  narrow as seeds are added. Cutting the planned arms to 3 seeds would save about 180 Primary demand
  fits and 70 solar fits, with no loss of interval width. The alternative is to keep 6 seeds and
  read a seed-averaged contrast with a bootstrap that resamples seeds inside each month draw, which
  needs a new helper whose move into `src/studies/` would need its own reviewed pull request.
- **The era of the placebo cloud.** For a scored run in 2025-12 to 2026-06, the real cloud comes
  from ECMWF's open data and every shuffled-run donor comes from the back-filled runs before
  2024-04-01, many of them before the 2023-06-27 resolution change. C1 and C2 are then an era
  contrast as well as an information contrast. The options are to draw donors from the recent
  segment, at least 14 days from the row's run, and to use the time-shifted rule where fewer than
  about 20 eligible recent runs exist (folds 2025-12 and 2026-01), or to make the time-shifted
  arms `D1t` and `P1t` the planned comparator with 6 and 5 seeds.
- **The replicate and the resolution change.** Fold 2023-06 straddles 2023-06-27. The `study` skill
  asks to drop the straddling part-month on both sides, add an era indicator to all three replicate
  arms, and restrict donors to the row's era.
- **A calibrated positive control.** The `K1` control is large and trivial, because `K1p` has no
  temperature. A calibrated known-answer control would add a synthetic column equal to the realised
  clear-sky index plus Gaussian noise, tuned on one fold so that the unmasked column gains about
  0.04 points, then masked with the cloud's pattern and contrasted with its placebo. The control
  passes if the masked gain is at least half the unmasked gain. The column reads outcome data, so
  the column is a control only. PLP's `longwave`, `relative_humidity`, `precipitation`, and
  `pressure_surface` inputs do not enter the physics-lite power, so a solar version is possible.
  Reporting `K1` minus `DF` on shared seeds would measure how much masking costs a strong signal.
- **A solar replicate.** A back-fill replicate for the farms with enough early history is possible
  and is left out to hold the experiment to one screen of results.
- **The solar holdout.** `evaluation-protocol-solar.md` would allow July and August 2026, two
  high-sun months, but the store refuses valid times from 2026-07-01. A labelled exploratory
  reading of those months is an option.

**One further limit applies to every result.** `R_F` is not the port target, so a gain on `R_F` may
not carry to the Candidate, which reads 14 thermal-lag and solar-gain columns that `R_F` lacks. The
cloud-scored folds cover December to June only, and fold 2025-12 holds only 10 recent runs with
cloud in its fine-tune window.

## Files

**Planned files; the finished page lists the files as written.**

- `ml-experiments/00711-ens-total-cloud-cover-for-primary-demand-and-solar-farms/`: this
  experiment's code folder, to be written (cloud build, placebo maps, raw matrices, solar inputs,
  fit, and report scripts, each named with the id in a valid module form, such as `e00711_fit.py`).
- `ml-experiments/00707-ens-backfill-pretraining-gain-against-ens-history-length/` (read,
  [write-up](../00707-ens-backfill-pretraining-gain-against-ens-history-length/result.md)): the
  `R_F` training code, `assert_origin_after_run` in `e00707_prep.py`, the donor rule in
  `e00707_placebo.py`, and the stored `R_F` and `A_F` forecasts.
- `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_solar_inputs.py` and
  `e00666_solar_fit.py` (read, then copied):
  the PLP inputs and fit, and the stored quarterly PLP forecasts. The write-ups are experiments 659
  and 666.
- `data/ml-experiments/00711-ens-total-cloud-cover-for-primary-demand-and-solar-farms/` (written):
  the per-run cloud summaries under `work/ens_cloud/`, the donor maps under `work/placebo/`, the
  solar tables under `solar_inputs/`, the fitted networks, and the predictions.
- `data/ml-experiments/briefs/total_cloud/` (read): the private findings, design, and design
  review that this plan follows.
