---
id: "00707"
slug: ens-backfill-pretraining-gain-against-ens-history-length
title: >-
  Does pre-training the demand network on ERA5 and CAMS still help when the ECMWF ENS history
  reaches back to 2020?
theme: retraining-and-training-history
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores about 456 fitted networks and per-row forecasts; (2) more than one design
  defensibly tests the hypothesis (which stages the long ENS window feeds, whether the placebo
  permutes weather); (3) it reuses the training code of experiments 397, 418, and 425, whose callers
  cannot be named without searching; (4) it touches no serving path or degradation rule; (5) it
  changes no contract. It waits for the ENS back-fill of 2020-10-15 to 2024-03-31.
hypothesis: >-
  The gain from pre-training the pooled mixture density network on ERA5 and CAMS before fine-tuning
  on ECMWF ENS is smaller when the ENS history is 47 to 58 months (from 2021-07-20) than when it is
  24 months.
prediction: >-
  Written before any fit. C1 (pre-trained minus ENS-only, both with the long ENS window) is between
  -0.04 and 0 points of CRPS at the six signed-MW Primaries and between -0.10 and 0 at the eight
  MVA-only Primaries, against -0.050 and -0.124 in experiment 425. The placebo contrast C5 is below
  zero at the 24-month window.
decision_rule: >-
  C1 is read per group at 99.69% (16 readings, month-block bootstrap on seed-averaged forecasts),
  on its interval [lower, upper], applying these tests in order. Lower limit above 0: refuted.
  Upper limit below -0.02: supported. Lower limit at or above -0.02: not supported (no longer worth
  porting). Upper limit below 0: small gain. Otherwise the interval straddles both -0.02 and 0:
  inconclusive, and the width is printed. The threshold of 0.02 is the protocol's minimum for a new
  fitted step, because pre-training adds one stage and two weather sources. C1 minus C2 is
  descriptive. The gate and the stop rule are in the Gate and Stop rule paragraphs.
benchmark: >-
  Stored A and R of experiment 425 (24-month window), and the ENS-only arm R_F with the long window.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS (twice the mean pinball loss over 11 quantile levels) at lead days 1 to 10 in points of
  effective capacity, with the normalised MAE of the median beside it.
production_status: not-for-production
adopted_in: none
---
**Status: planned. The sections below are the plan; the finished page replaces them.**

Our weather-forecast archive (ECMWF ENS, the ensemble weather forecasts from the European Centre for
Medium-Range Weather Forecasts) has covered 2024-04-01 onwards, and the neural demand forecaster
trains on it. Earlier experiments found that first training the network on estimates of past weather
(ERA5 reanalysis and CAMS solar data) and then on ENS beats training on ENS alone, but none had more
than about 24 months of ENS. dynamical.org now serves ENS back to 2020-10-15. This experiment asks
whether the gain from the ERA5 and CAMS stage shrinks when the ENS history is four to five years
long, by fitting the network of [experiment
425](../00425-best-mdn-architecture-assembly-with-ablations/result.md) with and without that stage on
24-month and 47-to-58-month ENS windows. It builds on [experiment
418](../00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens/result.md) and
[experiment 494](../00494-simplify-demand-network-refit-cadence-and-pretrain-once/result.md).

## Hypothesis, prediction, and decision rule

**Hypothesis.** The gain from pre-training the pooled mixture density network on ERA5 and CAMS is
smaller when the ENS training history is 47 to 58 months (window F, from 2021-07-20) than when it is
24 months.

**Prediction.** At the six signed-MW Primaries the pre-training contrast C1 lies between -0.04 and 0
points of CRPS, against -0.050 in experiment 425 and -0.061 in experiment 418. At the eight MVA-only
Primaries C1 lies between -0.10 and 0. The experiment cannot resolve a shrink from -0.05 to -0.02,
because experiment 425's minimum detectable effect was 0.037 and C1 − C2 has an interval about 1.4
times wider. The page says so.

**Decision rule.** The smallest gain worth porting is 0.02 points, the protocol's minimum for "a new
fitted step, a new input, or a new data source" (`evaluation-protocol-demand.md`), because
pre-training adds one fitted stage and two weather sources to maintain. C1 is read per group at the
Bonferroni level:

The tests apply in this order to the interval [lower, upper]. The first test that holds gives the
verdict.

- **Harmful:** the lower limit is above 0. Verdict `refuted`.
- **Still worth porting:** the upper limit is below -0.02. Verdict `supported`.
- **No longer worth porting:** the lower limit is at or above -0.02. Verdict `not supported`.
- **Small gain:** the upper limit is below 0. The lower limit is then below -0.02, so the interval
  straddles -0.02. Verdict `small gain`.
- **Cannot tell:** none of the above holds, so the interval straddles both -0.02 and 0. Verdict
  `inconclusive`, and the scorer prints the interval's width.

The experiment reports the verdict and the interval, and makes no adoption.

## Arms

**Every arm is the network of experiment 425 (arms `A` and `R`), widened by 1.15.** The code is
copied from `ml-experiments/00425-best-mdn-architecture-assembly-with-ablations/` and
`00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens/` into the new experiment's folder. The
copy keeps `FIRST_RUN = 2024-04-01`, because the held-out validation weeks key on it, and negative
days floor-divide correctly. The input caches (experiment 87's rows, experiment 397's run summaries,
and the raw `.npz` files) are rebuilt for 2020-10-15 onwards. Three stages combine:

- `S_era`: ERA5 and CAMS pseudo-runs from 2019-09-01, unweighted (experiment 418's recipe, which
  sets the `covid` input and keeps lockdown rows).
- `S_ens(W)`: an unweighted stage on the ENS rows in window W, with a fresh AdamW at a learning rate
  of 5e-4 (see Method).
- `FT(W)`: experiment 425's fine-tune on the last min(W, 730 days) of ENS, 0.5-year half-life.

The half-life gives a 2021 row about 0.2% of a new row's weight, so windows above 24 months need
the `S_ens` stage to be tested at all. Windows filter on valid time, so runs from 2023-06-17 to
2023-06-26 enter fold 2025-07's 24-month window; the page states this.

| Arm | Stages | ENS window W | Seeds |
|---|---|---|---|
| `A_24`, `R_24` (stored; gate) | `S_era`→`FT`; `FT` | 24 months | 0 to 4 |
| `A_E24`, `R_E24` | `S_era`→`S_ens`→`FT`; `S_ens`→`FT` | 24 months | 0 to 4 |
| `A_F`, `R_F` | `S_era`→`S_ens`→`FT`; `S_ens`→`FT` | 2021-07-20 to cut (47 to 58 months) | 0 to 4 |
| `A_12`, `R_12` | `S_era`→`FT`; `FT` | 12 months | 0 to 2 |
| `A_C`, `R_C` | as `A_F`, `R_F` | 2020-10-15 to cut (56 to 68 months, lockdown ENS rows kept with `covid`) | 0 to 2 |
| `A1x_24` (placebo) | shuffled `S_era`→`FT` | 24 months | 0 to 2 |
| `R_F+flags` | as `R_F`, two extra inputs | as F | 0 to 2 |

**Stored controls and references.** The gate reuses experiment 425's stored `A` and `R` (60 raw
files each, 12 folds × 5 seeds, seedless, unwidened, keyed on `time_series_id`,
`power_fcst_init_time`, `valid_time`, written outside `experiment_store.write_predictions`), under
row set `fold_a_daily_gap_v1`; no new row set is needed. Reference rows are `climatology`,
`manual_heuristic`, BG2 (experiment 87), and S0M (experiment 389), the last two from stored XGBoost
forecasts only. About 456 fine-tuned networks result (4 five-seed arms and 6 three-seed arms, 12
folds); every fitted state is saved in the experiment's data folder.

## Contrasts, scoring, controls

| Contrast | Definition | Question |
|---|---|---|
| C1 (headline) | `A_F` − `R_F` | Pre-training gain with long ENS |
| C2 | `A_E24` − `R_E24` | The same recipe at 24 months |
| C3 | `R_F` − `A_24` | Long ENS alone against the current recipe |
| C4 | `A_F` − `A_E24` | Long ENS inside the pre-trained recipe, recipe held fixed |
| C5 (placebo) | `A_24` − `A1x_24` | Weather, not power history, carries the gain |
| C6 | `A1x_24` − `R_24` | Power history alone |
| C7 | `R_C` − `R_F` | Lockdown ENS rows |
| C8 | `R_F+flags` − `R_F` | Resolution-upgrade and source-change steps |

**Scoring follows experiment 425.** The seeds that both arms hold are averaged in quantile space,
then `metrics.month_bootstrap_contrast` resamples whole months at 1 − 0.05/16 = 99.69%, for eight
contrasts in two groups, with 50,000 resamples (about 78 draws per tail). Every arm shares the row
set `fold_a_daily_gap_v1` and the 12 monthly folds 2025-07 to 2026-06. The scorer prints the minimum
detectable effect first. C1 − C2 is descriptive and carries its own interval. Descriptive readings,
labelled as such: with and without fold 2026-01 (which dominated experiments 449 and 487), days 1 to
6 against 7 to 10, the 12-month gain, per-Primary, and a t test on the 12 monthly differences,
because 99.69% undercovers at 12 months.

**Placebo.** `S_era` with whole weather days permuted within each Primary, power, calendar, and lags
left in place, drawn only from rows at least 3.5 hours before the fold's cut and within 15 days of
the day of year, across years. It passes if C5 lies wholly below zero. C6 asks whether the extra
power history alone helps.

**Gate.** (1) A prediction-only check loads experiment 425's stored states (`work/models/`,
`work/pretrained/`) and matches its stored predictions to about 1e-4. (2) A refit check: the seed
average of `A_24` and `R_24` on fold 2026-06 and 2026-01 matches the stored arms within two standard
errors of the seed-to-seed spread (the stored seed-mean difference has a standard deviation of 0.052
per fold, and 425 set only `torch.manual_seed`, so a fixed 0.005 tolerance is tighter than noise).

**Stop rule.** Stop and report to the manager if the gate fails, if a step check in section 5 fails,
or if `A_24` or `R_24` misses the stored arm on fold 2026-06 by more than the refit tolerance.

## Weather at the forecast origin

**Each ENS row takes its weather from one run: the 00Z run issued at least 9 hours before the row's
origin.** For the run of 2026-01-14 00:00 UTC the origin is 2026-01-14 09:00 UTC. The target at
2026-01-19 17:30 UTC (lead 128.5 hours) reads the member mean and spread of that run, never the
2026-01-19 run. The code asserts `nwp_init_time + 9 h <= origin` on every row and that every training
row's valid time is at least 3.5 hours before the cut, and prints one example row.

**No weather feeds the lag inputs.** The three origin lags are power readings at least 3.5 hours old
at the origin. The network has no lag-free head, so no score at any lead reads experiment 404's
lag-free table. `S_era` rows use ERA5 and CAMS at the valid time, in training only; experiment 418
bounds them 3.5 hours before the cut although ERA5 arrives 5 days late, and the page keeps that
bound for the gate and names it as a caveat.

## Data handling

- **Lockdown.** The back-fill holds 278 runs inside 2020-03-23 to 2021-07-19. Windows F and the
  24-month windows exclude them. `A_C` and `R_C` keep them with the `covid` input. `S_era` rows
  already include lockdown rows with `covid = 1` in every pre-trained arm, as in stored `A`. The
  window floor applies to origin and lag times too.
- **One missing run, 2024-01-03.** The back-fill holds 1,263 of the 1,264 daily runs from 2020-10-15
  to 2024-03-31. Dynamical.org holds only 26 of the 51 ensemble members for the run of 2024-01-03,
  so the ingest refuses it, and no experiment arm reads it. The code lists the run in its coverage
  report and treats it as a documented gap. A window that lacks any other run still stops the fit
  unless `--allow-partial-window` is passed.
- **Resolution upgrade, 2023-06-27, and source change, 2024-04-01.** Before any fit, compute monthly
  ENS mean minus ERA5 (CAMS for shortwave) at leads 12 to 36 hours, and the ENS spread, per variable
  at the 16 Primaries. A step at either date larger than three standard deviations of the adjacent 12
  monthly values fails the check. `R_F+flags` carries an indicator for each date.
- **Cleaned data and cuts.** `substation_zero` half-hours are dropped everywhere and counted per
  fold, separately for a flagged target and a flagged lag. The flagged half-hours come from the
  cleaned power table over 2019-09-01 to 2026-07-01, because experiment 398's list covers only
  2024-05-01 to 2026-03-05. D15 and D18 rows before 2025-03-01 are dropped. A row of D5, D10, D11,
  D14, or D16 is dropped if its valid time or its weekly lag is before the Primary's first usable
  date (D16: 2021-02-01). D9 (signed-MW six) has its first observed power on 2023-08-23, so long
  windows add nothing there; the page reports ENS months per Primary.
- **Precipitation type** is not among the network's seven weather variables, so the column is not
  read. **Shortwave** is a 6-hour mean after lead 144 hours; C1 is also read for days 1 to 6 and 7 to
  10 separately. **Calendar:** hour, day-of-week, and holiday columns only; early stopping holds out
  whole weeks.

## Folds and compute

**Folds are the 12 scored months 2025-07 to 2026-06, each with at least 12 months of history.**
**Screen first** on 2025-10, 2026-01, 2026-03, and 2026-06, seeds 0 to 2, all numbers interim. The
estimate is about 9 GPU-hours for the screen and 35 for the full run (extrapolated from experiments
418 and 494, to be re-measured on the first fold). `S_ens(F)` holds about 11 to 14 million rows (418
held 4.98 million over about 21 months). Request three streams on 4 cores, a GPU share of about 5 GB
per stream (unverified), and a memory cap near 14 GB per stream. Building run summaries for 1,264
back-filled runs is unmeasured CPU work.

## Method

**Five choices depart from experiment 425 or from the design note, and each is recorded here so that
a reader of the results can allow for it.**

- **`S_ens` starts a fresh AdamW at a learning rate of 5e-4.** The design note says that `S_ens`
  uses "`S_era`'s optimiser", and the code reads that as the same settings (AdamW, 5e-4, unweighted),
  not as the optimiser state carried over from `S_era`.
- **Every fine-tune that follows an earlier stage uses a learning rate of 2e-4.** The rule includes
  `R_E24`, `R_F`, and `R_C`, whose earlier stage is `S_ens` alone. Experiment 425 used 2e-4 only for
  `A`, which follows `S_era`. Only an arm with no earlier stage (`R_24`, `R_12`) fine-tunes from
  scratch at 5e-4.
- **The standardiser and the per-Primary target statistics come from the fine-tuning rows.**
  Experiments 425 and 418 do the same, and every stage, `S_era` and `S_ens` included, applies them
  unchanged. A stage with a longer window therefore sees inputs on the scale of the 24-month rows.
- **The minimum detectable effect uses the seed halves of `R_F`.** The scorer compares the mean of
  seeds 0 to 2 with the mean of seeds 3 and 4 of `R_F`. Experiment 425 took its minimum detectable
  effect from the stored `R`, which has a 24-month window.
- **The stored `A_24` and `R_24` saw less ENS than `A_E24` and `R_E24` do.** ENS in the old table
  starts on 2024-04-01, so before fold 2026-04 the stored arms saw 15 to 24 months of ENS. `A_E24`
  and `R_E24` read the back-filled runs and see a true 24 months in every fold. C2 and the stored
  425 line therefore differ in window as well as in code path, and the scorer prints both.

## Reports

Every table shows the normalised MAE of the median beside CRPS, per group, per Primary, and per lead
band. The page reports q01, q05, q95, and q99 coverage and pinball loss, cold winter nights, the
avoidable-curtailment cost, the protocol's load-shift slices and cold-start-months slice, a figure of
gain against ENS months, and a day-ahead and a 14-day example forecast. The Checks list gives median
minus mean per site, MAE beside each CRPS contrast, and the triggers fired. Likely triggers: 1
(placebo), 2 (C1 contradicts 418 and 425), 5 (`R_F` beats `A_24`), 6 (interval width), 7.

## Invalid conditions, risks, prerequisites

**The experiment is `invalid` if** any row reads a run issued after its origin, a training row
passes the cut, arms are scored on different rows, a valid time on or after 2026-07-01 is scored, or
lockdown ENS rows enter an arm other than `A_C` and `R_C`.

**Risks.** The 0.02 threshold is below the 0.037 minimum detectable effect. Window length and the
2023-06-27 era are confounded above 24 months, which C8 probes.

**Needed before the first long-window fit:** the back-fill (production pull request merged or the
maintainer's word to run from its branch), the `data-validation` checklist on it, and the two step
checks. **Work that can start now:** the plan commit, the copied code and its Opus review, the gate,
the placebo builder, and the 12-month and 24-month arms.
