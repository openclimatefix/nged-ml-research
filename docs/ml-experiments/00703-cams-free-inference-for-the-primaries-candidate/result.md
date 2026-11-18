---
id: "00703"
slug: cams-free-inference-for-the-primaries-candidate
title: >-
  How much skill does the Primaries Candidate lose when its two previous-day CAMS irradiance
  columns are not available at inference, and should they be replaced by the ENS analysis proxy or
  dropped?
theme: production-reference-and-best-guess
also_relevant_to:
  - demand-inputs
  - weather-products-and-blending
status: planned
kind: other
size: >-
  Complex. (1) It stores refitted K1 and Candidate networks, tables T, and per-seed quantile
  forecasts for two arms on four test months, plus one replay of the unchanged chain; (2) two
  replacements for the CAMS columns are defensible, and the plan fixes both; (3) it patches the
  input columns of a pipeline that spans experiments 397, 418, 437, 510, 560, 562, 569, 574, 591,
  613, and 667, whose callers cannot be named without searching; (4) it touches no serving path and
  no degradation rule; (5) it changes no contract. Reviews: plan, code, write-up against the printed
  report, and prose.
hypothesis: >-
  The Candidate, refitted from pre-training onwards without the previous day's CAMS irradiance at
  fine-tune and inference time, forecasts the 16 Primaries almost as well as the stored Candidate
  of experiment 667, whether the two columns are rebuilt from the ENS analysis proxy's shortwave
  (arm A) or set to zero in every stage (arm B).
prediction: >-
  Written before any fit. P1: arm A minus the control and arm B minus the control each change the
  CRPS by less than +0.03 points of capacity, pooled over the 16 Primaries, the four test months,
  and all leads. P2: arm A minus arm B lies within plus or minus 0.01 points, and its interval
  includes zero, so the rule below chooses arm B. P3: the normalised MAE of the median moves in the
  same direction as the CRPS in each contrast. P4: the replay of the unchanged chain reproduces the
  stored forecasts of experiment 667 to within 1e-5 of capacity at every quantile.
decision_rule: >-
  The decision contrast is C1, arm A minus arm B in CRPS, pooled over all scored rows, with a
  month-block bootstrap interval at the Bonferroni level for three contrasts (98.33%, conservative
  towards arm B because only C1 decides). Arm A is provisionally chosen only if the C1 interval lies
  wholly below zero, the point estimate is at least 0.01 points of capacity below zero, and both
  seeds agree in sign in every month. The 0.01-point threshold covers arm A's porting cost:
  production must keep the first day of the two previous ENS runs per Primary and compute a Haurwitz
  clear-sky ratio, where arm B needs no code at all. If arm A is provisionally chosen, both arms are
  refitted on the other four pre-July months (2025-12, 2026-01, 2026-03, and 2026-05), and the same
  rule is applied again to all eight months, whose result is final. Otherwise arm B, the simpler
  arm, is chosen. The decision to drop CAMS at inference is already made, so the experiment only
  chooses between the two replacements and measures the loss. C2 (arm A minus the control) and C3
  (arm B minus the control) quantify that loss and decide nothing.
benchmark: >-
  The stored per-seed normal-running (NRA) forecasts of experiment 667, refits 2025-11, 2026-02,
  2026-04, and 2026-06, seeds 0 and 1, which are the Candidate with the CAMS columns; climatology
  and the manual heuristic as stored for experiment 668.
row_set_id: backtest_year_2025_11_to_2026_10_daily_12utc
metric: >-
  CRPS (twice the mean pinball loss over the 11 levels, divided by effective capacity, times 100)
  and normalised MAE of the median, in points of capacity, by lead band (days 0 to 1, 1 to 3, 3 to
  7, 7 to 10, and 10 to 14); q05-to-q95 coverage and band width; q01, q05, q95, and q99 coverage
  and pinball loss; cold winter nights and the avoidable-curtailment cost; all four load-shift
  scenarios.
---

**Status: planned.** This page is the plan, committed before any fit. The write-up replaces it.

## The question

**Production will not have CAMS irradiance at inference, so the Candidate's two previous-day CAMS
columns need a replacement or have to go.** The Candidate is the demand neural network that
forecasts the 16 Primaries in the one-year backtest of [experiment
667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md). Its
weather encoder reads 14 thermal-lag and solar-gain columns from [experiment
437](../00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles/result.md) and
[experiment
569](../00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1/result.md).
Two of those columns read CAMS (the Copernicus Atmosphere Monitoring Service) at fine-tune and
inference time: `cams_prev_day_mean`, the mean CAMS global horizontal irradiance over the last
complete UTC day before the origin, in W/m2, and `cams_prev_day_csi`, that day's summed irradiance
over its summed CAMS clear-sky irradiance. The maintainer has decided that production reads no CAMS
at inference, and that the production forecast runs once a day at the 12:00 UTC origin on that
day's 00Z ENS run (the ECMWF 51-member ensemble), as the Candidate was trained. The experiment
measures how much skill the Candidate loses without those two columns, and chooses between two
replacements.

**CAMS may stay in pre-training.** Pre-training on ERA5 (ECMWF's reanalysis) and CAMS pseudo-runs
is training-only, so the two columns of the pre-training rows may still come from CAMS.

## Arms

- **Control.** The stored per-seed NRA forecasts of experiment 667 (the Candidate as backtested),
  reused and not refitted.
- **Replay (check only).** The unchanged chain of experiment 667, refitted from pre-training onwards
  for the 2026-04 fold, into this experiment's data folder. [Experiment
  684](../00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/result.md)
  found the chain from step 2 onwards deterministic, but no experiment has replayed the pre-training
  step. The replay tests prediction P4. If any quantile of the replay differs from the stored
  forecast by more than 1e-5 of capacity, the control is refitted on all four months and the refit
  replaces the stored forecasts as the control, because the arms would otherwise carry a re-run
  difference the control does not. The replay-against-stored difference is reported as the noise
  floor whether or not P4 holds. A pass in 2026-04 is assumed to hold in the other three months.
- **Arm A, the analysis-proxy columns.** The same two columns, computed from shortwave instead of
  CAMS. `cams_prev_day_mean` becomes the mean shortwave over the 48 half-hours of the last complete
  UTC day before the origin (half-hour stamps from 00:30 of that day to 00:00 of the origin day, as
  period ends). `cams_prev_day_csi` becomes that day's summed shortwave over its summed Haurwitz
  clear-sky irradiance, evaluated at the middle of each half-hour at the Primary's coordinates
  (`f437.clear_sky_on_grid`, as experiment 437's clear-sky index already uses). At fine-tune and
  inference time the shortwave is experiment 437's analysis proxy: for each half-hour before the
  origin, the ensemble mean of the freshest run with a positive lead that covers the half-hour. A
  run's rows start at its 09:00 UTC native origin, so the proxy holds leads of 0.5 to 24 hours after
  that origin, which are 9.5 to 33 hours after the 00Z initialisation. The previous UTC day
  therefore comes from the run of the day before the origin (09:30 to 00:00) and the run two days
  before (00:30 to 09:00), both published before the origin. The plan review found all 48 half-hours
  finite for every run and every Primary from 2024-04-01 to 2026-06-30, except the first two origins
  of the archive. In pre-training the shortwave is the dense CAMS stand-in that every other
  pre-training shortwave column already uses, so the two columns have one definition in every stage.
  That stand-in is CAMS hourly irradiance copied onto both half-hours of each hour, so arm A's mean
  column equals the control's in pre-training, and only the clear-sky ratio changes there. The
  ensemble mean is smoother than CAMS (on one Primary, April to June 2025, the daily proxy mean
  correlates 0.80 with the CAMS day mean, with a standard deviation of 53 against 60 W/m2), so the
  frozen weather encoder sees compressed values at fine-tune time, which is a reason arm A may
  under-deliver. A day with fewer than 48 finite half-hours gives a missing value, which the
  standardiser sets to 0, as a missing CAMS day does in the control.
- **Arm B, the columns dropped.** Both columns are set to the constant 0 in every stage, in
  pre-training rows and in fine-tune, table, and scoring rows. The standardiser then maps them to 0
  as well, so the weights that read them never see a non-zero input and the forecast cannot depend
  on them. The network keeps its 29-input weather encoder and uses the same seeds. The arm is
  therefore the same function class as a network with the two columns removed, and a port may drop
  the two columns and their weights.

**Both arms refit the whole chain from pre-training onwards.** The chain is experiment 667's
`chain.sh` (steps `pretrain`, `step2`, `table`, `fit`, `ptable`, and `pfit`): K1 pre-trained on
ERA5 and CAMS, K1's step 2 with a fresh mixer, table T from K1 with its lags masked, the
Candidate's fine-tune grafted onto K1, then table T and the Candidate's prediction pass over the
scored window. Arm A's pre-training rows change because their clear-sky ratio uses Haurwitz instead
of CAMS clear-sky, and arm B's change because the two columns are zero. Each arm runs both seeds, 0
and 1, as experiment 667 did.

## Rows, folds, and seeds

- **Four test months, one per season:** 2025-11 (autumn), 2026-02 (winter), 2026-04 (spring), and
  2026-06 (summer). Each month is one monthly refit with its cut-off at 00:00 UTC on the first day
  of the month, a 730-day fine-tune history, and the embargoes of experiment 667 (3 days at step 2,
  7 days at the fine-tune).
- **Scored rows:** the NRA output at the 12:00 UTC contract origin of every day of the month, 672
  half-hours of lead per origin, at the 16 Primaries. Rows with a valid time on or after 2026-07-01
  are dropped, which shortens the leads of the last 14 origins of the 2026-06 fold. The actuals are
  the cleaned power with the `substation_zero` half-hours dropped, and D15 and D18 are cut before
  2025-03-01. The page reports the dropped half-hours per fold.
- **Seeds:** 0 and 1, paired across arms by seed. The headline table also gives the seed-averaged
  forecast (the mean of the two seeds' quantiles, sorted within the row), which is the forecast
  experiment 667 published.

## How the contrasts are read

- **C1, arm A minus arm B** (the decision). **C2, arm A minus the control. C3, arm B minus the
  control** (the loss).
- **Interval.** `experiment_store.losses_for_bootstrap` pairs the rows, and
  `studies.bootstrap.bootstrap_difference_at_level` resamples whole months and a seed, at 98.33%
  (Bonferroni over three contrasts), with 20,000 resamples. Four months is below the bootstrap
  module's six-month minimum for evidence, so every number is labelled interim, and an interval
  that includes zero is the expected outcome. That is why the rule sends a win for arm A to the
  other four months before it counts.
- **Beside every CRPS contrast** the page gives the same contrast in normalised MAE of the median,
  by lead band, and per Primary. A contrast whose CRPS and MAE point in different directions fires
  audit trigger 3.
- **Per month and per seed**, the page tables C1, because with four months an interval wholly below
  zero means in practice that arm A wins in every month.
- **Climatology** is the month, half-hour, and weekend-flag quantiles of `ClimatologyForecaster` as
  stored for experiment 668, trained on the history before each fold, with its rows repeated once
  per seed.
- **The predict phase reads actuals after 2026-07-01** when it rebuilds table T for the June fold,
  and the scorer drops every row with a valid time on or after 2026-07-01.
- **Absolute scores** of every arm, of the control, of climatology, and of the manual heuristic, by
  lead band, on the same rows.

## Reporting for load-shift events

**Every arm and both baselines are scored on four sets of rows:** all rows; rows outside every
logged event and at least 7 days after one; rows inside a logged event; and the 7 days after a
logged event ends. The events are those `e00668_slices_primaries.py` uses (the table of experiment
653, plus the verified transfer between D5 and D16 from 2025-12-30 to 2026-01-15) that overlap the
four months. The page plots one day-ahead forecast and one full 14-day forecast against the
observed power, and at least six time series around logged events across both metering groups,
from 14 days before each event to at least 7 days after it, with its start and end marked. NGED's
logs may miss events, so the rows "outside every logged event" may hold events no label covers, the
harm shown there is optimistic, and the real-event sets undercount. No generator appears on the
page.

## Checks before the first scored run

- **Causality of arm A's columns.** Rebuild one Primary's table with every run after a sampled
  origin deleted, and require the two new columns to be unchanged at that origin (experiment 437's
  `check`, applied to the new columns).
- **Every other column is unchanged.** The 12 other thermal-lag and solar-gain columns of each
  arm's table are bit-equal to experiment 667's stored table on every key.
- **Arm B's columns are zero** in the standardised pre-training, fine-tune, and scoring matrices.
- **The replay**, as under "Arms".
- **Arm B's standardiser**, stored in `chains/<month>/work/stats/B1/`, holds mean 0 and standard
  deviation 1e-6 for the two columns.
- **The arm tables are the ones read.** `e00667_fold.redirect_to_extended_inputs` sets
  `f437.TRAILING_DIR` to experiment 667's tables, so the arm patch wraps that function and points
  `f437.TRAILING_DIR` at the arm's tables after it runs. A printed example row proves the switch.
- **Arm A's table builder** relaxes the no-missing assertion of `e00667_tables.build` for the two
  new columns only.
- **The held-out power stays hidden** in the train phase, as experiment 667's `e00667_fold.py`
  enforces.

## Compute

**Nine to 20 chains: two arms on four months and the replay, plus 8 more if arm A advances to eight
months and 3 more if the replay fails.** A fold of experiment 667's chain took about 35 minutes on
the workstation's RTX A6000, as one GPU process capped at 8 GB of host memory. The runner checks
`nvidia-smi`, `uptime`, and the running processes first, runs at most three chains at once inside
the free GPU memory and cores, and never stops another session's process.

## Standing practices

- **Followed:** monthly refits; cleaned data everywhere, with the dropped half-hours per fold; the
  D15 and D18 cut; every fitted network saved under this experiment's data folder; the stored
  predictions of experiment 667 as the control; a four-month screen with a pre-registered advance
  rule, every number labelled interim; tails, cold winter nights, and the avoidable-curtailment
  cost from `e00677_tails.py` and `e00677_protocol.py`; example forecasts and load-shift plots with
  the unlogged-event caveat; a draft pull request.
- **Not applicable:** no XGBoost model is fitted. The calendar control is not needed, because the
  date-time encoder is the Candidate's own and identical in every arm.
- **Advice followed:** experiment 677's monthly refit and its advice to judge a lag change on
  event-free error as well as on all rows (the four scenarios); experiment 684's use of a
  deterministic replay in place of a re-run noise floor, extended here to the pre-training step;
  experiment 699's reporting of band width beside CRPS. None of the `advice` items of experiments
  437, 569, 667, 668, 677, 684, and 699 argues against this design.

## Checks list for the write-up

The write-up carries the median minus the mean forecast per Primary, the MAE contrast beside every
CRPS contrast, the band width of every arm, and which audit triggers fired.

## Method

The Method section is written with the results: the architecture, the training regime, and the
measurement of the chosen arm, then the differences of the other arm.

## Files

- The table builder, the arm patch, the chain runners, and the scorer live in the folder of
  [experiment 704](../00704-cams-free-inference-for-the-primaries-step-5-network/result.md), which
  asks the same question of the step-5 network and runs first. The arm tables (copies of experiment
  667's `inputs/trailing437/` tables with the two columns replaced, which `c562.TableLookup` reads
  through `f437.table_path`) are shared with experiment 704.
- `data/ml-experiments/00703-cams-free-inference-for-the-primaries-candidate/` will hold each arm's
  tables, chains, weights, and forecasts.
