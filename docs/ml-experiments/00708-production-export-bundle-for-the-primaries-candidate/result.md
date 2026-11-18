---
id: "00708"
slug: production-export-bundle-for-the-primaries-candidate
title: >-
  Can production rebuild the Primaries Candidate's forecasts, with its K1 neural network and table T,
  from a plain directory of weights, standardisers, a config, and golden fixtures, without importing
  research code?
theme: production-reference-and-best-guess
also_relevant_to: []
status: finished
kind: other
size: >-
  Complex. (1) It stores one refit of the Candidate (arm B of experiment 703, refit 2026-10) and one
  export bundle; (2) the extension of experiment 705's bundle format to two linked neural networks
  and a table (K1, table T, and the Candidate) has several defensible layouts; (3) it imports the
  chain of experiments 574, 591, 598, 613, 667, 703, 704, and 705, whose callers cannot be named
  without searching; (4) it touches no serving path and no degradation rule; (5) it changes no contract.
  Reviews: code, write-up, and prose.
hypothesis: >-
  Experiment 705's bundle format, with a `components/k1/` and a `components/candidate/` folder and
  matching fixtures, is enough for a script that imports only torch, numpy, and polars to reproduce
  the Candidate's stored quantile forecasts of the Primaries, including table T, which the
  Candidate reads from K1's lags-masked forecasts.
prediction: >-
  At five fixture origins in October 2026 and both seeds, the standalone reference check reproduces
  K1's lags-masked medians and the table T entries the Candidate reads to 1e-5 of capacity, the
  Candidate's input matrix to 1e-5, and each seed's quantiles and the seed-mean quantiles to 2e-5 of
  capacity; and the exporter's own rerun of the prediction pass equals the chain's stored per-seed
  forecasts exactly.
decision_rule: >-
  Supported if every tolerance in the prediction holds at every fixture origin and seed; otherwise
  the bundle is not fit to hand to production.
benchmark: >-
  The per-seed forecasts that experiment 704's `chain667.sh` writes for arm `drop`, month 2026-10,
  and the input matrices and tables T that its prediction pass builds.
row_set_id: none
metric: >-
  Largest absolute difference between the reference check's values and the stored values, per
  stage (K1 lags-masked medians, table T entries, Candidate input matrix, raw network outputs,
  per-seed quantiles, seed-mean quantiles, contract megawatts over capacity).
verdict: supported
best_arm: >-
  Format version 1 with K1 and the Candidate; largest per-seed quantile error 1.7e-6 of capacity
  against a tolerance of 2e-5.
headline: >-
  A 71 MB bundle reproduces the Candidate's quantiles, including K1 and table T, from torch, numpy,
  and polars alone, to 1.7e-6 of capacity.
lessons: >-
  A port of the Candidate must run K1 every day to keep table T dense, must keep a power history of
  at least 150 days (the bundle uses 180) for the signal columns, and must write the two CAMS
  columns as 0.0; research's sparse table T differs from a dense one by at most 7.6e-5 of capacity.
take_home_message: >-
  Production can now rebuild the Candidate's forecasts for the Primaries from a folder of files and
  check its own K1, table T, and signal-column code against reference values. The port must run the
  K1 neural network every day to fill table T and must keep 180 days of power history. The sparse
  table T that the Candidate trained on differs little from the dense table that production will
  hold.
advice:
  - >-
    Hand production the Candidate as a bundle of plain files with golden fixtures per stage, and
    keep the standalone reference check in the bundle (Checks).
  - >-
    Run K1 at every daily origin and keep each lags-masked median in the table T store; the
    Candidate reads the freshest median that is at least 24 hours old (Finding 1).
  - >-
    Keep at least 150 days of power history for the signal columns, and 180 days for margin: the
    truncation error is 9.8e-5 at 120 days, 5.0e-6 at 150 days, and 7.2e-7 at 180 days, against a
    tolerance of 1e-5 (Finding 3).
  - >-
    Write the two CAMS columns as 0.0 and return 11 quantile columns with no `ensemble_member`
    column (Findings 4 and 7).
production_status: candidate
adopted_in: none
---

In plain terms, the experiment asked whether production can rebuild the Primaries Candidate's
forecasts from a folder of files, without research code. The answer came from a standalone check
that runs the two neural networks and the table between them from the files alone. The check
reproduces every stage to within its tolerance, and the largest error on a per-seed quantile is 1.7e-6
of capacity against a tolerance of 2e-5.

**Status: finished.** This experiment is a tool for the port rather than a test of a change: it
exports the Candidate, the demand neural network of [experiment
667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md) for the 16
primary substations (the Primaries), in arm B form from [experiment
703](../00703-cams-free-inference-for-the-primaries-candidate/result.md). The export follows the
format that [experiment
705](../00705-production-export-bundle-for-the-primaries-step-5-network/result.md) built for the
step-5 neural network. The refit is that of [experiment
704](../00704-cams-free-inference-for-the-primaries-step-5-network/result.md)'s chain, month 2026-10.
The hypothesis above is the acceptance test the code was written against, and the plan was committed
before the code (`cfa29c256`).

## The question

**The Candidate is not a single neural network: it reads the output of a second one.** K1 is the
step-2 neural network of experiment 667 (`AuxLagMdn`). Its forecasts, made with its lag inputs
masked, fill table T, the store of K1's lags-masked median for every Primary and half-hour. The
Candidate (`PermLagMdn`) reads table T through 60 extra signal columns, which are built from the
Primary's own power history, its neighbours' power history, and table T. Production re-implements
inference and never imports research code, so the bundle must carry both neural networks, the table
T rule, and the signal rules as files plus reference values. The experiment builds that bundle for
the 2026-10 refit, and proves with a standalone check that the files are complete.

## The bundle

**One bundle is one plain directory in format 1 of experiment 705, with a component folder per
neural network.** The bundle is 64 files and 71 MB (`bundles/drop/2026-10/`). The bundle was
exported from a clean tree at commit `e5133a8a3`, and the manifest records that tree as clean. The
format is in `bundle_readme.md`, which the exporter copies into the bundle as `README.md`.

- `manifest.json`, `config.json`, `primaries.json`, and `reference_check.py` are as in experiment
  705.
- `components/k1/`: K1's 60 input columns in order, the standardisers, the offset-knot dates, the
  expected-weather rows (`expected_weather_rows.parquet`, 6.7 MB, shared with the Candidate), and
  one `state_dict` per seed (seeds 0 and 1). The folder is 7.1 MB.
- `components/candidate/`: the Candidate's columns, the signal columns in order, each Primary's
  neighbour list, the standardisers, and one `state_dict` per seed (464 KB in all).
- `fixtures/k1/` and `fixtures/candidate/`: for each fixture origin, the inputs, K1's lags-masked
  quantiles, the table T entries that the Candidate reads, the input matrix raw and standardised,
  the raw outputs of the neural network, and the quantiles per seed, seed-averaged, and at the 12:00
  UTC contract origin in megawatts. `fixtures/k1/history_rows.parquet` holds K1's inputs at every
  daily origin of the history window, and the Candidate's fixtures hold the dense table T.

**Four things are new against experiment 705's bundle.**

- The bundle holds two neural networks, and the Candidate reads K1's output, so `config.json` states
  the table T rule and the signal rules, and the manifest lists which component reads which.
- `reference_check.py` rebuilds K1's lags-masked medians and the whole of table T, then the
  Candidate's signal columns from the power history and table T, before it runs the forward pass.
- A sixth fixture, `origin=2026-10-03_synthetic_missing_inputs`, sets inputs missing on purpose.
- The contract check compares the full output frame, not only the megawatt values.

**The fixture origins are those of experiment 705's production bundle:** 2026-10-01, 2026-10-03,
2026-10-24, 2026-10-25 (the clock change), and 2026-10-31. At none of the five is a lag missing, so
the synthetic fixture is the only missing-lag case. The synthetic fixture is built from 2026-10-03
with blanked spans of power history, table T, and expected weather. It has 1,722 week-lag-missing
input rows and 4,068 day-lag-missing input rows of 10,848 input rows (16 Primaries and 678
half-hours). The expected values of the synthetic fixture come from the research code path, never
from `reference_check.py`.

## The state production must keep between daily runs

**Production keeps three stores between daily runs.** The bundle's `config.json` lists them under
`state_between_daily_runs`, and the README describes each one.

- **Cleaned power:** per Primary and half-hour, as a fraction of effective capacity, with missing
  half-hours kept missing. The lags read the 336 half-hours that end 24 hours before the origin,
  which reach back about 8 days. The signal columns read the whole 180-day window.
- **K1's lags-masked medians (table T):** the median (q50) of K1's lags-masked forecast for the 48
  half-hours after each daily 09:00 origin, per Primary and per seed, stored with the origin
  (Finding 1). K1 runs on the same 60 base columns as the Candidate.
- **Expected-weather rows:** for each Primary and half-hour from 8 days before the origin to 24
  hours before it, the 28 expected-weather columns of the freshest run of the ECMWF ensemble (ENS)
  issued before that half-hour. The lag columns of both neural networks read each row at the lag's
  own half-hour (Finding 5).

**The neighbour lists are a static file of the bundle, not state.** The file is
`components/candidate/neighbours.json`. After a refit or a gap in the stores, the config's
`cold_start` rule rebuilds the missing days by running K1 over those days' origins with each day's
inputs, as `fixtures/k1/history_rows.parquet` does. Entries with no forecast are NaN, and the
Candidate then sees missing-gap flags.

## Checks

| Stage | Largest error, six fixtures | Synthetic fixture alone | Tolerance |
|---|---|---|---|
| K1 lags-masked median | 4.2e-7 | 0.0 | 1e-5 |
| K1 lags-masked q05 and q95 | 5.4e-7 | 0.0 | 1e-5 |
| Table T entries | 4.2e-7 | 3.9e-7 | 1e-5 |
| Candidate input matrix, 60 base columns | 0.0 | 0.0 | 1e-5 |
| Candidate input matrix, 60 signal columns | 7.6e-6 | 7.2e-6 | 1e-5 |
| Raw outputs of the neural network | 2.1e-6 | 1.7e-6 | 1e-5 |
| Quantiles per seed, stored input matrix | 1.1e-6 | 7.7e-7 | 2e-5 |
| Quantiles per seed | 1.7e-6 | 8.3e-7 | 2e-5 |
| Seed-mean quantiles | 8.3e-7 | 4.8e-7 | 2e-5 |
| Contract megawatts over capacity, per seed | 1.7e-6 | 8.3e-7 | 2e-5 |
| Contract megawatts over capacity, seed mean | 8.7e-7 | 5.1e-7 | 2e-5 |

**The reference check passes at all six fixtures and both seeds, in 23 seconds on a CPU.** The check
runs as `python -I reference_check.py <bundle>` and imports only the standard library, torch, numpy,
and polars. The K1 stages are checked at all 162,816 history rows, which are 212 daily origins from
2026-04-01 to 2026-10-29. Every origin before October is inside K1's training period, except in
held-out weeks, so the check covers in-sample origins as well as the October ones. The input matrix
of the 60 base columns reproduces exactly from the raw columns, the standardisers, and the cleaned
power.

**The signal columns have the least margin, 7.6e-6 against 1e-5, and the tolerance stays at 1e-5.**
The 7.6e-6 compares the check's CPU values with the signal columns that the capture stored. The
truncation errors in Finding 3 are a different measurement: they compare a shorter power history
with the full history, in the same CPU code.

**No scored contrast exists in this experiment, so the audit triggers do not apply and no audit
ran.** The experiment compares the bundle's values with stored values and compares no forecast
skill between arms.

**The exporter's rerun equals the stored research output exactly.** Re-running the prediction pass of
the stored weights gives per-seed forecasts equal to the chain's stored forecasts (336,288 rows per
seed, one per Primary, half-hour, and October daily origin, with a largest difference of 0.0 for
seeds 0 and 1). The rebuilt table T equals the stored table T (424,801 filled entries, one per
Primary and half-hour, with a largest difference of 0.0 for both seeds). The decision rule is met,
so the hypothesis is supported.

## The code review and what it changed

**The bundle's table T is dense.** The reference check rebuilds table T from this refit's K1 run at
every daily origin of the 180-day window, and the fixtures and the reference values use that dense
table. Research's table T is sparse. The training rows, the test rows, and the previous month's test
rows leave whole-day holes, which are the held-out weeks and the purge windows around them (15 days
before and 9 days after each held-out week). The exporter keeps the exact check that a rerun with the
sparse table equals the stored forecasts.

**The synthetic fixture exercises every missing-input branch.** The synthetic fixture's flag counts
are non-zero for the missing own-day, own-week, neighbour, table T, and expected-weather cases of
every signal group.

**The contract check compares the full frame.** The contract check compares per-seed megawatts,
`time_series_id`, `valid_time`, `power_fcst_init_time` (the 12:00 relabel), `nwp_init_time`, and the
dtypes.

**The first export ran before the code review, which departs from the order the `ml-experiment` skill
asks for.**

## Findings for the port

**Finding 1: table T needs a K1 run every day.** `T[p,g]` is K1's lags-masked median from the freshest
origin `o` with `0 < g - o <= 48` half-hours, and the Candidate reads table T up to 48 half-hours
before its own origin. Production therefore runs K1 at every daily origin and appends the median to
the store. The K1 forecast must use its lags-masked read-out: lag columns 37 to 42 are set to (0, 1,
age maximum), the quantiles are sorted and scaled by the target standard deviation and mean, and
only q05, q50, and q95 are kept.

**Finding 2: the dense table T that production holds differs a little from the sparse one that the
Candidate trained on.** At the five fixture origins, the per-seed quantiles differ by at most 7.6e-5
of capacity (seed 0 at 2026-10-01), and the seed-mean quantiles by at most 4.4e-5. The mean
difference of the per-seed quantiles is at most 2.9e-6, and the mean difference of the seed-mean
quantiles is at most 2.6e-6. The signal columns differ by up to 0.017 at 2026-10-01 and by 0.0009 at
2026-10-31. The research table is empty for 27% of the 180-day window at 2026-10-01 and for 15% at
2026-10-31, while the dense table is complete.

| Origin | Seed-mean quantiles, largest difference (fraction of capacity) | Seed-mean quantiles, mean difference |
|---|---|---|
| 2026-10-01 | 4.4e-5 | 2.6e-6 |
| 2026-10-03 | 3.9e-5 | 2.2e-6 |
| 2026-10-24 | 5.5e-6 | 3.1e-7 |
| 2026-10-25 | 4.3e-6 | 2.8e-7 |
| 2026-10-31 | 2.5e-6 | 1.7e-7 |

**Finding 3: the signal columns need a power history of at least 150 days, and the bundle uses
180.** The signal columns include a 7-day exponentially weighted average of the power history, and
that average has a memory of weeks. The largest error in the signal columns against the full history
is 0.33 with 30 days, 0.029 with 60 days, 1.7e-3 with 90 days, and 9.8e-5 with 120 days, all above
the tolerance of 1e-5. The error is 5.0e-6 with 150 days, which passes, and 7.2e-7 with 180 days. The
bundle uses 180 days for margin, because the signal columns are already the tightest stage (7.6e-6
against 1e-5), and the exporter asserts that the history window covers 180 days.

**Finding 4: the two Copernicus Atmosphere Monitoring Service (CAMS) columns are written as 0.0 in
every stage.** In arm B of experiment 703, the Candidate reads no CAMS data at inference. K1, table T,
and the Candidate all see 0.0, so production needs no CAMS feed for this arm.

**Finding 5: the expected-weather store is part of the state.** The lag columns read the expected
weather at the lag's own half-hour. A half-hour with no run has a `run_origin_time` at or before
1970-01-01 (zero or negative epoch microseconds), and a zero row. The Candidate's expected-missing
flag is 1 for those half-hours. The bundle's `components/k1/expected_weather_rows.parquet` is the
store for the fixture origins.

**Finding 6: `time_series_id` is `Int32` in the contract.** The contracts package declares
`PowerForecast.time_series_id` as `Int32`, and the contract check asserts that dtype.

**Finding 7: the contract output has 11 quantile columns and no `ensemble_member` column.** The
output holds one quantile forecast per series and half-hour: `q01_mw` to `q99_mw` for the seed mean,
and `q01_s0_mw` to `q99_s1_mw` per seed. The output matches the form that [issue #1708 (Store
PowerForecast as 11 quantile columns instead of one row per ensemble
member)](https://github.com/openclimatefix/nged-substation-forecast/issues/1708) describes. The
ensemble members enter only as the inputs' member mean and standard deviation.

## Method

**Architecture.** Both neural networks take 60 base columns per Primary and half-hour: 7 weather
means, 7 weather spreads, 21 datetime columns, 2 lead columns, 3 day-lag columns, 3 week-lag
columns, 2 lag grid-index columns, a knot-position column, and 14 B1 columns (the exact slices are
in `config.json`). The weather encoder and the datetime encoder are each a two-layer perceptron
(29 inputs, 64 hidden units, 32 outputs) that also reads an 8-dimensional embedding of the Primary.
The lag encoder is a two-layer perceptron (64 hidden units, 32 outputs) that reads 83 inputs in K1
and 143 in the Candidate, where the extra 60 inputs are the signal columns. Two mixer perceptrons
(pre-training and fine-tuning) end in 9 raw outputs, which are the logits, means, and log standard
deviations of a mixture of 3 Gaussians. The quantiles at 11 levels (0.01, 0.05, 0.10, 0.25, 0.40,
0.50, 0.60, 0.75, 0.90, 0.95, 0.99) come from a bisection on the mixture's cumulative distribution,
and the contract output is the seed mean of the sorted quantiles.

**Training regime** (from `training_provenance` in `config.json`; the training code is experiment
574's and experiment 667's). The loss is the negative log-likelihood of the mixture, weighted per row
by recency, minimised with AdamW and early stopping on a validation negative log-likelihood. The
training window is 730 days with a recency half-life of 90 days, and the targets stop at the refit
time minus 3.5 hours (2026-09-30 20:30 UTC for the 2026-10 refit). One week in 12 is held out on an
84-day cycle, with a purge of 15 days before and 9 days after each held-out week. The fine-tune
embargo is 7 days, the step-2 embargo is 3 days, and the seeds are 0 and 1. The refit is arm B of
experiment 703, which sets the two CAMS columns to 0.0 in every stage.

**Measurement.**

- `run_export.sh` runs the three steps for the 2026-10 refit of arm `drop`. `e00708_capture.py` runs
  through experiment 667's `gpu.sh` on the stored chain of [experiment
  704](../00704-cams-free-inference-for-the-primaries-step-5-network/result.md) (`chain667.sh drop
  2026-10`): it reruns the prediction pass in a scratch root, builds the dense K1 history, and dumps
  the input matrices, outputs, and synthetic fixture. `e00708_export.py` writes the bundle on the CPU
  and asserts the reproductions above. `e00708_check.py` reloads the bundle and runs
  `reference_check.py` in a clean interpreter.
- The refit month is 2026-10 because the cached inputs of [experiment
  607](../00607-bakeoff-extend-cached-inputs-to-2026-10-31/result.md) end at 2026-10-31.
- The exporter refuses a dirty tree for the files of this experiment unless `--allow-dirty` is given.
- Every command reads cached inputs from experiment folders, and none reads the shared NWP table.
- The compute was one GPU capture on the RTX A6000 (one process, below 16 GB), then the CPU export
  and check on one core under a memory cap.

## Files

- Code: `ml-experiments/00708-production-export-bundle-for-the-primaries-candidate/`
    - `e00708_capture.py`: the GPU capture, with the dense K1 history and the synthetic fixture.
    - `e00708_export.py`: the CPU exporter.
    - `e00708_reference_check.py`: the standalone check, copied into the bundle as
      `reference_check.py`.
    - `e00708_check.py`: runs the bundle's check in a clean interpreter.
    - `run_export.sh`: runs the three steps.
    - `bundle_readme.md`: the bundle format, copied into the bundle as `README.md`.
- Imported code: experiments 87, 397, 428, 437, 510, 558, 560, 562, 572, 574, 578, 598, 667 (its
  `e00667_extend.py`, `e00667_cand.py`, and `gpu.sh`), and 704 (its `e00704_arm.py`). The scripts
  import them by `PYTHONPATH`, and never modify them.
- Data: `data/ml-experiments/00708-production-export-bundle-for-the-primaries-candidate/`, with the
  bundle in `bundles/drop/2026-10/`, the capture dump in `scratch/drop_2026-10/`, and the logs in
  `logs/` (`run_export_clean.out` is the clean-tree rebuild).
