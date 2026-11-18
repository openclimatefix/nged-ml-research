---
id: "00708"
slug: production-export-bundle-for-the-primaries-candidate
title: >-
  Can production rebuild the Primaries Candidate's forecasts, with its K1 network and table T, from
  a plain directory of weights, standardisers, a config, and golden fixtures, without importing
  research code?
theme: production-reference-and-best-guess
also_relevant_to: []
status: finished
kind: other
size: >-
  Complex. (1) It stores one refit of the Candidate (arm B of experiment 703, refit 2026-10) and one
  export bundle; (2) the extension of experiment 705's bundle format to three linked networks (K1,
  table T, and the Candidate) has several defensible layouts; (3) it imports the chain of
  experiments 574, 591, 598, 613, 667, 703, 704, and 705, whose callers cannot be named without
  searching; (4) it touches no serving path and no degradation rule; (5) it changes no contract.
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
  A port of the Candidate must run K1 every day to keep table T dense, must keep 180 days of power
  history for the signal columns, and must write the two CAMS columns as 0.0; research's sparse
  table T differs from a dense one by at most 7.6e-5 of capacity.
take_home_message: >-
  Production can now rebuild the Candidate's forecasts for the Primaries from a folder of files and
  check its own K1, table T, and signal-column code against reference values. The port must run the
  K1 network every day to fill table T and must keep 180 days of power history. The sparse table T
  that the Candidate trained on differs little from the dense table that production will hold.
advice:
  - >-
    Hand production the Candidate as a bundle of plain files with golden fixtures per stage, and
    keep the standalone reference check in the bundle (Checks).
  - >-
    Run K1 at every daily origin and keep each lags-masked median in the table T store; the
    Candidate reads the freshest median that is at least 24 hours old (Finding 1).
  - >-
    Keep 180 days of power history for the signal columns, because a 150-day window leaves an error
    of 5.0e-6 on a tolerance of 1e-5 (Finding 2).
  - >-
    Write the two CAMS columns as 0.0 and return 11 quantile columns with no `ensemble_member`
    column (Findings 4 and 7).
production_status: candidate
adopted_in: none
---

**Status: finished.** This experiment is a tool for the port rather than a test of a change: it
exports the Candidate, the Primaries demand neural network of [experiment
667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md) (arm
`P7+D50`) in arm B form from [experiment
703](../00703-cams-free-inference-for-the-primaries-candidate/result.md), in the format that
[experiment 705](../00705-production-export-bundle-for-the-primaries-step-5-network/result.md) built
for the step-5 network. The hypothesis above is the acceptance test the code was written against,
and the plan was committed before the code (`cfa29c256`).

## The question

**The Candidate is not a single neural network: it reads the output of a second one.** K1 is the
step-2 network of experiment 667 (`AuxLagMdn`). Its forecasts, made with its lag inputs masked,
fill table T, and the Candidate (`PermLagMdn`) reads table T through 60 extra signal columns that
are built from the Primary's own power history, its neighbours' power history, and table T. Production
re-implements inference and never imports research code, so the bundle must carry both networks,
the table T rule, and the signal rules as files plus reference values. The experiment builds that
bundle for the 2026-10 refit, and proves with a standalone check that the files are complete.

## The bundle

**One bundle is one plain directory in format 1 of experiment 705, with a component folder per
network.** The bundle is 64 files and 71 MB (`bundles/drop/2026-10/`). It was exported from a clean
tree at commit `e5133a8a3`, and the manifest records that tree as clean. The format is in
`bundle_readme.md`, which the exporter copies into the bundle as `README.md`.

- `manifest.json`, `config.json`, `primaries.json`, and `reference_check.py` are as in experiment
  705.
- `components/k1/`: K1's 60 input columns in order, the standardisers, the offset-knot dates, the
  expected-weather rows (7.1 MB, shared with the Candidate), and one `state_dict` per seed (seeds 0
  and 1).
- `components/candidate/`: the Candidate's columns, the signal columns in order, each Primary's
  neighbour list, the standardisers, and one `state_dict` per seed (464 KB in all).
- `fixtures/k1/` and `fixtures/candidate/`: for each fixture origin, the inputs, K1's lags-masked
  quantiles, the table T entries that the Candidate reads, the input matrix raw and standardised,
  the raw network outputs, and the quantiles per seed, seed-averaged, and at the 12:00 UTC contract
  origin in megawatts. The Candidate's fixtures also hold the dense table T and K1's quantiles at
  every daily origin of the history window.

**Four things are new against experiment 705's bundle.**

- The bundle holds two networks that read each other's output, so `config.json` states the table T
  rule and the signal rules, and the manifest lists which component reads which.
- `reference_check.py` rebuilds K1's lags-masked medians and the whole of table T, then the
  Candidate's signal columns from the power history and table T, before it runs the forward pass.
- A sixth fixture, `origin=2026-10-03_synthetic_missing_inputs`, sets inputs missing on purpose.
- The contract check compares the full output frame, not only the megawatt values.

**The fixture origins are those of experiment 705's production bundle:** 2026-10-01, 2026-10-03,
2026-10-24, 2026-10-25 (the clock change), and 2026-10-31. At none of the five is a lag missing, so
the synthetic fixture is the only missing-lag case. It is built from 2026-10-03 with blanked spans
of power history, table T, and expected weather, and it has 1,722 week-lag-missing rows and 4,068
day-lag-missing rows of 10,848. The expected values of the synthetic case come from the research
code path, never from `reference_check.py`.

## The state production must keep between daily runs

**Production keeps four stores between daily runs.** The bundle's `config.json` lists them under
`state_between_daily_runs`, and the README describes each one.

- **Power history:** cleaned power per Primary and half-hour, with missing half-hours kept missing.
  The lags need the 336 half-hours that end 24 hours before the origin, which is about 8 days. The
  signal columns need 180 days.
- **Table T:** the lags-masked median of each K1 run, for every Primary and half-hour (Finding 1).
- **Expected-weather rows:** for each Primary and half-hour from 8 days before the origin to 24
  hours before it, the 28 expected-weather columns of the freshest ENS run that covered that
  half-hour. The lag columns read each row at the lag's own half-hour.
- **Neighbour lists:** one list per Primary, in `components/candidate/neighbours.json`.

## Checks

| Stage | Largest error, six fixtures | Synthetic fixture alone | Tolerance |
|---|---|---|---|
| K1 lags-masked median | 4.2e-7 | 0.0 | 1e-5 |
| K1 lags-masked q05 and q95 | 5.4e-7 | 0.0 | 1e-5 |
| Table T entries | 4.2e-7 | 3.9e-7 | 1e-5 |
| Candidate input matrix, 60 base columns | 0.0 | 0.0 | 1e-5 |
| Candidate input matrix, 60 signal columns | 7.6e-6 | 7.2e-6 | 1e-5 |
| Raw network outputs | 2.1e-6 | 1.7e-6 | 1e-5 |
| Quantiles per seed, stored input matrix | 1.1e-6 | 7.7e-7 | 2e-5 |
| Quantiles per seed | 1.7e-6 | 8.3e-7 | 2e-5 |
| Seed-mean quantiles | 8.3e-7 | 4.8e-7 | 2e-5 |
| Contract megawatts over capacity, per seed | 1.7e-6 | 8.3e-7 | 2e-5 |
| Contract megawatts over capacity, seed mean | 8.7e-7 | 5.1e-7 | 2e-5 |

**The reference check passes at all six fixtures and both seeds, in 23 seconds on a CPU.** The check
runs as `python -I reference_check.py <bundle>` and imports only the standard library, torch, numpy,
and polars. The K1 stages are checked at all 162,816 history rows, which are 212 daily origins from
2026-04-01 to 2026-10-29. Every origin before October is inside K1's training period, so the check
covers in-sample origins as well as the October ones. The signal columns have the least margin, 7.6e-6
against 1e-5, and the tolerance stays at 1e-5. The input matrix of the 60 base columns reproduces
exactly from the raw columns, the standardisers, and the cleaned power.

**The exporter's rerun equals the stored research output exactly.** Re-running the prediction pass of
the stored weights gives per-seed forecasts equal to the chain's stored forecasts (336,288 rows, a
largest difference of 0.0 for seeds 0 and 1), and a rebuilt table T equal to the stored table T
(424,801 filled entries, a largest difference of 0.0 for both seeds). The decision rule is met, so the
hypothesis is supported.

## The code review and what it changed

**The bundle's table T is dense.** The reference check rebuilds table T from this refit's K1 run at
every daily origin of the 180-day window. The fixtures and the reference values use that dense table.
Research's table T is sparse, because training rows, test rows, and the previous month's test rows
leave whole-day holes (held-out and purge windows). The exporter keeps the exact check that a rerun with
the sparse table equals the stored forecasts.

**The synthetic fixture exercises every missing-input branch.** Its flag counts are non-zero for the
missing own-day, own-week, neighbour, table T, and expected-weather cases of every signal group.

**The contract check compares the full frame.** It compares per-seed megawatts, `time_series_id`,
`valid_time`, `power_fcst_init_time` (the 12:00 relabel), `nwp_init_time`, and the dtypes.

**The first export ran before the code review.** The order departed from the `ml-experiment` skill,
which asks for a review before the first run, and the review's fixes led to a rerun of all three
steps.

## Findings for the port

**Finding 1: table T needs a K1 run every day.** `T[p,g]` is K1's lags-masked median from the freshest
origin `o` with `0 < g - o <= 48` half-hours, and the Candidate reads table T up to 48 half-hours
before its own origin. Production therefore runs K1 at every daily origin and appends the
median to the store. The K1 forecast must use its lags-masked read-out: lag columns 37 to 42 are set
to (0, 1, age maximum), the quantiles are sorted and scaled by the target standard deviation and mean,
and only q05, q50, and q95 are kept.

**Finding 2: the dense table T that production holds differs a little from the sparse one that the
Candidate trained on.** At the five fixture origins, the per-seed quantiles differ by at most 7.6e-5
of capacity (seed 0 at 2026-10-01), and the seed-mean quantiles by at most 4.4e-5. The mean difference
is 2.6e-6 or less. The signal columns differ by up to 0.017 at 2026-10-01 and by 0.0009 at 2026-10-31,
because the research table is empty for 27% of the 180-day window at 2026-10-01 and 15% at 2026-10-31
while the dense table is complete.

| Origin | Seed-mean quantiles, largest difference (fraction of capacity) | Seed-mean quantiles, mean difference |
|---|---|---|
| 2026-10-01 | 4.4e-5 | 2.6e-6 |
| 2026-10-03 | 3.9e-5 | 2.2e-6 |
| 2026-10-24 | 5.5e-6 | 3.1e-7 |
| 2026-10-25 | 4.3e-6 | 2.8e-7 |
| 2026-10-31 | 2.5e-6 | 1.7e-7 |

**Finding 3: the signal columns need a 180-day power history.** The 7-day exponentially weighted
average has a memory of weeks. The largest error in the signal columns against a full history is 0.33
with 30 days, 0.029 with 60 days, 1.7e-3 with 90 days, 9.8e-5 with 120 days, 5.0e-6 with 150 days, and
7.2e-7 with 180 days. The bundle uses 180 days, and the exporter asserts that the history window
covers 180 days.

**Finding 4: the two CAMS columns are written as 0.0 in every stage.** Production never needs CAMS
for the Candidate. K1, table T, and the Candidate all see 0.0 at inference, as in arm B of experiment
703.

**Finding 5: the expected-weather store is part of the state.** The lag columns read the expected
weather at the lag's own half-hour,
and the expected-missing flag is set when the weather origin is zero or less. The bundle's
`components/k1/expected_weather_rows.parquet` is the store for the fixture origins.

**Finding 6: `time_series_id` is `Int32` in the contract.** The contracts package declares
`PowerForecast.time_series_id` as `Int32`, and the bundle's contract schema says so. The first
export of this experiment wrote `Int8`, and the contract check now catches the dtype.

**Finding 7: the contract output has 11 quantile columns and no `ensemble_member` column.** The
output holds one quantile forecast per series and half-hour: `q01_mw` to `q99_mw` for the seed mean,
and `q01_s0_mw` to `q99_s1_mw` per seed. This matches the form of the forecast that production
issue #1708 describes. ENS members enter only as the inputs' member mean and standard deviation.

## Method

- `run_export.sh` runs the three steps for the 2026-10 refit of arm `drop`. `e00708_capture.py` runs
  experiment 704's chain (`chain667.sh drop 2026-10`, already stored) through experiment 667's
  `gpu.sh`: it reruns the prediction pass in a scratch root, builds the dense K1 history, and dumps the
  input matrices, outputs, and synthetic case. `e00708_export.py` writes the bundle on the CPU and
  asserts the reproductions above. `e00708_check.py` reloads the bundle and runs `reference_check.py`
  in a clean interpreter.
- The refit is arm B of experiment 703, with training targets up to 2026-09-30 20:30 UTC, because
  experiment 607's cached inputs end at 2026-10-31. Both seeds are 0 and 1.
- The exporter refuses a dirty tree for the files of this experiment unless `--allow-dirty` is given.
- Every command reads cached inputs from experiment folders and none reads the shared NWP table.
- Compute: one GPU capture on the RTX A6000 (one process, below 16 GB), then the CPU export and
  check on one core under a memory cap, inside slot C.

## Files

- Code: `ml-experiments/00708-production-export-bundle-for-the-primaries-candidate/`
    - `e00708_capture.py`: the GPU capture, with the dense K1 history and the synthetic case.
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
