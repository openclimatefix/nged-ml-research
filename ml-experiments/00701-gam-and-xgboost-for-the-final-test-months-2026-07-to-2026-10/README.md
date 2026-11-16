# Experiment 701: the XGBoost side (BG2 and its monthly-only variant)

**This section holds the scripts that forecast the 123 daily runs of 2026-07-01 to 2026-10-31 with
the XGBoost blend BG2 of [experiment
87](../00087-bg2-best-guess-assembled-with-drop-one-ablations/) and its monthly-only variant.** The
scripts reuse experiment 87's code unchanged. They import it through `PYTHONPATH`, and
`e00701_common.py` points its constants at this experiment's private folder. Nothing in the folders
or the data of experiments 87, 33, 34, 48, 607, or 700 is edited. The GAM side has its own section
(files `e00701_gam_*.py`).

All data go to `data/ml-experiments/00701-gam-and-xgboost-for-the-final-test-months-2026-07-to-2026-10/`
(called the sandbox below): `work/` for inputs, refit predictions, and arms, and `backtests/` for the
exported forecasts.

## Why the stored XGBoost files could not be reused

**Experiment 87 stops at 2026-06-30 in four places.** Its fit reads (1) experiment 48's demand frames
(`frames_demand`, 51 ensemble members of each run, 292 runs), (2) experiment 34's control-member
rows (`ens_after_training`, the training rows of the refits, 364 runs), (3) the solar offset's
ensemble-mean shape and anomaly, and (4) a power table that ends in October 2026 but a time grid
that ends 2026-07-31. It also hard-codes the last run (2026-06-30), the holdout (2026-07-01), and the
refit dates. The scripts here rebuild (1) to (3) for the new runs with the builders that made the
stored files, and move the constants.

## Files

- `e00701_common.py` patches experiment 87's `e00087_common`, `e00087_regimes`, and `e00087_offset`
  at import: experiment 607's observed power and ERA5 table, the grid end (2026-11-30), the runs
  (2026-07-01 to 2026-10-31), the holdout (2027-01-01), the frame and training-row folders, the
  sandbox outputs, and the refit schedule. Every script of this section imports it first, so the
  fit's worker processes are patched too.
- `e00701_build_inputs.py` builds and checks the inputs (subcommands below).
- `e00701_fit_demand.py` runs experiment 87's `e00087_fit_demand.py` through the patch.
- `e00701_smoke.py` refits one existing month and the frozen model and compares with the stored files.
- `e00701_assemble.py` links the stored and new refit files, then assembles the two arms with
  experiment 87's `e00087_assemble.py`, over the 365 stored and the 123 new runs.
- `e00701_export.py` writes the two forecast files in experiment 700's layout, using experiment
  700's `to_contract` and `write`.
- `e00701_xgb_run.sh` runs the fits, the assembly, and the export for `monthly` or `weekly`.

## Refit schedule of the new runs

- **Monthly refits (M, AN, PT, and the n1h monthly model):** 2026-07-01, 08-01, 09-01, 10-01, each
  at 00:00 UTC, serving every run of the month.
- **Weekly refits (W, the n1h weekly model):** every 7 days from 2026-06-30 to 2026-10-27, 18 refits.
  The refit of 2026-06-30 serves the runs 2026-07-01 to 2026-07-06. The refit of 2026-09-01 is also
  a monthly refit, and its file holds the union of both regimes' runs.
- **Frozen member (ST):** experiment 87 trained it once on rows to 2025-06-30 and saved no booster.
  It is refitted once, with the same rows, half-life, rounds, and seed, and forecasts the 123 new
  runs. `e00701_smoke.py frozen` compares the refit's forecasts of 2025-07-01 to 2025-07-03 with
  the stored ones.
- **A refit sees only rows whose valid time is at least 3.5 hours before it and whose run
  began at or before it.** The half-life is 0.5 years and the history is the whole of it (experiment
  87's rules, enforced by its `known_rows`).
- **Stored runs keep their stored forecasts.** A run on or before 2026-06-30 09:00 UTC keeps the
  refit that experiment 87 assigned to it (the weekly refit of 2026-06-23 serves the run of
  2026-06-30), so the assembly reproduces experiment 87's arms on those runs.

## Order of the build

Run every command from the repository root with these set:

```bash
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
export NGED_S3_BUCKET_URL=https://example.com NGED_S3_BUCKET_ACCESS_KEY=dummy NGED_S3_BUCKET_SECRET=dummy
m=ml-experiments
export PYTHONPATH=$m/00087-bg2-best-guess-assembled-with-drop-one-ablations:$m/00033-best-guess-architecture-vs-production:$m/00048-weekly-retraining-confirmation-and-extension-demand-generators:$m/00700-timing-run-gam-and-xgboost-14-day-monthly-refit:$m/00377-demand-v1-first-run-feasibility:$m/00701-gam-and-xgboost-for-the-final-test-months-2026-07-to-2026-10
```

1. **Inputs** (done; each is resumable, four worker processes, a minute or less):
   `e00701_build_inputs.py` with `frames-check`, `after-check`, `frames`, `after`, `links`, `u-mean`,
   `anomaly`, and `lags-check`, in that order.
2. **Smoke tests** (done): `e00701_smoke.py monthly` and `e00701_smoke.py frozen`.
3. **Monthly-only variant:** `ml-experiments/00701-*/e00701_xgb_run.sh monthly`.
4. **BG2 in full:** `ml-experiments/00701-*/e00701_xgb_run.sh weekly`.

The fits run as 6 worker processes with 12 CPU threads in all, sharing the GPU, as in experiment 700.
Each fit uses experiment 87's settings (500 rounds, `eta` 0.05, depth 6, 11 quantile levels).
`GPU hours logged so far: ... of 16` is experiment 87's own log, which the fit script prints; this
experiment's log is `work/gpu_hours.jsonl`.

## What was checked

- **Frames.** 15 stored runs of 2026-05 and 2026-06 were rebuilt from the ENS archive with
  experiment 48's builder and experiment 607's observed power. All 20 columns, 716,040 rows each,
  and every null position equal the stored frames, with a largest difference of 0.
- **Control-member rows.** 12 stored runs were rebuilt with experiment 33's builder and equal
  experiment 34's files exactly.
- **Solar offset inputs.** The rebuilt anomaly table equals experiment 87's stored table on all
  2,856,435 stored rows (largest difference 0) and adds 431,730 rows for the 123 new runs.
- **Timing rules.** On all 123 new frames, `power_fcst_init_time` is `nwp_init_time` plus 9 hours,
  every valid time is after the origin, and `origin_lags` (which raises if a lag is younger than the
  newest usable observation) passes on all 1,726,920 cases. The youngest observation any lag reads
  is 3.5 hours before its origin.
- **Assembly.** Assembling the 365 stored runs alone reproduces experiment 87's stored arms
  `p_s6_bg2` and `p_a3_monthly_in_weekly_slot` with a largest quantile difference of 0.
- **Export.** Exporting experiment 87's stored arms for 2026-06 reproduces experiment 700's two files
  on all 322,560 rows with a largest difference of 0 MW.
- **Refit.** The n1h monthly refit of 2026-06-01 (D2, D5, D9, and the offset twin of D9; runs
  2026-06-02 to 2026-06-04) differs from the stored forecast by a largest 0.014 and a mean 0.0007 of
  capacity over all 11 levels (median level: largest 0.010, mean 0.0009). The frozen model refitted
  at D2 and D9 reproduces the stored forecast of 2025-07-01 to 2025-07-03 with a largest 0.0068
  (D2 exactly, D9 mean 0.0002) and its offset twin at D9 with a largest 0.015, mean 0.0009.

## Limits

- **The capacity vintage of the solar offset is held at its last value.** Experiment 78's table of
  real-time capacities ends with the week of 2026-06-22, and extending it needs experiment 45's
  half-hourly inputs rebuilt to 2026-10. Experiment 87's own rule takes the last vintage for a run
  after the table's last week, so the offset at the 5 offset Primaries uses that value for July to
  September (the weight of the offset is zero from October). Over the last 8 weeks of the table the
  capacity of one series moves by up to 6% of its value.
- **The GPU fit is not bit-reproducible.** A rerun differs from the stored forecast by a largest
  0.014 of capacity and a mean of about 0.0007 in the smoke test.
- **The observed power ends at 2026-11-06 14:00 UTC in experiment 607's table.** Valid times of the
  last origins after that have no observation (the blend weights ignore them).

# Experiment 701: the GAM side (Primary demand research default 3)

**The scripts `e00701_gam_*.py` forecast the 123 daily runs of 2026-07-01 to 2026-10-31 with
Primary demand research default 3, the GAM, and write the forecasts in the layout of experiment 700's
`backtests/gam_year_sp1.parquet`.** The native 09:00 UTC origin is relabelled 12:00 UTC, as
experiment 700 did. The scripts import the code of experiments 103, 377, 378, 184, 202, 319, 320,
and 700 through `PYTHONPATH` and move the dates and paths from memory, in
`e00701_gam_extend.py` and `e00701_gam_extend377.py`. Nothing in `src/`, and nothing in an earlier
experiment's folder or stored data, is edited. The 15 Primaries are all except D15.

All GAM data go to `<data>/ml-experiments/00701-*/gam/`: `inputs/` (extended rows, series table,
sites, and combined CAMS file), `cache/` (the extended feature cache), `sandbox/` (copies of
experiment 700's sandbox, to which the stages append the new months), and `timing/` (logs). The
forecasts go to `backtests/gam_final_test_sp1.parquet`.

## GAM files, in the order they run

- `e00701_gam_inputs.py` rebuilds the rows of experiment 103 to 2026-10-31 with 103's own builder
  functions and checks that every shared row equals the stored row.
- `e00701_gam_cache.py` builds the extended feature cache and compares it with the stored cache.
- `e00701_gam_sandbox.py` makes the sandbox. It refuses to run if the sandbox exists.
- `e00701_gam_ens.py` has three subcommands: `check` (rebuilds 10 stored runs and compares), `build`
  (the 123 new runs of ensemble member arrays), and `gaps` (`NaN` and missing actuals per month).
- `e00701_gam_refit.py` refits the T-all GAM at the cuts 2026-07-01 to 2026-10-01
  (`--check-cut` refits a stored cut and writes nothing).
- `e00701_gam_stage1.py`, `e00701_gam_stage2.py`, and `e00701_gam_members.py` are experiment 700's
  stage scripts with `--first`, `--last`, and `--dry-run`. They replace the months' rows in the
  sandbox only.
- `e00701_gam_arms.py` runs experiment 378's S arm and daylight-blend arm (`low` and `long`).
  `--smoke MONTH` replays a stored month and compares.
- `e00701_gam_export.py` assembles the arms with the horizon thresholds 168 hours (S) and 240 hours
  (blend), then stage 2, and writes the contract file. `e00701_gam_check_export.py` compares the
  assembly with experiment 700's file for named Primaries and one stored month.
- `e00701_gam_run.sh` runs the refit, stages, arms, and export, at most 8 processes at a time.

## Commands

**Run each command from the repository root, after setting the variables of `CLAUDE.md`.** `DATA` is
`/home/jack/dev/nged-substation-forecast/data`.

```bash
export NGED_S3_BUCKET_URL=https://example.com NGED_S3_BUCKET_ACCESS_KEY=dummy NGED_S3_BUCKET_SECRET=dummy
E=ml-experiments
D=$E/00701-gam-and-xgboost-for-the-final-test-months-2026-07-to-2026-10
export DATA=/home/jack/dev/nged-substation-forecast/data
export PYTHONPATH=$E/00103-differentiable-physics-1-data-look:$E/00377-demand-v1-first-run-feasibility:$E/00700-timing-run-gam-and-xgboost-14-day-monthly-refit:$D
# 1. inputs and cache (real data folder), then the checks of the ensemble arrays
DATA_PATH_INTERNAL=$DATA uv run --no-sync python $D/e00701_gam_inputs.py
DATA_PATH_INTERNAL=$DATA uv run --no-sync python $D/e00701_gam_cache.py
DATA_PATH_INTERNAL=$DATA uv run --no-sync python $D/e00701_gam_sandbox.py
DATA_PATH_INTERNAL=$DATA uv run --no-sync python $D/e00701_gam_ens.py check
DATA_PATH_INTERNAL=$DATA uv run --no-sync python $D/e00701_gam_ens.py build
DATA_PATH_INTERNAL=$DATA uv run --no-sync python $D/e00701_gam_ens.py gaps
# 2. the four months (sandbox folder)
DATA_PATH_INTERNAL=$DATA/ml-experiments/00701-gam-and-xgboost-for-the-final-test-months-2026-07-to-2026-10/gam/sandbox \
  bash $D/e00701_gam_run.sh
```

Steps 1 are already done when the folders of `gam/inputs`, `gam/cache`, and `gam/sandbox` exist.
Each stage replaces its own months, so `e00701_gam_run.sh` can be run again after a failure.

## What was checked

- **Rows:** the extended rows table equals the stored table on all 1,624,499 shared rows, in 13
  columns (0 differences).
- **Cache:** the extended feature cache equals the stored cache array by array (largest absolute
  difference 0.0, no `NaN` on one side only).
- **Ensemble arrays:** the builder reproduces 10 stored runs at all 15 Primaries bit for bit. The 123
  new runs hold no `NaN`.
- **Refit:** a refit at a stored cut reproduces the stored fit to about 1e-11 in the GAM's
  coefficients (0.0 when refitted on the stored cache), so the difference is the environment.
- **Stages and export:** stage 1, stage 2, the member responses, the long-horizon arm, and the
  export reproduce experiment 700's June 2026 rows with a largest absolute difference of 0.0.

## Limits

- **Missing actuals:** a power outage on 2026-11-04 from 08:30 to 14:00 UTC removes 128 half-hours
  from the windows of the 11 October origins that reach it, at every Primary. July and August have
  gaps at some Primaries. `e00701_gam_ens.py gaps` lists them.
- **Capacity vintages:** the pinned PV capacity vintages end at the Monday 2026-08-03 origin, so the
  September and October cuts of the seven signed-MW Primaries hold the 2026-08-03 capacity.
- **Switching mask:** it is empty after 2026-07, because the labelled set has no later events.
- **Weather:** the ERA5 of September and October is the preliminary ERA5T.
- **Scoring:** a scorer must use the extended feature cache and the holdout dates of this
  experiment (the cache folder of `e00668_compare_primaries` and `eval_protocol.metrics.HOLDOUT_START`).
- **The run of 2026-06-30:** the GAM now forecasts it, which experiment 700 did not, but the export
  keeps origins from 2026-07-01 only.
- **Runtime:** experiment 700 measured about 15 minutes of single-thread GAM time per origin month
  (887 seconds), so the four months are expected to take 1 to 2 hours at 8 processes. The ensemble
  build is separate and is not included in that estimate.

## Scoring side

**The reference arms of the new runs are built by `e00701_refs.py`.** The `power_forecasts` Delta
table, which experiment 87 reads, ends with the run of 2026-06-30, so the manual heuristic and the
climatology of the 123 new runs are built from the production package's rules (`refs/arms/`). The
manual heuristic reproduces experiment 87's stored rows key for key, with quantiles within 1e-4 of
capacity, because the production forecasts hold the observed power on a coarser grid. The
climatology is the production lookup that the stored forecasts hold (the climatology of experiment
700, fitted on power before 2025-07-01), read for the new months, and reproduces the stored rows bit
for bit.

**Scoring runs in this order, once the three model files exist:** `e00701_ladder.py`,
`e00701_protocol.py`, `e00701_extra.py`, `e00701_report.py`, `e00701_plots.py`. The scripts need the
folders of experiments 377, 378, 668, 677, 700, and this one on `PYTHONPATH`. `--smoke` scores June
2026 from experiment 700's files, to `scores/smoke_protocol_final/`, and reproduces experiment 700's
per-row scores exactly.
