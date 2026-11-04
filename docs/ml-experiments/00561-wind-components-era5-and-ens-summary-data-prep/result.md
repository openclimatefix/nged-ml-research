---
id: "00561"
slug: wind-components-era5-and-ens-summary-data-prep
title: >-
  Wind components and total cloud cover for the ERA5 pretraining table and the ensemble summaries
theme: demand-inputs
also_relevant_to: []
status: finished
kind: other
protocol: demand-v2
size: >-
  simple. Stored: three tables under experiment 397's private work folder and the ERA5 inputs
  folder; no fits and no forecasts. Serving path: not touched (this repository holds none).
  Degradation rule: not touched. Designs: one defensible. Callers: none yet; later experiments read
  the tables. Data only, no sub-agent review.
hypothesis: >-
  The wind components (east and north, at 10 m and 100 m) can be added to the ERA5 pretraining
  table and to the ensemble summaries from data already on disk, and total cloud cover can be added
  to both from ERA5 on disk and from the Dynamical.org ensemble store.
prediction: >-
  Written before the build. The ERA5 box holds all four components and cloud cover, so no ERA5
  download is needed. The Dynamical.org store holds cloud cover for every member and run from
  2024-04, so both tables cover the whole period.
decision_rule: >-
  No score is compared. The tables are accepted when every key matches the existing tables, no cell
  is null or NaN, no key is duplicated, ranges are physical, the old columns are unchanged, and the
  origin rule holds row by row.
benchmark: none (data preparation)
row_set_id: none
metric: none
verdict: partly supported
best_arm: not applicable
headline: >-
  All four wind components exist for the whole ERA5 table and for all 821 ensemble runs. Ensemble
  cloud cover exists only from 2025-11-21 (222 of the 821 runs), because Dynamical.org holds no
  values before that date.
lessons: >-
  Check a variable's availability by run date before planning an experiment on it. The first
  prediction failed for ensemble cloud cover only.
take_home_message: >-
  The wind direction inputs and ERA5 cloud cover are ready for every run and every hour. Ensemble
  cloud cover is ready only for the 222 most recent runs (2025-11-21 to 2026-06-30), so an
  experiment that uses it cannot train on the earlier ensemble history. The production table does
  not store cloud cover yet.
advice:
  - >-
    Check by run date that an ensemble variable holds values before planning an experiment on it
    (README_00561.md; the ensemble cloud cover is all-NaN before 2025-11-21).
  - >-
    Rescale ensemble cloud cover (percent) or ERA5 cloud cover (fraction) before using both in one
    network (README_00561.md).
production_status: not-for-production
adopted_in: none
---

**Take-home message.** The wind direction inputs and ERA5 cloud cover are ready for every run and
hour, and no download was needed. Ensemble cloud cover exists only from 2025-11-21, so it covers 222
of the 821 summary runs. Next step: experiments that use ensemble cloud cover train on that shorter
window, or use ERA5 cloud cover for the earlier history.

## What was built

- **ERA5 table:** `era5_wind_components.parquet` beside `era5_features.parquet`, with
  `wind_u_10m`, `wind_v_10m`, `wind_u_100m`, `wind_v_100m` (m/s) and `total_cloud_cover` (fraction),
  2,667,434 rows with the same keys as the existing table. The existing file is untouched.
- **Ensemble summaries with wind components:** `ens_summary_uv/`, 821 runs, 9,221,472 rows. The 14
  original mean and standard-deviation columns match `ens_summary/` exactly (maximum absolute
  difference 0.0).
- **Ensemble cloud cover summaries:** `ens_summary_tcc/`, 222 runs (2025-11-21 to 2026-06-30),
  2,493,504 rows, mean and standard deviation over 51 members in percent.

## Checks

- **No lookahead.** The build asserts row by row that the weather comes from the run issued 9 h
  before the forecast origin and that every valid time follows the origin. Example row: run
  2024-04-01 00:00 UTC, origin 09:00, valid time 11:30, member 31.
- **Missing values.** 0 null or NaN cells in every table; 0 duplicate keys.
- **Physical ranges.** Ensemble mean wind components lie between -22 and 21 m/s; ERA5 components
  between -22 and 25 m/s; cloud cover between 0 and 100 %. ERA5 wind speed derived from the
  components equals the stored speed exactly.
- **Ensemble spread of cloud cover** grows from 12.8 % at day 0 to 32 % by day 12, as expected.

## Lineage

- Licences, attribution, units, and gotchas are in
  `ml-experiments/00561-wind-components-era5-and-ens-summary-data-prep/lineage_note.md`.
- Ensemble cloud cover is not in the production Delta table. Ingesting it is a human-reviewed
  production change that this experiment does not make.
- Source Cooperative's GRIB archive holds cloud cover for 2021-03-21 to 2024-03-31 and ends before
  the missing 2024-04 to 2025-11 window, so no source fills that window.
- Egress was a few hundred megabytes in total (about 6 MB per ensemble run), at no charge.
- Code: this folder imports experiments 87, 48, and 397 (`PYTHONPATH` as `e00561_ens_summary_uv.py`
  shows).
