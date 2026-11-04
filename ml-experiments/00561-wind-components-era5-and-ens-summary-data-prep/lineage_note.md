# Wind components and total cloud cover tables (experiment 561)

Built on 2026-11-04 by `ml-experiments/00561-wind-components-era5-and-ens-summary-data-prep/`.
Private; never committed.

## What is here

- `ens_summary_uv/init=YYYY-MM-DD.parquet`: the ensemble summaries of experiment 397's
  `ens_summary/` with four more weather columns. 821 files (runs 2024-04-01 to 2026-06-30),
  9,221,472 rows, 11,232 rows per run (16 Primaries). Keys `time_series_id`, `power_fcst_init_time`,
  `valid_time`; then `mean_<w>` and `std_<w>` (Float32, population standard deviation over the 51
  members) for the seven original columns and `wind_u_10m`, `wind_v_10m`, `wind_u_100m`,
  `wind_v_100m` (m/s, signed). The 14 original columns equal `ens_summary/` to a maximum absolute
  difference of 0.0.
- `ens_summary_tcc/init=YYYY-MM-DD.parquet`: `mean_total_cloud_cover` and `std_total_cloud_cover`
  (Float32, percent 0 to 100) with the same three keys. 222 files (runs 2025-11-21 to 2026-06-30),
  2,493,504 rows. `tcc_native/` holds the checkpoint it is built from: area-weighted cell values per
  member (7,698,960 rows, 8 H3 cells, 51 members).
- `data/studies/per_study/stage1_pilot/demand/inputs/era5_wind_components.parquet`:
  `time_series_id`, `valid_time`, `wind_u_10m`, `wind_v_10m`, `wind_u_100m`, `wind_v_100m` (m/s),
  `total_cloud_cover` (fraction 0 to 1), all Float32, on the half-hourly grid of
  `era5_features.parquet` (2,667,434 rows, identical keys, 2019-09-01 to 2026-07-31).

## Sources, licences, attribution

- ENS wind components: the production ENS archive (`data/NWP/nwp_model_id=ECMWF_ENS_0_25_degree`),
  ECMWF IFS ENS open data processed by Dynamical.org, CC-BY-4.0.
- ENS cloud cover: Dynamical.org store `ecmwf-ifs-ens-forecast-15-day-0-25-degree`, variable
  `total_cloud_cover_atmosphere`. Licence CC-BY-4.0. Attribution: "ECMWF IFS ENS forecast data
  processed by dynamical.org from ECMWF Open Data." Fetched 2026-11-04, about 6 MB per run, no
  charge.
- ERA5 wind components and cloud cover: the existing `ERA5/stage1_box/` tables (ARCO-ERA5, Google,
  `gs://gcp-public-data-arco-era5`; no new download). ERA5 is Copernicus data under the Copernicus
  licence (CC-BY-4.0 compatible): contains modified Copernicus Climate Change Service information;
  Hersbach et al. (2020), <https://doi.org/10.1002/qj.3803>.

## Rules and gotchas

- **Origin rule.** Every weather value for a target comes from the run issued at or before the
  forecast origin: `power_fcst_init_time` is the run's initialisation time plus 9 h, and every row's
  `valid_time` is after it. The build asserts this row by row and prints one example row
  (`nwp_init_time` 2024-04-01 00:00, `power_fcst_init_time` 09:00, `valid_time` 11:30). The ERA5
  table holds past weather (a pretraining input), not a forecast.
- **ENS cloud cover starts on 2025-11-21.** Dynamical.org holds all-NaN
  `total_cloud_cover_atmosphere` for runs 2024-04-01 to 2025-11-20 (checked by sampling every 14th
  run, then daily around the boundary). The Source Cooperative GRIB archive holds cloud cover for
  2021-03-21 to 2024-03-31 but ends the day before the open-data feed starts. No source covers
  2024-04 to 2025-11-20.
- **Units differ.** ENS cloud cover is a percentage (0 to 100); ERA5 cloud cover is a fraction (0 to
  1). Rescale before using both.
- **Cloud cover is interpolated linearly in time** between the native 3-hourly and 6-hourly steps,
  by the production resampler (under the placeholder name `wind_u_10m`, renamed back).
- **Cell aggregation.** The ENS cloud cover uses the production `h3_grid_weights` proportions over
  the 12 grid points of the 8 H3 cells that hold the 16 Primaries, renormalised over contributing
  points (none missing).
- The 14 original ENS summary columns are unchanged, so `ens_summary/` can be replaced by
  `ens_summary_uv/`; no model release uses cloud cover yet, and the production Delta table does not
  ingest it (a human-reviewed production change).

## Validation

`ens_summary_uv_validation.json`, `ens_summary_tcc_validation.json`, and
`era5_wind_components_validation.json` (beside the ERA5 table) hold the numbers: 0 null or NaN
cells, 0 duplicate keys, value ranges physically plausible (wind components within about -22 to 25
m/s, cloud cover 0 to 100 %), derived ERA5 wind speed equals the stored speed exactly, ERA5 keys
identical to the existing table.
