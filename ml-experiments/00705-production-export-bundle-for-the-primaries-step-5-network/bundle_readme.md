# Export bundle of the step-5 demand network (format version 1)

**This directory holds everything that production needs to re-implement the inference of one monthly
refit of the step-5 network at the 16 Primaries.** The network is arm `k1_5w_sp1` of research
experiment 677, with three seeds (0, 1, 2). `manifest.json` names the arm of the
CAMS-column experiment (`replay`, `proxy`, or `drop`), the refit month, the training cut-off, the
research commit, and the SHA-256 of every other file.

**Production re-implements every feature, so the fixtures are the defence against skew.** Each
fixture origin holds the raw weather inputs, every input column before and after standardisation,
the network outputs, and the quantiles, for all 16 Primaries. `reference_check.py` reproduces the
quantiles from the raw columns with `torch`, `numpy`, and `polars` only. Run it with
`python -I reference_check.py` (inputs and raw outputs to 1e-5, quantiles to 2e-5 of capacity,
because the stored forecasts come from a GPU pass).

## Forecast recipe at one daily origin

1. **Take the ENS 00Z run of the origin day.** The native origin is the run's initialisation time
   plus 9 hours (09:00 UTC). The contract origin is 12:00 UTC.
2. **Interpolate each of the 51 members alone to half-hours, then take the member mean and the
   population standard deviation (`ddof=0`).** The interpolation is `_upsample_nwp_to_half_hourly` of
   `ml_core` at the commit in `manifest.json` (`pinned_production_package_commit`). The 7 variables
   are listed in `config.json`.
3. **Build the 60 input columns** named in `components/step5/columns.json` for each (Primary,
   valid half-hour) with valid time up to 339 hours after the native origin, using
   `components/step5/standardisers.json`.
4. **Run the forward pass of `reference_check.py` for each seed**, read out the 11 quantiles, and
   sort them. Average the seeds' contract-origin values in megawatts, then sort again.
5. **Relabel the origin from 09:00 to 12:00 UTC, multiply by `capacity_mw`, and keep valid times
   from 12:00 UTC to just before 12:00 UTC 14 days later** (672 half-hours).

## Files

- `manifest.json`: format version, network name, arm, refit month and time, training cut-off,
  research git commit and dirty flags, pinned production-package commit, library versions, source
  paths on the research workstation, seeds, a `components` section, the export checks, and the
  SHA-256 of every file.
- `config.json`: every constant. `shared` holds the constants of the contract and the grid.
  `components.step5` holds the network's: input clip, lag rule and age codes, column slices, quantile
  levels, bisection range and steps, standard-deviation clamp, expected-weather rule, B1 constants,
  and the training provenance (half-life, embargoes, seeds). Training provenance is informational:
  inference does not read it.
- `primaries.json`: the 16 Primaries in `primary_index` order (0 to 15, the embedding index), with
  `time_series_id`, D-label, unit, H3 resolution-5 cell, latitude, longitude, `effective_capacity_mw`
  and `capacity_mw` (the two are equal; the target and the lags use the first, the output scaling
  uses the second).
- `reference_check.py`: the standalone reference forward pass and the check.
- `components/step5/columns.json`: the 60 input columns in order, with group and origin.
- `components/step5/standardisers.json`: mean and standard deviation of the 37 lag-free columns
  (float32 arithmetic, then clip at plus or minus 5), the day-lag and week-lag value statistics
  (float64), the 14 B1 columns' mean and standard deviation (float32, no clip), and the per-Primary
  target mean and standard deviation.
- `components/step5/offset_knots.json`: number of offset knots and the date of each knot. The values
  are in the weight files (`knots_old`, `knot_new`).
- `components/step5/expected_weather_rows.parquet`: the expected-weather rows that the fixtures read,
  raw and standardised, with the source run's origin time. The rule that builds every row is in
  `config.json` (`expected_weather.rule`).
- `components/step5/weights/seed<N>.pt`: the stored `state_dict` of class `AuxLagMdn` (stage `ft`),
  loadable with `torch.load(..., weights_only=True)`. `weights_index.json` lists every tensor's name,
  shape, and size, and marks the tensors that inference does not use (`mixer_pre.*`, `head.*`).
- `fixtures/step5/origin=<date>/`: one folder per fixture origin (native 09:00 UTC origin):
    - `ens_native.parquet`: the raw 51-member values at the native steps, before interpolation, for
      each Primary's H3 cell.
    - `ens_halfhourly.parquet`: the same members after interpolation, at half-hours, with wind speed
      and the lead in hours.
    - `ens_summary.parquet`: the member mean and standard deviation that the network reads
      (before the precipitation `log1p`), equal to the summary file that the research run read.
    - `input_matrix.parquet`: one row per (Primary, valid half-hour): keys, then `raw__<column>` and
      `std__<column>` for each of the 60 columns. For a lag, `raw__` holds the power as a fraction
      of effective capacity (NaN when missing).
    - `lag_and_expected_rows.parquet`: for each Primary, the 336 half-hours that the lags and the
      expected-weather table read at this origin: the cleaned power (fraction of effective capacity;
      NaN when missing) and the expected-weather row, raw and standardised.
    - `outputs_native.parquet`: nine raw network outputs and 11 quantiles per seed (fraction of
      capacity, native 09:00 origin, after the sort), and the seed mean after the sort.
    - `outputs_contract.parquet`: the 12:00 contract rows in megawatts: per seed, and the seed mean
      after the sort (672 half-hours per Primary).
    - `outputs_lags_removed.parquet`: the same quantiles with both lags set to the missing
      encoding (value 0, flag 1, largest age code), which the check reproduces from the standard
      matrix.

## Adding components

**A later bundle can add the K1 network and the Candidate without changing the format.** Each
network gets its own folder `components/<name>/`, its own `components.<name>` entry in
`manifest.json` (kind, seeds, file list), its own `components.<name>` section in `config.json`, and
its own `fixtures/<name>/`. The Candidate also needs the table T, the neighbour list, and the K1
step-2 weights; they become further files named in its manifest entry. Files that every component
shares (`config.json` `shared`, `primaries.json`) stay at the top level, and
`bundle_format_version` changes only when a file listed here changes meaning.

## Fixture origins

The refit month's fixture origins include a Saturday, the Good Friday bank holiday, and an origin
with a missing week lag at several Primaries. `manifest.json` lists them under `fixture_origins`
and gives the counts of rows with a missing lag under `export_checks`.
