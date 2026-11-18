# Export bundle of K1, table T, and the Candidate demand network (format version 1)

**This directory holds everything that production needs to re-implement the inference of one
monthly refit of the Primaries Candidate.** The Candidate is arm `P7+D50` of research experiment
667, in the form of arm B of experiment 703: the two previous-day CAMS columns
(`cams_prev_day_mean`, `cams_prev_day_csi`) are 0.0 in every stage, so production never needs CAMS.
The bundle has three linked parts: K1 (the step-2 network of experiment 667's chain), table T (K1's
lags-masked forecasts), and the Candidate network, which reads signal columns built from the
cleaned power and table T. Both networks have two seeds (0 and 1). `manifest.json` names the refit
month, the training cut-off, the research commit, and the SHA-256 of every other file. The format is
that of experiment 705's step-5 bundle (same manifest, `config.json`, `primaries.json`, `components/`
and `fixtures/` layout), with `components/k1/`, `components/candidate/`, `fixtures/k1/`, and
`fixtures/candidate/` added and nothing in the step-5 layout changed.

**Production re-implements every feature, so the fixtures are the defence against skew.**
`reference_check.py` reproduces, with `torch`, `numpy`, and `polars` only: K1's lags-masked
medians, the newest table T entries, the Candidate's 60 base columns (from the raw columns) and its
signal columns (from the power history and table T), the raw network outputs, and the quantiles
per seed, as a seed mean, and at the 12:00 contract origin in megawatts. It does not rebuild the
`raw__` columns themselves (ENS summary, solar and calendar columns, B1 columns): production must
compare its own values with `input_matrix.parquet`. Run it with `python -I reference_check.py`
(inputs, table T, K1 medians, and raw outputs to 1e-5; quantiles to 2e-5 of capacity, because the
stored forecasts come from a GPU pass).

## Forecast recipe at one daily origin

1. **Build the 60 base columns for K1 and the Candidate** exactly as in step 3 of experiment 705's
   recipe (`columns.json`, `standardisers.json`, `config.json`). Both networks read the same 60
   columns. The two CAMS columns are 0.0.
2. **Run K1 with both lags masked** (value 0, flag 1, largest age code in the day-lag and week-lag
   column groups; the two lag grid-index columns stay) for each seed, read out the sorted 11
   quantiles, and keep q05, q50, and q95 as a fraction of capacity. **Store the q50 of the first 48
   half-hours** after the origin, per Primary and seed. Table T needs nothing else of K1.
3. **Rebuild table T.** `T[primary, g]` is the stored K1 median with the largest origin index `o`
   for which `0 < g - o <= 48`. Seed 0 of the Candidate reads the table of K1 seed 0, and seed 1
   reads the table of K1 seed 1.
4. **Build the signal columns** (`components/candidate/signal_columns.json` gives the 60 names in
   order; `reference_check.py:signal_block` is the reference). The gap at a grid index is the cleaned
   power minus `T`. Each row reads, at its day lag and at its week lag, the gap, its exponentially
   weighted averages with half-lives of 1, 3, and 7 days, the matching absent flags, the expected
   weather-missing flag, and the sum, mean, minimum, and maximum of the same quantities over the
   Primary's electrical neighbours (`neighbours.json`, scaled by the ratio of capacities), the
   Primary's own value plus the neighbour sum, the neighbour count, and a no-neighbour flag.
5. **Run the Candidate** (`forward` with `n_extra = 60`) for each seed, read out the 11 quantiles,
   sort them, average the seeds' contract-origin values in megawatts, sort again, relabel the origin
   from 09:00 to 12:00 UTC, multiply by `capacity_mw`, and keep 672 half-hours.

## State that production must keep between daily runs

**The Candidate cannot be computed from one day's inputs: it needs a store of the cleaned power and
a store of K1's earlier lags-masked forecasts.** Each store must reach back `history_window_days`
before the Candidate's origin (see `config.json`, `state_between_daily_runs`, and
`components.candidate.signals.history_window_rule`).

- **The cleaned power of every Primary, every half-hour, as a fraction of effective capacity**, with
  missing half-hours kept as missing. The lags already need the last 14 days of it; the signal
  columns need the longer window because the 7-day exponentially weighted average has a memory of
  weeks.
- **K1's lags-masked median for the 48 half-hours after each daily origin**, per Primary and seed,
  with the origin of each forecast. The Candidate reads table T only up to 24 h before its own
  origin, so it reads forecasts that are at least 2 days old.
- **K1 must therefore run every day**, with the same 60 base columns as the Candidate, even though
  the Candidate's output does not use K1's output on the day.
- **After a refit, or when the store has gaps,** the research rebuilds the whole table by running
  the refitted K1 over the history, which makes the training days' entries in-sample. A production
  store of live forecasts would hold each day's forecast from whichever K1 was live, and has no
  entries before the first day of operation. The bundle does not test that difference.

## Files

- `manifest.json`, `config.json`, `primaries.json`, `reference_check.py`: as in experiment 705's
  bundle. `config.json` has a `components.k1` and a `components.candidate` section (the latter adds
  `signals`, `table_t`, and `n_extra`) and a `state_between_daily_runs` section.
- `components/k1/`: `columns.json`, `standardisers.json`, `offset_knots.json`,
  `expected_weather_rows.parquet` (shared with the Candidate), and `weights/seed<N>.pt`, the
  `state_dict` of K1's `AuxLagMdn` (stage `ft`).
- `components/candidate/`: `columns.json` and `standardisers.json` (copies of K1's), `offset_knots.json`,
  `signal_columns.json` (names and order of the 60 signal columns), `neighbours.json` (each
  Primary's neighbour indices and all capacities in megawatts), and `weights/seed<N>.pt`, the
  `state_dict` of the Candidate's `PermLagMdn` (first lag-encoder layer widened by 60 inputs).
- `fixtures/candidate/origin=<date>/`: `input_matrix.parquet` (keys; `raw__` and `std__` for the 60
  base columns; `extra_s<seed>__<name>` signal columns per seed), `lag_and_expected_rows.parquet` (the
  336-half-hour lag window: power and expected-weather rows), `signal_history.parquet` (power and
  table T's median per seed over the history window), `ens_summary.parquet` (the member mean and
  standard deviation that the research run read), `outputs_native.parquet`, and
  `outputs_contract.parquet` (one quantile forecast per Primary and half-hour: `q01_mw` .. `q99_mw`
  for the seed mean, and `q01_s<seed>_mw` .. per seed; there is no ensemble-member column).
- `fixtures/k1/origin=<date>/`: `history_rows.parquet` (the standardised K1 input rows of the three
  origins that fill the newest table T entries, and K1's stored lags-masked q05, q50, and q95 per
  seed) and `table_t_entries.parquet` (table T entries and their source origins for the 144
  half-hours up to 24 h before the fixture origin).

## What the bundle does not check

The ENS fixtures of experiment 705 (native members and interpolated half-hours) are not repeated;
the recipe is in that bundle. The 14 B1 columns, the solar and calendar columns, and the ENS summary
are checked only against the fixture files.
