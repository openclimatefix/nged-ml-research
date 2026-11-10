# Final models: architectures and training regimes for the production port

**This page states, for each forecast type, which neural network the one-year backtest used and how
it was trained.** It is written for the person who ports the models into `nged-substation-forecast`.
Each section covers one forecast type and has the same blocks: architecture, training regime,
post-processing, scores, code, input tables, and port notes. Most choices link to the experiment
page that decided them. The page adds no new scoring. Where two pages disagree, the page says so
and names the current one.

**No neural network on this page is adopted for production.** Every score comes from one backtest
year that overlaps the development months. The backtest is described in [experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md):
one forecast origin per day at 12:00 UTC from 2025-11-01 to 2026-10-31, and 672 half-hours (14 days)
per origin. The origin months from 2026-07 lie in the window that earlier experiments kept as a
final test, so those months are descriptive and no choice rests on them. The experiment 677 page,
which builds up the Candidate demand neural network one component at a time, is not merged. Its slot
in the Primaries section is marked.

**Where to start.** A porter should read, in order, the "Production code that exists today" section,
the "Output contract" section, the section of the forecast type to port, the "Open issues for the
porter" section, and the "Open questions" list. The "Tests to write" section lists the checks that a
port should pass before any score is compared. Each forecast-type section has a "Code" subsection
that names the research files that produced the neural network, with the pull request that last
changed each, and the production code that already does the same job. Each forecast-type section
also has an "Input tables" subsection that names the cached tables the fit reads, the script that
builds each table, and the order to run those scripts. The research folders are experiment folders
under `ml-experiments/`, and their scripts import modules from earlier experiment folders by file
name. The appendix lists those imports for each forecast type, and it covers imports only: a script
that writes a cached table but is imported by no fit appears in the "Input tables" subsections and
nowhere in the appendix. The production repository has no neural-network code and no `torch`
dependency, so every neural network on this page is new code there. Backtest files are private and
live in `data/ml-experiments/backtest_year_2025-11_to_2026-10/`, outside git.

**Four other pages hold rules that the neural networks follow, and this page does not repeat them.**
The [data-processing page](data-processing.md) states how the observed power, the weather, and the
series are cleaned and which dates get a regime flag. The three evaluation-protocol pages,
[demand](evaluation-protocol-demand.md), [wind](evaluation-protocol-wind.md), and
[solar](evaluation-protocol-solar.md), state the scoring rules for each forecast type. The
[glossary](glossary.md) defines the project's terms.

**Nine terms recur on this page, and the definitions below fix their meaning.** NGED is National
Grid Electricity Distribution. A Primary is a primary substation, a BSP is a bulk supply point, and
a GSP is a grid supply point. MVA is apparent power, which cannot be negative, and MW is real power.
The Candidate is the demand neural network that outputs the NRA forecast. The prevailing output is a
second forecast that allows for a load shift still in progress at the forecast origin, made by a
small neural network called the survival head, which estimates how likely the shift is to persist. A
calibrated copy of a backtest file is a copy whose quantile bands are rescaled (see "Walk-forward
band calibration"). The lag-enabled neural network reads recent observed power. CRPS is the
continuous ranked probability score, computed here as twice the mean pinball loss over the 11
quantile levels, divided by capacity, times 100. A CRPS of 4 therefore means 4% of capacity. The
normalised MAE (mean absolute error) of the median is the mean absolute error of the 50% quantile in
the same units. Smaller is better for both. The q05-to-q95 coverage is the share of observed power
inside the band from the 5% quantile to the 95% quantile, and its nominal value is 0.90. NRA is the
normal running arrangement, the electricity network configuration without any load shift. ECMWF is
the European Centre for Medium-Range Weather Forecasts, ENS is its ensemble forecast of 51 members,
ERA5 is its reanalysis of past weather, and CAMS is the Copernicus Atmosphere Monitoring Service. A
load shift is a transfer of demand between neighbouring Primaries, usually for days to weeks, that
NGED's logs record incompletely.

**The sections that follow, in order.** "Overview" tabulates the six outputs. "Series, identifiers,
and units" fixes the series names. "Production code that exists today" and "Output contract"
describe the production repository and the file format that the backtest used. One section per
forecast type follows: the 16 Primaries, the BSPs and GSPs, wind, and solar. Then come "Walk-forward
band calibration", "Production-port notes across forecast types", "Open issues for the porter",
"Where the pages disagree", "Open questions", "Tests to write", and the appendix.

## Overview

| Forecast type | Series | Neural network and output | Post-processing in the backtest files | Headline CRPS, days 0 to 1 and days 10 to 14 |
|---|---|---|---|---|
| Primaries, NRA output | 16 Primaries | The Candidate, a 3-component Gaussian mixture, 11 quantiles | Quantiles of the 8 MVA Primaries clamped at zero. No calibration. | 3.84 and 4.59 |
| Primaries, prevailing output | 16 Primaries | The Candidate's quantiles mixed with a shifted copy by a survival head with 14 extra lagged inputs | Same clamp. Walk-forward band scale in the calibrated copy. | 3.37 and 4.47 |
| BSPs | 2 BSPs | Lag-enabled mixture neural network, one per group | Walk-forward band scale in the calibrated copy | 4.27 and 6.41 |
| GSPs | 2 GSPs | Lag-enabled mixture neural network, one per group | Walk-forward band scale in the calibrated copy | 10.52 and 14.78 |
| Wind | 3 wind farms | Neural network S, a 3-component Gaussian mixture on ensemble weather | None | 4.58 and 9.86 |
| Solar | 6 solar farms | Neural network PLP, a 3-component Gaussian mixture on ensemble weather and physics-lite power | Night quantiles set to zero. Walk-forward band scale in the calibrated copy. | 4.83 and 7.18 |

The scores in the last column come from [experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md).
Section "Scores" of each forecast type gives the other lead bands, the coverage, and the climatology
and manual-heuristic references.

## Series, identifiers, and units

**The backtest files identify a series by its original `time_series_id`, and the experiment pages
mix that identifier with NGED's D-labels.** Pages [experiment
683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md)
and [experiment 688](00688-winter-normal-running-coverage-median-error-and-events-part-1/result.md)
write bare identifiers such as "Primary 13" for identifier 13, which is the Primary labelled D18.
The table fixes the correspondence for the 16 Primaries. The mapping comes from `site_map.csv` of
experiment 521, which the scoring code of [experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
reads.

| D-label | `time_series_id` | Unit | D-label | `time_series_id` | Unit |
|---|---:|---|---|---:|---|
| D2 | 25 | MVA | D12 | 14 | MVA |
| D3 | 3 | MVA | D14 | 16 | MW |
| D4 | 11 | MVA | D15 | 12 | MW |
| D5 | 17 | MW | D16 | 18 | MW |
| D8 | 26 | MVA | D18 | 13 | MW |
| D9 | 7 | MW | D19 | 9 | MVA |
| D10 | 10 | MW | D21 | 24 | MVA |
| D11 | 8 | MW | D22 | 15 | MVA |

- **Primaries.** Eight Primaries are metered in MW (D5, D9, D10, D11, D14, D15, D16, and D18) and
  eight in MVA, which is apparent power and cannot be negative (D2, D3, D4, D8, D12, D19, D21, and
  D22). D15 and D18 have a short history, and the backtest cuts their rows before 2025-03-01, as
  every recent experiment does.
- **BSPs and GSPs.** D13 Skegness is `time_series_id` 2 and D17 Boston is 1. D6 Walpole is 27 and D1
  Bicker Fen is 28. All four are signed MW series, so a negative value is net export.
- **Generators.** The three wind farms and the six solar farms are anonymised as Wind A to C and
  Solar A to F. This page lists no generator identifier, capacity, coordinate, or dated series.
- **Capacity.** Every score is divided by the series' effective capacity, the production
  effective-capacity table with one value per series. That capacity is a normalising constant and
  not a limit, for demand and for generators alike. The highest quantile reaches 2.1 times the table
  value at D15, 2.2 times at the GSP D1, 1.4 times at the GSP D6, and 1.35 to 1.7 times at five more
  Primaries (D5, D12, D16, D18, and D22). The wind and solar quantiles exceed it by up to 12%
  (experiment 668, Caveats). The files keep these quantiles on purpose, because the maintainer
  decided to leave quantiles above capacity uncapped.

## Production code that exists today

**The production repository (`openclimatefix/nged-substation-forecast`, read at commit `a0e66b220`
of `main`) has a model interface, an XGBoost forecaster, and orchestration, and none of the neural
networks, quantile output, or post-processing on this page.** Paths below are relative to that
repository. The sections for each forecast type say which of these pieces apply.

- **Model interface.** `packages/ml_core/src/ml_core/base_forecaster.py` defines `BaseForecaster`
  with the class constants `MODEL_NAME`, `CONFIG_CLASS`, and `feature_engineer`, and the abstract
  `train`, `predict`, `save`, `load`, and `trained_time_series_ids`. Both `train` and `predict` take
  a `pt.LazyFrame[AllFeatures]`, and `AllFeatures` is keyed by `ensemble_member`.
  `BaseForecasterConfig` carries `selected_features`, `ml_flow_experiment_id`, `experiment_name`,
  `weather_source`, `training_strategy`, `random_seed`, and `model_release`. `predict` stamps
  `power_fcst_model_name`, `power_fcst_model_version` (from `model_release`), `experiment_name`,
  `ml_flow_experiment_id`, and `fold_id` on every row. The interface allows per-series or pooled
  models.
- **Existing forecasters.** `XGBoostForecaster` (`packages/xgboost_forecaster`,
  `conf/model/xgboost.yaml`) trains one `xgb.Booster` per `time_series_id` on 24 features and a
  squared-error objective. `ManualHeuristicForecaster` and `ClimatologyForecaster` are in
  `packages/baseline_forecasters`. No forecaster branches on whether a series is a Primary, a BSP, a
  GSP, a wind farm, or a solar farm. `torch` appears in no `pyproject.toml`.
- **Output contract.** `PowerForecast` (`packages/contracts/src/contracts/power_schemas.py`, line
  414) holds one `power_fcst` per `ensemble_member`, with the primary key (`time_series_id`,
  `power_fcst_init_time`, `valid_time`, `ensemble_member`). It has no quantile columns, although
  `ClimatologyForecaster` encodes quantile levels as member ranks (member k stands for the level
  (2k+1)/102), and it has no column that separates a normal-running output from a prevailing output.
  `DELIVERY_QUANTILES` in `packages/contracts/src/contracts/common.py` lists 13 levels (0.01, 0.02,
  0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98, and 0.99) and is used for metric
  labels. The research files hold 11 quantiles (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75,
  0.90, 0.95, and 0.99). `TimeSeriesMetadata.units` is `MW` or `MVA`, and `power_fcst` carries the
  series' own unit. The next section maps the backtest file columns onto these fields.
- **Weather.** `Nwp` (`packages/contracts/src/contracts/weather_schemas.py`, line 121) stores ECMWF
  ENS only, at H3 resolution 5, with members 0 to 50, written by
  `packages/delta_store/src/delta_store/nwp.py`. Production does not store ERA5, CAMS, or UKV. No
  feature code computes the mean and standard deviation over the 51 members, although
  `packages/ml_core/src/ml_core/metrics.py` computes ensemble-mean metrics. The trained model sees
  the control member, and every member is scored separately at predict time.
  `_attach_containing_nwp_cell` in
  `packages/ml_core/src/ml_core/features/tabular_feature_engineer.py` maps each series to its H3
  cell.
- **Analysis proxy.** The production analysis proxy is the control member (member 0) of the freshest
  NWP run that holds a value for each valid time, filled column by column from the next-freshest run
  where a cell is null. It is `select_analysis_proxy` in
  `packages/weather_utils/src/weather_utils/analysis_proxy.py`, and the weather lags of the feature
  pipeline use it. The research analysis proxy is different: it is the ensemble mean of the freshest
  run that had been published by the forecast origin, which is the first day of the previous 00Z run
  (`ml-experiments/00437-*/e00437_features.py`, module docstring). The two definitions differ in the
  member (control against ensemble mean) and in the fill rule, so the production function does not
  reproduce the research columns.
- **Power inputs.** `packages/ml_core/src/ml_core/features/_lags.py` builds power and weather lags,
  and `_nullify_leaky_lags` nulls a power lag that is shorter than or equal to the lead time. The
  production code has no constant for the 6-hour power delay. `NWP_PUBLICATION_DELAY_HOURS` is 9
  (`packages/ml_core/src/ml_core/features/_nwp.py`). `scan_cleaned_power` in
  `packages/nged_data/src/nged_data/storage.py` reads only rows without a `drop_reason`, and
  `flag_nged_power` in `cleaning.py` has one rule, `substation_zero`.
- **Capacity.** `EffectiveCapacity` (`power_schemas.py`, line 605) is a static 99th percentile of
  the absolute power per series over the whole cleaned history, written by the `effective_capacity`
  asset in `src/nged_substation_forecast/defs/cv_assets.py`. It is a metric denominator, and
  `AllFeatures` has no capacity input.
- **Orchestration.** `trained_cv_model` and `cv_power_forecasts` in `defs/cv_assets.py` train and
  score on the folds of `conf/cv/default.yaml`, on demand. `promoted_model` in
  `defs/promotion_assets.py` copies a chosen MLflow run to the production model path.
  `live_forecasts` in `defs/live_forecast_assets.py` loads the promoted model with
  `load_forecaster_from_dir`, runs at 00:00, 06:00, 12:00, and 18:00 UTC over a 14-day horizon
  (`LIVE_FORECAST_HORIZON`), and writes with `write_power_forecasts`. The `ecmwf_ens` asset ingests
  the 00Z run from 10:30 UTC and retries every 30 minutes, up to 8 times, when the run is not yet
  complete (`defs/schedules.py`), so the 06:00 UTC slot uses the previous day's run, and the 12:00
  UTC slot does too when the day's run is late. No schedule refits a model.
- **Post-processing.** The production code has no clamp, calibration, or band-widening step.
  `power_fcst` is the raw output of `predict`.
- **Release record.** `docs/live_service/model-releases.md` records a numbered release (`model-N`),
  stamped as `power_fcst_model_version`. A release needs a cross-validation run named
  `model-release-N`, a promotion, a git tag, and a log row. The log has no rows yet.

## Output contract

**The backtest files have a fixed layout, and a golden test of a port needs every property below.**
The sources are the "File format" and "Known limitations" sections of the README in
`data/ml-experiments/backtest_year_2025-11_to_2026-10/`, the file schemas read from the stored
files, and the checks of [experiment
667](00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md) and
[experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md).

| Backtest column | Type and rule | Nearest production `PowerForecast` field |
|---|---|---|
| `time_series_id` | Int8, the original identifier | `time_series_id`, Int32 |
| `valid_time` | UTC microsecond timestamp, the end of the half-hour, so the lead-0 row covers the half-hour that ends at the origin (11:30 to 12:00) | `valid_time`, constrained to be strictly after `power_fcst_init_time` |
| `nwp_init_time` | UTC, the 00Z ENS run that the forecast used | `nwp_init_time` |
| `power_fcst_init_time` | UTC, always 12:00 | `power_fcst_init_time` |
| `q01` to `q99` | 11 Float32 columns (the BSP and GSP files were recast from Float64 on 2026-11-10) in the order 0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99, in the series' own unit | No quantile column; `power_fcst` is one value per `ensemble_member` |
| `NRA_or_prevailing` | String, `NRA` or `prevailing`; `primaries.parquet` only | No field |
| none | Model identity and run identity are not in the files | `power_fcst_model_name`, `power_fcst_model_version`, `experiment_name`, `ml_flow_experiment_id`, `fold_id` |

- **Columns and order.** The five files hold the columns in the order of the table's first five
  rows, and `primaries.parquet` ends with `NRA_or_prevailing`. The solar, wind, BSP, and GSP files
  have no such column.
- **Rows.** Each origin has 672 rows per series, at leads of 0 to 335.5 hours in half-hour steps,
  and `primaries.parquet` holds one row of each kind (`NRA` and `prevailing`) per key. The last
  half-hour of a forecast ends 30 minutes before the 14-day mark. The files hold 7,848,960 rows for
  the 16 Primaries (16 series, 365 origins, 672 half-hours, and 2 outputs) and 490,560 rows each for
  the BSPs and the GSPs.
- **Sort order and compression.** Each file is sorted by `time_series_id`, `power_fcst_init_time`,
  then `valid_time`, and `primaries.parquet` has a fourth key, `NRA_or_prevailing`. The Parquet
  metadata does not declare the order. Every column uses ZSTD compression.
- **Checks the files pass.** No duplicate key, no null or non-finite quantile, no crossing in any
  adjacent quantile pair, and UTC throughout (experiment 667, "Checks").
- **Two conflicts with the production schema, stated as facts.** The production `PowerForecast`
  constrains `valid_time` to be strictly after `power_fcst_init_time`, and the lead-0 row of the
  backtest files has `valid_time` equal to the origin. The research code also has no
  `ensemble_member` axis, because each row is a set of 11 quantiles and not a deterministic forecast
  per member. Neither conflict is settled on any page read for this one.

## Demand at the 16 Primaries

### Architecture

**The Primaries are forecast by the Candidate, a modular demand neural network that outputs 11
quantiles, plus a small prevailing head.** The Candidate is arm `P7+D50` of [experiment
612](00612-nra-combined-networks-three-class-roster-completion/result.md) and [experiment
613](00613-nra-roster-contrasts-with-intervals-five-seeds/result.md) (P7 is the neighbour power
summaries and D50 a 50% dropout of the signal columns, described under "Neighbour and gap signals"),
which [experiment 620](00620-bakeoff-final-test-demand-networks-at-primaries/result.md) tested on
the final-test months. The Candidate's output is the NRA output. The prevailing output comes from
the survival-hazard head of [experiment
652](00652-two-output-forecast-with-explicit-rules-and-survival-hazard/result.md), with the 14 extra
lagged inputs of [experiment
681](00681-extra-lagged-power-inputs-for-the-prevailing-forecast/result.md). [Experiment
667](00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md) is the first
backtest of the two outputs, and [experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
restates every detail below and is the current page where the two differ.

- **One row** is a (Primary, ensemble run, valid half-hour). The neural network runs once per
  ensemble run, on summaries of the 51 members. The weather is the ECMWF ensemble (ENS) 00Z run of
  the origin day, which the neural network treats as available at 09:00 UTC. The forecast is made at
  that native 09:00 UTC origin and relabelled to the origin of the production forecast contract,
  12:00 UTC, by adding 3 hours. Valid times before 12:00 UTC are dropped. The lead runs to 339
  hours, because ENS summaries reach about 351 hours.
- **Weather inputs.** The mean and the standard deviation (`ddof=0`) over the 51 members of 7
  variables: shortwave and longwave radiation, 2 m temperature, 2 m dew point, surface pressure,
  precipitation (after `log1p` of the value clipped at zero), and 10 m wind speed. All 14 columns
  are z-scored with the training rows' mean and standard deviation and clipped at plus or minus 5.
  Each series reads the weather of the H3 resolution-5 cell (a hexagonal grid cell) that holds it.
- **Calendar and sun inputs.** 21 columns: the cosine of the solar zenith angle, the sine and cosine
  of the azimuth, the sine and cosine of the local hour (Europe/London) at periods of 24 hours and
  12 hours, a one-hot day of week, a COVID flag, three Christmas columns, and three bank-holiday
  columns. The COVID flag is 1 from 2020-03-23 to 2021-07-19 inclusive. The three Christmas columns
  are a flag for a valid day within 14 days of Christmas, the signed days to Christmas over 14, and
  one minus the absolute value of that ratio. The three bank-holiday columns are a flag for a bank
  holiday, the signed days to the nearest bank holiday (clipped at 7 days) over 7, and a flag for a
  nearest holiday under 7 days away (`e00087_common.py`, `e00397_prep.py`). The lead enters as the
  lead in hours over 336 and its square root. The 7 means, the 7 standard deviations, the 21
  calendar and sun columns, and the 2 lead columns make the 37 "lag-free" columns, which the lag
  columns below extend.
- **Thermal-lag and solar-gain inputs.** 14 columns from [experiment
  437](00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles/result.md) and
  [experiment
  569](00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1/result.md):
  trailing temperature means (3, 6, and 24 hours), temperature changes (3 and 6 hours), the day's
  minimum and maximum, the count of cold-snap days, trailing shortwave means (3, 6, and 24 hours),
  the 3-hour change in the clear-sky index, and two columns from the previous day's CAMS irradiance.
  Before the origin they use the analysis proxy, a stand-in for observed weather. The research proxy
  is the ensemble mean of the freshest run published by the origin, which is the first day of the
  previous 00Z ensemble run (`e00437_features.py`). It uses no ERA5, because ERA5 arrives about 5
  days late. The production proxy in `packages/weather_utils/src/weather_utils/analysis_proxy.py`
  differs: it takes the control member (member 0) of the freshest run and fills each column from the
  next-freshest run where the freshest run is null. A port needs the ensemble-mean version.
- **Power lags.** A day lag and a week lag. Each is the latest same-half-hour reading at least 24
  hours before the neural network's own 09:00 origin, entered as a z-scored value, a missing flag,
  and an age code. The age code is the hours from the lag's time to the target time, over 336. A
  missing lag takes the largest age code (24 hours times the ceiling of 375 over 24, over 336 for
  the day lag, and 168 hours times the ceiling of 375 over 168, over 336 for the week lag, where 375
  is the longest lead of 351 hours plus 24). The 8 lag columns are the value, missing flag, and age
  code of each lag, plus the half-hour grid index of each lag's time (columns 43 and 44). Those two
  indices are not network inputs: they address the expected-weather table, which holds the
  standardised ensemble-mean weather (7 columns) and datetime (21 columns) at every grid time, taken
  from the freshest run issued at or before that time (`e00558_lags.py`, `e00560_model.py`). A knot
  position, the position of the valid time along the monthly offset knots, is one further input
  column (column 45). The newest lag is therefore the 09:00 reading of the day before the origin
  day, 27 hours before the contract's 12:00 origin, so the production contract's rule that power
  inputs be at least 6 hours old needed no change.
- **Neighbour and gap signals.** 60 columns read only by the lag encoder. For each lag: the gap
  between the observed power and a lag-free forecast table (the median of the lags-masked neural
  network of the same seed at its freshest earlier origin with a lead of at most 24 hours). The
  signals are exponentially weighted means of that gap with half-lives of 1, 3, and 7 days. Over the
  electrical neighbours of the Primary, the sum, mean, minimum, and maximum of those quantities are
  scaled by capacity ratio, and the own-plus-neighbour sum, a neighbour count, and a flag for no
  neighbour are added. The neighbours are the metered Primaries that NGED's "Primary Substation
  Interconnections" list links to the Primary ([experiment
  610](00610-nra-neighbour-summary-lag-encoder-inputs-iteration/result.md)). The list is two NGED
  files in `NGED/substation_hierarchy_2026-10-15/` under the research data folder. `Monitor
  Direction.csv` maps each `TimeSeriesID` to a `SubstationNumber`, and `Primary Substation
  Interconnections.csv` holds the pairs `PrimarySubstation1` and `PrimarySubstation2`. Two Primaries
  are neighbours when an interconnection pair joins their substation numbers (`nged_neighbours` in
  `e00400_blend.py`). The capacity ratio divides by `capacity_mw` of `site_map.csv`, read through
  `sites()` in `e00578_common.py`. During fine-tuning, all signal columns of a random 50% of rows
  are set to no information (the "D50" dropout).
- **Series identity.** An 8-dimensional embedding of the Primary. The target is cleaned power
  divided by effective capacity, then z-scored per Primary.

**The encoders and the mixer have fixed widths.** The weather encoder (7 means, the 8-wide
embedding, and the 14 thermal-lag and solar-gain columns, 29 inputs) and the datetime encoder (21
columns and the embedding, 29 inputs) each map to 64 hidden GELU units and then 32 outputs. One
shared lag encoder runs twice, on the day lag and then the week lag. Its 143 inputs are the lag's 3
columns, the 32-wide embeddings that the weather and datetime encoders give for the weather expected
at the lag's own time, the 8-wide embedding of the Primary, an 8-wide embedding of the lag's
identity, and the 60 signal columns. (The lag encoder of the K1 network that the Candidate is
grafted onto has 83 inputs, because it has no signal columns; see "Training regime".) It maps to 64
and then 32, and the two outputs are concatenated. The mixer takes 145 inputs (the weather and
datetime outputs of 32 each, the 64 lag outputs, the embedding, the seven spread columns, and the
two lead columns), has two hidden layers of 64 units with 10% dropout, and outputs 9 numbers: the
logits, means, and log standard deviations of a 3-component Gaussian mixture. No merged page states
the parameter count of the Candidate. The weights and biases of the linear layers named above add up
to 33,385 (4,000 for each of the weather and datetime encoders, 11,296 for the lag encoder, and
14,089 for the mixer). The embeddings and the offset knots (16 Primaries times the number of monthly
knots) add to that figure. The stored state dicts were not opened for this page, so a port should
assert its own count against a stored `.pt` file.

- **Offset.** A per-Primary piecewise-linear function of time, with one knot for each month start
  from 2019-09, is added to the three component means ([experiment
  565](00565-drift-embedding-variants-six-designs/result.md), [experiment
  572](00572-two-offset-part-1-lags-versus-the-slow-output-offset/result.md)). Fine-tuning trains
  only the newest knot, and a penalty of 10 times the mean squared difference between neighbouring
  knots keeps the function smooth.
- **Quantiles.** The 11 quantiles come from the mixture's cumulative distribution by 40 bisection
  steps on a range of plus or minus 40 standard deviations. The mixture weights are a log-softmax
  and each standard deviation is `exp` of the raw output clamped to plus or minus 3. The quantiles
  are rescaled to power and sorted within the row.
- **Loss.** The negative log-likelihood of the mixture on the standardised target, weighted, plus
  the knot penalty.

**Truncating the output distribution at zero is an option for the Primaries that cannot go negative,
and only [experiment
684](00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/result.md)
has tested it.** The backtest files carry a post hoc clamp instead (see "Post-processing").
Experiment 684 refitted the Candidate with the mixture conditioned on power at or above zero, so the
negative log-likelihood adds the log of the mixture's mass above zero and the quantiles are read
from the conditional cumulative distribution. It tested the family at four Primaries, D12, D22, D8,
and D5 (identifiers 14, 15, 26, and 17), on the test months 2026-02 and 2026-05, and on the NRA
output only. D5 is metered in MW and never goes negative in the observed power. The CRPS changed
against the stored forecasts by +0.030 (D12), -0.015 (D22), -0.070 (D8), and -0.010 (D5), with no
negative q01 left. The verdict was partly supported and the experiment's own status is candidate. A
gamma mixture trained inside the same shared neural network failed its stop rule at two Primaries
and was dropped, and it also worsened the Primaries that kept a Gaussian output. Whether the
truncated family helps the other five MVA Primaries (D2, D3, D4, D19, and D21), and the prevailing
output, was not tested.

**The prevailing output comes from a second small neural network, the head R-haz, which mixes the
Candidate's quantiles with a shifted copy of them.** The Candidate's 11 quantiles define a
piecewise-linear cumulative distribution with two tail knots, 1.5 times the outer spacing beyond the
1% and 99% quantiles. The prevailing distribution is the mixture `(1 - p)` times the Candidate's
distribution plus `p` times the Candidate's distribution shifted by a learned depth, with the depth
uncertain at three Gauss-Hermite points (defined under "Layers and outputs" below; [experiment
647](00647-two-output-forecast-departure-head-on-frozen-k1/result.md), with the form restated in
experiment 652 and experiment 668). The 11 prevailing quantiles are read off that mixture and are
monotone by construction. When `p` is 0, the prevailing rows equal the NRA rows. R-haz sits on the
frozen Candidate and has these parts:

- **Inputs, 93 in all.** The head reads 58 origin features (the gap and neighbour signals above,
  built from lags), 4 age columns (the log of the departure's age in days, that age over 30, a gate
  flag that is always set to zero, and the age of the freshest lag), and 3 row inputs (the lead over
  240 hours and the lower and upper half-widths of the Candidate's band). Together these make 65
  columns. The 14 extra lagged power inputs of experiment 681, each as a residual plus a missing
  flag, add 28 columns.
- **The 14 extra lagged inputs.** Every time is measured back from the neural network's origin
  (09:00 UTC), and every residual is the cleaned power minus the Candidate's lag-free median
  forecast for that time, as a fraction of capacity. The inputs are the residuals at 6, 9, 12, 18,
  24, 36, 48, and 72 hours, the means of the residuals over the last 6, 12, and 24 hours (each
  window ending 6 hours before the origin), and three differences between the means of four adjacent
  24-hour windows. A mean is missing when under half of its window holds a reading. The newest
  reading any input uses is 6 hours old, and a test perturbs every later half-hour and checks that
  no input moves.
- **Layers and outputs.** Two hidden layers of 64 ReLU units and 16 outputs: a signed depth (0.5
  times the hyperbolic tangent of the first output, so up to half of capacity), a spread, and 14
  daily hazards. The spread is `0.01 + 0.05 * softplus(output)` of the second output. Each hazard is
  `softplus(output - 3)`. The weights of the last layer are multiplied by 0.1 and its bias is set to
  zero at initialisation. The survival probability `p` is the exponential of minus the hazards
  accumulated over the days before the lead, and the share of day `d` that counts is `clamp(lead /
  24 - d, 0, 1)` for lead in hours. The value of `p` is 0 when the departure is younger than half a
  day. The shifted copy sits at the depth plus the spread times each of the three Gauss-Hermite
  nodes. Those nodes are -√3, 0, and √3 with weights 1/6, 2/3, and 1/6, which is the standard
  three-point rule for averaging over a normal distribution. The inputs are standardised with the
  mean and standard deviation of the training rows (the standard deviation floored at 0.001) and
  clipped at plus or minus 10. The head has 11,216 parameters (`e00647_head.py`, `p667_head.py`,
  `e00647_fit.py`).
- **The backtest files use the 14-input extra-lag set, and which set to port is undecided.**
  Experiment 681 found that 14 extra inputs (set L3) make the prevailing output react 20.4 hours
  sooner (95% interval 16.3 to 24.8) to the start of an injected load shift, and 8 inputs (set L1)
  19.1 hours sooner. The verdict was a small gain, under the 24 hours set in advance as worth
  porting. Only two real events could be read, and the 8-input set scored worst inside the real
  events. [Experiment
  687](00687-causal-detector-features-for-the-prevailing-forecast-heads/result.md) measured a larger
  gain for extra lagged power (1.25 days sooner than the head of experiment 652) on a different bank
  of injected events and different folds, so the two reaction times are not comparable. The 14-input
  set was adopted for the backtest files by nged-ml-research PR 353 (Replace the prevailing rows of
  the Primaries backtest file with the extra-lag head).
- **The fallback head B-haz.** With fewer than two earlier banks of injected shifts (the folds
  2025-11 and 2025-12; a bank is the set of injected-shift rows built from one earlier test month,
  described under "Training regime"), the prevailing rows come from B-haz: the Candidate plus its
  own recent residual times a survival curve fitted on synthetic durations, with no fit on observed
  power. The fit tries every distribution family in `s652.DISTS` (experiment 652's list), weights
  each injected shift's duration, and keeps the family with the lowest negative log-likelihood. The
  log-normal won in both folds, with a shape of 1.147 and a scale of 21.3 days for 2025-11, and
  1.142 and 22.1 days for 2025-12. The survival is conditional on the departure's age `a` in days:
  `S(a + 1 + lead / 24) / S(a + 1)`, with `a` clipped to 0.5 to 100 and `p` set to 0 below 0.5 days.
  The residual is the mean of observed power minus the Candidate's median over the 2 to 8 days
  before the origin, using forecast rows with a lead of 24 to under 48 hours. The prevailing
  quantiles are `sort(q + residual * p)`, where `q` holds the Candidate's quantiles and the residual
  is 0 where none exists (`p667_fit.py`, `e00652_score.py`, `e00647_data.py`). The prevailing rows
  equal the NRA rows on 74% of November origins and 65% of December origins, and on no origin from
  2026-01. B-haz has no inputs that the extra lagged power could widen.

**Experiment 687 tried causal step-detector features for the prevailing head and did not adopt
them.** The features reacted 1.59 days sooner than the head of experiment 652 (95% interval 1.00 to
2.26 days sooner), but extra lagged power alone reacted 1.25 days sooner, so the features are not
shown to beat plain lags (experiment 687). The prevailing head of the backtest therefore carries no
detector features.

!!! note "Pending experiment 677"

    Experiment 677 builds up the Candidate one component at a time, from the date-time encoder to
    the full Candidate, with reference lines from a generalised additive model and an XGBoost model.
    Its plan is in the open nged-ml-research PR 338 (Experiment 677: building up the Candidate
    demand network one component at a time (plan)). This slot will state which components the
    Candidate needs and what each adds once the page merges.

### Input tables

**Every fit of the Candidate reads cached input tables that earlier experiments built, and the
appendix does not list the scripts that build them.** The appendix follows `import` statements only,
so it cannot find a script that writes data without being imported. Paths that start with `00` are
under `$DATA_PATH_INTERNAL/ml-experiments/`, the other data paths are under `$DATA_PATH_INTERNAL/`,
and `<site>` is a Primary's D-label.

| Table | Path | Built by | Source data |
|---|---|---|---|
| Ensemble summaries: mean and standard deviation over the 51 members, one file per run (944 runs, 2024-04-01 to 2026-10-31) | `00607-*/work/ens_summary/init=<date>.parquet` | `00607-*/e00607_build.py uv`, which calls `00561-*/e00561_ens_summary_uv.py` (a copy of the build of experiment 397 that adds wind components); `e00607_build.py link` hard-links the earlier runs | ECMWF ENS 51-member archive, read through `e00048_build_frames` |
| ERA5 features, 11 by 11 cells | `00607-*/work/era5_features.parquet` | `e00607_fetch_era5_box_cds.py` (13 variables, Climate Data Store) and `e00607_fetch_ptype_arco.py` (precipitation type, ARCO-ERA5) extend the stored box to 2026-10-31, then `e00607_derive_era5.py` runs `studies/stage1_pilot/demand_build_inputs.py`'s feature builder on the merged box | ERA5 |
| Observed power on the half-hourly grid (cleaned, from 2019-09, with lags) | `00607-*/work/observed.parquet` | `e00607_build.py observed`, which calls `observed_power` in `studies/stage1_pilot/demand_build_inputs.py`; the original is `studies/per_study/stage1_pilot/demand/inputs/observed.parquet` | NGED power telemetry |
| CAMS irradiance at the Primaries | `studies/weather/CAMS/demand_points/cams_demand_points*.parquet`, combined in `00620-*/inputs/cams_demand_points_extended.parquet` | `00607-*/e00607_fetch_cams_demand.py` (2026-09-11 to 2026-10-31), and `00667-*/e00667_tables.py` for the combined file | CAMS |
| Raw input matrices per Primary (feature matrix, power over capacity, valid and origin times, lead) | `00667-*/inputs/raw/<site>.npz` | `00667-*/e00667_extend.py build` (then `check`), which reads the three tables above | Ensemble summaries, observed power, solar and calendar columns |
| Thermal-lag and solar-gain columns per (run, valid time) | `00667-*/inputs/trailing437/<site>.parquet` | `00667-*/e00667_tables.py build` (then `check`, `coverage`, `causality`), which calls `site_table` of `00437-*/e00437_features.py` | Ensemble summaries and CAMS |
| Expected-weather table (standardised ensemble-mean weather and datetime columns at every grid time) | `00667-*/inputs/w560/expected/expected_{ft,pre}_{cols,origin}.npy` | Built on first use by `raw_table` in `00560-*/e00560_data.py` | Raw input matrices (source `ft`); ERA5 and CAMS pseudo-run matrices of experiment 418 (source `pre`) |
| Stub frames for runs after 2026-06-29 | `00667-*/inputs/frames/init=<date>.parquet` | Created on demand by `e00667_fold.py` | The run's own ensemble-summary table |
| The lag-free forecast table T | `00667-*/tables/T_<month>_s<seed>.npz`, and `tables/predict/` for the prediction phase | `chain.sh` steps `table` and `ptable` | K1's step-2 network |
| Electrical neighbours | `NGED/substation_hierarchy_2026-10-15/Monitor Direction.csv` and `Primary Substation Interconnections.csv` | NGED files, read by `nged_neighbours` in `00400-*/e00400_blend.py` | NGED |

- **Run order.** First the three tables of experiment 607 (`e00607_build.py uv`, `tcc`, `observed`,
  and `link`, then the two ERA5 fetches and `e00607_derive_era5.py`, then the CAMS fetch). Then
  `e00667_extend.py build`, then `e00667_tables.py build`. The expected-weather table appears at the
  first fit. Then `chain.sh` runs per fold, and the prevailing scripts follow (see "Training
  regime"). The stage `tcc` of `e00607_build.py` writes ensemble cloud-cover summaries that the
  Candidate does not read.
- **Experiments 598, 606, and 610.** These experiments supply the code for table T and the signal
  columns (`e00598_signals.py`, `e00606_*.py`, `e00613_nbsignals.py`). The `tables/` folders that
  those three experiments stored are not read by experiment 667: `table_file` in `e00667_cand.py`
  always points at this experiment's own `tables/`, because the tables of another experiment come
  from another chain.
- **Experiment 560.** The weather table of the lag encoder and the lag-index code come from
  experiment 560's modules (`e00560_data.py`, `e00560_model.py`).
- **Open question.** Whether `capacity_mw` in `site_map.csv`, which sets the neighbour capacity
  ratio, equals the effective-capacity table of the production repository for the 16 Primaries was
  not checked. How the 3-hourly and 6-hourly ensemble steps are turned into half-hourly weather
  columns in the code of experiments 397 and 561 was not traced for this page.

### Training regime

**Each of the 12 refits of the Candidate, one per test month, is a chain of four stages that sees
only data from before its month: the K1 network, table T, the Candidate's fine-tune, and a
prediction pass.** The folds are the test months 2025-11 to 2026-10. [Experiment
667](00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md) and
[experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
give the settings below, which come from [experiment
545](00545-modular-lag-free-demand-network-weather-and-datetime-encoders/result.md) (the modular
design), [experiment
559](00559-m0-leak-ladder-three-step-schedule-frozen-encoders-and-penalty/result.md) (frozen
encoders and the three-step schedule), [experiment
418](00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens/result.md) (ERA5 and CAMS
pre-training), [experiment
574](00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/result.md) (the
embargoed schedule), and [experiment
599](00599-nra-refit-embargo-and-down-weighting-of-the-newest-rows-against-load-shift-absorption/result.md)
(the refit rule).

- **The chain.** `chain.sh MONTH` runs six steps in this order. `pretrain` and `step2` run K1
  through `e00667_k1.py`. `table` builds table T. `fit` is the Candidate's fine-tune with the
  held-out power hidden. `ptable` rebuilds table T over the scored window, and `pfit` is the
  Candidate's prediction pass that writes the forecasts. K1 is a separate network. `e00667_k1.py` is
  a copy of `e00613_arm.py` (experiment 613), which wraps arm A1 of `e00591_arm.py` (experiment
  591): a shared lag encoder of 83 inputs with a lag-identity embedding and no signal columns, no
  auxiliary loss, the design 5 offset of experiment 565, the synthetic lag-only load shifts of
  experiment 584's design C1, and the 7-day embargo of experiment 599's design E3. `e00667_k1.py`
  asserts these settings.
- **Grafting the Candidate onto K1.** The Candidate has 60 signal columns that K1 lacks.
  `PermLagMdn.load_state_dict` (`e00613_nbmodel.py`) pads K1's first lag-encoder layer with 60
  zero-weight columns, so the Candidate starts as K1 and learns its signal weights only in the
  fine-tune. The 143-input lag encoder therefore does not pre-train: the 83-input encoder of K1
  does.
- **Two phases.** In the `train` phase (`E00667_PHASE=train`) the cleaned power is NaN from the
  first day of the fold month, so no code path can read a held-out reading, and the scored window
  stays closed. In the `predict` phase the window is open: table T is rebuilt over the scored window
  plus the month before it, the standardiser is checked against a hash stored in the train phase
  (`check_fingerprint`), and the network fitted in the train phase is reloaded. A port should hide
  all power after the cut-off in every refit, and test that rule.
- **Fold mechanics.** `e00667_fold.py` patches experiment 574's pipeline to one cut-off per month
  (the first day of the month, at 00:00 UTC) and sets the longest lead to 339 hours. The control
  member's frames stop at the run of 2026-06-29, so a stub frame holding the run time stands in for
  later runs, and `e00667_extend.py` rebuilds every stored row from 2025-07-01, because the stored
  matrices dropped the half-hours that have no cleaned power.
- **Data.** Cleaned power, with the `substation_zero` half-hours dropped (the one cleaning rule the
  backtest applied), and the D15 and D18 histories cut before 2025-03-01. The ENS history starts on
  2024-04-01, so the early folds fine-tune on less ensemble data than the later folds.
- **Step 1, pre-training (K1).** The weather, datetime, and 83-input lag encoders of K1 pre-train on
  ERA5 and CAMS (shortwave) as stand-in weather, in pseudo-runs every 2 days from 2019-09-01 at
  09:00 UTC with leads of 0.5 to 351 hours (about 8.1 million rows for the 2026-07 fold). The
  pre-training rows end at least 3.5 hours before the refit time. Five Primaries start later (D11 on
  2019-09-17, D10 on 2019-12-01, D5 on 2020-02-01, D14 on 2020-07-01, and D16 on 2021-02-01). ERA5
  has no ensemble, so the seven spread columns hold the fold's median ensemble spread for each
  24-hour lead band. All weights train, with AdamW, learning rate 0.0005, weight decay 0.001, batch
  8,192, 20% of the rows per epoch, up to 60 epochs, patience 8, and gradient-norm clip 1.0.
- **Step 2, fresh mixer (K1).** The pre-trained weights load except the mixer. A fresh mixer is
  drawn with seed 12345, the weather and datetime encoders are frozen, and training runs on the
  730-day history with uniform weights and a 3-day embargo, at learning rate 0.0002 (0.0005 for the
  fresh mixer).
- **Table T.** The `table` step runs K1's step-2 network with its lags masked, and stores the median
  of each (Primary, grid time) at the freshest earlier forecast origin with a lead of at most 24
  hours (`masked_quantiles` and `build_lag_free_table` in `e00598_signals.py`). The signal columns
  and the 14 extra lags of the prevailing head are differences between observed power and table T.
- **Step 3, daily fine-tune (the Candidate).** The fine-tune starts from the step-2 weights, padded
  as above, with the same frozen encoders and learning rate 0.0002. Row weights are 0.5 raised to
  the power (age in days divided by 90), a 90-day half-life. The refit rule drops rows whose valid
  time lies in the newest 7 days before the cut-off and halves the weight of rows 7 to 14 days old.
  Early stopping holds out one week in every 12 weeks of origin days, with 15 days purged before and
  9 days after each held week, and uses at most 200,000 held rows. The ensemble weather is the input
  at this stage and at inference.
- **Seeds and widening.** Two seeds are averaged quantile by quantile and sorted again. The backtest
  applies no widening. The final test of [experiment
  620](00620-bakeoff-final-test-demand-networks-at-primaries/result.md) widened the band about the
  median by a factor of 1.0934, the mean of the five development-month factors of [experiment
  613](00613-nra-roster-contrasts-with-intervals-five-seeds/result.md), and experiment 668 states
  that it averaged five seeds.
- **Missing and delayed inputs.** A flagged or absent lag reads as value 0, a missing flag, and the
  maximum age code. In training, 30% of origin days mask both lags and, of those, 15% mask only the
  day lag. Absent signal columns read as 0 with a missing flag. Neighbour-signal columns whose
  training spread is under 80% of their spread in the real inputs of the month before the fold
  receive matching Gaussian noise during training. All weather columns must be finite.
- **Injected load shifts.** Each Primary receives random synthetic shifts, added to the lag inputs
  only, until they cover 5% of its row span: depth 15%, 30%, or 50% of capacity, length 3, 7, 15, or
  30 days, and random sign. Each shift on a Primary is subtracted, scaled by the capacity ratio,
  from one electrical neighbour, so the neighbour summaries see the transfer.
- **Banks.** Each month's R-haz trains on "banks" of injected-shift rows, one bank per earlier test
  month, built from the real history of that month with synthetic shifts and forecast by that
  month's Candidate chain (inference only), so fold M's head trains only on banks of months before M
  and on rows whose valid time is before the start of M. Each fit samples 60,000 real rows and 6,000
  bank rows. Origin days 0 to 17 of an injected shift train the head and origin days from 21 onward
  validate it. The lag inputs of an injected shift are built from the same shifted power as its
  target, and the lag-only shifts of the bank use the month's later power for proportional shifts
  ([experiment
  667](00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md), Caveats),
  which stays within the months before the fold being scored.
- **Duration mix.** The injected shifts have durations drawn half from 3 to 14 days and half from 28
  to 112 days, an assumed mixture with none from 15 to 27 days, and each injected shift counts
  equally.
- **Loss and optimiser.** The loss is the pinball loss over the 11 levels plus 0.05 times the binary
  cross-entropy of `p` against the observed state of an injected shift. The optimiser is Adam with
  learning rate 0.003, weight decay 0.0001, batch 4,096, up to 30 epochs, patience 4, and two seeds
  averaged by mean quantile. The fit samples with a generator seeded `1000 * seed + yyyymm % 997`,
  and each epoch shuffles with a generator seeded `seed * 100 + epoch`.
- **Hardware and run time.** About 0.5 GPU-hours per fold for the Candidate chains (experiment 667).
- **Capacity.** The effective capacity of the production table normalises the target. The neural
  network does not cap its output at that capacity.

### Post-processing

**The backtest file of the Primaries has one post hoc clamp, and the calibrated copy adds a scale
for the prevailing rows only.** Both act on stored quantiles after the neural networks run. The band
scale leaves the median unchanged. They are not the only steps that act on predicted quantiles: the
prevailing head (R-haz or B-haz) rebuilds the prevailing quantiles from the Candidate's quantiles,
and the seed average sorts the quantiles again. Those steps belong to the model chain above, and
"Port notes" below states what the chain means for a serving path.

- **Clamp at zero.** Every quantile of the eight MVA Primaries is at least zero, because apparent
  power cannot be negative. The clamp changes 206,487 of 7,848,960 rows (2.6%), in D12, D22, and D8
  only, and the largest change is 2.80 MVA. It is monotone, so no quantile crosses. The MW Primaries
  keep their negative lower quantiles, and D5 and D18 never go negative in the observed power yet
  have lower quantiles below zero in 703 and 6 rows ([experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md),
  Caveats). The clamp came from nged-ml-research PR 342 (Clamp night-time solar and MVA Primaries
  quantiles in the one-year backtest files (experiment 668 page)). The body of that pull request
  counts 202,598 changed rows, and the 668 page counts 206,487 for the current file.
- **Walk-forward band scale (calibrated copy only).** See "Walk-forward band calibration" below. It
  applies to the prevailing Primaries rows and not to the NRA rows. The scale of arm 2 raised the
  NRA coverage to 87.1% and the NRA CRPS by 0.081 points of capacity ([experiment
  683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md)).
- **Not applied.** The widening factor 1.0934 of [experiment
  613](00613-nra-roster-contrasts-with-intervals-five-seeds/result.md), and any cap at capacity.
  Experiment 683 tested the fixed factor 1.0934 on the NRA rows as its arm 1: the NRA coverage rose
  from 82.4% to 84.5% (stored) to 86.0% to 87.6%, each a range over the four lead bands, and the
  CRPS fell by 0.027 points of capacity (95% interval -0.031 to -0.023). The file does not apply it.

### Scores

**Both outputs beat the manual heuristic and climatology at every lead band, and the prevailing
output scores better than the NRA output overall.** The table gives CRPS in points of capacity,
pooled over the 16 Primaries, from [experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
(3,895,784 scored half-hours per output). The manual heuristic is NGED's lagged-power rule and
climatology is the month, half-hour, and weekend-flag quantiles of the series' own past power, both
as defined on that page.

| Lead band | NRA CRPS | Prevailing CRPS | Manual heuristic CRPS | Climatology CRPS |
|---|---:|---:|---:|---:|
| Days 0 to 1 | 3.84 | 3.37 | 4.89 | 5.94 |
| Days 1 to 3 | 3.96 | 3.60 | 4.89 | 5.98 |
| Days 3 to 7 | 4.23 | 4.00 | 4.90 | 6.07 |
| Days 7 to 10 | 4.45 | 4.30 | 5.11 | 6.16 |
| Days 10 to 14 | 4.59 | 4.47 | 5.12 | 6.27 |
| All | 4.31 | 4.09 | | |

- **Gain over climatology.** For the NRA output, 35.3% (95% interval 34.1 to 36.5) at days 0 to 1
  and 26.9% (25.7 to 28.0) at days 10 to 14. For the prevailing output, 43.2% (42.0 to 44.4) and
  28.8% (27.5 to 30.0).
- **Normalised MAE of the median, all rows.** 7.58 for the NRA output and 7.30 for the prevailing
  output.
- **Coverage.** The q05-to-q95 band holds 0.814 of observed half-hours for the NRA output and 0.851
  for the prevailing output, against a nominal 0.90. Both bands cover less than the nominal 0.90.
  [Experiment 688](00688-winter-normal-running-coverage-median-error-and-events-part-1/result.md)
  found that the winter NRA shortfall comes mostly from error in the median and from load transfers,
  and [experiment
  683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md)
  had only suggested that cause (see "Walk-forward band calibration").
- **Prevailing against NRA.** The prevailing output scores 0.03 points of capacity worse in the two
  fallback months (3.98 against 3.95), 0.17 better from 2026-01 to 2026-06, and 0.42 better from
  2026-07 to 2026-10. After the one verified load shift, at D5 and D16 in December 2025, the
  prevailing output scored worse than the NRA output for 7 days after the shift ended (experiment
  668).
- **Earlier prevailing numbers.** [Experiment
  667](00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md) scored the
  prevailing output without the extra lagged inputs (CRPS 4.14, median error 7.39). Experiment 668
  scores the current file (4.09 and 7.30). Use experiment 668. Its NRA numbers agree with experiment
  667 to within 0.01.

### Code

**Last change** is the merged nged-ml-research pull request that last touched the file on
`origin/main`, with its commit. Several experiment folders import modules from earlier folders, so a
fit needs the import closure listed in the appendix. Paths are relative to the repository root.

| File | Role | Last change |
|---|---|---|
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/e00667_cand.py` | Candidate fit entry point for one fold (copy of 620 and 613 code) | PR 327 (`2fcc55c49`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/e00667_fold.py` | Fold definitions, cut-offs, embargoes, 14-day leads | PR 327 (`2fcc55c49`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/e00667_extend.py` | Rebuilds the raw ensemble-summary matrices to 2026-10-31 (input table builder) | PR 327 (`2fcc55c49`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/e00667_export.py` | Native 09:00 UTC forecasts to the 12:00 UTC contract rows, with capacity scaling | PR 327 (`2fcc55c49`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/e00667_merge.py` | Merges NRA and prevailing rows into `primaries.parquet` and checks it | PR 327 (`2fcc55c49`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/chain.sh` | Runs one fold's fits in order: `pretrain`, `step2`, `table`, `fit`, `ptable`, `pfit` | PR 327 (`ed7bed47f`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/run_folds.sh` | Runs `chain.sh` for each listed month in turn | PR 327 (`a364a5d3f`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/e00667_k1.py` | K1 network: pre-training and step 2 of one fold (copy of 613's `e00613_arm.py` with the fold patched in) | PR 327 (`2fcc55c49`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/e00667_tables.py` | Rebuilds the thermal-lag and solar-gain tables to 2026-10-31 (input table builder) | PR 327 (`2fcc55c49`) |
| `ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/e00613_arm.py` | K1 arm: wraps arm A1 of 591 with the 7-day embargo and synthetic lag-only shifts | PR 271 (`8a5e60f9b`) |
| `ml-experiments/00591-shared-lag-encoder-auxiliary-shift-depth-loss/e00591_arm.py` | Arm A1: the shared lag encoder without an auxiliary loss, which K1 wraps | PR 246 (`87deec536`) |
| `ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/e00613_nbmodel.py` | The Candidate's network class (lag encoder with neighbour columns) | PR 271 (`b244bbe4c`) |
| `ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/e00613_nbsignals.py` | Neighbour and gap signal columns | PR 271 (`b244bbe4c`) |
| `ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/e00613_nbarm.py` | Fit and inject wrapper around the 574 pipeline | PR 271 (`8a5e60f9b`) |
| `ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/e00613_widths.py` | Width factors, source of 1.0934 | PR 271 (`8a5e60f9b`) |
| `ml-experiments/00606-nra-lag-encoder-signal-permutations-gap-averages-and-neighbours/e00606_model.py` | Shared lag encoder with a chosen set of inputs | PR 262 (`e93adb820`) |
| `ml-experiments/00591-shared-lag-encoder-auxiliary-shift-depth-loss/e00591_model.py` | Shared lag encoder and auxiliary depth head | PR 246 (`87deec536`) |
| `ml-experiments/00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/e00574_model.py` | Network with lag encoder and the design 5 offset add-ons of experiment 565 | PR 230 (`fb7e829af`) |
| `ml-experiments/00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/e00574_fit.py` | Three-step schedule, early stopping, coarse refits on days 0 and 14 of each fold (experiment 667 refits monthly) | PR 230 (`dd9977df8`) |
| `ml-experiments/00572-two-offset-part-1-lags-versus-the-slow-output-offset/e00572_model.py` | Network with the design 5 output offset of experiment 565 | PR 227 (`986ffd0eb`) |
| `ml-experiments/00572-two-offset-part-1-lags-versus-the-slow-output-offset/e00572_lags.py` | Day and week lags with a configurable age cap | PR 227 (`986ffd0eb`) |
| `ml-experiments/00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1/e00569_columns.py` | Thermal-lag and solar-gain input columns in pre-training and fine-tuning | PR 225 (`d8d614892`) |
| `ml-experiments/00559-m0-leak-ladder-three-step-schedule-frozen-encoders-and-penalty/e00559_model.py` | Modular encoders (weather, datetime) and mixer | PR 214 (`bbe099c19`) |
| `ml-experiments/00545-modular-lag-free-demand-network-weather-and-datetime-encoders/e00545_model.py` | Original modular lag-free network | PR 206 (`1d0595a71`) |
| `ml-experiments/00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles/e00437_features.py` | Trailing-window thermal-lag and solar-gain features | PR 95 (`5a84bafb2`) |
| `ml-experiments/00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens/e00418_prep.py` | ERA5 and CAMS pre-training rows and fold assembly | PR 75 (`a7d730b76`) |
| `ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand/e00397_models.py` | Gaussian mixture head and `mixture_quantiles` | PR 50 (`7b6422960`) |
| `ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand/e00397_fit.py` | Fit and forecast stages of the plain and attention mixture networks | PR 50 (`7b6422960`) |
| `ml-experiments/00598-nra-gate-against-signals-as-inputs-at-equal-band-width/e00598_signals.py` | Signals as per-row network inputs | PR 254 (`4c94f9f27`) |
| `ml-experiments/00599-nra-refit-embargo-and-down-weighting-of-the-newest-rows-against-load-shift-absorption/e00599_common.py` | Refit embargo and newest-row down-weighting | PR 255 (`f907d8435`) |
| `ml-experiments/00581-robust-multiweek-lag-encoder-x3/e00581_bank.py` | Injected load-shift bank for the lag inputs | PR 236 (`c9c3b259d`) |
| `ml-experiments/00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/e00684_family.py` | Truncated Gaussian (and gamma) output family patch | PR 348 (`27f018997`) |
| `ml-experiments/00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/e00684_run.py` | Runs the 667 chain under an output family | PR 348 (`8fe6b3df2`) |

**The prevailing head is separate code.** These files build and fit R-haz and its extra lagged
inputs on top of the Candidate's stored quantiles.

| File | Role | Last change |
|---|---|---|
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/prevailing/p667_head.py` | R-haz head with 14 daily hazards | PR 327 (`2fcc55c49`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/prevailing/p667_data.py` | Head rows for one fold month | PR 327 (`2fcc55c49`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/prevailing/p667_inject.py` | Candidate passes on injected-shift banks | PR 327 (`2fcc55c49`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/prevailing/p667_fit.py` | Fits the head of one fold month on earlier banks | PR 327 (`2fcc55c49`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/prevailing/p667_export.py` | Writes the prevailing contract rows | PR 327 (`2fcc55c49`) |
| `ml-experiments/00652-two-output-forecast-with-explicit-rules-and-survival-hazard/e00652_head.py` | R-haz survival head (12-output version) | PR 310 (`ec5122d0c`) |
| `ml-experiments/00652-two-output-forecast-with-explicit-rules-and-survival-hazard/e00652_fit.py` | Fits R-haz and its placebo | PR 310 (`88a622bc2`) |
| `ml-experiments/00647-two-output-forecast-departure-head-on-frozen-k1/e00647_head.py` | Departure heads and `mixture_quantiles` of the prevailing mixture | PR 305 (`77944a970`) |
| `ml-experiments/00650-extended-shift-departure-head-against-decayed-recent-error/e00650_bank.py` | Extended shift generator | PR 308 (`31c9df1cf`) |
| `ml-experiments/00660-shared-generator-of-injected-load-transfer-events/e00660_generator.py` | Shared generator of injected load-transfer events | PR 320 (`2f76ee2ac`) |
| `ml-experiments/00681-extra-lagged-power-inputs-for-the-prevailing-forecast/e00681_lags.py` | The 14 extra lagged-power inputs (28 columns) | PR 345 (`83c577da5`) |
| `ml-experiments/00681-extra-lagged-power-inputs-for-the-prevailing-forecast/e00681_fit.py` | Fits the head with the extra lagged inputs | PR 345 (`3ff7a104d`) |
| `ml-experiments/00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/e00668_extra_lags_prevailing.py` | Replaces the prevailing rows of `primaries.parquet` with the extra-lag head | PR 353 (`5ebfb9d74`) |

**The input-table builders are separate scripts.** These files write the cached tables of "Input
tables", and none of them is imported by a fit.

| File | Role | Last change |
|---|---|---|
| `ml-experiments/00607-bakeoff-extend-cached-inputs-to-2026-10-31/e00607_build.py` | Ensemble summaries, cloud-cover summaries, and the observed-power cache to 2026-10-31 | PR 266 (`8b97af818`) |
| `ml-experiments/00607-bakeoff-extend-cached-inputs-to-2026-10-31/e00607_derive_era5.py` | Merges the ERA5 box and derives `era5_features.parquet` | PR 266 (`8b97af818`) |
| `ml-experiments/00607-bakeoff-extend-cached-inputs-to-2026-10-31/e00607_fetch_era5_box_cds.py` | Fetches 13 ERA5 variables from the Climate Data Store | PR 266 (`356081f4d`) |
| `ml-experiments/00607-bakeoff-extend-cached-inputs-to-2026-10-31/e00607_fetch_ptype_arco.py` | Reads ERA5 precipitation type from ARCO-ERA5 | PR 266 (`8b97af818`) |
| `ml-experiments/00607-bakeoff-extend-cached-inputs-to-2026-10-31/e00607_fetch_cams_demand.py` | Fetches CAMS irradiance at the Primaries for 2026-09-11 to 2026-10-31 | PR 266 (`8b97af818`) |
| `ml-experiments/00561-wind-components-era5-and-ens-summary-data-prep/e00561_ens_summary_uv.py` | Ensemble summaries with wind components, one file per run | PR 216 (`0e9b782d3`) |
| `ml-experiments/00560-modular-lag-encoder-capped-lags-and-block-dropout/e00560_data.py` | Expected-weather table, and the lag columns of the lag encoder | PR 213 (`a08d677c0`) |
| `ml-experiments/00558-lags-in-pretraining-and-24-hour-cap-lag-light-demand-network/e00558_lags.py` | Capped day and week lags, lag ages, and block dropout | PR 212 (`ac219c797`) |
| `ml-experiments/00400-lagged-lagless-forecast-blend-around-load-shifts-stored/e00400_blend.py` | `nged_neighbours`: the electrical-neighbour list from the two NGED files | PR 54 (`fce59890a`) |

**Production code for the same job.** The production repository's documented extension point for a
new model is a `BaseForecaster` subclass with its own `FeatureEngineer`. Whether the neural networks
use it is not decided.

- **Exists.** The model interface, the cleaned-power reader, the effective-capacity table, the ENS
  weather store, and the lag machinery, as described under "Production code that exists today".
  `XGBoostForecaster` is the only trained forecaster and serves every series type.
- **Differs.** The production forecaster is a per-series XGBoost booster fed the control member, and
  it outputs one point forecast per ensemble member. The Candidate is a pooled neural network fed
  51-member summaries, ERA5 and CAMS pre-training rows, neighbour power summaries, and a 24-hour
  minimum lag age, and it outputs 11 quantiles per half-hour. Production `BaseForecaster.train` and
  `predict` take an `AllFeatures` frame keyed by `ensemble_member` (`base_forecaster.py`). The
  Candidate needs pooled rows of 51-member summaries, cross-series neighbour inputs, and multi-stage
  pre-training on other weather products, none of which that frame carries. The analysis proxy also
  differs: production's `select_analysis_proxy` takes the control member of the freshest run, and
  the research proxy is the ensemble mean of the freshest run published by the origin. The Candidate
  trained only on a 09:00 native origin from the same day's 00Z run, relabelled to 12:00. Production
  runs `live_forecasts` at 00:00, 06:00, 12:00, and 18:00 UTC, with `NWP_PUBLICATION_DELAY_HOURS =
  9`.
- **Missing.** Neural-network training and inference code, a `torch` dependency, ENS summary
  features, the neighbour list (the two NGED files named under "Architecture"), a quantile output
  column set, an NRA and prevailing output flag, the survival head and its bank of injected shifts,
  the truncated Gaussian family, the MVA clamp, the walk-forward band scale, a monthly refit
  schedule, and the effective-capacity scaling of the target. Four inputs are also absent at
  inference time. Production stores no CAMS irradiance, yet two Candidate columns read the previous
  day's CAMS irradiance at inference as well as in pre-training. Each monthly refit pre-trains on
  ERA5 and CAMS rows up to the refit time, and ERA5 arrives about 5 days late. The Candidate needs
  the lag-free forecast table T for the most recent days, which only a daily run of K1 produces. The
  Candidate also needs the history of its own stored forecasts for the prevailing head and for the
  band scale (see "Serving state" below).

### Port notes

- **Reproduce exactly.** The input definitions and scalings above, the 24-hour minimum age of the
  Candidate's own lags, the chain of K1, table T, and the Candidate's fine-tune with the 60
  zero-weight columns, the same frozen encoders and fresh mixer seed, the 90-day half-life, the
  monthly refit on data before the first of the month with a 3-day embargo at the mixer step and a
  7-day embargo at the fine-tune, the 11-quantile read-out by bisection, and the construction of the
  prevailing mixture from the Candidate's quantiles.
- **Serving state.** The signal columns and the 14 extra lags read table T, the median of the
  lags-masked K1 network at the freshest earlier origin with a lead of at most 24 hours. A service
  must therefore run K1 and store its forecast every day, for each Primary and for every neighbour
  that a signal reads. The walk-forward band scale reads the model's own stored forecasts of the
  previous 90 days beside the observed power (`LOOKBACK_DAYS = 90` in `e00683_calibrate.py`). The
  prevailing head reads the banks of injected-shift rows, one per earlier month. The production
  storage list under "What a production forecasting service has to store" holds the weights only, so
  table T and the forecast history are additions to it.
- **Serving path.** Inference for the Primaries is a chain, not a call to `predict`. K1 with its
  lags masked writes table T, and table T feeds the signals. Then come the Candidate for each seed,
  the seed average and sort, R-haz acting on the Candidate's quantiles, the clamp, and a calibration
  that reads 90 days of stored forecasts. That chain departs from two production design principles.
  The first is keeping the serving path close to "load a model, call `predict`". The second is
  putting a capability in the training loop when it could live in either place. The band scale could
  be fitted at refit time, and the calibrated copy fits it month by month from stored forecasts
  instead. What the chain buys is the lag signals and the prevailing output, whose gain is in the
  scores above. Whether to accept that trade is not decided.
- **Degradation.** The Candidate trained only on the 09:00 native origin of the same day's run, so
  the other 6-hourly production slots and a stale NWP run are untrained regimes. The `ecmwf_ens`
  schedule runs at 10:30 UTC and retries every 30 minutes up to 8 times, so on a late day the 12:00
  slot holds yesterday's run. Its leads would be 24 hours longer than the 339 hours trained and
  would pass the 351 hours that the summaries reach, and its lag ages would change. The research
  code states no rule for a stale run, a missing lag (the Candidate trained with 30% lag masking), a
  missing neighbour, missing CAMS, or fewer than two banks (B-haz covers that case in the first two
  folds). The production design requires degrading, widening the bands, and recording the
  degradation on the row, and the rule for each case is an open question below.
- **Unvalidated.** The truncated Gaussian output has been tested at only four Primaries and two
  months, and only for the NRA output. The prevailing head's behaviour on real load shifts rests on
  one verified transfer. The survival probability beyond day 10 is the bank generator's duration
  prior and not an estimate from observed events. The backtest used two seeds and no widening, where
  the final test used five seeds and a factor of 1.0934.
- **July and August level miss at D18 and D14.** [Experiment
  690](00690-slow-level-state-on-primaries-13-and-16-july-august/result.md) found that a masked,
  frozen slow-level state tracks the July to August 2026 level miss of the stored normal-running
  forecasts at D18 and D14 (identifiers 13 and 16; weekly correlation 0.66 and 0.87), and that
  adding it to `q50` halves the median absolute error at both Primaries (-50.3% and -51.3%), in
  hindsight. The level ramps over months and does not follow the step on 2026-07-01. Adding it moved
  the median absolute error of the two control Primaries by 5.5% and 7.5%, against a limit of 5% set
  in advance, so the verdict is partly supported. The added level is a hindsight diagnostic fitted
  on the whole record, and not a forecast, and experiment 690 proposes no new method. The miss stays
  listed as not explained by a load transfer ([experiment
  685](00685-event-table-on-the-final-test-window-all-primaries/result.md)).
- **Open questions.** Whether to port the 8-input or the 14-input extra-lag set (the backtest uses
  14). Whether the fallback head is needed in production, because the backtest needed it only for
  the first two folds. What experiment 677 concludes about the necessary components (pending). Which
  of the 00:00, 06:00, 12:00, and 18:00 UTC slots the Candidate must serve, and what the service
  does with a stale NWP run, a missing lag, or missing CAMS. Whether the band scale is fitted at
  refit time or in the serving path. Whether the porter rebuilds the cached inputs from source or
  ports the lineage of experiments 607, 437, and 560. Whether the Candidate's 2019-09 pre-training
  rows handle the regime change of D8's gas engines on 2023-08-15, which the code neither shows nor
  excludes.

## Demand at the BSPs and GSPs

### Architecture

**The two BSPs and two GSPs are forecast by the lag-enabled demand neural network of [experiment
633](00633-bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network/result.md), which is arm
`lagged_b1` of [experiment
622](00622-bsp-gsp-lagged-against-lag-free-matched-d5-thermal-lag-rematch/result.md), retrained with
fresher lags.** The backtest is [experiment
669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md). The
neural network has no neighbour inputs and no component that represents load shifts, and the series'
readings are net of embedded generation, which no input represents. One neural network is fitted per
group (the two BSPs together, the two GSPs together), so the weights are shared across the two
series of a group except for the series embedding and the offset.

- **Inputs.** The same columns as the Candidate, except that the 60 signal columns are absent, with
  one embedding of the series (2 entries per group). Each row holds the following groups.
    - **The 37 lag-free columns.** The mean and standard deviation over the 51 members of the 7
      weather variables (14 columns), 21 calendar columns (the cosine of the solar zenith angle, the
      sine and cosine of the azimuth, the sine and cosine of the local hour at periods of 24 and 12
      hours, a one-hot day of week, a COVID flag, three Christmas columns, and three bank-holiday
      columns), and 2 lead columns (the lead in hours over 336, and its square root). The weather
      columns are those of the Candidate, and the 51-member means and standard deviations are taken
      at the series' H3 resolution-5 cell.
    - **The 8 lag columns.** For each of the day lag and the week lag: the lag's power over
      capacity as a z-score (clipped at plus or minus 5), a missing flag, and an age code, which is
      the age of the lag in hours at the target time over 336. The remaining 2 columns are the grid
      indices of the two lag times. A missing lag reads as value 0, flag 1, and the age code of the
      longest lead: 1.0714 for the day lag and 1.5 for the week lag at the 3.5-hour cap.
    - **The knot position.** One column that places the row's valid time among the monthly knots of
      the output offset: the number of months from 2019-09-01 to the valid time, fractional within
      the month.
    - **The 14 thermal-lag and solar-gain columns.** As for the Candidate.
    - **The expected-weather table** for the lag encoder: the weather and calendar columns of the
      freshest run issued at or before the lag's own time, passed through the same encoders.
- **Lags.** The stored neural network of experiment 633 read lags at least 24 hours old. The
  backtest retrains it with the 3.5-hour cap of [experiment
  572](00572-two-offset-part-1-lags-versus-the-slow-output-offset/result.md), so the newest power is
  the 05:30 reading on the origin day, 6.5 hours before the 12:00 label.
- **Encoders, mixer, and output.** The same encoders and mixer as the Candidate, with a lag encoder
  of 83 inputs (3 lag columns, the two 32-wide expected-weather and expected-datetime embeddings,
  and the embeddings of the series and the lag) and a mixer with 145 inputs. The offset, loss,
  quantile bisection, and sorting are the Candidate's. No width factor is applied. The set-up
  follows [experiment 619](00619-bsp-gsp-lag-enabled-network-for-the-bake-off/result.md) (the
  lag-enabled neural network at the nodes), [experiment
  565](00565-drift-embedding-variants-six-designs/result.md), [experiment
  569](00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1/result.md),
  [experiment 591](00591-shared-lag-encoder-auxiliary-shift-depth-loss/result.md) (the shared lag
  encoder), [experiment
  574](00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/result.md),
  and experiment 572, and the node harness of [experiment
  564](00564-pooled-primary-bsp-gsp-m2-reduced/result.md).
- **Origin.** The neural network runs at its own 09:00 UTC origin (the 00Z run plus 9 hours) and its
  output is relabelled to 12:00 UTC. Every input read at 09:00 was available at 12:00.

### Input tables

**Every fit reads cached tables that earlier experiments built, and no script in the fit imports
builds them.** The appendix lists imports only, so a script that only writes data is not in it. The
table lists each input of the BSP and GSP fits, the script that builds it, and its source data.
Paths are relative to `data/ml-experiments/`, which is private and outside git.

| Input | Path | Built by | Source data |
|---|---|---|---|
| Node ensemble summaries (mean and standard deviation over 51 members, 7 variables, one file per daily 00Z run from 2024-04-01 to 2026-10-31) | `00633-*/inputs/ens_summary_nodes/init=<date>.parquet` | Runs to 2026-06-29: links to the files of `e00415_prep.py summaries`. Runs from 2026-07-01: `e00633_frames.py` (weather frames), then `e00633_nodes.py summaries` | The ECMWF ENS archive, read through `e00048_build_frames._half_hourly_rows` |
| Fine-tuning and step-2 matrices of the 4 nodes (observed, unflagged half-hours only) | `00633-*/inputs/raw/summary/<site>.npz` | `e00415_prep.py raw` writes experiment 415's matrix files, experiment 564's `prepare_folders()` links the 4 nodes' files, and `e00633_nodes.py raw` appends the rows of runs after 2026-06-29 | The node summaries above, the power grid, and the flag file below |
| Test matrices (every half-hour, observed or not) | `00669-*/inputs/raw_test/` | `e00669_testrows.py` | The same sources, without the observed-power filter |
| Pre-training matrices (ERA5 and CAMS pseudo-runs, ending 2026-07-31) | `00564-*/work/raw_pretrain/<site>.npz` | `e00564_fit.py build-pretrain-rows`, which calls experiment 418's `write_raw` | ERA5 and CAMS tables at the 4 nodes |
| Thermal-lag and solar-gain tables | `00633-*/inputs/trailing437_nodes/` | `e00633_tables.py build` (experiment 437's `site_table`, through `e00622_b1.py`) | The node summaries above, and CAMS irradiance from 2019-09-01 to 2026-11-01 in `00620-*/inputs/cams_demand_points_extended.parquet` |
| Extended CAMS irradiance at the demand series | `00620-*/inputs/cams_demand_points_extended.parquet` | `e00620_tables.py` merges the published `fetch_cams_demand_points.py` tables with `e00607_fetch_cams_demand.py` (2026-09-11 to 2026-10-31) | Copernicus Atmosphere Monitoring Service |
| Expected-weather tables of the lag encoder | `00633-*/inputs/w560_nodes/<group>/expected/*.npy` | No script of its own: experiment 560's `raw_table` builds and caches them at first use from the fine-tuning and pre-training matrices | The two matrix sets above |
| ERA5 features and the observed-power grid (to 2026-11-10 23:30) | `00607-*/work/era5_features.parquet` and `observed.parquet`, read through `e00633_common.redirect()` | `e00607_fetch_era5_box_cds.py` and `e00607_derive_era5.py` for ERA5, and `e00607_build.py observed` for power | The Climate Data Store (ERA5) and the NGED power table |
| Flag file | `00398-load-shift-event-rescore-without-flagged-zero-actuals/flagged_keys.parquet` | `e00398_flagged_keys.py` | The rows with `drop_reason` equal to `substation_zero` in the cleaned power Delta table of the production repository, 2024-04-01 to 2026-07-01 |
| Series table (series, capacity, coordinates, H3 cell) | `studies/per_study/stage1_pilot/demand/inputs/series.parquet`, read by `e00087_common.series_table()` | The stage-1 pilot | NGED metadata and the effective-capacity table |

**Each script reads the output of the scripts above it in the table.** The order that follows from
those reads is:

1. The ERA5, CAMS, and power extensions of experiments 607 and 620.
2. `e00415_prep.py summaries` and `raw`, and `e00564_fit.py build-pretrain-rows`.
3. `e00633_frames.py`, then `e00633_nodes.py summaries` and `raw`, then `e00633_tables.py build`.
4. `e00398_flagged_keys.py`, which needs only the cleaned power table.
5. `e00669_testrows.py`.
6. `00669-*/run_all.sh`, which runs `pretrain-wt`, `step2`, `fit`, and `assemble` for each group,
   and then `e00669_export.py` for the files.

### Training regime

- **Schedule.** The same three steps as the Candidate: ERA5 and CAMS pre-training, a step 2 with a
  fresh mixer of seed 12345 and frozen encoders, and a fine-tune with a 90-day half-life, with the
  same optimiser settings. Each group pre-trains its own encoders (`pretrain-wt`). The encoders
  pre-train on the matrices of [experiment
  564](00564-pooled-primary-bsp-gsp-m2-reduced/result.md), which end on 2026-07-31, so the folds
  from 2026-08 share the same pre-training rows.
- **Differences from the Candidate recipe.** The BSP and GSP fits differ in six ways.
    - **Lags.** The lag cap is 3.5 hours and not 24 hours.
    - **Embargo.** The embargo is 3 days at step 2 and at the fine-tune (`EMBARGO_DAYS` in
      `e00574_fit.py`). The Candidate's 7-day embargo with half weight for rows 7 to 14 days old is
      not applied.
    - **No signal columns and no dropout of them.** The neighbour and gap signals and their 50%
      dropout (D50) do not exist for these series.
    - **No synthetic load shifts.** `e00669_fit.py` calls the arm P5 of `e00574_fit.py` directly.
      The synthetic lag-only shifts of the Candidate's K1 chain enter only through
      `build_month_syn` in `e00591_arm.py`, which `e00669_fit.py` never imports. The file
      `e00574_fit.py` injects a shift only in its stress run, which experiment 669 turns off
      (`stress=False`). The docstring of `e00619_fit.py` also says "no synthetic load-shifts, no E3
      weighting". The fine-tune rows therefore carry no injected shifts, and no auxiliary depth loss
      runs, although the network class (`AuxLagMdn`) has the auxiliary heads.
    - **No prevailing output.** No survival head and no fallback head are fitted.
    - **Cleaning.** The Candidate and the BSP and GSP fits read the same flag file of experiment 398
      (above), not the live cleaning of the production repository. The file ends on 2026-07-01, and
      for these 4 series it holds no flags after 2025-09, so no half-hour is dropped in the test
      year. The cut of D15 and D18 before 2025-03-01 does not apply.
- **Fine-tuning rows.** The 730 days before the cut-off, with targets ending 3.5 hours before it.
  For the BSP group, 570,000 rows in the 2025-11 fold rising to 735,000 in 2026-10, and 2% to 3%
  fewer for the GSP group. The ENS history starts on 2024-04-01, so the first fold has 19 months of
  it.
- **Lag dropout.** 30% of days mask both lags, and 15% of those mask only the day lag.
- **Leakage guard.** In every refit, hide all power from the cut-off on. Experiment 669's
  `E00669_HIDE=1` sets the cleaned power to NaN from the cut-off, as experiment 667's `train` phase
  does. The switch makes the fold return no test rows, so only the pre-flight gate on the 2026-07
  fold ran with it. The 12 fits of the year relied on row selection alone, and the perturbation test
  of the Opus leakage review found no leak. A porter should hide the power in every refit and test
  that no weight changes.
- **Refits and seeds.** Twelve monthly refits, one per test month, and two seeds averaged quantile
  by quantile and sorted. The final-test neural network of [experiment
  633](00633-bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network/result.md) was one fit at
  2026-07-01. The early-stopping weeks are purged by 9 days, shorter than the 14-day leads, so a
  held-out week's targets can sit in training rows of later origins. [Experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md) records
  the shorter purge as a caveat, ran a perturbation test that found no leak, and states that it
  changed no result.
- **Test rows.** The stored test matrices drop half-hours without an observed reading, so experiment
  669 rebuilt them in `e00669_testrows.py` with no filter. A port has to forecast every half-hour.
- **Targets and capacity.** Power divided by effective capacity (D13 55.3 MW, D17 47.6 MW, D6 384.3
  MW, and D1 170.9 MW), z-scored per series. All four are signed MW series.
- **Weather at lag times.** The thermal-lag and solar-gain columns use the analysis proxy built from
  the first day of the previous 00Z run, which exists at forecast time. The research proxy is the
  ensemble mean of the freshest run published by the origin. The proxy in the production repository
  (`packages/weather_utils/src/weather_utils/analysis_proxy.py`) is the control member of the
  freshest run, filled column by column, so a port has to build the ensemble-mean version. Two of
  the 14 columns read the previous UTC day's CAMS irradiance, which CAMS publishes at midnight UTC,
  at inference as well as in pre-training. The last roughly 5 days of the ERA5 and CAMS
  pre-training rows could not be read at a real cut-off, because ERA5 arrives about 5 days late.
- **Hardware and run time.** About 2 GPU-hours on one process and 1 core (experiment 669).

### Post-processing

- **Walk-forward band scale (calibrated copy only).** See "Walk-forward band calibration" below. It
  applies to all BSP and GSP rows from 2025-12.
- **No clamp and no cap.** The files keep the quantiles above capacity: the highest quantile is 2.2
  times the table value at D1 and 1.4 times at D6. A negative quantile is net export, and the q01 is
  below zero in 94% of the D6 rows and 99.6% of the D1 rows.
- **Float32 files.** `bsps.parquet` and `gsps.parquet` were recast from Float64 to Float32 on
  2026-11-10, and the Float64 originals are in `pre_float32/`. The calibrated copies were fitted on
  the Float64 originals and hold Float32 quantiles.

### Scores

**The lag-enabled neural network beats climatology and the manual heuristic at every lead band, and
the margin shrinks with lead.** The table gives CRPS in points of capacity from [experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md),
with climatology as the reference. The manual-heuristic columns are on that page.

| Lead band | BSP CRPS | BSP climatology | GSP CRPS | GSP climatology |
|---|---:|---:|---:|---:|
| Days 0 to 1 | 4.27 | 7.23 | 10.52 | 18.69 |
| Days 1 to 3 | 4.67 | 7.24 | 11.44 | 18.74 |
| Days 3 to 7 | 5.55 | 7.24 | 13.18 | 18.75 |
| Days 7 to 10 | 6.13 | 7.28 | 14.52 | 18.83 |
| Days 10 to 14 | 6.41 | 7.32 | 14.78 | 18.87 |

- **Pooled over the four series,** [Experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md) reports
  CRPS of 7.40, 8.06, 9.37, and 10.47 at days 0 to 1, 1 to 3, 3 to 7, and 7 to 14, and q05-to-q95
  coverage of 0.883, 0.877, 0.869, and 0.870.
- **Coverage.** 0.867 for the BSPs and 0.877 for the GSPs, against a nominal 0.90 (experiment 668).
  The two sets of coverage figures pool differently: experiment 668 gives one figure per group, and
  experiment 669 pools both groups within each lead band. The group rows of experiment 669 (0.86 to
  0.88 for the BSPs and 0.87 to 0.89 for the GSPs) bracket the two group figures. The misses change
  side by month, so a level drift and not only a narrow band is at work.
- **Against the 24-hour-lag neural network.** On July to October 2026 origins at hours 0 to 72, the
  3.5-hour lags score 7.24 against 7.54, but only 0.16 of the 0.30 gain is attributable to the lag
  cut, and the rest comes from the monthly refits (experiment 669).
- **The largest miss.** The observed demand at D6 Walpole stepped up on 2026-10-19 while the median
  fell, a level step consistent with a load transfer that is not confirmed in NGED's logs and that
  this neural network does not model.

### Code

**Last change** is the merged nged-ml-research pull request that last touched the file on
`origin/main`, with its commit. Several experiment folders import modules from earlier folders, so a
fit needs the import closure listed in the appendix. The scripts that build the cached inputs are
named in "Input tables" and are not in the closure. Paths are relative to the repository root.

| File | Role | Last change |
|---|---|---|
| `ml-experiments/00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/e00669_fit.py` | Lag-enabled fit, 3.5-hour lag cut, one monthly fold | PR 330 (`5c3a002bc`) |
| `ml-experiments/00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/e00669_fold.py` | Twelve monthly folds patched onto the 574 pipeline | PR 330 (`ec222e24d`) |
| `ml-experiments/00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/e00669_testrows.py` | Test matrices that keep every half-hour | PR 330 (`ec222e24d`) |
| `ml-experiments/00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/e00669_export.py` | Writes `bsps.parquet` and `gsps.parquet` | PR 330 (`771ca7e20`) |
| `ml-experiments/00633-bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network/e00633_node_fit.py` | Lag-enabled network at the four nodes (final-test fold) | PR 290 (`dbc2f1cbc`) |
| `ml-experiments/00633-bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network/e00633_nodes.py` | Node weather inputs | PR 290 (`dbc2f1cbc`) |
| `ml-experiments/00622-bsp-gsp-lagged-against-lag-free-matched-d5-thermal-lag-rematch/e00622_b1.py` | Thermal-lag and solar-gain tables for the four nodes | PR 279 (`cc6cd8783`) |
| `ml-experiments/00619-bsp-gsp-lag-enabled-network-for-the-bake-off/e00619_fit.py` | Pooled-per-group lag-enabled network at the nodes | PR 275 (`18e6cdad2`) |
| `ml-experiments/00572-two-offset-part-1-lags-versus-the-slow-output-offset/e00572_lags.py` | Lag cap of 3.5 hours | PR 227 (`986ffd0eb`) |

**Production code for the same job.** The production repository's documented extension point for a
new model is a `BaseForecaster` subclass with its own `FeatureEngineer`. Whether the neural networks
use it is not decided.

- **Exists.** Series of type BSP and GSP exist in `TimeSeriesMetadata.substation_type`, and the same
  generic XGBoost forecaster could be run on them. No forecaster is specific to them.
- **Differs.** The research neural network reads 3.5-hour power lags, 51-member ENS summaries, and
  thermal-lag columns, is pooled per group, and outputs 11 quantiles. The production lag rule nulls
  any lag that is not older than the lead time and has no 6-hour power-delay constant. The
  production `BaseForecaster.train` and `predict` take `AllFeatures` rows keyed by
  `ensemble_member`, and the research network reads one row of 51-member summaries per half-hour,
  pooled over two series, after pre-training on ERA5 and CAMS rows. The flag file of experiment 398
  is a snapshot of the production cleaning, so a port reads the live cleaned table instead.
- **Missing.** The same pieces as for the Primaries, apart from the neighbour list and the
  prevailing head, and the group-pooled fitting.
- **Untrained origins and stale runs.** The neural network was trained and scored only at the
  09:00 native origin of the same day's 00Z run, relabelled to 12:00. The production
  `live_forecasts` asset runs at 00:00, 06:00, 12:00, and 18:00 UTC. The `ecmwf_ens` schedule starts
  at 10:30 UTC and retries every 30 minutes up to 8 times, so the run can arrive as late as about
  14:30 UTC. `live_forecasts` uses the freshest run present, so a late day forecasts from the
  previous day's run. With that run every lead is 24 hours longer, which exceeds the 351 hours seen
  in training, and every lag is older. The page gives no degradation rule for these cases. Rule 1
  of the production repository's
  [inherent-stability page](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/)
  requires the service to degrade, widen the bands, and record the degradation on the row. The
  training masks both lags on 30% of days, so an absent lag is a state the neural network has seen.

### Port notes

- **Reproduce exactly.** The 3.5-hour lag cap, the one-neural-network-per-group pooling, the offset
  of design 5 of [experiment 565](00565-drift-embedding-variants-six-designs/result.md), the
  thermal-lag columns of [experiment
  437](00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles/result.md), and the
  schedule of [experiment
  574](00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/result.md).
- **Unvalidated.** [Experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md) reports
  no curtailment cost against the manual heuristic for the BSPs and GSPs. Load shifts are not
  modelled, and NGED's event labels are incomplete.
- **Open questions.** Whether the 3.5-hour cap or a longer one gives the better score with the same
  refit schedule, which experiment 669 did not test on matched schedules. The D1 median is on
  average 18 MW below the observed median, and its monthly mean error swings between -39.8 MW
  and +25 MW. Which forecast slots (00:00, 06:00, 12:00, and 18:00 UTC) this neural network must serve,
  and what it does with a stale run, is not decided.

## Wind

### Architecture

**The three wind farms are forecast by the simple shared neural network S of [experiment
659](00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md), a
two-layer network with a Gaussian mixture output.** The backtest is [experiment
666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md). The only inputs are
ensemble weather and the lead.

- **Inputs, 16 in all.** The mean and standard deviation over the 51 members of 10 m wind speed,
  100 m wind speed, the sine and cosine of the 100 m wind direction, and 2 m air temperature (10
  columns). The lead as three columns: the sine and cosine of the lead in hours modulo 24, and the
  lead in hundreds of hours. A one-hot wind-farm column for each of the 3 farms, in the order of the
  farms' anonymous labels. One wind farm is a MW series and two are MVA series. There is no power
  lag, no calendar column, and no physics-lite input. The per-case summaries do hold the mean and
  standard deviation of a physics-lite power curve (`pl_mean`, `pl_sd`), but the fit does not read
  them.
- **Lead definition.** The wind lead is the time from the run's 09:00 UTC availability stamp to the
  valid time, so the first forecast half-hour (12:00 UTC) has a lead of 3 hours and the last has a
  lead of 338.5 hours. The solar network measures its lead from the 00Z run instead (see the solar
  section).
- **Standardisation.** Each input column is standardised with the mean and standard deviation of
  the fit rows, and the standard deviation has a floor of 1e-6.
- **Layers and output.** Two hidden layers of 64 units with GELU activations, feeding 9 outputs: the
  log weights (a log-softmax), the means, and the standard deviations of 3 Gaussian components.
  These describe the residual of power as a fraction of capacity from the mixture's centre, which
  is the mean of the training target. The last layer starts with zero weights, a bias of 0 for the
  weights, of -0.1, 0, and 0.1 for the means, and of log 0.08 for the log standard deviations. Each
  standard deviation is `exp(clamp(raw, -6, 1))`. The 11 quantiles are read from the mixture by 40
  bisection steps on the mixture's cumulative distribution function, over a residual range of -2
  to +2 (in fractions of capacity) around the centre. They are clamped to the range 0 to 1 of the
  neural network's own capacity scale, and then sorted within each row.
- **Weather.** The 00Z ENS run of the origin day, treated as available at 09:00 UTC, with leads of
  12 to 347.5 hours from the 00Z run. Where ensemble steps are 3 or 6 hours, the half-hourly value
  is interpolated as in [experiment 310](00310-metered-farm-long-lead-scoring/result.md) and
  [experiment 430](00430-unified-forecaster-solar-and-pretraining-ablations/result.md). No ERA5 or
  CAMS pre-training, and no ERA5 or CAMS input at inference time.
- **Cases kept.** Every case must hold all 51 members, numbered 0 to 50. The feature script raises
  an error when a case does not, and when a run from 2025-11-01 lacks any of its 672 half-hours.

### Input tables

**The wind fit reads tables that earlier scripts build, and it does not build them.** The page
appendix follows `import` statements only, so it lists the modules a fit imports and misses these
scripts, which write data and are never imported. All the data paths below sit under the data
folder (`DATA_PATH_INTERNAL`). The source data of every table is the production repository's own
stores: the ECMWF ENS `Nwp` Delta table (`NWP`, from 2024-04-01), the cleaned power Delta table
(`NGED/cleaned_power_time_series.delta`), the metadata table (`NGED/metadata.parquet`), and the
`effective_capacity` Delta table.

| Table (under `data/ml-experiments/`) | Built by | Source and content |
|---|---|---|
| `00259-*/work/hindcast/init=<date>.parquet` | `e00259_hindcast.py` | Member rows for the 00Z runs 2024-04-01 to 2025-06-16, with valid times before 2025-06-17 |
| `00615-*/wind_members/init=<date>.parquet` | `e00615_wind_members.py` | The same member rows for the runs 2025-06-17 to 2026-06-30, with valid times before 2026-07-01 |
| `00666-*/wind_members/init=<date>.parquet` | `e00666_wind_members.py` | The same member rows for the runs 2026-06-16 to 2026-10-31, with valid times before 2026-11-20 |
| `00659-*/wind_features/<farm>.parquet` | `e00659_wind_features.py` | Per-case summaries of experiment 659, read only by the gate in `e00666_wind_features.py` |
| `00666-*/wind_features/<farm>.parquet` | `e00666_wind_features.py` | The per-case summaries the fit reads |
| `00210-*/mask.parquet` | `e00210_data.py`, then `e00210_analysis.py` | The idle mask: `time_series_id`, `valid_time`, and `drop_reason` of every flagged half-hour |

- **Member rows are built by the production feature engineer.** Each member-row script calls the
  production `TabularFeatureEngineer().engineer` in bulk mode on one run's `Nwp` rows at the farms'
  H3 resolution-5 cells (`h3_res_5` of the metadata table) and on the cleaned power. The script
  keeps the key columns, the member, the 10 m and 100 m wind speeds, the sine and cosine of both
  wind directions, and the air temperature, as Float32. Rows without observed power are kept. The
  production feature engineer therefore already produces the wind inputs, member by member. The
  research code then summarises them in `e00666_wind_features.py`, which takes the mean and the
  standard deviation over the 51 members.
- **Files from experiment 615 and experiment 666 overlap.** Experiment 615 cut its runs at valid
  times before 2026-07-01, which truncated the runs from 2026-06-16. The 666 script rebuilds those
  runs to a full 15 days, and `member_paths` in `e00666_wind_features.py` takes the rebuilt file
  wherever both exist.
- **Run order.** The scripts below run from the repository root, with `DATA_PATH_INTERNAL` set. The
  second, third, and fourth read `e00259_common`, `e00615_common`, and `e00303_common` and need the
  experiment folders on `PYTHONPATH`, as the docstrings of the scripts say.
    1. `e00210_data.py` then `e00210_analysis.py` (the idle mask).
    2. `e00259_hindcast.py`.
    3. `e00615_wind_members.py`.
    4. `e00659_wind_features.py`.
    5. `e00666_wind_members.py`.
    6. `e00666_wind_features.py`, which asserts that its summaries reproduce experiment 659's stored
       summaries to 1e-5 before the valid time 2026-07-01.
    7. `e00666_wind_fit.py`, then the solar steps of the Solar section's "Input tables", then
       `e00666_export.py` (which writes `wind.parquet` and `solar.parquet` together, so it needs the
       solar fits as well).
- **The idle mask needs ERA5 wind.** `e00210_data.py` joins the farms' power to the ERA5 100 m wind
  speed in `data/handover/era5_wind_at_nged_wind_farms_anonymised.parquet`, interpolated linearly
  to each half-hour's midpoint. The wind-speed detector in `e00210_analysis.py` writes the mask from
  that table. We found no script in either repository that builds the handover file, so it is an
  open question (below). The mask ends on 2026-09-13 at 17:30 UTC.
- **Where "need only that folder" is wrong.** The appendix says wind and solar copy their code into
  the 666 folder and need only that folder. That holds for the `import` statements of the fit
  scripts. It does not hold for the inputs: the wind fit reads tables written by scripts in the
  experiment 210, 259, 615, and 659 folders.

### Training regime

- **Refits.** Wind refits on day 0 of each month (12 refits), on ensemble runs and valid times
  before the 62 days that precede the refit, and uses those last 62 days for early stopping
  ([experiment 666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md)).
  [Experiment
  659](00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md) refitted
  twice a month. One pooled network is fitted over the 3 farms for each refit and seed. A fit row
  needs a run stamp and a valid time that are both before the 62-day boundary, and a stopping row
  needs a run stamp from the boundary on and a valid time before the refit. The script asserts that
  no training row has a valid time at or after the refit, and the farm's capacity uses power from
  before the refit only. The last 62 days therefore never enter a gradient.
- **Optimiser.** AdamW, learning rate 0.002, batch 2,048, weight decay 0.0001, and 40 epochs. Every
  epoch runs, with no patience rule, and the epoch with the lowest negative log-likelihood on the
  last 62 days is kept. Training leads run to 351 hours from the 09:00 stamp.
- **Seeds.** Three seeds (0, 1, and 2), averaged quantile by quantile and then sorted.
- **Idle readings.** Readings flagged by the idle mask of [experiment
  210](00210-wind-idle-noise-mva-artefact/result.md) are set to zero in training and scoring until
  the mask ends. The mask is a research rule and not part of the production cleaning code, and from
  its end date idle readings count as observed, in scoring and in the training target of the last
  refit. That makes the last weeks of the backtest year a different scoring target.
- **Capacity.** The 99.9th percentile of the farm's cleaned power over the 730 days before each
  refit, which is 1.02 to 1.12 times the production effective capacity. The neural network trains
  with this scale and the scores divide by the production effective capacity.
- **Target and output scale.** The target is the observed power divided by the farm's production
  effective capacity (the last value in the `effective_capacity` table), with idle-mask rows set to
  zero, and then divided by the ratio of the new capacity to that effective capacity. The fit
  writes each quantile as the fitted fraction times the new capacity, in MW (or MVA for the two
  MVA farms).
- **Hardware and run time.** About 20 minutes of GPU time in 3 parallel processes for the 36 wind
  fits and 12 solar fits together (experiment 666).

### Post-processing

**The wind file has no post-processing and no calibrated copy.** Its stored coverage (0.913) already
exceeds 0.90, and a scale did not improve it ([experiment
683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md)).

### Scores

**The neural network beats climatology to day 10 and is 1.8% worse than climatology at days 10 to
14.** The CRPS gain over climatology at days 10 to 14 is -1.8% (95% interval -3.3 to -0.4). The
table gives CRPS in points of capacity from [experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md),
over 709,776 scored half-hours.

| Lead band | CRPS | Climatology CRPS |
|---|---:|---:|
| Days 0 to 1 | 4.58 | 9.72 |
| Days 1 to 3 | 5.47 | 9.73 |
| Days 3 to 7 | 7.68 | 9.67 |
| Days 7 to 10 | 9.36 | 9.69 |
| Days 10 to 14 | 9.86 | 9.68 |

- **Scale of the CRPS on the experiment 666 page.** [Experiment
  666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md) reports CRPS values half
  as large as these (for example 2.29 at days 0 to 1), because it defines CRPS as the mean pinball
  loss and experiment 668 as twice that. Compare numbers only within one page's convention, and use
  the experiment 668 convention, which [experiment
  667](00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md), [experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md), and
  [experiment
  683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md)
  share.
- **Coverage.** 0.913 over all rows (experiment 668). Wide bands at long leads may explain why the
  coverage stays near 0.90 (experiment 666).
- **Matched check.** On rows that [experiment
  659](00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md) also
  covers, the backtest scores 4.174 where experiment 659 scores 4.179, with both using the capacity
  of experiment 659. Both figures use the convention of the experiment 666 page.

### Code

**Last change** is the merged nged-ml-research pull request that last touched the file on
`origin/main`, with its commit. Several experiment folders import modules from earlier folders, so a
fit needs the import closure listed in the appendix. The input-building scripts in the second half
of the table are not imported by any fit and are not in the appendix. Paths are relative to the
repository root.

| File | Role | Last change |
|---|---|---|
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_common.py` | Constants, the capacity rule, the seeds, the quantile levels, and the data paths shared by wind and solar | PR 328 (`5acc5f4dc`) |
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_network.py` | Mixture head and training loop shared by wind and solar | PR 328 (`5acc5f4dc`) |
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_wind_members.py` | Ensemble member rows at the wind farms | PR 336 (`eb01acf75`) |
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_wind_features.py` | Per-case wind inputs | PR 336 (`eb01acf75`) |
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_wind_fit.py` | Monthly wind network S fits and forecasts | PR 336 (`eb01acf75`) |
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_export.py` | Writes `wind.parquet` and `solar.parquet` | PR 336 (`eb01acf75`) |
| `ml-experiments/00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/e00659_wind_fit.py` | Original network S fit | PR 319 (`1e3e197f3`) |
| `ml-experiments/00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/e00659_wind_features.py` | Original wind inputs; its output is the gate of `e00666_wind_features.py` | PR 319 (`6dc9c4f02`) |
| `ml-experiments/00259-wind-plant-plus-quantile-offset-head/e00259_hindcast.py` | Member rows for the runs 2024-04-01 to 2025-06-16 | commit `ea26a5465` |
| `ml-experiments/00615-bakeoff-wind-and-solar-development-month-backtests/e00615_wind_members.py` | Member rows for the runs 2025-06-17 to 2026-06-30 | PR 273 (`ae95c684e`) |
| `ml-experiments/00210-wind-idle-noise-mva-artefact/e00210_data.py` | Joins the farms' power to ERA5 100 m wind speed | commit `f7c12d0b4` |
| `ml-experiments/00210-wind-idle-noise-mva-artefact/e00210_analysis.py` | Detects idle readings and writes `mask.parquet` | commit `f7c12d0b4` |

**Production code for the same job.** The production repository's documented extension point for a
new model is a `BaseForecaster` subclass with its own `FeatureEngineer`. Whether the neural networks
use it is not decided.

- **Exists.** NWP wind components at 10 m and 100 m, derived wind speed and direction
  (`_derive_wind_features` in `tabular_feature_engineer.py`,
  `packages/weather_utils/src/weather_utils/wind.py`), and `time_series_type` with a wind value. The
  research member-row scripts call the production `TabularFeatureEngineer` in bulk mode, so the
  production code already yields the per-member wind inputs.
- **Differs.** Production has no wind-specific model. The research network S reads the mean and
  standard deviation over the 51 members and a one-hot farm column, and outputs a mixture's 11
  quantiles. Production trains on the control member.
- **Differs in the interface.** `BaseForecaster.train` and `BaseForecaster.predict` take
  `AllFeatures`, one row per time series, origin, valid time, and ensemble member, and `predict`
  returns a `PowerForecast` with an `ensemble_member` index and no column for a quantile level. The
  research network S needs one row per case that holds the 10 summaries over the 51 members, and
  one pooled fit over 3 farms for each refit. Its output is 11 quantiles at fixed levels.
- **Differs in the origin.** The research network trains and forecasts only from the 00Z run
  relabelled to a 12:00 UTC origin. Production starts ingesting the 00Z run at 10:30 UTC, retries
  every 30 minutes up to 8 times, and forecasts at 00:00, 06:00, 12:00, and 18:00 UTC. On a day
  when the 00Z run arrives after 12:00 UTC, the 12:00 slot would use the previous day's run, 24
  hours older. The
  wind lead feature would then exceed the 351 hours of training, and the last day of a 14-day
  forecast would lie beyond the 15 days of that run. The service must degrade in that case (the
  production rule is to widen the bands and record the degradation on the row), and no degradation
  rule for the neural networks is decided.
- **Missing.** Network S, the mixture head, ENS summary features, a capacity scale of the 99.9th
  percentile of the last 730 days, the idle mask (a research rule, not production cleaning), monthly
  refits, and quantile output.

### Port notes

- **Reproduce exactly.** The 64-by-64 layers with GELU, the zero-weight initial last layer and its
  biases, the 3-component mixture centred on the training mean, the standard-deviation clamp of
  `exp(clamp(raw, -6, 1))`, the input list, the lead measured from the 09:00 stamp, the 730-day
  99.9th-percentile capacity, and the monthly refit with a 62-day early-stopping window.
- **Unvalidated.** Wind has no power lag, so the neural network cannot react to a recent change of
  level. The idle mask is not production code. The behaviour on a stale or missing ensemble run, or
  on a case with fewer than 51 members, is untested: the research scripts raise an error there.
- **Gotchas.**
    - The quantiles are clamped at the network's own capacity, which is 1.02 to 1.12 times the
      production effective capacity. The README of the backtest folder lists runs of up to 9
      identical non-zero `q50` values at Wind A for that reason.
    - Two capacities are in play: the 730-day 99.9th percentile that the network trains on and the
      production effective capacity that the scores divide by.
    - The last 62 days before each refit serve for early stopping only.
- **Open questions.**
    - Whether to drop or damp the forecast beyond day 10, where the neural network scores worse than
      climatology.
    - Which script builds `data/handover/era5_wind_at_nged_wind_farms_anonymised.parquet`, which the
      idle mask needs. We found none.

## Solar

### Architecture

**The six solar farms are forecast by the simple shared neural network PLP of [experiment
659](00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md), which adds
a fixed physics-lite power to the ensemble weather.** The backtest is [experiment
666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md).

- **Inputs, 18 in all.** The mean and standard deviation over the 51 members of shortwave radiation,
  air temperature, and 10 m wind speed. The ensemble means of relative humidity, longwave radiation,
  precipitation, and surface pressure. The mean and standard deviation of the physics-lite power and
  of the Faiman cell temperature. The sun's elevation and the sine and cosine of its azimuth. The
  lead. There is no farm column and no calendar column. One pooled network is fitted over the 6
  farms for each refit and seed.
- **Lead definition.** The solar lead is the time from the 00Z run to the half-hour's valid time, so
  it runs from 12 to 347.5 hours. The wind network measures its lead from the 09:00 stamp instead.
- **Physics-lite power.** The physics-lite power is `clip((G / 1000) * (1 - 0.004 * (T_cell - 25)),
  0, 1)`, where G is shortwave radiation in W per square metre and `T_cell` is the Faiman cell
  temperature `T_air + G / (25 + 6.84 * wind speed)`. The formula runs on each of the 51 members
  separately, and the neural network reads the mean and standard deviation. The mixture is centred
  on the mean physics-lite power. It has no fitted parameters.
- **Layers and output.** As for wind: two hidden layers of 64 units with GELU activations and a
  3-component Gaussian mixture over power as a fraction of capacity, with the same last-layer
  initialisation, the same standard-deviation clamp, the same 40-step bisection over a residual
  range of -2 to +2 around the centre, and quantiles clamped to the range 0 to 1 of the neural
  network's capacity scale and then sorted. The input standard deviation has a floor of 1e-6.
- **Night gate inside the neural network.** The neural network forecasts only the half-hours where
  the mean cosine of the solar zenith angle over the ensemble step is above zero, and sets the
  quantiles of the other half-hours to 0.

### Input tables

**The solar fit reads one table per farm that `e00666_solar_inputs.py` builds from the production
stores, and it needs no earlier experiment's table, ERA5, or CAMS.** The table is
`data/ml-experiments/00666-*/solar_inputs/<site>.parquet`, with one file for each of the 6 farms. It
holds every half-hour of every 00Z run from 2024-04-01 to 2026-10-31, in the 14 day bands described
below, with `daylight`, the 18 inputs, and `actual`. The 18 inputs and `actual` are null where the
half-hour is not a daylight half-hour or a stamp is missing, and `actual` is also null where power
is unobserved. The script reads the ECMWF ENS `Nwp` Delta table (`NWP`), the cleaned power Delta
table, the `effective_capacity` table, and the metadata table, all under `DATA_PATH_INTERNAL`.

- **Run order.** `e00666_solar_inputs.py`, then `e00666_solar_fit.py`, then `e00666_export.py`
  (which writes `wind.parquet` and `solar.parquet` together). `e00666_solar_gate.py` is an optional
  check that compares the inputs with experiments 430 and 659. It imports modules from experiments
  310, 430, 461, and 659 and is not part of the pipeline.
- **The farm's weather comes from the nearest H3 cell by its centre.** `nearest_cell` picks the
  store's H3 cell whose centre is nearest the farm. The wind member rows instead use the cell that
  holds the farm (`h3_res_5`).
- **Complete stamps only.** The script keeps a stamp only when all 51 members are present, and it
  raises an error when a run from 2025-11-01 to 2026-10-31 lacks a complete stamp up to a lead of
  348 hours.
- **Stamps and day bands.** The ensemble stamps are 3-hourly up to a lead of 144 hours and 6-hourly
  after. Day `d` (1 to 14) holds the half-hours whose lead from the 00Z run is in `[12 + 24(d - 1),
  36 + 24(d - 1))` hours, so each half-hour of a run appears once and a run's 14 days cover 672
  half-hours.
- **Shortwave is a period mean spread by the zenith.** The ensemble value is a mean over the period
  that ends at the stamp. The script spreads it over the period's half-hours in proportion to the
  cosine of the solar zenith at the half-hour's midpoint, so that the period mean is unchanged.
- **Temperature and wind speed are interpolated.** Each is interpolated linearly between the
  previous stamp and the period's stamp, to the half-hour's midpoint, and falls back to the stamp
  when the previous stamp is missing. Relative humidity, longwave radiation, precipitation, and
  surface pressure are the ensemble means at the period's stamp, without interpolation.
- **Relative humidity is derived, not stored.** The production `Nwp` table has no relative-humidity
  column. The script computes it from the ensemble-mean 2 m temperature and 2 m dew point with the
  Magnus formula (constants 17.625 and 243.04): 100 times the ratio of the saturation vapour
  pressure at the dew point to the one at the air temperature, clipped to 0 to 100.
- **Sun position.** `pvlib` gives the apparent zenith and the azimuth at the half-hour's midpoint.
  The elevation input is 90 degrees minus the apparent zenith. A half-hour is a daylight half-hour
  when the mean cosine of the zenith over its 3-hour or 6-hour period is positive.
- **Constants.** Temperature coefficient -0.004, `U0` 25, `U1` 6.84, all in `e00666_common.py`.
- **Imports versus inputs.** The solar inputs come from the production stores through one script in
  the 666 folder, so the appendix statement that wind and solar need only that folder is right for
  solar's code and its data sources. It is wrong for wind (see the wind section).

### Training regime

- **Refits.** The solar neural network is refitted quarterly, on the first day of the first month of
  the backtest year and of each third month after it (2025-11-01, 2026-02-01, 2026-05-01, and
  2026-08-01). Each fit scores the next three months, uses three seeds (0, 1, and 2), and has no
  pre-training. Training uses valid times before one day earlier, with the
  last 61 days for early stopping, 20 epochs, daylight rows only, and leads below 348 hours from the
  00Z run, as [experiment 666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md)
  states ([experiment 430](00430-unified-forecaster-solar-and-pretraining-ablations/result.md) and
  [experiment
  659](00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md) stopped
  at 251.5 hours, so the lead range is extended). Solar trains on all daylight half-hours, including
  more low-sun half-hours than experiment 659 kept. The script asserts that no training row has a
  valid time at or after the refit.
- **Optimiser and seeds.** As for wind (AdamW, learning rate 0.002, batch 2,048, weight decay
  0.0001), keeping the epoch with the lowest negative log-likelihood on the last 61 days, with no
  patience rule. The target is the observed power over the farm's production effective capacity,
  divided by the ratio of the new capacity to that effective capacity. The network sees raw power,
  because solar has no idle mask.
- **Capacity.** As for wind: the 99.9th percentile of cleaned power over the 730 days before each
  refit. The fit writes each quantile as the fitted fraction times the new capacity, in MW.
- **Hardware and run time.** See the wind section.

### Post-processing

**The solar file carries a hard zero at night, set at each half-hour's own sun position, and a
calibrated copy adds a band scale.**

- **Night clamp.** Every quantile is zero on half-hours where the sun is not above the horizon at
  the middle of the half-hour. The clamp changes 271,808 of 1,471,680 rows (18.5%), the largest
  change is 6.1% of capacity, and the daylight rows are bit-identical to the unclamped file. It is
  needed because the gate inside the neural network uses the mean cosine over the 3-hour or 6-hour
  ensemble step, so a step that contains a sunrise or sunset leaves small non-zero quantiles at
  night.
- **The same rule as a gate.** [Experiment
  684](00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/result.md)
  applied the rule as a gate on the stored neural networks of [experiment
  666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md) and found no non-zero
  night quantile left and no change to any daylight or dawn-and-dusk score, with a lower night CRPS.
  The share of night observations above q99 rises, because the gate sets those quantiles to zero.
  Dawn and dusk (sun elevation under 5 degrees) stay badly calibrated: the q05-to-q95 coverage is
  0.69 and 0.75 at the two farms tested, and 0.30 and 0.24 of observations fall below q01. The gate
  was tested at two farms over three months of origins.
- **Walk-forward band scale (calibrated copy only).** See "Walk-forward band calibration" below,
  applied to daylight rows from 2025-12 with night quantiles kept at zero.

### Scores

**The neural network beats climatology at every lead band, and the gain shrinks from 37.1% at days 0
to 1 to 7.6% at days 10 to 14.** The table gives CRPS in points of capacity on daylight half-hours
(685,472 scored) from [experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md).

| Lead band | CRPS | Climatology CRPS |
|---|---:|---:|
| Days 0 to 1 | 4.83 | 7.68 |
| Days 1 to 3 | 5.33 | 7.69 |
| Days 3 to 7 | 6.35 | 7.71 |
| Days 7 to 10 | 6.89 | 7.73 |
| Days 10 to 14 | 7.18 | 7.76 |

- **Coverage.** 0.893 on daylight half-hours. About 11% of observations exceed q99 on the sunniest
  days of [experiment
  659](00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md), so the
  upper band fails on very sunny days.
- **Scale and row set.** [Experiment
  666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md) reports CRPS values
  about half as large, for the reason given in the wind section, on a daylight flag defined by the
  mean cosine over the ensemble step. Its solar numbers are therefore not comparable with the table
  above.
- **Seasonal bias.** The solar median is high in spring and low in autumn, and the neural network
  has no calendar input. One year with quarterly refits cannot separate a seasonal effect from refit
  age (experiment 666).
- **At days 10 to 14,** the neural network scores worse than climatology from November to March and
  better from April to September (experiment 666).

### Code

**Last change** is the merged nged-ml-research pull request that last touched the file on
`origin/main`, with its commit. Several experiment folders import modules from earlier folders, so a
fit needs the import closure listed in the appendix. Paths are relative to the repository root.

| File | Role | Last change |
|---|---|---|
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_common.py` | Constants, the capacity rule, the seeds, the quantile levels, and the data paths shared by wind and solar | PR 328 (`5acc5f4dc`) |
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_network.py` | Mixture head and training loop shared by wind and solar | PR 328 (`5acc5f4dc`) |
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_solar_inputs.py` | Solar inputs including physics-lite power | PR 336 (`eb01acf75`) |
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_solar_fit.py` | Quarterly solar network PLP fits and forecasts | PR 336 (`eb01acf75`) |
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_export.py` | Writes `wind.parquet` and `solar.parquet` | PR 336 (`eb01acf75`) |
| `ml-experiments/00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/e00659_solar_fit.py` | Original network PLP fit | PR 319 (`6dc9c4f02`) |
| `ml-experiments/00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/e00659_solar_wind.py` | Member arrays with 10 m wind speed | PR 319 (`6dc9c4f02`) |
| `ml-experiments/00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/e00684_solar_gate.py` | Night gate at each half-hour's own sun position | PR 348 (`70523c738`) |
| `ml-experiments/00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/e00668_clamp_fixes.py` | Night-time solar clamp and MVA lower-quantile clamp | PR 342 (`6640908c0`) |

**Production code for the same job.** The production repository's documented extension point for a
new model is a `BaseForecaster` subclass with its own `FeatureEngineer`. Whether the neural networks
use it is not decided.

- **Exists.** Shortwave and longwave radiation in `Nwp`, resampled to half-hours in
  `packages/ml_core/src/ml_core/features/_radiation.py`, and `time_series_type` with a PV value.
  `Nwp` also holds 2 m temperature, 2 m dew point, precipitation, and surface pressure.
- **Differs.** Production has no solar-specific model. The research network PLP adds a physics-lite
  power and a Faiman cell temperature computed per ensemble member, and outputs 11 quantiles with
  zeros at night.
- **Differs in the inputs.** Production's `_radiation.py` shapes shortwave with a clear-sky index
  and fits a piecewise cubic Hermite interpolating polynomial through longwave. The research code
  spreads shortwave by the cosine of the zenith, and takes longwave, precipitation, and pressure at
  the period's stamp. We did not compare the two numerically. Production has no relative-humidity
  column, so the port must derive it with the Magnus formula from the ensemble-mean temperature and
  dew point. The research code takes the H3 cell whose centre is nearest the farm.
- **Differs in the interface and the origin.** The interface mismatch and the origin mismatch are
  those described for wind: `BaseForecaster.train` and `predict` take `AllFeatures` keyed by
  ensemble member, where the research network PLP needs one row per daylight half-hour that holds
  the 18 inputs, a pooled fit over 6 farms, and 11 quantiles. The research network trains only on
  the 00Z run relabelled to a 12:00 UTC origin, and its lead (from the 00Z run) is trained to less
  than 348 hours. A stale run at the 12:00 slot would give leads of up to 371.5 hours, beyond
  training. No degradation rule is decided.
- **Missing.** Network PLP, the physics-lite power and cell-temperature code, the sun-position night
  gate and clamp, ENS summary features, the 730-day capacity scale, quarterly refits, and quantile
  output.

### Port notes

- **Reproduce exactly.** The physics-lite formula with the Faiman cell temperature, the 18 inputs,
  the mixture centred on the mean physics-lite power, the GELU layers and the last-layer
  initialisation, the daylight-only training rows, the lead measured from the 00Z run, and the post
  hoc night rule at each half-hour's own zenith.
- **Unvalidated.** The night gate was tested at two farms and one refit. Dawn and dusk coverage is
  poor. Arm b of [experiment
  684](00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/result.md),
  which would feed the physics-lite power at each half-hour's own zenith to the neural network, was
  not run. The behaviour on a missing stamp or a stale run is untested: the research script raises
  an error on a gap in the archive.
- **Gotchas.**
    - The quantiles are clamped at the network's own capacity, which differs from the production
      effective capacity that the scores divide by.
    - The last 61 days before each refit serve for early stopping only.
    - The solar daylight flag of the networks uses the mean cosine over the ensemble step, and the
      night clamp uses the half-hour's own zenith. The two differ at sunrise and sunset.
- **Open question.** Whether a calendar input or monthly refits remove the seasonal bias in the
  median.

## Walk-forward band calibration

**The backtest files hold one calibration, a scale on the width of the quantile bands fitted only on
earlier months, which brings the four calibrated outputs close to 0.90 coverage.** [Experiment
683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md)
tested it on the origin months 2025-12 to 2026-06, and the README of the backtest folder describes
the four calibrated copies. The scale is post-processing on stored quantiles and is not part of any
neural network above.

- **Method.** Each quantile's distance from `q50` is multiplied by one factor per series, output,
  origin month, and lead band (days 0 to 1, 1 to 3, 3 to 7, and 7 to 14). The factor is the 90th
  percentile of the ratio of the actual power's distance from `q50` to the stored half-width of the
  band on that side (`q95 - q50` above, `q50 - q05` below), which makes the scaled band cover 90% of
  the fitting rows. The fitting rows are the half-hours whose valid time lies in the 90 days that
  end 6 hours before the month starts, because power is available only up to 6 hours before an
  origin. The factor is bounded to 0.5 to 3. In the code, a cell with fewer than 20 fitting days
  keeps a scale of 1, and rows whose stored half-width is under 0.05 of capacity are left out of the
  fit. There is no level shift, so `q50` is unchanged. After scaling, the quantiles of MVA series,
  wind, and solar are clipped at zero.
- **Where it applies.** The prevailing Primaries rows, the BSPs, the GSPs, and solar. It does not
  apply to wind, whose coverage already exceeds 0.90, and it does not apply to the NRA Primaries
  rows. Two arms of experiment 683 tried the NRA rows. Arm 2, the walk-forward scale, brought their
  coverage to only 87.1% and raised their CRPS by 0.081 points. Arm 1, the fixed factor 1.0934 about
  the median that experiment 613 fixed, lowered their CRPS by 0.027 points (interval -0.031
  to -0.023) and raised their coverage to 86.0% to 87.6%. The backtest files apply neither arm to the
  NRA rows. The first month, 2025-11, has no earlier forecast to fit on and is left as stored.
- **Effect on coverage.** On 2025-12 to 2026-06, the q05-to-q95 coverage and CRPS change are as in
  the table, each coverage a range over the four lead bands (experiment 683).

| Output | Stored coverage | Calibrated coverage | CRPS change |
|---|---|---|---:|
| Prevailing Primaries | 84.0% to 87.9% | 88.0% to 88.4% | +0.047 |
| BSPs | 87.5% to 89.3% | 89.8% to 90.9% | +0.005 |
| GSPs | 86.4% to 90.3% | 89.1% to 89.8% | +0.003 |
| Solar | 87.8% to 89.5% | 89.4% to 91.0% | -0.002 |

- **Adding a level shift made the files tested worse.** A per-series level shift fitted on earlier
  months raised the NRA Primaries CRPS by 0.306 points and the GSPs' by 0.842, and experiment 683
  reports that every file got worse. The median bias changes from month to month.
- **A scale fitted through an unexplained stretch overshoots afterwards.** A scale fitted through
  the July and August 2026 load transfers overshoots after they end: D18 (identifier 13) reaches
  97.5% coverage in 2026-09 under the scale. The calibrated copies hold D18 and D14 (identifiers 13
  and 16) at their 2026-06 factor from 2026-07, a rule chosen after seeing the final-test months, so
  it is post hoc. D18 stays at 63% to 77% coverage in 2026-07, 2026-08, and 2026-10 even so.
- **The winter NRA shortfall is mostly error in the median.** [Experiment
  688](00688-winter-normal-running-coverage-median-error-and-events-part-1/result.md) dropped the
  half-hours inside labelled and verified load transfers and shifted each band by its month's
  observed median error, and the coverage of the NRA Primaries rose from 80.1% to 90.1% in 2025-12
  and from 80.3% to 88.6% in 2026-01. The shift used the month's own errors, so it is a diagnostic
  and not a method. The 80.1% and 80.3% baselines are the NRA bands after the walk-forward scale
  (arm 2 of experiment 683). The stored files give 81.7% and 77.7%. Experiment 688 supersedes the
  suggestion of experiment 683 to try a learned width head, because the stop rule that would have
  justified it did not fire.
- **A width scale fitted on a fixed lookback does not remove the steps in band width at a refit.**
  [Experiment
  689](00689-fixed-lookback-width-scale-versus-seed-averaging-at-refit-boundaries/result.md) took
  the 151 series-and-lead-band cells (of 384, 16 series by 6 refit boundaries by 4 lead bands) whose
  stored band width changes by more than 10% across a refit boundary from 2026-01 to 2026-06. Under
  the 90-day scale of experiment 683, 15.2% of those cells end with a before-to-after width ratio of
  0.90 to 1.10, against a predicted 90%. A 45-day lookback gives 9.9% and a 180-day lookback gives
  7.3%. The 90-day scale moves the ratio farther from 1 in 67% of the cells. Fitting the scale on
  each seed's quantiles and averaging gives nearly the same ratios: they lie within 0.05 of the
  seed-mean scale's in 89.4% of cells, just under the pre-registered 90%, so the steps are not an
  artefact of averaging the quantiles of two seeds.
- **Caveats of experiment 689.** The 151 cells share 6 refit boundaries, so they are not 151
  independent trials. The mechanism is untested: the scale for a month is fitted on forecasts of the
  neural network that the refit replaces, and the experiment fitted no scale on any forecast of the
  new network. The September step of D18 (identifier 13) lies after the last scored origin month and
  was not tested. Only the NRA output has per-seed files, so the prevailing output was not tested.
- **Calibrated copies on disk.** `primaries_calibrated.parquet` (prevailing rows),
  `bsps_calibrated.parquet`, `gsps_calibrated.parquet`, and `solar_calibrated.parquet`. The NRA rows
  of the first are copied unchanged. The prevailing rows of `primaries_calibrated.parquet` were
  built from the version of `primaries.parquet` from before the extra lagged inputs, so they do not
  yet carry them. The README of the backtest folder lists their SHA-256 prefixes.
- **Open questions.** Whether to refit the scale at every refit of a forecasting neural network (not
  decided, because the diagnostic was not run). The band width steps at some month-start refits, and
  the scale then overshoots for more than one month: the days 10 to 14 width of six Primaries (D11,
  D22, D5, D16, D21, and D2, identifiers 8, 15, 17, 18, 24, and 25) widens by 1.21 to 1.48 times at
  the February refit, and their calibrated coverage goes from 76.3% in January to 93.1% in February.
  D18 widens by 1.48 times at the September refit, and D4 (identifier 11) narrows by 0.76 times
  (experiments 683 and 689 list this as the April refit, and [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  as the refit that starts May 2026).
- **Research code.**
  `ml-experiments/00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/e00683_calibrate.py`
  holds `fit_month` (the scale), `apply_arm`, and the scoring, and `e00683_refit.py` measures band
  width across refits. The clamps are in
  `ml-experiments/00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/e00668_clamp_fixes.py`.
  The scripts that wrote the four calibrated copies are in
  `ml-experiments/00689-fixed-lookback-width-scale-versus-seed-averaging-at-refit-boundaries/calibrated_copies/`
  (see its `README.md`). `e00689_make_calibrated_supply_and_solar.py` writes the BSP, GSP, and solar
  copies from the arm 2 output of experiment 683, and `e00689_make_calibrated_primaries.py` writes
  the prevailing Primaries copy and its table of scales, including the hold of D18 and D14 at their
  2026-06 factor. That README states that the four copies reproduce the stored files byte for byte.
- **Production code for the same job.** None. The production repository has no clamp, calibration,
  or band-widening step.

## Production-port notes across forecast types

- **Series and units.** Carry the unit with the series, because the 8 MVA Primaries and 2 of the 3
  wind farms are apparent-power series and the rest are MW.
- **Origin and lags.** Every neural network reads the ENS 00Z run of the origin day as available at
  09:00 UTC and is relabelled to a 12:00 UTC origin. The Candidate's own lags are at least 24 hours
  old at its 09:00 origin, the BSP and GSP lags at least 3.5 hours old, and the prevailing head's
  extra lags at least 6 hours old. Wind and solar read no power lag. [Experiment
  572](00572-two-offset-part-1-lags-versus-the-slow-output-offset/result.md) tested a 3.5-hour lag
  cap at the Primaries, and none of the pages read for this one tests it together with the
  Candidate's neighbour signals and the prevailing head.
- **What a production forecasting service has to store.** For each demand and generator neural
  network, the fitted weights of each monthly (or quarterly, for solar) refit. For the prevailing
  head, a bank of injected-shift rows per earlier month. For calibration, a scale per series and
  lead band refreshed each month. For the Primaries, the lag-free forecast table: the median of the
  lags-masked neural network at the freshest earlier origin with a lead of at most 24 hours, for the
  Primary and for every neighbour, produced every day. The calibration also needs the neural
  network's own stored forecasts from the previous 90 days beside the observed power.
- **Serving path.** The chain for a Primaries forecast is longer than loading a model and calling
  `predict`. The lags-masked neural network writes the lag-free forecast table; the table feeds the
  neighbour and gap signals and the 14 extra lags; the Candidate runs for each seed (2 seeds) and
  the seeds are averaged; the survival head acts on the Candidate's quantiles to give the prevailing
  output; the MVA clamp acts on the quantiles of both outputs; and a calibration step reads 90 days
  of stored forecasts and observed power and rescales the bands. The clamps and the band scale are
  therefore not the only steps that act on predicted quantiles. Sources:
  `ml-experiments/00667-*/e00667_cand.py` (`ensure_tables`),
  `ml-experiments/00598-*/e00598_signals.py` (`build_lag_free_table`),
  `ml-experiments/00683-*/e00683_calibrate.py` (`LOOKBACK_DAYS = 90`), and experiment 668 (seeds and
  head). The production `CLAUDE.md` states two design principles that this chain trades away: keep
  the serving path close to "load a model, call `predict`", and put a capability in the training
  loop when it could live in either place. The band scale could be fitted at refit time, which is
  the second principle's case. The page records the trade and does not decide it.
- **Unmetered solar.** None of the neural networks above has an explicit term for unmetered solar
  behind a Primary. [Experiment
  673](00673-pv-capacity-for-nged-per-primary-with-register-cross-check/result.md) reads the
  capacity of unmetered solar from a neural-network probe of the response of net demand to sunshine,
  and calls its figures provisional. The probe gives 2.4 to 3.9 MW at six of the eight MW Primaries
  (D5, D9, D10, D11, D14, and D16) and 1.25 to 6.62 MW at five MVA Primaries (D2, D3, D4, D19, and
  D21). [Experiment 679](00679-pv-mva-conversion-loss-reactive-floor-at-half-level/result.md)
  measures the loss in recovery of an added solar farm when MW readings are converted to MVA, as 0.5
  percentage points (80% interval -1.0 to 7.7) if the meters read power factor 1 with a small
  reactive floor. These results are context and are not inputs to any neural network above.

## Gotchas across forecast types

**Each item below caused a wrong number or a wrong assumption in the backtest, or would in a port.**
The per-type sections give the detail where it belongs.

- **Forecast every half-hour.** The stored test matrices of the earlier experiments drop half-hours
  without an observed reading, because they are training matrices. Experiment 669 rebuilt them for
  the BSPs and GSPs in `ml-experiments/00669-*/e00669_testrows.py`, and experiment 667 rebuilt the
  Primaries' rows from 2025-07-01 onward (`REPLACE_FROM` in `e00667_extend.py`). A port must
  forecast every half-hour of the horizon.
- **The last 62 days of a wind refit window only stop training early.** Wind cases whose origin is
  at or after the cut-off (62 days before the refit date) and whose valid time is before the refit
  date are used to choose the best epoch and never enter a gradient (`STOP_WINDOW_DAYS` in
  `e00666_wind_fit.py`).
- **ERA5 arrives about 5 days late.** At a real cut-off, the last roughly 5 days of ERA5 and CAMS
  pre-training rows could not be read. The BSP and GSP section states this. The same lateness
  applies to any Candidate or wind input built from ERA5.
- **Injected lags use later power for proportional shifts.** In the Primaries' injected-shift bank,
  the lag inputs of a proportional shift use the month's later power. Experiment 667 lists this as a
  caveat.
- **The unclamped prevailing `q01` is more negative than the NRA `q01` at some MVA Primaries.** The
  lowest prevailing `q01` is -2.26 MVA at one MVA Primary against -1.21 for NRA, before the clamp
  (experiment 667, Caveats).
- **Wind A has flat `q50` runs.** Runs of up to 9 identical non-zero `q50` values occur at Wind A,
  because the quantiles are clamped at the neural network's own capacity (backtest README).
- **At least two capacities are in play.** The wind and solar networks train with the 99.9th
  percentile of the 730 days before each refit. Every score divides by the production effective
  capacity, one value per series. The neighbour power ratios of the demand neural networks divide
  the `capacity_mw` column of `site_map.csv` (set in `world()` of `e00647_data.py`), a third
  capacity that is not the effective-capacity table.
- **Hide all power after the cutoff in every refit.** Experiment 667 sets the power after the cutoff
  to NaN in its `train` phase (`TRAIN_ONLY` in `e00667_fold.py`). Experiment 669 has the same switch
  (`E00669_HIDE=1`), and its docstring says that only the pre-flight gate used it, so the year's
  fits relied on row selection alone. A port should hide the power by construction and test the rule
  (test 2 in "Tests to write").
- **Regime changes in the training history.** D8's reading includes one gas engine's output until
  2023-08-15, and the [data-processing page](data-processing.md) recommends an individual start date
  or a flag for such a series. The Candidate's pre-training spans 2019-09 onward. Whether the
  Candidate's rows handle this change is an open question below.

## Open issues for the porter: where production conflicts with the research fits

**Each item states a fact about production and the matching fact about the research code, with
sources. None is an agreed design.**

- **The neural networks were trained on one origin, and production runs at four origins with a
  stale-run fallback.** Every research network was trained only on a 09:00 native origin from the
  same day's 00Z run, relabelled to 12:00. Production `live_forecasts` runs at 00:00, 06:00, 12:00,
  and 18:00 UTC (`defs/live_forecast_assets.py`). The `ecmwf_ens` asset ingests at 10:30 UTC and
  retries every 30 minutes, up to 8 times, so the 12:00 slot can use the previous day's run
  (`defs/schedules.py`). With the previous day's run, every lead grows by 24 hours, past the
  training lead range (experiment 666 gives a cap of 348 hours for solar, and the Candidate's range
  was not checked), and the lag ages change. The research page set has no rule for this regime. Rule
  1 of `docs/design-philosophy/inherent-stability.md` in the production repository requires
  production to degrade, widen the bands, and record the degradation.
- **Some research inputs do not exist in production at inference time.** The previous day's CAMS
  irradiance enters two columns of the Candidate and of the BSP and GSP networks at inference, and
  production stores no CAMS (`weather_schemas.py` defines `Nwp` for ECMWF ENS only). The monthly
  pre-training reads ERA5 and CAMS from 2019-09 to the cutoff, and ERA5 arrives about 5 days late.
  The solar network's relative humidity is not a column of `Nwp`, which has temperature and dew
  point, so the research code derives it with the Magnus formula (constants 17.625 and 243.04 in
  `e00666_solar_inputs.py`) from the ensemble-mean values. The demand neighbour lists are read from
  two NGED files, `Monitor Direction.csv` and `Primary Substation Interconnections.csv`, in
  `e00400_blend.py`, and the production repository mentions the second file only in
  `docs/roadmap/data-sources.md`.
- **The analysis proxies differ.** The production proxy is the control member of the freshest run,
  filled column by column from older runs. The research proxy is the ensemble mean of the freshest
  run published by the origin, which is the previous run's first day
  (`ml-experiments/00437-*/e00437_features.py`). A port needs the research definition for every
  column that weather lags read.
- **The serving chain is longer than load and predict.** See "Serving path" in "Production-port
  notes across forecast types" for the chain and the two production principles it trades away.
- **The research models do not take `AllFeatures`.** The production `BaseForecaster.train` and
  `predict` take a `pt.LazyFrame[AllFeatures]`, keyed by `ensemble_member`. The Candidate needs
  pooled rows of 51-member ensemble summaries, inputs from neighbouring series, and multi-stage
  pre-training on ERA5 and CAMS. A different `FeatureEngineer` can supply a different data view,
  according to the production `CLAUDE.md`, but the page read no code that does so for these
  networks. The forecast-type sections list the mismatch under "Differs".
- **The output contract differs.** See "Output contract": production has no quantile columns and no
  output-kind column, and its `valid_time` constraint excludes the lead-0 row of the backtest files.

## Where the pages disagree

- **CRPS scale.** [Experiment
  666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md) defines CRPS as the mean
  pinball loss, and [experiment
  667](00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md), [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md),
  [experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md),
  [experiment
  683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md),
  and this page define it as twice that. The wind and solar numbers of experiment 666 are about half
  the numbers here. The current convention is the experiment 668 one.
- **Prevailing Primaries scores.** Experiment 667 reports a prevailing CRPS of 4.14, for the head
  without the extra lagged inputs. Experiment 668 reports 4.09 for the file after nged-ml-research
  PR 353. The current page is experiment 668.
- **Clamp counts.** The body of PR 342 counts 202,598 rows changed in the Primaries file, and
  experiment 668 counts 206,487. The current page is experiment 668.
- **The D4 and D18 width steps.** Experiment 683 places the days 10 to 14 narrowing of D4
  (identifier 11) at an April month start, with a factor of 0.76, and experiment 689 agrees.
  Experiment 668 places it at the refit that starts May 2026 and calls it about a third, and the
  README says about 37%. For D18, experiment 683 gives a widening of 1.48 times, experiment 668
  about half, and the README 55%.
- **The first month on the survival head.** The README says the refit that starts 2026-02 is the
  first month on the survival head that replaces the fallback head, and experiment 668 and the
  README's own "Known limitations" list say the fallback head serves 2025-11 and 2025-12 and R-haz
  serves 2026-01 onward. The current page is experiment 668.
- **Calibrated-copy scores.** The README gives CRPS changes of +0.044 (prevailing Primaries), +0.001
  (BSPs), +0.026 (GSPs), and +0.006 (solar), against +0.047, +0.005, +0.003, and -0.002 in
  experiment 683. The README's prevailing figures come from the file from before the extra lagged
  inputs. The current page for the experiment's own numbers is experiment 683.
- **"As stored" NRA coverage.** [Experiment
  688](00688-winter-normal-running-coverage-median-error-and-events-part-1/result.md) calls 80.1%
  and 80.3% the bands as stored. They are the scaled bands of experiment 683, whose stored values
  are 81.7% and 77.7%.
- **BSP and GSP coverage.** [Experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md) gives
  pooled q05-to-q95 coverage of 0.883 to 0.870 across its four lead bands for the four series
  together, and experiment 668 gives 86.7% for the BSPs and 87.7% for the GSPs, each as one figure
  per group. The two pages pool different rows. The pages do not state whether the row sets also
  differ, and this page did not check.
- **Solar lead cap.** Experiment 666 gives training leads below 348 hours and experiment 668 says
  the range was extended to 336 hours. The current page is experiment 666.
- **Width factor of the Candidate.** [Experiment
  612](00612-nra-combined-networks-three-class-roster-completion/result.md) mentions a band scaled
  by 1.12 to 1.18, and [experiment
  613](00613-nra-roster-contrasts-with-intervals-five-seeds/result.md) and [experiment
  620](00620-bakeoff-final-test-demand-networks-at-primaries/result.md) use 1.0934. This page uses
  1.0934 because the final test used it.
- **Extra-lag reaction gains.** [Experiment
  681](00681-extra-lagged-power-inputs-for-the-prevailing-forecast/result.md) reports 19.1 and 20.4
  hours sooner for 8 and 14 inputs, and [experiment
  687](00687-causal-detector-features-for-the-prevailing-forecast-heads/result.md) reports 1.25 days
  sooner for extra lags, on different banks and folds.
- **Calibrated copy of the prevailing Primaries rows.** `primaries_calibrated.parquet` was built
  before the extra lagged inputs, so its prevailing rows differ from those of `primaries.parquet`
  that this page scores.

## Open questions

These are listed for the maintainer. None is decided here.

- Whether the Primaries Candidate should read 3.5-hour lags, as the BSP and GSP neural network does,
  and what the answer does to the prevailing head.
- Which extra-lag set the prevailing head should use: the 8-input set or the 14-input set
  ([experiment 681](00681-extra-lagged-power-inputs-for-the-prevailing-forecast/result.md)).
- Whether the truncated Gaussian output should replace the post hoc clamp at all eight MVA
  Primaries, and what it does to the prevailing output ([experiment
  684](00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/result.md)).
- Whether to refit the band scale at every neural network refit, and how to treat the band-width
  steps at month-start refits ([experiment
  683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md)
  and [experiment
  689](00689-fixed-lookback-width-scale-versus-seed-averaging-at-refit-boundaries/result.md)).
  Experiment 689 shows that a fixed-lookback scale does not remove the steps, and it did not test a
  scale fitted on the new network's own forecasts.
- Whether the band scale belongs in the training loop or in the serving path (see "Serving path"
  above).
- Which forecast slots (00:00, 06:00, 12:00, and 18:00 UTC) each neural network must serve, and what
  a stale ENS run does to each (see "Open issues for the porter").
- Whether a port rebuilds the cached input tables from source or ports the lineage of experiments
  607, 437, and 564 (see the "Input tables" subsections).
- Whether the Candidate's pre-training rows handle D8's gas-engine regime change on 2023-08-15. The
  code was not read for this.
- Why the NRA bands of D18 and D14 (identifiers 13 and 16) cover only about 22% to 28% of
  observations in July and August 2026. [Experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  suggests a load shift that normal running ignores. [Experiment
  685](00685-event-table-on-the-final-test-window-all-primaries/result.md), an event table on the
  final-test window, finds no transfer between the two Primaries in those months, and names a level
  offset in the lag-free forecast as an untested candidate. [Experiment
  690](00690-slow-level-state-on-primaries-13-and-16-july-august/result.md) fitted a smooth slow
  level in hindsight, with the candidate windows masked, on a different forecast's residual. The
  level tracks the change (weekly correlation 0.66 at D18 and 0.87 at D14), and adding it to `q50`
  halves the median absolute error at both Primaries, in hindsight and not as a forecast. The level
  ramps over months and does not follow the step on 1 July. It moved the median absolute error of
  two control Primaries by 5.5% and 7.5%, against a limit of 5%, so the verdict is partly supported,
  and the experiment proposes no new method. The miss is still not explained by a load transfer
  (experiment 685), and the current pages are experiments 685 and 690.
- What the Candidate needs and what each component adds: pending experiment 677.
- Whether wind needs a power lag or a different long-lead rule, given its score against climatology
  at days 10 to 14.
- Whether a calendar input or more frequent refits remove the solar median's seasonal bias.
- How to model load shifts at the BSPs and GSPs, where the largest observed miss looks like one.
- Whether the BSP and GSP fine-tune carries synthetic lag-only shifts. The code path suggests that
  it does not (`e00669_fit.py` calls the fine-tune of experiment 574 directly, where the Candidate's
  chain uses synthetic shifts), and no run checked this.

## Tests to write

**A port should pass the checks below before any score is compared, and each check states the
property it protects and the source that supports it.** The thresholds come from the pages named. A
threshold that no page states is marked as a choice for the porter.

1. **No lookahead, by perturbation.** Property: no input depends on power or weather that was not
   available at the origin. Perturb all power after the cutoff and after the native origin minus
   each lag cap, which is 24 hours for the Candidate, 3.5 hours for the BSPs and GSPs, and 6 hours
   for the extra lags, and check that no input, no lag-free table entry, no signal, and no output
   moves. Repeat with the NWP of later runs. Sources: experiment 668 (a test perturbs every later
   half-hour for the extra lags), experiment 669 (a perturbation test found no leak; no lag index is
   newer than 05:30 on the origin day).
2. **Hidden-future training.** Property: a refit at month M reads no row at or after M. Fit one fold
   with all power after the cutoff set to NaN and require identical weights. Sources:
   `e00667_fold.py` (`TRAIN_ONLY`), `e00669_fold.py` (`E00669_HIDE=1`), and experiment 667 (reviews
   found no leak into any fold's training data).
3. **Golden output.** Property: the port reproduces the stored forecasts. Re-run one stored fold,
   for example 2026-03, and compare with the backtest file within a tolerance. The GPU re-run of the
   2026-03 GSP fold matched the stored forecasts to 5e-6 of capacity (experiment 669). For the
   Primaries, the 12:00 NRA rows equal the native 09:00 seed-mean quantiles times capacity exactly,
   with a maximum difference of 0.0 (experiment 667, "Checks").
4. **Contract.** Property: the output has the layout in "Output contract". Check 672 rows per origin
   and series, no duplicate key, no null or non-finite quantile, no crossing in any adjacent pair,
   UTC, Int8 identifiers, `valid_time` as the end of the half-hour, and one `NRA` row and one
   `prevailing` row per Primary key. Sources: experiments 667, 668, and 669, "Checks". Validation
   against the production `PowerForecast` needs a quantile column set that production does not have.
5. **Clamps and gates.** Property: the post-processing rules hold. MVA Primaries have no quantile
   below zero. MW Primaries are not clamped. Solar quantiles are zero where the zenith at
   mid-half-hour is 90 degrees or more, and daylight rows are bit-identical before and after the
   clamp. Sources: the backtest README ("Post-processing and checksums") and experiment 668.
6. **Calibration.** Property: the scale follows the rule in "Walk-forward band calibration". The
   `q50` is unchanged. Each factor lies in 0.5 to 3. Each factor uses only valid times at least 6
   hours before the month starts. A cell with fewer than 20 fitting days keeps a scale of 1.
   Coverage on the fitting rows is 90% when no bound binds. Sources: `e00683_calibrate.py` and the
   backtest README.
7. **Scores and coverage by lead band.** Property: the port reproduces the published scores.
   Recompute the tables of experiment 668. The Opus review of experiment 667 reproduced every score
   to 3 decimals, and the Sonnet audit of experiment 669 reproduced its scores within 0.03 points. A
   tolerance for a port is a choice for the porter.
8. **Monthly refit.** Property: the refit loop is walk-forward. A refit at month M trains only on
   data before M (check 2), and the band-width step at each refit is reported as experiments 683 and
   689 measured it.
9. **Degradation.** Property: the service degrades and never raises when an input is missing, as
   rule 1 of the production inherent-stability page requires. Check a missing lag (the Candidate
   trains with 30% lag masking), a missing neighbour, a stale or missing ENS run, missing CAMS, and
   fewer than 2 banks for the survival head. The service must always output a forecast, widen the
   bands, and record the degradation on the row. No page defines the widening for these cases.
10. **Architecture and parameter count.** Property: the port builds the architecture that was
    fitted. Assert the layer widths of each network and the parameter count against the stored `.pt`
    files in `data/ml-experiments/`. The widths are stated in each forecast-type section, and the
    count is not stated on any merged page.

## Appendix: modules each fit imports

**This appendix follows `import` statements only, so it lists the modules that a fit imports and
does not list the scripts that write the cached tables a fit reads.** The cached tables, and the
scripts that build them, are in the "Input tables" subsection of each forecast-type section. Wind
and solar copy their code into the 666 folder, and the fit imports of wind and solar need only that
folder, but their inputs are built elsewhere (see "Input tables"). The lists below come from
following every `import e00NNN_...` and `import p667_...` statement from the entry files, so they
name files that a porter must read or copy. The Candidate fit and the prevailing head share most of
their modules.

**The research code also imports modules of the `studies` package.** The package is at
`src/studies/` in the research repository, and the files that import it include the entry files
above. The modules found by a search of the fit, export, and calibration scripts are
`studies.sources` (data folder paths), `studies.power` (`scan_power`), `studies.pv_dataset`
(`pv_sites` and `wind_sites`), `studies.solar` (`zenith`), and, in the scoring and figure scripts of
the experiment folders listed, `studies.eval_protocol` (metrics and slices). `e00666_solar_gate.py`
imports no `studies` module and belongs to the wind and solar folder; the page read no evidence that
the backtest export imports it, and the closure of its own imports was not walked here.

**Candidate fit and export (entry files `e00667_cand.py` and `e00667_export.py`).** (97 files in 45
folders; folders of `ml-experiments/`, with the folder's slug omitted).

- `00011`: `e00011_term_dates.py`
- `00048`: `e00048_build_frames.py`, `e00048_common.py`, `e00048_gen_common.py`
- `00087`: `e00087_common.py`, `e00087_fit_demand.py`, `e00087_offset.py`, `e00087_regimes.py`
- `00377`: `e00377_adapter.py`, `e00377_score.py`
- `00381`: `e00381_score.py`
- `00391`: `e00391_stage1.py`
- `00397`: `e00397_fit.py`, `e00397_models.py`, `e00397_prep.py`, `e00397_score.py`
- `00400`: `e00400_blend.py`, `e00400_mixture.py`
- `00402`: `e00402_counterfactual.py`, `e00402_inject.py`, `e00402_mixture.py`, `e00402_worlds.py`
- `00404`: `e00404_data.py`, `e00404_fit.py`, `e00404_lagless.py`, `e00404_models.py`,
  `e00404_report.py`, `e00404_rows.py`, `e00404_score.py`, `e00404_trust.py`
- `00407`: `e00407_fit.py`, `e00407_models.py`
- `00410`: `e00410_stale_predict.py`
- `00418`: `e00418_prep.py`
- `00422`: `e00422_prep.py`
- `00425`: `e00425_prep.py`
- `00428`: `e00428_data.py`, `e00428_fit.py`, `e00428_history.py`, `e00428_models.py`,
  `e00428_predict.py`, `e00428_score.py`, `e00428_windows.py`
- `00435`: `e00435_table_p_provisional.py`, `e00435_tailcore.py`
- `00437`: `e00437_features.py`
- `00448`: `e00448_refs.py`, `e00448_run.py`
- `00454`: `e00454_build.py`, `e00454_score.py`
- `00470`: `e00470_build.py`, `e00470_heads.py`, `e00470_score.py`
- `00510`: `e00510_common.py`, `e00510_fit.py`
- `00511`: `e00511_common.py`, `e00511_fit.py`
- `00513`: `e00513_data.py`, `e00513_shape.py`
- `00515`: `e00515_features.py`, `e00515_fit.py`, `e00515_placebo.py`
- `00521`: `e00521_refs.py`, `e00521_stats.py`
- `00545`: `e00545_model.py`
- `00558`: `e00558_lags.py`
- `00559`: `e00559_extra.py`, `e00559_model.py`, `e00559_model_drift.py`, `e00559_wt.py`
- `00560`: `e00560_data.py`, `e00560_model.py`
- `00562`: `e00562_columns.py`
- `00565`: `e00565_design6.py`, `e00565_model_variants.py`
- `00568`: `e00568_spread.py`
- `00569`: `e00569_columns.py`
- `00572`: `e00572_lags.py`, `e00572_model.py`
- `00574`: `e00574_fit.py`, `e00574_model.py`
- `00578`: `e00578_common.py`
- `00581`: `e00581_bank.py`
- `00582`: `e00582_arms.py`, `e00582_common.py`, `e00582_score.py`
- `00583`: `e00583_common.py`, `e00583_score.py`
- `00591`: `e00591_model.py`
- `00592`: `e00592_common.py`
- `00598`: `e00598_signals.py`
- `00613`: `e00613_nbmodel.py`, `e00613_nbsignals.py`
- `00667`: `e00667_cand.py`, `e00667_export.py`, `e00667_extend.py`, `e00667_fold.py`

**Prevailing head, files not already in the Candidate list (entry files `p667_fit.py`,
`e00681_fit.py`, `e00681_lags.py`, and `e00668_extra_lags_prevailing.py`).** (27 files in 11
folders).

- `00601`: `e00601_common.py`
- `00606`: `e00606_signals.py`
- `00620`: `e00620_extend.py`, `e00620_score.py`
- `00637`: `e00637_detector.py`
- `00645`: `e00645_score.py`
- `00647`: `e00647_common.py`, `e00647_data.py`, `e00647_fit.py`, `e00647_head.py`
- `00650`: `e00650_bank.py`, `e00650_common.py`, `e00650_fit.py`, `e00650_score.py`
- `00652`: `e00652_common.py`, `e00652_fit.py`, `e00652_head.py`, `e00652_score.py`
- `00667`: `p667_data.py`, `p667_export.py`, `p667_fit.py`, `p667_head.py`
- `00668`: `e00668_clamp_fixes.py`, `e00668_extra_lags_prevailing.py`, `e00668_score_lib.py`
- `00681`: `e00681_fit.py`, `e00681_lags.py`

**BSP and GSP fit and export, files not already in the Candidate list (entry files `e00669_fit.py`
and `e00669_export.py`).** (6 files in 3 folders).

- `00611`: `e00611_pool.py`
- `00633`: `e00633_common.py`
- `00669`: `e00669_common.py`, `e00669_export.py`, `e00669_fit.py`, `e00669_fold.py`
