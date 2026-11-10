# Final models: architectures and training regimes for the production port

**This page states, for each forecast type, which neural network the one-year backtest used and how
it was trained.** It is written for the person who ports the models into `nged-substation-forecast`.
Each section covers one forecast type and has the same five blocks: architecture, training regime,
post-processing, scores, and port notes. Most choices link to the experiment page that decided them.
The page adds no new scoring. Where two pages disagree, the page says so and names the current one.

**No neural network on this page is adopted for production.** Every score comes from one backtest
year that overlaps the development months. The backtest is described in [experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md):
one forecast origin per day at 12:00 UTC from 2025-11-01 to 2026-10-31, and 672 half-hours (14 days)
per origin. The origin months from 2026-07 lie in the window that earlier experiments kept as a
final test, so those months are descriptive and no choice rests on them. The experiment 677 page,
which builds up the Candidate demand neural network one component at a time, is not merged. Its slot
in the Primaries section is marked.

**Where to start.** A porter should read, in order, the "Production code that exists today" section,
the section of the forecast type to port, and the "Open questions" list. Each forecast-type section
ends with a "Code" subsection that names the research files that produced the model, with the pull
request that last changed each, and the production code that already does the same job. The research
folders are experiment folders under `ml-experiments/`, and their scripts import modules from
earlier experiment folders by file name. The appendix lists those imports for each forecast type.
The production repository has no neural-network code and no `torch` dependency, so every neural
network on this page is new code there. Backtest files are private and live in
`data/ml-experiments/backtest_year_2025-11_to_2026-10/`, outside git.

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
  `train`, `predict`, `save`, `load`, and `trained_time_series_ids`. `BaseForecasterConfig` carries
  `selected_features`, `ml_flow_experiment_id`, `experiment_name`, `weather_source`,
  `training_strategy`, `random_seed`, and `model_release`. `predict` stamps `power_fcst_model_name`,
  `power_fcst_model_version` (from `model_release`), `experiment_name`, `ml_flow_experiment_id`, and
  `fold_id` on every row. The interface allows per-series or pooled models.
- **Existing forecasters.** `XGBoostForecaster` (`packages/xgboost_forecaster`,
  `conf/model/xgboost.yaml`) trains one `xgb.Booster` per `time_series_id` on 24 features and a
  squared-error objective. `ManualHeuristicForecaster` and `ClimatologyForecaster` are in
  `packages/baseline_forecasters`. No forecaster branches on whether a series is a Primary, a BSP, a
  GSP, a wind farm, or a solar farm. `torch` appears in no `pyproject.toml`.
- **Output contract.** `PowerForecast` (`packages/contracts/src/contracts/power_schemas.py`, line
  414) holds one `power_fcst` per `ensemble_member`, with the primary key (`time_series_id`,
  `power_fcst_init_time`, `valid_time`, `ensemble_member`). It has no quantile columns and no column
  that separates a normal-running output from a prevailing output. `DELIVERY_QUANTILES` in
  `packages/contracts/src/contracts/common.py` lists 13 levels (0.01, 0.02, 0.05, 0.10, 0.20, 0.35,
  0.50, 0.65, 0.80, 0.90, 0.95, 0.98, and 0.99) and is used for metric labels. The research files
  hold 11 quantiles (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, and 0.99).
  `TimeSeriesMetadata.units` is `MW` or `MVA`, and `power_fcst` carries the series' own unit.
- **Weather.** `Nwp` (`packages/contracts/src/contracts/weather_schemas.py`, line 121) stores ECMWF
  ENS only, at H3 resolution 5, with members 0 to 50, written by
  `packages/delta_store/src/delta_store/nwp.py`. Production does not store ERA5, CAMS, or UKV. No
  code computes the mean and standard deviation over the 51 members. The trained model sees the
  control member, and every member is scored separately at predict time.
  `_attach_containing_nwp_cell` in
  `packages/ml_core/src/ml_core/features/tabular_feature_engineer.py` maps each series to its H3
  cell, and `packages/weather_utils/src/weather_utils/analysis_proxy.py` builds the analysis proxy
  used by weather lags.
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
  `load_forecaster_from_dir`, runs at 00:00, 06:00, 12:00, and 18:00 UTC over a 14-day horizon, and
  writes with `write_power_forecasts`. No schedule refits a model.
- **Post-processing.** The production code has no clamp, calibration, or band-widening step.
  `power_fcst` is the raw output of `predict`.
- **Release record.** `docs/live_service/model-releases.md` records a numbered release (`model-N`),
  stamped as `power_fcst_model_version`. A release needs a cross-validation run named
  `model-release-N`, a promotion, a git tag, and a log row. The log has no rows yet.

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
  columns. The lead enters as the lead in hours over 336 and its square root.
- **Thermal-lag and solar-gain inputs.** 14 columns from [experiment
  437](00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles/result.md) and
  [experiment
  569](00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1/result.md):
  trailing temperature means (3, 6, and 24 hours), temperature changes (3 and 6 hours), the day's
  minimum and maximum, the count of cold-snap days, trailing shortwave means (3, 6, and 24 hours),
  the 3-hour change in the clear-sky index, and two columns from the previous day's CAMS irradiance.
  Before the origin they use the analysis proxy, a stand-in for observed weather built from the
  first day of the previous 00Z ensemble run, which exists at forecast time.
- **Power lags.** A day lag and a week lag. Each is the latest same-half-hour reading at least 24
  hours before the neural network's own 09:00 origin, entered as a z-scored value, a missing flag,
  and an age code. The newest lag is therefore the 09:00 reading of the day before the origin day,
  27 hours before the contract's 12:00 origin, so the production contract's rule that power inputs
  be at least 6 hours old needed no change.
- **Neighbour and gap signals.** 60 columns read only by the lag encoder. For each lag: the gap
  between the observed power and a lag-free forecast table (the median of the lags-masked neural
  network of the same seed at its freshest earlier origin with a lead of at most 24 hours). The
  signals are exponentially weighted means of that gap with half-lives of 1, 3, and 7 days. Over the
  electrical neighbours of the Primary, the sum, mean, minimum, and maximum of those quantities are
  scaled by capacity ratio, and the own-plus-neighbour sum, a neighbour count, and a flag for no
  neighbour are added. The neighbours are the Primaries that NGED's "Primary Substation
  Interconnections" list links to the Primary ([experiment
  610](00610-nra-neighbour-summary-lag-encoder-inputs-iteration/result.md)). During fine-tuning, all
  signal columns of a random 50% of rows are set to no information (the "D50" dropout).
- **Series identity.** An 8-dimensional embedding of the Primary. The target is cleaned power
  divided by effective capacity, then z-scored per Primary.

**The encoders and the mixer have fixed widths.** The weather encoder (7 means, the 8-wide
embedding, and the 14 thermal-lag and solar-gain columns, 29 inputs) and the datetime encoder (21
columns and the embedding, 29 inputs) each map to 64 hidden GELU units and then 32 outputs. One
shared lag encoder runs twice, on the day lag and then the week lag. Its 143 inputs are the lag's 3
columns, the 32-wide embeddings that the weather and datetime encoders give for the weather expected
at the lag's own time, the 8-wide embedding of the Primary, an 8-wide embedding of the lag's
identity, and the 60 signal columns. It maps to 64 and then 32, and the two outputs are
concatenated. The mixer takes 145 inputs (the weather and datetime outputs of 32 each, the 64 lag
outputs, the embedding, the seven spread columns, and the two lead columns), has two hidden layers
of 64 units with 10% dropout, and outputs 9 numbers: the logits, means, and log standard deviations
of a 3-component Gaussian mixture. No merged page states the parameter count of the Candidate.

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
uncertain at three Gauss-Hermite points ([experiment
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
  daily hazards. Each hazard is `softplus(output - 3)`. The survival probability `p` is the
  exponential of minus the hazards accumulated over the days before the lead, and `p` is 0 when the
  departure is younger than half a day. The head has 11,216 parameters.
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
  2025-11 and 2025-12), the prevailing rows come from B-haz: the Candidate plus its own recent
  residual times a log-normal survival curve fitted on synthetic durations, with no fit on observed
  power. The prevailing rows equal the NRA rows on 74% of November origins and 65% of December
  origins, and on no origin from 2026-01. B-haz has no inputs that the extra lagged power could
  widen.

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

### Training regime

**Each of the 12 refits of the Candidate, one per test month, runs three steps and sees only data
from before its month.** The folds are the test months 2025-11 to 2026-10. [Experiment
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

- **Data.** Cleaned power, with the `substation_zero` half-hours dropped (the one cleaning rule the
  backtest applied), and the D15 and D18 histories cut before 2025-03-01. The ENS history starts on
  2024-04-01, so the early folds fine-tune on less ensemble data than the later folds.
- **Step 1, pre-training.** The weather, datetime, and lag encoders pre-train on ERA5 and CAMS
  (shortwave) as stand-in weather, in pseudo-runs every 2 days from 2019-09-01 at 09:00 UTC with
  leads of 0.5 to 351 hours (about 8.1 million rows for the 2026-07 fold). Five Primaries start
  later (D11 on 2019-09-17, D10 on 2019-12-01, D5 on 2020-02-01, D14 on 2020-07-01, and D16 on
  2021-02-01). ERA5 has no ensemble, so the seven spread columns hold the fold's median ensemble
  spread for each 24-hour lead band. All weights train, with AdamW, learning rate 0.0005, weight
  decay 0.001, batch 8,192, 20% of the rows per epoch, up to 60 epochs, patience 8, and
  gradient-norm clip 1.0.
- **Step 2, fresh mixer.** The pre-trained weights load except the mixer. A fresh mixer is drawn
  with seed 12345, the weather and datetime encoders are frozen, and training runs on the 730-day
  history with uniform weights and a 3-day embargo, at learning rate 0.0002 (0.0005 for the fresh
  mixer).
- **Step 3, daily fine-tune.** The fine-tune starts from the step-2 weights with the same frozen
  encoders and learning rate 0.0002. Row weights are 0.5 raised to the power (age in days divided by
  90), a 90-day half-life. The refit rule drops rows whose valid time lies in the newest 7 days
  before the cut-off and halves the weight of rows 7 to 14 days old. Early stopping holds out one
  week in every 12 weeks of origin days, with 15 days purged before and 9 days after each held week,
  and uses at most 200,000 held rows. The ensemble weather is the input at this stage and at
  inference.
- **Seeds and widening.** Two seeds are averaged quantile by quantile and sorted again. The backtest
  applies no widening. The final test of [experiment
  620](00620-bakeoff-final-test-demand-networks-at-primaries/result.md) widened the band about the
  median by a factor of 1.0934, the mean of the five development-month factors of [experiment
  613](00613-nra-roster-contrasts-with-intervals-five-seeds/result.md), and experiment 668 states
  that it averaged five seeds.
- **Missing and delayed inputs.** A flagged or absent lag reads as value 0, a missing flag, and the
  maximum age code. In training, 30% of origin days mask both lags and, of those, 15% mask only the
  day lag. Absent signal columns read as 0 with a missing flag. Neighbour-signal columns whose
  training spread is under 80% of their spread in the last development month receive matching
  Gaussian noise during training. All weather columns must be finite.
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
  target.
- **Duration mix.** The injected shifts have durations drawn half from 3 to 14 days and half from 28
  to 112 days, an assumed mixture with none from 15 to 27 days, and each injected shift counts
  equally.
- **Loss and optimiser.** The loss is the pinball loss over the 11 levels plus 0.05 times the binary
  cross-entropy of `p` against the observed state of an injected shift. The optimiser is Adam with
  learning rate 0.003, weight decay 0.0001, batch 4,096, up to 30 epochs, patience 4, and two seeds
  averaged by mean quantile. The lag inputs of an injected shift are built from the same shifted
  power as its target.
- **Hardware and run time.** About 0.5 GPU-hours per fold for the Candidate chains (experiment 667).
- **Capacity.** The effective capacity of the production table normalises the target. The neural
  network does not cap its output at that capacity.

### Post-processing

**The backtest file of the Primaries has one post hoc clamp, and the calibrated copy adds a scale
for the prevailing rows only.** Both act on stored quantiles after the neural networks run. The band
scale leaves the median unchanged.

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
  applies to the prevailing Primaries rows and not to the NRA rows.
- **Not applied.** The widening factor 1.0934 of [experiment
  613](00613-nra-roster-contrasts-with-intervals-five-seeds/result.md), and any cap at capacity.

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
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/e00667_extend.py` | Rebuilds the ensemble-summary matrices to 2026-10-31 | PR 327 (`2fcc55c49`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/e00667_export.py` | Native 09:00 UTC forecasts to the 12:00 UTC contract rows, with capacity scaling | PR 327 (`2fcc55c49`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/e00667_merge.py` | Merges NRA and prevailing rows into `primaries.parquet` and checks it | PR 327 (`2fcc55c49`) |
| `ml-experiments/00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/chain.sh` | Runs one fold's fits in order | PR 327 (`ed7bed47f`) |
| `ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/e00613_nbmodel.py` | The Candidate's network class (lag encoder with neighbour columns) | PR 271 (`b244bbe4c`) |
| `ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/e00613_nbsignals.py` | Neighbour and gap signal columns | PR 271 (`b244bbe4c`) |
| `ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/e00613_nbarm.py` | Fit and inject wrapper around the 574 pipeline | PR 271 (`8a5e60f9b`) |
| `ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/e00613_widths.py` | Width factors, source of 1.0934 | PR 271 (`8a5e60f9b`) |
| `ml-experiments/00606-nra-lag-encoder-signal-permutations-gap-averages-and-neighbours/e00606_model.py` | Shared lag encoder with a chosen set of inputs | PR 262 (`e93adb820`) |
| `ml-experiments/00591-shared-lag-encoder-auxiliary-shift-depth-loss/e00591_model.py` | Shared lag encoder and auxiliary depth head | PR 246 (`87deec536`) |
| `ml-experiments/00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/e00574_model.py` | Network with lag encoder and D5 offset add-ons | PR 230 (`fb7e829af`) |
| `ml-experiments/00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/e00574_fit.py` | Three-step schedule, monthly refits, early stopping | PR 230 (`dd9977df8`) |
| `ml-experiments/00572-two-offset-part-1-lags-versus-the-slow-output-offset/e00572_model.py` | Network with the D5 output offset | PR 227 (`986ffd0eb`) |
| `ml-experiments/00572-two-offset-part-1-lags-versus-the-slow-output-offset/e00572_lags.py` | Day and week lags with a configurable age cap | PR 227 (`986ffd0eb`) |
| `ml-experiments/00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1/e00569_columns.py` | Thermal-lag and solar-gain input columns in pre-training and fine-tuning | PR 225 (`d8d614892`) |
| `ml-experiments/00559-m0-leak-ladder-three-step-schedule-frozen-encoders-and-penalty/e00559_model.py` | Modular encoders (weather, datetime) and mixer | PR 214 (`bbe099c19`) |
| `ml-experiments/00545-modular-lag-free-demand-network-weather-and-datetime-encoders/e00545_model.py` | Original modular lag-free network | PR 206 (`1d0595a71`) |
| `ml-experiments/00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles/e00437_features.py` | Trailing-window thermal-lag and solar-gain features | PR 95 (`5a84bafb2`) |
| `ml-experiments/00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens/e00418_prep.py` | ERA5 and CAMS pre-training rows and fold assembly | PR 75 (`a7d730b76`) |
| `ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand/e00397_models.py` | Gaussian mixture head and `mixture_quantiles` | PR 50 (`7b6422960`) |
| `ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand/e00397_fit.py` | Ensemble-summary rows and base fitting stages | PR 50 (`7b6422960`) |
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

**Production code for the same job.** The production repository's documented extension point for a
new model is a `BaseForecaster` subclass with its own `FeatureEngineer`. Whether the neural networks
use it is not decided.

- **Exists.** The model interface, the cleaned-power reader, the effective-capacity table, the ENS
  weather store, and the lag machinery, as described under "Production code that exists today".
  `XGBoostForecaster` is the only trained forecaster and serves every series type.
- **Differs.** The production forecaster is a per-series XGBoost booster fed the control member, and
  it outputs one point forecast per ensemble member. The Candidate is a pooled neural network fed
  51-member summaries, ERA5 and CAMS pre-training rows, neighbour power summaries, and a 24-hour
  minimum lag age, and it outputs 11 quantiles per half-hour.
- **Missing.** Neural-network training and inference code, a `torch` dependency, ENS summary
  features, the neighbour list, a quantile output column set, an NRA and prevailing output flag, the
  survival head and its bank of injected shifts, the truncated Gaussian family, the MVA clamp, the
  walk-forward band scale, a monthly refit schedule, and the effective-capacity scaling of the
  target.

### Port notes

- **Reproduce exactly.** The input definitions and scalings above, the 24-hour minimum age of the
  Candidate's own lags, the three-step schedule with the same frozen encoders and fresh mixer seed,
  the 90-day half-life, the monthly refit on data before the first of the month with a 3-day embargo
  at the mixer step and a 7-day embargo at the fine-tune, the 11-quantile read-out by bisection, and
  the construction of the prevailing mixture from the Candidate's quantiles.
- **Unvalidated.** The truncated Gaussian output has been tested at only four Primaries and two
  months, and only for the NRA output. The prevailing head's behaviour on real load shifts rests on
  one verified transfer. The survival probability beyond day 10 is the bank generator's duration
  prior and not an estimate from observed events. The backtest used two seeds and no widening, where
  the final test used five seeds and a factor of 1.0934.
- **Open questions.** Whether to port the 8-input or the 14-input extra-lag set (the backtest uses
  14). Whether the fallback head is needed in production, because the backtest needed it only for
  the first two folds. What experiment 677 concludes about the necessary components (pending).

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

- **Inputs.** The same 37 lag-free columns, 8 lag columns, knot position, expected-weather table for
  the lag encoder (the weather and calendar columns of the freshest run issued at or before the
  lag's own time, passed through the same encoders), and 14 thermal-lag and solar-gain columns as
  the Candidate, with one embedding of the series (2 entries per group) and no signal columns. The
  51-member means and standard deviations are taken at the series' H3 resolution-5 cell.
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

### Training regime

- **Schedule.** The same three steps as the Candidate: ERA5 and CAMS pre-training, a step 2 with a
  fresh mixer of seed 12345 and frozen encoders, and a fine-tune with a 90-day half-life, with the
  same optimiser settings and a 3-day embargo. The encoders pre-train on the matrices of [experiment
  564](00564-pooled-primary-bsp-gsp-m2-reduced/result.md), which end on 2026-07-31, so the folds
  from 2026-08 share the same pre-training rows.
- **Fine-tuning rows.** The 730 days before the cut-off, with targets ending 3.5 hours before it.
  For the BSP group, 570,000 rows in the 2025-11 fold rising to 735,000 in 2026-10, and 2% to 3%
  fewer for the GSP group. The ENS history starts on 2024-04-01, so the first fold has 19 months of
  it.
- **Lag dropout.** 30% of days mask both lags, and 15% of those mask only the day lag.
- **Refits and seeds.** Twelve monthly refits, one per test month, and two seeds averaged quantile
  by quantile and sorted. The final-test neural network of [experiment
  633](00633-bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network/result.md) was one fit at
  2026-07-01. The early-stopping weeks are purged by 9 days, shorter than the 14-day leads, so a
  held-out week's targets can sit in training rows of later origins. [Experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md) records
  the shorter purge as a caveat, ran a perturbation test that found no leak, and states that it
  changed no result.
- **Targets and capacity.** Power divided by effective capacity (D13 55.3 MW, D17 47.6 MW, D6 384.3
  MW, and D1 170.9 MW), z-scored per series. All four are signed MW series.
- **Weather at lag times.** The thermal-lag and solar-gain columns use the analysis proxy built from
  the first day of the previous 00Z run, which exists at forecast time. The last roughly 5 days of
  the ERA5 and CAMS pre-training rows could not be read at a real cut-off, because ERA5 arrives
  about 5 days late.
- **Hardware and run time.** About 2 GPU-hours on one process and 1 core (experiment 669).

### Post-processing

- **Walk-forward band scale (calibrated copy only).** See "Walk-forward band calibration" below. It
  applies to all BSP and GSP rows from 2025-12.
- **No clamp and no cap.** The files keep the quantiles above capacity: the highest quantile is 2.2
  times the table value at D1 and 1.4 times at D6. A negative quantile is net export, and the q01 is
  below zero in 94% of the D6 rows and 99.6% of the D1 rows.

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
  The misses change side by month, so a level drift and not only a narrow band is at work.
- **Against the 24-hour-lag neural network.** On July to October 2026 origins at hours 0 to 72, the
  3.5-hour lags score 7.24 against 7.54, but only 0.16 of the 0.30 gain is attributable to the lag
  cut, and the rest comes from the monthly refits (experiment 669).
- **The largest miss.** The observed demand at D6 Walpole stepped up on 2026-10-19 while the median
  fell, a level step consistent with a load transfer that is not confirmed in NGED's logs and that
  this neural network does not model.

### Code

**Last change** is the merged nged-ml-research pull request that last touched the file on
`origin/main`, with its commit. Several experiment folders import modules from earlier folders, so a
fit needs the import closure listed in the appendix. Paths are relative to the repository root.

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
  any lag that is not older than the lead time and has no 6-hour power-delay constant.
- **Missing.** The same pieces as for the Primaries, apart from the neighbour list and the
  prevailing head, and the group-pooled fitting.

### Port notes

- **Reproduce exactly.** The 3.5-hour lag cap, the one-neural-network-per-group pooling, the D5
  output offset of [experiment 565](00565-drift-embedding-variants-six-designs/result.md), the
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
  average 18 MW below the observed median, and its monthly mean error swings between -39.8 MW and
  +25 MW. The Float64 originals of the BSP and GSP files are not recast to the Float32 of the
  calibrated copies.

## Wind

### Architecture

**The three wind farms are forecast by the simple shared neural network S of [experiment
659](00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md), a
two-layer network with a Gaussian mixture output.** The backtest is [experiment
666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md). The only inputs are
ensemble weather and the lead.

- **Inputs.** The mean and standard deviation over the 51 members of 10 m and 100 m wind speed, the
  sine and cosine of the 100 m wind direction, and 2 m air temperature, plus the lead (the sine and
  cosine of the lead in hours modulo 24, and the lead in hundreds of hours) and a one-hot wind-farm
  column. One wind farm is a MW series and two are MVA series. There is no power lag, no calendar
  column, and no physics-lite input. Inputs are standardised with the fit rows' mean and standard
  deviation.
- **Layers and output.** Two hidden layers of 64 units feeding a 3-component Gaussian mixture over
  power as a fraction of capacity, centred on the training mean of the target. The 11 quantiles are
  read from the mixture and clamped to the range 0 to 1 of the neural network's own capacity scale.
- **Weather.** The 00Z ENS run of the origin day, treated as available at 09:00 UTC, with leads of
  12 to 347.5 hours from the 00Z run. Where ensemble steps are 3 or 6 hours, the half-hourly value
  is interpolated as in [experiment 310](00310-metered-farm-long-lead-scoring/result.md) and
  [experiment 430](00430-unified-forecaster-solar-and-pretraining-ablations/result.md). No ERA5 or
  CAMS pre-training.

### Training regime

- **Refits.** Wind refits on day 0 of each month (12 refits), on ensemble runs and valid times
  before the 62 days that precede the refit, and uses those last 62 days for early stopping
  ([experiment 666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md)).
  [Experiment
  659](00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md) refitted
  twice a month.
- **Optimiser.** AdamW, learning rate 0.002, batch 2,048, weight decay 0.0001, and 40 epochs,
  keeping the epoch with the lowest negative log-likelihood on the last 62 days. Training leads run
  to 351 hours from the 09:00 stamp.
- **Seeds.** Three seeds, averaged quantile by quantile and then sorted.
- **Idle readings.** Readings flagged by the idle mask of [experiment
  210](00210-wind-idle-noise-mva-artefact/result.md) are set to zero in training and scoring until
  the mask ends. The mask is a research rule and not part of the production cleaning code, and from
  its end date idle readings count as observed, in scoring and in the training target of the last
  refit. That makes the last weeks of the backtest year a different scoring target.
- **Capacity.** The 99.9th percentile of the farm's cleaned power over the 730 days before each
  refit, which is 1.02 to 1.12 times the production effective capacity. The neural network trains
  with this scale and the scores divide by the production effective capacity.
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
fit needs the import closure listed in the appendix. Paths are relative to the repository root.

| File | Role | Last change |
|---|---|---|
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_network.py` | Mixture head and training loop shared by wind and solar | PR 328 (`5acc5f4dc`) |
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_wind_members.py` | Ensemble member rows at the wind farms | PR 336 (`eb01acf75`) |
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_wind_features.py` | Per-case wind inputs | PR 336 (`eb01acf75`) |
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_wind_fit.py` | Monthly wind network S fits and forecasts | PR 336 (`eb01acf75`) |
| `ml-experiments/00666-one-year-backtest-of-simple-wind-and-solar-networks/e00666_export.py` | Writes `wind.parquet` and `solar.parquet` | PR 336 (`eb01acf75`) |
| `ml-experiments/00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/e00659_wind_fit.py` | Original network S fit | PR 319 (`1e3e197f3`) |
| `ml-experiments/00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/e00659_wind_features.py` | Original wind inputs | PR 319 (`6dc9c4f02`) |

**Production code for the same job.** The production repository's documented extension point for a
new model is a `BaseForecaster` subclass with its own `FeatureEngineer`. Whether the neural networks
use it is not decided.

- **Exists.** NWP wind components at 10 m and 100 m, derived wind speed and direction
  (`_derive_wind_features` in `tabular_feature_engineer.py`,
  `packages/weather_utils/src/weather_utils/wind.py`), and `time_series_type` with a wind value.
- **Differs.** Production has no wind-specific model. The research network S reads the mean and
  standard deviation over the 51 members and a one-hot farm column, and outputs a mixture's 11
  quantiles. Production trains on the control member.
- **Missing.** Network S, the mixture head, ENS summary features, a capacity scale of the 99.9th
  percentile of the last 730 days, the idle mask (a research rule, not production cleaning), monthly
  refits, and quantile output.

### Port notes

- **Reproduce exactly.** The 64-by-64 layers, the 3-component mixture centred on the training mean,
  the input list, the 730-day 99.9th-percentile capacity, and the monthly refit with a 62-day
  early-stopping window.
- **Unvalidated.** Wind has no power lag, so the neural network cannot react to a recent change of
  level. The idle mask is not production code.
- **Open question.** Whether to drop or damp the forecast beyond day 10, where the neural network
  scores worse than climatology.

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
  lead. There is no farm column and no calendar column.
- **Physics-lite power.** The physics-lite power is `clip((G / 1000) * (1 - 0.004 * (T_cell - 25)),
  0, 1)`, where G is shortwave radiation in W per square metre and `T_cell` is the Faiman cell
  temperature `T_air + G / (25 + 6.84 * wind speed)`. The formula runs on each of the 51 members
  separately, and the neural network reads the mean and standard deviation. The mixture is centred
  on the mean physics-lite power. It has no fitted parameters.
- **Layers and output.** As for wind: two hidden layers of 64 units and a 3-component Gaussian
  mixture over power as a fraction of capacity, with the quantiles clamped to the range 0 to 1 of
  the neural network's capacity scale.
- **Night gate inside the neural network.** The neural network forecasts only the half-hours where
  the mean cosine of the solar zenith angle over the ensemble step is above zero, and sets the
  quantiles of the other half-hours to 0.

### Training regime

- **Refits.** The solar neural network is refitted quarterly, on the first day of the first month of
  the backtest year and of each third month after it. Each fit scores the next three months, uses
  three seeds, and has no pre-training. Training uses valid times before one day earlier, with the
  last 61 days for early stopping, 20 epochs, daylight rows only, and leads below 348 hours from the
  00Z run, as [experiment 666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md)
  states ([experiment 430](00430-unified-forecaster-solar-and-pretraining-ablations/result.md) and
  [experiment
  659](00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md) stopped
  at 251.5 hours, so the lead range is extended). Solar trains on all daylight half-hours, including
  more low-sun half-hours than experiment 659 kept.
- **Optimiser and seeds.** As for wind (AdamW, learning rate 0.002, batch 2,048, weight decay
  0.0001), keeping the epoch with the lowest negative log-likelihood on the last 61 days. The target
  is the raw power as a fraction of capacity, because solar has no idle mask.
- **Capacity.** As for wind: the 99.9th percentile of cleaned power over the 730 days before each
  refit.
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
- **Differs.** Production has no solar-specific model. The research network PLP adds a physics-lite
  power and a Faiman cell temperature computed per ensemble member, and outputs 11 quantiles with
  zeros at night.
- **Missing.** Network PLP, the physics-lite power and cell-temperature code, the sun-position night
  gate and clamp, ENS summary features, the 730-day capacity scale, quarterly refits, and quantile
  output.

### Port notes

- **Reproduce exactly.** The physics-lite formula with the Faiman cell temperature, the 18 inputs,
  the mixture centred on the mean physics-lite power, the daylight-only training rows, and the post
  hoc night rule at each half-hour's own zenith.
- **Unvalidated.** The night gate was tested at two farms and one refit. Dawn and dusk coverage is
  poor. Arm b of [experiment
  684](00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/result.md),
  which would feed the physics-lite power at each half-hour's own zenith to the neural network, was
  not run.
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
  origin. The factor is bounded to 0.5 to 3. There is no level shift, so `q50` is unchanged. After
  scaling, the quantiles of MVA series, wind, and solar are clipped at zero.
- **Where it applies.** The prevailing Primaries rows, the BSPs, the GSPs, and solar. It does not
  apply to wind, whose coverage already exceeds 0.90, and it does not apply to the NRA Primaries
  rows, which the scale brought to only 87.1% coverage, with a CRPS rise of 0.081 points (experiment
  683). The first month, 2025-11, has no earlier forecast to fit on and is left as stored.
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
  justified it did not fire. The stop rule that would have justified a learned width head did not
  fire.
- **Calibrated copies on disk.** `primaries_calibrated.parquet` (prevailing rows),
  `bsps_calibrated.parquet`, `gsps_calibrated.parquet`, and `solar_calibrated.parquet`. The NRA rows
  of the first are copied unchanged. The prevailing rows of `primaries_calibrated.parquet` were
  built from the version of `primaries.parquet` from before the extra lagged inputs, so they do not
  yet carry them.
- **Open questions.** Whether to refit the scale at every refit of a forecasting neural network (not
  decided, because the diagnostic was not run). The band width steps at some month-start refits, and
  the scale then overshoots for more than one month: the days 10 to 14 width of six Primaries (D11,
  D22, D5, D16, D21, and D2, identifiers 8, 15, 17, 18, 24, and 25) widens by 1.21 to 1.48 times at
  the February refit, and their calibrated coverage goes from 76.3% in January to 93.1% in February.
  D18 widens by 1.48 times at the September refit, and D4 (identifier 11) narrows by 0.76 times
  (experiment 683 lists this as April and [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  as the refit that starts May 2026).

- **Research code.**
  `ml-experiments/00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/e00683_calibrate.py`
  holds `fit_month` (the scale), `apply_arm`, and the scoring, and `e00683_refit.py` measures band
  width across refits. The clamps are in
  `ml-experiments/00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/e00668_clamp_fixes.py`.
  The script that wrote the four calibrated copies, including the hold of D18 and D14 at their
  2026-06 factor, is not on `origin/main`.
- **Production code for the same job.** None. The production repository has no clamp, calibration,
  or band-widening step. A stored factor table applied to predicted quantiles would be new code, and
  where it would live is not decided.

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
  lead band refreshed each month.
- **Serving path.** The production design principle is to keep the serving path close to loading a
  fitted forecasting model and calling `predict`. The post hoc clamps and the band scale are the
  only steps here that act on predicted quantiles.
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
  (identifier 11) at an April month start, with a factor of 0.76. Experiment 668 places it at the
  refit that starts May 2026 and calls it about a third, and the README says about 37%. For D18,
  experiment 683 gives a widening of 1.48 times, experiment 668 about half, and the README 55%.
- **Calibrated-copy scores.** The README gives CRPS changes of +0.044 (prevailing Primaries), +0.001
  (BSPs), +0.026 (GSPs), and +0.006 (solar), against +0.047, +0.005, +0.003, and -0.002 in
  experiment 683. The README's prevailing figures come from the file from before the extra lagged
  inputs. The current page for the experiment's own numbers is experiment 683.
- **"As stored" NRA coverage.** [Experiment
  688](00688-winter-normal-running-coverage-median-error-and-events-part-1/result.md) calls 80.1%
  and 80.3% the bands as stored. They are the scaled bands of experiment 683, whose stored values
  are 81.7% and 77.7%.
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
  683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md)).
- Why the NRA bands of D18 and D14 (identifiers 13 and 16) cover only about 22% to 28% of
  observations in July and August 2026. [Experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  suggests a load shift that normal running ignores. [Experiment
  685](00685-event-table-on-the-final-test-window-all-primaries/result.md), an event table on the
  final-test window, finds no transfer between the two Primaries in those months, and names a level
  offset in the lag-free forecast as an untested candidate. The current page is experiment 685.
- What the Candidate needs and what each component adds: pending experiment 677.
- Whether wind needs a power lag or a different long-lead rule, given its score against climatology
  at days 10 to 14.
- Whether a calendar input or more frequent refits remove the solar median's seasonal bias.
- How to model load shifts at the BSPs and GSPs, where the largest observed miss looks like one.
- How the Float64 originals of the BSP and GSP files should be recast to match the Float32
  calibrated copies.

## Appendix: modules each fit imports

**Each research fit imports modules from earlier experiment folders, and the lists below give the
full set by folder.** The lists come from following every `import e00NNN_...` and `import p667_...`
statement from the entry files, so they name files that a porter must read or copy. The Candidate
fit and the prevailing head share most of their modules. Wind and solar copy their code into the 666
folder and need only that folder.

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
