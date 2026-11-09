---
id: "00668"
slug: documentation-of-the-one-year-backtest-files-models-training-and-performance
title: >-
  What do the one-year backtest files hold, how were the five models behind them built and trained,
  and how well do they forecast?
theme: production-reference-and-best-guess
also_relevant_to:
  - generator-forecasting
  - lagged-power-and-error-correction
status: planned
kind: other
size: >-
  Medium. (1) It stores no forecasts: the page reads the five backtest files and the observed power
  and writes figures and tables; (2) the page has one defensible structure, set in the plan below;
  (3) it spans the code of experiments 611, 620, 633, 652, 654, 659, 666, 667 and the BSP and GSP
  backtest, so every model section is checked against the code by a fact-check reviewer; (4) it
  touches no serving path or degradation rule; (5) it changes no contract.
hypothesis: >-
  Not a hypothesis test. The page documents the five one-year backtest files so that an engineer can
  re-implement each model and a colleague at NGED can read the files.
prediction: >-
  None. The page reports descriptive scores only.
decision_rule: >-
  None. The page is finished when the fact check, the numbers review and the prose review have no
  open finding.
benchmark: >-
  Climatology beside every model, and the manual heuristic or the earlier research default where
  one exists.
row_set_id: backtest_year_2025_11_to_2026_10
metric: >-
  CRPS in points of capacity, normalised mean absolute error of the median, q05 to q95 coverage, by
  lead band and by month.
---

This page documents the five Parquet files of the one-year backtest in
`data/ml-experiments/backtest_year_2025-11_to_2026-10/`. It has two readers: the colleague at NGED
who receives the files, and an engineer who must re-implement one of the four models behind them.
A Primary is a primary substation, a BSP is a bulk supply point, and a GSP is a grid supply point.
This page is a draft: the files, model sections, and plan are written, and the performance
sections (tables and figures) are added when each Parquet file exists.

**Take-home message.** To be written once the scores exist.

## Figures

Figures are added in phase 2, starting with the headline figure: score against lead time for each
model beside climatology.

## The files

The files are private. They hold forecasts and no observed power. The year overlaps the
development months and the final-test months of earlier experiments, so the scores are a
description of one year and not a ranking.

- **Period.** One forecast origin per day at 12:00 UTC, from 2025-11-01 to 2026-10-31 (365
  origins, no gaps).
- **Horizon.** Valid times run half-hourly from 12:00 UTC on the origin day to 11:30 UTC 14 days
  later: 672 half-hours per origin. The valid times of the last origin reach 2026-11-14.
- **Weather.** The 00Z run of the ECMWF ensemble (51 members) of the origin day, treated as
  available at 09:00 UTC. The ensemble's lead from 00Z is 12 to 348 hours, inside the 360 hours
  that the run provides. Where the ensemble steps are 3 hours or 6 hours, the half-hourly values
  come from the production resampler ([experiment
  661](../00661-pv-ens-time-resolution-shortwave-interpolation-audit-and-bias-adapter/result.md)).
- **Power delay.** At 12:00 UTC the observed power is available only up to 06:00 UTC of the same
  day. No input of any model reads newer power. Wind and solar read no power lag, so the delay
  moves only the training cut-off of each refit.
- **Files and series.**

| File | Series | Units | Rows (series x 365 origins x 672 half-hours) |
|---|---|---|---|
| `primaries.parquet` | 16 Primaries | MW; MVA for the eight MVA-only Primaries | 7,848,960 (two rows per key) |
| `bsps.parquet` | 2 BSPs (D13 Skegness, D17 Boston) | MW | 490,560 |
| `gsps.parquet` | 2 GSPs (D6 Walpole, D1 Bicker Fen) | MW | 490,560 |
| `wind.parquet` | 3 wind farms | MW | 735,840 |
| `solar.parquet` | 6 solar farms | MW | 1,471,680 |

- **Columns, in this order.**
    - `time_series_id`: the series' original identifier from the metadata table, as an 8-bit
      integer.
    - `valid_time`: the half-hour the forecast is for, a UTC timestamp.
    - `nwp_init_time`: the 00Z ensemble run used, a UTC timestamp.
    - `power_fcst_init_time`: the forecast origin, 12:00 UTC.
    - `q01`, `q05`, `q10`, `q25`, `q40`, `q50`, `q60`, `q75`, `q90`, `q95`, `q99`: the forecast
      quantiles, sorted within the row, in the series' own units. `q50` is the median forecast.
      Each quantile is the value that the observed power falls below with that probability, so
      `q05` to `q95` is a 90% band.
    - `NRA_or_prevailing` (`primaries.parquet` only): `"NRA"` or `"prevailing"`.
- **What `NRA_or_prevailing` means.** The normal running arrangement (NRA) rows forecast the demand
  the Primary would draw if the electricity network operator had made no load shift. The
  prevailing rows forecast the demand under the arrangement in force at the forecast origin,
  carrying a shift that is already running forward and widening the band with the probability
  that the shift ends. Each key (series, valid time, origin) has one row of each kind.
- **Checks.** Each file is checked for the row count above, no duplicate key, no null or
  non-finite quantile, no crossing quantiles, and UTC time-zone. Each worker's page states the
  results and a spot check that no input is newer than the limits above.

## Primaries

Primaries are forecast by the Candidate demand network ([experiment
620](../00620-bakeoff-final-test-demand-networks-at-primaries/result.md)), which gives the NRA
rows, with the survival-hazard prevailing head R-haz ([experiment
652](../00652-two-output-forecast-with-explicit-rules-and-survival-hazard/result.md)), which gives
the prevailing rows. The backtest itself is [experiment
667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md). Its page
fixes the choices that the earlier experiments left open, and this section restates both.

### Method

- **Inputs of the Candidate.** One row per (Primary, ensemble run, valid half-hour); the network
  runs once per ensemble run, on summaries of the 51 members.
    - Weather: the mean over the 51 members of 7 variables (shortwave and longwave radiation,
      2 m temperature, 2 m dew point, surface pressure, precipitation after `log1p` of the value
      clipped at zero, and 10 m wind speed), and the standard deviation (`ddof=0`) of the same 7.
      The means and `log1p` precipitation are z-scored with the training rows' mean and standard
      deviation and clipped at plus or minus 5.
    - Calendar and sun: 21 columns: solar zenith cosine, azimuth sine and cosine, the sine and
      cosine of the local hour at periods of 24 h and 12 h (Europe/London), a one-hot day of week,
      a COVID flag, three Christmas columns, and three bank-holiday columns.
    - Lead: the lead in hours divided by 336, and its square root.
    - Power lags: for each of a day lag and a week lag, the latest same-half-hour reading at least
      a minimum age before the origin, as a z-scored value, a missing flag, and an age code. The
      stored Candidate used a minimum age of 24 hours. The backtest lowers the minimum age to
      6 hours (a cap of 12 half-hours), pre-trains the lag encoder with that cap, and applies the
      same cut to the neighbour summaries.
    - Thermal-lag and solar-gain columns (14): trailing temperature means (3, 6, and 24 hours),
      temperature changes (3 and 6 hours), the day's minimum and maximum, the count of cold-snap
      days, trailing shortwave means (3, 6, and 24 hours), the 3-hour change in the clear-sky
      index, and two columns from the previous day's CAMS irradiance.
    - Neighbour and gap signals (60 columns, read only by the lag encoder): for each lag, the gap
      between the observed power and a lag-free forecast table (the median of the lags-masked
      network at its freshest earlier origin), exponentially weighted means of that gap with
      half-lives of 1, 3, and 7 days, and, over the electrical neighbours of the Primary,
      the sum, mean, minimum, and maximum of those quantities scaled by capacity ratio, the
      own-plus-neighbour sum, a neighbour count, and a flag for no neighbour. During fine-tuning,
      all signal columns of a random 50% of rows are set to no information (the "D50" dropout).
    - Series identity: an 8-dimensional embedding of the Primary.
    - Target: cleaned power divided by effective capacity, then z-scored per Primary.
- **Architecture of the Candidate.** The weather encoder (7 means, an 8-wide embedding, and 14
  thermal-lag and solar-gain columns, so 29 inputs) and the datetime encoder (21 columns and the
  embedding, 29 inputs) each map to 64 hidden GELU units and then 32 outputs. One shared lag
  encoder runs twice, on the day lag and then the week lag. Its 143 inputs are the lag's 3 columns,
  the 32-wide embeddings that the weather and datetime encoders give for the weather expected at
  the lag's own time, the 8-wide embedding of the Primary, an 8-wide embedding of the lag's
  identity, and the 60 signal columns; it maps to 64 and then 32, and the two outputs are
  concatenated. The mixer takes 145 inputs (the weather and datetime outputs of 32 each, the 64
  lag outputs, the embedding, the 7 spread columns, and the 2 lead columns), has two hidden layers
  of 64 units, and outputs 9 numbers: the logits, means, and log standard deviations of a
  3-component Gaussian mixture. Each hidden layer has 10% dropout.
- **Offset.** A per-Primary piecewise-linear function of time with one knot for each month start
  from 2019-09 (a slow drift term, "D5") is added to the three component means. Fine-tuning trains
  only the newest knot, and a penalty of 10 times the mean squared difference between neighbouring
  knots keeps the function smooth.
- **Quantiles.** The 11 quantiles come from the mixture's cumulative distribution by 40 bisection
  steps on a range of plus or minus 40 standard deviations. The values are then rescaled to power,
  sorted within the row, averaged over 5 seeds quantile by quantile, sorted again, and widened about
  the median by a factor of 1.0934. Experiment 613 fixed that factor before the final test.
- **Loss.** The negative log-likelihood of the mixture on the standardised target, weighted, plus
  the knot penalty.
- **Training, in three steps.**
    - Pre-training uses ERA5 and CAMS (shortwave) as stand-in weather, in pseudo-runs every 2 days
      from 2019-09-01 at 09:00 UTC with leads of 0.5 to 351 hours (about 8.1 million rows for the
      2026-07 fold). All weights train: AdamW, learning rate 5e-4, weight decay 1e-3, batch 8,192,
      20% of the rows per epoch, up to 60 epochs, patience 8, gradient-norm clip 1.0.
    - Step 2 loads the pre-trained weights except the mixer, draws a fresh mixer with seed 12345,
      freezes the weather and datetime encoders, and trains on the whole 730-day history with
      uniform weights and a 3-day embargo. Learning rate 2e-4, and 5e-4 for the fresh mixer.
    - The fine-tune starts from the step-2 weights with the same frozen encoders, learning rate
      2e-4 throughout, and weights of 0.5 to the power of age in days over 90 (a 90-day
      half-life). The refit rule drops rows with valid times in the newest 7 days before the
      cut-off and halves the weight of rows 7 to 14 days old.
    - Early stopping holds out one week in every 12 weeks of origin days, with 15 days purged
      before and 9 days after each held week, and uses at most 200,000 held rows.
- **Missing and delayed inputs.** A flagged or absent lag reads as value 0, a missing flag, and the
  maximum age code. In training, 30% of origin days mask both lags and, of those, 15% mask only
  the day lag. Absent signal columns read as 0 with a missing flag. Neighbour-signal columns whose
  training spread is under 80% of their spread in the last development month receive matching
  Gaussian noise during training. All weather columns must be finite.
- **Injected load shifts in training.** Each Primary receives random synthetic shifts, added to the
  lag inputs only, until they cover 5% of its row span: depth 15%, 30%, or 50% of capacity, length
  3, 7, 15, or 30 days, random sign. Each shift on a Primary is subtracted, scaled by the capacity
  ratio, from one electrical neighbour, so the neighbour summaries see the transfer.
- **R-haz, the prevailing head.** A small network on top of the frozen Candidate: it reads 58 origin
  features (the gap and neighbour signals above, built from lags), 4 age columns (the log of the
  departure's age in days, the age over 30, the gate flag, and the age of the freshest lag), and
  3 row inputs (the lead over 240 hours and the lower and upper half-widths of the Candidate's
  band). It has two hidden layers of 64 ReLU units and 12 outputs: a signed depth (0.5 times the
  hyperbolic tangent of the first output, so up to half of capacity), a spread, and daily hazards.
  Each hazard is `softplus(output - 3)`, and the survival probability `p` is the exponential of
  minus the hazards accumulated over the days before the lead. The probability is 0 when the
  departure is younger than half a day.
- **How the prevailing quantiles are formed.** The Candidate's 11 quantiles define a piecewise-linear
  cumulative distribution with two tail knots 1.5 times the outer spacing beyond the 1% and 99%
  quantiles. The prevailing distribution is the mixture `(1 - p)` times the Candidate's
  distribution plus `p` times the Candidate's distribution shifted by the depth, with the depth
  uncertain at three Gauss-Hermite points. The 11 prevailing quantiles are read off that mixture
  and are monotone by construction. When `p` is 0, the prevailing rows equal the NRA rows.
- **Loss and training of R-haz.** The loss is the pinball loss over the 11 levels plus 0.05 times
  the binary cross-entropy of `p` against the observed state of an injected shift. The training rows
  mix real rows with injected shifts, with durations drawn half from 3 to 14 days and half from 28 to
  112 days (an assumed mixture) and each injected shift counted equally. Adam, learning rate 0.003,
  weight decay 1e-4, batch 4096, up to 30 epochs, patience 4, 2 seeds averaged by mean quantile.
  The head has about 9,000 parameters and no dropout. The backtest extends the hazards from 10 to 14
  days and removes the gate flag from the inputs (experiment 667's plan).
- **Refits and embargo in the backtest.** Twelve monthly refits (test months 2025-11 to 2026-10),
  each trained on data before the first of the month with the 7-day embargo of the refit rule.
  The ensemble history starts 2024-04-01, so the early folds fine-tune on less ensemble history
  than the later folds. The D15 and D18 rows before 2025-03-01 are cut, as in every recent
  experiment.
- **Where each piece came from.** The modular network and its widths: experiment 545. Frozen
  encoders and the three-step schedule: 559. ERA5 and CAMS pre-training: 418. The offset: 565 and
  572. The thermal-lag columns: 437 and 569. The embargoed schedule: 574. The shared lag encoder:
  586 and 591. The signal columns: 598 and 606. The signal dropout: 610. The refit rule: 599. The
  roster and the width factor: 612 and 613. The prevailing head: 647, 650, and 652.

### Performance

To be added when `primaries.parquet` is written.

## BSPs and GSPs

The two BSPs and two GSPs are forecast by the lag-enabled network of [experiment
633](../00633-bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network/result.md), which is
experiment 622's arm `lagged_b1`. The backtest is [experiment
669](../00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md). The
network has no neighbour inputs and no model of load-shift events. Its readings are net of embedded
generation, which no input models.

### Method

- **Inputs.** The same 37 lag-free columns, 8 lag columns, knot position, and 14 thermal-lag and
  solar-gain columns as the Candidate network above, at the 4 series, with one embedding of
  the series (2 entries per group), and no signal columns and no neighbour inputs. The 51-member
  means and standard deviations are taken at the series' H3 resolution-5 cell.
- **Lags.** The stored network read lags at least 24 hours old. The backtest retrains it with the
  3.5-hour cap of experiment 572 (7 half-hours before the 09:00 origin), so the newest power is the
  05:30 reading, 6.5 hours before the 12:00 origin.
- **Architecture.** The same encoders and mixer as the Candidate with a lag encoder of 83 inputs
  (3 lag columns, the two 32-wide expected-weather and expected-datetime embeddings, and the
  embeddings of the series and the lag), and a mixer with 145 inputs. One network is fitted per group
  (the two BSPs together, the two GSPs together), so the weights are shared across the 2 series of a
  group except for the series embedding and the offset.
- **Offset, loss, and quantiles.** The offset, loss, quantile bisection, and sorting are those of
  the Candidate, with no width factor.
- **Targets.** Power divided by effective capacity (D13 55.3 MW, D17 47.6 MW, D6 384.3 MW, D1
  170.9 MW), z-scored per series. The series are signed MW series.
- **Training.** The same three steps as the Candidate (ERA5 and CAMS pre-training, a step 2 with a
  fresh mixer of seed 12345 and frozen encoders, and a fine-tune with a 90-day half-life), the same
  optimiser settings, and a 3-day embargo. Fine-tuning rows span 730 days before the cut-off, with
  targets ending 3.5 hours before it. Lag block dropout is 30% of days (both lags), of which 15% mask
  the day lag only. Two seeds are averaged quantile by quantile and sorted.
- **Origin.** The network runs at its own origin of 09:00 UTC, the 00Z run plus 9 hours, and its
  output is relabelled with 12:00 UTC. Every input that the network reads at 09:00 was available at
  12:00.
- **Refits.** Twelve monthly refits, one per test month 2025-11 to 2026-10, with the 3-day embargo
  of the schedule. The final-test network of experiment 633 was one fit at 2026-07-01.
- **Where each piece came from.** As for the Candidate; the node harness and pooling: 564 and 611;
  the lag network at the nodes: 619; the matched rematch: 622; the final test: 633.

### Performance

To be added when `bsps.parquet` and `gsps.parquet` are written.

## Wind

Wind farms are forecast by the simple network S of [experiment
659](../00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md). The
backtest is [experiment
666](../00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md).

### Method

- **Inputs.** The mean and standard deviation over the 51 members of 10 m and 100 m wind speed,
  the sine and cosine of the 100 m wind direction, and 2 m air temperature, plus the lead (the
  sine and cosine of its hour of day and the lead in hundreds of hours) and a one-hot wind-farm
  column. There is no power lag, no calendar column, and no physics-lite input in network S.
- **Architecture.** Two hidden layers of 64 units feeding a three-component Gaussian mixture over
  power as a fraction of capacity, centred on the training mean of the target. The 11 quantiles are
  read from the mixture and capped at capacity.
- **Training.** AdamW, learning rate 0.002, batch 2048, weight decay 0.0001, 40 epochs, stopping
  early on the last 62 days. Idle readings (experiment 210's idle mask) are set to zero in training
  and scoring. Three seeds are averaged quantile by quantile.
- **Capacity.** The 99.9th percentile of the farm's power over the 730 days before each refit.
- **Refits.** Wind refits on day 0 of each month (12 refits), training on runs before 62 days
  earlier. No ERA5 or CAMS pre-training.
- **Missing inputs.** The network reads only the ensemble weather. A farm with no idle mask after
  2026-09-13 is scored with its idle periods unmarked.

### Performance

To be added when `wind.parquet` is written.

## Solar

Solar farms are forecast by the simple network PLP of experiment 659. The backtest is
experiment 666.

### Method

- **Inputs.** The mean and standard deviation over the 51 members of shortwave radiation, air
  temperature, and 10 m wind speed; the ensemble means of relative humidity, longwave radiation,
  precipitation, and surface pressure; the sun's elevation and the sine and cosine of its azimuth;
  the lead; and the mean and standard deviation of a physics-lite power. There is no farm column and
  no calendar column.
- **Physics-lite power.** Irradiance over 1000 W per square metre, times one minus 0.004 per kelvin
  times the cell temperature minus 25 degrees Celsius, clipped to the range 0 to 1, with the Faiman
  cell temperature `T_air + G / (25 + 6.84 x wind speed)`. The formula runs on each of the 51 members
  separately, and the network reads the mean and standard deviation. The mixture is centred on the
  mean physics-lite power.
- **Architecture, loss, and optimiser.** As for wind, with 20 epochs and early stopping on the last
  61 days, and a lead range extended to 336 hours (experiments 430 and 659 stopped at 251.5 hours).
- **Refits.** Quarterly: 2025-11-01, 2026-02-01, 2026-05-01, and 2026-08-01, each scoring the next
  three months, 3 seeds each, with no pre-training.
- **Capacity.** As for wind.
- **Missing inputs.** As for wind.

### Performance

To be added when `solar.parquet` is written.

## Comparison baselines and Primaries slices

This section is written last. It will hold two items.

- **Baselines beside every model.** The manual heuristic and climatology for the demand files, and
  climatology for wind and solar (the manual heuristic only where one exists for the technology).
  Both are computed from the production baseline code for the same origins and valid times, and
  are not written into the Parquet files. The heuristic uses only lags old enough for each lead
  plus the 6-hour delay, or its fallback, and the section says how the 1-week lag is handled at long
  leads.
- **Four Primaries slices.** All rows; inside long and deep load-shift events (longer than 7 days
  and 8 noise units or more, the definition of experiment 665); the 7 days after such an event
  ends; and rows with no long and deep event and at least 7 days after one ended. The section gives
  the rows and events per slice first, names the event source, says where a slice is too small,
  and gives each slice a metrics table and at least one example time-series figure.

## Caveats

- **A descriptive year, not a ranking.** The year overlaps the development months and the
  final-test window.
- **Less ensemble history early.** The ensemble history starts 2024-04-01, so early folds train
  on less ensemble data.
- **Days 10 to 14 are untested for the demand networks.** The stored Candidate and the
  lag network were trained and scored to 10 days. The files report days 10 to 14 separately.
- **Solar's upper band fails on very sunny days.** About 11% of observations exceed `q99`
  on the sunniest days in experiment 659.
- **The shift labels are an incomplete dump.** NGED has said that load-shift events may be missing
  from its logs, so clean rows can hold unlabelled shifts.
- **The last origins cannot be fully scored.** Observed power ends on 2026-11-09 and the wind idle
  mask ends on 2026-09-13.

## Anonymisation

Wind farms appear as Wind A to Wind C and solar farms as Solar A to Solar F, with no
coordinates, no calendar dates on any generator's series, and no generator identifiers. Substations
carry their names and dates.
