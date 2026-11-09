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

## The ground-truth files

Five more private files hold the observed power, so a reader can score the forecast files without
the NGED data store: `ground_truth_primaries.parquet`, `ground_truth_bsps.parquet`,
`ground_truth_gsps.parquet`, `ground_truth_wind.parquet`, and `ground_truth_solar.parquet`. The
forecast files never hold observed power. The script is
`ml-experiments/00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/e00668_ground_truth.py`.

- **Rows.** One row per series per half-hour, from 2025-11-01 00:00 UTC to 2026-11-09 08:00 UTC,
  the last half-hour in the data store when the files were written. Every series has the full
  time grid, and a missing reading is an explicit null row. No row exists after 2026-11-09 08:00
  UTC, so the last origins of the forecast files have no observed power for their later valid
  times, and a join from a forecast file gives no match there.
- **Columns, in this order.**
    - `time_series_id`: the series' original identifier, an 8-bit integer. The data store and the
      metadata table hold it as a 32-bit integer, so a join to them needs a cast.
    - `valid_time`: the half-hour, a UTC timestamp, with the same name and meaning as in the forecast
      files so that the two join on `(time_series_id, valid_time)`. The stamp is the source `time`
      column, which the power contract defines as the end of the observation period.
    - `power`: the cleaned reading as a 32-bit float, null where the production cleaning flagged the
      half-hour. The value is `power_raw` wherever `drop_reason` is null.
    - `power_raw`: the reading as delivered, a 32-bit float, read from the raw power table, and null
      where no reading exists.
    - `drop_reason`: a nullable string, the production cleaning rule that flagged the half-hour.
      The only rule in the cleaning code when the files were written is `substation_zero`, an
      exact zero from a Primary, BSP, or GSP.
    - `research_flag`: a nullable string, null in every file but the wind file. In the wind file it
      holds the rule name of the idle mask of experiment 210 (`low_wind_idle` or
      `outage_suspected`). The mask is a research tool and not part of the production cleaning.
      It does not null `power`, and it ends at 2026-09-13 17:30 UTC, so a wind row after that
      carries no flag whether or not the farm was idle.
    - `power_unit`: `"MW"` or `"MVA"` for each series, never implicit. Some Primaries and two wind
      farms are MVA series.
- **How to read the nulls.** A null `power` means no usable reading and is not zero. Solar nulls
  are frequent at night, because the feed omits many night readings, so solar scores use daylight
  half-hours. Earlier experiments scored the half-hours that the wind idle mask flags as zero, by
  setting the target to 0; a scorer that reproduces those numbers does the same, and a scorer
  that drops them gets different numbers.
- **Capacity.** A private `series_capacity.parquet` in the same folder holds the capacity that the
  scores divide by, one row per series. The rule is the production effective-capacity table, which
  has one value per series. It is not the capacity that the wind and solar networks train with (the
  99.9th percentile of the 730 days before each refit). This page lists no generator capacity or
  identifier.
- **Baselines.** The manual heuristic and climatology need power from before 2025-11-01, which these
  files do not hold. They read the cleaned power table of the data store.
- **Checks.** The row count equals the series count times the half-hours in the grid for every
  file, no key repeats, the time-zone is UTC, and the series are 16 Primaries, 2 BSPs, 2 GSPs,
  3 wind farms, and 6 solar farms. A fresh Sonnet audit re-read all five files against the raw and
  cleaned tables for every series and found no mismatch.

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
      The 7 means (with `log1p` precipitation) and the 7 standard deviations are z-scored with the
      training rows' mean and standard deviation and clipped at plus or minus 5. Each series reads
      the
      weather of the H3 resolution-5 cell that holds it.
    - Calendar and sun: 21 columns: solar zenith cosine, azimuth sine and cosine, the sine and
      cosine of the local hour at periods of 24 h and 12 h (Europe/London), a one-hot day of week,
      a COVID flag, three Christmas columns, and three bank-holiday columns.
    - Lead: the lead in hours divided by 336, and its square root.
    - Power lags: for each of a day lag and a week lag, the latest same-half-hour reading at least
      a minimum age before the network's origin, as a z-scored value, a missing flag, and an age
      code. The network's own origin is 09:00 UTC (the 00Z run plus 9 hours), and the lag cap counts
      back from that origin. The stored Candidate used a cap of 24
      hours. The backtest uses the existing 3.5-hour cap of experiment 574 (7 half-hours, the V35
      arm), which reads power no newer than 05:30 UTC on the origin day: 6.5 hours before the 12:00
      file origin and earlier than the 06:00 cut of the contract. The only change from the stored
      Candidate is that the lag encoder is pre-trained at that cap (experiment 667's plan).
    - Thermal-lag and solar-gain columns (14): trailing temperature means (3, 6, and 24 hours),
      temperature changes (3 and 6 hours), the day's minimum and maximum, the count of cold-snap
      days, trailing shortwave means (3, 6, and 24 hours), the 3-hour change in the clear-sky
      index, and two columns from the previous day's CAMS irradiance.
    - Neighbour and gap signals (60 columns, read only by the lag encoder): for each lag, the gap
      between the observed power and a lag-free forecast table (the median of the lags-masked
      network of the same seed at its freshest earlier origin with a lead of at most 24 hours),
      exponentially weighted means of that gap with
      half-lives of 1, 3, and 7 days, and, over the electrical neighbours of the Primary,
      the sum, mean, minimum, and maximum of those quantities scaled by capacity ratio, the
      own-plus-neighbour sum, a neighbour count, and a flag for no neighbour. The neighbours are the
      Primaries that NGED's "Primary Substation Interconnections" list links to the Primary. During
      fine-tuning,
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
  steps on a range of plus or minus 40 standard deviations. The mixture weights are a log-softmax
  and each standard deviation is `exp` of the raw output clamped to plus or minus 3. The values are
  then rescaled to power and sorted within the row. The final test of experiment 620 averaged 5
  seeds quantile by quantile, sorted again, and widened the band about the median by a factor of
  1.0934, which experiment 613 fixed before that test. Experiment 667 plans 2 seeds and, by
  default, no widening.
- **Loss.** The negative log-likelihood of the mixture on the standardised target, weighted, plus
  the knot penalty.
- **Training, in three steps.**
    - Pre-training uses ERA5 and CAMS (shortwave) as stand-in weather, in pseudo-runs every 2 days
      from 2019-09-01 at 09:00 UTC with leads of 0.5 to 351 hours (about 8.1 million rows for the
      2026-07 fold). Five Primaries start later (D11 2019-09-17, D10 2019-12-01, D5 2020-02-01, D14
      2020-07-01, D16 2021-02-01). ERA5 has no ensemble, so the 7 spread columns hold the fold's
      median ensemble spread for each 24-hour lead band. All weights train: AdamW, learning rate
      5e-4, weight decay 1e-3, batch 8,192,
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
- **R-haz, the prevailing head.** Experiment 652 fitted it on top of the frozen K1 network of
  experiments 612 and 613, and experiment 667 plans to put it on the Candidate. It is a small
  network that reads 58 origin
  features (the gap and neighbour signals above, built from lags), 4 age columns (the log of the
  departure's age in days, the age over 30, the gate flag, and the age of the freshest lag), and
  3 row inputs (the lead over 240 hours and the lower and upper half-widths of the Candidate's
  band). The counts here are those of experiment 652: removing the gate flag leaves 64 inputs. It
  has two hidden layers of 64 ReLU units and 12 outputs: a signed depth (0.5 times the
  hyperbolic tangent of the first output, so up to half of capacity), a spread, and daily hazards.
  Each hazard is `softplus(output - 3)`, and the survival probability `p` is the exponential of
  minus the hazards accumulated over the days before the lead. The probability is 0 when the
  departure is younger than half a day.
- **How the prevailing quantiles are formed.** The Candidate's 11 quantiles define a
  piecewise-linear
  cumulative distribution with two tail knots 1.5 times the outer spacing beyond the 1% and 99%
  quantiles. The prevailing distribution is the mixture `(1 - p)` times the Candidate's
  distribution plus `p` times the Candidate's distribution shifted by the depth, with the depth
  uncertain at three Gauss-Hermite points. The 11 prevailing quantiles are read off that mixture
  and are monotone by construction. When `p` is 0, the prevailing rows equal the NRA rows.
- **Loss and training of R-haz.** The loss is the pinball loss over the 11 levels plus 0.05 times
  the binary cross-entropy of `p` against the observed state of an injected shift. The training rows
  mix real rows with injected shifts, with durations drawn half from 3 to 14 days and half from 28
  to
  112 days (an assumed mixture) and each injected shift counted equally. Adam, learning rate 0.003,
  weight decay 1e-4, batch 4096, up to 30 epochs, patience 4, 2 seeds averaged by mean quantile.
  The head of experiment 652 has 9,164 parameters (12 outputs) and no dropout. Experiment 667 plans
  14 daily hazards (16 outputs, so about 9,360 parameters) and no gate flag, so the numbers above
  are
  stored values and the plan changes them.
- **Refits and embargo in the backtest.** Twelve monthly refits (test months 2025-11 to 2026-10),
  each trained on data before the first of the month with the 7-day embargo of the refit rule.
  The ensemble history starts 2024-04-01, so the early folds fine-tune on less ensemble history
  than the later folds. The D15 and D18 rows before 2025-03-01 are cut, as in every recent
  experiment. The planned seeds are 2 for the Candidate and 2 for R-haz, an open item of that page.
- **Where each piece came from.** The modular network and its widths: experiment 545. Frozen
  encoders and the three-step schedule: 559. ERA5 and CAMS pre-training: 418. The offset: 565 and
  572. The thermal-lag columns: 437 and 569. The embargoed schedule: 574. The shared lag encoder:
  586 and 591. The signal columns: 598 and 606. The signal dropout: 610. The refit rule: 599. The
  roster and the width factor: 612 and 613. The prevailing head: 647, 650, and 652.

### Performance

To be added when `primaries.parquet` is written. It will include the verified load shift between
D5 and D16 (start 2025-12-30 01:00 UTC, end 2026-01-14 11:30 UTC). For each of D5 and D16 there will
be figures of the observed power from 14 days before the start to at least 7 days after the end,
with the start and end marked. One kind of figure shows the day-ahead series for the NRA and the
prevailing outputs with the q05 to q95 band of each, and the other shows the full 14-day forecasts
from several origins: before the start, early and late inside the shift, in the last 24 hours before
the end, and just after the end. K1 or the manual heuristic is added as a labelled third line where
it exists. The text states, from the file, how many hours the prevailing output took to move at the
start and to return after the end, and that this is one descriptive event.

## BSPs and GSPs

The two BSPs and two GSPs are forecast by the lag-enabled network of [experiment
633](../00633-bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network/result.md), which is
experiment 622's arm `lagged_b1`. The backtest is [experiment
669](../00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md). The
network has no neighbour inputs and no model of load-shift events. Its readings are net of embedded
generation, which no input models.

### Method

- **Inputs.** The same 37 lag-free columns, 8 lag columns, knot position, expected-weather table
  for the lag encoder (the weather and calendar columns of the freshest run issued at or before the
  lag's own time, passed through the same encoders), and 14 thermal-lag and solar-gain columns as
  the Candidate network above, at the 4 series, with one embedding of
  the series (2 entries per group), and no signal columns and no neighbour inputs. The 51-member
  means and standard deviations are taken at the series' H3 resolution-5 cell.
- **Lags.** The stored network read lags at least 24 hours old. The backtest retrains it with the
  3.5-hour cap of experiment 572 (7 half-hours before the 09:00 origin), so the newest power is the
  05:30 reading, 6.5 hours before the 12:00 origin.
- **Architecture.** The same encoders and mixer as the Candidate with a lag encoder of 83 inputs
  (3 lag columns, the two 32-wide expected-weather and expected-datetime embeddings, and the
  embeddings of the series and the lag), and a mixer with 145 inputs. One network is fitted per
  group
  (the two BSPs together, the two GSPs together), so the weights are shared across the 2 series of a
  group except for the series embedding and the offset.
- **Offset, loss, and quantiles.** The offset, loss, quantile bisection, and sorting are those of
  the Candidate, with no width factor.
- **Targets.** Power divided by effective capacity (D13 55.3 MW, D17 47.6 MW, D6 384.3 MW, D1
  170.9 MW), z-scored per series. The series are signed MW series.
- **Training.** The same three steps as the Candidate (ERA5 and CAMS pre-training, a step 2 with a
  fresh mixer of seed 12345 and frozen encoders, and a fine-tune with a 90-day half-life), the same
  optimiser settings, and a 3-day embargo. Fine-tuning rows span 730 days before the cut-off, with
  targets ending 3.5 hours before it. Lag block dropout is 30% of days (both lags), of which 15%
  mask
  the day lag only. Two seeds are averaged quantile by quantile and sorted.
- **Origin.** The network runs at its own origin of 09:00 UTC, the 00Z run plus 9 hours, and its
  output is relabelled with 12:00 UTC. Every input that the network reads at 09:00 was available at
  12:00.
- **Refits (planned).** Twelve monthly refits, one per test month 2025-11 to 2026-10, with the
  3-day embargo of the schedule. The final-test network of experiment 633 was one fit at 2026-07-01.
- **Where each piece came from.** As for the Candidate; the node harness and pooling: 564 and 611;
  the lag network at the nodes: 619; the matched rematch: 622; the final test: 633.

### Performance

#### 2 bulk supply points: all rows and by lead band

**The network beats both baselines at every lead band, and the margin shrinks with the lead.** The
tables give normalised error in points of capacity (percent of capacity; smaller is better), as the
mean over scored half-hours, and the gain of the network over each baseline with a 95% day-block
bootstrap interval from 2,000 resamples of origin days. The manual heuristic is the lagged-power rule
and climatology is the month, half-hour, and weekend-flag quantiles, both described under
[Comparison baselines](#comparison-baselines-slices-and-summary-figures). Days 10 to 14 are reported
separately because they lie beyond the 10-day horizon of the earlier stored backtests.

Normalised MAE of the median:

| Group | Half-hours | Model | Manual heuristic | Climatology | Gain over heuristic (%) | Gain over climatology (%) |
|---|---:|---:|---:|---:|---|---|
| days 0 to 1 | 35,014 | 7.71 | 12.09 | 13.07 | 36.3 (34.0 to 38.4) | 41.0 (38.9 to 43.2) |
| days 1 to 3 | 70,028 | 8.46 | 12.09 | 13.07 | 30.0 (28.0 to 31.9) | 35.3 (33.3 to 37.2) |
| days 3 to 7 | 139,970 | 10.13 | 12.09 | 13.07 | 16.2 (14.5 to 17.8) | 22.5 (20.8 to 24.0) |
| days 7 to 10 | 104,846 | 11.19 | 12.28 | 13.14 | 8.9 (6.9 to 10.8) | 14.9 (13.2 to 16.5) |
| days 10 to 14 | 138,560 | 11.68 | 12.30 | 13.22 | 5.1 (3.5 to 6.7) | 11.7 (10.1 to 13.0) |

CRPS (2 × mean pinball loss over the 11 levels):

| Group | Half-hours | Model | Manual heuristic | Climatology | Gain over heuristic (%) | Gain over climatology (%) |
|---|---:|---:|---:|---:|---|---|
| days 0 to 1 | 35,014 | 4.27 | 6.83 | 7.23 | 37.5 (35.3 to 39.7) | 41.0 (39.0 to 43.1) |
| days 1 to 3 | 70,028 | 4.67 | 6.83 | 7.24 | 31.5 (29.4 to 33.5) | 35.4 (33.5 to 37.3) |
| days 3 to 7 | 139,970 | 5.55 | 6.82 | 7.24 | 18.7 (17.0 to 20.4) | 23.4 (21.7 to 24.9) |
| days 7 to 10 | 104,846 | 6.13 | 6.98 | 7.28 | 12.3 (10.3 to 14.2) | 15.8 (14.2 to 17.4) |
| days 10 to 14 | 138,560 | 6.41 | 7.00 | 7.32 | 8.5 (6.9 to 10.1) | 12.5 (10.9 to 13.9) |

![Normalised MAE of the median by forecast horizon for 2 bulk supply points, three systems](bsps_mae_by_horizon.svg)

#### 2 bulk supply points: by season

Seasons are meteorological and refer to the season of the forecast origin. Autumn is split into the
months at the start of the backtest (2025-11) and at its end (2026-09 and 2026-10), because the
backtest year does not contain a full autumn.

Normalised MAE of the median:

| Group | Half-hours | Model | Manual heuristic | Climatology | Gain over heuristic (%) | Gain over climatology (%) |
|---|---:|---:|---:|---:|---|---|
| autumn (Nov 2025) | 40,320 | 9.18 | 12.04 | 10.96 | 23.7 (21.7 to 25.8) | 16.2 (13.7 to 18.5) |
| winter | 120,960 | 9.50 | 11.42 | 12.13 | 16.8 (14.5 to 19.0) | 21.7 (18.8 to 24.3) |
| spring | 123,620 | 11.63 | 13.36 | 15.19 | 12.9 (11.5 to 14.4) | 23.4 (21.6 to 25.3) |
| summer | 123,312 | 10.79 | 12.08 | 13.24 | 10.7 (9.4 to 12.1) | 18.5 (16.8 to 20.3) |
| autumn (Sep to Oct 2026) | 80,206 | 9.78 | 11.80 | 12.38 | 17.1 (13.2 to 21.3) | 21.0 (18.7 to 23.0) |

CRPS:

| Group | Half-hours | Model | Manual heuristic | Climatology | Gain over heuristic (%) | Gain over climatology (%) |
|---|---:|---:|---:|---:|---|---|
| autumn (Nov 2025) | 40,320 | 5.06 | 6.95 | 6.09 | 27.2 (24.9 to 29.5) | 16.9 (14.7 to 19.0) |
| winter | 120,960 | 5.21 | 6.53 | 6.64 | 20.3 (17.7 to 22.7) | 21.6 (18.7 to 24.3) |
| spring | 123,620 | 6.35 | 7.54 | 8.21 | 15.7 (14.3 to 17.1) | 22.6 (20.8 to 24.5) |
| summer | 123,312 | 5.90 | 6.83 | 7.50 | 13.6 (12.2 to 15.1) | 21.4 (19.7 to 23.2) |
| autumn (Sep to Oct 2026) | 80,206 | 5.45 | 6.61 | 7.00 | 17.6 (13.7 to 21.9) | 22.2 (19.6 to 24.5) |

![Normalised MAE of the median by season for 2 bulk supply points](bsps_mae_by_season.svg)

![Model error and CRPS gain over the manual heuristic, by season and lead band, for 2 bulk supply points](bsps_season_by_horizon.svg)

#### 2 bulk supply points: by calendar month

Normalised MAE of the median, by month of the forecast origin:

| Group | Half-hours | Model | Manual heuristic | Climatology | Gain over heuristic (%) | Gain over climatology (%) |
|---|---:|---:|---:|---:|---|---|
| 2025-11 | 40,320 | 9.18 | 12.04 | 10.96 | 23.7 (21.7 to 25.8) | 16.2 (13.7 to 18.5) |
| 2025-12 | 41,664 | 8.12 | 10.45 | 10.64 | 22.3 (19.1 to 25.4) | 23.7 (20.3 to 27.0) |
| 2026-01 | 41,664 | 9.26 | 10.51 | 12.33 | 11.9 (7.7 to 16.5) | 24.9 (20.3 to 29.3) |
| 2026-02 | 37,632 | 11.31 | 13.50 | 13.55 | 16.2 (13.3 to 19.2) | 16.6 (12.0 to 21.2) |
| 2026-03 | 41,636 | 12.85 | 14.84 | 16.16 | 13.4 (11.4 to 15.8) | 20.5 (18.4 to 22.9) |
| 2026-04 | 40,320 | 11.51 | 13.72 | 16.36 | 16.1 (13.6 to 18.6) | 29.6 (26.8 to 32.3) |
| 2026-05 | 41,664 | 10.52 | 11.51 | 13.08 | 8.6 (6.6 to 10.5) | 19.6 (17.6 to 21.6) |
| 2026-06 | 40,320 | 10.21 | 11.87 | 13.61 | 14.0 (12.4 to 15.5) | 24.9 (22.1 to 27.6) |
| 2026-07 | 41,664 | 10.92 | 11.77 | 12.74 | 7.2 (5.2 to 9.0) | 14.3 (12.1 to 16.3) |
| 2026-08 | 41,328 | 11.21 | 12.60 | 13.39 | 11.0 (8.0 to 14.0) | 16.3 (13.2 to 19.2) |
| 2026-09 | 40,320 | 9.52 | 10.97 | 12.44 | 13.2 (10.3 to 15.7) | 23.4 (20.6 to 26.1) |
| 2026-10 | 39,886 | 10.03 | 12.64 | 12.31 | 20.6 (13.6 to 27.4) | 18.5 (15.3 to 21.8) |

**Scored rows.** Of the 490,560 rows, 488,418 are scored. The rest have no observed power: the
ground-truth files end at 2026-11-09 08:00 UTC, so the last origins of 2026-10 cannot be scored in
full at long leads, and a few half-hours inside the year are missing or flagged by the cleaning.

#### 2 grid supply points: all rows and by lead band

**The network beats both baselines at every lead band, and the margin shrinks with the lead.** The
tables give normalised error in points of capacity (percent of capacity; smaller is better), as the
mean over scored half-hours, and the gain of the network over each baseline with a 95% day-block
bootstrap interval from 2,000 resamples of origin days. The manual heuristic is the lagged-power rule
and climatology is the month, half-hour, and weekend-flag quantiles, both described under
[Comparison baselines](#comparison-baselines-slices-and-summary-figures). Days 10 to 14 are reported
separately because they lie beyond the 10-day horizon of the earlier stored backtests.

Normalised MAE of the median:

| Group | Half-hours | Model | Manual heuristic | Climatology | Gain over heuristic (%) | Gain over climatology (%) |
|---|---:|---:|---:|---:|---|---|
| days 0 to 1 | 35,036 | 19.27 | 28.30 | 36.38 | 31.9 (28.5 to 35.0) | 47.0 (43.9 to 49.9) |
| days 1 to 3 | 70,072 | 21.06 | 28.41 | 36.46 | 25.8 (22.8 to 28.7) | 42.2 (39.4 to 45.0) |
| days 3 to 7 | 140,058 | 24.48 | 28.49 | 36.46 | 14.1 (11.8 to 16.3) | 32.9 (30.2 to 35.5) |
| days 7 to 10 | 104,912 | 27.02 | 28.93 | 36.65 | 6.6 (4.6 to 8.7) | 26.3 (23.5 to 29.1) |
| days 10 to 14 | 138,648 | 27.64 | 28.82 | 36.72 | 4.1 (2.5 to 5.7) | 24.7 (21.9 to 27.4) |

CRPS (2 × mean pinball loss over the 11 levels):

| Group | Half-hours | Model | Manual heuristic | Climatology | Gain over heuristic (%) | Gain over climatology (%) |
|---|---:|---:|---:|---:|---|---|
| days 0 to 1 | 35,036 | 10.52 | 15.60 | 18.69 | 32.6 (29.4 to 35.4) | 43.7 (40.7 to 46.4) |
| days 1 to 3 | 70,072 | 11.44 | 15.66 | 18.74 | 27.0 (24.3 to 29.5) | 39.0 (36.3 to 41.6) |
| days 3 to 7 | 140,058 | 13.18 | 15.69 | 18.75 | 16.0 (14.1 to 17.9) | 29.7 (27.5 to 32.1) |
| days 7 to 10 | 104,912 | 14.52 | 16.07 | 18.83 | 9.7 (7.8 to 11.6) | 22.9 (20.3 to 25.5) |
| days 10 to 14 | 138,648 | 14.78 | 16.03 | 18.87 | 7.8 (6.3 to 9.4) | 21.7 (19.1 to 24.2) |

![Normalised MAE of the median by forecast horizon for 2 grid supply points, three systems](gsps_mae_by_horizon.svg)

#### 2 grid supply points: by season

Seasons are meteorological and refer to the season of the forecast origin. Autumn is split into the
months at the start of the backtest (2025-11) and at its end (2026-09 and 2026-10), because the
backtest year does not contain a full autumn.

Normalised MAE of the median:

| Group | Half-hours | Model | Manual heuristic | Climatology | Gain over heuristic (%) | Gain over climatology (%) |
|---|---:|---:|---:|---:|---|---|
| autumn (Nov 2025) | 40,320 | 35.48 | 38.76 | 40.75 | 8.5 (5.6 to 10.9) | 12.9 (11.3 to 14.6) |
| winter | 120,960 | 31.93 | 38.59 | 37.44 | 17.3 (14.9 to 19.7) | 14.7 (11.4 to 18.0) |
| spring | 123,620 | 23.00 | 25.98 | 37.93 | 11.5 (9.1 to 13.7) | 39.4 (38.1 to 40.7) |
| summer | 123,620 | 16.28 | 18.50 | 34.17 | 12.0 (9.6 to 14.3) | 52.4 (51.0 to 53.6) |
| autumn (Sep to Oct 2026) | 80,206 | 26.16 | 28.35 | 34.74 | 7.7 (2.3 to 13.1) | 24.7 (14.8 to 33.3) |

CRPS:

| Group | Half-hours | Model | Manual heuristic | Climatology | Gain over heuristic (%) | Gain over climatology (%) |
|---|---:|---:|---:|---:|---|---|
| autumn (Nov 2025) | 40,320 | 18.10 | 19.99 | 20.08 | 9.5 (6.3 to 12.3) | 9.9 (8.1 to 11.7) |
| winter | 120,960 | 16.55 | 19.88 | 19.05 | 16.7 (14.8 to 18.8) | 13.1 (10.4 to 15.8) |
| spring | 123,620 | 12.80 | 14.78 | 19.82 | 13.4 (11.3 to 15.3) | 35.4 (34.0 to 36.8) |
| summer | 123,620 | 9.20 | 11.32 | 17.39 | 18.7 (15.9 to 21.5) | 47.1 (45.5 to 48.5) |
| autumn (Sep to Oct 2026) | 80,206 | 14.17 | 16.37 | 18.36 | 13.5 (8.7 to 18.3) | 22.9 (14.5 to 30.3) |

![Normalised MAE of the median by season for 2 grid supply points](gsps_mae_by_season.svg)

![Model error and CRPS gain over the manual heuristic, by season and lead band, for 2 grid supply points](gsps_season_by_horizon.svg)

#### 2 grid supply points: by calendar month

Normalised MAE of the median, by month of the forecast origin:

| Group | Half-hours | Model | Manual heuristic | Climatology | Gain over heuristic (%) | Gain over climatology (%) |
|---|---:|---:|---:|---:|---|---|
| 2025-11 | 40,320 | 35.48 | 38.76 | 40.75 | 8.5 (5.6 to 10.9) | 12.9 (11.3 to 14.6) |
| 2025-12 | 41,664 | 31.36 | 40.46 | 43.11 | 22.5 (17.8 to 27.0) | 27.3 (24.5 to 30.1) |
| 2026-01 | 41,664 | 32.44 | 37.27 | 32.82 | 12.9 (10.4 to 15.5) | 1.1 (-2.6 to 5.0) |
| 2026-02 | 37,632 | 31.99 | 37.98 | 36.29 | 15.8 (11.8 to 19.5) | 11.8 (5.8 to 17.1) |
| 2026-03 | 41,636 | 25.41 | 32.09 | 40.29 | 20.8 (18.7 to 22.9) | 36.9 (34.8 to 38.9) |
| 2026-04 | 40,320 | 21.40 | 23.13 | 34.83 | 7.5 (4.0 to 10.8) | 38.6 (35.8 to 41.3) |
| 2026-05 | 41,664 | 22.14 | 22.64 | 38.57 | 2.2 (0.9 to 3.5) | 42.6 (41.3 to 44.0) |
| 2026-06 | 40,320 | 15.99 | 18.81 | 34.75 | 15.0 (13.3 to 16.7) | 54.0 (53.0 to 54.9) |
| 2026-07 | 41,664 | 15.93 | 15.40 | 28.77 | -3.5 (-6.7 to -0.5) | 44.6 (42.6 to 46.5) |
| 2026-08 | 41,636 | 16.92 | 21.32 | 39.03 | 20.7 (18.1 to 23.2) | 56.7 (54.7 to 58.6) |
| 2026-09 | 40,320 | 20.07 | 25.77 | 41.34 | 22.1 (19.6 to 24.6) | 51.5 (49.4 to 53.4) |
| 2026-10 | 39,886 | 32.33 | 30.97 | 28.06 | -4.4 (-12.4 to 3.6) | -15.2 (-27.5 to -4.0) |

**Scored rows.** Of the 490,560 rows, 488,726 are scored. The rest have no observed power: the
ground-truth files end at 2026-11-09 08:00 UTC, so the last origins of 2026-10 cannot be scored in
full at long leads, and a few half-hours inside the year are missing or flagged by the cleaning.

## Wind

Wind farms are forecast by the simple network S of [experiment
659](../00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md). The
backtest is [experiment
666](../00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md).

### Method

- **Inputs.** The mean and standard deviation over the 51 members of 10 m and 100 m wind speed,
  the sine and cosine of the 100 m wind direction, and 2 m air temperature, plus the lead (the
  sine and cosine of the lead in hours modulo 24, and the lead in hundreds of hours) and a one-hot
  wind-farm
  column. There is no power lag, no calendar column, and no physics-lite input in network S.
- **Architecture.** Two hidden layers of 64 units feeding a three-component Gaussian mixture over
  power as a fraction of capacity, centred on the training mean of the target. The 11 quantiles are
  read from the mixture and clamped to the range 0 to 1 of capacity. Inputs are standardised with
  the
  fit rows' mean and standard deviation.
- **Training.** AdamW, learning rate 0.002, batch 2048, weight decay 0.0001, 40 epochs, and the
  epoch with the lowest negative log-likelihood on the last 62 days is kept. Idle readings
  (experiment 210's idle mask) are set to zero in training
  and scoring. Three seeds are averaged quantile by quantile.
- **Capacity.** The 99.9th percentile of the farm's power over the 730 days before each refit.
- **Refits (planned).** Wind refits on day 0 of each month (12 refits), training on runs before 62
  days earlier. No ERA5 or CAMS pre-training. Experiment 659 refitted twice a month.
- **Missing inputs.** The network reads only the ensemble weather. The end date of the idle mask
  is read at run time, and the performance section states it.

### Performance

To be added when `wind.parquet` is written.

## Solar

Solar farms are forecast by the simple network PLP of experiment 659. The backtest is
experiment 666.

### Method

- **Inputs.** The mean and standard deviation over the 51 members of shortwave radiation, air
  temperature, and 10 m wind speed; the ensemble means of relative humidity, longwave radiation,
  precipitation, and surface pressure; the sun's elevation and the sine and cosine of its azimuth;
  the lead; and the mean and standard deviation of a physics-lite power and of the cell
  temperature. There is no farm column and no calendar column.
- **Physics-lite power.** Irradiance over 1000 W per square metre, times one minus 0.004 per kelvin
  times the cell temperature minus 25 degrees Celsius, clipped to the range 0 to 1, with the Faiman
  cell temperature `T_air + G / (25 + 6.84 x wind speed)`. The formula runs on each of the 51
  members
  separately, and the network reads the mean and standard deviation. The mixture is centred on the
  mean physics-lite power.
- **Architecture, loss, and optimiser.** As for wind, with 20 epochs, the epoch with the lowest
  negative log-likelihood on the last 61 days kept, and a lead range extended to 336 hours
  (experiments 430 and 659 stopped at 251.5 hours). Solar has no idle mask, and the target is the
  raw output.
- **Refits (planned).** Quarterly: 2025-11-01, 2026-02-01, 2026-05-01, and 2026-08-01, each scoring
  the next
  three months, 3 seeds each, with no pre-training.
- **Capacity.** As for wind.
- **Missing inputs.** The network reads only the ensemble weather.

### Performance

To be added when `solar.parquet` is written.

## Performance by season

This section is written with the per-file performance, and holds one analysis for every forecast
type (Primaries, BSPs, GSPs, wind, solar). The seasons are the four meteorological seasons: winter
(December to February), spring (March to May), summer (June to August), and autumn (September to
November). The backtest year runs from 2025-11 to 2026-10, so autumn is split across its two ends,
and the section also reports by calendar month. For each type and season the tables give the
normalised mean absolute error of the median and the CRPS of the model, the manual heuristic where
one exists, and climatology, with the gain over the manual heuristic and over climatology as a
percentage and a day-block bootstrap interval. Each type has a figure of the normalised error by
season for the three arms, with a y-axis that does not start at zero. The demand types add the
split by lead band (days 0 to 1, 1 to 3, 3 to 7, and 7 to 14) within each season. The section says
what the tables show and asserts no cause that the numbers do not support. Generator series carry
season labels and no calendar dates.

## Comparison baselines, slices, and summary figures

This section is written last. It will hold five items.

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
- **One score-against-horizon plot per forecast type** (Primaries, BSPs, GSPs, wind, solar). The
  horizon (0 to 14 days) is on the x-axis and the normalised mean absolute error of the median on
  the y-axis, which does not start at zero. Three lines: the manual heuristic, climatology, and
  the model. Wind and solar have no manual heuristic unless one exists for that technology, and
  the plot says so; solar counts daylight half-hours only.
- **A dot-plot figure for each demand file** (Primaries, BSPs, GSPs), in the style of the
  bake-off pages: panels for MAE, CRPS, and twCRPS side by side, one dot per arm (manual
  heuristic, climatology, the model), and a half-hours column. One block is "By forecast horizon"
  (day 1, day 2, days 3 to 4, days 5 to 7, days 8 to 10, days 11 to 14). The other block is "By
  condition" (all days, winter, winter nights, cold nights, Economy 7 step hours, Christmas, bank
  holidays). The page reuses the metric, threshold, and condition code of `studies.eval_protocol`
  (`metrics.py`, `slices.py`) unchanged and restates each definition.
- **Two extra arms for the demand files:** Primary demand research default 3 and BG2, in the same
  tables and plots. Both reach 10 days at most, and their stored forecasts cover the development
  months and the final-test window, not the whole year, and may not exist for the BSPs and GSPs.
  The comparison uses stored forecasts on the common rows only and states the rows and months it
  uses. Gaps are not filled by guessing, and a CPU run to extend the comparison starts only after
  the maintainer agrees to its stated cost.

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
