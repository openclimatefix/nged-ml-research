# Final models: architectures and training regimes for the production port

**This page states, for each forecast type, what the final model is and how it is trained.** It is
written for the person who ports the models into `nged-substation-forecast`. Each section covers one
forecast type and has the same five blocks: architecture, training regime, post-processing, scores,
and port notes. Every choice links to the experiment page that decided it. The page adds no new
scoring. Where two pages disagree, the page says so and names the current one.

**None of these models is adopted for production, and every score comes from one backtest year that
overlaps the development months.** The backtest is described in [experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md):
one forecast origin per day at 12:00 UTC from 2025-11-01 to 2026-10-31, and 672 half-hours (14 days)
per origin. The origin months from 2026-07 lie in the window that earlier experiments kept as a
final test, so those months are descriptive and no choice rests on them. The experiment 677 page,
which builds up the Candidate demand neural network one component at a time, is not merged. Its slot
in the Primaries section is marked.

**Terms used throughout.** A Primary is a primary substation, a BSP is a bulk supply point, and a
GSP is a grid supply point. CRPS is the continuous ranked probability score, computed here as twice
the mean pinball loss over the 11 quantile levels, divided by capacity, times 100. A CRPS of 4
therefore means 4% of capacity. The normalised MAE of the median is the mean absolute error of the
50% quantile in the same units. Smaller is better for both. The q05-to-q95 coverage is the share of
observed power inside the band from the 5% quantile to the 95% quantile, and its nominal value is
0.90. NRA is the normal running arrangement, the electricity network configuration without any load
shift. A load shift is a transfer of demand between neighbouring Primaries, usually for days to
weeks, that NGED's logs record incompletely.

## Overview

| Forecast type | Series | Neural network and output | Post-processing in the backtest files | Headline CRPS, days 0 to 1 and days 10 to 14 |
|---|---|---|---|---|
| Primaries, NRA output | 16 Primaries | The Candidate, a 3-component Gaussian mixture, 11 quantiles | Quantiles of the 8 MVA Primaries clamped at zero. No calibration. | 3.84 and 4.59 |
| Primaries, prevailing output | 16 Primaries | The Candidate's quantiles mixed with a shifted copy by a survival head with 14 extra lagged inputs | Same clamp. Walk-forward band scale in the calibrated copy. | 3.37 and 4.47 |
| BSPs | 2 BSPs | Lag-enabled mixture neural network, one per group | Walk-forward band scale in the calibrated copy | 4.27 and 6.41 |
| GSPs | 2 GSPs | Lag-enabled mixture neural network, one per group | Walk-forward band scale in the calibrated copy | 10.52 and 14.78 |
| Wind | 3 wind farms | Network S, a 3-component Gaussian mixture on ensemble weather | None | 4.58 and 9.86 |
| Solar | 6 solar farms | Network PLP, a 3-component Gaussian mixture on ensemble weather and physics-lite power | Night quantiles set to zero. Walk-forward band scale in the calibrated copy. | 4.83 and 7.18 |

The scores in the last column come from [experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md).
The days 10 to 14 column is the band of that page's table that spans days 10 to 14. Section "Scores"
of each forecast type gives the other lead bands, the coverage, and the climatology and
manual-heuristic references.

## Series, identifiers, and units

**The backtest files identify a series by its original `time_series_id`, an 8-bit integer, and the
experiment pages mix that identifier with NGED's D-labels.** Page [experiment
683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md)
and page [experiment
684](00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/result.md)
write "Primary 13" for identifier 13, which is the Primary labelled D18. The table fixes the
correspondence for the 16 Primaries. The mapping comes from `site_map.csv` of experiment 521, which
the scoring code of [experiment
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
  Primaries (D5, D12, D16, D18, and D22), and the wind and solar quantiles exceed it by up to 12%
  ([experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md),
  Caveats).

## Demand at the 16 Primaries

### Architecture

**The Primaries are forecast by the Candidate, a modular demand neural network that outputs 11
quantiles, plus a small prevailing head.** The Candidate is arm `P7+D50` of [experiment
612](00612-nra-combined-networks-three-class-roster-completion/result.md) and [experiment
613](00613-nra-roster-contrasts-with-intervals-five-seeds/result.md), which [experiment
620](00620-bakeoff-final-test-demand-networks-at-primaries/result.md) tested on the final-test
months. Its output is the NRA output. The prevailing output comes from the survival-hazard head of
[experiment 652](00652-two-output-forecast-with-explicit-rules-and-survival-hazard/result.md), with
the 14 extra lagged inputs of [experiment
681](00681-extra-lagged-power-inputs-for-the-prevailing-forecast/result.md). [experiment
667](00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md) is the first
backtest of the pair, and [experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
restates every detail below and is the current page where the two differ.

- **One row** is a (Primary, ensemble run, valid half-hour). The neural network runs once per
  ensemble run, on summaries of the 51 members. The weather is the ECMWF ensemble (ENS) 00Z run of
  the origin day, which the neural network treats as available at 09:00 UTC. The forecast is made at
  that native 09:00 UTC origin and relabelled to the contract's 12:00 UTC origin by adding 3 hours.
  Valid times before 12:00 UTC are dropped. The lead runs to 339 hours, because ENS summaries reach
  about 351 hours.
- **Weather inputs.** The mean and the standard deviation (`ddof=0`) over the 51 members of 7
  variables: shortwave and longwave radiation, 2 m temperature, 2 m dew point, surface pressure,
  precipitation (after `log1p` of the value clipped at zero), and 10 m wind speed. All 14 columns
  are z-scored with the training rows' mean and standard deviation and clipped at plus or minus 5.
  Each series reads the weather of the H3 resolution-5 cell that holds it.
- **Calendar and sun inputs.** 21 columns: the cosine of the solar zenith angle, the sine and cosine
  of the azimuth, the sine and cosine of the local hour (Europe/London) at periods of 24 hours and
  12 hours, a one-hot day of week, a COVID flag, 3 Christmas columns, and 3 bank-holiday columns.
  The lead enters as the lead in hours over 336 and its square root.
- **Thermal-lag and solar-gain inputs.** 14 columns from [experiment
  437](00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles/result.md) and
  [experiment
  569](00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1/result.md):
  trailing temperature means (3, 6, and 24 hours), temperature changes (3 and 6 hours), the day's
  minimum and maximum, the count of cold-snap days, trailing shortwave means (3, 6, and 24 hours),
  the 3-hour change in the clear-sky index, and two columns from the previous day's CAMS irradiance.
  Before the origin they use the analysis proxy built from the first day of the previous 00Z run,
  which exists at forecast time.
- **Power lags.** A day lag and a week lag. Each is the latest same-half-hour reading at least 24
  hours before the neural network's own 09:00 origin, entered as a z-scored value, a missing flag,
  and an age code. The newest lag is therefore the 09:00 reading of the day before the origin day,
  27 hours before the contract's 12:00 origin, so the contract's 6-hour rule needed no change.
- **Neighbour and gap signals.** 60 columns read only by the lag encoder. For each lag: the gap
  between the observed power and a lag-free forecast table (the median of the lags-masked neural
  network of the same seed at its freshest earlier origin with a lead of at most 24 hours),
  exponentially weighted means of that gap with half-lives of 1, 3, and 7 days, and, over the
  electrical neighbours of the Primary, the sum, mean, minimum, and maximum of those quantities
  scaled by capacity ratio, the own-plus-neighbour sum, a neighbour count, and a flag for no
  neighbour. The neighbours are the Primaries that NGED's "Primary Substation Interconnections" list
  links to the Primary ([experiment
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
outputs, the embedding, the 7 spread columns, and the 2 lead columns), has two hidden layers of 64
units with 10% dropout, and outputs 9 numbers: the logits, means, and log standard deviations of a
3-component Gaussian mixture. No merged page states the parameter count of the Candidate.

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

**The output distribution may be truncated at zero for Primaries that cannot go negative, and only
[experiment
684](00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/result.md)
has tested that.** The backtest files carry a post hoc clamp instead (see "Post-processing").
[experiment
684](00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/result.md)
refitted the Candidate with the mixture conditioned on power at or above zero, so the negative
log-likelihood adds the log of the mixture's mass above zero and the quantiles are read from the
conditional cumulative distribution. It tested the family at four Primaries, D12, D22, D8, and D5
(identifiers 14, 15, 26, and 17), on the test months 2026-02 and 2026-05, and on the NRA output
only. D5 is metered in MW and never goes negative in the observed power. The CRPS changed against
the stored forecasts by +0.030 (D12), -0.015 (D22), -0.070 (D8), and -0.010 (D5), with no negative
q01 left. The verdict was partly supported and the experiment's own status is candidate. A gamma
mixture trained inside the same shared neural network failed its stop rule at two Primaries and was
dropped, and it also worsened the Primaries that kept a Gaussian output. Whether the truncated
family helps the other five MVA Primaries (D2, D3, D4, D19, and D21), and the prevailing output, was
not tested.

**The prevailing output is a second small neural network, the head R-haz, that mixes the Candidate's
quantiles with a shifted copy.** The Candidate's 11 quantiles define a piecewise-linear cumulative
distribution with two tail knots, 1.5 times the outer spacing beyond the 1% and 99% quantiles. The
prevailing distribution is the mixture `(1 - p)` times the Candidate's distribution plus `p` times
the Candidate's distribution shifted by a learned depth, with the depth uncertain at three
Gauss-Hermite points ([experiment
647](00647-two-output-forecast-departure-head-on-frozen-k1/result.md)). The 11 prevailing quantiles
are read off that mixture and are monotone by construction. When `p` is 0, the prevailing rows equal
the NRA rows. R-haz sits on the frozen Candidate and has these parts:

- **Inputs, 93 in all.** 58 origin features (the gap and neighbour signals above, built from lags),
  4 age columns (the log of the departure's age in days, that age over 30, a gate flag, and the age
  of the freshest lag), and 3 row inputs (the lead over 240 hours and the lower and upper
  half-widths of the Candidate's band). The gate flag is kept as a column and set to zero, which
  gives 65 columns. The 14 extra lagged power inputs of [experiment
  681](00681-extra-lagged-power-inputs-for-the-prevailing-forecast/result.md), each as a residual
  plus a missing flag, add 28 columns.
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
- **Which extra-lag set was chosen.** [experiment
  681](00681-extra-lagged-power-inputs-for-the-prevailing-forecast/result.md) found that 14 extra
  inputs (set L3) make the prevailing output react 20.4 hours sooner (95% interval 16.3 to 24.8) to
  the start of an injected load shift, and 8 inputs (set L1) 19.1 hours sooner. The verdict was a
  small gain, under the 24 hours set in advance as worth porting. Only 2 real events could be read,
  and the 8-input set scored worst inside the real events. Which set to port is open on that page.
  The backtest files use the 14-input set, adopted by nged-ml-research PR 353 (Replace the
  prevailing rows of the Primaries backtest file with the extra-lag head).
- **The fallback head B-haz.** With fewer than 2 earlier banks of injected shifts (the folds 2025-11
  and 2025-12), the prevailing rows come from B-haz: the Candidate plus its own recent residual
  times a log-normal survival curve fitted on synthetic durations, with no fit on observed power.
  The prevailing rows equal the NRA rows on 74% of November origins and 65% of December origins, and
  on no origin from 2026-01. B-haz has no inputs that the extra lagged power could widen.

**Experiment 687 tried causal step-detector features for the prevailing head and did not adopt
them.** The features reacted 1.59 days sooner than the head of [experiment
652](00652-two-output-forecast-with-explicit-rules-and-survival-hazard/result.md) (interval -2.26 to
-1.00), but extra lagged power alone reacted 1.25 days sooner, so the features are not shown to beat
plain lags ([experiment
687](00687-causal-detector-features-for-the-prevailing-forecast-heads/result.md)). The prevailing
head of the backtest therefore carries no detector features.

!!! note "Pending experiment 677"

    experiment 677 builds up the Candidate one component at a time, from the date-time encoder to
    the full Candidate, with reference lines from a generalised additive model and an XGBoost model.
    Its plan is in the open nged-ml-research PR 338 (Experiment 677: building up the Candidate
    demand network one component at a time (plan)). This slot will state which components the
    Candidate needs and what each adds once the page merges.

### Training regime

**The Candidate is refitted 12 times, once per test month, in three steps, and each fit sees only
data from before its month.** The folds are the test months 2025-11 to 2026-10. [experiment
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

- **Data.** Cleaned power, with the `substation_zero` half-hours dropped (the only rule in the
  production cleaning code when [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  was written), and the D15 and D18 histories cut before 2025-03-01. The ENS history starts on
  2024-04-01, so the early folds fine-tune on less ensemble data than the later folds.
- **Step 1, pre-training.** The weather, datetime, and lag encoders pre-train on ERA5 and CAMS
  (shortwave) as stand-in weather, in pseudo-runs every 2 days from 2019-09-01 at 09:00 UTC with
  leads of 0.5 to 351 hours (about 8.1 million rows for the 2026-07 fold). Five Primaries start
  later (D11 on 2019-09-17, D10 on 2019-12-01, D5 on 2020-02-01, D14 on 2020-07-01, and D16 on
  2021-02-01). ERA5 has no ensemble, so the 7 spread columns hold the fold's median ensemble spread
  for each 24-hour lead band. All weights train, with AdamW, learning rate 5e-4, weight decay 1e-3,
  batch 8,192, 20% of the rows per epoch, up to 60 epochs, patience 8, and gradient-norm clip 1.0.
- **Step 2, fresh mixer.** The pre-trained weights load except the mixer. A fresh mixer is drawn
  with seed 12345, the weather and datetime encoders are frozen, and training runs on the 730-day
  history with uniform weights and a 3-day embargo, at learning rate 2e-4 (5e-4 for the fresh
  mixer).
- **Step 3, daily fine-tune.** The fine-tune starts from the step-2 weights with the same frozen
  encoders and learning rate 2e-4. Row weights are 0.5 to the power of the age in days over 90, a
  90-day half-life. The refit rule drops rows whose valid time lies in the newest 7 days before the
  cut-off and halves the weight of rows 7 to 14 days old. Early stopping holds out one week in every
  12 weeks of origin days, with 15 days purged before and 9 days after each held week, and uses at
  most 200,000 held rows. The ensemble weather is the input at this stage and at inference.
- **Seeds and widening.** Two seeds are averaged quantile by quantile and sorted again. The backtest
  applies no widening. The final test of [experiment
  620](00620-bakeoff-final-test-demand-networks-at-primaries/result.md) averaged 5 seeds and widened
  the band about the median by a factor of 1.0934, which [experiment
  613](00613-nra-roster-contrasts-with-intervals-five-seeds/result.md) fixed before that test.
- **Missing and delayed inputs.** A flagged or absent lag reads as value 0, a missing flag, and the
  maximum age code. In training, 30% of origin days mask both lags and, of those, 15% mask only the
  day lag. Absent signal columns read as 0 with a missing flag. Neighbour-signal columns whose
  training spread is under 80% of their spread in the last development month receive matching
  Gaussian noise during training. All weather columns must be finite.
- **Injected load shifts.** Each Primary receives random synthetic shifts, added to the lag inputs
  only, until they cover 5% of its row span: depth 15%, 30%, or 50% of capacity, length 3, 7, 15, or
  30 days, and random sign. Each shift on a Primary is subtracted, scaled by the capacity ratio,
  from one electrical neighbour, so the neighbour summaries see the transfer.
- **Training the prevailing head.** Each month's R-haz trains on "banks" of injected-shift rows, one
  bank per earlier test month, built from the real history of that month with synthetic shifts and
  forecast by that month's Candidate chain (inference only), so fold M's head trains only on banks
  of months before M and on rows whose valid time is before the start of M. The training rows mix
  real rows with injected shifts, with durations drawn half from 3 to 14 days and half from 28 to
  112 days (an assumed mixture, none from 15 to 27 days), and each fit samples 60,000 real rows and
  6,000 bank rows. Origin days 0 to 17 of an injected shift train the head and origin days from 21
  onward validate it. The loss is the pinball loss over the 11 levels plus 0.05 times the binary
  cross-entropy of `p` against the observed state of an injected shift. The optimiser is Adam with
  learning rate 0.003, weight decay 1e-4, batch 4,096, up to 30 epochs, patience 4, and 2 seeds
  averaged by mean quantile. The lag inputs of an injected shift are built from the same shifted
  power as its target.
- **Hardware and run time.** About 0.5 GPU-hours per fold for the Candidate chains ([experiment
  667](00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md)).
- **Capacity.** The effective capacity of the production table normalises the target. The neural
  network does not cap its output at that capacity.

### Post-processing

**The backtest file of the Primaries has one post hoc clamp, and the calibrated copy adds a scale
for the prevailing rows only.** Both are applied to stored quantiles after the neural networks run,
and neither changes the median.

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
  for the prevailing output, against a nominal 0.90. Both bands are too narrow. [experiment
  683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md)
  found that the NRA shortfall is mostly error in the median, not band width (see "Walk-forward band
  calibration").
- **Prevailing against NRA.** The prevailing output scores 0.03 points of capacity worse in the two
  fallback months (3.98 against 3.95), 0.17 better from 2026-01 to 2026-06, and 0.42 better from
  2026-07 to 2026-10. After the one verified load shift, at D5 and D16 in December 2025, the
  prevailing output scored worse than the NRA output for 7 days after the shift ended ([experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)).
- **Earlier prevailing numbers.** [experiment
  667](00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md) scored the
  prevailing output without the extra lagged inputs (CRPS 4.14, median error 7.39). [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  scores the current file (4.09 and 7.30). Use [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md).
  Its NRA numbers agree with [experiment
  667](00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md) to within
  0.01.

### Port notes

- **Reproduce exactly.** The input definitions and scalings above, the 24-hour minimum age of the
  Candidate's own lags, the three-step schedule with the same frozen encoders and fresh mixer seed,
  the 90-day half-life, the monthly refit on data before the first of the month with a 3-day embargo
  at the mixer step and a 7-day embargo at the fine-tune, the 11-quantile read-out by bisection, and
  the construction of the prevailing mixture from the Candidate's quantiles.
- **Unvalidated.** The truncated Gaussian output has been tested at only 4 Primaries and 2 months,
  and only for the NRA output. The prevailing head's behaviour on real load shifts rests on 1
  verified transfer. The survival probability beyond day 10 is the bank generator's duration prior
  and not an estimate from observed events. The backtest used 2 seeds and no widening, where the
  final test used 5 seeds and a factor of 1.0934.
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
neural network has no neighbour inputs and no model of load shifts, and its readings are net of
embedded generation, which no input models. One neural network is fitted per group (the two BSPs
together, the two GSPs together), so the weights are shared across the 2 series of a group except
for the series embedding and the offset.

- **Inputs.** The same 37 lag-free columns, 8 lag columns, knot position, expected-weather table for
  the lag encoder (the weather and calendar columns of the freshest run issued at or before the
  lag's own time, passed through the same encoders), and 14 thermal-lag and solar-gain columns as
  the Candidate, with one embedding of the series (2 entries per group) and no signal columns. The
  51-member means and standard deviations are taken at the series' H3 resolution-5 cell.
- **Lags.** The stored neural network of [experiment
  633](00633-bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network/result.md) read lags at least 24
  hours old. The backtest retrains it with the 3.5-hour cap of [experiment
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
  and [experiment 572](00572-two-offset-part-1-lags-versus-the-slow-output-offset/result.md), and
  the node harness of [experiment 564](00564-pooled-primary-bsp-gsp-m2-reduced/result.md).
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
- **Refits and seeds.** Twelve monthly refits, one per test month, and 2 seeds averaged quantile by
  quantile and sorted. The final-test neural network of [experiment
  633](00633-bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network/result.md) was one fit at
  2026-07-01. The early-stopping weeks are purged by 9 days, shorter than the 14-day leads, so a
  held-out week's targets can sit in training rows of later origins. [experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md) found no
  leak from this and records it as a caveat.
- **Targets and capacity.** Power divided by effective capacity (D13 55.3 MW, D17 47.6 MW, D6 384.3
  MW, and D1 170.9 MW), z-scored per series. All four are signed MW series.
- **Weather at lag times.** The thermal-lag and solar-gain columns use the analysis proxy built from
  the first day of the previous 00Z run, which exists at forecast time. The last roughly 5 days of
  the ERA5 and CAMS pre-training rows could not be read at a real cut-off, because ERA5 arrives
  about 5 days late.
- **Hardware and run time.** About 2 GPU-hours on one process and 1 core ([experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md)).

### Post-processing

- **Walk-forward band scale (calibrated copy only).** See "Walk-forward band calibration" below. It
  applies to all BSP and GSP rows from 2025-12.
- **No clamp and no cap.** The files keep the quantiles above capacity: the highest quantile is 2.2
  times the table value at D1 and 1.4 times at D6. A negative quantile is net export, and the q01 is
  below zero in 94% of the D6 rows and 99.6% of the D1 rows.

### Scores

**The lag-enabled neural network beats the manual heuristic and climatology at every lead band, and
the margin shrinks with lead.** The table gives CRPS in points of capacity from [experiment
668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md).

| Lead band | BSP CRPS | BSP climatology | GSP CRPS | GSP climatology |
|---|---:|---:|---:|---:|
| Days 0 to 1 | 4.27 | 7.23 | 10.52 | 18.69 |
| Days 1 to 3 | 4.67 | 7.24 | 11.44 | 18.74 |
| Days 3 to 7 | 5.55 | 7.24 | 13.18 | 18.75 |
| Days 7 to 10 | 6.13 | 7.28 | 14.52 | 18.83 |
| Days 10 to 14 | 6.41 | 7.32 | 14.78 | 18.87 |

- **Pooled over the 4 series,** [experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md) reports
  CRPS of 7.40, 8.06, 9.37, and 10.47 at days 0 to 1, 1 to 3, 3 to 7, and 7 to 14, and q05-to-q95
  coverage of 0.883, 0.877, 0.869, and 0.870.
- **Coverage.** 0.867 for the BSPs and 0.877 for the GSPs, against a nominal 0.90 ([experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)).
  The misses change side by month, so a level drift and not only a narrow band is at work.
- **Against the 24-hour-lag neural network.** On July to October 2026 origins at hours 0 to 72, the
  3.5-hour lags score 7.24 against 7.54, but only 0.16 of the 0.30 gain is attributable to the lag
  cut, and the rest comes from the monthly refits ([experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md)).
- **The largest miss.** The observed demand at D6 Walpole stepped up on 2026-10-19 while the median
  fell, a level step consistent with a load transfer that is not confirmed in NGED's logs and that
  this neural network does not model.

### Port notes

- **Reproduce exactly.** The 3.5-hour lag cap, the one-neural-network-per-group pooling, the D5
  output offset of [experiment 565](00565-drift-embedding-variants-six-designs/result.md), the
  thermal-lag columns of [experiment
  437](00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles/result.md), and the
  schedule of [experiment
  574](00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/result.md).
- **Unvalidated.** [experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md) reports
  no curtailment cost against the manual heuristic for the BSPs and GSPs. Load shifts are not
  modelled, and NGED's event labels are incomplete.
- **Open questions.** Whether the 3.5-hour cap or a longer one gives the better score with the same
  refit schedule, which [experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md) says was
  not tested on matched schedules. The D1 median is on average 18 MW below the observed median, and
  its monthly mean error swings between -39.8 MW and +25 MW. The Float64 originals of the BSP and
  GSP files are not recast to the Float32 of the calibrated copies.

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
  column. There is no power lag, no calendar column, and no physics-lite input. Inputs are
  standardised with the fit rows' mean and standard deviation.
- **Layers and output.** Two hidden layers of 64 units feeding a 3-component Gaussian mixture over
  power as a fraction of capacity, centred on the training mean of the target. The 11 quantiles are
  read from the mixture and clamped to the range 0 to 1 of the neural network's own capacity scale.
- **Weather.** The 00Z ENS run of the origin day, treated as available at 09:00 UTC, with leads of
  12 to 347.5 hours from the 00Z run. Where ensemble steps are 3 or 6 hours, the half-hourly value
  is interpolated as in [experiment 310](00310-metered-farm-long-lead-scoring/result.md) and
  [experiment 430](00430-unified-forecaster-solar-and-pretraining-ablations/result.md). No ERA5 or
  CAMS pre-training.

### Training regime

- **Refits.** Wind refits on day 0 of each month (12 refits), on runs and valid times before 62 days
  earlier, with the last 62 days used for early stopping. [experiment
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
  fits and 12 solar fits together ([experiment
  666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md)).

### Post-processing

None. The wind file is not clamped, and wind has no calibrated copy, because its stored coverage
(0.913) already exceeds 0.90 and a scale did not improve it ([experiment
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

- **Scale of the CRPS on the experiment 666 page.** [experiment
  666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md) reports CRPS values half
  as large as these (for example 2.29 at days 0 to 1), because it defines CRPS as the mean pinball
  loss and [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  as twice that. Compare numbers only within one page's convention, and use the [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  convention, which [experiment
  667](00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md), [experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md), and
  [experiment
  683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md)
  share.
- **Coverage.** 0.913 over all rows ([experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)).
  The wind median at days 10 to 14 is low in some months, and wide bands at long leads may explain
  why the coverage stays near 0.90.
- **Matched check.** On rows that [experiment
  659](00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md) also
  covers, the backtest scores 4.174 where [experiment
  659](00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md) scores
  4.179, with both using the capacity of [experiment
  659](00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md). Both
  figures use the convention of the [experiment
  666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md) page.

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
- **Physics-lite power.** Irradiance over 1,000 W per square metre, times one minus 0.004 per kelvin
  times the cell temperature minus 25 degrees Celsius, clipped to the range 0 to 1, with the Faiman
  cell temperature `T_air + G / (25 + 6.84 x wind speed)`. The formula runs on each of the 51
  members separately, and the neural network reads the mean and standard deviation. The mixture is
  centred on the mean physics-lite power. It has no fitted parameters.
- **Layers and output.** As for wind: two hidden layers of 64 units and a 3-component Gaussian
  mixture over power as a fraction of capacity, with the quantiles clamped to the range 0 to 1 of
  the neural network's capacity scale.
- **Night gate inside the neural network.** The neural network forecasts only the half-hours where
  the mean cosine of the solar zenith angle over the ensemble step is above zero, and sets the
  quantiles of the other half-hours to 0.

### Training regime

- **Refits.** Quarterly, on the first day of the first month of the backtest year and of each third
  month after it, each fit scoring the next three months, with 3 seeds and no pre-training. Training
  uses valid times before one day earlier, with the last 61 days for early stopping, 20 epochs,
  daylight rows only, and leads below 348 hours from the 00Z run ([experiment
  430](00430-unified-forecaster-solar-and-pretraining-ablations/result.md) and [experiment
  659](00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md) stopped
  at 251.5 hours, so the lead range is extended). Solar trains on all daylight half-hours, including
  more low-sun half-hours than [experiment
  659](00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md) kept.
- **Optimiser and seeds.** As for wind (AdamW, learning rate 0.002, batch 2,048, weight decay
  0.0001), keeping the epoch with the lowest negative log-likelihood on the last 61 days. The target
  is the raw output, because solar has no idle mask.
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
- **The same rule as a gate.** [experiment
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
- **Scale and row set.** [experiment
  666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md) reports CRPS values
  about half as large, for the reason given in the wind section, on a daylight flag defined by the
  mean cosine over the ensemble step. Its solar numbers are therefore not comparable with the table
  above.
- **Seasonal bias.** The solar median is high in spring and low in autumn, and the neural network
  has no calendar input. One year with quarterly refits cannot separate a seasonal effect from refit
  age ([experiment 666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md)).
- **At days 10 to 14,** the neural network scores worse than climatology from November to March and
  better from April to September ([experiment
  666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md)).

### Port notes

- **Reproduce exactly.** The physics-lite formula with the Faiman cell temperature, the 18 inputs,
  the mixture centred on the mean physics-lite power, the daylight-only training rows, and the post
  hoc night rule at each half-hour's own zenith.
- **Unvalidated.** The night gate was tested at 2 farms and one refit. Dawn and dusk coverage is
  poor. Arm b of [experiment
  684](00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/result.md),
  which would feed the physics-lite power at each half-hour's own zenith to the neural network, was
  not run.
- **Open question.** Whether a calendar input or monthly refits remove the seasonal bias in the
  median.

## Walk-forward band calibration

**A scale on the width of the quantile bands, fitted only on earlier months, brings four of five
outputs close to 0.90 coverage and is the one calibration the backtest files hold.** [experiment
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
  rows, which the scale brought to only 87.1% coverage, with a CRPS rise of 0.081 points
  ([experiment
  683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md)).
  The first month, 2025-11, has no earlier forecast to fit on and is left as stored.
- **Effect on coverage.** On 2025-12 to 2026-06, the q05-to-q95 coverage moves from 84.0% to 87.9%
  (stored) to 88.0% to 88.4% for the prevailing Primaries, from 87.5% to 89.3% to 89.8% to 90.9% for
  the BSPs, from 86.4% to 90.3% to 89.1% to 89.8% for the GSPs, and from 87.8% to 89.5% to 89.4% to
  91.0% for solar, each as a range over the four lead bands. The CRPS change is +0.047 for the
  prevailing Primaries, +0.005 for the BSPs, +0.003 for the GSPs, and -0.002 for solar.
- **A level shift made every file worse.** Adding a per-series level shift fitted on earlier months
  raised the NRA Primaries CRPS by 0.306 points and the GSPs' by 0.842, because the median bias
  changes from month to month.
- **The scale must not be fitted through an unexplained stretch.** A scale fitted through the July
  and August 2026 load transfers overshoots after they end: D18 (identifier 13) reaches 97.5%
  coverage in 2026-09 under the scale. The calibrated copies hold D18 and D14 (identifiers 13 and
  16) at their 2026-06 factor from 2026-07, a rule chosen after seeing the final-test months, so it
  is post hoc. D18 stays at 63% to 77% coverage in 2026-07, 2026-08, and 2026-10 even so.
- **The winter NRA shortfall is mostly error in the median.** [experiment
  688](00688-winter-normal-running-coverage-median-error-and-events-part-1/result.md) dropped the
  half-hours inside labelled and verified load transfers and shifted each band by its month's
  observed median error, and the coverage of the NRA Primaries rose from 80.1% to 90.1% in 2025-12
  and from 80.3% to 88.6% in 2026-01. The shift used the month's own errors, so it is a diagnostic
  and not a method. The stop rule that would have justified a learned width head did not fire.
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
  ([experiment
  683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md)
  lists this as April and [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  as the refit that starts May 2026).

## Production-port notes across forecast types

- **Series and units.** Carry the unit with the series, because the 8 MVA Primaries and 2 of the 3
  wind farms are apparent-power series and the rest are MW.
- **Origin and lags.** Every network reads the ENS 00Z run of the origin day as available at 09:00
  UTC and is relabelled to a 12:00 UTC origin. The Candidate's own lags are at least 24 hours old at
  its 09:00 origin, the BSP and GSP lags at least 3.5 hours old, and the prevailing head's extra
  lags at least 6 hours old. Wind and solar read no power lag. Whether the Primaries should also
  read 3.5-hour lags is not tested on a page read for this one.
- **What a service has to store.** For each demand and generator network, the fitted weights of each
  monthly (or quarterly, for solar) refit. For the prevailing head, a bank of injected-shift rows
  per earlier month. For calibration, a scale per series and lead band refreshed each month.
- **Serving path.** The production design principle is to keep the serving path close to loading a
  model and calling `predict`. The post hoc clamps and the band scale are the only steps here that
  act on predicted quantiles.
- **Unmetered solar.** None of the networks above has an explicit term for unmetered solar behind a
  Primary. [experiment
  673](00673-pv-capacity-for-nged-per-primary-with-register-cross-check/result.md) reads the
  capacity of unmetered solar from the demand networks' response to sunshine, as 2.4 to 3.9 MW at
  the six Primaries metered in MW and 1.25 to 6.62 MW at five main Primaries metered in MVA, and
  [experiment 679](00679-pv-mva-conversion-loss-reactive-floor-at-half-level/result.md) measures a
  conversion loss of 0.5 points (80% interval -1.0 to 7.7) in the reactive-floor world. These are
  context and are not inputs to any network above.

## Where the pages disagree

- **CRPS scale.** [experiment
  666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md) defines CRPS as the mean
  pinball loss, and [experiment
  667](00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md), [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md),
  [experiment
  669](00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md),
  [experiment
  683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md),
  and this page define it as twice that. The wind and solar numbers of [experiment
  666](00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md) are about half the
  numbers here. The current convention is the [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  one.
- **Prevailing Primaries scores.** [experiment
  667](00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md) reports a
  prevailing CRPS of 4.14, for the head without the extra lagged inputs. [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  reports 4.09 for the file after nged-ml-research PR 353. The current page is [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md).
- **Clamp counts.** The body of PR 342 counts 202,598 rows changed in the Primaries file, and
  [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  counts 206,487. The current page is [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md).
- **Where the D4 width step falls.** [experiment
  683](00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files/result.md)
  places the days 10 to 14 narrowing of identifier 11 at an April month start, and [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  at the refit that starts May 2026.
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
  observations in two summer months. [experiment
  668](00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  suggests a load shift that normal running ignores, and the cause has not been checked against the
  event table.
- What the Candidate needs and what each component adds: pending experiment 677.
- Whether wind needs a power lag or a different long-lead rule, given its score against climatology
  at days 10 to 14.
- Whether a calendar input or more frequent refits remove the solar median's seasonal bias.
- How to model load shifts at the BSPs and GSPs, where the largest observed miss looks like one.
- How the Float64 originals of the BSP and GSP files should be recast to match the Float32
  calibrated copies.
