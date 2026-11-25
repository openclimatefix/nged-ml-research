---
id: "00729"
slug: msa-d8-neural-untangler-synthetic-known-answers
title: >-
  Can a neural network trained on synthetic days with a known answer turn a day of D8's unsigned
  apparent-power reading into a calibrated probabilistic estimate of D8's signed power, and which
  extra input carries the information?
theme: solar-disaggregation
also_relevant_to:
  - gross-demand-simulation
status: planned
kind: other
size: >-
  Anything else. (1) It stores about 42 fitted neural networks, synthetic training and test banks,
  per-half-hour predictions, and score tables, all private; (2) more than one design defensibly tests
  the hypothesis (three neural-network architectures, two output heads, and a ladder of inputs), so
  all are arms with a pre-registered selection rule; (3) it imports experiment 694's tracker as a
  baseline and reads the stored frame of experiment 39 and the engine window of experiment 194,
  whose callers cannot be named without searching the repository; (4) it touches no serving path or
  degradation rule; (5) it changes no contract. Reviews, in order: this Opus plan, a Sonnet plan
  review, Sonnet code, a fresh Opus code review before any first run, a real-file smoke test, then
  the results, write-up, and prose reviews.
hypothesis: >-
  A neural network that reads 48 hours of D8's apparent-power reading together with satellite
  sunshine, the half-hour steps of the bulk supply point meter above D8, and the calendar and
  temperature inputs of a demand forecast, and that is trained only on synthetic days built from
  D8's own readings before its gas engine was connected, returns for each half-hour of the middle
  24 hours a two-lobe distribution of D8's signed power. On held-out synthetic days the network's
  signed continuous ranked probability score (CRPS) beats the best rule that does not learn from the
  day's shape, and on real days the network neither invents reverse flow where there is none nor
  misses the metered engine.
prediction: >-
  Written before any code exists. Arm D on the chosen architecture beats the better of the two
  demand-climatology rules (B1, B2) by 1 to 4 points of D8's capacity in signed CRPS over all
  synthetic test half-hours, and almost all of the gain sits on reverse-flow half-hours. On the input
  ladder, adding sunshine (C3) and adding the bulk supply point steps (C4) each gain at least 0.3
  points, the BSP gain sits mostly in the worlds with an engine, and adding the calendar and
  temperature inputs (C5) changes the score by under 0.2 points. The quantile head (C6) is worse
  than the two-lobe head by at least 0.2 points. The 48-hour window beats the 24-hour window
  (C11). The harm checks pass, with arm D worse than reading-as-signed by 0.05 to 0.15 points on
  import half-hours. The likeliest miss is the sign calibration bar on solar-only worlds; the
  second likeliest is the engine head's real-record bar. The prediction fails if C1 does not show
  a gain, or if the ladder order differs.
decision_rule: >-
  Every bar and constant below is a named `Final` of `e00729_core.py`, set before any real data are
  read by experiment code. Headline contrast C1 is arm D minus B* (the better of B1 and B2 on the
  validation months) in signed CRPS over all synthetic test half-hours, read at 99.545%
  (Bonferroni, 1 - 0.05 / 11) from a block bootstrap of 7 consecutive base dates. The margin m is
  the larger of 0.25 points and twice the standard deviation over seeds 0 to 2 of arm D's
  validation-month signed CRPS, written once to a file that refuses to be overwritten before any
  test contrast is read. Harm limits: H1, arm D minus P0 on synthetic import half-hours, point
  estimate at most +0.20 points; H2, arm D minus P0 on the real held-out base days, point estimate
  at most +0.20 points, with at most 0.5% of those half-hours at P(X<0) above 0.5. Verdict, first
  row that matches: (0) stopped, no verdict; (1) `invalid`; (2) `failed`; (3) `not supported`, H1
  or H2 fails; (4) `supported`, C1 at or below -m with an upper end below zero, and the sign
  calibration, lobe coverage, and re-fold bars pass; (5) `partly supported`, C1 passes and one of
  those three bars fails; (6) `small gain`, C1's upper end below zero and its point above -m; (7)
  `inconclusive`, C1's interval includes zero and its half-width exceeds m; (8) `not supported`,
  every other case. The section "Verdict table" gives every row in full. The experiment ports
  nothing, so no smallest gain applies beyond m.
benchmark: >-
  P0 (the reading read as signed import); B1 (a demand climatology of D8's base days that sets the
  probability of reverse flow to the probability that demand lies below the reading); B2 (the same
  rule with a regression of base demand on calendar and temperature); experiment 694's M2 tracker on
  the engine-only test segments; arm A (the reading alone) as the no-hint network.
row_set_id: >-
  e00729_synthetic_d8_v1: the 48 half-hours of each UTC day of 8 held-out calendar months of D8's
  base days, each carried by 8 independent synthetic segment draws, scored in the middle of a 48-hour
  window. Real checks read the same months without injection, the metered engine record's 44 days,
  and D8's eras up to 2026-06-30. No valid time on or after 2026-07-01 is read. No forecast is made.
metric: >-
  Signed CRPS in points of D8's capacity (exact for the two-lobe distribution), the 11-quantile
  pinball loss, and the mean absolute error of the median normalised by D8's capacity, each on all,
  import, and reverse-flow half-hours; the reliability term of the Brier score of P(X<0); lobe
  coverage; q05 and q95 coverage; the engine head's mean absolute error and running F1 on the
  metered engine record; false-alarm shares on real days.
---

**We ask whether a neural network can turn a day of D8's unsigned reading into an estimate of D8's
signed power, with honest uncertainty.** D8 is a primary substation (a "Primary"), which steps the
voltage down to 11 kV. D8's meter reports apparent power in megavolt-amperes (MVA), a magnitude with
no sign. A gas engine and a large solar farm now connect below D8, so on some half-hours power may
flow out of D8 towards the grid (reverse flow), and the meter shows that export as a positive
reading. No meter tells us D8's signed power on any real day, so we cannot train on real answers.
We therefore build synthetic days with a known answer: we take D8's real readings from before the
engine was connected, when D8 only imported, subtract a simulated engine and a simulated solar
output, and fold the result into a magnitude the way the meter does. The network learns to undo the
fold. We then test the network on synthetic days that training never saw, on real days before
the engine where the answer is the reading itself, and on 44 real days when the engine's output was
metered.

**This page is the plan, written before any code exists, and the page holds no result.** The design
follows the denoising autoencoder and the other neural networks that [Kelly and Knottenbelt
(2015)](https://doi.org/10.1145/2821650.2821672) used to split a household's total power into its
appliances, and this experiment trains those neural networks on randomised synthetic data in the
spirit of domain randomisation ([Tobin et al. (2017)](https://doi.org/10.1109/IROS.2017.8202133)).

## Terms

**These terms are used across the page.**

- **Signed power X** is D8's real power in megawatts (MW), positive for import and negative for
  reverse flow.
- **The reading R** is what D8's meter reports: R = sqrt(X² + Q²), where Q is the reactive floor,
  the reactive power that keeps the reading above zero when X crosses zero. **Folding** is the
  name for this loss of the sign.
- **A point** is 1% of D8's effective capacity of 10.379 MVA, the normaliser that
  [data-processing](../data-processing.md) already uses for D8.
- **The bulk supply point (BSP)** above D8 is BSP A, the substation that feeds D8 and eight other
  Primaries. Its meter reports signed MW. **The BSP steps** are the half-hour changes of that meter.
- **A base day** is a UTC day of D8's real reading in the pre-engine era. **An injection** is a
  simulated engine output E and solar output S subtracted from the base day, so that X = base − E −
  S. **A segment** is one calendar month of base days carrying one continuous injection draw.
- **A world type** says what a segment holds: no injection, engine only, solar only, or engine and
  solar.
- **The two-lobe distribution** is the network's output for each half-hour: a probability of
  reverse flow, and a Gaussian magnitude for each of the two lobes (one at or above zero, one
  below). **P(X<0)** is the distribution's mass below zero.
- **The eras of D8** are the pre-engine era, the net era (engine only), the solar era (engine and
  the large solar farm), and the fade era, as [experiment
  697](../00697-gas-engine-state-tracker-part-4-running-patterns/result.md) defines them. This page
  uses the era labels only.
- **The metered engine record** is the 44-day record of D8's engine output that experiments 194,
  694, and 697 use. Experiments 694 and 697 call the record "Gas A" and describe it as a separate
  engine. A step test for this plan shows that the record is D8's engine (next section).

## What we know before the experiment

**D8's reading never falls below 1.0 MVA in the pre-engine era, and falls almost to zero once the
engine connects.** A read-only check written for this plan counted 1,354 pre-engine days with at
least 40 of 48 half-hours, and none with a half-hour below 1.0 MVA; the median daily minimum is 5.7
MVA. In the net era, 17% of days have a daily minimum below 1.0 MVA, and the lowest daily minima are
0.03 MVA. In the solar era the share rises to 56% of days, and to 87% to 97% in May to August. The
lowest half-hours sit in a one-half-hour V-shaped dip. That dip is what folding through zero
predicts, but a net flow that approaches zero without crossing it would give the same dip, so the
reading alone cannot show reverse flow.

**The reactive floor Q at D8 is small: the reading reaches 0.01 to 0.03 MVA.** The brief proposed a
floor that keeps the reading well clear of zero. The real reading does not stay clear of zero, so the
simulator draws Q from 0 to 0.3 MVA (0 to 3 points). The upper end covers the 0.14 MVA that
experiment 694 assumed. Treating a pre-engine base reading as signed power introduces an error of at
most Q² / (2R), which is under 0.02 MVA for Q of 0.3 MVA and the pre-engine minimum of about 3 MVA.

**Reverse flow is plausible in the net era as well as in the solar era.** The maintainer's working
idea was that D8 reverses only once both the engine and the solar farm are connected. The check does
not support "only after both", because the reading already reaches its floor in the engine-only era.
The synthetic worlds therefore contain reverse flow from the engine alone, from solar alone, and
from both.

**The metered engine record is D8's engine, and its output steps in the BSP meter but not in D8's
reading.** A step test regressed the half-hour changes of each meter on the changes of the record.
The BSP A meter falls by about 0.76 MW per MW of engine output, far outside 36 placebo shifts. D8's
reading does not move, because over those 44 days D8's reading adds the engine's output back, as
[data-processing](../data-processing.md) records. Two consequences follow for this plan:

- On those 44 days D8's reading is gross demand with no engine in it, so the network can be scored
  against the record only after the record is subtracted from the reading and folded (check K1
  below). The days without that subtraction are a placebo with a real engine in the BSP meter and
  none in D8's reading (check K2).
- The BSP's response is about 0.76 and not 1, so the simulator scales the engine and solar terms
  it adds to the BSP by a coefficient drawn from 0.6 to 1.0. That range is pre-registered here and is
  never refitted.

**The BSP steps are therefore a strong, real hint for the engine.** Three earlier pages agree.
[Experiment
194](../00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/result.md)
placed the engine at D8 from the timing of D8's shortfall and warned that the register's Primary
column does not name the meter that sees a generator. [Experiment
722](../00722-msa-mva-bsp-closure-and-fast-hint/result.md) found a slope of the BSP residual's 2-hour
change on the record's change of -0.85 (95% interval -1.16 to -0.39), close to the expected -1 but
void on its detection limit of 0.54 against a bar of 0.5. Experiment 723's page, on that
experiment's own branch and not yet merged, kept that reading void and replaced the two-placebo
rule with a permutation rule over circular day shifts for future records.
The half-hour step test of this plan reads the same relation with a z of about -7 against its
placebos. The test that found the BSP too noisy was a different one: the level of the BSP residual
at half-hours when D8's reading is near its floor, which cannot show a reverse flow of a few MW.

**The BSP A meter starts in the autumn of 2022, so only the last months of the base carry BSP
steps.** Arm C and the arms after it read the BSP steps where the meter exists and a mask channel
elsewhere. Their BSP contrast (C4) is read on the three test months that the meter covers.

## The synthetic world

### Base days and the split

**The base is D8's cleaned pre-engine reading, cut at the earlier of two boundaries.** The first is
the end of the pre-engine era label. The second is the engine's connection date in the private
constants file, read through `ml-experiments/engine_constants.py`. The page never states which
boundary binds or how many days the cut removes. The metered engine record's 44 days are never base
days. Half-hours that the `substation_zero` rule flags are dropped, and the count dropped per split
is reported. Days within 24 hours of a logged closure at D8 are dropped if the switching logs load,
and the count is reported.

**Whole calendar months are held out, because D8's demand has changed since the pre-engine era.**
The months are fixed here, before any data are read:

| Split | Months | Role |
|---|---|---|
| Test | 2020-10, 2021-02, 2021-06, 2022-04, 2022-08, 2022-11, 2023-02, 2023-05 | Every counted reading; the last three carry BSP steps |
| Validation | 2021-04, 2021-10, 2022-06, 2023-01 | Early stopping, the architecture choice, the choice of B*, and the margin m |
| Training | Every other base month | Fitting only |

No test or validation month touches the fixed lockdown window of `studies.dp_features`, which stays
in training. Training drops every base day within 2 days of a test or validation month, because a
window spans 48 hours.

### The engine simulator

**The engine simulator draws its parameters from wide ranges set from published descriptions, never
from a fitted real value.** The descriptions are those of experiments
[694](../00694-gas-engine-state-tracker-part-1/result.md),
[697](../00697-gas-engine-state-tracker-part-4-running-patterns/result.md), and
[159](../00159-switching-d8-multi-day-drops-recurrence-and-report/result.md): the engine runs more
on weekday evenings, often starts on weekday mornings, runs for spells of a few half-hours, reaches
most of its level in the first running half-hour, and runs on for days at a time in multi-day
spells. Each segment draws one parameter set:

| Parameter | Range |
|---|---|
| Number of units | 1 to 4 |
| Size of one unit | 10 to 50 points; total capped at 120 points |
| Share of days with no run | 0.2 to 0.7 |
| Weekday evening run (start 15:00 to 17:30 local, 2 to 6 hours) | probability 0.3 to 0.9 a day |
| Weekday morning run (start 05:30 to 08:30 local, 1 to 8 hours) | probability 0.1 to 0.6 a day |
| Midday run (start 10:00 to 14:00 local, 1 to 4 hours) | probability 0 to 0.3 a day |
| Weekend and bank-holiday factor on each run probability | 0.1 to 0.8 |
| Multi-day spell (2 to 19 days, at part or full output) | 0 to 2 per 30 days |
| Units running in a run | 1 to all, drawn per run |
| Output of a running unit | 0.6 to 1.0 of its size |
| Ramp to full output | 0 to 2 half-hours |
| Trip of one unit, per running half-hour | probability 0 to 0.02, lasting 1 to 4 half-hours |
| Start and stop times | continuous, so a half-hour that holds a switch carries a partial mean |

The simulator has no price input, so the engine's link with the system price (experiment 697's Q2)
is not reproduced. Runs carry no season, because experiment 697 found no winter-summer difference.

### The solar simulator

**Solar output comes from two sources, so that the sunshine input never equals the driver of the
injected solar.** Each segment draws one source:

- **Source S1 (half of the segments):** the normalised output of one of the metered solar farms A
  to F on the same base day, times a drawn capacity. The farms lie 5 to 25 km from D8, so their sky
  differs from the sky at D8 as a real embedded farm's would.
- **Source S2 (the other half):** the satellite irradiance of the Copernicus Atmosphere Monitoring
  Service (CAMS) at one of the farm sites, not at D8, on the same base day, through a plane-of-array
  solar model with a drawn tilt (10° to 40°), azimuth (120° to 240°), and ratio of direct-current to
  alternating-current rating (1.0 to 1.4, with clipping), times a drawn capacity and a multiplicative
  noise with a half-hour autocorrelation of 0.9 and a standard deviation of 0.05.

A segment with solar draws a capacity log-uniform from 5 to 150 points. Solar output is never
negative and is zero when the sun is below the horizon. The stored CAMS index changes once an hour
([experiment 722](../00722-msa-mva-bsp-closure-and-fast-hint/result.md)), so S2 and every CAMS input
interpolate the hourly index linearly to half-hours.

### Folding, the BSP steps, and the world mix

**Each segment folds its signed truth with its own reactive floor.** The segment draws Q from 0 to
0.3 MVA, and each half-hour multiplies Q by a factor from 0.8 to 1.0. The reading is R = sqrt(X² +
Q²).

**The synthetic BSP meter is the real BSP A meter of the same base day plus the injection's effect.**
It is BSP_real − β (E + S), with β drawn per segment from 0.6 to 1.0, so the BSP noise from the other
eight Primaries and the outside generators is real and has the same structure in training and test.
The network reads the half-hour change of this series.

**The world mix is fixed for training and test alike:** no injection 0.15, engine only 0.30, solar
only 0.20, and engine and solar 0.35. The share with no injection gives the network real pre-engine
days whose answer is the reading itself.

**Training draws come from fixed banks, and test draws are shared by every arm.** Each training
month gets 32 segment draws per network seed, from generator seed 7290 plus the seed. Each
validation and test month gets 8 segment draws from a separate generator seed space (729,000 and
up), built once, hashed, and read by every arm and baseline. Test draws never enter training.

### The realism gate before any fit

**Before any network is fitted, the synthetic readings must look like D8's real readings in the
eras they imitate, judged on folded aggregates only.** The gate compares the training banks' engine-
only worlds with the real net era, and the engine-and-solar worlds with the real solar era:

- the share of days whose daily minimum is below 1.0 MVA (real: 0.17 in the net era and 0.56 in the
  solar era) must lie within 0.5 to 2.0 times the real share;
- the share of half-hours below 1.0 MVA must lie within 0.5 to 2.0 times the real share.

**The gate also checks the synthetic BSP steps against the real relation on the metered engine
record's 44 days.** On the real days, the half-hour steps of the BSP A meter are regressed on the
record's steps, with circular shifts of 3 to 20 whole days as placebos, giving a real slope and z.
On the engine-only synthetic test segments the same regression is run on the synthetic BSP steps
against the injected engine's steps, cut to 44-day stretches. The gate passes when the real slope's
95% interval overlaps the simulated range of -β (-1.0 to -0.6), and the real z lies inside the 5th to
95th percentile of the synthetic stretches' z. The second condition checks that the synthetic BSP
noise around an engine step is neither cleaner nor noisier than the real noise.

**If the gate fails, the simulator's ranges may be re-tuned once, and the re-tune is recorded.** The
gate reads no network output and no target, so a re-tune cannot tune on a score. A second failure
stops the line (verdict row 0). The test banks must also hold at least 2,000 reverse-flow
half-hours outside a dead band of 2 points, and at least 200 in the three BSP test months, or the
line stops at row 0.

## Arms and architectures

### The input ladder

**Each arm adds one input to the arm before it, so each contrast measures one input.** Every
window is 48 hours long, from 12:00 UTC on the day before the scored day to 12:00 UTC on the day
after, and every arm scores the middle 24 hours. The clock is implicit in the window's position in
every arm.

| Arm | Inputs | Output |
|---|---|---|
| A | The reading R and its missing-value mask | Two-lobe |
| B | A plus the CAMS clear-sky index, clear-sky irradiance, and cosine of the solar zenith angle at D8 | Two-lobe |
| C | B plus the synthetic BSP steps and a BSP mask | Two-lobe |
| D | C plus local half-hour (sine and cosine), day of week, bank-holiday flag, and ERA5 2 m temperature at D8's grid cell | Two-lobe |
| D-quantile | D | 11 quantiles, pinball loss |
| D-24 | D on a 24-hour window holding only the scored day | Two-lobe |
| E | D, trained with half of the windows masked over a random 1 to 24 hours of the reading (with a mask channel), scored unmasked | Two-lobe |
| F | D plus the folded reading of the same segment 7 and 14 days earlier, each with a mask | Two-lobe |
| G1 | D | Two-lobe plus a solar-output head |
| G2 | D | Two-lobe plus an engine-output head |

**Arm D carries the coincident observed weather because this is a reconstruction of the past, not
a forecast.** The temperature is the hour's own ERA5 value, and no forecast origin exists, so the
origin-time weather rule of the forecasting experiments does not apply. Wind speed and
precipitation are not stored for D8's grid cell over the base years, so arm D omits them (see the
maintainer's decisions). No arm reads the month or the day of the year, so no arm can memorise the
calendar.

**Arm F's earlier days carry their own engine and solar output, never a copy of the scored day's.**
The segment is simulated as one continuous month, so the day 7 days earlier holds that day's own
engine runs and sky, drawn from the same parameter set. A lag that falls outside the window's own
month is masked, with the same rule in training and test.

### Three neural-network architectures

**Three architectures are fitted on arms A and D, and the best on the validation months carries
every other arm.**

- **Denoising autoencoder**, after Kelly and Knottenbelt: a 1-D convolution of 8 filters (width 4),
  a dense layer to a 128-unit bottleneck, a dense layer back to the window length, a 1-D
  convolution, and the output heads at every half-hour.
- **1-D U-net**, after [Ronneberger et al. (2015)](https://doi.org/10.1007/978-3-319-24574-4_28):
  four levels of 32, 64, 128, and 256 channels, kernels of width 5, group normalisation, GELU
  activations, and skip connections, about 1 million weights.
- **Bidirectional long short-term memory neural network**, after the recurrent network of Kelly and
  Knottenbelt: a 1-D convolution of 16 filters, two bidirectional layers of 128 units, and the
  output heads at every half-hour.

**The architecture rule is fixed here:** the lowest mean over seeds 0 to 2 of arm D's signed CRPS on
the validation months wins, and an architecture within m / 2 of the lowest wins if it has fewer
weights. The test scores of all three architectures on arms A and D are reported, and none of them
enters a counted contrast.

### The output heads

**The two-lobe head outputs, at each half-hour, a logit for the sign and a mean and a spread for each
lobe.** The import lobe is a Gaussian with a positive mean and the reverse-flow lobe a Gaussian with
a negative mean, in the manner of a mixture density network ([Bishop
(1994)](https://publications.aston.ac.uk/id/eprint/373/)). Each spread is 0.5 points plus a softplus,
so no lobe collapses. The loss is the negative log likelihood. The signed CRPS of the mixture is
computed exactly ([Grimit et al. (2006)](https://doi.org/10.1256/qj.05.235)).

**The quantile head outputs the 11 quantiles 0.05, 0.1, 0.2 to 0.9 in steps of 0.1, and 0.95**,
sorted, and is trained by the pinball loss ([Koenker and Bassett
(1978)](https://doi.org/10.2307/1913643)). Its CRPS is computed from the piecewise-linear
distribution through its quantiles.

**The auxiliary heads of G1 and G2 output a Gaussian for the solar output and for the engine output
at each half-hour**, each trained by negative log likelihood with a weight of 0.5 beside the main
loss.

## Baselines

**Four rules that learn nothing from a day's shape set the bar for the networks.**

- **P0** reads the reading as signed import: X = sqrt(max(R² − Q̂², 0)), with Q̂ = 0.1 MVA, as a point
  distribution.
- **B1** uses the fact that the injected output is never negative: if demand D is below the reading,
  X = +R is impossible, so P(X<0) = P(D < R). P(D < R) comes from the month-by-half-hour
  climatology of base demand (`baseline_forecasters.climatology.ClimatologyForecaster`, with local
  month, half-hour, and weekend flag) fitted on the training base days. The two lobes sit at ±R'.
- **B2** is B1 with a linear regression of base demand on local half-hour, day of week, bank
  holiday, and temperature, and a Gaussian residual.
- **M2**, experiment 694's grid filter, runs on the engine-only test segments with 28 days of
  injected history before each month, and its posterior of demand level minus engine output gives
  P(X<0). M2 has no solar term, so it is scored on the engine-only segments only, as a description.

**B\* is the better of B1 and B2 on the validation months**, chosen before any test score is read.

## Scoring

**Every counted reading pairs arms on the same synthetic test half-hours.** The rows are the 48
half-hours of each test base day, for each of the 8 draws of its segment. Rows are split into
import (X ≥ 0) and reverse flow (X < 0), and a dead band of 2 points around zero is reported apart.
Every table also reports the scores by world type, by solar source, and by daylight (09:00 to 13:00
UTC) against the other hours, following experiment 721's advice.

**Intervals come from a block bootstrap of 7 consecutive base dates within each test month.** Each
resample draws whole blocks with replacement, and carries every draw of every date in the block. The
point estimate is the mean over seeds 0 to 2, and each resample uses the seed-averaged losses. Each
counted reading uses 20,000 resamples and generator seed 729. The page reports the Kish effective
number of blocks beside each interval. Below 20 blocks the interval is widened about its point by
the ratio of the Student t quantile (with the Kish number minus 1 degrees of freedom) to the normal
quantile, as in [experiment 721](../00721-msa-mva-to-mw-rung-1-hidden-sign-fits-of-the-network/result.md).
The eight test months hold about 34 blocks, and the three BSP months about 13, so the BSP contrast
is widened.

**Eleven contrasts are planned, and each is read at 99.545% (Bonferroni across the 11).** A
negative difference is a gain. C1 carries the verdict; the other ten each read "gain", "loss", or
"no detectable difference" at a threshold of zero, because nothing here is ported.

| Contrast | Difference | Rows |
|---|---|---|
| C1 (verdict) | D minus B\* | All synthetic test rows |
| C2 | A minus P0 | All |
| C3 | B minus A | All |
| C4 | C minus B | The three BSP test months |
| C5 | D minus C | All |
| C6 | D-quantile minus D, on the 11-quantile pinball loss of both | All |
| C7 | E minus D | All |
| C8 | F minus D | All |
| C9 | G1 minus D | All |
| C10 | G2 minus D | All |
| C11 | D-24 minus D | All |

**The margin m is set from the spread between seeds before any test contrast is read.** It is the
larger of 0.25 points and twice the standard deviation over seeds 0 to 2 of arm D's signed CRPS on
the validation months. 0.25 points is about 26 kW, below which a difference in D8's reconstructed
power is smaller than the reactive floor's effect on the reading. The value is written to
`margin.json`, which refuses to be overwritten.

**Two bars decide between `supported` and `partly supported`.** Sign calibration: the reliability
term of the Brier score of arm D's P(X<0), outside the dead band, is at most 0.01, and no bin of at
least 200 half-hours sits off the diagonal by more than 0.1 (the bars of experiment 724's plan).
Lobe coverage: the 5% to 95% range of the lobe on the true side covers the truth on at least 85% of
half-hours. A lobe that covers more often is not penalised, because the reading fixes the magnitude
closely.

## Real-data checks

**Five checks run on real days, and each reports aggregates only.**

- **R1, false alarms on real pre-engine days.** The real test months, with no injection, have a
  known answer: X equals the reading. H2 reads arm D's signed CRPS minus P0's on these half-hours and
  the share of half-hours with P(X<0) above 0.5.
- **K1, the engine head against the metered engine record.** On the record's 44 days, the record's
  output is subtracted from D8's reading and folded with Q̂, and the real BSP meter, which already
  holds the engine's step, is read unchanged. Arm G2's engine head (the equal mixture of seeds 0 to
  2) is scored against the record. Bars: the head's median has a mean absolute error at most 0.5
  times that of an all-zero engine; and its F1 score for "running" (head median above 10 points,
  against the record's own unit count of at least 1) is at least M2's F1 on the same injected days.
  Arm D's signed CRPS on these days is reported beside B\*'s. The check reads a real engine schedule,
  real demand, and the real BSP response, so it is the one real known answer for the engine.
- **K2, the BSP placebo on the same 44 days.** D8's reading is read unchanged, so it holds no
  engine, while the BSP meter steps with the engine. Bars: at most 0.5% of half-hours have P(X<0)
  above 0.5, and G2's engine head median exceeds 10 points on at most 2% of half-hours.
- **R2, re-fold consistency on the real net, solar, and fade eras.** Arm D's output is folded with
  Q̂ and compared with the reading. Bar: the folded median lies within 2 points of the reading on at
  least 95% of half-hours in each era. The same bar is read on the 44 days of K2.
- **R3, descriptions with no bar.** The share of half-hours with P(X<0) above 0.5 by era and month
  and by era and local hour, set beside the share of days whose daily minimum is below 1.0 MVA; the
  share of predicted sign changes that sit next to a half-hour below 1.0 MVA, set beside the same
  share in the synthetic test; and the level of the matched BSP residual on half-hours with P(X<0)
  above 0.5, a level test that the plan's reverse-flow check found too noisy to show a few MW. The
  BSP steps, which carry the engine clearly, are validated by the realism gate and used by K1.

**K1 and K2 are counted readings that change no verdict word.** Each is reported as passes, fails,
or void, and a later experiment may use the engine head only where K1 passes. K1 is void if fewer
than 200 of the record's half-hours have the engine running. The simulator's ranges were set from
published descriptions of this same record, so K1 is not wholly independent of the simulator, and
the write-up says so.

## Verdict table

**The first row that matches is the verdict, and every row is reachable.**

| Row | Condition | Verdict |
|---|---|---|
| 0 | The line stops before C1 is read: the manager halts it; the timing projection exceeds the budget and the manager declines; the realism gate fails after its one re-tune; the reachability counts are short; or a review gate has not passed when a fit would start | Stopped, no verdict (`inconclusive` in the front matter) |
| 1 | A correctness check fails (the leakage checks below, the holdout guard, the private cut, or a known-answer unit test of the fold, the CRPS, or the bootstrap) | `invalid` |
| 2 | A fit that a counted reading needs raised and was not rerun under the rerun rule | `failed` |
| 3 | H1 or H2 fails | `not supported` |
| 4 | C1 at or below −m with an upper end below zero, and the calibration, lobe, and R2 bars pass | `supported` |
| 5 | C1 as in row 4, and the calibration, lobe, or R2 bar fails | `partly supported` |
| 6 | C1's upper end below zero and its point above −m | `small gain` |
| 7 | C1's interval includes zero and its half-width exceeds m | `inconclusive` |
| 8 | Every other case | `not supported` |

**The harm limits come from experiment 721, where both signed arms gained on reverse-flow rows and
lost 0.4 to 1.2 points on import rows.** H1 and H2 hold arm D to at most 0.20 points worse than P0
on import half-hours, the same limit as experiment 721's harm stop. In D8's real eras, under 5% of
half-hours read below 1.0 MVA, so reverse flow is rare. A reconstruction that lost 0.20 points on
every import half-hour would cancel a gain of 4 points on a 5% share of reverse-flow half-hours.

**A rerun is the same attempt only when a fit raised, or a check that reads no bar failed, the fix
changes no constant, and the Opus reviewer confirms the fix.** `inconclusive` and `not supported`
are never rerun. Every number is labelled interim until the counted reading it belongs to is read.

## Leakage checks

**Each of these checks is an assertion in the code, and a failure is verdict row 1.**

- No base day sits in two splits, and no training day lies within 2 days of a test or validation
  month.
- The test and validation banks come from a seed space that no training bank uses, and their hashes
  match across all arms.
- Solar source S1 reads a farm on the scored base day only, and S2 reads CAMS at a farm site, never
  at D8.
- The synthetic BSP is built the same way in every split, from the same base day's real meter.
- No simulator parameter lies outside the ranges on this page, and no simulator code reads the
  metered engine record or the private constants file.
- The private cut is applied, and the metered engine record's days are not base days.
- No valid time on or after 2026-07-01 is read, and no arm reads the month or the day of the year.
- The architecture, B\*, and m are chosen on validation months only, before any test score exists.

## Compute and gates

**The fits are small, and a timing gate measures their cost before the first fit.** One GPU process
of at most 6 GB runs at a time, with one data worker, on the core set that the manager assigns,
inside `systemd-run --user --scope` with MemoryMax 12G, MemorySwapMax 0, and one BLAS thread. There
are 18 fits in the architecture screen (3 architectures, arms A and D, 3 seeds) and 24 in the ladder
(8 arms, 3 seeds), 42 in all. The gate runs 5 epochs of each architecture on arm D and projects the
GPU-process-hours from the measured time per epoch. At an assumed 3 seconds an epoch, the cap of 200
epochs gives about 7 GPU-process-hours, and early stopping near 70 epochs about 2.5. The budget is 10
GPU-process-hours; a projection above it goes to the manager before any fit. M2's baseline runs on
one CPU core in under an hour.

**The gates run in the maintainer's order:** this plan, a Sonnet plan review, Sonnet code, a fresh
Opus code review before any first run, a real-file smoke test of every loader that reads a file the
synthetic-array tests do not exercise (D8's reading, the BSP meter, the frame of experiment 39, the
ERA5 temperature, the bank holidays, and the metered engine record), then a "fits starting" line to
the manager with the projection and the cores before each family of fits. Every fitted network is
saved in the experiment's data folder.

## Standing practices and earlier advice

**The plan follows the standing practices that apply to a reconstruction.** It uses cleaned data and
reports the dropped half-hours, saves every fitted network, screens the architectures on four
validation months, reports q05 and q95 coverage, and plots example days. Monthly refits, the D15 and
D18 cut, stored-prediction controls, cold winter nights, and the avoidable-curtailment cost do not
apply, because no forecast is made. The experiment is not about load-shift events, so the four
load-shift scenarios do not apply; days near a logged closure are dropped instead.

**The plan follows six items of earlier advice and departs from one.**

- It checks import rows before reverse-flow gains (721), as H1 and H2.
- It splits daylight from the other hours (721).
- It interpolates the hourly CAMS index before using its half-hour changes (722).
- It scores the engine on the real record alone as well as on synthetic data (694), as K1.
- It builds synthetic engine runs with ramps, part load, and units much smaller than the engine's
  (694).
- It does not read the tracker's spell lengths as engine behaviour (697): the simulator's spell
  ranges are wide and not taken from tracker labels.
- It departs from 694's advice to build synthetic schedules from the real record's switching
  structure, because the record is now a test set; the simulator uses published descriptions as
  ranges.

## Decisions that need the maintainer

**One decision belongs to the maintainer; the plan decides everything else and records the
decision above.**

| Decision | Default if no answer |
|---|---|
| Arm D could carry wind speed and precipitation, which need a new download of ERA5 at D8's grid cell for the base years (small, free) | Arm D omits wind speed and precipitation |

## Figures planned

**Every figure is safe to publish: synthetic series in detail, real series as aggregates only.** No
figure shows the metered engine record, an engine-head output on a real day, or a date of the engine.
Real-day panels carry no calendar dates.

1. D8's reading by era: the share of days with a daily minimum below 1.0 MVA by month, and the
   lower tail of the reading (aggregates).
2. A gallery of six synthetic segments drawn with solar source S2: base demand, engine output, solar
   output, signed truth, folded reading, and synthetic BSP steps, on days 1 and 2.
3. The realism gate: synthetic against real shares of near-zero days and half-hours.
4. Signed CRPS of every arm and baseline on all, import, and reverse-flow rows, as staggered dots on
   striped rows (`dot_chart` of experiment 700).
5. The 11 contrasts with their intervals and the margin m.
6. Example test windows: the two-lobe distribution as a band and P(X<0) as a strip against the
   truth, for an import-only, an engine-reversal, a solar-reversal, and a combined day.
7. The reliability diagram of P(X<0).
8. Real eras: the share of half-hours with P(X<0) above 0.5 by local hour and era, beside the share
   below 1.0 MVA.
9. The architecture screen on the validation months.

## Departures from the brief

- **The brief's "about 50 days" of engine metering is 44 days.**
- **The reactive floor is drawn from 0 to 0.3 MVA, not set to keep the reading clear of zero,**
  because D8's real reading reaches 0.01 to 0.03 MVA.
- **The base holds about 3.7 years by the era label, not about 1.5 years.**
- **The engine head's real check subtracts the record from D8's reading (K1),** because D8's
  reading adds the engine back over the record's 44 days; the unchanged days become a placebo (K2).
- **The BSP coefficient β is drawn from 0.6 to 1.0,** because the step test measured about 0.76.
- **Arm D omits wind speed and precipitation,** which are not stored for D8's grid cell.
- **The ladder is cumulative (A, then B, C, D)** and the variants E, F, G1, G2, the quantile head,
  and the 24-hour window are each added to arm D, so that every contrast changes one input.
- **The "pre-engine seasonal-median demand baseline" is B1**, which turns the climatology into a
  sign probability through the rule that injected output is never negative.

## Method

**Architecture:**

- The carrying arm is arm D on the architecture that the validation rule chooses, with the two-lobe
  head; the inputs, layers, and heads of each architecture are in "Three neural-network
  architectures" and "The output heads".
- Inputs are in points of D8's capacity, the CAMS fields are normalised by clear-sky values, and
  temperature is standardised on the training days.
- Every arm reads a 48-hour window and is scored on its middle 24 hours, except D-24.

**Training regime:**

- The training data are synthetic segments built on D8's pre-engine base days in the training
  months, from banks of 32 draws per month per seed, with the world mix, simulators, and fold above.
- An epoch is 8,000 windows; the optimiser is AdamW with a learning rate of 1e-3, weight decay of
  1e-4, batch size 256, and gradient clipping at 1.0; the learning rate halves after 10 epochs with
  no gain; training stops after 25 epochs with no gain in validation signed CRPS, or at 200 epochs.
- Seeds 0, 1, and 2 set the initial weights and the training bank of each fit.

**Measurement:**

- The scored rows, metrics, bootstrap, Kish widening, margin, contrasts, harm limits, real checks,
  and verdict table are in the sections above.
- Every table reports signed CRPS, the 11-quantile pinball loss, and the normalised mean absolute
  error of the median, on all, import, and reverse-flow rows.

## Files

- `ml-experiments/00729-msa-d8-neural-untangler-synthetic-known-answers/e00729_core.py` (planned):
  every named constant and bar on this page, the split months, and the verdict rule.
- `e00729_data.py` (planned): loaders for D8's cleaned reading, the BSP A meter, the CAMS and
  temperature fields of experiment 39's frame, the metered farms' output, the bank holidays, the
  metered engine record, and the private cut.
- `e00729_sim.py` (planned): the engine and solar simulators, the fold, and the synthetic BSP.
- `e00729_banks.py` (planned): builds and hashes the training, validation, and test banks.
- `e00729_nets.py` and `e00729_train.py` (planned): the three architectures, the heads and losses,
  the timing gate, and the fits.
- `e00729_baselines.py` (planned): P0, B1, B2, and a wrapper of experiment 694's M2.
- `e00729_score.py` (planned): CRPS, pinball, the bootstrap, the Kish widening, the margin file, and
  the verdict.
- `e00729_real.py` (planned): R1, K1, K2, R2, and R3.
- `e00729_figures.py` and `run.sh` (planned): the figures, and the launch wrapper.
- `ml-experiments/00729-msa-d8-neural-untangler-synthetic-known-answers/tests/` (planned): tests on
  synthetic arrays; those that need `torch` skip in CI.
- `data/ml-experiments/00729-msa-d8-neural-untangler-synthetic-known-answers/`: written. It will
  hold `banks/`, `networks/`, `predictions/`, `scores/`, `margin.json`, `verdict.json`, the run
  logs, and a `PRIVATE/` folder for anything dated. The two read-only checks written for this plan,
  `reverse_flow_check.md` and `engine_metering_check.md`, are there too and are never published.
- `data/ml-experiments/00039-pv-disaggregation-cams-injection/work/frame.parquet`: read. The CAMS
  index, clear-sky irradiance, zenith, and 2 m temperature at D8 and at the farm sites, from
  [experiment 39](../00039-pv-disaggregation-cams-injection/result.md).
- The metered engine record: read, from the window table of [experiment
  194](../00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/result.md).
  Never published with dates.
- `data/ml-experiments/private/engine_constants.json`: read through
  `ml-experiments/engine_constants.py` for the private cut only. Never committed.
