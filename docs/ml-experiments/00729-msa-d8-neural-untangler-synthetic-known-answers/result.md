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
  Anything else. (1) It stores about 47 fitted neural networks, synthetic training and test banks,
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
  24 hours a two-lobe distribution of D8's signed power. On held-out synthetic days the neural
  network's signed continuous ranked probability score (CRPS) beats the best rule that does not
  learn from the day's shape, and on real days before the engine was connected the neural network
  does not invent reverse flow where there is none.
prediction: >-
  Written before any code exists. Arm D on the chosen architecture beats the better of the two
  demand-climatology rules (B1, B2) by 1 to 4 points of D8's capacity in signed CRPS over all
  synthetic test half-hours, and almost all of the gain sits on reverse-flow half-hours. On the input
  ladder, adding sunshine (C3) and adding the bulk supply point steps (C4) each lower the signed
  CRPS by at least 0.3 points. The C3 gain rests on solar source S1, whose output shares no
  satellite product with the sunshine input. The C4 gain sits mostly in the worlds with an engine.
  Adding the calendar and temperature inputs (C5) changes the signed CRPS by under 0.2 points. The
  quantile head (C6) has an 11-quantile pinball loss at least 0.2 points higher than the two-lobe
  head. The 48-hour window has a signed CRPS lower than the 24-hour window (C11). The harm checks
  pass: arm D's signed CRPS exceeds P0f's by 0.01 to 0.08 points on import half-hours, against a
  limit of 0.12 points. The likeliest miss is the sign calibration bar on solar-only worlds. The
  second likeliest is the lobe coverage bar on half-hours within 3 points of zero. The prediction
  fails if C1 does not show a gain, or if the ladder order differs.
decision_rule: >-
  Every bar and constant below is a named `Final` of `e00729_core.py`, fixed before any fit or score
  exists. Headline contrast C1 is arm D minus B* (the better of B1 and B2 on the validation months)
  in signed CRPS over all synthetic test half-hours, read at 99.583% (Bonferroni, 1 - 0.05 / 12)
  from a bootstrap that resamples blocks of 7 consecutive base dates and the segment draws. The
  margin m is the larger of 0.25 points and twice the standard deviation over seeds 0 to 4 of arm
  D's validation-month signed CRPS, written once to a file that refuses to be overwritten before any
  test contrast is read. Harm limits, derived from the synthetic world's own mix in the section
  "Verdict table": H1, arm D minus P0f (P0's point forecast given the same 0.5-point lobe-width
  floor) on synthetic import half-hours, at most +0.12 points of signed CRPS and at most +0.25
  points of normalised mean absolute error of the median; H2, the same two limits on the real
  held-out base days, with at most 0.5% of those half-hours at P(X<0) above 0.5. H2 and R1 are weak
  checks. Verdict, first row that matches: (0) stopped, no verdict; (1) `invalid`; (2) `failed`; (3)
  `not supported`, H1 or H2 fails; (4) `supported`, C1 at or below -m with an upper end below zero,
  and the sign calibration and lobe coverage bars pass; (5) `partly supported`, C1 passes and one of
  those two bars fails; (6) `small gain`, C1's upper end below zero and its point above -m; (7)
  `inconclusive`, C1's interval includes zero and its half-width is at least m; (8) `not
  supported`, every other case. The section "Verdict table" gives every row in full. The experiment
  ports nothing, so no smallest gain applies beyond m.
benchmark: >-
  P0 (the reading read as signed import) and P0f (P0's point forecast given the same lobe-width
  floor as the neural networks, the reference for the harm checks); B1 (a demand climatology of D8's
  base days that sets the probability of reverse flow to the world's injected share times the
  probability that demand lies below the reading); B2 (the same rule with a regression of base demand
  on calendar and temperature);
  experiment 694's M2 tracker on the engine-only test segments; arm A (the reading alone) as the
  no-hint neural network.
row_set_id: >-
  e00729_synthetic_d8_v1: the 48 half-hours of each UTC day of 8 held-out calendar months of D8's
  base days, each carried by 8 independent synthetic segment draws, scored in the middle of a 48-hour
  window. Real checks read the same months without injection, the metered engine record's days, and
  D8's eras up to 2026-06-30. No valid time on or after 2026-07-01 is read. No forecast is made.
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
output, and fold the result into a magnitude the way the meter does. The neural network learns to
undo the fold. We then test the neural network on synthetic days that training never saw, on real
days before the engine where the answer is the reading itself, and on the real days when the
engine's output was metered.

**This page is the plan, written before any code exists, and the page holds no result.** The design
follows the denoising autoencoder and the other neural networks that [Kelly and Knottenbelt
(2015)](https://doi.org/10.1145/2821650.2821672) used to split a household's total power into its
appliances, and this experiment trains those neural networks on randomised synthetic data in the
spirit of domain randomisation ([Tobin et al. (2017)](https://doi.org/10.1109/IROS.2017.8202133)).

## Terms

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
- **The two-lobe distribution** is the neural network's output for each half-hour: a probability of
  reverse flow, and a Gaussian magnitude for each of the two lobes (one at or above zero, one
  below). **P(X<0)** is the distribution's mass below zero.
- **The eras of D8** are the pre-engine era, the net era (engine only), the solar era (engine and
  the large solar farm), and the fade era, as [experiment
  697](../00697-gas-engine-state-tracker-part-4-running-patterns/result.md) defines them. This page
  uses the era labels only.
- **The metered engine record** is the record of D8's engine output that experiments 194, 694, and
  697 use. Its length is on the [experiment
  194](../00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/result.md)
  page. Experiments 694 and 697 call the record "Gas A" and describe it as a separate engine. A step
  test for this plan shows that the record is D8's engine (next section).

## What we know before the experiment

**D8's reading never falls below 1.0 MVA in the pre-engine era, and falls almost to zero once the
engine connects.** A read-only check written for this plan counted the pre-engine days with at least
40 of 48 half-hours, and none has a half-hour below 1.0 MVA; the median daily minimum is 5.7 MVA. In
the net era, 17% of days have a daily minimum below 1.0 MVA, and the lowest daily minima are 0.03
MVA. In the solar era the share rises to 56% of days, and to 87% to 97% in May to August. The lowest
half-hours sit in a one-half-hour V-shaped dip. That dip is what folding through zero predicts, but a
net flow that approaches zero without crossing it would give the same dip, so the reading alone
cannot show reverse flow.

**The reactive floor Q at D8 is small: the reading reaches 0.01 to 0.03 MVA.** The real reading does
not stay clear of zero, so the simulator draws Q from 0 to 0.3 MVA (0 to 3 points). The upper end
covers the 0.14 MVA that experiment 694 assumed. Treating a pre-engine base reading as signed power
introduces an error of at most Q² / (2R), which is under 0.02 MVA for Q of 0.3 MVA and the
pre-engine minimum of about 3 MVA.

**Reverse flow is plausible in the net era as well as in the solar era.** The maintainer's working
idea was that D8 reverses only once both the engine and the solar farm are connected. The check does
not support "only after both", because the reading already reaches its floor in the engine-only era.
The synthetic worlds therefore contain reverse flow from the engine alone, from solar alone, and
from both.

**The metered engine record is D8's engine, and its output steps in the BSP meter but not in D8's
reading.** A step test regressed the half-hour changes of each meter on the changes of the record.
The BSP A meter falls by less than 1 MW per MW of engine output, far outside 36 placebo shifts. D8's
reading does not move, because on the record's days D8's reading is gross demand with no engine in
it, as [data-processing](../data-processing.md) records. Two consequences follow for this plan:

- On the record's days the neural network can be scored against the record only after the record is
  subtracted from D8's reading and folded (check K1 below). The same days without that subtraction
  are a placebo with a real engine in the BSP meter and none in D8's reading (check K2).
- The BSP's response is below 1, so the simulator scales the engine and solar terms it adds to the
  BSP by a coefficient β drawn from 0.6 to 1.0. That range is pre-registered here and is never
  refitted. The range comes from the same record that the realism gate and check K1 read, so the
  three are not independent (see "The realism gate before any fit").

**The BSP steps are therefore a strong, real hint for the engine.** Three earlier pages agree.
[Experiment
194](../00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/result.md)
placed the engine at D8 from the timing of D8's shortfall and warned that the register's Primary
column does not name the meter that sees a generator. [Experiment
722](../00722-msa-mva-bsp-closure-and-fast-hint/result.md) found a slope of the BSP residual's 2-hour
change on the record's change of -0.85 (95% interval -1.16 to -0.39), close to the expected -1 but
void on its detection limit of 0.54 against a bar of 0.5. Experiment 723's plan keeps that reading
void and replaces the two-placebo rule with a permutation rule over circular day shifts for future
records. The half-hour step test of this plan reads the same relation and finds it far outside its
placebos. The test that found the BSP too noisy was a different one: the level of the BSP residual
at half-hours when D8's reading is near its floor, which cannot show a reverse flow of a few MW.

**Only the last few months of the base carry BSP steps.** The BSP A meter covers a short span of the
base. Arms C and H and every arm after them read the BSP steps where the meter exists and a mask
channel elsewhere. A few months of base days carry real BSP steps: two of them are test months,
one is a validation month, and the rest are training months. The BSP contrasts (C4 and C12) are
read on the two test months that the meter covers.

## The synthetic world

### Base days and the split

**The base is D8's cleaned pre-engine reading, cut at the earlier of two boundaries.** The first is
the end of the pre-engine era label. The second is the engine's connection date in the private
constants file, read through `ml-experiments/engine_constants.py`. The page never states which
boundary binds or how many days the cut removes. Half-hours that the `substation_zero` rule flags
are dropped, and the count dropped per split is reported. Days within 24 hours of a logged closure at
D8 are dropped if the switching logs load, and the count is reported.

**Whole calendar months are held out, because D8's demand has changed since the pre-engine era.** A
rule chooses the months, `e00729_core.py` applies the rule to the private base calendar, and neither
this page nor the repository's code lists a month. The explicit list is in `PRIVATE/splits.md` of the
experiment's data folder. The rule has four steps:

- A base month is a calendar month with at least 20 base days.
- A base month that overlaps the fixed lockdown window of [data-processing](../data-processing.md)
  is always a training month, and never counts when the next step numbers the months.
- The remaining base months fall in two groups, numbered from 1 in time order: the months in which
  the BSP A meter has data on at least 20 base days, and all other months.
- In both groups, every third month from the second is a test month. A validation month is the last
  month of the BSP group, and every sixth month from the first in the other group. Every other
  base month is a training month.

The rule gives 8 test months, 2 of them in the BSP group, and 4 validation months, interleaved with
training months in time order, and every season holds at least one test month.

| Split | Rule | Role |
|---|---|---|
| Test | Every third month from the second in each group | Every counted reading |
| Validation | The last month (BSP group); every sixth month from the first (other group) | Early stopping, the architecture choice, the choice of B\*, and the margin m |
| Training | Every other base month, and every month in the lockdown window | Fitting only |

Training drops every base day within 2 days of a test or validation month, because a window spans 48
hours. A validation month that borders a test month also drops its days within 2 days of the test
month.

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

**Source S1 needs metered farm data, and a base day without it takes source S2 for that day.** A
read-only check counted the base days on which a farm has at least 40 of 48 half-hours of data. At
least one of the six farms has data on about 80% of base days, and no farm has data on about 19% to
25% of base days in each split. A segment that draws S1 uses one farm at a time: on a base day on
which the drawn farm lacks data, the segment takes another farm that has data, and on a base day on
which no farm has data, the segment takes source S2 at the segment's capacity for that day. The
code flags each such day, leaves flagged days out of the S1 rows of every by-source table, and
reports the count of flagged days per split. About 1 in 5 of the days in S1 segments is flagged.

**The sunshine input at D8 and the injected solar must not share a CAMS cell.** Arm B reads CAMS at
D8, and source S2 builds the injected solar from CAMS at a farm site, so a gain of arm B on S2 could
come from the shared satellite product and not from the sky that D8 sees. Three rules limit that
reading:

- The code asserts that the CAMS series at the injection's farm site differs from the CAMS series at
  D8. A read-only check found that no farm's daytime series equals D8's: the median gap is 4.5% to
  11% of D8's value, depending on the farm.
- The write-up reports arm B's gain (C3) separately for S1 and S2. The C3 prediction rests on S1,
  because the metered farm output shares no satellite processing with the sunshine input, so the S1
  gain is the evidence that the sunshine input helps. The S2 gain is an upper bound, because the
  injection and the input both come from one satellite product.
- Both sources are reported in every by-source table, with the S1-fallback days left out of S1.

### Folding, the BSP steps, and the world mix

**Each segment folds its signed truth with its own reactive floor.** The segment draws Q from 0 to
0.3 MVA, and each half-hour multiplies Q by a factor from 0.8 to 1.0. The reading is R = sqrt(X² +
Q²).

**The synthetic BSP meter is the real BSP A meter of the same base day plus the injection's effect.**
It is BSP_real − β (E + S), with β drawn per segment from 0.6 to 1.0, so the BSP noise from the other
eight Primaries and the outside generators is real and has the same structure in training and test.
The neural network reads the half-hour change of this series.

**The test world mix is fixed:** no injection 0.15, engine only 0.30, solar only 0.20, and engine
and solar 0.35, so 85% of test segments carry an injection. The share with no injection gives the
neural network real pre-engine days whose answer is the reading itself.

**The training mix is randomised, so that C1 does not measure knowledge of the fixed 85%.** Each bank
of 32 segment draws (one month, one seed) draws its own share of no-injection segments from a uniform
range of 0.05 to 0.60, which makes the injected share of a bank range from 0.40 to 0.95. The other
three world types share the remainder in the proportions 0.30 : 0.20 : 0.35. The test mix of 0.15
and the shifted mix below (0.50) both lie inside the range. The baselines B1 and B2 receive the
test world's injected share as a prior (see "Baselines"), so the neural network has no
advantage in knowing the mix.

**Training draws come from fixed banks, and test draws are shared by every arm.** Each training
month gets 32 segment draws per seed, from generator seed 7290 plus the seed. Each
validation and test month gets 8 segment draws from a separate generator seed space (729,000 and
up), built once, hashed, and read by every arm and baseline. Test draws never enter training.

**Every arm draws half of each epoch's windows from the training months that carry real BSP steps,
and half from all other training months.** A few training months carry real BSP steps, and the BSP
channel is empty everywhere else, so without this sampler arms C and later would see a filled BSP
channel on a small share of windows. Every arm uses the same sampler, including arms A and B, which
ignore the BSP channel, so that no contrast changes the training mix.

### The realism gate before any fit

**Before any neural network is fitted, the synthetic readings must look like D8's real readings in
the eras they imitate, judged on folded aggregates only.** The gate compares the training banks'
engine-only worlds with the real net era, and the engine-and-solar worlds with the real solar era:

- the share of days whose daily minimum is below 1.0 MVA (real: 0.17 in the net era and 0.56 in the
  solar era) must lie within 0.5 to 2.0 times the real share;
- the share of half-hours below 1.0 MVA must lie within 0.5 to 2.0 times the real share.

**The gate also checks the synthetic BSP steps against the real relation on the metered engine
record, with steps matched in size.** On the real days, the half-hour steps of the BSP A meter are
regressed on the record's steps, with circular shifts of 3 to 20 whole days as placebos. The
regression gives a real slope, the slope's standard error, and the residual standard deviation. The
same regression runs on the engine-only synthetic test segments, cut into stretches as long as the
record. A regression on larger steps has a smaller standard error, so the two sides are compared at
equal step sizes: both sides are split into five bins of the absolute size of the record's step, the
bins are the quintiles of the real record, and each synthetic stretch is weighted so that every bin
holds the real record's share of half-hours. The gate passes when the real slope's 95% interval
overlaps the simulated range of -β (-1.0 to -0.6), and the real standard error and the real residual
standard deviation each lie inside the 5th to 95th percentile of the size-matched synthetic
stretches. The second condition checks that the synthetic BSP noise around an engine step is neither
cleaner nor noisier than the real noise.

**The gate is not independent evidence about the BSP simulator.** The range of β was set from the
same record that the gate tests, and check K1 reads the same record. A pass therefore shows that the
simulator's BSP noise has the right size around an engine step, and does not validate β. The record
holds an engine and no solar farm, so β for solar has no real validation at all. The write-up reports
the BSP gain (C4 and C12) separately for engine-only worlds, solar-only worlds, and engine-and-solar
worlds, so that a reader can see how much of the gain rests on the solar value of β.

**If the gate fails, the simulator's ranges may be re-tuned once, and the re-tune is recorded.** The
gate reads no neural-network output and no target, so a re-tune cannot tune on a score. A second
failure stops the line (verdict row 0). The test banks must also hold at least 2,000 reverse-flow
half-hours outside a dead band of 2 points, and at least 200 in the two BSP test months, or the line
stops at row 0. The plan's simulation expects about 4,400 and about 1,100, so the counts have a
margin of about 2 and 5 (section "Verdict table").

## Arms and architectures

### The input ladder

**The ladder adds one input at a time, from arm A to arm D, and each other arm changes arm D or arm A
in one way.** Every window is 48 hours long, from 12:00 UTC on the day before the scored day to
12:00 UTC on the day after, and every arm scores the middle 24 hours. The clock is implicit in the
window's position in every arm.

| Arm | Inputs | Output |
|---|---|---|
| A | The reading R and its missing-value mask | Two-lobe |
| B | A plus the CAMS clear-sky index, clear-sky irradiance, and cosine of the solar zenith angle at D8 | Two-lobe |
| C | B plus the synthetic BSP steps and a BSP mask | Two-lobe |
| D | C plus local half-hour (sine and cosine), day of week, bank-holiday flag, and ERA5 2 m temperature at D8's grid cell | Two-lobe |
| H | A plus the synthetic BSP steps and a BSP mask, with no CAMS fields | Two-lobe |
| D-quantile | D | 11 quantiles, pinball loss |
| D-24 | D on a 24-hour window holding only the scored day | Two-lobe |
| E | D, trained with half of the windows masked over a random 1 to 24 hours of the reading (with a mask channel), scored unmasked | Two-lobe |
| F | D plus the folded reading of the same segment 7 and 14 days earlier, each with a mask | Two-lobe |
| G1 | D | Two-lobe plus a solar-output head |
| G2 | D | Two-lobe plus an engine-output head |

**The ladder is cumulative, so the BSP gain depends on the order of the inputs.** In arm C the BSP
steps join an input set that already holds sunshine, and the steps may carry less information once
sunshine is known. Arm H adds the BSP steps to arm A with no sunshine, so that C12 (H minus A) is the
BSP gain in the other order. A BSP gain that shows in C12 and not in C4 means that sunshine already
carries the information. Neither order is a result about which input matters most.

**Arm D carries the coincident observed weather because this is a reconstruction of the past, not
a forecast.** The temperature is the hour's own ERA5 value, and no forecast origin exists, so the
origin-time weather rule of the forecasting experiments does not apply. Wind speed and
precipitation are not stored for D8's grid cell over the base years, so arm D omits them (see the
maintainer's decisions). No arm reads the month or the day of the year, so no arm can memorise the
calendar.

**Arm F's earlier days carry their own engine and solar output, never a copy of the scored day's.**
The segment is simulated as one continuous month, so the day 7 days earlier holds that day's own
engine runs and sky, drawn from the same parameter set. A lag that falls outside the window's own
month is masked, with the same rule in training and test. The lags share the segment's parameter
set, which makes them more informative than the lags of a real engine whose behaviour drifts, so C8
is an upper bound on the gain from lags.

### Three neural-network architectures

**Three architectures are fitted on arms A and D, and the best on the validation months carries
every other arm.**

- **Denoising autoencoder**, after Kelly and Knottenbelt: a 1-D convolution of 8 filters (width 4),
  a dense layer to a 128-unit bottleneck, a dense layer back to the window length, a 1-D
  convolution, and the output heads at every half-hour.
- **1-D U-net**, after [Ronneberger et al. (2015)](https://doi.org/10.1007/978-3-319-24574-4_28):
  four levels of 32, 64, 128, and 256 channels, kernels of width 5, group normalisation, GELU
  activations, and skip connections, about 1 million weights.
- **Bidirectional long short-term memory neural network**, after the recurrent neural network of
  Kelly and Knottenbelt: a 1-D convolution of 16 filters, two bidirectional layers of 128 units, and
  the output heads at every half-hour.

**The architecture rule is fixed here:** the lowest mean over seeds 0 to 2 of arm D's signed CRPS on
the validation months wins, and an architecture within 0.125 points (half of the margin's floor) of
the lowest wins if it has fewer weights. The test scores of all three architectures on arms A and D
are reported, and none of them enters a counted contrast.

**The validation months do four jobs, and their scores are optimistic for the fitted networks.** The
months pick the architecture, stop each fit early, choose B\*, and set the margin m. Early stopping
picks the epoch that scores best on the same months, so a validation score is lower than the same
fit's score on months it never touched. The test months carry every counted reading for this reason.

### The output heads

**The two-lobe head outputs, at each half-hour, a logit for the sign and a mean and a spread for each
lobe.** The import lobe is a Gaussian with a positive mean and the reverse-flow lobe a Gaussian with
a negative mean, in the manner of a mixture density network ([Bishop
(1994)](https://publications.aston.ac.uk/id/eprint/373/)). Each spread is 0.5 points plus a softplus,
so no lobe collapses. The loss is the negative log likelihood. The signed CRPS of the mixture is
computed exactly ([Grimit et al. (2006)](https://doi.org/10.1256/qj.05.235)).

**The 0.5-point floor sets a lower limit on every score.** A Gaussian lobe of spread 0.5 points whose
mean is exact scores a CRPS of 0.5 × (2 × 0.399 − 0.564) = 0.117 points. No arm can score below about
0.12 points on a half-hour where its lobe is right, and the harm checks therefore use P0f (see
"Baselines") and not P0.

**The quantile head outputs the 11 quantiles 0.05, 0.1, 0.2 to 0.9 in steps of 0.1, and 0.95**,
sorted, and is trained by the pinball loss ([Koenker and Bassett
(1978)](https://doi.org/10.2307/1913643)). Its CRPS is computed from the piecewise-linear
distribution through its quantiles.

**The auxiliary heads of G1 and G2 output a Gaussian for the solar output and for the engine output
at each half-hour**, each trained by negative log likelihood with a weight of 0.5 beside the main
loss.

## Baselines

**Five rules that learn nothing from a day's shape set the bar for the neural networks.**

- **P0** reads the reading as signed import: X = sqrt(max(R² − Q̂², 0)), with Q̂ = 0.1 MVA, as a point
  distribution.
- **P0f** is P0's point X̂ given the same lobe-width floor as the neural networks: a Gaussian
  centred on X̂ with a spread of 0.5 points. P0f is the reference for the harm checks H1 and H2 and
  for contrast C2, so that the floor adds the same CRPS to the neural network and to its reference.
  P0 itself is reported in every table.
- **B1** uses the fact that the injected output is never negative: if demand D is below the reading,
  X = +R is impossible, so P(X<0) = π × P(D < R), where π is the share of segments that carry an
  injection in the world being scored (0.85 in the test banks). The prior π is the world-mix
  knowledge that a fixed-mix neural network would absorb in training, and giving it to B1 and B2
  keeps C1 from rewarding that knowledge. P(D < R) comes from the month-by-half-hour
  climatology of base demand (`baseline_forecasters.climatology.ClimatologyForecaster`, with local
  month, half-hour, and weekend flag) fitted on the training base days. The two lobes sit at ±R'.
- **B2** is B1 with a linear regression of base demand on local half-hour, day of week, bank
  holiday, and temperature, and a Gaussian residual.
- **M2**, experiment 694's grid filter, runs on the engine-only test segments with 28 days of
  injected history before each month, and its posterior of demand level minus engine output gives
  P(X<0). M2 has no solar term, so it is scored on the engine-only segments only, as a description.

**B\* is the better of B1 and B2 on the validation months**, chosen before any test score is read.
Both baselines use the prior π of the bank being scored.

## Scoring

**Every counted reading pairs arms on the same synthetic test half-hours.** The rows are the 48
half-hours of each test base day, for each of the 8 draws of its segment. Rows are split into
import (X ≥ 0) and reverse flow (X < 0), and a dead band of 2 points around zero is reported apart.
Every table also reports the scores by world type, by solar source (with the S1-fallback days left
out of S1), and by daylight (09:00 to 13:00 UTC) against the other hours, following experiment
721's advice.

**Intervals come from a two-way bootstrap that resamples date blocks and segment draws.** Within each
test month, a resample draws whole blocks of 7 consecutive base dates with replacement and, drawn
independently, whole segments (the 8 draws of that month) with replacement. A resampled cell pairs
one drawn block with one drawn segment and carries all 48 half-hours of each date in the block. The
point estimate is the mean over seeds 0 to 2, and each resample uses the seed-averaged losses. Each
counted reading uses 20,000 resamples and generator seed 729. Drawing the two factors independently
makes the interval count the variance between segment draws as well as the variance between dates,
and counts their interaction twice, so the interval is slightly conservative.

**The Kish effective number of this design is the product of the Kish numbers of its two factors.**
A resampled cell has weight a × b, where a counts how often its block was drawn and b how often its
segment was drawn. The Kish number is (Σw)² / Σw², and for product weights (Σw)² / Σw² equals
((Σa)² / Σa²) × ((Σb)² / Σb²). The page reports the two factors beside each interval. Below 20 of
either factor, the interval is widened about its point by the ratio of the Student t quantile (with
that factor's Kish number minus 1 degrees of freedom, at the contrast's Bonferroni level) to the
normal quantile, using the smaller factor, as in [experiment
721](../00721-msa-mva-to-mw-rung-1-hidden-sign-fits-of-the-network/result.md). The eight test months
hold about 33 blocks and 64 segments, so no interval on all test months is widened. The two BSP test
months hold about 8 blocks and 16 segments, so the BSP contrasts are widened by about 1.46 (the
ratio of the t quantile with 7 degrees of freedom to the normal quantile, 2.865).

**Twelve contrasts are planned, and each is read at 99.583% (Bonferroni across the 12).** A
negative difference is a gain. C1 carries the verdict. The other eleven each read "gain", "loss", or
"no detectable difference" at a threshold of zero, because nothing here is ported. C6 is read on the
11-quantile pinball loss, defined as twice the mean of the pinball losses over the 11 levels (in
points, comparable with CRPS), for the two-lobe head's quantiles and for the quantile head. Every
other contrast is read on signed CRPS.

| Contrast | Difference | Rows |
|---|---|---|
| C1 (verdict) | D minus B\* | All synthetic test rows |
| C2 | A minus P0f | All |
| C3 | B minus A | All, and by solar source |
| C4 | C minus B | The two BSP test months, and by world type |
| C5 | D minus C | All |
| C6 | D-quantile minus D, on the 11-quantile pinball loss of both | All |
| C7 | E minus D | All |
| C8 | F minus D | All |
| C9 | G1 minus D | All |
| C10 | G2 minus D | All |
| C11 | D-24 minus D | All |
| C12 | H minus A | The two BSP test months, and by world type |

**The margin m is set from the spread between seeds before any test contrast is read.** It is the
larger of 0.25 points and twice the standard deviation over seeds 0 to 4 of arm D's signed CRPS on
the validation months. Seeds 3 and 4 are fitted for arm D alone. With 3 seeds, a sample standard
deviation falls below half the true value with probability 0.22 (2 degrees of freedom); with 5
seeds, with probability 0.09 (4 degrees of freedom). The floor of 0.25 points absorbs a true seed
standard deviation of up to 0.125 points, and the standard deviation raises m only where the seeds
disagree by more. The validation months are optimistic (see "Three neural-network architectures"),
so the standard deviation measured there understates the spread on test months, and the floor
matters more than the standard deviation. 0.25 points is about 26 kW, below which a difference in
D8's reconstructed power is smaller than the reactive floor's effect on the reading. The value is
written to `margin.json`, which refuses to be overwritten.

**A shifted-mix read-out tests whether arm D has learned the training mix.** A second test bank is
built once from a separate seed space with 50% of its segments carrying no injection (an injected
share of 0.50, against 0.85 in the test banks), with the same months and draws. The read-out reports,
for arm D and for B\* with the prior set to 0.50, the signed CRPS, the reliability term and diagram
of P(X<0), and the share of half-hours with P(X<0) above 0.5. A neural network that had memorised the
injected share would over-predict reverse flow in this bank, and its reliability term would rise
well above its value on the test banks. The read-out enters no verdict row. The harm limit does not
change: it is derived from the test mix (section "Verdict table"), and the shifted bank has a
lower reverse-flow share, so it only makes harm easier to detect.

**Two bars decide between `supported` and `partly supported`.**

- **Sign calibration:** the reliability term of the Brier score of arm D's P(X<0), outside the dead
  band, is at most 0.01, and no bin of at least 200 half-hours sits off the diagonal by more than
  0.1 (the bars of experiment 724's plan). The reliability term is the mean squared gap between a
  forecast probability and the observed frequency, so 0.01 is a root-mean-square gap of 0.1, the same
  size as the bin bar. A calibrated forecast's term is about 0.0013 or less from sampling noise alone
  (10 bins, probabilities at worst 0.5, 64 segments of 30 days counted as independent), so the bar is
  about 8 times the noise.
- **Lobe coverage:** the 5% to 95% range of the lobe on the true side covers the truth on at least
  85% of half-hours. The floor makes coverage easy: in the plan's simulation, P0's magnitude with the
  floor alone covers the true magnitude on 99.8% of half-hours, and on 97.7% of reverse-flow
  half-hours. The 85% bar therefore detects a collapsed or biased lobe and says nothing about
  sharpness, and the write-up reports the mean width of the range beside the coverage. A lobe that
  covers more often is not penalised, because the reading fixes the magnitude closely.

## Real-data checks

**Five checks run on real days, and each reports aggregates only.**

- **R1, false alarms on real pre-engine days, a weak check.** The real test months, with no
  injection, have a known answer: X equals the reading. H2 reads arm D's signed CRPS minus P0f's on
  these half-hours and the share of half-hours with P(X<0) above 0.5. These days never read below
  1.0 MVA, so the neural network meets no reading near the fold. A pass shows that the neural network
  does not invent reverse flow far from zero, and says nothing about readings near zero. R1 and H2
  are weak checks.
- **K1, the engine head against the metered engine record.** On the record's days, the record's
  output is subtracted from D8's reading and folded with Q̂, and the real BSP meter, which already
  holds the engine's step, is read unchanged. Arm G2's engine head (the equal mixture of seeds 0 to
  2) is scored against the record. Bars: the head's median has a mean absolute error at most 0.5
  times that of an all-zero engine; and its F1 score for "running" (head median above 10 points,
  against the record's own running label) is at least M2's F1 on the same injected days. Arm D's
  signed CRPS on these days is reported beside B\*'s. The check reads a real engine schedule, real
  demand, and the real BSP response, so it is the one real known answer for the engine.
- **K2, the BSP placebo on the same days.** D8's reading is read unchanged, so it holds no
  engine, while the BSP meter steps with the engine. Bars: at most 0.5% of half-hours have P(X<0)
  above 0.5, and G2's engine head median exceeds 10 points on at most 2% of half-hours.
- **R2, re-fold consistency on the real net, solar, and fade eras, a report with no bar.** Arm D's
  output is folded with Q̂ and compared with the reading, and the report gives the share of
  half-hours on which the folded median lies within 2 points of the reading in each era, and on the
  K2 days. The neural network is trained to re-fold, so R2 is nearly guaranteed and enters no verdict
  row.
- **R3, descriptions with no bar.** The share of half-hours with P(X<0) above 0.5 by era and month
  and by era and local hour, set beside the share of days whose daily minimum is below 1.0 MVA; the
  share of predicted sign changes that sit next to a half-hour below 1.0 MVA, set beside the same
  share in the synthetic test; and the level of the matched BSP residual on half-hours with P(X<0)
  above 0.5, a level test that the plan's reverse-flow check found too noisy to show a few MW. The
  BSP steps, which carry the engine clearly, are validated by the realism gate and used by K1.

**K1 and K2 are counted readings that change no verdict word.** Each is reported as passes, fails,
or void. K1 is void if fewer than 200 of the record's half-hours have the engine running. The F1 bar
has little power: 200 running half-hours in spells of about 8 half-hours make about 25 independent
spells, so a proportion such as recall has a standard error near 0.10, and the bar separates two F1
scores only where they differ by about 0.2 or more. The simulator's ranges were set from published
descriptions of this same record, so K1 is not wholly independent of the simulator, and the write-up
says so.

## Verdict table

**The first row that matches is the verdict, and every row is reachable.** Row 3 sits before the gain
rows on purpose: a reconstruction that harms the import half-hours is not usable, however well it
scores on reverse flow.

| Row | Condition | Verdict |
|---|---|---|
| 0 | The line stops before C1 is read: the manager halts it; the timing projection exceeds the budget and the manager declines; the realism gate fails after its one re-tune; the reachability counts are short; or a review gate has not passed when a fit would start | Stopped, no verdict (`inconclusive` in the front matter) |
| 1 | A correctness check fails (the leakage checks below, the holdout guard, the private cut, or a known-answer unit test of the fold, the CRPS, or the bootstrap) | `invalid` |
| 2 | A fit that a counted reading needs raised and was not rerun under the rerun rule | `failed` |
| 3 | H1 or H2 fails | `not supported` |
| 4 | C1 at or below −m with an upper end below zero, and the calibration and lobe bars pass | `supported` |
| 5 | C1 as in row 4, and the calibration or lobe bar fails | `partly supported` |
| 6 | C1's upper end below zero and its point above −m | `small gain` |
| 7 | C1's interval includes zero and its half-width is at least m | `inconclusive` |
| 8 | Every other case | `not supported` |

**The harm checks compare arm D with P0f on import half-hours, because the lobe-width floor
penalises every arm on every row.** A Gaussian lobe with a spread of 0.5 points scores a CRPS of
0.117 points even when its mean is exact, whereas P0, a point distribution, scores only its absolute
error, near 0.02 points on import half-hours in the plan's simulation. A limit set against P0 would
fail every arm by about 0.1 points before the neural network had learned anything. P0f gives P0's
point forecast the same floor and scores 0.119 points on import half-hours in the simulation, so
arm D's excess over P0f measures only what the neural network adds on import half-hours: probability
placed on reverse flow, and a lobe centred away from P0's point.

**The limit is half of the margin's floor, spread over the import share of the synthetic world.** The
smallest gain that C1 counts as support is m, at least 0.25 points over all rows. Harm of h points
on every import half-hour lowers the overall gain by h × (1 − s), where s is the share of
reverse-flow half-hours. The limit lets the harm consume at most half of the smallest countable gain:
h ≤ 0.125 / (1 − s). In the synthetic world's own mix (no injection 0.15, engine only 0.30, solar
only 0.20, engine and solar 0.35), `e00729_plan_calcs.py` draws 3,000 segments from the parameter
tables of this page on D8's base readings, and s = 0.051 (0.059 in engine-only worlds, 0.010 in
solar-only worlds, 0.091 in engine-and-solar worlds). The limit is then 0.125 / 0.949 = 0.132 points.
On the real held-out base days every half-hour is import, so s = 0 and the limit is 0.125 points.
One limit serves both checks, rounded down to 0.12 points.

**The limit is fixed before any fit and before any score, and it does not depend on the simulation.**
The training mix is randomised, and the limit is derived on the fixed test mix, which scores the
harm checks. The one data input is D8's base readings, which the calculation script read to estimate
s, and no score exists. The simulation is approximate: solar output is a clear-sky shape times a
random cloud factor, and engine runs follow the parameter table without trips or ramps. A share s
between 0.03 and 0.08 would move the limit only between 0.129 and 0.136 points, so the rounded 0.12
holds across that range. The same simulation gives 4.7% of half-hours as reverse flow outside the
dead band of 2 points, which is about 4,400 of the 92,000 half-hours of the test banks (8 months ×
30 days × 8 draws × 48 half-hours) and about 1,100 of the 23,000 in the two BSP test months.

**The second harm limit is on the normalised mean absolute error of the median.** Arm D's error on
import half-hours may exceed P0's by at most 0.25 points, the margin's floor, because a median error
smaller than the margin is below the smallest difference that the plan counts. P0's own error is
near 0.02 points in the simulation. H1 and H2 each fail if either limit fails.

**A rerun is the same attempt only when a fit raised, or a check that reads no bar failed, the fix
changes no constant, and the Opus reviewer confirms the fix.** `inconclusive` and `not supported`
are never rerun. Every number is labelled interim until the counted reading it belongs to is read.

## Leakage checks

**Each of these checks is an assertion in the code, and a failure is verdict row 1.**

- No base day sits in two splits, no training day lies within 2 days of a test or validation month,
  and no validation day lies within 2 days of a test month.
- No test or validation month overlaps the fixed lockdown window.
- The test, validation, and shifted-mix banks come from a seed space that no training bank uses, and
  their hashes match across all arms.
- Solar source S1 reads a farm on the scored base day only, S2 reads CAMS at a farm site, never at
  D8, and the CAMS series of that farm site differs from the CAMS series at D8.
- The synthetic BSP is built the same way in every split, from the same base day's real meter.
- No simulator parameter lies outside the ranges on this page, and no simulator code reads the
  metered engine record or the private constants file.
- The private cut is applied, and no day of the metered engine record is a base day.
- No valid time on or after 2026-07-01 is read, and no arm reads the month or the day of the year.
- The architecture, B\*, and m are chosen on validation months only, before any test score exists.

## Compute and gates

**The fits are small, and a timing gate measures their cost before the first fit.** One GPU process
of at most 6 GB runs at a time, with one data worker, on the core set that the manager assigns,
inside `systemd-run --user --scope` with MemoryMax 12G, MemorySwapMax 0, and one BLAS thread. The
plan has 11 arms (A, B, C, D, H, D-quantile, D-24, E, F, G1, G2) and 47 fits:

- 18 fits in the architecture screen (3 architectures, arms A and D, 3 seeds);
- 27 fits in the ladder (the 9 arms other than A and D, 3 seeds each);
- 2 further fits of arm D, for seeds 3 and 4, which set the margin m.

The total of 47 holds only if the ladder reuses the fits of arms A and D from the screen, on the
architecture that the validation rule chooses.

**The GPU projection is 2.7 to 7.8 GPU-process-hours, and 10.2 if the bidirectional LSTM is slow.**
The timing gate runs 5 epochs of each architecture on arm D and projects the GPU-process-hours from
the measured time per epoch. At an assumed 3 seconds an epoch, 47 fits at the cap of 200 epochs take
about 7.8 GPU-process-hours, and 47 fits at an early stop near 70 epochs about 2.7. The bidirectional
LSTM may exceed 3 seconds an epoch. If the 6 LSTM fits (arms A and D, 3 seeds) take 10 seconds an
epoch at the cap, the total is about 10.2 GPU-process-hours. The default budget is 10
GPU-process-hours. The manager owns the GPU slots and may set another budget, and a projection above
the budget goes to the manager before any fit.

**The CPU work adds about 2 hours on one core.** These are estimates from the plan's simulation,
which drew 3,000 segments in a few seconds, and the timing gate measures them:

- Generating the training, validation, and test banks for 5 seeds takes under 15 minutes.
- Scoring 47 fits on test banks of about 92,000 half-hours each, and the bootstrap of 20,000
  resamples over the cell means of 12 contrasts, take about 30 minutes.
- M2's baseline runs on one CPU core in under an hour.

**The gates run in the maintainer's order:** this plan, a Sonnet plan review, Sonnet code, a fresh
Opus code review before any first run, a real-file smoke test of every loader that reads a file the
synthetic-array tests do not exercise (D8's reading, the BSP meter, the frame of experiment 39, the
ERA5 temperature, the bank holidays, and the metered engine record), then a "fits starting" line to
the manager with the projection and the cores before each family of fits. Every fitted neural network
is saved in the experiment's data folder.

## Standing practices and earlier advice

**The plan follows the standing practices that apply to a reconstruction.** It uses cleaned data and
reports the dropped half-hours, saves every fitted neural network, screens the architectures on four
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
5. The 12 contrasts with their intervals and the margin m.
6. Example test windows: the two-lobe distribution as a band and P(X<0) as a strip against the
   truth, for an import-only, an engine-reversal, a solar-reversal, and a combined day.
7. The reliability diagram of P(X<0).
8. Real eras: the share of half-hours with P(X<0) above 0.5 by local hour and era, beside the share
   below 1.0 MVA.
9. The architecture screen on the validation months.

## Choices of the plan

- **The reactive floor is drawn from 0 to 0.3 MVA,** because D8's real reading reaches 0.01 to 0.03
  MVA, and a floor that keeps the reading clear of zero does not fit it.
- **The engine head's real check subtracts the record from D8's reading (K1),** because D8's reading
  is gross demand on the record's days; the unchanged days become a placebo (K2).
- **The BSP coefficient β is drawn from 0.6 to 1.0,** because the step test found a response below 1.
- **Arm D omits wind speed and precipitation,** which are not stored for D8's grid cell.
- **The ladder is cumulative (A, then B, C, D)** and the variants E, F, G1, G2, the quantile head,
  and the 24-hour window are each added to arm D, so that every contrast changes one input. Arm H
  adds the BSP steps to arm A, so that the BSP gain is read in both orders.
- **The demand-climatology baseline is B1,** which turns the climatology into a sign probability
  through the rule that injected output is never negative.
- **The harm checks use P0f and not P0,** because the lobe-width floor adds 0.117 points to every
  arm and not to P0.
- **The test and validation months follow a rule and are listed only in a private file,** because a
  list of months, with the era cut, could let a reader recompute withheld dates.

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
- Half of each epoch's windows come from the training months that carry real BSP steps, in every
  arm.
- An epoch is 8,000 windows; the optimiser is AdamW with a learning rate of 1e-3, weight decay of
  1e-4, batch size 256, and gradient clipping at 1.0; the learning rate halves after 10 epochs with
  no gain; training stops after 25 epochs with no gain in validation signed CRPS, or at 200 epochs.
- Seeds 0, 1, and 2 set the initial weights and the training bank of each fit; seeds 3 and 4 are
  fitted for arm D alone, to set the margin m.

**Measurement:**

- The scored rows, metrics, two-way bootstrap, Kish widening, margin, contrasts, harm limits, real
  checks, and verdict table are in the sections above.
- Every table reports signed CRPS, the 11-quantile pinball loss, and the normalised mean absolute
  error of the median, on all, import, and reverse-flow rows.

## Files

- `ml-experiments/00729-msa-d8-neural-untangler-synthetic-known-answers/e00729_plan_calcs.py`
  (written): the plan's calculations. It applies the split rule to the private base calendar and
  writes `PRIVATE/splits.md`, simulates the synthetic world to find the reverse-flow share and the
  floor scores, counts the metered farms with data on each base day, and prints the arithmetic of the
  bootstrap widening, the seed floor, and the GPU projection. It prints aggregates only.
- `e00729_core.py` (planned): every named constant and bar on this page, the split rule (which
  derives the months from the private base calendar), and the verdict rule.
- `e00729_data.py` (planned): loaders for D8's cleaned reading, the BSP A meter, the CAMS and
  temperature fields of experiment 39's frame, the metered farms' output, the bank holidays, the
  metered engine record, and the private cut.
- `e00729_sim.py` (planned): the engine and solar simulators, the fold, and the synthetic BSP.
- `e00729_banks.py` (planned): builds and hashes the training, validation, and test banks.
- `e00729_nets.py` and `e00729_train.py` (planned): the three architectures, the heads and losses,
  the timing gate, and the fits.
- `e00729_baselines.py` (planned): P0, P0f, B1, B2, and a wrapper of experiment 694's M2.
- `e00729_score.py` (planned): CRPS, pinball, the two-way bootstrap, the Kish widening, the margin
  file, and the verdict.
- `e00729_real.py` (planned): R1, K1, K2, R2, and R3.
- `e00729_figures.py` and `run.sh` (planned): the figures, and the launch wrapper.
- `ml-experiments/00729-msa-d8-neural-untangler-synthetic-known-answers/tests/` (planned): tests on
  synthetic arrays; those that need `torch` skip in CI.
- `data/ml-experiments/00729-msa-d8-neural-untangler-synthetic-known-answers/`: written. It will
  hold `banks/`, `networks/`, `predictions/`, `scores/`, `margin.json`, `verdict.json`, the run
  logs, and a `PRIVATE/` folder for anything dated, which already holds `splits.md` (the month
  lists). The two read-only checks written for this plan, `reverse_flow_check.md` and
  `engine_metering_check.md`, are there too and are never published.
- `data/ml-experiments/00039-pv-disaggregation-cams-injection/work/frame.parquet`: read. The CAMS
  index, clear-sky irradiance, zenith, and 2 m temperature at D8 and at the farm sites, from
  [experiment 39](../00039-pv-disaggregation-cams-injection/result.md).
- The metered engine record: read, from the window table of [experiment
  194](../00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/result.md).
  Never published with dates.
- `data/ml-experiments/private/engine_constants.json`: read through
  `ml-experiments/engine_constants.py` for the private cut only. Never committed.
