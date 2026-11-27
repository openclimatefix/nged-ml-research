---
id: "00733"
slug: msa-forecaster-with-untangled-d8-history
title: >-
  Does the axial-attention demand forecaster forecast D8's apparent power better when its 14 history
  days also carry the untangling network's probability of reverse flow, unmetered solar output, and
  load-transfer steps, and can a head of the forecaster forecast the probability of reverse flow on
  synthetic days?
theme: solar-disaggregation
also_relevant_to:
  - lagged-power-and-error-correction
  - switching-labels-in-the-forecast
status: planned
kind: other
size: >-
  Anything else. (1) The experiment stores up to 92 fitted neural networks on the real folds and
  up to 9 on a synthetic forecasting world, a table of untangled history channels for D8, per-row
  forecasts, and score tables, all private; (2) more than one design defensibly tests the
  hypothesis (frozen or end-to-end untangler outputs, which outputs, warm start or fresh training,
  all 16 Primaries or D8 alone), so the plan records each choice and its reason; (3) the experiment
  reads the stored forecasts and stage 1 states of experiment 713, the training loop and history
  path of experiment 716, the banks and networks of experiment 730, the stored real-day outputs and
  networks of experiment 732, and the ENS loader of experiment 707, whose callers cannot be named
  without searching the repository; (4) the experiment touches no serving path or degradation rule;
  (5) the experiment changes no contract. Reviews, in order: this Opus plan, a Sonnet plan review,
  Sonnet code, a fresh Opus code review, a real-environment smoke test, then the results, write-up,
  and prose reviews.
hypothesis: >-
  Adding three untangled signals to each of the 14 history rows of experiment 713's axial-attention
  demand forecaster (the untangling network's probability that D8's power flows in reverse, its
  estimate of unmetered solar output, and its probability of a load-transfer step), computed by a
  frozen untangling network that never trained on any real day the forecaster reads, lowers the
  continuous ranked probability score (CRPS) of the forecast of D8's apparent power at leads of 24 h
  to 240 h, against the same forecaster with the folded reading alone, and the gain comes from the
  untangled values of each history day and not from the added parameters. On synthetic D8 days with
  a known sign, a head of the forecaster that forecasts the probability of reverse flow for each
  target half-hour, from origin-time weather, beats the persistence of the untangled history.
prediction: >-
  Written before any fit or score exists. (a) The forecast of D8's apparent power: C1 (FU minus
  M*) has a point estimate between -2% and +0.5% of M*'s CRPS at D8 and an interval that includes
  zero, so verdict row 7 (`not supported`, no detectable difference) is the likeliest row, row 5
  (`partly supported`) the next, and row 3 (`supported`) unlikely. Experiments 716 and 717 found no
  detectable gain from solar inputs in the history and target rows, and D8 is one Primary scored
  over 8 months, so C1 cannot resolve a gain much below about 3% of M*'s CRPS. Any gain sits in
  the midday half-hours of the spring and summer folds, where D8 folds. C2 (FU minus the placebo
  FU-S) has the sign of C1. The prediction of (a) fails if C1's lower end is above zero. (b) The
  sign forecast on synthetic days: C3 (YU minus the persistence rule B-hist) is a gain, and the
  sign-forecast word is "skilful": a Brier skill of at least 0.10 against the training-bank
  climatology, with a reliability term within its bar. C4 (YU minus Y0) is a gain, because the
  folded reading alone does not say which past half-hours were in reverse flow. The oracle-history
  arm YT beats YU by more than YU beats Y0. The prediction of (b) fails if C3's interval includes
  zero.
decision_rule: >-
  Every bar and constant is a named `Final` of `e00733_core.py`, fixed before any fit or score
  exists, and the two that need stored data (the smallest gain g and the climatology of reverse
  flow) are written to `limits.json` from stored reference files before any fit. Four contrasts are
  counted at 98.75% (Bonferroni, 1 - 0.05 / 4): C1 and C2 on the real folds and C3 and C4 on the
  synthetic forecasting world, defined in "Contrasts". The smallest gain worth porting g is 2% of
  M*'s CRPS at D8 on the headline band, because the change adds a neural network and a pipeline
  stage to production. Verdict, first row that matches: (0) stopped; (1) `invalid`; (2) `failed`;
  (3) `supported`, C1 and C2 gains, C1's point at or below -g, and the guards hold; (4) `small
  gain`, C1 and C2 gains and the guards hold, C1's point above -g; (5) `partly supported`, C1 a
  gain and C2 not a gain or a guard failing; (6) `refuted`, C1's lower end above zero; (7) `not
  supported`, every other case. Two reading words sit beside the verdict and do not change it: the
  sign-forecast word (from C3, a skill bar, and a reliability bar) and the history word (from C4,
  with the oracle-history reading R5 beside it). The section "Verdict table" gives every row in
  full.
benchmark: >-
  M*, experiment 713's selected arm M1, from its stored forecasts at D8 for seeds 0 to 4 of the 8
  folds; the month-by-half-hour climatology; the Candidate of experiment 667 where stored rows
  cover the scored keys. On the synthetic world, B-hist (the persistence of the untangled history's
  probability of reverse flow), B-clim (the training banks' share of reverse flow by local
  half-hour and day type), and O (the untangling network's probability on the target day itself,
  which reads the target day's reading and is a ceiling, not a forecast).
row_set_id: >-
  backtest_2025_11_to_2026_06_daily_12utc_before_holdout, restricted to D8, for the forecast of
  apparent power. e00733_synthetic_forecast_d8_v1 for the sign forecast: the target half-hours at
  lead days 2 to 10 of every origin of experiment 730's 64 test segments whose target rows lie
  inside the segment and whose base date has an ENS 00Z run. No valid time on or after 2026-07-01
  is read.
metric: >-
  CRPS in points of D8's capacity on leads from 24 h up to but not including 240 h after the 12:00
  UTC origin, with the mean absolute error of the median forecast normalised by capacity beside it;
  q01, q05, q95, and q99 pinball loss and coverage; cold winter nights; the avoidable-curtailment
  cost; the four load-shift scenarios. On the synthetic world, the Brier score, the log loss, the
  reliability term, and the Brier skill against B-clim of the forecast probability of reverse flow,
  by lead band and reading band, and the CRPS of the forecast reading.
production_status: not-for-production
adopted_in: none
---

**We ask whether the demand forecaster of D8 forecasts better when it is told what was hidden in
D8's past readings.** D8 is a primary substation (a "Primary") whose meter reports apparent power, a
magnitude with no sign. A gas engine and a solar farm connect below D8, so power sometimes flows out
of D8 towards the grid, and the meter shows that export as a positive reading, the same as import.
A neural network that forecasts D8 from its last 14 days of readings therefore cannot tell, from
the readings alone, which past half-hours were import and which were export.

**An earlier experiment trains a second neural network, the untangling network, to recover what
the meter hides.** For each past half-hour the untangling network gives the probability that D8's
power flowed in reverse, the output of unmetered solar panels in megawatts, and the probability that
a block of customers was moved to or from a neighbouring Primary (a load transfer). The untangling
network learned on synthetic days with a known answer and has never seen a real day with that
answer.

**This experiment adds the three untangled signals to each of the forecaster's 14 history days, and
compares the forecast of D8's reading with the forecast of the same forecaster without them.** The
untangling network is frozen, and its stored outputs enter the forecaster as three extra numbers per
past half-hour. A placebo forecaster reads the same signals from 28 days earlier, so that a gain from
the extra parameters is not mistaken for a gain from the signals.

**A second question asks whether the forecaster can forecast the probability of reverse flow itself,
for each half-hour of the next 10 days.** No meter records reverse flow at D8, so that forecast is
trained and scored on synthetic D8 days, where the true sign is known. The forecast must beat a rule
that copies the probability of reverse flow from similar past days.

**The design builds on the earlier work on the same substation.** The forecaster is the
axial-attention network of [experiment
713](../00713-msa-axial-attention-demand-network-part-1/result.md), an attention design that follows
the row-and-column attention of AlphaFold 2 ([Jumper et al.
(2021)](https://doi.org/10.1038/s41586-021-03819-2)). The extra history input follows the path of
[experiment 716](../00716-msa-unmetered-pv-part-1-history-estimate/result.md), and the placebo
follows experiments 716 and
[717](../00717-msa-unmetered-pv-part-2-target-row-forecast/result.md). The untangling network is
experiment 732's (MSA D8 untangling network on synthetic injections), whose synthetic days come from
[experiment 730](../00730-msa-d8-sign-detector-with-demand-head-and-masking/result.md). Every section
from "Why this design" to "Method" is the plan, written before any fit or score exists.

## Why this design

**The forecaster needs the untangled signals in its history rows only, because the untangling
network reads the weather observed at the same time as the reading.** The untangling network reads
D8's reading over a 48-hour window together with observed sunshine and temperature. For a past
half-hour, observed weather is a fact the forecaster may use. For a target half-hour, observed
weather lies in the future, so the target rows read only the weather forecast issued at or before
the forecast origin, and no untangled value of a target half-hour ever enters the forecaster.

**The untangling network enters frozen, through its stored outputs, and is never trained through
the forecaster's loss.** Three reasons decide this choice:

- The frozen network was fitted only on synthetic days built from D8's pre-engine readings. Those
  days lie inside the training window of experiment 713's stage 1, but stage 1 is not refitted
  here and its stored states read no untangled channel. Every pre-engine day precedes the first
  history day of stages 2 and 3, the only stages that read the untangled channels, so no fit of the
  untangling network saw a real day whose untangled channel the forecaster reads.
- An untangling network trained end to end would learn whatever helps the forecast loss, and its
  three outputs would no longer mean the probability of reverse flow, the solar output, and the
  transfer step that the synthetic world defines and scores.
- The stored outputs cost no GPU time, and the real-day outputs of experiment 732 are already on
  disk for every day the forecaster reads.

**The plan must work whichever way experiment 732 comes out, so a written rule picks each signal
before experiment 733 reads any of experiment 732's scores.** Experiment 732 is still running. The
section "Which untangled signals enter" states the rule, and the fallback for the probability of
reverse flow is experiment 730's sign-only network S, whose real-day checks passed.

**The forecaster keeps experiment 713's recipe and all 16 Primaries, and only D8 carries the new
signals.** A forecaster trained on D8 alone would be a different recipe from M*, the stored arm that
the counted contrast compares with. Experiment 725 found that refitting stages 2 and 3 from M*'s
stored stage 1 state with the same seed reproduces M* exactly, so the stored forecasts of M* are an
exact control at no GPU cost, and every new arm pairs with M* seed by seed. At the other 15
Primaries the new channels are absent and flagged absent.

**The sign forecast is trained and scored on a synthetic forecasting world, because no real day
has a known sign.** The world is built from experiment 730's synthetic D8 months, which hold a
simulated engine, simulated solar output, and the fold of the meter, so that each half-hour has a
true sign. Each forecasting sample takes 14 history days and up to 14 target days from one synthetic
month. The target rows read the real ENS weather forecast issued on the base day's date, so the
synthetic forecast faces the same weather uncertainty as a real one.

## Terms

- **The reading** is D8's apparent power, a magnitude in points. **Signed power X** is D8's real
  power, positive for import and negative for reverse flow. **Folding** is the loss of the sign.
- **A point** is 1% of D8's effective capacity, the normaliser that
  [data-processing](../data-processing.md) uses for D8.
- **P(X<0)** is the probability that a half-hour's signed power is negative.
- **The untangling network** is experiment 732's headline arm U\*, averaged over its seeds 0 to 2.
  Its three outputs used here are the sign output P(X<0), the solar output in points, and the
  transfer-step probability.
- **The untangled channels** are the untangling network's outputs placed on the forecaster's
  history grid, one value per history half-hour, with a presence flag.
- **The forecaster** is experiment 713's axial-attention network. **M\*** is its selected arm M1
  (frozen pre-trained encoders), stored for seeds 0 to 4 of 8 monthly folds.
- **The contract origin c** is 12:00 UTC on the origin day D, and **the native origin** is 09:00
  UTC on day D, when the ENS 00Z run of day D is available. **The visibility limit** is the last
  history half-hour whose power the forecaster may read, which experiment 713's layout fixes (27
  hours before the contract origin, as experiment 716's page states).
- **A history row** is one of the 14 rows of 48 half-hours at ages 2 to 15 days, and **the target
  row** is the row of one lead day k, as experiment 713 defines them.
- **A segment** is one calendar month of D8's pre-engine base days carrying one continuous draw of
  injected engine and solar output, as experiment 730 defines it. **A base day** is a UTC day of
  D8's real pre-engine reading.
- **The synthetic forecasting world** is the set of forecasting samples built inside segments, in
  "The synthetic forecasting world".

## What the earlier experiments found

**Six earlier results set the choices of this plan.**

- **Experiment 713:** the axial-attention network forecast the six Primaries metered in signed MW
  slightly better than the Candidate, and at the eight Primaries metered in MVA only the pooled
  difference rested on two Primaries, D8 and D12. A Primary that folds is where a forecaster
  reading only a magnitude has most to gain from knowing the sign, which is why D8 is the scored
  Primary.
- **Experiments 716 and 717:** a capacity-times-sky estimate of unmetered solar output in the
  history rows, and a solar forecast in the target row, gave no detectable change in CRPS. The
  forecaster already reads the sky. The untangled solar output differs from those estimates because
  the untangling network reads D8's own reading, so the untangled solar output can carry information
  the sky does not, such as a change of capacity.
- **Experiment 725:** history weather that a live service could build changed CRPS by no
  detectable amount, and a refit of stages 2 and 3 from M*'s stored stage 1 state with the same
  seed reproduced M* exactly.
- **Experiment 729** is invalid, because a bug left half of its synthetic solar farms silent. Every
  generator whose output this experiment reads is drawn in a gallery before the first fit.
- **Experiment 730:** the sign-only U-net S gave the best probability of reverse flow on synthetic
  days (Brier score 0.00853), and a demand output with a shared encoder made the sign worse. On real
  days its check H2 passed.
- **Experiment 732** trains one U-net, U\*, for the sign, the solar output,
  and the transfer steps, and stores U\*'s outputs for every real D8 day from the start of the
  forecaster's training window to the end of the last fold, apart from days with no reading.

## The forecaster and its history rows

**The forecaster is M\*, unchanged, with one zero-initialised input path for the untangled
channels.** Every arm on the real folds starts from M*'s stored stage 1 state of the same fold and
seed and runs stages 2 and 3 of experiment 713's recipe through experiment 716's copied training
loop. A network that loads M*'s state with every new channel absent gives M*'s forecast bit for
bit, and a unit test checks that.

**The history path is experiment 716's token path, widened from one channel to three.** For each
history token the path computes `γ ⊙ (W2 GELU(W1 [u·ok, v·ok, w·ok, ok]) / r)`, where u is the
untangled P(X<0), v is the untangled solar output divided by D8's capacity and by the fold's target
standard deviation (clipped to 0 to 10, in the units of the power input), w is the untangled
transfer-step probability, and ok is the presence flag. `W1` maps 4 to 32 with bias, `W2` maps 32
to 64 without bias, `γ` is a 64-element gain that starts at zero, and r is the root mean square of
the path's output at initialisation over the arm's training tokens. The read-out path of experiment
716 also carries the three channels: the attention weights that build the similar-day base give the
weighted means of u, v, and w, and a zero-initialised linear map adds those means and the cover to
the head's first layer. A channel that the rule of the next section drops is set to zero in every
token, with no change to the path's shape.

**The new parameters train in their own optimiser group, as in experiment 716.** The learning rates
are experiment 716's (3e-3 for `γ`, 2e-3 for `W1`, `W2`, and the head map, 1e-2 for the read-out
table), with no warm-up and no weight decay on the gains. Mask dropout hides the untangled channels
of the age-2 row and part of the age-3 row on 10% of training samples, and of ages 2 to 5 on
another 10%, as experiment 716 hid its estimate.

**The untangled channels exist only at D8.** At the other 15 Primaries every new channel is zero
and flagged absent. Only D8 is scored for the counted contrasts, and the other 15 Primaries are
scored as a report-only check that the new path harms nobody.

## Which untangled signals enter

**A rule written now, before any score of experiment 732 is read, decides which of the three
untangled signals enter the forecaster.** The channel stage reads experiment 732's printed verdict
row and reading words from its stored score files and writes the choice to `channels.json` before
any fit of this experiment. Experiment 732's predictions on real days are read, and its synthetic
test scores only through those words. The rule's reading of the real-day checks H2, P1, and P3
is one bit of selection per signal on real days, some of which fall in the fold months. Those checks
read aggregates of the untangling network's outputs and no forecast score, and the write-up states
the selection beside the verdict.

| Signal | Source when usable | Usable when | Otherwise |
|---|---|---|---|
| P(X<0) | Experiment 732's U\*, seeds 0 to 2 averaged | Experiment 732's verdict row is 4 (`supported`) or 5 (`inconclusive`), and its real-day check H2 passes | Experiment 730's S, seeds 0 to 2 averaged, from its stored real-day predictions |
| Solar output | Experiment 732's U\* | The P(X<0) source is U\*, experiment 732's solar word is "reasonable" or "partly", and its real-day checks P1 (night) and P3 (follows the sky) pass | Dropped |
| Transfer-step probability | Experiment 732's U\* | The P(X<0) source is U\*, and experiment 732's transfer word is "works" or "works only in Brier score" | Dropped |

**Experiment 732 stopped, `invalid`, or `failed` makes experiment 730's S the only source.** If
experiment 732 has not finished when the code of this experiment is ready, the line waits at verdict
row 0. A dropped signal is set to zero with its flag off, so the arms keep their shape whatever the
rule picks.

**The arm set follows the usable set.** FU carries every usable signal, and FS carries P(X<0)
alone. When P(X<0) is the only usable signal, FU and FS are the same arm, FS is not fitted, and the
report-only readings that need FS are `void`.

## How the outputs are placed on the history grid

**The stored outputs of experiment 732 cover every history day whose untangling window ends before
the forecaster's visibility limit.** Experiment 732 scores the middle 24 hours (a UTC day) of a
48-hour window running from 12:00 UTC on the day before to 12:00 UTC on the day after. The stored
output of a UTC day therefore depends on inputs up to 12 hours after that day ends. For an origin on
day D, every UTC day up to D − 3 has a window that ends at or before 12:00 UTC on day D − 2, which is
before the visibility limit, so the stored outputs of those days enter unchanged.

**For each origin the untangling network is run again on UTC days D − 2 and D − 1, with every input
after the visibility limit hidden.** The stored windows of those two days end after the visibility
limit, at 12:00 UTC on days D − 1 and D. The re-run hides D8's reading, the readings of the three
neighbours, and the BSP steps after the limit by setting each value and its presence mask to zero,
which is the hidden-span input that experiment 732 trained on. The CAMS sky after the limit is
hidden through its presence mask. The ERA5 temperature, wind speed, and precipitation after the
limit are replaced by the ENS-mean values of the latest 00Z run issued at or before the native
origin. The untangled channel of a history half-hour is present only where the forecaster's own power
token is visible. The re-run costs two windows per origin per seed, a few minutes of GPU time in all.

**The same rule builds the channels of the synthetic forecasting world and of the placebo.** The
placebo FU-S reads, for each history half-hour, the untangled values of the same half-hour 28 days
earlier, which lie before every history row of the sample. A history half-hour whose untangled value
is missing (a day with no reading) has its flag off. A change of regime inside the 28 days, such as
a load transfer or more solar capacity, makes the placebo's values wrong in level as well as in day,
so C2 tests the day-by-day values together with any recent change of level. The report prints the
share of absent flags in FU and in FU-S per fold, so that a C2 gain is not read as a gain from
coverage.

**The table of untangled channels must cover at least 90% of D8's visible history half-hours in
each fold's training window and in each fold month, or the line stops at verdict row 0.** The
real-environment smoke test prints the coverage per fold.

## Leakage and origin-time weather

**Every target row of every arm reads its weather from the NWP run issued at or before the forecast
origin, and an assertion checks this row by row.** On the real folds the target row reads the ENS
00Z run of day D, available at the native origin, through experiment 713's inputs. The assertion
reads, for every target token of every sample in every batch, the run's initialisation time and its
availability time, and fails unless the initialisation time is 00:00 UTC on day D and the
availability time is at or before the native origin. The same assertion checks that every other
target-row column (the trailing-temperature and solar-gain columns, and the CAMS day summary of day
D − 1) reads no valid time at or after the native origin. Those columns are observations of past
days that a live service may not hold at the native origin, so the code records them as a lab
shortcut, and every score is a ceiling on that account, as experiment 725's unrun read-out T3 says.
The date-time columns (solar geometry, calendar, and holiday flags) hold no weather. The assertion
prints one example row in
this form, with the real values of the first sample of fold 2026-01:

```text
origin-time weather: D8, contract origin 2026-01-14 12:00 UTC (native 09:00 UTC),
target valid 2026-01-17 13:30 UTC, lead day 4, ENS run 2026-01-14 00:00 UTC,
available 2026-01-14 09:00 UTC, latest valid time of any other target-row column
2026-01-14 00:00 UTC: pass
```

**The synthetic forecasting world runs the same assertion against the base day's date.** Its target
rows read the ENS 00Z run of the real date of the origin's base day, and the example of the
synthetic world is printed to the private log only, because a base date tells the reader which
months are pre-engine.

**Observed weather enters the untangling network's windows and the forecaster's history tokens
only, never a target row.** The forecaster's history tokens read the ERA5 and CAMS analyses that M\*
reads, as experiment 713 does. A live service could not build those analyses for ages 2 to 5, so
every score here is a ceiling for a live forecast. Experiment 725 found that history weather a live
service could build changed CRPS by no detectable amount.

**Each guard below is an assertion in the code, and a failure is verdict row 1.**

- Every base day of every bank that trained, stopped, or tested experiments 730 and 732 precedes
  the first history half-hour of any stage 2 or stage 3 sample of any fold, the only samples that
  read an untangled channel. The code reads the base calendar from experiment 730's banks and the
  first history half-hour from the fold layout, and prints only pass or fail. A second assertion
  counts the samples whose history overlaps a base day and the untangled channels present there,
  and requires both counts to be zero, so that a change to the fold layout cannot let a pre-engine
  day's in-sample channel into a fit.
- No untangled value of a target half-hour enters any input. The untangled channels are built on
  the history grid alone, and a poisoned copy of the channel table with every target-row value set
  to NaN gives the same forecast to 1e-6.
- No untangling window that feeds a history half-hour reads D8's reading, a neighbour's reading, the
  BSP steps, or the CAMS sky after the visibility limit, and no ERA5 value after the limit. A test
  poisons every such input after the limit with NaN and checks that the re-run's outputs are finite
  and unchanged.
- The placebo's values come from half-hours at least 28 days before the history half-hour they fill.
- Experiment 716's checks run unchanged: cleaned power set to NaN from each fold month's start in
  every training process, and prediction in a separate process.
- On the synthetic world, the forecaster trains on experiment 730's training banks of seeds 3 and 4,
  which no fit of experiment 732 read, stops on the validation bank, and is scored on the test bank,
  which no fit of experiments 730, 732, or 733 trained on. The untangled channels of every synthetic
  sample come from the untangling network's re-run on that sample's own segment.
- No input of the synthetic forecaster reads the segment's world type, its engine parameters, its
  solar capacity, or any truth array.
- No choice reads a test score: the smallest gain g and the climatology B-clim are written to
  `limits.json` from M*'s stored forecasts and the training banks before any fit, and every fit
  stops on validation before the first-test-score marker exists.

## The synthetic forecasting world

**A forecasting sample of the synthetic world takes its history rows and its target rows from one
synthetic month of experiment 730's banks.** A sample is (segment, origin day, lead day k). The
history rows of ages 2 to 15 and the target row of lead day k must lie inside the segment; a history
row outside the segment, or on a missing base day, has `row_exists` set to false, as in experiment
713. A sample is kept when at least 7 history rows exist. The banks are read and never rebuilt, and
the banks stage verifies each bank's SHA-256 digest against experiment 730's `banks/summary.json`.

**The target is the folded reading, and the sign head's target is the event X < 0 at each target
half-hour.** The forecaster's mixture head forecasts the folded reading in the units of experiment
713, and a second head gives the logit of P(X<0) for each target half-hour. Both heads read the same
trunk.

**The target rows read the real ENS forecast issued on the origin's base day, so a synthetic target
day faces real weather uncertainty.** The base days are real days, the injected solar output of
source S1 is a real farm's output on the same base day, and the injected solar output of source S2
comes from the CAMS sky of the same base day, so the ENS run of the base date forecasts the weather
that drives the synthetic solar output. ENS runs exist in the `NWP` Delta table from 2020-10-15, as
experiment 707 records, so base months without a run for every origin are left out. The banks stage
counts the left-out segments per split and writes the count to the private log; if more than half of
the test segments are left out, the line stops at verdict row 0.

**The synthetic forecaster starts from random weights and trains on D8's synthetic samples
alone.** Experiment 713's stage 1 states trained on D8's real readings of the base months, so a
warm start could carry the base demand of a test month into the test score. The synthetic forecaster
has experiment 713's architecture, the history path above, and the sign head, and trains in one
stage on ERA5 history weather and ENS target weather: AdamW with a learning rate of 5e-4 and weight
decay of 1e-3, batch 256, gradient clipping at 1.0, at most 80 passes with a patience of 20 on the
validation loss, which is the sum of the mixture negative log-likelihood and the sign head's binary
cross-entropy, each with weight 1.0. The patience is longer than experiment 713's 8, because
experiment 716 found that a patience of 8 stopped a slow history path after 4 to 23 passes.

**The sign head starts as the persistence rule.** The head reads `[h_0, ū, cover, one-hot lead
day]` through a layer of 32 units, where ū is the logit of the attention-weighted mean of the
history's untangled P(X<0) at the token's column. The output layer starts at zero, the bias starts
at the logit of the training banks' share of reverse flow, and a fixed skip adds ū times the cover.
An untrained head therefore gives the similar-day persistence of the history's untangled P(X<0),
which is close to the rule B-hist, and a unit test checks that.

**Load transfers are absent from the synthetic forecasting world.** Experiment 730's banks carry no
transfer, so the transfer-step channel is near zero there, and the synthetic world tests the sign and
solar channels only.

## Arms

**Seven arms are fitted, and M\* is read from stored forecasts.** The real-fold arms run 8 folds,
and the synthetic arms train on the synthetic world.

| Arm | Where | History input | Sign head | Seeds | Role |
|---|---|---|---|---|---|
| M\* | Real folds | The folded reading | No | 0 to 4, stored | The control of C1 |
| F0r | Real folds | The folded reading, every new channel absent | No | 0 and 1 on folds 2025-11 and 2026-05 | Gate: reproduces M* exactly |
| FU | Real folds | The folded reading and every usable untangled channel | No | 0 to 4 | The headline arm |
| FU-S | Real folds | As FU, with each channel taken 28 days earlier | No | 0 to 2 | The placebo of C2 |
| FS | Real folds | The folded reading and P(X<0) | No | 0 to 2 | Report only: the sign alone; not fitted when FU equals FS |
| Y0 | Synthetic world | The folded reading | Yes | 0 to 2 | The control of C4 |
| YU | Synthetic world | The folded reading and the untangled channels | Yes | 0 to 2 | The headline arm of the sign forecast |
| YT | Synthetic world | The folded reading and the true channels (the indicator of X < 0 and the true solar output) | Yes | 0 to 2 | Report only: the oracle history, a ceiling for the history path |

**F0r is a gate and not a reading.** F0r runs this experiment's loop with every new channel absent,
so its forecasts must equal M*'s stored forecasts at D8 to 1e-6, as experiment 725's Z did. A
difference stops the line at verdict row 0 and goes to the manager, because every counted contrast
on the real folds pairs a new arm with stored M\*.

**FU-S and FS run 3 seeds, and the contrasts that read them pair seeds 0 to 2 of FU.** The headline
C1 reads 5 seeds, because experiment 716 found a noise floor at 2 seeds that hid gains of the size
expected here.

**The fits run in an order that puts the verdict first.** The order is: the timing gate, F0r, FU on
the four screening folds at seeds 0 and 1, the screen rule, the rest of FU, FU-S, Y0, YU, and then
the report-only arms FS and YT. A stop at the stop line loses report-only arms before any counted
arm.

**The screen reads FU minus M\* on folds 2025-11, 2026-01, 2026-05, and 2026-06 at seeds 0 and 1,
and every screen number is interim.** If the screen's point estimate is above +2% of M*'s CRPS at
D8, the rest of FU and FU-S are not fitted, C1 and C2 are read on the screen alone with their
intervals, and the page reports the stop. The synthetic arms run either way.

## Baselines

**On the real folds the references are stored forecasts, and none needs a fit.** M\* is the
counted control. The month-by-half-hour climatology of
`baseline_forecasters.climatology.ClimatologyForecaster` is refitted on each fold's training window,
and its rows are repeated once per seed. The Candidate of experiment 667 is read where its stored
rows cover the scored keys.

**On the synthetic world three rules give P(X<0) for a target half-hour without a forecaster fit.**

- **B-hist, the persistence rule, is the counted reference of C3.** For each target half-hour, B-hist
  averages the history rows' untangled P(X<0) at the same column, with weights from experiment 713's
  starting day-type table (3.0 for the same day type, 1.5 for two different working days, 0
  otherwise, through a softmax). A sample with no visible history value at the column takes B-clim.
- **B-clim is the training banks' share of reverse flow by local half-hour, by weekday or weekend,
  and by calendar month,** from the training banks of seeds 3 and 4, written to `limits.json` before
  any fit. B-clim is the reference of the skill bar.
- **O is the untangling network's P(X<0) on the target day itself.** O reads the target day's
  reading and observed weather, so O is not a forecast. O is printed as a ceiling and enters no
  contrast.

## Contrasts

**Four contrasts are counted, each read at 98.75% (Bonferroni, 1 − 0.05 / 4).** A negative
difference is a gain of the first arm.

| Contrast | Difference | Rows | Question |
|---|---|---|---|
| C1 (verdict) | FU minus M\*, CRPS | D8, real folds, leads 24 h to 240 h, seeds 0 to 4 | Do the untangled history signals improve the forecast of D8's reading? |
| C2 | FU minus FU-S, CRPS | The same rows, seeds 0 to 2 | Does the gain come from the untangled values of each history day? |
| C3 | YU minus B-hist, Brier score of P(X<0) | Synthetic test target rows, lead days 2 to 10 | Does the sign head forecast reverse flow better than persistence of the untangled history? |
| C4 | YU minus Y0, Brier score of P(X<0) | The same rows | Do the untangled history signals help the sign forecast? |

**Every counted reading pairs arms on the same rows and averages seeds before the bootstrap.** On
the real folds each arm's quantiles are averaged over seeds and sorted, as in experiment 713, and the
loss frame from `experiment_store.losses_for_bootstrap` pairs rows on the run and the valid time. The
interval comes from experiment 713's month-block bootstrap with 100,000 resamples. With one Primary
the bootstrap has one factor, the 8 fold months, whose Kish effective number is 8, so the interval is
widened by the ratio of the Student t quantile (7 degrees of freedom) to the normal quantile, which
is the widening rule of experiments 729 to 732 applied to one factor. On the synthetic world the
interval comes from experiment 729's two-way bootstrap, called unchanged: within each test month,
blocks of 7 consecutive origin dates and, independently, whole segments are resampled, with 20,000
resamples, generator seed 733, the Kish effective number of each factor printed, and the same t
widening when either factor falls below 20.

**C1 reads 5 seeds and C2 reads 3, so the report prints the half-width of each interval beside
the verdict.** A C2 that includes zero only because its interval is wider than C1's is then visible
to the reader.

**The normalised mean absolute error of the median is printed beside every CRPS contrast, and the log
loss beside every Brier contrast.** A point of the opposite sign is audit trigger 3 of the
`ml-experiment` skill.

**Report-only readings enter no verdict row and no Bonferroni count, and each is read at 98.75%.**

| Reading | Content |
|---|---|
| R1 | FS minus M\*, CRPS at D8: the sign alone |
| R2 | FU minus FS, CRPS at D8: the solar and transfer channels beyond the sign |
| R3 | FU minus M\*, CRPS pooled over the other 15 Primaries: harm away from D8 |
| R4 | FU with every untangled channel masked at prediction, minus FU, CRPS at D8: whether the trained network uses the channels |
| R5 | YT minus Y0, Brier score of P(X<0): the ceiling of the history path, printed beside C4 |
| R6 | YU minus Y0, CRPS of the forecast reading on the synthetic world |
| R7 | YU minus O, Brier score of P(X<0): the distance to the untangling network on the target day |
| R8 | C1 by lead band (days 2 to 3, 4 to 6, 7 to 10), with each fold month left out in turn, in the midday half-hours (10:00 to 14:00 UTC) of folds 2026-04 to 2026-06, and on cold winter nights |
| R9 | The q01, q05, q95, and q99 pinball loss and coverage, and the avoidable-curtailment cost, for FU, FS, and M\* at D8 |
| R10 | The four load-shift scenarios at D8 for FU, FS, and M\*: all rows; rows outside logged events and at least 7 days after one; rows inside logged events; and the 7 days after an event ends |
| R11 | The sign head of YU by lead band and by reading band (under 1 MVA, 1 to 3 MVA, 3 MVA and above), with the mean P(X<0) on import half-hours of at least 1 MVA beside experiment 730's pooled harm limit of 0.0071 |
| R12 | FU scored with the stored untangled outputs of days D − 2 and D − 1 in place of the re-run, minus FU: how much the inputs after the visibility limit would have moved the forecast |

**R10 carries the unlogged-event caveat.** NGED's logs may miss load transfers, so the rows outside
logged events may hold events that no label covers, and the slices inside events undercount.

## Verdict table

**The first row that matches is the verdict, and every row is reachable.** A gain means the
interval's upper end is below zero, and a loss means its lower end is above zero. The guards are:
C1's point keeps its sign with each fold month left out in turn; the normalised mean absolute error
of the median moves the same way as CRPS; and FU is not worse than M\* by more than 0.05 points on
the q01 or q99 pinball loss at D8.

| Row | Condition | Verdict |
|---|---|---|
| 0 | The line stops before C1 is read: the manager halts it; F0r differs from M\*; the channel table covers less than 90% of a fold's visible history half-hours; a bank digest differs from experiment 730's; more than half of the synthetic test segments lack ENS runs; experiment 732 has not finished; a gallery panel does not show what its written list requires; the timing projection of the counted fits exceeds the stop line; or a review gate has not passed when a fit would start | Stopped, no verdict (`inconclusive` in the front matter) |
| 1 | A correctness check fails: an origin-time weather assertion, a leakage guard, the poisoned-copy checks, the bit-for-bit check of the zero-initialised path, the starting-rule check of the sign head, or a known-answer unit test of the bootstrap or the Brier score | `invalid` |
| 2 | A fit that C1 or C2 needs raised and was not rerun under the rerun rule | `failed` |
| 3 | C1 and C2 are gains, C1's point is at or below −g, and the guards hold | `supported` |
| 4 | C1 and C2 are gains and the guards hold, and C1's point is above −g | `small gain` |
| 5 | C1 is a gain, and C2 is not a gain or a guard fails | `partly supported` |
| 6 | C1 is a loss | `refuted` |
| 7 | Every other case (C1's interval includes zero) | `not supported` |

**Two reading words sit beside every verdict row and never change it.**

- **The sign-forecast word** answers whether the forecaster's sign head forecasts reverse flow well
  on synthetic days. The word is "skilful" when C3 is a gain, YU's Brier skill against B-clim is at
  least 0.10, and YU's reliability term is at most 0.1 times the test rows' share of reverse flow
  (experiment 730's calibration bar); "beats persistence, little skill" when C3 is a gain and the
  skill bar fails; "beats persistence, miscalibrated" when C3 is a gain, the skill bar passes, and the
  reliability bar fails; "no better than persistence" when C3's interval includes zero; and "worse
  than persistence" when C3 is a loss. The word is `void` when a seed of YU is missing or the scored
  target rows hold fewer than 500 reverse-flow half-hours.
- **The history word** answers whether the untangled history helps the sign forecast. The word is
  "helps" when C4 is a gain, "no detectable difference" when C4's interval includes zero, and
  "hurts" when C4 is a loss. R5 is printed beside it: when R5 is not a gain, even the true sign in
  the history does not help the sign head, so a null C4 says nothing about the untangling network.

**The skill bar of 0.10 asks the sign head to remove one tenth of the climatology's Brier score.**
The bar is a judgement and not a ported gain: a forecast that removes less than a tenth adds little
to a decision about reverse flow 2 to 10 days ahead.

**A failed report-only fit leaves the verdict untouched,** and its readings are `void`. A rerun is
the same attempt only when a fit raised, the fix changes no constant, and the Opus reviewer confirms
the fix. `not supported`, `refuted`, and `inconclusive` are never rerun.

## Compute, stop line, and gates

**The slot is A2: one GPU process of at most 6 GB, cores 16 to 21, a memory cap of 14 GB, and at
least 25 GB available at launch (the `available` column of `free -g`), with the load below 24.**
`run.sh` copies experiment 732's launcher. Every command runs inside `systemd-run --user --scope`
with `MemoryMax=14G` (and `SLOT_MEM=14G`), `MemorySwapMax=0`, `OOMPolicy=continue`, and
`AllowedCPUs=16-21`, under `taskset`, with one thread for every numerical library. The wrapper
refuses to start a GPU command while a 716 or 717 scorer or an F7 job runs, while another process of
this experiment holds the GPU, while two GPU processes of any agent already run, while the GPU lacks
6 GB, while less than 25 GB is available, or while the load is 24 or above. A refusal exits with the
launcher's "slot busy" code. The fits run one at a time.

**The chain reports at each step and stops on the first failure.** The launcher sends a "fits
starting" line to the manager with the projection and the cores before each family of fits, and a
"last fit ended" line when the last fit ends. A failed fit or check stops the chain, writes the
failure record, and exits with code 5, which the launcher does not treat as "slot busy". The session
that runs the experiment checks the chain's status at least every 30 minutes.

**The projection uses the stored timings of experiments 713, 716, 717, and 732.** The fit logs of
experiment 716 give a mean of 167 seconds for one fit of stages 2 and 3 from a stored stage 1 state
(arm H, 40 fits), and those of experiment 717 give 180 seconds (arm HT, 40 fits); experiment 716 had
planned 0.08 GPU-process-hours for one fit with its prediction and measured 0.047 hours for the fit
alone. The plan takes 0.08 hours for each real-fold fit. Experiment 713's fits of all three stages
took a mean of 630 seconds (arm M1, 40 fits). A synthetic fit has up to about 37,000 samples per pass,
against 21,396 in experiment 713's stage 1, and the plan takes 0.3 hours for one synthetic fit at
the cap of 80 passes. Experiment 732's `timing.json` gives 3.4 seconds an epoch of 8,000 windows for
U\*, so the re-runs of the untangling network take minutes. The peak GPU memory of experiments 716
and 717 was 3.5 GB, under the cap of 6 GB.

| Work | Fits | Hours each | GPU-process-hours |
|---|---|---|---|
| Real-environment smoke test (one pass of every arm) | | | 0.15 |
| Timing gate | | | 0.15 |
| Re-runs of the untangling network (real days and synthetic banks) | | | 0.2 |
| F0r | 4 | 0.08 | 0.32 |
| FU | 40 | 0.08 | 3.2 |
| FU-S | 24 | 0.08 | 1.92 |
| Y0 and YU | 6 | 0.3 | 1.8 |
| Counted work and gates | | | 7.74 |
| FS (report only) | 24 | 0.08 | 1.92 |
| YT (report only) | 3 | 0.3 | 0.9 |
| Total | | | 10.56 |

**The pre-registered cost is 10.56 GPU-process-hours, under the manager's limit of 12, and the stop
line is 12.** The smoke test, the timing gate, and the re-runs of the untangling network (0.5 hours)
count towards the 12 hours, as the table does. The timing gate runs before the first fit: 1 fold
of FU at seed 0 and 5 passes of YU,
recording the seconds per fit and per pass, the peak GPU memory, and the peak resident memory. The
gate projects every fit at its measured rate (FU-S, FS, and F0r at FU's rate; Y0 and YT at YU's).
If the projection exceeds 12 hours, report-only arms are dropped from the end of the fit order until
the projection fits: YT first, then FS. A dropped arm's readings are `void`. If the counted work
alone exceeds 12 hours, the line stops at verdict row 0 and goes to the manager. An early stop
shortens every fit, and the screen rule may remove 56 fits of FU and FU-S.

**The CPU work is about 2 hours of wall time on the 6 cores:** the channel table and its coverage
check under 15 minutes, the synthetic samples and their ENS rows under 30 minutes, the gallery under
20 minutes, and the scoring, bootstraps, and figures about 1 hour. The scorer loads one fold at a
time and keeps only the columns the bootstrap reads, as experiment 717's advice says.

**The gates run in the maintainer's order.**

1. This Opus plan, then a Sonnet plan review.
2. Sonnet code, then a fresh Opus code review before any run on real files, and a review of the
   fixes.
3. **A real-environment smoke test before the first fit.** One pass of every arm (F0r, FU, FU-S,
   FS, Y0, YU, and YT) on the real GPU and the real files, writing to a scratch folder outside the
   data folder. Each pass goes through the per-batch checks and the origin-time weather assertion,
   the prediction step, and a read of the stored output by the scoring code. The smoke test also
   reads experiment 732's real-day outputs and re-runs the untangling network for one origin, reads
   experiment 730's banks with their digests, builds the synthetic samples of one test segment with
   their ENS rows, prints the channel coverage per fold, and runs experiment 729's
   `tests/smoke_real_files.py`. Then every later stage (score, report, figures, loss curves, and
   attention maps) runs once on validation rows only.
4. The banks digest check, the channel stage, the limits, and the synthetic samples, on the CPU.
5. The synthetic world gallery, drawn before any fit. The session that runs the experiment checks
   each panel against the written list in `e00733_core.py` and records the result in
   `PRIVATE/gallery/panel_check.json`.
6. The timing gate and F0r, and then the fits in the order of "Arms", each family with its "fits
   starting" line.

**Every fitted neural network is saved in this experiment's data folder as its state file.**

## Standing practices and earlier advice

**The plan follows the standing practices of the `ml-experiment` skill, with two departures.** It
refits monthly, uses cleaned data (the half-hours that the `substation_zero` rule flags are dropped
and counted per fold), cuts the D15 and D18 rows before 2025-03-01 in training as experiment 713
does, saves every fitted network, reuses stored forecasts and outputs (M\*, experiment 730's banks
and S, experiment 732's outputs), screens on four folds with a stop rule, reports the tails, cold
winter nights, and the avoidable-curtailment cost, plots forecast examples and load-shift events,
and reports the normalised mean absolute error of the median. The first departure is that the
date-time encoder holds calendar columns, as in experiments 713 and 716, with no control without
them. The second is that D8 is metered in MVA only, so the load-shift plots cannot show both
metering groups that the skill names.

**The plan follows 16 items of earlier advice.**

- It reads the gain at one MVA-only Primary, D8, rather than pooled (713).
- It reports the other 15 Primaries beside D8 in R3 (713).
- It refits stages 2 and 3 from the stored stage 1 state with the same seed, so M\* is an exact
  control, and checks that with F0r (725).
- It reads the gain against a placebo that carries the wrong day's values (716, C2).
- It holds the 28-day-earlier placebo for an input whose own day's value carries the claimed gain
  (717).
- It gives the slow history path of the synthetic fits a longer patience, and adds an oracle-history
  arm whose ceiling the history path should reach (716, YT and R5).
- It loads one fold at a time in the scorer and keeps only the bootstrap's columns (717).
- It treats history weather that a live service could build as a caveat of no detectable size
  (725).
- It reads no rerun of the neighbour-row contrasts (715).
- It uses experiment 730's sign-only S as the reference and fallback for the sign (730).
- It reads the sign head's harm per reading band beside the pooled limit (730, R11).
- It draws every arm in a dot chart with its own colour or label (730).
- It aligns every pair of row sets by key before comparing them (730).
- It checks before any fit that every simulated component, and every untangled channel, is nonzero
  where it should be (729).
- It calls the real loaders in the smoke test, never a fixture built by hand (729).
- It counts every threshold over the half-hours where the quantity exists, such as the solar channel
  over sun-up half-hours (729).

**The plan departs from one item of earlier advice.** Experiment 715 advises 400 passes with a
patience of 30 before a known-answer test of the axial-attention network. The synthetic fits here
run at most 80 passes with a patience of 20, because 400 passes would cost more than the 12
GPU-process-hours of the slot; every fit logs whether the validation loss was still falling at the
cap, and a fit that hits the cap is reported.

## Decisions that need the maintainer

**None, as planned: the plan spends nothing, calls no large language model on substation data, adds
no data source or download, and changes no threshold after data are seen.** The ENS runs of the base
dates are in the `NWP` Delta table that experiment 707 reads. One case would become a decision for
the maintainer, and the line stops at verdict row 0 in that case: if the `NWP` table lacks D8's grid
cell for the base dates, a fresh extraction of ENS runs is a new download.

## Anonymisation

**The published page and its figures follow the anonymisation rule, and D8 is named because it is a
substation.** The page shows no engine size, no connection date, no distance, no register entry, no
series identifier, and no generator value. The untangled channels on a real day estimate when the
generators below D8 export and how much the solar farm produces, so no real-day untangled channel is
printed or drawn as a time series; the page shows their coverage and their effect on the forecast of
D8's reading only. Source S1 is a real farm's metered output, so a synthetic segment drawn from S1
appears only in aggregate panels, and every single S1 window stays in `PRIVATE/gallery/`. The test
and validation months of the synthetic world follow experiment 729's rule, and the month list stays
in experiment 729's private `PRIVATE/splits.md`. The base dates of the synthetic samples appear only
in the private log. Forecasts of D8's reading may be drawn on calendar dates, as experiment 713's
pages draw them.

## Synthetic world gallery

**The gallery shows the output of every generator whose output this experiment reads, before the
first fit, and again in the write-up.** Experiment 730 drew panels G1 to G9 from the banks that this
experiment reads, and the banks are digest-pinned, so those panels show this experiment's engine,
solar sources, fold, synthetic BSP meter, base days, and masks:

- G1: engine days (evening, morning, ramp, trip, part load, multi-day spell), 3 draws each.
- G2 and G3: solar output of sources S2 (single days) and S1 (seasonal bands only).
- G4 to G9: the BSP meter, the fold, the base days, the masks, the realism shares, and the component
  checks.

**Six new panels show what this experiment adds, each with a written list of what the panel must
show.**

- **G10, the layout of a synthetic sample:** for three test segments, the base days of the month on
  an axis of days, with the 14 history rows and the target rows of lead days 1 to 14 that lie inside
  the segment marked, the scored lead days 2 to 10 shaded, and the
  segment's edges shaded. The panel must show every history and target row inside the segment.
- **G11, synthetic target days:** for three S2 test segments, the folded reading, the signed truth,
  the injected engine, and the injected solar output over 10 target days, with reverse-flow
  half-hours shaded; S1 segments as an aggregate of the share of reverse flow by hour of day. The
  panel must show reverse flow on target days in at least two of the three segments.
- **G12, target weather against observed weather:** the correlation by lead day between the ENS
  shortwave of the base date's 00Z run and the CAMS sky of the same target half-hours, over the test
  segments. The panel must show a correlation that falls with lead day and stays above zero.
- **G13, the untangled channels against the truth on synthetic history:** for three S2 test
  segments, the history's untangled P(X<0) against the indicator of X < 0, and the untangled solar
  output against the injected solar output, over 14 history days. The panel must show a nonzero
  solar channel on sun-up half-hours of solar segments, and a P(X<0) above 0.5 on most reverse-flow
  half-hours of at least 1 MVA.
- **G14, the re-run of days D − 2 and D − 1:** the distribution of the difference between the stored
  and the re-run outputs on the synthetic validation bank, by hour after the visibility limit. The
  poison test must pass (NaN and 1e6 placed in every hidden input after the limit leave the re-run
  output unchanged), the mean absolute difference over origins more than 24 hours before the limit
  must be below 0.01, and the mean absolute difference must not increase with the hours before the
  limit (a monotone decay, in bins of 6 hours).
- **G15, the base rate the sign head forecasts:** the share of reverse flow in the synthetic target
  rows by lead day and by hour of day, with B-clim beside it. The panel must show a share above zero
  on sun-up half-hours.

**The real-day panels are private.** A panel of the real untangled channels by month and hour (the
coverage, the mean P(X<0), and the mean solar output) goes to `PRIVATE/gallery/` only; the
published page carries the coverage counts.

## Figures planned

**The write-up draws loss curves, attention maps, and forecast time series, as experiments 713 and
716 did, and every chart follows the `dataviz` skill.**

- **Loss curves:** the training and validation loss per pass of every fitted arm, in small multiples
  by fold for seed 0 of FU, FU-S, FS, and F0r (stages 2 and 3), and by seed for Y0, YU, and YT, each
  with the stored pass marked. The synthetic panels split the validation loss into the mixture
  negative log-likelihood and the sign head's cross-entropy.
- **Attention maps:** the across-day attention weights and the read-out weights of FU and M\* for
  D8's target tokens at lead day 2, on a sunny and a dull target day of fold 2026-05 and a winter day
  of fold 2026-01, from a forward pass of the saved states; and the same maps for YU and Y0 on a
  synthetic sample with reverse flow on the target day. The learned gain `γ` of the history path is
  printed beside each map.
- **Forecast time series at D8:** a day-ahead forecast and the full 14-day horizon of one origin, in
  winter and in summer, for FU, FS, and M\*, against D8's reading.
- **The sign forecast on synthetic days:** for three S2 test segments, YU's and Y0's forecast
  P(X<0) over 10 target days against the true sign, with B-hist beside them; and a reliability
  diagram of YU, Y0, and B-hist.
- **Contrasts:** a dot chart of C1, C2, R1 to R4, and R8 with their intervals, and a dot chart of C3,
  C4, R5, and R7, each arm in its own colour.
- **Load-shift events at D8:** at least 6 time series of logged events in the fold months, from 14
  days before to 7 days after each event, with the start and end marked, for FU and M\*. If the folds
  hold fewer than 6 logged events at D8, every logged event is drawn and the page says how many
  exist.

## Choices of the plan

**Each choice below was made without the maintainer, and each is recorded with its reason.**

- **Frozen stored outputs, not end-to-end training,** for the three reasons of "Why this design".
- **The untangling network's seed-averaged outputs,** because experiment 732 averages its counted
  readings over seeds 0 to 2 and a single seed would add seed noise to every history token.
- **Days D − 2 and D − 1 re-run with the inputs after the visibility limit hidden,** rather than
  dropping the newest days' channels, because the newest history days are the most informative and
  the hidden-span input is what the untangling network trained on. R12 measures what the stored
  outputs would have changed.
- **Experiment 732's check K2 is dropped from the rule (ruling of the design lead, 2026-11-26, set
  before any experiment 733 score exists).** Experiment 732's real-day stage computes H2, P1 to P4,
  and K3 only, and its page says K1 and K2 were not done, so the condition "H2 and K2 pass" reads
  "H2 passes" for the P(X<0) source. With experiment 732's verdict row 4, H2 passing, the solar word
  "not shown", and the transfer word "works only in Brier score", the rule gives P(X<0) from
  experiment 732's U\*, the solar output dropped (zeroed, flag off), and the transfer-step
  probability from U\*. FU carries P(X<0) and the transfer step, FS carries P(X<0) alone, and FU and
  FS are distinct arms.
- **G14 does not require a zero difference before the visibility limit (ruling of the design lead,
  2026-11-26, set before any experiment 733 score exists).** The premise that stored and re-run
  outputs are identical before the limit is wrong, because the untangling network reads the whole
  window. On the real networks the largest absolute difference is 0.06 at 24 to 48 hours before the
  limit, 0.13 at 12 to 24 hours, 0.39 at 6 to 12 hours, and 0.6 within 3 hours, and the mean is 3e-4
  to 5e-3. G14 passes when the poison test passes, the mean absolute difference over origins more
  than 24 hours before the limit is below 0.01, and the difference does not increase with the hours
  before the limit (binned by 6 hours, with a slack of 5e-4, which is 5% of the 0.01 limit, accepted
  by the design lead on 2026-11-27 for bin-to-bin noise). The panel plots this table for the real
  networks.
- **The top-row trailing columns of target-row B1 read the origin day's start (ruling of the design
  lead, 2026-11-26).** Experiment 713's getter cannot read observed valid-time columns for these
  columns, so the synthetic world and the real folds give them the weather of 00:00 UTC on the
  origin day. The page labels these columns a ceiling, as it does for the history rows.
- **The fold-coverage training window starts at 2024-04-01 (ruling of the design lead,
  2026-11-26),** the first run of experiment 607, because no ensemble run exists earlier, so no origin
  earlier has a re-run.
- **The history path is gated by the per-token flag, and F0r trains with the path's rates at zero
  (ruling of the design lead, 2026-11-27, set before any experiment 733 score or fit exists).** An
  ungated path added `gamma * W2 * GELU(b1) / r` to every history token even with every flag off, so
  after a one-pass fit `gamma` was up to 0.12 and F0r differed from M\* by up to 0.307. The path's
  output is now multiplied by the flag, so it is exactly 0 where the channel is absent, on every arm.
  F0r also trains with the new parameters' learning rates set to 0, and the fit stops with a
  correctness failure if any new parameter, `gamma` included, differs from its initial value
  afterwards.
- **All 16 Primaries with the channels at D8 alone,** so that M\* is an exact control and the recipe
  stays experiment 713's.
- **No sign head on the real folds,** because no real day has a known sign, and a head trained on
  the untangling network's own outputs would learn to copy the untangling network rather than
  forecast reverse flow.
- **A random start for the synthetic forecaster,** because experiment 713's stage 1 states trained on
  D8's real readings of the base months.
- **Training banks of seeds 3 and 4 for the synthetic forecaster,** because no fit of experiment 732
  read them. Only the injections are unseen: the base days of the training months are the same as
  in experiment 732's training banks, so the untangled channels of the training samples are cleaner
  than the test samples' channels, whose base months no fit saw, and C3 and C4 are read on the test
  bank alone.
- **Five seeds for C1 and three for the other real-fold arms,** to keep the cost under 12
  GPU-process-hours with the headline at full strength.
- **The smallest gain g of 2% of M*'s CRPS at D8,** because the change adds a neural network and a
  pipeline stage that production would have to run and monitor. The prediction expects C1 to
  resolve only gains of about 3% or more, so row 4 (`small gain`) is reachable only if C1's interval
  turns out narrower than expected; g is set by the cost of porting and not by the power of the
  design, and the page says so when row 4 cannot occur.
- **The month-block bootstrap with t widening on the real folds,** because one Primary leaves no
  second factor to resample, and 8 blocks need the t widening that experiment 713 found the plain
  percentile interval lacked.
- **No arm without the sign head on the synthetic world,** because whether the sign head harms the
  forecast of the reading is a side question, and R6 already reads the reading's CRPS.

## Method

### Architecture

- The real-fold arms are experiment 713's network (width 64, two axial blocks, the similar-day
  read-out, and the three-component mixture head; about 130,000 parameters), loaded from M*'s
  stored stage 1 state of the same fold and seed, with the history path and read-out path of "The
  forecaster and its history rows" (about 2,600 new parameters, all starting at zero effect).
- The history rows of ages 2 to 15 carry the folded reading, its visibility flag, and the usable
  untangled channels with their presence flag; the target row reads the ENS 00Z run of the origin
  day and no untangled value.
- The synthetic arms add the sign head (64 plus 16 inputs, 32 hidden units, one logit), starting
  as the persistence of the history's untangled P(X<0).

### Training regime

- The real-fold arms run stages 2 and 3 of experiment 713's recipe through experiment 716's loop, on
  the 16 Primaries with cleaned power, monthly refits for the 8 folds 2025-11 to 2026-06, and the
  seeds of "Arms".
- The untangled channels are experiment 732's stored real-day outputs of U\* (or experiment 730's S,
  by the rule), with days D − 2 and D − 1 re-run per origin.
- The synthetic arms train in one stage on experiment 730's training banks of seeds 3 and 4, stop
  on its validation bank, and read ERA5 and CAMS history weather and ENS target weather.
- The planned GPU cost is 10.56 GPU-process-hours, and the write-up records the measured cost.

### Measurement

- The real folds score D8 on experiment 713's row set at leads 24 h to 240 h against D8's cleaned
  reading, through experiment 713's scorer, with the month-block bootstrap and t widening.
- The synthetic world scores the sign head's P(X<0) against the true sign on the test bank's target
  rows at lead days 2 to 10, with experiment 729's two-way bootstrap.
- Every number is labelled interim until the counted reading it belongs to is read.

## Files

**Code, in `ml-experiments/00733-msa-forecaster-with-untangled-d8-history/`** (each script carries
the id as `e00733_`):

- `e00733_core.py` holds every constant, bar, and arm specification as a named `Final`, and the
  written list of each gallery panel.
- `e00733_paths.py` resolves this experiment's data folder and the read-only folders of experiments
  713, 716, 730, and 732.
- `e00733_channels.py` applies the rule of "Which untangled signals enter", builds the table of
  untangled channels on the history grid from the stored outputs, and builds the placebo.
- `e00733_rerun.py` re-runs the untangling network on days D − 2 and D − 1 of each origin with the
  inputs after the visibility limit hidden, on real days and on the synthetic banks.
- `e00733_leakage.py` holds the origin-time weather assertion, which prints its example row, and the
  leakage guards.
- `e00733_model.py` extends experiment 713's network with the history path, the read-out path, and
  the sign head.
- `e00733_inputs.py` adapts experiment 716's inputs to the three channels.
- `e00733_synth.py` builds the synthetic forecasting samples from experiment 730's banks, with the
  ENS rows of the base dates through experiment 707's loader.
- `e00733_train.py`, `e00733_fit.py`, and `e00733_predict.py` run the fits (experiment 716's loop on
  the real folds, a one-stage loop on the synthetic world), store the loss history of every pass
  with the stored pass marked, and write the forecasts.
- `e00733_baselines.py` computes B-hist, B-clim, and O.
- `e00733_score.py` and `e00733_report.py` compute every contrast, interval, guard, and reading
  word, and print every number of the page.
- `e00733_gallery.py`, `e00733_figures.py`, and `e00733_attention.py` draw the gallery, the result
  figures, and the attention maps.
- `e00733_series.py` draws the forecast time series, `e00733_signdata.py` joins the synthetic
  world's forecasts of reverse flow to the truth, and `e00733_figstages.py` runs the figure stages,
  optimises each SVG with `svgo`, and refuses a file over 500 KB.
- `e00733_smoke.py` runs the real-environment smoke test, and `e00733_smoke729.py` runs experiment
  729's real-file checks against experiment 730's banks.
- `e00733_stages.py` and `e00733_cli.py` run the stages in order and write the chain status.
- `run.sh` is experiment 732's launcher, copied, with this experiment's commands.
- `tests/` holds the guards the code list requires: a test that every attribute taken from an
  imported module exists; a test that every `.numpy()` call follows `.cpu()`; a test that a failure
  record stores the message and the traceback; a test that a failed chain exits with code 5 and the
  launcher does not read that code as "slot busy"; tests that touch every device branch (CUDA and
  CPU); a test that every sample of the other 15 Primaries carries every untangled channel at zero
  with its flag off; a test that the masking of R4 sets every untangled channel and flag of D8 to
  zero at prediction and changes nothing else; and the known-answer tests of verdict row 1 (the
  bit-for-bit zero path, the starting rule of
  the sign head, the poisoned copies, the origin-time weather assertion on a row that breaks it, and
  the bootstrap).

**Data, never committed, under `data/ml-experiments/00733-msa-forecaster-with-untangled-d8-history/`:**

- `channels.json` (the rule's choice), `channels/table.parquet` (the untangled channels on the
  history grid, with presence flags), and `channels/rerun/` (the re-run outputs per origin).
- `limits.json` (g, B-clim, and the counts of the synthetic world), written before any fit.
- `synthetic/` (the samples of the synthetic world and their ENS rows).
- `networks/` (every fitted state), `fits/` (the loss history of every fit), and `predictions/`
  (per-row forecasts, written with `experiment_store.write_predictions`).
- `scores/` (`timing.json`, the screen, the counted and report-only readings, and the chain status).
- `PRIVATE/gallery/` (the S1 windows, the real-day channel panels, and `panel_check.json`).

**Earlier experiments' files read, never written:** the stored forecasts and stage 1 states of
[experiment 713](../00713-msa-axial-attention-demand-network-part-1/result.md); the training loop and
history path of [experiment 716](../00716-msa-unmetered-pv-part-1-history-estimate/result.md); the
banks, `banks/summary.json`, networks, and real-day predictions of [experiment
730](../00730-msa-d8-sign-detector-with-demand-head-and-masking/result.md); the networks, real-day
predictions, and score files of experiment 732; the bootstrap and `tests/smoke_real_files.py` of
[experiment 729](../00729-msa-d8-neural-untangler-synthetic-known-answers/result.md); and the ENS
loader of [experiment 707](../00707-ens-backfill-pretraining-gain-against-ens-history-length/result.md).
