---
id: "00573"
slug: stepwise-output-offset-known-answer-step-recovery
title: >-
  Can a piecewise-constant output offset (daily knots with a total-variation penalty, or sparse
  jumps) recover abrupt steps in a Primary's level, where D5's monthly knots smear them?
theme: retraining-and-training-history
also_relevant_to:
  - differentiable-physics-long-history-and-drift
status: finished
kind: other
size: >-
  complex. Stored: offsets, fitted weights, and forecasts under this experiment's private data
  folder only; experiments 554 and 565 are read only. More than one defensible design: three
  (S1, S2, S3), which is the question. Callers: the scripts import experiments 397, 418, 422, 428,
  448, 510, 511, 559, and 565. Serving path and degradation rule: not touched (this repository
  holds none). Step 1 (the known-answer worlds) fits offsets only, on the outputs of a frozen
  network, and costs minutes of GPU time; Step 2 (the real-data arms) runs only if Step 1 passes.
hypothesis: >-
  D5 adds one offset per Primary and month, with a slope penalty, so a step in a Primary's level (a
  new estate, a persistent transfer, a metering change) is smeared over months and a short step
  cannot be seen. An offset that is constant between days and changes abruptly (daily knots held
  constant between knots, with an L1 total-variation penalty, or a sparse set of jumps chosen by
  hard-concrete gates) recovers the date and size of an injected step.
prediction: >-
  Written before any score is read. S1 or S2 recovers at least 80% of injected steps of 3% of
  capacity or more (date within 3 days, size within 25%) on at least 4 of the 5 folds' training
  spans, and D5's monthly offset (S0) recovers under 40% of them. Fails if S0 recovers 60% or more,
  or if neither S1 nor S2 reaches 60%. On the real series (Step 2, only if Step 1 passes), S1 or S2
  beats S0 by at least 0.03 CRPS points of capacity on all rows at one group, or lowers the error in
  the 30 days after the last jump by at least 0.05.
decision_rule: >-
  Step 1 passes when K-S1 (at least 80% of steps of 3% or more recovered on at least 4 of 5 folds'
  training spans, by S1 or S2), K-S2 (in the clean world no more than 1 false step of 1% or more per
  Primary-year), and K-S3 (in the ramp world the recovered path follows the ramp to within 25% of
  the injected rise, measured from 30 days before the ramp starts to 30 days after it ends, and
  the ramp window holds at least 5 steps of 0.3% of capacity or more) all hold for S1 or S2. If
  neither passes K-S1 and K-S2,
  the experiment stops, no real-data score is read, and the page says that stepwise offsets cannot
  be identified from weather-error-noisy targets at the tested penalties. Step 2 counts 12 CRPS
  contrasts (S1, S2, S3 against S0; times 2 groups; times 2 measures); a gain counts when the
  Bonferroni month-block interval (5 blocks, near the minimum to maximum of the five folds)
  excludes zero. The threshold is the smallest gain worth one more offset parameter per Primary and
  day, which this plan sets at the two-seed minimum detectable effect of 0.113 (the eight MVA
  Primaries) and 0.128 (the six MW Primaries) re-measured from the seed spread of S0; the
  detection of steps is judged by Step 1, which needs no gain.
benchmark: >-
  S0 (D5 as built in experiment 565, monthly knots with the slope penalty) on the same worlds and
  folds; M0 (stored, experiment 554) for the real-data CRPS.
verdict: not supported
best_arm: >-
  None. The highest recall of steps of 3% of capacity or more is 14% to 23% per fold (S1 at weight
  1000 and S2 started from S1 at 1000), with 2.5 to 8.6 apparent false steps per Primary-year in the
  clean world.
headline: >-
  Neither the daily-knot offset (S1) nor the gated-jump offset (S2) recovers injected level steps
  at the penalties tested: recall of steps of 3% of capacity or more is at most 23% on any fold
  against the required 80%, and the weights with under 1 apparent false step per Primary-year
  recover 3% or fewer. Monthly knots (S0) recover none. The real-data arms were not run.
lessons: >-
  At the three penalty weights tested, no weight both silences the offset's apparent false steps in
  the clean world and keeps steps of 2% to 6% of capacity. The clean world's apparent false steps
  include the real series' own unlabelled level changes, and may include the weather model's error;
  this experiment did not separate the two.
take_home_message: >-
  At the penalties we tried, an offset that is allowed to jump did not find steps of 2% to 6% of
  capacity in the demand forecast's residuals. The weights that stop apparent false steps also
  stop the real ones. The next step is to test whether a second signal, such as a neighbour
  series or a label, separates a step from slow drift, before relying on a free offset to date
  level changes.
advice:
  - >-
    Do not read step dates or sizes off a free per-Primary offset fitted to forecast residuals at
    daily resolution without a second signal; at three tested penalties the best recall of 3% steps
    was 23% (Results table).
  - >-
    Separate a step from the series' own drift with a second signal, such as a neighbour's series or
    a label, before trying a stepwise offset again (Results table).
  - >-
    Keep monthly knots (D5) for slow level change; they recovered no steps and showed under 0.1
    apparent false steps per Primary-year (S0 row).
production_status: not-for-production
adopted_in: none
row_set_id: experiment_470_intersection_v1
metric: >-
  Step 1: step recovery (date within 3 days, size within 25%), precision and recall, false steps per
  Primary-year in the clean world, and the ramp-following error. Step 2: CRPS in points of
  effective capacity at 24-120 h at the eight MVA Primaries and the six MW Primaries.
---

*Terms: CRPS is the continuous ranked probability score; GPU is graphics processing unit; MVA and MW
Primaries are the eight substations that report apparent power only and the six that report signed
power; D5 is experiment 565's output-offset design; M0 is the plain modular neural network; S0 to S3
and K-S1 to K-S3 are defined below.*

**We ask whether a model of a substation's slow level can see an abrupt step.** Experiment 565's D5
adds one number per Primary and month to the forecast's mean. A sudden change of a few percent of
capacity therefore appears as a slope over several months, and a short event does not appear. This
experiment tests two offsets that can change abruptly, on worlds where the answer is known: we add
steps of known date and size to real Primary series, refit the offset alone on top of a frozen
network, and check whether each offset finds the steps. Only if that check passes do we fit the
network and the offset together on the real series.

**Verdict: not supported. Take-home: at the penalties tested, a free jumpy offset did not find steps
of
2% to 6% of capacity; the weights that stop apparent false steps also stop real ones.** Step 1
failed, so the stop rule applied and no
real-data arm was run. The sections below are the plan, followed by the results.

## Arms

All arms use experiment 565's D5 base neural network, which is experiment 559's frontier arm and is
lag-free. D5 adds a knot value per Primary to the three mixture component means, in standardised
units (the Primary's target standard deviation).

- **S0 (reference):** D5 as built in experiment 565, monthly knots, linear interpolation, squared
  first-difference penalty.
- **S1 (daily knots, total variation):** one knot per Primary and day, held constant between knots
  (zero-order hold, so a step is not spread over a day), with an L1 penalty on the first differences
  of the knots. The penalty weight is swept over 3 values on the training days of the first fold of
  the worlds, and all three are reported.
- **S2 (sparse jumps):** the offset is the sum over every day k of a gate g_k, an amplitude a_k, and
  a step at that day, with hard-concrete gates ([Louizos et al.
  (2018)](https://arxiv.org/abs/1712.01312)) and a penalty on the expected number of open gates,
  swept over 3 values. If gradient fitting is unstable, S2 starts from S1's thresholded jumps, and
  the page says so.
- **S3 (no-labelled-event fit):** the better of S1 and S2 in Step 1, with the rows inside labelled
  events removed from the loss. S3 is an offset for the normal-running forecast and is not a
  detection source, because it never sees the events.

## Step 1: known-answer worlds

**Worlds.** Real weather, real Primary series, six injected Primaries (three MW, three MVA) with the
fewest labelled events, and the other ten untouched. The step is added to the cleaned power over
capacity at the one place every reader of the target calls (experiment 422's `cleaned_arrays`), the
MVA series are clipped at zero afterwards, and the truth is saved per Primary and half-hour.

- **Ks (steps):** 1 to 3 abrupt steps per injected Primary and year between 2022 and 2026, of
  2% to 6% of capacity with a random sign, at least 30 days apart, about half reverting after 5 to
  60 days and half persistent.
- **Kr (ramp):** a ramp of 3% of capacity over 12 months, with no step.
- **K0 (clean):** no injection.
- **Kt (transfer pairs, from experiment 402's `inject()`):** deferred to Step D, which the brief
  runs only after S1 or S2 passes Step 1.

**Cheap path.** The network is frozen at experiment 554's pretrained M0 weights. For each fold the
network's nine mixture outputs on the fold's ERA5 and CAMS pre-training rows are computed once, and
each arm fits only its offset on top of those outputs by minimising the mixture negative
log-likelihood of the injected targets. Step 2's arms fit the network and offset together, so
Step 1 tests the offset's recovery and not the joint fit.

**Scoring a recovered step.** A recovered offset path is turned into steps by thresholding its
first differences at 1% of capacity (merging adjacent days). A truth step is recovered when a
recovered step of the same sign lies within 3 days of it and its size is within 25% of the truth.
A recovered step with no truth step is a false step. Recall is the share of truth steps recovered
and precision the share of recovered steps that are true; both are reported with n and the binomial
half-width. An offset in the standardised unit is converted to capacity fractions through the
Primary's target standard deviation.

**Seeds and folds.** One seed; the five folds of 511, each fold's training span being its
pre-training rows. The injected series are the same for all five folds, so the folds differ only
in the training span and the fold's frozen weights.

## Step 2: real data (only if Step 1 passes)

Described in the brief and run only if K-S1 and K-S2 pass: CRPS at 24-120 h of S0, S1, S2, and S3
against M0, the part C slices, the absorbed share on the 74 real events, the error in the 30 days
after the last jump, the jump count per Primary-year, the share of the 9 long labelled events that
appear as a step on and a step off, and offset plots per Primary. Step D (the offset as a
load-shift detector) runs only if S1 or S2 passes Step 1. Seeds: 2; coarse refits (one fit per
fold). Cost is measured on the first pre-training and projected before the rest runs.

## Scoring protocol for Step 1 (fixed before any score is quoted)

- **Offset fits.** Each offset is fitted with Adam on a fixed random 25% of a fold's pre-training
  rows (about 1.8 million of 7 million; the weights below are per this subsample), 1,000 steps, rate
  0.02 decaying to 0.002. S0 uses experiment 565's weight
  of 10 on the mean squared first difference of the monthly knots. S1's total-variation weight is
  swept over 300, 1000, and 3000 nats per standardised unit, and S2's gate weight over 30, 100, and
  300
  nats per expected open gate, on the first fold (2025-10) of worlds Ks and K0. One weight per arm
  is then fixed: the one with the highest F1 (the harmonic mean of recall and precision) in world
  Ks among those with at most 1 false step of 1% or more per Primary-year in world K0.
- **Extracting steps.** A day's change of the offset counts toward a step when it is at least 0.1%
  of capacity; same-direction days up to 2 days apart merge; a merged run of total size 1% of
  capacity or more is a step, dated at the day where half of its change has happened.
- **Recall and precision.** Truth steps in the last 10 days of a fold's span are not scored. K-S1
  uses the truth steps of size 3% or more, pooled over the six injected Primaries, per fold; S1 or
  S2 passes when at least 4 of the 5 folds have recall 0.80 or more. Precision is the share of
  extracted steps that match a truth step.
- **K-S2.** Extracted steps of 1% or more in world K0, counted on the six Primaries that world Ks
  injects, divided by the six Primaries times the years of the fold's span; passing means 1 or fewer
  per Primary-year on at least 4 of 5 folds. These steps include real, unlabelled level changes of
  the real series, so the count is an upper bound on false steps. The count on the ten untouched
  Primaries is reported beside it.
- **K-S3.** For each injected Primary whose ramp ends at least 30 days before the span ends, the
  recovered rise from 30 days before the ramp starts to 30 days after it ends is within 25% of the
  truth, and the ramp window holds at least 5 steps of 0.3% of capacity or more (days of change of
  0.05% or more). S1 or S2 passes when at least two thirds of the (Primary, fold) pairs pass.
- **Which arm.** S1 and S2 are each scored on K-S1 to K-S3; "S1 or S2" means at least one of them
  passes all three. S0 is scored the same way for reference.

## Predictions that could fail (written before any score)

- K-S1 to K-S3 pass for S1 or S2, and S0 recovers under 40% of the steps. Fails if S0 recovers 60%
  or more, or if neither S1 nor S2 reaches 60%.
- S2 has at least 0.1 higher precision than S1 at a recall within 0.1. Fails if within 0.05.
- On real data, S1 or S2 beats S0 by at least 0.03 on all rows at one group, or the 30-day post-jump
  error is at least 0.05 lower. Fails if both are within 0.02.
- At least half of the 9 long labelled events appear as an on-and-off step pair at the labelled
  Primary under S1. Fails if under 30%.
- S3's absorbed share on the 74 real events is at most half of S1's. Fails if inside-event gain is
  within 0.05.
- S1 or S2 finds between 1 and 12 jumps of 2% or more per Primary-year. Fails if over 30 or under
  0.5.

## Timing and cost

The weather of every row comes from the ensemble run issued at or before the origin minus 9 h, as
in experiment 565 (its row-by-row assertions are inherited); no lag column exists; the end time of
every offset used at an origin is the refit cut. Ceiling about 2 GPU-hours, one GPU process under
8 GB. Step 1 is offset-only and costs minutes.

## Plan review and standing practices

Standing practices followed: cleaned data everywhere; the D15 and D18 cut before 2025-03-01 where
real series are scored; July 2026 unused; monthly or coarser refits; models saved in the private
data folder. Departures: none in Step 1. Earlier `advice` items followed: carry slow Primary-level
change as a per-Primary output offset (experiment 565), and state the minimum detectable effect
beside every contrast.

## Departures found in the code review (recorded before the scored run)

- **S2 starts from S1.** Gates started closed never opened, so S2 starts with a gate open at every
  day where S1's offset changes by 0.1% of capacity or more, and the gate penalty then closes the
  gates it does not need. A precision gain of S2 over
  S1 tests thresholding of S1, not independent sparse recovery.
- **Truth steps need data.** A truth step is scored only when the Primary has at least 100 cached
  rows in each of the 5 days either side (a quarter of a typical 400), because some Primaries have
  no
  rows before 2023 or 2024.
- **Per-Primary years.** K-S2's years run from each Primary's first cached day.
- **Unlabelled real drift.** The real series moves on its own, so the offset path holds real level
  changes as well as the injection. K-S3 therefore also reports the rise in world K0 beside the rise
  in world Kr, and a paired path (world Ks minus world K0) is reported as a descriptive diagnostic.

## Results (Step 1, five folds, one seed)

Recall is the share of injected steps of 3% of capacity or more that an arm recovers (date within 3
days, size within 25%), pooled over the six injected Primaries; n is 39 to 43 steps per fold, so one
step is about 2.5 points of recall. False steps are extracted steps of 1% of capacity or more per
Primary-year in the clean world K0, on the six Primaries that world Ks injects (real unlabelled
level changes count as false, so this is an upper bound).

| Arm | Recall per fold | Precision per fold | False steps per Primary-year (clean world) | Ramp pass share per fold |
|---|---|---|---|---|
| S0, monthly knots | 0, 0, 0, 0, 0 | 0 | 0.00 to 0.06 | 0 |
| S1, weight 3000 | 3%, 3%, 0%, 3%, 2% | 3% to 8% | 0.31 to 0.41 | 0 |
| S1, weight 1000 | 21%, 18%, 15%, 17%, 14% | 5% to 8% | 2.47 to 2.76 | 0.17 to 0.33 |
| S2, from S1 at 3000, gate 300 | 8%, 15%, 5%, 5%, 2% | 1% to 6% | 2.31 to 2.97 | 0 to 0.17 |
| S2, from S1 at 1000, gate 30 | 23%, 13%, 2%, 7%, 16% | 1% to 4% | 6.83 to 8.55 | 0 to 0.17 |

- **K-S1 fails for every arm:** no arm reaches 80% recall on any fold. The best fold is 23% (S2 from
  S1 at 1000, gate 30), with 7.8 apparent false steps per Primary-year; S1 at 1000 reaches 21%.
- **K-S2 holds only for S0 and S1 at weight 3000** (at most 1 false step per Primary-year), and
  those two recover at most 3% of the steps. No arm passes K-S1 and K-S2 together.
- **K-S3 fails for every arm:** the best ramp pass share is 0.33 (S1 at 1000, first two folds),
  against the required two thirds.
- **The prediction that S0 recovers under 40% holds** (it recovers none), but the prediction that S1
  or S2 reaches 60% fails, so the experiment stops by its own rule. The prediction that S2 has
  higher precision than S1 fails: no S2 row is more than 0.02 above the S1 row of the same start
  weight.
- **Penalty sweep (first fold, worlds Ks and K0):** S1 weights 300, 1000, and 3000 give recall 46%,
  21%, and 3% with 11.8, 2.5, and 0.41 false steps per Primary-year. The weight that satisfies the
  false-step limit of the selection rule is 3000, so 3000 is the selected weight; 1000 is shown as
  the sensitivity.
- **Untouched Primaries.** The ten Primaries with no injection show 0.26 to 0.43 (S1, 3000) and 2.88
  to 3.23 (S1, 1000) apparent false steps per Primary-year, close to the six injected ones, so the
  false steps are a property of the real series and not of the injection.
- **Why (a hypothesis, untested):** the clean world's steps may be the real series' own unlabelled
  level changes, or the weather model's error. In an unscored first look at three Primaries, the
  recovered path correlated 0.3 to 0.7 with the injected path and varied more than it did. The
  paired
  path (world Ks minus world K0) that would separate the causes was not computed.

## Caveats

- One seed, one frozen neural network (experiment 554's M0, seed 0), and a 25% row subsample, so the
  numbers
  carry no interval; they are known-answer checks of recovery, not significance tests.
- The fits ran on the neural network's outputs with the network frozen; a joint fit might behave
  differently, which Step 2 would have tested.
- S2 starts from S1's jumps, so a precision gain of S2 over S1 tests thresholding of S1.
- Steps called false are apparent: the clean world's real, unlabelled level changes count against
  precision, so precision is a lower bound and false-step rates are upper bounds.
- The first-fold sweep numbers are in the sweep output files in the private data folder, not in the
  printed report.
- Step 2, Step D, and world Kt were not run. Weather in these rows is the ERA5 and CAMS analysis of
  the target time, as in experiment 565's pre-training; no ensemble forecast is involved.
- Every number is interim and unaudited. GPU time used: about 0.3 GPU-hours.

## Checks

- The median-minus-mean forecast and the quantile-score contrasts do not apply: no forecast was
  scored.
- Audit triggers: none fired. A control (S0 on the clean world) behaved as expected; no result
  contradicted its prediction except the predicted success of S1 or S2, which is the recorded
  result.

## What the next experiment should answer

Whether a second signal (a neighbour Primary's offset, or the labelled events) lets a jumpy offset
separate a step from drift, for example by testing the paired path of world Ks minus world K0 first.
