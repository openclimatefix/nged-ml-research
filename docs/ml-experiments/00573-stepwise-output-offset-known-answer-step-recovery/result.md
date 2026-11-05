---
id: "00573"
slug: stepwise-output-offset-known-answer-step-recovery
title: >-
  Can a piecewise-constant output offset (daily knots with a total-variation penalty, or sparse
  jumps) recover abrupt steps in a Primary's level, where D5's monthly knots smear them?
theme: retraining-and-training-history
also_relevant_to:
  - differentiable-physics-long-history-and-drift
status: planned
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
  its end value, using at least 5 steps) all hold for S1 or S2. If neither passes K-S1 and K-S2,
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
row_set_id: experiment_470_intersection_v1
metric: >-
  Step 1: step recovery (date within 3 days, size within 25%), precision and recall, false steps per
  Primary-year in the clean world, and the ramp-following error. Step 2: CRPS in points of
  effective capacity at 24-120 h at the eight MVA Primaries and the six MW Primaries.
---

**We ask whether a model of a substation's slow level can see an abrupt step.** Experiment 565's D5
adds one number per Primary and month to the forecast's mean. A sudden change of a few percent of
capacity therefore appears as a slope over several months, and a short event does not appear. This
experiment tests two offsets that can change abruptly, on worlds where the answer is known: we add
steps of known date and size to real Primary series, refit the offset alone on top of a frozen
network, and check whether each offset finds the steps. Only if that check passes do we fit the
network and the offset together on the real series.

**Status: planned. No score has been read.** The sections below are the plan, in the order the
brief fixes.

## Arms

All arms use experiment 565's D5 base network, which is experiment 559's frontier arm and is
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
