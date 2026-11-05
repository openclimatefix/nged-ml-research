---
id: "00575"
slug: stepwise-offset-large-step-and-mirrored-pair-recovery
title: >-
  Can a piecewise-constant output offset recover large, label-sized level steps and mirrored pairs
  of steps, where experiment 573 found it recovered steps of 2% to 6% of capacity in none of its
  tests?
theme: retraining-and-training-history
also_relevant_to:
  - differentiable-physics-long-history-and-drift
status: planned
kind: other
size: >-
  medium. Stored: offsets, fitted paths, and per-event scores under this experiment's private data
  folder only; experiments 554 and 573 are read only. More than one defensible design: no, the
  arms and penalty grid are experiment 573's, unchanged. Callers: the scripts import experiments
  400, 428, 573, and the modules those use. Serving path and degradation rule: not touched (this
  repository holds none). Cost: the frozen network's cached outputs are reused, so each fit takes
  seconds; about 0.5 GPU-hours in total.
hypothesis: >-
  Experiment 573 injected steps of 2% to 6% of capacity, which may drown in the weather-error noise
  of the forecast's daily residuals. Steps as large as the labelled load-shifts (a median of 19% of
  capacity) are found by the daily-knot offset (S1) or the gated-jump offset (S2) when they last a
  week or more.
prediction: >-
  Written before any score is read. S1 or S2 recovers at least 80% of injected steps of 20% of
  capacity or more that last 7 days or more (date within 1 day, size within 25%), with no more than
  1 false step of 5% of capacity or more per Primary-year on the untouched Primaries. Fails if
  recall is under 50%. Also: recall for 1-day steps is at most 30% at every size below 50% (fails if
  50% or more); recall falls as duration falls at a fixed size, matching the detectability curve to
  within one duration bin; both halves of a mirrored pair are recovered in at least 70% of pairs
  with a step of 20% or more lasting 7 days or more (fails under 40%); S0 recovers under 20% of
  steps lasting under 30 days.
decision_rule: >-
  A known-answer test with no counted contrasts (Bonferroni count 0). The first prediction is judged
  on the pooled steps of the five folds, for each S1 and S2 arm of experiment 573's penalty grid,
  and passes if any arm meets both its recall and its false-step condition. The minimum detectable
  effect beside each recall is the binomial half-width 1.96 times the square root of p(1-p)/n, with
  n the number of injected steps in the bin; a bin with fewer than 20 steps is marked unresolved. If
  neither S1 nor S2 passes the first prediction, Step D (the offset as a load-shift detector) is
  cancelled. If one passes, Step D runs only over the sizes and durations where recall passes. The
  threshold is zero: this experiment adds no model, and it decides whether a later one is worth
  building.
benchmark: >-
  S0 (the monthly-knot offset of experiment 565) on the same worlds, and the clean-world false-step
  rates of experiment 573.
row_set_id: experiment_470_intersection_v1
metric: >-
  Recall of injected steps by size bin and duration bin, recall of both halves of a pair, false
  steps of 5% of capacity or more per Primary-year, and the detectability curve.
---

*Terms: S0, S1, and S2 are the monthly-knot, daily-knot, and gated-jump offsets of experiment 573.
Kl, Kp, and K0 are the worlds defined below. MVA and MW Primaries are the eight substations that
report apparent power only and the six that report signed power.*

**We ask whether the offsets that failed on small steps can find large ones.** [Experiment
573](../00573-stepwise-output-offset-known-answer-step-recovery/result.md) added steps of 2% to 6%
of capacity to real Primary series and found that no tested offset recovered more than 23% of the
steps of 3% or more. A labelled load-shift is much larger, so the failure may mean only that small
steps drown in weather-error noise. Here we add steps drawn from the sizes and durations of the
labelled events to the same series, with the network frozen and only the offset refitted, as in
experiment 573.

## Design

**Sizes and durations come from the labelled events.** The 184 events in experiment 400's
`events_corrected.parquet` have depth quartiles of 11%, 19%, and 31% of the Primary's capacity, and
length quartiles of 0.4, 2.1, and 3.6 days (the full printout is in the results). Only 65% last a
day or more. The injected events use depth 10% to 50% of capacity (log-uniform) and duration 1 to
180 days (log-uniform), the ranges the manager's note proposed, so the test covers the long events
that a daily-knot offset might reach. The labelled events themselves are mostly shorter, which the
results discuss.

- **Kl (large single steps):** per injected Primary, events starting between 2022-01-01 and
  2026-05-31 with 30 days or more between the end of one and the start of the next (about 2 events
  a year), a random sign, and a reversal at the end. At the three injected MVA Primaries an event is
  always a gain, because a loss could clip at zero. Three random draws of the world.
- **Kp (mirrored pairs):** a donor Primary changes by x and a neighbour on an edge of
  `Primary Substation Interconnections.csv` changes by -rho times x in MW (rho uniform from 0.6 to
  1.0), with the same onset and end; about 30% of events have no recipient. Donors and recipients
  are among the same six injected Primaries, so the ten untouched Primaries stay clean. No MVA
  Primary loses load. Three draws.
- **K0:** experiment 573's clean world. Its fitted paths are reused to count false steps of 5% or
  more; the ten untouched Primaries in Kl and Kp give a second count.
- **Arms:** S0, S1 at total-variation weights 3000 and 1000, and S2 at gate weights 300 and 30 on
  each, exactly as experiment 573's full run. No new tuning.

**Weather timing (standing rule).** The network is experiment 554's pre-trained M0 with its
outputs cached by experiment 573's `e00573_cache.py`, which asserts row by row that each row's origin
is not after the fold's refit, the target is not after the refit minus the cutoff age, and the origin
precedes the target, and prints one example row. This experiment reads that cache and changes no
weather input.

**Scoring.** A step is the change at an event's onset. It counts as recovered when the offset has a
step of the same sign within 3 days of the onset (within 1 day for steps of 20% or more) and within
25% of its size. The return step is scored separately. Only events that finish at least 10 days
before the fold's span ends and have at least 100 cached rows on each side of both changes are
scored. Recall is reported by size bin (10% to 20%, 20% to 35%, 35% to 50%) and duration bin (1 to 3
days, 3 to 14 days, 14 to 30 days, 30 to 180 days; the note's 3 to 14 days is split at 7 days for the
first prediction). For a pair, both halves must be recovered. The detectability curve uses, per
injected Primary, the standard deviation of the offset-free daily residual of the frozen network
(in the clean world), and the smallest step that exceeds three standard errors after n days is three
times that standard deviation over the square root of n.

**Statistics.** Per-fold results are shown beside the pooled results. No contrasts are counted.
Intervals on recall are binomial, not month-block bootstraps, because the quantity is a count of
injected steps.

**Advice followed.** The plan follows experiment 573's advice to use known-answer worlds before
real data, and keeps its monthly knots as the slow-drift reference.

## Results

To be written after the fits.
