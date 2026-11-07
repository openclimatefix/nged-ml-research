---
id: "00643"
slug: pv-explicit-term-against-bias-shifted-free-network-and-more-placebo-permutations
theme: solar-disaggregation
also_relevant_to: []
status: planned
kind: other
size: >-
  medium. Stored: pretrained network weights and test-window quantiles in this experiment's private
  folder. Serving path: not touched. Degradation rule: not touched. Designs: more than one way to
  remove a bias from the free network is defensible, so the two honest controls, the hindsight
  bound, the placebo permutations, and every bar are fixed below before any result is read.
  Callers: the scripts import the modules of experiments 634, 638, and 639 and edit none of them.
  Asset graph: not touched. One GPU process, about 90 fits of 35 s (about 0.9 GPU-hours) in stage
  1 and at most 54 more fits in stage 2, against a ceiling of 1.5 GPU-hours.
hypothesis: >-
  The sunny-midday gain of experiment 639's explicit solar term over the free-irradiance network is
  more than a bias in the free network. If the gain were only a bias, a free network whose sunny
  middays are shifted by a constant fitted on training months alone, or given a trainable
  sunny-midday offset with as many parameters as the explicit term's offset, would match the
  explicit network.
prediction: >-
  Fixed before any result was read. (C1, C2) In the real months the explicit network (arm `cal` of
  experiment 639) has a lower sunny-midday CRPS than both bias-shifted free networks by at least
  0.067 points. My expectation is a gain of 0.3 to 0.6 points over both, close to experiment 639's
  gain over the plain free network (0.75), because the bias in experiment 639 appeared between the
  training months and the test window, and a constant fitted on training months cannot see a shift
  that happens after training. (C3, C4) The same holds in the step world. (C5) Averaged over five
  permutations, the capacity-swap placebo loses to the explicit network in the step world by at
  least 0.181 points. A hindsight shift (fitted on the test window itself) is reported only as a
  bound, and I expect it to close most of the real-world gap, as in experiment 639.
decision_rule: >-
  Fixed before any result was read. CRPS is in points: percent of effective capacity, pooled over
  the six MW Primaries, the three cutoffs (2025-07, 2026-03, 2026-06), and the sunny half-hours of
  the 60 days after each cutoff (10:00 to 15:00 UTC, clear-sky index at least 0.7), averaged over
  seeds 0 to 4. A contrast is the control's CRPS minus the explicit network's, so a positive value
  favours the explicit network. Intervals are day-block bootstraps at the 99% level (95%,
  Bonferroni-corrected over the five contrasts C1 to C5). The bars are experiment 639's minimum
  detectable effects (MDE) and do not change: 0.067 in the real world and 0.181 in the step world.
  A contrast counts as a gain when its interval excludes zero and the point estimate is at least the
  MDE. The recommendation rule: if C1 and C2 are both gains, recommend building the version that uses
  ensemble weather forecasts and name its cost; otherwise report that the explicit term's gain is a
  bias correction and stop the explicit-PV line (a gain over only one of the two controls counts as
  otherwise). C3 and C4 are reported without a bar. The placebo C5 (the mean over the permutations
  of the permutation's CRPS minus the explicit network's) is decided "above" when it is a gain,
  "below" when its interval's upper end is under the MDE, and undecided otherwise. If undecided
  after stage 1, stage 2 adds seeds 5 to 7 of the explicit network and of all five permutations
  (54 fits), and the bar stays 0.181. If still undecided at the GPU-hour ceiling, the page says
  undecided. The added step (a capacity input) needs a threshold above zero, so the MDE is the
  threshold and no larger production threshold is set here.
benchmark: >-
  Experiment 639's arms `free` (seeds 0 to 4) and `cal` (seeds 0 to 4), both stored, and its `perm`
  placebo (shift 2). Everything else is new: `fbias`, `foff`, the hindsight bound, and `perm`
  shifts 1, 3, 4, and 5.
row_set_id: >-
  Experiment 639's test rows (not a published row set): the 60 days after each of the cutoffs
  2025-07, 2026-03, and 2026-06 (30 days at 2026-06), one row per Primary and valid time, cleaned
  power with farm-gap rows dropped. Nothing dated 2026-07-01 or later is used.
metric: >-
  CRPS in points of effective capacity from the 11 quantiles of each mixture, on sunny middays and
  on all half-hours; the median forecast's error on sunny middays; the learned sunny-midday shift
  or offset in points.
---

# Is the explicit solar term's gain only a bias in the free network?

*Plan, written before any result exists.*

We ask whether a demand neural network that is told a Primary substation's solar capacity forecasts
sunny middays better than a network that learns the solar response from irradiance, or whether the
gain of [experiment 639](../00639-pv-explicit-hindcast-probe-set-capacity-step-world/result.md) is
only that the free network's sunny-midday forecast runs too high. We give the free network two bias
corrections that use no capacity information, and compare the explicit network with each.

## Arms

All networks, rows, cutoffs, worlds, and seeds are experiment 639's. The free and explicit networks
are read from its stored weights and quantiles. Only the arms below are new.

- **`fbias`: a constant fitted on training months.** For each free network (cutoff, world, seed) and
  each of the six Primaries, take the mean of the median forecast's error (forecast minus actual)
  on the sunny-midday rows of the held-out weeks of the training data, and subtract that from every
  quantile of the network's sunny-midday test forecasts. The held-out weeks are the early-stopping
  rows, which the network did not fit directly.
- **`foff`: a trainable sunny-midday offset.** Train a new free network (experiment 639's loop, `train_explicit`) on the target `y + delta_p * inc * s`,
  where `s` is 1 on sunny middays and 0 elsewhere, and `delta_p` is a trainable offset in
  megawatts for each of the six Primaries (six parameters, the same number as the explicit term's
  offset, with the same prior of 0.25 MW). At forecast time subtract `delta_p * inc * s`. This is the
  explicit network with no physics curve, no capacity table, and no probed capacity.
- **`fhind`: the hindsight bound (not eligible for the decision).** Shift each free network's
  sunny-midday forecast by its own mean median error on the test window, per cutoff and Primary.
  Experiment 639 computed this once on the pooled rows; here it is a per-arm score with an interval.
- **`perm` shifts 1 to 5.** The explicit network with the capacity timelines of the six Primaries
  cyclically shifted by one to five places (experiment 639 used two). Shift 3 swaps three pairs.
  Capacity parameters are experiment 639's `cal` parameters, rearranged as in its `perm` arm.

## Contrasts

- **C1:** `fbias` minus `cal`, real world, sunny middays. **C2:** `foff` minus `cal`, same rows.
- **C3, C4:** the same two contrasts in the step world (4 MW added from 2023-04-01).
- **C5:** the mean over shifts of `perm` minus `cal`, step world.
- **Also reported, without a bar:** `free` minus `cal` (to reproduce experiment 639), `fhind` minus
  `cal`, each contrast on all half-hours, and the median forecast minus the mean forecast.

## Standing practices followed and departed from

- **Followed:** cleaned data, the D15 and D18 cut, saved weights, reuse of stored networks and
  predictions, a paired contrast on identical rows, five seeds, development months only, anonymised
  generators (farms A to F by letter), CAMS and ERA5 weather only, six MW Primaries only.
- **Departed from:** monthly refits (one fit per cutoff, as in experiments 634 and 639); the
  load-shift slices (no event is injected); the screen on four test months (three cutoffs); the
  forecast-example plots (recorded weather has no lead, so one example week is shown).
- **Earlier advice followed:** the shifted-free-network control and a placebo with an MDE below the
  expected loss (both from experiment 639's advice).

## Review order

A fresh Sonnet agent reviews the code before the first fit. A fresh Sonnet agent checks the write-up
against the printed report, and a second runs the prose review. An audit follows if a trigger in
the `ml-experiment` skill fires.
