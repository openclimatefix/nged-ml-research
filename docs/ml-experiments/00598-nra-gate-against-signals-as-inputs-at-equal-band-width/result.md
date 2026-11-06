---
id: "00598"
slug: nra-gate-against-signals-as-inputs-at-equal-band-width
title: >-
  Does a neural network that reads the gate's signals as inputs cover the normal running arrangement
  as well as the hand-built gate does, at the same band width?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. Answers to the triggers: (1) it stores new fitted network weights and forecasts in the
  private data folder; (2) more than one design could test the question (the plan fixes one and
  lists the others as departures); (3) it spans code whose callers cannot be named without
  searching (experiments 574, 581, 584, 591, 592, and 597); (4) it touches no serving path and no
  degradation rule; (5) it changes no contract. The plan review is one Sonnet agent, the code review
  one Sonnet agent, and the write-up review one Sonnet agent.
hypothesis: >-
  A neural network that is fine-tuned on synthetic lag-only load-shifts and is given the gate's
  signals as inputs can set its own band width as well as the hand-built G4 gate sets it, so the
  gate is not needed.
prediction: >-
  Written before any score. P1: arm S reaches arm R's day-4-to-end and after-end coverage within
  0.01, at R's event-free band width within 2% (fails if S is 0.02 or more below R). P2: S beats arm
  W (R's band widened by a constant factor to R's mean width on event rows) by at least 0.03 after
  the end. P3: S's follow ratio is at most 0.08. P4: arm S-p (the signals replaced by noise of the
  same variance) is at least 0.02 below S, otherwise the inputs carry nothing. P5: S's event-free
  CRPS is within the minimum detectable effect of A1.
decision_rule: >-
  P1 and P2 both hold: prefer S, because it has no hand-tuned threshold, no neighbour rule, and no
  false-trigger rate; say that one real pair and the step from synthetic to real events remain
  unproven. S is 0.03 or more below R from day 4 to the end in the first fold pair (2025-10 and
  2025-12): run only S-g next. S-g also fails: keep the gate and stop.
benchmark: >-
  Arm R, which is A1 + G4 from experiment 597 (stored forecasts, export in
  `backtest_export_candidates_2026-11-06`).
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the NRA inside q05-q95 and on the safe side of q05 or q95, by phase; median follow ratio;
  band width; event-free CRPS at 0-24 h and 24-120 h; tail pinball.
---

**Planned, no score read yet.** We ask whether a neural network that reads the signals the G4 gate
uses can learn to widen its own band during a load-shift, so that the hand-built G4 gate is not
needed. A load-shift moves part of one Primary substation's demand onto a neighbour for days or
weeks. The comparison is at equal band width, because a wider band alone raises coverage.

## Plan

- **Arms (all P5 recipe with A1 wiring, fine-tuned on lag-only synthetic shifts mirrored at one
  neighbour).** R: A1 + G4 band (stored from [experiment
  597](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00597-nra-assemble-c1-and-a1-with-g4-gate-as-single-forecasts/result/)).
  W: A1 with the band widened by a constant factor to R's mean width on event rows. S: signals as
  inputs, no gate. S-g: S plus the G4 state bit. S-p: S with noise of the same variance in place of
  the signals.
- **Reuse.** The A1 pre-training and step-2 weights of [experiment
  591](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00591-shared-lag-encoder-auxiliary-shift-depth-loss/result/)
  start every S arm, with the new input columns zero-initialised, so each S arm begins as A1. Only
  the fine-tune is new, which is why the GPU estimate is below the brief's 3 GPU-h ceiling.
- **Cadence.** Coarse refits (day 0 and day 14), as in experiments 584 and 591; seeds 0 and 1; five
  development months 2025-10, 2025-12, 2026-01, 2026-03, 2026-06, first fold pair first. D15 and D18
  are cut before 2025-03-01 and reported only.
- **Where the signals come from (a departure from the brief's wording).** The brief takes the
  disagreement from the lags-masked twin's median. Stored lag-free forecasts exist only for the
  five test months, so training rows would have none. Instead, a frozen copy of the fold's A1
  step-2 network, run with its lags masked, gives a lag-free median and band for every grid time and
  Primary, in training and test months alike (the table `T`). Every signal is then a lookup on `T`
  plus the observed power, as the G4 gate does with E2: the disagreement of each lag with `T` at the
  lag's own time, its sustain count, the fresh residual (observed power minus `T` up to the origin
  minus 4.0 h, never as a lag), the lag ages, the neighbour's disagreement, and `T`'s q05 and q95.
  `T` is in-sample on training rows and out-of-sample on test rows; this is a caveat that S-p tests.
- **No lookahead.** The disagreements and the sustain count use lags at least 24 h old. The fresh
  residual uses power up to the origin minus 4.0 h. A training row's signals use only its own
  synthetic lags and earlier rows.
- **Counted tests.** Seven: P1 (two phases), P2, P3, P4, P5 (two groups). The Bonferroni level and
  the minimum detectable effect are printed beside each contrast.
- **Stop rule.** As in the decision rule. One GPU process at a time under 8 GB; ask the manager
  before crossing 3 GPU-h.
- **Standing practices followed.** Monthly-equivalent coarse refits, cleaned data, the D15 and D18
  cut, saved weights, reuse of stored forecasts, tails and avoidable-curtailment cost, load-shift
  event plots (at least 6, both metering groups), example forecasts, and the unlogged-event caveat.

Reviews: one Sonnet plan review, one Sonnet code review before the first fit, one Sonnet write-up
review (details in the pull request).
