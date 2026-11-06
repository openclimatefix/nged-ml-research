---
id: "00594"
slug: persistent-and-share-matched-placebos-for-synthetic-load-shifts
title: >-
  Does the coverage gain from fine-tuning on synthetic load-shifts come from the shape of the shifts
  or from distrust of the lags in general? Two placebos that match the shifts' rows, size, and
  persistence.
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: two new arms, one new fine-tune per fold month, refit day, and seed (40 small
  neural-network fits, private data folder), plus tables and figures. More than one design could
  test the hypothesis (see "Arms"); the arms follow the manager's brief. Spans experiments 574, 581,
  584, and 590, whose code this experiment copies. Serving path and degradation rule: not touched
  (this repository holds none). Cost: well under the 0.6 GPU-h ceiling.
hypothesis: >-
  Experiment 590's noise placebo (T3) was matched to C1 in variance only: it perturbed about 97% of
  rows, not the roughly 7% that C1 perturbs, and its noise was not persistent. Experiment 590 could
  therefore not say whether C1's coverage gain after a load-shift ends comes from the shape of the
  synthetic shifts or from teaching the network to distrust its lags in general. A placebo on the
  same rows with the same magnitude, and a placebo with persistent shifts of random sign, separate
  the two explanations.
prediction: >-
  Written before any score. P1: P-share (noise on C1's event rows, standard deviation equal to the
  event depth, not persistent) covers the NRA on at least 0.03 fewer rows than C1 in the 7 days after
  the end of an injected 15 or 30 day load-shift, because a persistent offset is the pattern the
  injected bank tests. P2: P-persist (shifts of the same lengths and depths as C1, with the sign
  drawn independently for every block of one day, so the net shift over an event averages about zero)
  covers within 0.02 of C1 after the end, because it keeps the persistence. P3: event-free CRPS at 24
  to 120 hours of both placebos is within the minimum detectable effect of C1.
decision_rule: >-
  Counted contrasts: 8 (PS and PP minus C1, each for four measures: coverage of the NRA in q05-q95 in
  the 7 days after the end of the injected 15 and 30 day events, median follow ratio in the same
  window, event-free CRPS at the eight MVA Primaries, and event-free CRPS at the six MW Primaries).
  Intervals are at the Bonferroni level for 8 tests (event-block bootstrap with 2000 draws for the
  injected bank, five-block month bootstrap for event-free CRPS), and the half-width of each interval
  is printed beside it as the minimum detectable effect. Pre-registered reading: if both placebos
  match C1 within 0.01 on after-end coverage, the shape of the shifts does not matter; if a placebo
  beats C1 on after-end coverage by more than the minimum detectable effect, the page reports it
  and says which recipe to prefer. Experiment 590's T3 and experiment 574's P5 are shown for
  reference and are not counted. The experiment adopts nothing. A leak or slice cost is a flag unless
  it actively hurts: worse than C1 by more than max(0.03, the slice minimum detectable effect) with
  the interval excluding zero, or a harmful sign in 4 of 5 fold months.
benchmark: >-
  C1 of experiment 584 (stored forecasts and weights; the P5 recipe fine-tuned with lag-only
  synthetic load-shifts on 5% of each Primary's time), with experiment 590's T3 and experiment 574's
  P5 (both stored) as references.
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the NRA inside q05-q95 and q01-q99 by phase (before, days 0-3, day 4 to the end, 0-7 days
  after, 7-14 days after) on the injected bank and on the verified December pair, median follow
  ratio by phase, safe-side share, pinball loss at q01, q05, q95, q99, width, event-free CRPS in
  points of capacity, calibration on event-free rows.
---

## Plan

- **Arms.** C1 (stored, reference). PS (P-share): inside each of C1's synthetic events, the lags read
  the power plus independent Gaussian noise of standard deviation equal to the event's depth (0.15,
  0.30, or 0.50 of capacity), one draw per half-hour. PP (P-persist): inside each of C1's events the
  lags read the power plus a shift of the event's depth whose sign is drawn independently for every
  block of one day (48 half-hours), so a shift persists for a day but the event has no sustained
  sign. Both keep C1's event starts, lengths, and depths exactly; targets stay clean.
- **Interpretation of the brief.** The brief says "sign drawn per half-hour block". We read a block
  as one day, because a sign drawn per single half-hour would make PP the same as PS with a
  two-point instead of a Gaussian distribution.
- **Reused.** Experiment 584's C1, experiment 590's T3 and P5 (stored forecasts), experiment 574's
  P5 step-2 weights. Code copied from `ml-experiments/00590-*`.
- **Standing practices.** Refit days 0 and 14 per fold month, five development months, seeds 0 and
  1, cleaned power, the D15 and D18 cut inherited from experiment 574's rows, GPU one process under
  8 GB, fitted weights saved in the private data folder.
