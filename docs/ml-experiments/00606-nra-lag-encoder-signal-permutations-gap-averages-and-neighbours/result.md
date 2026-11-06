---
id: "00606"
slug: nra-lag-encoder-signal-permutations-gap-averages-and-neighbours
title: >-
  Which inputs should the lag encoder of the demand network read so that it follows a load-shift
  less: the lag's identity or age, the expected-input embeddings, the gap between a lag and the
  lag-free forecast, weighted averages of that gap, or the gaps at the electrical neighbours?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. Answers to the five triggers: (1) it stores fitted network weights and forecasts for 10
  to 14 arms in the private data folder; (2) more than one design could test the question (the plan
  fixes the arms and lists the other designs as departures); (3) it spans code whose callers cannot
  be named without searching (experiments 560, 574, 581, 584, 591, 592, 597, and 598); (4) it
  touches no serving path and no degradation rule; (5) it changes no contract. Reviews: one Sonnet
  agent for the code, one for the write-up against the printed report, one for the prose, and one
  fresh Sonnet agent that opens every rendered figure. Ceiling 2 GPU-h; ask before crossing.
hypothesis: >-
  The A1 demand network follows a load-shift (reads shifted lags as if they were normal) partly
  because its lag encoder has no input that says how far a lag is from what the lag-free forecast
  expected. Giving the lag encoder the gap between each lag and the lag-free forecast at the lag's
  own time, with exponentially weighted averages of that gap over the last days, lowers follow, and
  summaries of the gaps at all electrical neighbours lower it further.
prediction: >-
  Written before any score. (The removal arms P0, P1, and P2 are dropped; experiment 608 runs those
  removals.) P3: the per-lag
  gap lowers follow below A1's 0.119 by at least 0.015. P5: the weighted gap averages lower follow
  by at least 0.01 more than the sustain count does (P5 below P4). P7: the neighbour summaries lower
  follow by at least 0.01 more than P5, and the own-plus-neighbours sum matters (P7 below P7b by
  0.005 or more). P8 (named below): no arm reaches R's coverage after the end at R's band width (R
  minus the arm, both rescaled to R's width, is at least 0.01 for every arm). Placebo: the best arm
  follows at least 0.02 less than the same arm fed noise of the same variance. O2: the
  total-variation penalty on the D5 offsets lowers the best arm's follow by at least 0.01 and
  changes its event-free CRPS by less than the minimum detectable effect. Inference to check, not
  assume: with clean targets the loss never rewards a wider band when a lag is shifted, so a signal
  can lower follow without lifting coverage; follow and coverage are reported separately for every
  arm.
decision_rule: >-
  Event metrics are descriptive (event-block bootstrap), as in experiments 584 and 598. A signal set
  is worth keeping only if it lowers follow by its pre-registered margin and leaves event-free CRPS
  within the minimum detectable effect at the eight MVA Primaries and the six MW Primaries (D15
  and D18 reported only). Twelve counted contrasts, Bonferroni alpha 0.05/12 = 0.0042, are listed in
  the write-up with the minimum detectable effect beside each. Keep rule from the shared rules: a
  leak or slice cost is a flag unless it actively hurts.
benchmark: >-
  A1 (experiment 591, stored forecasts) for every contrast, and R (A1 plus the G4 gate, experiment
  597) for the width-matched coverage contrast.
row_set_id: experiment_470_intersection_v1
metric: >-
  Follow, share of the NRA inside q05-q95 by phase (day 4 to the end, 0 to 7 days after), band width,
  event-free CRPS at 0-24 h and 24-120 h, with the minimum detectable effect beside each contrast.
---

## Plan

**Standing practices followed.** Refit at day 0 and day 14 of each fold month (the coarse cadence of
experiments 584, 591, and 598), cleaned data everywhere, the D15 and D18 cut before 2025-03-01, every
fitted network saved in the data folder, stored predictions of A1 and R reused as controls, the
first fold pair screened before the other months, and July 2026 unopened. The advice lists of
earlier experiments were read; the plan follows 598's "build signals from out-of-sample lag-free
forecasts" advice by keeping its in-sample safeguard and a noise placebo, and departs from its
"keep the gate" conclusion only in testing a wider set of inputs.

## Design

All arms start from experiment 591's A1 step-2 weights and its shared lag encoder, and differ only in
step 3, the fine-tune on synthetic lag-only load-shifts of experiment 584's C1 (5% of each Primary's
time, depths 15, 30, and 50% of capacity, lengths 3, 7, 15, and 30 days, random sign, clean
targets). The shifts are applied only in step 3; pre-training and step 2 stay unperturbed (experiment
603 tests perturbing them, and the best arm is rerun with that change if 603 shows it helps).
Every extra column is computed from the shifted lags in training and from the real lags at test
time, by one code path (`e00606_signals.signal_columns`): a shifted lag changes the gap, and the
lag-free median at the lag's time is unchanged. Gap inputs use only lags and lag-free forecasts at
least 24 hours old; the fresh residual is not used. Training rows keep experiment 598's in-sample
safeguard: where the training-to-test standard deviation ratio of a continuous column is below 0.8,
noise brings the training variance up to the test variance.

The lag encoder of A1 reads, per lag, the lag's value, missing flag, and age, the weather and
datetime embeddings of the lag's own time with the freshest weather run issued at or before it (the
"expected" inputs), the Primary embedding, and an 8-number lag ID.

| Arm | Lag encoder reads |
|---|---|
| A1 | the control, stored |
| P3 | A1 + per-lag gap (4 columns) + expected-missing flag (2) |
| P4 | P3 + sustain count (2) |
| P5 | P3 + weighted gap averages at half-lives of 1, 3, and 7 days (12 columns), the default design |
| P6 | P5 + sustain count |
| P7 | P5 + summaries over all electrical neighbours (see below) |
| P7b | P7 without the own-plus-neighbours sum |
| P7c | P5 + the single best opposite-sign neighbour |
| P8 | the best subset of P3 to P7c, scored on the remaining months |
| O2 | the best arm with experiment 577's O2 total-variation penalty on the D5 offsets |
| Q-pair | the best neighbour arm and the best non-neighbour arm with the paired perturbation in step 2 and step 3 |
| Q-pre | one pre-training variant of the best arm (only if GPU time is left) |

**Stages.** Stage 1: P3 to P7c with the step-3 perturbation only. Stage 2: the best neighbour arm and
the best non-neighbour arm again, with the paired perturbation in step 2 and step 3 as well. The
paired perturbation lowers demand at one Primary and raises it by a smaller amount at its
electrically connected neighbours (a transfer of megawatts split across them), and the neighbour
summaries are computed from those perturbed lags. Stage 3, only if GPU time is left: one
pre-training variant of the best arm, whose lag-free table `T` comes from a clean-lag A1 step-2
network of the same fold and seed, held fixed and trained only on rows before the fold's test month.
That dependency is circular (pre-training now needs a step-2 network) and is a caveat of the page.
The neighbour arms (P7, P7b, P7c) are also scored on a target-side bank with a mirrored neighbour
(experiment 595's method), because the lag-only bank rewards any perturbation. At most 2 GPU
processes; ceiling 4 GPU-h.

The expected-missing flag is 1 where no stored weather run exists
at or before the lag's time (about 0.3% to 0.8% of rows); the page reports its share per Primary
and checks that adding it does not change all-rows CRPS by more than the minimum detectable effect.

**Neighbours.** An electrical neighbour is first one listed in `Primary Substation
Interconnections.csv`, then a sibling under the same bulk supply point among the 16 Primaries; the
rule is experiment 592's G4 neighbour table (`b400.nged_neighbours()`, reused through experiment
598). P7 gives the encoder, for the gap and for each of the three weighted averages at each lag's
time, the neighbours' sum, mean, minimum, and maximum (signed, scaled to the Primary's capacity),
the sum of the Primary's own value and the neighbours' sum (near zero for a pure transfer), the
neighbour count, and a flag that is 1 where the Primary has no neighbour with a gap (all values then
zero). The injected events mirror the shift at one neighbour by construction, so the injected bank
flatters every neighbour input; a real transfer may split across several neighbours or leave the 16
Primaries, and only one real pair is verified.

**Scoring.** First fold pair 2025-10 and 2025-12, seeds 0 and 1, the injected bank of 15 and 30 day
load-shifts, as in 598. The best arm is then scored on 2026-01, 2026-03, and 2026-06. Every arm
also has a version rescaled about its median to R's band width, so coverage compares at equal width.
The placebo arm replaces the best arm's extra columns with a column-wise resample of the training
block. The real December 2025 pair (D5 and D16) is shown descriptively with the maintainer's example
event.
