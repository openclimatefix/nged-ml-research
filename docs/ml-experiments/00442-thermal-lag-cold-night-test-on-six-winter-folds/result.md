---
id: "00442"
slug: thermal-lag-cold-night-test-on-six-winter-folds
title: >-
  Does the thermal-lag bundle lower the pooled neural network's error on cold winter nights, on six
  winter-weighted monthly folds?
theme: demand-inputs
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: change
protocol: demand-v2
size: >-
  simple. Nothing new is stored beyond this experiment's own forecasts and fitted networks in its
  private folder; every earlier experiment is read and never written. One design (experiment 437's
  bundle T, fitted again on two new folds and on two more seeds). Production serving path: not
  touched (this repository holds none). Degradation rule: not touched. Callers: the scripts import
  experiments 11, 87, 48, 377, 381, 397, 407, 418, 425, and 437, and copy the slice masks of
  experiment 441, whose pull request is not yet merged.
hypothesis: >-
  The maintainer expects thermal lags to help mostly on difficult winter rows, with no harm
  elsewhere. Experiment 441 (rescoring experiment 437) found that bundle T gains at least 0.05
  CRPS points on cold winter nights at both metering groups, but its intervals include zero, not
  every fold was negative, and the slice's minimum detectable effect (0.12 to 0.26) is above 0.05.
  Adding winter months to the folds lets the cold-night slice resolve a gain of that size.
prediction: >-
  Written before any fit and before any new score is read. P1. T minus R on cold winter nights is
  -0.30 to -0.05 CRPS points at the MVA-only eight and -0.30 to +0.10 at the signed-MW six, pooled
  over the five folds that hold cold winter nights. P2. T minus R on all rows outside the cold
  winter nights stays within +-0.04 at both groups. P3. The minimum detectable effect of the
  pooled cold-night slice stays above 0.05 at the signed-MW six, so the interval, not the rule,
  carries the conclusion. Each prediction can fail.
decision_rule: >-
  Interim; folds 2025-10, 2025-11, 2025-12, 2026-01, 2026-02, and 2026-03; seeds 0 to 4 averaged in
  quantile space; widened by 1.15; monthly refits; the rows and the control `R` of experiments 425
  and 437. Primary contrast: T minus `R` on cold winter nights (the mask of experiment 441),
  signed-MW six and MVA-only eight separately. T advances at a group only if (1) the CRPS
  difference is at most -0.05, (2) the month-block bootstrap interval (resampling whole origin
  months, 20,000 resamples, Bonferroni level 97.5% over the two groups) lies wholly below zero, (3)
  the difference is negative in at least 4 of the 5 folds that hold cold winter nights (the brief's
  6 of 8, scaled to 5 folds), (4) on all other rows the upper end of the same interval is at most
  +0.02, and (5) the increase in mean pinball loss at q95 and at q99 (point estimate) is at most
  +0.02 on the slice and on all other rows. The scorer prints the minimum detectable effect of the
  pooled cold-night slice first (the standard deviation, over 7-day valid-date blocks, of the
  difference in mean CRPS between the control's seeds 0 to 2 and its seeds 3 and 4, divided by the
  square root of two); if it exceeds 0.05 the write-up says so and reports the interval. Cold
  sunny days, cold overcast days, Christmas, and other bank holidays are descriptive. The smallest
  gain worth carrying forward is 0.05 because the brief fixes it. Every number is interim.
benchmark: >-
  `R` of experiment 425 (the ensemble-only pooled plain network, the control T was scored against
  in experiments 437 and 441), seeds 0 to 4, same rows; BG2, `manual_heuristic`, and `climatology`
  as absolute-skill references.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, and pinball loss and coverage at q01,
  q05, q95, and q99, in points of effective capacity, from `studies.eval_protocol`; leads 0.5 to 240
  hours (days 1 to 10).
verdict: not-yet-run
best_arm: none yet
production_status: not-for-production
adopted_in: none
take_home_message: >-
  Not yet run.
code: ml-experiments/00442-thermal-lag-cold-night-test-on-six-winter-folds/
---

**Take-home message: not yet run.**

## Question

CRPS is the continuous ranked probability score, a single error number for a probabilistic
forecast. The signed-MW six and the MVA-only eight are the two groups of Primary substations by
metering.

**This experiment asks whether thermal-lag columns help the pooled neural network on cold winter
nights.** Experiment 441 found a gain of at least 0.05 points there, but its intervals included
zero. The scored year holds five winter months (November to March); experiment 437 fitted three of
them and October. This experiment fits bundle T on the other two winter months, November 2025 and
February 2026, so that the cold-night slice rests on five folds.

## Design

- **Arms.** `T` (experiment 437's thermal-lag bundle, ensemble-only form; `T_pre` is skipped
  because it was worse) and the control `R`.
- **Base network: `R`, not S1 and not the brief's "R-inj".** The S1 confirmation (experiment 439)
  is still a plan, so S1 has not passed. The brief names "R-inj" as the reference, but experiment
  428's R-inj is the lagged network trained on injected load-shift worlds, which bundle T was
  never tested on. The contrast that experiments 437 and 441 measured is T minus `R`, so the
  contrast here is the same.
- **Folds: six, not eight.** The brief asked for two further winter months before 2025-07 (January
  and February 2025). They cannot be scored: the 51-member demand frames, the forecast origins,
  the stored references (BG2, `manual_heuristic`, `climatology`, the XGBoost recipe), and the
  protocol rows all cover only 2025-07-01 to 2026-06-30, so a 2025 fold would need those rebuilt
  first, and its training window (stored runs start on 2024-04-01) would be 9 to 10 months against
  the 18 to 24 of the other folds. The six folds are 2025-10, 2025-11, 2025-12, 2026-01, 2026-02,
  and 2026-03; October holds no cold winter nights.
- **Reuse and fits.** The control `R` is stored by experiment 425 for every month with seeds 0 to
  4, so it is not refitted. `T` is stored by experiment 437 for four folds with seeds 0 to 2. New
  fits: `T` on 2025-11 and 2026-02 with seeds 0 to 4 (10 fits) and `T` on the four old folds with
  seeds 3 and 4 (8 fits), 18 fits of about 75 seconds each, under 1 GPU-hour. One GPU process
  under 8 GB, one feeder core, `MemoryMax=8G`, `OMP_NUM_THREADS=1`. The fits wait for a free slot
  from the manager.
- **Slices.** The masks of experiment 441 (copied, because its pull request is not merged):
  cold winter nights (primary), cold sunny days, cold overcast days, Christmas, and other bank
  holidays; heat waves cannot be scored. Row and day counts per slice and fold are printed before
  scoring.
- **Standing practices.** Monthly refits (stored networks); cleaned data (flagged exact zeros
  dropped, count per fold printed); the D15 and D18 cut at 2025-03-01; tails reported;
  every number labelled interim.
- **Figures.** Gain-with-interval chart for T on each slice; fold-by-fold table; example winter
  night week (forecast, actual, `R`, `T`); a day-ahead and a 14-day example forecast.
- **Reviews.** Draft pull request; plan review (fresh Sonnet); code review (fresh Sonnet) before
  the first fit; results review with the audit triggers; write-up review; prose review.

## Counts and minimum detectable effect, printed before `T` was scored

The scorer's `pre` stage read only the control `R` (stored seeds 0 to 2 against seeds 3 and 4).

- **The pooled cold-winter-night slice is large enough to decide on but not to detect 0.05.** It
  holds 16,185 scored half-hours on 35 local dates at the signed-MW six, and 22,459 half-hours on
  42 dates at the MVA-only eight. Both groups have rows in five folds: 2025-11 (7 dates at each
  group), 2025-12 (13), 2026-01 (15 at the six, 22 at the eight), 2026-02 (7), and 2026-03 (3).
  2025-10 has none.
- **The minimum detectable effect of the pooled cold-night slice exceeds 0.05 at both groups.**
  Over 7-day valid-date blocks it is 0.19 at the six (15 blocks) and 0.15 at the eight (16 blocks).
  Over the five folds it is 0.063 at the six and 0.090 at the eight. A gain of 0.05 can therefore
  meet the decision rule only if the interval is unusually narrow, and a failure to meet it is not
  evidence of no effect. The write-up reports the interval whatever the rule's reading.
- **Cold sunny days** hold 40 dates at the six and 35 at the eight (minimum detectable effect 0.20
  and 0.29 over blocks); **cold overcast days** 12 and 10 dates; **Christmas** 10 dates in two
  folds; **other bank holidays** 4 dates in one fold. The last three are descriptive.

## Caveats registered before the fits

- **Six folds, not eight,** for the reason above; the sign rule is scaled from 6 of 8 to 4 of 5.
- **Two of the five cold-night folds are new, three were seen in experiment 441,** so the pooled
  result is not an independent replication. The new folds alone are reported beside the pooled
  result.
- **Seeds 3 and 4 of `T` on the old folds are fitted after experiment 441's result was read.**
  They fix the number of seeds, not the design.
- **The in-slice month-block interval is nearly all-or-nothing.** It resamples 5 origin months, so
  an upper end below zero needs almost every fold to be negative, which is stricter than the
  "4 of 5" condition.
- **The mask thresholds use runs up to the refit, whose later valid times pass it.** Only ENS
  weather enters the thresholds, never a score or an actual, and experiment 441 used the same
  window.
