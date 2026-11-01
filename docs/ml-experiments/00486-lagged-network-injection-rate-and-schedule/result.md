---
id: "00486"
slug: lagged-network-injection-rate-and-schedule
title: >-
  Does the share of training rows with a corrupted lag, and how it changes during training, alter
  how far the lagged network follows a level change in its lags?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
date: 2026-11-01
size: >-
  medium. Stored: new training worlds, fitted networks, forecasts and sweep tables in this
  experiment's private folder only; experiment 404's folders are read and never written. Serving
  path: not touched (this repository holds none), and the regime costs nothing at serving. One
  design is defensible for each arm because the brief fixes the rates and the schedules. Callers:
  the scripts import the folders of experiments 404, 428, 402, 397, 422, and 87 unchanged. One
  Sonnet plan review, one Sonnet code review before the first fit, one Sonnet results review, and
  two Sonnet write-up reviews.
hypothesis: >-
  The share of training rows whose lags lie inside an injected event sets the prior the lagged
  network (A0-inj) learns about how often a jump in its lags is an event. A lower share leaves more
  of a lag step in the median forecast, and a higher share makes the network trust fresh lags less.
  A schedule that trains first at a high share and ends at a low share (Cur-down) reaches the lower
  leak without the lost score of a constant high share.
prediction: >-
  Written before any score is read. P1: at 3 lags the median of R05 follows a 5%-of-capacity step by
  at least 0.05 more (as a share of the step) than A0-inj, and R50 follows it by at least 0.05 less
  (the best-posed prediction; the sweep share is the quantity the rates are meant to move). P2: R50
  costs under 0.05 points of capacity on rows with no labelled event at the signed-MW six against
  A0-inj (an equivalence check, not a pass, because 0.05 is below the minimum detectable effect).
  P3: Cur-down is within the minimum detectable effect of the better of R05 and R50 on both rows with
  no labelled event and injected harm rows. P4: no arm beats B-mix on rows with no labelled event at
  the six, as no learned arm has so far; the question is whether an arm narrows the gap
  (A0-inj is +0.35 behind at the six).
decision_rule: >-
  Sixteen contrasts, Bonferroni k = 16: each of R05, R50, Cur-down, and Cur-up minus A0-inj and minus
  B-mix, on rows with no labelled event (real data, clean world) and on injected harm rows (scoring
  worlds w200 to w202), at the signed-MW six, in points of capacity. Real harm rows and rows inside
  real events are descriptive. Contrasts with the 50/50 mixture of each arm with the lagless table
  are descriptive and not in k. A prediction below the minimum detectable effect is reported as an
  equivalence check (is the arm within the minimum detectable effect of A0-inj?) and never as a
  pass. The smallest gain worth porting is zero on injected harm rows with an interval that excludes
  zero, with no loss on rows with no labelled event beyond the minimum detectable effect and q99
  pinball loss within +0.02, because the rate and the schedule cost nothing at serving. Stop rule:
  if R05 and R50 are within the minimum detectable effect of A0-inj on injected harm rows, rows with
  no labelled event, and the sweep share at 3 lags, say that the rate does not matter between 5% and
  50% of rows, and write the schedules up as not run on more seeds.
benchmark: >-
  The stored A0-inj and B-mix forecasts of experiment 404 (seed 0 in five months; seeds 1 and 2 in
  2026-06), read from experiment 404's private folder and never refitted.
row_set_id: >-
  Experiment 404's rows: the intersection of finite actuals with flagged zeros dropped.
metric: >-
  Mean CRPS from the 11 quantiles in points of effective capacity, absolute beside every contrast;
  q01, q05, q95, and q99 coverage and pinball loss; the readings-level encroachment sweep of
  experiment 404 on every arm; avoidable curtailment at f = 0.9, 0.75, and 0.5.
---

**Plan only: no fit of the four new arms has run.** The world shares and the reproduction check below
come from `e00486_fit.py share` and from re-fitting A0-inj at 2026-06 seed 0 (arm `A0-repro`).

## What this experiment asks

**The lagged neural network (A0-inj of experiment 404) learns from training worlds in which
power readings have been corrupted by injected load-shift events, and the share of training rows
with a corrupted lag has never been varied.** Experiment 404 chose it once. This experiment trains
the same network, on the same rows and with the same loss, on worlds with a different share, and
on two schedules that change the share during training. It then asks whether the network's median
still follows a step in its lags, and what that costs on rows with no labelled event.

## Arms

All arms are experiment 404's A0-inj network (mixture density network, three origin lags, six
epochs, experiment 397's recency weights, monthly refit, lagless table frozen), trained on injected
worlds saved to disk, one per rate and seed. Only the world in each epoch changes.

| Arm | Share of training rows with a lag inside an injected event | Schedule |
|---|---|---|
| A0-inj (reference, stored) | 27.8% measured at 2026-06 (experiment 404 called it 20%) | constant |
| R05 | 4.9% | constant |
| R50 | 49.3% | constant |
| Cur-down | 49.3%, then 27.8%, then 4.9% | epochs 1 to 2, 3 to 4, 5 to 6 |
| Cur-up | 4.9%, then 27.8%, then 49.3% | epochs 1 to 2, 3 to 4, 5 to 6 |

- **Rate definition.** Spikes count as events, and the shares are for 2026-06 seed 0 (the
  write-up reports the share in every cell). The share counts training rows with at least one of the three origin lags
  inside an injected event, measured on the 2026-06 training pool at seed 0 (`e00486_fit.py share`).
  The brief's 5%, 20%, and 50% are the labels; the stored world measures 27.8% by this definition.
  The new worlds use experiment 428's injection with the event coverage changed to 0.016 (R05) and
  0.18 (R50), set by measuring candidate worlds (0.02, 0.04, 0.20, 0.30 gave 6.4%, 11.4%, 54.1%,
  and 73.4%).
- **Worlds are per seed and new for each rate.** Training seeds 100 to 102; the scoring worlds w200
  to w202 and the clean world w-1 are experiment 404's, unchanged, so every arm is comparable with
  the stored A0-inj and B-mix.
- **Reproduction.** `A0-repro` re-fitted A0-inj for 2026-06 seed 0: on world w200 its quantiles
  differ from the stored A0-inj by at most 1.4e-4 of capacity. The check passes if every arm-level
  score matches the stored one within the minimum detectable effect; it is also the noise floor.
  No other cell is refitted.
- **Mixtures.** Each new arm is also scored as the fixed 50/50 mixture with the cross-fitted lagless
  table (`X+lagless`), read alone and in the mixture; the mixture contrasts are descriptive.

## Minimum detectable effect, row sets, and the stop rule

- **Minimum detectable effect (MDE).** Computed from the stored A0-inj and B-mix before any new arm
  is read, and printed first. Source 1: the seed spread at 2026-06 (seeds 0 to 2). Source 2:
  experiment 404's month-to-month spread (0.4 to 1.0 points on harm rows, 0.3 to 0.9 inside events,
  0.3 to 0.8 on clean rows). Clean-row equivalence uses experiment 439's bar of 0.061. For the sweep
  share, equivalence means within 0.05 of A0-inj.
- **Row sets of the 16 contrasts.** Signed-MW six only (the MVA-only eight apart, never pooled).
  Rows with no labelled event: the clean real world w-1. Injected harm rows: pooled over w200 to
  w202. Contrasts use the five months at seed 0; seeds 1 and 2 at 2026-06 give the MDE and
  descriptive rows. Intervals are month-block bootstraps with the sign count over the five months
  beside each; event slices use whole-event intervals.
- **Stop rule.** Evaluated after the seed-0 cells of R05 and R50. If it fires, Cur-down and Cur-up
  still run on those cells, but 2026-06 seeds 1 and 2 are not fitted for any arm.
- **References.** The gain is read against A0-inj; q99 pinball loss within +0.02 is against B-mix;
  no loss on rows with no labelled event is judged against the 0.061 bar.

## What the write-up will state

- One sentence an operator could read on what the regime changes in the forecast; that the regime
  needs no event labels.
- Serving cost per arm and per reference: networks per pass (one for an arm, two for B-mix), tables
  stored and refreshed (none, against the lagless table), detector or gate (none), and the score gap
  to B-mix on rows with no labelled event, harm rows, and injected harm rows.
- q01, q05, q95, and q99 coverage and pinball loss, the 12 to 36 h lead band, raw and after table P
  with the 1.15 widening; row counts per row class (including rows inside real events, after events,
  and cold nights); avoidable curtailment at f = 0.9, 0.75, and 0.5.
- The readings-level sweep and the bad-data injections of experiment 465 (copied from its branch
  with credit, since it is unmerged), the time series plots, and the `dataviz` skill's rules.

## Compute

- **Fits:** 4 arms × 7 cells (five scored months at seed 0; 2026-06 at seeds 0 to 2) = 28 fits at
  about 26 s each on the GPU, about 0.2 GPU-hours, one GPU process under 8 GB and one feeder core.
- **Reused:** the stored A0-inj and B-mix forecasts on all 7 cells; one reproduction fit.
- **Scoring:** forecasts on w200 to w202 and w-1 for 8 arms (four networks alone and mixed), the
  sweep, the corruption injections, curtailment, and the bootstrap dominate the wall time.

## Standing practices followed

Monthly refits; cleaned data with flagged `substation_zero` half-hours dropped (the count per cell
is printed with the scoring); the D15 and D18 start cut before 2025-03-01; fitted networks saved
under this experiment's data folder; stored forecasts reused; tails and curtailment reported; the
unlogged-event caveat stated: NGED's logs may miss events, so rows with no labelled event may hold
events, the harm table is optimistic, and real-event slices undercount. Weather inputs are the
ECMWF ENS mean and spread of experiment 404 (no past-weather lags enter the network, so the
past-weather caveat does not apply).

## Reading earlier advice

The plan follows experiment 404's advice to use the fixed 50/50 mixture B-mix as the reference and to
report the absolute score and event counts beside every contrast. It departs from nothing.
