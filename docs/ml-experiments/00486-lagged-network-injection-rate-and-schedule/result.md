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
status: finished
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
verdict: not supported
best_arm: >-
  None. R05 (4.9% of training rows with a lag inside an injected event) scores 0.10 points of
  capacity better than A0-inj on rows with no labelled event (interval -0.144 to -0.048, better in
  7 of 7 cells), which is within the 0.14 seed spread, and it follows a step in its lags more
  (0.224 against 0.100 of the step at 3 lags).
headline: >-
  The rate sets the leak but not the score on injected harm rows. At 3 lags the median follows
  0.224 of a 5% step under R05, 0.100 under A0-inj, and 0.058 under R50. All four new arms score
  0.06 to 0.07 points worse than A0-inj on injected harm rows, and every network alone stays 0.25 to
  0.47 points behind B-mix on rows with no labelled event.
lessons: >-
  A higher share of corrupted rows shrinks the leak and costs score on rows with no labelled event;
  a lower share does the reverse. The share in the last third of training decides the leak, so a
  schedule is no better than the constant rate it ends at.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  How many training rows carry a corrupted lag trades the leak against clean-row score: 50% leaks
  less than half as much as 5% but scores 0.2 points worse on rows with no labelled event. No rate
  or schedule beats the fixed 50/50 mixture of the network and the lagless table, so the way to
  stop the forecast following a level change in its lags is not found in the rate. All numbers are
  interim, from five months.
advice:
  - >-
    Do not try to remove the lag leak by changing the share of injected rows or its schedule; the
    share only trades leak against clean-row score (sweep table, contrasts C1 to C8).
  - >-
    Keep the fixed 50/50 mixture with the lagless table as the reference, because no rate or
    schedule of injected rows closed the 0.25 to 0.47 point gap to it (contrasts C5 to C8).
code: ml-experiments/00486-lagged-network-injection-rate-and-schedule/
---
 **How many training rows carry a corrupted lag trades the leak against the score on ordinary rows;
it does not remove the leak for free.** We trained the same neural network, which forecasts a
substation's power from its recent readings (its lags), on worlds in which readings had been
corrupted by injected load shifts, with 5%, 28%, and 50% of training rows holding a corrupted lag.
We then added a step to the lags and measured how much of the step the forecast followed. At 5% the
forecast followed 22% of a step in all three lags, at 28% it followed 10%, and at 50% it followed
6%. The 50% arm scored 0.12 points of capacity worse than the 28% arm on rows with no labelled
event, and the 5% arm scored 0.10 better. Changing the rate during training did not help: the share
of the step the forecast follows tracks the rate of the last third of training. No arm beat the
fixed 50/50 mixture with the lagless table (B-mix).

**Verdict: not supported.** The hypothesis was that a schedule from a high rate to a low rate
(Cur-down) reaches the lower leak without the lost score. Cur-down follows 18% of the step, more
than the 10% of A0-inj, and none of the four new arms improves on A0-inj on injected harm rows. All
numbers are interim (five months, one to three seeds, no 99% claim), the sweep and the bad-data
injections cover leads under 48 hours, and NGED's logs may miss events, so rows with no labelled
event may hold events.

**Terms.** A lag is a reading of the substation's power from an earlier time (the same half-hour a
day and a week before the newest usable reading). A harm row is a forecast target outside every
event whose lags lie inside one, so a forecast that trusts its lags is misled. Injected harm rows
are harm rows in the injected scoring worlds (w200 to w202), where many events exist. Points of
capacity are the continuous ranked probability score (CRPS, from 11 quantiles) as a percentage of
the Primary's capacity. The signed-MW six are the Primaries that report signed power. B-mix is the
50/50 mixture of the stored A0-inj forecast and the lagless forecast.

![Each new arm minus A0-inj and minus B-mix](figure_1_contrasts.svg)

![Median shift as a share of a 5% step in the lags](figure_2_sweep.svg)

![A day-ahead and a 10-day forecast](figure_3_examples.svg)

![Six forecasts before real labelled events](figure_4_events.svg)

## Results

- **Arms and rates.** All arms are experiment 404's A0-inj network, trained on one injected world
  per epoch (6 epochs). The share of training rows with at least one of the 3 lags inside an
  injected event, at 2026-06 seed 0: R05 4.9%, A0-inj 27.8% (experiment 404 called it 20%), R50
  49.3%. Cur-down uses 49.3%, then 27.8%, then 4.9% for 2 epochs each; Cur-up uses the reverse. The
  shares in the other cells are 4.4% to 5.7%, 25.9% to 28.8%, and 45.4% to 49.6%.
- **Reproduction.** The re-fitted A0-inj at 2026-06 seed 0 matches the stored forecasts to at most
  1.4e-4 of capacity at every quantile.
- **Minimum detectable effect.** From the seed spread of the stored A0-inj at 2026-06 (seeds 0 to
  2), 2.8 standard deviations: 0.144 on rows with no labelled event and 0.155 on injected harm rows.
  Experiment 404's month-to-month spread (0.3 to 1.0 points) is larger; experiment 439's clean-row
  bar is 0.061.
- **Contrasts at the signed-MW six (k = 16 for the injected harm rows and rows with no labelled
  event; points of capacity, negative is better; 95% interval resampling months; cells better of
  7).**

| Arm | Absolute, no labelled event | Minus A0-inj | Minus B-mix | Absolute, injected harm | Minus A0-inj | Minus B-mix |
|---|---|---|---|---|---|---|
| A0-inj (stored) | 3.698 | | +0.350 (0.171 to 0.598) | 4.019 | | +0.365 (0.177 to 0.566) |
| R05 | 3.590 | -0.101 (-0.144 to -0.048), 7 | +0.250 (0.068 to 0.530) | 4.077 | +0.066 (0.019 to 0.113), 1 | +0.431 (0.229 to 0.679) |
| R50 | 3.793 | +0.119 (0.006 to 0.220), 2 | +0.469 (0.248 to 0.768) | 4.065 | +0.071 (-0.008 to 0.150), 3 | +0.436 (0.220 to 0.717) |
| Cur-down | 3.676 | -0.009 (-0.087 to 0.081), 5 | +0.342 (0.151 to 0.609) | 4.065 | +0.071 (0.012 to 0.130), 2 | +0.436 (0.257 to 0.662) |
| Cur-up | 3.768 | +0.092 (-0.001 to 0.186), 3 | +0.443 (0.233 to 0.745) | 4.056 | +0.061 (-0.021 to 0.152), 3 | +0.427 (0.233 to 0.691) |

- **Predictions.** P1 partly holds: at 3 lags and leads under 48 hours R05 follows 0.124 more of the
  step than A0-inj (0.224 against 0.100), but R50 follows only 0.042 less (0.058), below the 0.05
  asked; at leads of 48 to 120 hours R50 is 0.037 less than A0-inj (0.054 against 0.091) at 3 lags,
  so the second half of P1 also misses there, by a smaller margin of 0.013. P2 fails: R50 costs
  0.119 on rows with no labelled event, above the 0.061 bar, though inside the 0.144 seed spread. P3
  holds as an equivalence check: Cur-down is within 0.09 of R05 and 0.001 of A0-inj on rows with no
  labelled event, and within 0.005 of R05 on injected harm rows. P4 holds: every network alone is
  0.25 to 0.47 behind B-mix on rows with no labelled event, and R05 narrows the gap from 0.35 to
  0.25.
- **Stop rule.** It does not fire: R05 and R50 differ from A0-inj on the sweep share by more than
  0.05 at 3 lags.
- **Mixtures (descriptive, not in k).** Mixed with the lagless table, R05 is 0.073 better than B-mix
  on rows with no labelled event (-0.094 to -0.048, better in 7 of 7 cells) and 0.030 better on
  injected harm rows (-0.060 to 0.000). R50, Cur-down, and Cur-up mixed are within 0.06 of B-mix.
- **Sweep, median shift as a share of a 5% step at 3 lags, leads under 48 hours (mean of 5 months;
  range of months).** A0-inj 0.100 (0.084 to 0.128); R05 0.224 (0.161 to 0.268); R50 0.058 (0.052 to
  0.070); Cur-down 0.177 (0.152 to 0.213); Cur-up 0.067 (0.050 to 0.085). The band-width ratio stays
  within 1.006 for every arm. The control that hides all three lags was not run (experiment 404's
  sweep builds it only for its own arms starting with "B"); the lagless head follows 0 by
  construction, so the mixture follows about half of the network's share.
- **Bad-data injections (median shift as a share of the corruption, mean of 5 months).** Stuck 24
  hours: A0-inj 0.056, R05 0.102, R50 0.028, Cur-down 0.086, Cur-up 0.030. Dropout to zero for 6
  hours: 0.024, 0.032, 0.019, 0.051, 0.017. Spike of 4 times the week-on-week change: 0.024, 0.032,
  0.023, 0.036, 0.022. Spike of 20 times: 0.004, 0.014, 0.010, 0.013, 0.011. The corruption
  definitions are experiment 465's (unmerged at the time), copied with the change that the size is
  measured over the corrupted readings only.
- **Tails, signed-MW six, rows with no labelled event, raw (coverage of q01, q05, q95, q99; q99
  pinball in points of capacity).** A0-inj 0.011, 0.043, 0.813, 0.915; 0.349. B-mix 0.005, 0.025,
  0.856, 0.950; 0.259. R05 0.011, 0.043, 0.822, 0.928; 0.305. R50 0.011, 0.041, 0.796, 0.901; 0.415.
  Every network alone is under-covered at q95 and q99 and its q99 pinball loss is 0.046 (R05) to
  0.156 (R50) above B-mix's, beyond the +0.02 limit; the mixtures with the lagless table are within
  0.016 of B-mix. The full table, including injected harm rows and the 12 to 36 hour band, is in
  `work/scores/tails.csv`.
- **Avoidable curtailment (signed-MW six; million pounds summed over the cells of the five months,
  95% interval resampling cells).** At f = 0.9, R05 minus A0-inj is -0.09 (-0.44 to 0.17) and R50 is
  -0.06 (-0.35 to 0.14). At f = 0.75, R50 is +1.58 (0.25 to 3.32) and Cur-up +1.21 (0.23 to 2.42).
  At f = 0.5, R05 is -3.12 (-7.94 to 0.37), R50 +4.78 (1.46 to 8.82), and Cur-up +3.31 (0.49 to
  6.57). A0-inj minus B-mix is +0.19, +0.90, and +2.69 at f = 0.9, 0.75, and 0.5, all intervals
  including zero.
- **Serving cost.** Each arm serves one network per pass, no table to refresh, no detector or gate,
  and the regime costs nothing at serving. B-mix serves two networks (the lagged network and the
  lagless head), the cross-fitted lagless table, and a mixture step. The price of removing that
  complexity is the gap to B-mix: +0.25 (R05) to +0.47 (R50) on rows with no labelled event and
  +0.43 to +0.44 on injected harm rows.
- **What the regime would change, for an operator.** A forecast trained with a higher share of
  corrupted readings follows a sudden step in the last few days of power readings less, and is a
  little less accurate on ordinary days. The regime needs no event labels.
- **Row counts, signed-MW six, summed over the 7 cells:** 626,771 rows with no labelled event and
  1,138,838 injected harm rows. Rows inside real events, rows after events, and cold-night rows were
  not scored separately in this experiment.

## Checks

- **Median minus mean forecast per site and the error of the mean forecast.** Not computed: the
  stored forecasts hold 11 quantiles and not the mixture's mean, so neither check could be run, and
  no audit was run.
- **Triggers.** None fired by the criteria of the `ml-experiment` skill that could be checked: no
  control failed, no nested arm beat the arm that contains it, and the headline intervals are of the
  same order as experiment 404's. The R50 half of P1 missed its threshold by 0.008, which is a
  contradiction of the prediction that this page explains by the small size of the miss.
- **Reviews.** One Sonnet review of the plan and code before the first fit, one Sonnet review of the
  scoring scripts before the numbers were quoted; each finding was triaged in the pull request. The
  numbers review and the prose review follow.

## What to try next

- **Do not tune the rate further.** The rate trades leak against clean-row score; the 5% to 50%
  range spans the whole trade, and the best-scoring arm (R05) leaks the most.
- **Stage the difficulty of events** (the brief's follow-up) only if the maintainer wants the leak
  lower at constant score; this experiment gives no evidence for it.
- **Stored artefacts.** Fitted networks:
  `data/ml-experiments/00486-lagged-network-injection-rate-and-schedule/work/models/`; forecasts
  under `work/predictions/`; tables under `work/scores/`.

## Plan (written before any score was read)

**The lagged neural network (A0-inj of experiment 404) learns from training worlds in which power
readings have been corrupted by injected load-shift events, and the share of training rows with a
corrupted lag has never been varied.** Experiment 404 chose it once. This experiment trains the same
network, on the same rows and with the same loss, on worlds with a different share, and on two
schedules that change the share during training. It then asks whether the network's median still
follows a step in its lags, and what that costs on rows with no labelled event.

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

- **Rate definition.** Spikes count as events, and the shares are for 2026-06 seed 0 (the write-up
  reports the share in every cell). The share counts training rows with at least one of the three
  origin lags inside an injected event, measured on the 2026-06 training pool at seed 0
  (`e00486_fit.py share`). The brief's 5%, 20%, and 50% are the labels; the stored world measures
  27.8% by this definition. The new worlds use experiment 428's injection with the event coverage
  changed to 0.016 (R05) and 0.18 (R50), set by measuring candidate worlds (0.02, 0.04, 0.20, 0.30
  gave 6.4%, 11.4%, 54.1%, and 73.4%).
- **Worlds are per seed and new for each rate.** Training seeds 100 to 102; the scoring worlds w200
  to w202 and the clean world w-1 are experiment 404's, unchanged, so every arm is comparable with
  the stored A0-inj and B-mix.
- **Reproduction.** `A0-repro` re-fitted A0-inj for 2026-06 seed 0: on world w200 its quantiles
  differ from the stored A0-inj by at most 1.4e-4 of capacity. The check passes if every arm-level
  score matches the stored one within the minimum detectable effect; it is also the noise floor. No
  other cell is refitted.
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
events, the harm table is optimistic, and real-event slices undercount. Weather inputs are the ECMWF
ENS mean and spread of experiment 404 (no past-weather lags enter the network, so the past-weather
caveat does not apply).

## Reading earlier advice

The plan follows experiment 404's advice to use the fixed 50/50 mixture B-mix as the reference and
to report the absolute score and event counts beside every contrast. It departs from nothing.
