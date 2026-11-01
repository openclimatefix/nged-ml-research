---
id: "00495"
slug: lagged-network-hard-negative-mining-on-injected-events
title: >-
  Does training the lagged network a second time on the injected rows where it follows an event in
  its lags lower how far its median follows a step in its lags?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
date: 2026-11-01
size: >-
  medium. Stored: one fitted network per cell, one hard-row table per cell, forecasts and sweep
  tables in this experiment's private folder only; experiments 404, 428, and 486 are read and never
  written. Serving path: not touched (this repository holds none), and the regime costs nothing at
  serving. One design is defensible because the brief fixes the arm. Callers: the scripts import
  the folders of experiments 404, 428, 402, 397, 422, and 87 unchanged and copy experiment 486's
  scoring scripts. One Sonnet plan review, one Sonnet code review before the first fit, one Sonnet
  results review, and two Sonnet write-up reviews.
hypothesis: >-
  The lagged network (A0-inj) is trained on a world where about 28% of training rows have a lag
  inside an injected event. Its loss is dominated by the rows with clean lags. Training its last 3
  epochs with the rows where it follows an injected step wrongly, or its loss is high, shown 3 times
  each (hard-negative mining, arm HN) lowers how much of a step in its lags appears in its median
  forecast, because those are the rows that teach it to discount a jump in its lags.
prediction: >-
  Written before any score is read. P1 (the falsifiable one): at 3 lags and leads under 48 hours,
  HN's median follows at most 0.10 of a 5%-of-capacity step, which is at least 0.05 less than
  A0-inj's stored share of 0.100 in this harness (experiment 486 measured 0.100 and 0.058 for
  R50); the pass bar of the brief is at most 0.10 together with a cost on rows with no labelled
  event under 0.05 points of capacity against A0-inj. P2: the injected-harm-row score is within the
  minimum detectable effect of A0-inj (an equivalence check, never a pass). P3: HN does not beat
  B-mix on rows with no labelled event. The prior is low: every single network stayed 0.25 to 0.47
  points behind B-mix in experiment 486, and B-mix's lead is a combination effect, so this
  experiment is expected to cut the following, not to close the gap to B-mix.
decision_rule: >-
  Four contrasts, Bonferroni k = 4 (level 1 - 0.05/4): HN minus A0-inj and HN minus B-mix, each on
  rows with no labelled event and on injected harm rows (scoring worlds w200 to w202), at the
  signed-MW six, in points of capacity. The MVA-only eight, real harm rows, and rows inside real
  events are descriptive. The minimum detectable effect is printed before any contrast is read.
  Pass bar: the median follows at most 0.10 of a 5% step at 3 lags, with a clean-row cost under 0.05
  against A0-inj. The smallest gain worth porting is zero on injected harm rows with an interval
  that excludes zero, no loss on rows with no labelled event beyond the minimum detectable effect,
  and q99 pinball loss within +0.02, because mining adds training cost only. Stop rule (fixed
  before any score is read): HN sits on the trade-off curve of experiment 486 if its clean-row
  score minus A0-inj is within 0.061 points (experiment 439's clean-row bar) of the value of the
  piecewise-linear curve through the points (follow share, clean-row score minus A0-inj) of R05
  (0.224, -0.101), A0-inj (0.100, 0), and R50 (0.058, +0.119) at HN's follow share. If HN sits on
  that curve, the injection-regime family is closed.
benchmark: >-
  The stored A0-inj and B-mix forecasts of experiment 404 (seed 0 in five months; seeds 1 and 2 in
  2026-06), read from experiment 404's private folder and never refitted. Experiment 486's R05 and
  R50 results are quoted from its merged page, not rerun.
row_set_id: >-
  Experiment 404's rows: the intersection of finite actuals with flagged zeros dropped.
metric: >-
  Mean CRPS from the 11 quantiles in points of effective capacity, absolute beside every contrast;
  q01, q05, q95, and q99 coverage and pinball loss; the readings-level encroachment sweep of
  experiment 404 on HN, A0-inj, and B-mix; avoidable curtailment at f = 0.9, 0.75, and 0.5.
code: ml-experiments/00495-lagged-network-hard-negative-mining-on-injected-events/
---
**Plan, written before any score was read.** Experiment 486 showed that the share of training rows
with a corrupted lag trades how far the lagged network's median follows a step in its lags against
its score on ordinary rows. Hard-negative mining is the one regime of the injection family that
experiment does not duplicate: it keeps the 28% share and changes which rows the last epochs see.

## Arm and mining rule

- **HN.** Experiment 404's A0-inj network (mixture density network, three origin lags, experiment
  397's recency weights, monthly refit, the lagless table frozen) trained on the stored injected
  world of experiment 428 for 3 epochs. The epoch-3 checkpoint is saved. A mining pass scores every
  training row once. The last 3 epochs shuffle the training rows together with two extra copies of
  each hard row, so each hard row appears 3 times per epoch.
- **Hard row.** (a) The row has a lag inside an injected event, its three lags differ from the clean
  readings by at least 0.5% of capacity on average (the step), and the median on the injected lags
  minus the median on the clean lags exceeds 0.25 of the step; or (b) its pinball loss against the
  clean target (mean of the 11 quantiles) exceeds the 80th percentile of its class (a lag inside an
  injected event, or none). The hard set is computed once.
- **Why 3 plus 3 epochs.** The brief says 60% of 6 epochs (3.6) and an epoch-3 checkpoint; the
  checkpoint falls on a whole epoch, so the split is 3 plain epochs and 3 mined epochs. The first 3
  epochs reproduce the stored A0-inj recipe, whose re-fit matched the stored scores to 1.4e-4 of
  capacity in experiment 486.
- **Dropped from the brief.** Arms W3 and W3+HN (weight 3 on event rows), dropped by the load-shift
  Discuss session after experiment 486 (PR 148) showed that weight 3 on 20% event rows is about the
  effective share of R50. The earlier brief's conditional dependence on brief 1 is therefore gone.
- **Reproduction.** Not refitted: experiment 486's `A0-repro` already matched the stored A0-inj to
  1.4e-4 of capacity; HN's first 3 epochs use that recipe.

## Cells, row sets, and scoring

- **Cells.** Seven: five scored months at seed 0 (2025-10, 2025-12, 2026-01, 2026-03, 2026-06) and
  2026-06 at seeds 1 and 2, as for experiments 404 and 486. Seven full refits. The scoring worlds
  w200 to w202 and the clean world w-1 are experiment 404's, unchanged.
- **Minimum detectable effect.** Both sources, printed first: the seed spread of the stored A0-inj
  and B-mix at 2026-06 (seeds 0 to 2), and experiment 404's month-to-month spread (0.4 to 1.0 points
  on harm rows, 0.3 to 0.9 inside events, 0.3 to 0.8 on rows with no labelled event). Experiment
  439's clean-row bar of 0.061 judges clean-row equivalence. Harm-row claims are judged on the
  injected harm rows; real harm rows and rows inside real events are descriptive.
- **Contrasts.** The four of the decision rule, as month-block bootstraps with the sign count over
  the five months. The 50/50 mixture of HN with the lagless table, and the MVA-only eight, are
  descriptive and outside k.
- **Reported beside CRPS.** q01, q05, q95, and q99 coverage and pinball loss (raw and after table P
  with the 1.15 widening, with the 12 to 36 hour band); the encroachment sweep of experiment 404
  and the bad-data injections of experiment 465 (stuck 24 hours, dropout to zero 6 hours, spikes of
  4 and 20 times) on HN, A0-inj, and B-mix; avoidable curtailment at f = 0.9, 0.75, and 0.5;
  row counts per row class; a figure of the hard set by lead band and by event depth; the plots of
  the standing rules (a day-ahead and a 14-day forecast, and at least 6 real labelled events across
  both metering groups, with generators anonymised).
- **Serving cost.** One network per pass, no table, mixture, or detector, and no change at serving
  from the mining pass (training only). B-mix serves two networks, a stored lagless table, and a
  mixture step. The write-up states the score gap to B-mix on rows with no labelled event and on
  injected harm rows as the price of removing that complexity.
- **Operator sentence.** "The network was shown its own worst mistakes about events and trained on
  them more." The regime needs no event labels.

## Compute

- **Fits.** 7 fits at about 20 to 36 s plus one mining pass each (a forward pass over the training
  rows, minutes at most), about 0.1 GPU-hours. One GPU process under 8 GB and one feeder core. A fit
  starts only when `free -g` shows more than 20 GB available. The first fit is timed and the total
  reported to the manager before the rest.
- **Reused.** The stored A0-inj and B-mix forecasts on all 7 cells, and experiment 486's results.
- **Scoring.** Forecasts on w200 to w202 and w-1, the sweep, the corruption injections,
  curtailment, and the bootstrap dominate the wall time.

## Standing practices followed

Monthly refits; cleaned data with flagged `substation_zero` half-hours dropped (the count per cell
is printed with the scoring); the D15 and D18 start cut before 2025-03-01; fitted networks saved
under this experiment's data folder; stored forecasts reused; tails and curtailment reported; the
unlogged-event caveat stated: NGED's logs may miss events, so rows with no labelled event may hold
events, the harm table is optimistic, and real-event slices undercount. Weather inputs are the ECMWF
ENS mean and spread of experiment 404 (no past-weather lags enter the network, so the past-weather
caveat does not apply).

## Reading earlier advice

The plan follows experiment 486's advice to keep B-mix as the reference and not to tune the share
of injected rows, and experiment 488's finding that real labelled events add nothing detectable
beyond injected events. It departs from nothing. Experiment 484 weights hard rows in demand network
fine-tuning, a different network and loss; it is cited as related work only.
