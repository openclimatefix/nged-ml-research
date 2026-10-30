---
id: "00406"
slug: pooled-mixture-density-network-without-embedding-and-with-short-history
title: >-
  Does the pooling gain of the mixture density network come from sharing parameters or from the
  Primary embedding, and does it grow when each Primary's training history is short?
theme: hierarchy-and-pooling
also_relevant_to:
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: forecasts, fitted networks, and small result tables in this experiment's private
  folder only; experiments 397 and 403 are read and never written. Designs: more than one
  defensible (which history lengths, which controls), so every arm is registered. Callers: the
  scripts import experiments 87, 48, 377, 381, and 397 and the `studies` evaluation code, whose
  callers sit in those folders. Serving path and degradation rule: not touched (this repository
  holds neither); no change to what is stored by earlier experiments.
hypothesis: >-
  Most of the gain that one pooled mixture density network shows over one network per Primary
  (experiment 403) comes from sharing parameters, not from the learned Primary embedding, and the
  gain does not grow when each Primary's training history is cut to 3, 6, or 12 months.
prediction: >-
  Written before any score is read. P1: the pooled network with no embedding minus the per-Primary
  network is at or below -0.123 points (two-thirds of experiment 403's -0.185) at the signed-MW six,
  both fitted with the D15 and D18 start cut. P2: the embedding adds little: the pooled network with
  the embedding minus the pooled network without it lies between -0.05 and +0.05 points at the
  signed-MW six and at the MVA-only eight. P3: the pooled gain does not grow as history shortens:
  the contrast of the pooled network with the embedding minus the per-Primary network at 3 months
  is no more than 0.10 points below the same contrast at the full 24 months, at both groups
  (experiment 403 found no correlation between gain and months of history, -0.02). P4: every
  shortening of the history raises both networks' CRPS: at 3 months each is at least 0.10 points
  above its own full-history value at the signed-MW six. P5: the D15 and D18 start cut lowers the
  CRPS at D15 and D18 together (cut minus uncut below zero) for both the per-Primary and the pooled
  network, and moves the signed-MW six and MVA-only eight by less than 0.03 points.
decision_rule: >-
  Each contrast is read per group (signed-MW six; MVA-only eight; D15 and D18, report-only) on CRPS
  at days 1 to 10, with the month-block bootstrap of `studies.eval_protocol` at 95% and 20,000
  resamples. The five screen months give wide intervals, so every number is interim and no claim of
  benefit rests on it. C1 pooled without embedding minus per-Primary; C2 pooled without embedding
  minus pooled with embedding; C3 pooled with embedding minus per-Primary; C4 the C3 contrast at 3,
  6, and 12 months of history; C5 each arm with the start cut minus the same arm from experiment 397
  without it. The hypothesis is supported if P1, P2, and P3 all hold. It is partly supported if P1
  holds and P3 fails, or if P3 holds and P1 fails. It is not supported if both fail. The smallest
  gain worth acting on is any C1 or C3 gain whose interval excludes zero, because one pooled network
  costs one fit instead of 16.
benchmark: >-
  The per-Primary and pooled plain networks of experiment 397 (without the start cut), on the same
  half-hours.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, in points of effective capacity, from
  `studies.eval_protocol`; leads 0.5 to 336 hours; the headline band is days 1 to 10.
code: ml-experiments/00406-pooled-mixture-density-network-without-embedding-and-with-short-history/
---

**Take-home message (to be written when the result is known).**

This page is the plan. It is committed before any network is fitted, so that its commit order proves
the predictions came first.

## Question and design

**We split the pooling gain into two parts and test whether it depends on history length.** [Campagne
et al. (2026)](https://doi.org/10.48550/arXiv.2610.35074) separate the gain from sharing parameters
across sites from the gain from graph edges. [Experiment
403](../00403-pooled-versus-per-primary-mixture-density-network/result.md) found that one pooled
network scores lower than 16 per-Primary networks. This experiment asks how much of that gain
survives when the pooled network has no Primary input at all, and whether the gain grows where
history is short.

- **The networks.** Experiment 397's plain mixture density network (setting `g09`, summary scheme,
  3 Gaussian components), on its five screen folds (test months 2025-10, 2025-12, 2026-01, 2026-03,
  and 2026-06), with three seeds averaged in quantile space, and the same folds, refits, row
  weights, and scoring rows. No XGBoost model is used.
- **Arms at the full 24-month window (with the D15 and D18 start cut).** `per_primary` (one network
  per Primary), `pooled_emb` (one network with an 8-wide Primary embedding), and `pooled_noemb` (one
  network with no Primary input).
- **Arms at 3, 6, and 12 months.** `per_primary_h<H>` and `pooled_emb_h<H>`: each fold trains only
  on each Primary's last `H` months of rows before the refit.
- **The start cut.** D15 and D18 lose every training row before 2025-03-01, and every row whose
  weekly lag reaches back before that date (maintainer's rule of 2026-10-30, from [experiment
  399](../00399-xgboost-short-history-start-cut-d15-d18/result.md)). The scoring rows start in
  2025-10, so no scoring actual or inference input is affected.
- **Controls without the cut.** Experiment 397's `plain_own` and `plain_pooled` forecasts, read
  and never written, give contrast C5. They are the same networks without the start cut.

## Fit counts, time, and memory

- **Fits.** `pooled_noemb`, `pooled_emb`, and `per_primary` at the full window: 3 arms by 5 folds by
  3 seeds, so 15 pooled fits for each of the two pooled arms and 15 sets of 16 per-Primary
  networks. Three history lengths add 45 pooled fits and 45 sets of 16.
- **Time.** Experiment 397 fitted one pooled plain network in about 50 s and one set of 16
  per-Primary networks in about 50 s on the GPU. Shorter histories train on fewer rows, so the
  whole experiment is expected to take under 3 hours.
- **Memory.** One GPU process with at most 8 GB of host memory (the 397 screen peaked at 7.9 GB for
  a 4.2-million-row fold), and one feeder core. A fold's raw rows are the per-Primary files that
  experiment 397 wrote once, so no table is rebuilt per arm.

## Checks the write-up reports

- Absolute CRPS of every arm beside experiment 397's uncut networks, per group and per Primary.
- **Example forecasts.** For D10 (signed-MW) and D4 (MVA-only), the pooled network's median, 50%
  band, and 90% band against what happened: one day-ahead forecast and one full 14-day forecast
  from a single run, in winter (made on 2026-01-12) and summer (made on 2026-06-15).
- C4 per Primary against months of history, with the correlation at each history length.
- **Unlogged load-shift events.** NGED says load-shift events may be missing from its logs.
  Training rows in every arm can therefore hold events no label covers, so a gain measured here is
  measured on rows that may include them.

## Counting towards the demand-v2 stop rule

**This experiment does not count.** It tests where the pooling gain comes from, not a new demand
model, so no idea's count rises whatever the verdict.

## Problems caused by the move

None so far.
