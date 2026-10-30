---
id: "00407"
slug: pooled-mixture-density-network-tail-arms-and-recency-weights
title: >-
  Which change to the pooled mixture density network's output, or to its row weights, improves the
  tails of the Primary demand forecast beyond a global widening factor?
theme: calibration-and-uncertainty
also_relevant_to:
  - production-reference-and-best-guess
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: forecasts, fitted networks, widening factors, and small result tables in this
  experiment's private folder only; experiments 397 and 406 are read and never written. Designs:
  more than one defensible (output family, component count, how the widening is fitted, row-weight
  half-life), so every arm is registered. Callers: the scripts import experiments 87, 48, 377, 381,
  and 397 and the `studies` evaluation code, whose callers sit in those folders. Serving path and
  degradation rule: not touched (this repository holds neither).
hypothesis: >-
  The pooled plain network's tails at the six signed-MW Primaries are too narrow (q95 coverage 0.855
  and q99 coverage 0.941 in experiment 397), and at least one of five changes lowers the tail loss
  below a globally widened pooled plain network without raising the CRPS: 5 mixture components,
  split-normal (skewed) components, a widening fitted per lead band inside each refit, a quantile
  head trained with the pinball loss, or a different half-life for the row weights.
prediction: >-
  Written before any score is read. The reference is `ctrl_w115`, the pooled plain network with its
  quantiles q01, q05, q95, and q99 widened by 1.15 about the median (q95 coverage 0.891 and q99
  coverage 0.960 at the signed-MW six in experiment 397). T1: with 5 components and the same 1.15
  widening, the pinball loss at q99 at the signed-MW six is at least 3% below the reference's, and
  the CRPS is no more than 0.02 points above it. T2: the split-normal arm, unwidened, reaches q95
  coverage of at least 0.90 at the signed-MW six, with a CRPS no more than 0.02 points above the
  reference's. T3: the per-lead-band widening of the pooled plain network brings q95 coverage at the
  signed-MW six within 0.03 of 0.95 (0.92 to 0.98) and q99 coverage within 0.02 of 0.99, with a
  CRPS no more than 0.02 points above the reference's. T4: the pinball quantile head's CRPS lies
  within 0.03 points of the reference's at the signed-MW six, and its pinball loss at q01 and q05 on
  sunny low-demand half-hours is no higher than the reference's. H1: a row-weight half-life of 1
  year is no better than the 0.5-year control by more than 0.02 points at either group. H2: a
  half-life of 0.25 years is no better than the control by more than 0.02 points at either group.
  C6: the D15 and D18 start cut lowers the CRPS at D15 and D18 together for the group-attention
  network (arm A0), and moves the signed-MW six and the MVA-only eight by less than 0.03 points.
  C7: with the cut, A0 minus the per-Primary plain network is at or below -0.10 points at the
  signed-MW six (experiment 403 found -0.185 without the cut).
decision_rule: >-
  Each contrast is read per group (signed-MW six; MVA-only eight; D15 and D18, report-only) on CRPS
  at days 1 to 10, with the month-block bootstrap of `studies.eval_protocol` at 95% and 20,000
  resamples, and alongside the tail table (coverage and pinball loss at q01, q05, q95, and q99, on
  all half-hours and on sunny low-demand half-hours). The five screen months give wide intervals, so
  every number is interim, and a result ranks changes for further testing without supporting a
  claim of benefit. An arm is carried forward if its prediction holds and its CRPS interval does
  not lie wholly above zero. The hypothesis is supported if at least one of T1 to T4 holds; it is
  not supported if all four fail. The half-life arms and C6 and C7 are reported on their own and do
  not count towards that decision. The smallest change worth carrying forward is a q99 pinball loss
  3% lower at the signed-MW six at no CRPS loss, because a different output family adds a model to
  the pipeline.
benchmark: >-
  The widened pooled plain network (`ctrl_w115`) fitted in this experiment, the pooled plain network
  and A0 of experiment 397 without the start cut, and the per-Primary plain network of experiment
  406, all on the same half-hours.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, coverage and pinball loss at q01, q05,
  q95, and q99, in points of effective capacity, from `studies.eval_protocol`; leads 0.5 to 336
  hours; the headline band is days 1 to 10.
code: ml-experiments/00407-pooled-mixture-density-network-tail-arms-and-recency-weights/
---

**Take-home message (to be written when the result is known).**

This page is the plan. It is committed before any network is fitted, so that its commit order proves
the predictions came first.

## Question and design

**We change one thing at a time in the pooled plain network and compare each change with the same
network widened by a global factor.** [Experiment
397](../00397-plain-and-attention-mixture-density-network-for-primary-demand/result.md) found that
the networks cover too few outcomes with their 95% quantile at the six signed-MW Primaries, and
that a widening of 1.15 only lifts the coverage to about 0.89. This experiment tests four ways to
fix the tails and two row-weight half-lives.

- **The networks.** The pooled plain network of experiment 397 (setting `g09`, summary scheme, an
  8-wide Primary embedding) on its five screen folds (test months 2025-10, 2025-12, 2026-01,
  2026-03, and 2026-06), with three seeds averaged in quantile space and the same refits and
  scoring rows.
- **`ctrl`.** 3 Gaussian components, row-weight half-life 0.5 years. Its widened version
  `ctrl_w115` is the reference.
- **`k5`.** 5 Gaussian components.
- **`skew`.** 3 split-normal components. A split-normal component has one standard deviation below
  its mode and another above it, so a component can be skewed. Each component's density and
  distribution function have closed forms, so the quantiles come from the same bisection as the
  Gaussian mixture. A Student-t family is not tested, because the Student-t distribution function
  has no closed form in the neural-network library used.
- **`ctrl_bw`: widening fitted per lead band.** For each of five lead bands (day 1, days 2 to 3,
  days 4 to 7, days 8 to 10, days 11 to 14) and each of q01, q05, q95, and q99, the factor that
  makes the share of held-out rows below the widened quantile equal the nominal level. The held-out
  rows are the one week in every 12 that the network did not train on (the same rows that stop
  its training), so the widening is fitted inside each refit on rows the network has not seen. The
  factor is limited to between 0.5 and 4. Every arm also gets its `_bw` version, and the 1.15 `_w115`
  version.
- **`qhead`.** 11 quantiles output directly (a median and cumulative positive steps either side, so
  they cannot cross) and trained with the pinball loss averaged over the 11 levels.
- **`hl365` and `hl91`.** `ctrl` with row weights that halve every 1 year and every 0.25 years
  (the control halves every 0.5 years, as the XGBoost member does).
- **`a0`.** Experiment 397's group-attention network with uniform weights, fitted with the start cut,
  to redo experiment 403's contrast.
- **The start cut.** D15 and D18 lose every training row before 2025-03-01, and every row whose
  weekly lag reaches back before that date (maintainer's rule of 2026-10-30, from [experiment
  399](../00399-xgboost-short-history-start-cut-d15-d18/result.md)). The scoring rows start in
  2025-10, so no scoring actual or inference input is affected.
- **Input ports from XGBoost** (the solar summary column and sunny flag with a 1-year half-life,
  and trailing temperature means) are not in this experiment and follow as a separate experiment.

## Fit counts, time, and memory

- **Fits.** `ctrl`, `k5`, `skew`, `qhead`, `hl365`, and `hl91`: 6 arms by 5 folds by 3 seeds, so 90
  pooled fits, and 15 more for `a0`. The widenings need no refit.
- **Time.** Experiment 397 fitted a pooled plain network in about 50 s and A0 in about 130 s on the
  GPU, so the experiment is expected to take about 2.5 hours.
- **Memory.** One GPU process with at most 8 GB of host memory and one feeder core. The per-Primary
  raw rows that experiment 397 wrote are read once per fold, a fold's tables are built once per
  half-life and shared by every arm and seed with that half-life, and the held-out rows' widening
  is computed once per fit.

## Checks the write-up reports

- Absolute CRPS of every arm beside experiment 397's uncut networks, per group and per Primary.
- **Tails first.** Coverage and pinball loss at q01, q05, q95, and q99, on all half-hours and on
  sunny low-demand half-hours, which are the sunny half-hours (experiment 381's flag) whose
  reference median is in the lowest quarter for the Primary. The rows are chosen from the
  forecast, never from the outcome.
- **Example forecasts.** For D10 (signed-MW) and D4 (MVA-only), the reference's median, 50% band,
  and 90% band against what happened: one day-ahead forecast and one full 14-day forecast from a
  single run, in winter (made on 2026-01-12) and summer (made on 2026-06-15).
- **Unlogged load-shift events.** NGED says load-shift events may be missing from its logs, so the
  training rows and the scoring rows can hold events no label covers.

## Caveats registered before the scores

- **Widening factors are fitted per seed and applied per seed**, and the scorer then averages the
  widened quantiles over the three seeds, so the averaged `_bw` forecast is not exactly calibrated
  at the nominal level.
- **The widening is fitted on the unweighted held-out weeks across the whole 24-month window.** For
  `hl91`, whose network concentrates on recent rows, the factors reflect older rows that the network
  fits less closely.
- **The sunny low-demand quarter is taken over all leads of the sunny half-hours**, and the tail
  table then uses the days 1 to 10 rows of it.

## Counting towards the demand-v2 stop rule

**This experiment does not count.** It tests changes to the tails of an existing model, not a new
demand model, so no idea's count rises whatever the verdict.

## Problems caused by the move

None so far.
