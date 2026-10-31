---
id: "00415"
slug: pooled-mixture-density-network-at-bulk-and-grid-supply-points
title: >-
  Does the pooled plain mixture density network forecast demand at two bulk supply points and two
  grid supply points as well as it does at Primaries, and do the Primaries beneath a node help?
theme: hierarchy-and-pooling
also_relevant_to:
  - production-reference-and-best-guess
  - calibration-and-uncertainty
status: done
kind: other
protocol: demand-v2
size: >-
  medium. Stored: network weights and per-arm forecast tables in this experiment's private folder;
  experiments 87, 386, 397, and 403 are read and never written. Production serving path: not touched
  (this repository holds none). Degradation rule: not touched. Designs: more than one would test the
  question (fit the nodes in the Primary network or alone, feed the children as lags or as
  forecasts); the plan fixes one. Callers: the scripts import experiment 397's modules and
  experiment 87's rows, which sit in earlier folders.
hypothesis: >-
  A pooled plain mixture density network fitted across the 16 Primaries and the 4 nodes (D13 and D17
  are bulk supply points; D1 and D6 are grid supply points) with a node-type input scores lower
  CRPS at days 1 to 10 at the nodes than the same network fitted at the 16 Primaries only and
  applied to the nodes, and is not worse than the stored monthly XGBoost control of experiment 386
  by more than 0.30 points of effective capacity at either group.
prediction: >-
  Written before any score is read. P1. N1 minus P0 at the nodes is negative at both groups, between
  -1.5 and -0.3 points, because P0 has never seen a node's level, scale, or reverse flow. P2. N1
  minus the stored monthly XGBoost control (experiment 386 stage D, widened 1.15) is between -0.1
  and +0.6 points at the bulk supply points and between 0.0 and +1.5 at the grid supply points: the
  grid supply points aggregate wind, solar, and storage, as in experiment 386. P3. N2 minus N1 at
  each group of nodes is between -0.15 and +0.05 points, and N2 minus N2rw is no larger in
  size than N2 minus N1 (the neighbours act as a noise average). Experiments 161 and 204 found that
  the children's forecasts added at most 0.04 points to a GAM at the two bulk supply points, so
  the children's lags are unlikely to help more. P4. q95 coverage of N1 at the nodes is 0.84 to 0.92
  (experiment 397's Primary networks cover 0.84 to 0.86), and q05 coverage 0.06 to 0.10. P5. N1
  minus P0 at the 14 countable Primaries is within +-0.05 points (adding nodes does not hurt them).
decision_rule: >-
  See "Stop rule and decision rule" in the body, registered before any score is read.
benchmark: >-
  Stored forecasts of experiment 386 on `fold_a_daily_gap_v1`: BG2 (`b_weekly`, weekly refit), the
  monthly XGBoost control (raw and widened by 1.15), the GAM stack, `manual_heuristic`, and
  `climatology`, restricted to the rows of the folds' test months. The Primaries' controls are the P0
  fits of this experiment.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles, in points of effective
  capacity, from `studies.eval_protocol`; leads 0.5 to 336 hours; headline band days 1 to 10; q01,
  q05, q95, and q99 coverage and pinball loss beside it.
verdict: supported
code: ml-experiments/00415-pooled-mixture-density-network-at-bulk-and-grid-supply-points/
best_arm: >-
  N1 (the pooled plain network fitted at the 16 Primaries and the 4 nodes): CRPS 4.52 (D13), 6.80
  (D17), 14.41 (D1), and 14.26 (D6) at days 1 to 10, four test months
headline: >-
  N1 minus BG2, days 1 to 10, CRPS in points of effective capacity: bulk supply points -0.47 (95%
  interval -0.67 to -0.23), grid supply points -0.02 (-0.15 to +0.13); N1 is a 3-seed average and
  BG2 a single fit. N1 minus the P0 network fitted at the Primaries only: -4.1 (-5.7 to -3.1) and
  -15.2 (-23.2 to -10.6), where P0's input scaling makes the grid figure unfair to P0. Four test
  months, so every interval is interim.
take_home_message: >-
  A pooled neural network trained on the readings of two bulk supply points and two grid supply
  points together with the 16 Primaries forecasts the bulk supply points a little better than the
  weekly XGBoost forecast (0.47 points of capacity lower, in all four months) and the grid supply
  points about as well, with very wide bands. A copy fitted at the Primaries alone fails at these
  points, partly because of how its inputs were scaled. Giving the network the readings of the
  Primaries beneath a node did not help.
lessons:
  - >-
    A network fitted at the Primaries only scores 4.1 (bulk) and 15.2 (grid) points worse than one
    that also trains on the nodes, with q95 coverage of 0.60 to 0.81; the grid figure is inflated
    by P0's input scaling (audit finding 5), so the size of the benefit of node training is not
    measured.
  - >-
    The Primaries beneath a node add nothing as inputs: N2 minus N1 is +0.01 (bulk) and +0.32 (grid,
    within refit noise), and randomly chosen Primaries (N2rw) score the same as N1.
advice:
  - >-
    Fit the pooled plain network on the nodes' own readings together with the Primaries, and do not
    apply a network fitted at the Primaries only to a bulk or grid supply point without scaling
    the node's inputs by its own statistics (C1).
  - >-
    Do not add the lags of the Primaries beneath a node to the network at bulk or grid supply
    points, because neither the true nor a random set of Primaries lowered the score (C3, C4).
  - >-
    Report the network's upper-tail coverage at the bulk supply points before using it there,
    because its q95 coverage is 0.88 at D13 against a nominal 0.95 (tails table).
production_status: not-for-production
adopted_in: none
date: 2026-10-30
---

**Take-home message: one pooled neural network, fitted on the 16 Primaries and the four supply
points together, forecasts the bulk supply points better than the weekly XGBoost forecast does.** At
the grid supply points it ties the XGBoost forecast, and only with prediction bands about 100 points
of capacity wide. The Primaries beneath a supply point do not help as inputs.

We asked whether the pooled plain mixture density network of [experiment
403](../00403-pooled-versus-per-primary-mixture-density-network/result.md) works at two bulk supply
points (BSPs: D13, D17) and two grid supply points (GSPs: D1, D6). We call these four the nodes.
They are the nodes of [experiment
386](../00386-bsp-gsp-demand-v1-top-down-bg2-gam-simple-xgboost/result.md). A BSP connects the 132
kV electricity network to the 33 kV electricity network, and a GSP connects the transmission network
to the 132 kV electricity network. A Primary is one of the 16 primary substations that experiments
397 and 403 forecast. "Pooled" means one neural network fitted jointly to many substations. A
mixture density network is a neural network that outputs a probability distribution as a mixture of
three bell curves. BG2 is the weekly-refitted XGBoost forecast of experiment 87.

For each of four test months we fitted the network on the 24 months before the month and scored it
on that month. We compared it with a copy fitted at the Primaries only and with the forecasts that
experiment 386 stored. The verdict is `supported`. "Interim" applies to every interval, because four
test months give wide intervals. The BSP gain over BG2 is the one claim made.

![Figure 1: CRPS per node and arm](figure_1_crps_per_node.svg)

![Figure 2: day-ahead forecast at a BSP](figure_2_day_ahead_D13.svg)

![Figure 3: day-ahead forecast at a GSP](figure_3_day_ahead_D1.svg)

![Figure 4: 14-day forecast at a BSP](figure_4_14_day_D13.svg)

![Figure 5: 14-day forecast at a GSP](figure_5_14_day_D1.svg)

## Stop rule and decision rule

We registered these rules before reading any score.

- **Stop rule A.** Stop if N1 is worse than `climatology` at 3 or more of the 4 nodes in both stage
  1 test months.
- **Stop rule B.** Stop if N1 minus P0 at the 14 countable Primaries exceeds +0.10 points in both
  stage 1 test months.
- **Stop rule C.** Skip stage 3 if N2 minus N1 is 0 or higher in both stage 2 test months at both
  groups of nodes.
- **Decision rule.** The hypothesis is `supported` if N1 beats P0 at the nodes at both groups and is
  not worse than the monthly XGBoost control by more than 0.30 points at either group.

## Arms and results

CRPS is the continuous ranked probability score estimated from 11 quantiles, in points of effective
capacity (the percentage of the substation's effective capacity), at days 1 to 10. The four test
months are 2025-10, 2026-01, 2026-03, and 2026-06. Each CRPS is the average of 3 seeds, averaged
level by level over the quantiles. P0 and N1 count 14 Primaries as countable, because D15 and D18
share a substation number and lack data before 2025-03-01. The 95% intervals come from resampling
the four test months.

- **P0:** the pooled network fitted at the 16 Primaries only, applied to a node with the mean of
  the 16 Primary embeddings and the node's own target statistics.
- **N1:** fitted at the 16 Primaries and the 4 nodes, with a node-type input.
- **N2:** N1 plus the mean of three lagged readings of the Primaries beneath the node (origin lags:
  the newest reading before the forecast origin, and the readings at the target's time of day on the
  latest day and the latest week before the origin) (7 beneath D13 and D1, 9 beneath D17 and D6;
  NGED's hierarchy file) and the share of those Primaries with a reading.
- **N2rw:** N2 with a random set of the same size in place of the true Primaries (overlap with the
  true set: 0 of 7 at D13 and D1, 2 of 9 at D17 and D6).
- **CRPS (D13, D17, D1, D6):** P0 6.65, 12.93, 33.09, 25.92; N1 4.52, 6.80, 14.41, 14.26; N2 4.52,
  6.82, 14.51, 14.81; N2rw 4.66, 6.80, 14.35, 14.35; BG2 4.94, 7.31, 14.52, 14.19; monthly XGBoost
  control widened by 1.15 4.99, 7.39, 14.70, 14.33; GAM stack 5.66, 8.93, 17.77, 15.81;
  `manual_heuristic` 5.80, 8.82, 19.18, 15.92; `climatology` 6.06, 8.89, 18.29, 15.37.
- **C1, N1 minus P0:** BSPs -4.13 (-5.66 to -3.09); GSPs -15.17 (-23.16 to
-10.55). P0 is not a fair transfer, because its lag inputs are scaled by the Primaries' statistics,
and a grid supply point's lags (net flow near zero, often negative) lie far outside them. At D1 P0's
median has a bias of +33 points and a correlation of -0.06 with the actual; its q95 coverage (the
share of actuals below its 95th percentile; 0.95 is perfect) is 0.60 to 0.81 across the four nodes.
Treat -4.1 as the better bound on the benefit of node training and -15.2 as inflated by an amount we
did not measure.
- **C2, N1 minus the widened monthly XGBoost control:** BSPs -0.53 (-0.76 to -0.28); GSPs -0.18
  (-0.68 to +0.23). N1 minus BG2: BSPs -0.47 (-0.67 to -0.23); GSPs -0.02 (-0.15 to +0.13). N1 is a
  3-seed average and BG2 and the control are single fits. Single-seed N1 minus BG2 averages -0.39 at
the BSPs and +0.21 at the GSPs, so the BSP gain holds for a single seed, and the GSP tie holds only
once the three seeds are averaged. BG2 refits weekly and N1 refits once per test month, so the BG2
contrast depends on cadence. The BSP gain keeps its sign when either BSP is left out and in each
test month; the GSP contrast changes sign with the node left out (D6 +0.08, D1 -0.11).
- **C3, N2 minus N1:** BSPs +0.01 (-0.26 to +0.33); GSPs +0.32 (+0.19 to +0.44). The interval
  resamples test months only. Per seed the GSP difference is -0.04, +0.44, and +0.61 (standard
error about 0.20), and the difference comes mainly from D6 (+0.54; D1 +0.09). A harm of about 0.3 at
the GSPs is not established. A likely reason for no gain is that the Primaries' lags barely track a
grid supply point (correlation -0.26 to +0.03 at D1 and D6, +0.59 to +0.87 at the BSPs).
- **C4, N2 minus N2rw:** BSPs -0.06 (-0.22 to +0.09); GSPs +0.31 (+0.21 to +0.38). N2rw minus N1 is
  +0.07 (BSPs) and +0.02 (GSPs).
- **C5, N1 minus P0 at the 14 countable Primaries:** -0.022 (-0.048 to +0.005); at D15 and D18 -0.46
(-1.43 to +0.31). Adding the nodes to the training rows did not measurably change the Primaries'
score.
- **C6, against the simple references:** N1 minus `manual_heuristic` is -1.65 (BSPs) and -3.21
  (GSPs); N1 minus `climatology` is -1.82 and -2.49; all intervals exclude zero.
- **Tails (N1, days 1 to 10; D13, D17, D1, D6).** q95 coverage 0.88, 0.92, 0.95, 0.91; q99 coverage
  0.97, 0.98, 0.99, 0.98; q05 coverage 0.04, 0.05, 0.09, 0.04; q01 coverage 0.006, 0.011, 0.014,
0.006. BG2's q95 coverage is 0.90 (D13), 0.94 (D17), 0.98 (D1), and 0.94 (D6), so N1's upper tail is
narrower than BG2's at all nodes except D1 (0.95 against 0.98). The upper tail at D13 is too narrow.
- **Predictions.** P1 failed in size: we predicted N1 would beat P0 by 0.3 to 1.5 points, and it
  beat P0 by 4 to 15. P2 failed because N1 did better than predicted: it beat the control by 0.53 at
  the BSPs (predicted: 0.1 better to 0.6 worse) and by 0.18 at the GSPs (predicted: 0 to 1.5 worse).
  P3 held in its first clause at the BSPs (N2 minus N1 +0.01) and failed at the GSPs (+0.32). Its
  second clause failed narrowly at the BSPs (N2 minus N2rw -0.06, against N2 minus N1 +0.01). P4
  held in part: q95 coverage 0.88 to 0.95, q05 coverage 0.04 to 0.09. P5 held.

## Checks

- **Median minus mean.** Not computed, because the neural network stores quantiles only.
- **Mean-forecast error.** Not computed. The pinball loss at q95 agrees with CRPS in direction for
  N1 against BG2 at all four nodes, and at q99 at three nodes.
- **Triggers fired.** Two of the protocol's audit triggers fired: a result contradicts a prediction
  (P1, P2, and P3 at the GSPs), and an arm with extra inputs (N2) scores worse than the arm it
  extends (N1) at the GSPs. A fresh Sonnet audit re-derived every headline contrast to within 0.01
  and found no leakage or missing-value fault in the neighbour columns The audit also found that N1
  at the GSPs ties BG2 only with bands about 100 points of capacity wide, and its median correlates
  0.29 (D6) and 0.65 (D1) with the actual, against 0.82 and 0.87 at the BSPs. A separate Sonnet
  audit of the stage 1 numbers reproduced the printed CRPS and coverage to within 0.31 points and
  found the stored actuals equal to the cleaned readings.

## Setup and caveats

- **Recipe.** Experiment 397's `plain_pooled` network with its chosen setting (width 64, two layers,
  three components, dropout 0.1), the ensemble mean and standard deviation of seven weather columns
  as inputs, the three origin lags, one fit per test month on the 24 months before, monthly
  refitting, as the standing rules ask. Code: `ml-experiments/00415-*/` imports experiment 397 and
  experiment 87's rows.
- **Cleaned data and the cut.** All actuals and lags come from experiment 87's cleaned panel. Of the
  65,166 half-hours that BG2 forecast at each node in the four months, 2,747 are not scored, because
  some arm lacks a forecast or a cleaned actual for them. Rows of D15 and D18 before 2025-03-01 are
  dropped everywhere.
- **Weather at lag times.** The lags are power, not weather. The weather inputs are the ensemble
  forecast for the valid time, available at forecast time.
- **Not scored.** The curtailment cost (the metric needs generator tables that experiment 386 also
  lacked), load-shift event plots (no label is used), and a weekly-refit check.
- **Missing run.** The 2026-06-30 run is absent from the 2026-06 test month, because the stored rows
  end at the 2026-06-29 run.
- **Interim.** Four test months make the intervals wide. The BSP gain over BG2 is the one claim
  made, and it rests on four test months.
- **Stored.** Weights and forecasts are in this experiment's private folder under `work/models` and
  `work/seed_preds`.

## What to try next

- A fair transfer arm, with the node's lags scaled by its own statistics, before quoting the value
  of training on the nodes.
- More seeds of N1 and N2 at D1 and D6, to settle whether the neighbour columns harm the GSPs.
- A weekly-refit check of N1 at the GSPs, where it only ties BG2.
- Widening or recalibrating the upper tail at D13.
- Embedded generation and storage under each node as explicit inputs, which experiment 386 named
  and this experiment did not test.

## Problems caused by the move

None. This section records any difficulty that moving the research code to this repository caused;
this experiment met none.
