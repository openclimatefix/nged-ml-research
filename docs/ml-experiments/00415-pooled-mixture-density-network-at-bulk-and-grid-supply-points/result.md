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
status: planned
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
  the two bulk supply points is between -0.15 and +0.05 points, and N2 minus N2-rw is no larger in
  size than N2 minus N1 (the neighbours act as a noise average). Experiments 161 and 204 found that
  the children's forecasts added at most 0.04 points to a GAM at these two bulk supply points, so
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
code: ml-experiments/00415-pooled-mixture-density-network-at-bulk-and-grid-supply-points/
---

**Plan only. This page is written before any code runs; results replace this notice.**

## Question

Experiment 386 scored four forecast families at two bulk supply points (BSPs: D13, D17) and two
grid supply points (GSPs: D1, D6). BG2, the weekly XGBoost model, was best there, and the Primary
GAM stack scored 1.0 (BSPs) and 2.0 (GSPs) points worse. Experiment 403 found a pooled plain
mixture density network (MDN) better than one network per Primary. This experiment asks whether the
same pooled network forecasts the four nodes, and whether the Primaries beneath a node add anything.

## What changed since the brief, and why the plan differs

- **Four folds, not the full year.** The standing screen (2026-06, 2026-01, 2025-10, 2026-03; 24-month
  training window, one fit per fold) replaces the brief's open-ended scope. The brief predates it.
  Every number is interim and no claim of benefit rests on it.
- **One-pass ENS mean and standard deviation inputs** (the `summary` scheme of experiment 397), as
  the maintainer expects to become the default; experiment 394 found the mean carries the gain.
- **Cleaned data and the start cut.** All rows come from experiment 87's cleaned panel (flagged
  half-hours dropped, counted per fold), and D15 and D18 rows before 2025-03-01 are dropped
  everywhere (training, lags, inference, scoring).
- **Monthly refit cadence**, applied as one fit per fold. The stored BG2 refits weekly, so the
  contrast against BG2 depends on cadence; the stored monthly control is the like-for-like row.
- **Reuse.** The stored forecasts of experiment 386 are the references and are not refitted. Primary
  controls come from P0, so no experiment 397 forecast is reused (its fits cover five other months
  and no node).
- **Experiment 411 and 405.** Neither bears on this plan: the lag-trust and solar-term networks are
  separate arms of other experiments, and this experiment keeps the plain network of 403.
- **Gap in the brief.** The brief asks for the Primaries beneath each node. The repository records
  them only for the BSPs (D13: D9, D10, D11; D17: D5, D14, D16; experiment 161). No record of the
  Primaries beneath D1 and D6 exists, so the neighbour arms cover the two BSPs only, and the GSPs
  get the neighbour columns as missing. The manager is asked to supply the GSP children if they
  exist.

## Arms

All arms are the plain pooled network of experiment 397 (`plain_pooled` setting, three mixture
components, Primary embedding, `summary` scheme, three origin lags), trained on the cleaned rows
of the fold's 24-month window, 3 seeds, one fit per fold.

- **P0 (reference).** Fitted at the 16 Primaries only. Applied to a node with the mean of the 16
  Primary embeddings and the node's own standardisation of the target (its training mean and
  standard deviation), because an unseen scale would make the transfer meaningless.
- **N1.** Fitted at the 16 Primaries and the 4 nodes, with its own embedding per series and a
  node-type input (Primary, BSP, GSP).
- **N2.** N1 plus neighbour inputs: the mean over the children of each of the three origin lags in
  capacity-normalised units, with a missing flag. At the GSPs and Primaries the columns are missing.
- **N2-rw.** N2 with each BSP's children replaced by a degree-matched random set of three other
  Primaries (fixed seed, never the BSP's own children).
- **Stored references:** BG2, monthly XGBoost control, GAM stack, `manual_heuristic`, `climatology`.

## Stop rule and decision rule

Registered before any score is read. Scores are CRPS at days 1 to 10 in points of effective
capacity from `studies.eval_protocol`, BSPs and GSPs reported apart and never pooled, paired
month-block bootstrap, interim.

- **Stage 1.** P0 and N1, folds 2026-06 and 2026-01, seed 1 (4 fits).
- **Stop A (nothing learned).** If N1 scores worse than the stored climatology at three or more of
  the four nodes in both stage 1 folds, stop and write up. The network has learned nothing at the
  nodes that a month-by-half-hour average does not give.
- **Stop B (nodes hurt the Primaries).** If N1 minus P0 at the 14 countable Primaries pooled exceeds
  +0.10 points in both stage 1 folds, skip the neighbour arms and write up P0 and N1 only.
- **Stage 2.** N2 and N2-rw, the same two folds, seed 1 (4 fits).
- **Stop C (neighbours do nothing).** If N2 minus N1 at the BSPs pooled is 0 or higher in both folds,
  run no further seeds or folds of N2 or N2-rw, and report them as screened out.
- **Stage 3.** Whatever survives, on all four folds with 3 seeds (at most 48 fits in all).
- **Contrasts.** C1 N1 minus P0 at the nodes; C2 N1 minus the stored monthly XGBoost control and
  minus BG2; C3 N2 minus N1 at the BSPs; C4 N2 minus N2-rw at the BSPs (neighbour identity);
  C5 N1 minus P0 at the Primaries; C6 each against `manual_heuristic` and `climatology`. The
  smallest gain worth acting on is any gain whose interval excludes zero, because the change adds
  rows and an input column to a network we would train anyway. The refit noise floor is the spread
  across the 3 seeds of P0, and a contrast smaller than it is called level.
- **Verdict words.** `supported` needs C1 negative at both groups with an interval below zero and
  C2 within +0.30 at both groups. Anything that stops at A or B is `refuted` or `not supported`
  with the take-home saying how far behind.

## Reporting

Tail calibration (q01, q05, q95, q99 coverage and pinball loss) per node; a day-ahead and a
14-day example forecast at one BSP and one GSP, at substations, so no anonymisation is needed; the
count of flagged half-hours dropped per fold. Curtailment cost and load-shift event plots are not
scored: the curtailment metric needs generator tables that experiment 386 also lacked, and no
load-shift label is involved.

## Compute

Timed per fit before any scored run; the estimate and the memory per fit go here. One GPU process
under 8 GB, one feeder core, `MemoryMax=8G`, until the manager raises them.
