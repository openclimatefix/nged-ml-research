---
id: "00475"
slug: recipe-on-bulk-and-grid-supply-points-against-pooled-plain-network
title: >-
  Does the first-release candidate recipe forecast the two bulk supply points and two grid supply
  points better than the pooled plain network of experiment 415?
theme: hierarchy-and-pooling
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: network weights and per-arm forecast tables in this experiment's private folder
  only; experiments 87, 386, 415, 425, and 428 are read and never written. Production serving path:
  not touched (this repository holds none). Degradation rule: not touched. Designs: more than one
  would test the question (which recipe, pooled with the Primaries or alone); the plan fixes one.
  Callers: the scripts import experiment 397, 415, 418, and 425 modules, whose callers sit in those
  folders.
hypothesis: >-
  The first-release candidate recipe (experiment 425's arm A: ERA5-and-CAMS pretraining, the
  Primary-and-node embedding, hand-made calendar columns, the D15 and D18 cut, and flagged
  exact-zero half-hours dropped), fitted pooled with the 16 Primaries and a node-type input, scores
  lower CRPS at days 1 to 10 than experiment 415's N1 at both bulk supply points and does not lose
  at the grid supply points.
prediction: >-
  Written before any fit and before any score is read. P1. Arm A minus 415's N1 is between -0.30
  and 0.0 points of effective capacity at the bulk supply points D13 and D17, because experiment
  425 found pretraining lowers CRPS by 0.05 to 0.12 points at the Primaries and the nodes add
  reverse-flow and generation that pretraining on ERA5 cannot teach. P2. At the grid supply points
  D1 and D6 the difference is between -0.30 and +0.30, inside the minimum detectable effect of about
  0.4. P3. q99 pinball loss of arm A is within 5% of 415's N1 at every node. P4. Arm A is also
  lower than BG2 at the bulk supply points and within 0.2 of BG2 at the grid supply points.
decision_rule: >-
  See "Decision rule" in the body, registered before any score is read.
benchmark: >-
  Experiment 415's stored N1 and the stored forecasts of experiment 386 on `fold_a_daily_gap_v1`:
  BG2 (`b_weekly`), the monthly XGBoost control, the GAM stack, `manual_heuristic`, and
  `climatology`, on the equal-row set shared with experiment 386.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, in points of effective capacity, from
  `studies.eval_protocol`; leads 0.5 to 336 hours; headline band days 1 to 10; q01, q05, q95, and q99
  coverage and pinball loss, raw and at the 1.15 widening.
take_home_message: >-
  Planned; no score has been read.
---

**Take-home message: planned; no score has been read.**

## Question and what experiment 415 already answered

[Experiment 415](../00415-pooled-mixture-density-network-at-bulk-and-grid-supply-points/result.md)
fitted one pooled plain network (experiment 397's recipe) on the 16 Primaries and the four nodes
(D13 and D17 are bulk supply points; D1 and D6 are grid supply points) with a node-type input. It
beat BG2 by 0.47 points of effective capacity at the bulk supply points, tied it at the grid supply
points, and found that the Primaries' lags add nothing. This experiment does not repeat 415's N1,
N2, N3, or N4(a). It asks only whether the better recipe, which the first-release candidate will
use, improves on 415's N1 at the four nodes. Only four nodes have a usable series (2 bulk, 2 grid),
so every node result is a case study with n = 4, and a pooled-by-type network would pool two series.

## Arms

- **Reused, not refitted:** 415's N1 (3 seeds, 4 folds; `00415.../work/seed_preds/N1`), and
  experiment 386's BG2, monthly XGBoost control, GAM stack, `manual_heuristic`, and `climatology`.
- **A-nodes (fitted):** experiment 425's arm A recipe (loop `e00425_fit.py`, which imports
  experiment 418's ERA5-and-CAMS pretraining rows) extended from 16 to 20 series, with 415's
  node-type input and 415's ensemble-summary node weather. ERA5 rows exist for all four nodes.
- **S1-nodes (fitted if the timing gate passes):** the S1 base with lags and surprise inputs
  (loop of experiment 428). Experiment 415 used experiment 397's loop, which has no surprise inputs;
  A-nodes uses 425's loop and S1-nodes uses 428's, and the page says which.
- **Report-only:** a per-type pooled network (two series each) and the metered generators as
  children (N4(b), which experiment 223 left untested because unmetered solar and the battery
  confound it). Experiment 470 (the demand assembly) has not reported; if it reports before the
  fits finish, its base replaces A-nodes.

## Setup

- Four folds: 2025-10, 2026-01, 2026-03, 2026-06; 24 months of training before each; monthly refits;
  three seeds. A fit sees only data before its forecast time.
- Cleaned data with flagged exact-zero half-hours dropped; D15 and D18 cut before 2025-03-01
  (applied). The dropped counts per fold are reported.
- Compute: one GPU process under 8 GB, one feeder core, `MemoryMax=8G`, `OMP_NUM_THREADS=1`. One
  fit is timed first and the total (fits and GPU-hours) goes to the manager before the rest.
- Lags: the nodes' series support the three origin lags used for the Primaries (newest reading, same
  time of day on the latest day and the latest week); the plan checks that each node has them in all
  four folds.
- Process: code review by a fresh Sonnet agent before the first fit, results review, write-up
  review; Bonferroni count 8 (4 nodes, 2 contrasts each: against 415's N1 and against BG2).

## Decision rule

Registered before any score is read.

- Minimum detectable effect: the seed spread of 415's N1 at the grid supply points (about 0.4),
  stated before any contrast.
- The recipe is a candidate for the nodes if its point estimate beats 415's N1 at both bulk supply
  points and does not lose at the grid supply points, with q99 pinball loss within 5%.
- A node-level contrast inside the minimum detectable effect is inconclusive.
- Otherwise 415's N1 stays the node forecaster and BG2 is the reference at the grid supply points.
- The curtailment cost does not apply at nodes (no generator tables), and the page says so.
- Interim: four test months give wide intervals; the bootstrap over four month blocks cannot reach
  95%.

## Caveats

- 415's N1 is a 3-seed average and BG2 a single weekly fit; the contrast against BG2 depends on
  cadence and seed count and is stated that way.
- NGED's event labels are incomplete: rows with no labelled event may hold unlabelled events.
- The forecast frames omit the 2026-06-30 run, as in 415.
