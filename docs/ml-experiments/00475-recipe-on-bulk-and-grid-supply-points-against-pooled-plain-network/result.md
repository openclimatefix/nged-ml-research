---
id: "00475"
slug: recipe-on-bulk-and-grid-supply-points-against-pooled-plain-network
title: >-
  Does the first-release candidate recipe forecast the two bulk supply points and two grid supply
  points better than the pooled plain network of experiment 415?
theme: hierarchy-and-pooling
also_relevant_to:
  - production-reference-and-best-guess
status: done
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
verdict: not supported
production_status: not-for-production
adopted_in: none
best_arm: >-
  A (experiment 425's arm A recipe fitted pooled with the 16 Primaries and the 4 nodes): CRPS 4.61
  (D13), 6.66 (D17), 13.92 (D1), and 13.69 (D6) at days 1 to 10, four test months, 3-seed average
headline: >-
  A minus 415's N1, days 1 to 10, points of effective capacity: bulk supply points -0.025 (95%
  interval -0.104 to +0.053), grid supply points -0.532 (-0.815 to -0.215). A minus BG2: -0.49 (BSPs)
  and -0.55 (GSPs). Four test months, so every interval is interim.
take_home_message: >-
  The assembled recipe (ERA5-and-CAMS pretraining, cuts, flagged zeros dropped) does not improve on
  the plain pooled network at the two bulk supply points: it is better at D17 (-0.14) and worse at
  D13 (+0.09), both inside the noise. At the two grid supply points the recipe is about 0.5 points of
  capacity better than the plain network, which is close to the size of the refit noise (about
  0.4). Against BG2 the recipe is better at all four nodes, with the largest gain at the grid supply points.
---

**Take-home message: the first-release recipe does not clearly beat the plain pooled network at the
bulk supply points, but it gains about 0.5 points of capacity at the grid supply points.** The
registered rule needs a win at both bulk supply points, so the verdict is `not supported`. The
recipe lowers CRPS below BG2 at all four nodes.

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

- Minimum detectable effect: the seed spread of 415's N1 at the grid supply points (per-seed spread of 415's N1: 0.44 at D1 and 0.79 at D6; about 0.4 is the low end),
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

## Results

CRPS at days 1 to 10 in points of effective capacity, four test months (2025-10, 2026-01, 2026-03,
2026-06), equal-row set of 62,419 half-hours per node across all leads, then restricted to days 1 to 10 (shared with experiment 386; 2,747 of 65,166
not scored). A and N1 are 3-seed averages; BG2 is a single weekly fit.

| Node | A | 415's N1 | BG2 (weekly-refit XGBoost best guess) | Monthly XGBoost, bands widened by 1.15 | `manual_heuristic` | `climatology` |
|---|---|---|---|---|---|---|
| D13 (bulk supply point) | 4.61 | 4.52 | 4.94 | 4.99 | 5.80 | 6.06 |
| D17 (bulk supply point) | 6.66 | 6.80 | 7.31 | 7.39 | 8.82 | 8.89 |
| D1 (grid supply point) | 13.92 | 14.41 | 14.52 | 14.70 | 19.18 | 18.29 |
| D6 (grid supply point) | 13.69 | 14.26 | 14.19 | 14.33 | 15.92 | 15.37 |

- **A minus N1:** BSPs -0.025 (-0.104 to +0.053); GSPs -0.532 (-0.815 to -0.215). Per node: D13
  +0.09, D17 -0.14, D1 -0.50, D6 -0.57.
- **A minus BG2:** BSPs -0.491 (-0.661 to -0.299); GSPs -0.550 (-0.943 to -0.116). A minus the
  widened monthly XGBoost: -0.557 and -0.710. A minus `manual_heuristic`: -1.68 and -3.75. A minus
  `climatology`: -1.84 and -3.03.
- **Decision rule:** not met. A loses to N1 at D13 by 0.09, inside the minimum detectable effect
  (about 0.4). Its gain at the grid supply points (0.5) is only just above the minimum detectable
  effect, and the intervals resample four test months, so they are interim.
- **Tails (A against N1, days 1 to 10), q99 pinball loss:** D13 0.246 against 0.242 (+1.7%), D17
  0.355 against 0.362 (-1.8%), D1 0.787 against 0.797 (-1.3%), D6 0.604 against 0.626 (-3.5%); all
  within 5%. q95 coverage of A: 0.878 (D13), 0.935 (D17), 0.952 (D1), 0.920 (D6); the D13 upper tail
  is still too narrow, as in N1 (0.876).
- **Predictions:** P1 failed (we predicted -0.30 to 0.0 at the BSPs; D13 was +0.09). P2 failed in
  size (we predicted -0.30 to +0.30 at the GSPs; the gain was -0.5). P3 held. P4 held: A is below
  BG2 at all four nodes.
- **Bonferroni count:** 8 (four nodes by two contrasts); the intervals above are unadjusted 95%
  intervals.

## Setup and caveats

- **Recipe.** Experiment 425's arm A: ERA5-and-CAMS pretraining (about 10.7 million rows),
  fine-tuning on 7.17 million ensemble-summary rows (2026-06 fold), setting `g09`, monthly refits, 3
  seeds. Pooled with the 16 Primaries and a node-type input. The loop is experiment 425's
  (`e00425_fit.py`); experiment 415 used experiment 397's loop. Code: `ml-experiments/00475-*/`,
  importing experiments 87, 397, 407, 415, 418, and 425.
- **Cuts.** D15 and D18 rows before 2025-03-01 dropped; flagged exact-zero half-hours dropped from
  training rows (2026-06 fold: 68,680 start-cut and 4,073 flagged-zero fine-tuning rows). Scoring
  uses experiment 386's equal-row set, which is not re-cleaned for A.
- **Not like for like with 425.** The standardiser and the ensemble-spread table are fitted on all
  20 series; experiment 425 used the 16 Primaries. No early-history cut was set for the nodes (their
  data start 2019-09-17, D17 in 2022-07).
- **Not done:** S1 (experiment 428's loop; needs new code), the per-type pooled arm, the
  generators-as-children arm, curtailment cost (does not apply at nodes), load-shift plots, example
  forecast plots, and the results and write-up reviews. A fresh Sonnet agent reviewed the code
  before the first fit (no bugs; the findings on node early-history cuts and flagged-zero keys were
  checked and needed no change).
- **Interim.** Four test months; NGED's event labels are incomplete, so clean rows may hold
  unlabelled events.

## What to try next

- Run S1 on the nodes if the demand assembly (experiment 470) adopts it.
- Repeat the grid supply points with more seeds to settle whether 0.5 exceeds the refit noise.

## Problems caused by the move

None.
