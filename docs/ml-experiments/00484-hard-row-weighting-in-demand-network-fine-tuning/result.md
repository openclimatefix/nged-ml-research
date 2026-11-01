---
id: "00484"
slug: hard-row-weighting-in-demand-network-fine-tuning
title: >-
  Does weighting the fine-tuning rows toward cold winter nights, high-flow half-hours at the signed-MW
  six, or winter improve the demand network on those rows without losing elsewhere?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
  - curtailment-cost
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: one 11-quantile table per arm, fold, and seed (45 tables) and the 45 fine-tuned
  networks in this experiment's private folder. More than one design would defensibly define the
  curtailment weight (which fraction f, which flow), so the plan fixes it before any fit. 45
  fine-tunes of a stored pretrained network, about 0.8 GPU-hours on one GPU process under 8 GB and
  one feeder core. Production serving path: not touched (this repository holds none). Degradation
  rule: not touched. Callers: the scripts import experiments 87, 397, 404, 418, 425, 448, 454, and
  470, whose callers sit in those folders.
hypothesis: >-
  Experiments 435 and 470 left cold winter nights worse than the tail target, and avoidable
  curtailment cost ranks the candidates differently from CRPS. Fine-tuning the pretrained network
  with larger weights on the rows that decide NGED's costs lowers the error on those rows without
  raising it elsewhere.
prediction: >-
  Written before any fit. P1 (W-cold): CRPS on cold winter nights falls by at least the larger of
  0.05 points and the cold-night minimum detectable effect, at the eight and at the six, relative to
  experiment 425's stored fit. A smaller contrast is inconclusive. P2 (all-row equivalence): the
  all-row CRPS of W-cold, W-curt, and W-season differs from the stored fit by less than the minimum
  detectable effect of the all-row slice (0.065 at the eight and 0.100 at the six for one seed in
  experiment 470; recomputed here for the mean of three seeds), so the weighting does not trade away
  the other rows. P3 (W-curt): avoidable-curtailment cost at f = 0.75 at the signed-MW six falls by at
  least 0.1 million pounds, with the q99 pinball loss at the eight and the six rising by at most 0.02
  points. P4 (W-season): no prediction of gain; W-season is the winter-only variant of P1 and is read
  on cold winter nights and all rows. Experiment 447 found the demand network insensitive to its
  settings, so the null is likely.
decision_rule: >-
  Arms are W-cold, W-curt, and W-season, each fine-tuned from the stored pretrained weights of
  experiment 425's arm A at seeds 0 to 2 on the five folds of experiment 449 (2025-10, 2025-12,
  2026-01, 2026-03, 2026-06). The matched reference W0 is experiment 425's own stored fit (its
  fine-tune rows already carry the recency weight 0.5 ** (age_days / SHORT_HALF_LIFE_DAYS); every
  new weight multiplies it, and the early-stopping loss is the unweighted held-out-week negative
  log-likelihood, the same for every arm). One (month, seed) of W0 is refitted to prove the code
  path reproduces the stored fit; it reproduced it exactly (10 epochs, validation NLL 0.36901423
  at 2026-01, seed 0). Scored on experiment 470's intersection of rows (gam14: the eight MVA-only
  Primaries and the signed-MW six), 1.15 widening headline, CRPS in points of effective capacity,
  month-by-origin-day block bootstrap (20,000 resamples). Bonferroni count 27 (3 arms times 9
  tests: CRPS on the eight, the six on rows with no labelled event, the six on real-event rows
  against normal-running-arrangement actuals, cold winter nights at the eight and at the six, the
  12 to 36 hour band at the eight and at the six, plus q99 pinball at the eight and at the six on
  rows with no labelled event). Christmas week, D15 and D18, the other tails, and avoidable-curtailment
  cost at f = 0.9, 0.75, and 0.5 (cell bootstrap, as experiment 470) are descriptive and enter no
  count, except that P3 reads the f = 0.75 interval. The minimum detectable effect (experiment 428's
  formula on the seed spread of the stored arm A single seeds, for the mean of three seeds,
  including the cold-night slices) is printed before any contrast is read; a contrast inside it is
  inconclusive. An arm is a candidate if it meets its prediction, loses at most 0.02 CRPS points on
  the eight and the six, and keeps q99 pinball loss within +0.02 points absolute (the rule of
  experiments 452 and 470). Stop rule: if neither W-cold nor W-curt clears its prediction, report
  that row weighting does not move the hard rows and close the idea. The smallest gain worth porting
  is zero in cost terms (a weight is a training-loop change with no extra serving network), so a
  candidate needs only a Bonferroni interval that excludes zero on its gating slice. Every score is
  selection-biased, because the five folds that choose also score; July 2026 stays unused.
benchmark: >-
  Experiment 425's stored arm A fits (seeds 0 to 2 for the contrasts; the five-seed A-pool of
  experiment 470 as the reference row), BG2, manual_heuristic, climatology.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS from the 11 quantiles in points of effective capacity; coverage and pinball loss at q01, q05,
  q95, and q99 raw and with the 1.0 to 1.15 widening sweep; avoidable-curtailment cost at f = 0.9,
  0.75, and 0.5 at the signed-MW six with cell-bootstrap intervals; per group and per Primary.
---

## Does weighting the fine-tuning rows toward the rows NGED cares about help?

**Plan only; the write-up replaces this section when the fits finish.**

### Arms and weights

- **Reference W0.** Experiment 425's stored arm A fits, seeds 0 to 2 (`seed_preds/A/{month}_s{0-2}.parquet`
  in its data folder). Pretrained weights: `00425-.../work/pretrained/A/{month}_s{0-4}.pt`.
- **W-cold.** Weight 3 on a half-hour whose sun is below -5 degrees and whose ERA5 temperature is
  3 degrees Celsius or lower. This is the scorer's `cold_nights` rule (`studies.eval_protocol.slices`),
  not a per-Primary temperature decile, so the weighted rows and the scored rows are the same kind of
  row. 1 elsewhere.
- **W-curt.** Weight 3 (1 plus 2) on a half-hour of one of the signed-MW six (D5, D9, D10, D11, D14,
  D16) whose flow is at least 0.75 times that Primary's maximum flow in the fold's fine-tuning
  window (the scaled limit of `studies.node_curtailment` at f = 0.75). 1 elsewhere. No forecast is
  needed.
- **W-season.** Weight 3 on every half-hour in local December, January, or February (the winter
  fine-tune of the dropped winter-specialist brief; the ENS rows start in 2024-04, so the fold
  windows hold one or two winters).
- **W-both** (the product of W-cold and W-curt, capped at 6) is not run.

### Compute and reuse

- 45 fine-tunes (3 arms times 5 folds times 3 seeds), each about 1 minute, one GPU process under
  8 GB with one feeder core, about 0.8 GPU-hours. The 15 W0 refits are saved because experiment
  425's stored fits are the reference.
- Reused: experiment 425's pretrained weights and stored arm A forecasts (seeds 0 to 2), experiment
  470's row context and scorer. Fitted: the 45 weighted fine-tunes.
- Monthly refits, cleaned data, and the D15 and D18 start cut come from experiment 425's fold
  builder unchanged.
- Production cost: no extra serving network; the change is a training-loop weight, so the monthly
  refit costs the same number of fits.
