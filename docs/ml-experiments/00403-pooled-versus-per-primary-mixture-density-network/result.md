---
id: "00403"
slug: pooled-versus-per-primary-mixture-density-network
title: >-
  Does one pooled mixture density network beat one network per Primary for Primary demand, and where
  does the gain fall?
theme: hierarchy-and-pooling
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  simple. Stored: a small score table in this experiment's private folder; experiment 397's stored
  forecasts are read and never written. Production serving path: not touched (this repository holds
  none). Degradation rule: not touched. Designs: one defensible (a re-score of two stored forecast
  sets on identical rows). Callers: the script imports experiments 377, 381, and 87 and the
  `studies` evaluation code, whose callers sit in those folders. No fit runs.
hypothesis: >-
  One network trained across all 16 Primaries scores lower CRPS than one network per Primary, with
  the same inputs, rows, folds, and seeds, and the gain is largest at the Primaries with the
  shortest history.
prediction: >-
  Written before the contrast is computed. The pooled plain network is already known to score
  0.12 to 0.24 points lower than the per-Primary network in experiment 397's group means, so P1 is
  that pooled A0 minus the per-Primary network is negative at the signed-MW six and at the
  MVA-only eight, with a month-block 95% interval that excludes zero at the signed-MW six. P2: the
  gain correlates negatively with months of history across the 14 countable Primaries (a larger gain
  where the history is shorter), with a correlation below -0.3. P3: D15 and D18 gain more than the
  median countable Primary. P4: the gain does not concentrate: the best three Primaries carry less
  than 60% of the summed gain.
decision_rule: >-
  The contrast is pooled A0 minus the per-Primary plain network on CRPS at days 1 to 10, per group
  (signed-MW six; MVA-only eight, with and without D8; D15 and D18 report-only) and per Primary, by
  the month-block bootstrap of `studies.eval_protocol` at 95% (20,000 resamples). The five screen
  months give wide intervals, so every number is labelled interim and no claim of benefit rests on
  it. Pooling is supported if the signed-MW six and the MVA-only eight both have a negative
  difference with an interval below zero, and at least 10 of the 14 countable Primaries gain. A
  Primary-level interval that includes zero is reported as such. The smallest gain worth acting on
  is any gain whose interval excludes zero, because a pooled network costs less to train than many
  per-Primary networks and so needs no gain to pay for its complexity.
benchmark: >-
  The per-Primary plain network of experiment 397 stage 1, the pooled plain network of stage 2, and
  the stored monthly XGBoost forecasts of experiment 389, all on the same half-hours.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, in points of effective capacity, from
  `studies.eval_protocol`; leads 0.5 to 336 hours; the headline band is days 1 to 10.
code: ml-experiments/00403-pooled-versus-per-primary-mixture-density-network/
---

**Take-home message (to be written when the result is known).**

This page is the plan, committed before the contrast is computed.

## Question and design

**We re-score two sets of stored forecasts and fit nothing.** Experiment 397 stage 1 fitted one
mixture density network per Primary. Stage 2 fitted one pooled network with a learned Primary
embedding and uniform input-group weights (arm A0). Both use the ensemble mean and standard
deviation of each weather variable, the same training rows, the same five test months (2025-10,
2025-12, 2026-01, 2026-03, and 2026-06), the same three seeds, and the same scoring rows. The
earlier experiments that compared pooled and per-site models used XGBoost, a generalised additive
model, or a neural network at full history, and found per-site models better (experiments 24, 87,
105, and 109), so the pooled result needs a control. It matters for the 1,300 substations of
Flexpectation v2, because one pooled network trains far more cheaply than 1,300 networks.

- The script `e00403_rescore.py` averages each arm's seeds in quantile space, restricts the arms to
  the half-hours they share, and prints the absolute scores, the contrast per group and per
  Primary, and how the gain spreads over the Primaries and relates to months of history.
- The stage 1 and stage 2 rows are the same, because both come from experiment 397's preparation
  script, so no refit is needed.
- Reused forecasts: experiment 397's `plain_own`, `plain_pooled`, and `A0` sets (summary scheme)
  under its private `work/seed_preds/summary/`.
- Tails (coverage and pinball loss at q05 and q95) are reported in experiment 397 for these arms and
  are not repeated here.

## Problems caused by the move

None so far.
