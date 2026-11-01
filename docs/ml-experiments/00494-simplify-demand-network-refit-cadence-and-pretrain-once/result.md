---
id: "00494"
slug: simplify-demand-network-refit-cadence-and-pretrain-once
title: >-
  Can the Primary demand network be refitted less often, or pretrained once and only fine-tuned at
  each refit, without losing forecast skill?
theme: retraining-and-training-history
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: refit-cadence and pretrain-once forecasts and the new networks under this
  experiment's private data folder only; every earlier experiment is read and never written.
  Serving path: not touched (this repository holds none). Degradation rule: not touched. Designs:
  more than one defensible (how to age a network, and what "pretrain once" reuses), so the plan fixes
  both before any score is read. Callers: this folder's scripts import experiments 397, 425, 449, and
  470, and the `studies` evaluation code. One Sonnet plan review, one Sonnet code review before the
  first fit, one Sonnet results review, and two Sonnet write-up reviews. Part of the neural
  simplification series (step 3 of 7); step 6 (input and output pruning) follows in the same pull
  request.
hypothesis: >-
  A pooled demand network that was refitted 3 months before the scored month forecasts as well as a
  freshly refitted network, and so does a network fine-tuned from one pretraining run that stops
  before the first scored month. Staler networks (6 months) lose a little more. If so, production
  needs one pretraining run, fine-tunes at most every 3 months, and no ERA5-and-CAMS replay at refit.
prediction: >-
  Written before any score is read. P1: the 3-month-old network (Stale3) is within 0.05 points of
  CRPS of the fresh two-network pool (K2) at both the MVA-only eight and the signed-MW six, on rows
  with no labelled event. P2: the pretrain-once network (PretrainOnce) is within 0.05 points of K2
  at both groups. P3: the 6-month-old network (Stale6) loses 0.03 to 0.10 points at the eight and
  0.05 to 0.15 at the six, more on cold winter nights. P4: q99 coverage moves by less than 0.01 for
  every arm.
decision_rule: >-
  An arm is equivalent to K2 when the upper Bonferroni bound of its CRPS difference to K2 (month-block
  bootstrap, 20,000 resamples) is at most the limit, at the eight and at the six, and the upper
  Bonferroni bound of its q99 pinball-loss difference is at most +0.02 points. The limit is the larger
  of the brief's cap and the printed minimum detectable effect (MDE): Stale3 and PretrainOnce
  max(0.05, MDE); Stale6 max(0.10, MDE). An interval that reaches past the limit is inconclusive,
  not equivalent, and an equivalence claim at the MDE cannot show a loss smaller than the MDE.
  Bonferroni count 12 (3 arms x 2 groups x [CRPS, q99 pinball loss]). Stale11 and the cold-night,
  band, Christmas, and curtailment slices are descriptive and not counted. Recommend the stalest
  cadence whose arm is equivalent; recommend pretrain-once if PretrainOnce is equivalent. If a
  claim holds only with the 2025-07 fold (15 months of ENS history, not 24) included, say so.
benchmark: >-
  K2: the quantile average of arm A seeds 0 and 1 of experiment 425 (fresh pretraining and
  fine-tuning in the scored month), stored; scored on the same rows.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity, q01, q05, q95, and q99 coverage and pinball loss,
  avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the signed-MW six.
---

**TODO at the finished commit: take-home message.**

## Plan

### Question

Production refits the demand network every month, pretraining on ERA5-and-CAMS rows and then
fine-tuning on ENS rows. Fewer fits per year and no pretraining at each refit would simplify
serving and training. This experiment ages the network on purpose, and also fine-tunes from one
pretraining run reused at every refit. The baseline is experiment 425's arm A with 2 seeds.

### Design

- **Rows and folds.** The 5 folds of experiment 449 (2025-10, 2025-12, 2026-01, 2026-03,
  2026-06) on the rows of experiment 470 (`experiment_470_intersection_v1`). The D15 and D18 start
  cut and the flagged exact-zero drop apply. Cleaned data everywhere; monthly refits.
- **K2 (baseline).** Quantile average of seeds 0 and 1 of arm A, stored in experiment 425
  (`work/seed_preds/A`). The 1.15 widening is applied to every arm as in experiments 470 and 489.
- **Stale3, Stale6 (cadence).** For scored month M, forecast with the arm A networks (seeds 0 and 1)
  that were pretrained and fine-tuned at refit R = M minus 3 (or 6) months. The network's own
  standardiser and target statistics are used. No retraining: the networks are experiment 425's
  stored `work/models/A/<R>/s<seed>`. A cadence of 3 months means the network is 0, 1, or 2 months
  old when scored, so Stale3 (3 months old) is the worst month of a 3-month cadence, and Stale6 the
  worst month of a 6-month cadence.
    - Stale3 needs R in 2025-07, 2025-09, 2025-10, 2025-12, 2026-03: all stored, so no fits.
    - Stale6 needs R in 2025-04, 2025-06, 2025-07, 2025-09, 2025-12. Experiment 425 stores only the
      last three, so the first two are fitted here: arm A, seeds 0 and 1, pretrain and fine-tune,
      with `e00425_fit.run_arms` unchanged. Their ENS fine-tuning windows hold 12 and 14 months, not
      24, because the ENS history starts in 2024-04; the plan reads Stale6 on all 5 folds and on
      the 3 folds with stored networks.
    - Stale12 cannot be run: refits before 2025-04 would have under 12 months of ENS history. The
      single fold 2026-06 from the stored 2025-07 networks (Stale11) is reported as descriptive.
- **PretrainOnce.** Each of the 5 scored months is fine-tuned from the pretrained arm A network of
  the first stored cut (2025-07) of the same seed, instead of the network pretrained at that
  month's cut. Fine-tuning is experiment 425's (learning rate 2e-4, same rows, same loop, seeds 0
  and 1). The fold's standardiser is the scored month's own, so the fine-tuning sees an input scale
  that differs slightly from the pretraining's. That mismatch is part of what the arm tests. Ten
  fine-tunes; no pretraining.
- **Not run.** A cadence on the fine-tune alone with fresh pretraining (a pure fine-tune staleness
  arm): the question the brief asked, but it needs a pretraining cut at each scored month paired with
  a fine-tune R months earlier, and production would pretrain once, which PretrainOnce covers.

### Scoring and stop rules

- Experiment 489's scorer pattern: the experiment 470 scorer, tail script, and curtailment script
  pointed at this folder. Because Stale6 and Stale11 cover fewer folds, each arm is scored in its
  own call against K2 on the folds the arm has.
- The scorer prints the MDE before any contrast (experiment 428's definition from the seed spread of
  arm A, as in experiment 489). Experiment 481 uses another definition; the page reports both.
- Report per arm: CRPS at the eight, the signed-MW six, cold winter nights, the 12 to 36 hour band;
  q01, q05, q95, q99 coverage and pinball loss; avoidable-curtailment cost at f = 0.9, 0.75, 0.5 at
  the six. Christmas week and D15 and D18 are report-only.
- **Stop.** Do not run Stale6's two new fits if Stale3 already fails its limit at either group.
- **Serving table (required).** Networks per serving pass, refits per year, pretrainings per year,
  GPU seconds per refit (experiment 481: about 80 per seed, 46 pretraining and 34 fine-tuning), and
  the score cost against K2.

### Compute

- Stale3: 0 fits, 10 month-forecast passes. Stale6: 4 networks (2 cuts x 2 seeds), about 5 GPU
  minutes, plus passes. PretrainOnce: 10 fine-tunes, about 6 GPU minutes. One GPU process under 8 GB;
  one feeder core.

### Standing practices followed

Monthly refits; cleaned data; D15 and D18 cut; saved networks; stored forecasts reused; tails and
cost; example forecasts and the 14-day plot; the process order of the ml-experiment skill.
Not applicable: load-shift event plots (no event slice is headline).

### Series

Step 3 of the neural simplification series (brief of 2026-11-01); step 6 shares this pull request.
Steps 1 and 4 run in other pull requests.
