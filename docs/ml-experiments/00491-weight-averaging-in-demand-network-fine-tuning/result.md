---
id: "00491"
slug: weight-averaging-in-demand-network-fine-tuning
title: >-
  Can one demand network, trained once with an exponential moving average or a stochastic weight
  average of its weights, recover a useful share of the five-network pool's gain?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: one 11-quantile table per arm, fold, and seed (60 tables for the four main
  arms, plus 12 for the one-fold pretraining arms) and the fitted networks in this experiment's
  private folder. More than one design would defensibly define the averaging window, so the plan
  fixes it before any score. 15 fine-tunes of stored pretrained networks (each yielding three
  networks), plus 3 pretrainings and 12 fine-tunes at one fold, about 0.5 GPU-hours on one GPU
  process under 8 GB and one feeder core. Production serving path: not touched (this repository
  holds none). Degradation rule: not touched. Dependencies: the scripts import experiments 87, 397,
  404, 418, 425, 448, 454, and 470, whose callers sit in those folders.
hypothesis: >-
  Experiment 481 found that averaging the quantiles of several networks that differ only in fit
  seed lowers the score at the signed-MW six, and that the first-release demand forecaster serves
  five such networks. Averaging the weights of one network along one training trajectory costs one
  network at serving time, and may recover a useful share of the gain from seed averaging.
prediction: >-
  P1: either the exponential moving average (EMA) or the stochastic weight average (SWA) closes at
  least 40% of the gap between the fixed-budget single seed (S0f) and the stored five-seed pool
  (Pool5) at the eight, on rows with no labelled event, with the q99 pinball loss at the eight and
  the six at most 0.02 points above S0f's. P1 is weak by construction. In experiment 481, the six
  scores 3.257 points at one seed and 3.200 at five, a gap of 0.057 points; 40% of that is 0.023
  points, which is below the seed-noise minimum detectable effect (0.100 at the six for one seed in
  experiment 470), so the closure share at the six is reported as a point estimate and cannot be a
  test. P2 (testable): in at least 12 of the 15 fine-tunes, the EMA weights have a lower
  held-out-week loss at the last epoch than the raw weights. A first fit at a 10-epoch budget
  (since replaced by 4 epochs) had shown 0.3704 for EMA against 0.3748 for the raw weights, so P2
  is not blind. P3 (the likely outcome): the closure share is under 20%, because the five pool
  members differ in their pretraining seed, and averaging weights during fine-tuning starts from
  one pretrained network. P4 (descriptive): averaging weights during pretraining as well (SWA-pre,
  one fold) closes more of the gap than fine-tune averaging alone.
decision_rule: >-
  Arms are S0f, EMA, SWA, and EMA99 (a descriptive EMA with decay 0.99), four networks read from one
  fixed-budget fine-tune per fold and seed (seeds 0 to 2, the five folds of experiment 449: 2025-10,
  2025-12, 2026-01, 2026-03, 2026-06), fine-tuned from the stored pretrained weights of experiment
  425's arm A. S0f is the matched control. The other references are experiment 425's stored fits,
  not refitted: S0 (the early-stopped single seed, three seed cells; the production recipe, a
  descriptive reference), Pool2 (the quantile mean of two seeds, three pairs: seeds (0,1), (0,2), and
  (1,2)), Pool3 (seeds 0 to 2), and Pool5 (seeds 0 to 4, the A-pool of experiment 470). Scored on
  experiment 470's intersection of rows (the eight MVA-only Primaries and the signed-MW six), 1.15
  widening headline, CRPS in points of effective capacity, month-by-origin-day block bootstrap
  (20,000 resamples). Bonferroni count 54 (6 contrasts times 9 tests): the contrasts are EMA minus
  S0f, SWA minus S0f, EMA minus Pool2, SWA minus Pool2, EMA minus Pool5, and SWA minus Pool5, each on
  the three-seed mean. The nine tests are (1) CRPS at the eight; (2) CRPS at the six on rows with no
  labelled event; (3) CRPS at the six on real-event rows against normal-running-arrangement actuals;
  (4) and (5) CRPS on cold winter nights at the eight and at the six; (6) and (7) CRPS in the 12 to
  36 hour band at the eight and at the six; (8) and (9) q99 pinball loss at the eight and at the six
  on rows with no labelled event. Descriptive and uncounted: S0f, EMA, and SWA against S0, EMA99
  against S0f, Pool-of-EMA and Pool-of-SWA over three seeds against Pool3, the one-fold pretraining
  arms, Christmas week, D15 and D18, the other tails, and avoidable-curtailment cost at f = 0.9,
  0.75, and 0.5 at the signed-MW six (cell bootstrap, as experiment 470). The brief lists
  curtailment cost as a gating row; the plan keeps it outside the count deliberately, because three
  more tests per contrast would raise the count to 72 and widen every interval. The minimum
  detectable effect (experiment 428's formula on the seed spread of the stored arm A single seeds,
  for the mean of three seeds, including the cold-night slices) is printed before any contrast is
  read; a contrast inside it is inconclusive. The gaps from S0f and from S0 to Pool5 are printed in
  points at the eight and the six before any averaged arm is read. The eight decides, because the
  gap at the six (0.057 points in experiment 481) is smaller than the six's minimum detectable
  effect (0.100), so the six cannot separate S0f from Pool5. Recommendation: serve a weight-averaged
  single network only if the eight's gap from S0f to Pool5 exceeds 0.15 points, the arm's contrast
  with Pool5 is inside the minimum detectable effect at the eight, its q99 pinball loss at the eight
  and the six is at most 0.02 points above Pool5's (the rule of experiments 452 and 470), and its
  contrast with S0f is a gain. If the eight's gap is 0.15 points or less, the outcome is
  inconclusive: the experiment cannot tell whether seed diversity or weight averaging carries the
  gain. If the gap exceeds 0.15 points and the closure share is between 40% and 70%, report the
  score against serving cost for the maintainer to decide. If the share is under 20%, report that
  seed diversity, not weight averaging, carries the gain. SWA-pre runs at the 2026-01 fold only,
  descriptively, because experiment 481's curve shows that seed diversity carries the gain (K = 2
  gains 0.035 points over K = 1). Every score is selection-biased, because the five folds that
  choose also score; July 2026 stays unused.
benchmark: >-
  Experiment 425's stored arm A fits (S0 at seeds 0 to 2; Pool2, Pool3, and Pool5 of the stored
  seeds), BG2, manual_heuristic, climatology.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS from the 11 quantiles in points of effective capacity; coverage and pinball loss at q01, q05,
  q95, and q99 raw and with the 1.0 to 1.15 widening sweep; avoidable-curtailment cost at f = 0.9,
  0.75, and 0.5 at the signed-MW six with cell-bootstrap intervals; per group and per Primary.
---

## Can one network trained with weight averaging replace a pool of five?

**Plan only; the write-up replaces this section when the fits finish.**

### Arms and settings

- **Fixed budget.** Every averaged run fine-tunes for exactly 4 epochs, the median best epoch of
  experiment 425's 25 stored arm A fine-tunes. The stored lengths (median 10, range 8 to 15) include
  the 6 patience epochs after the best epoch, so the best epoch is the length minus 6. The learning
  rate stays at 2e-4, constant. The batch size is 16,384 rows, so a fold has about 270 optimiser
  steps per epoch (about 1,080 steps in all). Early stopping is off.
- **One run, four networks.** The run is a copy of experiment 397's training loop with the
  averaging added, so the batch order and the optimiser match experiment 425's; run through the
  averaged loop, 2026-01 seed 0 reproduces the stored best-epoch validation loss exactly (0.36901423
  at epoch 4). **S0f** is the weights after epoch 4. **EMA** is the exponential moving average of the
  weights over optimiser steps, `ema = 0.999 * ema + 0.001 * weights`, divided by `1 - 0.999 ** step`
  (zero-debiased); its memory is about 1,000 steps, so it averages the whole 4-epoch trajectory.
  **EMA99** uses decay 0.99 (memory about 100 steps, descriptive). **SWA** is the uniform average of
  the weights over every optimiser step of the last 25% of the budget (the last epoch).
- **Row weights.** Experiment 425's recency weight `0.5 ** (age_days / SHORT_HALF_LIFE_DAYS)`
  multiplies each training row's loss, unchanged in every arm. No other weight is added.
- **Which weights early stopping sees.** None: no network is chosen by the held-out-week
  validation loss. The loss is the unweighted negative log-likelihood on the held-out week, logged
  after each epoch for the raw weights and the EMA weights as a diagnostic.
- **References.** S0f is the matched control. S0 is experiment 425's stored fit, which stops early
  and restores the best epoch by the held-out-week loss, so it also has the selection advantage of
  choosing its epoch on the validation week; S0f has not.
- **SWA-pre (one fold, 2026-01, three seeds).** Pretrain from scratch for 32 epochs (the median of
  the 25 stored pretraining lengths minus 6), keeping the final and the SWA weights (last 25% of the
  steps), saved under `work/pretrained/`. Fine-tune each as above (2 fine-tunes per seed, 6 in all,
  each yielding the networks below). Arms: P0f (final pretrained weights, final fine-tuned weights),
  Pswa (SWA pretrained weights, final fine-tuned weights), Pswa+EMA, and Pswa+SWA. P0f is the
  matched control for the pretraining.
- **Not run.** Pool-of-EMA and Pool-of-SWA over more than three seeds, and any averaging beyond
  these schemes.

### Compute and reuse

- 15 fine-tunes (5 folds times 3 seeds) at under 2 minutes each, one GPU process under 8 GB with
  one feeder core. The SWA-pre arm adds 3 pretrainings and 6 fine-tunes at one fold. Total about
  0.5 GPU-hours.
- Reused: experiment 425's pretrained weights and stored arm A forecasts (seeds 0 to 4), experiment
  470's row context and scorer. Fitted: the 15 fine-tunes and the one-fold pretrainings.
- Monthly refits, cleaned data, and the D15 and D18 start cut come from experiment 425's fold
  builder unchanged.
- Production cost: S0f, EMA, and SWA each serve one network per pass and one fit per monthly refit.
  Pool5 serves five networks and five fits. Pool-of-EMA over three seeds serves three.
