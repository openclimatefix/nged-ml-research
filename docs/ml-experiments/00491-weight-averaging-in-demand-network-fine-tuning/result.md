---
id: "00491"
slug: weight-averaging-in-demand-network-fine-tuning
title: >-
  Can one demand network, trained once with an exponential moving average or a stochastic weight
  average of its weights, recover a useful share of the five-network pool's gain?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: finished
date: 2026-11-01
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
verdict: inconclusive
production_status: not-for-production
adopted_in: none
best_arm: >-
  None. At the eight MVA-only Primaries, the weight-averaged single networks (EMA and SWA) score
  within 0.005 points of the fixed-budget single network, and Pool5 (five networks) scores 0.082
  points better.
headline: >-
  Averaging the weights of one network along its fine-tuning run closed 2% (EMA) and 6% (SWA) of the
  0.082-point gap between a single network and the five-network pool at the eight MVA-only Primaries,
  and 22% and 12% of the 0.073-point gap at the six. A pool of three networks that each used weight
  averaging closed 79% to 86% of the gap at the eight.
take_home_message: >-
  One network trained with a moving average of its weights cannot stand in for a pool of seeds.
  The averaged network scores 0.002 to 0.005 points better than a single network at the eight (the
  seed-noise minimum detectable effect is 0.083), while averaging three seeds gains 0.065 to 0.071.
  The gain of a pool comes from networks that differ in their seed, so serve two seeds rather than
  one averaged network. The gap between one network and five (0.082 points at the eight) is below
  the 0.15 points that the plan needed before a closure share could settle the question, so the
  verdict stays inconclusive.
lessons: >-
  Early stopping on the held-out week is not worth much here: the fixed 4-epoch network (S0f) scores
  within 0.007 points of the early-stopped network (S0) at the eight and the six, though 0.14 points
  better on the six's real-event rows. Weight averaging lowers the held-out-week loss at epoch 4 in
  only 7 of 15 fits, so the averaged weights are no better than the raw weights at the end of a short run. The best tail
  widening is 1.05 at the eight and 1.05 to 1.10 at the six for every arm, so averaging seeds does
  not change the widening.
advice:
  - >-
    Do not replace a seed pool with one weight-averaged network: EMA and SWA close at most 22% of the
    gap to Pool5 on any slice (Figure 2).
  - >-
    Serve two seeds, as experiment 481 found; Pool2 closes 59% of the gap at the eight and 71% at
    the six (Figure 2).
  - >-
    Do not repeat averaging within a run for this network without a longer budget: the fine-tune
    stops at its best epoch after 4 epochs, so there is little trajectory to average (Checks).
---

## Take-home

**One averaged network gains almost nothing over one ordinary network, while a pool of seeds gains
most of the available score.** At the eight MVA-only Primaries the fixed-budget single network
scores 4.762 points, the five-network pool 4.680, EMA 4.760, and SWA 4.757. The three-seed pool of
EMA networks scores 4.697. At the six, on rows with no labelled event, the figures are 3.524, 3.451,
3.508, 3.515, and 3.451. The gap to the five-network pool is 0.082 at the eight and 0.073 at the
six, and the seed-noise minimum detectable effect is 0.083 and 0.112, so the experiment can show
only that weight averaging gains little, and cannot rank the arms closely. Five months and one
fine-tune trajectory per seed make these results interim.

## Figures

![CRPS of each arm](figure_1_score_by_arm.svg)

![Share of the gap closed](figure_2_closure_share.svg)

![Counted contrasts](figure_3_contrasts.svg)

![Example forecasts, Primary D10, day ahead](figure_4_example_D10_day_ahead.svg)

![Example forecasts, Primary D10, 14 days](figure_4_example_D10_14_days.svg)

## Results

Scores are points of effective capacity, 1.15 widening, on experiment 470's `gam14` rows, the mean
of three seeds. The scorer printed the Bonferroni count (54) and the minimum detectable effect for
the mean of three seeds before any contrast: 0.083 at the eight, 0.112 at the six on rows with no
labelled event, 0.169 at the eight on cold nights, and 0.236 at the six on cold nights. Selection
bias applies to every score: the five folds that choose also score.

- **The gap from one network to five is small.** From S0f to Pool5 it is 0.082 points at the eight
  and 0.073 at the six (0.119 and 0.036 on cold nights). Every gap is below the 0.15 points the plan
  set, so the closure shares are reported without a test.
- **Weight averaging gains 0.002 to 0.016 points on the counted slices at the eight and the six.**
  EMA minus S0f: -0.002 at the eight (Bonferroni interval -0.015 to +0.011) and -0.016 at the six
  (-0.033 to -0.002). SWA minus S0f: -0.005 (-0.009 to -0.001) and -0.008 (-0.013 to -0.004). The
  closure shares of the gap to Pool5 are 2% (EMA) and 6% (SWA) at the eight, and 22% and 12% at the
  six. The 95% intervals of the EMA share run from -8% to 11% at the eight and 10% to 36% at the
  six.
- **Averaged networks stay far from the pool.** EMA minus Pool5 is +0.080 at the eight (+0.063 to
  +0.099) and +0.057 at the six (+0.036 to +0.078). SWA minus Pool5 is +0.077 and +0.064. The
  differences from Pool5 are within the minimum detectable effect at the eight (0.083), and the
  plan's condition that the gap exceed 0.15 points fails, so the recommendation rule does not
  recommend a weight-averaged network.
- **Two seeds do what weight averaging cannot.** Pool2 (three pairs of seeds) closes 59% of the gap
  at the eight and 71% at the six. A pool of three EMA networks closes 79% and 99%, and a pool of
  three SWA networks 86% and 91%. A pool of three plain networks (Pool3) scores 4.697 at the eight
  and 3.457 at the six.
- **Weight averaging helps slightly on cold nights and worsens real-event rows.** EMA minus S0f on
  cold nights at the eight is -0.024 (-0.057 to +0.006). SWA minus S0f is -0.014 (-0.021 to
  -0.008), and +0.023 on the six's real-event rows (+0.012 to +0.036). Both are within the minimum
  detectable effect.
- **The q99 rule holds.** EMA, SWA, and S0f differ by at most 0.002 points of q99 pinball loss at
  the eight and the six, and the averaged networks are 0.004 to 0.012 above Pool5 (rule: at most
0.02). Coverage at q99 is 0.991 (EMA) and 0.990 (SWA) at the eight, against 0.992 for Pool5, and
0.985 and 0.983 at the six against 0.985.
- **The best widening is 1.05 at the eight and 1.05 to 1.10 at the six for every arm,** including
  S0 and Pool5, with differences of at most 0.004 points from the 1.15 headline, so averaging seeds
  or weights does not change the widening.
- **The fixed budget scores about the same as early stopping.** S0f minus S0 is -0.002 at the eight,
  +0.007 at the six (+0.004 to +0.010), and -0.14 on real-event rows at the six (-0.14 to -0.08).
- **SWA in pretraining (2026-01 only, three seeds, descriptive).** At the eight: S0f 4.552, P0f
  4.557, Pswa 4.560, Pswa+EMA 4.586, Pswa+SWA 4.564, Pool5 4.492. At the six: S0f 3.932, P0f 3.891,
  Pswa 3.866, Pswa+EMA 3.818, Pswa+SWA 3.857, Pool5 3.889. Averaging weights in pretraining does not
  help at the eight and lowers the six's score by 0.025 against P0f, one fold and below the minimum
  detectable effect.
- **Avoidable-curtailment cost is the one metric on which weight averaging looks as good as the
  pool, within wide intervals.** At f = 0.75 at the signed-MW six on rows with no labelled event,
  over the five months, the cost in millions of pounds is 7.36 for S0f, 6.80 for EMA, 7.18 for SWA,
  7.16 for Pool2, 7.05 for Pool5, 6.52 for the pool of three EMA networks, and 12.88 for the manual
  heuristic. EMA minus S0f is -0.56 (95% interval -1.52 to -0.02), EMA minus Pool5 is -0.25 (-1.00
  to +0.19), and SWA minus Pool5 is +0.14 (-0.01 to +0.28). At f = 0.9 the EMA minus S0f difference
  is -0.12 (-0.34 to +0.00), and at f = 0.5 it is -0.95 (-2.48 to -0.05). The intervals span the
  differences between arms, and cell bootstrap intervals treat each month and Primary as
  independent, so the curtailment ranking of EMA above Pool5 is within noise. These contrasts are
  descriptive and uncounted.
- **Cost.** Each fine-tune took a median of 104 s on a loaded workstation (all four networks of a
run come from that one fit). A pretraining took about 40 s. S0f, EMA, and SWA each serve one network
per pass and need one fit per monthly refit. Pool5 serves five networks and five fits. Serving
latency was not measured.

## Predictions

- **P1 (EMA or SWA closes at least 40% of the gap at the eight): not supported.** The shares are
  2% and 6%. The test itself is inconclusive because the gap is below 0.15 points.
- **P2 (EMA lowers the held-out-week loss in at least 12 of 15 fits): refuted.** EMA is lower in 7
  of 15, by -0.004 to +0.009.
- **P3 (the closure share is under 20%): supported at the eight (2% and 6%),** and not at the six
  for EMA (22%, interval 10% to 36%).
- **P4 (SWA in pretraining closes more of the gap than fine-tune averaging alone): not supported at
  the eight, descriptive at the six.**

## Checks

- **Reproduction.** The early-stopped loop and the new averaged loop both reproduce experiment 425's
  stored best-epoch validation loss for 2026-01, seed 0 (0.36901423, epoch 4 of 10).
- **A plan review found the budget wrong and it was fixed before any scored fit.** The stored
  fine-tune lengths (median 10 epochs) include 6 patience epochs, so the median best epoch is 4. An
  earlier 10-epoch fit was deleted.
- **Which triggers fired.** None of the seven audit triggers fires: no control failed, the results
agree with experiment 481 (a second seed is the largest gain), and no leave-one-out flips a verdict
by construction (month-block intervals only).
- **Median against mean.** Not computed; the contrasts here are quantile-score contrasts only.
- **Limits.** One pretrained network per seed is shared by S0f, EMA, and SWA, so the three are
  correlated. Five months set every score. Pool2 is the quantile mean of seeds (0,1), (0,2), and
  (1,2). The curtailment arm costs are descriptive.

## Plan

The plan in the pull request body and `decision_rule` above was followed, with two deviations
recorded here: the fine-tune budget is 4 epochs (not 10), and the EMA99 arm and the second
pretraining control P0f were added.
