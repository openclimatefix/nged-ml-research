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
  fixes it before any score. 15 fine-tunes of stored pretrained networks (each yielding four
  networks), plus 3 pretrainings and 6 fine-tunes (12 networks) at one fold, about 0.5 GPU-hours on
  one GPU process under 8 GB and one feeder core. Production serving path: not touched (this repository
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
  Averaging the weights of one neural network along its fine-tuning run closed 2% (exponential moving
  average, EMA) and 6% (stochastic weight average, SWA) of the 0.082-point gap in CRPS (a probabilistic
  forecast score; lower is better) between a single network and a five-network pool, at the eight
  MVA-only Primaries. At the six signed-MW Primaries, the same averaging closed 22% and 12% of a
  0.073-point gap. A pool of three networks, each from a different seed and each trained with weight
  averaging, closed 79% to 86% of the gap at the eight.
take_home_message: >-
  One neural network trained with a moving average of its weights did not match a pool of seeds.
  The averaged network scores 0.002 to 0.005 CRPS points better than a single network at the eight
  (the seed-noise minimum detectable effect is 0.083), while a pool of three averaged networks from
  different seeds gains 0.065 to 0.071. In this experiment, the pools that gained came from networks
  that differ in their seed. The gap between one network and five is 0.082 points at the eight. The
  plan needed a gap above 0.15 points before a closure share could settle the question, so the
  verdict stays inconclusive.
lessons: >-
  Early stopping on the held-out week does not improve the score here: the fixed 4-epoch network
  (S0f) scores within 0.007 points of the early-stopped network (S0) at the eight and the six,
  though 0.11 points better on the six's real-event rows. The exponential moving average lowers the
  held-out-week loss at epoch 4 in only 7 of 15 fits, so the averaged weights are no better than the
  raw weights at the end of a short run. The best tail widening is 1.05 at the eight and 1.05 to
  1.10 at the six for every arm, so averaging seeds does not change the widening.
advice:
  - >-
    Do not replace a seed pool with one weight-averaged network: at the eight and the six, EMA and
    SWA close at most 22% of the gap from S0f to Pool5 (Figure 2).
  - >-
    Pool2, the cheapest pool scored here, closes 59% of the gap at the eight and 71% at the six
    (Figure 2); experiment 481 found that a second seed gave the largest gain among the seed counts
    it scored.
  - >-
    Do not repeat averaging within a run for this network without a longer budget: the fine-tune
    reaches its best epoch after 4 epochs, so little trajectory remains to average (Checks).
---

## Take-home

**One averaged neural network gains almost nothing over one ordinary neural network, while a pool of
seeds closes most of the gap to the five-network pool.** The demand networks here forecast power at
16 substations, and CRPS (continuous ranked probability score, in points of effective capacity;
lower is better) scores each forecast. At the eight MVA-only Primaries, the fixed-budget single
network scores 4.762 points, the five-network pool 4.680, the exponential moving average (EMA)
network 4.760, and the stochastic weight average (SWA) network 4.757. The pool of three EMA networks
scores 4.697. At the six signed-MW Primaries, on rows with no labelled event, the same five arms
score 3.524, 3.451, 3.508, 3.515, and 3.451. The gap to the five-network pool is 0.082 points at the
eight and 0.073 at the six. The seed-noise minimum detectable effect is 0.083 at the eight and 0.112
at the six. Each gap is therefore about the size of the noise, so the experiment can show only that
weight averaging gains little. It cannot rank the arms closely. These results are interim, because
they rest on 5 months of data and one fine-tune trajectory per seed.

## Figures

![CRPS of each arm at the eight and the six](figure_1_score_by_arm.svg)

Figure 1. CRPS of each arm.

![Share of the gap closed: EMA and SWA close at most 22%, pools of seeds 59% to 99%](figure_2_closure_share.svg)

Figure 2. Share of the gap from S0f to Pool5 that each arm closes.

![Counted contrasts of EMA and SWA with S0f and Pool5, with Bonferroni intervals](figure_3_contrasts.svg)

Figure 3. Counted contrasts with Bonferroni intervals.

![Example forecasts, Primary D10, day ahead](figure_4_example_D10_day_ahead.svg)

![Example forecasts, Primary D10, 14 days](figure_4_example_D10_14_days.svg)

Figure 4. Example forecasts of one Primary.

## Results

**Every score is in CRPS points of effective capacity at the 1.15 tail widening, the mean of three
seeds on experiment 470's rows.** Experiment 470 (the first-release demand forecaster comparison)
supplies the rows, which hold the eight MVA-only Primaries (substations metered in MVA only) and
the six signed-MW Primaries (substations metered in signed MW). Each arm is scored over 15
fine-tunes: 5 held-out months times 3 seeds. A negative contrast means the first arm scores lower,
which is better. The scorer printed the Bonferroni count (54) and the minimum detectable effect
for the mean of three seeds before any contrast: 0.083 at the eight, 0.112 at the six on rows with
no labelled event, 0.169 at the eight on cold nights, and 0.236 at the six on cold nights. Every
score is selection-biased, because the five folds that choose also score.

Arms:

- **S0f:** one network fine-tuned for a fixed 4 epochs, final weights.
- **S0:** one network fine-tuned with early stopping (experiment 425's stored fit; the production
  recipe).
- **EMA, EMA99, SWA:** the same fine-tune read out as an exponential moving average of the weights
  (decay 0.999, or 0.99 for EMA99) or a stochastic weight average over the last epoch.
- **Pool2, Pool3, Pool5:** the quantile mean of 2, 3, or 5 seeds of S0 (experiment 425's stored
  seeds). **PoolEMA3, PoolSWA3:** the quantile mean of 3 EMA or SWA networks from different seeds.
- **P0f, Pswa, Pswa+EMA, Pswa+SWA:** the one-fold pretraining arms (final or SWA pretrained
  weights, then the fine-tune read out as in the lines above).

- **The gap from one network to five is small.** From S0f to Pool5, the gap is 0.082 points at
  the eight and 0.073 at the six (0.119 and 0.036 on cold nights). The plan required a gap above
  0.15 points before a closure share could be tested. Every gap here is below that, so the closure
  shares are reported without a test.
- **Weight averaging gains 0.002 to 0.016 points at the eight and the six.** EMA minus S0f: -0.002
  at the eight (Bonferroni interval -0.015 to +0.011) and -0.016 at the six (-0.033 to -0.002). SWA
  minus S0f: -0.005 (-0.009 to -0.001) and -0.008 (-0.013 to -0.004). The closure shares of the
  gap to Pool5 are 2% (EMA) and 6% (SWA) at the eight, and 22% and 12% at the six. The 95% intervals
  of the EMA share run from -8% to 11% at the eight and 10% to 36% at the six.
- **Averaged networks stay far from the pool.** EMA minus Pool5 is +0.080 at the eight (+0.063 to
  +0.099) and +0.057 at the six (+0.036 to +0.078). SWA minus Pool5 is +0.077 and +0.064. The
  differences from Pool5 are within the minimum detectable effect at the eight (0.083). The plan's
  condition that the gap exceed 0.15 points fails. The plan's recommendation rule is therefore not
  met.
- **Two seeds do what weight averaging cannot.** Pool2 (three pairs of seeds) closes 59% of the gap
  at the eight and 71% at the six. A pool of three EMA networks closes 79% and 99%, and a pool of
  three SWA networks 86% and 91%. A pool of three plain networks (Pool3) scores 4.697 at the eight
  and 3.457 at the six.
- **Weight averaging helps slightly on cold nights.** EMA minus S0f on cold nights at the eight is
  -0.024 (-0.057 to +0.006), and SWA minus S0f is -0.014 (-0.021 to -0.008). Both are within the
  cold-night minimum detectable effect (0.169).
- **SWA worsens the six's real-event rows.** SWA minus S0f is +0.023 on those rows (+0.012 to
  +0.036). EMA minus S0f is -0.008 (-0.050 to +0.033). The scorer printed no minimum detectable
  effect for the real-event rows.
- **The q99 rule holds.** EMA, SWA, and S0f differ by at most 0.005 points of q99 pinball loss at
  the eight and the six (EMA is 0.004 better than S0f at the six). The averaged networks are 0.006
  to 0.010 above Pool5 (rule: at most 0.02). Coverage at q99 is 0.991 (EMA) and 0.990 (SWA) at the
  eight, against 0.992 for Pool5, and 0.985 and 0.983 at the six against 0.985.
- **The best widening is 1.05 at the eight and 1.05 to 1.10 at the six for every arm,** including
  S0 and Pool5, with differences of at most 0.004 points from the 1.15 headline, so averaging seeds
  or weights does not change the widening.
- **The fixed budget scores about the same as early stopping.** S0f minus S0 is -0.002 at the eight,
  +0.007 at the six (+0.004 to +0.010), and -0.11 on real-event rows at the six (-0.14 to -0.08).
- **SWA in pretraining (2026-01 only, three seeds, descriptive) does not help at the eight and
  helps at the six.** At the eight: S0f 4.552, P0f 4.557, Pswa 4.560, Pswa+EMA 4.586, Pswa+SWA
  4.564, Pool5 4.492. At the six: S0f 3.932, P0f 3.891, Pswa 3.866, Pswa+EMA 3.818, Pswa+SWA
  3.857, Pool5 3.889. At the six, Pswa scores 0.025 below P0f and Pswa+EMA 0.073 below P0f, and
  Pswa+EMA scores below Pool5. This is one fold and the differences are below the six's minimum
  detectable effect.
- **Avoidable-curtailment cost is the one metric where EMA looks as good as the pool, within wide
intervals.** At f = 0.75 at the signed-MW six on rows with no labelled event, over the 5 months, the
cost in millions of pounds is 7.36 for S0f, 6.80 for EMA, 7.18 for SWA, 7.16 for Pool2, 7.05 for
Pool5, 6.52 for the pool of three EMA networks, and 12.88 for the manual heuristic. EMA minus S0f is
-0.56 (95% interval -1.52 to -0.01), EMA minus Pool5 is -0.25 (-1.00 to +0.19), and SWA minus Pool5
is +0.14 (-0.01 to +0.28). At f = 0.9 the EMA minus S0f difference is -0.12 (-0.33 to +0.00), and at
f = 0.5 it is -0.95 (-2.47 to -0.05). SWA is not as good as the pool at f = 0.5: SWA minus Pool5 is
+0.66 (+0.32 to +1.00).
- **The curtailment ranking of EMA above Pool5 is within noise.** The intervals of EMA minus Pool5
  include zero, and the cell bootstrap treats each month and Primary as independent. The
  curtailment-cost contrasts are descriptive and uncounted.
- **A fine-tune takes a median of 104 s on a loaded workstation, and all four networks of a run come
  from that one fit.** A pretraining takes about 40 s. S0f, EMA, and SWA each serve one network per
  pass and need one fit per monthly refit. Pool5 serves five networks and five fits. Serving latency
  was not measured.

## Predictions

- **P1 (EMA or SWA closes at least 40% of the gap at the eight): not supported.** The shares are
  2% and 6%. The P1 test is inconclusive because the gap is below 0.15 points.
- **P2 (EMA lowers the held-out-week loss in at least 12 of 15 fits): refuted.** EMA is lower in 7
  of 15. The range of EMA minus raw weights is -0.009 to +0.004.
- **P3 (the closure share is under 20%): supported at the eight (2% and 6%), not supported at the
  six for EMA,** because the point estimate of 22% exceeds 20%, although its interval (10% to 36%)
  reaches below.
- **P4 (SWA in pretraining closes more of the gap than fine-tune averaging alone): not supported at
  the eight, and descriptively supported at the six** (Pswa+EMA 3.818 against 3.865 for EMA alone).

## Checks

- **Both fine-tune loops reproduce experiment 425's stored validation loss.** The loop that stops at
  the best epoch and the loop that averages weights each give 0.36901423 for 2026-01, seed 0 (epoch
  4 of 10).
- **The fine-tune budget is 4 epochs, not the 10 planned.** The stored fine-tune lengths (median
  10 epochs) include 6 patience epochs, so the median best epoch is 4.
- **None of the seven audit triggers fired.** No control failed, and the results agree with
  experiment 481 (a second seed gives the largest gain among the seed counts it scored).
- **The median forecast against the mean forecast was not computed.** The contrasts here are
  quantile-score contrasts only.
- **Five months, shared pretrained networks, and descriptive curtailment arms limit these
  results.** One pretrained network per seed is shared by S0f, EMA, and SWA, so their scores are
  correlated. Pool2 is the quantile mean of seeds (0,1), (0,2), and (1,2).

## Plan

**The plan was followed, with two deviations.** The fine-tune budget is 4 epochs, and the EMA99 arm
and the P0f pretraining control were added.
