---
id: "00449"
slug: pretraining-with-lags-and-surprise-inputs-on-the-s1-network
title: >-
  Does pretraining the S1 demand network on ERA5 and CAMS weather help when the pretraining stage
  itself sees the recent-power lags and the surprise inputs?
theme: production-reference-and-best-guess
also_relevant_to:
  - lagged-power-and-error-correction
status: done
kind: other
protocol: demand-v2
size: >-
  complex. Stored in this experiment's private folder only: 58 pretrained networks, 73 fine-tuned
  networks (131 in all), and 25 cross-fitted lagless tables built on ERA5 and CAMS weather (5 folds
  by seeds 0 to 4, 6 modules each, 150 lagless-module fits), with their forecasts; experiments 404,
  428, 439, and 448 are read and never written. More than one design is defensible (how the lagless
  forecast is built for the pretraining years, whether the lag inputs are masked in pretraining), so
  the plan fixed each choice before any score was read. Production serving path: not touched (this
  repository holds none). Degradation rule: not touched. Callers: the scripts copy experiment 448's
  fit and score code and import the folders of experiments 397, 404, 418, 422, 428, and the
  `studies` evaluation code.
hypothesis: >-
  Pretraining on ERA5 and CAMS weather did not help the S1 network in experiment 448 because the
  pretraining stage hid the lag and surprise inputs, so the lag pathway was learned only from the
  shorter ensemble history. If the pretraining stage instead sees the three origin lags and the
  surprise at each lag (the surprise computed from a cross-fitted lagless forecast on ERA5 and CAMS
  weather), the fine-tuned S1 network scores lower days 1 to 10 CRPS than S1.
prediction: >-
  Written before any fit and before any score is read. P0. The reproduction arm PT0 matches
  experiment 448's S1PT: +0.05 CRPS points against S1 at the signed-MW six and at the MVA-only
  eight, within the seed spread. P1. PT1 (lags and surprise live in pretraining) beats S1 by 0.03 to
  0.10 points at the signed-MW six and by 0.05 to 0.15 at the MVA-only eight, which is the size
  experiment 425 found for a network without lags (-0.050 and -0.124). P2. PT1 beats PT0 by at
  least 0.05 points, because PT0 lost the S1 lead. P3. PT2 (token dropout 0.3 on the lag inputs)
  scores within 0.03 points of PT1 on clean rows (C5); its q99 pinball loss on rows with a lag jump
  is reported, not tested. P4. Fine-tuning at the full learning rate (PT1-full) is worse than at
  2e-4 by under 0.03 points (C9). P5. PT1b (lags live, surprise masked in pretraining) recovers part
  of PT1's gain without any lagless table; if PT1 loses to S1 and PT1b does not, the ERA5-built
  surprise is the cause (C8). The
  prediction could fail in either direction: PT1 could also lose to S1, which would mean the
  surprise inputs built from ERA5 differ too much from the ones built from the ensemble.
decision_rule: >-
  Four screen folds (test months 2026-06, 2026-01, 2025-10, 2026-03), seeds 0 to 2, monthly refits,
  1.15 widening unless stated, CRPS at days 1 to 10 in points of effective capacity (a negative
  difference is a gain), per metering group (signed-MW six; MVA-only eight; D15 and D18
  report-only, with the experiment 399 start cut) and per Primary. Every number is interim and
  carries no 99% claim. Contrasts are paired by fold and seed with the month-block bootstrap of
  `studies.eval_protocol`, at a level Bonferroni-corrected over 13 contrasts at the signed-MW six
  and the MVA-only eight (26 tests; D15 and D18 are outside the count). C1 PT1 minus S1. C2
  PT1-full minus S1. C3 PT2 minus S1. C4 PT1 minus PT0 (the
  effect of showing the lags to the pretraining stage). C5 PT2 minus PT1. C6 PT0 minus S1 (the
  reproduction of experiment 448). C7 PT1b minus S1. C8 PT1 minus PT1b (the effect of the surprise
  inputs in pretraining). C9 PT1-full minus PT1. C-pool the pool contrasts above. All contrasts are
  read at both groups, on the four screen folds and on the five folds including 2025-12. Pretraining
  adds a stage to the pipeline, so the
  smallest gain worth porting is 0.03 points. A PT arm advances if its pooled point gain against S1
  is at least 0.03 points and the interval of that contrast lies wholly below zero (the brief's
  rule), at the signed-MW six or at the MVA-only eight, the point gain is negative in at least 3 of
  the 4 folds (the pooled paired difference of the fold's three seeds, no interval per fold), the
  other group is not above +0.02, and q95 and q99 pinball loss on clean rows is no more than +0.02
  above S1's. The minimum detectable
  effect is computed per group from the paired fold-by-seed differences of S1 seeds 0 to 2 and
  printed before any contrast is read. Real-event rows are reported with and without the largest
  event, and so is the cold-winter-night slice. The encroachment sweep of experiment 392 runs
  prediction-only on the saved networks of any arm that advances. Reported beside the all-rows
  result for every arm: the Christmas and bank-holiday slice (22 to 31 December, fold 2025-12), the
  cold-winter-night slice, D15 and D18, and all 16 Primaries together. Experiment 425's arm A
  (stored seeds 0 to 4, widened by 1.15) and BG2 are reference rows on the same half-hours. The
  experiment ends at this screen; the full scored year is a later experiment.
benchmark: >-
  S1 seeds 0 to 2 from experiment 428 (raw forecasts, 1.15 widening); PT0 is experiment 448's S1PT
  (seeds 0 to 2); the monthly XGBoost recipe, BG2, `manual_heuristic`, and `climatology` as read by
  experiment 448.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS from the 11 quantiles at days 1 to 10; coverage and pinball loss at q01, q05, q95, q99; cold
  winter nights; sunny half-hours; avoidable-curtailment cost; clean rows and rows with a lag jump,
  with and without the largest event. Per group and per Primary.
verdict: partly supported
best_arm: >-
  PT1-pool5, a five-seed pool of S1 pretrained on ERA5 and CAMS rows that carry real origin lags and
  the cross-fitted surprise: CRPS 2.740 at the signed-MW six and 3.416 at the MVA-only eight (days 1
  to 10, four screen folds, the 579,044 half-hours that every arm and reference shares, 1.15
  widening), against 2.857 and 3.556 for one fit of S1. PT1-pool5 is not the best pretrained demand
  network: experiment 425's arm A averaged over five seeds scores 2.763 and 3.373 and is ahead over
  all 16 Primaries.
headline: >-
  Pretraining with real lags and the cross-fitted surprise reversed experiment 448's loss: PT1 beat
  S1 by 0.082 points at the signed-MW six and 0.068 at the MVA-only eight, and beat the masked-lag
  pretraining by 0.129 and 0.110. Most of the gain at the six comes from one fold: without 2026-01
  it falls from 0.082 to 0.036, below the minimum detectable effect of 0.065. Experiment 425's arm
  A, averaged over five
  seeds, is ahead of PT1-pool5 at D15 and D18 (0.58) and over all 16 Primaries (0.078, Bonferroni
  interval above zero), 0.043 ahead at the eight (interval spans zero), and 0.023 behind it at the
  six (interval spans zero, all from 2026-01). PT1-pool5 is ahead of BG2
  at the six and the eight, ties it over all 16 Primaries, and is behind it at D15 and D18 (by about
  1.2 points) and on cold winter nights at the six. The evidence does not support PT1-pool5 as the
  first-release demand candidate.
lessons: >-
  Showing live lags to the pretraining stage removed the harm of experiment 448 (PT1 against PT0
  -0.129 and -0.110). Showing the surprise inputs lowered CRPS by 0.042 at the six (PT1 against
  PT1b) and
  0.016 at the eight (interval spans zero). Token dropout on the lags (PT2) was slightly worse at
  the six and slightly better at the eight, and the full fine-tuning learning rate (PT1-full) was
  slightly worse at the six; the effects are small but their Bonferroni intervals exclude zero.
  Pooling five seeds of PT1 gained a further 0.035 and 0.037 over one PT1 fit. All 55 PT1, PT1b, and
  PT2 pretraining runs stopped at the 60-epoch cap, so pretraining may be too short.
take_home_message: >-
  Pretraining the S1 network on past-weather estimates helps once the pretraining rows carry real
  recent-power inputs: the S1 neural network pretrained this way scores 0.08 points better than S1
  at the signed-MW six and 0.07 at the
  MVA-only eight, and 0.13 and 0.11 better than the earlier masked pretraining. Most of the gain
  at the six comes from one test month, and experiment 425's simpler recipe (no surprise inputs,
  five seeds averaged) is ahead of the pooled network at the hardest substations (D15 and D18) and
  over all 16 Primaries. BG2 is still ahead at D15 and D18 and on cold winter nights at the six.
  All numbers are interim, from four test months.
advice:
  - >-
    Pretrain demand networks on ERA5 and CAMS rows that carry real origin lags. Masking the lags in
    pretraining worsened CRPS by 0.13 and 0.11 points against live lags (C4).
  - >-
    The surprise inputs lowered CRPS by 0.042 at the six and 0.016 at the eight (C8), but they are
    not shown to be necessary: experiment 425's arm A, which has none, is ahead of PT1-pool5 over
    all 16 Primaries, although the two recipes differ in more than the surprise inputs.
  - >-
    Do not adopt PT1-pool5 as the first-release demand candidate. The candidates the evidence
    supports are 425's arm A, its pool of five, and S1. A first-release choice also needs the
    avoidable-curtailment cost, where BG2 leads at the scaled limit f = 0.9.
production_status: not-for-production
adopted_in: none
code: ml-experiments/00449-pretraining-with-lags-and-surprise-inputs-on-the-s1-network/
---

**Take-home message: pretraining helps S1 once the pretraining rows carry real recent-power inputs,
but the gain is small, rests on one test month at the six, and experiment 425's simpler recipe is
ahead of the pooled network over all 16 Primaries.** PT1 scores 0.082 CRPS points below S1 at the
signed-MW six and 0.068 below at the MVA-only eight (four screen folds, seeds 0 to 2, injected-world
clean rows). Experiment 448's masked-lag pretraining scored 0.048 and 0.042 above S1, so the live
lags are the difference (PT1 against PT0: -0.129 and -0.110). All numbers are interim, from four
test months.

## Results

**Which rows each table uses.** The first table and the cold-night columns use the 579,044
half-hours that every arm and reference shares (real-world clean rows, four folds). The contrasts
below use injected-world clean rows (275,534 at the six, 381,834 at the eight, 9,567 at D15 and D18)
unless a row says "real rows" (83,458 at D15 and D18). D15 and D18 are headlined on real rows; the
injected-row figure follows, labelled.

**The minimum detectable effect is 0.065 points at the six, 0.080 at the eight, and 0.98 (injected
rows) to 1.18 (real rows) at D15 and D18.** The minimum detectable effect comes from the seed spread
of S1, for two single-seed
arms. For a pool against a pool or a fixed reference the minimum detectable effect is a loose upper
bound. The Bonferroni
intervals count 26 tests (13 contrasts at the six and the eight); D15 and D18 are outside that
count.

| Arm | Six | Eight | D15 and D18 | Six, cold nights | Eight, cold nights |
|---|---|---|---|---|---|
| S1 | 2.857 | 3.556 | 9.309 | 2.784 | 2.957 |
| PT0 (448's masked-lag pretraining) | 2.910 | 3.598 | 10.046 | 2.784 | 2.928 |
| PT1 | 2.772 | 3.467 | 9.205 | 2.569 | 2.799 |
| PT1-full | 2.784 | 3.465 | 9.326 | 2.607 | 2.826 |
| PT1b (live lags, no surprise) | 2.817 | 3.487 | 9.377 | 2.674 | 2.811 |
| PT2 (lag dropout 0.3) | 2.796 | 3.442 | 9.341 | 2.697 | 2.775 |
| S1-pool5 | 2.802 | 3.501 | 8.952 | 2.722 | 2.821 |
| PT1-pool5 | 2.740 | 3.416 | 8.865 | 2.537 | 2.708 |
| 425 arm A, five seeds | 2.763 | 3.373 | 8.283 | 2.636 | 2.656 |
| BG2 | 2.845 | 3.543 | 7.687 | 2.405 | 2.848 |
| Monthly XGBoost (1.15 widening) | 3.026 | 3.621 | 8.852 | 2.811 | 3.020 |
| `manual_heuristic` | 3.554 | 4.171 | 12.088 | 4.820 | 4.450 |
| `climatology` | 3.946 | 4.599 | 22.740 | 5.159 | 4.799 |

Days 1 to 10 CRPS in points of effective capacity, 1.15 widening. The 425 arm A row averages the
quantiles of five seeds and used experiment 397's fit loop; PT1-pool5 mixes the distributions of
five seeds. The two recipes differ in more than the surprise inputs.

**Contrasts (injected-world clean rows, month-block bootstrap, 95% interval, Bonferroni interval in
brackets):**

- **C1 PT1 minus S1.** Six -0.082 (-0.108 to -0.057; -0.124 to -0.044). Eight -0.068 (-0.103 to
  -0.035; -0.124 to -0.018). D15 and D18 on real rows -0.044 (Bonferroni -0.191 to +0.113; by fold
  +0.917, +0.082, -0.158, -0.264). On 9,567 injected rows D15 and D18 is -0.169 (95% -0.287 to
  -0.047; Bonferroni -0.357 to +0.029). Both D15 and D18 figures are far smaller in magnitude than
  the minimum detectable effect of about 1.0.
- **C4 PT1 minus PT0.** Six -0.129, eight -0.110, D15 and D18 on real rows -0.742. Every Bonferroni
  interval lies wholly below zero.
- **C6 PT0 minus S1.** Six +0.048 and eight +0.042, which reproduces experiment 448's loss. The
  plan's refit check (a validation loss equal to the stored one) was run only for 2025-12 and its
  result is not stored, so this page does not claim it.
- **C8 PT1 minus PT1b.** Six -0.042 (-0.063 to -0.022 Bonferroni), eight -0.016 (interval spans
  zero).
- **C3 PT2 minus S1.** Six -0.057, eight -0.086, D15 and D18 on real rows +0.102 (Bonferroni -0.047
  to +0.280).
- **C5 PT2 minus PT1.** Six +0.025 (Bonferroni +0.009 to +0.041) and eight -0.018 (-0.030 to
  -0.005), D15 and D18 on real rows +0.146 (-0.035 to +0.321): PT2 is slightly worse at the six and
  slightly better at the eight.
- **C9 PT1-full minus PT1.** Six +0.011 (Bonferroni +0.003 to +0.019), eight +0.001, D15 and D18 on
  real rows +0.113 (+0.061 to +0.170): the full learning rate is slightly worse.
- **Pool contrast 1, PT1-pool5 minus S1-pool5.** Six -0.058, eight -0.058, D15 and D18 -0.052 on
  real rows (-0.210 on injected rows). **Pool contrast 3, PT1-pool5 minus PT1.** Six -0.035, eight
  -0.037.

**By fold at the six, the gain of PT1 over S1 is -0.281 in 2026-01 and -0.004, -0.047, and -0.047 in
the other three folds.** Leaving 2026-01 out, the four-fold gain is -0.036, below the minimum
detectable effect of 0.065. The gain is negative in 4 of 4 folds on injected rows, but on real rows
2025-10 is +0.018, so the count is 3 of 4 on real rows. The gain at the eight is carried by 2026-03:
without it the eight is -0.026. In 2026-01 S1 is unusually poor (3.188 at the six, against 2.907 for
BG2).

**The fifth fold, 2025-12, cannot be read as a gain at the six.** Over five folds C1 is -0.272 at
the six, -0.114 at the eight, and +0.200 at D15 and D18 on real rows (a harm; Bonferroni +0.017 to
+0.407). The minimum detectable effect at the six rises to 0.34 on five folds because S1's seeds
diverge in 2025-12, where S1 fails across the whole month (2025-12 alone is -1.006). Over 1 to 21
December the six score S1 3.514, PT1 3.021, and BG2 3.181. Over 22 to 31 December (days 1 to 10)
they score S1 4.223, PT1 2.441, PT1-pool5 2.312, BG2 2.323, and 425 arm A 2.092. At D15 and D18 that
week BG2 scores 2.344 and PT1-pool5 3.268.

**Experiment 425's arm A, averaged over five seeds, is ahead of PT1-pool5 overall.** PT1-pool5 minus
425 arm A on shared rows over four folds is -0.023 at the six (Bonferroni -0.059 to +0.012; the
whole gap comes from 2026-01, -0.132, with +0.001, +0.003, and +0.029 in the other folds), +0.043 at
the eight (Bonferroni -0.003 to +0.086), +0.582 at D15 and D18 (Bonferroni +0.330 to +0.830), and
+0.078 over all 16 Primaries (Bonferroni +0.030 to +0.138). Over five folds the figures are +0.003,
+0.036, +0.574, and +0.076. The surprise inputs plus the pool are therefore not shown to add to
425's recipe.

**Against the other comparators, PT1-pool5 minus each, four folds (Bonferroni intervals), at the
six, the eight, D15 and D18, and all 16 Primaries:**

- **S1-pool5.** -0.062 (-0.100 to -0.029), -0.084 (-0.143 to -0.029), -0.087 (-0.246 to +0.085),
  -0.077 (-0.116 to -0.039).
- **BG2.** -0.105 (-0.184 to -0.024), -0.127 (-0.199 to -0.047), +1.178 (+0.459 to +2.046), +0.022
  (-0.082 to +0.131). On five folds the six is -0.118 (-0.206 to -0.027) and all 16 Primaries tie.
  BG2 leads on cold winter nights at the six by 0.10 to 0.13 (interval spans zero).
- **Monthly XGBoost (1.15 widening).** -0.286, -0.204, +0.013 (-0.376 to +0.360), -0.210.
- **Experiment 448's S1PT-pool5.** -0.121, -0.129, -0.957, -0.216. `manual_heuristic` and
  `climatology` are beaten everywhere by 0.76 to 14 points.

**Avoidable-curtailment cost: no candidate wins at every limit, so the maintainer must choose the
threshold f.** The table gives the cost over the four folds, signed-MW six, lead 12 to 36 hours, in
millions of pounds, with a 95% interval from resampling the 24 cells (6 series by 4 test months)
with replacement 20,000 times. Each cost is the energy-proxy cost of the forecast minus that of a
perfect forecast on the same half-hours. The intervals on totals are wide because a few cells
dominate; the paired differences below are narrower.

| Arm | f = 0.9 | f = 0.75 | f = 0.5 |
|---|---|---|---|
| BG2 | 1.28 (0.09 to 3.47) | 6.40 (2.21 to 12.35) | 19.31 (13.46 to 26.19) |
| PT1 | 1.33 (0.07 to 3.69) | 4.82 (1.77 to 9.48) | 16.47 (11.33 to 22.93) |
| PT1-pool5 | 1.35 (0.08 to 3.73) | 4.86 (1.79 to 9.55) | 16.32 (11.29 to 22.52) |
| 425 arm A, five seeds | 1.70 (0.08 to 4.74) | 5.81 (1.63 to 12.55) | 17.38 (10.97 to 26.26) |
| S1 | 1.76 (0.07 to 5.02) | 6.27 (1.87 to 13.52) | 18.98 (11.92 to 29.32) |
| S1-pool5 | 1.76 (0.07 to 5.01) | 5.82 (1.86 to 12.23) | 18.01 (11.67 to 26.87) |
| `manual_heuristic` | 1.83 (0.14 to 4.97) | 10.28 (3.35 to 20.77) | 29.92 (19.12 to 44.80) |

- **PT1-pool5 minus BG2.** +0.07 at f = 0.9 (-0.11 to +0.33), -1.54 at f = 0.75 (-3.26 to -0.29),
  and -3.00 at f = 0.5 (-4.52 to -1.75). PT1-pool5 is cheaper than BG2 at the two lower limits and
  level at f = 0.9.
- **PT1-pool5 minus 425 arm A.** -0.35 (-1.02 to +0.00), -0.96 (-3.08 to +0.30), and -1.06 (-4.04 to
  +0.78): not separable.
- **425 arm A minus S1.** -0.07 (-0.36 to +0.13), -0.46 (-1.19 to +0.17), and -1.60 (-3.37 to
  -0.09).
- **BG2 against 425 arm A and S1.** BG2 is cheaper at f = 0.9 (S1 minus BG2 +0.48, interval -0.15 to
  +1.64; 425 arm A minus BG2 +0.42, -0.02 to +1.28), and dearer at f = 0.5 (S1 minus BG2 -0.34, 425
  arm A minus BG2 -1.94), with every one of those intervals spanning zero.

**PT1 has lower q95 and q99 pinball loss than S1 at the six, and the lower tail is over-wide for
every S1-family arm.** On clean rows q95 and q99 pinball loss of PT1 is at or below S1's at the
six (0.676 and 0.184 against 0.718 and 0.191). The lower tail is over-wide for every S1-family arm
(q05 coverage 0.010 to 0.013 at the six, against 0.05 nominal). At D15 and D18 PT1b has a poor q95
pinball loss (2.739 against 2.261 for S1) and is the one pretrained arm to avoid there.

**Raw forecasts, and widening of 1.05 and 1.10, leave the contrasts unchanged.** PT1 minus S1 at the
six is -0.089 raw, -0.086 at 1.05,
-0.084 at 1.10, and -0.082 at 1.15. The pool contrast 1 is -0.062, -0.061, -0.059, and -0.058. The
1.15 widening is slightly too wide for a pool: PT1-pool5 at the six scores 2.8096 raw, 2.8052 at
1.05 and at 1.10, and 2.8090 at 1.15, and at D15 and D18 7.753 raw against 7.840 at 1.15.

**The advance rule is met by PT1 against S1 on injected rows.** Point gain at least 0.03 with the
interval wholly below zero (six and eight), negative in at least 3 of 4 folds, the other group not
above +0.02, and the clean-row tails within +0.02 at the six and the eight (D15 and D18 q99 is
+0.030). The rule does not compare with 425's arm A, and the comparison with 425's arm A is where
the evidence is weakest. PT1 was chosen as the best arm for the pool by a near
tie (mean gain 0.075 against 0.072 for PT2 on injected rows). The encroachment sweep of experiment
392, required of an advancing arm, has not been run.

### What the evidence supports

- **Supported.** Live lags in pretraining reverse experiment 448's harm. PT1 beats S1 on 3 of 4
  folds at the eight and on 4 of 4 (injected rows) or 3 of 4 (real rows) at the six. PT1-pool5 beats
  BG2 at the six and the eight on four and five folds.
- **Not supported.** PT1-pool5 as the first-release demand candidate, because 425's arm A is ahead
  over all 16 Primaries. Any claim at D15 and D18, where BG2 leads PT1-pool5 by about 1.2 points.
  Any claim on cold winter nights at the six, where BG2 leads by 0.10 to 0.13. A tie with BG2 over
  all 16 Primaries is all that is shown (+0.022 on four folds, -0.020 on five).
- **Production cost.** Serving a pool of five needs five networks, five forward passes, and five
  ensemble-weather lagless modules (experiment 404's further module, one per seed), because each
  member reads its own seed's table. The ERA5 lagless modules are used in training only. A monthly
  refit needs about 75 fits against eight for S1: 5 ERA5 tables of 6 modules (30), 5 ensemble tables
  of
  6 cross-fit modules plus 1 further module (35), five pretraining runs of 60 epochs on about 7.4
  million rows, and five fine-tunes. Sharing one lagless table across the members is untested.
  Training
  also needs an ERA5 and CAMS feed.

### Checks

- **Mean-forecast checks.** Median minus mean, mean bias, and mean absolute error per group are in
  `figure_tables/mean_forecast_checks.csv`. They rank the arms as CRPS does at the six and the eight
  (PT1 4.923 against S1
  5.072 at the six; PT1-pool5 4.931 against S1-pool5 5.056). Every arm under-forecasts the mean:
  about -1.7 points at the six and -8.0 to -11.2 at D15 and D18. No audit trigger 3 signal. The
  checks omit 425's arm A and BG2.
- **Pretraining length.** All 55 PT1, PT1b, and PT2 pretraining runs stopped at the 60-epoch cap, so
  pretraining may be too short.
- **Leakage.** Each of the 25 ERA5 lagless tables (seeds 0 to 4, five folds) covers 2019-09-01 to
  the half-hour before the refit, fits each module on 5 of 6 contiguous blocks and forecasts the
  sixth, and repeats no `(primary, grid)`. No table holds a half-hour at or after its refit.
- **Lag semantics.** PT1 pretrains through the same input class as fine-tuning, so the lags come
  from the same origin-lag function over the same power from 2019-09. The stages differ in the
  surprise source, in the spread columns (set to the fold's median per lead band in pretraining),
  and, for PT2, in the dropout.
- **Not done.** The encroachment sweep, the figures (headline chart by lead day, contrasts by fold,
  surprise distribution by source, example forecasts, lag-jump event plots), and the nulled-input
  interval check for PT2. The tables for them are under `figure_tables/`.

## Question

Experiment 448 pretrained the S1 network on ERA5 and CAMS weather with every recent-power input
masked, then fine-tuned it on ensemble rows with the inputs live. The result lost to S1 by 0.05
points at the signed-MW six and at the MVA-only eight. The maintainer expected that pretraining
across all history and all Primaries would add signal. This experiment asks whether the loss came
from the masking: the lag pathway of the pretrained network never saw a lag, so fine-tuning on the
shorter ensemble history could not recover it.

## Context

- **S1 recipe.** Experiment 447's width-256 finding is interim and unaudited, so the S1 reference
  keeps hidden width 64.
- **Reused stored forecasts.** Only seeds 0 to 2 of S1 and PT0 are used (both are stored to seed 4);
  the minimum detectable effect comes from S1's seeds, so the PT arms' own seed spread is read from
  the result. S1 seeds 0 to 2 (experiment 428) and PT0, which is experiment 448's
  S1PT (stored networks and forecasts in experiment 448's private folder). Their rows are the same
  four folds, worlds, and row set, so no S1 network is refitted and PT0 is refitted for 2025-12
  only. The reproduction check (a refit of one PT0 fold and seed with a validation loss equal to the
  stored value) was run only for 2025-12, and its result is not stored.
- **Experiment 425 pretrained with real lags (Opus audit of experiment 448).** Experiment 425's
  pretraining rows carried real lagged power, so its pretraining gained 0.044 points at the
  signed-MW
  six and 0.107 at the MVA-only eight on experiment 448's rows, while experiment 448's masked-lag
  pretraining lost. 425's single-seed arm A (no surprise inputs, no pool) ties the five-seed S1 pool
  at the six (2.809) and is 0.069 better at the eight. PT1 is therefore 425's arm A with the S1
  surprise inputs added: pretrained with real lags and the cross-fitted lagless surprise. The
  prediction P1 used 425's gains (-0.050 and -0.124 on its own rows) as the expected size.
- **Fifth fold for the holiday slice.** The Christmas and bank-holiday slice (22 to 31 December,
  where
  BG2 leads) lies in the 2025-12 fold, which is not one of the four screen folds. Every arm is
  therefore also fitted for 2025-12 (experiment 448 stored S1 for it; PT0 is refitted there). The
  advance rule reads the four screen folds only. The fifth fold is read for the Christmas slice and
  for the five-fold table.
- **Standing practices followed.** Monthly refits; cleaned data (flagged and start-cut half-hours
  dropped); the D15 and D18 cut at 2025-03-01 applied; every network saved in the data folder; draft
  PR; Sonnet for every review (code before the first fit, results, and write-up) and one Opus audit.
  Pools used the spawn method.

## Arms

| Arm | Pretraining stage | Fine-tuning stage | New neural networks |
|---|---|---|---|
| S1 | none | ensemble rows, lr 2e-3 | 0 (stored) |
| PT0 | ERA5 and CAMS, lags and surprise masked | lr 2e-4 | 0 (experiment 448's S1PT) |
| PT1 | ERA5 and CAMS, lags and surprise live | lr 2e-4 | 12 + 12 |
| PT1-full | the PT1 pretrained network | lr 2e-3 | 12 |
| PT1b | ERA5 and CAMS, lags live, surprise masked (zero) | lr 2e-4 | 12 + 12 |
| PT-pool5 | five seeds of the best of PT1, PT1-full, and PT2 (fixed before reading C-pool) | as the arm | 5 networks per refit |
| PT2 | ERA5 and CAMS, each lag masked with probability 0.3 | lr 2e-4 | 12 + 12 |

**The best PT arm** is the one among PT1, PT1-full, and PT2 with the lowest mean of its point
differences against S1 at the signed-MW six and the MVA-only eight over the four screen folds, fixed
before any pool contrast is read. Its seeds 3 and 4 are fitted in a second stage (with the lagless
tables for those seeds) and pooled with seeds 0 to 2 by experiment 448's equal-weight mixture pool.
C-pool is the PT-pool5 minus S1-pool5 (experiment 448's five-seed pool of S1) and PT-pool5 minus
PT0-pool5 (experiment 448's S1PT-pool5), at both groups. PT1b is a diagnostic that needs no lagless
table: it separates the effect of the lags from the
effect of the surprise. Each lag token (the lag value, its missing flag, and its surprise; the age
column stays)
is masked as a unit in PT2,
so the network keeps a pathway that works without the lags. Fine-tuning shows all lags. PT1 and
PT1-full share one pretrained network per fold and seed.

## How the surprise is built for the pretraining years

**The surprise at a lag is the lag minus a lagless forecast of the same half-hour, divided by the
lagless forecast's half band width.** The lagless forecast must not have seen the half-hour it
forecasts, or the surprise shrinks towards zero in pretraining and the network learns a lag pathway
that fails on ensemble rows. The plan therefore builds one cross-fitted lagless table per fold and
seed
on ERA5 and CAMS weather, by the method of experiment 404's table:

- **Span and blocks.** 2019-09-01 to the fold's refit, split into 6 contiguous blocks. A lagless
  module fitted on 5 blocks forecasts the sixth. Nothing at or after the refit enters.
- **Module and inputs.** Experiment 404's lagless module (a pooled mixture density network with a
  Primary embedding, 64 hidden units, 6 epochs, learning rate 2e-3). The inputs are the 7 ERA5 and
  CAMS
  weather means, the solar-position, calendar, and holiday columns of experiment 397, and no lag and
  no lead. Experiment 404's ensemble spread columns are dropped because ERA5 has no spread. Rows
  have
  equal weights.
- **Early cut.** Targets before experiment 418's first-usable date of D11, D10, D5, D14, and D16 are
  left out of the fits, as in the pretraining targets.
- **Target.** Cleaned power over capacity (world 0), with flagged half-hours and the D15 and D18
  start
  cut as missing and left out of the fits.
- **Table.** Per `(primary, grid)`: the q50 and half the q05-to-q95 width, in the layout the S1 code
  already reads.
- **Rescale.** The smoke check on 2026-01 seed 0 found the ERA5 table's surprise standard deviation
  (1.245) well above the ensemble table's (0.850), over 2025-02 to 2026-01. The ERA5 half band width
  is therefore multiplied, per Primary, by the ratio of the two surprise standard deviations over
  the half-hours before the refit where both tables exist (the ensemble years), so the pretraining
  surprise has the scale the network sees in fine-tuning. The ratio uses no data at or after the
  refit. The ratios are printed per fold and Primary; a ratio outside 0.25 to 4 stops the run
  (widened from 0.3 to 3 after the first 2025-12 build gave 0.295 at one Primary; the other builds
  gave 0.41 to 2.31).
- **Overlap years.** Pretraining rows after 2024-04 use the ERA5 table in pretraining and the
  ensemble table in fine-tuning, for the same half-hours.
- **Non-causal and unpurged.** A module forecasting a block is trained on later blocks too, as in
  experiment 404, and the held-out and purged weeks of pretraining are not excluded from the other
  blocks' fits. Neither uses anything at or after the refit.
- **What differs from experiment 404's recipe.** The span is 2019-09 to the refit rather than 24
  months, rows have equal weights, and the inputs are ERA5 and CAMS, so the ERA5 modules are not
  identical in quality to 404's. The further module of experiment 404 (scoring months) is not
  needed.
- **What differs from use.** In fine-tuning, and in use, the surprise comes from experiment 404's
  table, built from ensemble weather. The pretraining surprise comes from ERA5 and CAMS weather,
  which
  is more accurate, so its half-hours are less surprising on average. A smoke check prints the
  surprise standard deviation by source and asserts that the pretraining table differs from the
  fine-tuning table. ERA5 arrives about 5 days late and is not available at
  forecast time; the live service uses the ensemble-based table, as the fine-tuned networks do.

## Compute

131 networks were fitted (lagless tables, pretraining, and fine-tuning over five folds and up
to five seeds), on up to three GPU processes under 8 GB each.
