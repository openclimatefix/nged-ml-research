---
id: "00447"
slug: demand-network-hyperparameter-sweep-on-the-s1-skeleton
title: >-
  Does changing any one setting of the S1 demand network, or a random search over the three most
  influential settings, lower its quantile score by the agreed margin?
theme: lagged-power-and-error-correction
also_relevant_to:
  - calibration-and-uncertainty
  - production-reference-and-best-guess
status: planned
kind: change
protocol: demand-v2
size: >-
  medium. Stored: fitted networks and forecasts in this experiment's private folder only;
  experiments 404, 428, and 439 are read and never written. One design would test the hypothesis
  (one factor at a time on a fixed skeleton), so only the factor levels are open, and this plan
  fixes them. Production serving path: not touched (this repository holds none). Degradation rule:
  not touched. Callers: the scripts import experiment 428's modules, experiment 397's network and
  loss, and read experiment 428's and 439's stored S1 forecasts. Sonnet writes and reviews; no Opus.
hypothesis: >-
  S1 (the pooled plain mixture-density network of experiment 397's setting g09, given the three
  origin lags and the surprise at each lag) sits near a flat optimum, so changing one of its hidden
  width, depth, Primary-embedding width, number of mixture components, dropout, weight decay,
  learning rate, learning-rate schedule, batch size, or optimiser lowers days 1 to 10 CRPS by less
  than the margin.
prediction: >-
  Written before any fit and before any score is read. CRPS in points of effective capacity, days 1
  to 10, 1.15 widening, pooled over the signed-MW six and the MVA-only eight, clean rows. P1. Most
  of the 22 one-factor arms change the pooled clean CRPS by less than the minimum detectable
  effect (0.064 points at two seeds, from S1's six stored seeds). P2. A wider network (128 or 256)
  and a Primary embedding of 16 are the likeliest to help, each by 0.03 to 0.10 points; one
  mixture component is not tested, and 2 components lose 0.02 to 0.10 points. P3. SGD and a hidden width of 32
  are the likeliest to hurt. P4. The cosine and one-cycle
  schedules change the score by under 0.05 points because early stopping already ends training.
  P5. No change lowers the q95 or q99 pinball loss on real harm rows by more than the seed
  spread. The prediction that could fail: at least one arm gains by the margin on three folds by
  two seeds and again on five folds by three seeds. If none does, the experiment stops after the
  one-factor stage and runs no random search.
decision_rule: >-
  Screen folds: test months 2026-06, 2026-01, 2025-10 (summer, winter, autumn); seeds 0 and 1;
  monthly refits on the 24 months before each test month (experiment 428's recipe, unchanged except
  for the arm's setting). The reference is S1 itself (experiment 428's stored forecasts, same folds
  and seeds). Contrast = arm minus S1 in CRPS points on the pooled 14 Primaries (signed-MW six and
  MVA-only eight added block by block, reported separately too), clean rows, with a month-and-day
  block bootstrap 95% interval. Margin = the larger of 0.03 points (the agreed demand margin) and
  the seed minimum detectable effect (1.96 x sqrt(2) x seed sd / sqrt(seeds), the sd from S1's six
  stored seeds on the same folds). A one-factor arm moves when its gain is at least the margin on
  clean pooled rows, it improves at least 2 of the 3 folds, and no q95 or q99 pinball loss rises by
  more than 0.02 points on clean, real harm, or injected harm rows. If no arm moves, the experiment
  stops and says so. Otherwise the three factors with the largest best-level gain are searched:
  20 configurations drawn at random (seed 447) from their levels and S1's level, screened on the
  same folds and seeds, and the top 2 by pooled clean CRPS plus S1 are confirmed on five folds
  (the four above plus 2025-12) by seeds 0 to 2, with the same margin and tail rule applied to the
  confirmation alone. A configuration is reported as beating S1 only if it passes on the
  confirmation. No arm is adopted by this experiment; the winner's configuration is handed to the
  assembly worker. The smallest gain worth porting is the margin, because every setting is a
  constant in the training script and costs nothing in production. Interim; no 99% claim.
benchmark: >-
  S1 of experiment 428 (seeds 0 to 2) and experiment 439 (seeds 3 to 5), stored forecasts, as the
  reference of every contrast.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, pinball loss and coverage at q01, q05,
  q95, and q99, median minus mean forecast, seed spread, at the 1.15 widening; clean, injected harm,
  real harm, and real event rows.
verdict: not-yet-run
best_arm: none yet
production_status: not-for-production
adopted_in: none
code: ml-experiments/00447-demand-network-hyperparameter-sweep-on-the-s1-skeleton/
---

**Take-home message: not yet run.**

## Question

Does any single setting of the S1 demand network matter more than the seed noise, and if one does,
does a combination of the influential settings help further?

## Plan

- **Base.** S1 of experiment 428: hidden width 64, 2 hidden layers, Primary embedding 8, 3 mixture
  components, dropout 0.1, AdamW with weight decay 1e-3 and learning rate 5e-4, constant schedule,
  batch 8,192, up to 60 epochs with early stopping (patience 8).
- **Levels (22 one-factor arms).** Hidden width 32, 128, 256. Depth 3, 4. Embedding 4, 16.
  Components 2, 5. Dropout 0, 0.2. Weight decay 1e-4, 1e-2. Learning rate 2.5e-4, 1e-3. Schedule
  cosine, one-cycle (peak 2 x 5e-4). Batch 4,096, 16,384. Optimiser Adam (no decay), Lion (learning
  rate 1.5e-4, weight decay 1e-2), SGD with Nesterov momentum 0.9 (learning rate 5e-2, weight decay
  1e-3). The optimiser arms use each optimiser's customary learning-rate scale, so they test a
  recipe, not the optimiser alone.
- **Screen.** 3 folds by 2 seeds, 6 fits per arm, 132 fits. S1's forecasts are stored, so the
  reference costs no fits.
- **Gate.** One refit of S1 (2026-01, seed 0) under this experiment's training loop must match
  experiment 428's stored validation loss and forecasts, before any arm is read.
- **Random search.** Up to 20 configurations, 6 fits each (120 fits).
- **Confirmation.** The top 2 configurations and S1 on 5 folds by 3 seeds: 30 fits plus 3 S1 refits
  (the fifth fold, 2025-12, has no stored S1 forecast).
- **Compute.** About 0.25 GPU-hours per fit (experiment 439: 24 fits in about 6 GPU-hours), so the
  screen is about 33 GPU-hours, the random search about 30, and the confirmation about 8 (33 fits).
- **Cleaned data and the short-history cut.** Experiment 428's rows are used unchanged.
- **Reports.** Absolute CRPS beside every contrast, the signed-MW six and the MVA-only eight
  separately, clean, injected harm, real harm, and real event rows as in experiment 439, tail
  pinball and coverage, and the seed spread.
