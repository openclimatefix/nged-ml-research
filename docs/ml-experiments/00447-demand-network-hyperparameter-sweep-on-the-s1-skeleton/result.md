---
id: "00447"
slug: demand-network-hyperparameter-sweep-on-the-s1-skeleton
title: >-
  Does changing any one setting of the S1 demand neural network, or a random search over the three most
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
  and a Primary embedding of 16 are the likeliest to help, each by 0.03 to 0.10 points; the levels are 2, 3, and 5
  components, and 2 components lose 0.02 to 0.10 points. P3. SGD and a hidden width of 32
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
  block bootstrap 95% interval. Minimum detectable effect = 1.96 x sqrt(2) x sd / sqrt(2 seeds),
  where sd is the residual standard deviation of S1's six stored seeds (0 to 5) about each fold's
  mean, on the three screen folds, computed on pooled 14 clean rows (0.064 points); it covers seed
  noise only, so it is a lower bound. Margin = the larger of 0.03 points (the agreed demand margin)
  and that minimum detectable effect, so the screen margin is the minimum detectable effect. A
  screen at two seeds detects a gain of 0.064 points about half the time and a gain of about 0.09
  points 80% of the time, so a gain between 0.03 and 0.064 is reported as undetectable and is not
  evidence of a flat optimum. A one-factor arm moves when its pooled clean gain is at least the
  margin, it improves at least 2 of the 3 folds, and no pooled q95 or q99 pinball loss (mean over
  the three folds, points of capacity, seed-averaged quantiles) rises by more than 0.02 points on
  clean, real harm, or injected harm rows (six cells; a real harm cell fails only if its rise also
  exceeds that cell's seed spread). If no arm moves, the experiment stops and says so. Otherwise the
  search factors are the three factors, among those whose best level has a positive pooled clean
  gain, with the largest best-level gain (ties broken by the larger mean fold gain, then the lower
  absolute CRPS); if fewer than 3 factors have a positive gain, only those are searched. The optimiser
  levels are bundles that fix their own learning rate and weight decay, so a draw that uses a
  non-AdamW optimiser keeps S1's schedule, and the learning rate, weight decay, and schedule
  factors cannot be searched in the same draw. The search samples up to 20 distinct configurations
  uniformly from the product of the searched factors' levels and S1's level, excluding S1 itself
  and the single-factor arms, with seed 447; each is screened on the same folds and seeds. The top
  2 by pooled clean CRPS that also meet the screen margin and tail rule are confirmed with S1 on 5
  folds (2026-06, 2026-01, 2025-10, 2026-03, 2025-12) by seeds 0 to 2. A configuration passes the
  confirmation if its pooled clean gain is at least the screen margin, its 95% block-bootstrap
  interval excludes zero, it improves at least 3 of the 5 folds, and the tail rule holds. The
  confirmation shares 3 folds and 2 seeds with the screen, so the gain on the two new folds
  (2026-03, 2025-12) is reported separately beside it. A configuration is reported as beating S1
  only if it passes the confirmation. No arm is adopted by this experiment; the winner's
  configuration is handed to the assembly worker. The smallest gain worth porting is the margin,
  because every setting is a constant in the training script and costs nothing in production.
  Interim; no 99% claim.
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

Does any single setting of the S1 demand neural network matter more than the seed noise, and if one does,
does a combination of the influential settings help further?

## Plan

**Every change is made to S1 one at a time, and the order of the stages is set by the stop rule.**

- **Base.** S1 of experiment 428: hidden width 64, 2 hidden layers, Primary embedding 8, 3 mixture
  components, dropout 0.1, AdamW with weight decay 1e-3 and learning rate 5e-4, constant schedule,
  batch 8,192, gradient clip 1.0, training on clean and injected worlds, up to 60 epochs with early
  stopping (patience 8). All of these except the setting under test carry over to every arm.
- **Levels (22 one-factor arms).** Hidden width 32, 128, 256. Depth 3, 4. Embedding 4, 16.
  Components 2, 5. Dropout 0, 0.2. Weight decay 1e-4, 1e-2. Learning rate 2.5e-4, 1e-3. Schedule
  cosine over the 60-epoch cap, one-cycle (peak 2 x 5e-4, 60 epochs of steps; S1 runs all 60 epochs,
  so the schedules complete). Batch 4,096, 16,384. Optimiser Adam (no decay), Lion (learning rate
  1.5e-4, weight decay 1e-2), SGD with Nesterov momentum 0.9 (learning rate 5e-2, weight decay
  1e-3). The optimiser arms use each optimiser's customary learning-rate scale, so they test a
  recipe, not the optimiser alone.
- **Stages.** Stage 1a runs hidden width, depth, embedding width, and mixture components (9 arms).
  The other factors (dropout, weight decay, learning rate, schedule, batch size, optimiser; 13 arms)
  run only if a stage 1a arm moves S1; if none does, the page says the neural network is
  insensitive to these settings and stops.
- **Fits.** 6 fits per arm (3 folds by 2 seeds): 54 for stage 1a, 78 for the rest, up to 120 for
  the random search, and 33 for the confirmation (2 configurations by 5 folds by 3 seeds, plus 3 S1
  refits for 2025-12, which has no stored S1 forecast; the refits stand in as S1 for that fold).
- **Compute.** One fit of S1 took 38 seconds on the GPU (2026-11-01), so the full programme of up to
  285 fits is about 3 GPU-hours plus scoring; width 256, depth 4, and small batches are slower.
- **Gate.** The refit of S1 (2026-01, seed 0) under this experiment's training loop reproduced
  experiment 428's stored validation loss exactly (0.43363991, 60 epochs). Any later disagreement
  beyond 1e-3 in validation loss stops the sweep.
- **Seeds.** Seeds 0 to 5 of S1 each have their own cross-fitted lagless table (experiments 404 and
  439), so the six stored seeds are independent draws.
- **Reports.** Absolute CRPS beside every contrast, the signed-MW six and the MVA-only eight
  separately, clean, injected harm, real harm, and real event rows as in experiment 439, tail
  pinball and coverage, and the seed spread.
