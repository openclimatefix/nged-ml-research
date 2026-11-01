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
status: finished
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
  effect (0.064 points at two seeds, from S1's six stored seeds). P2. A wider neural network (128
  or 256) and a Primary embedding of 16 are the likeliest to help, each by 0.03 to 0.10 points; the
  levels are 2, 3, and 5 components, and 2 components lose 0.02 to 0.10 points. P3. SGD and a
  hidden width of 32 are the likeliest to hurt. P4. The cosine and one-cycle schedules change the score by under 0.05 points because early stopping already ends training.
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
verdict: not supported
best_arm: >-
  The unchanged S1 (pooled-14 clean CRPS 3.011 points of capacity on the three screen folds). The
  best single change, hidden width 256, gains 0.042 points there (95% interval 0.029 to 0.056),
  below the 0.064 margin.
headline: >-
  None of the 9 pre-registered one-factor changes (width, depth, Primary embedding, mixture
  components) moves S1 by the 0.064-point margin, so the 13 other arms and the random search did not
  run in the pre-registered plan. Run afterwards as a post hoc extra, none of those 13 moves S1
  either. Post hoc, a hidden width of 256 gains 0.087 points on 5 folds by 3 seeds, but one new fold
  carries it and real events get worse.
lessons: >-
  S1 is insensitive to its settings at the resolution of two seeds. Weight decay 1e-4 and Adam
  without decay are identical to S1 to four decimals because the learning rate is small. A wider
  hidden layer is the only direction with a hint of gain, and it trades clean-row CRPS against
  real-event CRPS and tail coverage.
take_home_message: >-
  Changing one setting of the S1 demand neural network at a time does not improve it by the
  margin we required (0.064 CRPS points): the network is insensitive to its hidden width,
  depth, Primary embedding, mixture components, dropout, weight decay, learning rate and schedule,
  batch size, and optimiser at this resolution. Only a hidden width of 256 shows a gain (0.047 on
  the four folds with stored S1 forecasts, 0.250 on one new fold), and it makes real load-shift
  events and the upper tail worse. Keep S1 as it is; combinations of settings were not tested.
advice:
- >-
  Keep S1 at hidden width 64, 2 hidden layers, Primary embedding 8, 3 mixture components, dropout
  0.1, AdamW, and learning rate 5e-4: no single change beat it by the margin (Results).
- >-
  Do not run a random search over these settings: the stop rule fired, and no arm moved (Results).
- >-
  Do not put hidden width 256 into the assembled network on this evidence: it gains on clean rows
  and loses on real events and q99 coverage (Results, post hoc).
- >-
  Look for demand gains in inputs and in the lag-trust design before further tuning (Results).
production_status: not-for-production
adopted_in: none
code: ml-experiments/00447-demand-network-hyperparameter-sweep-on-the-s1-skeleton/
---

**Changing one setting of the S1 demand neural network at a time does not improve it by the margin
we required, so S1 stays as it is.** S1 is the pooled neural network of experiment 428 that reads
the three recent power readings and the surprise at each. We changed one setting at a time on 3
test months by 2 seeds and compared each change with S1 on clean rows, pooled over the 14
Primaries. The margin was 0.064 CRPS points (the seed minimum detectable effect; the agreed 0.03
never binds). No pre-registered arm reached it, so the stop rule ended the pre-registered
experiment: no random search, and no further arms. The verdict is **not supported**.

**Post hoc, the other 13 arms and a width-256 confirmation were run on the manager's request, and
the pre-registered verdict is unchanged.** None of the 13 arms moves S1. Hidden width 256 passes
the confirmation rule on 5 folds by 3 seeds (gain 0.087 points, interval 0.067 to 0.107, better on
5 of 5 folds). That gain comes mostly from the new fold 2025-12 (+0.250), the four folds with stored
S1 forecasts give +0.047 (below the margin), and real events get worse, so we do not recommend
width 256.

![Each arm minus S1 on clean rows with 95% intervals](figure_1_clean.svg)

*Figure 1. Gain over S1 in CRPS points on clean rows (positive is better), 3 folds by 2 seeds, all 22
arms (13 of them post hoc). Dashed lines mark 0 and the 0.064 margin.*

![Gain on injected harm and real harm rows](figure_2_harm.svg)

*Figure 2. The same on injected harm and real harm rows.*

![Gain in each fold](figure_3_folds.svg)

*Figure 3. Each arm's gain in each screen fold, clean pooled rows.*

![Width 256 on 5 folds](figure_6_confirmation_folds.svg)

*Figure 6. Width 256's gain in each of the 5 confirmation folds (post hoc), 3 seeds.*

![Day-ahead example](figure_4_example_day_ahead.svg)

*Figure 4. A day-ahead forecast of one signed-MW Primary, S1 and width 256 (seeds 0 and 1).*

![10-day example](figure_5_example_10_day.svg)

*Figure 5. The full 10-day horizon of one forecast of the same Primary.*

## Results

- **Pre-registered stage (width, depth, embedding, components; 54 fits).** Gains over S1 in CRPS
  points, pooled clean: width 256 +0.042 (0.029 to 0.056, 3 of 3 folds), width 128 +0.033, depth 4
  +0.016, 5 components +0.007, 2 components -0.002, depth 3 -0.006, embedding 16 -0.013, embedding
  4 -0.024, width 32 -0.070. S1's absolute score is 3.011 (screen folds).
- **Post hoc, the other 13 arms (78 fits).** SGD +0.012, learning rate 1e-3 +0.011, one-cycle
  +0.009, batch 4,096 +0.009, dropout 0 +0.001, weight decay 1e-4 -0.0001, Adam -0.0001, weight
  decay 1e-2 -0.004, batch 16,384 -0.010, cosine -0.010, Lion -0.012, dropout 0.2 -0.022, learning
  rate 2.5e-4 -0.029. All sit inside the 0.064 minimum detectable effect except the clearly worse.
- **Post hoc, width 256 on 5 folds by 3 seeds (33 fits).** Pooled clean CRPS 3.300 against S1's
  3.387 (+0.087, 0.067 to 0.107). Per fold: 2026-06 +0.068, 2026-01 +0.027, 2025-10 +0.047, 2026-03
  +0.046, 2025-12 +0.250. The four folds with stored S1 give +0.047; the screen folds give +0.047.
- **Where width 256 loses (confirmation).** Real-event CRPS 6.956 against 6.673 (-0.283; MVA-only
  eight -0.519); real-harm MVA-only eight -0.087. Clean q95 coverage 0.918 against 0.933 and q99
  0.979 against 0.986; real-event q99 coverage 0.857 against 0.891 (signed-MW six 0.774 against
  0.830). Tail pinball loss on clean, real-harm, and injected-harm rows stays within 0.02 points.
- **Checks.** Error of the median forecast, clean rows (screen folds): signed-MW six 4.700 (S1)
  against 4.668 (width 256) and MVA-only eight 5.685 against 5.671; the median minus mean forecast
  is -0.33 and -0.29 at the six, so the CRPS gain is not a median-versus-mean artefact. Audit
  triggers fired: one fold carries the gain (2025-12), and the gain is on clean rows while real
  events lose. Seeds 0 to 5 of S1 each use their own cross-fitted lagless table, so the stored
  seeds are independent draws.
- **Reference.** S1 is experiment 428's and 439's stored forecasts; the refit under this
  experiment's loop reproduced the stored validation loss exactly (0.4336399, 60 epochs). For
  2025-12, which has no stored S1, the S1 refit of this experiment is the reference.

## Caveats

- **Resolution.** At two seeds the screen detects 0.064 points about half the time and about 0.09
  points 80% of the time, so a gain between 0.03 and 0.064 is undetectable here and is no evidence of
  a flat optimum.
- **Winner's curse.** Width 256 was picked as the best of 22 arms and confirmed on rows and seeds that
  overlap the screen (3 folds, seeds 0 and 1); the new folds carry the gain.
- **Optimiser arms** change the learning rate and weight decay together with the update rule, so
  they test a recipe, not the optimiser alone. Schedules ran all 60 epochs.
- **Schedule.** Monthly refits (the research default); a weekly-refit re-check was not run.
- **What it rules out.** Tuning these settings one at a time will not give S1 a demand gain of the
  size already agreed. Combinations were not tested because no single arm moved.

## Next steps

- Look for demand gains in the inputs and the lag-trust design, not in these settings.
- If width 256 is of interest, test it inside the assembled network and report real-event CRPS and
  q99 coverage beside the clean-row gain.

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
