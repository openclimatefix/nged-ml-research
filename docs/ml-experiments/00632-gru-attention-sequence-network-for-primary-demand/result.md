---
id: "00632"
slug: gru-attention-sequence-network-for-primary-demand
title: >-
  Does a network with two gated recurrent units and three self-attention layers, pre-trained on
  ERA5 and CAMS weather, forecast Primary demand better than the existing pre-trained networks?
theme: lagged-power-and-error-correction
also_relevant_to:
  - demand-inputs
status: planned
kind: other
size: >-
  Complex. Answers to the five triggers: (1) it stores fitted network weights and forecasts in the
  private data folder only; (2) more than one design could test the question (joint attention
  against cross-attention, token length, forecast length), so this plan fixes them in advance; (3)
  it reuses code from experiments 397, 418, 560, 565, 574, 610, and 614, whose callers cannot be
  named without searching; (4) it touches no serving path and no degradation rule; (5) it changes
  no contract. Ceiling 6 GPU-hours (process hours); ask before crossing.
hypothesis: >-
    A sequence neural network (Seq: an encoder gated recurrent unit over the last 7 days, a
  decoder gated recurrent unit over the next 7 days that starts from the encoder's state, and
  three self-attention layers over all tokens), pre-trained on ERA5 and CAMS weather with one
  Primary embedding for all 16 Primaries, forecasts the next 7 days of demand better than the
  existing pre-trained networks, because it reads the last 6 days of power and not only a day lag
  and a week lag.
prediction: >-
    Written before any fit. Seq lowers CRPS at 24 to 168 h against P2-pre by at least 0.06 points
  at the eight MVA Primaries, and is not worse by 0.06 or more at the six MW Primaries. The
  evidence from experiments 585, 560, and 558 predicts a tie, so the prediction can fail.
decision_rule: >-
    (1) Four counted contrasts (Bonferroni count 4): Seq minus P2-pre and Seq minus P2-hist6, CRPS
  at 24 to 168 h, at the eight MVA Primaries and at the six MW Primaries. Seq minus K3-pre is
  reported without a count. (2) The prediction holds if, at the MVA Primaries, the whole interval
  of Seq minus P2-pre lies below zero and its point estimate is at most -0.06, and, at the MW
  Primaries, the upper end of the interval is below +0.06. (3) The margin 0.06 is the interval
  half-width experiment 591 measured for network contrasts at 24 to 120 h (0.058 and 0.059); it is
  also the smallest gain worth porting two GRUs and an attention stack. (4) Stage 3 (fine-tuning
  on ensemble weather) is a separate later experiment and runs only if the prediction holds and
  q05-q95 coverage lies in 0.85 to 0.95 in both groups. (5) Stop rule: after seed 0 on the folds
  2025-12 and 2026-06 (a point estimate over two month blocks), stop if Seq is worse than K3-pre
  by more than 5% CRPS at days 1 to 2 in either group (the maintainer's rule), or worse than
  P2-pre by more than 0.10 points in both groups (larger than the 0.06 margin, so not noise).
benchmark: >-
    P2-pre (the lag neural network's pre-training, shared by K1, K2, and the Candidate), P2-hist6
  (a per-row control that reads 6 daily lags), and K3-pre (the lag-free pre-training of experiment
  565), all scored on ERA5 and CAMS inputs.
row_set_id: roster_export_days_1_to_7_v1
metric: >-
  CRPS (twice the mean pinball loss over 11 levels) by forecast day and pooled at 24 to 168 h; mean
  absolute error of the median; q05-q95 coverage; share of rows above q90; pinball loss and coverage
  at q01, q05, q95, and q99.
---
---

**Draft plan, written before any fit. The Sonnet plan review ran and its findings are triaged
below.** Nothing here is adopted for production. Every stage 2 number is a ceiling for a live
forecast, because ERA5 arrives about 5 days late.

**Terms.** Seq is the sequence neural network of this experiment. ERA5 is ECMWF's reanalysis of past
weather, and CAMS is the Copernicus Atmosphere Monitoring Service irradiance product; both are
estimates of past weather. P2-pre is the pre-training of the lag neural network of [experiment
574](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/result/)'s
arm P2, from which K1, K2, and the Candidate start. K3-pre is the lag-free pre-training of
[experiment
565](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00565-drift-embedding-variants-six-designs/result/).
D5 is the slow monthly output offset, and B1 is the set of 14 trailing-temperature and solar-gain
columns of experiment 569 that the weather encoder of P2 reads. A sequence is one Primary and one
forecast origin (09:00 UTC); a pass is one sweep over a random 20% or all of the training sequences.
The MVA and MW groups are the eight MVA Primaries and the six MW Primaries; D15 and D18 are
report-only.

## Question and stages

**The question is whether a neural network that reads the last 6 days of power uses that history
better than the two power lags the existing neural networks read.** Stage 1 pre-trains Seq for all
16 Primaries on ERA5 and CAMS weather with an 8-number Primary embedding. Stage 2 scores Seq beside
P2-pre, P2-hist6, and K3-pre on ERA5 and CAMS weather only, and is this experiment's headline. Stage
3 (fine-tuning on ensemble weather, comparison with K1, the Candidate, and K3) is a separate later
experiment.

## Architecture

- **Tokens.** 336 history tokens (7 days to the origin) and 336 future tokens (leads 0.5 to 168 h),
  half-hourly, width 64. Memory is not a constraint (`e00632_budget.py`: about 1 GB at batch 32 with
  plain attention). Time is the risk.
- **Inputs per token.** The weather encoder reads the 7 weather columns and the 14 B1 columns at the
  token's own time (the 2 CAMS previous-day columns take the origin's value). The datetime encoder
  reads the 21 datetime columns. Every token adds a position embedding (steps from the origin) and a
  segment embedding (history or future). A history token also has a lag encoder reading power as a
  share of capacity, a missing flag, the age in days, the token's weather and datetime embeddings,
  and the Primary embedding. Future tokens take a learnt "no lag" vector in that slot.
- **No lookahead.** A token carries power only if its half-hour is at least 24 h before the origin,
  the same cap the lag neural networks use (lags observed at least 24 h before the origin). The 48
  newest history tokens carry power 0 and missing flag 1, as do flagged tokens. The loss sits on
  future tokens only.
- **Recurrence and attention.** The encoder GRU reads the history embeddings. The decoder GRU reads
  the future embeddings and starts from a linear map (tanh) of two encoder states at fixed
  positions: the last one and the one at origin minus 24 h. Three joint self-attention layers (4
  heads, feed-forward width 256, no causal mask, because every input is known at the origin) follow.
  Seq has about 250,000 parameters against about 30,000 for P2.
- **Output.** A three-component Gaussian mixture per future token, trained with the negative
  log-likelihood of experiment 397 and read out as 11 quantiles. The D5 offset is added to the
  component means with the module of P2 (monthly knots from 2019-09, a penalty on first differences,
  flat extrapolation after the newest knot). Knots start at zero and are fitted on training targets
  only; a test-month origin reads the newest knot.

## Training

- **Sequences and cut-offs.** Daily 09:00 UTC origins from ERA5 and CAMS pseudo-weather, within the
  24 months before each fold's refit, cleaned power only, D15 and D18 cut at 2025-03-01. Every
  target at or after the refit minus 3.5 h is set to missing in training and in validation, so a
  late origin contributes only its early targets. Validation is the last 30 origin days before the
  refit, and training origins end 7 days before the first validation origin.
- **Standardisation.** Seq uses the fold's stored statistics that P2-pre used (the standardiser and
  target mean and standard deviation of `e00428_data.build_fold`, and B1's `stats/B1/<month>.npz`),
  asserted equal to a recomputation within 1e-6.
- **Fit.** AdamW, learning rate 5e-4, 300 warm-up steps, cosine decay, weight decay 1e-2, dropout
  0.1, gradient clip 1.0, batch 256, patience 8 passes, at most 60 passes; block dropout as in the
  lag neural networks (all power masked on about 25% of sequences, the newest 24 h of power on about
  5%). Seq starts fresh, as P2-pre and P2-hist6 did, so no arm has extra epochs.
- **Seeds and folds.** Seeds 0, 1, and 2 on 2025-10, 2025-12, 2026-01, 2026-03, and 2026-06, monthly
  refits. Comparators are scored at the same seeds; the plan lists K3-pre's seeds before scoring.
- **July 2026.** The dense grid ends at 2026-07-01. Origins in 2026-06 are kept only if their 168 h
  window ends before 2026-07-01 (origins through 23 June), for every arm, so all arms are scored on
  identical keys.
- **Saved models.** Each fitted state dict goes to `data/ml-experiments/00632-*/models/`, outside
  git.

## Comparison (stage 2)

- **Arms.** Seq; P2-pre and K3-pre from stored weights through the pre-training head (inference
  only), on ERA5 test rows built like experiment 574's pre-training rows at the test-month origins;
  P2-hist6, a per-row control trained here with the P2 recipe whose lag encoder reads up to 6 daily
  lags 24 h or older, mean-pooled; and a month-by-half-hour climatology row
  (`ClimatologyForecaster`).
- **Row set.** 09:00 UTC origins, leads 0.5 to 168 h, valid times before 2026-07-01, event-free rows
  of the roster export, cleaned observed power as the target; a join assertion checks that every
  arm's (Primary, origin, valid time) keys are identical before any score is read.
- **Interval.** Month-block bootstrap through `bootstrap_difference_at_level` at the Bonferroni
  level. With five month blocks the 98.75% interval is close to the range of the block means, so
  "excludes zero" here means nearly every block agrees in sign.
- **Scores.** CRPS by forecast day and pooled; MAE of the median; q05-q95 coverage; share above q90;
  q01, q05, q95, and q99 pinball loss and coverage, per group and per Primary; cold winter nights.
  Avoidable-curtailment cost does not apply (no load-shift analysis in stage 2). Half-hours dropped
  by cleaning are reported per fold.
- **Checks.** Median minus mean forecast per Primary; the MAE of the median beside each CRPS
  contrast; which audit triggers fired.
- **Figures.** Headline CRPS by forecast day with intervals (all arms, climatology first-class);
  tail calibration; a day-ahead example and the full 7-day horizon of one forecast, in winter and
  summer; each checked by a fresh Sonnet reviewer against the figure checklist.
- **Anonymisation.** Demand Primaries only, so not applicable. No leaderboard post.

## Run order and compute

1. Score P2-pre and K3-pre (no fits) and print the MDE on this row set.
2. Run P2-hist6 (about 80 s per fit), then time one Seq fit (fold 2026-06, seed 0).
3. Run seed 0 on 2025-12 and 2026-06 and apply the stop rule; then all folds and seeds.
4. Ablations, one seed on all five folds, descriptive: A-noAttn (GRUs only), A-noGRU (attention
   only), and A-NB (neighbour summaries added: mean power of the electrical neighbours and the share
   of neighbours with a reading, same 24 h cap, 50% signal dropout).

- **Ceiling.** 6 GPU-hours; ask before crossing. If a Seq fit exceeds 20 min: 20% of sequences per
  pass, then hourly tokens, then a 2-day head. If the ceiling still binds: drop ablations first,
  then seeds (3 to 1), then folds. One GPU process, `MemoryMax=8G`, `OMP_NUM_THREADS=1`, memory
  fraction 8/48, bf16.

## Triage of the plan review

- **Accepted:** target cut-off and validation (F1), July weather (F2), poisoned-copy check per
  sample with a negative control (F3), D5 offset (F4), stored standardisation and B1 (F7), key join
  assertion (F8), neighbours moved to an ablation (F9, F10), fresh start instead of initialising
  from P2-pre (F11), seeds (F12), fixed margin 0.06 and an MW rule (F15 to F17), interval caveat
  (F18), K3-pre uncounted (F19), reasons for thresholds (F20), ablation drop order (F21), figures,
  tails, checks, climatology, counts, saved models (F22 to F25), prose findings.
- **Rejected:** dropping the two-state decoder link (F6 asks for fixed positions, which are adopted;
  the second state stays because the A-noAttn and A-noGRU arms leave it untested only as a pair).
- **Cut:** the A-shared and A-2d ablations, the lead columns, and the stage 3 detail (stage 3 is a
  later experiment).
