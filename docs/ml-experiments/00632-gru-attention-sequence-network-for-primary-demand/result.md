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
  A sequence network pre-trained on ERA5 and CAMS weather, with one Primary embedding for all 16
  Primaries, forecasts the next 7 days of demand better than the existing pre-trained networks,
  because it reads the last 6 days of power and not only a day lag and a week lag. The network is an
  encoder gated recurrent unit (GRU) over the last 7 days, a decoder GRU over the next 7 days that
  starts from the encoder's hidden state, and three self-attention layers over all tokens.
prediction: >-
  Written before any fit. Seq's CRPS at 24 to 168 h is lower than P2-pre's by at least the minimum
  detectable effect (MDE) at the eight MVA Primaries, and no worse than P2-pre beyond the MDE at the
  six MW Primaries. The evidence from experiments 585, 560, and 558 predicts a tie, so the
  prediction can fail.
decision_rule: >-
  Four counted contrasts (Bonferroni count 4): Seq minus P2-pre and Seq minus K3-pre, CRPS at 24 to
  168 h, at the eight MVA Primaries and the six MW Primaries, with month-block bootstrap intervals
  at the Bonferroni level. The prediction fails if the interval of Seq minus P2-pre includes zero at
  the eight MVA Primaries. Stage 3 (fine-tuning on ensemble weather) is a separate later experiment
  and runs only if the prediction holds and q05-q95 coverage lies between 0.85 and 0.95 in both
  groups. Stop rule: after seed 0 on the folds 2025-12 and 2026-06, stop if Seq is worse than K3-pre
  by more than 5% CRPS at days 1 to 2 in either group, or worse than P2-pre by more than 0.10
  points at 24 to 168 h in both groups. The smallest gain worth porting is the MDE, because the
  network adds two GRUs and an attention stack to the pipeline.
benchmark: >-
  P2-pre (the lag network's pre-training, shared by K1, K2, and the Candidate) and K3-pre (the
  lag-free pre-training of experiment 565), scored afresh on ERA5 and CAMS inputs from stored weights.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS (twice the mean pinball loss over 11 levels) by forecast day and pooled at 24 to 168 h; mean
  absolute error of the median; q05-q95 coverage; share of rows above q90; pinball loss and coverage
  at q01, q05, q95, and q99.
---

**Draft plan, written before any fit; the Sonnet plan review is pending.** The plan folds in the
Opus plan of 2026-11-07. Nothing here is adopted for production, and every stage 2 number is a
ceiling for a live forecast, because ERA5 arrives about 5 days late.

## Question and stages

**The question is whether a network that reads the last 6 days of power uses that history better
than the two power lags the existing networks read.** Stage 1 pre-trains one sequence network (Seq)
for all 16 Primaries on ERA5 weather and CAMS irradiance, with an 8-number Primary embedding. Stage 2
scores Seq beside the existing pre-trained networks on ERA5 and CAMS weather only, and is this
experiment's headline. Stage 3 fine-tunes on ECMWF ensemble weather and compares with K1, the
Candidate, and K3 from the roster export. Stage 3 is a separate later experiment and runs only if
stage 2 passes.

## Architecture

- **Tokens.** 336 history tokens (the 7 days before the origin) and 336 future tokens (the 7 days
  after), half-hourly, at width 64. Memory is not a constraint (`e00632_budget.py`: about 1 GB at
  batch 32 with plain attention; the Opus plan estimates about 3 GB at batch 256 with the fused
  kernel). Time is the risk, so the first fit is timed (section "Compute").
- **Inputs per token.** The roster's weather encoder and datetime encoder at the token's own time. A
  history token also has a lag encoder (64 hidden, 32 out) reading power as a share of capacity, a
  missing flag, the age in days, the token's weather and datetime embeddings, the Primary embedding,
  and neighbour summaries (sum, mean, minimum, and maximum over the electrical neighbours of their
  power as a share of the Primary's capacity, a neighbour count, and an absent flag). Future tokens
  take a learnt 32-number "no lag" vector in the lag encoder's slot. Every token adds a position
  embedding (steps from the origin) and a segment embedding (history or future). Future tokens also
  read the roster's two lead columns.
- **No lookahead.** A token carries power only if its half-hour is at least 24 h before the origin.
  The 48 newest history tokens therefore carry power 0 and missing flag 1, as do tokens whose cleaned
  reading is null. Neighbour summaries obey the same rule. `e00632_data.assert_no_lookahead` checks
  this on every sequence and against a copy of the power array with the newer readings replaced by
  1e6. The loss sits on future tokens only.
- **Block dropout.** As the roster: all power masked on about 25% of training sequences, the newest
  24 h of power-bearing tokens masked on about 5%. Signal dropout of 50% on the neighbour summaries.
- **Recurrence.** An encoder GRU reads the history embeddings. A decoder GRU reads the future
  embeddings and starts from a linear map (with tanh) of two encoder states: the final one and the
  one at the newest power-bearing token (origin minus 24 h).
- **Attention.** One stack of 3 joint self-attention layers over all 672 tokens (4 heads, feed-forward
  width 256), no causal mask, because every input is known at the origin. Cross-attention would need
  a second module and memory does not require it. Seq has about 250,000 parameters, against about
  30,000 for the roster network.
- **Output.** A three-component Gaussian mixture per future token, trained with the roster's negative
  log-likelihood (`e00397_models.mixture_nll`) and read out as 11 quantiles by bisection
  (`mixture_quantiles`). The slow output offset D5 is added to the component means exactly as in
  P2-pre.

## Training

- **Rows.** Daily origins at 09:00 UTC from ERA5 and CAMS pseudo-weather (the pre-training rows of
  experiments 418 and 574), within the 24 months before each fold's refit, with the D15 and D18 cut
  at 2025-03-01 and cleaned power only. A batch is gathered by index on the GPU from per-Primary
  half-hour arrays.
- **Fit.** AdamW, learning rate 5e-4 with 300 warm-up steps and cosine decay, weight decay 1e-2,
  dropout 0.1, gradient clip 1.0, batch 256, patience 8, at most 60 passes. Validation is the last 30
  origin days before the cutoff, and training sequences end 7 days before the first validation origin.
  The weather encoder, datetime encoder, and Primary embedding start from P2-pre's weights of the same
  fold and seed, at learning rate 1e-4. The lag encoder is fresh.
- **Seeds and folds.** Seeds 0, 1, and 2 on the five development months (2025-10, 2025-12, 2026-01,
  2026-03, 2026-06). July 2026 onwards is unused. Retraining is monthly, as the standing rule says.

## Comparison (stage 2)

- **Arms.** Seq; P2-pre (stored pre-training of the lag network, `00613-*/chains/d5/work/pretrained_P2`);
  K3-pre (stored lag-free D5 pre-training of experiment 565). Both comparators are scored by
  inference from stored weights through their pre-training head, on test rows built with experiment
  574's pre-training row builder at the test-month origins.
- **Row set.** 09:00 UTC origins on every day of the five months, leads 0 to 167.5 h, valid times
  before 2026-07-01, event-free rows as defined in the roster export, cleaned observed power as the
  target. D15 and D18 are report-only.
- **Contrasts and MDE.** The four contrasts above. The MDE is the half-width of the Seq minus P2-pre
  interval, printed beside each contrast. P2-pre minus K3-pre is scored first, at no GPU cost, to give
  the MDE on this row set and to check that lags help in the ERA5 world.
- **Tails.** q01, q05, q95, and q99 pinball loss and coverage, and the share above q90 (the roster
  is known to exceed q90 on about 14% of rows).

## Compute and fallbacks

- **Ceiling.** 6 GPU-hours. Fifteen pre-training fits (5 folds, 3 seeds) at 5 to 30 min each cost 1.3
  to 7.5 GPU-hours, so the timing run decides the route. Time one fit (fold 2026-06, seed 0) first.
- **Fallbacks, in order.** If a fit exceeds 20 min: sample 20% of sequences per pass; then hourly
  tokens (168 in, 168 out); then a 2-day forecast head. The write-up says which was used.
- **Resources.** One GPU process, `MemoryMax=8G`, `OMP_NUM_THREADS=1`, 4 cores,
  `torch.cuda.set_per_process_memory_fraction(8/48)`, bf16 autocast. If the final bake-off run
  (experiment 620) holds the GPU, wait.

## Run order

1. Score P2-pre and K3-pre on the row set (no fits).
2. Run P2-hist6 first, as a cheap test: P2's per-row network whose shared lag encoder reads every
   same-half-hour reading in the 7 days before the origin that is at least 24 h old (up to 6 lags,
   each with its age), mean-pooled. If it does not beat P2-pre by half the MDE, more history days
   carry little that a per-row network can use, and Seq can win only through within-day shape.
3. Time one Seq fit, then run seed 0 on the folds 2025-12 and 2026-06 and apply the stop rule.
4. Run all folds and seeds, then the ablations below (one seed, all five folds, descriptive, not
   counted): A-noAttn (GRUs alone), A-noGRU (attention alone), A-2d (2-day history), A-shared (one GRU
   for history and future), and A-noNB (no neighbour summaries, which separates the neighbour effect
   from the sequence effect, because P2-pre has none).

## Why Seq might not win

- About 1,030 pre-training origin days per Primary, and about 8 times the roster's parameters.
- Experiments 585, 560, and 558 found that more or longer lags did not help beyond the MDE.
- The 24 h cap empties the 48 newest history tokens, where a sequence network's edge usually lies.
- The roster is tuned over about 50 experiments, and Seq gets one configuration and three seeds.
- Reading 6 days of raw power may raise load-shift follow above the Candidate's 0.047 (stage 3).

## Stage 3 outline (not part of this experiment)

Fine-tune on ensemble weather with the roster's schedule (step 2, then refits on days 0 and 14 with the
E3 embargo), swap the neighbour columns for the gap summaries of experiment 610's arm P7, add the
9 h weather-run assertion, and compare with K1, the Candidate, and K3 on identical keys, days 1 to 7.
The worker brief's prediction (Seq matches the Candidate within the MDE at days 1 to 3 and is worse
than K3 beyond day 4 if it overfits) and its stop rule (first seed worse than K3 by more than 5% CRPS
at days 1 to 2) carry over.
