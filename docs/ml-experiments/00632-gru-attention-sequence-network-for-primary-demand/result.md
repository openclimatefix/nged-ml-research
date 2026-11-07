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
  Complex (draft; the Opus plan will replace this paragraph). Answers to the five triggers: (1) it
  stores fitted network weights and forecasts in the private data folder only; (2) more than one
  design could test the question (joint attention against cross-attention, token length, window
  length), so the plan fixes them in advance; (3) it reuses code from experiments 397, 418, 560,
  610, and 614, whose callers cannot be named without searching; (4) it touches no serving path and
  no degradation rule; (5) it changes no contract. Ceiling 6 GPU-hours (process hours).
hypothesis: >-
  DRAFT, pending the Opus plan. A demand network built as an encoder GRU over the last week, a
  decoder GRU over the next week, and three self-attention layers over all tokens, pre-trained on
  ERA5 and CAMS weather with one Primary embedding for all 16 Primaries, scores better on ERA5 and
  CAMS weather than the existing pre-trained networks.
prediction: >-
  DRAFT, pending the Opus plan. The sequence network matches or beats the existing pre-trained
  networks at days 1 to 3 and loses at days 4 to 7 if it overfits.
decision_rule: >-
  DRAFT, pending the Opus plan. Stop rule: if the first seed is worse than the lag-free network by
  more than 5% CRPS at days 1 to 2, stop and report. Stage 3 (fine-tuning on ensemble weather) is a
  separate later experiment and runs only if stage 2 shows a win.
benchmark: >-
  The pre-training-stage scores of the existing networks (experiments 560, 608, 612, and 613). The
  Opus plan names the stored files.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS and MAE by horizon (days 1 to 7), q05-q95 coverage, and upper-tail exceedance, on the five
  development months and 16 Primaries.
---

**Draft plan, written before any fit.** The Opus plan, requested by the manager, replaces the
placeholders above. This page records only what is settled so far.

## Settled so far

- **Architecture.** An encoder GRU reads the last 7 days (336 half-hour tokens). A decoder GRU,
  initialised from the encoder GRU's final hidden state, reads the next 7 days (336 tokens). Both GRUs
  receive embeddings from a weather encoder, a datetime encoder, and (history only) a lag encoder
  that also sees the electrical neighbours of the Primary. Three self-attention layers then let every
  token attend to every other token, past and future together. The output is a three-component
  Gaussian mixture per future half-hour, read out as 11 quantiles, with the loss of the roster.
- **Token budget.** 672 tokens fit: with width 64, 4 heads, 3 layers and batch 32, the pessimistic
  estimate in `e00632_budget.py` is 1.0 GB, and the fused attention kernel uses less. Hourly tokens
  (168 in, 168 out) and a 2-day forecast head stay as fall-backs.
- **No lookahead.** Own power and neighbour power are visible only at half-hours at least 24 h before
  the forecast origin, so the last 24 h of the history window carry weather and datetime inputs and
  no power. `e00632_data.assert_no_lookahead` checks this row by row and against a copy of the power
  array whose newer readings are replaced by 1e6.
- **Stage 1 and stage 2.** Stage 1 pre-trains one network for all 16 Primaries on ERA5 and CAMS
  weather with a Primary embedding. Stage 2 pauses and compares the pre-trained network with the
  existing approaches on ERA5 and CAMS weather only. Stage 3, fine-tuning on ensemble weather, is a
  separate later experiment.
- **Caveat.** ERA5 arrives about 5 days late, so scores on ERA5 weather are a ceiling for a live
  forecast.
