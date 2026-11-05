---
id: "00568"
slug: attention-mixer-modular-network-with-lead-and-spread-token
title: >-
  Does a self-attention mixer over the encoder outputs, with a token for lead time and the spread of
  the weather ensemble, beat the multi-layer-perceptron mixer of the modular lag-free demand
  network?
theme: retraining-and-training-history
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fine-tuned weights of the final candidate arms, step-2 weights, pretrained
  weights, forecasts, and attention weights in a private folder. More than one defensible design
  (token set, spread normalisation, how the mixer is pretrained). Spans code whose callers need a
  search (experiments 545, 554, 559, 560, 566). Serving path and degradation rule: not touched (this
  repository holds none). Cost ceiling 5 GPU-h, set by the maintainer; the brief estimates 4.5.
hypothesis: >-
  The modular network M0 of experiment 545 concatenates its encoders' outputs and feeds them to a
  multi-layer-perceptron (MLP) mixer. A self-attention mixer that reads each encoder's output as a
  separate token, plus a token built from the lead time and the ensemble spread of several weather
  variables, extracts more from the same encoders, and does so without raising the injected-shift
  or real-event leak.
prediction: >-
  Written before any fit. P1: T1 (attention, full) beats T0 (the MLP mixer on the same encoders) by
  at least 0.05 CRPS at 24-120 h at the eight and by at least 0.03 at the six; fails if within 0.03
  (eight) or if the sign differs in 2 or more folds. P2: T1 beats T2 (no spread token) by at least
  0.03, with the gain concentrated in the top spread tercile; fails if within 0.02. P3: T1 beats T3
  (temperature spread alone) by at least 0.02; fails if within 0.01. P4: T4 (MLP with the same spread
  columns) is at least 0.02 worse than T1; fails if within 0.01 (then the spread columns, not
  attention, carry the gain). P5: T5 (shuffled-spread placebo) is within 0.02 of T2; fails if T5
  gains as much as T1. P6: T1's injected-shift leak is at most T0's plus 1.0 point of capacity, and
  the attention weight on the weather token exceeds that on the Primary token in every lead band;
  fails if the Primary token's weight rises with lead. P7 (T6, run only after experiment 567 has
  merged): T6 is within 0.04 of A-pool-24 at 24-120 h at the eight and has a lower real-event leak
  than the MLP stack.
decision_rule: >-
  An arm passes screening if CRPS at 24-120 h at the eight or at the six is at least 0.03 better than
  T0 with a Bonferroni interval excluding zero, and its injected-shift leak is at most T0's plus 1.0
  point of capacity. At most 3 passers get daily refits and the real-event leak harness of experiment
  566, where the real-event mean absorbed share is the primary leak measure and the injected 6.0
  points is a flag. The threshold of 0.03 is set by the cost of porting: attention adds a new network
  block and a new input group to the serving path. Bonferroni count 36 for screening (30 CRPS
  contrasts for T1 against T0, T2, T3, T4 and T5 against T2, each at 2 groups and 3 lead bands, plus
  6 for T6), and 6 for each of the cold-night, Christmas-week, and spread-tercile families. Wording
  is "interval excludes zero" or "not shown".
benchmark: >-
  T0, the MLP mixer on the same encoders under the same schedule, with its parameter count raised to
  T1's within 10%; stored daily-refit M0 (experiment 545), A1 (experiment 559) and LAG-v3 (experiment
  560) as report-only context.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at 24-120 h and by the seven lead bands of experiment 521 and
  0-6 h, at the eight MVA-only Primaries and the six signed-MW Primaries (D15 and D18 report-only);
  injected-shift leak in points of capacity; q01, q05, q95, q99 coverage and pinball loss.
---

**This page is the plan; the write-up replaces it when the experiment is scored.** We ask whether the
demand network that forecasts a Primary substation's load gets better when its final block, the
mixer, reads each encoder's output as its own token and attends over the tokens, instead of reading
the outputs glued into one vector. A new token carries the lead time and how much the members of the
weather ensemble disagree about several variables. We fit six variants and the original, score them
on five test months, and check that none of them leaks a sudden shift in recent demand into the
forecast more than the original.

## Arms

- **Base recipe:** experiment 554's M0 encoders (weather, datetime, Primary embedding) and ERA5/CAMS
  pretraining, then experiment 559's A1 schedule (a step 2 that trains a fresh mixer and the
  encoders on all rows up to the first cutoff, then the fine-tune with the 90-day half-life). The
  attention arms pretrain their own encoders under an attention mixer without the lead-and-spread
  token (one pretraining per fold and seed); T0 and T4 reuse experiment 554's pretrained M0 weights.
- **T0 (control):** the MLP mixer of M0, with its hidden width raised so that its parameter count is
  within 10% of T1's. **T1:** the full attention mixer. **T2:** T1 without the spread token (lead on
  the query token only). **T3:** T1 with the spread token built from 2 m temperature spread alone.
  **T4:** T0 with the multi-variable spread columns of T1 added. **T5:** T1 with the spread columns
  shuffled between days of the same Primary and month. **T6:** T1 with own-lag and neighbour-lag
  tokens, run only after experiment 567 has merged; this plan does not cover it.
- **Tokens (d = 32, 4 heads, 2 pre-norm layers, dropout 0.1, learned token-type embedding):** weather,
  datetime, Primary embedding, lead-and-spread, and a learned query token that feeds the 3-component
  Gaussian mixture head. Each non-weather token is dropped with probability 0.1 in training.
- **Spread columns:** the standard deviation across the 51 members of 2 m temperature, 10 m wind
  speed, shortwave radiation, longwave radiation, dew point, surface pressure, and precipitation (the
  seven stored spread columns of M0), the u and v components of 10 m wind (experiment 559's
  `UvSource`), and total cloud cover (`CloudSource`; present only from ENS run 2025-11-21, and a
  missing flag marks earlier rows). Each is divided by its per-Primary median over the training
  days. The multi-variable token uses temperature, wind speed, shortwave, the two wind components,
  and cloud; T3 uses temperature alone. The drift token is left out because experiment 565's
  five-seed replication of A10 has not merged.

## Scoring and rules

- **Coarse refit (one pretrain, one step 2, and one fine-tune per fold and seed), 5 folds of 511
  (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), 2 seeds.** The injected 7-day shift of experiment
  511 (seed 0) is the cheap leak check. July 2026 is unused. Rows with no labelled event.
- **Lead-and-spread columns of a row come from the ensemble run issued at or before the forecast
  origin;** the script asserts it row by row and prints one example row with the run's initialisation
  time and the origin.
- **Statistics:** 5-block month bootstrap (near the minimum to maximum of the folds), per-fold signs,
  the minimum detectable effect from the 2-seed spread beside every contrast (experiment 545's paired
  values were 0.134, 0.139, and 0.101 at the eight), leave-D8-out beside every contrast, D15 and D18
  report-only. Every number is interim and unaudited until the reviews have run.
- **Compute:** one GPU process under 8 GB (`MemoryMax=8G`, `OMP_NUM_THREADS=1`), started only when
  available memory exceeds 20 GB, 5 GPU-hours in total; the maintainer is asked before this is
  exceeded. The fine-tuned weights of the final candidate arms are saved.
- **Anonymisation and wording:** metered generators appear only as Farm A to C and Solar A to F, and
  the page says "load-shift".
