---
id: "00565"
slug: drift-embedding-variants-six-designs
title: >-
  Which design of a slow-change ("drift") signal helps the modular lag-free demand network: a monthly
  drift embedding, a slow-trend summary input, a random-walk embedding, a shared global-time
  embedding, an output-side slow offset, or a slow residual correction?
theme: retraining-and-training-history
also_relevant_to:
  - differentiable-physics-long-history-and-drift
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: forecasts, fitted weights, and stress-run forecasts under this experiment's
  private data folder only; experiments 545, 554, and 559 are read only. More than one defensible
  design: six, which is the question. Callers: the scripts import experiments 454, 470, 510, 511,
  521, and 545 and copy the model code of experiment 559. Serving path and degradation rule: not
  touched (this repository holds none). Stage 1 (design 6) is CPU only and re-scores stored
  forecasts; the GPU stages start after experiment 559 has merged.
hypothesis: >-
  Substations change slowly over years, and the Primary embedding is static, so the modular
  lag-free demand network follows slow change only through its daily fine-tune. A designed slow
  signal can carry that change. Stage 1 asks the cheapest version: an exponentially weighted mean
  of the network's own past residuals, older than 24 hours, added to the forecast quantiles.
prediction: >-
  Written before any score is read. Design 6 with the 28-day half-life closes at least 25% of the
  base network's CRPS gap to A-pool at 24 to 120 h at the eight, and fails if it raises the
  injected-shift leak by more than 1.0 point of capacity. The later stages carry the predictions of
  the brief (design 1 keeps at least 70% of M0's gain and lowers the injected leak by 1.0 point;
  design 2 gains at least 0.05 at the eight with the 365-day window beating the 90-day; design 3
  lies within 0.03 of design 1; design 4 gains at least 0.03 at both groups; design 5 gains at
  least 0.05 at the eight with leak at most the base's plus 1.0).
decision_rule: >-
  A gain counts when the Bonferroni month-block interval (5 blocks, near the minimum to maximum of
  the five folds) excludes zero and the gain is at least 0.03 CRPS points of capacity. Stage 1
  counts 12 CRPS contrasts (2 half-lives x 2 groups x 3 lead bands of 24-48, 48-72, and 72-120 h)
  against M0; the same contrasts against the other stored bases are reported uncounted. The whole
  experiment counts 36 CRPS contrasts (6 designs); the cold-night and Christmas-week families are
  counted separately. The threshold of 0.03 is the smallest gain worth one extra input column or
  one stored vector per Primary. Design 6 needs no model change, so its threshold is zero with a
  lower bound that excludes zero.
benchmark: >-
  M0, M0lr, L-1d, and A-pool (experiments 545 and 511), the 559 arms A1, A10, and A11 (read only),
  and experiment 500's 28-day median shift as the reference for design 6.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity by lead band at the eight MVA-only Primaries and the six
  signed-MW Primaries (D15 and D18 report-only); injected-shift leak in points of capacity.
---

**Planned.** Stage 1 (design 6) is written first; stages 2 to 4 are added to this page, with their
own commit, before any GPU fit starts.

## Stage 1: design 6, a slow residual correction (CPU only, no fit)

- **Method.** For every stored forecast row of a base arm, with origin `o` and Primary `p`, the
  correction is the exponentially weighted mean of the residuals (actual minus `q50`) of that
  Primary's earlier forecast rows whose target is at most `o` minus 24 h and whose own origin is at
  most `o`. Each row's weight is `0.5^(age / H)`, with age the days between `o` and the row's
  target and `H` 28 or 90 days. The sum is divided by `max(sum of weights, 48)`, so the correction
  shrinks to zero where history is short, and the same shift is added to all 11 quantiles. The
  constants (24 h, 48, and the two half-lives) are fixed here and are not tuned.
- **Why 24 h and not the 6 h delivery delay.** "Lag-free" means no input from the target's own
  series inside 24 h of the origin. The cut is asserted row by row, and one example row is printed.
- **History.** Only the fold's own month supplies residuals, because the stored forecasts cover the
  five fold months and nothing earlier. The 90-day half-life therefore behaves like a mean over the
  fold so far, and the first days of each fold carry little correction.
- **Bases scored.** M0, A1, A10, and A11 (the pooled quantiles of seeds 0 and 1). The base for the
  GPU stages is declared in the stage 2 commit, from experiment 559's write-up, before any fit.
- **Injected-shift leak.** Computed in closed form for the stress world of experiment 511: the
  injected block adds `sign x shift` (capacity fraction) to the actuals of the 7-day block, and
  the forecasts issued before the block's end are unchanged, so the correction's response is the
  weight share of block rows. The response of the base network's own leak to the correction is not
  included, so the figure is an upper bound.
- **Reference.** Experiment 500's 28-day median shift (A-pool, per lead band) cut q99 pinball by
  13%; this stage uses a per-Primary mean over all leads and does not reproduce 500's table.

## Step R: five-seed replication of A10 against M0 (GPU)

- **Fits.** A10 seeds 2 to 4 are new fits with experiment 559's code, run in place; seeds 0 and 1 are
  experiment 559's. M0 seeds 0 to 4 are experiment 554's. Both sets are daily-refit fits on the five
  folds of 511, so the pairs match by fold and seed. Each seed is scored on its own (widened by 1.15,
  as the stored row tables are) and the contrast of a row is the mean over seeds of the paired
  difference.
- **Counted contrasts: 8.** Two row sets (the eight and the six) x three lead bands (24-48, 48-72, and
  72-120 h), and the cold-night rows of each set at 24-120 h (3 folds hold cold nights). The
  minimum detectable effect is 1.96 x the seed-to-seed standard deviation of the contrast, pooled
  within folds, divided by the square root of 5.
- **Leak.** The injected 7-day shift of experiment 511 on all five seeds of both arms (M0 seeds 1 to
  4 are new fits run with experiment 554's code). Real-event leak from experiment 556's harness is not
  run here.

## Stages 2 to 4: the model designs (coarse screening, GPU)

- **Base.** Experiment 559 names no arm on the gain-and-leak frontier without a drift term. The base
  is therefore the coarse refit of M0 (C-M0: ERA5 and CAMS pretrained weights of experiment 554 and a
  fresh mixer, 90-day half-life; CRPS 4.465 at the eight and leak 7.60 points in the daily harness).
  The reference design is the coarse refit of A10 (C-A10).
- **Coarse refit.** The screening fits twice per fold, on days 0 and 14 (cadence `h`), so that the
  injected-shift control exists at day 14; the daily harness is not run. Seeds 0 and 1, five folds.
  The injected-shift leak is a single fit at day 14 with the block injected, seed 0, scored over the
  forecast window that follows (days 14 to the fold's end).
- **Arms.** D3m (design 3, monthly knots, L2 on first differences, encoders not frozen), D3q (the same
  with quarterly knots), D4 (design 4, one embedding per month shared by all Primaries, per-Primary
  scale, encoders frozen) and D4-null (permuted month index), D5 (design 5, per-Primary slow offset
  added to the component means, first-difference penalty, encoders frozen). The penalty weight is
  10 for every design and is not tuned. Design 2 (slow-trend summary input) follows when its builder
  exists.
- **Pass rule.** An arm passes when its CRPS at 24-120 h is at least 0.03 better than C-M0 at the eight
  or the six, with a Bonferroni month-block interval (36 counted contrasts) that excludes zero, and
  its injected-shift leak is at most C-M0's plus 1.0 point. At most 3 passing arms get the daily and
  per-event refits.
- **Predictions that could fail.** As in the brief: design 3 within 0.03 of design 1; design 4 gains
  at least 0.03 at both groups and its placebo does not match it; design 5 gains at least 0.05 at the
  eight with leak at most the base's plus 1.0.

## Step R result (interim and unaudited)

**A10 (the monthly drift embedding) is no better than M0 at the eight, is better at the six, is worse
on cold nights, and leaks about 1 point less.** The pre-registered prediction fails: the leak falls by
0.95 points rather than the 1.0 predicted, and the sign of A10 minus M0 at the eight is not the same
in 4 of 5 seeds.

- **Eight, 24-120 h.** A10 minus M0 is +0.010 (positive is worse; Bonferroni interval -0.31 to 0.29,
  minimum detectable effect 0.133), with per-seed values -0.001, -0.035, +0.085, -0.066, and +0.068.
  A10 keeps about 94% of M0's gain over L-1d, but the difference between the two arms is not shown.
- **Six, 24-120 h.** A10 minus M0 is -0.134 (interval -0.37 to 0.15, minimum detectable effect 0.073).
  All 5 seeds are negative and 4 of 5 folds favour A10, but the interval includes zero.
- **Cold nights (3 folds hold them).** A10 is worse by +0.183 at the eight (interval 0.08 to 0.23) and
  by +0.454 at the six (interval -0.39 to 0.86, 5 of 5 seeds worse). Experiment 559's two-seed gain on
  cold nights at the six (-0.348) does not replicate.
- **Injected load-shift leak (all 5 seeds, mean of 5 folds).** A10 6.81 points of capacity (seed
  range 5.96 to 7.28) against 7.76 for M0 (7.50 to 8.03). A10 is lower in all 5 seeds, by 0.31 to 2.08.
