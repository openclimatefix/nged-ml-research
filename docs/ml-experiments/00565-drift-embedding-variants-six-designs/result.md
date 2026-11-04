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

## Stages 2 to 4: the model designs (GPU, after experiment 559 has merged)

Step R first (A10 against M0, seeds 0 to 4, five folds), then designs 3, 5, 4, and 2, as the brief
`drift_embedding_variants_brief.md` orders them. The details are added in the stage 2 commit.
