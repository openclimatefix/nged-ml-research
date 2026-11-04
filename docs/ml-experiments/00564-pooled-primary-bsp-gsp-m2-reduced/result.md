---
id: "00564"
slug: pooled-primary-bsp-gsp-m2-reduced
title: >-
  Does training the modular lag-free demand network on 2 bulk supply points and 2 grid supply points
  as well as the Primaries lower the Primaries' forecast error?
theme: hierarchy-and-pooling
also_relevant_to:
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: pretrained and fine-tuned network weights, window forecasts, and pretraining rows
  for four extra series, all under this experiment's private data folder; no earlier experiment's
  files are written. Serving path: not touched (this repository holds none). Degradation rule: not
  touched. Designs: more than one is defensible (where the extra series enter, how their loss is
  weighted, how the series type is fed in), so this plan fixes each before any score is read.
  Callers: this folder's scripts import experiments 87, 397, 418, 422, 428, 448, 510, 511, 545, and
  554. One fresh Sonnet code review before the first scored run.
hypothesis: >-
  Adding the 2 bulk supply points (BSPs) and 2 grid supply points (GSPs) to the training rows of the
  modular lag-free network M0 (experiment 545, replicated in 554), with a one-hot input saying
  whether a series is a Primary, a BSP, or a GSP, lowers the Primaries' CRPS at leads of 24 to 120 h,
  because the weather and calendar encoders see more examples of how weather and calendar drive load.
prediction: >-
  Written before any score is read. P1. M2-a (extra series in pretraining and fine-tuning) beats M0
  by at least 0.05 points of capacity (CRPS, pooled 24-120 h) at the eight MVA-only Primaries and is
  within 0.03 of M0 at the six signed-MW Primaries; the prediction fails if M2-a is within 0.02 of M0
  at both groups. P2. M2-b (pretraining only) is no better than M2-a. P3. The injected-shift leak
  (all leads, mean of five folds) of M2-a does not rise by more than 1.0 point of capacity above M0's,
  if the stress run is affordable. Four extra series are few, so the expected gain is small.
decision_rule: >-
  A gain counts when the Bonferroni month-block interval (5 blocks, near the minimum to maximum of
  the five folds; 42 counted tests: 3 arms times 2 Primary groups times 7 lead bands) excludes zero
  and the pooled 24-120 h point estimate meets P1. The threshold of 0.05 is the cost of a pooled
  training set (four more series to clean, align, and refit daily), and sits below the minimum
  detectable effect, so a pass reads "interval excludes zero". The experiment fails if M2-a is within
  0.02 of M0 at both groups. The BSP and GSP scores are reported, not tested.
benchmark: >-
  M0 of experiment 554 (5 seeds; seeds 0 and 1 are experiment 545's stored fits), on experiment 511's
  rows. The arms here use seeds 0 and 1, so the contrast is against M0's seeds 0 and 1 (the same 2
  seeds); the 5-seed M0 is a report-only reference. L-1d (experiment 511), `manual_heuristic`, and
  `climatology` are report-only references.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity by the seven lead bands of experiment 521, at the eight
  MVA-only Primaries and the six signed-MW Primaries (D15 and D18 report-only), and at the 4 extra
  series separately.
---

**Plan only: no fit has run.** The question is whether one network trained on 20 series (the 16
Primaries, 2 bulk supply points, and 2 grid supply points) forecasts the Primaries better than M0,
which trains on the 16 Primaries alone. The maintainer asked for this reduced version of the test
after [experiment 563](../00563-pooled-primary-bsp-gsp-data-check/result.md) found that NGED's data
holds only 4 such series, below the 8 the original brief required.

## Design

- **Series.** The 16 Primaries keep their indices 0 to 15. The extra series follow as indices 16 to
  19: D1 and D6 (GSPs), D13 and D17 (BSPs). All four are signed MW. Weather rows, cleaned power, and
  ERA5 and CAMS pretraining rows come from the same recipe as the Primaries, built for the four
  series from files already on disk (no download).
- **Series-type input.** A fixed one-hot vector (Primary, BSP, GSP) per series is concatenated to the
  datetime encoder's input, beside the learned 8-wide embedding. It is not a column of the input
  matrix, so every other part of the rows is unchanged.
- **Arms.** M2-a: extra series in pretraining and in fine-tuning. M2-b: extra series in pretraining
  only; fine-tuning uses the 16 Primaries. M2-c: extra series in fine-tuning only, starting from M0's
  pretrained weights. M0 (experiment 554) is the control. 2 seeds (0 and 1), 5 folds of 511, daily
  refit with a 90-day half-life, as in M0.
- **Loss weight.** Primary rows keep M0's recency weights. Each extra series' rows are scaled so that
  the series' total weight equals the mean total weight of a Primary, so the extra series raise the
  total training weight by about 4 in 20 and never outweigh a Primary.
- **Leak rule.** Extra-series rows enter training only before each cutoff minus 3.5 h, as every row
  does. A GSP is a near-duplicate of the BSP beneath it, and the network reads no power input, so
  no series' target is another series' input.
- **Scoring.** The Primaries are scored with experiment 554's scorer on 511's rows. The extra series
  are scored separately (CRPS beside `climatology` and `manual_heuristic`), because 511's row set
  holds Primaries only.

## Checks planned

- Origin-time weather asserted row by row for training and test rows, with one example row printed
  (experiment 554's assertions, applied to all 20 series).
- Leave-D8-out beside every Primary contrast; per-fold signs; the minimum detectable effect from the
  2-seed spread beside every contrast.
- The median forecast minus the mean forecast per site, and the error of the mean forecast beside the
  CRPS contrasts, as the Checks list requires.
