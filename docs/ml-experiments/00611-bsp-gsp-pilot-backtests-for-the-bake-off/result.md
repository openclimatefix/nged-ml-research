---
id: "00611"
slug: bsp-gsp-pilot-backtests-for-the-bake-off
title: >-
  How well does the lag-free demand network forecast the 2 bulk supply points and 2 grid supply
  points when each group is pooled on its own?
theme: hierarchy-and-pooling
also_relevant_to: []
status: planned
kind: other
date: 2026-11-06
size: >-
  medium, by the five triggers. Stored: fitted weights, forecasts, and a Parquet export, all under
  this experiment's private data folder. More than one design is defensible, which is why the plan
  fixes two (Design 1, Design 2) from experiment 604 before any score is read. Callers: the scripts
  import experiments 87, 397, 415, 428, 448, 510, 511, 559, 564, 565, and 577. No serving path or
  degradation rule is touched (this repository holds none). One Sonnet code review before the first
  fit, one Sonnet write-up review, one Sonnet figure review.
hypothesis: >-
  The lag-free network with the steadier offset O2 (experiment 577), pooled per group, forecasts
  the 2 bulk supply points (BSPs) and the 2 grid supply points (GSPs) at least as well as a
  month-by-half-hour climatology, and at the BSPs as well as the weekly XGBoost model (BG2).
prediction: >-
  Written before any score is read. Design 1 beats the climatology at all 4 series at 24 to 120 h
  (CRPS lower by at least 1 point of effective capacity, as experiment 564 found at 24 to 120 h).
  Design 1 is within the minimum detectable effect of BG2 at the BSPs and worse than BG2 by more
  than 1 point at the GSPs, because the GSPs carry embedded generation. The q05 to q95 band covers
  between 0.85 and 0.95 of rows at every lead band. The GSP that mostly exports power has a
  coverage below the other three series.
decision_rule: >-
  This is a pilot that produces forecasts for the bake-off, so there is no adoption rule. Design 1
  against Design 2 and each design against the references are descriptive; the Bonferroni count is
  stated beside each contrast, with its minimum detectable effect. A contrast counts as a gain when
  its Bonferroni month-block interval (5 blocks) excludes zero.
benchmark: >-
  Climatology of the fold's training window; weekly XGBoost model (BG2) of experiment 415 where its
  stored forecasts cover the same rows, and otherwise stated as missing.
row_set_id: experiment_611_nodes_v1
metric: >-
  CRPS in points of effective capacity, q05 to q95 coverage, and q05 to q95 width, by lead band
  (0 to 24, 24 to 120, 120 to 240 h) and by group (BSPs, GSPs), with the GSP that mostly exports
  power reported separately.
code: ml-experiments/00611-bsp-gsp-pilot-backtests-for-the-bake-off/
---

**Plan only. No fit has run.** The design comes from the merged page of
[experiment 604](../00604-bsp-gsp-pilot-design-for-the-bake-off/result.md), whose recommended
Designs 1 and 2 this experiment runs.

## Design

- **Series.** Two BSPs (D13 Skegness, D17 Boston) and two GSPs (D1 Bicker Fen, D6 Walpole), pooled
  per group, so each network has 2 series.
- **Design 1.** The lag-free network of experiment 565 with the offset O2 of
  [experiment 577](../00577-static-between-steps-offsets-and-probabilistic-offset-widening/result.md)
  (the total-variation arm), pretrained on ERA5 and CAMS rows, then fine-tuned with a fresh mixer.
- **Design 2.** The lag network of
  [experiment 415](../00415-pooled-mixture-density-network-at-bulk-and-grid-supply-points/result.md),
  with the shared lag encoder A1, if a GPU slot is free after Design 1.
- **Folds.** The five development months 2025-10, 2025-12, 2026-01, 2026-03, and 2026-06; coarse
  refits on days 0 and 14; seeds 0 and 1. July 2026 onwards is never read.
- **Weather.** The run issued at or before the forecast origin minus 9 h.
- **Compute.** At most 2 GPU processes, each under 8 GB, ceiling 3 GPU-h.
- **References.** A month-by-half-hour climatology from the fold's training window, and BG2 from
  stored results if they cover these rows.
