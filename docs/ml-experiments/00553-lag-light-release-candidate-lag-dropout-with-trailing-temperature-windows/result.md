---
id: "00553"
slug: lag-light-release-candidate-lag-dropout-with-trailing-temperature-windows
title: >-
  Does one demand network that reads the day-ago and week-ago power readings, trailing temperature
  windows, and is trained with those readings randomly removed keep most of the gain of the readings
  and still work without them?
theme: lagged-power-and-error-correction
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
size: >-
  medium. Stored: fitted network weights, window forecasts, and score tables under this experiment's
  private data folder only. Serving path: not touched (this repository holds none). Degradation
  rule: not touched. Designs: the brief fixes the recipe and asks for two dropout rates, so one
  defensible design. Callers: this folder's scripts copy experiment 550's fit script and experiment
  544's dropout code, and reuse the scorers and slices of experiments 521, 544, 550, and 551. One
  fresh Sonnet code review before the first fit, one Sonnet review of the write-up against the
  printed report, and one Sonnet prose review.
hypothesis: >-
  Experiment 544's lag dropout cost most of the gain of the day-ago and week-ago power readings (it
  closed 31% to 40% of L-1d's gap to A-pool where experiment 539's network closed 76% to 83%). The
  trailing temperature windows of experiments 548 and 550 may restore part of that gain, and a
  lower dropout rate may lose less of it. A network with all three (readings, windows, and
  dropout) could therefore close most of the gap while still working when the readings are absent.
prediction: >-
  Written before any score is read. P1: with the readings present, the better of the two dropout
  rates closes at least 85% of L-1d's gap to A-pool at 24-120 h at the eight (this may well fail;
  experiment 539's network closed 76% to 83% without dropout). P2: with every reading removed, the
  candidate is within 0.15 CRPS of the lag-free recipe of experiment 548 at 24-120 h at the eight.
  P3: in Christmas week the candidate is no worse than L-1d by more than 0.10 CRPS at the eight. P4:
  the injected-shift leak is at most 6.0 points of capacity.
decision_rule: >-
  The candidate is the dropout rate that closes more of the gap at 24-120 h at the eight among the
  rates whose leak is at most 6.0 (this rule is fixed before scoring). All four predictions must hold
  for the candidate to count as a release candidate; any failure is reported as a failure. If the
  gap closed is under 70%, the lag-light line stops. There is no gain threshold for porting,
  because the change reuses inputs and a training trick already built. Contrasts are shown with
  month-block bootstrap intervals (5 blocks, near the minimum to maximum of the folds), per-fold
  signs, the minimum detectable effect from L-1d's two-seed spread, and the Bonferroni count as a
  number (84 = 2 rates x 3 contrasts x 7 lead bands x 2 row sets).
benchmark: >-
  L-1d (experiment 511), R-DW (experiment 539), R-DROP (experiment 544), A-pool (experiment 425),
  the lag-free recipe with trailing windows (experiment 548, arm E), the retrained K2-S1
  (experiment 541), and `manual_heuristic` for scale, all read from stored forecasts.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at the eight MVA-only and the six signed-MW Primaries by
  seven lead bands; D15 and D18 report-only; July 2026 unused; injected-shift leak in points of
  capacity (experiment 511's stress world).
production_status: not-for-production
adopted_in: none
---

Planned. Written before any fit. The arms are `RDWT-D30` and `RDWT-D15` (R-DW plus the three trailing
temperature windows of experiment 515, trained with the six lag inputs removed in 30% and 15% of
fine-tuning rows), each scored with the readings present and with every reading removed. Five folds
of 511, seeds 0 and 1 for every contrast (the seeds of all stored comparators) and seed 2 as a
replication. Cost estimate 2 GPU-hours.
