---
id: "00544"
slug: lag-light-demand-network-robust-same-slot-median-and-lag-dropout
title: >-
  Can a median of the same half-hour over past days, or training with the power readings randomly
  removed, make the lag-light demand network less sensitive to load-shift events and to missing
  readings without losing its gain?
theme: lagged-power-and-error-correction
also_relevant_to:
  - demand-inputs
status: planned
kind: other
size: >-
  medium. Stored: fitted network weights, window forecasts, and score tables under this experiment's
  private data folder only. Serving path: not touched (this repository holds none). Degradation
  rule: not touched. Designs: more than one is defensible (which half-hours enter each median, and
  how an absent reading is coded), so the plan fixes both before any score is read. Callers: this
  folder's scripts import experiments 428, 454, 470, 510, 511, and 521 and the unmerged scripts of
  experiment 539 (read from its private data folder, copied here). One Sonnet code review before the
  first fit, one Sonnet review of the write-up against the printed report, one Sonnet prose review.
hypothesis: >-
  Experiment 539's lag-light network (the lag-free network plus the day-ago and week-ago power
  readings, each with a missing flag and an age) closes 76% to 83% of the lag-free network's gap to
  the lag-enabled network A-pool, but it follows an injected load shift more than the lag-free
  network does. Two changes could keep the gain and cut the weakness. (1) R-MED: replace each single
  reading by the median of the same half-hour over many days, which one shifted day cannot move.
  (2) R-DROP: train with the readings removed in 30% of rows, so the network also works when they
  are late or absent.
prediction: >-
  Written before any score is read. P1 (R-MED): within 0.03 CRPS of R-DW (experiment 539) at 24-120 h
  at the eight, and its injected-shift leak is at least 0.5 points of capacity below R-DW's 5.68 (at
  most 5.2). P2 (R-DROP with the readings present): keeps at least 90% of R-DW's closed gap at
  24-120 h at the eight. P3 (R-DROP with the readings absent in every scored row): costs at most 0.15
  CRPS more than plain L-1d at 24-120 h at the eight. Either arm may fail, and the page says so.
decision_rule: >-
  A prediction counts as met only when its stated margin holds in the pooled 24-120 h band at the
  eight and its paired interval does not contradict it. Contrasts R-MED minus R-DW, R-DROP minus
  R-DW, and R-DROP (absent) minus L-1d are counted, per lead band and group; all others are
  report-only. Every contrast is shown with its minimum detectable effect from L-1d's seed spread.
  There is no gain threshold for porting: the change adds inputs already in 539, so the question is
  whether either variant removes a weakness without losing the gain.
benchmark: >-
  L-1d (experiment 511), A-pool (experiment 425, rows of 533), R-DW (experiment 539), and R9
  (experiment 538), all read from their stored forecasts; `manual_heuristic` for scale.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at the eight MVA-only and the six signed-MW Primaries by seven
  lead bands; D15 and D18 report-only; injected-shift leak (experiment 511's stress world) in points
  of capacity by block age.
---

# Lag-light variants: median of the same half-hour, and lag dropout

*Plan only; the write-up replaces this page.*

## Plan

- **Arms, on experiment 539's harness and rows.** R-MED, R-DROP, each with seeds 0 and 1 over the
  five folds of experiment 511 (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), daily refit from the
  month's pretrained weights, 90-day half-life. The six lag inputs are appended with zero
  first-layer weights.
- **R-MED inputs.** The day-ago slot holds the median of the same half-hour on the 7 most recent
  days, and the week-ago slot the median of the same half-hour on the 28 most recent days. Every
  member is at least the newest usable observation age old (origin minus 3.5 h). A median needs at
  least 3 of 7 (or 10 of 28) members observed, otherwise the slot is missing. The age of both slots is
  the age of the newest member.
- **R-DROP training.** In 30% of rows (training and early-stopping rows alike, fixed per refit) all
  six inputs are set to "absent": value 0 after standardising, missing flag 1, age at the training
  maximum. This follows the data convention for a missing reading; the brief's "flag 0" is read as
  "present flag 0".
- **R-DROP is scored twice**: with the readings present, and with every row's readings absent.
- **Checks:** every weather input from the run issued at or before the origin, and every power value
  at least 3.5 h older than the origin, asserted row by row with one printed example.
