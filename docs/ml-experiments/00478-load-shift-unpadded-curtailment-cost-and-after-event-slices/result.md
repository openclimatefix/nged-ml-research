---
id: "00478"
slug: load-shift-unpadded-curtailment-cost-and-after-event-slices
title: >-
  Part F2 of experiment 395: what do load-shift events cost in avoidable curtailment, and how long
  does the damage last after an event ends?
theme: lsc-detectors
also_relevant_to:
  - production-reference-and-best-guess
  - lagged-power-and-error-correction
status: planned
kind: other
protocol: demand-v1
size: >-
  simple. Stored: result tables and one figure per forecast in this experiment's private folder; no
  forecast is written and no model is fitted. Designs: one defensible design, because experiment
  395 Part F already tagged every scored row against the unpadded events, and this part adds
  columns and scores. Callers: the scripts import experiment 395's Part F build and the scorers of
  experiments 381 and 390, and nothing imports them. The repository holds no production serving path.
hypothesis: >-
  Rows with a lag inside an unpadded load-shift event raise the avoidable-curtailment cost of the
  weekly bundle of experiment 381 above that of the GAM stack of experiment 377 at the signed-MW six,
  and the extra harm lasts for the 7 days after the event ends.
prediction: >-
  Written before any score is read. P1. In the 7 days after events, the weekly bundle's
  avoidable-curtailment cost minus the plain weekly member's is positive at the signed-MW six at
  f = 0.9. P2. The CRPS harm of one to three lags in an event is largest for targets 0 to 24 hours
  after the event ends and falls to within its interval of zero by 3 to 7 days after. P3. Experiment
  391's surprise arm (n1) has a smaller harm than its control (b0). P4. Events of over 7 days do
  more harm per row than events of under 1 day.
decision_rule: >-
  Descriptive re-scoring of stored forecasts under `demand-v1`: nothing is refitted, so no adoption
  claim is made. Intervals resample whole events (95% per Primary, 99% pooled, k = 6 planned
  contrasts, so a level of 99.17%). A contrast with under 5 events is descriptive only. The
  threshold is zero, because the experiment changes nothing in production.
benchmark: >-
  The plain weekly member (experiment 383 arm d) and the GAM stack of experiment 377, on
  `fold_a_daily_gap_v1`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Avoidable-curtailment cost in pounds (`metrics.avoidable_curtailment_cost`, signed-MW six, node
  only, limit fraction f = 0.9, 0.75, and 0.5, 12 to 36 hour leads), continuous ranked probability
  score in points of effective capacity, and coverage at q95 and q99.
code: ml-experiments/00478-load-shift-unpadded-curtailment-cost-and-after-event-slices/
---

# Experiment 478: Part F2 of experiment 395, avoidable-curtailment cost and after-event slices

**Plan only. This page is rewritten with the results when the scoring has run.**

## Question

Part F of [experiment 395](../00395-xgboost-level-state-winter-spread-anchor-stored-forecasts/result.md)
scored stored demand forecasts against 184 unpadded, hand-labelled load-shift events. This part adds
the pieces Part F left out.

## What changed since the brief was written

- **Part F already holds the unpadded harm table.** Experiment 395 is merged, so this is a follow-up
  and not a new re-score.
- **Flagged zeros need no new rule.** Part F drops the flagged exact-zero half-hours, and
  [experiment 398](../00398-load-shift-event-rescore-without-flagged-zero-actuals/result.md) showed
  that dropping them moves no contrast. This part reads Part F's `flagged` column and drops the same
  rows.
- **NGED's event labels are incomplete.** Every table writes "rows with no labelled event", never
  "clean".

## Method

- **Rows.** Part F's per-row files under `data/ml-experiments/00395-.../work/part_f/` (the weekly
  bundle, the monthly control, the GAM stack, experiment 383's `d`, `c1`, and `c2`), plus experiment
  391's `b0`, `n1`, and `o`, tagged by the same code and the same unpadded events.
- **Corrected signs.** Events take the signed depth of Part H's corrected label table. The share of
  the depth that the median follows is recomputed per forecast, and the harm table gains the share
  followed beside the harm.
- **New slices.** Target position after the event (0 to 24 hours, 1 to 3 days, 3 to 7 days), event
  length (under 1 day, 1 to 7 days, over 7 days), and depth quartiles.
- **Avoidable-curtailment cost** at the signed-MW six for f = 0.9 (headline), 0.75, and 0.5, in
  three row sets: during events, in the 7 days after, and across all rows. Each forecast is read
  only as a difference from the plain weekly member and from the GAM stack, per Primary, with
  whole-event intervals. The MVA-only eight have no defined cost.
- **Widening.** Every forecast is scored raw. The 1.15 widening applies only to experiment 383's
  arms, so the contrasts between forecasts that differ in widening are labelled and never mixed with
  input changes.
- **Chart.** One figure per forecast of q95 coverage by lag count and event length.
- **Reviews.** A fresh Sonnet agent reviews the code before the first scored run, another checks the
  write-up against the printed report, and a third runs the prose review.
- **Compute.** CPU only, 2 cores, no fit, run under `systemd-run` with `MemoryMax=16G`.

## Contrasts

C1 to C6: for each of the bundle, the monthly control, experiment 391's `b0` and `n1`, and
experiment 383's `c1` and `c2`, the avoidable-curtailment cost minus the plain weekly member's at the
signed-MW six, f = 0.9, in the 7 days after events.
