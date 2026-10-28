---
id: "00374"
slug: mva-fold-stage-two-at-eight-primaries
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
size: >-
  medium. Stored: re-run forecasts, fits, and small result files in the private folder of this
  experiment only; experiment 152's folder is read and never written. Production serving path: not
  touched. Degradation rule: not touched. Designs: more than one would test the hypothesis (a new
  tuning period for stage 2 from experiment 152's engine, or donor tuning rows from other
  Primaries); the engine re-run is chosen because the maintainer asked for the full comparison.
  Callers: reuses the engine of experiment 152 and the stage-2 code of experiment 252, whose callers
  are all in those folders; this folder's runner redirects the engine's output paths.
hypothesis: >-
  At the eight Primaries that report only apparent power, the fit that the rule of experiment 366
  chooses (the fit through the meter at D12, the naive fit at the other seven), followed by
  experiment 252's stage-2 corrections, scores no worse than experiment 252's arm A1 over all
  half-hours, pooled over the eight Primaries.
prediction: >-
  Written after the raw-fit scores of experiment 372 were known (chosen minus BG2 over all
  half-hours: D8 +3.36, D12 +0.76, D22 +0.42, the other five within 0.1), and before any stage-2
  score of this experiment is read. P1. Chosen with stage 2 minus A1, over all half-hours pooled
  over the eight, lies between -0.30 and +0.30 points of effective capacity (experiment 353 found
  B within 0.5 of A1 at D8, and A1 differs from the chosen fit in having a T-all design, a solar
  clip, and a 60% clip). P2. At D12, the fit through the meter with stage 2 minus the naive fit with
  stage 2 is below zero on sunny daylight half-hours, with an interval that may include zero. P3.
  Stage 2 narrows the gap to BG2 at D22 and D12 relative to experiment 372's raw fits (D22 +0.42,
  D12 +0.76), and leaves D8's gap above +2.0. P4. The three references rescored here reproduce
  experiment 252's stored scores, and the re-run raw fits reproduce experiment 372's scores within
  0.05 points at each Primary (the control gate).
decision_rule: >-
  P1 bears the verdict, by a paired month-resampled bootstrap with 20,000 resamples and a 95%
  interval, as in experiment 353. `supported` when the interval's upper end is below +0.10 points
  (non-inferiority: the added cost of the chosen fit's choice is zero) and the control gate passes.
  `not supported` when the upper end is at or above +0.10 points, or when the interval lies wholly
  above zero. `inconclusive` when the control gate fails. P2 to P4 are descriptive. Every result
  holds for reactive power zero, and the rule fails at power factor 0.97 (experiments 358 and 366).
benchmark: >-
  Experiment 252's arm A1; BG2, climatology, and manual_heuristic as stored by experiment 87; the raw
  fits of experiment 372
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity,
  scored year 2025-07-02 to 2026-06-30, leads 0.5 to 168 hours, rows outside the hand-filtered
  events; slices sunny daylight, winter nights, cold nights; tail coverage at q01, q05, q95, q99
verdict: ""
production_status: not-for-production
adopted_in: none
date: 2026-10-28
---

# MVA fold: stage-two corrections at the eight MVA-only Primaries

**Planned. Written before any stage-2 score of this experiment is read.**

This is the full form of step 4 of the MVA sequence, which the maintainer asked for after
[experiment 372](../00372-mva-fold-rule-chosen-fit-at-all-primaries/result.md) scored the raw fits.

## Plan

- **Re-run.** Run experiment 152's engine for the eight MVA-only Primaries (arms N, A, and F_fold)
  for the tuning, replicate, and scored periods, into this experiment's own work folder. The
  stored ensemble inputs of experiment 152 are copied, not shared, so no file in experiment 152's
  folder changes. Use the "spawn" start method, 4 workers of 2 threads each, and a 16 GB memory cap.
- **Stage 2.** Apply experiment 252's stage-2 corrections to each arm as experiment 353 did at D8.
  Primaries with no usable tuning month borrow the spread of the donor Primaries, as in
  experiment 353.
- **Report.** Score the chosen arm with stage 2 against A1, BG2, climatology, and the manual
  heuristic, per Primary and pooled, with tail coverage and the median-minus-mean check.
