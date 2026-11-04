---
id: "00556"
slug: lag-free-finetune-absorption-of-load-shift-events-and-unlearn-time
title: >-
  How much of a load-shift event does the lag-free demand network absorb through its fine-tune, and
  how fast does it unlearn the event?
theme: retraining-and-training-history
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted control weights and per-cell shift tables under this experiment's private
  data folder only. Serving path: not touched. Degradation rule: not touched. Designs: more than
  one defensible (how a step is injected, how a mirrored recipient is defined, what the
  residual-clipped weight is measured against), fixed here before any shift is read. Callers: the
  folder imports experiments 428, 510, and 511. One Sonnet code review before the first run, one
  Sonnet results review, two Sonnet write-up reviews.
hypothesis: >-
  When a load-shift event lies in the rows the lag-free demand network is fine-tuned on, the
  network absorbs part of the event into its median forecast, and the absorbed share grows with the
  event length. A rule inside the scheduled fine-tune (an embargo on the newest 3 days of targets,
  or weights clipped by a label-free residual rule) cuts the absorbed share without slowing
  adaptation to a genuine lasting level change or worsening the score.
prediction: >-
  Written before any shift is read. P1: the leak share at 10% depth rises with the step length L
  from at most 0.05 at L = 1 day to at least 0.15 at L = 7 days for F7d, and F1d leaks more than F7d
  at the same L. P2: at the daily cadence the leak falls below 10% of its first value within 7 days
  of the step's end, at the 7-day cadence within 14 days. P3: F1d adapts to 90% of a lasting
  10%-of-capacity step within 14 days and F7d within 28 days. P4: F3d-E3 cuts the leak at L of 3
  days or less by at least 70% and delays genuine adaptation by about 3 days, at a CRPS cost of at
  most +0.05. P5: F3d-R cuts the leak at L of 3 days or less by at least 30% at a CRPS cost of at
  most +0.05, and does not cut it at L of 7 days or more. P6: the real-event absorption share is
  under 0.10 of the labelled depth.
decision_rule: >-
  A control is worth porting if it cuts the leak at L of 3 days or less by at least 30% against
  the arm it modifies, with a CRPS cost of at most +0.05 and q99 pinball loss within +0.02, and the
  delay in genuine adaptation reported beside it. A control that passes the leak bar but delays
  adaptation by more than 7 days is reported as "more robust, slower to adapt". Stop rule: if
  F1d's leak at 10% depth and L of 3 days or less is under 1.0 point of capacity in the first four
  lead bands, in both groups, fine-tune absorption is not material and F3d-E3 and F3d-R are not
  run. Bonferroni count k = 24.
benchmark: >-
  L-Ww (experiment 510), L-3d and L-1d (experiment 511) as F7d, F3d, F1d; K2-S1 for the gap to the
  lagged network.
row_set_id: experiment_470_intersection_v1
metric: >-
  Sign-adjusted median shift in points of capacity by lead band, days to 50% and 90% adaptation,
  CRPS in points of capacity, q01, q05, q95, and q99 coverage and pinball loss.
---

# Fine-tune absorption of load-shift events (planned)

**Plan only; the write-up replaces this page when the run finishes.** Design choices fixed before
any shift is read:

- **Injection.** A step is added to the fine-tuning targets of the 16 Primaries together (as in
  experiment 510), in the L days before the refit origin, at 10% of capacity (and 3 standard
  deviations of the 7-day mean for L = 7 only). The mirrored sensitivity gives odd-numbered
  Primaries the opposite sign.
- **Origins.** Day 14 (week 3) and day 7 (week 2) of each of the five folds; adaptation starts at
  day 0.
- **Controls.** F3d-E3 leaves the newest 3 days of targets out. F3d-R multiplies each row's weight by
  min(1, 3 / |z|), where z is the row's residual against the plain F3d median three days earlier
  (under the same injection) in units of the Primary's residual MAD over the last 90 days.
- **Compute.** About 4 GPU-hours, approved by the manager.
