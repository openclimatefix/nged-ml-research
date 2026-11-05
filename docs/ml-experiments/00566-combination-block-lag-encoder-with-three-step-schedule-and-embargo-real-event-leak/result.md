---
id: "00566"
slug: combination-block-lag-encoder-with-three-step-schedule-and-embargo-real-event-leak
title: >-
  Do the three-step schedule and a 3-day embargo cut the real-event load-shift leak of the modular
  lag-encoder demand network, and how much of its accuracy gain do they keep?
theme: lagged-power-and-error-correction
also_relevant_to:
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fine-tuned weights of the final candidate arms, step-2 weights, forecasts, and
  per-event leak tables in a private folder. More than one defensible design (what the corrected-target
  refit changes when the network also reads lags, how the embargo and the three-step schedule combine).
  Spans code whose callers need a search (experiments 556, 559, 560, 510, 511, 545). Serving path and
  degradation rule: not touched (this repository holds none). Cost ceiling 6 GPU-h for the whole
  combination programme; this first part is planned at about 3 GPU-h.
hypothesis: >-
  The lag encoder of experiment 560 (variant 3, capped 24-hour lags, block dropout) lowers 24-120 h CRPS
  at the eight but leaks an injected load shift more than L-1d does. Training the mixer once on the
  whole history (the three-step schedule, experiment 559's A1), leaving the newest 3 days out of each
  fine-tune (the embargo, experiment 556), or both, lowers the leak measured on real labelled events and
  keeps most of the lag encoder's accuracy gain.
prediction: >-
  Written before any fit. P1: C4 (lag encoder plus three-step schedule plus embargo) keeps at least 70%
  of C1's gain over C0 (lag-free M0) at 24-120 h at the eight, and its real-event leak is within the
  lag-free network's (mean absorbed share 0.015 of the labelled depth in experiment 556). P1 fails if C4
  keeps under 50% of the gain, or if embargo plus three-step schedule adds more than 0.10 CRPS over C1.
  P2: the real-event leak of C1 (no protection) is larger than the lag-free network's 0.015 and C3 and C4
  cut it by at least 30%.
decision_rule: >-
  The primary leak measure is the mean absorbed share over the real events (the largest event dropped
  beside it), read at the flag level fixed here before any number is read: a mean without the largest
  event above 0.03 of the labelled depth (twice the lag-free network's 0.015) flags the arm. The injected
  6.0-point limit is a flag, not a gate. Contrasts are paired by row; intervals are 5-block month
  bootstraps (near the minimum to maximum of the folds) with per-fold signs and the minimum detectable
  effect beside each. Counted contrasts: C1-C0, C2-C1, C3-C1, C4-C1 at 24-120 h at the eight and at
  the six, so 8 tests (Bonferroni count 8). The threshold for porting a protection is the leak cut of
  P2 at a CRPS cost of at most 0.05 at 24-120 h at the eight. Wording is "interval excludes zero" or
  "not shown".
benchmark: >-
  C0 (lag-free M0 fitted at the same coarse cadence from experiment 554's pretrained weights); stored
  daily-refit M0 (experiment 545) and LAG-v3 (experiment 560) as report-only context; the lag-free
  network's real-event leak from experiment 556.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity by the seven lead bands of experiment 521 and by 24-120 h, at the
  eight MVA-only Primaries and the six signed-MW Primaries (D15 and D18 report-only); absorbed share of a
  real labelled event's depth; q01, q05, q95, q99 coverage and pinball loss.
---

# Combination block, part 1: real-event leak harness and arms C0 to C4

**This page is a plan; the results follow when the experiment is scored.** Every number is interim and
unaudited until the reviews have run.

## Question

Experiment 560 found that a lag encoder lets the modular demand network use power lags at least 24
hours old and score better than the lag-free network at the eight, but it leaks an injected load shift
(6.1 to 8.1 points of capacity against L-1d's 4.49). Experiment 559 found that the three-step schedule
cuts the lag-free network's injected leak at a cost in accuracy, and experiment 556 found that a
3-day embargo removes the leak of a short step. This part asks which protection keeps the lag encoder's
gain while lowering the leak measured on real events.

## Method

- **Real-event leak harness (ported from experiment 556).** For every labelled event, the cutoff is the
  first daily cutoff at least 1 day after the event's end (experiment 556 used 3 days, at which a 3-day
  embargo cannot drop any of the event's rows, so 1 day is the test of the embargo); C1 is also run at
  3 days, for comparison with experiment 556. C0 is run at 1 day as the lag-free reference. The network
  is fine-tuned once on the metered targets and once on targets
  with that event's labelled step removed (experiment 435's correction at depth factor 1.0); the lag
  inputs stay as observed. The absorbed share is the sign of the event times the difference of the two
  median forecasts over the 7 days after the cutoff at the event's Primary, divided by the labelled
  depth. The events are the 184 rows of experiment 400's `events_corrected.parquet` that have a cutoff
  inside the five scored months; experiment 556 counted 74 (month, event) pairs, and this harness uses
  the same pairs.
- **Arms (all with block dropout of lags).** C0: lag-free M0. C1: LAG-v3. C2: C1 with the three-step
  schedule. C3: C1 with a 3-day embargo. C4: C1 with both. Coarse cadence: one fit per fold at the month's
  first cutoff, scored over the whole month. Two seeds, five folds of 511 (2025-10, 2025-12, 2026-01,
  2026-03, 2026-06); July 2026 unused.
- **Reuse.** C1's fits are experiment 560's day-0 fine-tunes. C0 is fitted here from experiment 554's
  pretrained weights with a fresh mixer.
- **Scoring.** Experiment 560's rows and scorer, with the lags removed and aged by 1, 2, and 3 extra days
  (inference only) for every lag arm. Leave-D8-out beside every contrast.
- **Assertions.** Every weather input comes from the run issued at or before the forecast origin, and every
  lag is at least 24 hours older than the origin; both are asserted row by row and one example row is
  printed (experiment 560's assertions, unchanged). Training events end before the refit cut minus
  3.5 hours.
- **Not in this part.** The T factor (corrected target with abnormal lags kept, synthetic ARAs) and arms
  C5 to C10 follow in a later part; they are not planned here.

## Standing practices

Cleaned data everywhere; D15 and D18 rows before 2025-03-01 cut; saved weights for the final arms; stored
predictions reused as controls; tails and cost reported; examples plotted; the unlogged-event caveat
stated (NGED's logs may miss events, so clean-lag rows may hold events no label covers and the real-event
counts are lower bounds).
