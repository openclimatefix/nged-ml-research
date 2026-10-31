---
id: "00435"
slug: stage-0-recalibration-table-for-untrusted-inputs-and-tails
title: >-
  Does a stratified recalibration table, fitted on stored forecasts, keep the demand bands honest
  when inputs are untrusted and fix the tail coverage on clean rows?
theme: calibration-and-uncertainty
also_relevant_to:
  - lagged-power-and-error-correction
status: planned
kind: other
date: 2026-10-31
size: >-
  medium. Stored: tables and figures under
  data/ml-experiments/00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails/ only; no
  forecast model is fitted. Serving path: not touched. Degradation rule: not touched. Designs: the
  stratification and the pass rules are fixed by the maintainer-accepted brief, so one design.
  Callers: this folder's scripts only, which read stored forecasts of experiments 410, 422, 425,
  428, 429, and 430 and call no reviewed module beyond `studies.*`. One Sonnet plan review, one
  Sonnet code review before the first run, one Sonnet results review, and two Sonnet write-up
  reviews.
hypothesis: >-
  The target is the normal running arrangement (NRA). If a factor on the distance of each quantile
  from the median is fitted per metering group, lead band, tail side, and input-condition class on
  the months before the scored month, then coverage on held-out months in every harm class lies
  within the accepted tolerance of the clean-row coverage, without worsening CRPS on clean rows.
prediction: >-
  Written before any score is read. P1: for R-inj of experiment 428 the fitted factors are near 1
  (within 0.05) and every class passes. P2: for the untrained arms (R of experiment 428, DROP of
  experiment 410) the factors exceed 1 on the lower tail of harm rows (q05 coverage 0.058 raw).
  P3: clean-row q95 coverage after the table lies in 0.93 to 0.97 for the six signed-MW series,
  and the per-Primary variant lifts D14 from about 0.81. P4: wind q99 coverage rises from about
  0.972 to 0.98 or above with the wind-speed-band class. P5: three seeds pooled widen the bands on
  harm rows by less than 5% against clean rows, so seed pooling alone does not fix them.
decision_rule: >-
  A class passes if, on held-out months, q95 coverage lies within 0.03 of the clean-row q95
  coverage and q99 within 0.015, and on clean rows q95 lies in 0.93 to 0.97, q99 in 0.98 to 0.995,
  q05 in 0.03 to 0.07, and q01 in 0.005 to 0.02, with CRPS at most 0.03 points and tail pinball at
  most 0.02 points worse than the 1.15 widening. Real event rows are scored against NRA-corrected
  actuals (headline) and metered actuals (report), with and without the largest event. If the
  table passes every class on R-inj, it is the production recommendation and Brief B runs only U3
  and the experiment 428 iteration 1 merge.
benchmark: >-
  The stored forecasts raw, and with the 1.15 widening about the median.
row_set_id: >-
  The row set of each source experiment, unchanged.
metric: >-
  Coverage, pinball loss, and CRPS (points of effective capacity) at q01, q05, q95, and q99, per
  metering group, lead band, and condition class.
---

# Stage 0: a recalibration table for untrusted inputs and tails

**Plan only; no score has been read.** The steps follow the Stage 0 brief of 2026-10-31.

- Step 1: build NRA-corrected actuals for the scored months of experiment 428's four folds and
  2025-12, with the 0.75 and 1.25 depth variants and the count of dropped events.
- Step 2: tail-coverage diagnostics on stored forecasts of experiments 425, 429, and 430.
- Step 3: fit table P once the cross-fit table of experiment 404 is frozen (commit 63d3799f) and
  experiment 428 seeds 1 and 2 are scored; apply it to the stored forecasts of 410, 422, 425, 428,
  429, and 430.
- Step 4: pool the three seeds of experiment 428 and report the band-width ratio and coverage on
  harm rows against clean rows.
- Compute: no fits, one CPU core, `nice 19`, under 16 GB of memory.
