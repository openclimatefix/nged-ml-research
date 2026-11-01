---
id: "00463"
slug: lag-trust-cross-fitted-trust-fit-seed-zero
title: >-
  Does fitting the trust weight on out-of-fold lagged heads close the gap between a learned trust
  weight and the fixed 50/50 mixture?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
date: 2026-11-01
kind: other
protocol: demand-v2
size: >-
  complex. Stored: 30 out-of-fold lagged-head networks (not kept), 5 trust networks, and 5 months
  of forecasts and learned weights in this experiment's private data folder; every earlier
  experiment is read and never written. More than one design would test the hypothesis (which head
  is the lagged head, the number of folds, whether the trust head or a constant weight is fitted),
  so each choice is fixed here before any score is read. Production serving path: not touched (this
  repository holds none). Degradation rule: not touched. Callers: the scripts import the modules of
  experiments 87, 48, 397, and 404 and copy the scoring helpers of experiment 458.
hypothesis: >-
  Experiment 404's learned trust weight (B1) loses to the fixed 50/50 mixture (B-mix) partly
  because the lagged head is fitted in sample on the same rows where the trust weight is fitted,
  while the lagless table is out of sample on those rows (mechanism M11). A trust weight fitted on
  out-of-fold lagged heads and the cross-fitted lagless table, so that both heads are out of sample
  on the rows where the trust is fitted, matches B-mix on rows with no labelled event and beats it
  inside real load-shift events.
prediction: >-
  Written before any score is read. P1. The learned lagged weight (the product of the three
  per-lag trust weights) falls from about 0.75 at 0-12 h to about 0.5 by days 6-10, as experiment
  458's fitted weights did, and swings by 0.2 or more between months. P2. Held-out scores are no
  better than B-mix: G1-trust minus B-mix on rows with no labelled event at the signed-MW six lies
  between -0.03 and +0.10, and on real-event rows it lies within the minimum detectable effect of
  zero. P3. G1-trust is better than B1 (the in-sample twin) on rows with no labelled event, by
  0.05 to 0.3 points at the signed-MW six, because the out-of-fold lagged head is less
  over-confident than the in-sample one. P4. Seed 1 is not run, because the weight swings by more
  than 0.1 between months.
decision_rule: >-
  G1 fails (M11 does not explain the gap; close learned trust in the 404 harness and keep B-mix as
  the reference) if G1-trust minus B-mix on rows with no labelled event is above +0.03 points at
  the signed-MW six, or if its CRPS on real-event rows (NRA-corrected actuals, depth factor 1.0,
  headline; metered actuals reported beside it) is not lower than B-mix's by more than the minimum
  detectable effect. G1 passes only if both hold and its 95% leave-one-month-out interval on
  real-event rows excludes zero; nothing is adopted unless G1 passes at the signed-MW six and is no
  worse than +0.03 at the MVA-only eight. The minimum detectable effect is 2.8 times the standard
  error of the monthly differences between B-mix and A0-inj on the same rows, computed before any
  G1 contrast is read, and the standard error of each contrast's own monthly differences is
  reported beside it. The weight curve alone is a diagnostic and decides nothing. Seed 1 runs only
  if the learned weight swings by less than 0.1 between months.
benchmark: >-
  B-mix, experiment 404's stored fixed 50/50 mixture of A0-inj and the cross-fitted lagless table,
  on identical rows. B1, experiment 404's stored in-sample learned trust, is the second reference
  (the in-sample twin of G1-trust).
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of effective capacity from the 11 quantiles of
  each forecast, as in experiment 404's report, on the real-data world (world -1) with leads of 0
  to 240 hours (lead days 1 to 10).
code: ml-experiments/00463-lag-trust-cross-fitted-trust-fit-seed-zero/
---

**Plan only: no score has been read.** The question is whether the learned trust weight of
experiment 404 loses to the fixed 50/50 mixture because it was fitted on unequal footing.
NGED's event labels are incomplete, so "rows with no labelled event" may hold real events that
nobody logged, and every real-event slice undercounts. Every number will be interim and unaudited.

## Design

- **Harness.** Experiment 404's, seed 0, five scored months (2025-10, 2025-12, 2026-01, 2026-03,
  2026-06). Each month's training window, injected world (seed 100), recency weights, and
  cross-fitted lagless table (`FOLDS = 6` blocks of target time) are experiment 404's, read and not
  rewritten.
- **Out-of-fold lagged heads.** For each month, 6 plain lag networks (experiment 404's A0-inj
  recipe: 3 lags, no lag dropout, 6 epochs, Adam at 2e-3, batch 4096) are each fitted on the
  training rows whose target lies outside one of the 6 blocks, and forecast the rows inside that
  block. The block of a row is the lagless table's block of its target time, so each row has an
  out-of-fold lagged head and a cross-fitted lagless head from the same fold. Each fold refits the
  per-Primary target mean and standard deviation on its fitting rows only. The input standardiser
  is experiment 404's, fitted on all training rows (it uses no target); this is a small
  approximation, stated here.
- **Trust fit.** One trust fit per month: the trust head of B1 (the three per-lag trust weights,
  combined by their product and smoothed over three half-hours) is fitted on the training rows with
  both heads frozen (the out-of-fold lagged mixture and the cross-fitted lagless mixture are
  constants), by the mixture negative log-likelihood, 6 epochs, Adam at 2e-3, batch 4096, with
  experiment 404's trust inputs and recency weights. The final forecast is the 6-component mixture
  of the two heads, exactly as in B1.
- **Test-time forecast.** The lagged head is experiment 404's stored full-data A0-inj network of
  the same month and seed, the lagless head is the table's further module, and the trust head is
  the one just fitted. Nothing from the scored month or later enters any fit.
- **Fit count.** 5 months x 6 folds = 30 lagged-head fits, plus 5 trust fits = 35 fits (seed 0
  only). The count and the time of one fit are reported to the manager before the remaining fits.
- **Arms.** B-mix (reference); B1 (stored in-sample learned trust); G1-trust. A0-inj and the
  lagless head alone are reported for absolute scores.
- **Check before the grid.** A fold-free refit of the lagged head with this script's code must
  reproduce experiment 404's stored A0-inj training loss to three decimals; the scored-month rows
  of G1-trust must have the same keys as B-mix.

## Standing practices

- **Retraining.** Monthly (one fit per scored month); the contrast is about a weight fitted on a
  training window, not about the cadence.
- **Cleaned data and the short-history cut.** Inherited from experiment 404's rows and tables
  (`substation_zero` half-hours dropped; D15 and D18 before 2025-03-01 dropped).
- **Reuse.** The lagless table, the A0-inj full-data networks, the trust inputs, the worlds, and the
  B-mix and B1 forecasts are experiment 404's; the lagless table is not refitted.
- **Compute.** One GPU process under 8 GB (`MemoryMax=8G`), `OMP_NUM_THREADS=1`, one feeder core.
  The GPU is used because the lagged heads are single large networks.
- **Past-weather caveat.** Weather at lag times is as in experiment 404 (it is available at
  forecast time in that harness).

## Scoring

- Signed-MW six and MVA-only eight separately; D15 and D18 are report-only.
- Row sets as in experiments 455 and 458: no-label rows (no labelled event in the day windows, lags
  or target), harm rows (a day window inside an event, target outside), real-event rows (target
  inside a padded event or in the 96 half-hours after one). Real-event rows are scored against
  NRA-corrected actuals at depth factor 1.0 (headline; 0.75 and 1.25 as sensitivity) and against
  metered actuals.
- Leads 0 to 240 hours; drop-largest-event and leave-one-month-out intervals; the Bonferroni count
  is 14 (G1-trust against B-mix on no-label, harm, event with NRA-corrected actuals, and event with
  metered actuals, at both groups = 8; G1-trust against B1 on the first three row sets at both
  groups = 6); MDE beside each contrast; absolute scores beside contrasts.
- Tail calibration: coverage and pinball loss at q01, q05, q95, q99. Curtailment cost does not
  apply in the 404 harness.
- The learned lagged weight is read by lead band (0-12 h, 12-36 h, 36-72 h, days 4-5, days 6-10) and
  by month, beside the stored B1 weight.
- Figures: contrasts with intervals (headline first), learned weight by lead band and month, a
  day-ahead and a 10-day forecast across a real event at D5, and the same on rows with no labelled
  event.

## Advice followed and departed from

- Follows experiment 458: keep B-mix as the reference; do not adopt a learned weight unless it
  clearly passes.
- Departs from experiment 458 only in scoring five months (no warm-up month is lost), because the
  trust is fitted inside each month's own training window.
