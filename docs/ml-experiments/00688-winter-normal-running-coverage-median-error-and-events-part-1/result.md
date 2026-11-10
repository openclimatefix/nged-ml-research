---
id: "00688"
slug: winter-normal-running-coverage-median-error-and-events-part-1
title: >-
  Do the 90% bands of the normal-running Primaries miss in winter because of median error and load
  transfers, or because the bands are too narrow?
theme: calibration-and-uncertainty
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
date: 2026-11-10
size: >-
  medium. Stored: tables and figures only; no forecasting model is fitted and no prediction table
  is written (the diagnostics re-score the stored normal-running quantiles of experiment 683's arm
  2). Two defensible designs exist for the oracle re-centring (one shift per series and origin
  month, or one per series, origin month, and lead band), so the plan fixes the first before any
  score is read and reports the second as a sensitivity. Callers: this folder's scripts only, which
  import `studies.*`, experiment 668's scoring library, and experiment 683's calibration module. No
  contract, serving path, or degradation rule is touched. One Sonnet code review before the scoring
  run, one Sonnet review that re-derives the headline numbers, one Sonnet prose review; no separate
  plan review, because the brief fixes the question and the stop rule.
hypothesis: >-
  The winter shortfall of the normal-running Primaries' 90% bands (q05-to-q95 coverage 80% in
  2025-12 and 2026-01 after experiment 683's walk-forward scale) comes from a month-long error in
  the median forecast and from half-hours inside load transfers, not from bands that are too
  narrow for the errors the median makes.
prediction: >-
  Written before any score is read. Scored rows are the normal-running output of the 16 Primaries,
  origin months 2025-12 and 2026-01 (headline) and 2026-02 to 2026-06 (reported). P1 (the
  pre-registered prediction). After step 1 (drop the half-hours inside verified and labelled load
  transfers) and step 2 (oracle re-centring) together, the pooled q05-to-q95 coverage of
  experiment 683's arm 2 exceeds 0.86 in both 2025-12 and 2026-01. P2. Step 2 explains more of the
  shortfall (0.90 minus arm 2's coverage) than step 1 in both months, because the events cover 2
  of 16 Primaries for 16 days of the verified transfer plus short labelled events. P3. At least
  two series stay below 0.86 in 2025-12 or 2026-01 after both steps, because a one-month shift
  cannot remove a within-month drift. P4. After both steps, the half-width of arm 2's band is
  within 25% of the empirical 90% half-width of the re-centred residuals in every lead band for
  the pooled Primaries (ratio between 0.75 and 1.25).
decision_rule: >-
  Fixed before any score is read. Stop rule (the brief's): if the pooled coverage after steps 1 and
  2 stays at or below 0.86 in either 2025-12 or 2026-01, the width is wrong beyond bias and events,
  the write-up says so plainly, and a learned per-lead width head at Primaries 11, 13, 16, and 18
  (about 4 GPU-hours) is justified as the next experiment. Otherwise the finding is "median error
  and events, not width" and that experiment is dropped. Step 1 mask (valid time of the scored
  half-hour inside any span, at the Primary the span names): the 34 labelled events of experiment
  580's label table whose labelled span overlaps 2025-12 to 2026-06, and the 4 verified transfers
  of experiment 580 in that period (the D5 and D16 transfer of 2025-12-29 to 2026-01-13, and the
  May 2026 transfers among D10, D11, D12, and D16), at both Primaries of each. Step 2: for each
  series and origin month, the median over the month's unmasked scored rows of the observed power
  minus the stored median is added to all 11 quantiles of arm 2 (rows from every lead band share
  one shift). The shift is computed on the same rows it is scored on, so it is a diagnostic and
  never a forecast. Fraction explained: (coverage after the step minus arm 2's coverage) divided by
  (0.90 minus arm 2's coverage), for each step alone and for both together, in both orders of
  application, with the order effect stated. Series below 0.86: named per month. Intervals are
  day-block bootstraps over forecast-origin days (2000 resamples, the picks shared across stages),
  95%, as in experiment 683. Sensitivities, reported and never deciding: (a) the shift per series,
  origin month, and lead band; (b) the mask widened to the fitted spans of experiment 580 and to
  the candidate events of experiment 685 before 2026-07; (c) the final-test window 2026-07 to
  2026-10 is neither scored nor used. Porting threshold: not applicable, because nothing here is
  ported.
benchmark: >-
  Arm 2 of experiment 683 (the walk-forward scale) on the same rows, and the stored normal-running
  quantiles (arm 0).
row_set_id: backtest_year_2025_11_to_2026_10
metric: >-
  q05-to-q95 coverage and the shares of rows below q05 and above q95, per origin month and series,
  with day-block bootstrap 95% intervals; the ratio of the band half-width to the empirical 90%
  half-width of the residuals, per series and lead band.
---

*Planned: no score has been read. The page is replaced by the write-up when the scoring has run.*

**Plan.** Experiment 683 widened the stored bands by a scale fitted on earlier months. That brought
four of five outputs close to the 90% coverage the q05 and q95 quantiles promise, and left the
normal-running Primaries output at 80% in the two winter months. This experiment asks why, by
re-scoring the stored quantiles of experiment 683's arm 2 under two diagnostic changes and by
comparing the band width with the spread of the errors.

- **Standing practices followed.** Stored forecasts reused (nothing is fitted); cleaned data
  (experiment 683's scored rows, which come from the cleaned observed power); D15 and D18 are
  scored from the files as stored, because the files fix their own training history and the
  diagnostic compares the same rows as experiment 683; the final-test window is untouched.
- **Earlier advice followed.** The advice of experiment 683 to test a winter median correction and
  a learned width head: this experiment decides whether the width head is needed.
- **Reviews.** One fresh Sonnet code review before the scoring run; one fresh Sonnet review that
  re-derives the headline numbers; one fresh Sonnet prose review.
