---
id: "00547"
slug: lag-light-and-lag-free-blend-by-lead-band-stored-forecasts
title: >-
  Does a fixed-weight blend of the lag-free network L-1d and the lag-light network R-DW, with one
  weight per lead band, forecast better than either network alone?
theme: lagged-power-and-error-correction
also_relevant_to:
  - demand-inputs
status: planned
kind: other
size: >-
  Simple to run, scored as anything else. Nothing is fitted and nothing new is stored beyond the
  blended quantiles in the private data folder (stored rows only, CPU). One design is defensible
  (one weight per group and lead band, fitted walk-forward by fold), and the script wraps the
  scoring code of experiments 470, 521 and 539, whose callers cannot be named without searching.
  No serving path or degradation rule is touched.
hypothesis: >-
  Averaging the 11 stored quantiles of L-1d (a daily-refit lag-free network, experiment 511) and
  R-DW (the same network plus day-ago and week-ago power, experiment 539) with a fixed weight per
  lead band forecasts better than either parent, because the two networks make partly different
  errors and the best weight on the lag-light network differs by lead.
prediction: >-
  At the eight MVA-only Primaries, the walk-forward blend beats both parents by at least 0.05 points
  of capacity (CRPS) in each of the bands 24-48 h, 48-72 h, and 72-120 h. It fails if it only
  equals the better parent (R-DW). Written 2026-11-04 before any blend was scored.
decision_rule: >-
  Supported if, at the eight, the blend's CRPS is at least 0.05 below both parents in each of the
  three bands 24-48 h, 48-72 h, and 72-120 h (point estimates), and the month-block interval of the
  difference to R-DW excludes zero in the pooled 24-120 h rows. Not supported if the pooled 24-120 h
  difference to R-DW is above -0.02 (equal to the better parent within the weights' noise). Between
  the two, partly supported. Stop and report if the stored predictions cannot be blended. The
  threshold is zero in production cost only if the blend replaces R-DW; running two networks is a
  second model to serve, so the 0.05 is what decides the verdict word. Intervals are month-block
  bootstrap with 4 blocks (the scored folds; near the min-to-max range), with per-fold signs, the
  minimum detectable effect beside every contrast, and a Bonferroni count of 28 (2 counted
  contrasts x 7 bands x 2 sets).
benchmark: >-
  L-1d (experiment 511) and R-DW (experiment 539), both on the rows of experiment 533; A-pool
  (experiment 425 arm A) and manual_heuristic as references.
row_set_id: >-
  main16 of experiment 533 (1,142,520 rows on the five folds 2025-10, 2025-12, 2026-01, 2026-03,
  and 2026-06), scored from the second fold (2025-12) on because the first fold has no earlier fold
  to fit a weight on.
metric: >-
  CRPS in points of capacity at the eight MVA-only and the six signed-MW Primaries (rows with no
  labelled event), by lead band. D15 and D18 are report-only. July 2026 is not scored.
---
**Plan.** Experiment 539's R-DW closes most of the gap between L-1d and A-pool. This experiment asks
whether a blend of L-1d and R-DW does better than R-DW alone, using the stored forecasts only.
Experiment 528's learned gate collapsed to a calendar-only rule, so the weights here are fixed
numbers, one per lead band and Primary group.

- Blend: quantile averaging on the stored 11 quantiles, `w * R-DW + (1 - w) * L-1d`, because
  mixture parameters are not stored. Quantile averaging keeps the median between the two parents.
  The blend is widened by the headline factor 1.15 (the same as both parents) before scoring.
- Weights: for each scored fold, `w` is the grid value in 0, 0.1, ..., 1 that minimises mean CRPS
  of the widened blend on the earlier folds, per group (eight, six with no labelled event, report)
  and per lead band of experiment 521. The first fold supplies weights only, so the scored folds
  are 2025-12, 2026-01, 2026-03, and 2026-06. The folds are not contiguous months, so "earlier"
  means earlier calendar order.
- Weather rule: no weather input is read here. Both parents used the run issued at or before the
  forecast origin (asserted row by row in experiment 539's checks). The script asserts that every
  weight was fitted on folds before the scored fold and prints one example row.
- Not computed: the avoidable-curtailment cost (stored-row re-score only).
