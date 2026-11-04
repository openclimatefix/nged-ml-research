---
id: "00549"
slug: k2-s1-blend-and-climatology-stored-check
title: >-
  Does a blend of the lag-light network R-DW with the retrained K2-S1 beat both, and did any stored
  blend with climatology help at 120 to 240 hours?
theme: lagged-power-and-error-correction
also_relevant_to:
  - demand-inputs
status: planned
kind: other
size: >-
  Simple. Part 1 re-uses experiment 547's blend code on stored quantiles (CPU, 2 cores, nothing
  fitted but the weights, nothing stored beyond blended scores in the private data folder). Part 2
  reads stored result pages and fits nothing. One design is defensible for the blend (one weight per
  group and lead band, walk-forward). The script copies the scoring code of experiments 547, 541, and
  539. No serving path or degradation rule is touched.
hypothesis: >-
  Part 1. Averaging the 11 stored quantiles of R-DW (experiment 539) and the retrained K2-S1
  (experiment 541, a different kind of network: monthly lag-free heads plus lagged heads) with a
  fixed weight per lead band forecasts better than either, because the two networks have different
  inputs and architectures and so make less correlated errors than L-1d and R-DW did in experiment
  547. Part 2. A blend with a calendar-only or climatology forecast helps the demand forecast only at
  120 to 240 h.
prediction: >-
  Part 1 (written 2026-11-04 before any blend was scored). At the eight MVA-only Primaries, the
  walk-forward blend beats both parents by at least 0.05 points of capacity (CRPS) at 24-120 h. It
  fails if it only equals R-DW. Part 2. A stored climatology or calendar-only blend helps by at
  least 0.05 at 120-240 h at the Primaries. If no stored blend shows that, we stop and say so.
decision_rule: >-
  Part 1. Supported if at the eight the blend's pooled 24-120 h CRPS is at least 0.05 below both
  R-DW and the retrained K2-S1 and the month-block interval of the difference to R-DW excludes zero.
  Not supported if the pooled difference to R-DW is above -0.02. Between the two, partly supported.
  The 0.05 is set by the cost of serving a second network. Intervals are month-block bootstrap with 4
  blocks (the scored folds; near the min-to-max range), per-fold signs, the minimum detectable effect
  beside every contrast, and a Bonferroni count of 28 (2 counted contrasts x 7 bands x 2 sets).
  Part 2. Reported as a table of stored results; no new statistics.
benchmark: >-
  R-DW (experiment 539) and the retrained K2-S1 (experiment 541, cell K2-S1-origin-trained); A-pool
  (experiment 425 arm A) and manual_heuristic as references.
row_set_id: >-
  main16 of experiment 533 (the five folds 2025-10, 2025-12, 2026-01, 2026-03, and 2026-06), scored
  from the second fold on because the first fold has no earlier fold to fit a weight on.
metric: >-
  CRPS in points of capacity at the eight MVA-only and the six signed-MW Primaries (rows with no
  labelled event), by lead band. D15 and D18 are report-only. July 2026 is not scored.
production_status: not-for-production
adopted_in: none
---
Planned. The hypothesis, prediction, and decision rule above were written before any blend was
scored.
