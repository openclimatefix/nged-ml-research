---
id: "00546"
slug: teacher-student-distillation-into-lag-free-demand-network
title: >-
  Can a demand network that sees no recent power learn what a network with day-ago and week-ago
  power knows, by being trained on that network's forecasts?
theme: lagged-power-and-error-correction
also_relevant_to:
  - demand-inputs
status: planned
kind: other
size: >-
  Anything else, so a code review before the first fit. Stored: network weights and forecasts for 5
  folds x 2 seeds of daily refits (private data folder). More than one design could test the idea
  (loss form, teacher weight), and the fit script wraps experiment 511's and 539's code, whose
  callers cannot be named without searching. No serving path or degradation rule is touched.
hypothesis: >-
  A lag-free network (the student) fine-tuned against the forecast of a teacher that sees the
  day-ago and week-ago lags (experiment 539's R-DW), in addition to the real target, scores closer to
  A-pool than the lag-free network L-1d does, because the teacher's forecast carries information about
  the weekly pattern that the weather and calendar inputs can recover but the real target alone
  teaches slowly. The student reads no lag at any time, so it adds no load-shift leak route.
prediction: >-
  At the eight, 24 to 120 h, the student closes at least 30% of L-1d's gap to A-pool (L-1d minus
  A-pool is +0.288, +0.254, +0.200 points of capacity at 24-48 h, 48-72 h, 72-120 h), with the
  injected-shift leak no higher than 4.49 + 0.3 points of capacity. It fails if it closes under 15%.
  Written 2026-11-04 before any student fit.
decision_rule: >-
  Supported if the share of the gap closed is at least 30% in each of the three bands (point
  estimates) and the leak is at most 4.79. Not supported if the share closed is under 15% in any
  band. Between 15% and 30%, or with leak over 4.79, the verdict is partly supported. Stop after the
  first 2 folds if the student closes under 15%. Intervals are month-block bootstrap with 5 blocks
  (near min-to-max), with per-fold signs, the minimum detectable effect, and a Bonferroni count of
  28 (2 counted contrasts x 7 bands x 2 sets). The threshold is zero in production cost (the student
  is an ordinary lag-free network), so any gain whose interval excludes zero is worth a port, but the
  pre-registered 30% is what decides the verdict word.
benchmark: >-
  L-1d (experiment 511), A-pool (experiment 425 arm A), R-DW (experiment 539, the teacher), K2-S1
  rebuilt on origin-time weather (experiment 541) when it is available, manual_heuristic.
row_set_id: >-
  main16 of experiment 533 (1,142,520 rows on the five folds 2025-10, 2025-12, 2026-01, 2026-03,
  and 2026-06), the rows scored in experiments 537 to 539.
metric: >-
  CRPS in points of capacity at the eight MVA-only and the six signed-MW Primaries, by lead band,
  and the injected load-shift leak in points of capacity. D15 and D18 are report-only.
advice: []
production_status: not-for-production
adopted_in: none
---
**Planned. No student has been fitted and no number exists yet.**
