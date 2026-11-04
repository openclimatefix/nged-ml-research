---
id: "00546"
slug: teacher-student-distillation-into-lag-free-demand-network
title: >-
  Can a demand network that sees no recent power learn what a network with day-ago and week-ago
  power knows, by being trained on that network's forecasts?
theme: lagged-power-and-error-correction
also_relevant_to:
  - demand-inputs
status: finished
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
verdict: refuted
best_arm: >-
  None. The student scored 0.013 to 0.028 points of capacity worse than L-1d at 24 to 120 h at the
  eight, so it closed a negative share of L-1d's gap to A-pool.
headline: >-
  Training a lag-free network on the forecasts of a network that sees the day-ago and week-ago power
  did not help: on the first 2 folds the student was 0.013 to 0.028 points of capacity worse than
  the lag-free network L-1d at 24 to 120 h (eight MVA-only Primaries), against a pre-registered
  requirement of closing at least 15% of the 0.26 to 0.35 gap to A-pool. The stop rule ended the
  experiment after 2 of 5 folds. Interim and unaudited.
lessons: >-
  The information the lags carry is not in the weather and calendar inputs, so a student that
  cannot see recent power cannot recover it from a teacher's forecasts. The loss changed the
  student's training target but gave it no new input.
take_home_message: >-
  A demand network that cannot see recent power learned nothing useful from a teacher that can:
  it scored slightly worse than the same network trained on the real target alone, and stayed 0.26
  to 0.38 points of capacity behind A-pool at 1 to 5 days ahead. The day-ago and week-ago power
  must therefore be an input at forecast time, or the gap stays. The next step is to choose between
  accepting the injected load-shift leak that those lags bring and a robust same-slot summary of the
  recent weeks.
advice:
  - >-
    Do not distil a lag-enabled teacher into a lag-free demand network at 24 to 120 h: the student
    was 0.013 to 0.028 points worse than L-1d on 2 folds (figure 1).
  - >-
    To close the lag-free gap at the eight, give the network a lag input, such as a robust
    same-slot summary, because training signals alone did not (figure 1).
production_status: not-for-production
adopted_in: none
---
**A demand network that sees no recent power did not learn what a network with day-ago and
week-ago power knows, even when trained on that network's forecasts, so the lags must stay an input
at forecast time.** We asked whether a lag-free "student" network could be taught by a "teacher"
network that reads the power from the same half-hour a day ago and a week ago. The student was
fine-tuned on a loss that gave half its weight to the real target and half to draws from the
teacher's forecast distribution. The teacher is R-DW of experiment 539, which closed 76% to 83% of
the gap between the lag-free network L-1d and A-pool. The student follows
[experiment 511](../00511-lagless-network-refit-cadence-half-life-and-mixed-pretraining/result.md)'s
L-1d recipe, with the same rows and the same pretrained weights. The idea is knowledge distillation
([Hinton et al., 2015](https://doi.org/10.48550/arXiv.1503.02531)).

Verdict: refuted. At the eight MVA-only Primaries the student scored 0.028, 0.013, and 0.026 points
of capacity (CRPS) worse than L-1d at 24-48 h, 48-72 h, and 72-120 h, so it closed a negative share
(-8%, -5%, and -10%) of L-1d's gap to A-pool. The pre-registered stop rule (under 15% on the first
2 folds) ended the experiment after folds 2025-10 and 2025-12. The other 3 folds, the injected-shift
leak, and the rebuilt K2-S1 comparison were not run. Every number is interim and unaudited.

![CRPS minus A-pool at the eight, by lead band, for L-1d, the student, and the teacher](figure_1_gap_to_a_pool.svg)

## Results (first 2 folds, the eight)

- Student minus L-1d: +0.028 (24-48 h), +0.013 (48-72 h), +0.026 (72-120 h), +0.012 (120-240 h).
  Intervals are month-block bootstrap with 2 blocks (the range of the 2 folds): 0.017 to 0.039,
  0.003 to 0.023, 0.011 to 0.041, and -0.001 to 0.026. Both folds are positive, except at 120-240 h
  (1 of 2). The minimum detectable effect is 0.21, 0.21, 0.15, and 0.18, so the differences are
  far below what the experiment could resolve; they show no gain, not a loss.
- Absolute CRPS at 24-48 h: student 4.552, L-1d 4.524, teacher 4.257, A-pool 4.177. At 72-120 h:
  4.753, 4.727, 4.581, 4.464.
- At the six (signed MW) the student is also 0.027 to 0.052 worse than L-1d at 24 to 240 h.
- The Bonferroni count is 28 for the counted contrasts of this scoring script. No interval here
  was widened to it, and the claim does not rest on one.

## Checks and caveats

- Weather input for every row comes from the run issued at or before the forecast origin, asserted
  row by row for 62 test origins and all training origins. Example: target 2025-10-03 07:00 UTC,
  origin 2025-10-01 09:00 UTC, weather run 2025-10-01 00:00 UTC. The teacher's lags were observed at
  or before origin minus 3.5 h, also asserted per row. The student has no lag column.
- The teacher fitted on the same rows it labels, so its forecast on them is sharper than on new
  rows. The 0.5 weight on the real target limits the effect. A student cannot beat an honest teacher
  and did not match it.
- Median-minus-mean forecast, tail scores, and the leak were not computed because the experiment
  stopped. No audit trigger was tested beyond the prediction itself: a refuted prediction has one
  untested explanation, that the lag information is not recoverable from the weather and calendar
  inputs. One seed pair, one teacher weight (0.5), and one loss form were tried.
- Cost: about 0.15 GPU-h for the 2 folds. Code review: one Sonnet review, all five findings
  triaged (four applied, one a design note).

## What the next experiment should answer

Whether a robust same-slot summary (the median of the same half-hour over the last 4 weeks) keeps
the lags' gain with a smaller injected load-shift leak than the 5.68 points of the day-ago and
week-ago lags.
