---
id: "00599"
slug: nra-refit-embargo-and-down-weighting-of-the-newest-rows-against-load-shift-absorption
title: >-
  Can a longer embargo, or down-weighting the newest rows, lower the share of an injected load-shift
  that the daily fine-tune of the A1 demand neural network absorbs into its median forecast?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: refits of three arms on event-carrying power, forecasts and score tables in the
  private data folder only. One design is defensible (experiment 595's harness, unchanged, with one
  training-side setting changed per arm), so the plan fixes the arms, the quantity, and the decision
  rule. Spans code of experiments 574, 587, 591, and 595, which this experiment wraps without
  editing. Serving path and degradation rule: not touched (this repository holds none). Compute:
  about 0.6 GPU-h for three arms (experiment 595's A1 run took about 0.2 GPU-h), under the 1.5 GPU-h
  ceiling.
hypothesis: >-
  Lengthening the embargo of the daily fine-tune from 3 to 7 days (E1), or halving the weight of
  the newest 7 days of fine-tune rows (E2w), or both (E3), lowers the total share of an injected
  load-shift's depth that the median forecast carries 7 days after the shift ends, because the
  newest rows hold the shift's latest days and the refit then sees fewer of them.
prediction: >-
  Written before any score. P1: at least one of E1, E2w, and E3 lowers the total share (q50, day-14
  refit, 7 days after the end, mean over the 12 injected designs, seeds 0 and 1) by at least 0.04
  against E0, which is A1 as stored by experiment 595 (0.160). P2: that arm's event-free CRPS
  (arm fitted on the no-event power, all rows, days 1 to 10) rises by no more than the minimum
  detectable effect, separately for the eight MVA Primaries and the six MW Primaries. P3: CRPS at
  leads 0 to 24 h rises more than CRPS at leads 24 h to 10 days for the 7-day embargo (E1 and E3).
  P1 fails if no arm lowers the share by 0.04.
decision_rule: >-
  All shares are descriptive and the verdict reads P1 and P2. Counted tests: 3 (E1, E2w, and E3
  minus E0, total share at q50, day-14 refit, 7 days after the end, paired by design); a difference
  counts only if its Bonferroni interval (3 tests) excludes zero. Intervals resample designs with
  2000 draws (both Primaries of one transfer in one block), and each printed half-width is the
  minimum detectable effect (MDE) of the share. For event-free CRPS the MDE is the half-width of the
  95% interval of the arm-minus-E0 difference, resampling Primary-by-month cells within a metering
  group; P2 holds if the point difference is at most that MDE. D15 and D18 are report-only. The
  experiment adopts nothing.
benchmark: >-
  E0, the A1 arm of experiment 595 (3-day embargo, equal-age weights as in 595, total share 0.160
  with interval 0.099 to 0.221 at 7 days after the end), recomputed from the stored forecasts.
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the event's depth that the median forecast moves toward the event, by phase (before,
  first 4 days, day 4 to the end, 7 days after the end), for the total effect and for the refit
  alone, as in experiments 587 and 595; event-free CRPS (mean pinball loss over the 11 quantiles)
  overall and at leads 0 to 24 h.
---

**Take-home message (to be written after scoring).** Planned.

**The question.** In [experiment 595](../00595-nra-target-side-absorption-for-p5-lag-encoder-with-synthetic-shifts/result.md),
about half of what the A1 and C1 networks absorb of a load-shift in their training power (0.08 to
0.09 of the depth) came from the refit itself. This experiment tests whether a training-side change
lowers that share: a longer embargo of the newest rows, down-weighting the newest rows, or both.

- **Base arm.** A1 of experiment 595 (shared lag encoder, lag-identity embedding, fine-tuned on
  experiment 581's synthetic lag-only shifts). Stored step-2 weights are read only; they never see
  the injected shift, so the step-2 embargo is unchanged at 3 days in every arm.
- **Arms.** E0: A1 as stored by experiment 595 (3-day embargo, existing 90-day recency weights).
  E1: the daily fine-tune embargo is 7 days instead of 3. E2w: 3-day embargo; every fine-tune row
  in the newest 7 days that remain after the embargo (age 3 to 10 days at the cutoff) has its
  weight multiplied by 0.5, and every older row by 1. The multiplier is a step, applied on top of
  the existing 90-day recency weight, and applies to training and early-stopping rows alike. E3:
  E1 and E2w together, so the embargo is 7 days and the rows of age 7 to 14 days are halved. (The
  down-weighted window follows the embargo, because the rows of the newest 7 days would otherwise
  all be embargoed in E3 and E3 would equal E1.)
- **Harness.** Experiment 595's: 12 injected designs, a shift in the training power and inputs,
  mirrored neighbour, coarse refits on day 0 and day 14, seeds 0 and 1, the five development
  months, July 2026 unused.
- **Reported.** Total and refit-only share by phase, tails, MDE and the Bonferroni count (3) as
  numbers; event-free CRPS and tail coverage by metering group, overall and at 0 to 24 h; four
  plots of the median against the no-event forecast.
- **Compute.** One GPU process at a time, `MemoryMax=8G`, ceiling 1.5 GPU-h; ask before crossing.
