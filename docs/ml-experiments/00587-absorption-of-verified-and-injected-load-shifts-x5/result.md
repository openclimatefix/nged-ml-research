---
id: "00587"
slug: absorption-of-verified-and-injected-load-shifts-x5
title: >-
  How much of a load-shift does the fine-tuned demand network absorb into its forecast, with and
  without the slow output offset, when the load-shift is a telemetry-verified one or an injected one
  with a known depth?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: refits of three arms on event-carrying power, forecasts and score tables in the
  private data folder only. More than one design could define "absorbed share" (see the plan), so
  the plan fixes three quantities before any score. Spans experiments 565, 574, 580, and 581, whose
  code this experiment wraps without editing them. Serving path and degradation rule: not touched
  (this repository holds none). Compute: under the 1.5 GPU-h ceiling (each refit takes about 10
  seconds on the GPU, plus about 45 seconds to build a fold month's rows).
hypothesis: >-
  A load-shift in the fine-tuning targets moves the fine-tuned network's median forecast toward the
  event by only a small share of the event's depth, and the slow output offset (D5) and the lag
  encoder (P5) do not raise that share much above the lag-free network's (M0).
prediction: >-
  Written before any score. P1: on the injected 15 and 30 day events at 30% and 50% depth, for the
  day-14 refit (event rows in training), the refit-only share (the event fit run on uninjected
  inputs, against the refit on uninjected power) is at most 0.05 of depth at D5 and at P5 in the
  7 days after the event ends, and the newest D5 knot moves by at most 0.03 of capacity (the
  prediction of the plan review of 2026-11-05, row X5). P1 fails if either number is doubled (0.10
  of depth, 0.06 of capacity). P2: M0's share is within 0.03 of D5's in every phase (the offset does
  not catch up faster than the network). P3: for the day-0 refit, where the event is not in the
  training rows, M0 and D5 shares are 0 by construction, and P5's total share equals its lag
  following alone. P4: on the verified D5/D16 pair, the shares at the day-14 refit of 2026-01 are
  within 0.05 of the injected-bank mean for the same depth and length.
decision_rule: >-
  All shares are descriptive. Intervals resample events (design blocks, with both Primaries of a
  transfer in one block) with 2000 draws, and the printed half-width of each interval is the
  minimum detectable effect. Counted tests: 6 (D5 minus M0 and P5 minus D5, each for the first 4
  days, day 4 to the end, and the 7 days after the end, on the day-14 refit, injected events). A
  difference counts only if its Bonferroni interval excludes zero. A share of at least 0.10 of
  depth in the 7 days after the end is read as material absorption, and then a long-event embargo
  or an offset change is queued. The experiment adopts nothing.
benchmark: >-
  The corrected-target refit (the same fold month refitted on a target with the event removed),
  which is the stored fit of experiments 565 and 574 for injected events and a new refit for the
  verified pair.
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the event's depth that the median forecast moves toward the event (sign times the
  difference of medians over depth) by phase (before, first 4 days, day 4 to the end, 7 days after
  the end), for the total effect and for the refit alone, at q50 and at q05, q95, q01, and q99.
---

**Plan only; this page is rewritten when the experiment finishes.**

## Plan

- **Question.** When the power readings carry a load-shift (a transfer of load between two Primaries),
  how much of its depth does a refit of the demand network absorb into the median forecast? The
  experiment follows the plan review's check X5 (`briefs/nra_plan_review_opus_answer.md`) and the
  erratum on experiments 565 and 566, whose absorbed shares used labels that experiment 580 could
  not confirm.
- **Three quantities, fixed before any score.** Each case has an event world (power carries the
  event) and a no-event world (the event removed, which is the uninjected power for an injected
  event and the observed power minus the fitted depth for the verified pair). With `f(model,
  inputs)` the median forecast at the event's Primary, `s` the sign of the event's depth at that
  Primary (positive: the Primary carries more), and `D` the depth:
    - **Total share** = `s * (f(fit in event world, inputs of event world) - f(fit in no-event
      world, inputs of no-event world)) / D`.
    - **Refit share** = `s * (f(fit in event world, inputs of no-event world) - f(fit in no-event
      world, inputs of no-event world)) / D`. The no-event fit is the corrected-target refit of
      question 2. M0 and D5 read no power lags, so their total and refit shares are equal.
    - **Lag share** = total share minus refit share, which is non-zero only for P5.
- **Arms.** M0 (experiment 565's lag-free network, arm `C-M0`), D5 (experiment 565's arm `D5`, the
  slow output offset, no lags), and P5 (experiment 574's embargoed three-step recipe with the lag
  encoder and the offset). Refits on days 0 and 14 of the fold month, seeds 0 and 1. Step 2 and the
  pre-training are the stored ones and never see an injected event.
- **Cases.** The verified D5 and D16 transfer of 2025-12-29 to 2026-01-13 (experiment 580,
  `events_verified.parquet`, transfer 23), scored at the 2025-12 day-14 refit and at the 2026-01
  day-0 and day-14 refits. Twelve designs from experiment 580's bank of 358, drawn with a fixed seed
  (20261106): the nine cells of depth 15%, 30%, 50% by length 7, 15, 30 days at start day 3, and
  three 15-day designs at start day 12 (one per depth). The bank's depths are 15%, 30%, and 50% of
  the smaller capacity, not the 10%, 25%, and 50% of the brief, and its lengths include 3 days.
- **Phases** (by valid time, relative to the event): before the onset, the first 4 days, day 4 to the
  end, and the 7 days after the end. A phase with fewer than 48 half-hours at a Primary is dropped
  from that case and counted.
- **Reuse.** Event-world forecasts of the verified pair and no-event forecasts of every injected
  design are the stored fits of experiments 565 and 574 (`fits/<arm>/<month>_d<day>_s<seed>`), after
  a determinism check that a refit on uninjected power reproduces the stored forecast. Refitted:
  every injected-world fit (3 arms x 12 designs x 2 refits x 2 seeds = 144) and the 3 arms x 3
  refits x 2 seeds = 18 no-event fits of the verified pair.
- **Standing practices followed.** Monthly cadence in the sense of the coarse refits (days 0 and 14);
  cleaned power throughout (the stack of `e00510_common.clean_stack`); D15 and D18 are report-only
  and excluded from the bank; every fitted model saved as a `.pt` file in the experiment's data
  folder; stored forecasts reused as above; July 2026 unused. Not followed, with the reason: the
  avoidable-curtailment cost and the example 14-day forecast plots are not produced, because the
  measure is a share of depth, not a forecast score; event plots are produced.
- **Weather.** The weather run at or before the origin minus 9 hours (the harness asserts it); lag
  inputs at least 24 hours old.
- **Earlier advice followed.** Judge long-event behaviour on injected events (experiment 580);
  use no labelled event for a claim (experiment 580); report the minimum detectable effect and the
  Bonferroni count as numbers.
- **Compute.** One GPU process under 8 GB (`MemoryMax=8G`, `OMP_NUM_THREADS=1`), ceiling 1.5 GPU-h.
- **Caveat to carry into the write-up.** The real-event numbers rest on one verified pair. NGED's
  logs may miss events, so fits on the other months may contain unlogged events.
