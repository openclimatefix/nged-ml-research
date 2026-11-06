---
id: "00590"
slug: synthetic-load-shifts-in-the-training-targets
title: >-
  Does the gain from fine-tuning on synthetic load-shifts survive when the synthetic load-shift also
  moves the training targets, as a real load-shift does, and does it teach the network to follow
  shifts in its lag inputs?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: three new arms, one new fine-tune per fold month, refit day, and seed (60 small
  neural-network fits, private data folder), plus tables and figures. More than one design could
  test the hypothesis (see "Arms"); the arms follow the manager's brief. Spans experiments 574, 581,
  584, and 587, whose code this experiment wraps. Serving path and degradation rule: not touched
  (this repository holds none). Cost: well under the 1.5 GPU-h ceiling (fits of about 10 seconds
  each, row building of about 45 seconds per fold month, inference on the injected bank).
hypothesis: >-
  In experiments 581 and 584 the synthetic load-shift is added to the lag inputs only, so the
  network learns that its lags can lie. A real load-shift also moves the target on the same days.
  Applying the synthetic shift to the targets as well teaches the network that a shift in its lags
  predicts a shift in the target, so it follows lag shifts more and the coverage gain after a
  load-shift ends shrinks; a mixture of lag-only and lag-plus-target shifts keeps part of the gain.
prediction: >-
  Written before any score. P1: T1 (shift in lags and targets on 5% of the time) follows the
  lag-only injected events more than T0 (lag-only, stored as C1 in experiment 584) by at least 0.05
  of the depth in the 7 days after the end, and covers the NRA on at least 0.03 fewer rows. P2: T2
  (mixture) sits between T0 and T1 on both, and covers the NRA on at least 0.82 of rows in the 7 days
  after the end. P3: T3 (placebo noise) covers the NRA on at least 0.03 fewer rows than T0, so the
  gain of T0 over C0 comes from the shifts and not from noisy lags. P4: event-free CRPS at 24-120 h
  of T1, T2, and T3 is within the minimum detectable effect of T0. P5: the refit-only share of
  experiment 587 (the share of a shift in the training power that the weights absorb) is higher for
  T1 than for T0 by at least 0.03 of depth, because T1's weights have seen target shifts.
decision_rule: >-
  Counted contrasts: 12 (T1, T2, and T3 minus T0, each for four measures: coverage of the NRA in
  q05-q95 in the 7 days after the end of the injected 15 and 30 day events, median follow ratio in the
  same window, event-free CRPS at the eight MVA Primaries, and event-free CRPS at the six MW
  Primaries). Intervals are at the Bonferroni level for 12 tests (event-block bootstrap with 2000
  draws for the injected bank, five-block month bootstrap for event-free CRPS), and the half-width
  of each interval is printed beside it as the minimum detectable effect. T1 or T2 beats T0 on a
  contrast only if the difference favours it (higher coverage, lower follow ratio, lower CRPS) by
  more than the minimum detectable effect. The experiment adopts nothing. A leak or slice cost is a
  flag unless it actively hurts: worse than T0 by more than max(0.03, the slice minimum detectable
  effect) with the interval excluding zero, or a harmful sign in 4 of 5 fold months.
benchmark: >-
  T0 = arm C1 of experiment 584 (stored forecasts and weights; the P5 recipe fine-tuned with lag-only
  synthetic load-shifts on 5% of each Primary's time), and C0 (experiment 572's v3+D5, stored) as the
  reference without synthetic shifts.
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the NRA inside q05-q95 and q01-q99 by phase (before, days 0-3, day 4 to the end, 0-7 days
  after, 7-14 days after) on the injected bank and on the verified December pair, median follow
  ratio by phase, safe-side share, pinball loss at q01, q05, q95, q99, width, event-free CRPS in
  points of capacity, calibration on event-free rows, and the experiment 587 refit-only share for T0
  and T1.
---

**Plan only.** Nothing has been scored. The rest of this page is written when the results exist.

## Question

A load-shift moves part of one primary substation's demand onto another, so the recent power
readings (the lag inputs) no longer show the normal running arrangement (NRA). Experiments 581 and
584 fine-tune the forecast network on lag inputs that carry synthetic load-shifts while the targets
stay clean. The network therefore learns that its lags can lie. A real load-shift moves the lags and
the target on the same days. We ask whether the coverage gain survives, and whether the network learns
to follow lag shifts, when the synthetic load-shift is applied to the targets too.

## Arms

All arms are experiment 584's C1 recipe (experiment 574's P5, one new daily fine-tune per fold month,
refit day, and seed) with the same event schedule (depths 15%, 30%, 50% of capacity, lengths 3, 7, 15,
30 days, random sign, seeded by the fold month). Only the fine-tune's inputs and targets differ.

- **T0 (control, stored):** lag-only shifts on 5% of each Primary's time. These are experiment 584's
  C1 forecasts and weights; nothing is refitted.
- **T1:** the same schedule applied to the lag inputs and to the targets of every row whose valid
  time lies inside an event.
- **T2:** lag-only shifts on 2.5% of the time and lag-plus-target shifts on a further 2.5%.
- **T3 (placebo):** the lag inputs read the power plus independent Gaussian noise, whose standard
  deviation at each Primary equals that of the T0 shift schedule over the month, with no persistent
  shift. The targets stay clean.

The standardisation and target moments stay those of the clean power for every arm. T3's noise is
independent per half-hour, so it has the variance of the shifts but not their persistence; a
persistent placebo is not tested.

## Scoring

- **Injected bank:** experiment 581's lag-only injection (24 passes; a constant shift in the lag
  inputs only), so the test reads whether the network follows the lags. Follow ratio and coverage by
  phase, the safe-side share, and tail pinball losses, from experiment 581's and 584's scoring code
  with the same uninjected twin.
- **Verified December pair:** D5 and D16, descriptive (two Primary-events).
- **Event-free rows, 24-120 h:** CRPS (2 × mean pinball loss over the 11 stored quantiles), coverage
  and pinball loss at q01, q05, q95, q99, per metering group, eight MVA and six MW Primaries.
- **Absorption (experiment 587's measure):** the refit-only share of an event in the training power,
  for T0 and T1, on the six 15-day designs of experiment 587 at the day-14 refit, 7 days after the
  end, against experiment 587's stored plain P5. If wiring T0 and T1 into experiment 587's harness
  proves more work than it is worth, the report says so and leaves this measure out.
- **Reuse:** T0 and C0 are stored forecasts of experiments 584 and 572. Their rows match the new
  arms' because the new arms use experiment 584's wrapper and row set.

## Standing practices followed

Coarse refits (days 0 and 14), seeds 0 and 1, five development test months (2025-10, 2025-12,
2026-01, 2026-03, 2026-06), July 2026 unused, cleaned data, D15 and D18 cut before 2025-03-01, weather
from the run issued at or before the origin minus 9 hours, lag inputs at least 24 hours old, fitted
weights saved in the private data folder, tails reported beside CRPS. Stored forecasts cover 0 to 120
hours, so no 14-day horizon plot exists; NGED care about 1 to 10 days, so every number is a short-lead
result. The avoidable-curtailment cost needs node-level forecasts and is not computed. NGED's logs may
miss load-shift events, so clean rows may hold unlogged events.

Scripts: `ml-experiments/00590-synthetic-load-shifts-in-the-training-targets/`.
