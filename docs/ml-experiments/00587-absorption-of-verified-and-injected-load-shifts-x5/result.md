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
status: finished
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
verdict: refuted
best_arm: >-
  P5 (embargoed three-step recipe with the lag encoder and the offset) absorbs least: a refit share
  of +0.082 (0.040 to 0.123) of the injected event's depth in the 7 days after the end, against
  +0.186 (0.064 to 0.337) for M0 and +0.232 (0.099 to 0.380) for D5.
headline: >-
  A refit absorbs far more of a load-shift than predicted: at the day-14 refit M0 and D5 move the
  median forecast by 0.19 and 0.23 of the injected depth in the 7 days after the event ends
  (medians over designs 0.08 and 0.11), against a predicted 0.05 or less. The newest offset knot
  of D5 moves by only 0.012 of capacity on average, so the absorption sits in the network weights.
lessons: >-
  The share depends strongly on how much of the newest training data the event covers: 15-day
  events that start on day 3 give D5 shares of 0.58 to 0.70, 30-day events 0.20 to 0.26, 7-day
  events 0.06 to 0.14, and 15-day events that start on day 12 0.01 to 0.07. P5's 3-day embargo
  removes the newest event rows, so its refit share is lower, and the P5 minus D5 contrast
  measures the embargo and the three-step schedule together with the lag encoder.
take_home_message: >-
  When a load-shift is in the power readings used to refit the demand network, the refit moves the
  median forecast toward the event by about 0.1 to 0.3 of the event's depth for the networks
  without an embargo (M0, D5), and by about 0.08 for the embargoed recipe (P5). That is several
  times the 0.05 predicted, so the earlier shares of 0.014 to 0.040 on labelled events understate
  the leak, and a long-event embargo is worth testing next. The slow offset is not where the
  absorption happens: its newest knot moves by 0.012 of capacity on average.
advice:
  - >-
    Treat a refit that includes a running load-shift as absorbing about 0.1 to 0.3 of its depth
    into the median forecast, unless the newest 3 days of targets are embargoed (figure 1, the
    refit share at the day-14 refit).
  - >-
    Test a longer embargo or an event-aware down-weighting of fine-tuning rows next, on injected
    events, because the embargoed recipe still absorbs 0.08 of the depth (figure 1, P5).
  - >-
    Do not read the D5 offset's newest knot as the absorption channel; it moved by 0.012 of
    capacity on average while the share was 0.23 (figure 3).
  - >-
    Report absorbed shares as medians over designs beside means, with the number of designs per
    cell, because three designs carry the mean (table of design cells).
production_status: not-for-production
adopted_in: none
---

**We asked how much of a load-shift (a transfer of load between two Primaries, the primary
substations of the electricity distribution network) a demand network learns when it is refitted on
power readings that contain the load-shift.** We took one load-shift confirmed in the telemetry
(Primaries D5 and D16, 2025-12-29 to 2026-01-13, [experiment
580](../00580-verified-load-shift-event-set-and-label-audit/result.md)) and 12 load-shifts of known
depth that we added to the readings of two connected Primaries. For each, we refitted three
demand networks twice (on days 0 and 14 of a fold month), once on the readings with the load-shift
and once on the readings without it, and measured how far the median forecast of the first fit sits
from the second toward the load-shift, as a share of the load-shift's depth. The earlier shares of
[experiment 565](../00565-drift-embedding-variants-six-designs/result.md) and [experiment
566](../00566-combination-block-lag-encoder-with-three-step-schedule-and-embargo-real-event-leak/result.md)
used labels that experiment 580 could not confirm.

**Take-home message: the refit absorbs several times more of a running load-shift than predicted,
and the slow offset is not where the absorption happens.** At the day-14 refit, injected 7 to 30 day
load-shifts move the median forecast by 0.19 (M0) and 0.23 (D5) of their depth in the 7 days after
the end (95% intervals 0.06 to 0.34 and 0.10 to 0.38), and by 0.08 (0.04 to 0.12) for P5, whose
3-day embargo removes the newest event rows. The prediction was at most 0.05. The newest offset knot
of D5 moves by 0.012 of capacity on average, so the absorption sits in the network weights. Three
15-day designs carry the means: the medians over the 12 designs are 0.08 (M0), 0.11 (D5), and 0.07
(P5). The verdict is "refuted".

**Terms.** M0 is the lag-free network without an offset ([experiment
565](../00565-drift-embedding-variants-six-designs/result.md), arm `C-M0`). D5 is M0 plus a slow
per-Primary output offset (arm `D5`). P5 is the embargoed three-step recipe with a power-lag encoder
and the offset ([experiment
574](../00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/result.md)).
The **refit share** is the sign of the event times the median forecast of the fit on event readings,
run on no-event inputs, minus the median forecast of the fit on no-event readings, over the event's
depth. The **total share** runs the first fit on event inputs, so it adds P5's power lags. Positive
means toward the event; 1 means the forecast moved by the whole depth.

![Refit and total share of the injected load-shifts' depth, by phase and arm](figure_1_share_by_phase.svg)

![Verified D5 and D16 load-shift: observed, normal running arrangement, and the three arms](figure_2_verified_pair.svg)

![Newest offset knot movement by arm](figure_3_newest_knot_movement.svg)

![Four injected load-shifts](figure_4_injected_examples.svg)

## Results

- **Arms and recipe.** M0, D5, and P5 as above; refits on days 0 and 14, seeds 0 and 1; cleaned
  power; D15 and D18 excluded; step 2 and pre-training are experiment 565's and 574's stored
  weights, which never see an injected event. Weather from the run at or before the origin minus
  9 hours (asserted by the harness); lag inputs at least 24 hours old. Refits reproduce the stored
  forecasts to at most 5e-6 of capacity, so every fit here is a refit, run through the same code as
  the stored one.
- **Row set.** The 12 designs (fixed seed 20261106): the nine cells of depth 15%, 30%, 50% of the
  smaller capacity by length 7, 15, 30 days at start day 3, and three 15-day designs at start day
  12. Each design has two Primary-events, so 24 per arm. The bank has no 10% or 25% depths.
- **Refit share, day-14 refit, injected, mean (95% interval from resampling designs; designs).**
  Days 4 to the end: M0 +0.305 (0.144 to 0.484; 9), D5 +0.332 (0.178 to 0.498; 9), P5 +0.082
  (0.031 to 0.141; 9). The 7 days after the end: M0 +0.186 (0.064 to 0.337; 12), D5 +0.232 (0.099
  to 0.380; 12), P5 +0.082 (0.040 to 0.123; 12). First 4 days (2 designs, so no interval is
  meaningful): M0 +0.114, D5 +0.090, P5 0.000.
- **Medians over designs, day-14 refit.** Days 4 to the end: M0 +0.263, D5 +0.318, P5 +0.047. The 7
  days after the end: M0 +0.083, D5 +0.107, P5 +0.069.
- **By design cell** (days 4 to the end and 7 days after pooled, 3 designs per cell, mean and
  range). D5: 7 days +0.095 (0.057 to 0.140); 15 days from day 3 +0.642 (0.579 to 0.704); 30 days
  +0.224 (0.196 to 0.255); 15 days from day 12 +0.050 (0.012 to 0.071). M0: +0.064, +0.595, +0.142,
  +0.060. P5: +0.108, +0.156, +0.077, 0.000. The cause of the spread between cells is not tested.
- **Day-0 refit (event not in training).** Refit shares are zero to 1e-8 for all arms, which is the
  placebo. P5's total share is +0.014, +0.074, and +0.136 in the three phases, which is lag
  following alone.
- **Total minus refit (P5's lag share), day-14 refit.** About 0.18 in days 4 to the end (0.262
  against 0.082) and 0.14 in the 7 days after the end (0.219 against 0.082).
- **Contrasts of the refit share, day-14 refit, Bonferroni over 6 tests (minimum detectable effect
  is the interval's half-width).** D5 minus M0: days 4 to the end +0.028 (-0.029 to +0.106; 0.068;
  9 designs); 7 days after the end +0.046 (-0.000 to +0.098; 0.049; 12 designs). P5 minus D5: days 4
  to the end -0.250 (-0.411 to -0.100; 0.155); 7 days after the end -0.151 (-0.333 to -0.019; 0.157).
  The first-4-days contrasts rest on 2 designs and are not interpreted. P5 minus D5 mixes the
  embargo, the three-step schedule, and the lag encoder, so it does not isolate the lag encoder.
- **Newest offset knot, day-14 refit, injected, toward the event (share of capacity).** D5: mean
  +0.012, largest +0.086 (2 of 24 Primary-events above 0.03); P5: mean +0.001, largest +0.006.
- **Tail shares (refit, day-14 refit, days 4 to the end).** M0: q01 +0.40, q05 +0.37, q50 +0.31, q95
  +0.36, q99 +0.38. D5: +0.41, +0.38, +0.33, +0.35, +0.35. P5: +0.07, +0.07, +0.08, +0.12, +0.13.
  The whole distribution moves with the median; for P5 the upper tail moves slightly more.
- **Verified D5 and D16 pair (one pair, so descriptive).** At the 2026-01 day-14 refit, 7 days
  after the end, refit share at D5 and D16: M0 +0.177 and +0.085, D5 +0.357 and +0.165, P5 +0.038
  and +0.113; P5's total share +0.172 and +0.293. At the 2025-12 refits and at the 2026-01 day-0
  refit (which holds about 3 days of the event) the shares are 0 to 0.13. The no-event world
  subtracts the fitted constant depth, which carries a load mismatch of 0.017 of capacity.
- **Replacing the erratum numbers.** Experiments 565 and 566 reported 0.014 to 0.040 of labelled
  depth for events that ended before the refit. The injected 7-day designs, which ended 5 days
  before the day-14 refit, give M0 +0.064, D5 +0.095, and P5 +0.108 (3 designs each; ranges 0.03 to
  0.19). Those figures replace the unproven ones for the comparison of recipes. The real-event
  shares on 565 and 566's pages stay unproven.

## Checks

- **Median minus mean forecast per site:** not applicable (only quantiles are stored).
- **Error of the mean forecast beside the quantile contrast:** not applicable (the measure is a
  difference of forecast quantiles; all five quantile shares have the same sign).
- **Triggers fired.** Trigger 2 (the result contradicts the prediction): fired, and a fresh
  Sonnet audit re-derived the shares from the raw forecast files and reproduced every printed number
  to 3 decimals (the audit report is in the private briefs folder). Trigger 5
  (a nested arm beats the arm that contains it) fired in form: P5 absorbs less than D5 because of
  the embargo, not the lag encoder. Triggers 1, 3, 4, 6, and 7 did not fire (leaving one design out
  flips no sign; dropping design 234 shrinks D5 minus M0 in days 4 to the end to +0.009).
- **Triage of the audit's findings.** Accepted: report medians beside means (done); relabel P5
  minus D5 (done); no interval for the 2-design first-4-days cell (done); early-stopping epochs
  differ between worlds by 2 to 4 on average (placebo Primaries move by 0.001 of capacity, so this
  does not explain the shares). Not acted on: why shares fall with the distance from the training
  rows is untested.
- **Reviews.** The brief asked for one code review and one write-up review, so no plan review ran. A Sonnet
  code review before the first run (5 must-fix findings, all fixed); Sonnet audit of the results;
  write-up reviews below.

## Caveats and next steps

- **Predictions.** P1 failed (D5 0.23, over twice 0.10 as a mean, and 0.11 as a median). P2 failed
  in the 7 days after the end (D5 minus M0 +0.046). P3 held. P4 failed: the verified pair's
  2026-01 day-14 shares (0.09 to 0.36) are not within 0.05 of the injected mean of 0.64 (D5) for
  15-day events from day 3, and sit close to the injected 7-day and 30-day cells.
- **Lead range.** The shares pool all leads of the forecast (up to 10 days).
- **Unlogged events.** NGED's logs may miss load-shifts, so the fits on the other months can
  contain events that no label covers.
- **Designs per cell.** Only 3 designs sit in each cell; a design's month and Primary pair differ
  between cells.
- **Next.** Test a longer embargo and an event-aware down-weighting of the fine-tuning rows on the
  same bank.
