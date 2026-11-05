---
id: "00572"
slug: two-offset-part-1-lags-versus-the-slow-output-offset
title: >-
  Do power lags still help the modular demand network once a slow per-Primary output offset (D5)
  exists, and does the offset make lags unnecessary at 24 to 240 hours?
theme: lagged-power-and-error-correction
also_relevant_to:
  - differentiable-physics-long-history-and-drift
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted weights, forecasts, and scores in the private data folder only. More than
  one defensible design: how the offset is learnt (pre-training or fine-tuning only) and whether a
  lag network carries it. Callers: the code copies experiments 560 and 565 and imports 545, 554,
  558, 559, and 566 (more than two callers, found by search). Serving path and degradation rule: not
  touched (this repository holds none). Cost: ceiling 2 GPU-h (the manager's limit for Part 1).
hypothesis: >-
  A slow per-Primary output offset (D5) carries the Primary's current level into the forecast, as
  lagged power does. If so, a lag-free modular network with an offset learnt in pre-training (E2)
  scores about as well as the best lag network (v3) at 24 to 240 hours, and adding the offset to the
  lag network (v3+D5) gains little. Lags should still matter in the first 24 hours.
prediction: >-
  Written before any fit. P1: E2 is within 0.05 of v3 at 24-120 h at the eight MVA Primaries and at
  the six MW Primaries (fails if v3 is better by 0.10 or more at either). P2: v3-3.5 (lags up to
  3.5 h old) beats E2 by at least 0.15 at 0-24 h (fails if within 0.05). P3: at 120-240 h, E2 is
  within 0.03 of v3 and of v3+D5 (fails if v3 is better by 0.05 or more). P4: v3+D5 beats v3 by at
  least the minimum detectable effect at 24-120 h (fails if within 0.03, which would make the two
  redundant). P5: E2 beats E2-ft by at least 0.03 (fails if within 0.02). P6: E2's absorbed share
  of the depth of the 74 real load-shift events is above M0's and below v3's (566: 0.026).
decision_rule: >-
  A gain counts when the Bonferroni month-block interval (5 blocks, near the minimum to maximum of
  the five folds) excludes zero and the gain is at least 0.03 CRPS points of capacity (the offset
  and the lag encoder each add a network part, so zero would not pay for the port). Bonferroni count
  24: 4 contrasts (E2 against M0, E2 against v3, E2 against v3+D5, v3+D5 against v3) x 2 groups x 3
  lead bands (0-24, 24-120, 120-240 h). Everything else is descriptive and labelled so. The
  stop rule of the brief: if v3+D5 beats E2 by at least 0.10 at 24-120 h at both groups, the
  report says Part 2 should continue on v3+D5 as well as E2.
benchmark: >-
  M0 coarse (experiment 565's C-M0, no offset, no lags), E2 (experiment 565's D5, stored), v3
  (experiment 560, refitted coarse), with L-1d (511), A-pool-24 (558) as report-only references.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity in three lead bands at the eight MVA Primaries and the six
  MW Primaries (D15 and D18 report-only); leak: injected 7-day shift (stress) and real-event
  absorbed share (primary).
---

**Plan only; the write-up replaces this page when the fits finish.** The brief is Part 1 of
`briefs/two_offset_nra_and_prevailing_forecast_brief.md`. Part 2 (two offsets and a prevailing
forecast) is a later experiment and is not started here.

## Step 0: what experiment 565's D5 did

- **Where the offset was learnt.** In pre-training: `pretrained_D5/real/<fold>_s<seed>.pt` holds one
  scalar per Primary and month from 2019-09 (monthly knots on the ERA5 and CAMS rows). Fine-tuning
  keeps the older knots as fixed buffers and trains only the newest knot. Flat extrapolation
  clamps every position after the cutoff to the newest knot.
- **Penalty and read-out.** `OutputOffsetDrift`: an L2 penalty of weight 10 on the mean squared first
  difference of each Primary's knots (a slope penalty). The offset is added to the three mixture
  component means, in standardised units.
- **So E2 is experiment 565's arm D5 as stored** (coarse refits on days 0 and 14, seeds 0 and 1 at
  least, folds 2025-10 to 2026-06). E2 needs no new fit. **E2-ft is new**: the same network, with the
  pre-training of M0 (experiment 554) and every knot learnt in fine-tuning only.

## Arms (all on the same rows, folds, seeds 0 and 1, coarse refits)

| Arm | Offset | Lags | Source |
|---|---|---|---|
| M0 | none | none | stored: 565's C-M0 |
| E2 | learnt in pre-training and continued | none | stored: 565's D5 |
| E2-ft | learnt in fine-tuning only | none | new fit |
| v3 | none | day and week lags at least 24 h old, block dropout | experiment 560's pre-trained v3 weights, refitted coarse |
| v3+D5 | learnt in pre-training and continued | as v3 | new pre-training and fit |
| v3-3.5 (report only) | none | day and week lags at least 3.5 h old, block dropout | new pre-training and fit |

L-1d (experiment 511) and A-pool-24 (experiment 558) are stored report-only references; the brief's
question of whether production may read lags newer than 24 h is the maintainer's, and v3-3.5 only
measures it. v3-3.5 runs last and is dropped first if the GPU-hour ceiling is at risk.

## Scoring

- CRPS in lead bands 0-24, 24-120, and 120-240 h (the 6-hourly half of the ensemble: the
  half-hourly upsampling rule of experiment 521), at the eight MVA Primaries and the six MW
  Primaries; slices: cold winter nights, Christmas week, rows with no labelled event, D8 alone, D14
  alone, leave-D8-out, with D15 and D18 report-only.
- Leak: the injected 7-day shift (stress, flag) and the absorbed share of depth of the 74 real events
  (primary), with lags removed and aged by 1, 2, and 3 extra days (inference only) for every lag arm.
- Minimum detectable effect beside every contrast, re-measured per lead band from the seed spread
  (two-seed values of the brief: 0.113, 0.128, 0.092), per-fold signs, 5-block month bootstrap.

## Timing asserts (printed row by row, one example per arm and band)

The weather comes from the ensemble run at or before the origin minus 9 h; every lag and observation
is at or before the origin minus 3.5 h for v3-3.5 and minus 24 h for v3 and v3+D5; the end time of
every slow summary precedes the origin.

## Advice followed and departed from

Follows 565 (carry slow change as an output offset, not an embedding), 560 (score lags removed and
aged beside the normal score), and the standing rules (one fit per fold, cleaned data, D15 and D18
cut before 2025-03-01, saved weights). Departs from none.
