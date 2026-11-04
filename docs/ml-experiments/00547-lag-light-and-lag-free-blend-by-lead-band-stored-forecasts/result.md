---
id: "00547"
slug: lag-light-and-lag-free-blend-by-lead-band-stored-forecasts
title: >-
  Does a fixed-weight blend of the lag-free network L-1d and the lag-light network R-DW, with one
  weight per lead band, forecast better than either network alone?
theme: lagged-power-and-error-correction
also_relevant_to:
  - demand-inputs
status: finished
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
verdict: not supported
best_arm: >-
  None. The blend scored 4.404 at the eight, 24-120 h, against R-DW's 4.403 (difference +0.000,
  interval -0.018 to +0.013).
headline: >-
  A fixed-weight blend of the lag-free network L-1d and the lag-light network R-DW did not beat
  R-DW: at the eight MVA-only Primaries, 24-120 h, the blend was +0.000 points of capacity from
  R-DW (month-block interval -0.018 to +0.013, 4 blocks, MDE 0.04), against a pre-registered gain
  of at least 0.05 over both parents. The fitted weight on R-DW was 0.7 to 1.0, so the blend is R-DW
  with a little L-1d added. Interim and unaudited.
lessons: >-
  R-DW already holds what L-1d knows at 1 to 5 days ahead, so averaging the two quantile forecasts
  only moves the answer towards the worse network. The two networks share inputs and a pretrained
  start, so their errors are too alike to cancel.
take_home_message: >-
  Averaging the lag-free network with the lag-light network gives no better forecast than the
  lag-light network alone: at 1 to 5 days ahead the two scores differ by under 0.01 points of
  capacity, far below the 0.05 we asked for. The lag-free network adds nothing the lag-light
  network lacks, so the way to improve further is a different kind of input or network, not a
  blend of these two. The next step is to test a blend with a network of a different kind (for
  example K2-S1 rebuilt on origin-time weather).
advice:
  - >-
    Do not blend L-1d with R-DW by lead band: the walk-forward blend equalled R-DW at 24-120 h
    at the eight (figure 1, contrast table).
  - >-
    Blend only networks with different inputs or architectures, because two networks with the same
    inputs and start gave no diversity gain (fitted weights 0.7 to 1.0 on R-DW, figure 2).
production_status: not-for-production
adopted_in: none
---
**Averaging the lag-free network L-1d with the lag-light network R-DW, with a fixed weight per lead
band, forecasts no better than R-DW alone.** We asked whether the two networks make different
enough errors that their blend beats both. The blend averages the 11 stored quantiles of each
network (mixture parameters are not stored), with the weight on R-DW fitted on the earlier folds
and applied to the next. Nothing was fitted except the weights, and no new forecasts were made.

Verdict: not supported. At the eight MVA-only Primaries and 24-120 h, the blend scored 4.404
points of capacity (CRPS) against R-DW's 4.403 and L-1d's 4.597. The pre-registered gain was at
least 0.05 below both parents in each of 24-48 h, 48-72 h, and 72-120 h, and the gain over R-DW was
+0.003, +0.002, and -0.002 in those bands. The first fold (2025-10) supplies weights only, so
everything below is the four folds 2025-12, 2026-01, 2026-03, and 2026-06. July 2026 is unused.
Every number is interim and unaudited.

![CRPS minus A-pool at the eight for L-1d, R-DW and the blend, by lead band](figure_1_gap_to_a_pool.svg)

## Results (four scored folds)

- **Eight, blend minus R-DW:** +0.003 (24-48 h), +0.002 (48-72 h), -0.002 (72-120 h), +0.000 (24-120
  h). The 24-120 h interval is -0.018 to +0.013 (month-block bootstrap, 4 blocks, near min-to-max),
  with 3 of 4 folds positive (the blend worse) and an MDE of 0.04. The difference is far below both
  the 0.05 goal and what the experiment could resolve.
- **Eight, blend minus L-1d:** -0.19 at 24-120 h (interval -0.28 to -0.11, 0 of 4 folds positive);
  this is R-DW's own gain over L-1d, not a gain from blending.
- **Six (signed MW, rows with no labelled event):** blend minus R-DW is -0.001 to -0.004 in each of
  the three bands (-0.002 at 24-120 h, interval -0.016 to 0.006, MDE 0.08); blend minus L-1d is
  -0.06 at 24-120 h.
- **Weights:** at the eight the fitted weight on R-DW is 1.0 at 6-24 h, 0.9 at 24-72 h, 0.7 to 0.8
  at 72-120 h, and 0.5 to 0.6 at 120-240 h (figure 2). At 120-240 h the blend is -0.010 from R-DW
  (interval -0.064 to 0.052).
- **A fixed half-half blend** is worse than R-DW at the eight (+0.041 at 24-120 h, interval 0.006
  to 0.086): equal weights move the forecast towards the worse network.
- **Versus A-pool** at the eight, 24-120 h: blend +0.021 (interval -0.027 to 0.061).
- **Cold winter nights (eight, 24-120 h, 13,092 rows):** blend 3.976, R-DW 3.945, L-1d 4.318.
  **22-31 December (fold 2025-12 only, 15,360 rows):** blend 3.987, R-DW 4.062, L-1d 3.975.
- **Tails (eight, 24-120 h, blend):** coverage at q01 0.25%, q05 1.5%, q95 96.2%, q99 99.4%, so
  the 1.15 widening leaves the bands slightly too wide.
- **D15 and D18 (report-only), 24-120 h:** blend minus R-DW +0.084 (folds -0.16, +0.14, +0.33,
  0.00); blend minus L-1d -0.92.
- Counted tests: 28 (2 contrasts x 7 bands x 2 sets). No interval above is widened to that count,
  and no claim rests on one.

![Fitted weight on R-DW by lead band, eight MVA-only Primaries, one line per scored fold](figure_2_weights.svg)

![A day-ahead and a full 10-day forecast of the blend for one Primary, December 2025](figure_3_example_forecast.svg)

## Checks and caveats

- Weather inputs: none are read here. Both networks used the weather run issued at or before the
  forecast origin, asserted row by row in experiment 539's checks. This script asserts that every
  weight was fitted on earlier folds only and prints one example row: fold 2025-12, Primary 0,
  origin index 109602, target index 109705 (lead 51.5 h), weight 0.9, L-1d median 0.786, R-DW
  median 0.791, blend median 0.790, actual 0.892 (fractions of capacity).
- The parents' CRPS recomputed from the stored quantiles matches experiment 539's stored per-row
  CRPS to under 0.01 on every row (asserted).
- The weights come from quantile averaging, a scored-fold-by-fold walk-forward on four
  non-contiguous months, so earlier folds are not always the same season. Four month blocks are
  near the min-to-max of the folds.
- The first fold is not scored. The MDE comes from the two seeds of each parent, blended with the
  same weights.
- The avoidable-curtailment cost was not computed. The related experiment 458 (a weight per lead
  band on the lag-free forecast in a lagged and lag-free mixture) also found no gain.
- Code review: one Sonnet review of the pull request, findings triaged in the pull request.
- Cost: no GPU, under 1 minute on 2 cores.

## What the next experiment should answer

Whether a blend of R-DW with a network of a different kind, such as K2-S1 rebuilt on origin-time
weather (experiment 541), has errors different enough to give the 0.05 gain.
