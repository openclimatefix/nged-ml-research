---
id: "00706"
slug: inference-time-level-tracking-part-1-look-back-offsets-by-driver-on-stored-forecasts
title: >-
  Can the slow level of a Primary substation's net demand be tracked at forecast time, from the
  forecast errors of the last 14 days, split by irradiance and temperature?
theme: lsc-detectors
also_relevant_to:
  - lagged-power-and-error-correction
  - calibration-and-uncertainty
  - switching-labels-in-the-forecast
status: finished
kind: other
protocol: demand-v2
size: >-
  Anything else. (1) Stored: forecasts of four arms, the residual table, and, in stage 2 only, the
  head's state files, all private under the experiment's data folder; nothing under `src/`. (2) More
  than one design defensibly tests the hypothesis (binned or regressed drivers, which stored
  forecast
  is "expected", how the share is learned). (3) Only this folder's scripts call the new code, and
  they import the folders of experiments 647, 660, 667, and 688 and `studies.*`. No download, no
  spend. Reviews: Fable design, Opus design review, Opus code review before each first run, Opus
  result review, Sonnet numbers review, Sonnet prose review, and an audit if a trigger fires.
hypothesis: >-
  A look-back over the last 14 days of the stored normal-running forecast's errors, computed at each
  forecast origin and split by driver, lowers the forecast error on rows free of load-shift events
  beyond what the Candidate's monthly-knot offset and its lagged-power and recent-gap inputs already
  give. In stage 2, a share learned from the size, timing, and electrical neighbours of a shift lets
  the normal-running forecast take slow drift and leave a switching shift out.
prediction: >-
  Written before any score is read, CRPS in points of capacity, leads 0 to 240 h, 2025-12 to
  2026-06. C1: the lead-band global offset minus the stored normal-running forecast on event-free
  rows
  is -0.05 to -0.15, with day 1 showing less gain because the Candidate already tracks a 1-to-7-day
  level. C2: the driver split minus the global offset is between -0.02 and +0.02 (expected to fail),
  with one predicted gain of -0.05 on sunny daylight half-hours from April to June at the signed-MW
  six. C3 (stage 2): the learned share cuts the copied share of an injected shift by a third to a
  half
  against the no-network rule. C4 (stage 2): the real neighbour graph and a rewired graph differ by
  under 0.1 points, for each of N-flag and N-gate. C6 (stage 2): N-gate minus N-flag on the copied
  share
  and on event-free CRPS is within 0.1 points, and we expect N-gate to be no worse. C5: the faithful
  prevailing arm minus the stored prevailing rows is 0 to +0.1.
decision_rule: >-
  Contrasts are paired on series, origin, and valid time and read at 99.17% (Bonferroni over six
  contrasts). A real-row contrast counts when its month-block bootstrap interval and the t test on
  the monthly paired differences both exclude zero (critical value 3.71 for 6 degrees of freedom),
  and each interval's half-width and minimum detectable effect is printed. C1 and C2 are read in the
  signed-MW six and the MVA-only eight separately, and both groups must pass. The smallest
  worthwhile
  gain for C1 is 0.05 points, because the offset adds a serving step that keeps 14 days of issued
  forecasts and telemetry, and 0.05 equals what fresher lags bought in experiment 572. C1 passes
  when
  the 99.17% interval lies wholly below -0.05. C2 passes at zero with an interval that excludes
  zero. A
  point estimate worse than +0.02 in either group blocks `supported`. The full verdict table covers
  every combination of C1, C2, and C3 and is in the design file. A failed known-answer gate caps the
  affected contrast at `inconclusive`.
benchmark: >-
  The stored normal-running and prevailing rows of the one-year backtest files (experiment 668),
  which carry the Candidate's monthly-knot offset D5, 24-hour and 7-day lags, and recent-gap and
  neighbour inputs; recipe R (experiment 115), a 28-day median shift (experiment 642), and a
  month-by-half-hour climatology.
row_set_id: backtest_year_2025_12_to_2026_06_daily_12utc_pre_holdout
metric: >-
  CRPS and normalised mean absolute error of the median forecast, in points of capacity, with q01,
  q05, q95, and q99 coverage and pinball loss, cold winter nights, and sunny April to June
  half-hours.
verdict: inconclusive
production_status: not-for-production
adopted_in: none
best_arm: >-
  None. The stored normal-running forecast (CRPS 3.235 points of capacity, pooled over 14 Primaries)
  was not beaten by any look-back arm. The best look-back arm, the lead-band global offset (G-lb),
  scored
  +0.026 worse (99.17% interval -0.070 to +0.179).
headline: >-
  A 14-day look-back offset gains 0.13 points of capacity on day 1 and 0.10 on day 2, and loses more
  than
  it gains from day 5 onward, so on all lead times it is +0.026 worse than the stored forecast
  (interval
  -0.070 to +0.179). Splitting the offset by irradiance and cold gains nothing more (+0.012,
  interval
  -0.047 to +0.080), and the learned gate and flag heads of stage 2 (post hoc) do not beat the
  no-network
  rule on real rows.
lessons: >-
  The recent forecast error carries level information for the next 1 to 2 days only, and applying
  one
  offset at every lead time spends that gain on days 5 to 10. The gain also depends on the look-back
  window being free of an event: windows that contain a load shift gave +0.337 (audit), while clean
  windows gave -0.026 (interval -0.079 to +0.020). A learned gate learned to read neighbour inputs
  on
  injected shifts, where the generator puts the mirrored step on the real neighbours, but on real
  rows
  the real graph and a rewired graph scored within 0.015.
take_home_message: >-
  Correcting the forecast with the last 14 days of its own errors helps for the first one to two
  days
  and hurts after day 5, so one offset applied to every lead time does not improve the forecast. The
  fixed look-back offset also copies a load shift forward for a week after the shift ends. A learned
  gate
  copies less of an injected shift than a fixed rule, but only when its neighbour inputs hold the
  mirrored
  step, which real data did not show.
advice:
  - >-
    Fade a look-back level offset with lead time, or apply it to day 1 and day 2 only, never at
    every
    lead time up to 10 days (stage 1 lead bands, figure 1).
  - >-
    Take the look-back window from the smallest-lead residual, not one residual per lead band: the
    smallest-lead offset beat the lead-band offset by 0.041 (interval -0.083 to -0.001) (stage 1
    sensitivities).
  - >-
    Test any inference-time level offset on windows with no load shift inside them, and report the
    windows that contain one separately, because the shift is copied forward for a week
    (clean-window
    check, scenarios table).
  - >-
    Do not score a level correction on seven month blocks and expect to resolve 0.05 points: the
    minimum detectable effect was 0.16 (C1).
  - >-
    Do not read a gain on injected shifts as a neighbour signal when the generator places the
    mirrored
    step on the real neighbours: the neighbour test is the real-row contrast of the real against a
    rewired graph (C4).
  - >-
    Build any ramp-recovery known-answer gate from the ceiling of the offset it tests: a 0.75 times
    14-day offset reaches a copied share of at most 0.57 on a 60-day ramp, so the gate of 0.6 was
    unreachable (KA3).
---


**We asked whether the slow drift in a substation's net demand can be corrected at forecast time,
with no new fitted network.** At each forecast origin we took the errors the stored forecast made
over the last 14 days, averaged them, and added a share of that average to the forecast. We also
split the errors by sunshine and by cold temperature, to see whether solar drift and cold-night
drift need separate corrections. We scored the corrected forecasts against what happened at 14
Primary substations from 2025-12 to 2026-06. Stage 2 then trained two small neural networks (post
hoc, see below) to decide how much of each offset to apply, using synthetic load shifts. The method
follows recipe R of [experiment
115](../00115-differentiable-physics-w1-diurnal-shape-state-on-ens/result.md), and the question
follows the advice of [experiment
690](../00690-slow-level-state-on-primaries-13-and-16-july-august/result.md) to fit the level
causally.

**The verdict is inconclusive, and the take-home message is that one look-back offset helps for two
days and hurts from day 5.** In stage 1, the lead-band global offset (G-lb) scored +0.026 CRPS
points of capacity worse than the stored forecast (99.17% interval -0.070 to +0.179, pooled over 14
Primaries), against a pre-registered bar of -0.05. It gained 0.127 points on day 1 and 0.096 on day
2, and lost 0.043 on days 5 to 7 and 0.155 on days 8 to 10. The interval is too wide to rule out
either a small gain or a small loss (minimum detectable effect 0.164). Stage 2 ran post hoc, because
stage 1 missed the pre-registered gate of 0.02 that allowed it, and the maintainer told the worker
to run it anyway.

## Results

**Stage 1: the global offset and the driver split do not beat the stored forecast.** CRPS is in
points of capacity, event-free rows, origins 2025-12 to 2026-06, and a negative difference favours
the first arm.

| Contrast | Pooled 14 | Signed-MW six | MVA-only eight |
|---|---|---|---|
| C1: G-lb minus stored forecast | +0.026 (-0.070 to +0.179) | +0.029 (-0.225 to +0.350) | +0.023 (-0.060 to +0.104) |
| C2: driver split (D) minus G-lb | +0.012 (-0.047 to +0.080) | -0.016 (-0.095 to +0.063) | +0.032 (-0.019 to +0.095) |
| G-lb minus recipe R (95% interval) | +0.044 (+0.011 to +0.077) | +0.077 (+0.048 to +0.111) | +0.021 (-0.019 to +0.061) |

- **Absolute scores (stage 1 rows, pooled 14, CRPS):** stored forecast 3.235, G-lb 3.260, D 3.272,
  month-by-half-hour climatology 4.772, manual heuristic 4.051. The normalised mean absolute error
  of the median is 5.838 for the stored forecast, 5.862 for G-lb, and 5.887 for D.
- **The gain sits at short lead.** [Figure 1](fig1_lead_bands.svg) shows every arm below zero on
  days 1 and 2 and most arms above zero by days 8 to 10; only the decayed offset R0-168 stays below
  zero.
- **Recipe R beat G-lb.** Recipe R (0.75 times the 14-day mean error by half-hour of day) scored
  0.044 better than G-lb (interval +0.011 to +0.077 on G-lb minus recipe R).
- **C5 (faithful prevailing arm minus stored prevailing rows)** is +0.150 (+0.026 to +0.417, pooled
  14), so the 7-day look-back is no better than the stored prevailing forecast.

![CRPS of each arm minus the stored forecast by lead band](fig1_lead_bands.svg)

**The offset copies a load shift forward.** On rows that follow a labelled event by up to 7 days,
G-lb scored 4.20 against 3.49 for the stored forecast, and on the 7 days after the 4 verified
transfers G-lb scored 6.21 against 4.07. Inside labelled events G-lb scored 7.73 against 7.98.
[Figure 2](fig2_events.svg) shows 8 events at Primaries with both metering types.

![Observed power and day-ahead medians around events](fig2_events.svg)

**Stage 2 (post hoc): the learned heads do not beat the no-network rule on real rows.** Origins
2026-01 to 2026-06, event-free rows, pooled 14, CRPS: stored forecast 3.255, N-rule 3.264, N-gate
(learned gate, no switching flag) 3.293, N-flag (switching probability) 3.322.

| Contrast (post hoc, 99.17%) | Difference (pooled 14) |
|---|---|
| C3: N-gate minus N-rule, real rows | +0.029 (-0.101 to +0.121) |
| C3: N-flag minus N-rule, real rows | +0.058 (+0.007 to +0.204; the monthly t test does not exclude zero) |
| C4: N-gate real minus rewired graph, real rows | -0.015 (-0.055 to +0.030) |
| C4: N-flag real minus rewired graph, real rows | +0.006 (-0.017 to +0.025) |
| C3 on injected shifts: copied share, N-gate minus N-rule | -0.077 (-0.104 to -0.051) |
| C3 on injected shifts: copied share, N-flag minus N-rule | -0.015 (-0.034 to +0.006) |
| C6: copied share, N-gate minus N-flag | -0.062 (-0.094 to -0.032) |

- **On injected shifts the gate copies less than the rule.** The copied share was 21.4% for N-rule,
  19.9% for N-flag, and 13.8% for N-gate (14.4% on the rewired graph). With the neighbour inputs
  zeroed N-gate copied 25.3%. [Figure 4](fig4_copied_share.svg) shows the shares.
- **This is a mechanism check, not evidence for neighbours.** The generator puts the mirrored step
  on the real-graph neighbours only, so the neighbour inputs hold the answer by construction. On
  real rows the real and rewired graphs differ by 0.015 or less (C4).
- **The flag detects any injected step, weakly.** The switching probability of N-flag had an area
  under the ROC curve of 0.60 on real rows and 0.68 on bank entries (0.61 and 0.69 on the rewired
  graph).

![Copied share of an injected shift](fig4_copied_share.svg)

![A day-ahead forecast and a 14-day forecast](fig3_examples.svg)

## Checks

- **Median minus mean forecast, per site:** the offsets shift all 11 quantiles by one amount, so no
  arm changes the shape of the forecast distribution. The audit found the median-minus-mean gap of
  -0.64 for both the stored forecast and G-lb.
- **Error of the median beside every CRPS contrast:** the mean absolute error of the median points
  the same way as CRPS for G-lb (+0.024), D minus G-lb (+0.025), and every stage 2 arm, so trigger 3
  did not fire.
- **Triggers that fired, and the audit.**
    - A control failed (trigger 1): the ramp known-answer gate (copied share at least 0.6) failed
      for every arm (0.41 to 0.52). The audit computed that a 0.75 times 14-day offset reaches at
      most 0.57 on that ramp, so the gate could not be passed by construction, and it does not
      separate the arms.
    - Leaving one Primary out flips the sign of C1 (trigger 4): dropping D16 turns +0.026 into
      -0.014. January alone makes the pooled mean positive (+0.327 in January, about -0.024 without
      it).
    - A nested arm beats the arm that contains it (trigger 5): G-lb beats D in the MVA-only group
      (+0.032 for D), and the smallest-lead offset (G-sl) beats G-lb by 0.041 (interval -0.083 to
      -0.001), so the lead-band choice made before scoring was the worse variant.
    - An independent Sonnet audit re-derived C1, C2, and the stage 2 real-row contrasts from the
      stored predictions and reproduced every real-row number within bootstrap noise. It did not
      re-run the bank contrasts or the stage 1 known-answer tests. Its three requested reports
      (clean-window C1, the D16 and January drivers, and the ramp ceiling) are in this section.
- **Clean-window C1.** On rows whose 14-day look-back window holds no event, G-lb minus the stored
  forecast is -0.026 (interval -0.079 to +0.020, pooled 14). On rows whose window holds an event it
  is +0.337. The pre-registered verdict uses the all-event-free reading above.
- **Known-answer tests that passed:** KA1a (the offset removes 98.4% of an additive step at 14 days
  and 48.4% at 7 days, against an exact 0.984 and 0.484), and KA2 (the driver split removes 87% of a
  PV-shaped error at midday and leaves 0.3 points at night). The injection sizes were raised from 10
  to 30 points (step), 20 points (PV), and 3 points per unit (cold) after the readability check
  showed them under 5 times the noise, and before any contrast was read.
- **Placebos.** PL1 (a non-neighbour Primary's window) scored +0.453 worse than the stored forecast,
  so a wrong-series offset hurts. PL2 (shuffled drivers within each 14-day window) scored +0.048,
  against +0.038 for D.
- **Weather look-ahead.** Target weather comes from the 00:00 UTC ensemble run of the origin's day,
  available at 09:00 UTC, and the look-back window ends 6 hours before the origin. The script
  asserts this row by row. Example: Primary 17 (D5), origin 2026-03-10 12:00 UTC, valid time
  2026-03-15 13:00 UTC, lead 121 h from the origin and 133 h from the 00:00 UTC run, window
  2026-02-24 06:30 to 2026-03-10 06:00 UTC.
- **Unlogged-event caveat.** NGED's logs may miss events, so the event-free rows may hold events no
  label covers, and the real-event slices undercount.
- **Not done.** The hindsight ceilings of experiments 690 and 688, the survival-curve variant
  P-surv, the curtailment cost, and bootstrap intervals for the lead-band and day-of-month splits
  were planned in the design and not implemented. The scored prediction sets carry a `dirty` flag in
  their manifests because the scoring fixes were committed after the fits; the fitted code is the
  committed code of the commit that follows.

## Method

Architecture:

- **G-lb:** for each forecast origin, the mean of the residuals (cleaned power minus the stored
  normal-running median) over the 14 days that end 6 hours before the origin, taken per lead band
  from the forecasts issued for those half-hours, times 0.75, added to all 11 quantiles at every
  lead.
- **D:** a ridge regression of the window residuals on the ensemble-mean shortwave radiation over
  800 W/m2 and the cold index max(0, 5 C minus temperature) over 5 K; the correction at the target
  time uses the target's ensemble-mean drivers; a slope is dropped to zero when fewer than 24 window
  half-hours have its driver above 0.25.
- **P-7d, P-cp:** the same form with factor 1.0, a 7-day window (P-7d) or a change-point split of a
  28-day window (P-cp), with the window ending at 03:00 UTC to match the stored prevailing rows.
- **N-rule:** G-lb's offset times a weight that is 0 when a step larger than 3 noise units arrived
  within 24 hours, else 1.
- **N-flag:** a network with 2 hidden layers of 32 units outputs a switching probability p (trained
  with a binary cross-entropy term of weight 0.05 against the generator's transfer flag) and applies
  the slow level in full and (1 minus p) times the step part.
- **N-gate:** a network with 2 hidden layers of 32 units outputs one gate in [0, 1] for each of 16
  look-back offsets (global, irradiance, cold, and neighbours' offsets, over windows of 1, 3, 7, and
  14 days, exponentially weighted), with the lead time as an input, and applies the sum of gate
  times offset; CRPS loss only.
- **Neighbours:** NGED's interconnection list as coded in experiment 400, or a degree-preserving
  rewired graph.

Training regime:

- **Stage 1 has no training.** Stage 2 trained on the injected-shift bank of [experiment
  667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md) and real
  rows of the months before each test month, with a 7-day embargo, 10 seeds, folds 2026-01 to
  2026-06, and early stopping on valid-time weeks inside the training period. Half the steps use
  own-only additive ramps of 10 points over 60 days.
- **Compute:** 240 fits (2 arms, 2 graphs, 6 folds, 10 seeds) took 37 minutes on 2 CPU cores and one
  GPU share at about 1 GB; stage 1 needed under 5 minutes on CPU.

Measurement:

- **Scored rows:** origins 12:00 UTC daily, 2025-12 to 2026-06 (stage 1) and 2026-01 to 2026-06
  (stage 2), leads 0 to 240 h, valid times before 2026-07-01, cleaned power, event-free rows, 14
  Primaries in two groups (signed-MW six and MVA-only eight; D15 and D18 report-only).
- **Metrics:** CRPS as twice the mean pinball loss over the 11 quantiles, mean absolute error of the
  median, q01 to q99 coverage and pinball loss, against cleaned power (clean power without the
  injected shift on the bank).
- **Intervals:** month-block bootstrap with 20,000 resamples at 99.17% (six contrasts), plus a t
  test on the monthly paired differences; bank contrasts resample (month, Primary) clusters with
  seeds inside.
- **Decision rule:** see the front matter. C1 needed its interval wholly below -0.05 in both groups.

## Files

- `ml-experiments/00706-.../e00706_common.py`: loaders, dense layout, ridge by cumulative sums,
  residual tables, group and event masks.
- `e00706_residuals.py`: stage 0; builds the residual table, runs the look-ahead assertions, prints
  the example.
- `e00706_arms.py`: stage 1; writes the prediction sets `g`, `g-sl`, `g-refit`, `d`, `d-pv`,
  `d-cold`, `d-plus`, `p`, `p-7d`, and the window variants.
- `e00706_known_answer.py`: KA1a, KA1b, KA2, PL1, PL2, the perturbation test, and event counts.
- `e00706_score.py`: stage 1 contrasts and tables, written to
  `data/ml-experiments/00706-.../tables/`.
- `e00706_s2_data.py`, `e00706_s2_models.py`, `e00706_s2_fit.py`, `e00706_s2_score.py`: stage 2
  caches, networks, fits, and scoring.
- `e00706_figures.py`: the four figures beside this page.
- `data/ml-experiments/00706-.../predictions/`: private stored forecasts (stage 1 and stage 2 sets)
  and `s2/models/` (state files); read from the one-year backtest files of [experiment
  668](../00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md).
- Audit and design notes: `data/ml-experiments/briefs/level_tracking/`.
