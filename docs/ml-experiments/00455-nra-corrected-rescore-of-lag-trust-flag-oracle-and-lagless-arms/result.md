---
id: "00455"
slug: nra-corrected-rescore-of-lag-trust-flag-oracle-and-lagless-arms
title: >-
  Does an exact record of which power readings lie inside a load-shift event help a lagged and
  lagless mixture when the forecast is scored against the normal running arrangement?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: done
kind: other
protocol: demand-v2
size: >-
  simple. Stored: one table of per-cell contrasts in this experiment's private folder. No fit and
  no new forecast: it rescores stored forecasts of experiments 404 and 445 against experiment 435's
  NRA-corrected actuals, on one CPU core. One design (experiment 404's audit script, extended to
  more arms). Production serving path: not touched (this repository holds none). Degradation rule:
  not touched. Callers: the script imports experiment 404's modules.
hypothesis: >-
  Scored against NRA-corrected actuals instead of metered actuals, the hard flag-oracle mixture
  (O-hard) scores lower than B-mix (a fixed 50/50 mixture of the lagged and lagless forecasts)
  on rows inside real load-shift events, so detecting events would matter under the NRA target.
prediction: >-
  Written before any score is read. Experiment 445 found O-hard +0.465 worse than B-mix against
  metered actuals at the signed-MW six. P1. Against NRA-corrected actuals the contrast moves
  towards zero by under 0.5 points of capacity but stays above zero at depth factor 1.0, so the
  445 refutation stands. P2. The lagless head scores worse than B-mix at every depth. P3. Dropping
  the largest event of each month changes the sign of at least one O-hard contrast.
decision_rule: >-
  Contrast = arm minus B-mix on the same rows inside padded real events (target inside a padded
  event or in the 96 half-hours after one), mean of month means (seeds within a month averaged
  first), signed-MW six as the headline and the MVA-only eight reported apart. Interval = month-block
  bootstrap (resampling the 5 months, 20,000 draws) at the Bonferroni level for 48 contrasts (8 arms
  against B-mix, at 3 depth factors, in 2 groups; the with-and-without-largest-event variants
  repeat the same contrast and are not counted again). If O-hard minus B-mix is below zero with a
  Bonferroni interval that excludes zero, with and without the largest event of each month, at
  all three depth factors (0.75, 1.0, 1.25), detection matters under the NRA target. Otherwise
  the 445 refutation stands under both targets. A 5-month bootstrap at the Bonferroni level cannot
  resolve a tail beyond the smallest and largest resampled mean, so the rule is conservative: a
  contrast that passes it is large relative to the spread of months. The minimum detectable effect
  (2.8 times the standard error of the monthly differences) is reported beside each contrast.
  The smallest gain worth reporting is zero, because the O-rule is a diagnostic ceiling and is not
  ported. Interim and unaudited.
benchmark: >-
  B-mix, experiment 404's stored fixed 50/50 mixture of A0-inj and the lagless table.
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of capacity from 11 quantiles, as in experiment
  404's report; no widening beyond what the stored forecasts carry.
verdict: refuted
best_arm: >-
  None beat B-mix inside real events at the signed-MW six. The closest is O-floor (exact flags, lagless
  weight 0.5 plus half the flagged share) at +0.01 (Bonferroni interval -0.23 to +0.26) at depth 1.0.
headline: >-
  Scored against NRA-corrected actuals instead of metered actuals, a mixture that switches to the
  lagless forecast whenever an exact flag says a lag lies in a load-shift event (O-hard) is still
  0.16 to 0.19 points of capacity worse than B-mix at the six Primaries metered in signed megawatts
  (95% month-block intervals exclude zero at all three depth factors, Bonferroni intervals for 48
  contrasts include zero), and stays worse when the largest event of each month is dropped
  (Bonferroni intervals exclude zero). Interim and unaudited.
lessons: >-
  Correcting the actuals for the load that real events moved shrinks O-hard's loss against B-mix
  from 0.38 to 0.17 points but does not reverse it, so the 445 refutation stands under both targets.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  Knowing exactly which recent power readings lie inside a load-shift event still does not make a
  switching mixture beat the fixed 50/50 mixture, even when the forecast is scored against the
  normal running arrangement instead of the metered flow. The gap shrinks from 0.38 to 0.17 points
  of capacity at the six Primaries metered in signed megawatts and stays above zero at every depth
  factor, so the lag-trust line should not wait on better event detection. Five months and about 25
  events can detect a difference of only about 0.2 points. Every number is interim and unaudited.
advice:
  - >-
    Keep the fixed 50/50 mixture as the reference for any lag-trust test, and score inside-event rows
    against NRA-corrected actuals as well as metered actuals (this page).
  - >-
    Do not build an event detector for the mixture rule on the argument that metered actuals mislead:
    exact flags lose to B-mix under the NRA target too (this page, O-hard).
code: ml-experiments/00455-nra-corrected-rescore-of-lag-trust-flag-oracle-and-lagless-arms/
---

**Knowing exactly which recent readings lie inside a load-shift event still does not make a mixture
beat B-mix, even when forecasts are scored against the normal running arrangement (NRA).** B-mix is
the fixed 50/50 mixture of the plain network and the lagless forecast. At the six Primaries metered
in signed megawatts (the signed-MW six), O-hard (which switches to the lagless forecast whenever an
exact flag marks a lag) scores 0.16 to 0.19 points of capacity worse than B-mix at the three depth
factors of experiment 435's NRA correction, against 0.38 worse on metered actuals. The numbers are
interim and unaudited.

This experiment fits nothing. It rescores forecasts stored by experiments 404 and 445 on the rows
inside real events, adding the lagless head and the three flag-oracle mixtures of experiment 445 to
the rescore that experiment 404's audit ran for its own arms.

- **Row 1 of the review is answered by experiment 404's audit**, not here: `e00404_audit.py` and the
  files `cells.parquet`, `pooled_event_weighted.csv`, and `event_counts.csv` in 404's `work/audit/`
  folder. It found B3 and B6 behind B-mix when months are weighted by events (+0.13 and +0.15), and
  that the 1.15 widening and table P move every entry by 0.03 or less, so B-mix minus A0-inj on
  clean rows stays near +0.29.
- **Harm row.** The scoring code (experiment 428's window flags) calls a row a harm row when its
  target lies outside every padded real event and outside the 96 half-hours after one, but any
  observed half-hour of the three day-windows of its lags lies inside a padded event. The 404 plan
  wrote the definition on the three origin lags only. A clean row has no real event near it. A row
  "inside real events" has its target inside a padded event or in the 96 half-hours after one; this
  page scores those rows.
- **Event counts.** The five scored months overlap 46 corrected events (21 at the signed-MW six, 16
  at the MVA-only eight, 9 at D15 and D18) and 42 padded runs. Experiment 445 counts 49 and
  experiment 435 counts 41 with different windows. At the six the bootstrap clusters are 18 on harm
  rows, 25 inside events, and 37 on clean rows. The whole history holds 184 corrected events (170
  padded runs).
- **Unlogged-event caveat.** NGED says load-shift events may be missing from its logs. The 46 events
  are a lower bound, so rows called clean may hold unlabelled real events, and the harm and
  inside-event rows undercount. The NRA-corrected actuals correct only labelled events (a measured
  deviation for 34 of 184, the labelled depth for the other 150).

![Each arm minus B-mix inside real events at the signed-MW six](figure_1_signed_six.svg)

![Each arm minus B-mix inside real events at the MVA-only eight](figure_2_mva_eight.svg)

![The same at the signed-MW six with the largest event of each month dropped](figure_3_signed_six_drop.svg)

## Results

- **Rows and scores.** Rows are the real-event rows of forecasts with leads of 0 to 240 hours, as in
  404's audit. The score is the mass-weighted pinball loss in points of capacity from 11 quantiles.
  A contrast is the arm's score minus B-mix's on the same rows, mean of month means (seeds within a
  month averaged first), so negative is better than B-mix. B-mix scores 13.13 at the signed-MW six
  and 9.36 at the MVA-only eight at depth 1.0 (5.94 and 7.76 on metered actuals). B6 has fit seed 0
  only; the other arms have seeds 0 to 2 in 2026-06.
- **Intervals.** Resampling the months gives a 95% interval and a Bonferroni interval for 48
  contrasts (8 arms against B-mix, at 3 depth factors, in 2 groups; the metered column and the
  drop-largest variants repeat contrasts and are not counted again). Five months (four at the
  MVA-only eight, which has no 2026-06 real-event rows) cannot resolve the Bonferroni tail beyond
  the smallest and largest resampled mean, so a Bonferroni interval is nearly the range of the
  monthly values and is conservative. The minimum detectable effect (2.8 times the standard error of
  the monthly differences) is in the last column.

Signed-MW six, depth factor 1.0, inside real events (25 events, 5 months):

| Arm | Arm score | Arm minus B-mix [95%] (Bonferroni) | Largest event of each month dropped (Bonferroni) | Minimum detectable effect |
|---|---|---|---|---|
| A0-inj | 13.48 | +0.35 [+0.19, +0.62] (+0.17, +0.76) | +0.20 (+0.04, +0.44) | 0.38 |
| S1 | 13.55 | +0.42 [+0.19, +0.66] (+0.12, +0.76) | +0.26 (+0.01, +0.47) | 0.38 |
| B3 | 13.21 | +0.08 [-0.16, +0.31] (-0.30, +0.42) | +0.34 (+0.16, +0.64) | 0.40 |
| B6 | 12.56 | -0.58 [-1.70, +0.23] (-2.82, +0.45) | +0.23 (-0.02, +0.44) | 1.64 |
| lagless | 13.22 | +0.10 [-0.09, +0.23] (-0.18, +0.30) | +0.07 (+0.01, +0.15) | 0.26 |
| O-hard | 13.30 | +0.17 [+0.04, +0.27] (-0.04, +0.28) | +0.21 (+0.10, +0.43) | 0.20 |
| O-soft | 13.28 | +0.15 [+0.02, +0.24] (-0.10, +0.26) | +0.15 (+0.01, +0.44) | 0.18 |
| O-floor | 13.14 | +0.01 [-0.13, +0.16] (-0.23, +0.26) | +0.03 (+0.00, +0.10) | 0.21 |

MVA-only eight, depth factor 1.0 (26 events, 4 months):

| Arm | Arm score | Arm minus B-mix [95%] (Bonferroni) | Largest event of each month dropped (Bonferroni) | Minimum detectable effect |
|---|---|---|---|---|
| A0-inj | 10.36 | +1.00 [+0.85, +1.15] (+0.77, +1.20) | +0.63 (+0.38, +0.89) | 0.27 |
| S1 | 10.16 | +0.80 [+0.66, +0.99] (+0.63, +1.07) | +0.57 (+0.42, +0.77) | 0.27 |
| B3 | 10.33 | +0.97 [-0.03, +1.97] (-0.05, +2.19) | +0.91 (+0.25, +1.73) | 1.63 |
| B6 | 9.85 | +0.49 [+0.32, +0.65] (+0.32, +0.72) | +0.27 (+0.00, +0.49) | 0.28 |
| lagless | 9.24 | -0.12 [-0.42, +0.17] (-0.51, +0.31) | -0.02 (-0.15, +0.19) | 0.51 |
| O-hard | 9.89 | +0.53 [+0.09, +0.85] (-0.12, +0.94) | +0.33 (+0.22, +0.39) | 0.64 |
| O-soft | 9.89 | +0.53 [+0.22, +0.82] (+0.11, +0.92) | +0.20 (-0.01, +0.34) | 0.46 |
| O-floor | 9.29 | -0.07 [-0.25, +0.03] (-0.34, +0.05) | -0.01 (-0.06, +0.02) | 0.25 |

- **O-hard minus B-mix at the signed-MW six** is +0.19 [+0.07, +0.29] at depth 0.75, +0.17 [+0.04,
  +0.27] at 1.0, and +0.16 [+0.01, +0.25] at 1.25 (95% intervals, positive means B-mix is better).
  With the largest event of each month dropped it is +0.26, +0.21, and +0.17, with Bonferroni
  intervals above zero at all three (+0.10 to +0.56, +0.10 to +0.43, +0.08 to +0.35). The rule for
  "detection matters" needs a negative contrast, so it is not met at any depth.
- **Metered actuals** give O-hard +0.38 [+0.21, +0.62] at the six. The NRA correction therefore
  removes about half of the gap but not the sign.
- **At the MVA-only eight** O-hard is +0.50 to +0.55 behind at the three depths (Bonferroni
  intervals include zero for O-hard and exclude it for A0-inj, S1, and B6). The lagless head and
  O-floor are the only arms ahead of B-mix there (-0.12 to -0.22 and -0.07 to -0.12), and no
  interval excludes zero.
- **Against the predictions.** P1 holds (the O-hard contrast stays above zero, moving from +0.38 to
  +0.16 to +0.19). P2 holds at the six (the lagless head is +0.08 to +0.11 behind at every depth,
  with intervals that include zero). P3 fails: dropping the largest event of each month never
  changes the sign of an O-hard contrast.
- **Checks.** No new forecast is written, so the median-minus-mean and mean-forecast-error checks do
  not apply. None of the audit triggers fired: the numbers for A0-inj, S1, B3, and B6 reproduce
  404's audit tables (+0.35, +0.42, +0.08 at depth 1.0 at the six).

## Caveats

- **Few months and events.** Five months and 25 events at the six cannot detect a difference below
  about 0.2 points of capacity for the oracle arms and 0.4 or more for the others, and the MVA-only
  eight has four months.
- **Depth.** The NRA correction uses the labelled depth for 150 of 184 events, so its error is
  repeated by the three depth factors, which scale the same correction.
- **Tails.** This page scores CRPS only. The tails and the avoidable-curtailment cost are not
  rescored; experiment 445 found B-mix's q99 covers 71.8% inside real events at the six.
- **Row set.** Rows with leads above 240 hours are excluded to match 404's audit; experiment 445's
  +0.465 includes them, which is why the metered figure here is +0.38.
- **Unlogged events** may sit in clean rows and make every event slice an undercount (see above).

## What to try next

- **Stop work on event detection for the mixture rule.** Exact flags lose to B-mix under both
  targets.
- **Row 3 of the review** (a mixture and ensemble family) and the ambiguous-world test are the live
  lines; this page changes neither.
