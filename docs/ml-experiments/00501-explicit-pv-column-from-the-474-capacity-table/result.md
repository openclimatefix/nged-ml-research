---
id: "00501"
slug: explicit-pv-column-from-the-474-capacity-table
title: >-
  Does an explicit solar-output column, built from a capacity fitted only on earlier months, improve
  the demand forecast at the six Primaries metered in signed megawatts?
theme: solar-disaggregation
also_relevant_to:
  - differentiable-physics-embedded-generation
  - production-reference-and-best-guess
status: done
kind: other
protocol: demand-v2
date: 2026-11-02
verdict: refuted
production_status: not-for-production
adopted_in: none
size: >-
  medium. Stored: fitted networks, forecasts, and the per-fold capacity table, in this experiment's
  private folder. Serving path: not touched (this repository holds none). Degradation rule: not
  touched. Designs: one defensible (experiment 425's arm A with one added input column), because
  the brief fixes the column, the folds, and the seeds. Callers: this folder's scripts, which import
  experiments 269, 474, 405, 397, 425 and the reviewed `studies.*` modules. Reviews, all Sonnet and
  all before the first fit or score that is quoted: one plan review, one code review, one results
  review, and one write-up review plus one prose review.
hypothesis: >-
  Experiment 425's pooled network (arm A) already reads irradiance, so it can learn solar output
  from weather alone. Adding one input column, the solar output that a fixed capacity would give
  (the capacity fitted by experiment 474's method on months before the fold, times experiment 405's
  physics shape of the ensemble-mean weather), lowers the CRPS of the sunny-midday half-hours at the
  signed-MW six.
prediction: >-
  Written before any fit. The prior is low: experiments 349, 352, and 416 found a flat response to
  explicit solar information. The prediction that could fail is that the CRPS of sunny-midday
  half-hours (CAMS clear-sky index of 0.7 or more, 10:00 to 15:00, days 1 to 10) at the signed-MW
  six falls by at least the minimum detectable effect (printed before any contrast; experiment 416
  found 0.28 to 0.37 points with 5 seeds, so with 3 seeds the value is larger). A gain would not be
  attributable to solar capacity, because experiment 474's capacities fail the placebo gate at
  every rung.
decision_rule: >-
  Three contrasts, all PV column minus arm A, all at the signed-MW six on the 5 folds of experiment
  449 (2025-10, 2025-12, 2026-01, 2026-03, 2026-06) and 3 seeds: C1 CRPS of the sunny-midday
  half-hours, C2 CRPS of all daylight half-hours, C3 CRPS of all half-hours at days 1 to 10. The
  Bonferroni count is 3 (two-sided level 98.33%), from month-block bootstrap resamples with equal
  rows. A contrast is a gain only if its interval excludes zero on the negative side and the point
  estimate is at least the minimum detectable effect. The smallest gain worth porting is the
  minimum detectable effect, because the column adds a per-Primary capacity table and an annual
  capacity refit to production. Stop rule: stop after this arm; the 10th and 90th percentile
  capacity arms run only if the maintainer asks. Every verdict carries the statement that
  placebo-failed capacities make any gain unattributable to solar capacity.
benchmark: >-
  Experiment 425's arm A on the same 5 folds, seeds 0 to 2 (stored), with experiment 425's
  references (`bg2`, `manual_heuristic`, `climatology`, `S0M`) scored beside it.
row_set_id: >-
  Experiment 425's scoring rows (months 2025-10, 2025-12, 2026-01, 2026-03, 2026-06; days 1 to 10),
  with flagged exact-zero half-hours dropped and the experiment 399 start cut for D15 and D18.
  July 2026 stays unused.
metric: >-
  CRPS (points, equal weight per Primary) by slice and group; q01, q05, q95, q99 coverage and
  pinball loss; cold winter nights; avoidable curtailment cost at f = 0.9, 0.75, and 0.5; the
  12 to 36 hour band.
best_arm: >-
  Neither arm: experiment 425's arm A (3.220 CRPS points at the signed-MW six, days 1 to 10) and the
  column arm (3.219) are indistinguishable.
headline: >-
  The explicit solar-output column changed the sunny-midday CRPS at the signed-MW six by -0.014
  points (98.33% interval -0.027 to +0.000), against the 0.225 points the experiment could detect.
take_home_message: >-
  Giving the demand network an explicit solar-output column, built from a capacity that was fitted
  only on earlier months, did not improve the forecast at the six Primaries metered in signed
  megawatts. On sunny middays the CRPS moved by at most 0.027 points, about a tenth of the 0.225
  points the experiment could detect, and every other slice and the tails stayed unchanged. The
  network already reads irradiance, so the capacities from experiment 474, which fail the placebo
  test, are not worth carrying into the serving forecast.
lessons: >-
  A fixed capacity times a physics shape adds nothing that arm A's irradiance input and Primary
  embedding do not already hold. The capacities stay unvalidated, and this experiment gives no
  reason to try the 10th and 90th percentile arms.
advice:
  - >-
    Do not add a fixed explicit solar-output column to the demand network at the signed-MW six;
    sunny-midday CRPS moved by at most 0.027 points against 0.225 detectable (C1).
  - >-
    Keep the network's own irradiance input and Primary embedding as the solar model; an explicit
    capacity-times-shape column duplicates them (C1 to C3).
  - >-
    Judge any future solar capacity by a known-answer or placebo test, not by the demand forecast
    score, because a flat forecast response cannot validate a capacity (C1, leave-one-out).
---

**Giving the demand network a column of "solar output if the capacity were X" did not improve the
forecast at the six Primaries metered in signed megawatts.** We asked whether one extra input, the
output that a solar capacity fitted by experiment 474's method on earlier months would give under the
forecast weather, lowers the error of experiment 425's best network (arm A). We fine-tuned arm A with
the column on 5 test months and 3 seeds, and compared it with arm A's stored forecasts. On sunny
middays the continuous ranked probability score (CRPS, a score for a whole forecast distribution, in
the same units as megawatts) moved by -0.014 points. The 98.33% interval is -0.027 to +0.000, and
the experiment could detect 0.225 points. The verdict is `refuted`: the largest gain the data allow
is about a tenth of the predicted one.

## Figures

![Contrasts](figure_1_contrasts.svg)

![Capacity by fold](figure_2_capacity_by_fold.svg)

![Winter 14 days](figure_3_example_winter_14_days.svg)

![Winter day-ahead](figure_3_example_winter_day_ahead.svg)

![Summer 14 days](figure_3_example_summer_14_days.svg)

![Summer day-ahead](figure_3_example_summer_day_ahead.svg)

## Results

- **Arms.** Arm A (experiment 425, seeds 0 to 2, stored) and the column arm (arm A fine-tuned from
  the same pretrained weights, 3 seeds, 5 folds: 2025-10, 2025-12, 2026-01, 2026-03, 2026-06).
  Monthly refits, learning rate 2e-4, forecasts widened by 1.15. 15 fine-tunes of 28 to 60 seconds
  each on the GPU.
- **Column.** Capacity per Primary and fold (3.0 to 4.1 MW AC at the six, from
  `e00501_capacity.py`; the full-history fit reproduces experiment 474's stored values exactly)
  times experiment 405's unit shape, divided by 3 MW; zero at the other 10 Primaries.
- **Tested contrasts** (column minus arm A, CRPS, days 1 to 10, signed-MW six, Bonferroni count 3,
  level 98.33%, rows equal across arms):

| Slice | Difference | Interval | Detectable |
|---|---|---|---|
| C1 sunny-midday half-hours (34,432) | -0.014 | -0.027 to +0.000 | 0.225 |
| C2 all daylight half-hours (158,999) | +0.000 | -0.001 to +0.001 | 0.134 |
| C3 all half-hours (429,350) | -0.001 | -0.003 to +0.000 | 0.104 |

- **Report-only groups.** MVA-only eight: C1 +0.008 (+0.001 to +0.016), C2 -0.000, C3 +0.000. D15 and
  D18: C1 -0.015 (-0.038 to +0.002), C2 -0.013 (-0.033 to -0.001), C3 -0.005 (-0.011 to -0.000).
  The column is zero at the eight, so their small shifts come from the shared network weights.
- **By fold (C1, points).** 2025-10 -0.030, 2025-12 -0.008, 2026-01 +0.012, 2026-03 -0.025, 2026-06
  -0.011. Leaving out each Primary in turn leaves C1 between -0.008 and -0.018.
- **Absolute scores** (signed-MW six, days 1 to 10, rows shared with the references): arm A 3.220,
  column arm 3.219, BG2 3.428, monthly XGBoost 3.569, manual heuristic 4.134, climatology 4.392.
  Sunny-midday CRPS by fold: 4.41, 3.73, 5.59, 4.08, and 3.07 for arm A.
- **Lead bands** (signed-MW six): 12 to 36 hours 2.822 (A) and 2.820 (column); 36 hours to 10 days
  3.268 and 3.268.
- **Tails** (signed-MW six, days 1 to 10): coverage at q01, q05, q95, q99 is 0.007, 0.022, 0.895,
  0.962 for the column arm and 0.007, 0.022, 0.895, 0.963 for arm A; pinball loss at q99 is 0.326
  and 0.325. Cold winter nights: 5.049 for both.
- **Curtailment** (avoidable cost over the six, pounds, as the scoring machinery reports it; the
  manual-heuristic baseline is the maintainer's unconfirmed understanding): at f = 0.9, 6.857 million
  (column) and 6.865 million (arm A); at 0.75, 16.94 and 16.97 million; at 0.5, 38.16 and 38.21
  million; the manual heuristic at 0.5 is 61.66 million.

## Checks

- **Median minus mean forecast per Primary** (mean of the 11 quantiles, MW) differs between the arms
  by under 0.001 at every Primary, and the mean-forecast absolute error moves together with the
  median's.
- **Triggers.** None fired: no control failed (the capacity fit reproduces experiment 474), the
  result does not contradict an earlier experiment (349, 352, and 416 found a flat response), the
  quantile score and the mean-forecast error agree, leaving one Primary out does not flip the sign,
  the column arm contains arm A and does not beat it by more than its interval, and no sister
  experiment on the same rows exists to compare interval widths.
- **Unattributable.** Experiment 474's capacities fail the placebo gate at every rung, so a gain
  would not have shown that the capacities are right, and the absence of a gain does not show that
  they are wrong. Experiment 474's D5 (export limit), D9 and D16 (pinned capacity too high) caveats
  apply.
- **Rows.** Fine-tuning rows dropped per fold: 68,680 for the D15 and D18 start cut and 2,155 to
  2,925 flagged exact-zero half-hours. 1,179 scoring actuals were dropped for a flagged exact zero
  across the 12 stored months. Clean rows may hold events that NGED's logs do not label.
- **Leaks left.** The recovery divisor, the farm-output normalisation, and `u99` use data after the
  fold (each is a constant scale factor), and the 2025-10 fold identifies its capacity increment from
  three months. Weather at lag times is arm A's; its past-weather caveat applies unchanged.

## Serving table

- **Stored:** one capacity per Primary at the six (6 numbers), refitted with each monthly network
  refit (30 fits of about 13 seconds each in this experiment) or once a year if the capacity is
  held.
- **Computed at serving:** experiment 405's physics shape of the ensemble-mean shortwave radiation
  and temperature per half-hour, times the capacity.
- **Cost against gain:** a per-Primary capacity table and an annual or monthly capacity refit, for
  no measurable gain.

## What to try next

Nothing in this direction: a capacity cannot be validated by the demand forecast score, so the
10th and 90th percentile arms were not run. Experiment 479's anomaly estimator remains the
candidate for a capacity that passes a known-answer test.
