---
id: "00388"
slug: xgboost-lagged-power-column-sets-weekly-demand
title: >-
  Which lagged-power columns improve the simple weekly XGBoost demand model at the 16 Primaries?
theme: lagged-power-and-error-correction
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v1
size: >-
  medium. Stored: forecasts and small result files in this experiment's private folder only;
  experiments 87, 377, 379, and 381 are read and never written. Designs: more than one would test
  the hypothesis (the anomaly window, the set of analogue weeks, and the summary statistic are all
  choices); the plan fixes each before any score is read. Callers: the fit script imports
  experiment 87's folder, the scoring script imports experiment 377's folder and
  `studies.eval_protocol`. This repository holds no serving path or degradation rule, so neither is
  touched.
hypothesis: >-
  A single XGBoost model per Primary, refitted weekly on the ENS control member with experiment 87's
  recipe (the BG2 weekly member), scores better at days 1 to 10 when its three origin lags are
  joined by further lagged-power columns: the six weekly analogues (arm A), their mean and median
  (arm B), or two anomaly columns that subtract a 4-week same-weekday median (arm C).
prediction: >-
  Written before any score is read. P1. Arm A and arm B move the days 1 to 10 CRPS by between
  -0.10 and +0.05 points of effective capacity pooled per group, because experiment 30 found -0.30
  on the old rows but the gain vanished once a 14-day error correction was added, and the current
  recipe has no correction. P2. The gain is largest in the 72 to 168 h band and near zero in the
  0 to 24 h band, where the origin lags already carry the information. P3. Arm C moves the score by
  under 0.05 points in either direction, because experiments 5, 18, and 23 found that lag
  anomalies did not beat a single-stage model. P4. No arm moves the D2 and D4 placebos by more than
  the half-width of that Primary's interval, unless the same arm moves D8.
decision_rule: >-
  Every arm is a column change with no second model and no blend, so the threshold for porting an
  arm is any gain whose 99% interval excludes zero (the cost of porting a column is nil). Weekly
  refits throughout, `colsample_bytree` and `colsample_bynode` fixed at 1.0 in every arm. Contrasts,
  registered before any score is read, all CRPS at days 1 to 10 in points of effective capacity,
  paired month-resampled bootstrap of `studies.eval_protocol`, 20,000 resamples. The level is
  Bonferroni for k = 6 contrasts: 99.83% pooled per group, 95% per Primary. C1: arm A minus the
  baseline. C2: arm B minus the baseline. C3: arm C minus the baseline. C4: arm D (B plus C)
  minus the baseline, fitted only if C2 and C3 each lie wholly below zero in some group. C5: arm E
  minus the baseline. C6: arm F minus the baseline. C7: arm F minus arm E (read once, not counted
  in the Bonferroni level). Each
  contrast is read per group (signed-MW six, MVA-only eight with and without D8; D15 and D18 report
  only), per lead band (0 to 24, 24 to 72, 72 to 168, 168 to 240 h), and per Primary, with
  leave-one-out and the tail, cold-night, sunny, and load-shift slices beside it. An arm is a
  candidate when its interval lies wholly below zero in a group and it breaks no tolerance of the
  protocol. Placebos: D2 and D4 for any gain at D8. No adoption claim.
benchmark: >-
  The baseline arm (refitted at all 52 weekly refits; experiment 87's stored weekly `n1h` member is a reference), BG2, the default 3 GAM stack (experiment 377), `manual_heuristic`, and `climatology`, on
  `fold_a_daily_gap_v1`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles, mean absolute error of the
  median, tail pinball and coverage, and twCRPS, in points of effective capacity, from
  `studies.eval_protocol`; leads 0.5 to 336 hours; headline band days 1 to 10.
code: ml-experiments/00388-xgboost-lagged-power-column-sets-weekly-demand/
production_status: not-for-production
adopted_in: none
date: 2026-10-29
---

**Plan (status `planned`).** The question is whether giving the plain weekly XGBoost demand model
more lagged-power columns than its current three improves the forecast. The three origin lags are
the largest known input: removing them costs D8 +0.550 points (+0.237 to +0.944) and D2 +0.109 and
D4 +0.283 (experiment 371). We test three further sets of columns, each a column change that needs
no second model, so a gain is cheap to port. The design follows experiment 30, which found the six
weekly analogues helpful on older rows ([Experiment 30](../00030-lagged-power-feature-sets-primary-demand/result.md)),
and the fitting and scoring code of experiments 381 and 87.

## Pre-registered arms

All arms refit weekly with experiment 87's `n1h` recipe (half-life 0.5 years, 500 rounds, `eta`
0.05, whole history), one seed. `colsample_bytree` and `colsample_bynode` stay at 1.0, because
column subsampling would hand a wider arm a free win. A column is usable only when its meter reading
is at least 3.5 hours older than the forecast origin; otherwise it is null.

- **Baseline.** The three origin lags as `e00087_common.origin_lags` defines them: the newest usable
  reading, the reading at the latest earlier time with the valid time's half-hour of day, and the
  reading at the latest earlier time with the valid time's weekday and half-hour. The reproduction
  gate (experiment 383's: mean CRPS within 0.01 points at every Primary) refits the baseline at 4
  of the 52 weekly refits and compares it with experiment 87's stored weekly `n1h` member. The
  gate failed at one Primary: D22 differed by 0.0195 points, and the other 15 Primaries differed
  by at most 0.0045 (the GPU fit is not bit-reproducible, and the largest median gap sat at the
  refit of 2025-12-30). Because a stored baseline would carry that noise into every contrast, all
  contrasts use a baseline refitted at all 52 refits with the same code and settings as the arms.
  The stored `n1h` member remains a reference row, with the gate's gaps reported.
- **A, weekly analogues.** Baseline plus power at the valid time minus 1, 2, 3, 4, 5, and 6 weeks
  (6 columns); the 1-week analogue is null beyond about 6.9 days of lead.
- **B, analogue summaries.** Baseline plus the mean and median of whichever of those six are usable
  at the lead (2 columns; this is experiment 87's `a7` pair).
- **C, lag differences.** Baseline plus two columns. The origin anomaly is the newest usable reading
  minus the median reading at the same half-hour and weekday over the previous 4 weeks. The day-ago
  anomaly is the reading at the valid time minus 1 day minus the median reading at that half-hour
  and weekday over the previous 4 weeks, null when the one-day-earlier reading is not usable.
  The 4-week window is a choice, not a tested value.
- **E, drop the newest reading** (maintainer, 2026-10-30). The baseline without `lag_origin`, which
  is one noisy value that is the same for every valid time of a forecast origin. It keeps the
  same-time latest-day and latest-week lags. Prediction that can fail: E beats the baseline beyond
  24 hours and does not lose from 0 to 24 hours, so E is read by lead band before any pooled number.
- **F, strict 24-hour lag** (maintainer, 2026-10-30). E with the day lag replaced by the power
  exactly 24 hours before the valid time, null where that reading is not yet usable (leads beyond
  about 20.5 hours). F is also read against E, which shows what the older latest-day values add at
  long leads.
- **D, B plus C.** Fitted only if C2 and C3 each help alone (see the decision rule).

## Change of cadence to monthly refits (decided before any arm was scored)

The maintainer asked for monthly refits in most research. The headline contrasts are therefore
monthly: arms a, b, and c refitted on the 1st of each month from 2025-07-01 to 2026-06-01 (192 fits
per arm), against experiment 381's stored monthly control, which has experiment 87's `n1h`
recipe with no changes (the same columns, 500 rounds, `eta` 0.05, half-life 0.5 years, column
subsampling 1.0). The weekly baseline and weekly arm a (832 fits each) were finished first and give
a descriptive cadence check beside the monthly contrast. Weekly arm b is partial (384 of 832 fits)
and weekly arm c was not fitted, so neither is scored. Arm d is fitted only if b and c each help
alone at monthly refits. The decision rule above applies to the monthly contrasts.

## Lag-encroachment sweep (requested by the manager, rule fixed before any score is read)

After the main fits, the baseline and arms a, b, and c are refitted once more at the latest weekly
refit of the scored year (2026-06-23), with the boosters saved, for a four-Primary screen to save
compute (D8 the target, D4 a placebo, D10 and D5 signed-MW; 16 fits in all). A prediction-only sweep then adds a
synthetic step of depth D (plus and minus 2%, 5%, and 10% of capacity) to the lagged-power inputs:
first the most recent lag only (k = 1), then the two most recent (k = 2), then all of them,
including the weekly-analogue columns that lie inside the step. Per arm and lead band it reports
the shift of the median as a share of D, the band-width ratio against k = 0, and whether the
quantiles span both levels. It also reports, per arm: the share of event-level readings inside the
1-99% and 5-95% bands, whether q95 and q99 reach the event level at each k, and the price of
spanning (band widening, and CRPS outside events). NGED would accept a forecast whose median is the
normal running arrangement while its tails always span the normal and the abnormal levels. The sweep
uses at most 3 threads.

## Constraints and compute

- Sonnet only. At most 3 threads and `MemoryMax=16G`; `uv run --no-sync`, the `spawn` start method.
- About 76 s per fit in experiment 381, 832 fits per weekly arm: each arm takes about 6 hours of
  wall-clock time at 3 threads, and the refitted baseline is a fourth arm.
- References and slices as the brief lists: tails (0.01, 0.05, 0.95, 0.99), cold nights, sunny
  half-hours, load-shift event half-hours, and the avoidable-curtailment cost in three row sets.
