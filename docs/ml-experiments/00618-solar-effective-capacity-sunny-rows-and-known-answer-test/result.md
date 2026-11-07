---
id: "00618"
slug: solar-effective-capacity-sunny-rows-and-known-answer-test
title: >-
  Can the capacity of a metered solar farm be tracked from its very sunny half-hours alone, and did
  Solar A, Solar B, and Solar D really produce more in 2021 to 2023?
theme: metered-solar-farms
also_relevant_to:
  - generator-forecasting
status: planned
kind: other
date: 2026-11-07
size: >-
  medium. Stored: private tables and fitted levels under data/ml-experiments/ only. Designs: more than
  one defensible estimator (penalty form, day versus half-hour observations), so the penalty is
  fixed by placebos before any real fit. Callers: this folder's scripts only. CPU only, at most 4
  cores. No forecast is scored (experiment 464 found no gain from slow scaling). Serving path and
  degradation rule: not touched.
hypothesis: >-
  A farm's capacity moves rarely, so a piecewise-constant level fitted to one daily number taken from
  its very sunny half-hours (clear-sky index at least 0.9, clear-sky irradiance at least 700 W/m2,
  six half-hours in a row) recovers injected capacity steps, raises few false steps, and shows whether
  Solar A, Solar B, and Solar D produced more in 2021 to 2023 than now.
prediction: >-
  Written before any script ran. Step 1 passes: at least 3 of the 5 testable farms (Solar A, B, C, D,
  F) have a median of 20 or more sunny half-hours per April to August month. Step 2: injected steps
  of 8% or more are recovered at least 80% of the time at Solar A, B, C, and F; Solar D (fewest sunny
  rows) is weaker. The real fits show a fall in level for Solar A, B, and D between 2021 to 2023
  and 2025 to 2026, as experiment 453 saw.
decision_rule: >-
  Step 1 stop rule (fixed by the brief): under 20 sunny half-hours a month at 3 of the 5 testable
  farms stops the line. A farm counts as under 20 when its median over April to August months with
  data is under 20. Step 2 pass bars, per farm: recall at least 80% for injected steps of 8% or
  more; at most 1 false step a year on permuted days; recovered step size within 25% of the
  injected size in at least 80% of detected steps; 90% interval covering the true level on 85% to
  95% of injected informative days; smooth-drift placebo at most 3 steps a year with the final
  level within 3% of the truth. The penalty is the smallest on a fixed grid that meets the
  false-step bar on tuning permutations, then checked on fresh ones.
benchmark: >-
  Experiment 453's reading that Solar A, Solar B, and Solar D produced more in 2021 to 2023 than now.
row_set_id: sunny_rows_to_2026_06_30
metric: >-
  Recall, false steps per year, size error, and interval coverage of the recovered level; sunny
  half-hours per month.
---

# Solar effective capacity: sunny rows and a known-answer test

Planned. The write-up replaces this page when the scripts have run.

- **Rows:** CAMS irradiance and clear-sky irradiance from experiment 218's `farm_rows_cams.parquet`
  (no new fetch), farm output as a share of the farm's effective capacity, to 2026-06-30. July 2026
  onwards is untouched.
- **Estimator:** one number a day (median of output over a modelled expectation, on sunny rows),
  then an exact L0 change-point fit (optimal partitioning) with a penalty per step.
- **Not done:** no forecast test, no wind (other briefs), no scoring on July 2026 onwards.
