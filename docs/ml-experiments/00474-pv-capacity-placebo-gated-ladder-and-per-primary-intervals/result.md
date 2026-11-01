---
id: "00474"
slug: pv-capacity-placebo-gated-ladder-and-per-primary-intervals
theme: solar-disaggregation
also_relevant_to:
  - differentiable-physics-embedded-generation
status: planned
kind: other
size: >-
  medium. Stored: small private result files only. Serving path: not touched. Degradation rule: not
  touched. Designs: more than one way to gate a placebo, so the gate is fixed below before any
  score is read. Callers: this folder's scripts, which copy experiments 269 and 129 and call the
  reviewed `studies.*` modules without editing them. Asset graph: not touched.
hypothesis: >-
  Experiment 269's pooled empirical plant table (raw recovery 0.892 of an injected farm) can be
  gated against a placebo that keeps the season and the time of day but breaks the day-to-day
  weather link, and a physics-shaped constraint ladder on top of it raises or holds recovery while
  the placebo stays under the gate. Month-block intervals on the capacity then cover an injected
  capacity in at least 70% of injections at the nominal 80% level, including a load that follows
  the clouds.
prediction: >-
  Written before any fit. (1) The placebo (day-of-year profile plus a within-month shuffled
  anomaly) recovers under 0.20 raw on the GAM base, because the convex fit has no free
  day-of-year term. (2) The pooled table (C0) keeps raw recovery at or above 0.85 for 2 MW and
  4 MW injections at 4 of the 5 decision Primaries. (3) The physics shape (C1, tilt 20 degrees)
  moves raw recovery by under 0.03 either way. (4) The month-block 80% interval covers the injected
  capacity in 70% or more of injections in the easy world, and under 30% with a cloud-following
  load (experiment 241 found 0 to 2%).
decision_rule: >-
  Fixed before any score is read. Recovery is the raw capacity ratio of experiment 269's read-out
  (recovered over injected, ratio of 99th percentiles of alternating-current output over daylight
  half-hours), not divided by anything. Gate: the 95th percentile, over 20 within-month shuffles,
  of the raw placebo recovery at the same rung. A rung passes when real raw recovery is at least
  0.85 and the rung's own placebo 95th percentile is under 0.20. A failed gate is reported and the
  next rung is tried; it does not stop the work. Stage 1 uses the best rung that passes; if none
  passes, C0 is carried with the failed gate marked on every row of the table.
benchmark: >-
  Experiment 269's pooled empirical shape (0.892 raw at the ratio of 99th percentiles, 0.966 as a
  capacity ratio), and experiment 129 arm 2 (0.75 to 0.88, placebo 0.12, coverage 23 of 30).
row_set_id: >-
  Experiment 269's: the feature cache of experiment 106 before 2026-07-01 at D5, D9, D10, D11, D14,
  and D16 (injection test), then every Primary in the cache for the stage 1 table. No forecast is
  scored.
metric: >-
  Raw recovery by rung with month-block 80% intervals; placebo recovery by rung (20 shuffles at C0
  and the final rung, 5 elsewhere); interval coverage of the injected capacity; per-Primary AC and
  DC capacity by season-year with 10th, 50th and 90th percentiles.
---

# PV capacity: a placebo-gated ladder and a per-Primary table with intervals

**How to read the NGED table.** Each row is one Primary. The capacity columns give the 10th, 50th,
and 90th percentile of the estimate in megawatts of solar panels behind that substation's meter. AC
capacity is what the inverters can export, and DC capacity is the panels' rating, assumed to be 1.1
to 1.5 times the AC capacity. A narrow range does not mean a certain estimate: the Confidence column
says how well the method found an injected metered farm in tests, and every row's placebo gate
failed. Rows marked MVA-only, reverse flow, or report-only need extra care (see Status).

**Take-home message (GAM side, interim): the pooled plant table finds 0.87 of an injected farm, no
solar shape passes the placebo gate, and a cloud-load nuisance term is what makes the capacity
intervals honest.** Recovery is the share of an injected metered farm's output that the convex fit
finds again (1 is all of it). Every rung finds 0.75 to 0.94 of the injected farm. The placebo (the
farm's day-of-year profile plus the anomaly of another day of the same month) recovers 0.24 to 0.38
at the 95th percentile, above the 0.20 limit, because the season and the time of day alone give 0.13
to 0.26. A cloud-following load drops the 80% interval's coverage of the injected capacity to 0% at
every rung; adding two smooth cloud terms to the fit (day-mean and half-hour cloud response)
restores it to 73% to 93%. The per-Primary table below therefore uses the rung with the best
coverage in both worlds, and marks the failed gate on every row.

## Figures

![Estimated unmetered solar capacity at each Primary](figure_01_capacity_overview.svg)

![Capacity by season-year at each Primary](figure_01_capacity_panels.svg)

## Ladder results

Raw recovery over 5 Primaries, 5 farms, and injections of 2 and 4 MW.
Numbers are in the private file `work/rung_summary.parquet`; the main rows follow.

- **C0, pooled plant table (CAMS sky):** recovery 0.874, placebo 95th percentile 0.36 (maximum over
  Primaries), coverage 27% (bootstrap alone) and 80% (bootstrap plus recovery spread); with a cloud-
  following load 0% and 7%.
- **C1, physics shape, tilt 20 degrees:** 0.823, placebo 0.28, coverage 75%; cloud load 0%.
- **C1b, physics shape, tilt 35 degrees:** 0.752, placebo 0.25, coverage 75%; cloud load 0%.
- **R1, nearest farm's recorded output as the shape (instead of CAMS):** 0.876, placebo 0.35,
  coverage 79%; cloud load 24%.
- **R2, nearest farm's learned plant response:** 0.881, placebo 0.36, coverage 86%.
- **R3, pooled table plus nearest farm's output (as well as CAMS):** 0.938, placebo 0.37, coverage
  79%.
- **R4, pooled table driven by the mean of ERA5 and CAMS irradiance:** 0.940, placebo 0.40, coverage
  85%; cloud load 3%.
- **R5, pooled table driven by ERA5 alone:** 0.814, placebo 0.40, coverage 77%.
- **With the cloud terms:** C0 0.882 (placebo 0.49, coverage 77%, cloud load 73%); C1 1.030 (0.51,
  80%, 80%); R4 0.868 (0.51, 93%, 93%).

## Table for NGED

The table is in [nged_table.md](nged_table.md) (latest season-year) and in the private file
`work/nged_table.parquet` (every season-year). It uses rung R4 with the cloud terms. DC capacity is
AC capacity times an assumed ratio between 1.1 and 1.5 (centre 1.25), because experiment 73 found
that the ratio cannot be fitted.

## Neural estimate and pros and cons

The neural estimate and the pros-and-cons scores are on the page of experiment 477, which also holds
the side-by-side plot of neural and GAM capacities. It reads this experiment's files: the table
`nged_table.md` (latest season-year), and the private files `work/nged_table.parquet` (every
season-year, with the columns `ac_p10`, `ac_p50`, `ac_p90`, `dc_p10`, `dc_p50`, `dc_p90`,
`pinned_mw`, `status`) and `work/rung_summary.parquet`, under
`data/ml-experiments/00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals/`.

## Caveats

- No rung passes the placebo gate, so every capacity carries a "gate failed" note.
- D12 uses the physical fleet shape and the fit through the meter; D15 is a generator case with no
  estimate; D18 has data only from 2025-03-01.
- MVA-only Primaries are fitted to apparent power with reactive power assumed zero, which experiment
  370 found fails at a power factor of 0.97.
- The cloud-load test uses one form of load (experiment 129's), and the cloud terms match that form
  exactly, so the restored coverage is an optimistic figure.
- NGED's event labels are incomplete, so the readings may hold unlabelled events.
