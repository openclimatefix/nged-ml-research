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

**Plan only; no fit has run.** The maintainer set this line high priority on 2026-11-01. NGED wants
a rough installed-PV estimate for every Primary, so the work gives every Primary a row and marks how
far to trust it. This first pull request covers the GAM side (stage 0) and the per-Primary table
(stage 1) on CPU. The neural-network side and the explicit PV forecast (stage 2) wait for the
manager's go.

## Plan

- **Base.** Experiment 269's pooled empirical plant table (the mean learned response of the other
  farms, read at the Primary's CAMS irradiance), fitted with experiment 129's convex model on rows
  before 2026-07-01. This is rung C0, reproduced from stored code and checked against 269's
  stored `result_D*.parquet`.
- **Placebo, calibrated as in experiment 409.** The farm's series is split into a day-of-year
  profile and an anomaly. The placebo injects the profile plus the anomaly of another day shuffled
  within the same month. A profile-only benchmark (anomaly set to zero) states how much recovery
  the season and time of day give alone. Shuffling within month and clear-sky class is not used,
  because it keeps the sky link.
- **Known answer.** Metered farms A to F by letter (E only where its history allows) are injected
  at 2 MW and 4 MW AC (1 MW is below seed noise). Raw recovery and interval coverage are read per
  rung.
- **Ladder.** C1: the physics chain from `studies.pv_physics` (tilt centred near 20 degrees,
  temperature derate, clip at the AC rating). C2: a fixed DC-to-AC ratio of 1.25 (range 1.1 to
  1.5), applied after the fit, so the DC interval carries that spread. C3: a post-hoc asymmetric
  penalty on the capacity increments (a rise is cheap, a fall is dear). C4 and C5 (nearby-farm shape;
  CAMS with ERA5 temperature) are experiments 238, 267 and 269, whose stored results are cited, not
  re-run. Every rung is reported, including failures.
- **Stage 1 table.** Per Primary and season-year: AC and DC capacity at the 10th, 50th and 90th
  percentiles from a month-block bootstrap of the chosen rung, a recovery column from the known
  answer, the method and rung, and a status column (D5 export limit, D9 and D16 pin, D12 reverse
  flow, D15 and D18 history only from 2025-03-01, report-only). The MVA-only Primaries reuse
  experiment 366's rule and 420's export gate at reactive power zero, with the power-factor-0.97
  failure reported.
- **Coverage.** The known-answer coverage test includes the cloud-following load case (code in
  experiments 129, 235 and 241).
- **Privacy.** No metered generator's series is shown with a name or ID; distance bands replace any
  farm-to-Primary pairing. The table carries no generator data.
- **Compute.** CPU only, at most 4 cores, under 12 GB. Fits cost about 0.5 s each.
- **Reviews.** A fresh Sonnet agent reviews the code before the first scored run; results, write-up
  and prose reviews follow, with findings recorded in the PR.
