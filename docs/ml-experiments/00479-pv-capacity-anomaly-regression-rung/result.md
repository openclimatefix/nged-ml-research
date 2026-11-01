---
id: "00479"
slug: pv-capacity-anomaly-regression-rung
theme: solar-disaggregation
also_relevant_to:
  - differentiable-physics-embedded-generation
status: planned
kind: other
size: >-
  medium. Stored: small private result files only. Serving path: not touched. Degradation rule: not
  touched. Designs: more than one way to build the estimator and its placebo, so both are fixed
  below before any score is read. Callers: this folder's scripts, which call experiment 474's
  helpers and the reviewed `studies.*` modules without editing them. Asset graph: not touched.
hypothesis: >-
  A capacity estimate that regresses the within-month anomaly of net demand on the within-month
  anomaly of the physics PV shape, with the temperature response fitted jointly, recovers an
  injected metered farm (raw recovery of 0.85 or more), and its placebo passes. A placebo that
  passes here is a property of the construction. The estimator is
  useful only if it also recovers the injected farm (the removed means did not erase the signal)
  and its 80% interval covers the injected capacity when the load follows the clouds.
prediction: >-
  Written before any fit. (1) The post-removal signal share of the PV shape stays between 0.15 and 0.30
  at every Primary (stage 0, below, measures it). (2) The
  within-month day-shuffle placebo of the irradiance anomaly recovers under 0.20 of the real
  capacity at every signed-MW Primary, by construction. (3) The profile-aware placebo (the farm's
  day-of-year profile kept, its anomaly shuffled; experiment 474's placebo) also falls under
  0.20 (the group means remove the profile except its within-month ramp; experiment 474's
  residual of 0.25 to 0.40 is the quantity under test, so a pass is expected but not guaranteed).
  (4) Raw recovery of the 2 MW and
  4 MW injections is 0.75 to 0.95 without cloud terms in the easy world. (5) With a
  cloud-following load and no cloud term the 80% interval covers the injection in under 30% of
  injections; with the two cloud terms it covers 70% or more in the load of experiment 129's
  form, and less when the load follows the squared cloud response.
decision_rule: >-
  Fixed before any score is read. Recovery is the raw capacity ratio of experiment 474 (recovered
  over injected, ratio of 99th percentiles of alternating-current output over daylight
  half-hours, not divided by anything), read at the season-year July 2024 to June 2025. A
  Primary passes the placebo when the 95th percentile of the profile-aware placebo's recovery,
  over 20 shuffles and the five farms, is under 0.20. It passes the known answer when the median
  recovery of the 2 MW and 4 MW injections is at least 0.85 and the nominal 80% interval covers
  the injected capacity in at least 70% of injections in the easy world. Each cell is one estimator
  variant (without or with the two cloud terms) in one world; the cells gated are the easy world and
  the linear cloud-load world, so four gated cells, each judged on its own. The squared
  cloud-load world is reported only. The coverage gate uses the interval widened by 1.28 standard
  deviations of the recovery over the five farms (experiment 474's definition), compared on the
  recovery scale where the injected truth is 1. A variant is useful if its gated cells pass the
  placebo and the known answer at 4 or more of the 6 signed-MW Primaries. Only a signed-MW
  Primary that individually passes all gates of a useful variant is marked validated in the NGED
  table. The 8 MVA-only Primaries, D5, D18, and any signed-MW Primary that fails are marked
  unvalidated (the MVA-only ones add the power-factor caveat), and keep experiment 474's estimate
  beside this one. The comparison with experiment 474's recovery on the same farms is reported,
  not gated. A pass or fail is reported either way. The 15-fit forecast
  step is dropped.
benchmark: >-
  Experiment 474's rungs C0 to R5 (raw recovery 0.75 to 0.94, profile-aware placebo 0.25 to 0.40,
  0.39 to 0.51 with the cloud terms; coverage 0% to 24% with a cloud-following load and no cloud
  term, 73% to 93% with two cloud terms), and experiment 477's neural capacities.
row_set_id: >-
  Experiment 474's: the feature cache of experiment 106 before 2026-07-01, the default rows with a
  finite reading and a finite CAMS shape, and for D15 and D18 only rows from 2025-03-01. The
  injection test uses the signed-MW Primaries D5, D9, D10, D11, D14, and D16 and the farms A, B,
  C, D, and F, with the rows that have a farm reading. The NGED table covers every Primary except
  D15 (a generator case with no estimate).
metric: >-
  Raw recovery by Primary, world, and estimator; the profile-aware placebo's 95th percentile and
  the irradiance-shuffle placebo's share of the real capacity; interval coverage; per-Primary AC
  and DC capacity by season-year with 10th, 50th, and 90th percentiles from 2,000 month-block
  draws; post-removal signal share, variance inflation factor, and minimum detectable capacity.
---

# PV capacity: an anomaly-regression rung on experiment 474's ladder

**Status: planned. The rest of this page is written after the fits.**

## Plan

- **Estimator.** For each Primary, remove from the reading (megawatts) and from every regressor the
  mean of the same calendar year-month, local hour of day, and weekday class (Monday to Friday,
  Saturday, Sunday). Regress the reading's anomaly on the physics PV shape's anomaly: tilt 35
  degrees, DC to AC ratio 1.15 as in experiment 405, with the temperature derate, one capacity
  coefficient per season-year (July to June). The coefficient is the negative of the capacity in
  MW AC. Temperature enters the same regression, so heating and cooling load is fitted jointly
  with the PV capacity: ERA5 temperature and its 24-hour trailing mean, each with piecewise-linear
  hinges at 5, 10, 15, and 20 degrees Celsius, in three local-hour bands (0 to 8, 8 to 16, 16 to
  24), all as anomalies. The comparison arm adds the two cloud columns of experiment 474 (the
  day-mean and half-hour cloud response).
- **Capacity-rises prior.** A post-hoc step after each fit: clip each draw's capacity path at zero
  and take its running maximum over the season-years, so a fall is resisted. Both versions are
  reported.
- **Uncertainty.** A month-block bootstrap of 2,000 draws, resampling months within each
  season-year and calendar quarter (experiment 269's stratification). The 10th, 50th, and 90th
  percentiles of AC capacity. DC capacity is AC capacity times 1.25, with 1.1 and 1.5 as a
  post-hoc rescaling (experiment 73 found the ratio not identified, and the AC capacity does not
  change).
- **Known-answer worlds** (experiment 474's harness, same farms and injection sizes). The injected
  series is subtracted from the reading (so the reading's anomaly falls by the series' anomaly)
  before the group means are removed. The easy world, with no load that follows the clouds, injects
  2 MW and 4 MW of Solar A to F's output (five farms, E left out for its short history). The
  cloud-load world adds experiment 129's cloud-following load (a load whose mean equals 2 MW of
  capacity, linear in the day's mean cloud response). A third world adds a load that follows the
  squared day-mean cloud response, scaled to the same mean, to test a load the cloud terms do not
  match exactly. Each world runs with and without the cloud terms.
- **Placebos.** (a) The profile-aware placebo of experiment 474: the farm's day-of-year profile
  plus the anomaly of another day of the same month, injected at 2 MW. The group-mean removal
  deletes the profile, so this placebo can pass trivially. (b) The irradiance-shuffle placebo: the
  PV shape's days and the two cloud columns' days are permuted together, by one
  permutation within each calendar year-month, and the real reading is refitted, and
  the placebo capacity is reported as a share of the real capacity. This placebo is true by
  construction. Both use 20 shuffles. The page states plainly that a placebo that passes by
  construction can hide a real signal that was removed together with the means, so the known
  answer (recovery of the 2 MW and 4 MW injections) is the check that the means did not erase
  the signal.
- **Coverage.** The nominal 80% interval is the 10th to 90th percentile of the bootstrap's
  increment (injected-world capacity minus base capacity, same draw), reported alone and widened
  by 1.28 times the standard deviation of the point recoveries over the five farms (experiment
  474's definition).
- **Scope.** Six signed-MW Primaries run the known answer. The eight MVA-only Primaries (D2, D3, D4,
  D8, D12, D19, D21, D22) are fitted to the apparent-power reading with reactive power assumed zero,
  by experiment 366's rule and experiment 420's export gate: the rule fails at a power factor of
  0.97, so these capacities carry that caveat. D5 (export limit; the clip-free solar term does not
  represent it), D12 (reverse flow), D15 (generator case, not run), and D18 (history only from
  2025-03-01, report-only) are marked. The 6 signed-MW Primaries and the 8 MVA-only Primaries are
  reported separately, never pooled.
- **Not applicable.** No forecast is scored, so the avoidable-curtailment cost, tail calibration
  (including the rule that q99 coverage must stay within 5%), and the forecast-use step do not
  apply. The tails reported are the 5th and 95th percentiles of the bootstrap capacity. Experiment
  474's forecast-use step repeats experiments 405, 409, 416, and 477 and is dropped.
- **Anonymisation.** Solar A to F by letter; no nearest-farm names; distance bands only
  ("under 15 km", "15 to 30 km", "30 km or more"). Rows with no labelled event: NGED's event
  labels are incomplete, so the readings may hold unlabelled events.

## Stage 0: numbers fixed before any recovery or placebo is read

Ordinary-least-squares standard errors on the real reading (no injection, no placebo), at the
latest season-year (July 2025 to June 2026). The standard error assumes independent rows, so it is
a lower bound (rows within a day are strongly correlated); the bootstrap interval will be wider.

- **Post-removal signal share:** 0.23 to 0.25 of the PV shape's variance at each of the 14
  Primaries, and 0.21 at D18. Experiment 405 found 0.21 on its farm. About one quarter of the
  shape's variance survives the group-mean removal.
- **Variance inflation factor of the latest season-year's capacity column** against all other
  columns: 1.02 to
  1.05 without the cloud terms, 1.35 to 1.65 with them, 2.4 for D18.
- **Minimum detectable capacity** (80% power, two-sided level 0.05 divided by the
  Bonferroni count of 36 tests: 6 signed-MW Primaries, 2 estimators, 3 gates): 0.05 to 0.35 MW
  AC without the cloud terms and 0.06 to 0.40 MW with them (D8 is widest, D21 narrowest). The count
  suits the 6 signed-MW Primaries only, and the figure tests a capacity against zero, which none of
  the gates does, so it is indicative. A capacity under those figures cannot be told from zero by
  this estimator.
- **Bonferroni count:** 36.

The full table is in the private file `work/stage0_signal.json`.

## Checks and reviews

Plan review, code review before the first fit, results review, and write-up review: Sonnet
reviewers, as the `ml-experiment` skill requires. Their findings and triage are in the PR body.
