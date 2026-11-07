---
id: "00621"
slug: pv-probe-injected-known-answer-and-placebo-test
theme: solar-disaggregation
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
size: >-
  medium. Stored: pretrained network weights, probe frames, and probe results in this experiment's
  private folder. Serving path: not touched. Degradation rule: not touched. Designs: more than one
  injection and probe design is defensible, so the worlds, the probe regression, and every pass bar
  are fixed below before any recovery is read. Callers: this folder's scripts import experiments 405,
  409, 448, 510, and 554 and edit none of them. Asset graph: not touched. One GPU process, about 0.7
  GPU-hours.
hypothesis: >-
  A lag-free neural network pretrained on ERA5 and CAMS estimates of past weather, probed with its
  datetime inputs held fixed and donor-day weather swapped in, returns the AC capacity that was
  injected into its training target at the six MW Primaries (raw recovery 0.85 to 1.15), reads a
  profile-only and a shuffled-sky placebo near zero, and is not fooled by a cloud-following load.
prediction: >-
  Fixed from the Opus design review of 2026-11-07 before any recovery was read. (1) Median recovery
  of a 4 MW real-farm injection is 0.75 to 0.90 and lies in 0.85 to 1.15 at 5 of the 6 Primaries;
  for 2 MW at 4 of 6. (2) The physics-curve world recovers more than the real-farm world. (3) Each
  placebo reads under 0.20 of the injected MW (95th percentile over cutoffs and seeds) at 5 of 6.
  (4) With the cloud template, the cloud-following load fakes under 0.15 of a 2 MW farm. (5) The
  nominal 80% interval covers the injected capacity in at least 70% of injections in the physics-curve
  world and in the linear cloud-load world.
decision_rule: >-
  Fixed before any recovery is read. Each bar passes or fails as written in the prediction; a failed
  bar names its world and stops the line with no rescue tuning. The step world is not run (see the
  deviations). Recovery is the probed capacity of the injected network minus that of the uninjected
  twin with the same seed, divided by the injected megawatts; it is never divided by the recovery at
  other farms. The threshold for acting on a probe capacity in production is not set here.
benchmark: >-
  The uninjected twin (same cutoff and seed, same rows). Experiment 474's rung R4 and experiment
  479's anomaly rung for the real-data capacities, as context only.
row_set_id: >-
  Not a forecast score. Pretraining rows are experiment 554's ERA5-and-CAMS rows at the 2026-03 and
  2026-06 cutoffs (development data only; nothing dated 2026-07-01 or later). Probe dates are 12
  Wednesdays from 2025-04-09 to 2025-09-10, half-hour ends from 10:00 to 15:00 UTC.
metric: >-
  Effective AC capacity in MW per Primary from the donor-day regression; recovery as a share of the
  injected megawatts; day-block bootstrap 80% interval.
production_status: not-for-production
adopted_in: none
---

# Injected known-answer and placebo test of the donor-day PV probe

*Planned. The write-up replaces this page when the fits and probes have run.*

**Worlds.** Each pretraining injects one world at all six MW Primaries (D5, D9, D10, D11, D14, D16)
and is paired with an uninjected twin of the same cutoff and seed. The worlds are: a real farm's
output scaled to 2 MW and 4 MW (farms A, B, C, and F, two rotations over the Primaries); the physics
unit curve driven by the Primary's own CAMS shortwave at 2 MW and 4 MW; a profile-only placebo (the
farm's year-month by half-hour mean) at 4 MW; a shuffled-sky placebo (that profile plus the farm's
anomaly from another day of the same month) at 4 MW; and a cloud-following load (linear and squared
forms of experiment 129's day-long cloud response, averaging 0.10 and 0.30 of 2 MW over daylight
half-hours).

**Probe.** Datetime inputs and the Primary are held at the probe date and half-hour. The weather
inputs come from every other date within 21 days of the probe date's day of year, at the same
half-hour. Probed net demand is regressed, with an intercept per probe cell, on the physics unit
curve of the donor weather and the donor temperature (`plain`), and also on a flat `1 - clear-sky
index` cloud template (`cloud_tpl`). The capacity is minus the slope on the unit curve.

**Deviations from the design review, fixed before any recovery was read.** (a) The network is
experiment 554's stage `pre` network (M0), which has no year or drift input, so the step world, whose
point is time resolution, cannot be tested and is not run. (b) Farms D and E are not used, because
their daylight gaps (14% and 24 to 78%) would remove most rows; the six Primaries take farms A, B, C,
F, A, B. Rows with a gap in any of the four farms are dropped from every arm, which is 15.5% of the
six Primaries' rows at the 2026-06 cutoff. (c) The farm series has no zero-fill after 2026-03-26;
those half-hours are dropped like any other gap. (d) The cloud load is run at two sizes because the
size that experiment 129 used may be too small to test the 0.15 bar.

**How each bar is computed (fixed before the grid was read; the only recoveries seen were a timing
run of one seed at the 2026-06 cutoff).** The unit of evaluation is a Primary. For each bar, a
Primary's value is the median (or, for the placebo bar, the 95th percentile) over the 2 cutoffs, the
3 seeds, and, for the farm worlds, the 2 farm rotations. A bar passes when it holds at 5 of the 6
Primaries (4 of 6 for the 2 MW real-farm bar). Recovery uses the `plain` regression for the
real-farm, physics-curve, and placebo worlds, and the `cloud_tpl` regression for the cloud-load
worlds. The nominal 80% interval of an injection is its recovery plus or minus 1.28 times the root of
the bootstrap variance and the across-seed variance of the probed capacity difference; the
bootstrap-only interval is reported beside it. In the cloud-load worlds, the interval has to cover
zero, because no PV was injected. The cloud load fakes capacity when the `cloud_tpl` difference
exceeds 0.15 of 2 MW.
