---
id: "00649"
slug: pv-capacity-table-season-year-from-stored-probes
theme: solar-disaggregation
also_relevant_to: []
status: planned
kind: other
size: >-
  simple. Stored: one small private parquet of table rows and the figure source. Designs: one
  defensible construction (stored probe outputs divided by stored recovery factors), fixed below.
  Callers: this folder's scripts read, and edit none of, the stored outputs of experiments 621, 634,
  and 638 and the stored tables of experiments 267, 474, and 479. Serving path and degradation rule:
  not touched. No fit and no GPU; CPU only, under 1 minute.
hypothesis: >-
  The stored donor-day probes of the lag-free neural networks (experiments 621 and 634), divided by
  the leave-one-farm-out recovery of an injected real farm and widened by the spread of recoveries
  across farms, give an 80% capacity interval per Primary and summer half-year that overlaps
  experiment 474's R4 interval at 4 or more of the 6 MW Primaries.
prediction: >-
  Written before any table row was computed. (1) The calibrated median lies within 0.75 to 1.33
  times experiment 474's R4 median at 4 or more of 6 Primaries, and the 80% intervals overlap at 4 or
  more of 6. (2) The calibrated median is 1.4 to 1.6 times the raw median (the inverse of the 0.62
  to 0.71 recovery). (3) The calibrated 90th percentile is at or above the Feed-in Tariff register
  floor at every Primary that has a floor. (4) The widened interval covers the injected truth in
  60% to 80% of leave-one-farm-out constant injections, below the nominal 80% (experiment 634 read
  0.66). (5) The energy-basis row is within 10% of the megawatt-basis row.
decision_rule: >-
  A descriptive product, so the verdict is read from prediction (1) only. Supported if the intervals
  overlap at 4 or more of 6 Primaries with experiment 474's R4 table for the same half-year;
  partly supported if they overlap at 2 or 3; not supported at 0 or 1. The same count is reported
  against experiment 479's anomaly rung and the register floor, without changing the verdict. No
  threshold for use in production is set here.
benchmark: >-
  Experiment 474's R4 table (pooled table, ERA5 and CAMS mean irradiance, cloud terms), experiment
  479's anomaly rung, and the Feed-in Tariff register floor used in experiment 267 (experiment 129's
  `registers.parquet`, arm 2, 1 MW cut).
row_set_id: >-
  Not a forecast score. The probe cells of experiments 621, 634, and 638: 12 Wednesdays, half-hour
  ends 10:00 to 15:00 UTC, in April to September 2025 (headline) and April to September 2024
  (`probes_shift364`). Development data only; nothing dated 2026-07-01 or later.
metric: >-
  AC capacity in MW at the 10th, 50th, and 90th percentiles per Primary, raw and calibrated; energy
  recovery per injected megawatt-hour; overlap of two 10th-to-90th-percentile intervals in MW and as
  a share of the narrower one.
---

# Capacity of unmetered solar behind the six MW Primaries, from the stored probes (planned)

This page is the plan. The experiment builds a table of the AC capacity of unmetered solar at each of
the six signed-megawatt Primaries for the summer half-years of 2024 and 2025. It uses the stored
outputs of experiments 621, 634, and 638 and refits nothing. The write-up replaces this section.

## Plan

- **Why.** Experiment 644 closed the explicit-PV line as a forecast gain. The capacity estimate
  remains useful as an explainability and cross-check product for NGED.
- **Raw value.** Per cutoff, probe half-year, and Primary: the median over 3 seeds of the `plain`
  control probe's capacity. The spread combines the 200-draw day-block bootstrap and the standard
  deviation over seeds. The 10th and 90th percentiles are the median minus and plus 1.2816 standard
  deviations, as in experiment 634.
- **Calibrated value.** The raw value divided by the leave-one-farm-out factor: the median
  recovery of the injected 4 MW real farms (A, B, C, F), leaving the farm out. The spread adds the
  standard deviation of the four farms' median recoveries, in quadrature, to the relative spread.
  The headline is the 2026-06 cutoff (the most training data). The other three cutoffs
  (2025-07, 2025-10, 2026-03) are a sensitivity row. No cloud template is used in the headline.
- **Energy-basis sensitivity.** Recovery per injected megawatt-hour, from experiment 634's
  `energy_recoveries.csv`, gives the capacity of a physics-shaped plant carrying the same energy
  (raw divided by the energy recovery), and the AC capacity of a farm-shaped fleet (that, divided by
  the median farm-over-unit-curve energy ratio). The second is close to the megawatt basis by
  construction, so prediction (5) is a consistency check.
- **Comparison.** Each probe half-year is set against experiment 474's R4 table for the two
  season-years (July to June) that its probe dates fall in, experiment 479's anomaly rung
  (`cloud = false`, `prior = none`), and the register floor. Per Primary: whether the intervals
  overlap, by how many megawatts, and as a share of the narrower interval.
- **Resolution statement.** Values are season-year, not monthly. Experiment 638 found that a capacity
  step is followed at 0.82 to 0.88 only in an ordinary spring, and experiment 634 found that a
  sunny spring is read as solar capacity.
- **Standing practices.** The data are stored probe outputs (cleaned power, the experiment 554
  pretraining rows). No forecast is scored, so the tails, curtailment-cost, and example-forecast
  rules do not apply. The D15 and D18 cut does not apply (the six signed-megawatt Primaries only).
  Nothing from 2026-07-01 is used. Generators are not shown; the injected farms appear as A, B, C,
  and F only in the recovery bullets.
- **Review order.** Sonnet writes the script; a fresh Sonnet agent reviews the code before the first
  run; a fresh Sonnet agent re-derives the headline numbers from the stored files; a fresh Sonnet agent
  reviews the write-up; and a fresh Sonnet agent runs the prose review.

## Stop rules

- If the seed standard deviation of the raw capacity exceeds 0.5 MW at 3 or more Primaries, report
  the table as unreliable and keep experiment 474's R4 table as the product.
- If the cutoffs' calibrated medians differ by more than 30% at 3 or more Primaries, headline the
  spread across cutoffs rather than the 2026-06 cutoff alone.
