---
id: "00506"
slug: mva-sign-evidence-dip-depth-and-gated-solar-rule
title: >-
  What can the data not yet used say about the sign of real power at the eight Primaries whose
  meters report apparent power only?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
kind: other
date: 2026-11-02
size: >-
  medium. Stored: small result tables under data/ml-experiments/00506-*/work/ (private). Serving
  path: not touched. Degradation rule: not touched. Designs: more than one statistic could test the
  question, so the plan fixes two (dip depth, gated rule). Callers: reads the stored results of
  experiments 152, 372, 393, 408, 412, 415, 474, and 475. No model is fitted. Reviews: one Sonnet
  plan review (done), one Sonnet code review before the first run, one Sonnet results review, and
  two Sonnet write-up reviews.
hypothesis: >-
  Real power at a Primary can change sign only if its apparent power dips close to its reactive floor
  at sunny middays, so the depth of that dip, and a rule that calls reverse flow only when expected
  solar output exceeds expected gross demand, carry usable evidence of the sign at the eight
  Primaries that no fitted treatment has recovered.
prediction: >-
  Written before any number is read. The dip statistic detects crossings at the six signed-MW
  Primaries (with injected solar farms) with an area under the curve of at least 0.8 at reactive
  floor Q = 0, and falls to 0.6 to 0.7 at power factor 0.97. Of the eight, only D12 (experiment 370)
  shows a dip below the calibrated threshold, and D8 (which NGED marks "Yes") does not. The
  threshold is frozen on the six before any of the eight is read. The gated rule with a capacity from outside the
  labelled rows does not match experiment 408's recall at a lower false-reverse rate.
decision_rule: >-
  Part 1 counts as a usable diagnostic only if the area under the curve at Q = 0 is at least 0.8 with
  a month-block bootstrap interval whose lower end is above 0.6 (the lower-bound clause is this
  plan's addition to the brief); otherwise it is a diagnostic only and the page says so. At power
  factor 0.97, the realistic case, an area under the curve below 0.7 means the delivered label may
  not claim to detect a crossing. Part 4's
  rule is an improvement only if it matches experiment 408's recall at a lower false-reverse rate than
  experiment 408's 3.6% at injection level 2.0 at the six, and experiment 412's S1c at the matched
  false-reverse rate at D5 and D10 only; otherwise the line is closed. No gain in forecast score is claimed, because no
  forecast is fitted. The threshold for porting is not applicable: nothing here enters production.
benchmark: >-
  Experiment 408's shortwave-threshold rule at the six (false-reverse 3.6% at level 2.0); experiment
  412's S1c at D5 and D10 only (false-reverse 6.7% in March to 16.9% in June).
row_set_id: not_applicable_no_forecast
metric: >-
  Area under the curve of the dip statistic for detecting a sunny-midday crossing, balanced accuracy,
  recall, and false-reverse rate of the gated rule, each with a month-block bootstrap interval
  (20,000 resamples).
---

# Plan: what the data can say about the sign of real power at the eight MVA-only Primaries

*Status: planned. Nothing below has been run.*

## Question and scope

Eight Primaries (D2, D3, D4, D8, D12, D19, D21, D22) have meters that report apparent power (MVA)
only, so no signed history exists. Six earlier fold treatments (experiments 152, 372, 374, 387, 408,
and 412) did not recover the sign. This experiment fits nothing. It asks what four sources can say: the
depth of the apparent-power dip at sunny middays, a gated physical rule with an independent
capacity, the upstream meters, and a metered generator.

## Order of work (as the manager's edits of 2026-11-02 fix it; the manager hears at plan PR open, code review done, results, and merge)

1. **Part 1, with Part 4's dip comparison merged in: one table of the eight.** Per Primary and
   month, the median, 10th percentile, and minimum of the apparent power (the reading at the eight,
   the absolute value of MW at the six) on sunny middays (CAMS clear-sky index at least 0.7, 10:00 to
   15:00 UK time), beside the same months' dark-half-hour median, in points of effective capacity.
   Cleaned data (`studies.power.scan_power`), flagged exact zeros dropped, D15 and D18 cut before
   2025-03-01 (experiment 399), July 2026 untouched.
2. **Known-answer calibration at the six.** Injected solar farms as in experiment 408 (Solar A to F
   in anything written), reactive floor Q = 0, 2, and 4 points, and a Q-in-proportion arm (power
   factor 0.97, Q about 25% of the dark-hour demand level, which needs no sign). Area under the curve and best threshold of the dip statistic
   for detecting a crossing, with counts of crossings per Primary. Only D10 has real reverse flow, so
   the injections carry the calibration.
3. **Part 4, the gated rule.** Call reverse flow only if the unit PV shape is at least 0.3 and the
   expected solar output exceeds expected gross demand by a margin. Capacity from a pre-2026 vintage
   of experiment 474's table with the tested Primary's own signed rows excluded; if that cannot be
   built, the test is labelled circular and diagnostic only (474's table failed its placebo gate at
   every rung, 95th-percentile placebo recovery 0.25 to 0.51). Compare with experiment 408 at the
   six and with experiment 412 at D5 and D10 only. Apply to the eight and report agreement with
   Part 1 only, without tuning on the eight. The rule reports P(C u > D) with an interval for C;
   expected gross demand D comes from the stored first-release demand forecast on dark and cloudy
   half-hours of the same day type, adjusted for temperature on sunny middays, where demand and
   solar output are confounded (experiment 412 found temperature can stand in for irradiance).
   Capacity sources: the vintage above, the embedded-PV-capacity study's table (a fitted product,
   not an installed-capacity register), and experiments 477 and 479, which recover 0.62 to 0.72 of
   a known capacity. Injected farm sizes are fixed as experiment 408 fixed them (a multiple of the
   median sunny-midday reading), never from the capacity table. The vintage is circular unless it
   is refitted without the tested Primary's rows.
4. **Part 2, a short check.** Units, share of negative readings, and children of the two grid supply
   points and two bulk supply points; closure residual cited from experiment 152 (D12 coefficient
   1.02, variance cut 0.7%, interval includes zero); which of the eight have a metered generator
   behind them (anonymised labels), the share of each node's children that are metered, and whether
   the generator's output exceeds the estimated demand (the demand model's dark-hour level) at any
   half-hour, which would imply export. The brief's closure statement is kept: the closure residual
   is below half the typical reverse flow, or the closure cannot resolve the sign; expected to
   resolve at most one or two Primaries.
5. **Part 3, one page of wording.** Evidence per flagged Primary, the false-reverse rates of the
   methods that have known-answer tests, the wording a delivered forecast may carry, and results at
   power factor 0.97 stated as the realistic case.

## Statistics

- Bonferroni count: 70 (Part 1: 16 Primaries by 4 reactive settings is 64; Part 4: 6 Primaries).
- Minimum detectable effect stated beside every difference in area under the curve.
- Month-block bootstrap, 20,000 resamples. July 2026 untouched. "Rows with no labelled event" is the
  wording for NGED's incomplete event labels.
- No forecast score is computed. The folded-versus-unfolded contrast is quoted only from experiment
  372 (D12 only) and experiment 393 (the six only).

## Compute

CPU only, 2 cores, no fits, about 4 to 7 hours. Each script runs under
`systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue`.

## Problems caused by the move

Found so far: `torch` is not in the default `uv sync` environment, so
experiment 412's `e00412_inputs.py` cannot be imported by a CPU-only experiment; this experiment
reads the power, CAMS parquet and metadata directly.
