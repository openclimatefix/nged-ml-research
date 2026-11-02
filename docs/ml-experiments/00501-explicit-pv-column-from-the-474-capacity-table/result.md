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
status: planned
kind: other
protocol: demand-v2
date: 2026-11-02
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
---

**Planned. This page holds the plan only; the question, the answer, and the take-home message are
written after the run.**

## Question

Experiments 474, 477, and 479 leave the unmetered solar capacities at the six Primaries metered in
signed megawatts unvalidated: no rung of 474's ladder passes the placebo gate. A fixed capacity
from the 474 method is the one cheap use not yet tried. This experiment asks whether it helps
experiment 425's arm A when supplied as one extra input column.

## Plan

- **Column.** Capacity in megawatts of alternating current (the median arm: the point estimate of
  the 474 method divided by the median known-answer recovery of its rung R4 with cloud terms) times
  experiment 405's unit shape of the ensemble-mean shortwave radiation and temperature (DC to AC
  ratio 1.15, tilt 35 degrees, scaled by experiment 405's `u99`). The column is zero at the 10
  other Primaries, so the pretrained network is unchanged there.
- **Capacity per fold, out of fold.** For each fold the capacity is refitted with the experiment
  474 GAM (rung R4, cloud terms, all history before the fold's first day, month-block bootstrap
  not run) after setting experiment 269's cut to that day. The farm response tables that build rung
  R4's shape are refitted on farm output before the same day, so no scored month reaches the
  column. The capacity is read at the latest season-year the fit has rows for. 30 fits (6
  Primaries, 5 folds) on the CPU.
- **Arm.** Experiment 425's arm A, fine-tuned from its stored pretrained weights (cut-specific, 3
  seeds) with the column appended as a new input whose first-layer weights start at zero, at
  experiment 425's fine-tuning learning rate (2e-4). 15 fine-tunes (5 folds, 3 seeds), about 0.15
  GPU-hours, one GPU process under 8 GB. The reference is experiment 425's stored arm A, seeds 0 to
  2.
- **Standing practices followed.** Monthly refits; cleaned data with the flagged exact-zero
  half-hours dropped (count reported per fold); the experiment 399 start cut for D15 and D18;
  every fitted network saved; stored predictions reused for arm A; examples and tail plots in the
  write-up. The ensemble-mean inputs are those of arm A. Weather at lag times is not used; the
  network's lag inputs are those of arm A, so the experiment 425 caveat on past weather applies
  unchanged.
- **Caveats fixed in advance.** D5 is an export-limit case that the clip-free solar term cannot
  represent; D9 and D16 have a pinned capacity that is too high in the fit through the meter. The
  six are signed-MW Primaries; D15 and D18 are report-only. Every score is selection-biased.
  NGED's event labels are incomplete, so "rows with no labelled event" is the correct phrase for
  clean rows.

## Minimum detectable effect

To be printed by `e00501_score.py` before any contrast and copied here before the first fit: the
standard deviation over folds of the sunny-midday CRPS difference between arm A's seeds {0, 1, 2}
and seeds {3, 4}, divided by the square root of two.

## Serving table (to complete after the run)

What is stored or computed at serving, and the refits per year, written after the run.
