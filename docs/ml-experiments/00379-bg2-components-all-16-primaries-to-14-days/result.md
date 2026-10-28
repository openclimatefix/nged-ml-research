---
id: "00379"
slug: bg2-components-all-16-primaries-to-14-days
title: >-
  What does each component of BG2 contribute at all 16 Primaries, out to 14 days, under demand-v1?
theme: production-reference-and-best-guess
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: other
protocol: demand-v1
size: >-
  medium. Stored: forecasts and small result files in this experiment's private folder only;
  experiments 87, 360, and 377 are read and never written. Designs: one (re-blend the stored
  members of experiment 87 by its own rule, as experiment 360 did at four sites). Callers: the
  scripts copy experiment 87's assembly code and experiment 377's adapter and scorer, whose callers
  are all in those folders. Serving path and degradation rule: not touched (this repository holds
  neither).
hypothesis: >-
  Under the demand-v1 protocol at all 16 Primaries, no single component of BG2 carries its lead
  over a single weekly-refitted XGBoost model, so dropping any one of the five blend members, the
  solar offset, or the spread factor moves the CRPS of days 1 to 10 by less than 0.10 points of
  effective capacity.
prediction: >-
  Written before any score is read. P1. Dropping any one of the five members raises the pooled
  CRPS of days 1 to 10 by between -0.05 and +0.10 points in each group (the drop-one arms of
  experiment 360 at D8 moved by at most 0.09). P2. Weekly refitting (member W) beats monthly
  refitting (member M) by 0.05 to 0.20 points in the signed-MW group (experiment 87 found 0.123
  over all 16 Primaries). P3. The solar offset lowers the CRPS of days 1 to 10 by under 0.03
  points pooled over the signed-MW six, because it touches only daylight April to September at 5
  of the 6 (0.010 over all 16 Primaries in experiment 87), and by 0 at the MVA-only eight, which
  carry no offset. P4. Each member alone scores within 0.30 points of BG2 except the frozen
  stack member (ST), which trails by more than 0.50. P5. The blend's lead over its best single
  member shrinks from days 1 to 7 to days 8 to 14, because the members share one NWP input
  and its errors become common at long leads.
decision_rule: >-
  Descriptive, no adoption and no threshold: the experiment ranks BG2's components, it does not
  change a research default. Contrasts, all registered before any score is read, all CRPS in
  points of effective capacity on the shared rows by the paired month-resampled bootstrap of
  `studies.eval_protocol`, 99% interval with 20,000 resamples as the headline and 95% interval with
  2,000 resamples per Primary. C1: BG2 minus BG2 without member k, for each of the five members.
  C2: BG2 minus the same blend with no solar offset (stored arm S5). C3: BG2 minus the blend with
  the spread factor 1.0 (stored arm A5). C4: each stage of the staircase minus the stage before it
  (frozen, stack, monthly, weekly, blend, blend with offset). C5: each member alone minus BG2.
  Every contrast is read in the days 1 to 10 band, with days 11 to 14 and each lead band shown
  beside it, once per group (signed-MW six; MVA-only eight with and without D8; D15 and D18 report
  only). One reading per group, never one for all 16. No tally counts D15 or D18.
benchmark: >-
  BG2 (stored arm `p_s6_bg2` of experiment 87), `manual_heuristic`, `climatology`, and Primary
  demand research default 1 extended to 336 hours (experiment 377's stored stack), all scored on
  the same rows. Production does not run BG2.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles (the quantile score), mean
  absolute error of the median, tail pinball and coverage, and twCRPS, in points of effective
  capacity, from `studies.eval_protocol`; leads 0.5 to 336 hours; the headline band is days 1 to 10.
code: ml-experiments/00379-bg2-components-all-16-primaries-to-14-days/
---

## BG2's components at all 16 Primaries, out to 14 days, under demand-v1 (plan)

This page is the plan. The write-up replaces it when the scores exist.

**Components.** The list comes from the assembly page of [experiment
87](../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md) and the drop-one page of
[experiment 360](../00360-bg2-member-drop-one-at-d8/result.md). BG2 at a Primary is a quantile blend
of five members: weekly-refitted XGBoost (W), monthly-refitted XGBoost (M), an analogue model (AN),
an XGBoost model pooled by substation type (PT), and the frozen stack of experiment 48 with a
14-day error correction (ST). Every member has the spread factor 1.15 about its median. At D9, D10,
D11, D14, and D16, members W, M, AN, and ST use their solar-offset twins. The blend weights are one
vector per origin month, chosen on a grid of quarters by earlier cases only.

**Arms.** Each arm is read from the stored member forecasts of experiment 87, with no refit and no
write to its folders.

- **Staircase** (stored arms): S1 frozen, S2 stack, S3 monthly, S4 weekly, S5 blend without the
  offset, S6 BG2.
- **Each member alone**, in BG2's configuration (spread factor, offset twin where the member has one):
  W, M, AN, PT, ST.
- **Drop-one:** BG2 re-blended without each member, the weights re-chosen on the grid for each subset,
  as experiment 360 did. Dropping the offset is S5; dropping the spread factor is the stored arm A5.
- **References:** BG2, `manual_heuristic`, `climatology`, and the experiment 377 stack.

**Rows.** The stored arms cover 365 daily runs at all 16 Primaries and leads to 351 hours. The
scorer keeps valid times before 2026-07-01 and leads to 336 hours, and the slice table drops
rows inside hand-filtered events. The stored quantiles are fractions of effective capacity, so the
adapter of experiment 377 converts them to MW. **A gate** requires the re-blended five-member arm to
reproduce the stored BG2 quantiles to within 0.001 points of capacity, and the unit check requires each
stored actual to equal the feature cache's reading.

**Groups and D8.** The three groups are reported separately. Every MVA-only number carries the
label "reactive power assumed zero". BG2's members fit D8's own readings, so BG2 has one arm at D8,
and it is D8's row; the donor-spread arm belongs to the stack of experiment 377 only.

**Compute.** Three threads in total (`OMP_NUM_THREADS=1` per process, at most 3 processes), and
`MemoryMax=16G` per job. No refit is expected. If a stored member file is missing, the plan stops
and asks the manager.

**What the next experiment should answer** is written after the scores exist.
