---
id: "00456"
slug: primary-demand-unmirrored-level-step-scan
title: >-
  How often does a Primary's demand show a sustained level step that no neighbouring Primary mirrors?
theme: lsc-detectors
also_relevant_to:
  - lagged-power-and-error-correction
status: planned
kind: other
date: 2026-11-01
size: >-
  medium. Stored: tables, figures, and cross-fitted forecasts under
  data/ml-experiments/00456-primary-demand-unmirrored-level-step-scan/ only. Serving path: not
  touched (this repository holds none). Degradation rule: not touched. Designs: the definitions are
  fixed by the maintainer-cleared brief and tightened below before any result is read, so one
  design. Callers: this folder's scripts import experiment 87's series loader and nothing else
  beyond `studies.*`. One Sonnet plan review, one Sonnet code review before the first run, one
  Sonnet results review, and two Sonnet write-up reviews. CPU only, 1 to 2 cores.
hypothesis: >-
  Row 4 of the lag-trust design needs a rate of genuine level changes (a new connection, a long
  curtailment, a demand-side change) in its training worlds. A label-free scan of the daily
  weather-corrected level of the 16 Primaries finds steps that persist for 7 days or more, and
  separates those that a neighbouring Primary mirrors (load-shift candidates) from those that none
  mirrors (genuine level-change candidates).
prediction: >-
  Written before any result is read. P1: the scan finds mirrored steps at the Primaries with
  labelled events, and recalls at least half of the labelled events that last 7 days or more. P2:
  unmirrored steps occur at under 1 per Primary-year at the thresholds below, and the share of
  half-hours inside an unmirrored step is under 2% (the lower of row 4's two placeholder rates). P3:
  injected steps of 3 times the spread lasting 14 days or more are recalled at 0.8 or higher.
decision_rule: >-
  Descriptive, no verdict on a model. The scan is trusted for the row 4 rate if injected steps of
  1.5 times the detection threshold lasting 14 days (a size the baseline thresholds can detect) are recalled at
  0.7 or higher and the false-alarm rate on block-bootstrap surrogates is under 0.2 per
  Primary-year. If not, the unmirrored count is reported as
  an upper bound (false alarms dominate) or a lower bound (recall is low), and the page says which.
benchmark: >-
  NGED's labelled events (experiments 392, 395, 401, and 435) for recall, and injection into
  step-free periods for the known answer.
row_set_id: >-
  Not applicable: no forecast is scored.
metric: >-
  Steps per Primary-year by class, recall and false-alarm rate of the scan, and the share of
  half-hours inside an unmirrored step.
code: ml-experiments/00456-primary-demand-unmirrored-level-step-scan/
---

**Plan (written before any result exists).** This page is committed as `planned`; the finished
write-up replaces it.

## Question

How often does a Primary's demand series show a sustained step (a new level lasting days to
months) that no neighbouring Primary mirrors with an opposite-sign step, and how large are such
steps? A mirrored step is a load-shift candidate. An unmirrored step is a candidate genuine level
change. The answer sets the rate of genuine level changes in the training worlds of row 4 of the
lag-trust design, where 2% and 5% are placeholders.

## Data and tightened definitions

- **Series.** The 16 Primaries of experiment 87's loader (`observed.parquet`, flagged exact-zero
  half-hours already null), from 2019-09-01 to 2026-07-31, divided by effective capacity. D15 and
  D18 drop every row before 2025-03-01 (the standing cut). The number of null half-hours inside
  each Primary's active span is reported.
- **Level.** The daily median of the residual, in points of effective capacity (1 point is 1% of
  effective capacity), over days with at least 24 valid half-hours. Two residuals are compared.
  Method (a) is the series minus the mean over the five seeds of the stored cross-fitted lagless
  `q50` of experiment 449 (25 tables, five months: 2025-10, 2025-12, 2026-01, 2026-03, 2026-06; a
  single month is too short to hold a 7-day step reliably away from its edges, so method (a) is a
  cross-check on those months only). Method (b), used for the full history, is the series minus a
  lagless XGBoost median forecast (13 calendar and ERA5 features of experiment 5, absolute-error
  objective, 300 rounds, depth 6, eta 0.1), cross-fitted in 6-month blocks: each block is
  predicted by a model trained on all other blocks of that Primary. Blocks have no time feature,
  so a step that lasts more than one block is partly absorbed; candidate steps that start within 3
  days of a block boundary are flagged as a possible artefact. ERA5 is the weather at all times
  (a lab condition: ERA5 arrives about 5 days late, so the scan is a retrospective finder, not a
  live input).
- **Spread.** The median absolute deviation (unscaled) of each Primary's daily residual about its
  median, per method.
- **Change points.** Optimal partitioning of the daily residual with a squared-error cost, a
  minimum segment length of 7 days, and a penalty of 3 times `log(n)` times the robust noise
  variance (from the median absolute successive difference), so the penalty is set by the data and
  not by a threshold of ours. The scan builds its own dynamic programme (the `ruptures` package is
  not installed). A step is the difference in mean level between two adjacent segments, and it
  counts if it is at least 3 times the Primary's spread and at least 2 points. Persistence is
  therefore at least 7 days by construction.
- **Sensitivity grid.** Spread multiple 2, 3, and 4; size floor 1, 2, and 4 points; minimum segment
  14 days against 7 (the 3-day variant of the brief is dropped because a segment under 7 days
  would be reported by the penalty alone). The grid is one factor at a time around the baseline.
- **Temporary step.** A step followed by an opposite-sign step of similar size (within 50%) after
  fewer than 30 days. Both are reported apart, as one temporary event.
- **Mirror.** A step at another Primary is a mirror if it has the opposite sign, a magnitude in MW
  within 50% of the first step's magnitude in MW, and a start within 1 day. Magnitudes are compared
  in MW because load moves in MW. The primary rule uses a single mirroring step. A sensitivity rule
  also accepts the sum of coincident opposite-sign steps at several neighbours, and a start within
  3 days. The neighbour sets are the pairs named in the corrected label table of experiment 395
  (`sw_block_detector_label_table_corrected.csv`); the second test uses all 15 other Primaries. A
  step whose neighbours have no data on its dates is reported as unassessable, not as unmirrored.
- **Classes.** Mirrored (load-shift candidate), unmirrored (genuine level-change candidate),
  temporary (reverses within 30 days), unassessable.

## Validation

- **Known answer from injection.** Steps of 1, 1.5, and 2 times the detection threshold
  `max(3 x spread, 2 points)` (not 1, 2, and 3 times the spread, which sit below or on the
  threshold by construction; the spread and noise scale are fixed at the un-injected series'
  values) lasting 7, 14, 30, and 90 days
  are added to the daily residual, alone and with an opposite-sign mirror of the same MW at one
  neighbour, only into windows (the duration plus 30 days either side) in which the scan finds no
  change point and no labelled event overlaps. The number of half-hours that the step-free rule
  excludes is reported. Recall is the share of injected steps detected within 2 days of their start
  with a size within 50%, per Primary and per cell of the 3 by 3 grid. Injection is done on the
  daily residual (the cross-fitted model is not refitted), which is a lab shortcut: a refitted
  model would partly absorb a long injected step.
- **False alarms.** The surrogate is built without the scan: each Primary's daily residual is
  resampled in blocks of 14 days (block bootstrap, 20 draws), which keeps short-range
  autocorrelation and destroys the order of any long step. The scan is re-run on each draw, and
  every detected step is a false alarm, reported per Primary-year. Real steps remain inside the
  blocks, so the rate is a floor.
- **Recall on labelled events.** The share of event-by-Primary rows of experiment 400's corrected
  table (184 rows) found by the scan as a mirrored step within 2 days of the label, in all and in
  the subset lasting 7 days or more. The brief says 78 events; the table holds 184 event-by-Primary
  rows and the plan reports against the table. NGED's logs are incomplete, so this recall is a
  lower bound on recall and says nothing about unlabelled events.
- **Look at the candidates.** A chart of each unmirrored step (the Primary's residual, its
  neighbours', the segment means) and a second Sonnet reviewer who classifies each as plausible
  level change, weather or calendar artefact, outage or data fault, or unclear. No cause is claimed.

## Pre-registered parameters (added after the plan review, before any result)

- **Penalty.** `3 * ln(n) * sigma^2`, with `n` the Primary's number of valid days and `sigma` the
  median absolute difference of the daily residual at lag 7, times 1.4826, divided by `sqrt(2)`
  (lag 7 limits the understatement from autocorrelation). Sensitivity: multipliers 2 and 5. The
  dynamic programme is O(n^2) for about 2,600 days per Primary and is timed on one Primary first.
- **Mirror test.** Symmetric: the ratio of the two steps' magnitudes in MW lies between 0.5 and 2,
  with MW = points * effective capacity / 100. The neighbour's step is its raw change in segment
  mean across the same dates, and it need not pass the neighbour's own thresholds. The report also
  gives the share of unmirrored steps that become mirrored under the 3-day start rule and under the
  sum rule. The unmirrored count under the 1-day rule is an upper bound. Neighbour set for the
  headline: experiment 395's label-table pairs, because that table is the corrected one. A
  temporary event is tested for a mirror first, and counts as temporary only if unmirrored.
- **Found.** An injected or labelled step counts as detected if a step of the same sign starts
  within 2 days of it, with a size within 50%.
- **Windows.** Injection windows are chosen with a looser scan (penalty multiplier 2, spread
  multiple 2, floor 1 point) that finds no change point, no labelled event, and no mirrored
  candidate at any Primary. Recall is reported on these windows and on random windows as a control.
  Recall is conditional on scan-quiet windows.
- **Long steps.** Injected durations are 7, 14, 30, and 90 days. The headline covers steps of 7 to
  60 days. Longer steps are partly absorbed by the 6-month cross-fit, so their count is a lower
  bound. One Primary and 30-day steps are also injected into the half-hourly series and refitted
  to measure the difference from daily injection.
- **Bases of the numeric thresholds.** The false-alarm limit of 0.2 per Primary-year and the P2
  limit of 1 per Primary-year are judgement calls, chosen so that false alarms stay below a
  tenth of the expected real steps.
- **Method disagreement.** Steps in the five months found by both methods, by one only, are
  tabulated.
- **Share of half-hours inside an unmirrored step.** The half-hours of the segment between an
  unmirrored step and the next step, within the step's persistence, divided by all valid
  half-hours.
- **Recall per event.** Also reported (an event is found if any of its rows is found).

## Deviations from the brief

- Injected sizes are multiples of the detection threshold, not of the spread (code review
  finding: sizes of 1 to 3 times the spread are undetectable or on the threshold by construction).
  The block-bootstrap surrogate is scanned with the real series' spread and noise scale. The sum
  rule ignores neighbour changes under a quarter of the step. Method (a) is compared with method
  (b) by daily level (correlation) on the five months, because a 30-day month cannot hold a
  7-day-persistence step away from its edges.
- A step is flagged `near_block_boundary` when it starts within 3 days of a 6-month block
  boundary, and rates are also given without those steps.
- The 3-day minimum-persistence variant is replaced by 14 days (the penalty alone would report
  segments under 7 days).
- Recall is measured against 184 event-by-Primary rows, not 78 events.
- Method (b) is a 6-month-block cross-fitted XGBoost, not a monthly-refit model.
- Injection is into the daily residual, with one refit check.
- The share of training rows is replaced by the share of half-hours.

## Outputs

- A table of candidate steps (Primary, start, end, size in MW and in points, class, the nearest
  neighbour's change at the same time).
- Counts and rates per Primary-year by class, with the sensitivity grid, and the share of
  half-hours inside an unmirrored step, before and after removing steps that match a labelled
  event. The rate is a lower bound if the scan misses steps shorter than 7 days.
- The mirrored steps that match no labelled event, per Primary-year, listed for the maintainer as
  candidate unlabelled load-shift events.
- At least two charts of unmirrored steps and one of a mirrored step.
- A plain statement of what the scan cannot tell: a genuine level change that a neighbour mirrors
  by coincidence, a step shorter than 7 days, and any cause. Candidates are reviewed knowing that
  NGED's labels are incomplete, so an unlabelled candidate may be a real event.

## Standing practices

- Cleaned data: flagged zeros are nulls throughout. Fitted boosters are saved as `.ubj` under the
  experiment's data folder. No forecast is scored, so tail calibration, curtailment cost, and
  example-forecast plots do not apply. Parallel pools use `spawn`, one thread per process.
  Cadence: not applicable (cross-fit, not refit).
- Review order: Sonnet writes the code, a fresh Sonnet reviews it before the first run, results
  review, write-up reviews. All interim numbers are unaudited.
