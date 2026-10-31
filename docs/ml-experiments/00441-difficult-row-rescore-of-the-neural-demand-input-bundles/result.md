---
id: "00441"
slug: difficult-row-rescore-of-the-neural-demand-input-bundles
title: >-
  Do the input bundles of experiment 437 lower the neural network's error on cold winter nights,
  cold sunny days, Christmas, and heat waves, without raising it elsewhere?
theme: demand-inputs
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  simple. A free re-score of stored forecasts: no model is fitted, nothing is downloaded, and the
  only new files are slice masks and cached score tables in this experiment's private folder.
  Experiment 437's folders are read and never written. One design (the slices are fixed from ENS
  weather and the calendar before any score is read). Production serving path: not touched (this
  repository holds none). Degradation rule: not touched. Callers: the scripts import experiments
  11, 87, 377, 381, 396, 397, 407, 418, 425, and 437 and the `studies` evaluation code.
hypothesis: >-
  The annual mean score of experiment 437 hides a gain that is concentrated on difficult rows. The
  maintainer expects thermal-lag columns (bundle T) to help mostly on cold winter nights, solar-gain
  columns (bundle S) to help on cold sunny days, look-ahead columns (bundle F) to help on cold winter
  nights and heat waves, and calendar and tariff columns (bundle C) to help at Christmas and on bank
  holidays. NGED needs good forecasts on exactly those rows, with no harm on the others.
prediction: >-
  Written before any arm is scored and before any score is read. Numbers are the change in CRPS
  (points of effective capacity, days 1 to 10, widened by 1.15) of an arm minus its control; a
  negative number is a gain. P1. T and T_pre lower the CRPS on cold winter nights by 0.05 to 0.30 at
  the MVA-only eight (experiment 437 found its small gains at the eight in the two winter folds) and
  by -0.20 to +0.10 at the signed-MW six. P2. S and S_pre lower the CRPS on cold sunny days by 0.00
  to 0.20 at the six; the 437 reading of S on cold sunny days was +0.31 at the six (worse), so this
  prediction can fail. P3. F and F_pre change the CRPS on cold winter nights by -0.10 to +0.10. P4.
  On all rows outside each primary slice, every arm stays within +-0.06 of its control (the range
  of 437's annual contrasts). P5. No slice detects 0.05: the minimum detectable effect printed
  before scoring is above 0.05 in every cell, so at most one or two arm-slice cells can meet the
  decision rule, and a failure to meet it is not evidence of no effect.
decision_rule: >-
  Interim; four folds (2025-10, 2025-12, 2026-01, 2026-03), seeds 0 to 2 averaged in quantile space,
  widened by 1.15, monthly refits, the rows and controls of experiment 437 (`R` for T, S, C,
  C_tariff, and F; `A` for T_pre, S_pre, and F_pre). Slices are defined from the ENS-mean weather of
  each forecast's own run and from the calendar (the definitions are in `e00441_masks.py`), and are
  fixed before any arm is scored. Primary contrasts (Bonferroni over these only, 10 tests, level
  99.5%): T and T_pre on cold winter nights at the six and at the eight; F and F_pre on cold winter
  nights at the six and at the eight; S and S_pre on cold sunny days at the six. Every other arm-slice
  cell is descriptive. The tests the counts allow were fixed after reading the row and day counts
  and before reading any score: a slice with fewer than 20 distinct valid dates in a group is
  descriptive only, which leaves cold sunny days at the eight (19 dates), cold overcast days (6),
  Christmas (10), other bank holidays (4, one fold), and heat waves (none, because no fold holds a
  summer month) descriptive. An arm advances to a focused follow-up at a group if, on its primary
  slice, (1) the CRPS difference is at most -0.05, (2) the month-block bootstrap interval (resampling
  whole origin months, 20,000 resamples, level 99.5%) lies wholly below zero, (3) the difference is
  negative in at least 3 of 4 folds or in every fold that holds rows of the slice, (4) on all other
  rows the upper end of the same interval is at most +0.02, and (5) the increase in mean pinball loss
  at q95 and at q99 is at most +0.02 both on the slice and on all other rows (point estimates).
  Where fewer than 3 origin months hold rows of a slice, the interval resamples 7-day blocks of
  valid dates instead, is labelled so, and cannot support condition (2). The minimum detectable
  effect of each cell is the standard deviation, over 7-day valid-date blocks, of the difference in
  mean CRPS between the control's seeds 0 to 2 and its seeds 3 and 4, divided by the square root of
  two (experiment 437's convention, a lower bound). It is printed by `e00441_score.py pre` before
  any arm is scored. The smallest gain worth carrying forward is 0.05 because the brief fixes it.
benchmark: >-
  `R` and `A` of experiment 425, as in experiment 437, with BG2, `manual_heuristic`, and
  `climatology` as absolute references on the shared half-hours.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles and pinball loss at q01, q05, q95, and
  q99, in points of effective capacity, from `studies.eval_protocol`; leads 0.5 to 240 hours (days 1
  to 10).
verdict: not-yet-run
best_arm: none yet
production_status: not-for-production
adopted_in: none
take_home_message: >-
  Not yet run.
code: ml-experiments/00441-difficult-row-rescore-of-the-neural-demand-input-bundles/
---

**Take-home message: not yet run.**

## Question

CRPS is the continuous ranked probability score, a single error number for a probabilistic forecast.
The signed-MW six and the MVA-only eight are the two groups of Primary substations by metering.

**This re-score asks whether the input bundles that did not help on average help on the rows NGED
cares about most.** Experiment 437 added eight bundles of extra input columns to the pooled neural
network and judged them on the annual mean. The maintainer expects thermal-lag columns and similar
bundles to help mostly in winter, so a gain on cold winter nights, Christmas, or heat waves would be
averaged away. No network is refitted: the stored forecasts of experiment 437 are scored again on
slices of difficult rows.

## Design

- **Forecasts.** The eight arms of experiment 437 (T, S, C, C_tariff, F, T_pre, S_pre, F_pre) and
  controls `R` and `A`, four folds, three seeds, loaded by `e00437_score.load_frames`. Experiment 437
  folder is imported; nothing in it is edited.
- **Slices.** Each is defined from the ENS-mean 2 m temperature and clear-sky index of the
  forecast's own run, from solar elevation, and from the calendar, per Primary. Thresholds that are
  percentiles come from the 730 days of runs before each fold's refit (runs exist from 2024-04-01, so
  the first fold's window is 18 months).
    - **Cold winter nights.** Elevation below 0, local month November to March, ENS-mean temperature
      at or below the Primary's 10th percentile of winter-night ENS temperatures in the window.
    - **Cold sunny days and cold overcast days.** Daily-mean ENS temperature below 5 C; sunny if the
      daily-mean clear-sky index is at least 0.7, overcast if at most 0.3. Daylight half-hours only.
    - **Christmas and other bank holidays.** 24 December to 2 January; and Good Friday to Easter
      Monday and the May and August bank holidays of England and Wales.
    - **Heat-wave days.** Daily-maximum ENS temperature at or above the Primary's 95th percentile of
      June to August daily maxima, on at least 2 consecutive days of the same run.
- **Counts before scoring** (from `work/pre_output.txt`, days are distinct local valid dates, rows are
  scored half-hours at days 1 to 10 summed over the folds and Primaries):

  | Slice | Signed-MW six | MVA-only eight | Folds with rows |
  |---|---|---|---|
  | Cold winter nights | 11,793 rows, 21 days | 16,958 rows, 28 days | 3 (2025-12, 2026-01, 2026-03) |
  | Cold sunny days | 7,005 rows, 24 days | 8,340 rows, 19 days | 3 |
  | Cold overcast days | 971 rows, 6 days | 1,011 rows, 5 days | 1 (2026-01) |
  | Christmas | 23,040 rows, 10 days | 30,720 rows, 10 days | 2 |
  | Other bank holidays | 5,664 rows, 4 days | 7,552 rows, 4 days | 1 (2026-03) |
  | Heat-wave days | 0 | 0 | 0 |

- **Minimum detectable effect, from the seed spread, before any arm is scored** (7-day valid-date
  blocks; control `R` or `A`; six or eight):

  | Slice | `R`, six | `R`, eight | `A`, six | `A`, eight |
  |---|---|---|---|---|
  | Cold winter nights | 0.23 | 0.16 | 0.21 | 0.12 |
  | Cold sunny days | 0.19 | 0.18 | 0.13 | 0.26 |

  Every cell is above the 0.05 bar, so **no slice can detect a gain of 0.05**: an arm meets the
  decision rule only if its gain is several times the bar. A failure to meet it is not evidence of
  no effect. The same figures over folds are smaller (0.02 to 0.16) but rest on three folds.
- **Standing practices.** Monthly refits (stored); cleaned data (flagged exact zeros dropped, 1,179
  half-hours in the scored set); the D15 and D18 cut at 2025-03-01; stored controls reused and no
  fit; tails reported; four-fold screen with every number labelled interim.
- **Figures.** Per slice, a gain-with-interval chart for every arm; example winter-night week of
  forecast, actual, control, and best arm; the sunny-minus-overcast gap of S and S_pre; a day-ahead
  and a 14-day example forecast.
- **Reviews.** Draft pull request; fresh Sonnet code review before the arms are scored (it found that the protocol's own `christmas` column overrode the registered mask, now fixed, which moved the Christmas count from 9 to 10 dates; the six tested cells were unaffected); results
  review with the audit triggers; write-up; prose review.

## Caveats registered before scoring

- **The folds hold no summer month,** so heat waves cannot be scored at all.
- **Christmas, other bank holidays, and cold overcast days have 10 dates or fewer and one or two
  folds,** so they are descriptive.
- **Winter nights come mostly from two folds** (2025-12 and 2026-01), with 3 dates in 2026-03.
- **The first fold's percentile window is 18 months,** not 24, because ENS runs are stored from
  2024-04-01.
