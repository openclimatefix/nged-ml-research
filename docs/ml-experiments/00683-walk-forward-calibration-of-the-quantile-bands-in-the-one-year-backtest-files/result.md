---
id: "00683"
slug: walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files
title: >-
  Can a scale on the width of the stored quantile bands, fitted only on earlier months, bring the
  q05-to-q95 coverage of the one-year backtest files to 90%?
theme: calibration-and-uncertainty
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
date: 2026-11-10
size: >-
  medium. Stored: one table of calibrated quantiles per arm per file in this experiment's private
  data folder (`predictions/`), and tables and figures; no model is fitted. Two defensible designs
  exist (a scale on the half-width, or a scale on each side separately; a fixed lookback of 45, 90,
  or 180 days), so the plan fixes the first and the 90-day lookback before any score is read.
  Callers: this folder's scripts only, which import `studies.*` and the scoring library of
  experiment 668. No contract, serving path, or degradation rule is touched. One Sonnet plan review,
  one Sonnet code review before the scoring run, one Sonnet results review, one Sonnet write-up
  review, one Sonnet prose review.
hypothesis: >-
  The quantile bands stored in the one-year backtest files are too narrow for the Primaries, the
  bulk supply points (BSPs), and the grid supply points (GSPs) (q05-to-q95 coverage 81% to 88%
  against 90%), and the shortfall is a width error that a per-series, per-lead-band scale on the
  half-width of the band removes, when the scale is fitted only on forecasts whose actual power was
  already known at the start of the month.
prediction: >-
  Written before any score is read. P1. Arm 2 (the walk-forward scale) brings the pooled q05-to-q95
  coverage within 0.03 of 0.90 in each of the four lead bands (days 0 to 1, 1 to 3, 3 to 7, 7 to
  14) for each of the five files, over the calibrated months before 2026-07 and over 2026-07 to
  2026-10 separately. P2. Arm 2 raises the CRPS by at most 0.05 points of capacity against arm 0
  in every file and lead band, and lowers it in the Primaries. P3. Arm 3 (arm 2 plus a per-series
  level shift) adds less than 0.05 points of capacity of CRPS gain over arm 2, because the stored
  medians are not biased on average. P4. The fixed factor 1.0934 (arm 1) leaves the Primaries below
  0.87 coverage, because the shortfall there (81%) is larger than the 9% widening closes. P5.
  Primaries 13 and 16 in 2026-07 and 2026-08 stay far below 0.90 under every arm (one-sided misses
  in opposite directions), because a scale fitted on earlier months cannot follow a load transfer.
  P6. The bands of the Primaries 11 and 13 and the six February-widening Primaries change at the
  month-start refit by more than the seed-to-seed spread of the same band, so a scale fitted before
  the refit mis-sizes the first month after it.
decision_rule: >-
  Fixed before any score is read. Rows: those that the pages of experiments 666 to 669 score (cleaned
  observed power finite; wind idle half-hours set to zero up to 2026-09-13 17:30 UTC; solar
  daylight half-hours only; the Primaries scored per output, normal running (NRA) and prevailing),
  restricted to the origin months with a fitted calibration, 2025-12 to 2026-10 (2025-11 has no
  earlier forecast to fit on, so all arms are scored on 2025-12 to 2026-10 only). Arm selection
  uses the origin months 2025-12 to 2026-06 only; 2026-07 to 2026-10 (the final-test window) is
  reported separately and never used to choose an arm, a lookback, or a clamp. `supported` if P1 and
  P2 hold on the months before 2026-07 for all five files. `partly supported` if they hold for
  some files; the write-up names the files. `not supported` if arm 2 leaves any file more than 0.03
  from 0.90 in two or more lead bands before 2026-07. Stop rule: if the pooled NRA coverage at the
  Primaries (all 16 series, scored rows, per origin month) stays below 0.86 in any calibrated month
  after arm 2 or arm 3, the write-up says that width is not the whole problem, names the months, and
  the experiment stops there (no further arms); the Discuss session then proposes a learned width
  head as a separate experiment. The stop rule is also evaluated without Primaries 13 and 16 in
  2026-07 and 2026-08, and the write-up gives both results. The lookback is fixed at 90 days; 45
  and 180 days are reported as a sensitivity on the months before 2026-07 only. Threshold for
  porting: the calibration is a post-processing step with no model change, so any scale whose
  coverage gain has an interval (day-block bootstrap, 95%) excluding zero and whose CRPS change is
  inside P2 is worth porting. Intervals are day-block bootstraps over forecast-origin days (the
  method of experiment 668).
benchmark: >-
  Arm 0, the files as stored (`primaries.parquet`, `bsps.parquet`, `gsps.parquet`, `wind.parquet`,
  and `solar.parquet`, the clamped versions that experiment 668 documents), and arm 1, the fixed
  width factor 1.0934 about the median that experiment 613 fixed.
row_set_id: backtest_year_2025_11_to_2026_10
metric: >-
  q05-to-q95 coverage and CRPS (twice the mean pinball loss over the 11 stored levels, in points of
  capacity) per file, lead band, origin month, and series; pinball loss and coverage at q01, q05,
  q95, and q99; the share of rows below q05 and above q95 per month; the width and median level of
  days 10 to 14 across month-start refits.
---

# Walk-forward calibration of the quantile bands in the one-year backtest files

*This page is the plan. It is rewritten as the write-up after the scoring run.*

## Question

**The stored quantile bands contain the actual power less often than their label says.** For the
Primaries, 81% of the half-hours fall inside the q05-to-q95 band in normal running (nominal 90%),
and the figures for the BSPs, GSPs, wind, and solar files are 87%, 88%, 91%, and 89% (the
[thorough check of the files](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result/)
documents the files). We ask whether a scale on the width of each band, fitted only on earlier
months, closes the gap, and whether a level shift on the median adds anything.

## Arms

All arms rescale the stored forecast and keep its median unless stated. After every arm, the
quantiles of MVA series, wind, and solar are clipped at zero, as the clamps of experiment 668 do.

- **Arm 0:** the file as stored.
- **Arm 1:** every quantile moves away from `q50` by the factor 1.0934 that [experiment
  613](../00613-nra-roster-contrasts-with-intervals-five-seeds/result.md) fixed.
- **Arm 2:** every quantile moves away from `q50` by a scale `s`, fitted per series, per output
  (Primaries), and per lead band (days 0 to 1, 1 to 3, 3 to 7, 7 to 14). At the start of each origin
  month, `s` is the 0.90 quantile of `z`, where `z` is the distance of the actual power from `q50`
  divided by the stored half-width on the side where the actual lies (`q95 - q50` above, `q50 -
  q05` below). That choice makes the scaled q05-to-q95 band cover 90% of the fitting rows exactly.
  The fitting rows are the 90 days of valid times ending 6 hours before the month starts, because
  power is available only up to 6 hours before an origin. No forecast with an origin or valid
  time after that cut-off enters the fit. `s` is clipped to 0.5 to 3. Months with fewer than 200
  fitting rows in a cell keep `s = 1` and are counted.
- **Arm 3:** arm 2 with a per-series, per-output level shift. The shift is the median of the actual
  power minus `q50` over the same fitting rows, added to every quantile, and `s` is then fitted on
  the shifted band.

The method follows the calibration idea of [experiment
630](../00630-roster-upper-demand-tail-calibration-event-days-and-bias-correction/result.md), which
fitted a shift and a rescale of the upper tail on earlier months for the stored demand networks
and advised fitting by metering group and season; we do not, because the width error is the
question here, and a per-series fit on a rolling window follows the season.

## Reuse and standing practices

- **Reused:** the five stored forecast files, the ground-truth files, and the capacity table of the
  backtest folder, and the scoring library of [experiment
  668](../00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
  (`e00668_score_lib.py`, imported on `PYTHONPATH`). Nothing is fitted, so no model file is saved.
- **Cadence:** the stored forecasts are the monthly refits of experiments 666, 667, and 669; the
  calibration is refitted every month, at the same boundaries.
- **Cleaned data:** scoring actuals are the cleaned power, as in experiment 668. The D15 and D18
  rows before 2025-03-01 lie outside the year, so the cut does not apply.
- **Final-test window:** every table is also split into the origin months before 2026-07 and 2026-07
  to 2026-10.
- **Tails and cost:** pinball loss and coverage at q01, q05, q95, and q99 are in the metric. The
  curtailment cost needs the node-plus-generators frame and a thermal limit, so it is not reported,
  and the write-up says so.
- **Anonymisation:** wind farms are Wind A to C and solar farms Solar A to F, with no coordinates and
  no calendar dates on a generator's series.

## Refit boundaries

For Primaries 11, 13, and the six Primaries (8, 15, 17, 18, 24, 25) whose bands widen in February,
the write-up gives the width of the q05-to-q95 band and the median level for days 10 to 14, by
origin, across the month-start refits. It reads the per-seed exports of [experiment
667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md) (two seeds
per fold) for the spread between seeds, and the training row count per fold. It then says whether
the calibration scale should be refitted at every model refit: it compares the first month after
each refit with the later months, under arm 2 (scale fitted before the refit) and under a variant
that holds the scale at its last value before the refit and rescales it by the ratio of the new
fold's median band width to the old one (a diagnostic only; not an arm). Experiment 677's interim
notes put monthly refits at 7.53 against 7.99 for quarterly refits (all-rows MAE), which is the
reason the calibration follows the monthly boundary.

## Process

- Draft pull request with this plan, one Sonnet plan review, one Sonnet code review before the scoring
  run, then the run, a results review, the write-up, a Sonnet write-up review, and a Sonnet prose
  review. The audit triggers are run on the results before the prose review.
- CPU only, at most 4 cores.
