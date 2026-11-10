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
  Written before any score is read. Scored months are the origin months 2025-12 to 2026-06
  ("before July") unless stated. P1. Arm 2 (the walk-forward scale) brings the pooled q05-to-q95
  coverage within 0.03 of 0.90 in each of the 20 cells formed by the five files and the four lead
  bands (days 0 to 1, 1 to 3, 3 to 7, 7 to 14). P2. Arm 2 raises the CRPS by at most 0.05 points
  of capacity against arm 0 in every cell, and lowers it in the Primaries pooled over lead bands.
  P3. Arm 3 (arm 2 plus a per-series level shift) gains less than 0.05 points of capacity of CRPS
  over arm 2, because the stored medians are not biased on average (the plan reports arm 0's mean
  signed error to check this). P4. Arm 1 (the fixed factor 1.0934) leaves the pooled NRA coverage
  of the Primaries, over all lead bands, below 0.87, because the shortfall (81%) is larger than the
  9% widening closes. P5. Primaries 13 and 16 in 2026-07 and 2026-08 stay below 0.80 pooled
  coverage under every arm (one-sided misses in opposite directions), because a scale fitted on
  earlier months cannot follow a load transfer. P6. At the month-start refits of Primaries 11 and
  13 and of the six February-widening Primaries, the days 10 to 14 band width changes by more than
  the width difference between the two seeds of the same fold, and arm 2's coverage error in the
  first month after such a refit exceeds 0.03 while the later months stay within it. The final-test
  window 2026-07 to 2026-10 is described and never enters a verdict.
decision_rule: >-
  Fixed before any score is read. Rows: those that the pages of experiments 666 to 669 score
  (cleaned observed power finite; wind idle half-hours set to zero up to 2026-09-13 17:30 UTC;
  solar daylight half-hours only; the Primaries scored per output, normal running (NRA) and
  prevailing). The fitting rows use the identical mask. Lead is measured from the power forecast
  origin, in four bands (days 0 to 1, 1 to 3, 3 to 7, 7 to 14); the five bands of experiment 668
  merge its last two, and days 10 to 14 are a sub-slice used only for the refit boundaries. All
  arms are scored on origin months 2025-12 to 2026-10 only (2025-11 has no earlier forecast to fit
  on). CRPS is the pooled mean over rows of each series' CRPS normalised by its capacity (points
  of capacity). Verdict, on the months before July only, by the 20 file-by-band cells of P1 and P2:
  `supported` if all 20 cells pass both; `not supported` if 5 or more fail either; `partly
  supported` otherwise, naming the failing cells. Choice of arm, on the months before July:
  arm 3 over arm 2 only if its pooled CRPS gain has a day-block bootstrap 95% interval excluding 0
  and exceeds 0.05 points of capacity; arm 2 over arm 1 if its absolute deviation of coverage from
  0.90 is smaller in more cells; otherwise the simpler arm. Stop rule: if the pooled NRA coverage
  of all 16 Primaries (all lead bands together, per origin month from 2025-12 to 2026-06) stays
  below 0.86 in any month after arm 2 or arm 3, with all months of all series, width is not the
  whole problem; the write-up names the months, and the sensitivity runs and the refit-ratio
  diagnostic are not run. A single noisy month, such as 2025-12 with its 30 fitting days, can
  trigger the stop; that is intended. The version without Primaries 13 and 16 in 2026-07 and
  2026-08 is reported as context and does not decide the stop. If the stop rule triggers, the
  Discuss session proposes a learned width head as a separate experiment. The lookback is fixed at
  90 days; 45 and 180 days are a sensitivity on the months before July, run only if the stop rule
  has not triggered. Porting threshold: the calibration is post-processing with no model change,
  so any scale that reduces the absolute deviation of coverage from 0.90 in a file and lead band,
  with a day-block bootstrap 95% interval of the reduction excluding 0 and a CRPS change inside
  P2, is worth porting. Refit question: the scale is refitted at every model refit unless the
  ratio variant (scale held at its last value, multiplied by the ratio of the median days 10 to 14
  band width in the first 7 origins after the refit to the last 7 before it) lowers the first-month
  coverage error by at least 0.01 pooled over the affected Primaries. Intervals are day-block
  bootstraps over forecast-origin days (the method of experiment 668).
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
  time after that cut-off enters the fit. `s` is clipped to 0.5 to 3. A cell with fewer than 20 distinct
  valid days of fitting rows keeps `s = 1`; rows whose stored half-width is under 0.05 points of
  capacity (clamped rows) are left out of the fit. The write-up counts the fallbacks and the
  cells where the clip binds. The shift of arm 3 is per series and output, over all lead bands.
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

## Figures

Take-home first. Then: coverage by lead band for each arm, per file (headline); band-width ratio by
month at the refit boundaries; example forecast fans before and after calibration for two Primaries
(a day-ahead forecast and the full 14 days, with the 50% and 90% bands). Lines are 1 px, panels at
least 280 px high, hues distinct. Tables of wind and solar are pooled over months when split by
series, and pooled over series when split by month, so no calendar date sits beside a generator.

## Caveats stated in advance

- The first calibrated month, 2025-12, fits on about 30 days, and the newest usable rows for the
  days 7 to 14 band are about a week older than the month start.
- The cleaned power comes from a flagging rule run over the whole table, which may use later data.
- One scale applies to all 11 levels, so q01 and q99 are only approximately calibrated, and one
  scale does not equalise the one-sided misses; the tail metrics show both.

## Process

- Draft pull request with this plan, one Sonnet plan review (triaged in the pull request), one Sonnet code
  review before the scoring run, then the run, a results review, the write-up, and a Sonnet prose
  review. The audit triggers are run on the results before the prose review.
- CPU only, at most 4 cores.
