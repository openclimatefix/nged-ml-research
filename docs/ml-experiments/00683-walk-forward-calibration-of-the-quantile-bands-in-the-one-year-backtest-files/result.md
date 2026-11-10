---
id: "00683"
slug: walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files
title: >-
  Can a scale on the width of the stored quantile bands, fitted only on earlier months, bring the
  q05-to-q95 coverage of the one-year backtest files to 90%?
theme: calibration-and-uncertainty
also_relevant_to:
  - production-reference-and-best-guess
status: finished
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
verdict: partly supported
best_arm: >-
  Arm 2 (the walk-forward scale) for the bulk supply points, grid supply points, solar, and the
  prevailing Primaries output: pooled q05-to-q95 coverage 88.0% to 90.9% (stored 84.0% to 89.3%),
  CRPS change +0.047 or less points of capacity. No arm closes the normal-running Primaries gap
  without a CRPS rise: arm 2 gives 87.1% coverage at +0.081 points (interval 0.064 to 0.099).
headline: >-
  Of 20 file-by-lead-band cells, 16 pass both tests on the months before July. The 4 that fail are
  the normal-running Primaries, where arm 2 reaches 87.1% coverage in every band (within 0.03 of
  0.90, so P1 holds) but raises the CRPS by 0.081 points of capacity (interval 0.064 to 0.099),
  above the 0.05 limit. The stop rule fired: arm 2's pooled normal-running Primaries coverage is
  80.1% in 2025-12 and 80.3% in 2026-01. The level shift (arm 3) makes every file worse (Primaries
  CRPS +0.306, interval 0.253 to 0.362).
lessons: >-
  A scale on the band half-width fixes coverage where the stored bands are a little narrow
  everywhere (the bulk and grid supply points, solar, the prevailing Primaries) and cannot fix
  normal-running Primaries coverage in winter, where the shortfall (77.7% in 2026-01) comes from
  error in the median that widening does not remove. A scale fitted through a load transfer
  overshoots after it ends (Primary 13: 97.5% in 2026-09), and a level shift fitted per series
  pools those events into the median and lowers skill.
take_home_message: >-
  Widening the stored bands by a factor fitted only on earlier months brings the bulk supply
  points, grid supply points, solar, and the prevailing Primaries output to within 2 points of the
  nominal 90% coverage without costing accuracy. It does not rescue the normal-running Primaries:
  coverage stays at 87% on average and at 80% in two winter months, so the bands there are
  wrong by more than their width. A learned width head, or a fix to the median in winter, is the
  next thing to try, and a median-level shift fitted per series made every file worse.
advice:
  - >-
    Calibrate the half-width of the bands with a per-series, per-lead-band scale fitted on the
    previous 90 days, for bulk supply points, grid supply points, solar, and the prevailing
    Primaries output, not for wind (coverage there is already above 90%) (arm 2 against arm 0).
  - >-
    Do not add a per-series level shift to a scale calibration: it raised the Primaries CRPS by
    0.306 points of capacity and the grid supply points' by 0.842 (arm 3 against arm 0).
  - >-
    Test a calibration against months that contain a load transfer before porting it: a scale
    fitted through the July and August 2026 transfers gave 97.5% coverage at Primary 13 in the next
    month, so hold the scale back while a transfer is under way (Primary 13 slice).
  - >-
    Expect no calibration scale to bring normal-running Primaries coverage to 90% in winter:
    2026-01 stays at 80.3% under a scale, so propose a learned width head or a median correction
    as a separate experiment (stop rule, monthly figure).
production_status: candidate
adopted_in: none
---

**The stored quantile bands cover the actual power less often than their 90% label says, and a
scale fitted on earlier months fixes that for five of the six outputs but not for the normal-running
Primaries.** We took the five stored one-year forecast files (the 16 Primaries, with a normal-running
and a prevailing output, the 2 bulk supply points, the 2 grid supply points, 3 wind farms, and 6
solar farms) and rescaled the distance of every quantile from the median by a number fitted only
on forecasts whose actual power was already known at the start of each month. We then counted how
often the actual power fell between the q05 and q95 quantiles. On the months from December 2025 to
June 2026, the scale brought the bulk supply points to 89.8% to 90.9%, the grid supply points to
89.1% to 89.8%, and solar to 89.4% to 91.0% (stored: 87.5% to 89.3%, 86.4% to 90.3%, and 87.8%
to 89.5%). It left the normal-running Primaries at 87.1% (stored 82.4% to 84.5%), at a cost of 0.081
points of capacity in CRPS. It also gave 80.1% in December and 80.3% in January, which triggers the
pre-registered stop rule: width is not the whole problem there. The method is a post-processing
step, and it uses a calibration idea from [experiment
630](../00630-roster-upper-demand-tail-calibration-event-days-and-bias-correction/result.md).

**Verdict: partly supported.** 16 of the 20 file-by-lead-band cells pass both tests of the plan
(coverage within 0.03 of 0.90, and CRPS no more than 0.05 points of capacity above the stored file).
The 4 cells that fail are the normal-running Primaries in each lead band, on the CRPS test. The
plan was committed first, in the same pull request. Numbers below are from the
origin months 2025-12 to 2026-06 ("before July") unless a line says otherwise. The months from
2026-07 are the final-test window, described but never used to choose.

## Figures

![Coverage by lead band](coverage_by_lead_band.svg)

*Pooled q05-to-q95 coverage by lead band, for each arm and file, on the months before July. The
dashed line is the nominal 90%. Wind and solar are pooled over farms.*

![Monthly Primaries coverage](monthly_primaries_coverage.svg)

*Pooled normal-running coverage of the 16 Primaries per origin month, all lead bands. The dashed
line is the 86% stop line and the solid line is 90%.*

![Band width at refits](refit_width_ratio.svg)

*Days 10 to 14 band width at each month start, over its value in the last 7 origins of the month
before, for the eight Primaries of the refit question. The grey band is the 5th to 95th percentile
of the ratio between the two seeds of one fold in experiment 667 (0.89 to 1.10).*

![Fan, 14 days](fan_14_days.svg)

*One 14-day forecast (origin 2026-01-14 12:00 UTC) of Primaries 13 and 17: stored on the left, arm
2 on the right. Blue is the median, with the 50% and 90% bands; orange is the observed power.*

![Fan, first day](fan_first_day.svg)

*The first day of the same forecasts.*

## Results

- **Arms.** Arm 0: the stored file. Arm 1: every quantile moved from `q50` by 1.0934. Arm 2: a
  per-series, per-output, per-lead-band scale, the 0.90 quantile of the distance of the actual power
  from `q50` over the stored half-width, fitted on the 90 days of valid times ending 6 hours before
  each month starts. Arm 3: arm 2 plus a per-series median-level shift. After each arm, the
  quantiles of MVA series, wind, and solar are clipped at zero.
- **Rows.** Cleaned observed power as in experiments 666 to 669 (wind idle half-hours set to zero;
  solar daylight only). The scored set is 2025-12 to 2026-10 origins: 3,574,946 half-hours per
  Primaries output, 448,098 at bulk supply points, 448,406 at grid supply points, 651,550 for wind,
  and 644,760 for solar. Experiment 668 also required finite manual-heuristic and climatology
  forecasts, so its row counts differ slightly (3,893,864 against our 3,897,273 for the Primaries
  over all 12 months).
- **Normal-running Primaries (before July).** Coverage is 82.4% to 84.5% stored, 86.0% to 87.6%
  under arm 1, 87.1% under arm 2, and 82.8% to 83.6% under arm 3. CRPS changes against the stored
  file are -0.027 (arm 1; interval -0.031 to -0.023), +0.081 (arm 2; 0.064 to 0.099), and +0.306
  (arm 3; 0.253 to 0.362) points of capacity.
- **Prevailing Primaries.** Arm 2: 88.0% to 88.4% (stored 84.0% to 87.9%), CRPS +0.047 (0.039 to
  0.055).
- **Bulk supply points.** Arm 2: 89.8% to 90.9%, CRPS +0.005 (-0.001 to 0.011). **Grid supply
  points:** 89.1% to 89.8%, +0.003 (-0.012 to 0.017). **Solar:** 89.4% to 91.0%, -0.002 (-0.006
  to 0.002). **Wind:** 90.4% to 92.0%, +0.007 (0.000 to 0.015). The stored wind file is already
  above 90% in every band, so the scale does not improve it.
- **Closer to 90% than the stored file:** arm 2 in 20 of the 24 file-by-band cells (the Primaries
  count as two files); arm 1 in fewer, because it overshoots wind, solar, and the bulk and grid
  supply points. Arm 3 is not preferred over arm 2: its CRPS is worse by 0.23 to 0.84 points.
- **Stop rule fired.** Pooled normal-running Primaries coverage under arm 2 is 80.1% in 2025-12
  (fitted on about 30 days) and 80.3% in 2026-01, below the 86% line, and under arm 3 78.7% and
  66.2%. The stored file has 81.7% and 77.7%. So the sensitivity runs at 45 and 180 days and the
  refit-ratio diagnostic were not run.
- **Primaries 13 and 16 in 2026-07 and 2026-08.** Stored coverage is 28.3% and 22.9% at Primary 13
  and 24.5% and 22.1% at Primary 16. Arm 2 gives 25.5% and 49.4% at Primary 13 (P5 holds: below
  0.80). Without those two series in those two months, pooled normal-running coverage of the other
  Primaries is 80.6% and 79.5% stored and 88.9% and 86.0% under arm 2 (all 16 series: 73.9% and
  72.4% stored, 81.1% and 82.6% under arm 2). After the event, the scale overshoots: Primary 13 has
  97.5% in 2026-09 and 97.8% in 2026-10 under arm 2.
- **Final-test window (2026-07 to 2026-10, descriptive).** Arm 2 coverage: normal-running
  Primaries 85.9% to 86.7% (stored 76.9% to 77.5%), prevailing 88.7% to 89.0%, bulk supply points
  86.8% to 88.5% (stored 83.7% to 84.9%), grid supply points 87.5% to 88.1% (stored 86.9% to
  89.0%), wind 89.4% to 93.3%, and solar 87.3% to 88.2% (stored 89.5% to 91.3%). The scale lowers
  the stored coverage at the grid supply points' long leads and at solar, so the gain is not
  uniform across the year.
- **Tails, normal-running Primaries (before July, all lead bands).** Share of actual power below
  q05: 8.06% stored, 5.43% arm 2 (nominal 5%); above q95: 8.99% stored, 7.47% arm 2; below q01:
  2.47% and 1.60%; above q99: 2.94% and 2.56%. Pinball loss rises under arm 2 at q05 (0.920 to
  0.939) and q95 (1.164 to 1.286). The curtailment cost is not reported, because it needs the
  node-plus-generators frame and a thermal limit. Wind's q01 hit share is 9% in every arm because
  the idle half-hours are scored as zero and the quantile is at zero. One scale applies to all 11
  levels, so the tails are only approximately calibrated.
- **Refit boundaries.** At the February month start the days 10 to 14 band width of Primaries
  8, 15, 17, 18, 24, and 25 widens by 1.26, 1.31, 1.39, 1.48, 1.21, and 1.24 times, and Primary 13
  widens by 1.48 times in September. Primary 11 narrows by 0.76 in April. The seed-to-seed ratio of
  the same band in one fold ranges from 0.89 to 1.10 (5th to 95th percentile; extremes 0.68 and
  1.29), so the February and September steps lie outside the seed spread. Under arm 2 the
  coverage at the six Primaries goes from 76.3% in January to 93.1% in February, 94.1% in March,
  and 91.0% in April, so the overshoot lasts more than the first month; P6's second clause is not
  supported. Whether the scale should be refitted at every model refit is not decided by this
  experiment, because the diagnostic that would decide it was not run. Experiment 667's folds had
  4.2 to 5.5 million training rows each (4,220,716 in 2025-11 to 5,472,275 in 2026-05).

## Checks

- **Median minus mean forecast:** not applicable, since the stored files hold no mean forecast.
  Arms 1 and 2 leave `q50` unchanged, so the error of the median is the same as the stored file's;
  arm 3 moves it. The mean fitted shift (stored median bias, points of capacity) is 0.9 for the
  normal-running Primaries, -0.2 for the prevailing, -0.4 for bulk supply points, 5.9 for grid
  supply points, 1.7 for wind, and 0.2 for solar; the grid supply points' bias is the reason arm 3
  hurts them most (CRPS +0.842 points of capacity).
- **Triggers fired:** trigger 5 (arm 3, which contains arm 2, is worse than arm 2). The explanation
  is the grid supply points' and the load transfers' effect on a per-series shift pooled over all
  lead bands; a fresh Sonnet audit is listed in the pull request. Triggers 1, 3, 4, 6, and 7 do
  not apply; trigger 2 does not fire, because the P2 failure is explained by the monthly and
  Primary 13 and 16 slices above.
- **Scale clipping.** The scale hit its bounds in 24 of 704 normal-running Primaries cells
  (bounds 0.5 and 3), in none elsewhere, and no cell fell back to a scale of 1.
- **Caveats stated in advance.** The first calibrated month, 2025-12, fits on about 30 days. The
  cleaned power comes from a flagging rule run over the whole table, so a fit uses cleaning
  decisions made with later data. NGED's event logs may miss load transfers, so the fitting rows may
  hold events that no label covers.

## What the next experiment should answer

A scale cannot make up for a median that is wrong in winter at the Primaries. The next experiment
is a learned width head on the normal-running Primaries (conditioned on temperature and the
recent error), scored on 2025-12 to 2026-01 against arm 2, with a scale that holds while a
load transfer is under way.
