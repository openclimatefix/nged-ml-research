---
id: "00689"
slug: fixed-lookback-width-scale-versus-seed-averaging-at-refit-boundaries
title: >-
  Does a width scale fitted on a fixed lookback remove the steps in band width at the model refits,
  or does the seed average of the quantiles matter?
theme: calibration-and-uncertainty
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
date: 2026-11-10
size: >-
  medium. Stored: tables and figures only, no forecasting model is fitted and no prediction table is
  written. Two defensible designs exist for the seed arm (a scale fitted per seed and averaged, or
  one scale fitted on the seed mean), so the plan fixes the first and reports a cross-seed variant
  beside it. Callers: this folder's scripts only, which import `studies.*`, experiment 668's scoring
  library, and a copy of experiment 683's fitting rule. No contract, serving path, or degradation
  rule is touched. One Sonnet code review before the scoring run and one Sonnet prose review; no
  plan review, because the brief fixes the question, the prediction, and the stop rule.
hypothesis: >-
  The steps in band width at the model refits (Primaries 11 and 13, and six widenings in February)
  come from the new model's raw band width differing from the old one's, so a scale fitted on the
  previous 90 days of the old model's forecasts cannot remove them; the seed average of the
  quantiles is not the cause.
prediction: >-
  Written before any score is read. Scored origin months are 2025-12 to 2026-06; refit boundaries
  are the starts of 2026-01 to 2026-06, because the month before 2025-12 is uncalibrated and the
  boundary of 2026-09 lies in the final-test window. A cell is one series, one boundary, and one
  of the four lead bands (days 0 to 1, 1 to 3, 3 to 7, 7 to 14). The ratio of a cell is the median
  band half-width over the first 7 origin days after the boundary, divided by the median over the
  last 7 origin days before it. A cell is affected if the stored (uncalibrated) ratio is above 1.10
  or below 1/1.10. P1 (the brief's prediction, which can fail). Under arm A (the 90-day walk-forward
  scale of experiment 683), at least 90% of the affected cells have a ratio between 0.90 and 1.10.
  P2. The prediction is expected to fail, because the scale for a month is fitted on forecasts of
  the model that the refit replaces, so the calibrated ratio stays close to the stored ratio. P3.
  Seed averaging does not matter: arm B's ratio differs from arm A's by at most 0.05 in at least
  90% of the affected cells.
decision_rule: >-
  Fixed before any score is read. Verdict on P1 only: `supported` if at least 90% of the affected
  cells of arm A are inside 0.90 to 1.10; `not supported` otherwise, and the write-up then says
  plainly that the scale alone does not remove the boundary steps and names the series and
  boundaries that remain outside. No new method is proposed. P3 is reported beside it with the
  share of affected cells and its day-block bootstrap interval. Arm C (45 and 180 days) is
  reported, and decides nothing. Intervals are day-block bootstraps over origin days within the
  before and after windows of each cell.
benchmark: >-
  Arm 0, the stored seed-mean files (`primaries.parquet`), the quantity the ratios are measured
  against.
row_set_id: backtest_year_2025_11_to_2026_10
metric: >-
  The boundary band-width ratio per cell and the share of affected cells inside 0.90 to 1.10, per
  arm; q05-to-q95 coverage by lead band for each arm on origin months 2025-12 to 2026-06.
---

**Planned.** Experiment 683 measured steps in band width at the model refits. This experiment
asks whether a single width scale fitted on a fixed lookback removes them, or whether the seed
average of the quantiles drives them.

## Arms

- **Arm 0:** the stored normal-running quantiles. The prevailing output is reported separately
  where its per-seed forecasts exist; if they do not, the page says so.
- **Arm A:** the fixed 90-day walk-forward scale of experiment 683, fitted on the seed-mean
  quantiles (rule copied from `e00683_calibrate.py`).
- **Arm B:** the same rule fitted on each seed's quantiles from the per-seed files of experiment
  667, applied to the same seed's forecasts, then averaged over the two seeds. Variant B-cross
  fits on one seed and applies to the other.
- **Arm C:** arm A with a 45-day and a 180-day lookback.

## Row set and compute

- The 16 Primaries, normal-running output, origin months 2025-12 to 2026-06; the same rows as
  experiment 683.
- CPU only, at most 2 cores, no refits, no GPU.
