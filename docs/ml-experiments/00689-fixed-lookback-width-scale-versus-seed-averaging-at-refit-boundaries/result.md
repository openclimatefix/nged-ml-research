---
id: "00689"
slug: fixed-lookback-width-scale-versus-seed-averaging-at-refit-boundaries
title: >-
  Does a width scale fitted on a fixed lookback remove the steps in band width at the model refits,
  or does the seed average of the quantiles matter?
theme: calibration-and-uncertainty
also_relevant_to:
  - production-reference-and-best-guess
status: finished
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
verdict: not supported
best_arm: >-
  None. No arm removes the steps: 15.2% of the 151 affected cells have a calibrated ratio of 0.90
  to 1.10 under arm A (90 days), 15.9% under arm B (per seed), 9.9% and 7.3% under the 45-day and
  180-day lookbacks (stored: 0.7%).
headline: >-
  A fixed lookback does not remove the band-width steps at the refits. 23 of 151 affected cells
  (15.2%, percentile interval 14.6% to 24.5%) end inside 0.90 to 1.10 under the 90-day scale,
  against the predicted 90%. The scale widens the step in 67% of the affected cells, because it is
  fitted on the model that the refit replaces. Seed averaging does not drive the steps: arm B's
  ratio is within 0.05 of arm A's in 89.4% of cells.
lessons: >-
  The steps at the refits come from the new model's raw band width differing from the old model's,
  not from the seed average and not from the lookback. The scale carries the old model's width
  correction into the new model, so after a refit it can add to the step: Primary 18 at the
  February refit goes from a stored ratio of 1.39 to 3.62 at days 7 to 14.
take_home_message: >-
  A single width scale fitted on the previous 90, 45, or 180 days does not remove the steps in
  band width at the model refits, and often makes them larger (the median step of an affected
  cell goes from 15% stored to 24%). Fitting the scale per seed and averaging gives nearly the
  same ratios, so seed averaging is not the cause. The scale for a month is fitted on forecasts of
  the model that the refit replaces; a calibration that follows refits needs the new model's own
  forecasts, which do not exist before the refit.
advice:
  - >-
    Do not rely on a fixed-lookback width scale to hold band width steady across a refit: 15.2% of
    affected cells ended inside 0.90 to 1.10, and Primary 18's days 7 to 14 step grew from 1.39 to
    3.62 (arm A against stored).
  - >-
    A scale fitted per seed and averaged is not needed for the boundary question: its ratios are
    within 0.05 of the seed-mean scale's in 89.4% of affected cells (arm B against arm A).
production_status: not-for-production
adopted_in: none
---

**A width scale fitted on a fixed lookback does not remove the steps in band width at the model
refits, and seed averaging is not their cause.** Under the 90-day scale of experiment 683 (arm A),
23 of the 151 affected cells (15.2%) end with a ratio of 0.90 to 1.10 across the refit, against the
predicted 90%. The scale makes the step larger in 67% of the affected cells, because the scale for
a month is fitted on forecasts of the model that the refit replaces. Fitting the scale on each
seed's quantiles and then averaging (arm B) changes the ratios by at most 0.05 in 89.4% of the
cells, so seed averaging does not matter for the steps.

The question was whether the steps that experiment 683 measured (Primaries 11 and 13, and six
widenings in February) are removed by one scale per series and lead band, or whether they come from
the seed average of the quantiles changing at a refit. The answer is neither: the new model's raw
band width differs from the old model's, and a lookback of 45, 90, or 180 days changes the share of
cells that pass by a few points only. This narrows the search for a calibration that follows refits
to one that has some forecasts of the new model, such as a scale refitted a few weeks after each
refit, or a refit-aware scale; this experiment does not test either.

## Figures

![Share of affected cells inside 0.90 to 1.10](share_inside_ten_percent.svg)

*Share of the 151 affected cells whose after-to-before ratio of band width lies between 0.90 and
1.10, by arm. The dashed line is the predicted 90%. Whiskers are 95% percentile intervals over
resampled origin days, with one set of draws per refit boundary.*

![Ratio scatter](ratio_scatter.svg)

*Calibrated ratio against stored ratio for each affected cell, by lead band, arms A and B. The grey
band is 0.90 to 1.10 and the diagonal is no change. Points above the diagonal on the right are
steps that the scale has enlarged.*

## Arms and rows

- **Rows:** the 16 Primaries, normal-running output, origin months 2025-12 to 2026-06 (refit
  boundaries 2026-01 to 2026-06; 2025-11 has no scale and 2026-07 onward is not loaded). 384 cells
  (16 series, 6 boundaries, 4 lead bands); 151 are affected (stored ratio above 1.10 or below
  1/1.10). Affected cells by boundary: 36, 20, 15, 25, 31, 24 (January to June).
- **Cell ratio:** median band width (q95 minus q05, after the zero clamp for the MVA Primaries) of
  the first 7 origin days after the boundary over the median of the last 7 days before it. Ratios of
  the width and of the half-width agree away from the clamp.
- **Arm 0:** stored seed-mean file. **Arm A:** experiment 683's rule, 90 days, fitted on the seed
  mean. **Arm B:** the same rule fitted on each seed's quantiles of experiment 667's fits (moved
  3 hours to the 12:00 UTC origin, times capacity), applied to the same seed, averaged, then clamped.
  The seed fits are clamped at zero for the fit only. **B-cross:** each seed scaled by the other
  seed's fit. **Arm C:** arm A at 45 and 180 days.
- **Seed check:** the mean of the two seed files reproduces the stored file to 4e-6 (3,924,480
  rows). Only the normal-running output has per-seed files, so the prevailing output is not
  tested.
- **Code reuse:** the fitting rule is copied from `ml-experiments/00683-*/e00683_calibrate.py`,
  which also supplies `z_scores`, `band_index`, and `load_truth`; scoring helpers come from
  experiment 668. No cost cap or retraining involved: CPU only, no fits.

## Results

- **P1 (at least 90% of affected cells within 0.90 to 1.10 under arm A): not supported.** 15.2%
  (23 of 151; interval 14.6% to 24.5%). By lead band: 7 of 44, 8 of 46, 4 of 27, 4 of 34 for days 0
  to 1, 1 to 3, 3 to 7, and 7 to 14. Stored: 1 of 151.
- **P2 (expected to fail): held.** The median absolute step of an affected cell is 15% stored and
  24% under arm A; arm A is farther from 1 than stored in 67% of cells.
- **P3 (arm B within 0.05 of arm A in at least 90% of cells): narrowly not met, at 89.4%** (135 of
  151); B-cross is 86.8%. Arm B passes 24 of 151 cells (15.9%), against 23 for arm A.
- **Lookback (arm C):** 15 of 151 cells (9.9%) at 45 days and 11 (7.3%) at 180 days, against 23 at
  90 days. The steps do not follow the lookback.
- **Named steps:** at the February refit, days 7 to 14 ratios (stored, arm A, arm B) are 1.21, 1.25,
  1.26 for Primary 8; 1.22, 1.64, 1.59 for 15; 1.30, 1.51, 1.50 for 17; 1.39, 3.62, 3.45 for 18;
  1.17, 1.24, 1.24 for 24; 1.16, 1.27, 1.27 for 25. Primary 11 at the April refit goes from 1.30 to
  1.37 at days 0 to 1 and from 0.76 to 0.79 at days 7 to 14 (arm A). The September refit of
  Primary 13 lies after July and is not scored.
- **Coverage, q05 to q95, pooled by lead band, origin months 2025-12 to 2026-06 (weekly-block 95%
  intervals about 1.7 points wide each side):** stored 82.4% to 84.5%, arm A 87.1%, arm B 88.0% to
  88.1%, B-cross 88.0% to 88.2%, 45 days 86.8% to 86.9%, 180 days 86.9% to 87.0%. Arm B's coverage is
  about 1 point above arm A's; this experiment does not test why.

## Checks

- **Median minus mean forecast, mean-forecast error, scoring triggers:** not applicable, because the
  experiment writes no forecast and scores no CRPS. No audit trigger fired.
- **Code review:** a fresh Sonnet reviewer found the clamp before averaging in arm B, a fit on
  unclamped seed quantiles, independent day draws per cell, the final-test months being loaded, and
  duplicate-key and month-pairing guards. All were fixed before the numbers above were produced; the
  basic interval the reviewer suggested was tried and rejected, because cells chosen for an extreme
  stored ratio regress towards 1 under resampling and a basic interval then falls below 0.
- **Bias in the intervals:** the percentile interval for the stored arm (11.9% to 29.2%) lies above
  its point share (0.7%) for the same reason, so read the intervals as spread.

## Caveats and what to try next

- **Caveat:** the 7-day windows are short; one cell's ratio has noise of the order of the 10% limit.
  The pass share rests on 151 cells that share 6 boundaries.
- **Caveat:** the per-seed files exist for the normal-running output only, and for two seeds.
- **Next:** test a scale refitted from the first weeks of forecasts after each refit, or fitted on
  the new model's own back-test forecasts; either needs a refit-aware design.
