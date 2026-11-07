---
id: "00626"
slug: widen-the-tails-of-the-51-scenario-member-values
title: >-
  Can the tails of the 51 scenario values be widened until the q05 and q95 exceedance rates are
  within 1 point of 5%, without changing the Schaake-shuffled scenario paths' variogram and energy
  scores by more than 2%?
theme: calibration-and-uncertainty
also_relevant_to:
  - hierarchy-and-pooling
status: finished
kind: other
protocol: demand-v2
date: 2026-11-07
size: >-
  medium. Stored: tables and figures in this experiment's private folder only; no forecast is fitted
  or written for reuse. Designs: more than one defensible (rescale about the median, or re-draw the
  51 values from a wider quantile function), so the plan registers each. Callers: this folder's
  scripts only, which read experiment 436's stored run tables and Schaake-shuffled scenario paths
  (J3) and call no reviewed module beyond `studies.*`. No fit of a forecasting model. One Sonnet
  code review before the first run, one Sonnet results review, one Sonnet write-up review.
hypothesis: >-
  In experiments 434 and 436 the 51 values per node and half-hour were too narrow in the tails: the
  q95 of the 51 values was exceeded by 7.6% to 8.8% of actuals (target 5%). A factor per family and
  lead band, fitted on earlier months only, that rescales the 51 values about their median (arm B),
  or re-draws them from a wider quantile function (arm C), brings the q05 and q95 exceedance rates
  to within 1 point of 5% in every family and lead band on later months, while the variogram and
  energy scores of the Schaake-shuffled paths move by less than 2% and the CRPS of the 51 values
  moves by less than 1%.
prediction: >-
  Written before any score is read. P1. Arm A (unchanged) reproduces the exceedance rates of
  experiments 434 and 436 (7% to 9% at q95 for demand). P2. Arm B and arm D bring q05 and q95
  exceedance within 1 point of 5% in the pooled bands, and fail in at least one family and lead band
  cell, because one factor cannot fix demand's tails at 12 to 36 hours and at days 5 to 10 together.
  P3. Widening raises the CRPS of the 51 values by 0.5% to 3% in the bands where the rates were
  already near 5% and lowers it elsewhere, so the 1% CRPS limit fails in at least one family. P4.
  Path scores move by under 2% for demand and wind and by more than 2% for solar (the solar
  variogram is dominated by the daylight steps whose spread the widening changes most). P5. The
  fitted factors differ by more than 30% between folds for wind (3 farms, few runs).
decision_rule: >-
  Fixed before any score is read. Rows: the 44 Tuesday runs of experiment 436, 2025-09 to 2026-06.
  Folds: test months 2025-12 to 2026-06 (7 months); the factors of a test month are fitted on all
  earlier runs (at least 3 months), never on the test month. Scored rows: half-hours with an actual
  and a non-degenerate forecast (the largest of the 51 values above the smallest), in leads 0 to 240
  hours, in four lead bands (0 to 12, 12 to 36, 36 to 120, 120 to 240 hours). Exceedance rates use
  the weibull quantile of the 51 values (position k/52) at levels 0.05, 0.10, 0.90, 0.95; an actual
  exceeds q95 if it is strictly above, and is below q05 if strictly below. `supported` if one arm
  brings both the q05 and the q95 exceedance rate within 1 point of 5% (4% to 6%) in all 12 cells (3
  families x 4 lead bands), with the CRPS of the 51 values no worse than 1% (per family, pooled over
  the bands), and the variogram and energy scores of the paths no worse than 2% in the days 1 to 10
  joint, spatial and temporal blocks. `partly supported` if the same holds only for the rates pooled
  over all 25 nodes in each band, or only for some families. Otherwise `not supported`. Stop rule:
  if a fitted factor for a family and band differs by more than 30% (largest divided by smallest)
  between folds, stop, report the factors, and do not score the arm that used them. No rescue tuning
  after a failed bar. Intervals: month-block bootstrap (10,000 resamples of the 7 test months), 95%,
  with no multiplicity correction because there is no single gating contrast beyond the stated
  cells.
benchmark: >-
  Arm A, the unchanged Schaake-shuffled scenarios (J3) of experiment 436, scored on the same rows.
row_set_id: joint_tuesdays_v1
metric: >-
  Exceedance rate at q05, q10, q90, q95 per family and lead band; CRPS of the 51 values; energy
  score and variogram score (order 0.5) of the paths in experiment 436's blocks; slices of cold
  winter nights (demand, valid 18:00 to 06:00 UTC in December to February) and very sunny half-hours
  (solar, median of the 51 values at least 50 points of capacity).
verdict: not supported
best_arm: >-
  None. Arm D (separate upper and lower factors) came closest: 6 of 12 family-by-band cells had a
  q95 exceedance rate within 1 point of 5% and 7 of 12 had a q05 rate within 1 point, against 7 and
  4 for the unchanged arm A. Pooled over the 25 nodes, arm D put q05 at 4.4% to 5.1% and q95 at 5.6%
  to 6.1% in the four lead bands, missing the bar by 0.1 point at 0 to 12 hours.
headline: >-
  Widening the 51 values did not fix the tails, because the demand forecast looks biased low rather
  than too narrow: at 12 to 36 hours the share of Primary actuals above q90 was 14.2% against 10%,
  and
  below q10 was 6.4% against 10%. A factor fitted on earlier months moved demand's upper tail the
  wrong way (arm B raised its q95 exceedance from 4.7% to 7.1% at 12 to 36 hours), and separate
  factors (arm D) brought the lower tail to 4.1% but left q90 at 15.6%. CRPS moved by -0.4% to +0.4%
  and the joint variogram score of the paths by +0.6% to +1.1%.
lessons: >-
  The tails of the stored forecasts are not uniformly too narrow. Demand's lower tail is too wide
  and its upper tail too narrow, and wind and solar exceed q95 at rates of 4.6% to 8.0%. Arm A did
  not reproduce the 7.6% to 8.8% q95 exceedance quoted in experiment 436, and the table does not
  show why; the definition of the quantile is the first suspect. A shift of demand's median, not a
  widening, is the likely need, but no table measures the median bias directly.
take_home_message: >-
  Rescaling or stretching the 51 values about their median did not bring the q05 and q95 exceedance
  rates to within 1 point of 5% in every family and lead band, and it cannot fix demand, whose
  forecast sits too low (12% to 17% of actuals above q90, 6% to 8% below q10). The scenario paths
  and the CRPS were hardly affected (joint variogram score +0.6% to +1.1%, CRPS -0.4% to +0.4%), so
  widening cost little on the stored forecasts. The next step is a bias correction for demand, then
  a re-check of the tails.
advice:
  - >-
    Check the median bias of demand before widening the tails of the 51 values, because the demand
    forecast exceeded q90 for 12% to 17% of actuals and fell below q10 for 6% to 8% (figure 1,
    exceedance table).
  - >-
    Quote tail exceedance with the weibull quantile of the 51 values (position k/52), and state the
    definition, because experiment 436's 7.6% to 8.8% q95 figures do not reproduce under it
    (exceedance table).
  - >-
    Fit one factor for the upper side and one for the lower side, not one factor for both, when the
    two tails miss in opposite directions (arm D against arm B).
code: ml-experiments/00626-widen-the-tails-of-the-51-scenario-member-values/
production_status: not-for-production
adopted_in: none
---

**Widening the 51 scenario values about their median did not bring both tails to 5% in every family
and lead band, mainly because the demand forecast is biased low, so its upper tail is too narrow
while its lower tail is already too wide.** The question was whether a factor per family and lead
band, fitted on earlier months, could bring the q05 and q95 exceedance rates to within 1 point of 5%
without disturbing the Schaake-shuffled scenario paths. The answer is no for all four arms: the best
arm (D) met the bar in 6 of 12 cells at q95 and 7 of 12 at q05. The scenario paths and the CRPS
barely moved, so the cost of widening is small. What it narrows: the tail problem of experiments 434
and 436 is a demand median bias plus modest wind and solar tails, not a uniform shortage of spread.

![Exceedance rate at q95 and q05 by arm, family, and lead band; 95% month-block intervals](figure_1_exceedance_by_arm.svg)

![Winter example: Primary D1, day ahead and ten days, arms A and D](figure_2_example_winter.svg)

![Summer example: Primary D1, day ahead and ten days, arms A and D](figure_2_example_summer.svg)

The figures were written by the scripts and not rendered by the author, because no renderer was
available; a reader should check them.

- **Arms.** A unchanged (experiment 436's stored 51 values and J3 Schaake ranks); B one factor per
  family and lead band applied above and below the median; C values re-drawn at levels stretched
  about the median, with one stretch per family and band (end values extrapolated by the end
  slopes); D separate upper and lower factors. Wind and solar values are clipped to 0 to 100 points
  of capacity. Every arm keeps J3's ranks.
- **Rows.** 44 Tuesday runs of experiment 436; 7 test months (2025-12 to 2026-06), 36 runs; each
  month's factors fitted on all earlier runs (13 to 39 runs). Scored rows: leads to 240 hours with
  an actual and a non-degenerate forecast. Data: experiment 436's cleaned actuals. No model was
  fitted; retraining cadence is not applicable.
- **Exceedance, arm A (q95 / q05, share of actuals beyond the weibull quantile of the 51 values).**
  Demand 3.9% to 6.2% / 1.8% to 2.5%; wind 4.6% to 8.0% / 5.2% to 6.0%; solar 5.2% to 6.5% / 4.4% to
  7.3%. Demand above q90: 11.7% to 16.8% (target 10%); below q10: 6.2% to 7.8%.
- **Arm D, pooled over families, by band (0 to 12, 12 to 36, 36 to 120, 120 to 240 hours).** q05
  5.1%, 4.5%, 4.4%, 4.8%; q95 6.1%, 5.6%, 5.8%, 5.7%. Arm A pooled: q05 3.3%, 3.0%, 2.7%, 3.4%; q95
  4.3%, 4.8%, 5.6%, 6.3%.
- **Cells within 1 point of 5% (of 12).** q95 / q05: A 7 / 4; B 3 / 5; C 5 / 4; D 6 / 7. Arm B
  raised demand's q95 exceedance in every band (for example 4.7% to 7.1% at 12 to 36 hours) because
  its single factor, below 1 for demand, narrowed the upper tail.
- **Cold winter nights (demand).** Arm A: above q90 16.9% to 18.3%, above q95 6.2% to 7.4%. Arms B
  and C raised above-q95 to 8.9% to 11.2%; arm D 7.4% to 10.1%. **Very sunny half-hours (solar).**
  Arm A above q95 5.4% to 7.3%; arms B to D 8.1% to 13.5%, so widening hurts the sunny slice.
- **CRPS of the 51 values, relative to A (bands pooled, 95% month-block interval).** Demand B +0.4%
  (0.0% to +0.6%), C +0.1% (-0.0% to +0.3%), D -0.2% (-0.6% to +0.1%); wind B -0.2%, C -0.2% (-0.4%
  to -0.0%), D -0.4% (-1.1% to +0.8%); solar B +0.3% (-0.0% to +0.9%), C +0.4%, D +0.4%. All within
  the 1% limit. Absolute CRPS of A: demand 5.08, wind 10.47, solar 8.61 points of capacity.
- **Path scores, days 1 to 10, relative to A.** Joint variogram B +1.1% (+0.7% to +1.5%), C +0.6%
  (+0.2% to +1.0%), D +0.7% (+0.5% to +1.0%); joint energy within 0.2%. Spatial variogram +0.6% to
  +1.0%; temporal variogram within 0.2%. All point estimates are within the 2% limit; the spatial
  variogram interval for arm B reaches +2.5%. Joint variogram of A: 7.39.
- **Tail calibration at q01 and q99 and pinball loss.** Not reported: the stored run files hold 51
  values, not the q01 and q99 quantile forecasts, and the scenario values stop at levels 1/52 and
  51/52.
- **Registered predictions.** P1 failed (arm A does not reproduce 7% to 9%; the 7.6% to 8.8% quoted
  by experiment 436 used a different definition). P2 held: arm D pooled came within 0.1 point of the
  bar (6.1% at 0 to 12 hours), arm B pooled missed in all bands (q95 6.2% to 7.4%, q05 3.1% to
  4.5%), and arms B to D failed in at least one cell. P3 failed (pooled CRPS rises were at most
  0.4%, and the largest per-band rise, solar arm D at 0 to 12 hours, was about 0.9%, so none
  exceeded the 1% limit). P4 cannot be judged: the path scores are pooled over families, and none
  moved by more than 1.1%. P5 failed (wind factors changed by at most 14% between folds; the largest
  change was 28%, for solar).
- **Stop rule.** Largest-to-smallest ratio of a fitted factor across folds: 1.28 (solar, arm D upper
  factor, 0 to 12 hours), under the 30% limit, so all arms were scored.
- **Caveats.** Demand's lower factor reached the grid floor of 0.8 in six of seven folds up to 120
  hours (never at 120 to 240 hours), so the grid, not the data, set that factor; the registered grid
  was kept. Solar's lower quantile is often 0 after clipping, so a solar q05 cell can sit well under
  5% for that reason. Training runs from late in a month have leads of up to 10 days whose actuals
  fall in the next month, the test month (a small leak in the letter of "earlier months"). Seven
  test months and 36 runs; intervals are month-block. The fit's clipping is applied to the
  interpolated quantile and the scoring's clipping to the values before interpolating (negligible).
  Experiments 577 and 608 were not reused because their widening applies to a different forecast.
- **Reviews.** A fresh Sonnet agent reviewed the code before the first scoring run and found no
  blocking bugs (items above are its low-severity points, kept as caveats). The numbers were not
  re-derived by a second reviewer, and no separate prose review ran.
- **What to try next.** Remove the demand median bias (for example a per-Primary, per-lead offset
  fitted on earlier months), then refit arm D on the debiased values and re-check the tails; then
  compare with experiment 483's table P.
