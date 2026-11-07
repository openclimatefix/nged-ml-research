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
status: planned
kind: other
protocol: demand-v2
date: 2026-11-07
size: >-
  medium. Stored: tables and figures in this experiment's private folder only; no forecast is
  fitted or written for reuse. Designs: more than one defensible (rescale about the median, or
  re-draw the 51 values from a wider quantile function), so the plan registers each. Callers: this
  folder's scripts only, which read experiment 436's stored run tables and Schaake-shuffled
  scenario paths (J3) and call no reviewed module beyond `studies.*`. No fit of a forecasting model.
  One Sonnet code review before the first run, one Sonnet results review, one Sonnet write-up
  review.
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
  exceedance within 1 point of 5% in the pooled bands, and fail in at least one family and lead
  band cell, because one factor cannot fix demand's tails at 12 to 36 hours and at days 5 to 10
  together. P3. Widening raises the CRPS of the 51 values by 0.5% to 3% in the bands where the rates
  were already near 5% and lowers it elsewhere, so the 1% CRPS limit fails in at least one family.
  P4. Path scores move by under 2% for demand and wind and by more than 2% for solar (the solar
  variogram is dominated by the daylight steps whose spread the widening changes most). P5. The
  fitted factors differ by more than 30% between folds for wind (3 farms, few runs).
decision_rule: >-
  Fixed before any score is read. Rows: the 44 Tuesday runs of experiment 436, 2025-09 to 2026-06.
  Folds: test months 2025-12 to 2026-06 (7 months); the factors of a test month are fitted on all
  earlier runs (at least 3 months), never on the test month. Scored rows: half-hours with an actual
  and a non-degenerate forecast (the largest of the 51 values above the smallest), in leads 0 to
  240 hours, in four lead bands (0 to 12, 12 to 36, 36 to 120, 120 to 240 hours). Exceedance rates
  use the weibull quantile of the 51 values (position k/52) at levels 0.05, 0.10, 0.90, 0.95; an
  actual exceeds q95 if it is strictly above, and is below q05 if strictly below. `supported` if one
  arm brings both the q05 and the q95 exceedance rate within 1 point of 5% (4% to 6%) in all 12
  cells (3 families x 4 lead bands), with the CRPS of the 51 values no worse than 1% (per family,
  pooled over the bands), and the variogram and energy scores of the paths no worse than 2% in the
  days 1 to 10 joint, spatial and temporal blocks. `partly supported` if the same holds only for
  the rates pooled over all 25 nodes in each band, or only for some families. Otherwise `not
  supported`. Stop rule: if a fitted factor for a family and band differs by more than 30% (largest
  divided by smallest) between folds, stop, report the factors, and do not score the arm that used
  them. No rescue tuning after a failed bar. Intervals: month-block bootstrap (10,000 resamples of
  the 7 test months), 95%, with no multiplicity correction because there is no single gating
  contrast beyond the stated cells.
benchmark: >-
  Arm A, the unchanged Schaake-shuffled scenarios (J3) of experiment 436, scored on the same rows.
row_set_id: joint_tuesdays_v1
metric: >-
  Exceedance rate at q05, q10, q90, q95 per family and lead band; CRPS of the 51 values; energy
  score and variogram score (order 0.5) of the paths in experiment 436's blocks; slices of cold
  winter nights (demand, valid 18:00 to 06:00 UTC in December to February) and very sunny
  half-hours (solar, median of the 51 values at least 50 points of capacity).
---

**Planned.** The arms are A (unchanged), B (the 51 values rescaled about their median by one factor
per family and lead band, fitted on earlier months), C (the 51 values re-drawn at levels spread
wider than k/52 from the quantile function, extended past its end values by the end slopes, with
one level-stretch factor per family and lead band), and D (B with a separate factor for the upper
and the lower side). Every arm keeps experiment 436's Schaake ranks (J3), so only the values
change. Experiment 577's tail widening (jump-risk widening of a sparse-step offset) does not apply,
because these forecasts hold no offset, and experiment 608's constants are fitted for a different
forecast; neither is reused.
