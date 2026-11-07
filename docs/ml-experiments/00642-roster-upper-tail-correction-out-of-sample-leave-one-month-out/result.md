---
id: "00642"
slug: roster-upper-tail-correction-out-of-sample-leave-one-month-out
title: >-
  Can a correction fitted on other months bring the upper demand tail of K1 and the Candidate
  within 1.5 percentage points of nominal without raising CRPS by more than 1%?
theme: calibration-and-uncertainty
also_relevant_to: []
status: planned
kind: other
protocol: demand-v2
date: 2026-11-07
size: >-
  medium. Stored: tables and figures in this experiment's private folder only; no forecast is
  fitted and none is written for reuse. Several arms (five corrections), so more than one defensible
  design exists, but the arms are fixed by the series brief. Callers: this folder's scripts, which
  import experiment 630's folder (`e00630_common`, `e00630_prep`) and `studies.*`. One Sonnet plan
  review, one Sonnet code review before the first scored run, one Sonnet results review, one Sonnet
  write-up review, one Sonnet prose review.
hypothesis: >-
  A correction to the upper quantiles that is fitted without looking at the month it is scored on
  can bring the Candidate's and K1's event-free q90 and q95 exceedance within 1.5 percentage
  points of nominal (10% and 5%) on held-out months, because experiment 630 found the excess on
  ordinary days at most Primaries and a constant shift fitted on three months removed only part of
  it. The arms that let the correction vary with the Primary, lead band, hour of day, or the last 8
  weeks of errors can follow the season that one constant could not.
prediction: >-
  Written before any score is read. P1. At least one arm brings the Candidate's pooled held-out
  event-free q90 and q95 exceedance each within 1.5 points of nominal (8.5% to 11.5% and 3.5% to
  6.5%) with a CRPS ratio at or under 1.01. P2. The rolling conformal arm (C) tracks the season and
  has the smallest spread of q90 exceedance across the five held-out months, but it leaves the
  first weeks of each month uncorrected when the 8 weeks before hold too few residuals, so its
  pooled exceedance misses the bar by more than the per-Primary scale arm does. P3. The
  quantile-mapping arm (Q) is the only arm that repairs q90 and q95 together without a CRPS
  increase over 1%, because it fits the shape of the whole distribution; it also moves the lower
  tail, and its q05 exceedance leaves 3.5% to 6.5%. P4. Combining the 28-day median shift with the
  per-Primary scale (arm M) beats the per-Primary scale alone on the March 2026 fold, the month
  with the largest excess in experiment 630, and does not on the June 2026 fold. P5. K1 behaves like
  the Candidate (same arms pass), and K3 needs a larger correction.
decision_rule: >-
  Fixed before any score is read. Rows: the five development months of the roster export
  (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), lead 0 to 240 hours, every row with an observed
  power reading; the month is the month of the forecast origin. Never the July to October 2026
  window. Data and row preparation: experiment 630's frames (`e00630_prep.py`). Evaluation is
  leave-one-month-out: for each held-out month, every correction is fitted on the event-free rows of
  the other four months and scored on the held-out month, so no row is scored by a correction that
  saw it. This is a lab design (the fitting months include later months than the held-out month);
  only arm C is causal. The bar, per network (K1 and the Candidate are gating; K3 is a reference):
  an arm passes when, on the pooled event-free rows of all five held-out months, (a) its q90
  exceedance lies in 8.5% to 11.5%, (b) its q95 exceedance lies in 3.5% to 6.5%, and (c) its CRPS
  (twice the mean pinball loss over the 11 quantiles) is at most 1.01 times the uncorrected
  network's CRPS on the same rows. The verdict is `supported` if some arm passes for both K1 and
  the Candidate, `partly supported` if some arm passes for one of the two or passes (a) and (b)
  but not (c), `small gain` if no arm passes but an arm brings both levels at least half the way
  from the uncorrected exceedance to nominal at both networks with a CRPS ratio at or under 1.01,
  and `not supported` otherwise. The best arm is the passing arm with the lowest CRPS ratio.
  Corrections are never retuned after a score is read. Arms (fitted per Primary unless stated, on
  event-free fitting rows): (S) per-Primary scale c, applied to the distance above q50 of every
  quantile above q50, chosen on a grid from 0.5 to 3.0 in steps of 0.05 so that the training q90
  exceedance is closest to 10%; (B) per-lead-band and per-hour scale: S, followed by a pooled
  factor per lead band (0 to 24, 24 to 72, 72 to 120, 120 to 240 hours) and 4-hour UTC block of
  valid time, chosen by the same rule on the rows already scaled by S; (C) rolling conformal:
  for each origin day and Primary and lead band, the adjustment to each quantile above q50 is the
  empirical quantile (at the quantile's own level) of the residuals y minus that quantile over the
  event-free rows whose valid time is before 00:00 UTC of the origin day and whose origin day lies
  in the previous 56 days; a cell with fewer than 2,000 such residuals is left uncorrected and
  counted; (Q) quantile mapping of the whole distribution: for each Primary and lead band, the
  probability integral transform of each training row (the level at which y sits on the forecast's
  piecewise-linear quantile function, extended past q01 and q99 by the end segments' slope) is
  collected; the corrected quantile at level tau is the forecast's quantile function evaluated at
  the training PIT's tau-quantile; (M) the rolling 28-day median shift of y minus q50 (computed as
  in C, from the previous 28 days, causal) added to every quantile, followed by the per-Primary
  scale of S fitted on the shifted training residuals. Sort every corrected forecast across the 11
  levels. Intervals: 95% percentile intervals from 20,000 resamples of origin days within each
  held-out month (the independent unit is the day, since half-hour rows are autocorrelated), and,
  as the brief requires, a month-block bootstrap of the five held-out months for the pooled
  quantities; five months are too few for the month-block interval to be narrow, so the bar uses the
  point estimate and the intervals are descriptive. No multiplicity correction: the bar is a
  three-part threshold, not a hypothesis test.
benchmark: >-
  The uncorrected exported forecasts of the same network on the same rows, the nominal levels (10%,
  5%), and experiment 630's out-of-sample result (shift S: 12.6% and 7.2%; rescale R: 11.7% and 6.5%
  on the last two months).
row_set_id: roster_export_2026-11-07_dev_months
metric: >-
  Pooled and per-month event-free exceedance at q90 and q95 (and q05, q10 for the lower tail),
  CRPS and its ratio to the uncorrected network, threshold-weighted CRPS at each Primary's 95th
  percentile of the fitting months' power, and the avoidable curtailment cost at the six signed-MW
  Primaries; by Primary, lead band, and month.
code: ml-experiments/00642-roster-upper-tail-correction-out-of-sample-leave-one-month-out/
---

**This is the plan.** The finished write-up replaces this page.

**Question.** Experiment 630 found that the stored neural-network demand forecasts K1 and the
Candidate have too little weight on high demand: 13% to 17% of event-free half-hours exceed the
forecast 90th percentile (q90) against a nominal 10%. One correction fitted on three months and
applied to two removed only part of the excess, because the bias moves with the season and
differs between the metering groups. This experiment tests five corrections that can follow those
changes, each fitted on the other months and scored on the held-out month, and asks whether any
brings q90 and q95 exceedance within 1.5 points of nominal without raising the continuous ranked
probability score (CRPS) by more than 1%.

**Context and reused work.** The rows, the event-free flag, the temperature joins, and the
day-resampling bootstrap come from [experiment
630](../00630-roster-upper-demand-tail-calibration-event-days-and-bias-correction/result.md)
(its folder is imported). The rolling conformal arm follows the idea of [experiment
500](../00500-rolling-conformal-recalibration-of-the-demand-tails/result.md) and the conformal
quantile adjustment of [Romano et al. (2019)](https://doi.org/10.48550/arXiv.1905.03222); the
quantile-mapping arm is a recalibration of the probability integral transform in the style of
[Kuleshov et al. (2018)](https://doi.org/10.48550/arXiv.1807.00263). The stored forecasts already
include the D5 slow output offset, so arm M adds a rolling median shift on top of it, which is the
reading of "a combination with the D5 offset" used here.

**Earlier advice followed.** Experiment 630: fit by metering group and by season, check the
cold-night bias, report exceedance beside CRPS. Experiment 483: fit the recalibration on earlier
months only before quoting coverage (arm C is causal; the other arms are leave-one-month-out and
labelled as such); do not widen the lower tail without checking it (every arm reports q05 and q10).
Experiment 500: the rolling window and the 28-day median shift.

**Standing practices.** Monthly refit does not apply (no model is fitted); cleaned data and the D15
and D18 cut before 2025-03-01 come from the export's cleaned `observed_power` (the cut is applied
here to rows scored and fitted, which 630 did not do, so D15 and D18 rows before that date are
excluded and the count is reported). Tails (q01, q05, q95, q99 pinball and coverage), cold nights, the
avoidable curtailment cost, and example forecasts are reported.

**Compute.** CPU only, no GPU, well under an hour. Predictions are read from the export; no model
is saved.

**Reviews.** The plan, code, results, and write-up each get one fresh Sonnet review; the prose
review follows the merge rules.
