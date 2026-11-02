---
id: "00500"
slug: rolling-conformal-recalibration-of-the-demand-tails
title: >-
  Does a rolling correction from the last days of delivered errors fix the upper tail of the
  signed-MW six?
theme: calibration-and-uncertainty
also_relevant_to: []
status: planned
kind: other
date: 2026-11-02
size: >-
  medium. Stored: tables and figures under the experiment's data folder only; no model is fitted.
  Serving path: not touched. Degradation rule: not touched. Designs: one defensible test, because
  the correction, buffers, and rule come from the series brief and the queue-order check. Callers:
  this folder's scripts, which import experiments 435, 470, 483, and 492. One Sonnet code review
  before the first scored run, one Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  A rolling correction, which shifts each quantile of A-pool by the empirical quantile of the
  residuals the forecaster delivered over the last 7, 14, or 28 days (rolling conformal), repairs
  the two clauses that table P misses at the signed-MW six in experiment 483: q95 coverage when
  table P is fitted on earlier folds only (0.927 against a floor of 0.93), and q99 pinball loss
  (+6.1% against a limit of 5%). It does so without pushing q01 or q05 out of range.
prediction: >-
  Written before any score is read. P1: at least one of RC7, RC14, and RC28 brings q95 coverage at
  the six into 0.93 to 0.97 and q99 into 0.98 to 0.995, and keeps q01 in 0.005 to 0.02 and q05 in
  0.03 to 0.07. P2: q99 pinball loss at the six is within 5% of the raw forecast's for the same
  arm. P3: coverage is more stable across the five folds than table P's, because the correction
  follows each month. P4: the 7-day buffer is the noisiest and the 28-day buffer the slowest to
  follow a change, so the 14-day buffer is the best compromise. P5: cold winter nights stay below
  target at q95, because the buffer averages over all conditions and the errors on cold nights are
  larger than on other nights.
decision_rule: >-
  Rolling conformal is a candidate (RC7, RC14, RC28, or RC14 followed by a constant widening of
  1.05) if, at the signed-MW six on clean rows pooled over the five scored folds, q95 lies in 0.93
  to 0.97, q99 in 0.98 to 0.995, q01 in 0.005 to 0.02, and q05 in 0.03 to 0.07; the mean q99
  pinball loss is within 5% of the raw forecast's; and CRPS is within 0.03 points of the best arm.
  Coverage is judged on the point estimate, and the Bonferroni interval is reported beside it. If
  none of the four arms passes, the tails at the six need a different forecaster, not a different
  correction. The cost of porting is a store of the last 28 days of delivered errors per Primary
  and lead band, updated at each origin, instead of a stored table, so a pass is worth porting only
  if it passes where table P (fitted on earlier folds only) fails; a pass that merely matches
  table P is a candidate for serving only if the maintainer prefers the stateful store.
benchmark: >-
  The raw A-pool forecast, the symmetric 1.15 widening, table P (leave-one-fold-out, and fitted on
  earlier folds only), and the two-constant rule of experiment 492, all on experiment 483's cache.
row_set_id: >-
  The `gam14` row set of experiment 470 (experiment 483's cache), clean rows; the eight and D15 and
  D18 are reported, not judged.
metric: >-
  Coverage and pinball loss at q01, q05, q95, and q99, CRPS in points of effective capacity,
  avoidable-curtailment cost at f = 0.9, 0.75, and 0.5, and the spread of coverage across folds.
---

**Plan (committed before any score is read).** Experiment 483 found that table P passes all four
tail coverages at the signed-MW six but misses two clauses: q99 pinball loss is +6.1% against the
widening (limit 5%), and q95 coverage is 0.927 when the table is fitted on earlier folds only (floor
0.93). This experiment asks whether a rolling conformal correction fixes those two clauses. The idea
comes from conformalized quantile regression ([Romano et al.
(2019)](https://doi.org/10.48550/arXiv.1905.03222)) and its online form that recomputes the
correction as new errors arrive ([Gibbs and Candès
(2021)](https://doi.org/10.48550/arXiv.2106.00170)). For each Primary, lead band, and quantile
level, the correction shifts the stored quantile by the empirical quantile of the residuals (actual
minus the forecast's quantile) of the last 7, 14, or 28 days of delivered forecasts.

## Design

- **Forecasts.** The five-seed quantile average of experiment 425's arm A (A-pool), which experiment
  483 scored. A-gam-in is dropped: its forecasts exist for the five scored folds only, so no buffer
  can reach before a fold, and it exists for the eight only. The 12 months of A-pool predictions in
  experiment 425's `seed_preds/A` (2025-07 to 2026-06, each month scored with a model fitted before
  it) supply the buffers; no in-sample forecast is used.
- **Buffer.** At origin t (daily at 09:00 UTC) and for one Primary and one lead band, the buffer
  holds every stored forecast row in that lead band whose target time is at most t minus 6 hours
  (an actual is delivered after 6 hours) and within the last 7, 14, or 28 days of target times. A
  fold's earliest origins use the previous month's forecasts, which come from the previous month's
  model; a serving system has the same boundary at each monthly refit. The buffer uses every
  delivered reading with a flagged exact zero removed, including rows within a load-shift event,
  because a serving system holds no event labels. A buffer cell with fewer than 100 residuals
  leaves the forecast unchanged and is counted.
- **Shift.** Each of the 11 stored quantiles is shifted by the empirical (linear-interpolation)
  quantile of the buffer's residuals at its own level. The scorer sorts the 11 quantiles per row, as
  in experiment 483, and the script reports how many rows needed sorting.
- **Arms.** `raw`; `w1.15` (symmetric widening of the four tail quantiles); `P` (table P,
  leave-one-fold-out, and `P-earlier`, fitted on earlier folds only); `two-constant` (experiment 492,
  leave-one-fold-out and earlier-only); `RC7`, `RC14`, `RC28`; and `RC14+w1.05` (RC14 followed by a
  symmetric 1.05 widening of the four tail quantiles). The rolling arms can score all five folds.
  Two row sets keep the arms on equal rows: the five folds (arms without `-earlier`) and the four
  folds 2025-12 to 2026-06 (adds `P-earlier` and the earlier-only two-constant rule).
- **Scoring.** At the signed-MW six (judged), the eight, and D15 and D18 (reported): q01, q05, q95,
  and q99 coverage and pinball loss, CRPS, by lead band (including 12 to 36 hours), on clean rows
  (rows with no labelled event), on cold winter nights, and on real-event rows against
  NRA-corrected actuals; avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the six; and
  coverage by fold.
- **Intervals.** Month-by-origin-day block bootstrap, 20,000 resamples. The Bonferroni count is 8
  (four candidate arms, RC7, RC14, RC28, and RC14+w1.05, each minus raw, on CRPS and q99 pinball
  loss, clean rows at the six); the interval level is 99.375%. Contrasts against table P, the
  widening, and the two-constant rule are descriptive, at 95%.
- **Minimum detectable effect** (A-pool five single seeds, 1.15 widening, clean rows; computed by
  `e00500_conformal.py mde` before any arm is scored): at the six, CRPS 0.087 points, q01 coverage
  0.0009, q05 coverage 0.0032, q95 coverage 0.0145, q99 coverage 0.0076, and q99 pinball loss 0.018
  points; at the eight, CRPS 0.044, q01 0.0008, q05 0.0040, q95 0.0096, q99 0.0028, and q99 pinball
  loss 0.0095. The coverage clauses use floors 0.005 to 0.02 (q01) and 0.03 to 0.07 (q05), wider than
  the minimum detectable effects, but a q95 or q99 coverage within 0.0145 or 0.0076 of a floor is
  within seed noise and is reported as such.
- **Standing practices.** Monthly refits in the stored forecasts (no fit here); cleaned data
  (flagged exact-zero half-hours dropped, the count per fold reported); the experiment 399 start cut
  for D15 and D18 (reported, never judged); July 2026 unused; every score selection-biased because
  the same five folds choose and score. Reused predictions: experiment 425's seed predictions and
  experiment 483's cache. Nothing is fitted, so no model is saved.
- **Serving cost** (stated on the page). Table P is a stored table of 20 factors per metering group,
  fitted at training. Rolling conformal replaces the table with a store of the last 7 to 28 days of
  delivered forecast errors per Primary and lead band, updated at every origin. Serving would hold
  that state, fail back to the 1.15 widening when actuals are missing, and add a data dependency on
  the delivered power readings the forecast already reads for its lags.
- **Advice followed.** Experiment 483 and 492 advice: the earlier-only fit is the planning figure;
  recalibrate the eight with an upper-tail widening or one factor per tail; keep table P at the six.
  This experiment tests the six only against that last advice.
