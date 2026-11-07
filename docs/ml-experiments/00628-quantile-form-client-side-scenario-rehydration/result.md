---
id: "00628"
slug: quantile-form-client-side-scenario-rehydration
title: >-
  Can NGED rebuild coherent scenarios from stored quantiles alone, and which tail rule should the
  client use?
theme: calibration-and-uncertainty
also_relevant_to:
  - generator-forecasting
  - hierarchy-and-pooling
status: planned
kind: other
protocol: demand-v2
date: 2026-11-07
size: >-
  medium. Stored: tables in this experiment's private folder only; no forecast is fitted. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one defensible (the tail
  rule, and how the client couples 1,000 draws), so the plan registers each. Callers: this folder's
  scripts only, which import experiments 434, 436, and 625 and `studies.*`. One Sonnet code review
  before the first run, one Sonnet results review, and one Sonnet write-up review.
hypothesis: >-
  NGED reads stored forecasts from S3 and builds scenarios in Python. The stored forecasts of
  experiments 425, 429, and 430 hold 11 quantiles (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75,
  0.90, 0.95, 0.99) per site and half-hour. The claim is that a client holding only these 11
  quantiles, plus a small dependence table, draws 1,000 coherent scenarios per run that score within
  1% of the 51 scenarios the pipeline builds (experiment 436's J5 and J3) on CRPS, and within 1
  percentage point on exceedance at the 0.01, 0.05, 0.95, and 0.99 levels, provided the client
  extends the quantile function past the outer levels with a good tail rule.
prediction: >-
  Written before any score is read. P1. Between the 0.01 and 0.99 levels all tail rules agree, so
  CRPS differs between rules by under 0.5% for demand and by up to 2% for solar and wind, where the
  tail beyond 0.01 holds the mass at zero and at capacity. P2. Exceedance at the four outer levels is
  within 1 percentage point of nominal for every rule, because those levels are knots of the stored
  quantile function; the 51 pipeline scenarios cannot resolve the 0.01 and 0.99 levels at all, so
  they sit further from nominal than the client draws. P3. A Gaussian tail gives the most stable
  lowest-wind and 5th-percentile-solar picks, clamping the least. P4. The pure-copula client
  (1,000 draws, no ensemble-rank component) scores within 3% of J5 on the spatial variogram score
  for demand and wind, and up to 10% worse for solar, whose J5 mixing weight was 0.25 to 0.5.
decision_rule: >-
  Fixed before any score is read. Rows: experiment 436's 44 Tuesday runs of 2025-09 to 2026-06,
  days 1 to 10 (the July to October 2026 window is not scored), the 16 Primaries, 3 wind farms, and 6
  solar farms. Intervals: month-block bootstrap over the 10 calendar months, 20,000 resamples,
  98.33% (Bonferroni over three registered contrasts). `Q11 sufficient` if the best tail rule is
  within 1% of the in-pipeline arm N on CRPS and within 1 percentage point of N on exceedance at
  the four outer levels, in every lead band (0 to 12 h, 12 to 36 h, 36 to 120 h, 120 to 240 h),
  for every family. N is experiment 436's J0 for CRPS (the marginal does not depend on the coupling)
  and the stored quantile knots for exceedance. `Quantiles not sufficient` otherwise; and if a
  client arm scores worse than the in-pipeline scenarios or independent draws by more than 2% on
  the variogram score or CRPS, the session stops and tells the manager before rebuilding anything.
  The mixture arm M is not run: the Primaries' native output is an XGBoost quantile fit, and the
  wind and solar mixture parameters were never stored. A 13-level run would need the quantile
  forecasts rewritten at the 13 NGED delivery levels, and is not done. No rescue tuning after a
  failed bar.
benchmark: >-
  Experiment 436's stored 51-path arms J0 (independent), J3 (Schaake shuffle), and J5 (fitted
  copula) for scenarios; the stored quantile knots for exceedance.
row_set_id: joint_tuesdays_v1
metric: >-
  CRPS, exceedance rate at the 0.01, 0.05, 0.95, and 0.99 levels, energy and variogram scores at 51
  paths, and the movement of two scenario picks, per lead band.
code: ml-experiments/00628-quantile-form-client-side-scenario-rehydration/
---
Planned. The write-up follows the run.
