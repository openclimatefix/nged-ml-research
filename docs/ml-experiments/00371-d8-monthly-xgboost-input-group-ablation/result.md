---
id: '00371'
slug: d8-monthly-xgboost-input-group-ablation
title: >-
  Which input group gives monthly XGBoost its lead over the GAM stack at substation D8?
theme: production-reference-and-best-guess
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: other
size: >-
  simple. Stored: private files under data/ml-experiments/00371-d8-monthly-xgboost-input-group-ablation/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one per arm (each arm
  changes one input group of the stored monthly XGBoost recipe, named below). Callers: the scripts
  import experiment 87's folder (fit, common, regimes, stack code) and read experiment 360's stored
  scoring frames.
hypothesis: >-
  One input group of monthly XGBoost (the weather inputs the GAM lacks, the three origin lags, the
  0.5-year recency weighting, or the 51-member mixture) accounts for most of its 0.64-point lead
  over the GAM stack at substation D8.
prediction: >-
  Written before any score is read. At D8, removing at least one input group raises monthly
  XGBoost's quantile score by 0.3 points or more (the gap is 8.338 for the GAM stack minus 7.694
  for monthly XGBoost, 0.644). Best guess for which group: the three origin lags (+0.3 to +0.5).
  At the placebos D2 and D4 no group moves the score by more than 0.1 points.
decision_rule: >-
  Each arm minus the control, paired on the same rows, whole valid months resampled (20,000
  resamples, seed 0), 95% intervals. A group carries the gap when its removal raises the score by
  at least 0.3 points and the interval lies wholly above zero. Stop rule: if no group reaches 0.2
  points, report that GAM input work is not the route. Gate: the refitted control reproduces the
  stored monthly XGBoost score of 7.694 at D8 within 0.05 points, on the 96,986 rows of experiment
  354, before any other arm is read.
benchmark: >-
  Monthly XGBoost as stored (7.694 at D8, experiment 360), the GAM stack (8.338 weekly, experiment
  355), BG2 (7.593)
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity;
  MAE of the median in MW; Pearson correlation of the median with actual demand; bias in MW; by
  lead band
code: ml-experiments/00371-d8-monthly-xgboost-input-group-ablation/
---

**Plan only; results follow.** Experiment 360 found that monthly XGBoost alone scores 7.694 at
substation D8 against 8.338 for the GAM stack, and experiment 364 found that recent power explains
only part of the analogue member's lead. This experiment refits monthly XGBoost with one input group
changed at a time, to find which group carries the gap.

## Plan

- **Control and arms.** Monthly XGBoost as experiment 87 fitted it (13 weather and calendar columns,
  lead hours, three origin lags, 500 rounds, eta 0.05, recency half-life 0.5 years, control-member
  training rows, 51-member mixture, spread factor 1.15), refitted at the 12 monthly refits. Arms:
    - **no extra weather:** drop longwave radiation, dew point, surface pressure, precipitation, and
      10 m wind speed, the five weather inputs the GAM stack does not use (the GAM uses temperature
      and irradiance only).
    - **no origin lags:** drop the three lags measured from the forecast origin.
    - **equal weights:** every training row weight 1 in place of the 0.5-year half-life.
    - **ensemble mean:** predict from the ensemble-mean inputs, one forecast per valid time in place
      of the 51-member mixture.
    - **GAM inputs only:** drop the five extra weather inputs and the three origin lags together.
- **Rows and sites.** The 96,986 rows of experiment 354 at D8; D2 and D4 as placebos on all stored
  rows to 168 hours.
- **Scoring.** Quantile score, MAE of the median, correlation, and bias, overall and by lead band,
  with month-resampled 95% intervals; the same score for each arm minus the control.
- **Compute.** GPU, 4 worker processes, 3 threads each, `MemoryMax=16G` per job.
