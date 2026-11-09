---
id: "00667"
slug: one-year-backtest-of-the-two-output-demand-network-at-primaries
title: >-
  What does a one-year backtest of the class 2 Candidate demand network with a survival-hazard
  prevailing output score at the 16 Primaries, with every input at least 6 hours old?
theme: production-reference-and-best-guess
also_relevant_to:
  - lagged-power-and-error-correction
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores 12 monthly refits of a neural network and a survival head and one private
  Parquet file; (2) several designs are defensible (which base network, with or without the gate
  input, widening or not), settled in part by the designer session; (3) it spans code from
  experiments 574, 591, 610, 612, 613, 620, 647, 650, and 652, whose callers cannot be named without
  searching; (4) it touches no serving path and no degradation rule; (5) it changes no contract.
hypothesis: >-
  Descriptive, not a hypothesis test. The one-year backtest is the first real-rows test of the
  Candidate plus the R-haz head. Prediction that could fail: the Candidate's NRA output scores
  within 0.10 points of capacity of its development-month score once its power lags are cut from
  24 hours to 6 hours.
prediction: >-
  Written before any fit. P1: Candidate NRA CRPS with 6-hour lags is not worse than the 24-hour-lag
  Candidate on the same fold by more than 0.10 points of capacity (a fresher lag should help). P2:
  the survival output stays calibrated at days 10 to 14 (predicted survival within 0.10 of the
  observed share still on). P3: q05 to q95 coverage of the NRA output is within 0.05 of nominal
  in every month.
decision_rule: >-
  Descriptive; no ranking and no claim of benefit. A pre-flight gate on one fold decides whether
  the full year runs: P1 must hold on that fold, or the fold's result goes to the manager before
  the year starts.
benchmark: >-
  Experiment 620's stored Candidate (24-hour lags, 10 days) and the manual heuristic, as context
  only.
row_set_id: backtest_year_2025_11_to_2026_10_daily_12utc
metric: >-
  CRPS in points of capacity and the normalised MAE of the median, by lead band (days 0 to 1, 1 to
  3, 3 to 7, 7 to 14) and by month; q01, q05, q95, and q99 coverage.
---
A Primary is a primary substation. The Candidate is the class 2 neural network of experiments 612
and 613 (arm P7 with D50: neighbour power summaries, 50% signal dropout, no explicit gate). This
page is planned and holds the plan only.

## Method (plan)

- **Base network:** the Candidate (arm `P7+D50`, experiments 610, 612, 613, 620), as the designer
  session advised. Pre-training chain: lag, weather, and date-time encoders pre-trained on ERA5 and
  CAMS (experiments 559, 565, 574), a fresh mixer on the whole ENS history (starts 2024-04-01),
  then the daily fine-tune with a 90-day recency half-life.
- **Lags of at least 6 hours, against two origins:** the stored networks use an origin of 09:00 UTC
  (the 00Z ENS run plus 9 hours). This backtest uses the contract's origin of 12:00 UTC, so the
  network's origin time moves to 12:00. Experiment 574's arm V35 caps lag age at 3.5 hours
  (`CAP_3_5H` = 7 half-hours); this plan adds a cap of 12 half-hours (6 hours), measured from the
  12:00 origin. The newest power any lag reads is therefore at 06:00 UTC on the origin day, which is
  exactly 6 hours before 12:00. A 12-half-hour cap measured from the native 09:00 origin would read
  power as new as 03:00 only by accident of arithmetic, and would leave unused power between 03:00
  and 06:00; the plan does not use that reading. The lag encoder is pre-trained with the 6-hour cap.
  The neighbour summaries obey the same cut.
- **Survival head:** R-haz of experiment 652 sits on the Candidate (`P7+D50`), not on K1. Its
  gate-state input is removed, so no explicit gate enters. The Candidate's frozen 11 quantiles
  are the NRA output. The prevailing output is the 11 quantiles of the mixture
  `(1 - p) NRA + p (NRA shifted by a learned depth d and spread by a learned sigma)`
  (experiment 647's `mixture_quantiles`: the shifted part is 3 Gauss-Hermite components at
  `d + g sigma`, the quantiles are read from the mixture's cumulative distribution at the 11
  levels). The head has 14 daily hazards instead
  of 10, so `p = exp(-cumulative hazard)` keeps falling to day 14. One ablation arm keeps the gate
  input, on the development months only.
- **Folds:** 12 monthly refits, test months 2025-11 to 2026-10, trained on data before the
  month with the 7-day embargo of experiment 613. Origins daily at 12:00 UTC; the weather is the
  00Z ENS run of the same day.
- **Seeds:** 2 (experiment 612's choice), quantiles averaged across seeds.
- **Full-row test matrices:** the stored test matrices drop every half-hour with no observed
  reading, which leaves gaps after 2026-10-17. The inputs here are built for every valid time from
  12:00 to 14 days later for every origin, and only the scoring is restricted to rows with observed
  power.
- **Compute:** ceiling 10 GPU-hours (raised by the manager from 6), up to 3 GPU processes at once with at most 4 CPU cores each, 2 seeds and 14 daily hazards kept.
- **Standing practices followed:** cleaned data, the D15 and D18 cut before 2025-03-01, monthly
  refits, saved weights, and tail and example-forecast plots.
- **Open items before the first fit:** a one-fold timing run (the stored Candidate fit takes about
  65 s per seed on a final-test fold, but the cost of the daily fine-tune, the injected bank for the
  head, and a 6-hour pre-train is not yet known); whether the Candidate's unscaled band is scaled
  to K1's width (default: not scaled, with the width ratio stated).

## Files

To be written: `primaries.parquet` in
`data/ml-experiments/backtest_year_2025-11_to_2026-10/` (private).
