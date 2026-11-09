---
id: "00666"
slug: one-year-backtest-of-simple-wind-and-solar-networks
title: >-
  What does a one-year backtest of the simple shared wind network S and the simple shared solar
  network PLP forecast for the 3 metered wind farms and the 6 metered solar farms?
theme: generator-forecasting
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: wind-v1
date: 2026-11-09
size: >-
  Medium. (1) Stores private Parquet forecasts and fitted-network state files in the data folder
  only. (2) The refit cadence and the lead range were the open design choices; the Discuss session
  fixed them (below). (3) Spans code from experiments 659, 615, 430, and 259, which this folder
  copies. (4) Touches no serving path and no degradation rule. (5) Changes no contract.
hypothesis: >-
  Descriptive. Over daily forecast origins from 2025-11-01 to 2026-10-31, the networks of
  experiment 659 (wind S, solar PLP), retrained monthly (wind) or quarterly (solar) on data
  available before each refit, produce calibrated 11-quantile forecasts out to 14 days, and solar
  PLP trained on leads out to 336 h scores no worse at days 10 to 14 than climatology.
prediction: >-
  Written before any score. (1) Wind S over days 0 to 10 scores within 0.15 points of capacity of
  its experiment 659 score (3.684) on the overlapping origin months. (2) Solar PLP at days 10 to
  14 scores better than `climatology` but by less than at days 1 to 3. (3) The q05-q95 coverage of
  both stays between 0.85 and 0.95 in every lead band, except that very sunny days still exceed
  q99 on 10% or more of rows.
decision_rule: >-
  Descriptive; nothing is ranked or adopted. The file is a deliverable for NGED. No contrast is
  tested.
benchmark: >-
  `climatology` (the farm's own past power) for the 10 to 14 day solar band; the scores of
  experiment 659 for the overlapping months.
row_set_id: year_backtest_2025-11_to_2026-10_v1
metric: >-
  CRPS (11-level quantile score) and normalised MAE of the median in points of capacity, by lead
  band and by month; q05-q95 coverage; tail shares above q99 and below q01.
production_status: not-for-production
---
**Planned (not yet run).** This experiment writes two private Parquet files, `wind.parquet` and
`solar.parquet`, in `data/ml-experiments/backtest_year_2025-11_to_2026-10/`. They hold a one-year
backtest of the simple pooled wind network S and the simple pooled solar network PLP from
[experiment 659](../00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md).
The backtest is descriptive and not a ranking: the year overlaps the development months, and the
origins from 2026-07-01 to 2026-10-31 lie in the window that earlier experiments kept as a final
test, so those months are no longer a held-out test for choosing between S, PLP, and the defaults.

## Plan

- **Origins and weather.** One forecast origin per day at 12:00 UTC from 2025-11-01 to 2026-10-31.
  The weather is the 00 UTC ECMWF ENS run of the same day (51 members), treated as available at
  09:00 UTC. Valid times run half-hourly from 12:00 on the origin day to 14 days later (672
  half-hours), so the lead from the 00 UTC run is 12 to 348 hours.
- **Power lags.** Neither network reads a power lag: wind S reads the mean and spread of the raw
  weather only, and solar PLP reads weather, sun position, and the physics-lite power. The 6-hour
  power delay therefore changes no feature. It sets only the training cut-off of each refit.
- **Refits (fixed by the Discuss session).** Wind S refits on day 0 of each month (12 refits, 3
  seeds each). Solar PLP refits on 2025-11-01, 2026-02-01, 2026-05-01, and 2026-08-01 (4 refits, each scoring the next three months), 3 seeds each. No ERA5 or
  CAMS pre-training. Training uses only runs and valid times before the refit date, and the
  62-day stopping window of experiment 659.
- **Solar leads to 14 days.** The solar network trains on leads out to 336 h (experiments 430 and
  659 stopped at 251.5 h), because the lead is an input column. The page checks how daylight
  half-hours are interpolated after the ensemble steps coarsen from 3 hours to 6 hours at day 6,
  checks that the archive holds 14 days for every run, and reports days 10 to 14 as a separate
  lead band beside `climatology`, with coverage.
- **Files and checks.** The output columns, row counts, and checks are those of the shared
  contract in `data/ml-experiments/briefs/year_backtest_contract_brief.md`. Series carry their
  original `time_series_id`.
- **Reviews.** One Sonnet plan review, one Sonnet code review before the first fit, a results
  review, and one Sonnet prose review.

## Files

To be written when the experiment finishes.
