---
id: "00669"
slug: one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points
title: >-
  What does a one-year backtest of the lag-enabled demand network score at the 2 bulk supply points
  and the 2 grid supply points, with every power input at least 6 hours old and forecasts out to 14
  days?
theme: production-reference-and-best-guess
also_relevant_to:
  - hierarchy-and-pooling
status: planned
kind: other
protocol: demand-v2
size: >-
  Medium. (1) It stores 12 monthly refits of a neural network at 2 groups and 2 seeds, and 2
  private Parquet files; (2) one design is defensible (the recipe is fixed by experiments 622 and
  633, and the maintainer fixed the lag cut and the cadence); (3) it spans code from experiments
  87, 397, 418, 422, 428, 437, 510, 511, 559, 560, 565, 569, 572, 574, 591, 611, 622, and 633,
  whose callers cannot be named without searching; (4) it touches no serving path and no
  degradation rule; (5) it changes no contract. One Sonnet plan review, one Sonnet code review
  before the first fit, one Sonnet prose review.
hypothesis: >-
  Descriptive, not a hypothesis test. The lag-enabled network of experiment 633 (experiment 622's
  arm `lagged_b1`) forecasts the 2 bulk supply points (BSPs: D13 Skegness and D17 Boston) and the
  2 grid supply points (GSPs: D6 Walpole and D1 Bicker Fen) every day for a year from 2025-11-01,
  at leads of 0 to 14 days, with no neighbour inputs and no model of load-shift events.
prediction: >-
  Written before any fit. P1: with a lag cut at 3.5 hours before the network's 09:00 origin (6.5
  hours before the 12:00 origin), the CRPS at days 0 to 3 is no worse than the 24-hour-lag network's
  stored 7.45 points of capacity (experiment 633, July to October 2026) by more than 0.5 points,
  because a fresher lag should help. P2: the lag adds nothing beyond day 6, so CRPS at days 7 to 14
  matches the CRPS at days 4 to 10 of experiment 633 within 1.0 point. P3: q05 to q95 coverage is
  between 0.80 and 0.95 in every month. Each can fail.
decision_rule: >-
  Descriptive. No ranking, no claim of benefit, and nothing adopted for production. The year
  overlaps the development months and the final-test months, so no number here is a held-out
  score. A pre-flight gate on the 2026-07 fold decides whether the full year runs: the gate's
  checks (no feature reads power newer than 06:00 UTC of the origin day, the row count of the
  fold, no crossing quantiles) must pass.
benchmark: >-
  Experiment 633's stored lag-enabled network (24-hour lags, leads to 10 days, one fit at
  2026-07-01), as context for July to October 2026 only.
row_set_id: backtest_year_2025_11_to_2026_10_daily_12utc
metric: >-
  CRPS in points of capacity and the normalised MAE of the median, by lead band (days 0 to 1, 1 to
  3, 3 to 7, 7 to 14) and by month, with q01, q05, q95, and q99 coverage.
---
A bulk supply point (BSP) is a substation that takes power from the transmission network, and a
grid supply point (GSP) is a substation where the distribution network meets the transmission
network. This page is planned and holds the plan only.

## Method (plan)

- **Network:** experiment 633's lag-enabled network, which is experiment 622's arm `lagged_b1`:
  the pooled, group-of-2 mixture density network of experiment 619 with the D5 output offset of
  experiment 565, the thermal-lag and solar-gain columns of experiment 569, the shared lag encoder
  of experiment 591, frozen encoders in fine-tuning, and the three-step schedule with a 3-day
  embargo of experiment 574 (arm P5). Pre-training on ERA5 and CAMS, then the ENS mixer, then the
  fine-tune, as in those experiments. No neighbour inputs, no gate, and no load-shift model.
- **Lags:** the stored network reads power at least 24 hours before its 09:00 origin (the
  `24h` cap of experiments 560 and 572), which already satisfies the contract's cut. The
  maintainer asked for lags of at least 6 hours, so this plan retrains with experiment 572's
  `3.5h` cap (7 half-hours before the 09:00 origin, so the newest power is the 05:30 reading, 6.5
  hours before the 12:00 origin). The lag encoder is pre-trained and fine-tuned with that cap.
- **Origin:** the network runs at its own origin, 09:00 UTC on the day (the 00Z ENS run plus 9 hours),
  using the same 00Z ENS run the contract names. Its output is relabelled `power_fcst_init_time` =
  12:00 UTC, and only valid times from 12:00 to 11:30 of day 14 are kept. Every input the network
  read at 09:00 was available at 12:00, so the relabelling does not leak.
- **Leads:** the training rows already hold leads of 0.5 to 351 hours, so no retraining for the
  longer horizon is needed. Only the scoring window of experiment 428 stops at 240 hours, and this
  experiment lifts that limit to 351 hours. Days 11 to 14 were never scored before, so they are
  reported separately.
- **Folds:** 12 monthly refits, test months 2025-11 to 2026-10. Each trains on data before the
  first of the month and forecasts every origin in the month. A lag may read power observed inside
  the month, but only from 3.5 hours before the 09:00 origin or earlier. The training phase hides
  power from the cutoff on, as experiment 633's training phase does.
- **Seeds:** 2 (experiment 622's choice); the quantiles are averaged across seeds, then sorted
  within the row.
- **Units:** the network forecasts shares of effective capacity; the Parquet files hold MW (all 4
  series are MW series), obtained by multiplying by `effective_capacity_mw`.
- **Standing practices followed:** cleaned data (flagged half-hours dropped, counted per fold),
  monthly refits, saved weights, tail and example-forecast plots. The D15 and D18 cut does not
  apply (no Primaries here).

## Files

To be written: `bsps.parquet` and `gsps.parquet` in
`data/ml-experiments/backtest_year_2025-11_to_2026-10/` (private).
