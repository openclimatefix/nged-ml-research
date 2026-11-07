---
id: "00633"
slug: bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network
title: >-
  How does BG2 score on the held-out months July to October 2026 beside the three demand networks
  at the 16 Primaries, and how does the lag-enabled network score at the bulk and grid supply
  points?
theme: production-reference-and-best-guess
also_relevant_to:
  - hierarchy-and-pooling
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. Answers to the five triggers: (1) it stores new XGBoost boosters, network weights,
  forecasts, and a lock file; (2) the BG2 recipe admits more than one defensible reduction (see the
  hypothesis) and the freeze fixes one; (3) it spans code from experiments 48, 87, 574, 591, 619,
  620, and 622 whose callers cannot be named without searching; (4) it touches no serving path and
  no degradation rule; (5) it changes no contract.
hypothesis: >-
  Part 1. BG2 rebuilt as one fit at 2026-07-01 ("BG2-single-fit": experiment 87's weekly XGBoost
  member, fitted once on data before 2026-07-01, a mixture over the 51 ensemble members with spread
  factor 1.15) scores within 0.10 CRPS points of K1, the ungated class 2 candidate, and K3 of
  experiment 620 at the 16 Primaries, and the ranking of BG2 against the three networks on the
  held-out months equals its ranking on the five development months. Part 2. The lag-enabled
  network of experiment 622 (arm lagged_b1) beats BG2 and the climatology at days 1 to 3 at the
  bulk and grid supply points that have readings in the window.
prediction: >-
  Written before any fit, and before any look at July to October 2026 power. P1: BG2-single-fit
  scores worse than K1 on CRPS at days 1 to 10, as it does on the development months (experiment
  613 stored BG2 forecasts for comparison). P2: BG2-single-fit scores better than the manual heuristic
  and the climatology. P3: at the bulk supply points the lag-enabled network beats BG2 at days 1 to 3
  by about 0.1 CRPS points (experiments 619 and 622); at the grid supply points the two tie. P4:
  event-free q05-q95 coverage of BG2 lies between 0.85 and 0.95.
decision_rule: >-
  Descriptive, scored once per candidate set. No tuning and no second look. A system is reported
  as beating another when the day-block bootstrap interval of the CRPS difference excludes zero,
  and the Bonferroni count is printed beside every interval. Nothing is adopted for production.
benchmark: >-
  K1, the class 2 candidate, K3, the manual heuristic, and the climatology of experiment 620
  (exports read only); at the bulk and grid supply points, the manual heuristic and the
  climatology.
row_set_id: final_test_origins_2026_07_01_to_2026_10_31
metric: >-
  The metrics of experiment 620: CRPS, MAE, threshold-weighted CRPS, curtailment cost, q05-q95
  coverage, and event-free CRPS, with day-block bootstrap intervals.
---
## Part 1: BG2 beside the three demand networks at the 16 Primaries

**BG2-single-fit scored lower CRPS than all three networks on the held-out months, but it saw
fresher power lags, so the margin is not a like-for-like win.**
BG2-single-fit is experiment 87's weekly XGBoost member, fitted once on data before 2026-07-01,
forecasting each of the 51 ensemble members and pooling them into 11 quantiles with the 1.15 spread
factor.
It is scored on the same 929,870 half-hourly rows, 16 Primaries, and 123 forecast origins
(2026-07-01 to 2026-10-31) as the three networks of experiment 620.
Every figure is in percent of capacity, scored once, with no row dropped.

**Two caveats limit the comparison.**

- BG2's power lags come from the newest reading at least 3.5 hours before the forecast origin. The
  networks use lags at least 24 hours old. BG2 therefore holds an information advantage, and a BG2
  win over the networks is not like-for-like on lags.
- The weekly-refit arm, "BG2-weekly", was pre-agreed as optional and was skipped. Refitting 20
  boosters on each of 17 weeks was estimated at 3.7 hours or more of fitting on a shared GPU, over
  the 3-hour cap. BG2-single-fit is the like-for-like comparator (one fit at the training cutoff, as
  for the networks); the production cadence is not shown.

BG2-single-fit omits the monthly, analogue, pooled, and stacked members and the solar offset of the
full five-member BG2. It also takes all power from the cleaned grid and drops D15 and D18 before
2025-03-01.

![MAE by forecast day: BG2 lowest at every day, climatology highest](mae_by_day.svg)

![CRPS by forecast day](crps_by_day.svg)

| System | CRPS | MAE | Threshold-weighted CRPS | q05-q95 coverage |
|---|---|---|---|---|
| BG2 | 4.23 (4.15 to 4.32) | 7.39 | 0.467 | 0.888 |
| Manual heuristic | 4.85 (4.74 to 4.96) | 8.49 | 0.467 | 0.787 |
| K1 | 4.94 (4.88 to 5.01) | 8.30 | 0.550 | 0.747 |
| Candidate | 5.00 (4.94 to 5.06) | 8.42 | 0.560 | 0.773 |
| K3 | 5.32 (5.25 to 5.39) | 8.85 | 0.553 | 0.729 |
| Climatology | 8.12 (7.96 to 8.29) | 14.75 | 0.596 | 0.875 |

Coverage is the share of observations inside the q05-q95 interval (nominal 0.90). BG2 over-covers
slightly with its 1.15 spread factor; the three networks under-cover (0.73 to 0.77).

| Contrast (CRPS) | Difference | Bonferroni 95% interval |
|---|---|---|
| K1 minus Candidate | -0.06 | -0.07 to -0.04 |
| K1 minus K3 | -0.38 | -0.42 to -0.34 |
| K1 minus BG2 | +0.71 | +0.58 to +0.84 |
| K1 minus Manual heuristic | +0.10 | -0.11 to +0.31 |
| K1 minus Climatology | -3.18 | -3.45 to -2.92 |
| Candidate minus BG2 | +0.77 | +0.63 to +0.90 |
| K3 minus BG2 | +1.09 | +0.93 to +1.23 |
| BG2 minus Manual heuristic | -0.61 | -0.72 to -0.51 |
| BG2 minus Climatology | -3.89 | -4.13 to -3.65 |

**BG2's lead is concentrated at a few Primaries; the two sets of forecasts trade places at the
others.** The table gives mean CRPS per Primary.

| Primary | K1 | Candidate | K3 | BG2 | Manual heuristic |
|---|---|---|---|---|---|
| D2 | 2.39 | 2.40 | 2.64 | 2.20 | 2.56 |
| D3 | 1.29 | 1.34 | 1.61 | 1.35 | 1.50 |
| D4 | 1.94 | 1.91 | 2.79 | 2.45 | 3.08 |
| D5 | 2.78 | 2.81 | 2.75 | 2.96 | 3.67 |
| D8 | 7.56 | 7.68 | 8.17 | 7.24 | 7.92 |
| D9 | 2.19 | 2.18 | 2.43 | 2.40 | 2.80 |
| D10 | 4.88 | 4.94 | 4.64 | 4.17 | 5.57 |
| D11 | 2.63 | 2.66 | 2.55 | 2.75 | 2.78 |
| D12 | 6.38 | 6.38 | 6.69 | 6.59 | 7.96 |
| D14 | 5.97 | 6.16 | 8.04 | 2.96 | 3.08 |
| D15 | 13.91 | 13.56 | 13.63 | 11.58 | 12.73 |
| D16 | 3.43 | 3.66 | 4.10 | 2.64 | 2.74 |
| D18 | 14.98 | 15.55 | 15.68 | 9.38 | 11.12 |
| D19 | 3.04 | 3.03 | 3.32 | 3.29 | 3.72 |
| D21 | 2.42 | 2.41 | 2.56 | 2.46 | 2.68 |
| D22 | 3.66 | 3.68 | 3.86 | 3.56 | 3.94 |

**Predictions.** P1 (BG2 worse than K1) is refuted: BG2 scores 0.71 CRPS points lower than K1
(Bonferroni interval above), though the lag advantage explains part of it. The hypothesis that BG2
sits within 0.10 CRPS points of the networks is also refuted. P2 holds: BG2 beats the manual
heuristic (by 0.61 CRPS points) and the climatology. P4 holds: BG2's q05-q95 coverage is 0.888,
inside 0.85 to 0.95. The ranking against the development months was not rechecked here.

**Threshold-weighted CRPS (the upper tail) shows no BG2 advantage over the manual heuristic**
(difference -0.0001, interval spanning zero).

## Part 2: the lag-enabled network at the bulk and grid supply points

Not yet run.
