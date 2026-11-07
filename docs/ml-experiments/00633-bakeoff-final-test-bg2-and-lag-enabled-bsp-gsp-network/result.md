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
**Plan only.** This page is filled in after the held-out months are scored. The maintainer's brief
of 2026-11-07 fixes the training cutoff (2026-07-01), the 123 forecast origins (2026-07-01 to
2026-10-31), the 16 Primaries, and the single scoring pass per candidate set.
