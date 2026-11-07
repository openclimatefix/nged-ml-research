---
id: "00620"
slug: bakeoff-final-test-demand-networks-at-primaries
title: >-
  How do the three frozen candidate demand networks score on the held-out months July to October
  2026 at the 16 Primaries?
theme: production-reference-and-best-guess
also_relevant_to:
  - lagged-power-and-error-correction
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. Answers to the five triggers: (1) it stores new network weights, forecasts, and a
  lock file; (2) the freeze admits more than one defensible design (how the candidate's band width
  and the gate are set without the test months); (3) it spans code from experiments 510 to 614
  whose callers cannot be named without searching; (4) it touches no serving path and no
  degradation rule; (5) it changes no contract.
hypothesis: >-
  Each of the three frozen candidate demand networks (K1 with the gate, the ungated class 2
  candidate, and the lag-free K3), trained only on data before 2026-07-01, is better than the
  climatology and manual-heuristic comparators on the held-out forecast origins 2026-07-01 to
  2026-10-31, and the ranking of the three candidates on those months agrees with their ranking on
  the five development months.
prediction: >-
  Written before any fit or any look at July to October 2026 power. P1: all three networks have a
  lower mean continuous ranked probability score (CRPS) than climatology and the manual heuristic
  at 0-240 h. P2: event-free q05-q95 coverage of each network lies between 0.85 and 0.95. P3: the
  ranking by CRPS of K1, the class 2 candidate, and K3 on the held-out months equals their ranking
  on the development months in experiment 613.
decision_rule: >-
  Descriptive, scored once. No tuning and no second look. A candidate is reported as beating a
  comparator when the day-block bootstrap interval of the CRPS difference excludes zero. The
  Bonferroni count is printed beside every interval. Nothing is adopted for production.
benchmark: >-
  BG2, the manual heuristic, climatology, and the weak production-style XGBoost model if its
  artefact exists.
row_set_id: final_test_origins_2026_07_01_to_2026_10_31
metric: >-
  CRPS, MAE, threshold-weighted CRPS, curtailment cost, q05-q95 coverage, follow ratio and
  absorption on verified events in the window, event-free CRPS.
---
**Plan only.** This page is filled in after the held-out months are scored. The roster, the
training cutoff (2026-07-01), and the single scoring pass are fixed by the maintainer's brief of
2026-11-07. BSP and GSP networks are not scored here.
