---
id: "00613"
slug: nra-roster-contrasts-with-intervals-five-seeds
title: >-
  Do five seeds give intervals tight enough to show that the class 2 candidate can replace the
  gate, that O2 is equivalent to D5, and that dropping E3 changes absorption?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. Answers to the five triggers: (1) it stores fitted network weights and forecasts for
  about six arms in the private data folder only; (2) more than one design could test the question
  (seed averaging against per-seed contrasts), so this plan fixes the seed mean and checks it
  against two disjoint seed sets; (3) it spans code whose callers cannot be named without searching
  (experiments 565, 574, 577, 584, 591, 592, 595, 597, 599, 601, 608, 610, and 612); (4) it touches
  no serving path and no degradation rule; (5) it changes no contract. Ceiling 10 GPU-hours
  (process hours); ask before crossing.
hypothesis: >-
  Experiment 612 left three claims resting on point estimates inside the tolerance. With seeds 0 to
  4 on the five development months, the interval of each contrast lies inside the tolerance (0.01
  coverage, 0.03 absorption, the printed minimum detectable effect in CRPS) for (1) the class 2
  candidate against K1, (2) O2 against D5 in K1, K2, and K3, and (3) E3 against the 3-day embargo
  on coverage.
prediction: >-
  Written before any score. P1: the whole interval of the candidate minus K1 lies inside 0.01 for
  day-4-to-end and after-end coverage at K1's band width. P2: O2 minus D5 lies inside 0.01 for
  event-free coverage in K1 and K2 but not in K3 at 120-240 h (experiment 612 saw -0.017 there). P3:
  E3 lowers absorption by at least 0.03 against the 3-day embargo (experiment 599 saw -0.045 to
  -0.062). P4: the cell-level bootstrap intervals shrink by less than a third against experiment
  612's, because the unit of resampling is still the 16 Primaries and 5 months.
decision_rule: >-
  A contrast is "equivalent" only if its whole Bonferroni interval lies inside the tolerance
  (coverage 0.01, follow 0.01, absorption 0.03, event-free CRPS 0.0006, the minimum detectable
  effect of experiment 591). The class 2 candidate can replace the G4 gate if its day-4-to-end and
  after-end coverage contrasts with K1 are equivalent and its event-free CRPS contrast with K2 is
  equivalent. O2 is an equivalent swap for D5 if every counted coverage and CRPS contrast is
  equivalent. Anything else is reported as inconclusive, better, or worse, with the side the
  interval falls on. The Bonferroni count is stated as a number beside every family. The experiment
  adopts nothing.
benchmark: >-
  K1, K2, and K3 of experiment 612 (D5 offset), and the candidate of experiment 612, all rescored
  on seeds 0 to 4.
row_set_id: experiment_470_intersection_v1
metric: >-
  Event-free q05-q95 coverage and CRPS at 24-120 h and 120-240 h; coverage, follow, and safe-side
  share of the normal running arrangement by phase on the injected bank; absorption; coverage 0-7
  days after the end.
verdict: inconclusive
best_arm: none yet
headline: Planned; no score yet.
lessons: Planned; no score yet.
take_home_message: Planned; no score yet.
advice:
  - >-
    Planned; no score yet.
production_status: not-for-production
adopted_in: none
---

**Experiment 612 left three claims resting on point estimates inside the tolerance. This experiment
refits the arms behind them on three more seeds (2 to 4) and replaces the point estimates with
intervals.** The claims are that the class 2 candidate can replace the gate, that the steadier
offset O2 is equivalent to the slow offset D5, and that dropping the E3 refit rule (a 7-day embargo
and half weight on the next 7 days) changes absorption.

## Plan

- **Seeds.** Seeds 0 and 1 are experiment 612's stored fits. Seeds 2 to 4 are new fits of K1 (D5 and
  O2), the lag-free K3 (D5 and O2), the class 2 candidate, and K1 with the 3-day embargo, each on
  the five development months. The forecast of a system is the mean of its five seeds' quantiles,
  as experiment 612's was the mean of two.
- **Checks on the seed mean.** Each headline contrast is repeated on seeds {0, 1} alone and on
  seeds {2, 3, 4} alone, which are disjoint sets of networks, to show whether the sign and size
  are stable.
- **Chains.** Pre-training and step-2 weights for seeds 2 to 4 are fitted afresh into a chain
  folder of this experiment, because experiment 591's stored chain holds seeds 0 and 1 only.
  A reproduction check fits one cell on seed 0 in the new chain and compares it with the stored
  fit.
- **Scoring.** Experiment 612's scoring, with the seed set raised to 0 to 4, tail constants fitted
  on earlier months only, and the candidate at K1's band width.
- **Absorption.** E3 against the 3-day embargo uses experiment 599's stored per-design shares
  (experiment 595's method, 12 injected designs, seeds 0 and 1).
- **Standing practices followed.** Cleaned data, July 2026 onwards unused, refits on days 0 and 14,
  every fitted network saved, the December 2025 D5/D16 example event drawn with one arm per panel.
