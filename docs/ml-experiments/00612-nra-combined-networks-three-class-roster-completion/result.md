---
id: "00612"
slug: nra-combined-networks-three-class-roster-completion
title: >-
  Which network should each of the three load-shift classes use, once the tail constants, the
  steadier offset, a neighbour-summary class 2 candidate, and the missing measures are added to
  experiment 608's table?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. Answers to the five triggers: (1) it stores fitted network weights and forecasts for
  about 8 new arms in the private data folder only; (2) more than one design could test the
  question (the class 2 candidate takes experiment 610's closest arm, P7 with 50% signal dropout, but
  other arms are possible); (3) it spans code whose callers cannot be named without searching
  (experiments 565, 574, 577, 584, 591, 592, 595, 597, 599, 601, 606, 608, and 610); (4) it touches
  no serving path and no degradation rule; (5) it changes no contract. Ceiling 10 GPU-hours (process
  hours); ask before crossing.
hypothesis: >-
  Once the two-constant tail widening is fitted on each network's own forecasts, the steadier offset
  O2 is equivalent to D5 in K1, K2, and K3, and a class 2 network that reads neighbour summaries
  with signal dropout (K2 plus experiment 610's P7 with D50) matches K1's day-4-to-end coverage at
  the gate's band width without the gate.
prediction: >-
  Written before any score. P1: with tail constants fitted on each arm's own forecasts, O2 is within
  0.01 of D5 on event-free q05-q95 coverage at both metering groups (experiment 608 saw O2 lower
  coverage by 0.027 without tail constants). P2: O2 stays at least 0.01 below D5 on day-4-to-end
  coverage in K1 (experiment 608 saw 0.041). P3: the class 2 candidate covers within 0.015 of K1
  day 4 to the end at the gate's band width, and is worse than K1 on event-free CRPS by less than the
  minimum detectable effect. P4: dropping E3 raises absorption by 0.03 or more. P5: dropping the
  offset lowers event-free coverage by 0.02 or more at the six MW Primaries.
decision_rule: >-
  O2 is an equivalent swap for D5 if, with tail constants refitted, every Bonferroni interval of
  its difference from D5 lies inside the tolerance (coverage 0.01, absorption 0.03, event-free CRPS
  the printed minimum detectable effect); otherwise its cost is reported. The class 2 candidate can
  replace the G4 gate if its day-4-to-end and after-end coverage at the gate's band width are within
  0.01 of K1's and its event-free CRPS is within the printed minimum detectable effect of K2's.
  Curtailment cost at limits 0.9, 0.75, and 0.5 is reported only. The Bonferroni count is stated as
  a number. The experiment adopts nothing.
benchmark: >-
  K1, K2, and K3 of experiment 608 (D5 offset), and the stored A1 + G4 of experiment 597.
row_set_id: experiment_470_intersection_v1
metric: >-
  Event-free q05-q95 coverage and CRPS at 24-120 h and 120-240 h; coverage of the normal running
  arrangement by phase on the injected bank; absorption; follow; curtailment cost at limits 0.9,
  0.75, and 0.5.
---

## Plan

Scope, order, and compute are those of the brief: tail constants and O2 first; the class 2
candidate second; the no-E3 and no-offset ablations on the first fold pair third; the missing
measures on all five development months fourth; plots and the roster table last. July 2026 onwards
is never opened. Seeds 0 and 1; refits on days 0 and 14. The fits of experiment 608 are reused
(read only); new fits are listed in the write-up.
