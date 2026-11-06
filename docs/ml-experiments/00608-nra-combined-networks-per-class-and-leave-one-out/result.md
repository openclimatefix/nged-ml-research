---
id: "00608"
slug: nra-combined-networks-per-class-and-leave-one-out
title: >-
  Which parts of the combined lag-and-gate, lag-without-gate, and lag-free demand neural networks are
  needed, when each part is removed alone?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. Stored: refits, forecasts, and score tables in the private data folder only; no weights
  leave it. More than one design could test the question (the order of the removals, and whether the
  steadier offset is built inside the lag network or added after). The experiment spans the code of
  experiments 574, 577, 584, 587, 591, 592, 595, 597, and 599. It touches no serving path and no
  degradation rule (this repository holds none). Compute: about 12 process-hours expected, 14 as the
  ceiling for the D5 arms, and 20 in total for the steadier-offset (O2) chains.
hypothesis: >-
  Each part of the three combined networks (K1, lag and gate; K2, lag without an explicit gate; K3,
  lag-free) earns its place: removing any one lowers a measure it was added for by more than that
  measure's tolerance.
prediction: >-
  Written before any score. P1: removing the gated band (G4) lowers day-4-to-end coverage by 0.03 or
  more (experiment 598 saw 0.044). P2: removing the refit rule (E3) raises absorption by 0.03 or more
  (experiment 599 saw 0.045). P3: the D5 offset and the steadier offset (O2) are equivalent on every
  measure (experiment 577). P4: under G4, A1's gain over the C1 wiring is below 0.015, so A1 is
  inconclusive (experiment 597 saw 0.012). P5: removing the synthetic shifts raises follow above 0.15.
  P6: removing the expected embeddings raises follow by 0.01 or more. P7: removing the spread inputs
  stays within tolerance (experiment 496). P8: removing the tail constants lowers event-free q05-q95
  coverage at the six MW Primaries by 0.01 or more. P9: K1 with D5 reproduces A1 + G4 of experiment
  597 (day-4-to-end coverage 0.925) within 0.003.
decision_rule: >-
  A part is needed if any Bonferroni interval of a measure lies wholly on the worse side of that
  measure's tolerance; removable only if every interval lies inside the tolerance; otherwise
  inconclusive (kept, and flagged). Tolerances: coverage 0.01, follow 0.01, absorption 0.03, event-free
  CRPS the printed minimum detectable effect, event-free q05 and q95 coverage 0.01; curtailment cost is
  reported only. The Bonferroni count is stated as a number in the write-up. No arm is read until the
  K1 reference (K1-ref) reproduces experiment 597 within 0.003. The experiment adopts nothing.
benchmark: >-
  A1 + G4 of experiment 597 (day-4-to-end coverage 0.925), and K1 itself as the reference for every
  removal.
row_set_id: experiment_470_intersection_v1
metric: >-
  Coverage of the normal running arrangement inside q05-q95 by phase on the injected bank, follow,
  absorption, event-free CRPS and q05/q95 coverage at 24-120 h and 120-240 h, curtailment cost at
  limits 0.9, 0.75, and 0.5.
---

**Plan only; the write-up follows the scoring.** This page is the pre-registration. The design is
the manager's order review of 2026-11-06 (sections 2 and 3); the maintainer's decisions on it are:
three neural classes, the steadier offset O2 replaces D5 in K1, K2, and K3, and the aim is one combined
network per class by 08:00Z.

- **Rows and months.** Development months 2025-10, 2025-12, 2026-01, 2026-03, and 2026-06; seeds 0
  and 1; the first fold pair (2025-10, 2025-12) for every arm, the other three months only for K1,
  K2, K3, and inconclusive arms. Nothing reads a row from 2026-07-01 onwards.
- **Order of work.** Arms that use the stored D5 chains run first (K1-ref, tail constants, G4 off, C1
  wiring), then the O2 chain per fold, so K1 with D5 and K1 with O2 are reported side by side.
- **Gate.** O2 and E3 change the lag-free twin, so the G4 gate state and its false-trigger rate are
  recomputed for every arm that uses G4; experiment 592's stored gate state is never reused.
- **Stop rules.** Stop if K1-ref misses 0.003 of 0.925. If K1 falls below A1 + G4 by 0.01 on coverage
  or is worse beyond the minimum detectable effect on event-free CRPS, stop the removals and run
  add-one arms instead.
- **Out of scope.** Add-one arms from experiments 603 and 606 are added when the maintainer
  messages; July 2026 is never opened.
