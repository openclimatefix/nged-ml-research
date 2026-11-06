---
id: "00597"
slug: nra-assemble-c1-and-a1-with-g4-gate-as-single-forecasts
title: >-
  Do C1 and A1 with the G4 gated band, assembled as single forecasts, hold the normal running
  arrangement inside the q05-q95 band on 0.90 of rows from day 4 of a load-shift to its end?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Medium. Nothing is stored except tables, figures, and an export of forecasts in the private data
  folder; no fit and no new weights. The experiment wraps the stored forecasts of experiments 584 and
  591 and the gate of experiment 592. It touches no serving path and no degradation rule. It runs on
  CPU only.
hypothesis: >-
  The G4 band of experiment 592 and the C1 forecast of experiment 584 hold up when assembled into one
  forecast and scored by phase, and the A1 forecast of experiment 591 does at least as well.
prediction: >-
  Written before any score. P1: C1 + G4 reproduces experiment 592's day-4-to-end coverage of 0.913
  (stop and report if it differs by more than 0.002). P2: A1 + G4 covers at least 0.92 of rows in the
  0-7 days after the end of an injected load-shift. P3: the event-free pinball loss at q05 and q95
  of each assembled forecast is within the minimum detectable effect of its plain network. P4: the
  verified December pair is unchanged from experiment 592.
decision_rule: >-
  P1 to P4 hold or fail as stated. The counted tests are 6: the assembled forecast minus its plain
  network at q05 and q95 for each of C1 and A1 (4), and A1 + G4 minus C1 + G4 CRPS for the two
  metering groups (2), all at 24-120 h on event-free rows. Each contrast uses a five-block month
  bootstrap at the Bonferroni level, and the half-width is printed beside it as the minimum
  detectable effect.
benchmark: C1 and A1 without the gated band, and experiment 592's G4 result.
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the NRA inside q05-q95 and on the safe side of q05 or q95, by phase; median follow
  ratio; tail pinball; CRPS on event-free rows.
production_status: not-for-production
adopted_in: none
---

Planned. The page is filled in when the scores are in.
