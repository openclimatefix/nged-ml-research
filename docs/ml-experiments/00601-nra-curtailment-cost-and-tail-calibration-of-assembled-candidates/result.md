---
id: "00601"
slug: nra-curtailment-cost-and-tail-calibration-of-assembled-candidates
title: >-
  Does the G4 gated band lower or raise the curtailment cost of C1 and A1, and how well are the
  upper tails of the assembled load-shift forecasts calibrated?
theme: curtailment-cost
also_relevant_to:
  - lagged-power-and-error-correction
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Medium. Stored: tables and figures in the private data folder; no fit, no new weights, no new
  forecast. Designs: one defensible design, because the cost metric and the stored forecasts are
  fixed and only the row sets are chosen. Callers: the scripts import experiments 581 to 584, 592,
  and 597 and `studies.eval_protocol.metrics`; nothing imports them. It touches no serving path and
  no degradation rule. CPU only.
hypothesis: >-
  The G4 gated band does not raise the curtailment cost of C1 or A1 on event-free rows, and it
  lowers that cost in the 7 days after an injected load-shift ends.
prediction: >-
  Written before any score. P1: on event-free rows of the six MW Primaries (12 to 36 hours ahead,
  limit fraction f = 0.9), the avoidable curtailment cost of C1 + G4 minus C1 and of A1 + G4 minus A1
  is not above the minimum detectable effect of the month-block contrast. P2: in the 7 days after the
  end of an injected load-shift, the avoidable cost of each gated forecast is lower than its plain
  network's. P3: the q95 exceedance rate of each assembled forecast on event-free rows lies within
  0.01 of 0.05, and the q99 rate within 0.005 of 0.01.
decision_rule: >-
  P1 holds if the upper end of the Bonferroni interval of the event-free contrast is not above its
  half-width, read as the minimum detectable effect. P2 holds if the Bonferroni interval of the
  after-end contrast lies below zero. The counted tests are four: (C1 + G4 minus C1, A1 + G4 minus A1)
  x (event-free rows, 7 days after an injected end). Tail calibration and the real December pair are
  descriptive.
benchmark: C1 and A1 without the gated band; the lags-masked twin and the lag-free E2 as references.
row_set_id: experiment_470_intersection_v1
metric: >-
  Avoidable curtailment cost in pounds (`metrics.avoidable_curtailment_cost`, node only, limit
  fraction f = 0.9 headline and 0.75 and 0.5, 12 to 36 hour leads, six MW Primaries); reliability
  of q95 and q99; pinball loss at q05, q95, and q99; over-forecast and under-forecast shares.
---
**Plan only. No score has been read.**

- Reuse unchanged: the cost metric of experiments 277 and 478 (`metrics.avoidable_curtailment_cost`
  over `studies.node_curtailment`, node-only mode), the injected-bank machinery of experiment 597.
- Systems: C0, E2, C1, C1 + G4, A1, A1 + G4 from `backtest_export_candidates_2026-11-06`. The manual
  heuristic and climatology are not stored for these months, so they are not scored (no fits).
- Row sets: injected bank by phase; real data on event-free versus event rows; the real December pair.
