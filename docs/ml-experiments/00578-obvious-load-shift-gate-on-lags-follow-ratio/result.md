---
id: "00578"
slug: obvious-load-shift-gate-on-lags-follow-ratio
title: >-
  Does the lag network follow long, deep load-shift events, and would an inference-time gate on its
  lags stop it?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted weights and forecasts in the private data folder only. More than one
  defensible design (what the gate reads, what it does on a trigger). Spans code whose callers need
  a search (experiments 556, 565, 566, 572, 575, 576 and 400). Serving path and degradation rule:
  not touched (this repository holds none). Cost: ceiling 4 GPU-h; projected about 1 GPU-h.
hypothesis: >-
  The v3+D5 lag network (lags plus the slow output offset) follows long, deep, obvious load-shift
  events at 24-120 h, and a gate that falls back to the network without lags while the Primary's own
  lags sit far from the lag-free expectation stops it without harming the rows with no labelled
  event.
prediction: >-
  Written before any follow ratio or gate score. P1 (problem is real): the lag effect of v3+D5
  (median forecast with the observed lags minus median forecast with the NRA-corrected lags, over the
  labelled depth, 24-120 h, after day 4 of the event) is at least 0.30 on E-obv events, and the
  lag-free D5 network follows at most 0.05; fails if the lag effect is under 0.15. Stop rule: if it
  is under 0.15 the gate arms are not run. P2 to P7 are the brief's: G2 triggers on at least 70% of
  E-obv events within 4 days (fails under 50%) at no more than 1 false trigger per Primary-year; A1
  or A2 lowers the lag effect by at least 0.15 (fails under 0.07); no loss beyond the minimum
  detectable effect on rows with no labelled event; A1 and A2 agree within 0.05; A1 and A2 recover 60%
  of the oracle A3 gain; the error after release stays within the minimum detectable effect.
decision_rule: >-
  Counted tests: 8 (lag effect, A1 and A2 against the ungated network, pooled over groups: 2; CRPS
  inside E-obv events: 2; CRPS on rows with no labelled event, 24-120 h and 0-24 h at the eight MVA
  Primaries and the six MW Primaries: 4). A contrast counts when the Primary-and-month block
  bootstrap interval (5 blocks, near the minimum to maximum of the folds) excludes zero. With fewer
  than 30 events only a large difference can be resolved, and the page says so. Threshold for porting
  a gate: zero (a gate adds one inference-time rule), kept only if no other pre-registered slice is
  hurt by more than the larger of 0.03 and its minimum detectable effect.
benchmark: >-
  The ungated v3+D5 network (experiment 572, stored), the lag-free D5 network (experiment 565, stored),
  and the same fitted weights with the lags masked (the lags-masked twin of experiment 572).
row_set_id: experiment_470_intersection_v1
metric: >-
  Lag effect (share of the labelled depth), CRPS in points of effective capacity at 0-24 h and 24-120
  h, false triggers per Primary-year, days from onset to trigger.
production_status: not-for-production
adopted_in: none
---
**Planned.** Step 0 (the label counts, no score) was run before this plan was committed: of 184
labelled events, 23 last at least 7 days, and 12 of those are at least as deep as the median depth of
the 23 (22.1% of effective capacity). Only 3 of the 12 fall in the five scored fold months of
experiment 572, so two folds (2025-07 and 2025-08) are added; 5 of the 12 (ids 122, 85, 183, 157, and
136) cannot be tested because the ensemble weather summaries start on 2024-04-01.
