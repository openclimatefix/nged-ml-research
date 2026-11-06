---
id: "00602"
slug: forecasts-by-encoder-stage-of-the-modular-network
title: >-
  What does each encoder of the modular demand network (datetime, weather, power lag) contribute to
  its forecast, on the December 2025 load-shift, on injected load-shifts, and on an event-free
  fortnight?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  simple. Inference only on stored weights of experiments 574, 584, and 591 (no fit, no new data), a
  descriptive probe with no contrast that decides anything. One defensible design. Serving path and
  degradation rule: not touched (this repository holds none). Compute under 1 GPU-h.
hypothesis: >-
  Descriptive. The datetime encoder alone gives a smooth daily and weekly shape, the weather encoder
  moves it with temperature and sun, and the lag encoder adds the level that follows recent power,
  including a load-shift in the lag inputs.
prediction: >-
  Written before any figure was drawn. P1: with the weather and lag embeddings replaced by their
  training-set mean, the datetime-only median is a plausible demand shape (within 0.25 of capacity
  of the full network's median on at least 90% of event-free rows). P2: the lag encoder's
  contribution (stage 3 minus stage 2 median) follows the sign of an injected load-shift, with a
  magnitude between 0.1 and 1.0 of its depth, for C1 and A1. P3: for plain P5 (clean lags) the lag
  contribution is smaller in magnitude than for C1 on the same injected events. P1 failing sends the
  probe to frozen linear heads on each embedding. P2 and P3 are descriptive.
decision_rule: >-
  Descriptive only: nothing is adopted. Every plot is drawn from one seed (0) and the stored weights
  of the refit in force on the plotted dates. No contrast is tested, so no Bonferroni count applies
  (count: 0).
benchmark: >-
  The full network (stage 3) of each arm, which is the forecast stored by experiments 574, 584, and
  591.
row_set_id: experiment_470_intersection_v1
metric: >-
  Median and q05-q95 band of each stage; the lag encoder's contribution (stage-3 median minus
  stage-2 median) in shares of capacity, against injected or real depth.
production_status: not-for-production
adopted_in: none
---

Planned. A staged probe through the trained mixer replaces the other encoders' activated
embeddings with their training-set mean embedding, giving three stages: (1) datetime only,
(2) datetime and weather, (3) datetime, weather, and power lag (the full network). The probe is
off-distribution, and every caption will say so. Networks: P5 (experiment 574), C1 (experiment 584,
P5 plus lag-only synthetic load-shifts), and A1 (experiment 591, shared lag encoder). Windows: the
December 2025 pair (D5 and D16), four injected designs, and one event-free fortnight.
