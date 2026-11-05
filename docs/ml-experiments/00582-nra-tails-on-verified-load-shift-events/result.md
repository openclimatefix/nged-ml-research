---
id: "00582"
slug: nra-tails-on-verified-load-shift-events
title: >-
  Do the forecast's tails hold the normal running arrangement during and after long load-shift
  events that the telemetry confirms?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: forecasts and fitted weights in the private data folder only. More than one
  defensible design (how the control is recalibrated, how the mixture and the union band are built,
  what the after-snap-back rule masks). Spans code whose callers need a search (experiments 565, 572,
  578, 580). Serving path and degradation rule: not touched (this repository holds none). Cost:
  CPU plus about 0.3 GPU-h.
hypothesis: >-
  On load-shift events that the telemetry confirms (experiment 580), the day-ahead forecast of the
  lag network covers the normal running arrangement (NRA) well during the event and badly in the 7
  days after the load-shift ends, because the week lag still holds the event. A rule that keeps
  the week lag masked for 7 days after the event restores that coverage without a cost on rows with
  no labelled event, and a mixture or union band does no better than the same network recalibrated
  on earlier months.
prediction: >-
  Written before any score. P1: after the end of a verified event, the v3+D5 network's q05-q95 band
  holds the NRA on at most 0.60 of rows at 24-120 h. P2: the gated arm with the after-snap-back rule
  holds it on at least 0.75 of rows, a lift of at least 0.15. P3: on rows with no labelled event, every
  mixture and union arm is within 0.01 points of effective capacity of the recalibrated control R in
  pinball loss at q05 and at q95. The prediction fails if the mixture beats R on those rows by more
  than 0.01, or if the rule lifts after-end coverage by less than 0.15. Stop rule: if R alone lifts
  after-end coverage to 0.75, no blend or gate is needed for the tails.
decision_rule: >-
  Descriptive. Fewer than 10 verified real events exist (4 transfers, 8 Primary-events), so
  every event-phase number is descriptive, with an event-block bootstrap interval that leaves one
  Primary out at a time. Rows with no labelled event carry the counted tests: pinball loss at q05
  and at q95, for the eight MVA Primaries and the six MW Primaries, each arm against R (Bonferroni
  count 4 per arm), with a month-block bootstrap of 5 blocks (near the minimum to maximum of the
  folds) and the minimum detectable effect beside each contrast. Threshold for porting a rule: zero
  (a rule adds one inference-time step), kept only if no pre-registered slice is hurt by more than the
  larger of 0.03 and its minimum detectable effect.
benchmark: >-
  The v3+D5 network (experiments 572 and 578, stored), its lags-masked twin, the lag-free D5 network
  (experiment 565, stored), and the recalibrated control R built here from the v3+D5 forecasts.
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the NRA inside q05-q95 and q01-q99 by phase, tail follow ratio, pinball loss at q01, q05,
  q95, and q99, width inflation against R, false-alarm cost on rows with no labelled event.
---

# Do the forecast's tails hold the normal running arrangement during and after verified load-shift events?

Written before any score. The question, the prediction, and the decision rule are in the front
matter. Design (all inputs stored from experiments 565, 572, 578, and 580, plus one new fold month):

- **Events:** only the telemetry-verified transfers of experiment 580 (`events_verified.parquet`).
  The labelled corrections are not used, because experiment 580 verified 1 of 184. The NRA reference
  is the observed power minus the fitted depth inside the event.
- **Forecasts:** the v3+D5 network, its lags-masked twin, the lag-free D5 network, and a
  corrected-lag twin whose lags are rebuilt from the verified counterfactual. A new fold month
  (2026-05) is fitted with the 572 recipe, because three verified transfers end in May 2026.
- **Arms built from stored quantiles:** the recalibrated control R (per group and lead band, widths
  fitted on earlier months), mixtures of the lag network and the lag-free D5 network, the union band
  of the lag network and its masked twin, and a gated arm with and without an after-snap-back rule.
- **Standing practices followed:** cleaned power everywhere, D15 and D18 cut before 2025-03-01, two
  seeds, the weather run issued at or before the forecast origin (asserted row by row by the 572
  code), month-block bootstrap with 5 blocks, per-fold signs, minimum detectable effect beside every
  contrast.
