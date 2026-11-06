---
id: "00588"
slug: nra-event-gallery-for-the-candidate-forecast
title: >-
  How does the candidate normal-running-arrangement forecast behave, plot by plot, on verified,
  labelled, and injected load-shifts and on windows with no load-shift?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  simple. A free re-draw of the forecasts stored by experiment 584 (no fit, no new data, no GPU),
  stored: figures and tables in the private data folder only. One defensible design (the
  maintainer asked for plots). Spans experiments 580, 581, 582, 583, and 584, whose code this
  experiment wraps without editing it. Serving path and degradation rule: not touched (this
  repository holds none).
hypothesis: >-
  Plots of experiment 584's stored forecasts show that the gated band (C2) covers the normal running
  arrangement (NRA) better than the earlier v3+D5 forecast (C0) after a load-shift ends, and that C2
  is the same forecast as the P5 recipe trained on synthetic load-shifts (C1) wherever the gate is
  off.
prediction: >-
  Written before any figure was drawn. P1: in each of the 4 event-free windows, C2's q05 and q95
  differ from C1's by under 0.001 of capacity on at least 99% of rows. P2: in at least 10 of the 12
  injected designs, at each of the two Primaries, C2 covers the NRA on more rows in the 7 days after
  the end than C0. P3: on the verified December pair, the plotted C2 band is wider than C0's during
  the gate-on days. The prediction fails if P1 or P2 fails. P3 is descriptive.
decision_rule: >-
  Descriptive only: the experiment adopts nothing and reads no new contrast. Every coverage number
  is shown with its row count, and the real-event numbers (one verified pair, 11 labelled events of
  3 days or more in fold months) are labelled descriptive. Where a contrast is quoted from
  experiment 584, its minimum detectable effect is quoted with it.
benchmark: >-
  C0 (the v3+D5 forecast of experiment 572, stored) against C2 (experiment 584), with C1 as the
  gate-off reference.
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the NRA inside q05-q95 by phase (before, days 0-3, day 4 to the end, 1-7 days after the
  end), band width, and gate state, for each plotted event.
---

Plan only. The write-up follows the figures.

## Plan

- **Reuses stored forecasts of experiment 584** (C0, C1, C2, C3), five fold months, 24 to 120 h
  ahead, July 2026 unused, D15 and D18 cut before 2025-03-01, cleaned data. No fit.
- **Figures:** verified events, labelled events of 3 days or more, 12 injected designs, 4
  event-free windows, summary figures.
- **Reviews:** one fresh Sonnet code review before the numbers are quoted, one Sonnet write-up
  review.
