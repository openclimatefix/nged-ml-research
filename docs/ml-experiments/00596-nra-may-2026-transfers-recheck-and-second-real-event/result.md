---
id: "00596"
slug: nra-may-2026-transfers-recheck-and-second-real-event
title: >-
  Do the May 2026 load-shift transfers show up in the readings, and can they serve as a second
  real test of the candidate forecast?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: a check table (CPU, no fit) and, only if a transfer passes, small fine-tuned
  weights and forecasts in the private data folder. More than one defensible design (the pass rule
  and the reference window), which is why the rule is fixed here before any gap is computed. Spans
  code whose callers need a search (experiments 574, 582, 584, 591, 592). Serving path and
  degradation rule: not touched (this repository holds none). Cost: CPU, plus at most 0.5 GPU-h.
hypothesis: >-
  At least one of the four May 2026 load-shift events (Primaries D10, D11, D12, and D16) shows a
  gap between the observed power and a forecast that reads no lags, of the sign and about the size
  of its fitted depth, and the gap is confined to the event's span. The candidate forecast (C1 with
  the G4 band) then covers the normal running arrangement (NRA) on at least 0.90 of rows from day 4
  to the end of the event, and the G4 gate fires within 2 days of the onset.
prediction: >-
  Written before any gap is computed. P1: no May 2026 event passes the rule below, because
  experiment 582 found the lag-free median only 0.01 to 0.03 of capacity from the observed power
  at D10, D11, and D16, against fitted depths of 0.11 to 0.15. The prediction fails if D10, D11, or
  D16 passes. P2 (only if an event passes): C1 plus G4 covers the NRA on at least 0.90 of rows from
  day 4 to the end, and G4 fires within 2 days. P2 fails if coverage is below 0.80 or the gate
  fires later than 4 days after the onset.
decision_rule: >-
  The pass rule is below (fixed before the first gap was computed). If at least one transfer
  passes, fine-tune C1 and A1 for 2026-05 and score them by phase; those scores are descriptive
  (at most 2 Primary-events), with no counted test. If none passes, stop and list what to ask NGED.
  The threshold for porting anything is not applicable: nothing here is ported.
benchmark: >-
  The lag-free D5 network (experiment 565 recipe, 2026-05 forecasts stored by experiment 582), the
  v3+D5 network and its lags-masked twin (experiment 582), and, if an event passes, the
  December 2025 pair (D5 and D16) as the reference real event.
row_set_id: experiment_470_intersection_v1
metric: >-
  Mean gap between observed power and the lag-free median inside each event span, minus the same
  gap in the week after the span, as a share of fitted depth; that gap against a per-Primary
  placebo distribution. If an event passes: NRA coverage by phase, follow ratio, and the day the
  G4 gate fires.
---

# Experiment 596 (planned)

**Pass rule, fixed before any gap is computed (2026-11-06).** For each Primary-event (D10, D11,
D12, and D16, from the `events_verified` table of experiment 580, with D12's three overlapping
transfers merged as in experiment 582), using the lag-free D5 median at 24 to 120 h lead:

1. **Signal.** `S` is the mean of (observed power minus lag-free median) over the rows whose valid
   time lies from 1 day after the onset to 1 day before the end, minus the same mean over the 7 days
   that start 1 day after the end. All values are shares of effective capacity.
2. **Sign and size.** The event passes only if `S` has the sign of the fitted depth and `S` lies
   between 0.5 and 1.5 times the depth.
3. **Placebo.** `|S|` must also exceed the 95th percentile of `|S|` for the same Primary and the
   same span length, computed over placebo windows slid by one day across the fold months that have
   lag-free forecasts and lie more than 7 days from every verified transfer.
4. **Transfer.** A transfer passes when the Primary on each side passes, the two signals have
   opposite signs, and the signed MW differ in sum by less than 30% of the larger. D12 enters with
   its merged depth, and the three May transfers are judged through D10, D11, and D16 against D12.
5. **Neighbours.** For each passing transfer, the daily mean of the partner Primary is plotted
   beside it, to look for the mirror by eye.

The lag-free forecast carries a level bias of 5 to 9 points of capacity in December 2025
(experiment 589), which is why the signal subtracts the same gap after the span.

**Reported whatever the outcome:** the table of `S`, depth, ratio, placebo p95, and verdict per
Primary-event; time series plots of observed power, lag-free median, and the fitted NRA at all four
Primaries, with the span marked; the daily means of the neighbouring Primaries. The caveat that
NGED's logs may miss events applies throughout. July 2026 is not used.
