---
id: "00600"
slug: nra-long-lead-scoring-of-assembled-candidates-at-24-120-and-120-240-h
title: >-
  Do C1, A1, and their G4 assembled forecasts still hold the normal running arrangement inside the
  q05-q95 band at 120-240 h ahead, where NGED's horizon reaches 10 days?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Medium. Triggers: stores only tables and figures in the private data folder (no); touches no
  serving path or degradation rule (no); one defensible design, because the experiment re-scores
  stored forecasts with experiment 597's code at a longer lead (no); spans code whose callers can be
  named, because it imports experiments 582, 583, 584, 592, and 597 (no). No fit and no GPU.
hypothesis: >-
  The G4 gated band keeps its coverage of the normal running arrangement (NRA) from day 4 of a
  load-shift to its end at 120-240 h ahead, and it helps there at least as much as at 24-120 h.
prediction: >-
  Written before any score is read. P1: at 120-240 h, on injected 15 and 30 day load-shifts, the
  coverage of the NRA by the q05-q95 band from day 4 to the end is at least 0.90 for C1 + G4 and for
  A1 + G4 (both metering groups pooled). P2: the gate's value, A1 + G4 minus A1 coverage in the 0-7
  days after the end, is not smaller at 120-240 h than at 24-120 h; the same for C1. P3: the
  event-free pinball loss at q05 and q95 of each assembled forecast is within the minimum
  detectable effect of its plain network at 120-240 h. P4: the rebuilt G4 state equals experiment
  592's stored state, and the 24-120 h numbers reproduce experiment 597's (C1 + G4 day-4-to-end
  coverage 0.913).
decision_rule: >-
  P1 holds if both point estimates are at least 0.90; the 95% event-bootstrap interval is printed
  and a lower bound below 0.90 is reported as "not shown". P2 holds if the point estimate of the
  difference between the two gains is at least 0 for A1; it is refuted if the upper bound of its
  Bonferroni interval is below 0. P3 holds if every counted interval lies within plus or minus the
  minimum detectable effect (MDE). P4 is a stop-and-report check. The counted tests are eight: P1
  for C1 + G4 and A1 + G4 (2), P2 for C1 and A1 (2), and the pinball loss of the assembled forecast
  minus its plain network at q05 and q95 for C1 and A1 at 120-240 h on event-free rows (4). Intervals
  use five-block month bootstraps (event-free rows) or event bootstraps (injected bank) at the
  Bonferroni level, with the half-width printed beside each as the MDE.
benchmark: C1 and A1 without the gated band, C0, and E2 (the lag-free neural network of experiment 565).
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS; share of the NRA inside q05-q95 by phase; safe-side share; median follow ratio; tail
  pinball at q01, q05, q95, and q99.
---

**Planned, no score read yet.** Experiment 597 scored C1 + G4 and A1 + G4 only 24 to 120 h ahead,
because it cut the stored forecasts at 120 h. NGED's horizon is 1 to 10 days, so every 597 coverage
number is a short-lead number. The stored forecasts of C0, C1, A1, and E2, and their injected-bank
forecasts, reach 240 h, so this experiment re-scores them with 597's code and no fit.

## Plan

- **Reused, not fitted.** The stored seed forecasts of C1 (experiment 584), A1 (experiment 591), C0
  (experiment 572), their lags-masked twins, their injected-bank forecasts (experiment 581), and E2
  (experiment 565). A probe found 0 to 240 h in all of them. The G4 gate reads only observed power
  up to 4 hours old, E2 at short lead, and neighbours' readings, so its state is the same as in
  experiment 597.
- **What changes.** `e00582_arms.MAX_LEAD_H` is raised from 120 to 240 before the table is built, and
  every row is labelled 0-24, 24-120, or 120-240 h. Everything else is experiment 597's scoring.
- **Rows and recipe.** `experiment_470_intersection_v1`, five fold months, 16 Primaries, July 2026
  unused, cleaned data, D15 and D18 cut before 2025-03-01 (report-only), coarse refits at days 0 and
  14, seeds 0 and 1 averaged. Weather is from the run issued at or before the forecast origin minus
  9 hours, as in the stored forecasts.
- **Scored sets.** Event-free rows, the injected bank (experiment 581), and the real December 2025
  pair (descriptive). Groups: the eight MVA Primaries and the six MW Primaries separately.
- **Standing practices followed.** Monthly refits (inherited from the stored forecasts), cleaned
  data, the D15 and D18 cut, plots under the 2026-11-06 plot rules, and the unlogged-event caveat.
  Weekly refits and the avoidable-curtailment cost are not in scope: no forecast is refitted, and the
  cost metric needs node-level forecasts.
- **Checks.** A fresh Sonnet agent reviews the code before the numbers are quoted; another reviews
  the results; another reviews the write-up; a fourth opens every rendered figure.
