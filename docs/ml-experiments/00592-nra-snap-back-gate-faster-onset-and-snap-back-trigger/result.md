---
id: "00592"
slug: nra-snap-back-gate-faster-onset-and-snap-back-trigger
title: >-
  Can a faster gate, or a gate that fires at the snap-back, lift the share of rows with the normal
  running arrangement inside the band from day 4 of a load-shift to its end to 0.90, with fewer than
  1 false trigger per Primary-year?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: tables and figures only (private data folder); no fitted weights and no new
  forecasts. Six gates, each one defensible design from the maintainer's brief. Spans experiment
  583's gate and scoring code and experiment 584's C1 forecasts, which this experiment wraps.
  Serving path and degradation rule: not touched (this repository holds none). Cost: CPU only.
hypothesis: >-
  The gate of experiment 583 reads power that is at least 24 h old and needs 3 days of a sustained gap, so
  it fires a median 4 to 6 days after a load-shift begins, and the C2 band of experiment 584 covers the
  normal running arrangement (NRA) on only 0.889 of rows from day 4 to the end. A gate that reads the
  freshest observed power (up to the forecast origin minus 3.5 h) against a lag-free expectation fires
  within about 1 to 2 days, and lifts that coverage to 0.90 or more at under 1 false trigger per
  Primary-year.
prediction: >-
  Written before any score. P1: G3 (the fast gate) fires a median of at most 2 days after onset on
  injected 15 and 30 day events, against 4 to 6 days for G0. P2: G3 lifts day-4-to-end coverage of the
  C2 band on the injected bank (all 1,248 Primary-events) to at least 0.90. P3: G3 raises false
  triggers on event-free rows above 1 per Primary-year, and G4 (G3 plus a mirrored opposite-sign gap at an
  electrical neighbour) brings them back to 1 or below. P4: G1 and G2 (sustain of 2 and 1 days on
  the day-old lags) lift coverage by under 0.01 each and raise false triggers. P5: G5 (the snap-back rule) ends the
  wider band earlier than G0, with a smaller after-end share of rows covered than G0 or G3. The
  prediction fails for P2 if no gate reaches 0.90, and for P3 if G4 exceeds 1 false trigger per Primary-year.
decision_rule: >-
  A gate is best if, among the gates with at most 1 false trigger per Primary-year on event-free rows
  (Primary-years counted on event-free origin days), it has the highest day-4-to-end coverage on the
  injected bank; it lifts that coverage to 0.90 only if the lower end of its 95% event-bootstrap interval
  is at least 0.90, and is worth porting only if its extra event-free pinball loss at q05 and q95 over
  G0 is no more than the minimum detectable effect. Counted tests: 10 (five gates G1 to G5 against G0,
  times the q05 and q95 pinball loss on event-free rows at 24 to 120 h, pooled over both metering
  groups). Each contrast uses a five-block month bootstrap at the Bonferroni level, and the half-width of
  that interval is printed beside it as the minimum detectable effect. The real December pair
  (D5 and D16, 2 Primary-events) is descriptive only.
benchmark: >-
  G0, experiment 583's gate, with experiment 584's C1 forecast and its lags-masked twin (stored).
row_set_id: experiment_470_intersection_v1
metric: >-
  Onset delay in days, detection rate, end-detection delay in days, false triggers per Primary-year on
  event-free rows, share of the NRA inside q05-q95 by phase (before, days 0-3, day 4 to the end, 0-7 days
  after), and pinball loss at q05 and q95 on event-free rows.
---

Planned. Gates: G0 experiment 583's gate (control); G1 sustain of 2 days; G2 sustain of 1 day; G3 the
freshest observed power up to the forecast origin minus 3.5 h against the lag-free expectation over
the last 24 h; G4 G3 plus a mirrored opposite-sign gap at an electrical neighbour; G5 a snap-back rule.
