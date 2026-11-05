---
id: "00583"
slug: after-snap-back-event-band-on-x3-syn
title: >-
  Does keeping a wider forecast band for 7 days after a load-shift ends make the forecast bands hold
  the normal running arrangement again, at a small loss on days with no load-shift?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: tables and figures only (private data folder); no fitted weights, no forecasts. One
  defensible design per arm; the arms follow the NRA plan review (X4). Spans experiment 582's scoring
  code, which this experiment copies and extends. Serving path and degradation rule: not touched (this
  repository holds none). Cost: CPU only, zero GPU hours (inference-free re-scoring of stored
  forecasts).
hypothesis: >-
  After a load-shift ends, the lag network of experiment 581 (X3-syn) still holds old readings, so its
  q05-q95 band holds the normal running arrangement (NRA) on too few rows. Keeping the union of the
  lag network's band and its lags-masked twin's band for the 7 days after the lag-versus-expectation
  gap collapses restores most of that coverage, at a small loss on days with no load-shift.
prediction: >-
  Written before any score. P1: after the end of an injected lag-only event of 7, 15, or 30 days, arm
  A1 holds the NRA inside q05-q95 on at least 0.80 of rows at 24-120 h, and from day 4 of the event to
  its end on at least 0.80 of rows too (arm A2 only for the latter). P2: on rows with no event, arm A1
  stays within 0.01 points of capacity of X3-syn in the mean pinball loss at q05 and q95, with at most
  1 false trigger per Primary-year. The prediction fails if the false triggers exceed 3 per
  Primary-year, or if the event-free pinball cost exceeds 0.02 points. P3: arm A4 (random trigger days)
  lifts after-end coverage by under a quarter of arm A1's lift. On the verified D5 and D16 pair the
  numbers are descriptive (2 Primary-events).
decision_rule: >-
  Descriptive for every event metric (event-block bootstrap, leave-one-Primary-out range on the
  verified events). Counted tests: 16 (arms A1 to A4 against X3-syn, times the eight MVA Primaries and
  the six MW Primaries, times the q05-q95 band's pinball loss and the q01-q99 band's pinball loss, on
  event-free rows at 24-120 h); month-block bootstrap with 5 blocks at the Bonferroni level, and the
  half-width of that interval printed as the minimum detectable effect beside each contrast. The 0-24
  h band is reported without a test. Threshold for porting a rule: zero extra event-free loss beyond
  the minimum detectable effect, because the rule adds only one inference-time step on stored
  forecasts, and a coverage of 0.80 or more after the end.
benchmark: >-
  Experiment 581's X3-syn lag network and its lags-masked twin (stored), and experiment 582's union
  band as the control (the band on every row).
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the NRA inside q05-q95 and q01-q99 by phase (before, days 0-3, day 4 to the end, 0-7 days
  after, 7-14 days after), the one-sided share on the NRA side, pinball loss at q01, q05, q95, q99,
  width, and on event-free rows the width inflation, the share of days the rule is active, and the
  false triggers per Primary-year.
production_status: not-for-production
---

**Planned; the numbers will follow the scoring run.** The plan is committed before any score is read.

## Question

Experiment 582 found that, in the 7 days after a verified load-shift ends, the forecast bands hold
the normal running arrangement (NRA) on too few rows: 0.40 of rows for v3+D5, 0.59 for X3-syn, and
0.67 for a union band. The target is 0.90. This experiment asks whether a rule that watches the
lag-versus-expectation gap, and widens the band for 7 days after the gap collapses, closes that gap.

## Arms (all on experiment 581's stored X3-syn forecasts; no fit)

- **V:** X3-syn with its lags (the reference).
- **M:** the same weights with every lag masked (the only lag-free forecast X3-syn has stored).
- **Union:** the widest of the two bands at every quantile level, with the lag median, on every row
  (experiment 582's union band, the control).
- **A1:** the union band for 7 days from the origin day a snap-back is seen, X3-syn elsewhere.
- **A2:** the union band on origin days when the gate is on and for 7 days after the snap-back.
- **A3:** A1 with the lags-masked median in place of the lag median during the 7 days.
- **A4 (placebo):** A1 with the snap-backs moved to random origin days (same number per Primary, 20
  draws).

**The gate** (experiment 578's G1, own gap only): the daily mean of (observed minus the lags-masked
median) at 24-48 h lead, the gap, is sustained when it keeps one sign and exceeds
max(3 standard deviations of the Primary's event-free daily gap, 5% of capacity) on 3 consecutive
days. **A snap-back** is the first origin day on which the newest gap day falls below half that
threshold. Every day the gate reads is at least 24 h old at the 09:00 UTC origin (the newest day read
is the origin day minus 2).

## Evidence and scoring

- **Verified events:** the load-shifts of experiment 580 (descriptive; the verdict rests on the
  December pair D5 and D16).
- **Injected bank:** experiment 581's 1,248 Primary-events (lag-only injection, so the observed power
  is the NRA); the gate reads the injected lags, which is the clean gap plus the injected shift.
- **Event-free rows:** the false-alarm cost, width inflation, share of days active, false triggers per
  Primary-year.
- Row sets, phase windows, and the metric code are experiment 582's; weather comes from the run issued
  at or before the origin minus 9 h; July 2026 is unused.
