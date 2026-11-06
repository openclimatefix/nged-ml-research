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
status: finished
kind: other
protocol: demand-v2
size: >-
  medium. Stored: tables and figures only (private data folder); no fitted weights and no new
  forecasts. Six gates, each one defensible design from the maintainer's brief. Spans experiment
  583's gate and scoring code and experiment 584's C1 forecasts, which this experiment wraps.
  Serving path and degradation rule: not touched (this repository holds none). Cost: CPU only.
hypothesis: >-
  The gate of experiment 583 reads power that is at least 24 h old and needs 3 days of a sustained
  gap, so it fires a median 4 to 6 days after a load-shift begins, and the C2 band of experiment 584
  covers the normal running arrangement (NRA) on only 0.889 of rows from day 4 to the end. A gate
  that reads the freshest observed power (up to the forecast origin minus 3.5 h) against a lag-free
  expectation fires within about 1 to 2 days, and lifts that coverage to 0.90 or more at under 1
  false trigger per Primary-year.
prediction: >-
  Written before any score. P1: G3 (the fast gate) fires a median of at most 2 days after onset on
  injected 15 and 30 day events, against 4 to 6 days for G0. P2: G3 lifts day-4-to-end coverage of
  the C2 band on the injected bank (all 1,248 Primary-events) to at least 0.90. P3: G3 raises false
  triggers on event-free rows above 1 per Primary-year, and G4 (G3 plus a mirrored opposite-sign gap
  at an electrical neighbour) brings them back to 1 or below. P4: G1 and G2 (sustain of 2 and 1 days
  on the day-old lags) lift coverage by under 0.01 each and raise false triggers. P5: G5 (the snap-
  back rule) ends the wider band earlier than G0, with a smaller after-end share of rows covered
  than G0 or G3. The prediction fails for P2 if no gate reaches 0.90, and for P3 if G4 exceeds 1
  false trigger per Primary-year.
decision_rule: >-
  A gate is best if, among the gates with at most 1 false trigger per Primary-year on event-free
  rows (Primary-years counted on event-free origin days), it has the highest day-4-to-end coverage
  on the injected bank; it lifts that coverage to 0.90 only if the lower end of its 95% event-
  bootstrap interval is at least 0.90, and is worth porting only if its extra event-free pinball
  loss at q05 and q95 over G0 is no more than the minimum detectable effect. Counted tests: 10 (five
  gates G1 to G5 against G0, times the q05 and q95 pinball loss on event-free rows at 24 to 120 h,
  pooled over both metering groups). Each contrast uses a five-block month bootstrap at the
  Bonferroni level, and the half-width of that interval is printed beside it as the minimum
  detectable effect. The real December pair (D5 and D16, 2 Primary-events) is descriptive only.
benchmark: >-
  G0, experiment 583's gate, with experiment 584's C1 forecast and its lags-masked twin (stored).
row_set_id: experiment_470_intersection_v1
metric: >-
  Onset delay in days, detection rate, end-detection delay in days, false triggers per Primary-year
  on event-free rows, share of the NRA inside q05-q95 by phase (before, days 0-3, day 4 to the end,
  0-7 days after), and pinball loss at q05 and q95 on event-free rows.
verdict: partly supported
best_arm: >-
  G4 (the fast gate plus a mirrored opposite-sign gap at an electrical neighbour). On injected 15
  and 30 day load-shifts (672 Primary-events), 24 to 120 h ahead, G4's band holds the normal running
  arrangement on 0.913 (0.906 to 0.920) of rows from day 4 to the end, against 0.889 for G0, with 1
  false trigger in 4.97 Primary-years.
headline: >-
  G3, G4 and G5 fire a median 1 day after onset (G0: 6 days). From day 4 to the end the band holds
  the NRA on 0.915 (G3, G5) and 0.913 (G4), against 0.889 for G0. G3 and G5 fire falsely 2.6 times
  per Primary-year; G4 0.20 (1 trigger; one-sided 95% upper bound 0.95), so G4 alone clears both
  bars. On the verified December pair no gate changes day-4-to-end coverage (D16 0.685 for every
  gate).
lessons: >-
  A gate that reads fresh power fires 3 to 5 days sooner, and a mirrored neighbour removes its false
  alarms at the cost of 2.6 points of detection rate. The idealised neighbour in the injected bank,
  the 5 Primary-years of false-trigger evidence, and the unchanged verified pair mean the gain is
  not yet shown on real events.
take_home_message: >-
  A gate that watches the freshest power instead of day-old power switches the wider band on about 5
  days sooner, and raises coverage from day 4 of a load-shift to its end from 0.889 to 0.913 on
  injected events, but it also fires falsely 2.6 times per Primary-year unless it requires an
  opposite-sign gap at a neighbouring Primary (0.2 per year). The gain is not seen on the two real
  December events, so the next step is to test it on real load-shifts before relying on it.
advice:
  - >-
    Read the freshest observed power (up to 4 h before the forecast) against a lag-free expectation
    to switch on a wider band, rather than day-old lags, and require a mirrored opposite-sign gap at
    a neighbouring Primary to keep false triggers at 1 per Primary-year or below (G3 against G4,
    false triggers). - >- Do not add a snap-back rule to the fresh-power gate: it left coverage
    unchanged and released the band early in 21% of events (G5 against G3, release table). - >-
    Count false triggers over many more Primary-years before trusting a rate: 5 Primary-years give 1
    trigger and an upper bound of 0.95 (false-trigger table). - >- Test any new gate on real load-
    shifts as well as injected ones: on the verified December pair every gate scored the same (real
    pair figure).
production_status: candidate
adopted_in: none
---

**A gate that reads the freshest power turns the wider band on about 5 days sooner, and a mirrored
opposite-sign gap at a neighbouring Primary keeps its false alarms low.** On injected 15 and 30 day
load-shifts, the share of rows from day 4 to the end with the normal running arrangement (NRA)
inside the q05-q95 band rises from 0.889 to 0.913 (G4). The same gate without the neighbour rule
reaches 0.915 but fires falsely 2.6 times per Primary-year. The two verified December events do not
move at all, so the gain is shown on injected events only. Verdict: partly supported.

A load-shift is a period in which part of one Primary substation's demand is supplied through
another Primary, so recent power readings no longer show the NRA (the normal running arrangement).
The forecast widens its band while a gate says a load-shift is in progress, and [experiment
584](../00584-nra-candidate-pack-x6/result.md)'s gate, copied from [experiment
583](../00583-after-snap-back-event-band-on-x3-syn/result.md), needs 3 days of day-old readings, so
it fires a median 4 to 6 days after the load-shift begins. We asked whether a faster gate, or a gate
that reacts to the end of the load-shift, closes the gap to the 0.90 target. We re-scored the stored
experiment 584 forecasts under six gates on the 1,248 injected Primary-events of [experiment
581](../00581-robust-multiweek-lag-encoder-x3/result.md) and on the verified pair of [experiment
580](../00580-verified-load-shift-event-set-and-label-audit/result.md). Nothing is refitted. Nothing
is adopted for production.

## Figures

![Coverage from day 4 to the end against false triggers, per gate](headline_coverage_against_false_triggers.svg)

G3 sits exactly under G5 (identical coverage and false triggers), so only G5's green dot shows.

![Share of injected events with the gate on by day after onset](gate_on_by_day_after_onset.svg)

![Coverage by phase and gate, injected 15 and 30 day events](coverage_by_phase.svg)

![When each gate is on, eight random injected events](gate_timelines.svg)

![Delay from the end of the load-shift to the gate releasing](gate_release_delay.svg)

![False-alarm cost on event-free rows](false_alarm_cost.svg)

![Coverage on the verified December pair](real_pair_coverage.svg)

Day-ahead forecasts under G0, G3, and G4 around the two verified events (the 14-day horizon cannot
be drawn, because the stored forecasts reach only 120 h):

![D16 verified event](event_verified_D16.svg)

![D5 verified event](event_verified_D5.svg)

Four injected events (15 and 30 days, up and down, both metering groups):

![Injected event 1](event_injected_1.svg)

![Injected event 2](event_injected_2.svg)

![Injected event 3](event_injected_3.svg)

![Injected event 4](event_injected_4.svg)

## Gates

- **G0** (control): experiment 583's gate. The gap is observed power minus the lags-masked forecast
  from day-old data. It turns on after 3 days above the threshold, and off when the gap falls below
  half the threshold.
- **G1, G2:** the same gate with a sustain of 2 days and of 1 day.
- **G3:** one reading per forecast origin: the mean of observed power minus the median of the
  lag-free forecast of [experiment 565](../00565-drift-embedding-variants-six-designs/result.md)
  (E2), over the 40 half-hours from the previous origin up to 4.0 h before this origin (forecasts
  are issued at 09:00 UTC with a first lead of 0.5 h). It turns on after one reading above the
  threshold (the larger of 3 standard deviations and 5% of capacity, taken on event-free days) and
  off when a reading falls below half the threshold.
- **G4:** G3, turning on only if a metered electrical neighbour (NGED's interconnection list) has a
  reading of the opposite sign above its own threshold on the same day.
- **G5:** G3, also releasing (a snap-back) when the reading changes sign, or falls by more than 60%
  in one day.
- **Band under every gate:** experiment 584's C2. It is the widest of C1's band and its lags-masked
  twin's band while the gate is on and for 7 days after it releases, and C1's own band elsewhere.

## Results

- **Row set and recipe:** `experiment_470_intersection_v1`, five fold months (2025-10, 2025-12,
  2026-01, 2026-03, 2026-06), 24 to 120 h unless stated, July 2026 unused, cleaned data, D15 and D18
  cut before 2025-03-01, seeds 0 and 1 averaged, coarse refits (days 0 and 14). The contrast could
  depend on the retraining cadence. No fit, so no booster was saved. Forecasts reused: experiment
  584's C1 (`c1/work`), its lags-masked twin, its injected bank, and E2 (every scored row has an E2
  median).
- **Coverage from day 4 to the end** (injected 15 and 30 day events, 756 Primary-events with a day
  4, q05-q95, 95% event-bootstrap interval): G0 0.889 (0.882 to 0.897), G1 0.896 (0.888 to 0.903),
  G2 0.904 (0.896 to 0.910), G3 0.915 (0.908 to 0.922), G4 0.913 (0.906 to 0.920), G5 0.915 (0.908
  to 0.922). With the 3 and 7 day events added (1,092 Primary-events): G0 0.884, G2 0.899, G3 0.910,
  G4 0.908 (0.902 to 0.915).
- **Coverage 0 to 7 days after the end** (672 Primary-events): G0 0.927, G3 0.925, G4 0.923. The
  gates are equal within the intervals, and the band was already near 0.92 after the end in
  experiments 583 and 584.
- **How fast each gate turns on** (injected events of 7, 15, and 30 days, 1,260 Primary-events,
  median days from onset to the first gate-on origin day, share detected): G0 6 (0.892), G1 5
  (0.915), G2 4 (0.956), G3 1 (0.931), G4 1 (0.905), G5 1 (0.931). The median days from the end to a
  release is 2 for G0, G1 and G2, and 1 for G3, G4 and G5.
- **False triggers per Primary-year** (event-free origin days at 16 Primaries, 4.97 Primary-years
  for the fresh gates; one-sided 95% upper bound in brackets): G0 0.23 (1.1), G1 0.68 (1.7), G2 3.28
  (5.0), G3 2.61 (4.2), G4 0.20 (0.95), G5 2.61 (4.2). Every Primary has at least one neighbour in
  the interconnection list. With 1 to 15 triggers behind each rate, the rates are rough.
- **False-alarm cost** (pinball loss, event-free rows at 24 to 120 h, eight MVA and six MW
  Primaries, minus G0's, points of capacity, Bonferroni-level interval over 10 contrasts, minimum
  detectable effect in brackets). For q05: G3 +0.0014 (0.0001 to 0.0038; 0.0019), G4 +0.0002
  (-0.0004 to 0.0012; 0.0008). For q95: G3 -0.0026 (-0.0078 to 0.0000; 0.0039), G4 -0.0003 (-0.0011
  to 0.0000; 0.0005). The event-free width of the q05-q95 band is 19.97 (G0), 20.07 (G3), and 20.02
  (G4) points of capacity, and its coverage is 0.883 to 0.884 for every gate. The month-block
  interval has only 5 blocks, so it is close to the range of the five monthly differences.
- **Tails** (day 4 to the end, injected 15 and 30 days): q01-q99 coverage is 0.978 (G0), 0.986 (G3),
  0.985 (G4). Pinball loss at q05, q95, q01, q99 is 0.708, 0.849, 0.178, 0.226 points of capacity
  for G0 and 0.695, 0.808, 0.177, 0.212 for G3. The curtailment-cost metric needs a node and its
  generators' forecasts and is not computed here.
- **Verified December pair** (2 Primary-events, descriptive): from day 4 to the end D16's coverage
  is 0.685 under every gate, although the share of origin days in event mode rises from 0.64 (G0) to
  0.89 (G3, G4, G5). D5's coverage is 0.868 under every gate and 0.583 after the end. D5's gate was
  already on before the event under every gate, as in experiment 583.
- **Early release inside an event:** the share of injected events (7, 15, and 30 days) whose gate
  turned off at least once before the event's end is 0.007 (G0), 0.013 (G1), 0.042 (G2), 0.131 (G3),
  0.122 (G4), and 0.207 (G5).

## Checks

- **Median forecast minus mean forecast per site:** not applicable. Every gate picks between two
  stored bands and keeps C1's median, so the median and its error are identical across gates. The
  contrasts are in band coverage and quantile loss only.
- **Reproduction:** G0 under this experiment's scorer reproduces experiment 584's C2 coverage
  exactly (largest difference 0.0 on 10,440 injected rows and 28 verified-event rows).
- **Triggers of "Audit before merge":** none fired. No control failed, no nested arm beat the arm
  that contains it, and G3 and G5 are equal. A leave-one-Primary-out check of the false-trigger
  counts was not run.
- **Code review:** a fresh Sonnet agent reviewed the code before the first run that fed this page.
  The findings were triaged: the month-edge treatment of the fresh reading was made uniform, the
  false-trigger table gained a "with a neighbour" row, the G0 reproduction check was added, the
  uncounted intervals are labelled, and two docstrings were corrected.
- **Bias of the fresh gap:** the mean fresh reading on days with no gate on is at most 0.17 of the
  threshold (median 0.06), so the threshold and the raw gap agree.

## Caveats

- **The mirrored neighbour is idealised.** In the injected bank the neighbour (one metered
  neighbour, chosen by `(pass + Primary) mod number of neighbours`) receives exactly the opposite
  load, scaled by capacity (MW and MVA capacities are mixed in the ratio). Real load-shifts may be
  split over several Primaries, or go to a neighbour that has no meter.
- **The false-trigger sample is 4.97 Primary-years.** One false trigger is 0.20 per Primary-year, so
  "at most 1" holds for G4 with a one-sided 95% upper bound of 0.95.
- **The two verified events are 2 Primary-events.** Their coverage did not move, so the
  injected-bank gain is not confirmed on real events. NGED may have load-shifts missing from its
  logs, so clean rows may hold unlogged events, the false-trigger counts may include real events,
  and the real-event slices undercount.
- **The old-gate injection is a whole-day shift and the fresh-gate injection is exact to the
  half-hour,** as experiment 583 injected the old gate. A partial first day favours G0 to G2
  slightly.
- **The lead range is 0 to 120 h**, shorter than the 1 to 10 days NGED uses, so every number here is
  a short-lead result.
- **The same gaps decide the threshold and the scoring,** as in experiment 583 (in-sample
  thresholds).

## What to try next

- Test G4 on real load-shifts beyond the December pair, and on a neighbour the real data mirror.
- Replace the single-reading release with a release that needs two quiet readings, to stop G3 and G5
  turning off inside an event (13% and 21% of events).
- Count false triggers over a longer event-free history before any rate is relied on.
