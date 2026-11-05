---
id: "00578"
slug: obvious-load-shift-gate-on-lags-follow-ratio
title: >-
  Does the lag network follow long, deep load-shift events, and would an inference-time gate on its
  lags stop it?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: finished
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
verdict: not supported
best_arm: >-
  Not applicable (no gate was run). Lag effect of v3+D5 on the 7 testable long-and-deep events:
  mean 0.087, median 0.029 of the labelled depth at 24-120 h.
headline: >-
  On the 7 long-and-deep load-shift events that can be tested, reading the lags moves the 24-120 h
  forecast by a median of 0.03 of the event's depth (one event above 0.15). The forecast already
  holds most of the event's level without event lags.
lessons: >-
  The forecast follows long events without the lags: across the 12 long events in the folds, the
  forecast with corrected lags still reproduces a median 0.84 of the event's level. A gate on the
  lags cannot stop that. The slow output offset and the fine-tune are candidates, but no arm without
  them was run. Lags add the most at D8 event 173 (0.31) and at D16 event 55 (0.17).
take_home_message: >-
  The lag network follows long, deep load-shift events only a little through its lags (a median of 3%
  of the event's depth), so a gate on the lags would have little to stop. Most of the event's level
  is already in the forecast without event lags, so a control has to act elsewhere; the slow output
  offset and the fine-tune are the candidates, and were not isolated here. Of the 12 long-and-deep
  events, 5 cannot be tested.
advice:
  - >-
    Do not build a lag gate for long, deep events yet: the lag effect has a median of 0.03 of the
    depth on the 7 testable events (Results).
  - >-
    Run an arm without the slow output offset to find out whether the offset or the fine-tune carries
    a long event's level: the forecast with corrected lags keeps a median 0.84 of it (Results).
  - >-
    Include D8 event 173 (lag effect 0.31) and D16 event 55 (0.17) in any later control of lags, as
    they are the two events where lags add more than 0.15 of the depth (Results).
production_status: not-for-production
adopted_in: none
---
**The lag network follows long, deep load-shift events only a little through its lags, so there is
little for a gate on the lags to stop.** We asked whether the network that reads power lags follows
long, deep events in the logs of load transfers between Primaries, and whether a gate that drops the
lags during such an event would stop it. We re-forecast experiment 572's v3+D5 network (lags plus the
slow output offset) with the lags taken from the NRA-corrected power, which has the labelled events
removed. We compared the two forecasts, in 24-120 h ahead forecasts, for every long event we could
test. The lags moved the forecast by a median 0.03 of the event's depth, so the gate arms were not
run. All numbers are interim and unaudited. **Next step:** if the event's level matters to NGED, test
a control on the offset and the fine-tune, not on the lags. Terms: NRA is NGED's normal running
arrangement; the corrected power is the metered power with the labelled load transfer added back; a
fold is one scored month of experiment 572; D3 to D16 are anonymised Primary labels.

**Verdict: not supported.** The prediction that the lag effect is at least 0.30 on obvious events
fails, and the stop rule (under 0.15) applies. Obvious events among the long-and-deep ones (E-obv, defined in the
plan) number 0: no long-and-deep event shows 3 consecutive days with a daily gap above 3 event-free standard
deviations from the lag-free D5 expectation. One shorter event passes the rule (event 55, D16, 16%
deep, not long and deep). Bonferroni count: 0 of the 8 planned tests were run, and
the lag effect is descriptive, with per-event values in the table (n = 7 long-and-deep events, so only
a large difference could be resolved; no bootstrap interval was computed).

![Follow ratio per event](follow_ratio_per_event.svg)

![14-day plots of the D5 and D16 example events](example_events_14_days.svg)

## Results

**Which events can be tested.** Of 184 labelled events (experiment 400), 23 last at least 7 days.
Their median depth is 22.1% of effective capacity, and 12 are at least that deep (long and deep).
Five of the 12 (ids 122, 85, 183, 157, and 136) cannot be tested, because the ensemble weather
summaries start on 2024-04-01. The other 7 lie in the five folds of experiment 572 (ids 115, 146,
178) or in the two folds added here, 2025-07 and 2025-08 (ids 143, 172, 173, 175). Duration quartiles
(days): 0.15, 0.4, 2.1, 3.6, 39.7; depth quartiles (% of capacity): 1.6, 10.9, 19.2, 31.3, 99.6.

**Follow ratio.** The follow ratio is the mean of (forecast with observed lags minus forecast with corrected lags)
divided by the labelled depth, over valid times in the event, at 24-120 h. Five more events of 7
days or more (ids 28, 55, 112, 113, 174) lie in the folds and are shown beside the 7.

| Event | Primary | Days, depth | Follow ratio | Days 0-3, 4-7, 8-14 after onset |
|---|---|---|---|---|
| 115 | D3 | 7.0, 25% | 0.029 | 0.05, 0.09, - |
| 143 | D5 | 8.4, 43% | 0.018 | 0.04, 0.08, - |
| 146 | D5 | 39.7, 51% | 0.132 | 0.07, 0.18, 0.18 |
| 172 | D8 | 8.4, 24% | 0.077 | 0.13, 0.25, - |
| 173 | D8 | 19.3, 23% | 0.311 | 0.21, 0.32, 0.36 |
| 175 | D8 | 14.2, 22% | 0.017 | -0.04, 0.12, 0.15 |
| 178 | D8 | 7.0, 31% | 0.028 | 0.05, 0.11, - |
| 55 (D16, the example pair) | D16 | 15.5, 16% | 0.169 | 0.08, 0.27, 0.26 |

On the 7 long-and-deep events the mean is 0.087, the median 0.029, and one event (173) is above
0.15. Across all 12 long events that lie in the folds the mean is 0.110, the median 0.070, and three
(55, 173, 174) are above 0.15. The ratio is lowest in days 0 to 3 and higher after day 4 (all-event
means 0.07 and 0.18). The lags-masked version gives the same sign for about 7 of the 12 events and a
different sign for events 28, 112, 115, and 175, so single-event values are noisy.

**The D5 offset absorbs the event.** The level share is the sum of (forecast minus corrected power)
over the sum of (observed minus corrected power) inside the event, at 24-120 h. With observed lags
the median level share over the 12 long events in the folds is 0.95, with corrected lags 0.84, with masked lags 0.96, and for
the lag-free D5 network 0.73 (4 events, all in the five old folds). So the forecast reproduces most
of a long event's level even when it reads no event lags. That comparison does not isolate the offset: the
corrected-lag forecast uses the weights fine-tuned on the uncorrected targets, so the offset and the
fine-tune are both candidates. The exception is the example pair at D16 (event 55): the level share
is 0.24 with observed lags and 0.12 with corrected or masked lags and for the lag-free network.

**Not run.** The gates (G1 to G3), the actions A1 to A4, the injected-world tuning, and the cost on
rows with no labelled event, because the stop rule applied. The lag-free D5 network (E2) was not
re-fitted for 2025-07 and 2025-08, so event 173 and the other new-fold events have no E2 figures.

## Method notes

- The re-forecast of the five old folds equals experiment 572's stored v3+D5 forecasts exactly
  (maximum absolute difference 0.0 over 1,142,520 rows).
- Weather: every forecast row uses the ensemble run issued at or before the origin minus 9 h
  (example row: origin 2026-06-08 09:00, run 2026-06-08 00:00). Every lag, observed or corrected, is
  at least 24 h before the origin (asserted by experiment 572's checks; the corrected variant leaves
  the lag times unchanged).
- Event-free gap standard deviations and the lag-free expectation use experiment 565's D5 forecasts,
  the shortest stored lead per half-hour.
- GPU time: about 0.5 GPU-h (unmeasured), against a ceiling of 4. Reused code: experiments 400, 435
  (NRA correction), 556, 565, 572, and 575.
- Unlogged events: NGED's logs may miss events, so the clean-lag rows may hold events no label
  covers, and the counts here are lower bounds.
