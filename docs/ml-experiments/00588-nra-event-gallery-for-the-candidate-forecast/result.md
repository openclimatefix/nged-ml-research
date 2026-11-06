---
id: "00588"
slug: nra-event-gallery-for-the-candidate-forecast
title: >-
  How does the candidate normal-running-arrangement forecast behave, plot by plot, on verified,
  labelled, and injected load-shifts and on windows with no load-shift?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: finished
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
verdict: partly supported
best_arm: >-
  C2 (gated band) against C0: on the injected designs, C2 covers the normal running arrangement on
  more rows in the 7 days after the end than C0 in 17 of 24 examples (12 of 15 distinct cases). It
  does
  not in 6 of the 12 designs at the MVA-only Primaries, where C0 already covers 0.92 or more, and in
  1 of 12 at the MW Primaries.
headline: >-
  C2 matches C1 exactly on the 4 event-free windows (P1 holds). C2 beats C0 after the end on both
  verified December Primaries (D16 0.85 against 0.33, D5 0.57 against 0.34), but not before the
  event at D5 (0.46 against 0.80). On injected load-shifts C2 beats C0 in 11 of 12 designs at the
  MW Primaries and in 6 of 12 at the MVA-only Primaries (P2 fails there; P3 does not hold).
lessons: >-
  The gate at D5 was on for the whole plotted December window, so C2 used the union band
  throughout and still covered D5's normal running arrangement on only 0.46 of rows before the
  event. In the labelled events that the power readings did not confirm, the band does not follow
  the label's
  dates and sign.
take_home_message: >-
  The plots show the candidate forecast covering the normal running arrangement more often than the
  earlier forecast in the week after a load-shift ends, with no change on days with no load-shift,
  but
  less often during the load-shift. They also show that the real December pair is a case where the
  gate is on for a month
  and the band still misses before the event. For labelled events that the power readings do not
  confirm, the band does not follow the label, which is expected and not an error.
advice:
  - >-
    Judge a band rule on negative-sign injected events at the eight MVA-only Primaries separately,
    because C0 already covers 0.92 or more there and the gated band adds no coverage (injected
    gallery, D8 and D12).
  - >-
    Read the gate timeline beside every real-event plot, because a gate that is on for the whole
    window, as at D5 in December, widens the band without placing it (verified D5 figure and gate
    timeline).
production_status: not-for-production
adopted_in: none
---

**The plots show the candidate forecast (C2) covering the normal running arrangement (NRA) on more
rows than the earlier v3+D5 forecast (C0) in the 7 days after a load-shift ends, and changing
nothing on days with no load-shift.** The NRA is the power a substation would have drawn without the
load-shift. C2 does worse than C0 while the load-shift is under way and, at one real substation,
before it starts.

Verdict: **partly supported**. The pre-registered rule said the prediction fails if P1 or P2 fails,
and P2 failed at one metering group, so the prediction as a whole failed. Outcomes:

- **P1 held.** C2 equals C1 on all 4 event-free windows.
- **P2 held at the six MW substations (11 of 12 designs) and failed at the eight MVA-only
  substations (6 of 12).** The run drew one substation per design from each group, not the same two
  substations for every design.
- **P3 did not hold.** On the December pair, C2's band is narrower than C0's on the gate-on days
  before and during the event, and wider only after the end (verified table).

Nothing here is adopted for production.

**How to read the plots.** A load-shift moves part of one primary substation's demand onto another,
so recent power readings no longer show the NRA. A primary substation (a Primary) is named D2, D3,
and so on. MW (megawatts) is real power. MVA (megavolt-amperes) is apparent power, and the eight
"MVA-only" Primaries report only apparent power; the six "signed-MW" Primaries report signed real
power. Each figure shows day-ahead forecasts (issued the day before the valid day) for one Primary,
with C0 in the top panel and C2 in the bottom panel, and a key under both. Both panels draw the
forecast the same way: the orange band is the q05 to q95 range (the 90% band) and the dark orange
line is the median. The black line is the observed power. The pale lilac shade is the load-shift.
The sky-blue bar under the C2 panel marks origin days (days on which the forecast was issued) when
the gate was on, or held for 7 days after it released. The gate is a rule that fires when the lagged
power disagrees with a forecast made without it. No figure draws the NRA, because it is only an
estimate (observed power minus the fitted load-shift); every coverage number on this page is scored
against that estimate.
Where a fold month
is mentioned, it is one of the five test months.

**The arms.** C0 is the forecast of [experiment
572](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00572-two-offset-part-1-lags-versus-the-slow-output-offset/result/).
C1 is the P5 recipe of [experiment
584](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00584-nra-candidate-pack-x6/result/):
the same network and training as C2, trained with synthetic load-shifts in its lag inputs, with no
extra band. C2 is C1 with a wider band (the union of C1's band and the band of its lags-masked twin)
while the gate is on and for 7 days after. The union band is that wider band.

## Verified real events

The December 2025 pair is the one verified transfer in the event set of [experiment
580](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00580-verified-load-shift-event-set-and-label-audit/result/)
that falls in a fold month with stored forecasts. Each Primary has one figure for the whole window
(14 days before the start to 14 days after the end) and one zoom on the 7 days after the end. The
lilac shade is the verified window of experiment 580 (2025-12-29 to 2026-01-13). The dashed vertical
lines are the maintainer's switch times for the example event (2025-12-30 01:00 and 2026-01-14
11:30). The power readings place the end of the load-shift on 14 January at both Primaries, so the
maintainer's end time matches the readings and the verified window ends about one day early (see
[experiment
597](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00597-nra-assemble-c1-and-a1-with-g4-gate-as-single-forecasts/result/)
for the step sizes).

![D5, verified load-shift](verified_D5_2025-12-29.svg)

D5 carried 1.5 MW less during the load-shift. The gate was on for the whole window, so C2 used the
union band throughout and the sky-blue bar spans it. Before the start the observed power (black) is
the NRA, and C2's band follows it less well than C0's. The stored gate state of [experiment
584](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00584-nra-candidate-pack-x6/result/)
(`score_c1/gate_state.npz`) is on for every origin day from at least 16 days before the start to 26
days after it, and held for the next 4 days, so the bar really does span the whole axis. The G4 gate
of experiment 597 is a different gate.

![D5, verified load-shift, zoom on the 7 days after the end](verified_D5_2025-12-29_zoom.svg)

The same D5 window, zoomed on the 7 days after the end. Look at how often the black line leaves
each band.

![D16, verified load-shift](verified_D16_2025-12-29.svg)

D16 carried 1.5 MW more. Look at the zoom below: after the end, C2's band is wider than C0's and
contains the observed power more often (after the end the observed power is the NRA).

![D16, verified load-shift, zoom on the 7 days after the end](verified_D16_2025-12-29_zoom.svg)

The same D16 window, zoomed on the 7 days after the end. Only the second switch time falls inside
this window.

![Gate state around each verified event](summary_gate_timeline.svg)

The gate at D5 was on for every origin day in the window. At D16 it fired 5 days after the start and
released 19 days after.

Share of the NRA inside the q05 to q95 band (coverage), by phase: before (7 days before the start),
days0-3 (the first 3 days), day4-end (day 4 to the end), and after0-7 (7 days after the end). Rows
are day-ahead rows only, so the numbers differ slightly from the 24 to 120 h rows of [experiment
584](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00584-nra-candidate-pack-x6/result/).
Widths are in per cent of capacity. Both events belong to one transfer, so every number is
descriptive. During the event C2 covers less than C0 at both Primaries (D16 day4-end 0.68 against
0.90).

| Event | Phase | Rows | C0 coverage | C2 coverage | C0 width | C2 width |
|---|---|---|---|---|---|---|
| D5 2025-12-29 | before | 336 | 0.80 | 0.46 | 23.0 | 19.6 |
| D5 2025-12-29 | days0-3 | 144 | 0.58 | 0.48 | 22.8 | 19.3 |
| D5 2025-12-29 | day4-end | 597 | 0.75 | 0.87 | 22.4 | 20.3 |
| D5 2025-12-29 | after0-7 | 336 | 0.34 | 0.57 | 20.3 | 21.5 |
| D16 2025-12-29 | before | 336 | 0.95 | 0.95 | 14.9 | 12.1 |
| D16 2025-12-29 | days0-3 | 144 | 0.81 | 0.72 | 14.8 | 12.4 |
| D16 2025-12-29 | day4-end | 567 | 0.90 | 0.68 | 17.8 | 15.0 |
| D16 2025-12-29 | after0-7 | 336 | 0.33 | 0.85 | 17.9 | 20.4 |

## Labelled events (label not verified by the power readings)

These are NGED's labelled events of 3 days or more that fall in a fold month with stored forecasts
(11 events; the page shows up to 12). The dashed blue line is the observed power minus the labelled
depth, which is an estimate of the NRA if the label were right. **NGED's labels are known to be
imperfect, so a plot where the forecast does not follow the label is not an error.** The status
comes from [experiment
580](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00580-verified-load-shift-event-set-and-label-audit/result/):
only event 55 is verified, and event 146's label has the wrong dates and sign. "Not seen by the
power rule" is the status that [experiment
580](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00580-verified-load-shift-event-set-and-label-audit/result/)
calls "rejected". Events 177, 178, 181, and 182 are the mirrored halves of labelled pairs (the
partner has the opposite depth on the same dates). Event 179 has no partner. Look at whether the
orange band moves toward the dashed blue line.

![D5, labelled event 146](labelled_146_D5_40d.svg)

D5, event 146 (+2.90 MW, 39.6 days from 2025-12-05). 580 status: label has wrong dates or sign.
Gate-on origin days in the window: 42.

![D16, labelled event 55](labelled_055_D16_15d.svg)

D16, event 55 (+1.50 MW, 15.5 days from 2025-12-30). 580 status: verified. Gate-on origin days in
the window: 14.

![D3, labelled event 115](labelled_115_D3_7d.svg)

D3, event 115 (+3.22 MW, 7.0 days from 2025-12-15). 580 status: not seen by the power rule. Gate-on
origin days in the window: 0.

![D8, labelled event 178](labelled_178_D8_7d.svg)

D8, event 178 (-3.22 MW, 7.0 days from 2025-12-15). 580 status: not seen by the power rule. Gate-on
origin days in the window: 0.

![D5, labelled event 149](labelled_149_D5_5d.svg)

D5, event 149 (+2.66 MW, 4.8 days from 2026-01-27). 580 status: unpaired step (possible level
change). Gate-on origin days in the window: 6.

![D8, labelled event 182](labelled_182_D8_5d.svg)

D8, event 182 (-2.66 MW, 4.8 days from 2026-01-27). 580 status: not seen by the power rule. Gate-on
origin days in the window: 0.

![D5, labelled event 145](labelled_145_D5_4d.svg)

D5, event 145 (+2.41 MW, 4.2 days from 2025-10-11). 580 status: not seen by the power rule. Gate-on
origin days in the window: 0.

![D8, labelled event 177](labelled_177_D8_4d.svg)

D8, event 177 (-2.41 MW, 4.2 days from 2025-10-11). 580 status: not seen by the power rule. Gate-on
origin days in the window: 0.

![D5, labelled event 147](labelled_147_D5_4d.svg)

D5, event 147 (+3.07 MW, 3.7 days from 2026-01-15). 580 status: not seen by the power rule. Gate-on
origin days in the window: 18.

![D8, labelled event 181](labelled_181_D8_4d.svg)

D8, event 181 (-3.07 MW, 3.7 days from 2026-01-15). 580 status: not seen by the power rule. Gate-on
origin days in the window: 0.

![D8, labelled event 179](labelled_179_D8_3d.svg)

D8, event 179 (-2.90 MW, 3.3 days from 2025-12-28). 580 status: not seen by the power rule. Gate-on
origin days in the window: 0.

## Injected events

Each figure injects one design (a length, a depth as a share of capacity, and a sign) into the lag
inputs of two Primaries, one MVA-only and one signed-MW (March 2026, the bank of [experiment
581](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00581-robust-multiweek-lag-encoder-x3/result/)).
Each figure's caption gives the Primary and the design; the Primary differs between designs. **The
injected shift is added to the lag inputs only and the target stays clean, so the plots show the
lag-input effect**, and the observed power is also the NRA. In each figure, compare the width of the
band in the bottom panel (C2) with the band in the top panel (C0) after the lilac shade ends. All
events in a month end on the same day, so a 30-day event begins before the plotted window (the x
axis then starts above 0). Coverage of the NRA in the 7 days after the end. The sky-blue bar comes
from the gate recomputed on the injected inputs; it is not read from the stored gate state of
experiment 584. Each cell gives C0 coverage / C2 coverage:

| Design | MVA-only Primary (C0 / C2) | MW Primary (C0 / C2) |
|---|---|---|
| 7 days, +15% | D3: 0.62 / 0.89 | D16: 0.66 / 0.97 |
| 7 days, -30% | D8: 0.93 / 0.89 | D9: 0.68 / 0.91 |
| 7 days, -50% | D12: 0.92 / 0.91 | D14: 0.73 / 0.88 |
| 7 days, +50% | D12: 0.70 / 0.84 | D14: 0.42 / 0.90 |
| 15 days, +15% | D4: 0.93 / 0.94 | D16: 0.66 / 0.97 |
| 15 days, -30% | D8: 0.93 / 0.89 | D10: 0.91 / 0.98 |
| 15 days, -50% | D12: 0.92 / 0.91 | D14: 0.73 / 0.88 |
| 15 days, +50% | D12: 0.70 / 0.84 | D14: 0.42 / 0.90 |
| 30 days, +15% | D22: 0.88 / 0.93 | D5: 0.83 / 0.95 |
| 30 days, -30% | D8: 0.93 / 0.89 | D11: 0.99 / 0.99 |
| 30 days, -50% | D12: 0.92 / 0.91 | D16: 0.71 / 0.99 |
| 30 days, +50% | D12: 0.70 / 0.84 | D16: 0.26 / 0.98 |

D8, D12, D14, and D16 appear in several designs because the bank assigns each Primary to one design
per pass. All events end on the same day, so their after-end rows are the same rows and their
coverage repeats. The 24 examples hold 15 distinct Primary and design cases, and C2 covers more than
C0 in 12 of them. At the MVA-only Primaries C2 covers more than C0 in 6 of 12 designs. The 6
exceptions are all negative-sign events at D8 and D12, where C0 already covers 0.92 or more.

![D3, injected 7 days, +15%](injected_7d_up_15pct_mva.svg)

D3 (MVA-only eight): +15% of capacity for 7 days.

![D16, injected 7 days, +15%](injected_7d_up_15pct_mw.svg)

D16 (signed-MW six): +15% of capacity for 7 days.

![D8, injected 7 days, -30%](injected_7d_down_30pct_mva.svg)

D8 (MVA-only eight): -30% of capacity for 7 days.

![D9, injected 7 days, -30%](injected_7d_down_30pct_mw.svg)

D9 (signed-MW six): -30% of capacity for 7 days.

![D12, injected 7 days, +50%](injected_7d_up_50pct_mva.svg)

D12 (MVA-only eight): +50% of capacity for 7 days.

![D14, injected 7 days, +50%](injected_7d_up_50pct_mw.svg)

D14 (signed-MW six): +50% of capacity for 7 days.

![D12, injected 7 days, -50%](injected_7d_down_50pct_mva.svg)

D12 (MVA-only eight): -50% of capacity for 7 days.

![D14, injected 7 days, -50%](injected_7d_down_50pct_mw.svg)

D14 (signed-MW six): -50% of capacity for 7 days.

![D4, injected 15 days, +15%](injected_15d_up_15pct_mva.svg)

D4 (MVA-only eight): +15% of capacity for 15 days.

![D16, injected 15 days, +15%](injected_15d_up_15pct_mw.svg)

D16 (signed-MW six): +15% of capacity for 15 days.

![D8, injected 15 days, -30%](injected_15d_down_30pct_mva.svg)

D8 (MVA-only eight): -30% of capacity for 15 days.

![D10, injected 15 days, -30%](injected_15d_down_30pct_mw.svg)

D10 (signed-MW six): -30% of capacity for 15 days.

![D12, injected 15 days, +50%](injected_15d_up_50pct_mva.svg)

D12 (MVA-only eight): +50% of capacity for 15 days.

![D14, injected 15 days, +50%](injected_15d_up_50pct_mw.svg)

D14 (signed-MW six): +50% of capacity for 15 days.

![D12, injected 15 days, -50%](injected_15d_down_50pct_mva.svg)

D12 (MVA-only eight): -50% of capacity for 15 days.

![D14, injected 15 days, -50%](injected_15d_down_50pct_mw.svg)

D14 (signed-MW six): -50% of capacity for 15 days.

![D22, injected 30 days, +15%](injected_30d_up_15pct_mva.svg)

D22 (MVA-only eight): +15% of capacity for 30 days.

![D5, injected 30 days, +15%](injected_30d_up_15pct_mw.svg)

D5 (signed-MW six): +15% of capacity for 30 days.

![D8, injected 30 days, -30%](injected_30d_down_30pct_mva.svg)

D8 (MVA-only eight): -30% of capacity for 30 days.

![D11, injected 30 days, -30%](injected_30d_down_30pct_mw.svg)

D11 (signed-MW six): -30% of capacity for 30 days.

![D12, injected 30 days, +50%](injected_30d_up_50pct_mva.svg)

D12 (MVA-only eight): +50% of capacity for 30 days.

![D16, injected 30 days, +50%](injected_30d_up_50pct_mw.svg)

D16 (signed-MW six): +50% of capacity for 30 days.

![D12, injected 30 days, -50%](injected_30d_down_50pct_mva.svg)

D12 (MVA-only eight): -50% of capacity for 30 days.

![D16, injected 30 days, -50%](injected_30d_down_50pct_mw.svg)

D16 (signed-MW six): -50% of capacity for 30 days.

## Event-free control

Four 14-day windows with no labelled or verified load-shift within 7 days. Rule: the lowest-numbered
Primary of the stated group whose 14 days from day 8 of the month hold no event, for 2025-10 and
2026-06 (MVA-only) and 2026-03 and 2026-06 (MW). The gate was off on every origin day in all four
windows, and C2's 11 stored quantiles equal C1's on every one of the 672 rows in each window
(maximum difference 0.0). Each panel shows C2 only, because C1 is identical on every row.

![Event-free windows, first two](event_free_windows_1.svg)

The first two event-free windows. Look for any gap between the band and the black line: the
gate is off, so the band is C1's band.

![Event-free windows, last two](event_free_windows_2.svg)

The last two event-free windows, read the same way.

## Summary figures

![Coverage by phase](summary_coverage_by_phase.svg)

Injected events of 15 and 30 days, both groups, 24 to 120 h, from the tables of [experiment
584](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00584-nra-candidate-pack-x6/result/).
Look at after0-7: C2 covers 0.927, against 0.860 for C1 and 0.759 for C0 (from [experiment
584](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00584-nra-candidate-pack-x6/result/)).

![Band width by phase](summary_width_by_phase.svg)

Look at the phases where the gate is on: C2's band is wider than C1's there, and equal before.

![Daily band width, verified events](summary_width_over_time.svg)

The orange line (C2) lies above the purple line (C1) on gate-on days. Colours differ from the other
figures: here purple is C1 and orange is C2.

![Coverage by phase for each verified event](from584_coverage90_verified_events.svg)

Copied from [experiment
584](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00584-nra-candidate-pack-x6/result/);
look at how coverage changes between phases. The follow ratio by phase is not redrawn, because
[experiment
584](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00584-nra-candidate-pack-x6/result/)
has no follow-ratio figure. [experiment
582](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00582-nra-tails-on-verified-load-shift-events/result/)
has one for earlier arms on the same events.

![Coverage by phase, injected events of 15 and 30 days](from584_coverage90_15_and_30_days_both_groups.svg)

The same coverage as the first summary figure, with the earlier arms of the experiment.

Look at the q05 and q95 bars: C2's pinball loss (the penalty for a quantile forecast; lower is
better) is lowest after the end.

![Tail pinball loss after the end](from584_tail_pinball.svg)

## What the plots show

- **Event-free days are unchanged.** C2's quantiles equal C1's on all 4 windows (event-free figure).
- **After the end, C2 beats C0 on both verified Primaries.** D16 0.85 against 0.33 and D5 0.57
  against 0.34 (verified table; descriptive, one transfer).
- **Before and during the event, C2 covers the NRA less often than C0.** At D5 before the event 0.46
  against 0.80, with the gate on for the whole window. At D16 from day 4 to the end 0.68 against
  0.90 (verified table). [experiment
  584](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00584-nra-candidate-pack-x6/result/)
  reports the same.
- **On injected events C2 covers more than C0 after the end in 17 of 24 examples (12 of 15 distinct
  cases).** The exceptions are negative-sign events at D8 and D12, and one near-tie at D11 (0.991
  against 0.988; injected table).
- **For labelled events that the power readings did not confirm, the band does not follow the
  label's dates and sign.** The gate was on for 18 origin days of event 147, 6 of event 149, and 42
  of event 146, and for none of the others except event 55 (labelled figures).

## Checks and caveats

- **Row set:** `experiment_470_intersection_v1`, five fold months (2025-10, 2025-12, 2026-01,
  2026-03, 2026-06), July 2026 unused, D15 and D18 cut before 2025-03-01, cleaned data, seeds 0 and
  1 averaged, coarse refits (days 0 and 14). Weather comes from the run issued at or before the
  forecast origin minus 9 h, and lags are at least 24 h old. Stored forecasts cover 0 to 120 h, so
  NGED's 1 to 10 day horizon is not shown.
- **Descriptive only:** one verified pair, 11 labelled events, and 12 injected designs, each at one
  MVA-only and one MW Primary. No contrast is tested. Where [experiment
  584](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00584-nra-candidate-pack-x6/result/)
  quotes a contrast, its minimum detectable effect is on that page.
- **Unlogged events:** NGED's logs may miss events, so the event-free windows may hold unlabelled
  events, and the real-event slices undercount.
- **Injected events** make any rule keyed on the lags look better than it would on real events,
  because the gate reads the injected lags and not a change in the power ([experiment
  583](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00583-after-snap-back-event-band-on-x3-syn/result/)).
- **Not computed:** the avoidable-curtailment cost (the cost of curtailing a generator more than a
  perfect forecast would require; it needs node forecasts with generators), tail tables beyond those
  of [experiment
  584](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00584-nra-candidate-pack-x6/result/),
  and weekly refits.
- **Injected rows are cut by the March origin mask:** rows before the start and late after the end
  are missing, so injected coverage here is not comparable with the 24 to 120 h bank tables of
  [experiment
  584](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00584-nra-candidate-pack-x6/result/).
- **Reviews:** a fresh Sonnet agent reviewed the script before the numbers were quoted and found no
  leak and no generator data. A second reviewed the write-up against the tables.

Scripts: `ml-experiments/00588-nra-event-gallery-for-the-candidate-forecast/e00588_gallery.py`,
which wraps the code of experiments 582, 583, and 584 without editing it. Tables are in
`data/ml-experiments/00588-nra-event-gallery-for-the-candidate-forecast/` (private).
