---
id: "00597"
slug: nra-assemble-c1-and-a1-with-g4-gate-as-single-forecasts
title: >-
  Do C1 and A1 with the G4 gated band, assembled as single forecasts, hold the normal running
  arrangement inside the q05-q95 band on 0.90 of rows from day 4 of a load-shift to its end?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: finished
kind: other
protocol: demand-v2
size: >-
  Medium. Nothing is stored except tables, figures, and an export of forecasts in the private data
  folder; no fit and no new weights. The experiment wraps the stored forecasts of experiments 584 and
  591 and the gate of experiment 592. It touches no serving path and no degradation rule. It runs on
  CPU only.
hypothesis: >-
  The G4 band of experiment 592 and the C1 forecast of experiment 584 hold up when assembled into one
  forecast and scored by phase, and the A1 forecast of experiment 591 does at least as well.
prediction: >-
  Written before any score. P1: C1 + G4 reproduces experiment 592's day-4-to-end coverage of 0.913
  (stop and report if it differs by more than 0.002). P2: A1 + G4 covers at least 0.92 of rows in the
  0-7 days after the end of an injected load-shift. P3: the event-free pinball loss at q05 and q95
  of each assembled forecast is within the minimum detectable effect of its plain network. P4: the
  verified December pair is unchanged from experiment 592.
decision_rule: >-
  P1 to P4 hold or fail as stated. The counted tests are six: the assembled forecast minus its plain
  network at q05 and q95 for each of C1 and A1 (4), and A1 + G4 minus C1 + G4 CRPS for the two
  metering groups (2), all at 24-120 h on event-free rows. Each contrast uses a five-block month
  bootstrap at the Bonferroni level, and the half-width is printed beside it as the minimum
  detectable effect.
benchmark: C1 and A1 without the gated band, and experiment 592's G4 result.
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the NRA inside q05-q95 and on the safe side of q05 or q95, by phase; median follow
  ratio; tail pinball; CRPS on event-free rows.
verdict: supported
best_arm: >-
  A1 + G4 (the A1 network with the G4 gated band). On injected 15 and 30 day load-shifts (756
  Primary-events with a day 4), 24 to 120 h ahead, it holds the normal running arrangement inside
  q05-q95 on 0.925 (0.918 to 0.931) of rows from day 4 to the end and 0.928 (0.922 to 0.934) in the
  7 days after, against 0.913 and 0.923 for C1 + G4.
headline: >-
  C1 + G4 reproduces experiment 592's day-4-to-end coverage of 0.913 exactly. A1 + G4 reaches 0.925
  and 0.928 after the end. Neither gated forecast adds more event-free pinball loss than the minimum
  detectable effect, and the median still follows 0.11 of an injected shift. On the verified December
  pair, the coverage is below 0.80 in three phases at D5 and two at D16.
lessons: >-
  The gated band works as one forecast, so the remaining weaknesses are the median's following (0.11
  of the depth, up to 20 points of capacity in single 50% events), the level miss before the real
  December load-shift at D5, and refit absorption, which this experiment did not test.
take_home_message: >-
  We built two single forecasts from stored parts, C1 with the G4 band and A1 with the G4 band, and
  scored them on injected load-shifts and on the two real December 2025 load-shifts. On the
  injected load-shifts both hold the normal running arrangement inside the q05-q95 band on at least
  0.91 of rows from day 4 to the end, and A1 + G4 on 0.925, at no measurable cost on event-free
  rows. The median still follows about 0.11 of an injected shift, and the two real events show gaps
  that the injected shifts do not, so the next test worth running is a shift in the power itself.
advice:
  - >-
    Use A1 + G4 as the reference forecast for later load-shift work: it clears the injected-bank
    coverage bars in the day-4-to-end and 0-7-days-after phases (A1 + G4 against C1 + G4 headline bullets).
  - >-
    Run the target-side absorption test (the shift in the training power and the lag inputs) on
    both networks next, because the injected bank changes only the lags and flatters the lag
    networks (caveats bullet).
  - >-
    Do not read a gain on the real December pair from this experiment: the two events show the same
    level miss before the load-shift in every earlier forecast (real-pair bullet).
production_status: candidate
adopted_in: none
---

**We assembled two single forecasts from stored parts and tested them without any new fit. On the
injected load-shifts, both hold the normal running arrangement (NRA) inside the q05-q95 band on at
least 0.91 of rows from day 4 of a load-shift to its end, with pinball loss at q05 and q95 on
event-free rows within the minimum detectable effect.
The median still follows about 0.11 of the shift, and the two real December 2025 events show gaps
that the injected shifts do not.** Verdict: supported. All four pre-registered predictions held: P1
reproduced (difference 0.000000), P2 held (A1 + G4 0.928), P3 held, and P4 held for C1.

A load-shift moves part of one Primary substation's demand onto a neighbour for days or weeks, so
recent power readings stop showing the NRA, the way the electricity network is normally arranged to
supply that Primary. Earlier experiments built the parts of a forecast that copes with this: the C1
neural network ([experiment
584](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00584-nra-candidate-pack-x6/result/)),
the A1 neural network with a shared lag encoder ([experiment
591](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00591-shared-lag-encoder-auxiliary-shift-depth-loss/result/)),
and the G4 rule that widens the band ([experiment
592](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00592-nra-snap-back-gate-faster-onset-and-snap-back-trigger/result/)).
None of the three experiments joined the parts and scored the result as one forecast. We did, and we
exported both forecasts
for others to score. Nothing is adopted for production.

**How the forecast is assembled.** The median is the neural network's own median. While the G4 gate
is on,
and for 7 days after the G4 gate turns off, every quantile is the widest of the network's value and
the value
of its lags-masked twin (the same weights with every lag removed). On other days the quantiles are
the network's own. G4 reads three inputs: observed power up to 4.0 hours old, the lag-free neural
network
E2, and a mirrored opposite-sign gap at a neighbouring Primary. The state is therefore the same for
C1 and A1. The
rebuilt state equals experiment 592's stored state exactly.

## Figures

![Coverage by phase](coverage_by_phase.svg)

Share of rows with the NRA inside q05-q95, on injected 15 and 30 day load-shifts, by phase. Dot:
mean over Primary-events. Line: 95% interval from resampling events. Dashed line: the 0.90 target.
Grey and sky blue are the plain neural networks; orange and blue are the assembled forecasts. The
gated
band lifts coverage in the two phases where the gate is on, and changes nothing 7 to 14 days after
the end.

![Safe-side share by phase](safe_side_by_phase.svg)

Share of rows with the NRA on the safe side of q05 or q95: below q95 when the shift lowered the
load, above q05 when it raised it. Dashed line: the 0.95 target.

![Coverage by phase, all lengths](coverage_by_phase_all_lengths.svg)

This figure repeats the first figure for all injected events (3, 7, 15, and 30 days). Only the 3 and
7 day
events have rows before the shift and in days 0 to 3, because the 15 and 30 day events start before
the scored month begins.

![Follow histogram](follow_histogram_by_depth_and_length.svg)

How far the median moved towards the injected shift, event by event, from day 4 to the end. 0 means
the median ignored the shift. The shift is 15, 30, or 50 points of capacity. Most events move the
median by under 5 points, and a few move it by 10 to 20 points in 50% events.

![D5, day-ahead forecasts through the December 2025 load-shift](real_pair_D5_day_ahead.svg)

D5 through the verified December 2025 load-shift: C1 + G4 above, A1 + G4 below, day-ahead, from 14
days before the start to 14 days after the end. The grey shade is the verified window of experiment
580 (2025-12-29 to 2026-01-13). The dashed vertical lines are the maintainer's switch times
(2025-12-30 01:00 and 2026-01-14 11:30). Observed power falls below both forecasts for most of the
load-shift, and the dashed NRA stays close to the bands. The two bands lie close together.

![D16, day-ahead forecasts through the December 2025 load-shift](real_pair_D16_day_ahead.svg)

The same for D16, whose power rises during the load-shift. **Which window matches the readings:**
the maintainer's switch times match the power readings better than the window of experiment 580,
which is about one day early at both ends. At D5 the 24-hour average power falls by 2.3 MW around
2025-12-30 00:00 and rises by 2.1 MW around 2026-01-14 11:30, matching the maintainer's times. At D16
the two steps (a rise of 1.7 MW and a fall of 1.8 MW) come about 7 hours later, around 08:30 on 30
December and 19:30 on 14 January. A 24-hour average cannot place a step to the hour, so the times
are approximate.

Day-ahead forecasts on 12 injected designs (7, 15, and 30 days; 15%, 30%, and 50% of capacity, up
and down), one Primary from each metering group per design, March 2026 (24 figures, each with C1 +
G4 above and A1 + G4 below). The target stays clean, so the observed power is also the NRA. The
window runs from 14 days before the shift to 14 days after its end. The stored forecasts stop at
120 hours, so no panel shows the full 14-day horizon.

![D3, injected 7 days, +15%](injected_7d_up_15pct_mva.svg)

D3 (eight MVA Primaries): +15% of capacity for 7 days.

![D16, injected 7 days, +15%](injected_7d_up_15pct_mw.svg)

D16 (six MW Primaries): +15% of capacity for 7 days.

![D8, injected 7 days, -30%](injected_7d_down_30pct_mva.svg)

D8 (eight MVA Primaries): -30% of capacity for 7 days.

![D9, injected 7 days, -30%](injected_7d_down_30pct_mw.svg)

D9 (six MW Primaries): -30% of capacity for 7 days.

![D12, injected 7 days, +50%](injected_7d_up_50pct_mva.svg)

D12 (eight MVA Primaries): +50% of capacity for 7 days.

![D14, injected 7 days, +50%](injected_7d_up_50pct_mw.svg)

D14 (six MW Primaries): +50% of capacity for 7 days.

![D12, injected 7 days, -50%](injected_7d_down_50pct_mva.svg)

D12 (eight MVA Primaries): -50% of capacity for 7 days.

![D14, injected 7 days, -50%](injected_7d_down_50pct_mw.svg)

D14 (six MW Primaries): -50% of capacity for 7 days.

![D4, injected 15 days, +15%](injected_15d_up_15pct_mva.svg)

D4 (eight MVA Primaries): +15% of capacity for 15 days.

![D16, injected 15 days, +15%](injected_15d_up_15pct_mw.svg)

D16 (six MW Primaries): +15% of capacity for 15 days.

![D8, injected 15 days, -30%](injected_15d_down_30pct_mva.svg)

D8 (eight MVA Primaries): -30% of capacity for 15 days.

![D10, injected 15 days, -30%](injected_15d_down_30pct_mw.svg)

D10 (six MW Primaries): -30% of capacity for 15 days.

![D12, injected 15 days, +50%](injected_15d_up_50pct_mva.svg)

D12 (eight MVA Primaries): +50% of capacity for 15 days.

![D14, injected 15 days, +50%](injected_15d_up_50pct_mw.svg)

D14 (six MW Primaries): +50% of capacity for 15 days.

![D12, injected 15 days, -50%](injected_15d_down_50pct_mva.svg)

D12 (eight MVA Primaries): -50% of capacity for 15 days.

![D14, injected 15 days, -50%](injected_15d_down_50pct_mw.svg)

D14 (six MW Primaries): -50% of capacity for 15 days.

![D22, injected 30 days, +15%](injected_30d_up_15pct_mva.svg)

D22 (eight MVA Primaries): +15% of capacity for 30 days.

![D5, injected 30 days, +15%](injected_30d_up_15pct_mw.svg)

D5 (six MW Primaries): +15% of capacity for 30 days.

![D8, injected 30 days, -30%](injected_30d_down_30pct_mva.svg)

D8 (eight MVA Primaries): -30% of capacity for 30 days.

![D11, injected 30 days, -30%](injected_30d_down_30pct_mw.svg)

D11 (six MW Primaries): -30% of capacity for 30 days.

![D12, injected 30 days, +50%](injected_30d_up_50pct_mva.svg)

D12 (eight MVA Primaries): +50% of capacity for 30 days.

![D16, injected 30 days, +50%](injected_30d_up_50pct_mw.svg)

D16 (six MW Primaries): +50% of capacity for 30 days.

![D12, injected 30 days, -50%](injected_30d_down_50pct_mva.svg)

D12 (eight MVA Primaries): -50% of capacity for 30 days.

![D16, injected 30 days, -50%](injected_30d_down_50pct_mw.svg)

D16 (six MW Primaries): -50% of capacity for 30 days.

![Event-free cost](false_alarm_cost.svg)

Extra pinball loss at q05 and q95 on event-free rows from the G4 band. All intervals contain 0 or
lie within the minimum detectable effect.

## Results

- **Row set and recipe.** The rows are `experiment_470_intersection_v1`: five fold months (2025-10,
  2025-12, 2026-01, 2026-03, 2026-06), 16 Primaries, forecasts 24 to 120 h ahead unless stated, July
  2026 unused, cleaned data, and D15 and D18 cut before 2025-03-01. Refits ran on days 0 and 14 of
  each month, so a contrast could differ at another cadence. Seeds 0 and 1 are averaged quantile by
  quantile, then sorted across the 11 levels. No fit ran and no weights were saved. Reused: the
  stored forecasts of C1 (experiment 584) and A1 (experiment 591) with their lags-masked twins and
  injected-bank forecasts, E2, and the injected bank of [experiment
581](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00581-robust-multiweek-lag-encoder-x3/result/).
- **Reproduction.** C1 + G4, day 4 to end, 15 and 30 day events: 0.9132 (0.9062 to 0.9201),
  experiment 592 0.9132, 756 Primary-events.
- **Headline, injected 15 and 30 day events, both metering groups, q05-q95 coverage (95%
  event-bootstrap interval), safe-side share, median follow ratio.**

| Forecast | Day 4 to end | 0-7 days after | Safe side, 0-7 days after | Follow, day 4 to end |
|---|---|---|---|---|
| C1 | 0.854 (0.844 to 0.863) | 0.860 (0.850 to 0.869) | 0.876 | 0.106 |
| C1 + G4 | 0.913 (0.906 to 0.920) | 0.923 (0.917 to 0.930) | 0.939 | 0.106 |
| A1 | 0.878 (0.869 to 0.886) | 0.885 (0.876 to 0.894) | 0.901 | 0.110 |
| A1 + G4 | 0.925 (0.918 to 0.931) | 0.928 (0.922 to 0.934) | 0.944 | 0.110 |

- **By metering group, day 4 to end.** C1 + G4: 0.923 at the eight MVA Primaries and 0.899 at the
  six MW Primaries. A1 + G4: 0.935 and 0.910. The safe-side share in the same phase is 0.940 and
  0.920 (C1 + G4) and 0.949 and 0.927 (A1 + G4).
- **Other phases, all event lengths (3, 7, 15, and 30 days).** Before the shift: C1 + G4 0.882, A1 +
  G4 0.885. Days 0 to 3: 0.872 and 0.883. These coverages are below 0.90, and the gate does not
change them,
  because it is off until it fires about 1 day after onset. In the 7 to 14 days after the end,
coverage is 0.890 and
  0.885, unchanged by the gate.
- **Tail pinball, day 4 to end, points of capacity.** q05 and q95 for C1: 0.73 and 0.90; for C1 +
  G4: 0.69 and 0.81. For A1: 0.72 and 0.88; for A1 + G4: 0.71 and 0.82. The 0.01 and 0.99 pinball
  loss and the NRA-side 99% share are in the printed report. The median's tail follow ratio on the
  safe side falls from 0.053 to -0.000 (C1) and from 0.036 to -0.012 (A1).
- **Median follow.** The median is the network's own, so the gate cannot lower following: 0.106 (C1)
  and 0.110 (A1) of the depth, which is a mean move of 2.1 to 2.3, 2.9 to 3.5, and 3.9 to 4.6 points
  of capacity at depths of 15%, 30%, and 50%. The worst single events move the median by 17 to 20
  points in the 50% events (histogram above).
- **Gate.** Detection rate 0.905 and a median onset delay of 1 day on the 840 injected 15 and 30 day
  Primary-events. The median release delay is 1 day after the end. False triggers: 1 in 4.97
  Primary-years on event-free days, 0.20 per Primary-year (one-sided 95% upper bound 0.95); the wide
  band is on for 1.2% of event-free origin days (the export marks 1.4% of event-free rows, which
includes the 7-day hold).
- **Event-free pinball loss and CRPS (CRPS, points of capacity).** The continuous ranked probability
  score (CRPS) is
  twice the mean pinball loss over the 11 quantile levels. At 24-120 h, eight MVA and six MW
  Primaries: C1 3.081 and 2.465; C1 + G4 3.084 and 2.466; A1 3.057 and 2.445; A1 + G4 3.061 and
  2.447. At 0-24 h: C1 2.876 and 2.265; C1 + G4 2.883 and 2.266; A1 2.843 and 2.218; A1 + G4 2.851
  and 2.219. The stored forecasts stop at 120 h, so no CRPS at 120-240 h exists. D15 and D18
  (report-only): C1 8.073, C1 + G4 8.076, A1 8.204, A1 + G4 8.209.
- **Counted contrasts and minimum detectable effect (MDE).** six counted tests at the Bonferroni
  level, month-block intervals over 5 months, so each is near the range of the five monthly
  differences. Pinball loss, both groups, 24-120 h, assembled minus plain network, in points of
  capacity: C1, q05 +0.0004 (0.0000 to 0.0011; MDE 0.0006) and q95 -0.0002 (-0.0010 to 0.0002; MDE
  0.0006); A1, q05 +0.0006 (0.0000 to 0.0015; MDE 0.0007) and q95 +0.0002 (0.0000 to 0.0005; MDE
  0.0003). All four contrasts are inside the minimum detectable effect (MDE), so P3 holds. A1 + G4
minus C1 + G4 CRPS: -0.025 (-0.079 to
  +0.021; MDE 0.050) at the eight MVA Primaries and -0.023 (-0.072 to +0.032; MDE 0.052) at the six
  MW Primaries: indistinguishable.
- **Real December pair (D5 and D16, 2 Primary-events, descriptive only), 24-120 h.** C1 + G4 equals
  experiment 592's G4 coverage (D5 0.868 and D16 0.686 from day 4 to the end), so P4 holds. Phases
follow experiment 592: the merged verified span of each Primary, with day 4 starting 3 days after
onset; a cut on the export's `in_verified_event` rows gives 0.89 and 0.66 for these two cells, which
flags the same phases. Coverage
  by phase, C1 + G4 then A1 + G4: D5 before 0.43 and 0.59, days 0 to 3 0.49 and 0.57, day 4 to end
  0.87 and 0.86, 0-7 days after 0.58 and 0.56. D16 before 0.95 and 0.92, days 0 to 3 0.71 and 0.77,
  day 4 to end 0.69 and 0.70, 0-7 days after 0.84 and 0.84. Phases below 0.80 are flagged: D5
  before, days 0-3, and after the end; D16 days 0-3 and day 4 to the end. [Experiment
589](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00589-p5-coverage-on-d5-before-the-december-load-shift/result/)
  traced the D5 miss before the shift to a level bias that every earlier forecast shares.
- **Export.** `data/ml-experiments/backtest_export_candidates_2026-11-06/` (private, not committed):
  `C1`, `C1_G4`, `A1`, and `A1_G4` for the five months, in the layout of experiment 593 (the earlier
  export, which has no result page) with a `gate_on` column in the `_G4` files. `checks.txt`
  reproduces C1's CRPS of experiment 584 (3.081 and 2.465), shows the C1 file equals experiment
  593's, and shows the median is unchanged by the gate.

## Checks

- **Median minus mean forecast.** Not available: the stored forecasts hold quantiles only. The
  median of each assembled forecast equals its network's median on every row (0 rows changed), so
  the median error is the same with and without the gate.
- **Mean-forecast error beside the quantile contrasts.** The gated and plain forecasts share the
  median, so the median's absolute error is identical; the contrasts above are in the band only.
- **Triggers.** None of the seven audit triggers fired: no control failed, the reproduction matched,
  the quantile and median contrasts agree in direction, and no nested arm beat its container.
- **Code review.** A fresh Sonnet agent found no bug that changes a headline number. It noted that
  the gate thresholds use the whole backtest's event-free days (the readings are causal, the
  thresholds are in-sample), that the G4 safe-side follow ratio mixes the network's movement and the
  band widening, and that the intervals resample events inside the same 5 months, so they are
  narrower than a month-block interval would be.

## Caveats

- **The injected bank gives the lag neural networks an easier test than a real load-shift would.**
  It changes the lag inputs only, not the power, so
  it says nothing about a load-shift that enters the training power. That test (target-side
  absorption) has not been run on C1 or A1.
- **One real pair.** Real-event numbers rest on two Primary-events. The network operator's (National
  Grid Electricity Distribution, NGED) labels are imperfect, and
  NGED says load-shifts may be missing from its logs, so clean rows and training periods can hold
  events no label covers.
- **Horizon.** The forecasts stop at 120 hours; NGED cares about 1 to 10 days.
- **The G4 false-trigger rate rests on 1 trigger in 5 Primary-years.**
- **Real events at D5.** C1 + G4 and A1 + G4 both miss the level before the shift and the band does
  not cover
  the NRA after it.

## What to try next

- Target-side absorption for A1 + G4 and C1 + G4 (a shift in the training power and the inputs, 12
  or more designs).
- A second real event: score both assembled forecasts on May 2026.
- A gate-free network that receives the gate's signals as inputs, judged at equal band width.
