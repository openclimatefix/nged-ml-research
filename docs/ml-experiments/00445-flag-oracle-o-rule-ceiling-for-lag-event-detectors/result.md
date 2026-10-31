---
id: "00445"
slug: flag-oracle-o-rule-ceiling-for-lag-event-detectors
title: >-
  If a detector knew exactly which recent power readings lie inside a load-shift event, how much
  would a lagged and lagless forecast mixture gain?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: done
kind: other
protocol: demand-v2
size: >-
  simple. Stored: one table of forecasts per rule, month, and world in this experiment's private
  folder. No fit, only forward passes of saved experiment 404 networks on one CPU core. One design
  (the lagged head is experiment 404's A0-inj network and the lagless head is its frozen cross-fitted
  table, the two heads of its B-mix arm). Production serving path: not touched (this repository
  holds none). Degradation rule: not touched. Callers: the scripts import experiment 404's modules.
hypothesis: >-
  A mixture of the lagged forecast and the lagless forecast whose weight is set by the exact record
  of which of the 3 lags lie inside a load-shift event (an oracle detector, the O-rule) scores
  lower than B-mix (a fixed 50/50 mixture of the same two forecasts) on rows whose time being
  forecast lies inside a real event.
prediction: >-
  Written before any score is read. CRPS in points of capacity, mass-weighted pinball loss, all
  horizons of experiment 404's rows, no widening beyond what the stored forecasts carry. P1. On the
  injected scoring worlds the hard and soft O-rule beat B-mix on injected harm rows at the
  signed-MW six (the flags are exact and the lagless head is known to be better there). P2. Inside
  real events at the signed-MW six the hard O-rule minus B-mix lies between -0.4 and +0.2, so the
  event-cluster interval includes zero. P3. On clean rows the hard O-rule equals the lagged head
  (A0-inj) exactly, so its contrast with B-mix there is A0-inj minus B-mix (+0.13 on real clean
  rows at the six in experiment 404's interim table).
decision_rule: >-
  Contrast = O-rule arm minus B-mix on the same rows, mean mass-weighted pinball loss, with a 95%
  event-cluster bootstrap interval (real events for real rows; world, Primary, and origin day for
  injected rows), with and without the largest event. Primary row set: rows whose target lies inside
  a padded real event ("inside real events"), signed-MW six, pooled over the five months
  2025-10, 2025-12, 2026-01, 2026-03, 2026-06 at fit seed 0 (2026-06 seeds 1 and 2 are a robustness
  check, not pooled). Reading rule. If the hard or the soft O-rule beats B-mix inside real events at
  the six, the interval excluding zero both with and without the largest event, detection is a
  bottleneck for the mixture rule: a better detector would help, and its ceiling is that contrast.
  If neither does, the limit is the lagged and lagless heads or the labelling, because the exact
  flags cannot make this pair of forecasts score better inside events. Labelling means rows inside
  events whose actuals are not normal running arrangement (NRA, the usual network configuration)
  and events missing from the logs. The smallest gain worth reporting is zero, because the O-rule
  is a diagnostic ceiling and is not ported. Interim; no 99% claim.
benchmark: >-
  B-mix, experiment 404's fixed 50/50 mixture of A0-inj and the lagless table. S1, B1, B3, and B6
  (experiment 404's stored final forecasts) are reference rows, where stored for the cell.
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of capacity from 11 quantiles, by row class,
  group, and world, as in experiment 404's report.
verdict: refuted
best_arm: >-
  None beat B-mix. The closest is O-floor (post hoc: weight 0.5 + share of flagged lags / 2 on the
  lagless head), +0.326 (-0.004 to +0.540) points against B-mix inside real events at the signed-MW
  six.
headline: >-
  Giving a mixture of the lagged and lagless forecasts the exact record of which lags lie inside a
  load-shift event did not make it score better than the fixed 50/50 mixture B-mix. Inside real
  events at the six Primaries metered in signed megawatts the hard rule scored 0.465 points of
  capacity worse than B-mix (95% interval 0.198 to 0.644, 20 events), and the soft rule 0.429 worse.
lessons: >-
  Detection is not the bottleneck for the mixture rule. The lagged head and the lagless head score
  alike even when all 3 lags are flagged, and a 50/50 average of the two beats either alone in every
  row set, so picking one head with a perfect flag throws that averaging gain away.
take_home_message: >-
  A perfect record of which recent power readings lie inside a load-shift event did not help:
  choosing the lagless forecast (or weighting it by the share of flagged readings) scored worse than
  simply averaging the lagged and lagless forecasts 50/50, inside real events (0.465 points of
  capacity worse, 95% interval 0.198 to 0.644) and on simulated events. The two forecasts are about
  equally good even when all three readings are flagged, and their average beats both, so a better
  event detector cannot lift this pair of forecasts. All numbers are interim and unaudited.
advice:
  - >-
    Do not build a lag-event detector to drive a hard or soft switch between the lagged and the
    lagless forecast, because with exact flags the switch scores 0.465 (0.198 to 0.644) points worse
    than the fixed 50/50 mixture inside real events at the signed-MW six (O-hard, Figure 1).
  - >-
    Keep a fixed 50/50 mixture of the lagged and lagless forecasts as the reference for any rule
    that reweights the two, because it beat both forecasts alone in every row set (Figure 2).
  - >-
    Test whether hiding the flagged lags from the lagged network (experiment O-mask) improves the
    lagged head itself, because the heads, not the detector, limit this rule (Figure 2).
production_status: not-for-production
adopted_in: none
code: ml-experiments/00445-flag-oracle-o-rule-ceiling-for-lag-event-detectors/
---

**Take-home message: telling the forecast exactly which of its three recent power readings lie
inside a load-shift event does not beat simply averaging the lagged and lagless forecasts 50/50, so
the limit is the two forecasts and not the detector. The hypothesis is refuted.**

We asked how much a perfect event detector could gain. The lagged forecast sees recent power
readings (lags) and the lagless forecast does not. B-mix averages the two with equal weight. We
replaced the equal weight with rules that use an exact record of which of the 3 lags lie inside an
event: the hard rule uses the lagless forecast whenever any lag is flagged, and the soft rule
weights the lagless forecast by the share of flagged lags. Neither rule beat B-mix, in real events
or in simulated ones. The reason is in Figure 2: even with all 3 lags flagged, the lagless forecast
is barely better than the lagged forecast on simulated events, and worse inside real events, while
the average of the two beats both. This rules out a better detector as the way to improve this pair
of forecasts. It narrows the search to making the lagged forecast itself ignore the flagged lags,
which is experiment O-mask.

**How to read the numbers.** Every number is interim and unaudited. The score is the
continuous ranked probability score (CRPS) in points of capacity, from 11 quantiles (smaller is
better), pooled over five test months (2025-10, 2025-12, 2026-01, 2026-03, 2026-06) at fit seed 0.
A contrast is the arm minus B-mix on the same rows, so a positive number favours B-mix. Intervals
resample whole events (95%). "Inside real events" are rows whose valid time lies inside a padded
real event or in the 96 half-hours after it. "Real harm" rows lie outside every event but have a
day window of recent power that touches one. The lagged head is the A0-inj network of experiment 404
and the lagless head is its frozen cross-fitted forecast, so B-mix is exactly experiment 404's arm.
The signed-MW six are the Primaries metered in signed megawatts, the MVA-only eight the others.

![Figure 1: arm minus B-mix](figure_1_contrasts.svg)

![Figure 2: score by number of flagged lags](figure_2_by_count.svg)

![Figure 3: day-ahead example](figure_3_example_day_ahead.svg)

![Figure 4: 14-day example](figure_4_example_14_day.svg)

![Figure 5: six real events](figure_5_real_events.svg)

## Headline numbers

- **Arms.** O-hard, O-soft, O-lagless (the lagless head alone), and O-floor (post hoc). O-floor puts
  weight 0.5 + (flagged lags) / 6 on the lagless head, so it equals B-mix when no lag is flagged. The
  stored experiment 404 arms A0-inj, B1, and B3 are reference rows. B6 and S1 are not stored for
  all five months, so they are not scored.
- **Inside real events, signed-MW six, minus B-mix (20 events):** O-hard +0.465 (+0.198 to +0.644),
  and +0.374 (+0.154 to +0.592) without the largest event; O-soft +0.429 (+0.180 to +0.607);
  O-lagless +0.414 (+0.065 to +0.630); O-floor +0.326 (-0.004 to +0.540); A0-inj +0.155 (-0.025 to
  +0.448). B-mix scores 7.855, O-hard 8.320.
- **Inside real events, MVA-only eight (18 events):** O-hard +0.677 (+0.291 to +1.333), O-soft
  +0.423 (+0.133 to +0.815), O-floor +0.323 (+0.048 to +0.779).
- **Real harm rows, six:** O-hard +0.255 (+0.151 to +0.371), O-soft +0.197 (+0.093 to +0.320),
  O-floor +0.073 (+0.015 to +0.159). B-mix scores 3.351.
- **Injected harm rows, six:** O-hard +0.292 (+0.261 to +0.325), O-soft +0.238 (+0.215 to +0.263),
  O-floor +0.029 (+0.010 to +0.050). B-mix scores 3.987.
- **Clean rows:** O-hard equals A0-inj, which is 0.390 worse than B-mix on real clean rows at the
  six (3.959 against 3.569). O-floor equals B-mix exactly on clean rows (control passed).
- **Number of flagged lags (six, pooled):** on injected harm rows with 3 flagged lags the lagged
  head scores 4.10, the lagless head 4.03, and B-mix 3.49. Inside real events with 3 flagged lags
  they score 9.48, 10.19, and 9.51.
- **Every cell.** O-hard minus B-mix inside real events is positive in all 7 cells (2026-06 seeds 0
  to 2 and the four other months): +0.21 to +0.70 at the six.
- **Tails, six, inside real events** (pinball loss; share of actuals below the quantile): B-mix q01
  0.600 (7.9%), q99 2.307 (71.8% below); O-hard q01 0.806 (9.0%), q99 2.794 (69.1%); O-floor q01
  0.807, q99 2.694. The upper tail is where flags hurt most: O-hard's q99 loss is 0.487 above
  B-mix's.
- **Event counts.** `events_corrected.parquet` holds 184 corrected events: 78 at the six, 88 at the
  eight, and 18 at the report-only D15 and D18. Neighbouring events on one Primary merge into 170
  padded runs. The five scored months overlap 49 corrected events (2, 4, 15, 10, and 18; an event
  straddling a month boundary counts in both). The earlier brief's 112 matches none of these counts.
- **Cadence and data.** No fit: forward passes of experiment 404's saved A0-inj networks (monthly
  refits, cross-fitted lagless tables) on one CPU core. Cleaned power with the D15 and D18 cut as
  in experiment 404. Copies of experiment 404's modules at PR 58 commit `1ddeab7e`
  (`e00445_*.py`, code-identical after renaming). A0-inj and B-mix are unaffected by the later
  commits of PR 58.
- **Stored forecasts.** `data/ml-experiments/00445-flag-oracle-o-rule-ceiling-for-lag-event-detectors/work/predictions/<arm>/`.

## Checks

- **Median minus mean forecast (points of capacity, per Primary, clean world).** For O-hard it
  ranges from -2.33 (D15) to -0.03 (D22), mostly small and negative, as for the lagless head alone
  (-1.75 at D12 to +0.29 at D18). No Primary has a gap of more than 2.6 points (O-soft at D15).
- **Error of the median beside each contrast.** The mean absolute error of q50 inside real events at
  the six is 9.81 for B-mix, 10.06 for A0-inj, 10.06 for O-lagless, 10.19 for O-hard, 10.17 for
  O-soft, and 9.99 for O-floor, so the median agrees in direction with every quantile-score
  contrast (trigger 3 did not fire).
- **Triggers.** Trigger 2 fired: the result contradicts predictions P1 (O-rules beat B-mix on
  injected harm rows: they lose by 0.29 and 0.24) and P2 (hard minus B-mix between -0.4 and +0.2
  inside real events: it is +0.465). The explanation, tested in this experiment, is that the 50/50
  average gains from averaging two forecasts that score alike (Figure 2) and that O-floor, which
  keeps the average, removes most of the loss on injected rows (+0.029). Trigger 4 did not fire:
  removing the largest event leaves every inside-event contrast positive. Triggers 1, 5, 6, and 7
  did not fire (the control O-floor equals B-mix on clean rows). P3 held: O-hard equals A0-inj on
  clean rows.

## Caveats

- **Interim.** One fit seed per month, five months, 20 events at the six. 2026-06 seeds 1 and 2 give
  the same sign (+0.34 and +0.28 for O-hard inside real events), scored separately.
- **Unlogged events.** NGED says load-shift events may be missing from their logs. Clean rows can
  therefore hold unlogged events, a harm table measured on clean rows is optimistic, and real-event
  slices undercount.
- **Flags are exact only on the simulated worlds.** On real data the flag is the padded
  hand-labelled event, so labels that miss an event, or rows inside an event whose actuals are not
  the normal running arrangement, would make the real-event result worse than the oracle name
  suggests.
- **Avoidable-curtailment cost not computed.** The measure needs node and generator forecasts that
  experiment 404 does not store.
- **O-floor is post hoc.** Its weights were chosen after the first scores. Its rows, rule, and
  intervals were not tuned on the scored rows.

## What to try next

- **Run O-mask.** Train S1 and score it with each flagged lag replaced by "missing". If the lagged
  head improves inside events once it cannot see the flagged lags, the gain comes from the head and
  not from a switch.
- **Make the lagless head better inside events.** Inside real events with 3 flagged lags the
  lagless head is 0.71 points worse than the lagged head, so the lagless forecast also misses the
  events.

## Reuse and credit

The lagged head, lagless table, row classes, scoring, and event-cluster bootstrap come from
experiment 404 (PR 58, copied as `e00445_*.py`). The row classes and flag construction follow
experiment 428, and the flag oracle idea follows experiment 402.
