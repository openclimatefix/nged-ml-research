---
id: "00581"
slug: robust-multiweek-lag-encoder-x3
title: >-
  Does reading five weeks of power lags through a median and a spread, instead of one day lag and
  one week lag, stop the forecast following a load-shift event, without losing the lag gain?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: finished
kind: change
protocol: demand-v2
size: >-
  complex. Stored: fitted weights and forecasts in the private data folder only. More than one
  defensible design (a fixed median and spread, a learned set encoder, synthetic contamination in
  fine-tuning). Spans code whose callers need a search (experiments 560, 565, 572, 576, and 578).
  Serving path and degradation rule: not touched (this repository holds none). Cost: ceiling 3
  GPU-h; projected under 1 GPU-h, because experiment 572's whole v3+D5 fit took 8 minutes.
hypothesis: >-
  A lag encoder that reads the median and the median absolute deviation of five weekly lags (and of
  three daily lags), plus each lag's robust z-score, keeps most of the gain from reading lags and
  stops the forecast following a load-shift event that contaminates only one or two weekly lags.
prediction: >-
  Written before any fit or any injected-event score (X3 of the NRA plan review, briefs/
  nra_plan_review_opus_answer.md). On injected lag-only events of 3, 7, and 15 days, the median
  follow ratio after the end is at most 0.05 for X3-med against at least 0.15 for v3+D5, and the
  tail follow ratio at q95 (or q05 when the event raised load) is at most 0.05. CRPS on all rows at
  24-120 h is within 0.05 points of v3+D5 at the eight MVA Primaries. The prediction fails if
  X3-med loses more than 0.08 points of CRPS against v3+D5 at the eight MVA Primaries, or if follow
  after the end stays above 0.10. Stop rule: if X3-med keeps under 50% of v3+D5's lag gain at the
  eight MVA Primaries, the X3-set and X3-syn arms are not run.
decision_rule: >-
  Counted tests: 12 (X3-med, X3-set, and X3-syn against v3+D5, times the eight MVA Primaries and
  the six MW Primaries, times the lead bands 0-24 h and 24-120 h). A contrast counts as a gain or a
  loss when the month-block bootstrap interval (5 blocks, near the minimum to maximum of the folds)
  at the Bonferroni level excludes zero; the two-seed minimum detectable effect is stated beside
  each contrast. X3-set runs only if X3-med keeps at least 80% of v3+D5's lag gain at the eight MVA
  Primaries. Threshold for porting: zero CRPS loss beyond the minimum detectable effect, because
  the change adds only inputs to an existing encoder, and the tail coverage after the event ends
  must rise (one-sided share on the NRA side at least 0.90 at q95).
benchmark: >-
  v3+D5 (the lag network with the slow output offset, experiment 572, stored weights and forecasts),
  its lags-masked twin, and the lag-free D5 network (experiment 565, stored).
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at 0-24 h and 24-120 h; on injected events, the follow ratio
  at q05, q50, q95, and q99, the coverage of the event-free power inside q05-q95 and q01-q99, the
  pinball loss at q01, q05, q95, and q99, and the width inflation, by phase.
verdict: partly supported
best_arm: >-
  v3+D5 fine-tuned with lag-only synthetic load-shift events on 5% of the time (post hoc variant of the
  planned X3-syn). Day 4 to the end of an injected event, the median forecast follows 0.104 of the
  depth against 0.196 for v3+D5 at the eight MVA Primaries.
headline: >-
  Replacing the week lag by the median of 5 weekly lags keeps the whole lag gain (CRPS 4.423 against
  4.427 at the eight MVA Primaries, 24-120 h) but trims following of an injected lag-only event only
  from 0.196 to 0.172 of its depth. Training with synthetic lag-only events halves following (0.104)
  and lifts the share of readings on the safe side of q95 or q05 from 0.77 to 0.89.
lessons: >-
  The day lag, not the week lag, carries most of the following during an event: following is the same
  for events of 7, 15, and 30 days, so a median over weekly lags cannot remove it. Showing the
  network contaminated lags during fine-tuning is what lowers following, and v3+D5 with the synthetic
  events does nearly as well as X3-med with them. After a 30-day event the 5-week median is followed
  more than the single week lag, as expected.
take_home_message: >-
  A median over five weeks of power lags does not stop the forecast following a load-shift event,
  because the day-old lag carries the event too, and this network follows the day lag most. Fine-tuning
  on lag-only synthetic events roughly halves the following and moves the safe-side coverage of q95
  and q05 from 0.77 to 0.89, at a small loss of CRPS. On the one verified 15-day transfer the
  coverage in the week after it ends is still only 0.5 to 0.6, so the tails stay below the 0.90 target
  there.
advice:
  - >-
    Train lag networks with lag-only synthetic events on about 5% of the fine-tune time, not with a
    robust summary of the weekly lags: following falls from 0.196 to 0.104 of the depth (Injected
    bank).
  - >-
    Judge a lag control on the injected bank and on days 4 to the end and 0 to 7 after the end, where
    following peaks, not on the whole event (Injected bank).
  - >-
    Do not expect a median over weekly lags to help against events shorter than 30 days, because the
    day lag carries them (follow by length, Injected bank).
  - >-
    Report the share of readings on the safe side of q95 or q05 with the follow ratio, because the
    follow ratio of 0.1 to 0.2 hides a fall in coverage from 0.94 to 0.77 (Injected bank).
  - >-
    Count verified transfers before designing a real-event test: only 1 transfer (2 Primaries) lies
    inside the five folds, so real-event tail numbers are descriptive (Verified events).
production_status: candidate
adopted_in: none
---
**A median over five weeks of power lags does not stop the forecast following a load-shift event,
but training on synthetic lag-only events does roughly halve it.** We asked whether the neural
network that reads a day-old and a week-old reading of a Primary's power can be made to ignore an
event that sits in a few of its lags. We built a variant that reads the median of five same-weekday
readings and their spread (X3-med). We then added a shift of 15%, 30%, or 50% of capacity to the lag
inputs of every Primary for 3 to 30 days, in 24 passes that give every Primary every depth and
length with both signs, and measured how far the day-ahead forecast moved. The real power, the
target, was never shifted, so the unshifted power is the reference (the normal running arrangement,
NRA). We also scored the one verified real transfer of the telemetry set of [experiment
580](../00580-verified-load-shift-event-set-and-label-audit/result.md) that lies inside the five
fold months, and the May 2026 transfers. All numbers are interim and unaudited.

**Verdict: partly supported.** X3-med passes its accuracy condition and misses its following target;
the synthetic-event variant, which is a post hoc change of the planned X3-syn, meets the target's
direction but not its size. In CRPS (points of capacity, 24-120 h) at the eight MVA Primaries,
lag-free D5 scores 4.651, v3+D5 4.427, and X3-med 4.423, so X3-med keeps 102% of the lag gain. At
the six MW Primaries v3+D5 scores 3.327 and X3-med 3.328. The median follow ratio (the share of the
injected depth that the median forecast moves towards) is 0.065 after the end against a predicted
0.05 or less, and 0.172 from day 4 to the end against 0.196 for v3+D5. The prediction that v3+D5
follows at least 0.15 after the end also fails: it follows 0.075.

![Following and tail coverage by phase](tails_by_phase.svg)

## Injected bank

Each row is a mean over events of 3, 7, and 15 days (480 events at the eight MVA Primaries and 360
at the six MW Primaries per phase from day 4 to the end), forecasts 24-120 h ahead, two-seed means.
"Safe side" is the share of NRA readings on the side of q95 (event raised load: q05) where the
shifted forecast is wrong. The nominal value is 0.95.

| Arm | Eight MVA: follow, day 4 to end | Eight MVA: follow, 0-7 d after | Eight MVA: safe side q95/q05, day 4 to end | Six MW: follow, day 4 to end | Six MW: safe side, day 4 to end | Six MW: safe side, 0-7 d after |
|---|---|---|---|---|---|---|
| v3+D5 | 0.196 | 0.075 | 0.770 | 0.176 | 0.735 | 0.841 |
| X3-med | 0.172 | 0.065 | 0.785 | 0.155 | 0.750 | 0.858 |
| X3-syn | 0.085 | 0.038 | 0.894 | 0.064 | 0.856 | 0.900 |
| v3+D5-syn | 0.104 | 0.040 | 0.885 | 0.077 | 0.842 | 0.890 |
| Placebo (weeks 6-10) | 0.163 | 0.041 | 0.792 | 0.134 | 0.757 | 0.876 |
| X3-3.5h | 0.196 | 0.055 | 0.760 | 0.180 | 0.744 | 0.875 |

- **The day lag carries the following.** From day 4 to the end, v3+D5 follows 0.181 (7-day events),
  0.194 (15-day), and 0.198 (30-day) of the depth. A 7-day event contaminates 1 or 2 of the 5 weekly
  lags, a 30-day event 4 or 5, and the following is the same, so the weekly lags are not the
  channel. X3-med follows more than v3+D5 for 30-day events (0.231 against 0.198 from day 4, 0.148
  against 0.082 in the week after the end), the design limit of a median over 5 weeks.
- **The follow ratio falls as the depth rises** (v3+D5: 0.209, 0.193, 0.172 at depths of 15%, 30%,
  and 50%), so the network does not follow a large shift in proportion.
- **The placebo** reads weeks 6 to 10 and keeps about half of the lag gain at the eight MVA
  Primaries (4.532), so recency matters there. At the six MW Primaries it scores 3.289, better than
  v3+D5 (3.327), so recency does not matter there.
- **X3-3.5h** (report only) lowers CRPS at the eight MVA Primaries (4.402 at 24-120 h, contrast at
  0-24 h of -0.089 against v3+D5, interval -0.154 to -0.023, minimum detectable effect 0.108) and
  follows as much as v3+D5.
- **CRPS cost of the synthetic events** (positive is worse), counted contrasts against v3+D5: at the
  eight MVA Primaries X3-syn +0.037 (0-24 h) and +0.032 (24-120 h); at the six MW Primaries +0.081
  and +0.071. All 8 counted intervals that were run include zero (Bonferroni count 12, of which the
  4 X3-set cells were not run). The same cost for v3+D5-syn is +0.012 to +0.058 and is descriptive.
  The two-seed minimum detectable effect is 0.06 to 0.15.
- **Event-free calibration is unchanged.** On rows with no labelled or verified event within 7 days
  (24-120 h), the share of readings at or below q95 is 0.919 (v3+D5) and 0.918 (X3-syn) at the eight
  MVA Primaries, and 0.897 and 0.894 at the six MW Primaries. Pinball loss at q05 and q95 is 0.007
  and 0.009 for all arms. Every arm's upper tail is too narrow, as the plan review found for v3+D5.

![Injected 30% shift for 15 days at one MW Primary](injected_event_d5_30pct_for_15_days.svg)

## Verified events

The telemetry set of experiment 580 holds 7 transfers from 2024-04-01. Demand frames start in July
2025, so only transfer 23 (D5 and D16, 29 December 2025 to 13 January 2026, 26% and 15% of capacity)
and the May 2026 transfers 24 to 26 can be scored. The numbers below are descriptive: 1 transfer is
far fewer than the 10 needed for a test. The NRA reference is the observed power minus the fitted
transfer.

| Arm | Transfer 23: safe side q95/q05, day 4 to end | Transfer 23: 0-7 d after | Transfer 23: q05-q95 coverage, 0-7 d after | Transfer 23: follow, day 4 to end |
|---|---|---|---|---|
| v3+D5 | 0.839 | 0.494 | 0.488 | 0.199 |
| X3-med | 0.864 | 0.516 | 0.504 | 0.169 |
| X3-syn | 0.903 | 0.593 | 0.587 | 0.104 |
| v3+D5-syn | 0.900 | 0.454 | 0.449 | 0.131 |

- **After the transfer ends, the tails miss the NRA about half the time**, for every arm (0.45 to
  0.59 against 0.90), including X3-syn. Part of that is an ordinary miss: before the event the
  q05-q95 coverage at D5 is 0.73 to 0.77.
- **The May 2026 transfers** at D10, D11, D12, and D16 lie in a fold where v3+D5 and X3-med cover
  the NRA on only 0.17 to 0.45 of rows before and during the events, so they say little about lags.
  In the June fold, 7 to 14 days after the ends, all arms cover 0.92 to 0.94 of the NRA.

![Verified transfer 23 at D16](verified_event_23_at_d16.svg)

![Verified transfer 23 at D5](verified_event_23_at_d5.svg)

## What changed from the plan

- **X3-set** was not run. The condition (X3-med keeps 80% of the lag gain) held, but X3-med did not
  move the following, so a learned set encoder on the same inputs had no stated reason to.
- **X3-syn** was built on X3-med, not on X3-set, and v3+D5 with the same synthetic events
  (v3+D5-syn) was added as a control. It shifts the lag inputs on 5% of each Primary's time, with
  random depths, lengths, and signs, seeded by the fold month, and leaves the targets clean. This is
  post hoc.
- **Fold months.** Only the five folds of experiment 572 were used. The extra months planned for the
  verified transfers of 2024 cannot be built, because the demand frames start in July 2025. The
  2026-05 fold reuses the v3+D5 weights of experiment 582.
- **The lag-free D5 reference** and the recalibrated control of experiment X2 were not built for the
  tail tables.
- **Statistics.** Intervals for CRPS are the 5-block month bootstrap at the Bonferroni level of 12
  tests; the injected-bank and real-event numbers carry no intervals. The unlogged-event caveat
  applies: NGED's logs may miss events, so the "no labelled or verified event" rows may hold events.

## Checks

- **Median forecast minus mean forecast per site and the error of the median beside each quantile
  contrast** were not computed; the contrasts above are quantile scores only.
- **Audit triggers.** Trigger 2 fired: v3+D5's following after the end (0.075) is below the 0.15 the
  prediction assumed, taken from experiment 578's labelled events. The explanation tested here is
  that the following is nearly independent of event length, so it is carried by the day lag, and
  that the labelled events of experiment 578 were in part mislabelled ([experiment
  580](../00580-verified-load-shift-event-set-and-label-audit/result.md)). No other trigger fired;
  leaving one Primary out was not run.
- **Leakage checks.** Each lag is at least 24 h (X3-3.5h: 3.5 h) before the forecast origin, each
  lag's weather run is at or before the lag time, and each forecast's weather run is at or before
  the origin (asserted row by row; the example rows print in the fit logs). One Sonnet code review
  before the first fit found no problems.

## What the next experiment should answer

Whether the synthetic-event fine-tune keeps its gain when events are also present in the training
targets, and whether a gate or a union band (experiments X2 and X4 of the plan review) lifts the
coverage after an event ends from about 0.5 to the 0.90 target, since the lag network alone does
not.

Scripts: `ml-experiments/00581-robust-multiweek-lag-encoder-x3/` (`e00581_arm.py` wraps experiment
572's fit; `e00581_report.py` prints every table above). Code reused from experiments 560, 565, 572,
578, and 580.
