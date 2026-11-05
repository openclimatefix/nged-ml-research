---
id: "00583"
slug: after-snap-back-event-band-on-x3-syn
title: >-
  Does keeping a wider forecast band for 7 days after a load-shift ends make the forecast bands hold
  the normal running arrangement again, at a small loss on days with no load-shift?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: finished
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
verdict: not supported
best_arm: >-
  No rule beats chance. On injected 15 and 30 day lag-only events (672 Primary-events, 24-120 h, 0-7
  days after the end) the 7-day band after the snap-back (A1) holds the normal running arrangement
  inside q05-q95 on 0.848 of rows, against 0.826 for X3-syn and 0.847 for the same rule triggered on
  random days. Keeping the union band on while the gate is on (A2) reaches 0.915, close to the 0.921
  of a union band that is always on.
headline: >-
  A band widened for 7 days after the gap collapses covers the normal running arrangement no better
  than a band widened for 7 random days (0.848 against 0.847 after the end, bank). The only arms that
  reach 0.90 are the ones that are on almost all the time after the end (A2 0.915, always-on union
  0.921). On the verified D5 and D16 pair, A2 lifts the share after the end from 0.68 to 0.82 (D16) and
  from 0.45 to 0.55 (D5); the 0.90 target is not met. On days with no load-shift the rule costs under
  0.001 points of pinball loss.
lessons: >-
  The snap-back is seen 2 days after the end at best, and the lag contamination is gone by then, so a
  band held from that day widens forecasts that no longer need it. X3-syn's coverage after the end is
  flat in the origin day (0.82 to 0.83 on days +1 to +4), which a better-timed rule cannot improve
  much. Whether the gain of A2 and of the union band is more than extra width is untested.
take_home_message: >-
  Widening the forecast band for 7 days after a load-shift ends does not work better than widening it
  on random days, because the lag network is already back to normal by the time the rule notices the
  end. Only a band that is wide whenever the gate has been on lifts coverage after the end to about
  0.92 on injected events, and on the real December pair to 0.82 and 0.55, below the 0.90 target, at
  no measurable cost on days without a load-shift. The next step is a width-matched control, to test
  whether those gains are more than extra width.
advice:
  - >-
    Do not widen the band only for a fixed 7 days after the snap-back: the rule gains nothing over
    random placement because the lags are clean by the time the gap collapses (A1 against A4, bank).
  - >-
    Compare any event-mode band with a width-matched control (the lag network scaled to the same mean
    width) before reading a coverage gain as detection (A2 and Union, bank).
  - >-
    Report the real D5 and D16 pair as one event pair, and check a gate's state before the event
    starts, because the D5 gate was already on 19 days earlier (verified events).
production_status: not-for-production
---

**Widening the band for 7 days after the gap collapses covers the normal running arrangement (NRA) no
better than widening it for 7 random days.** The rule is seen too late: the lag network is already
back to normal when the gap collapses, so the wider band lands on forecasts that no longer need it.
Only the arm that stays wide whenever the gate has been on (A2) lifts coverage after the end to about
0.92 on injected events, and A2 is almost the same as a union band that is always on. Verdict: not
supported.

Experiment 582 found that after a load-shift ends, the forecast's q05-q95 band holds the NRA (the
power the substation would have drawn without the load-shift) on too few rows. We asked whether a rule
that watches the gap between the recent power readings and a forecast made without them, and widens
the band for 7 days after that gap collapses, closes the shortfall. We re-scored the stored X3-syn
forecasts of [experiment 581](../00581-robust-multiweek-lag-encoder-x3/result.md) on 1,248 injected
events and on the verified December load-shift, and compared the rule with the same rule fired on
random days. The rule is a version of experiment 578's gate on the lags and experiment 582's union
band; nothing is refitted.

## Figures

![Share of the NRA inside q05-q95 by phase, injected events of 15 and 30 days, both groups](coverage90_15_and_30_days_both_groups.svg)

![Share of the NRA on the safe side of the band, injected events of 15 and 30 days, both groups](nra_side95_15_and_30_days_both_groups.svg)

![Coverage by phase on the verified D5 and D16 events](coverage90_verified_events.svg)

![False-alarm cost: pinball loss against X3-syn on days with no load-shift](false_alarm_cost.svg)

![Share of days in event mode on days with no load-shift](share_days_in_event_mode.svg)

Day-ahead forecasts around the verified pair (coloured strips mark origin days in event mode):

![D5, verified load-shift](verified_D5_2025-12-29.svg)

![D16, verified load-shift](verified_D16_2025-12-29.svg)

Day-ahead forecasts around injected events (a fixed selection rule: month 2025-10, the
lowest-numbered Primary of each group for three length, depth, and sign combinations):

![Eight MVA Primaries, 15 days up 50%](injected_mva-only_15d_up_50pct.svg)

![Eight MVA Primaries, 30 days down 30%](injected_mva-only_30d_down_30pct.svg)

![Eight MVA Primaries, 7 days down 50%](injected_mva-only_7d_down_50pct.svg)

![Six MW Primaries, 15 days up 50%](injected_signed-mw_15d_up_50pct.svg)

![Six MW Primaries, 30 days down 30%](injected_signed-mw_30d_down_30pct.svg)

![Six MW Primaries, 7 days down 50%](injected_signed-mw_7d_down_50pct.svg)

Full stored horizon of one run (X3-syn stores 0 to 120 h only, so no 14-day horizon exists here):

![Winter run](horizon_2025-12-30.svg)

![Summer run](horizon_2026-06-10.svg)

Other cuts (all lengths, each group alone) sit beside this page as `coverage90_*` and `nra_side95_*`.

## Results

- **Arms:** V is X3-syn with lags; M is the same weights with lags masked; Union is the widest of both
  bands on every row (experiment 582's control); A1 is the union band for 7 days from the origin day
  a snap-back is seen; A2 is the union band while the gate is on plus those 7 days; A3 is A1 with the
  lags-masked median; A4 is A1 at random days (20 draws, same count per Primary).
- **Gate:** own gap only (experiment 578's G1), 3 sustained days above max(3 standard deviations, 5%
  of capacity), reading only days at least 24 h old (newest day read: origin day minus 2). Snap-back:
  the newest gap day falls below half the threshold. Row set: the five fold months with stored X3-syn
  forecasts; 24-120 h unless stated; July 2026 unused. Retraining: none; the weights are
  experiment 581's.
- **Injected bank, 24-120 h, 0-7 days after the end, 15 and 30 day events (672 Primary-events, both
  groups), q05-q95 coverage with 95% event-bootstrap interval:**

| Arm | Coverage after end | Days 4 to end |
|---|---|---|
| V (X3-syn) | 0.826 (0.816 to 0.837) | 0.844 |
| M (lags masked) | 0.876 | 0.882 |
| Union (always on) | 0.921 (0.915 to 0.927) | 0.921 |
| A1 | 0.848 (0.839 to 0.857) | 0.844 |
| A2 | 0.915 (0.909 to 0.922) | 0.880 |
| A3 | 0.848 | 0.844 |
| A4 (placebo) | 0.847 (0.837 to 0.856) | 0.856 |

- **A1 against the placebo:** A1 minus A4 is +0.0014 (interval -0.0002 to +0.0031, audit). The hold
  covers 25% of after-end rows, and a random hold covers 21%.
- **Per group and length:** the same pattern holds at the eight MVA Primaries and the six MW Primaries
  and at 7, 15, and 30 days (A1 within 0.004 of A4 except the six MW Primaries at 15 days, 0.871
  against 0.867).
- **Tails (24-120 h, injected 15 and 30 days, after the end):** NRA-side share at q05 or q95 is 0.848
  (V), 0.869 (A1), 0.936 (A2), 0.941 (Union), against 0.95 nominal. Pinball at q01 and q99 and the
  coverage at q01-q99 are in `bank_summary.csv` (private data folder).
- **Verified D5 and D16 pair, after the end (descriptive; one transfer pair, not two independent
  events):**

| Event | V | A1 | A2 | Union |
|---|---|---|---|---|
| D16 | 0.677 | 0.703 | 0.822 | 0.822 |
| D5 | 0.447 | 0.447 | 0.552 | 0.552 |

- **D5 is not a detection result:** its gate turned on 19 days before the event, on a gap of the
  opposite sign, and did not release until 14 days after the end. A2 equals Union there because the
  gate was on, and A1 equals V because no snap-back fell inside the window.
- **Timing on the bank:** the gate triggered on 83% to 88% of events, a median 4 to 6 days after the
  onset, and the snap-back came a median 2 days after the end (87% of events). The gate does not fire
  at Primaries with a high threshold (largest: 0.61 of capacity at one MVA Primary).
- **Cost on days with no load-shift (24-120 h, mean pinball in points of capacity, A1 and A2 against
  V):** at the eight MVA Primaries, q05-q95 band -0.0003 and +0.0002, q01-q99 band 0.0000 and
  -0.0001; at the six MW Primaries exactly 0. The half-width of the Bonferroni interval (the minimum
  detectable effect here) is under 0.001 for A1 and A2. For comparison, the always-on union changes
  the same loss by -0.007 to +0.005 and widens the 90% band by 9% (eight) and 8% (six).
- **Activity and false triggers:** the gate is on for 1.1% of event-free origin days at the eight MVA
  Primaries and 0% at the six MW Primaries; there were 2 false triggers, 0.87 per Primary-year at the
  eight MVA Primaries and 0.47 over all 16 Primaries (prediction: at most 1; fails above 3).
- **Predictions:** P1 holds numerically (A1 0.848, and V already covers 0.844 from day 4 to the end)
  but not because of the rule. P2 holds (cost under 0.001 points, under 1 false trigger per
  Primary-year). P3 fails: the placebo's lift (+0.021) is 94% of A1's (+0.022), not under a quarter.
  Bonferroni count: 16 (4 arms, 2 groups, 2 coverage bands, event-free rows).

## Checks

- **Median minus mean forecast, per site:** only 11 quantiles are stored, so the mean forecast does
  not exist. As a proxy, X3-syn's median minus the mean of its 11 quantiles is -0.008 of capacity on
  average (standard deviation 0.026), pooled over sites.
- **Mean-forecast error beside the quantile contrasts:** not available, for the same reason; the
  median's absolute error is in `event_free.csv` (`ae50`) and does not change for A1, A2, and A3
  except through A3's median swap on held days.
- **Triggers fired:** trigger 1 (a placebo control matched the rule). A fresh Sonnet audit
  re-derived every headline number from the stored forecasts with its own code, and all reproduced.
  It showed the placebo equals the rule because the injected contamination is gone before the
  snap-back is seen. Triaged: accepted, and it is the finding of this page.
- **Code review (before the first run):** a fresh Sonnet review found no leakage and six issues; four
  were fixed (event-free rows and activity shares polluted by holds after real events, false-trigger
  rate understated by month-start days that cannot trigger, misleading trigger-time rows, and a
  point estimate that did not match its interval). Two were kept as caveats: the design limit that a
  rule keyed on origin day covers about half of after-end rows (valid-time phases), and the gate
  threshold being computed on all five months.

## Caveats and what to try next

- **Width is untested:** Union and A2 may gain coverage only because they are wider. A control that
  scales X3-syn's band to the same mean width is the next test (not run).
- **Only 2 verified events,** one transfer pair. The injected bank uses lag-only shifts with a
  constant depth, with the observed power unchanged, so the gate sees the injected lags and not a
  change in the power.
- **Unlogged events:** NGED's logs may miss events, so the event-free rows may hold events no label
  covers, the cost on them is optimistic, and the real-event slices undercount.
- **Horizon:** X3-syn stores 0 to 120 h. NGED cares about 1 to 10 days, so these results cover the
  short end only.
- **Next:** a rule fired by the lags themselves (a robust z-score of the newest day lag) would act
  at the snap-back itself rather than 2 days later; the gate cannot see the end earlier than the
  24 h age rule allows.
- **Reuse:** code in `ml-experiments/00583-after-snap-back-event-band-on-x3-syn/`, importing
  experiments 578, 581, and 582; no fitted models, no new forecasts; tables in
  `data/ml-experiments/00583-after-snap-back-event-band-on-x3-syn/`.
