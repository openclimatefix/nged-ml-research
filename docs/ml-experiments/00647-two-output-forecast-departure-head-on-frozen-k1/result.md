---
id: "00647"
slug: two-output-forecast-departure-head-on-frozen-k1
title: >-
  Can a small departure head on the frozen K1 network forecast the arrangement now in force better
  than K1 shifted by its own recent error?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Medium. (1) The experiment stores head weights (a few thousand parameters each), one scored table
  per cut, and no new K1 weights; (2) more than one design would defensibly test the hypothesis
  (a head with a survival mixture, a free quantile head, a no-fit shift), so it runs all three side
  by side; (3) it reuses the injection bank of experiment 581, the gap signals of experiment 606,
  the lag-free table of experiment 598, the gate of experiment 592, and the detector of experiment
  637, and the callers of each are named in the plan; (4) it touches no serving path and no
  degradation rule; (5) it changes no contract.
hypothesis: >-
  The arrangement now in force (the prevailing forecast) can be forecast better by adding a
  separate departure head to the normal-running-arrangement forecast of K1 (frozen, gradient
  stopped) than by shifting K1's quantiles by its own recent error with a decaying weight (R0),
  without changing the normal-running-arrangement output.
prediction: >-
  Written before any head is fitted. P2 (injected bank, valid times inside the shift): the head with
  a survival mixture (H1) scores a CRPS at least 15% below K1 and below R0 by more than the
  minimum detectable effect. P4 (flagged days and the verified D5 and D16 pair, against observed
  power): H1's CRPS is below R0's by more than the minimum detectable effect, and its q05-q95
  coverage is at least 0.75 against R0's 0.17 to 0.25. P1 (event-free rows): H1 costs no more than
  0.03 points of capacity in any cell. The noise placebo (H1 with its signal inputs replaced by
  noise of the same variance) scores within the minimum detectable effect of K1 on P2.
decision_rule: >-
  The experiment follows the design answer's scorecard in
  `briefs/two_output_network_opus_design_answer.md`. N1 and N2 (the normal-running-arrangement
  output) hold by construction, because no head code path writes to the NRA rows, and the scoring
  script asserts that the NRA columns are unchanged. H1 or H2 counts as beating R0 on a bar when
  the month-block 95% interval of the CRPS difference excludes zero in the predicted direction and
  the point estimate exceeds the minimum detectable effect. The minimum detectable effect is a
  fixed 0.05 points of capacity (CRPS), chosen before any head row is scored, so that the bar does
  not move with the baseline. The first month pair (2025-10 and 2025-12) is a feasibility gate and
  not a test, because two month blocks cannot support a 95% interval. The experiment stops, and
  the prevailing output is R0 itself, when neither H1 nor H2 has a point estimate below R0's on
  both P2 and P4 on that pair. Claims need the pooled month-block interval over all five months.
  The smallest gain worth porting is set by the added pipeline step (a second model whose inputs
  are K1's outputs and the lag signals): 15% on P2 and the interval bars above, because R0 already
  gives 10% to 19% on the verified pair with no fit. The 0.03-point P1 and P3 limits are the
  design answer's.
benchmark: >-
  K1 (experiments 612 and 613, stored export), R0 (K1's quantiles shifted by K1's own mean residual
  over valid days 2 to 8 before the origin, times exp(-lead/tau), tau = 72 h and 168 h).
row_set_id: dev_months_origins_2025_10_to_2026_06_export_roster
metric: >-
  CRPS (twice the mean pinball loss over the 11 stored quantile levels) in points of capacity,
  q05-q95 coverage, and q01, q05, q95, and q99 pinball loss, by phase of the shift and lead band.
production_status: not-for-production
---

## Status

**Plan only: no head has been fitted and no score read.** The page is rewritten as the write-up
after the run.

## Question and design

NGED want two forecasts from one demand network. The normal running arrangement (NRA) forecast is
the power a Primary (primary substation) would draw with no load shift, and K1 already gives it.
The prevailing forecast follows the arrangement in force now. The
design answer in the private `data/ml-experiments/briefs/` folder
for the two outputs proposes that the prevailing output equals the NRA output plus a departure term
from a separate small head, and not a second quantile head on a shared trunk. This experiment is
that design's first test, on K1's stored forecasts. It follows
[experiment 590](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00590-synthetic-load-shifts-in-the-training-targets/result/),
which found that training one network on shifted targets doubles how far its median follows a shift.

**The NRA output is K1, frozen and unchanged.** On real rows it is the stored export of
experiments 612 and 613 (the mean of five seeds, with the tail constants and the G4 gated band). On
the injected bank the stored files are raw per-seed forecasts, so the script rebuilds K1 from them
with experiment 613's scoring path (seed mean, tail constants, G4 state) and first asserts that the
rebuild reproduces experiment 613's K1 score tables. Tail constants for 2025-10 are borrowed from
later months, as in experiment 613. The head code has no path that writes to the NRA rows, and the
scoring script asserts the NRA columns are identical before and after the head runs. N1 and N2
are therefore reported, not tested.

## Arms

- **R0 (no fit).** K1's 11 quantiles shifted by the mean of observed power minus K1's day-ahead
  median over valid days 2 to 8 before the origin, times exp(-lead / tau), for tau of 72 h and
  168 h. The experiment reproduces the design answer's R0 numbers with its own code and reports
  them.
- **H1 (departure head with a survival mixture).** A small neural network (two hidden layers of 64
  units) outputs a signed depth *d* and a depth uncertainty. The prevailing quantiles are those of
  a mixture of K1's distribution (weight 1 - p) and K1's distribution shifted by *d* (weight p),
  where *p* is the survival probability that the departure still holds *h* hours later, given its
  age. The duration prior is a stated choice: an equal mixture of a uniform 3 to 14 day duration
  and a uniform 28 to 112 day duration. A sensitivity run uses a short-heavy (80% short) and a
  long-heavy (80% long) mixture.
- **H2 (free quantile head).** The same inputs and loss, with 11 offsets added to K1's quantiles
  and sorted. It tests whether the mixture structure helps.
- **H1-p (noise placebo).** H1 with every signal input replaced by Gaussian noise that matches the
  column's mean and variance. It must score close to K1.

**Head inputs.** The gap between each power lag and the lag-free forecast, its exponentially
weighted averages (half-lives of 1, 3, and 7 days), and the neighbour summaries (experiment 606's
`gap`, `ew`, and `nb` groups, which cut the median's follow from 0.119 to 0.035 to 0.065); the G4
gate state; the age of the freshest lag; and the age of the current departure. The head has its own
inputs and no shared encoder with K1, so its loss cannot flow into K1.

## Training and scoring rows

- **Prevailing target.** On an injected row, observed power plus the synthetic shift for valid
  times before the synthetic end; elsewhere, observed (cleaned) power. The injection bank is
  experiment 581's: 24 passes per month, depths of 0.15, 0.30, and 0.50 of capacity, lengths of 3,
  7, 15, and 30 days, ending on day 14, in both signs. The stored forecasts on that bank are
  experiment 613's `inject` folders, so no K1 inference is needed. The lags carry the shift; the
  stored forecasts already reflect it.
- **No tuning on the scored month.** The head is fitted leave-one-month-out: its training months
  are the other development months and its scored month is held out. The first fold pair is
  2025-10 and 2025-12 with head seeds 0 and 1. The survivor then runs on all five months with
  head seeds 0 to 4. The first pair's hyperparameters (learning rate, weight decay, epochs) are
  fixed in the plan below, and are not tuned on any scored month.
- **Hyperparameters.** Adam, learning rate 0.003, weight decay 0.0001, batch 4096, 30 epochs,
  early stop by validation loss on a held-out training month, and the validation split grouped by
  month. This is the whole search.
- **Compute.** The fits are a few thousand weights on stored forecasts and run on CPU (about 8 of
  the slot's cores). The GPU ceiling of 2 GPU-hours is not expected to be used. K1 inference is
  needed only if the lag-free table has missing days before a month's first origin; the plan then
  times one fold first.
- **Cadence.** No refit cadence applies: K1's forecasts are stored as made (monthly refit of
  experiment 612 and 613's E3 rule), and the head is fitted once per held-out month.

## Scorecard (from the design answer; pre-registered)

| Id | Output | Rows | Measure | Bar |
|---|---|---|---|---|
| N1 | NRA | injected 15 and 30 day bank | q05-q95 coverage day 4 to end; follow; safe-side share | within 0.01 of K1 (0.927, about 0.11, 0.944) |
| N2 | NRA | event-free | CRPS | within 0.0006 of capacity of K1 |
| P1 | Prevailing | event-free, not flagged | CRPS by group and lead band (day 1, days 2 to 5, days 6 to 10) | no worse than K1 by more than 0.03 points in any cell |
| P2 | Prevailing | injected bank, during the shift | CRPS against the shifted target | at least 15% below K1, and below R0 by more than the MDE |
| P3 | Prevailing | injected bank, after a synthetic end inside the horizon | CRPS against the NRA | no worse than K1 by more than 0.03 points |
| P4 | Prevailing | experiment 637 detector flags on development days, and the verified D5 and D16 pair | CRPS against observed; coverage | below R0 by more than the MDE; coverage at least 0.75 |
| P5 | Prevailing | event-free | q05-q95 coverage | 0.88 to 0.92 |

**Flagged-day caveat.** Experiment 637's detector selects days on which observed power leaves the
manual heuristic, so P4 favours any forecast that follows a shift. P2 and P3 are the unbiased
tests, and P4 is the real-data check. The NRA is never drawn as truth in a figure.

**Reported with every score.** Tail calibration (coverage and pinball loss at q01, q05, q95, q99)
and the avoidable-curtailment cost in three row sets (inside each hand-labelled event, the 7 days
after, and all rows) per group and per Primary; the unlogged-event caveat; the median minus the
mean forecast per site; and the CRPS contrast beside the same contrast on the median's error.

## Inputs, controls, and limits

- **Input sources.** Injected lags come from `e00581_bank.contaminate`; the lag-free forecasts
  from experiment 598's table; the signals from experiment 606's code. The first step lists, per
  input, its source file and its first and last available date. Rows where R0 or a lag input
  cannot be formed (a 30 day shift starting in the previous month, or a history shorter than the
  7 day half-life needs) are dropped from every arm equally, and the counts are reported. If the
  lag-free table is missing days before a month's first origin, K1 inference fills them, with one
  fold timed first and the 2 GPU-hour ceiling kept.
- **R0 specification.** The first step writes R0 as code and asserts that, on the stored
  non-injected export, it reproduces the design answer's table (event-free all-16 Primaries 3.752
  for K1, 3.637 for tau 72 h, 3.577 for tau 168 h; verified rows 14.93, 13.42, 12.79) within 0.01.
  Observed power for R0 on the injected bank carries the same shift as the lags, with the same
  staleness (at least 33 hours). Tau is the design answer's pair of values, and both are reported.
- **No calendar or date input enters either head.** The age of the current departure is the only
  duration signal. Because every injected shift ends on day 14, an extra control gives the head
  the age shuffled across rows. P2 and P3 are reported per injected length (3, 7, 15, and 30
  days). The 28 to 112 day part of the duration prior cannot be exercised by this bank and rests on
  the stated prior.
- **Leakage.** The leave-one-month-out variant trains on later months as well as earlier ones, as
  experiment 613's tail constants do. A forward-chained variant (earlier months only) runs on the
  first month pair. Rows flagged by experiment 637's detector or by G4, and the verified D5 and
  D16 days, are excluded from head training (the real rows in them have unknown NRA), and the
  training row count per fold is reported.
- **Extra controls.** An age-and-gate-only head (no gap inputs); H2 with a noise placebo; a plain
  widened K1 band with the same coverage as H1 on P2 and P4; an oracle head given the true depth
  and onset on injected rows (the ceiling). The mean q95-q05 band width is reported beside every
  coverage. The five seeds of K1 enter the intervals by resampling seeds.
- **Figures chosen before scores.** Six time series plots: pass 2, 5, and 9 of 2025-12, one
  Primary of each metering group each, plus the verified D5 and D16 pair.

## Read-out: the gap as a detector

If the head works, the gap between the prevailing and NRA outputs is a detector of departures. A
detector is needed anyway to mask rows when scoring NRA forecasts on new data, because NGED will
send no new labels. The run scores two signals as detectors with no extra fit: the gap at lead 0
(the head's depth *d*) and the head's survival probability. The scoring reports recall on the
injected bank (per depth and length) and on the verified D5 and D16 pair, the false-alarm rate per
month on event-free days, and the days from onset to the first alarm, beside experiment 637's
frozen detector on the same days. Limits stated in the write-up: the gap is built from the same
lag-minus-lag-free signal as G4; the bank holds four lengths and one verified real pair; and NGED's
event labels are incomplete, so false alarms are an upper bound.

## Standing practices

Followed: cleaned data throughout (the `substation_zero` rule's drops reported per fold); the noisy
start of D15 and D18 cut; every head saved under the experiment's data folder; stored predictions of
experiments 612 and 613 reused; tails and cost reported; the day-ahead and the 14-day horizon
plotted against what happened; at least 6 time series plots across both metering groups of the
shifts and of the two outputs. Not followed: refit monthly (no K1 refit occurs). The July to
October 2026 window is closed.

## Advice taken from earlier experiments

Followed: experiment 590's placebo and no-shifted-target rules; experiment 598's rule to keep gate
signals out of the NRA side; experiment 606's gap and neighbour inputs; experiment 613's tolerance;
experiment 645's pairing of K1 against BG2 on the development months. Departures: none.

## Next steps (rewritten after the run)

A permanent change in load appears as a departure that never ends, which the first experiment does
not test. The design answer's follow-up 1 (injected 4 to 16 week shifts) and follow-up 2 (joint
training) wait for this result.
