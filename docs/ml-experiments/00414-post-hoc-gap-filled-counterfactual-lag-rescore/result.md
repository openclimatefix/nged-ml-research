---
id: "00414"
slug: post-hoc-gap-filled-counterfactual-lag-rescore
title: >-
  Post hoc: does filling the gaps in the stage-1 median, so that more load-shift events are scored,
  change the counterfactual-lag gain on harm rows?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: done
kind: other
protocol: demand-v2
size: >-
  simple. Stored: one small table of per-row scores in this experiment's private folder; every
  earlier experiment is read and never written. One defensible fill rule is fixed below before any
  score is read. Callers: the scripts copy experiment 411's and import the folders of experiments 87,
  391, 392, and 400. Production serving path: not touched (this repository holds none). Degradation
  rule: not touched.
hypothesis: >-
  Post hoc, because it changes the design of experiment 411 after that experiment's result was read.
  When the gaps in experiment 391's stage-1 median are filled with the mean reading at the same
  half-hour of the 4 weeks before, so that more of the 184 corrected load-shift events are scored,
  replacing the readings inside each event with clean values lowers the CRPS of experiment 392's R1
  lagged boosters on harm rows by at least 0.1 points of effective capacity at the six signed-MW
  Primaries.
prediction: >-
  Written before any score of the filled run is read. At least 40 of the 184 events are scored,
  against 15 in experiment 411. The gain at the six is between -0.5 and +0.3 points, and its 95%
  event-cluster interval includes zero. Reason: experiment 411 found -0.836 on 9 events and +0.460
  without one event, and the events added here come from the same Primaries, so a clear positive
  gain would be a surprise.
decision_rule: >-
  The same as experiment 411: if the gain at the six is under 0.1 points, batches 1 and 2 of
  experiment 404 are skipped, as the experiment 404 plan states; the interval is reported beside
  it. The report states the gain for all scored events, for the 15 events that experiment 411 also
  scored, and for the added events alone, so that a reader can see whether the fill changes the
  answer. The decision belongs to the maintainer.
benchmark: >-
  Experiment 411's result on its 15 events, kept beside this one (the same boosters, rows, and
  scoring; the 15 events are scored again here and should give the same numbers).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Exact CRPS of the distribution linear between the 11 quantiles, in points of effective capacity;
  leads up to 240 hours (day 10).
verdict: inconclusive
best_arm: >-
  none. Filling the gaps added no event, so the result is experiment 411's: real lags 5.061 and clean
  lags 5.897 CRPS points at the six signed-MW Primaries (gain -0.836, 95% -2.468 to +1.301, 9 events).
headline: >-
  Filling the gaps in the stage-1 median added no scored event, because 129 of the 184 corrected
  events lie outside the reach of the 4 test refit months and the 3 events within reach that have a
  gap also lack observed power. The 15 events of experiment 411 reproduce exactly, so the question
  stays open and the sample is capped by the test window, not by the median.
lessons:
  - >-
    A filter listed first in a loss count takes the blame for events that later filters would also
    drop: experiment 411 blamed gaps for 99 events, and 96 of those lie outside the test window.
  - >-
    The reachable events are few: 55 of 184 lie within reach of the 4 test months, and 25 of those
    pass the sign, step, and bias filters.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  The gap-filled re-score changed nothing. Filling the gaps in the stage-1 median added no scored
  event, and the 15 events of experiment 411 reproduced to three decimals, so the six signed-MW gain
  stays at -0.836 CRPS points (interval -2.468 to +1.301) and still rests on 9 events. The sample is
  limited by the 4 test months, not by the gaps. The maintainer still has to decide whether
  experiment 404 skips batches 1 and 2 on this evidence.
advice:
  - >-
    To score more load-shift events, produce boosters and rows for more months, or loosen the
    event filters, and do not fill the median.
  - >-
    Count events lost to filters in the order the code applies them, and count the events outside the
    test window first.
date: 2026-10-30
code: ml-experiments/00414-post-hoc-gap-filled-counterfactual-lag-rescore/
---

**Status: done. Filling the gaps added no scored event.** The sections from "The fill rule" onward
are the plan, written before any score of the filled run was read. The experiment is **post hoc**: it
exists because [experiment 411](../00411-counterfactual-lag-rescore-harm-rows-r1-boosters/result.md)
scored only 15 of 184 corrected events, and the design change (the fill rule) was chosen after that
result was read. The fill rule below was fixed from counts of the input gaps, not from any score.

## Results

**The filled run scored the same 15 events as experiment 411, with the same numbers.** The
prediction (at least 40 events scored) failed, and so did its reason: the fill rule fixes the gaps
in the stage-1 median, and the gaps were not what limited the sample. Intervals are 95% event-cluster
bootstraps (2,000 resamples). They differ from experiment 411's in the third decimal only because
the resampling draws differ.

| Group | Harm rows | Events | CRPS real lags | CRPS clean lags | Gain | 95% interval |
|---|---|---|---|---|---|---|
| Six signed-MW | 5,354 | 9 | 5.061 | 5.897 | -0.836 | -2.468 to +1.301 |
| Eight MVA-only | 3,241 | 6 | 6.514 | 6.489 | +0.024 | -0.568 to +1.550 |
| All 14 | 8,595 | 15 | 5.609 | 6.120 | -0.512 | -1.798 to +0.863 |

- **Added events:** none. The report's "added events" section is empty.
- **Leave-one-event-out gain range:** -1.278 to +0.460 at the six, and -0.354 to +0.877 at the eight.
  Dropping D5's one event still flips the six-Primary sign.
- **No figure:** the scored events are experiment 411's, so its figure applies unchanged.

## Why no event was added

**129 of the 184 corrected events lie outside the reach of the 4 test refit months.** A sampled run
reads lags from the week or two before its origin, so an event can only affect a run if it falls
between 14 days before the first test origin and the last test origin (the count allows 14 days, a
wide margin). Experiment 411's count of "99
events lost to gaps" took the gap filter first, and 96 of those events are in the 129. Counting in
the order of the sweep:

- **Outside the reach of the test months:** 129 events.
- **Gap in the observed power:** 3 events within reach. These also have a gap in the stage-1 median,
  and the fill cannot repair observed power.
- **Sign, step share, and bias filters:** 12, 8, and 7 events within reach.
- **Pass every event filter:** 25 events, of which 15 produce harm rows in the sampled runs.

After the fill, no event within reach has a gap left in the stage-1 median, so the fill worked as
intended on the 3 events it could touch.

## Checks

- **Median minus mean forecast per site and the error of the mean forecast:** not computed, because
  the boosters' mean forecast is not stored; the scored forecasts are the mixture quantiles.
- **Reproduction:** the 15 events reproduce experiment 411 to three decimals in every row of the
  table, including the per-Primary and lead-band gains.
- **Triggers fired:** trigger 2 (the result contradicts its prediction: 15 events scored against at
  least 40 predicted) and trigger 7 (the experiment 404 plan depends on the headline). The cause is
  explained and tested above (the reach count), so trigger 2 has an explanation recorded.
- **Audit:** a fresh Sonnet agent reviewed the code before the first run and found that the fill
  would change the pre-event bias of the original events; the fix computes the bias on the
  unfilled median, and the reproduction above confirms it.

## The fill rule

**The rule is fixed before any score is read.** In experiment 411, 99 of the 184 events were never
scored. The input counts show why: 46% of the event half-hours (14,184 of 30,832) have no stage-1
median. The gaps are median-only in 75 events and in both the median and the observed power in 24
events; the observed power alone is never missing. Of the 75 median-only events, the longest gap is
at most 12 half-hours in 8 events, at most 48 in 30 events, at most 96 in 39, and at most 336 in 70.
Linear interpolation across gaps that long would invent a median, so the fill is different:

- **Where the stage-1 median is missing, it is replaced by the mean observed reading at the same
  half-hour of the previous 4 weeks.** The mean uses only readings that are finite and outside
  every labelled event of the Primary. It is left missing when fewer than 2 of the 4 weeks qualify.
- **Every reading used is older than the half-hour it fills,** so a lag never receives a value from
  after itself.
- **The pre-event bias is computed on the unfilled median,** so the 15 original events keep the
  bias that experiment 411 used and should reproduce its scores.
- **Nothing else changes.** Events with missing observed power stay skipped (24 events). The
  pre-event bias, the 25% step filter, the sign filter, the bias filter, the run sample (at most 6 runs
  per event and refit month), and the scoring are the same as in experiment 411.

## What the experiment answers

**The experiment tests whether the fill rule changes the answer, not whether the filled median is
the true clean load.** A seasonal profile ignores weather and holidays, so the clean lags are an
estimate with an error that grows with the gap. The report therefore gives the gain separately for
the 15 original events, which have no filled values, and for the added events, which do.

## Caveats written before the run

- **The sample is selected.** Events with missing observed power stay out, and the sign, step, and
  bias filters still apply, so the scored events are not a random sample of the 184.
- **The filled median for a long gap is a seasonal average.** It is as good as the previous weeks
  are for the event's own weeks; an event with a holiday in it is badly filled.
- **Single seed and four refit months.** Every number is interim.

## Method

- **Code.** The scripts copy experiment 411's, with one added function (`fill_median`). They
  re-predict experiment 392's R1 lagged boosters twice per sampled run, with the observed series and
  with the clean series, and score both forecasts against the observed power on harm rows (target
  outside every event, at least one lag inside it, lead up to 240 hours).
- **Reviews.** A fresh Sonnet agent reviews the code before the first run. The results get a fresh
  Sonnet review, then the write-up gets a prose and numbers review.

## Problems caused by the move

None so far.
