---
id: "00414"
slug: post-hoc-gap-filled-counterfactual-lag-rescore
title: >-
  Post hoc: does filling the gaps in the stage-1 median, so that more load-shift events are scored,
  change the counterfactual-lag gain on harm rows?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
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
code: ml-experiments/00414-post-hoc-gap-filled-counterfactual-lag-rescore/
---

**Take-home message (to be written when the result is known).**

This page is the plan. It is committed before any score exists. The experiment is **post hoc**: it
exists because [experiment 411](../00411-counterfactual-lag-rescore-harm-rows-r1-boosters/result.md)
scored only 15 of 184 corrected events, and the design change (the fill rule) was chosen after that
result was read. The fill rule below was fixed from counts of the input gaps, not from any score.

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
