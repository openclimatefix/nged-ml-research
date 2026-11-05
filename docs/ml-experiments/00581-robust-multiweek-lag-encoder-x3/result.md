---
id: "00581"
slug: robust-multiweek-lag-encoder-x3
title: >-
  Does reading five weeks of power lags through a median and a spread, instead of one day lag and
  one week lag, stop the forecast following a load-shift event, without losing the lag gain?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
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
---
**Plan only; the write-up replaces this page when the experiment finishes.**

## Question

We ask whether the neural network that reads power lags can be made to ignore a load-shift event
that sits in one or two of its weekly lags, without giving up the gain it gets from reading lags.
Experiment 578 and the Opus review of the normal-running-arrangement (NRA) plan found that the
forecast follows an event most in the week after it ends, when the week lag still holds the event
and the day lag is already clean.

## Design (X3 of the review)

- **Base.** v3+D5 of experiment 572: the lag network of experiment 560 (mode v3) plus the D5 output
  offset of experiment 565. Five folds (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), refits on days
  0 and 14, seeds 0 and 1, the same rows and scoring as experiment 572.
- **X3-med.** The week lag of v3 becomes the median of the same-weekday half-hours at weeks 1 to 5.
  Seven columns are appended: the median absolute deviation of the five weekly lags, the median and
  the median absolute deviation of the day lags at days 1 to 3, the robust z-scores of the day lag
  and of the newest weekly lag against the weekly median, and the robust z-score of the day lag
  against the day median. The week lag is missing when fewer than 3 of the 5 weeks are observed.
- **Controls.** v3+D5 (stored). A placebo with the same encoder and columns, but the five weekly
  lags are weeks 6 to 10 old.
- **X3-35 (report only).** X3-med with the newest reading 3.5 h before the origin.
- **X3-set and X3-syn** run only if X3-med passes the 80% rule above.
- **Injected bank (lag-only).** Contamination is added to the cleaned power of one Primary at the
  lag times, never to the targets: depths 15%, 30%, and 50% of effective capacity, lengths 3, 7, 15,
  and 30 days, both signs, ending on day 14 of each fold month, so that the 7 days after the end are
  scored. Every Primary meets every depth-and-length combination across 24 passes, so a pass
  contaminates all 16 Primaries at once (the network reads one Primary at a time). The reference is
  the same network on uncontaminated lags, so the follow ratio isolates the lags, and the
  uncontaminated power is the NRA reference. The verified event set of the sister experiment 580 is
  scored too, once it exists.
- **Known-answer check.** Follow during the event should rise with the number of contaminated weekly
  lags: near 0 for at most 2 of 5 and near v3+D5's for 3 or more. A 30-day event is followed from
  about day 15 (a design limit).
- **Tail metrics, by phase** (before; days 0 to 3; day 4 to the end; 0 to 7 days after the end; 7
  to 14 days after): follow ratio at q05, q50, q95, q99, coverage inside q05-q95 and q01-q99,
  pinball loss at q01, q05, q95, q99, width inflation, and false-alarm cost on rows with no labelled
  event, against v3+D5 (the recalibrated control of experiment X2 is not built here).
- **Standing practices.** Cleaned data everywhere; D15 and D18 rows before 2025-03-01 cut; every
  fitted model saved; each lag's weather run asserted at or before the lag time, and every lag at
  least 24 h before the origin, row by row.
- **Compute.** One GPU process under 8 GB of host memory, 1 OpenMP thread. Fits are resumable: each
  finished cell is reused and files are written atomically.
