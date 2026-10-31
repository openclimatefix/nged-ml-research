---
id: "00440"
slug: windowed-and-neighbour-relative-surprise-on-the-lag-network
title: >-
  Does reading the surprise in a window around each lag, and against what the neighbouring Primaries
  did, help the lag network tell a load-shift event from a genuine level shift?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: 36 fitted networks (plus 12 if the attention version runs), their forecasts, and
  sweep tables in this experiment's private folder only; experiment 428's and 404's folders are read
  and never written. More than one design would test the hypothesis (which statistics, which window,
  how neighbours are weighted), so each choice is fixed here before any score is read. Production
  serving path: not touched (this repository holds none). Degradation rule: not touched. Callers:
  the scripts import experiment 428's modules, experiment 402's world builder, and a copy of
  experiment 404's window-statistics module.
hypothesis: >-
  Adding to S1 (the three lags plus the surprise at each lag) the statistics of the surprise in a
  12-hour window either side of each lag (S2) lowers the error on rows whose lags are near a
  load-shift event. Adding the same statistics for the neighbouring Primaries and the surprise
  relative to the neighbours' (S3) lowers it further, because a load-conserving event shows an
  opposite shift at neighbours and a genuine level shift does not. The gain comes from the
  neighbours' identity, not from any extra inputs: S3 beats S3-rw (random neighbours of the same degrees).
prediction: >-
  Written before any fit and before any score is read. CRPS in points of effective capacity at days 1
  to 10, 1.15 widening, signed-MW six, seeds 0 to 2. P1. S2 minus S1 on clean rows lies between -0.06
  and 0.00 and on injected harm rows between -0.15 and -0.02. P2. S3 minus S2 on injected harm rows
  lies between -0.15 and 0.00 and on real harm rows between -0.20 and +0.05. P3. S3 minus S3-rw on
  injected harm rows lies between -0.12 and -0.01 (the injected events are load-conserving between
  graph neighbours, so random neighbours show no mirror). P4. On clean rows S3 minus S2 lies within
  +-0.04. P5. In the encroachment sweep (step on the lags only, so no mirror at neighbours), S3
  follows at least as large a share of the step as S2 at k = 3, because an unmirrored step is a
  genuine level shift. P6. The MVA-only eight show the same signs with smaller sizes.
decision_rule: >-
  Four folds (test months 2026-06, 2026-01, 2025-10, 2026-03), seeds 0 to 2, monthly refits,
  contrasts per group (signed-MW six decides; MVA-only eight and D15 and D18 are reported) on CRPS
  at days 1 to 10 at the 1.15 widening, on clean, injected harm, real harm, and real event rows,
  with a month-and-day block bootstrap interval and experiment 428's minimum detectable effect from
  R-inj's seed spread. Pre-registered margins: a contrast is a gain when its point estimate is
  below minus the larger of 0.03 points (the margin agreed for demand) and the minimum detectable
  effect, and its interval excludes zero; it is a loss when it is above plus that margin; the tail
  harm limit is 0.02 points of pinball loss at q95 and q99. Contrasts: W1 S2 minus S1; W2 S3 minus
  S2; W3 S3 minus S3-rw; W4 S3 minus S1; W5 S3-rw minus S2 (do random neighbours add anything).
  S2 advances over S1 if W1 is a gain on injected harm or real harm rows at the signed-MW six with
  no loss on clean rows and no tail harm. S3 advances over S2 if W2 is a gain on injected harm or
  real harm rows, W3 is a gain on the same rows (the neighbours matter, not the extra inputs), and
  there is no loss on clean rows and no tail harm. The attention version of S3 runs only if W2 is a
  gain on clean or injected harm rows. Otherwise the page reports the arm as not advancing, and
  S1 stays the base. The smallest gain worth porting is the margin above, because S2 and S3 add
  hand-built inputs that need the window statistics and the neighbour graph in the serving path.
  Interim; no 99% claim.
benchmark: >-
  S1 and R-inj of experiment 428 (seeds 0 to 2, stored forecasts), and E-oracle of experiment 438 as
  the ceiling for perfect trust information.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, coverage and pinball loss at q01, q05,
  q95, and q99, at the 1.15 widening; the lag-encroachment sweep of experiment 392 with its
  masked-lag control.
verdict: not-yet-run
best_arm: none yet
production_status: not-for-production
adopted_in: none
code: ml-experiments/00440-windowed-and-neighbour-relative-surprise-on-the-lag-network/
---

**Take-home message: not yet run.**

## Question

CRPS is the continuous ranked probability score. The signed-MW six and the MVA-only eight are the two
groups of Primary substations by metering. A Primary's neighbours are the Primaries that share a
supply route with it, from the adjacency file of experiment 383.

**Experiment 428 gave the lag network the surprise at each lag, and the maintainer asks whether the
surprise is better read in a window and against the neighbours.** A load-conserving load-shift
event shows an opposite shift at the neighbours, so the lag should be ignored. A genuine level shift
shows no mirror, so a large surprise should raise the lag's weight. A gate that can only down-weight a
surprising lag cannot express this, so these arms feed the features to the network as plain inputs
and let the network learn either direction. The window statistics are experiment 404's (copied, with
credit).

## Design

- **Arms.** S2 (S1 plus 24 window statistics), S3 (S2 plus 30 neighbour columns), S3-rw (S3 with
  random neighbours keeping each Primary's degree). All three use experiment 428's rows, injected
  worlds, training loop, and the frozen cross-fitted lagless tables of experiment 404.
- **Statistics.** Per lag: the mean, signed sum, standard deviation, minimum, maximum, counts above
  the 95% and below the 5% lagless quantile, and a triangular-tent mean of the surprise in a window
  of 12 hours either side of the lag, cut at the newest usable reading. Experiment 404's median is
  omitted (prefix sums cannot give it). Neighbour columns are capacity-weighted means of the same
  statistics, the lag's surprise minus the weighted neighbour surprise, and the weighted sign
  agreement.
- **Sweep.** Prediction only, on the saved networks, stepping the same lags (k = 1, 2, 3) in every
  arm including R-inj and S1, with the masked-lag control. A step shifts the window's mean, sum,
  minimum, maximum, and tent by the step and leaves the standard deviation and counts unchanged
  (an approximation).
- **Compute.** 36 fits (3 arms, 4 folds, 3 seeds), about 6 to 10 GPU-hours, one process under 8 GB
  (a one-epoch timing used 2.9 GB). Twelve more only if the attention version runs.
- **Reported.** Absolute CRPS beside every contrast, tails, the real event count, and example
  forecasts and labelled-event plots.
