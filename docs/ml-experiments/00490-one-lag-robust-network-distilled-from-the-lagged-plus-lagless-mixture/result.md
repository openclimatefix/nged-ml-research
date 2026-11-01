---
id: "00490"
slug: one-lag-robust-network-distilled-from-the-lagged-plus-lagless-mixture
title: >-
  Can one neural network that reads recent power be trained to forecast as well as a mixture of four
  networks and a lagless table, even when its recent power readings are wrong?
theme: lagged-power-and-error-correction
also_relevant_to:
  - calibration-and-uncertainty
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: one table of 11-quantile forecasts per arm, month, seed, and scoring world in this
  experiment's private folder, plus 45 new saved neural networks and one cached table of teacher
  mixtures per month and seed. More than one design would defensibly test the hypothesis (which
  teacher, which loss, which rows carry the teacher term), so the plan fixes them in advance. The
  scripts import experiment 404's modules, experiment 454's scoring helpers, and experiment 435's
  tail helpers. Production serving path: not touched (this repository holds none). Degradation
  rule: not touched.
hypothesis: >-
  A single lagged network (A0-inj's architecture: the Primary's measured power at three earlier
  times, no lagless input) trained on injected load-shift worlds, with the forecast of experiment
  454's K2-S1 mixture as an extra training target, forecasts as well as that mixture, so the live
  service could run one network and store no lagless table, mixture step, or store of past lagless
  forecasts.
prediction: >-
  Written before any score is read. P1. D-K2 is within the printed minimum detectable effect of
  K2-S1 on rows with no labelled event at the signed-MW six and at the MVA-only eight (K2-S1 is
  0.12 points of capacity better than B-mix at the six in experiment 454, so a student that reaches
  B-mix but not K2-S1 fails). P2. D-K2 is no worse than K2-S1 by more than the minimum detectable
  effect on rows inside real events (scored against normal-running-arrangement actuals) and on
  held-out injected harm rows. P3. D-K2's q99 pinball loss is within +0.02 points of K2-S1's, with
  or without the 1.15 widening. P4. D-Bmix lands between A0-inj and B-mix, because B-mix is a
  weaker teacher. P5 (fails if the teacher term only copies in-sample sharpness): D-K2's q05 and
  q95 coverage on the held-out months is no further from nominal than K2-S1's.
decision_rule: >-
  Score = mass-weighted pinball loss (CRPS-like) in points of capacity on experiment 404's rows,
  seeds averaged within a month, then a mean over the 5 scored months (2025-10, 2025-12, 2026-01,
  2026-03, 2026-06), with a month-block bootstrap 95% interval at a Bonferroni family of 18
  (2 student arms times 9 tests: clean rows, harm rows, real-event rows on NRA actuals, injected
  harm rows, cold winter nights, the 12 to 36 hour band, q99 pinball loss, q01 pinball loss, and
  avoidable-curtailment cost at f = 0.9 at the six). The minimum detectable effect is printed first:
  0.065 at the eight and 0.100 at the six from experiment 470's single seeds, and the seed spread of
  B-mix at 2026-06 (seeds 0 to 2) in this harness, with a cold-night minimum detectable effect from
  the same spread. Recommend D-K2 if its contrasts against K2-S1 on rows with no labelled event at
  the six and the eight lie inside the printed minimum detectable effect, it is no worse than K2-S1
  by more than that effect on the real-event and injected harm rows, and its q99 pinball loss is
  within +0.02 absolute. That cuts serving from 4 networks plus a lagless table to 1 network. If it
  falls between B-mix and K2-S1, report where it lands (the gap closed is a descriptive number only).
  Stop rule: if it scores worse than B-mix on rows with no labelled event beyond the minimum
  detectable effect, report that one network cannot absorb the lagless module and close the idea.
  The smallest gain worth porting is zero in score and large in serving cost, so the verdict is an
  equivalence claim, not a superiority claim. "Clean" always means rows with no labelled event,
  because NGED's event labels are incomplete. The MVA-only eight are reported separately and never
  pooled with the six; D15 and D18 are report-only. These results are interim: five months, so no
  99% claim, and July 2026 stays unused.
benchmark: >-
  K2-S1 (experiment 454: two S1 networks and two lagless forecasts, equal weights, cells from seed
  pairs 0-1, 0-2, 1-2), B-mix, A0-inj (the single-seed network, called R-inj), and S1, all read from
  experiment 454's stored forecasts on the clean world and recomputed on the injected scoring worlds
  where 454 stored none.
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of capacity from 11 quantiles, by row class and
  group, as in experiment 404's report; tail coverage and pinball loss at q01, q05, q95, and q99,
  raw and after the 1.0 to 1.15 widening sweep; avoidable-curtailment cost at the six.
---

## Plan

**This page is the plan; the write-up replaces it after the fits.** The text below is fixed before
any score is read.

- **Question.** Experiment 454's K2-S1 mixture needs 4 networks per forecast (2 S1 and 2 lagless),
  the lagless table, and a store of past lagless forecasts. Can one lagged network, trained with
  K2-S1's forecast as an extra target on injected worlds, match it, so serving needs one network?
- **Harness.** Experiment 404's: five monthly refits (2025-10, 2025-12, 2026-01, 2026-03, 2026-06),
  fit seeds 0 to 2 (each trains on its own injected world, 100 to 102), 6 epochs, experiment 397's
  recency weights, the frozen cross-fitted lagless table, scoring worlds 200 to 202 and the clean
  world.
- **Student.** A0-inj's architecture (three lags, no lagless input, no surprise inputs). The brief
  named S1 as the student, but S1 reads the distance from each lag to the lagless median, which
  needs the lagless table at serving, so an S1 student would keep part of the module this
  experiment tries to remove. The manager is asked to confirm.
- **Teachers.** K2-S1 for seed c is the equal-weight mixture of the S1 networks and lagless
  forecasts of experiment 454's seed pair for cell c (c = 0: seeds 0 and 1; c = 1: seeds 0 and 2;
  c = 2: seeds 1 and 2). B-mix for seed c is A0-inj seed c with lagless table seed c. Training-row
  targets come from forward passes of the stored networks on the student's own training rows and
  world. **The teacher networks are in sample on those rows**; only the lagless table is
  cross-fitted.
- **Loss.** For every row, weight times (0.5 times the negative log likelihood of the true target
  plus 0.5 times a teacher term). The teacher term is the negative log likelihood, under the
  student's mixture, of 8 samples drawn from the teacher's mixture (a Monte Carlo estimate of the
  forward divergence from teacher to student). The weight is experiment 397's recency weight; no
  early stopping is used, so no validation loss is weighted.
- **Arms.** D-K2 and D-Bmix (teacher term on all rows, the arms of the contrasts); D-K2-corr
  (teacher term only on rows where one of the three lags lies in an injected or padded real event,
  the literal reading of the brief; descriptive, outside the Bonferroni family); D-0 (weight 0 on the
  teacher term, one fit at 2026-06 seed 0, as the reproduction check of the stored A0-inj).
  References: A0-inj, S1, B-mix, K2-S1.
- **Fits.** 3 student arms times 15 (5 months by 3 seeds) plus D-0, 46 fits. The cost is reported to
  the manager after the first timed fit.
- **Distillation risk.** A teacher's outputs on rows it trained on are sharper than on new rows. The
  student sees the true target at weight 0.5, and the write-up reports q05, q95, and q99 coverage on
  held-out months (the 5 scored months), raw and widened. A student cannot exceed an honest teacher.
  A variant that distils only from months earlier than the scored month is not run (it needs teacher
  refits).
- **Serving table.** The write-up prints, for K2-S1, B-mix, A0-inj, and each student: networks per
  pass, tables and stores at serving, fits per monthly refit (two S1 seeds, two lagless seeds, and
  the student), and the score cost of each removal.
- **Standing rules followed.** Monthly refits; cleaned data with flagged exact-zero half-hours
  dropped; the D15 and D18 start cut at 2025-03-01; networks saved under the experiment's data
  folder; stored forecasts reused as references; the example forecasts, real-event plots, tail
  table, avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the six, the readings-level
  encroachment sweep, and the bad-data injections of experiment 465 on the arm's own lags;
  unlogged-event caveat.
