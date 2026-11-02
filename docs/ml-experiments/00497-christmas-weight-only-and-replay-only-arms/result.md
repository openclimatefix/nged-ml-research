---
id: "00497"
slug: christmas-weight-only-and-replay-only-arms
title: >-
  Which part of the Christmas weight with ERA5 replay acts: the weight alone or the replay alone?
theme: production-reference-and-best-guess
also_relevant_to:
  - retraining-and-training-history
status: planned
kind: other
date: 2026-11-02
size: >-
  medium. Stored: a quantile forecast per arm, month, and seed, and the fitted weights, in the
  experiment's private data folder. One design is defensible (experiment 485's own H2 split into
  its two components, same code and settings). Callers: the scripts import experiments 87, 397, 407,
  418, 422, 425, 470, and 485, whose callers sit in those folders. Fits: 30 fine-tunes (2 arms, 5
  folds, 3 seeds). One Sonnet code review before the first fit, one Sonnet results review, and two
  Sonnet write-up reviews.
hypothesis: >-
  Experiment 485 bundled a Christmas-row weight of 5 with a 25% ERA5 replay of earlier Christmases
  (H2), so it could not say which component moved the Christmas days or which cost skill on other
  days. Either component alone lowers CRPS on 22 to 31 December 2025, and at least one of them does
  so without the loss outside the window that the bundle showed.
prediction: >-
  Written before any fit or score. Either the weight-only arm (WO) or the replay-only arm (RO)
  lowers 22 to 31 December 2025 CRPS at the six signed-MW Primaries and at the eight MVA-only
  Primaries by at least the Christmas minimum detectable effect (0.27 at the eight and 0.26 at the
  six in experiment 485, recomputed and printed first), and passes the gate outside the window. The
  prior is against it: experiment 485's bundle H2 lowered the eight's Christmas CRPS by 0.222, below
  that effect, and lost 0.081 elsewhere at the eight, so the likeliest outcome is that neither arm
  clears the effect and the stop rule applies.
decision_rule: >-
  Score = CRPS in points of effective capacity with the 1.15 widening, seeds scored cell by cell
  and averaged, month-block bootstrap (month by origin day, 20,000 resamples) with Bonferroni
  intervals. The Bonferroni count is 18: 2 arms against H0 times 9 tests (the 7 CRPS row sets of
  experiment 470 and q99 pinball loss at the eight and at the six on rows with no labelled event),
  as in experiment 485. The Christmas contrasts use a day-block bootstrap, are descriptive, and
  enter no count. The minimum detectable effect is printed first, from the three-seed spread of the
  stored arm A fits, for the Christmas days, the gate rows, the cold winter nights, and the 12 to
  36 hour band. Gate for each arm: on all folds' days outside 22 December to 2 January, CRPS may not
  exceed H0's by more than 0.02 points on the eight and on the six rows with no labelled event, and
  q99 pinball loss by more than 0.02 absolute. The 0.02 gate sits below the minimum detectable
  effect on the six rows (0.122 in experiment 485), so a pass there is inconclusive, not shown. An
  arm is a candidate only if it meets the gate and lowers Christmas CRPS by at least the Christmas
  minimum detectable effect at both groups. One scored Christmas is suggestive at best; the check is
  Christmas 2026. Stop rule: if neither arm moves Christmas CRPS by the minimum detectable effect,
  report that the network handles the holiday as well as the data allow. The smallest gain worth
  porting is the Christmas minimum detectable effect, because both arms are a training-loop change
  with no serving cost.
benchmark: >-
  H0, experiment 425's stored arm A fine-tunes (seeds 0 to 2 as cells), the bundle H2 from
  experiment 485 (quoted from its page, not rescored), BG2, `manual_heuristic`, and climatology from
  experiment 470 (curtailment cost: `manual_heuristic` only).
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity, q01 to q99 pinball loss and coverage, and avoidable
  curtailment cost at f = 0.9, 0.75, and 0.5.
---

**Planned.** Experiment 485 found that the Christmas weight with ERA5 replay (H2) lowered Christmas
CRPS by 0.222 points at the eight MVA-only Primaries, inside one holiday's noise, and cost 0.08
points on other days. This experiment fits the two components separately so the next step knows
which one to keep or drop. The eight are the eight MVA-only Primaries, and the six are the six
signed-MW Primaries.

## Design (fine-tune experiment 425's stored arm A pretraining, three seeds, five folds)

- **Folds, cadence, cuts:** the folds of experiment 485 (2025-10, 2025-12, 2026-01, 2026-03,
  2026-06), one monthly refit per fold, the experiment 399 start cut for D15 and D18 and the flagged
  exact-zero drop exactly as in experiments 425 and 485, with the per-fold counts logged.
- **H0 (reference, not refitted):** experiment 425's stored fits. Experiment 485's `H0r` showed the
  widened network reproduces them to 1e-8 (fold 2026-06, seed 0).
- **WO:** window rows (valid time 22 December to 2 January, by UTC date) weighted 5, multiplying
  experiment 425's recency weight `0.5 ** (age_days / SHORT_HALF_LIFE_DAYS)`. No replay.
  Early stopping sees the unweighted held-out-week negative log-likelihood.
- **RO:** 25% of each batch drawn with replacement from the ERA5 and CAMS rows of every Christmas
  window before the cut (2019 to 2024 for fold 2025-12), with their real lags. No weight. The
  window rows are the same as experiment 485's H2 replay.
- **Same code as experiment 485:** both arms run through experiment 485's `run`, `train_ext`, and
  `pretrain_rows`, with the three zero-initialised extra input columns left at zero, so the
  network matches H2's shape.
- **Scored Christmas:** 22 to 31 December 2025 (fold 2025-12) only. December 2024 is not scored:
  no pretrained weights exist for that cut (the stored cuts cover 2025-07 to 2026-06), and the
  ensemble history starts 2024-04-01, which leaves 8 months of ensemble fine-tuning.
- **Confound, stated up front:** RO takes 1.33 times the optimiser steps per epoch, so a RO effect
  is a replay effect plus a step-count effect. A step-matched control is not run.
- **Compute:** about 0.6 GPU-hours (30 fits of 36 to 70 seconds, one GPU process under 8 GB, one
  feeder core).

## Scoring (all through experiment 485's scorer, pointed at this folder)

- **Event days (suggestive):** CRPS on 22 to 31 December and the days either side, at the six and
  the eight separately, day-by-day fan charts, q01, q05, q95, and q99 coverage and pinball loss, and
  avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the signed-MW six.
- **Gate rows:** all folds' days outside the window, the 12 to 36 hour band, and cold winter nights
  (lowest temperature decile per Primary, as experiment 470's slice).
- **Serving table:** neither arm stores or computes anything at serving; both change the training
  loop only. Refits per year: unchanged (12 monthly refits).
- **Standing rules followed:** monthly cadence; cleaned data; the D15 and D18 cut; fitted weights
  saved; stored predictions reused (H0); Bonferroni count as a number; July 2026 unused; every
  score is selection-biased; NGED's event labels are incomplete, so "rows with no labelled event"
  may hold events.
- **Items not applicable:** load-shift event plots (the experiment does not score load-shift events
  as such); the 12 to 36 hour wind and solar bands (demand only).
