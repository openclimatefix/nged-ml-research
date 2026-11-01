---
id: "00485"
slug: christmas-period-demand-forecast-regime-arms
title: >-
  Does treating 22 December to 2 January as its own regime improve the pretrained demand
  network's forecast over Christmas?
theme: production-reference-and-best-guess
also_relevant_to:
  - lagged-power-and-error-correction
status: planned
kind: change
date: 2026-11-01
size: >-
  complex. Stored: a quantile forecast per arm, month, and seed, and the fitted weights, in the
  experiment's private data folder. More than one design would defensibly test the question (a
  day-type input, a lag substitute, a weight, a replay share, and a Christmas-weighted pretraining
  each have their own form), so the plan fixes each before any score is read. New code in the lag
  builder (the lag substitute) and in the training loop (the replay share). Callers: the scripts
  import experiments 87, 397, 407, 418, 422, 425, and 470, whose callers sit in those folders.
  Fits: 1 reproduction fit, 75 fine-tunes (5 arms, 5 folds, 3 seeds), and 15 pretrainings (arm P,
  5 folds, 3 seeds). One Sonnet code review before the first scored fit, one Sonnet results review,
  and two Sonnet write-up reviews.
hypothesis: >-
  Experiment 425's arm A (A-pool) already has Christmas flags and day-of-week inputs, yet its lags
  on 26 to 31 December point at holiday days, and its Christmas weeks are seen at ordinary weight
  by pretraining and fine-tuning. Treating 22 December to 2 January as a regime (a day-type input,
  a lag taken from the latest same-type day outside the window, a Christmas-row weight with an
  ERA5 replay, or a Christmas-weighted pretraining) lowers CRPS on the Christmas days without
  harming other days.
prediction: >-
  Written before any fit or score. P1: H3 (lag substitute) lowers CRPS on 22 to 31 December 2025
  at the signed-MW six and at the eight by more than the Christmas-day minimum detectable effect,
  and helps most on 26 and 27 December, when the day and weekly lags fall inside the holiday. P2:
  H1 (day-type columns) helps least, because the network already has holiday flags. P3: H4 lowers
  Christmas-day CRPS by the larger of 0.10 points and the minimum detectable effect, and narrows the
  gap to BG2 on those days, while passing the non-inferiority gate elsewhere. P4: the Christmas
  weight alone (H2) and the Christmas-weighted pretraining (P) move the other days by an amount
  that is inside the minimum detectable effect (an equivalence claim against it). The prediction can
  fail in either direction: the network may already handle the holiday as well as the data allow.
decision_rule: >-
  Score = CRPS in points of effective capacity with the 1.15 widening, seeds scored cell by cell and
  averaged, a month-block bootstrap (month by origin day, 20,000 resamples) with Bonferroni
  intervals. The Bonferroni count is 45: 5 arms against H0 times 9 tests (the 7 CRPS row sets of
  experiment 470 and q99 pinball loss at the eight and at the six on rows with no labelled event).
  The Christmas contrasts use a day-block bootstrap, are descriptive, and enter no count. The
  minimum detectable effect is printed first, from the three-seed spread of the stored arm A fits
  (including the cold-night rows and the Christmas days). Gate for every arm: on all folds' days
  outside 22 December to 2 January the CRPS may not exceed H0's by more than 0.02 points on the
  eight and on the six rows with no labelled event, and q99 pinball loss by more than 0.02
  absolute. An arm is a candidate for the first release only if it meets the gate. A Christmas gain
  on one scored period is suggestive; the check is Christmas 2026. Recommend H1 or H3 only if it
  gains and costs no more than a calendar column or a small store. Stop rule: if no arm beats H0 on
  the Christmas days by more than the Christmas minimum detectable effect, report that the network
  already handles the holiday as well as the data allow and that BG2's lead is not closed by these
  features. The smallest gain worth porting is the Christmas minimum detectable effect for H1 and
  H3 (a column and a small store of past lags), and the same for H2 and P (a training-loop change
  with no serving cost).
benchmark: >-
  H0, experiment 425's stored arm A fine-tunes (seeds 0 to 2 as cells, and the stored five-seed
  A-pool), BG2, `manual_heuristic`, and climatology from experiment 470.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity, q01 to q99 pinball loss and coverage, and avoidable
  curtailment cost at f = 0.9, 0.75, and 0.5.
---

**Plan (no result yet).** This page is committed before any fit so that the commit order proves the
hypothesis came first. The write-up replaces it when the fits finish.

## Question and what the data can say

- **Scored Christmases: one.** Fold 2025-12 holds 22 to 31 December 2025 (10 days); fold 2026-01
  holds 1 and 2 January and trains on Christmas 2025 beside them, so those two days are report-only
  (lag leakage). The window is 22 December to 2 January by UTC date everywhere.
- **Training Christmases:** pretraining (ERA5 and CAMS rows with real lags, from 2019-09) holds
  Christmas 2019 to 2024; the ENS fine-tuning window (from 2024-04, 730 days) holds Christmas 2024
  and, for folds after the new year, Christmas 2025. Fold 2025-12 never trains on its own Christmas.
- **Consequence:** the Christmas minimum detectable effect on 10 days is large. It is printed before
  any contrast, every Christmas contrast is suggestive, and Christmas stays report-only for the
  usual gates. A second scored Christmas (December 2024 held out) needs a new fold cut and is not
  run.

## Arms (all fine-tune experiment 425's stored arm A pretraining, three seeds, five folds)

- **Folds:** 2025-10, 2025-12, 2026-01, 2026-03, 2026-06 (experiment 449's), one refit per fold,
  monthly cadence as in 425. The experiment 399 start cut for D15 and D18 and the flagged exact-zero
  drop are applied exactly as in 425; the per-fold counts are logged.
- **H0 (reference, not refitted):** 425's own stored fits; `H0r` refits seed 0 of 2026-06 to show
  the widened network reproduces them within noise.
- **H1:** adds two input columns, `sat_like` and `sun_like`, with zero-initialised first-layer
  weights. On a window day the Primary's own history gives the day type: weekday-like (both zero),
  Saturday-like, or Sunday-like. A year votes by taking the nearest of three reference levels (mean
  daily power of weekdays, Saturdays, and Sundays from 15 November to 14 December and from 10
  January to 10 February); the calendar dates are voted by mode over earlier years whose window
  ended at least 4 days before the refit. Bank holidays outside the window are not remapped (425 has
  a bank-holiday flag; experiment 396's holiday columns raised January error, the prior for the
  gate). The network already has `christmas_inside`, `christmas_days`, `christmas_tent`, and the
  day-of-week inputs, so only the day type is new.
- **H2:** Christmas-period rows (valid time in the window) get weight 5 in fine-tuning, multiplying
  425's recency weight `0.5 ** (age_days / SHORT_HALF_LIFE_DAYS)`. ERA5 and CAMS rows whose valid
  time is in the window (about 0.27 million rows, held-out weeks excluded) are replayed: each batch
  of 16,384 holds 12,288 fine-tuning rows and 4,096 replay rows drawn with replacement, the fold's
  mean recency weight (so replay carries about a quarter of the loss), so an epoch (one pass over
  the fine-tuning rows) takes 1.33 times the steps. Early stopping sees the unweighted held-out-week
  negative log-likelihood of the fine-tuning rows only.
- **H3:** if a row's day or weekly lag half-hour lies in the window, the lag becomes the cleaned
  power at the same half-hour of the latest earlier day outside the window with the target date's
  day type (voted type in the window, weekday type outside; the day-type vote for a date uses only
  earlier years, and a substituted lag's age input grows by the days stepped back); a third input
  column `lag_substituted` flags it. The newest-usable-observation rule is kept because the search
  only moves back. The origin lag is not touched. The rule is the same in training and at forecast
  time.
- **H4:** H1 plus H2 plus H3.
- **P (requested 2026-11-01 by the manager for the maintainer):** a new pretraining of arm A's
  network with the window rows weighted 5 over Christmas 2019 to 2024 (pretraining cap and patience
  as 425), then the plain 425 fine-tune. It separates "Christmas skill from pretraining on all six
  Christmases" from "fine-tune emphasis" (H2).

## Scoring (experiment 470's rows, slices, and widening)

- Minimum detectable effect from H0's three-seed spread, printed first (all slices, cold nights,
  Christmas).
- Christmas days (22 to 31 December 2025) and the days either side (17 to 21 December, 3 to 6
  January), at the six and the eight separately: CRPS, q01, q05, q95, q99 coverage and pinball loss,
  avoidable curtailment at f = 0.9, 0.75, and 0.5 at the six, day-by-day fan charts for two
  Primaries per group, with BG2, `manual_heuristic`, and climatology as reference rows.
- Gate rows: all folds' days outside the window, rows with no labelled event, plus cold winter
  nights.
- Every score is selection-biased; July 2026 stays unused. Production cost: whether the serving path
  needs the voted day-type table, a store of past same-type lags, and the three extra inputs.

## Cost

About 1.3 GPU-hours: 76 fine-tunes at 30 to 70 seconds, 15 pretrainings at about 1 minute, plus fold
building. The timing of one fit is reported to the manager before the remaining fits.
