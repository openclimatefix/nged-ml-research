---
id: "00485"
slug: christmas-period-demand-forecast-regime-arms
title: >-
  Does treating 22 December to 2 January as its own regime improve the pretrained demand
  network's forecast over Christmas?
theme: production-reference-and-best-guess
also_relevant_to:
  - lagged-power-and-error-correction
status: finished
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
verdict: not supported
best_arm: >-
  None. H2 (Christmas weight 5 with ERA5 replay) has the lowest Christmas CRPS at the eight MVA-only
  Primaries (difference from H0 -0.222 on 22 to 31 December 2025), below the Christmas minimum
  detectable effect of 0.271, and it fails the gate elsewhere (+0.081 at the eight).
headline: >-
  No arm lowers Christmas CRPS by more than the minimum detectable effect (0.27 at the eight MVA-only
  Primaries, 0.26 at the signed-MW six, one scored Christmas), so no gain was detectable. H2, H4, and
  P lose 0.05 to 0.08 points at the eight outside the window, more than the 0.02 gate. H1 passes the
  gate and changes nothing. H3 lowers the six's CRPS slightly on rows with no labelled event (-0.018,
  inside the noise) and raises it on real-event and cold-night rows (+0.30 and +0.31).
lessons:
  - >-
    No treatment gained over Christmas by more than the experiment can detect; that is not proof of
    equivalence. P comes within 0.012 of BG2 at the six over Christmas (4.002 against 3.990), so the
    six's gap to BG2 is within the noise of one holiday.
  - >-
    Emphasising Christmas rows, by weight and ERA5 replay or by weighted pretraining, costs skill at
    the eight outside the window for a Christmas gain that stays inside the noise.
  - >-
    Replacing a lag that falls in the holiday by a same-day-type lag lowers the six's CRPS slightly
    on rows with no labelled event and raises it on real-event and cold-night rows.
take_home_message: >-
  None of the five ways of treating 22 December to 2 January as its own regime lowered Christmas
  CRPS by more than one Christmas can detect. The three that moved the Christmas days at all also
  made the other days worse at the eight MVA-only Primaries. The next check is Christmas 2026.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Do not add the tested Christmas weight with ERA5 replay, or the Christmas-weighted pretraining,
    to the first release: H2, H4, and P each exceed the 0.02-point gate outside the window at the
    eight (Results, first bullet).
  - >-
    Do not add a same-day-type lag store for the holiday: H3 raises CRPS on real-event rows and cold
    nights at the six by about 0.3 points (Results, real-event and cold-night bullets).
  - >-
    Score Christmas 2026 as the second Christmas before any further holiday work, because one
    holiday cannot separate a 0.2-point gain from noise (Figure 2).
---

**No arm gained more over Christmas than one holiday can detect, and three cost skill on other
days.** The experiment tested five treatments of 22 December to 2 January (H1 a day-type input, H2
a Christmas weight with ERA5 replay, H3 a lag substitute, H4 all three, P a Christmas-weighted
pretraining) against H0, experiment 425's arm A. The eight are the eight MVA-only Primaries, and the
six are the six signed-MW Primaries. The Christmas minimum detectable effect is 0.27 CRPS points at
the eight and 0.26 at the six, from the three-seed spread of the stored fits on 22 to 31 December
2025. The largest Christmas differences (H2 -0.222 at the eight, P -0.190 at the six) lie below
those values, so the pre-registered stop rule applies and no gain was detected. The minimum
detectable effect reflects seed spread only, so it does not prove equivalence.

![Each arm minus H0, Bonferroni intervals](figure_1_contrasts.svg)

![CRPS by valid date over the holiday](figure_2_crps_by_date.svg)

![Day-ahead forecast of one Primary, H0 and H2](figure_3_example_day_ahead.svg)

![14-day forecast of one Primary, H0 and H2](figure_4_example_fourteen_day.svg)

## Results (CRPS in points of effective capacity, 1.15 widening, difference from H0)

- **Row sets:** differences and intervals use rows that H0 and all five arms hold. Levels such as
  H0's 4.777 use the smaller set that BG2, `manual_heuristic`, and climatology also hold, so a level
  can differ from a difference by up to 0.005.
- **Gate, days outside the window, eight:** H1 -0.004, H3 -0.002, P +0.047, H4 +0.077, H2 +0.081.
  H0's own score is 4.777. The gate allows +0.02, so H2, H4, and P fail. The minimum detectable
  effect at the eight is 0.054.
- **Gate, six on rows with no labelled event:** H3 -0.018, H1 -0.001, H4 +0.009, P +0.018, H2
  +0.037. H0 scores 3.563. H2's point estimate exceeds the 0.02 gate, but its Bonferroni interval
  (-0.009 to 0.089) spans zero. The minimum detectable effect is 0.122, so every arm is inside it.
- **q99 pinball loss on the gate rows:** H2 +0.009 and H4 +0.008 at the eight (Bonferroni intervals
  exclude zero), everything else within 0.003. The +0.02 absolute rule is met by all arms.
- **Cold winter nights, outside the window:** at the six H3 is +0.309 (outside the minimum
  detectable effect of 0.26), P +0.210 and H4 +0.127 (inside it), and H2 -0.124; at the eight P is
  +0.213, H4 +0.255, and H2 +0.178, all outside the minimum detectable effect of 0.15.
- **Real-event rows at the six (normal-running-arrangement actuals):** H3 +0.301, H4 +0.226, P
  +0.222. H3's lag substitute therefore hurts when an event is present.
- **Christmas, 22 to 31 December 2025 (suggestive, one holiday, day-block bootstrap, no count):** at
  the eight H2 -0.222 (uncorrected 95% interval -0.312 to -0.131; Bonferroni with 45 tests -0.367 to
  -0.077), H4 -0.180, P -0.050, H3 -0.037, H1 +0.033; at the six P -0.190, H2 -0.097, H3 -0.017, H4
  -0.015, H1 +0.014. H0 scores 4.604 (eight) and 4.199 (six); BG2 scores 5.173 and 3.990;
  `manual_heuristic` 5.783 and 6.801; climatology 6.929 and 7.133.
- **Days either side (17 to 21 December, 3 to 6 January):** H2, H3, H4, and P are worse than H0 at
  both groups (+0.07 to +0.31); H1 is slightly better (-0.04, -0.06).
- **1 and 2 January of fold 2026-01 (report-only, lag leakage):** H3, H4, and P are worse (+0.2 to
  +0.9) at the six.
- **Tails on Christmas:** q95 coverage at the eight is 0.968 for H0 and 0.949 to 0.971 for the
  others; at the six it is 0.873 for H0 and 0.832 to 0.875 for the others (below 0.95 for all).
- **Pre-registered predictions:** P1 (H3 gains by more than the minimum detectable effect) failed.
  P2 (H1 helps least) held. P3 (H4 gains and passes the gate) failed on both counts. P4 (H2 and P
  stay inside the minimum detectable effect elsewhere) failed for H2 and H4 (+0.081 and +0.077
  against 0.054) and for P (+0.047, inside 0.054 only narrowly).
- **Compute:** 1 reproduction fit (H0r, 2026-06 seed 0: 15 epochs, val NLL 0.3641, the stored
  value), 75 fine-tunes of 36 to 70 seconds, and 15 P pretrainings; about 1.3 GPU-hours.

## Caveats

- The experiment did not score avoidable-curtailment cost at f = 0.9, 0.75, and 0.5, the "Checks"
  bullets (median minus mean forecast, mean-forecast error beside each score), the load-shift event
  time-series plots, or a second scored Christmas (December 2024 held out). Figures 3 and 4 show one
  Primary and one run.
- The Christmas and either-side slices come from one holiday, so every number there is suggestive.
- H2 and H4 take 1.33 times the optimiser steps per epoch, so their loss differs from H0's in the
  batch mix and the step count as well as in the weight. Their gate losses cannot be separated from
  that.
- The voted day type uses only earlier Christmases, so fold 2025-12 has years 2019 to 2024 for the
  Primaries with that history and none for D15 and D18; for such dates the day type is the
  weekday's.
- Every score is selection-biased: the five folds that choose the arm also score it. July 2026 is
  unused. NGED's event labels are incomplete: "rows with no labelled event" may hold events.
- Reference H0 is experiment 425's stored fits; the reproduction fit matched the stored fit.

## Audit (fresh Sonnet agent, from the raw stored forecasts)

- **Trigger:** three pre-registered predictions failed (P1, P3, P4) with no tested explanation.
- **Numbers reproduce:** the audit re-derived H2 +0.081, H4 +0.077, P +0.047, H1 -0.004, and H3 -0.002
  at the eight outside the window, and H3 +0.301 on real-event rows at the six. Every H2 and H4 seed
  loses to the matching H0 seed, and all arms are scored on the same 1,145,256 rows. It found no
  leakage in the day-type vote, the replay rows, or the lag substitution.
- **Accepted: the H2, H4, and P loss sits in two winter folds and is not specific to Christmas.**
  The loss at the eight is +0.139 (H2) and +0.172 (P) in fold 2025-12, and +0.195 (H2) and +0.132 (P)
  in fold 2026-01, against at most +0.05 elsewhere. In fold 2025-12 H2 loses +0.112 on 1 to 16
  December, more than five days before the window. P has neither the weight nor the replay and
  shares the loss, so a factor common to the winter folds may explain part of it. H2 bundles the
  weight with the replay (each replay row is seen about 70 times in a 10-epoch fit), so the
  experiment cannot say which component hurts. The advice below therefore rejects the bundles as
  tested, not the ideas separately.
- **Accepted: the H3 real-event harm rests on one episode.** The six's real-event rows are mostly
  two Primaries and one long episode. In fold 2026-01 the harm is +0.725 on targets from 3 to 16
  January that have a substituted lag, and -0.005 elsewhere, which fits the substituted lag
  removing event state. In fold 2025-12 the harm is +0.405 on rows with no substituted lag, so the
  +0.30 mixes substitution and a changed fit. Excluding 3 to 13 January the figure is +0.113.
- **Accepted: the gate slice includes targets up to about 16 January,** which H3 and H4 affect,
  because long leads carry a lag back into the window. The gate numbers keep that definition.
- **Not run:** a weight-only arm, a replay-only arm, and H0r in folds 2025-12 and 2026-01. The
  H0r run in 2026-06 matches the stored fit to 1e-8.

## What to try next

- Score Christmas 2026 with H0 and H1, the arm that passes the gate.
- Run a weight-only arm and a replay-only arm, and H0r in folds 2025-12 and 2026-01, before ruling
  out either idea.
- Hold out December 2024 as a second scored Christmas before spending more on the question.

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

## Design (all fine-tune experiment 425's stored arm A pretraining, three seeds, five folds)

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
