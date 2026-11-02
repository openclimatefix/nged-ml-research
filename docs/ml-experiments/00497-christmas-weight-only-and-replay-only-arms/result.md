---
id: "00497"
slug: christmas-weight-only-and-replay-only-arms
title: >-
  Which part of the Christmas weight with ERA5 replay acts: the weight alone or the replay alone?
theme: production-reference-and-best-guess
also_relevant_to:
  - retraining-and-training-history
status: finished
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
  report that neither arm improved the neural network's Christmas forecast beyond what the three-
  seed spread can resolve. The smallest gain worth porting is the Christmas minimum detectable
  effect, because both arms are a training-loop change with no serving cost.
benchmark: >-
  H0, experiment 425's stored arm A fine-tunes (seeds 0 to 2 as cells), the bundle H2 from
  experiment 485 (quoted from its page, not rescored), BG2, `manual_heuristic`, and climatology from
  experiment 470 (curtailment cost: `manual_heuristic` only).
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity, q01 to q99 pinball loss and coverage, and avoidable
  curtailment cost at f = 0.9, 0.75, and 0.5.
verdict: not supported
best_arm: >-
  None. RO (replay only) has the lowest Christmas CRPS at the eight MVA-only Primaries (-0.230 from
  H0 on 22 to 31 December 2025), below the Christmas minimum detectable effect of 0.271, and it
  fails the gate outside the window (+0.073 at the eight).
headline: >-
  Neither arm lowers Christmas CRPS by the Christmas minimum detectable effect (0.27 at the eight,
  0.26 at the six, one scored Christmas). The replay alone carries both the Christmas gain at the
  eight (-0.230) and the loss on other days (+0.073, over the 0.02 gate). The weight alone does
  nothing measurable (+0.009 outside the window, +0.148 on the Christmas days at the eight).
lessons:
  - >-
    The ERA5 replay is the part of experiment 485's H2 that moved the Christmas days and the part
    that lowered skill elsewhere. RO reproduces H2's pattern (-0.230 against -0.222 at the eight
    over Christmas, +0.073 against +0.081 outside the window), so the Christmas gain and the loss
    are one effect, not two.
  - >-
    The Christmas weight of 5 alone changes nothing the experiment can detect outside the window,
    and its Christmas contrast at the eight points the wrong way (+0.148, uncorrected interval 0.038
    to 0.238, smaller than the 0.27 detectable effect).
  - >-
    One scored Christmas cannot separate a 0.23-point gain from noise, so the check stays Christmas
    2026.
take_home_message: >-
  The ERA5 replay of earlier Christmases lowers Christmas CRPS at the eight MVA-only Primaries by
  about 0.23 points, a difference smaller than the 0.27 that one holiday can resolve, and the same
  replay makes the other days worse by 0.07 points. The Christmas weight on its own moves the other
  days by 0.01 points and Christmas by 0.15 points the wrong way. Neither idea is worth porting
  before Christmas 2026 gives a second holiday to score.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Do not add the ERA5 Christmas replay (25% of each fine-tune batch) to the first release: it
    worsens CRPS by 0.073 points outside the window at the eight, over the 0.02 gate (RO, Results).
  - >-
    Do not add a Christmas-row weight of 5 to the fine-tune: it moves other days by only +0.009 and
    Christmas the wrong way (WO, Results).
  - >-
    Christmas 2026 would give a second scored holiday; one holiday cannot
    separate a 0.23-point gain from noise (Figure 2).
---

**In this experiment the ERA5 replay carried both the Christmas gain and the loss on other days,
and the weight carried neither; the replay's extra optimiser steps are not ruled out as the cause.**
We asked which half of experiment 485's combined arm (labelled H2 there: a weight of 5 on Christmas
rows plus a replay of earlier Christmases drawn from ERA5 weather estimates, 25% of each batch)
did the work. We fitted each half alone (WO the weight, RO the replay) from the same stored
pretrained neural network, on five folds (test months) and three seeds, and scored both on the
same rows as H0 (experiment 425's unmodified fine-tune). Scores are the continuous ranked
probability score (CRPS) in points of effective capacity. The eight are the eight Primary
substations whose meters report MVA only, and the six are the six whose meters report signed MW.
RO lowers CRPS over 22 to 31 December 2025 at the eight by 0.230 points and raises it by 0.073
points on days outside 22 December to 2 January. WO moves the days outside that window by 0.009.
The Christmas minimum detectable effect (from the three-seed spread of the stored fits) is 0.27 at
the eight and 0.26 at the six. Neither arm clears that effect, so the stop rule applies: we report
no demonstrated Christmas gain. The next test is the Christmas of 2026.

![Each arm minus H0, Bonferroni intervals](figure_1_contrasts.svg)

![CRPS by valid date over the holiday](figure_2_crps_by_date.svg)

![Day-ahead forecast of one Primary, H0, WO, and RO](figure_3_example_day_ahead.svg)

![14-day forecast of one Primary, H0, WO, and RO](figure_4_example_fourteen_day.svg)

## Results (CRPS in points of effective capacity, 1.15 widening, difference from H0)

- **Minimum detectable effect, printed first (three-seed spread of the stored fits):** 0.054
  (eight, outside the window), 0.122 (six, no labelled event), 0.148 and 0.260 (cold nights, eight
  and six), 0.082 (12 to 36 hour band, eight), 0.27 and 0.26 (22 to 31 December, eight and six).
  The 0.02 gate is below the first two, so a pass there is inconclusive.
- **Absolute scores outside the window:** H0 4.780 (eight) and 3.569 (six, no labelled event). WO
  4.788 and 3.592; RO 4.853 and 3.585.
- **Gate, eight:** WO +0.009 (Bonferroni -0.018 to 0.031), RO +0.073 (0.037 to 0.112). RO fails.
- **Gate, six, no labelled event:** WO +0.023 (-0.007 to 0.058), RO +0.016 (-0.017 to 0.054).
  Both are inside the minimum detectable effect, so the gate is inconclusive for both.
- **q99 pinball loss on the gate rows:** RO +0.0087 at the eight (Bonferroni 0.002 to 0.019),
  which stays inside the +0.02 limit; WO +0.0023 at the eight and +0.0016 at the six.
- **Christmas, 22 to 31 December 2025 (suggestive, one holiday, day-block bootstrap, uncorrected
  95% intervals):** at the eight RO -0.230 (-0.315 to -0.145) and WO +0.148 (0.038 to 0.238); at
  the six RO -0.078 (-0.238 to 0.067) and WO -0.031 (-0.234 to 0.157). H0 scores 4.613 (eight) and
  4.171 (six). Experiment 485 reported H2 at -0.222 (eight) and -0.097 (six).
- **12 to 36 hour band, outside the window:** eight WO +0.006, RO +0.054 (Bonferroni 0.010 to
  0.099); six WO +0.015, RO -0.036 (-0.077 to 0.004).
- **Cold winter nights, outside the window:** eight WO +0.017, RO +0.154 (-0.002 to 0.327); six WO
  -0.075, RO -0.089. Minimum detectable effects are 0.148 and 0.260.
- **Tails on Christmas, eight:** q95 coverage H0 0.968, WO 0.923, RO 0.976; q99 coverage 0.996,
  0.987, 0.998. At the six q95 coverage is 0.874 (H0), 0.816 (WO), 0.859 (RO), below 0.95 for all.
  WO's six q99 pinball loss on Christmas is 0.477 against H0's 0.341 (day-block interval 0.012 to
  0.309, uncorrected).
- **Avoidable-curtailment cost at the signed-MW six, rows with no labelled event, all five folds
  (units of experiment 470's table; difference from H0, 95% cell-bootstrap interval):** at f = 0.9
  H0 1.852, WO -0.139 (-0.503 to 0.072), RO -0.167 (-0.557 to 0.049); at f = 0.75 H0 7.408, WO
  -0.178 (-1.046 to 0.411), RO -0.515 (-1.705 to 0.187); at f = 0.5 H0 22.711, WO +0.168 (-1.248 to
  1.439), RO -0.970 (-2.908 to 0.315). `manual_heuristic` costs 2.017, 12.883, and 37.789. On the
  Christmas half-hours alone the costs are 0.000 to 0.002 at f = 0.9, 0.19 at f = 0.75, and 1.2 to
  1.3 at f = 0.5 for all three arms, with no contrast distinguishable from zero.
- **Where RO's loss outside the window sits:** mostly the two winter folds, +0.150 (2025-12) and
  +0.174 (2026-01) at the eight, as in experiment 485's H2.
- **Pre-registered prediction:** failed. Neither arm lowered Christmas CRPS by the minimum
  detectable effect at both groups.
- **Fits and compute:** 30 fine-tunes of 30 to 41 seconds each (0.29 GPU-hours), no reproduction
  fit needed because experiment 485 matched H0 to 1e-8. Rows dropped per fold: 68,680 for the
  D15 and D18 start cut and 2,155 to 2,925 flagged exact-zero half-hours. Replay sets hold 218,689
  (folds 2025-10 and 2025-12), 274,171, 286,075, and 286,075 rows. Fitted networks are in the
  experiment's data folder under `work/models/{WO,RO}`.

## Checks

- **Median minus mean of the 11 quantiles, per group (points; gate rows / Christmas rows):** H0
  -0.847 / -0.189 at the eight and -0.471 / -0.672 at the six; WO and RO differ from H0 by at most
  0.12 on gate rows, and by up to 0.12 at the eight and 0.24 at the six on Christmas rows. No arm
  shifts the forecast in a way the CRPS contrast does not show.
- **Mean absolute error of the median forecast beside each CRPS contrast:** outside the window RO
  is +0.100 at the eight and +0.020 at the six against H0 (WO +0.007, +0.016); on Christmas RO is
  -0.399 at the eight and -0.125 at the six (WO +0.180, -0.144). Every sign agrees with the
  matching CRPS contrast.
- **Audit triggers:** none fired. The prediction failed, but it was written with a stated prior
  against it and the cause (the replay) is what the split tests. RO and WO add up to H2's pattern
  (RO +0.073 plus WO +0.009 against H2's +0.081 outside the window at the eight), so no nested arm
  beats its container.

## Caveats

- One Christmas is scored, so every Christmas number is suggestive. December 2024 is not scored:
  no pretrained weights exist for that cut, and the ensemble history starts 2024-04-01.
- RO takes 1.33 times the optimiser steps per epoch, so an RO effect is a replay effect plus a
  step-count effect. No step-matched control was run.
- Fold 2026-01 trains on Christmas 2025, so its rows for 1 and 2 January are reported but not
  counted in any test.
- Every score is selection-biased: the five folds that choose the arm also score it. July 2026 is
  unused. NGED's event labels are incomplete: "rows with no labelled event" may hold events.
- The serving cost is nil for both arms: each changes the training loop only, and the refit count
  stays at 12 monthly refits a year.
- Code reuse: the scripts import experiment 485's fit loop, regime helpers, scorer, and figure
  code, which in turn import experiments 87, 397, 407, 418, 422, 425, and 470.

## Review of the work

- **Code review (fresh Sonnet agent, before the first fit):** no blocking problems; two wording
  fixes (the curtailment table has no BG2 column; H2 is quoted from experiment 485, not rescored),
  both applied.
- **Results review (fresh Sonnet agent, from the raw stored forecasts):** re-derived the seven
  headline numbers to four decimals, found identical rows and correct seed pairing across arms,
  and found no leakage in the replay set.
- **Prose review (fresh Sonnet agent):** 25 findings. Accepted: the lead's missing definitions and
  its overclaim about the replay (step count not ruled out), the wording of the weight's
  Christmas contrast, and the "network", "costs", "meets", and "which" ambiguities. Rejected: the
  request to explain every statistical term (Bonferroni, day-block bootstrap) inline, because
  experiment 485's page, which this one follows, defines them once.

## What to try next

- Christmas 2026 would give a second scored holiday for H0 and RO.
- A step-matched control and a smaller replay share would show whether the loss outside the window
  comes from the extra optimiser steps.
