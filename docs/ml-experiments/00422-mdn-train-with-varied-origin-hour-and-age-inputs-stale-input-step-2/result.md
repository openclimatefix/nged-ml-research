---
id: "00422"
slug: mdn-train-with-varied-origin-hour-and-age-inputs-stale-input-step-2
title: >-
  Does training the pooled mixture density network at many origin hours, with the age of the weather
  forecast and of the newest power reading as inputs, remove its loss from stale inputs?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: fitted networks, forecasts, and a score table in this experiment's private folder
  only; experiments 397 and 410 are read and never written. Production serving path: not touched
  (this repository holds none). Degradation rule: not touched, but the result informs one (what the
  forecast does when the weather forecast or the power readings are late, or the readings are
  missing). Designs: more than one defensible (which delays and origin hours to train on, whether
  the older weather run is shown in training), so the ladder of arms registers each. Callers: the
  scripts import experiments 87, 397, and 410 and the `studies` evaluation code, whose callers sit
  in those folders.
hypothesis: >-
  The pooled plain mixture density network loses CRPS when the newest power reading falls at a
  time of day, or lies further back, than it saw in training, and when the weather forecast is
  older than in training. Training with the forecast origin at many hours of the day, the newest
  reading delayed by a random amount, and the previous weather run sometimes shown, with the age of
  the weather forecast and the age of the newest reading as inputs, removes that loss without
  raising the score on fresh inputs.
prediction: >-
  Written before any fit and before any score is read. The arms are T0 (origin 09:00 only), T1
  (origin hour varied), T2 (T1 plus delayed readings and the two age inputs), T2n (T2 without the
  two age inputs), T3 (T2 plus the previous weather run sometimes), and T4 (T3 plus the three origin
  lags sometimes missing). Losses are CRPS at days 1 to 10 in points of effective capacity against
  the same network's own fresh case, at the signed-MW six. P1. In T1 the P12 loss is under 0.10
  points (experiment 410 found 0.197 at the signed six and 0.151 at the MVA-only eight for the
  unmodified network; the 97% rests on one Primary, D5, one month, and one seed), because the audit
  traced 97% of the P12 loss to the time of day of the newest reading; T1's P6 and P24 losses stay
  under 0.05. P2. In T2 every power-delay arm (P6, P12, P24) loses under 0.05 points, W24 loses
  within 0.02 of T0's W24 loss, and W24+P12 and W24+P24 lose under 0.10. P3. In T3 W24, W24+P12, and
  W24+P24 each lose under 0.05 points, and W48, which lies beyond the training range, loses more
  than W24. P4. At a constant hour of day the loss grows with delay beyond the 24 hours trained on:
  in T2 and T3 the loss at P48 exceeds that at P24, at P36 that at P12, and at P30 that at P6, each
  by at least 0.03 points. P2n. T2n and T2 differ by under 0.02 points in every stale arm at both
  groups, because the delay is already visible to the network as the difference between the lag age
  and the lead; a larger difference means the age input is not redundant. P5. The fresh-case CRPS of
  T1 to T4 (and T2n) is within +0.03 points of T0's at both groups. P6. In T4 the DROP arm loses
  under 2 points (T0: 9.7 at the signed six, 6.5 at the MVA-only eight in experiment 410), and its
  q95 coverage falls by under 0.05 (T0: 0.33 at the signed six, 0.27 at the MVA-only eight). P7. In
  T2 to T4, q95 and q99 coverage fall by under 0.02 in every arm except P30, P36, P48, W48, and (for
  T2 and T3) DROP. P8. T3 loses no more than the XGBoost S1M model of experiment 410 in P12 and
  W24+P12 (S1M: 0.306 and 0.338 at the signed six, 0.098 and 0.130 at the MVA-only eight), a test
  that binds at the MVA-only eight, because T0 already beats 0.306 at the signed six. P9 (off the
  training grid). In T2 and T3 the loss at P10, a delay between the 9 and 12 hours on the 3-hour
  training grid, is within 0.02 points of the mean of the losses at P6 and P12 interpolated by
  delay. P4 and P7 to P9 are reported and sit outside the decision rule.
decision_rule: >-
  Each contrast is read per group (signed-MW six; MVA-only eight, with and without D8; D15 and D18
  report-only) on CRPS at days 1 to 10, with the month-block bootstrap of `studies.eval_protocol` at
  95% and 20,000 resamples, on the half-hours that the arm and its reference share. Planned
  contrasts. C1: each stale arm minus the same network's fresh case (the experiment 410 contrast).
  C2: each of T1 to T4 minus T0 in the same stale arm, which measures what the training change
  gained. C3: the fresh case of each of T1 to T4 minus T0's fresh case, which measures what the
  training change cost. The hypothesis is supported if the primary arm, T3 (T2 and T4 are read the
  same way and named if one of them meets the limits when T3 does not), meets every experiment 410
  limit at both groups (CRPS loss under 0.05 for P6 and P12, under 0.10 for W24, P24, W24+P12, and
  W24+P24, q95 and q99 coverage fall under 0.02) and C3 is under +0.03 points at both groups; partly
  supported if the best of T1 to T4 cuts the P12 loss by more than half at both groups (0.098 at the
  signed six, 0.076 at the MVA-only eight) but some limit still fails at either group, or the limits
  hold and C3 exceeds +0.03; not supported otherwise. Every limit is tested on the point estimate,
  and the page also prints the upper 95% bound beside it. The experiment 410 limits were proposed by
  the planning session without evidence, so the page reports every number beside them and the
  maintainer judges. The five screen months give wide intervals, so every number is interim. The
  smallest change worth porting is any that meets the limits, because the added cost is a
  training-row builder and two input columns, with no new data or model. The XGBoost S1M forecasts
  of experiment 410 are the reference for P8 and are not refitted.
benchmark: >-
  The pooled plain network of experiment 397 as re-scored by experiment 410 (stored FRESH and stale
  forecasts), the XGBoost S1M model of experiment 410, and T0, a refit of the pooled plain network
  at origin 09:00 under the cleaned-data and start-cut rules, which is the control for C2 and C3.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, coverage and pinball loss at q01, q05,
  q95, and q99, and interval width, in points of effective capacity, from `studies.eval_protocol`;
  leads 0.5 to 336 hours; the headline band is days 1 to 10.
code: ml-experiments/00422-mdn-train-with-varied-origin-hour-and-age-inputs-stale-input-step-2/
---

**Plan: can training teach the pooled neural network to cope with late weather and late readings?**
The status is `planned`. Nothing below has been run.

## Question and design

**Experiment 410 found that the pooled plain mixture density network tolerates most stale inputs,
and fails when the newest power reading falls at a time of day it never saw.** Readings delayed by 6
or 24 hours cost under 0.1 points of CRPS, and readings delayed by 12 hours cost 0.10 to 0.31
points, because training always put the newest reading at 05:30. The audit traced 97% of the 12-hour
loss (on the primary substation D5 in 2026-01) to that move from 05:30 to 17:30. This experiment
trains the network so that the newest reading lands at many times of day and lies back by many
delays, and checks whether the loss disappears.

- **Evidence checked on 2026-10-31.** A search of the experiment pages for stale or delayed inputs
  found no page since experiment 410 that stresses a neural network with them. Experiment 388,
  merged on 2026-10-31, asks which lags the XGBoost model needs, not what happens when the readings
  are late. No standing rule in `common.md` changed the brief's design. The brief and this plan
  agree except for four additions, each registered below: the delay of the newest reading is
  randomised, not only the origin hour (otherwise the age of the newest reading is a constant 3.5
  hours and carries no information), the previous weather run is sometimes shown (otherwise the
  weather-age input never leaves its training range in the W24 arms), a control arm T0 is refitted
  under the current cleaning rules, and an arm T4 trains with missing origin lags, which experiment
  410 named as the next step.
- **The base network.** The pooled plain network of experiment 397 (`plain_pooled`, the same
  setting as its stored meta files, 3 Gaussian components, the Primary embedding, the ECMWF ENS mean
  and standard deviation passed through once), on the five screen folds (test months 2025-10,
  2025-12, 2026-01, 2026-03, 2026-06), monthly refits on the 24 months before the refit, three
  seeds. Experiment 397's code is imported, and its training loop, early stopping on every
  twelfth week, and 1.15 widening are unchanged.
- **How the training rows change.** Experiment 397's rows hold one (run, valid time) pair per
  Primary with the origin fixed at 09:00 UTC (the 00:00 UTC run plus the 9-hour publication
  delay). Each arm below redraws, per row and per seed, an origin offset *h* after the run's start,
  a delay *d* of the newest reading, and whether to use the previous run. The weather columns of
  the row are those of the run chosen. The lead, the three origin lags, their ages, and the
  missing flags are recomputed from the drawn origin with `e00087_common.origin_lags`, so the
  newest usable reading is at `origin - 3.5 h - d`. A row survives only if its valid time is
  after the drawn origin.
- **The two new inputs.** The age of the weather forecast is the hours from the weather run's start
  to the forecast origin, divided by 48. The age of the newest power reading is the hours from the
  reading's timestamp to the forecast origin (3.5 plus *d*), divided by 48. The reading age is not
  new information: the three lag-age inputs of experiment 397 equal the lead plus 3.5 hours plus
  *d* (the training rows and the test arms both build them that way, as experiment 410 does), so
  the network can already read *d* as the lag age minus the lead. The input is a convenience, and
  arm T2n (T2 without either age input) separates what the delays in training add from what the
  inputs add. In T1 and T2 the weather age is a function of the origin hour, so it works as a
  time-of-day input there. When the three origin lags are missing (arm T4 in training, the DROP
  test arm), the reading age keeps its true value (3.5 plus *d*) and the three lag-missing flags
  say the readings are absent. No weather-missing flag is added, because no arm removes the weather
  forecast.
- **How the rows are drawn.** The origin offset, the delay, the run, and the lag dropout are drawn
  once per row and seed, so each (run, valid time) pair keeps one draw for the whole fit. The
  held-out rows that stop training early use the same draws as the arm's training rows, so each
  arm stops on its own distribution; the early-stopping score is therefore not comparable across
  arms, and the contrasts use the test-month scores only.
- **Cleaned data and the start cut.** Every arm's lags come from the cleaned grid (the flagged
  `substation_zero` half-hours missing), flagged half-hours are dropped from the training targets,
  and D15 and D18 rows before 2025-03-01 are dropped from training rows, lag inputs, and scoring
  actuals. The count of dropped half-hours is printed per fold.
- **Leakage guards.** A row's valid time must be at or before the refit minus 3.5 hours and after
  its drawn origin, so no lag, weather run, or target comes from after the refit. The purge around
  each held-out week (14 days before it, 7 days after it in experiment 397) becomes 15 days
  before and 9 days after: a run reaches valid times 14.4 days past its start, and a delay of up to
  24 hours pushes a lag a day further back. A check before the first fit asserts, on a sample of
  rows, that no lag index is later than the newest usable reading and that a drawn origin never
  precedes the run's publication (09:00).

## Arms

**Training arms.** Origin offsets are in hours after the 00:00 UTC run start; each draw is
uniform over its set and independent of the others.

| Arm | Origin offset h (hours) | Reading delay d (hours) | Previous run | Origin lags missing | Age inputs |
|---|---|---|---|---|---|
| T0 | 9 | 0 | never | as in the data | not added |
| T1 | 9, 12, 15, 18, 21, 24, 27, 30 | 0 | never | as in the data | not added |
| T2 | 9, 12, 15, 18, 21, 24, 27, 30 | 0, 3, 6, 9, 12, 15, 18, 21, 24 | never | as in the data | added |
| T2n | as T2 | as T2 | never | as in the data | not added |
| T3 | as T2 | as T2 | with probability 0.25 | as in the data | added |
| T4 | as T2 | as T2 | with probability 0.25 | all three, with probability 0.10 | added |

An offset of 24 or more is an origin on the next calendar day, before that day's run is published
(its run is available at offset 33). A row that uses the previous run has a weather age 24 hours
older, up to 57 hours, and needs the previous run to cover the valid time; where it does not, the
fresh run is used and the age input says so. T1 has the original inputs and tests the time-of-day
explanation alone. T2 adds the two age inputs and the delays. T3 adds the older weather. T4 adds
sometimes-missing lags, which tests the degradation case directly.

**Test arms.** Every fitted network is scored at the experiment 397 test origins (09:00), on the
arms of experiment 410 (FRESH, W24, W48, P6, P12, P24, DROP, W24+P12, W24+P24) and four more that
land on the same hour of day as P6, P12, and P24: P30, P36, P48, and P10 (an off-grid delay). P6,
P12, P24 put the newest reading at 23:30, 17:30, and 05:30; P30, P36, and P48 put it at those hours
one day further back, so a comparison of P6 with P30, of P12 with P36, and of P24 with P48 isolates
the delay from the time of day. The network sees the lead and lag ages of the delayed origin as in
experiment 410, plus the two new ages (weather age 9 hours for a fresh run, 33 for W24, 57 for W48;
reading age 3.5 plus the delay).

## Reuse and compute

**Stored forecasts reused.** Experiment 410's stored forecasts (the stored experiment 397 pooled
network, `FRESH` and the stale arms; the XGBoost S1M model) are the references for P8 and for a
reproduction check: T0's FRESH should land within seed noise (0.03 points) of the stored network's
FRESH, which differs only by the cleaning and start-cut rules. No XGBoost model is refitted, and
the extra arms P30, P36, and P48 are scored for the neural networks only. T0 is refitted because
the stored network trained on unfiltered power rows and without the D15 and D18 cut, so a contrast
against it would mix those rules with the training change.

**Compute.** 6 arms × 5 folds × 3 seeds = 90 fits. The experiment 397 estimate is 2 to 4 minutes
per pooled fit on the GPU. One fold's training matrix is about 1.6 GB (float32), so one GPU process
fits under 8 GB of video memory, and the row builder works in float32 and in chunks to stay under 8
GB of host memory (the first timing reports peak resident memory). Building the rows for an arm adds
numpy work, which the first task after the slot is granted times (one fit of T3 on one fold, with
the build time reported). Prediction scores 12 test arms × 5 networks × 3 seeds × 5 months on the
GPU in batches; experiment 410 needed about 6 minutes per month for 10 arms on CPU. The slot is one
GPU process under 8 GB, one feeder core, `MemoryMax=8G`, `OMP_NUM_THREADS=1`; the stop line is 12 GB
in the available column of `free -g`.

## Standing practices

- **Followed:** monthly refits; cleaned data with the flagged-half-hour count per fold; the D15 and
  D18 start cut; every fitted network saved (`.pt` state files and the standardiser, in the
  experiment's private folder); stored forecasts reused as references; a screen on the five folds of
  experiment 397 labelled interim; tails (q01, q05, q95, q99 coverage and pinball loss) beside
  CRPS; the nulled-input widening report (interval width and coverage on stale rows against fresh
  rows, and a simple widening rule if coverage falls short); two example forecasts per season (a
  day-ahead and a 14-day horizon) under the fresh and 24-hour stale arms.
- **Not applicable:** load-shift event plots (the experiment scores no load-shift events); the
  avoidable-curtailment cost (a stress test of inputs, scored on CRPS and tails, as experiment 410).
- **Past-weather caveat:** all weather is the ECMWF ENS mean and spread of the 00:00 UTC run, which
  exists at forecast time. No ERA5 is used.

## Problems caused by the move

None so far.
