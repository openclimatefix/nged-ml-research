---
id: "00410"
slug: mdn-stale-weather-and-delayed-power-rescore
title: >-
  How much does the pooled mixture density network lose when the weather forecast is a day old or
  the power readings are delayed?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: one prediction set per arm and a small score table in this experiment's private
  folder; experiment 397's networks and the experiment 389 boosters are read and never written.
  Production serving path: not touched (this repository holds none). Degradation rule: not touched,
  but the result informs one (what the forecast does when the weather forecast or the power readings
  are late). Designs: more than one defensible for the stale inputs (see the arms), so every
  variant is registered. Callers: the scripts import experiments 87, 377, 381, 389, and 397 and the
  `studies` evaluation code, whose callers sit in those folders. No fit runs.
hypothesis: >-
  The stored pooled plain mixture density network of experiment 397 loses little CRPS and little
  tail coverage when its weather forecast is one run (24 hours) older or its power readings are up
  to 24 hours later than in training, because it receives the age of each power lag as an input and
  has a missing-lag path, but loses more than the XGBoost model with the same inputs because it has
  no stale-input training.
prediction: >-
  Written before any score is read. P1. The loss against the fresh case at the signed-MW six is
  under 0.05 points at P6 and P12, and under 0.10 at W24, P24, W24+P12, and W24+P24. I expect P1 to
  hold for W24 and P6 to P12 and to fail for P24 and for the arms that combine W24 with a delayed
  power reading, because the origin lags carry most of the short-lead score. P2. The loss grows
  with the delay and is concentrated at days 1 to 2, where the newest lag matters. P3. Dropping the
  three origin lags (arm DROP) loses more than P24, because the network's missing-lag path was
  trained on few rows. P4. The q95 and q99 coverage at the signed-MW six falls by under 0.02 in W24
  and P6 to P12 and by more in P24. P5. The XGBoost model with mean and spread columns (S1M) loses
  no more than the network in every arm.
decision_rule: >-
  Arms are listed in the plan. The contrast is each stale arm minus the fresh arm on the same rows,
  CRPS at days 1 to 10, per group (signed-MW six; MVA-only eight, with and without D8; D15 and D18
  report-only) and per Primary, by the month-block bootstrap of `studies.eval_protocol` at 95%
  (20,000 resamples). The five screen months give wide intervals, so every number is labelled
  interim. No further experiment is needed if the pooled plain network, in each of the arms P6, P12,
  W24, P24, W24+P12, and W24+P24, loses under 0.05 points (P6, P12) or under 0.10 points (the other
  four) at both the signed-MW six and the MVA-only eight, and its q95 and q99 coverage fall by under
  0.02 against the fresh case. Otherwise a second experiment trains with random staleness and two
  age inputs. These thresholds were proposed by the neural MDN Discuss session without evidence and
  belong to the manager and the maintainer to change. The smallest loss worth a second experiment
  is the threshold itself, because a second experiment adds a training step.
benchmark: >-
  The fresh case of the same networks on the same rows (the same as experiment 397's pooled plain
  network, summary scheme, with the 1.15 widening), and the XGBoost S1M boosters of experiment 389
  with the same stale inputs.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, coverage and pinball loss at q01, q05,
  q95, and q99, and interval width, in points of effective capacity, from `studies.eval_protocol`;
  leads 0.5 to 336 hours; the headline band is days 1 to 10.
code: ml-experiments/00410-mdn-stale-weather-and-delayed-power-rescore/
---

**Take-home message (to be written when the result is known).**

This page is the plan, committed before any score is read.

## Question and design

**We re-score experiment 397's stored networks with stale inputs and fit nothing.** In production
the weather forecast can arrive late, and the power readings arrive every 6 hours, so the forecast
sometimes pairs recent readings with an old weather forecast, or a fresh weather forecast with
delayed readings. Experiment 397 trained its networks with one pairing. A small loss means no
training with random staleness is needed.

### What the networks see (read from experiment 397's code)

- **The lead input is measured from the forecast origin**, not from the weather forecast start. The
  origin is the run's initialisation time (00:00 UTC) plus 9 hours of publication delay. Each daily
  run therefore reaches the network as a 9-hour-old weather forecast at lead 0.
- **The three power lags are computed from the origin.** The newest usable reading is 3.5 hours
  before the origin; the day and week lags are the latest earlier readings at the valid time's
  half-hour and weekday. Each lag has a missing flag and an age input (hours divided by 336).
- **A stale weather forecast looks like a normal forecast with the wrong lead.** The network is told
  the lead from the origin, so a day-old weather forecast matched on valid time arrives as if it
  were 24 hours fresher than it is. This is what a production system that does not tell the network
  the weather forecast's age would do.

### Change from the brief (stored data limits the arms)

- **Only the 00:00 UTC run is stored, one per day.** The brief asks for weather forecasts 6 and 12
  hours older, which do not exist in the stored data (the live service also ingests one run per day).
  The stale-weather arms are therefore one missed run (W24, the previous day's run) and two missed
  runs (W48), matched on valid time. The 12-hour threshold of the brief is applied to the
  delayed-power arm P12 only.
- **The XGBoost comparator is S1M, not S0M.** Experiment 389 stored S1M boosters for all 16 Primaries
  and 12 monthly refits, but S0M boosters for two Primaries. S1M is also the XGBoost model with the
  same one-pass inputs as the summary-scheme network.

### Arms

| Arm | Weather forecast | Power readings |
|---|---|---|
| FRESH | run of the origin | newest reading 3.5 hours before the origin |
| W24 | previous day's run, same valid time | fresh |
| W48 | run two days earlier | fresh |
| P6, P12, P24 | fresh | origin moved back 6, 12, 24 hours (lags and ages recomputed) |
| DROP | fresh | the three origin lags set missing, through the network's missing-flag path |
| W24+P12, W24+P24 | previous day's run | origin moved back 12, 24 hours |

- **Delayed power moves the origin of the lags, not of the lead.** The lead input is unchanged; the
  lag values and ages are those of an origin 6, 12, or 24 hours earlier.
- **Where the older run does not reach the valid time**, the arm scores only the valid times it
  covers, and the fresh arm is restricted to the same rows. The count is reported.
- **Networks:** the pooled plain network (summary scheme, 3 seeds, 5 folds: 2025-10, 2025-12,
  2026-01, 2026-03, 2026-06) and the per-Primary plain network, from `work/models/summary/` of
  experiment 397. The per-Primary network is a secondary report.
- **XGBoost:** the S1M boosters of experiment 389 for the same folds, with the same stale inputs.

### Scoring and reporting

- Same rows, folds, and 1.15 widening of the outer quantiles as experiment 397. Standing practices
  followed: cleaned actuals (the flagged `substation_zero` half-hours dropped, counted per fold),
  and the D15 and D18 start cut at 2025-03-01 applied to the lag inputs and scoring actuals (the
  stored networks were trained before the cut; this experiment does not retrain them).
- Tails and widening: q01, q05, q95, and q99 coverage and pinball loss on stale rows against fresh
  rows, interval width, and whether a simple widening rule would repair a coverage shortfall.
- Figures: CRPS loss against the fresh case for every arm with intervals first, loss by lead, two
  example probabilistic forecasts (a day-ahead and a full 14-day horizon, fresh against W24 and
  P24), and at least two time series charts of the stale arms against the fresh arm.

## Standing practices followed

Monthly refits as stored, cleaned data, the start cut, reuse of stored networks and boosters, the
tails and the example forecasts. The load-shift plots do not apply, because the experiment does not
score load-shift events.

## Problems caused by the move

None so far.
