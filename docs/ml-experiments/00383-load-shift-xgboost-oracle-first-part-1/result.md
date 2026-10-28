---
id: "00383"
slug: load-shift-xgboost-oracle-first-part-1
title: >-
  Do nulled load-shift lags and wider bands fix the weekly XGBoost model's tails after a load shift?
theme: lsc-detectors
also_relevant_to:
  - lagged-power-and-error-correction
  - calibration-and-uncertainty
  - curtailment-cost
status: planned
kind: other
protocol: demand-v1
size: >-
  complex. Stored: forecasts and small result files in this experiment's private folder only;
  experiments 87, 371, 379, and 381 are read and never written. Production serving path: not
  touched (this repository holds none). Degradation rule: not touched, but arm c2 tests a lagless
  fallback and arm W tests widening on missing inputs, which are candidate degradation rules for a
  later port. Designs: more than one would test the hypothesis (nulled lags with dropout, or a lagless
  fallback; a fixed or an as-of widening factor); the plan fixes all of them before any score is read.
  Callers: the fit script imports experiment 87's folder and copies experiment 100's pseudo-event
  recipe; the scorer imports experiment 377's folder and `studies.eval_protocol`.
hypothesis: >-
  On the rows whose power lags lie inside a hand-labelled load-shift event, the weekly XGBoost
  demand model's 5% to 95% band covers about 0.80 of the readings against 0.90 nominal. Nulling
  those lags (with 30% random lag dropout in training) or serving a lagless model on those rows, plus
  widening the band, brings the coverage within 0.03 of 0.90 and lowers the q95 and q99 pinball loss
  by 10% or more, without harming rows outside events.
prediction: >-
  Written before any score is read. P1. No arm moves the pooled CRPS of days 1 to 10 by more than
  0.01 points in either group (about 2% of the main-score rows have a contaminated lag, so the gain
  cannot reach the protocol's 0.02-point bar). P2. Arm c1 nets slightly worse than arm d over all
  rows, from the dropout harm on ordinary rows (experiment 100 found +0.035 on 98% of rows). P3. Arm c2
  equals arm d on every row without a contaminated lag, bit for bit. P4. On the contaminated-lag
  rows, arm d's 5% to 95% coverage is about 0.80 (experiment 100: 0.799); widening by k = 1.28 lifts
  it to 0.87 to 0.92 and changes the CRPS by under 0.1 points. P5. Widening arm d alone, with no
  nulling, gets most of the coverage gain. P6. Gate O fails on the CRPS test and passes or fails
  on the tail test with a wide interval, because the contaminated rows sit mostly at D5 and D16
  in two long events over Christmas.
decision_rule: >-
  Descriptive, no adoption claimed. Gate O (decides whether the detector arm b2 runs) passes when
  all five hold, with the contrasts read at 95% by a month-block bootstrap, Bonferroni-corrected
  across the arms run: (1) the better of (c1+W) and (c2+W), minus d, on the contaminated-lag rows
  has a CRPS interval wholly below zero, or a mean q95 and q99 pinball loss that falls by 10% or more
  with an interval excluding zero; (2) its 5% to 95% coverage on those rows is within 0.03 of 0.90;
  (3) the main headline of each group stays within 0.02 points on a one-sided 95% bound; (4) no
  Primary is worse than 0.05 points on its point estimate; (5) the placebos D2, D9, D11, and D21 are
  non-inferior (arms c2 and W bit-identical to d there). The widening rule W wins when it raises
  coverage on nulled rows towards nominal without raising the pinball loss on those rows by more than
  0.05 points. Arm b2 wins when it recovers at least 50% of the oracle gain on the contaminated-lag
  rows, marks at most twice the hand labels' half-hours, and passes the same harm conditions, with
  D8 and the placebos included. Stop rules: stop if d fails its reproduction gate (within 0.01
  points of the stored weekly member on shared rows); stop after the c arms if gate O fails; stop b2
  if its marked half-hours exceed 3 times the hand labels' count in the tuning months, if the
  known-answer gate fails, if pooled false alarms at the placebos exceed 4 per Primary-year, or if
  it fails the D8 or placebo checks.
benchmark: >-
  Arm d (the weekly member of BG2, experiment 87), BG2, `manual_heuristic`, and `climatology`, on
  `fold_a_daily_gap_v1`, all 16 Primaries to 336 hours.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles, pinball loss and coverage at
  the 0.01, 0.05, 0.95, and 0.99 quantiles, band width and coverage on nulled against un-nulled
  rows, twCRPS, and avoidable-curtailment cost, in points of effective capacity, from
  `studies.eval_protocol`; headline band days 1 to 10; the pre-registered headline rows are the
  contaminated-lag rows.
code: ml-experiments/00383-load-shift-xgboost-oracle-first-part-1/
production_status: not-for-production
adopted_in: none
date: 2026-10-28
---

# Nulled load-shift lags and wider bands in the weekly XGBoost model

**Plan (status `planned`).** The question is whether telling the weekly XGBoost demand model which of
its recent-power inputs lie inside a load-shift event, and widening its band on those rows, repairs
its tails after a load shift. Hand-labelled events touch about 2% of the main-score rows, so the
experiment cannot move the pooled score. It can show whether the tails and interval widths on the
contaminated rows improve. A positive answer would make a detector worth building. A negative answer
leaves the detector deprioritised.

The design follows `briefs/opus_load_shift_detector_review_answer.md` (the experiment brief at its
end) and `briefs/opus_detector_distillation_answer.md` (the detector). The nulled-lag and dropout
recipe is experiment 100's, the lagless model is experiment 371's `no_lags` arm, the detector is
built on arm A of experiment 327 with the neighbour veto of experiment 315, and the scoring is
experiment 379's.

## Sizing (the five triggers)

- **What gets stored:** forecasts and small result files in this experiment's private folder only.
- **Production serving path:** not touched; this repository holds none.
- **Degradation rule:** not touched, but c2 and W are candidates for one.
- **More than one defensible design:** yes (nulled lags or lagless fallback; fixed or as-of widening
  factor). The plan fixes all of them.
- **Callers unknown without searching:** no. The scripts import experiments 87, 100, 298, 315, 327,
  371, 377, and 379, whose callers are all in their own folders.

## Arms

All arms use experiment 87's `n1h` weekly recipe (the weekly member of BG2), seed 0,
`colsample_bytree = colsample_bynode = 1.0`, ENS control-member training, and weekly retraining. All
16 Primaries are scored to 336 hours under `demand-v1`.

- **d, baseline.** No nulling. Reproduction gate: within 0.01 points of the stored weekly member
  (experiment 379 `only_W`) on shared rows. Experiment 381's own weekly baseline arm is not stored
  yet, so this experiment refits d with the same code and reads experiment 381's code only.
- **c1, oracle, lags nulled.** Training targets inside a hand-labelled event are dropped. Each of the
  three power lags is set to missing where its time lies inside an event, in training and at test
  time. Training adds 30% random pseudo-event lag dropout (experiment 100's `drop30`).
- **c2, oracle, lagless fallback.** A lagless weekly model (experiment 371's `no_lags` recipe, event
  targets dropped) is served on every row with any lag inside an event. Every other row is arm d,
  unchanged. A script check asserts that c2 equals d on those rows.
- **W, widening.** Post-processing with no refit, on the nulled or fallback rows of c1 and c2 (and of
  d, for comparison). Each quantile's distance from the median is multiplied by k = 1.28 (the ratio of
  z(0.95) = 1.645 to z(0.90) = 1.28). The k is labelled "tuning-adjacent", because experiment 100
  looked at the same year. A second, as-of k sets each month's k from the coverage on nulled rows in
  all earlier scored months, capped at 2, with k = 1.28 until any exist.
- **b2, detector (runs only if gate O passes).** One arm replaces the brief's (b) and (b2). It is the
  12-hour mean of each reading's position in the lagless model's quantile bands, with the neighbour
  veto of experiment 315. The exact rule, the thresholds, and the known-answer gate are fixed in the
  section "Detector b2" below before any score is read.
- **a, target nulling only.** Optional, last, descriptive. Experiment 75 measured +0.004.

## Rows, groups, and slices

- **Event labels.** The padded hand-filtered `switching_mask` in the DP feature cache, read through
  experiment 377's `adapter.events_from_masks`. In 2025-07 to 2026-06 it holds 33 Primary-events: at
  D3, D4, D5, D10, D12, D14, D16, D19, D22, D15, and D18. D5 has a 42-day event and D16 a 17.5-day
  event, both over Christmas.
- **Targets:** D3, D4, D5, D10, D12, D14, D16, D19, and D22. **Pre-registered placebos** (no events
  in the scored year): D2, D9, D11, and D21. **Harm watch for b2:** D8, reported apart and not
  exempted. **Report-only:** D15 and D18.
- **Headline rows.** Real rows outside the hand-filtered events, days 1 to 10, with at least one of
  the three lags inside an event (the contaminated-lag rows). The signed-MW six and the MVA-only eight
  are read separately.
- **Metrics on the headline rows.** CRPS; pinball loss and coverage at 0.01, 0.05, 0.95, and 0.99;
  mean width and coverage of the 5% to 95% and 1% to 99% bands, on nulled against un-nulled rows, per
  arm, group, and Primary.
- **Also reported.** The full `demand-v1` report for every arm (per-Primary table, worst Primary,
  leave-one-out, groups, lead bands to 14 days, cold nights, Economy 7 step hours, sunny daylight,
  Christmas, the event slice, the 7-days-after slice, the four tail measures), with BG2,
  `manual_heuristic`, and `climatology`. The avoidable-curtailment cost during each event, in the 7
  days after it ends, and across all rows, per group and per Primary (signed-MW only; the MVA-only
  eight carry "not defined"). The take-home names the tail result and the curtailment cost.
- **Event-block views.** A per-event table and leave-one-event-out contrasts for the D5 and D16
  events; every contrast with and without 20 December to 10 January; the gain by days since onset
  (0 to 1, 1 to 3, over 3); the number of events counted, not only half-hours.
- **Injected NRA scoring (report-only, aim 2).** The stored injected transfers of experiments 88 and
  100, families A and B, scored against the unmodified series, for d, c1, c2, and b2.

## Detector b2

Fixed in full in this section before arm b2 starts, following the section "Design for tonight" of
`briefs/opus_detector_distillation_answer.md`. Arms d, c1, c2, and W do not depend on it.

## Process notes

- Experiment 381's weekly baseline arm is not stored; arm d is refitted here from experiment 87's
  code, which experiment 381 also imports. Experiment 381's folder is never written.
- Sonnet only, including sub-agents. At most 3 threads, `OMP_NUM_THREADS=1` per process,
  `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`, and `free -g` before each
  launch. No production code is changed, and nothing is fetched. No metered generator series is named.
