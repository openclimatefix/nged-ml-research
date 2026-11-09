---
id: "00654"
slug: two-output-network-end-to-end-with-end-detector
title: >-
  Does one network that outputs both the normal-running forecast and the prevailing forecast, with a
  learned end detector, beat a frozen K1 plus a separate survival head?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores per-row forecasts, fine-tuned network weights for each arm, fold, and seed,
  and the release-candidate parquet files; (2) more than one design defensibly tests the hypothesis
  (which parts of K1 are fine-tuned, whether the prevailing loss reaches the lag encoder, whether
  the gate is an input); (3) it spans K1's training and injection code from experiments 559, 574,
  591, 612, 613, 647, 650, and 652, whose callers cannot be named without searching; (4) it touches
  no serving path or degradation rule; (5) it changes no contract. Full set of reviews: plan, code,
  results, write-up, prose.
hypothesis: >-
  One network, trained end to end (K1's body plus a prevailing head that reads the lag-encoder
  outputs, with no hand-built gate), gives a prevailing forecast at least as good as the frozen K1
  plus the survival-hazard head R-haz of experiment 652, with no loss to the normal-running-
  arrangement (NRA) output; and a learned end detector that reads the freshest half-hours lets the
  prevailing forecast drop a shift faster after it ends.
prediction: >-
  Written before any fit. E2 (no gate, neighbour summaries, joint training) minus E0 (frozen K1 plus
  R-haz) in scaled CRPS over lead days 1 to 9 where a shift is in force, on unseen training-family
  shifts, is below -0.05 points of capacity (the minimum detectable effect of experiment 650). E2d
  minus E2 is negative in the "ended in the last 24 hours" and "near" strata by at least 0.3 points,
  and is within 0.1 points of zero elsewhere. For shifts shallower than about 5 points of capacity
  the end detector does no better than the fitted survival curve B-haz. The NRA output of each
  end-to-end arm is within 0.03 points of K1 on event-free rows. The placebo for E2d scores within
  0.1 points of E2.
decision_rule: >-
  Primary contrast: E2 minus E0, scaled CRPS averaged over lead days 1 to 9 on origins where a shift
  is in force (far, middle, and near strata pooled), on unseen training-family shifts, injected
  bank. It counts as a gain when the month-block 95% interval (percentile bootstrap over the 5
  development months, seeds averaged within a month) excludes zero in the negative direction and
  the point estimate is below -0.05 points. Second contrast: E2d minus E2 in the "ended in the last
  24 hours" and "near" strata. Third: E2 minus E1. NRA check: each end-to-end arm's NRA CRPS minus
  K1's, on event-free rows and on injected rows scored against the unshifted base, both directions
  reported. One veto: on event-free origins, the upper bound of the 95% month-block interval of
  each end-to-end arm minus K1 (unscaled CRPS) is below 0.03 points of capacity, and the NRA output
  of each arm is non-inferior to K1 on the same rows. Every other contrast is descriptive. The first
  two months (2025-10 and 2025-12) are a feasibility gate, not a test.
benchmark: >-
  E0 (frozen K1 plus R-haz, experiment 652's stored fits), R0 (168 h), B-haz, K1 as context only,
  and, at the single final-test step, the manual heuristic.
row_set_id: dev_months_origins_2025_10_to_2026_06_export_roster
metric: >-
  CRPS (twice the mean pinball loss over the 11 stored quantile levels) in points of capacity after
  equal width scaling, by lead day (1 to 9) and by the state of the shift at the forecast origin;
  q05-q95 coverage; q01, q05, q95, and q99 pinball loss; survival reliability; follow; false-alarm
  rate per Primary-month; detection delay after an injected end; avoidable-curtailment cost.
production_status: not-for-production
adopted_in: none
---
**Status: planned. The sections below are the plan; the finished page replaces them.**

A Primary is a primary substation. A load shift is a period when the electricity network operator
moves part of one Primary's demand onto, or off, a neighbouring Primary. K1 is the neural network of
experiments 612 and 613 that forecasts the normal running arrangement (NRA), the demand with no load
shift. [Experiment 652](../00652-two-output-forecast-with-explicit-rules-and-survival-hazard/result.md)
built a prevailing forecast (the arrangement now in force) as a small head on the frozen K1 and found
that a learned survival output ties a survival curve fitted to shift durations. This experiment asks
whether training one network end to end, with an NRA output and a prevailing output, and then adding
a learned detector for the end of a shift, does better than that frozen K1 plus head.

## Updates to the brief after reading the merged pages

- Experiments 647, 650, and 652 are merged; the reference E0 is experiment 652's R-haz and its
  stored fits. The newest merged experiment, 651, does not change the design.
- The brief asks that the final-test window (2026-07-01 to 2026-10-31) be opened once. The
  `ml-experiment` skill still says no experiment scores a valid time on or after 2026-07-01, so the
  worker messages the manager, and waits for the maintainer's confirmation, before the window is
  touched.

## Feasibility finding that sets the design

K1 is not one network. It is a pre-trained network per held-out month and seed (five seeds), followed
by a daily fine-tune (d0 and d14 cut-offs), stored as `.pt` files under
`data/ml-experiments/00612-*/arms/k1_d5/work/models` (seeds 0 and 1) and
`data/ml-experiments/00613-*/chains/d5/work/` (seeds 2 to 4). Experiment 650 injects shifts by
replacing the lag block of the stored input matrix and running the saved networks. Retraining the
whole chain on injected shifts would repeat experiments 559, 574, 591, 612, and 613, which is not
affordable. The plan therefore **starts each end-to-end arm from the saved K1 weights of the fold and
seed, and fine-tunes jointly on a stored bank of input matrices with injected lags**, as one extra
stage. Everything before that stage is K1 as stored. The cost is that the end-to-end arm differs from
a network trained from scratch with the prevailing loss; the page says so.

## Arms (matched rows, epochs, seeds, folds: leave one development month out, 2 seeds)

- **E0:** frozen K1 plus R-haz (experiment 652's stored fits; reference).
- **E1:** K1's body initialised from the frozen weights and fine-tuned jointly with a prevailing head
  that reads the lag-encoder outputs and the G4 gate state. First variant: stop-gradient from the
  prevailing loss into the NRA output. Second variant: the prevailing loss reaches the lag encoder.
- **E2:** as E1, with no gate and with neighbour summaries (experiment 610's class 2 recipe, 50%
  signal dropout). The gate is not an input and G4 is not applied.
- **E1d, E2d:** E1 and E2 plus the end detector.
- **Placebo for E2d:** E2d refitted with each training row's shift targets, labels, and weights
  permuted among the rows of the same Primary.
- **No-fit references:** R0 (168 h), B-haz, and K1 as context.

## The end detector

The detector takes the freshest half-hours up to the origin of the observed power minus the NRA
forecast, in noise units, with the slow seasonal level removed, the all-Primaries median residual
removed, and the same quantity at connected neighbours (29 edges of `neighbours_nged.json`). Two
extra inputs: the opposite-sign step at a connected neighbour (a score boost, never a requirement),
and a per-Primary routine-step rate from a 60-day context. Its output is the probability that the
shift has ended, multiplying the survival output. Placebo for the neighbour input: a
degree-preserving rewired copy of the edges. Prediction that could fail: below about 5 points of
capacity of depth, the detector does no better than the fitted survival curve. Cost to watch: false
alarms on event-free origins and on origins more than 14 days from any event, per Primary-month.

## Generator

Real unshifted bases; onsets and ends are instantaneous steps; duration log-uniform from 1 to 112
days; depth log-uniform from 3 to 50 points of capacity with constant-MW and proportional shapes;
donor and receiver, hard negatives, held-out family, and start times from logged starts, as in the
brief. **Step 0 measures the real edge width at D5/D16 and D3/D22 before any fit**; if every edge is
within one half-hour, the staged-return arm is dropped. Un-injected time is unlabelled, not negative.

## Strata, scoring, and final test

Strata, lead days 1 to 9, scaling across strata, and the single veto are those of experiment 652,
plus the strata "event-free and more than 14 days from any event" and "ended in the last 24 hours".
The final test freezes candidates in a file before the window is read, then scores E0, the best
end-to-end arm, and the manual heuristic once, in the four scenarios of the brief, with counts of
real events per scenario stated as lower bounds.

## Compute and process

- One GPU process at a time. One fold is timed first. The estimate is 4 to 8 GPU-hours, unmeasured;
  the worker asks the manager before crossing 8.
- Standing practices followed: monthly refit, cleaned data, D15 and D18 cut before 2025-03-01, save
  every fitted network, reuse stored predictions (E0, K1, R0, B-haz), tails and avoidable-curtailment
  cost reported, example forecasts and event time series plotted, draft pull request.
- Reviews: one Sonnet plan review, one Sonnet code review before the first fit, one results review,
  one write-up review, one prose review.
