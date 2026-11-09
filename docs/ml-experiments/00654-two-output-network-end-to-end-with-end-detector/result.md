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

## Results so far

**Status: development-month fits and the first final-test scoring are done. The end detector, E0 on
the final-test window, the Step 0 edge widths, and the figures are not run.** Numbers are CRPS in
points of capacity. "J1" is the end-to-end network whose prevailing loss does not reach the lag
encoder (stop-gradient); "J2" lets the prevailing loss reach the lag encoder. Neither reads the G4
gate state. "C0" is the matched-compute control with no prevailing head.

**On the development months, the end-to-end networks lose to the frozen K1 plus R-haz (E0) while a
shift is in force.** J2 minus E0 is +1.74 scaled points over the in-force strata (month-block 95%
interval +1.00 to +2.47), so the predicted gain did not appear. J2 beats J1 by 0.32 points (interval
-0.54 to -0.05), and both beat the matched-compute control C0.

**The NRA veto fails on the development months.** On injected rows scored against the unshifted
base, the NRA output of J2 scores 3.90 in the far stratum against 3.76 for K1 (+0.14). On event-free
rows J2's NRA output scores 3.81 against 3.77 (+0.04). Both exceed the 0.03 limit, so the veto is
reported as failed for the variant that lets the prevailing loss reach the lag encoder. The cause is
the joint fine-tune recipe (cuts at day 0 and day 14, synthetic shifts), which is weaker than the
daily fine-tune of the stored K1.

**J2 does better than E0 near the end of a shift.** In the "ended in the last 24 hours" stratum J2
scores 6.52 and E0 scores 7.44 on the scaled measure, and in the "near" stratum 6.44 against 7.34.

| Stratum (origin state) | K1 | R0 (168 h) | B-haz | E0 | C0 | J1 | J2 NRA | J2 prevailing |
|---|---|---|---|---|---|---|---|---|
| Far from the end | 10.50 | 7.24 | 5.86 | 5.94 | 10.60 | 8.79 | 3.87 | 8.41 |
| Middle | 7.79 | 6.36 | 6.75 | 6.55 | 7.90 | 7.32 | 3.76 | 7.06 |
| Near the end (24 h) | 5.93 | 5.89 | 7.41 | 7.34 | 5.94 | 6.66 | 3.84 | 6.44 |
| Ended in the last 24 h | 5.95 | 6.08 | 7.83 | 7.44 | 5.99 | 6.69 | 3.85 | 6.52 |
| Event-free | 5.50 | 4.40 | 4.19 | 3.90 | 5.86 | 5.02 | 3.78 | 4.77 |

Table: scaled CRPS on the injected bank, training family, 5 development months, seeds averaged.
These are injected shifts on real bases, not real events. The NRA column is scored against the
unshifted base, so it is not comparable with the other columns. The unscaled values and the
normalised mean absolute error of the median sit in `score/tables.json`.

### Final-test window, scored once for J2, J1, C0, K1, and the manual heuristic

The frozen candidates are in `final/freeze.json` (hashes included). The window has no labelled
events in the export, so "inside" means rows where the hand-built G4 gate is on: five spells, of
20 to 121 days (median 37 days), so these are long detected states, not injected events. Lead 24 to
240 h; intervals are blocked by origin month (4 months) and are coarse.

| Rows | Count | K1 | Manual heuristic | J2 NRA | J2 prevailing | J1 prevailing | C0 |
|---|---|---|---|---|---|---|---|
| All | 837,868 | 4.98 | 4.86 | 4.82 | 5.05 | 5.15 | 4.84 |
| Gate off, 7 or more days from a spell | 694,709 | 4.11 | 4.43 | 3.94 | 4.45 | 4.54 | 3.98 |
| Gate on | 137,676 | 9.40 | 6.97 | 9.30 | 8.11 | 8.29 | 9.20 |
| Gate off, under 7 days after a spell | 5,483 | 4.36 | 5.28 | 4.42 | 4.46 | 4.48 | 4.40 |

The "after a spell" row holds a single month and one spell end, so it carries almost no weight.

**Prevailing is not best on all rows, and the most likely reason is the false-alarm cost on rows
with no gate.** On the 86% of rows with the gate off, J2's prevailing output scores 4.54 against 4.05
for its NRA output. Its median differs from its NRA median by 2.75 points on average there, with a
mean survival probability of 0.15, so the head carries a shift that is not in force. On gate-on rows
that really depart from K1's median by 3 points or more, the prevailing output beats the NRA output
(8.97 against 10.56). On the 12% of gate-on rows that do not depart (1.68 against 1.41) it is
worse.

**The manual heuristic wins on gate-on rows because the shift outlasts its lags.** The heuristic is
the median-style quantile mix of the observed power 1 to 6 weeks and 49 to 55 weeks earlier. For
gate-on rows, 94% of the 1-week members and 76% of the 6-week members lie inside the same gate-on
spell, so the old lags already hold the shifted level. The spells last 20 to 121 days, which is
close to the long half of the injected durations (28 to 112 days). The other half of the injected
durations is 3 to 14 days, which no lag of a week or more can follow; on the window the old lags
help for the reason the maintainer suspected, because the real spells are long.

## Updates to the brief after reading the merged pages

- The maintainer approved opening the final-test window once for this candidate set on 2026-11-09
  (the brief records it). The freeze-first rule and the message to the manager still apply.
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
seed, and fine-tunes jointly on input matrices with injected lags**, as one extra stage. Nothing stores
those matrices: experiment 591's `run_inject` rebuilds them per month (`_build_month_p5`,
`build_test`) and overwrites the lag block with the injected power, so this experiment adds a cache
of them on disk, built with the fold month's standardiser. Everything before that stage is K1
as stored. The cost is that the end-to-end arm differs from a network trained from scratch with the
prevailing loss; the page says so.

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

## Fold structure and leakage

- K1's weights are keyed by (held-out month, daily cut, seed). The joint fine-tune uses the other
  four development months for training and early stopping (origin days 0 to 17 train, 21 and later
  validate, as in experiment 650); the scored month's bank passes and rows never enter either.
  Fine-tuning from fold-month weights on later months is accepted, as in experiment 652's heads,
  and the page states it.
- K1's own daily fit uses month-m rows before its cut. The joint fine-tune trains on no month-m
  row, so no row has a valid time at or after an origin it was trained under.
- Equal-width scaling is pooled across strata, one factor per arm, month, and seed, as in 652.
- The end detector's "observed minus NRA forecast" for past half-hours comes from the NRA forecast
  issued at an earlier origin for that half-hour (a frozen K1 output), never from the network
  scoring its own training rows. Whether the freshest half-hours exist at the origin in production
  is stated on the page; the detector's gain is a ceiling if they do not.
- Detector training rows contain the hard negatives (D8-style routine steps, pulses under 3 hours,
  ramps, placebo pairs, common-mode perturbations) at 30% of injected rows, and detection is
  reported separately on them.
- Mirrored receiver shifts are not in K1's inputs, as in experiments 647 and 650.

## Staging and the fit count

The full design is about 5 months x 2 cuts x 2 seeds x 7 arms, roughly 140 joint fits. Stage 1
fits E2 and E2d only, at the d0 cut, one seed, on 2025-10 and 2025-12, and times them; the
head-only end detector is first fitted on the frozen body, as in experiment 652. Stage 2 adds E1
(both variants), the second seed, and the other months if E2 minus E0 passes the feasibility gate:
**E2 is no worse than E0 by more than 0.1 points and the veto bound holds on both months.** The
placebo runs only if E2d shows an effect. The d14 cut is added only if d0 results are not
cut-dependent in the stage 1 timing check. The manager is told the staging; the GPU-hour
estimate is restated from the timed fold against the fit count before 8 hours are crossed.

## Added controls and selection rules

- **Matched-compute control C0:** K1 fine-tuned the same way and for the same epochs with only the
  NRA loss and no prevailing head, so that E2 minus E0 is not read as a gain from extra epochs.
  E2 minus E1 is the clean read on dropping the gate; the neighbour-summary inputs are new to the
  pretrained body, and the page says so.
- **Second contrast bar:** E2d minus E2 counts as a gain in a stratum when its month-block 95%
  interval excludes zero and the point estimate is below -0.3 points there; elsewhere it is
  descriptive.
- **Best end-to-end arm for the final test:** the arm with the lowest pooled in-force scaled CRPS on
  development months among those that clear the veto, fixed before the window is read.
- The NRA prediction (within 0.03 of K1) is tested as the veto's interval bound, not as a point
  estimate.

## Deliverables

- Figures (at most two panels each): headline loss by lead day and stratum; the time series of
  observed power, K1, the NRA output, and the prevailing output from 14 days before to 7 days after
  the verified D5/D16 pair and a few injected events; survival reliability on training-family and
  held-out durations; the end-detector figure (delay by depth, false alarms); the Step 0 edge-width
  figure; the four final-test scenario plots; day-ahead and 14-day example forecasts.
- Sensitivity: short-heavy and long-heavy duration mixtures, with the sign of the primary contrast
  under each; results by shape; the in-event level drift on a share of events; the staged-return arm
  unless Step 0 finds pure steps.
- Release candidate: one parquet per arm under
  `data/ml-experiments/00654-*/release_candidate/` with the brief's columns, and a plain-words
  README in that folder.
- The Method section lists K1's body widths, read from experiments 545, 560, 591, and 612. The
  manual-heuristic forecasts for the final-test origins are looked up first and computed if absent.
- One line to the manager at: plan PR open, fits started, fits finished, final-test scored, PR
  merged.

## Priority and reporting (manager, 2026-11-09)

- The maintainer needs backtest forecasts on disk by the end of 2026-11-09, so the release-candidate
  parquet files and README come first. The first file is built from E0's stored outputs (experiment
  652's R-haz fits and K1's rebuilt quantiles on the real rows); each end-to-end arm adds its own
  file when its fits finish.
- Every metric table also reports the normalised MAE of the median forecast (MAE divided by the
  Primary's effective capacity, as the other tables do).

## Four scenarios on the development months too (skill, PR 314)

The ml-experiment skill (merged 2026-11-09) requires four scenarios for any experiment related to
load-shift events, whatever the testing period: all rows; rows outside events and at least 7 days
after one; rows inside events; and the 7 days after an event ends. The development-month tables
therefore report all four scenarios for each arm, on the injected bank (events known exactly) and,
descriptively, on the real labelled and detected events; the final test reports the same four. Each
scenario gets its metrics (CRPS, normalised MAE of the median, coverage, tails) and an example time
series from 14 days before to at least 7 days after an event.

## Compute and process

- One GPU process at a time. One fold is timed first. The estimate is 4 to 8 GPU-hours, unmeasured;
  the worker asks the manager before crossing 8.
- Standing practices followed: monthly refit, cleaned data, D15 and D18 cut before 2025-03-01, save
  every fitted network, reuse stored predictions (E0, K1, R0, B-haz), tails and avoidable-curtailment
  cost reported, example forecasts and event time series plotted, draft pull request.
- Reviews: one Sonnet plan review, one Sonnet code review before the first fit, one results review,
  one write-up review, one prose review.
