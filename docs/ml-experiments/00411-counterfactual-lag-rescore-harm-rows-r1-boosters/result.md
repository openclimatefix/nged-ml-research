---
id: "00411"
slug: counterfactual-lag-rescore-harm-rows-r1-boosters
title: >-
  By how much would the forecast with power lags improve on harm rows if the lags inside each real
  load-shift event were replaced by clean values?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  simple. Stored: one small table of per-row scores in this experiment's private folder; every
  earlier experiment is read and never written. One defensible design: experiment 402's own
  counterfactual harness, scored. Callers: the script copies experiment 402's `real` sweep and
  imports the folders of experiments 87, 391, 392, and 400. Production serving path: not touched
  (this repository holds none). Degradation rule: not touched.
hypothesis: >-
  Replacing the readings inside each real load-shift event with clean values, in the lags that
  experiment 392's R1 lagged XGBoost boosters read, lowers the CRPS of those boosters' forecasts on
  harm rows (rows whose target lies outside every event but whose lags lie inside one) by at least
  0.1 points of effective capacity at the six signed-MW Primaries.
prediction: >-
  Written before any score is read. The gain, real-lag CRPS minus counterfactual-lag CRPS, is
  between 0.02 and 0.15 points at the six signed-MW Primaries, and under 0.05 at the eight MVA-only
  Primaries. Reason: experiment 401 and the experiment 402 step 2 redo found a real event moves the
  median forecast by only 5 to 10% of the step (real over synthetic share ratio 1.05 to 1.10, so
  the factor-2 rule does not fire), so removing the event cannot move the score by much. The
  difference between the forecast with lags (5.119) and without lags (4.292) on those rows is then
  mostly the two forecasts differing as whole forecasts.
decision_rule: >-
  The gain at the six is read with a 95% event-cluster bootstrap interval (2,000 resamples of whole
  events). If the point estimate is under 0.1 points, batches 1 and 2 of experiment 404 are skipped,
  as that plan states. The threshold 0.1 is the one in the plan of experiment 404, chosen there as
  the gain that pays for adding a trust head to a network; this experiment does not set it. If the
  point estimate is at least 0.1 but the interval includes 0.1 or zero, the report says so and
  leaves the choice to the maintainer.
benchmark: >-
  The same boosters with the observed lags (real), on the same rows. The stored CRPS of the weekly
  bundle with and without lags on harm rows (experiment 402 `ceilings3.txt`) is context only.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Exact CRPS of the distribution linear between the 11 quantiles, in points of effective capacity
  (experiments 400 and 402); leads up to 240 hours (day 10).
code: ml-experiments/00411-counterfactual-lag-rescore-harm-rows-r1-boosters/
---

**Take-home message (to be written when the result is known).**

This page is the plan. It is committed before any score exists. It is batch 0 of the
[experiment 404 plan](../00404-mdn-lagless-and-calibration-module-trust-weights/result.md), run as
its own experiment so that the experiment 404 session is not blocked. It needs no neural network,
no GPU, and no fit.

## Question

**We measure the most any lag-trust method can gain on harm rows.** A lag-trust method tries to
down-weight the lagged power readings that sit inside a load-shift event. If the readings inside the
event were replaced by clean values, the forecast could not be misled by them, so the score gain
from that replacement is an upper bound on the gain of any method that does not change anything else
about the forecast. If the upper bound is under 0.1 points, a learned trust weight cannot be worth
building.

## Stored forecasts versus re-prediction

**The stored forecasts cannot be re-scored with replaced lags, so the script re-predicts with the
boosters.** Experiment 402 step 1 read experiment 400's stored forecast files (experiment 381's
weekly bundle, with and without lags). Those files hold forecasts made from the observed lags only.
Replacing the lags means predicting again, which needs the model. Experiment 402's step 2
already re-predicts with experiment 392's R1 lagged boosters, which are on disk, so this experiment
uses those. This experiment reuses that harness and adds a score. The
consequence is that the gain is measured on the R1 boosters, not on the weekly bundle that
experiment 402's other tables use. The report states this, and also states the R1 boosters' own
CRPS with real lags, so that a reader can compare it with the bundle's 5.119.

## Method

- **Events and clean values.** Experiment 400's corrected events per Primary. Inside each event, the
  reading is replaced by experiment 391's stage-1 median plus the mean of (reading minus median)
  over the 7 days before the event. Events whose measured step is under 25% of the labelled depth,
  whose sign disagrees with the label, or whose pre-event bias exceeds half the step are skipped, as
  in experiment 402 step 2.
- **Runs.** For each event and each of the four refit months (2025-10, 2026-01, 2026-03, 2026-06),
  up to 6 daily forecast runs, spread evenly over the runs whose lags touch the event. Each run is
  predicted twice by the refit's booster for that Primary: with the observed series, and with the
  clean series. The actual power is the observed series in both cases.
- **Harm rows.** Rows with at least one of the three lags inside the event, a target time outside
  every event of that Primary, a finite actual, and a lead of up to 240 hours.
- **Scores and contrast.** Row CRPS for both forecasts. The gain is the mean CRPS with observed lags
  minus the mean with clean lags, over all harm rows. The interval resamples whole events.
- **Groups.** The six signed-MW Primaries (D5, D9, D10, D11, D14, D16) carry the decision. The eight
  MVA-only Primaries (D2, D3, D4, D8, D12, D19, D21, D22) are reported beside them. Gain by lead
  band (days 1 to 2, 3 to 5, 6 to 10) and per Primary are reported.

## Caveats written before the run

- **The sample of runs is not the full harm-row set.** At most 6 runs per event and refit month are
  scored, so events with many runs are under-weighted against the full set. Row counts will be
  smaller than experiment 402's 43,024.
- **A row can appear under two events.** Each event is scored on its own, with only that event's
  lags replaced, so a row whose lags touch two events is scored twice, once per event.
- **The replacement is itself an estimate.** The stage-1 median plus a bias is not the true clean
  load, so the gain is an estimate of the upper bound, not the bound.
- **Five test months, one seed.** The R1 boosters are single-seed, so every number is interim.

## Problems caused by the move

None so far. Experiment 402's modules sit on an unmerged branch, so this experiment copies the files
it needs.
