---
id: "00681"
slug: extra-lagged-power-inputs-for-the-prevailing-forecast
title: >-
  Do extra lagged power inputs make the prevailing output of the two-output demand network react
  faster to a load shift, without worsening the NRA output?
theme: lagged-power-and-error-correction
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: change
protocol: demand-v2
size: >-
  Medium. (1) It stores per-fold head weights and one private Parquet file of predictions per arm;
  (2) more than one lag set is defensible, so the sets are fixed here and checked with the designer
  session before any fit; (3) it spans code from experiments 647, 650, 652, and 667, whose callers
  cannot be named without searching; (4) it touches no serving path and no degradation rule; (5) it
  changes no contract.
hypothesis: >-
  Giving the prevailing head of experiment 667's Candidate more lagged power inputs, newer than the
  24-hour and 7-day lags it reads today, shortens the time the prevailing output takes to cover half
  of a new load level after a confirmed load shift starts.
prediction: >-
  Written before any fit. P1: the best lag arm covers half of the new level at least 24 hours
  sooner than the baseline on the verified D5 and D16 transfer (181 and 52 hours in experiment
  668), and 12 hours sooner on the injected events of experiment 660. P2: the NRA output is
  identical to the baseline's, because the head is a separate network on frozen Candidate
  quantiles. P3: the share of event-free rows on which the prevailing output departs from the NRA
  output by 3 points of capacity or more rises by at most 1 percentage point. P4: the placebo arm,
  with the same number of lag inputs taken from an unrelated Primary, gains less than a third of
  the best arm's reaction-time gain.
decision_rule: >-
  Support needs all of: the best arm's reaction-time gain over the baseline, with a 95% interval
  over events, excludes zero on the injected events and has the right sign on both verified
  transfers; the NRA CRPS equals the baseline's to four decimals; the false-alarm share and the
  prevailing CRPS on event-free rows exceed the baseline's by no more than the baseline's own
  seed-to-seed spread; and the placebo gain is below a third of the arm's gain. The smallest gain
  worth a port is 12 hours of reaction time at no cost on event-free rows, because the extra
  inputs add a column block to the head and nothing else. Any failed condition is reported plainly,
  with no rescue tuning.
benchmark: >-
  Experiment 667's stored Rhaz head (the baseline arm L0) on the same folds, rows, and seeds, and
  the placebo arm.
row_set_id: backtest_year_2025_11_to_2026_06_daily_12utc
metric: >-
  Time for the prevailing output to cover half of the new level after a confirmed load shift starts
  and ends; CRPS, normalised MAE of the median, and q05 to q95 coverage by lead band; the share of
  event-free rows with a 3-point departure from the NRA output.
---

**This plan tests whether more lagged power inputs let the prevailing output follow a load shift
sooner.** Experiment 668 measured that the prevailing output of experiment 667's Candidate took 181
hours (D5) and 52 hours (D16) after the start of the verified transfer to move half-way. This
experiment changes only the inputs of the prevailing head.

## Design

- **Baseline (L0).** Experiment 667's Rhaz head with 14 daily hazards, its quantiles, folds,
  training windows, pre-training chain, seeds, and exports, as stored. Nothing in
  `ml-experiments/00667-*` is edited; the new folder copies `p667_data.py`, `p667_fit.py`, and
  `p667_head.py` and adds a lag block to the head's origin features.
- **Why the NRA output cannot change.** The NRA output is the Candidate's seed-mean quantiles. The
  prevailing head is a separate network trained afterwards on those frozen quantiles, so each arm
  re-fits the head only. The write-up still reports the NRA score next to the baseline, and the
  check is an equality test of the stored NRA table.
- **Arms.** Each extra lag is a residual in points of capacity: the cleaned power at that time
  minus the Candidate's lag-free median for the same time, as the head's existing gap columns are.
  Every lag ends at least 6 hours before the origin.
    - L1: power at 6, 9, 12, 18, 24, 36, 48, and 72 hours before the origin (8 inputs).
    - L2: L1 plus rolling means of the last 6, 12, and 24 hours, each ending 6 hours before the origin
      (11 inputs).
    - L3: L2 plus the differences between consecutive daily means for the last 3 days (14 inputs).
    - P: placebo with the width of the best of L1 to L3: its lag inputs come from another Primary of
    the same metering group (a fixed permutation, no Primary maps to itself).
- **Bank rows.** The injected shift is applied to the new lags exactly as to the target, from the
  same shifted power series. The residual's lag-free median is not shifted, because the lag-free
  network reads no power. The code checks that bank shifts start or end 6 to 72 hours before some
  origins, or the new lags cannot be learned.
- **Delay assertion.** The code asserts, for every row, that each lag time, rolling-mean end, and
  difference input is at most origin minus 6 hours.
- **Veto (designer, 2026-11-10).** Experiment 663's end detector read the freshest half-hours and
  worsened the prevailing forecast (+0.30 in force, +0.20 net). The one veto here is an event-free
  CRPS cost against L0 above 0.03 points of capacity. Scores are reported in the strata event-free,
  event-free more than 14 days from any event, in force, near the end, and ended in the last 24
  hours, with the off-shift false-alarm rate.
- **Folds.** The development folds of experiment 667: 2025-11 to 2026-06. Folds 2025-11 and 2025-12
  have fewer than two earlier banks and use the fallback head with no trained weights, so the six
  folds 2026-01 to 2026-06 carry the contrasts. The final-test window (from 2026-07-01) is not read.
- **Events.** The verified D5 and D16 transfers inside the development months, the other confirmed
  real events, and the injected events of experiment 660's bank. The label table is an incomplete
  dump, so every count is a lower bound; cells with fewer than 5 events are marked as too few to read.
- **Reaction time.** The first hours after the event's start (and end) at which the prevailing
  median differs from its normal-running median by at least half of the fitted transfer, as in
  experiment 668.
- **Seeds.** The two seeds of experiment 667; each arm is fitted with each seed.

## Scoring and figures

- CRPS, normalised MAE of the median, and q05 to q95 coverage for both outputs by lead band (days 0
  to 1, 1 to 3, 3 to 7, 7 to 14), for all rows, inside events, and on event-free rows more than 14
  days from any event; tail coverage and pinball loss at q01, q05, q95, and q99.
- Cost of false alarms: the share of event-free rows on which the prevailing output departs from
  the NRA output by 3 points or more, and its CRPS.
- Figures: reaction time against the number of lags with intervals (headline); the verified D5 and
  D16 transfer from 14 days before to 14 days after, baseline and best arm, prevailing panel under
  NRA panel; plus at least four example time series of forecasts.

## Compute

- The head is a 2-hidden-layer network of 64 units fitted on the CPU in experiment 667. The plan
  fits the head only, so it needs no GPU hours (ceiling 4 GPU-h unused). It uses at most 4 CPU cores
  and a 16 GB memory cap per job. Experiment 667's stored Candidate quantiles and bank are reused;
  the new fits are 4 arms x 2 seeds x 6 folds.

## Reviews

A Sonnet reviewer checks this plan, a fresh Sonnet reviewer checks the code before the first fit, and
a fresh Sonnet reviewer checks the results and the write-up.
