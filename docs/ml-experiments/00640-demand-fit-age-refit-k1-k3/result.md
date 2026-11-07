---
id: "00640"
slug: demand-fit-age-refit-k1-k3
title: >-
  How much worse does the demand network score the longer ago it was fitted: 0, 4, 8, 16, or 26
  weeks before the forecast month?
theme: retraining-and-training-history
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  Medium. Answers to the five triggers: (1) it stores new network weights and forecasts in the
  private data folder only; (2) one design is defensible, because the refit schedule is the only
  thing varied; (3) it spans code from experiments 510 to 620 whose callers cannot be named without
  searching, so it reuses that code by import and patches only the cutoffs; (4) it touches no
  serving path and no degradation rule; (5) it changes no contract.
hypothesis: >-
  A demand network fitted on data that ends A weeks before the forecast month scores worse the larger
  A is, and the loss is under 1% of the continuous ranked probability score (CRPS) for every 4 weeks
  of fit age, for K1 (the lag-using network with the gate) and for K3 (the lag-free network, the
  control).
prediction: >-
  Written before any fit. P1: the mean CRPS rises by less than 1% per 4 weeks of fit age, fitted as
  a straight line through the five fit ages, for K1 and for K3 (percent of the age-0 CRPS, 0 to 240
  h). P2: K1 loses more per 4 weeks than K3 at 24 to 120 h, because K1's lag inputs and fine-tune
  rows go stale with the fit. P3: the Primaries with a labelled event in the test month lose more
  per 4 weeks than the others. P4: the age-0 fit scores within 1% of experiment 613's roster on the
  first 14 days of each month (the roster's fit cutoff of day 0 applies to those days).
decision_rule: >-
  The slope is the CRPS change per 4 weeks of fit age, as a percentage of the age-0 CRPS, with a
  month-block bootstrap interval (resampling the 5 development months, with Primaries paired).
  Weekly refits in production are worth running if the interval's lower end exceeds 1% per 4 weeks,
  monthly refits if the CRPS at 4 weeks of age exceeds the age-0 CRPS by more than 1% with an
  interval excluding zero, and neither if the upper end of the slope is under 1%. There is no
  Bonferroni correction because the slope per network is the one planned contrast per network (2 in
  total, so the printed level is 97.5%).
benchmark: >-
  Experiment 613's roster scores for K1 and K3 (day 0 and day 14 refits), climatology, and the
  manual heuristic, on the same rows.
row_set_id: development_months_2025_10_to_2026_06_16_primaries
metric: >-
  CRPS and MAE by lead band (0-24, 24-120, 120-240 h) and by weeks since the fit; q05-q95 coverage.
---

*Plan only. No fit has run and no result is written yet.*

## Plan

- **Question.** The production service will refit on some schedule. Experiments
  [494](00494-simplify-demand-network-refit-cadence-and-pretrain-once/result.md) and
  [511](00511-lagless-network-refit-cadence-half-life-and-mixed-pretraining/result.md) measured
  the price of an old fit on the lag-free network only, and the July to October window of
  [experiment 620](00620-bakeoff-final-test-demand-networks-at-primaries/result.md) used one
  fit with no refit, so its fit age rose from 0 to 17 weeks. This experiment measures the loss
  per 4 weeks of fit age on the development months, where an answer cannot touch the closed window.
- **Design.** For each development test month (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), fit
  K1 and K3 once with the cutoff A weeks before the month starts, for A = 0, 4, 8, 16, and 26. Each
  fit has no refit during the month, so the fit age of a forecast origin is A plus its days into the
  month, and the score by weeks since fit covers 0 to about 30 weeks. The recipe, seeds, 16
  Primaries, and pretraining rules are experiment 613's. Pretraining reruns at each cutoff, with
  its newest row older than the cutoff by the usual refit age, as experiment 620's final fold does.
  Each fit's cleaned power is hidden after its cutoff, as in experiment 620's training phase, so no
  code path can read a later reading.
- **Cheapest recipe.** One cutoff per fit and no day-14 refit, with seeds 0 to 2 in place of
  experiment 613's five. Experiment 620's logs show about 11 minutes of GPU time for K1 over 5
  seeds, so 25 cutoffs cost about 3.3 GPU-hours for K1 at 3 seeds and under 0.5 GPU-hours for K3.
  Prediction P4 is the check that this recipe reproduces the roster within 1%; if it fails, the
  page says so and quotes only the contrast between ages, not absolute scores.
- **Scoring.** Reuse experiment 613's scorer (`e00613_score.py`) for the cells by Primary, month,
  and lead band, with K1 assembled with the gate from the K3 fit of the same cutoff, and group the
  rows by weeks since the fit. Report separately the Primaries with a labelled event in the
  test month (NGED event labels, a lower bound because labels are incomplete) and the others.
  Quote the age-0 roster rows, climatology, and the manual heuristic beside the networks, and the
  tail table (q01, q05, q95, q99 coverage) beside the CRPS.
- **Standing practices followed.** Cleaned data everywhere; the D15 and D18 cut before 2025-03-01;
  models saved as `.ubj` or state files in the data folder; stored 613 predictions reused as the
  reference; the closed window (valid time from 2026-07-01) is never opened; the run is a screen on
  the development months, not a protocol change, so `kind: other`.
- **Compute.** Ceiling 6 GPU-hours (process hours). One GPU process at a time, 8 GB host cap and
  one OMP thread, as experiment 620's `gpu.sh` does, and `systemd-run` with `-p MemoryMax=8G -p
  OOMPolicy=continue`.
- **Reporting.** The headline chart is CRPS against weeks since the fit for K1, K3, climatology,
  and the manual heuristic, with month-block intervals. Further figures: slope per lead band, the
  event and non-event split, and per-month curves.

## What would change production

- If the slope interval's upper end is under 1% per 4 weeks, a refit every 4 to 8 weeks is
  enough, and the production service should not spend effort on weekly refits.
- If the lower end exceeds 1% per 4 weeks, weekly or monthly refits are worth running, and the
  experiment names which by the age at which the loss passes 1%.
