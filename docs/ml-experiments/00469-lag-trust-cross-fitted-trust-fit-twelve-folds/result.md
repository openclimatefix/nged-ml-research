---
id: "00469"
slug: lag-trust-cross-fitted-trust-fit-twelve-folds
title: >-
  Does a finer cross-fit (12 folds instead of 6) let the learned trust weight approach 0.5 and beat
  the fixed 50/50 mixture?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
date: 2026-11-01
kind: other
protocol: demand-v2
size: >-
  complex, following experiment 463. Stored: 60 out-of-fold lagged-head networks (not kept), 5 trust
  networks, a rebuilt 12-fold lagless table (65 module fits) and 5 months of forecasts and learned
  weights, all in this experiment's private data folder; every earlier experiment is read and never
  written, and experiment 404's frozen 6-fold lagless table is never touched. One design only: the
  recipe of experiment 463's G1-trust arm with the fold count changed. Production serving path: not
  touched (this repository holds none). Degradation rule: not touched. Callers: the fit script
  imports experiment 463's fit loop and the modules of experiments 87, 48, 397, and 404; the lagged
  head is experiment 404's A0-inj network.
hypothesis: >-
  The out-of-fold lagged head of experiment 463 is too weak because each of its 6 folds trains on
  only 5/6 of the rows, so its learned trust weight sits below 0.5 and loses to the fixed 50/50
  mixture (B-mix). With 12 folds, each fold trains on 11/12 of the rows, the out-of-fold lagged
  head is stronger, the learned lagged weight approaches 0.5, and the trust weight beats B-mix.
prediction: >-
  Written before any fit. P1. G1-trust-12 minus B-mix on rows with no labelled event at the
  signed-MW six improves from experiment 463's -0.021 to below -0.06, with a 95% interval wholly
  below zero, and the learned lagged weight rises above 0.45. The two numbers are expected to
  fail: 12 folds raise the training share only from 83% to 92%, while experiment 463's
  out-of-fold negative log-likelihood (0.905) is far above its in-sample value (0.242) because of
  extrapolation to a held-out block of time, not because of the 8% of rows lost.
decision_rule: >-
  Mechanism verdict. The weaker-fold-head explanation is supported if G1-trust-12 minus B-mix on
  rows with no labelled event at the signed-MW six is below -0.06 with an interval wholly below
  zero and the learned lagged weight is above 0.45. It is rejected, and the trust line closes, if
  that contrast stays within +-0.03 and the learned weight stays below 0.45. A contrast between
  -0.06 and -0.03, or a weight between 0.40 and 0.45, is inconclusive: report partial support,
  do not close the line, do not adopt, and run no 24-fold follow-up unless it is pre-registered.
  Adoption. Adopt only if G1-trust-12 also beats K2-S1 (experiment 454) on rows with no labelled
  event, on NRA-corrected real-event rows (depth factor 1.0, headline), and on avoidable-curtailment
  cost (signed-MW six, f = 0.9, 0.75, 0.5), because K2-S1 needs no cross-fit table in production.
  The gate is hard to reach (K2-S1 is -0.121 against B-mix, P1 is below -0.06); the run is a
  diagnostic. Intervals. Months are the resampling unit (5 months, 4 degrees of freedom). The 95%
  interval is reported with the Bonferroni interval beside it; the Bonferroni count is 24 (three contrasts, G1-trust-12 against B-mix, against G1-trust, and
  against K2-S1, on four slices, which are rows with no labelled event, harm rows, NRA-corrected
  real-event rows, and metered real-event rows, at both substation groups), and the minimum
  detectable effect is stated as a number beside every contrast. The paired contrast G1-trust-12
  minus G1-trust (6 folds, experiment 463) is the direct answer to the question, with its own
  minimum detectable effect.
benchmark: >-
  B-mix (experiment 404's stored fixed 50/50 mixture of A0-inj and the cross-fitted lagless table).
  References: experiment 463's G1-trust and G1-insample (6 folds; there is no 12-fold insample
  control), and K2-S1 from experiment 454.
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of effective capacity from the 11 quantiles of
  each forecast, as in experiment 404's report, on the real-data world (world -1) with leads of 0
  to 240 hours (lead days 1 to 10).
code: ml-experiments/00469-lag-trust-cross-fitted-trust-fit-twelve-folds/
---

**Planned.** This page is the pre-registration; the write-up replaces it when the fits finish.

- **Question.** Experiment 463 fitted the trust weight on lagged heads from 6 folds. Its learned
  weight (about 0.4) did not beat the fixed 50/50 mixture. Does a finer split, 12 folds, move the
  weight towards 0.5?
- **Fits.** The lagless table is rebuilt at 12 folds in `lagless_12fold/` (65 module fits, 6
  minutes). Then, per scored month, 12 out-of-fold lagged-head networks plus 1 trust network (60
  plus 5 fits, about 1.3 GPU-hours). One fold fit takes 11 s, a trust fit about 12 minutes.
- **Reused, not refitted.** Experiment 404's B-mix and full-data A0-inj networks, experiment 454's
  K2-S1 forecasts, and experiment 463's six-fold G1-trust and G1-insample forecasts.
- **Scoring.** Equal rows across arms; signed-MW six and MVA-only eight reported separately; rows
  with no labelled event; NRA-corrected real-event rows with and without the largest event; harm
  rows (experiment 455's definition); absolute CRPS; q01, q05, q95, q99 coverage and pinball;
  month-block intervals; the learned weight per month and per lead band; avoidable-curtailment cost
  via experiment 454's `e00454_curtailment.py`; the production cost of a 12-fold cross-fit (12
  out-of-fold fits, 1 trust fit, and 13 lagless fits per refit, against 6, 1, and 7 at 6 folds).
- **Caveat.** NGED's event labels are incomplete, so rows with no labelled event may hold unlogged
  events.
