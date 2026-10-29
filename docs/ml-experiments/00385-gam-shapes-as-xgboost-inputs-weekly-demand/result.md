---
id: "00385"
slug: gam-shapes-as-xgboost-inputs-weekly-demand
theme: production-reference-and-best-guess
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: other
protocol: demand-v1
size: >-
  complex. Stored: forecasts, out-of-fold GAM column tables, and small result files in this
  experiment's private folder only; experiments 87, 377, and 379 are read and never written.
  Production serving path: not touched. Degradation rule: not touched. Designs: more than one would
  test the hypothesis (how the out-of-fold blocks are cut, how often the GAM columns are refreshed);
  the plan fixes one before any score is read. Callers: the fit script imports experiment 87's and
  experiment 377's folders and `studies.dp_gam`.
hypothesis: >-
  Giving the monthly-refitted XGBoost model (experiment 87's recipe, the BG2 monthly member) the shapes that
  the GAM learns, as extra input columns, lowers its CRPS at days 1 to 10, most at the Primaries
  where the GAM stack beats BG2 today.
prediction: >-
  Written before any score is read. P1. The stage-1 fitted value alone lowers the pooled days 1 to 10
  score of the monthly model by 0.02 to 0.08 points of effective capacity. P2. The partial effects
  alone lower it by less than the fitted value does. P3. Both together are no better than the fitted
  value alone by more than 0.02 points. P4. Any gain is largest at the Primaries where the stack
  trails BG2 by least, and the best arm stays behind the GAM stack at the signed-MW six.
decision_rule: >-
  Contrasts, pre-registered, each arm minus the baseline (experiment 87's monthly member refit; gate:
  within 0.05 points of experiment 379's stored `only_M` arm before any other arm is read), CRPS at
  days 1 to 10 in points of effective capacity, pooled per group (signed-MW six, MVA-only eight, and
  the two others as the protocol groups them), paired month-resampled bootstrap of
  `studies.eval_protocol`: C1 stage-1 value, C2 partial effects, C3 both. Read at the 98.3% level
  (Bonferroni, k = 3). The cost of an arm is a GAM fit at serving time plus one out-of-fold table per
  refit, so the threshold for porting is 0.05 points pooled in either group, with the interval
  wholly below zero. Also reported, not decisive: the best arm against the GAM stack and BG2. No
  adoption claim.
benchmark: >-
  Experiment 87's monthly member refit (baseline), the GAM stack from experiment 377, BG2 and its
  monthly member alone (experiment 379 `only_M`), `manual_heuristic`, and `climatology`, on
  `fold_a_daily_gap_v1`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles, mean absolute error of the
  median, and twCRPS, in points of effective capacity, from `studies.eval_protocol`; leads 0.5 to
  336 hours; headline band days 1 to 10.
code: ml-experiments/00385-gam-shapes-as-xgboost-inputs-weekly-demand/
production_status: not-for-production
adopted_in: none
date: 2026-10-28
---

# GAM-learnt shapes as input columns for the monthly-refitted XGBoost model

**Plan (status `planned`).** The question is whether an XGBoost model that is refitted every week
gains from seeing shapes that a GAM (generalised additive model) has already learnt from the same
history. The maintainer called the idea wacky (2026-10-28). Experiment 381 named it as arm A6 and
left it to its own experiment.

## Design

- **Baseline.** Experiment 87's `n1h` recipe (whole history, half-life 0.5 years, 500 rounds,
  `colsample_*` at 1.0), refitted monthly (maintainer, 2026-10-29: monthly refits, and every fitted booster saved under
  `work/models/monthly/<arm>/` in this experiment's data folder). The gate against experiment 379's
  `only_M` is 0.05 points. Weekly refits are a follow-up if the monthly contrast is promising.
- **GAM.** The T-all fit of experiment 377 (refitted at each cut with its code; the 2025-07-01 refit matches
  its stored fit to 5e-11). The three input groups are its stage-1 value (the fitted net power in
  points, from the member's forecast weather), the partial effects of its calendar terms (the
  columns of the design for the daily profile and day-type terms, times their coefficients), and the
  partial effect of its temperature (heating) terms.
- **No row sees a GAM fitted on its own target.** For a refit at time T, the GAM columns of the
  training rows come from out-of-fold fits: the training window to the latest GAM cut at or before T
  is cut into calendar-month blocks, each block is predicted by a fit that omits it, and the fit of
  the GAM cut itself predicts the rows of the scored runs. The scored rows use only that cut's
  full-window fit, whose cut is at or before the run's origin.
- **The GAM is refitted at each XGBoost refit**, at the cut 3.5 hours before the refit time, on its
  24-month window. Each fit takes under a second, so the out-of-fold fits are cheap.
- **Arms.** Baseline; `fitted` (stage-1 value); `shapes` (partial effects); `both`.
- **Row set and scoring.** All 16 Primaries to 14 days, `demand-v1`: per group, per Primary,
  leave-one-out, tails (pinball loss and coverage at 0.01, 0.05, 0.95, 0.99), the cost in the
  protocol's curtailment measure in the three load-shift row sets, and the median-minus-mean check.
- **Nulled lags.** The baseline already carries missing recent-power lags. The write-up reports the
  interval width and coverage on half-hours with missing lags against those with lags present.
- **Cost, stated.** The GAM must run at serving time, one fit per cut and one ensemble-driven
  prediction per run.

## Constraints

Sonnet only. At most 3 threads and `MemoryMax=16G` per job. Compute time is no object. The 381 plan,
code, and result are read-only. The id is the next free after 384.
