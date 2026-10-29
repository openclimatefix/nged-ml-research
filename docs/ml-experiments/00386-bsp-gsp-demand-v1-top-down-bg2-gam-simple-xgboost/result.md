---
id: "00386"
slug: bsp-gsp-demand-v1-top-down-bg2-gam-simple-xgboost
theme: hierarchy-and-pooling
also_relevant_to:
  - production-reference-and-best-guess
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v1
size: >-
  complex in scale, medium in risk. Stored: forecasts and small result files in this experiment's
  private folder only; experiments 87, 161, 377, 379, and 381 are read and never written.
  Production serving path: not touched. Degradation rule: not touched. Designs: more than one would
  test the question (fit the GAM stack on the BSP and GSP readings as stand-alone series, or reuse
  the bulk supply points' older chain); the plan fixes the first. Callers: scripts copy experiment
  377's stack and experiment 161's bulk-supply-point inputs, and call `studies.eval_protocol`.
hypothesis: >-
  At the four bulk and grid supply points that NGED forecasts, fitted and scored top-down on the
  aggregate series, the best Primary-demand GAM stack is not worse than BG2 (the weekly-refitted
  XGBoost model alone, `b_weekly`) by more than 0.20 points of effective capacity at days 1 to 10,
  separately for the two bulk supply points (BSPs: D13, D17) and the two grid supply points (GSPs:
  D1, D6).
prediction: >-
  Written before any score is read. P1. BSPs, stack minus BG2, days 1 to 10, pooled over the two:
  between -0.10 and +0.40 points. P2. GSPs, stack minus BG2: between +0.00 and +0.80 points, because
  a GSP aggregates wind, solar, and storage that the stack has no term for (experiment 36 found
  regional wind and solar columns helped there). P3. The stack beats climatology at all four series
  and `manual_heuristic` at least at three. P4. Coverage at q05 is above 0.07 for the stack at the
  BSPs (the lower tail is too narrow, as in experiment 377). P5. The simple weekly XGBoost arm
  equals BG2 until experiment 381 stores a better arm, because BG2 at these series is already the
  weekly member alone.
decision_rule: >-
  Descriptive: the BSP and GSP family has no research default, so nothing is adopted. Contrasts,
  registered before any score is read, CRPS in points of effective capacity on the shared rows
  (`fold_a_daily_gap_v1`, 292 daily runs of 2025-07 to 2026-06), paired month-resampled bootstrap of
  `studies.eval_protocol`: C1 stack minus BG2 pooled over the BSPs, C2 the same pooled over the
  GSPs, never one pooled number across both; each at the 99% level (20,000 resamples) with the
  largest contributor left out beside it, and per series at 95% (2,000 resamples). C3 stack minus
  `manual_heuristic` and minus `climatology`, per series. C4 simple weekly XGBoost minus BG2 and
  minus the stack, once experiment 381's best simple arm is stored. Lead bands as in the protocol;
  days 1 to 10 headline, days 11 to 14 report only. Tail calibration (pinball loss and coverage at
  0.01, 0.05, 0.95, 0.99) is reported beside CRPS and twCRPS and named in the take-home.
benchmark: >-
  BG2 (`b_weekly`, experiment 87's stored predictions, which hold leads up to 351 hours at these
  four series), `manual_heuristic`, `climatology`, and, as a labelled extra, the bottom-up sum of
  Primary forecasts where experiment 161 or 204 built it.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles, mean absolute error of the
  median, and twCRPS, in points of effective capacity, from `studies.eval_protocol`; leads 0.5 to
  336 hours; headline band days 1 to 10.
code: ml-experiments/00386-bsp-gsp-demand-v1-top-down-bg2-gam-simple-xgboost/
production_status: not-for-production
adopted_in: none
date: 2026-10-29
---

# BG2, the GAM stack, and a simple weekly XGBoost model at the bulk and grid supply points

Plan only. Nothing has been fitted. Stages, in order, each a separate commit so the pull request
can be split if it grows too large:

- **Stage A, no new fitting.** Score BG2 (`b_weekly`), `manual_heuristic`, and `climatology` at D1,
  D6, D13, and D17 on the demand-v1 metrics. BSP and GSP are their own groups. Tail calibration and
  slices included.
- **Stage B, the GAM stack at the BSPs.** Reuse experiment 161's ensemble stores for D13 and D17 and
  experiment 377's stack code; report what could not be fitted.
- **Stage C, the GAM stack at the GSPs.** D1 and D6 have no ensemble store in the stack's format, so
  they need one built (the largest cost). If it cannot be built within the compute budget, the GSPs
  appear as "not fitted" rows, as the protocol requires.
- **Stage D, the simple weekly XGBoost arm.** Until experiment 381 stores its best arm, the
  placeholder is BG2's weekly member, which is the same model as BG2 at these four series, labelled
  "placeholder (equals BG2)". The real arm is refitted with experiment 381's fit script once it has
  merged.
