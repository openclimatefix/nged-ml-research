---
id: "00380"
slug: gam-stack-ablation-and-jaggedness-at-16-primaries
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v1
size: >-
  simple. Stored: forecasts and small result files in this experiment's private folder only;
  experiment 377's folders are read and never written. Production serving path: not touched.
  Degradation rule: not touched. Designs: the ablation has one defensible design (switch one stage
  of experiment 377's stack off at a time, and build up from the plain model); the jaggedness
  measures are chosen below, before any score. Callers: the scripts import experiment 377's
  modules, whose callers are in that folder, and `studies.eval_protocol`. Nothing is refitted: the
  stage 1 monthly fits are experiment 377's stored fits, and the Kalman-off arm forecasts them
  again with no filter correction.
hypothesis: >-
  (H1) Each stage of experiment 377's GAM stack (the Kalman filter on stage 1's residuals, the winter
  night offset, recipe R, and the per-fold stage 2 spread) earns its place only if removing it
  raises the days 1 to 10 quantile score by more than that stage's production cost justifies; some
  stages will not. (H2) The stack trails BG2 most at the Primaries whose demand is least regular
  (low share of variance in the weekly profile and in temperature, many steps), because a GAM
  captures regular patterns and not largely stochastic generation.
prediction: >-
  Written before any score is read. P1: removing the Kalman filter raises the lead-day-1 score by
  more than 0.10 points but the days 1 to 10 score by less than 0.05, in every group. P2: removing
  recipe R raises the days 1 to 10 score by 0.01 to 0.05. P3: removing the winter offset changes it
  by under 0.01. P4: replacing the per-fold spread by stage 1's tuning-month spread raises it by
  under 0.03 and worsens q05 coverage. P5: the plain model (no filter, no corrections, stage 1
  spread) trails the full stack by 0.15 to 0.40 points. P6: the Spearman correlation (n = 16)
  between the stack-minus-BG2 gap and the share of variance in the weekly profile (M1) is at most
  -0.5, and with the step rate (M4) at least +0.5.
decision_rule: >-
  No adoption. Stage contrasts, each by the paired month-resampled bootstrap of
  `studies.eval_protocol` at 99% (the level a 5% Bonferroni correction across the eight planned
  contrasts sets is 99.4%, so 99% is reported and a contrast near the edge is called
  inconclusive), days 1 to 10, per group. A stage "earns its place" when removing it raises the
  score by more than its threshold with an interval excluding zero. Thresholds, set by what the
  stage costs in production: the Kalman filter 0.05 points (a filter state to run and monitor per
  Primary and band), the winter offset 0.02, recipe R 0.02 (each a small fitted or averaged term
  on past errors), the per-fold spread 0.02 (monthly refits of the spread tables). A stage whose
  interval excludes zero but whose point estimate is under the threshold is a "small gain" and
  could go for a small, measured loss. H2 is supported when M1 and M4 (the two direct readings of
  the hypothesis) both have the registered sign with |rho| at least 0.5 and a permutation p
  value under 0.025 each (n = 16); the other measures are descriptive. With 16 Primaries the test
  is weak: a failure at n = 16 is reported as not shown, not as the hypothesis being false.
benchmark: >-
  BG2, `manual_heuristic`, and `climatology` as stored by experiment 87 on `fold_a_daily_gap_v1`,
  as scored in experiment 377.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles, mean absolute error of the
  median, and twCRPS, in points of effective capacity, from `studies.eval_protocol`; leads 0.5 to
  336 hours; the headline band is days 1 to 10.
code: ml-experiments/00380-gam-stack-ablation-and-jaggedness-at-16-primaries/
---

# The GAM stack taken apart at all 16 Primaries, and why it is jagged

**Status: planned. This page is the plan; the results replace it.**

**Questions.** (1) Which stages of the GAM stack could be removed for a small, measured loss? (2)
Is the stack's gap to BG2 larger at the Primaries whose demand is less regular?

**Design.**

- Start from the stack that [experiment
  377](../00377-demand-v1-first-run-feasibility/result.md) scored ("Primary demand research
  default 1, extended to 336 h", no solar summary S). Its stages: the monthly-refitted GAM with a
  Kalman filter on its residuals (stage 1), the winter night offset and recipe R (stage 2 mean
  corrections), and the per-fold spread tables (stage 2 bands). Not in the stack, so not ablated:
  the solar summary S, the correction network H, and the daylight heuristic blend. Experiment 378
  builds default 3 with S and the blend; this experiment does not wait for it.
- Arms: `full` (experiment 377's stack, rebuilt, which must reproduce its stored numbers),
  `drop_k` (Kalman filter off), `drop_o` (offset off), `build3` (recipe R off), `drop_f` (stage 1
  tuning-month spread in place of the per-fold spread), and a build-up from the plain model:
  `build0` (no filter, no corrections, stage 1 spread), `build1` (+ per-fold spread), `build2`
  (+ Kalman filter), `build3` (+ offset), `full` (+ recipe R). The eight planned contrasts are
  `drop_k`, `drop_o`, `build3`, `drop_f` against `full`, and the four build-up steps.
- The Kalman-off stage 1 is experiment 377's stored monthly fits forecast again with no filter
  correction; the stage 2 corrections and spreads are then rebuilt with experiment 377's code on the
  unfiltered forecasts. D8 uses its own-residual arm (from fold month 2025-08); the donor-spread arm
  is not repeated.
- Scoring: `studies.eval_protocol` on the shared rows of all arms and the three references; per
  Primary, per group (signed-MW six, MVA-only eight with "reactive power assumed zero", D15 and D18
  report only), per slice and lead band, with the per-Primary table, leave-one-out contrasts, and
  tail calibration. Retraining schedule: none (stage 1 fits are experiment 377's monthly fits).
- Jaggedness measures, from readings before 2025-07-01 only, registered here: M1 the share of
  variance explained by a half-hour-of-week mean profile; M2 the extra share of the remaining
  variance explained by binned temperature; M3 the lag-1 autocorrelation of what remains; M4 the
  step rate (share of half-hour changes above 5% of the Primary's capacity); M5 the share of
  half-hours with reverse flow (signed-MW six only, so n = 6 and descriptive). Outcome: the
  stack-minus-BG2 gap, days 1 to 10, per Primary. Tests: Spearman rank correlation at n = 16 and at
  n = 14 (D15 and D18 dropped), a leave-one-out range, a permutation placebo (10,000 shuffles of the
  measure across Primaries), and a time-shuffled-readings placebo for M1 to M4.
- Weather at lag times: stage 1 reads the ensemble forecast and the analysis proxy of experiment
  377 only, so every input exists at forecast time. The jaggedness measures use ERA5 temperature in
  the training period only and are not forecast inputs.

**Earlier advice followed.** Compare with BG2 on experiment 377's shared rows; report the sunny
half-hour gap; treat D8 as its own problem (its own-residual arm is its row).
