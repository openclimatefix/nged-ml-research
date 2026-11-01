---
id: "00457"
slug: gam-stage-1-forecast-as-an-input-to-the-pooled-mixture-density-network
title: >-
  Does the out-of-fold forecast of the convex GAM, given to the pretrained pooled mixture density
  network as an input or as a base, improve Primary demand forecasts?
theme: production-reference-and-best-guess
also_relevant_to:
  - hierarchy-and-pooling
status: planned
kind: change
protocol: demand-v2
size: >-
  complex. Stored in this experiment's private folder only: the stage-0 GAM forecasts, and 50
  pretrained and 50 fine-tuned neural networks (two new arms, five folds, seeds 0 to 4). Experiments
  155, 201, 252, 425, and 449 are read and never written. More than one design is defensible
  (extra input against additive base, and how stage 0 is built), so the plan fixes each choice
  before any score is read. Production serving path: not touched (this repository holds none).
  Degradation rule: not touched. Callers: the scripts import the fit loop of experiment 397 through
  experiment 425 and the GAM code of `studies.dp_gam`, whose callers sit in those folders.
hypothesis: >-
  If the pooled mixture density network of experiment 425's arm A is given the out-of-fold stage-1
  GAM forecast of Primary demand, either as one extra input feature (A-gam-in) or as a
  zero-initialised additive base that the network corrects (A-gam-base), its CRPS at days 1 to 10
  is lower than arm A's on the eight MVA-only Primaries, because the GAM carries the smooth
  calendar, temperature, and solar shapes that the network learns less well from five years of data.
prediction: >-
  Written before any fit and before any score is read. P1. A-gam-base beats A-ref by 0.05 to 0.15
  points of effective capacity at the eight (experiment 201's correction head on arm D gained
  0.086 at the signed-MW six, a weaker base than arm A). P2. A-gam-in gains less than A-gam-base,
  because the network may ignore one more input among many. P3. Cold winter nights and Christmas
  week do not worsen by more than 0.02. The prediction could fail in either direction: the GAM may
  add nothing the pretrained network has not learnt (experiment 413 found learnt shape libraries
  0.17 to 0.23 worse), and an additive base may pull the quantiles towards the GAM's errors.
decision_rule: >-
  CRPS at days 1 to 10 in points of effective capacity (negative is a gain), monthly refits, 1.15
  widening unless stated, seeds 0 to 4 averaged by quantile as experiment 425 scores arm A, on the
  half-hours every arm and A-ref share, five folds (2026-06, 2026-01, 2025-10, 2026-03, 2025-12 of
  experiment 449). Gating rows are the eight MVA-only Primaries, cold winter nights, and Christmas
  week (22 to 31 December, one fold, one year, report-only if its interval spans zero), each with a
  month-block bootstrap and Bonferroni intervals; the count of tested contrasts is stated as a
  number before any score is read (2 arms times 3 gating row sets = 6, plus the q99, q01, q05, and
  q95 checks counted separately in the scorer's printout). D15 and D18 are real rows,
  report-only. The scorer prints the minimum detectable effect (half the five-seed spread of A-ref
  seed-matched pools over folds) before any contrast is read. Adopt an arm over A-ref only if its
  CRPS gain at the eight is at least the minimum detectable effect, its cold-night and
  Christmas-week CRPS do not worsen by more than 0.02, and its q01, q05, q95, and q99 pinball loss
  do not worsen by more than 0.02 (one-sided 95% upper bound). If neither arm clears the rule, the
  GAM shapes do not help the pooled network and the question closes. Production cost, stated in
  either case: the GAM must run in the serving path with its monthly refit.
benchmark: >-
  A-ref (experiment 425's arm A, seeds 0 to 4 and the five-seed quantile average, stored), the new
  stage-0 GAM alone, BG2, climatology, and `manual_heuristic` as stored references.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS from the 11 quantiles at days 1 to 10; coverage and pinball loss at q01, q05, q95, q99;
  cold winter nights; Christmas week; avoidable-curtailment cost at f = 0.9, 0.75, 0.5 at the
  signed-MW six, with block intervals as in experiment 449; widening sweep 1.0 to 1.15. Per group
  and per Primary.
---

## Plan (interim, unaudited; no score has been read)

- **Stage 0 (CPU): one pure stage-1 GAM forecast per Primary, month and ensemble run.** No Kalman
  filter, no lagged-error input. Horizon to 240 hours. Each monthly GAM is fitted on data strictly
  before the scored month. Which GAM code and which stored fits are used is fixed in the stage-0
  section of this page once the code review has run.
- **Stage 1 (GPU): two new arms on experiment 425's arm A recipe.** Experiment 397's loop
  (`g09`, pretraining then fine-tuning at learning rate 2e-4), five seeds, the five folds of
  experiment 449. No injected-event generator, no frozen tables of experiment 404, no surprise
  inputs. Arm A itself is scored again, never refitted.
- **Leakage check.** The last training timestamp of every monthly GAM is read from the fit
  metadata and must be before the start of the scored month.
- **Process.** Seven steps of `common.md`: code, fresh Sonnet code review before any number is
  used, run, results review, write-up, write-up review.
- **Caveats carried onto the page.** NGED's event labels are incomplete, so rows with no labelled
  event may hold unlabelled real events. Past-weather: the GAM is driven by the calibrated
  ensemble mean at forecast time, so no past-weather caveat applies; the ERA5 pretraining years are
  handled as the stage-0 section states.
