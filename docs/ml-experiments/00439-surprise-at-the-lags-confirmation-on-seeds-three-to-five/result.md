---
id: "00439"
slug: surprise-at-the-lags-confirmation-on-seeds-three-to-five
title: >-
  Does adding the surprise at the three lags to the lagged network still lower the error on three
  new seeds?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  simple. Stored: 24 fitted networks and their forecasts in this experiment's private folder, plus 15
  new cross-fitted lagless tables (seeds 3 to 5) added beside experiment 404's frozen ones. No new
  method; the arms and rows are experiment 428's. Production serving path: not touched (this
  repository holds none). Degradation rule: not touched. Callers: the scripts import experiment
  428's modules.
hypothesis: >-
  S1 (the three lagged readings plus the surprise at each lag) lowers the continuous ranked
  probability score against R-inj (the three lags, trained on injected events) on clean rows, on
  new fit seeds 3 to 5, by about as much as it did on seeds 0 to 2 (0.083 points at the signed-MW six).
prediction: >-
  Written before any new fit. CRPS in points of effective capacity at days 1 to 10, 1.15 widening.
  P1. On seeds 3 to 5, S1 minus R-inj on clean rows at the signed-MW six lies between -0.11 and
  -0.05. P2. On injected harm rows it lies between -0.09 and -0.03. P3. At the MVA-only eight it has
  the same sign and a smaller size. P4. At least 3 of 4 folds show a negative contrast on clean rows.
  P5. On real event rows the contrast is negative (experiment 428: -0.170) but its interval is wide.
decision_rule: >-
  Seeds 3 to 5, four folds (test months 2026-06, 2026-01, 2025-10, 2026-03), monthly refits, signed-MW
  six and MVA-only eight, rows as in experiment 428 (clean, injected harm, real harm, real event).
  S1 is confirmed if, on seeds 3 to 5 alone at the signed-MW six, S1 minus R-inj on clean rows is
  below minus the minimum detectable effect (computed from R-inj's seed spread on seeds 3 to 5, by
  experiment 428's formula), its 95% block-bootstrap interval excludes zero, and at least 3 of 4
  folds are negative. If the point estimate is negative but inside the minimum detectable effect, S1
  is reported as unconfirmed and R-inj stays the reference. If the contrast is positive, S1 is
  refuted as a base. The pooled estimate over seeds 0 to 5 is reported beside, with per-fold and
  per-seed contrasts. The smallest gain worth porting is the minimum detectable effect, because S1
  adds a lagless module to the serving path (a design cost the maintainer must accept before the
  first release). Interim; no 99% claim.
benchmark: >-
  R-inj refitted on seeds 3 to 5 with the same folds and rows. Seeds 0 to 2 are experiment 428's
  stored forecasts.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, coverage and pinball loss at q01, q05,
  q95, and q99, at the 1.15 widening.
verdict: not-yet-run
best_arm: none yet
production_status: not-for-production
adopted_in: none
code: ml-experiments/00439-surprise-at-the-lags-confirmation-on-seeds-three-to-five/
---

**Take-home message: not yet run.**

## Question

CRPS is the continuous ranked probability score. The signed-MW six and the MVA-only eight are the two
groups of Primary substations by metering.

**This experiment checks a result that cleared its minimum detectable effect by 0.006.** Experiment
428 found S1 lowered CRPS against R-inj by 0.083 on clean rows (minimum detectable effect 0.077),
and one of four folds did not clear the minimum. New fit seeds are the cheapest independent check.

## Design

- **Arms.** S1 and R-inj of experiment 428, fitted with its code unchanged on seeds 3, 4, and 5
  (4 folds, so 24 fits, about 6 GPU-hours, one process under 8 GB).
- **Lagless tables.** S1's surprise reads experiment 404's cross-fitted lagless forecast (K = 6
  blocks). Experiment 404 froze tables for seeds 0 to 2 only, so tables for seeds 3 to 5 are built
  with the same frozen code (15 builds of about 1 minute each) so each fit seed has its own table.
- **Other.** Cleaned data, D15 and D18 start cut, injected worlds, and scoring rows as in 428. The
  frozen tables for seeds 0 to 2 are untouched; experiment 428's stored forecasts supply seeds 0 to 2.
- **Reported.** Absolute CRPS beside each contrast, the minimum detectable effect, per-fold and
  per-seed contrasts, the pooled contrast over seeds 0 to 5, tails, example forecasts, and whether
  the arm can run in experiment 404's harness.
