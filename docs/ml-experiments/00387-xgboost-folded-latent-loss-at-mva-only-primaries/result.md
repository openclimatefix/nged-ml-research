---
id: "00387"
slug: xgboost-folded-latent-loss-at-mva-only-primaries
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
  - demand-inputs
status: planned
kind: other
protocol: demand-v1
size: >-
  complex. Stored: forecasts and small result files in this experiment's private folder only;
  experiments 87, 379, 381, 365, and 366 are read and never written. Designs: more than one would
  test the hypothesis (folded squared loss, folded pinball loss, a sign-then-magnitude pipeline),
  so the plan fixes the order and the trigger for each before any score is read. Callers: the fit
  script imports experiment 87's folder and copies experiment 381's fitter; the scorer imports
  `studies.eval_protocol`. Serving path and degradation rule: not touched (this repository holds
  neither).
hypothesis: >-
  A single XGBoost model per Primary, refitted weekly with experiment 87's recipe but with a loss
  that scores the absolute value of a signed latent power against the metered apparent power (a
  folded loss), recovers the reverse-flow V-shape at the MVA-only Primaries that the plain
  model misses, and recovers a signed power forecast that agrees with known signed truth.
prediction: >-
  Written before any score is read. P1 (known answer). On the six signed-MW Primaries with
  synthetic reverse flow injected and the sign then discarded, the folded pinball model (F2)
  recovers the sign on at least 85% of reverse-flow half-hours and the signed power to within 1.5
  points of effective capacity (mean absolute error), against more than 3 points for the plain
  model read as if it were signed. P2. At D12 the plain model's median under-forecasts sunny
  half-hours by about 5 points (experiment 381); F2 cuts that bias to under 2.5 points. P3. F2
  moves the days 1 to 10 CRPS by -0.05 to -0.25 points at D12 and by -0.03 to +0.03 points at the
  other MVA-only seven. P4. At the signed-MW six, the folded loss applied to the true signed series is
  not used (those meters need no fold), and the folded model fed unsigned synthetic MVA scores
  within 0.10 points of the plain signed-meter model there. P5. F1 (folded squared loss, quantiles
  by residual scaling) recovers the sign about as well as F2 but its tails are 10% to 30% worse in
  pinball loss at the 0.01 and 0.99 quantiles. P6. F3 is not needed.
decision_rule: >-
  Pre-registered contrasts, all CRPS or sign accuracy on the shared rows of `fold_a_daily_gap_v1`.
  Gate G (known answer, read before any D12 or MVA-only number): an arm passes when its recovered
  sign accuracy on reverse-flow half-hours is at least 80% and its signed-power MAE is under half
  that of the plain model read as signed, at injection level 1.0 and the first reactive setting (pf 1.0, Q = 0), with the best (kappa, Q0)
  chosen on gate data; the other reactive settings are reported, not gated. An arm that fails G is not read at D12. If F1 and F2 both fail G, F3 (sign
  probability model, then magnitude model) is built, after the plan is amended and reviewed, and
  nothing else is read. Contrasts after G: C1 (F2 minus plain weekly, CRPS days 1 to 10, MVA-only eight
  pooled and D12 alone), C2 (F1 minus plain), C3 (F2 minus F0, the unfolded custom-objective
  control, to separate the fold from the custom objective), C4 (F2 minus plain at the signed-MW
  six, the harm check; the loss must not exceed the protocol's per-Primary tolerance),
  C5 (F2 minus the GAM stack and BG2, descriptive). Paired month-resampled bootstrap of
  `studies.eval_protocol`, 99% pooled with k = 4 (level 1 - 0.01/4, 20,000 resamples) and 95% per
  Primary. Screening at monthly refits; the arm that passes G and has C1 wholly below zero
  at D12 or the MVA-only pooled group is refitted weekly. Threshold: a loss change costs nothing
  in production, so any interval excluding zero counts. Placebos: D5 and D10 (signed, no
  reverse flow). No adoption claim.
benchmark: >-
  The plain weekly XGBoost model of experiment 87 (`only_W` of experiment 379, and the refit
  reproduced here), BG2, the GAM stack of experiment 377, `manual_heuristic`, and `climatology`, on
  `fold_a_daily_gap_v1`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, mean absolute error of the median,
  tail pinball and coverage at 0.01, 0.05, 0.95, and 0.99, and twCRPS, in points of effective
  capacity, from `studies.eval_protocol`; leads 0.5 to 336 hours; headline band days 1 to 10. For the
  signed forecast: mean absolute error against the signed truth, sign accuracy on reverse-flow
  half-hours, and the CRPS of the recovered signed quantiles.
code: ml-experiments/00387-xgboost-folded-latent-loss-at-mva-only-primaries/
production_status: not-for-production
adopted_in: none
date: 2026-10-29
---

# A folded loss for one weekly XGBoost model at the MVA-only Primaries

**Plan (status `planned`).** Eight Primaries have meters that record apparent power (MVA), which
has no sign, so reverse flow from embedded solar shows up as demand. A plain XGBoost model under-
forecasts sunny half-hours at D12 by about 5 points (experiment 381). This experiment gives the
model a signed latent power *l* (demand minus embedded generation) and scores |*l*| against the
metered reading, so the V-shape can appear without telling the model where reverse flow happens.
The prize is a signed MW forecast for MVA-metered substations (maintainer, 2026-10-29).
Reactive power is assumed zero ("|P| as the working simplification"), and the loss carries a power
factor so a reactive term can enter later.

## Setup

- **Base recipe.** Experiment 87's `n1h` weekly member (ENS control member rows, whole history,
  half-life 0.5 years, 500 rounds, eta 0.05, depth 6, `colsample` 1.0, the three recent-power lags,
  `lead_hours`), copied from experiment 381's fitter. The plain refit must reproduce experiment 379's stored
  `only_W` (and `only_M`) within 0.05 points before any new arm is read.
- **Fold.** The fold is the smooth S(*l*) = sqrt(*l*² + Q0²) / pf. Q0 is the reactive-power floor,
  because an MVA reading never reaches zero while some reactive power flows (maintainer, 2026-10-29);
  pf = 1.0 is the working assumption (NGED recommends it). Q0 is fixed before any score is read, as
  a share of effective capacity, at three values: 0 (the strict |*l*| case), Q_lo, and 2 Q_lo,
  where Q_lo is the 0.5th percentile of a Primary's training readings. The sensitivity to Q0 is
  reported. Inputs and lags are the metered series, folded or not.
- **Identification of the sign.** The folded loss is symmetric in the sign of *l*, and a trial on
  synthetic data (a V-shaped |*l*| against shortwave) showed that gradient descent from a positive start
  fits the V on the positive branch and never crosses zero (sign accuracy 0.50). Sign recovery
  needed two structural choices, both pre-registered here and both tested in gate G: a monotone
  constraint (non-increasing) on every shortwave input, so that generation can only lower *l*, and a
  start *l*0 = (the plain model's level) - kappa x shortwave, with kappa in {0, 0.5, 1.0} times
  the 95th percentile of the training readings per 1000 W/m². kappa is chosen on the gate's synthetic
  data only, never on D12 or any MVA-only score. The write-up says that the sign then comes from
  this structure and not from the loss alone.
- **F0 (control).** The same custom multi-quantile pinball objective with no fold (*l* unsigned),
  unit hessian, started from the plain model's per-level fit. It separates the fold from the loss
  implementation, because XGBoost's built-in `reg:quantileerror` also adjusts leaf values, which a custom
  objective cannot.
- **F1 (folded squared loss).** One model on the mean: loss (y - S(*l*))², gradient 2(S - y)·sign(*l*)/pf,
  hessian 2/pf² (a Gauss-Newton step), started from a constant. Quantiles by residual scaling:
  the empirical quantiles, by lead band and sunny flag, of y - S(*l̂*) over the last 13 weeks of the
  training window, with a second fit that stops 13 weeks earlier supplying the residuals. Signed
  quantiles are *l̂* plus the same residual quantiles (an assumption that the error is symmetric
  about the latent, stated beside every signed number).
- **F2 (folded pinball).** The 11-level pinball loss of y against S(*l*), one output per level,
  gradient -(α - 1[y < S])·*l*/(S pf), hessian constant 1, started from the plain model's
  per-level fit with the tilt. Quantiles are sorted at the end to avoid crossing. The 11
  signed quantiles are the signed outputs *l*, not folded.
- **F3 (sign then magnitude).** Built only if F1 and F2 both fail gate G.
- **Weekly refits for the final arm; monthly refits for screening.** Seed, rows, runs, and scoring rows
  are the plain model's.

## Gate G: the known-answer test (read before any MVA-only number)

At each of the six signed-MW Primaries (D5, D9, D10, D11, D14, D16), add a synthetic photovoltaic
plant (the physics model of `studies.pv_physics`, two plants: east-west and south-west, as in
experiment 110) to the signed series at levels 0, 1.0, and 2.0, sized so the injected output
makes reverse flow common on sunny middays (the sizing of experiments 358 and 366). The result is a signed
truth with real reverse flow. Convert the series to synthetic MVA S = sqrt(P² + Q²) at four reactive settings: pf 1.0 with Q = 0, pf 0.97
(Q = P tan(arccos pf)), a constant Q of 1.0 point of effective capacity, and pf 0.97 with that constant Q
(sign discarded). Replace the training labels and the lags by that synthetic MVA, and train the folded models on
it alone. Report, against the true signed series: mean absolute error in points of effective
capacity, sign accuracy on reverse-flow half-hours, and the CRPS of the recovered signed
quantiles, for F0 (read as signed), F1, and F2, at each reactive setting with the fold assuming pf 1.0
and Q0 in {0, Q_lo, 2 Q_lo}, plus pf 0.97 with the fold assuming 0.97, and for kappa in {0, 0.5, 1.0}, with and
without the monotone constraint. No metered generator output enters, so nothing needs
anonymising.

## Reading at the MVA-only Primaries

- Per-Primary table, worst Primary, leave-one-out, tails, and the load-shift curtailment rows follow
  `demand-v1` (curtailment cost is not defined for MVA-only meters; the label says so).
- Sunny-half-hour bias (median minus actual) at D12 and D8, by lead band, for plain, F0, F1, and F2.
- Time-series figures at D12 (a Primary, so it may be shown) for sunny weeks: observed MVA, the plain median,
  the folded fit, and the recovered signed MW.
- Recovery check at D12: the recovered latent against the rule-chosen fit and the signed fit of
  experiments 366, 370, and 374 (capacity and reverse-flow estimates). D12 has no signed truth, so the page
  says what is and is not validated.
- Placebos D5 and D10. Tails (pinball and coverage at 0.01, 0.05, 0.95, 0.99) per group and Primary.
  Interval width and coverage on half-hours whose lags are missing, with one simple widening rule if coverage falls short.
  Reactive power assumed zero, stated beside every MVA-only number.

## Plan review triage (Sonnet reviewer, 2026-10-29)

Accepted, and now part of the plan:

- **Signed quantiles.** F2's 11 outputs are folded quantiles, so F2 reports a signed *point*
  forecast only (the median latent). Signed-quantile CRPS is reported for F1 only, with the residual
  multiplied by sign(*l̂*) so that it moves the signed value the right way.
- **Does the fold act?** Report the share of training rows and of sunny middays with *l̂* < 0 at
  the last round. If F2 equals F0 with no crossings, the hypothesis was not tested, and the page
  says so. The mechanism is the function class plus the tilted start, not the loss alone.
- **Details.** The smooth fold removes the kink at zero. The pinball hessian is 1/pf². Every
  DMatrix (fit and predict) gets the explicit base margin, and `predict(output_margin=True)` returns
  the latents. A finite-difference gradient test and a K = 3 smoke test run first. In the
  known-answer test the start fit is the plain fit on synthetic MVA.
- **Gate G.** Pooled over the six Primaries with no Primary under 60% sign accuracy. Reverse flow means
  true P below -1 point of effective capacity, with at least 200 such half-hours per Primary. Sign
  accuracy is reported as balanced accuracy with the false-reverse rate. Baselines: predict +|y|, and
  a threshold rule on ENS shortwave fitted on the training window. The 80% gate is a threshold and the
  85% in P1 is a prediction.
- **Contrasts.** The confirmatory family is C1 and C2 at the MVA-only pooled group and D12 (k = 4);
  C3, C4, and C5 are descriptive. Per-Primary 95% intervals are descriptive. Weekly intervals are
  post-selection from monthly screening, and the page says so. An arm that passes G but shows no
  C1 gain changes nothing.
- **C4 and P4.** At the signed-MW six, F2 trains on |P| as MVA and is scored against |P| for both
  arms, against the protocol's per-Primary tolerance of +0.05 points.
- **Known-answer design.** Plant size comes from training-window data only. A constant-Q setting
  (above) tests the floor. The page says injection is the easy case and that D12 stays unvalidated by it.
- **F1 residuals.** The calibration window respects the protocol's embargo, and the sunny flag
  uses ENS shortwave. Residuals are slightly wide because the final model sees more data.
- **D12 recovery check.** Sign agreement on sunny middays and mean absolute difference in points
  of effective capacity against the signed fits of experiments 366, 370, and 374, labelled a
  consistency check, not validation.
- **Code to read.** Experiment 365 and 366 rule code and experiment 358's known-answer code are
  copied for the synthetic-MVA device. The 11 levels include 0.01 and 0.99 (checked in
  `e00087_common.LEVELS`).

## Constraints

Sonnet only. At most 3 threads and `MemoryMax=16G` per job. The GPU is shared; a custom objective runs
its gradient on the CPU, so the fits use `hist` on the CPU if the GPU transfer is the bottleneck. No
metered generator series with its name or ID. Never touches production code.
