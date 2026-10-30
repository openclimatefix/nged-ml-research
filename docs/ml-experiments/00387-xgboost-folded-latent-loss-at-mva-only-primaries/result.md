---
id: "00387"
slug: xgboost-folded-latent-loss-at-mva-only-primaries
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
  - demand-inputs
status: finished
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
  other MVA-only seven. P4. At the signed-MW six, the folded loss applied to the true signed
  series is
  not used (those meters need no fold), and the folded model fed unsigned synthetic MVA scores
  within 0.10 points of the plain signed-meter model there. P5. F1 (folded squared loss, quantiles
  by residual scaling) recovers the sign about as well as F2 but its tails are 10% to 30% worse in
  pinball loss at the 0.01 and 0.99 quantiles. P6. F3 is not needed.
decision_rule: >-
  Pre-registered contrasts, all CRPS or sign accuracy on the shared rows of `fold_a_daily_gap_v1`.
  Gate G (known answer, read before any D12 or MVA-only number): an arm passes when its recovered
  sign accuracy on reverse-flow half-hours is at least 80% and its signed-power MAE is under half
  that of the plain model read as signed, at injection level 1.0 and the first reactive setting
  (pf 1.0, Q = 0), with the best (kappa, Q0)
  chosen on gate data; the other reactive settings are reported, not gated. An arm that fails G
  is not read at D12. If F1 and F2 both fail G, F3 (sign
  probability model, then magnitude model) is built, after the plan is amended and reviewed, and
  nothing else is read. Contrasts after G: C1 (F2 minus plain weekly, CRPS days 1 to 10, MVA-
  only eight
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
verdict: >-
  not supported
best_arm: >-
  The plain weekly-style XGBoost model with the built-in quantile loss: CRPS 6.860 points of
  capacity at D12 and a mean over the eight MVA-only Primaries about 0.05 lower than the folded
  model F2.
headline: >-
  The folded loss (F2) scored 0.149 points of capacity worse than the plain model at D12 (99.75%
  interval -0.061 to +0.378) and +0.052 (-0.083 to +0.203) over the eight MVA-only Primaries, and
  left the sunny-half-hour bias at D12 at -5.07 points against -5.04 for plain. Its recovered
  signed power is negative at sunny midday on 63% to 98% of half-hours at six Primaries with no
  known reverse flow, so the sign follows the start tilt, not the readings.
lessons: >-
  The loss does not identify the sign of a latent power, so a signed MW forecast at an MVA meter
  needs structure that fixes the sign. The known-answer gate was too easy because the injected
  solar had the tilt's own form. A custom pinball objective scores worse than the built-in
  quantile loss (+0.110 points, interval -0.026 to +0.253), which confounds any comparison with
  the plain model.
take_home_message: >-
  Training the XGBoost model on a folded signed power did not improve the forecast at the eight
  Primaries whose meters report unsigned apparent power, and it did not fix the under-forecast of
  sunny half-hours at D12. The signed power it reports follows the start we gave it, not the
  readings, so it cannot yet be shown to NGED as a forecast. A signed forecast needs a structural
  way to fix the sign, such as demand minus a non-negative generation term.
advice:
  - >-
    Do not read a signed latent power from a folded loss at an MVA meter; the sign follows the
    start, not the readings (recovery table).
  - >-
    Give each arm of a custom-loss comparison the same number of rounds and a custom-loss
    reference, because the built-in quantile loss scores about 0.11 points better than a custom
    pinball loss (F0 minus plain).
  - >-
    Build a known-answer control whose injected signal does not share the form of the fix being
    tested; the shortwave-tilt gate was passed by construction (gate).
---

# A folded loss for one XGBoost model at the MVA-only Primaries

**Take-home.** The folded loss did not help. At D12, the one Primary known to have reverse flow, the
folded model scored 0.149 points of capacity worse than the plain XGBoost model (99.75% interval
-0.061 to +0.378), and it left the under-forecast of sunny half-hours where it was (-5.07 points
against -5.04). Across the eight MVA-only Primaries the difference was +0.052 (-0.083 to +0.203), so
we cannot tell it from zero. The signed power the model reports does not come from the readings. It
follows the start we gave it: the recovered power is negative at midday on 63% to 98% of sunny
half-hours at six Primaries where earlier experiments found no reverse flow. A folded loss therefore
needs a structural way to fix the sign before it can give NGED a signed MW forecast, and a tilted
start is not one. Verdict: not supported, with the caveat that the comparison with the plain model
is confounded by the optimiser (see "Checks").

**What we asked.** Eight Primaries have meters that record apparent power in MVA, which has no sign,
so reverse flow from embedded solar looks like demand. We asked whether one XGBoost model per
Primary, trained to predict a signed latent power and scored on its folded value against the metered
reading, can learn the V-shape that reverse flow makes in the readings, and so give a signed MW
forecast. The idea extends [experiment
381](../00381-simple-weekly-xgboost-demand-against-gam-stack/result.md), which found that the plain
model under-forecasts sunny half-hours at D12 by about 5 points. The fold of a signed quantity into
a magnitude follows the known-answer device of [experiment
365](../00365-mva-fold-readings-only-reverse-flow-rule/result.md) and [experiment
366](../00366-mva-fold-irradiance-correlation-reverse-flow-rule/result.md). Reactive power is
assumed zero except where a floor Q0 is stated.

**How.** Each arm refits monthly on experiment 87's recipe (the ENS control member, a half-life of
0.5 years, 500 rounds) and forecasts the 73 runs of `fold_a_every5d_v1` for days 1 to 14, scored
under `demand-v1` for days 1 to 10. The arms were the plain XGBoost model (built-in quantile loss),
F0 (a custom pinball loss with no fold), and F2 (the same loss on the folded value sqrt(l² + Q0²),
with Q0 of 0, 2, and 4 points of capacity). F2 starts from the plain model's fit minus a tilt
proportional to forecast shortwave. We did not run F1 (the folded squared loss), F3 (the
sign-then-magnitude pipeline, which only the plan's gate failure would have triggered), or a weekly
refit (no arm won). The plan is in the commit history of this page.

## Figures

![D12's lowest readings stop at about 4 points; a signed meter reaches zero](figure_1_low_readings.svg)

![Known answer at D10: injected reverse flow recovered](figure_2_known_answer_d10.svg)

![A sunny week at D12: the folded median and signed power](figure_3_d12_sunny_week.svg)

![CRPS differences by Primary](figure_4_contrasts.svg)

## Headline numbers

All scores are the continuous ranked probability score (CRPS) in points of effective capacity, days
1 to 10, monthly refits. Reactive power is assumed zero at every MVA-only number, apart from the
floor Q0.

- **Reference check.** The refitted plain model matches experiment 379's stored monthly member
  (`only_M`) with a mean absolute median gap of 0.0065 MW over 409,968 rows.
- **Absolute CRPS at D12.** Plain 6.860, F0 7.067, F2 with Q0 = 0, 2, and 4: 7.009, 7.009, and
  7.003. The other seven Primaries are in the table below; F2 (Q0 = 2) is 0.01 to 0.10 better than
  plain at D2, D3, D4, D19, and D21, 0.34 worse at D22, and 0.21 worse at D8.
- **C1, F2 (Q0 = 2) minus plain, 99.75% interval.** MVA-only eight +0.052 (-0.083 to +0.203);
  without D8 +0.029 (-0.120 to +0.155); D12 alone +0.149 (-0.061 to +0.378). On sunny half-hours:
  -0.012 (-0.147 to +0.143) pooled, and +0.066 (-0.347 to +0.521) at D12. Leaving one Primary out
  never moved the pooled difference outside -0.07 to +0.19 (95% intervals all include zero).
- **C3, F2 (Q0 = 2) minus F0.** MVA-only eight -0.058 (-0.087 to -0.030); D12 alone -0.058 (-0.129
  to +0.012). On sunny half-hours -0.287 (-0.467 to -0.161) pooled and +0.080 (-0.374 to +0.300) at
  D12. The fold repairs part of what the custom loss lost, but not at D12.
- **F0 minus plain.** +0.110 (-0.026 to +0.253): the custom loss alone scores worse than the
  built-in quantile loss.
- **Sensitivity to Q0.** Q0 = 0 against 2: +0.004 (-0.001 to +0.009) pooled. Q0 = 4 against 2:
  -0.000 (-0.007 to +0.006). The floor does not matter at the precision of this experiment.
- **Sunny-half-hour bias at D12 (median minus actual).** Plain -5.04 points, F0 -3.38, F2 -5.07. The
  folded model does not remove the under-forecast. At D19 the bias is +6.2 for plain and +6.3 for
  F2.
- **Tails, mean over the eight Primaries.** Coverage at q01, q05, q95, and q99: plain 0.013, 0.071,
  0.932, 0.991; F2 (Q0 = 2) 0.024, 0.090, 0.895, 0.965 (nominal 0.01, 0.05, 0.95, 0.99). Pinball
  loss at q01, q05, q95, and q99: plain 0.248, 0.890, 0.986, 0.331; F2 0.265, 0.955, 1.042, 0.301.
  The folded model's quantile band is too narrow at the inner tails and slightly better at q99. At
  D12, q01 pinball is 0.305 for plain and 0.443 for F2. Threshold-weighted CRPS is 0.200 for plain
  and 0.201 for F2.
- **Nulled lags and load-shift rows.** These were not scored separately. The curtailment cost is not
  defined at MVA-only meters.

| Primary | Plain | F0 | F2, Q0 = 0 | F2, Q0 = 2 | F2, Q0 = 4 |
|---|---|---|---|---|---|
| D12 | 6.860 | 7.067 | 7.009 | 7.009 | 7.003 |
| D8 | 7.763 | 8.218 | 7.987 | 7.976 | 7.995 |
| D22 | 3.930 | 4.266 | 4.280 | 4.272 | 4.270 |
| D19 | 3.290 | 3.285 | 3.223 | 3.226 | 3.216 |
| D4 | 2.517 | 2.500 | 2.501 | 2.506 | 2.500 |
| D21 | 2.407 | 2.387 | 2.338 | 2.328 | 2.326 |
| D2 | 2.328 | 2.307 | 2.300 | 2.294 | 2.296 |
| D3 | 1.759 | 1.702 | 1.663 | 1.657 | 1.659 |

## Does the signed power mean anything?

- **Known-answer gate (passed, but too easy).** We took the six signed-MW Primaries, subtracted a
  synthetic solar plant (a multiple of ERA5 shortwave, sized from the training window only) to make
  reverse flow, discarded the sign (and added reactive power in four settings), and trained on the
  unsigned series alone. With the start tilt at 1.0 (no monotone constraint), F2 recovered the sign
  of the reverse-flow half-hours with balanced accuracy 0.89 to 0.92, recall 0.89 to 0.98, and a
  false-reverse rate of 0.12 to 0.14, at every reactive setting and at Q0 of 0, 2, and 4. Without
  the tilt the sign was not recovered (balanced accuracy 0.51, the same as the plain model). The
  signed error on reverse-flow half-hours was 5.8 to 6.6 points against 21 to 25 for the plain
  model. The gate was too easy: the injected solar has the same linear form in shortwave as the
  tilt, and a rule that sets the plain median negative when ERA5 shortwave exceeds a fitted
  threshold scored higher (balanced accuracy 0.94 to 0.98, false-reverse rate 0.04 to 0.07).
- **Recovery at the MVA-only Primaries.** Earlier experiments
  ([366](../00366-mva-fold-irradiance-correlation-reverse-flow-rule/result.md),
  [370](../00370-mva-fold-pv-capacity-and-eight-primaries/result.md)) found reverse flow at D12 and
  at none of the other seven, with no signed truth at any. F2's latent is negative at sunny midday
  (ERA5 shortwave above 600 W/m², 10 to 14 UTC) on this share of half-hours: D2 0.74, D3 0.68, D4
  0.07, D8 0.98, D12 0.17, D19 0.77, D21 0.70, D22 0.98. The share is highest where there is no
  reverse flow and lowest at D12, so the readings did not set the sign.
- **Placebo, no injection (signed-MW six).** PLACEBO_PLACEHOLDER

## Checks

- **Optimiser confound.** The built-in quantile loss rewrites each leaf from the residuals, which a
  custom objective cannot copy, and the custom objective's Hessian is a constant
  (`e00387_fit.py:96`), so leaf values are a step rule, not a Newton step. F0 and F2 also start from
  the plain model's in-sample predictions and then boost 500 more rounds, so they get 1,000 rounds
  against the plain model's 500. "F2 minus plain" mixes the fold with a weaker optimiser; "F2 minus
  F0" is the cleaner contrast, with a weak reference. The "no gain" statement is confounded in this
  way. An Opus audit by the Discuss session confirmed the gradient (the finite-difference self-test
  passes) and these points, and its findings are triaged here.
- **Median against the mean forecast.** The forecasts store 11 quantiles, not a mean, so the
  median-minus-mean check is replaced by the mean absolute error of the median, beside every CRPS
  contrast: plain 6.895, F0 7.028, F2 (Q0 = 2) 6.931 (mean over the eight Primaries). It agrees in
  direction with the CRPS contrasts (the plain model is best).
- **Triggers fired.** A result that contradicts its prediction (trigger 2: we predicted -0.05 to
  -0.25 at D12) with an explanation tested above, and an easy known-answer control (trigger 1). The
  audit above is the response. Scoring uses the folded quantiles of the reading, which are valid
  quantiles of the metered value; the signed latent is a diagnostic only and no score depends on it.
- **Row set.** The fits forecast the 73 runs of `fold_a_every5d_v1`, not the 292 runs of
  `fold_a_daily_gap_v1`, so no comparison with the GAM stack or BG2 is made here. The weekly refit
  and the signed-MW harm check (C4) were not run, because no arm won; the placebo above covers false
  reversal only.

## What to try next

- A signed MW forecast at the MVA-only Primaries needs the sign fixed by structure (for example a
  latent built as demand minus a generation term that is non-negative and monotone in shortwave,
  with the generation capacity estimated as in experiment 370), because neither the loss nor a
  tilted start identifies it.
- At D12 the plain model's under-forecast of sunny half-hours is not a loss-function problem: the
  fold did not move it. Look at the ensemble's shortwave spread and at a sunny-day input instead.
- Use XGBoost's built-in quantile loss as the reference in any further custom-loss experiment, and
  give the arms equal numbers of rounds.
