---
id: "00381"
slug: simple-weekly-xgboost-demand-against-gam-stack
theme: production-reference-and-best-guess
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: other
protocol: demand-v1
size: >-
  medium. Stored: forecasts and small result files in this experiment's private folder only;
  experiments 87, 371, 377, and 379 are read and never written. Production serving path: not
  touched. Degradation rule: not touched. Designs: more than one would test the hypothesis (the
  Economy 7 input could be a flag or a ramp, the solar summary could be a daily or a trailing
  mean); the plan fixes one of each before any score is read. Callers: the fit script imports
  experiment 87's folder; the scoring script imports experiment 377's folder and `studies.eval_protocol`.
hypothesis: >-
  A single XGBoost model per Primary, refitted weekly on the ENS control member with experiment 87's
  recipe (the BG2 weekly member), plus at most four small input or weighting changes, scores no worse
  than the GAM stack ("stack without S below 168 h", experiment 377) at every one of the 14 countable
  Primaries at days 1 to 10.
prediction: >-
  Written before any score is read. P1. The BG2 weekly member alone already matches or beats the
  stack at 9 to 11 of the 14 Primaries (experiment 379 stores it; the stack trails BG2 at 6 of 14).
  P2. The Economy 7 input lowers the score by 0.03 to 0.10 points on Economy 7 step hours and by
  under 0.02 points on all hours. P3. The solar-summary stand-in lowers the score on sunny
  half-hours by 0.05 to 0.20 points at the six signed-MW Primaries and by under 0.05 points at the
  MVA-only eight. P4. The lead-dependent half-life moves the days 1 to 10 score by under 0.03 points
  in either direction. P5. At D12 the plain model's residuals on sunny half-hours are positive (it
  under-forecasts reverse flow shown as demand), and the sunny-flag interaction removes at least
  a third of that bias.
decision_rule: >-
  Screening runs at monthly refits (12 refits, P-roll recipe is not used: whole-history, half-life
  0.5 years, 500 rounds, as experiment 87). Contrasts, pre-registered: C1 to C4, each candidate arm
  minus the monthly control, CRPS at days 1 to 10 in points of effective capacity, pooled per group,
  paired month-resampled bootstrap of `studies.eval_protocol`, read at the 99% level with k = 4
  (level 1 - 0.01/4, 20,000 resamples). An arm is kept when its interval lies wholly below zero in
  the signed-MW group or the MVA-only group and it breaks no tolerance of the protocol's table.
  The kept arms form one bundle, fitted weekly as the final arm (C5: bundle minus the stored weekly
  baseline) and scored against the stack (C6: bundle minus stack, per Primary and per group, 99%
  pooled and 95% per Primary), BG2, `manual_heuristic`, and `climatology`. Headline: the number of
  the 14 countable Primaries at which the bundle is no worse than the stack (point estimate at or
  below +0.05 points, the protocol's per-Primary tolerance) and the number at which its 95% interval
  lies wholly below zero, with the worst Primary named. Placebos: D5 and D9 for the MVA remedy and
  D10 and D16 for the Economy 7 input, chosen here before scoring. No adoption claim.
benchmark: >-
  The GAM stack from experiment 377, BG2 and its weekly member alone (experiment 379 `only_W`),
  `manual_heuristic`, and `climatology`, on `fold_a_daily_gap_v1`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles, mean absolute error of the
  median, and twCRPS, in points of effective capacity, from `studies.eval_protocol`; leads 0.5 to
  336 hours; headline band days 1 to 10.
code: ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack/
production_status: not-for-production
adopted_in: none
date: 2026-10-28
---

# A single weekly XGBoost model for Primary demand against the GAM stack

**Plan (status `planned`).** The question is whether a plain XGBoost model, retrained every week and
with no stacked parts, can match the GAM stack at each of the 16 Primaries. The baseline is the
weekly member of BG2 (experiment 87), scored under the `demand-v1` protocol by experiment 379. The
baseline already holds the three recent-power lags (experiment 371 found these carry the gap at D8),
the local hour and day of week, the distance to the nearest bank holiday and to Christmas, and
`lead_hours`, so those candidates are in the control and are not tested again.

## Pre-registered arms

- **Control.** Experiment 87's `n1h` recipe, refitted monthly for screening (the stored `only_M` arm
  of experiment 379 is the reference; the refit must reproduce it within 0.05 points before any other
  arm is read) and weekly for the final comparison (the stored `only_W` arm). `colsample_bytree` and
  `colsample_bynode` stay at 1.0 in every arm, so no arm gains from column subsampling.
- **A1, Economy 7.** Adds two inputs: a 0/1 flag for the local half-hours 1 to 6 (the protocol's
  Economy 7 step hours) and the number of half-hours since local 00:30, clipped to 0 to 12.
- **A2, solar summary stand-in.** Adds the mean forecast shortwave over the local day of the valid
  time (the same ensemble member, the same run) and the mean over the previous local day. It is a
  cheap stand-in for the GAM's two-number solar summary S.
- **A3, lead-dependent recency.** Row weights use a half-life of 0.25 years for rows at leads under
  48 hours and 1 year for rows at longer leads, in place of the single 0.5-year half-life.
- **A4, MVA remedy (eight MVA-only Primaries only; the signed-MW six are controls).** Adds a sunny
  flag (forecast shortwave over the Haurwitz clear-sky irradiance above 0.6) and its product with
  shortwave. A second variant reads the rule-chosen fit of experiments 366 and 374 as an input column
  if the stored fit exists for the run origins and uses data from before each origin only; the plan
  records at the start of the work whether it does, and drops the variant otherwise.
- **Bundle.** The arms kept by the decision rule, fitted weekly.

## Diagnostics before the arms

- **MVA handling.** Residuals of the control (median minus actual) on sunny half-hours at D12 (known
  reverse flow), at the other seven MVA-only Primaries, and at the signed-MW six, by group and lead
  band, before and after A4.
- **Reference rows.** The stack from experiment 377, BG2, `manual_heuristic`, `climatology`, and the
  stored weekly baseline, each reported with its absolute score.

## Tails and nulled inputs (maintainer, 2026-10-28)

Tail pinball loss and coverage at the 0.01, 0.05, 0.95, and 0.99 quantiles are reported per group
and per Primary beside CRPS and twCRPS, and the take-home names the tail result. The control and every
arm already carry the three recent-power lags, which are missing (NaN) when the telemetry has a gap
near the origin. The write-up reports mean 5%-to-95% interval width and coverage on half-hours whose
lags are missing against half-hours whose lags are present, and tries one simple widening rule
(scaling the quantile distances from the median by a factor fitted on the earlier months of the
same fold) if coverage on the missing rows falls short.

## Avoidable-curtailment cost (maintainer, 2026-10-28)

Because the arms feed missing lags on event rows, the write-up reports the protocol's avoidable
curtailment cost (`studies.eval_protocol`) during each hand-labelled load-shift event, in the 7 days
after it ends, and across all rows, per group and per Primary, beside CRPS and the tail measures. The
cost is defined only for the signed-MW six, so the MVA-only eight carry the label "not defined".
The take-home names it.

## Constraints

Sonnet only. At most 3 threads and `MemoryMax=16G` per job. The GPU is shared. Tail calibration,
slices, load-shift rows, per-Primary table, leave-one-out, and the D8 own-residual and donor-spread
arms follow the protocol page. Reactive power is assumed zero at the MVA-only Primaries.
