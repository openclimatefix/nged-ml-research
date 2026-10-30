---
id: "00408"
slug: xgboost-signed-label-and-cams-columns-at-mva-only-primaries
theme: production-reference-and-best-guess
also_relevant_to:
  - demand-inputs
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v1
size: >-
  complex. Stored: fitted boosters, per-arm quantile forecasts, and small result files in this
  experiment's private folder only; experiments 381, 387, 399, 366, and 152 are read and never
  written. Designs: more than one would test the hypothesis (a signed label, a sign-then-magnitude
  pipeline, a custom objective), and experiment 387 already ruled out the custom objective, so the
  plan fixes a label-based design and its stop rules before any score is read. Callers: the fit
  script copies experiment 381's `run_job` and experiment 399's cleaned-power patch, and the known-
  answer scripts copy experiment 387's synthetic-series and gate helpers. Serving path and
  degradation rule: not touched (this repository holds neither).
hypothesis: >-
  At the eight MVA-only Primaries, the monthly XGBoost demand model gives a signed real-power
  forecast, without losing skill on the MVA reading, when (a) two CAMS columns (the unit solar
  shape and the clear-sky index of the half-hour itself) are added to its inputs and (b) reverse
  flow is labelled before training: a reading is flipped to negative only on very sunny
  half-hours, where a physics rule says the substation exports power in the shape of solar output.
prediction: >-
  Written before any score is read. P1 (known answer, injection level 2.0). On the six signed-MW
  Primaries with a metered solar farm's output injected and the sign then discarded, the A2 rule
  recovers the sign with pooled balanced accuracy of at least 0.85 and every Primary at least 0.70,
  a false-reverse rate of at most 5%, and a signed mean absolute error below the plain model read
  as signed. P2. The placebos stay under 1% false reverse flow on sunny middays at D5, D9, D11, D14,
  and D16. P3. The two CAMS columns (A1) move days 1 to 10 CRPS by -0.05 to -0.20 points of
  effective capacity at D12 and by -0.03 to +0.03 points at the other seven MVA-only Primaries.
  P4. The signed label (A2 at tau = 0.7) moves CRPS by -0.10 to +0.10 points at D12 against A1 and
  by -0.03 to +0.03 at the other seven. P5. The shuffled-clear-sky-index arm (A2-shuffle) loses the
  gain A2 shows at D12. P6. Sunny-half-hour bias at D12 falls from about -5 points (experiment 381)
  towards 0 at day 1 and stays near -5 at days 8 to 10, because CAMS arrives about a day late and
  cannot describe the future weather.
decision_rule: >-
  Stage 1 (known answer) is read before any D12 or MVA-only number. It passes at level 2.0 when
  all of: pooled balanced accuracy at least 0.85 with every Primary at least 0.70; false-reverse
  rate at most 5%; balanced accuracy at least 0.03 above a shortwave-threshold rule fitted on the
  training window; signed MAE over all daylight half-hours no worse than the plain model read as
  signed. If it fails, stop and report: the A2 arms are not run. If it passes, repeat at level 1.0
  with the same criteria. The placebos can each fail. Stage 2 contrasts, all in CRPS points of
  effective capacity on the shared rows of `fold_a_daily_gap_v1`, days 1 to 10, paired month
  bootstrap, 20,000 resamples, 99% with Bonferroni k = 3: C1 = A1 - A0 pooled over the eight
  MVA-only Primaries; C2 = A2 (tau 0.7) - A1 pooled over the eight; C3 = A2 (tau 0.7) - A1 at
  D12. Descriptive: A2 (tau 0.5), A2-shuffle, the GAM stack (experiment 377's stack without S
  below 168 h), BG2, the worst Primary, leave-one-out. Porting rule for the CAMS columns: C1's
  interval lies below zero and no Primary is worse than +0.05 (a column costs nothing in
  production, so the threshold is zero). Porting rule for A2 at D12 only, labelled "inferred MW":
  the gate and every placebo pass, the upper ends of C2 and C3 are at or below +0.05, and D12's
  tail coverage is within 0.02 of A1. Stop rule: stop signed-latent XGBoost work at the MVA-only
  Primaries if the gate fails at level 2.0, a placebo invents reverse flow above 1%, or C3's upper
  end exceeds +0.10. This would be the fifth fold treatment after experiments 152, 372, 374, and
  387, so the next step is the maintainer's decision.
metric: >-
  Continuous ranked probability score from the 11 quantiles, mean absolute error of the median,
  tail pinball and coverage at 0.01, 0.05, 0.95, and 0.99, twCRPS, and avoidable-curtailment cost,
  in points of effective capacity from `studies.eval_protocol`; leads 0.5 to 336 hours; headline
  band days 1 to 10. For the signed forecast: balanced accuracy, recall and false-reverse rate of
  its sign, and signed mean absolute error against known signed truth (injected data only).
code: ml-experiments/00408-xgboost-signed-label-and-cams-columns-at-mva-only-primaries/
production_status: not-for-production
adopted_in: none
date: 2026-10-30
row_set_id: >-
  fold_a_daily_gap_v1
benchmark: >-
  A0, experiment 381's `monthly/control` recipe refitted here on cleaned data (see the plan below).
  The GAM stack, BG2, manual_heuristic, and climatology are scored as references from stored
  forecasts.
---

# Can the monthly XGBoost model infer signed MW from MVA readings, in lab conditions?

**Plan only; no score has been read.** At 8 of the 16 Primaries ("MVA-only": D2, D3, D4, D8, D12,
D19, D21, D22) the meter records apparent power, which has no sign, so reverse flow from embedded
solar looks like demand. Experiments 152, 372, 374, and 387 each tried a different fold treatment
and none gave a signed forecast that could be trusted. This experiment gives the monthly XGBoost
model two solar columns from CAMS (the Copernicus satellite irradiance product, a realised estimate
of past weather) and labels reverse flow by a physics rule before training. It first checks that
rule on six Primaries whose meters do record sign, after adding the recorded output of a metered
solar farm, so that the true answer is known.

## What changed since the brief was written

- **Experiment 387 merged** (not supported: a folded loss leaves the sign set by the start, and its
  custom objective scored about 0.11 points worse than the built-in quantile loss). This plan
  therefore uses only the built-in `reg:quantileerror` loss and a label, as the brief says, and
  never a custom objective.
- **Standing rules added to `common.md` on 2026-10-30** are applied: monthly refits only; cleaned
  data everywhere with the `substation_zero` half-hours dropped (through experiment 399's
  `scan_power` patch), with the count per fold reported; the D15 and D18 rows before 2025-03-01
  cut in training, lags, inference inputs, and scoring actuals; every fitted booster saved as
  `.ubj` in the experiment's data folder; tails and avoidable-curtailment cost reported; two
  example probabilistic forecasts plotted (a day-ahead forecast and a 14-day forecast, at D12 in
  winter and summer, with any metered generator anonymised); the seven-step process with a fresh
  Sonnet agent at each review.
- **Stored control not reused for A0.** Experiment 381's `monthly/control` was fit on the uncleaned
  cache without the D15 and D18 cut. A0 is therefore refit here, as the brief says, and experiment
  381's stored forecasts serve only the code check (D5 and D12, within 0.05 CRPS).
- **Scope unchanged.** The plan's contrasts, thresholds, and stop rules are the brief's.

## Design

- **Base recipe (all arms).** Experiment 381's monthly control: 12 refits with origins 2025-07-01
  to 2026-06-01, built-in `reg:quantileerror` with 11 quantiles, 500 rounds, eta 0.05, depth 6,
  seed 0, the default constant `base_score`, `hist`, 3 threads. Fits at 3 threads, no GPU (many
  small per-Primary fits, as experiment 381 found), memory capped at 8 GB.
- **CAMS source and delay.** `cams_demand_points.parquet` (CAMS solar radiation time series,
  hourly global horizontal irradiance at each Primary's coordinates, 2019-09 to 2026-09); each
  hour-ending value is repeated on the two half-hours that end inside the hour
  (`studies.dp_features`). The service publishes with about a 1-day delay, so the columns cannot
  describe the first day of lead in production. Lab conditions use the half-hour's own CAMS value.
  `u_t` is `studies.dp_features.unit_pv_shape`; `k_t` is CAMS GHI over the Haurwitz clear-sky GHI
  (`studies.baselines.haurwitz_w_m2`) at the half-hour's solar zenith.
- **Signed-label rule (A2), refreshed at each refit from the training window only.** The gate is
  open at a Primary when the Spearman correlation of the reading with `u` on summer-midday rows
  with `k` at least 0.7 exceeds +0.3, or the 1st percentile is below 1.0 point of capacity
  (experiment 366's rule, not re-tuned). `D_ref(h)` is the median reading at UTC hour `h` on
  April-to-September training rows with `k` below 0.3. `C` is the median of `(y + D_ref)/u` on
  clear summer-midday training rows. A reading is reverse flow when the gate is open, `k` is at
  least tau, `C u` exceeds `D_ref(h)`, and `C u - y` is at least 2 points. The label is `-y` for
  reverse flow and `+y` otherwise. The lags stay unsigned readings.
- **Shape check (maintainer's rule), pre-registered.** For each Primary, the correlation between
  the inferred reverse-flow magnitude and `C u` on reverse half-hours is reported. A correlation of
  at least 0.8 counts as following the shape; below 0.5 counts as not following it.
- **Fold at serving.** Build a piecewise-linear CDF `F` from the 11 signed quantiles and set
  `H(x) = F(x) - F(-x)` for `x` at least 0. The MVA quantile at level alpha is the inverse of `H`.
  This is the identity whenever q01 is at least 0.
- **Arms.** A0 (control, refit here). A1 (A0 plus `u_t` and `k_t`). A2 (tau = 0.7, registered),
  A2 (tau = 0.5), A2-off (tau infinite; must equal A1 to the last bit, a code check), and A2-shuffle
  (`k` permuted across days within the same month and hour before labelling). A1 is also run at the
  other 8 Primaries as a no-harm check. The A2 arms are run only after the known-answer gate
  passes at level 2.0.
- **Known-answer test, run first.** Sites: D5, D9, D10, D11, D14, D16. Three different metered
  solar farms rotate across the Primaries (the injection frame of experiment 39 that experiment 152
  uses), scaled to 2.0 times, then 1.0 times, the median sunny-midday reading of the training
  window. The farm output never enters the inputs, `C`, or `u`. Reading is
  `y' = sqrt(P'^2 + Q^2)` with `Q = 0`. The farm data stay in the private store and no farm name,
  ID, or series is published. Placebos on the real readings: at most 1% false reverse flow on sunny
  middays at D5, D9, D11, D14, and D16; D10's 130 real reverse half-hours reported; forcing the gate
  open at D5 must leave the MVA score about unchanged and make the signed MAE worse.
- **Scoring.** `demand-v1` at all 16 Primaries on `fold_a_daily_gap_v1`; days 1 to 10 headline and
  by lead band; q01, q05, q95, q99 tails; avoidable-curtailment cost; cold winter nights; sunny
  half-hours shown three ways (realised `sunny_clear_sky_index`, CAMS `k` at least 0.7,
  `sunny_unit_shape`); D12's sunny bias by lead band for A0 and A1. Half-hours dropped as
  `substation_zero` are counted per fold.
- **Reuse.** Stored GAM stack, BG2, `manual_heuristic`, and `climatology` forecasts are read as
  references; 16 Primaries x 12 refits x 3 fit arms are fitted here.
- **Reviews.** (1) Code is written by Sonnet. (2) A fresh Sonnet agent reviews it before the first
  fit (leakage, the `train_end` cut, scoring, matching the copied recipe). (3) A second review if
  the first flags many problems. (4) Run. (5) A fresh Sonnet agent re-derives the headline numbers
  and the audit triggers are run. (6) Write-up. (7) A fresh Sonnet agent reviews the write-up for
  numbers and prose.

## Risks

- D12's reverse flow is inferred, not measured (fitted PV capacity about 148 points, 6.9 MW,
  unverified), and D12's solar response bottoms out at 100 to 200 W/m2, so a "very sunny" gate
  probably mislabels moderate-sun reverse flow there.
- A solar-only gate mislabels reverse flow from other generators; the assumption is unsafe at
  Primaries with other generation (D15 is negative on 78% of its half-hours, 63% of them at night).
- Reactive power is unknown, so every result assumes `Q = 0`, except a power-factor-0.97 check at
  the July, November, and March refits.
- The brief's exploration used ERA5 with Haurwitz, not CAMS, so `k` is recomputed from CAMS here.
