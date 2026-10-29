---
id: "00382"
slug: origin-lags-as-gam-terms-at-d8-and-the-stack-primaries
title: >-
  Do three recent-power lags, added to the GAM stack as smooth terms, close the gap to monthly
  XGBoost at substation D8?
theme: lagged-power-and-error-correction
also_relevant_to:
  - production-reference-and-best-guess
status: finished
kind: change
protocol: demand-v1
size: >-
  complex. Stored: forecasts and small result files in this experiment's private folder only;
  experiments 371 and 377 are read and never written. Serving path: not touched (this repository
  has none). Degradation rule: not touched. Designs: more than one would test the hypothesis (the
  lags as terms of the stage 1 GAM design, or as terms of the stage 2 correction); this plan fixes
  the stage 2 form and says why. Callers: the scripts import experiment 377's folder
  (`e00377s2_common`, `e00377s2_corrections`, `e00377s3_stack`, `e00377_adapter`) and
  `studies.eval_protocol`, and nothing imports them.
hypothesis: >-
  Experiment 371 found that monthly XGBoost leads the GAM stack at D8 mainly through three lagged
  values of recent power (the newest usable reading, the reading one day earlier at the valid
  half-hour, and the reading one week earlier). Adding the same information to the GAM stack as
  three smooth terms in its stage 2 mean correction closes most of the D8 gap to BG2 without
  harming the other Primaries.
prediction: >-
  Written before any score is read. P1. At D8, "default plus origin lags" scores better than
  "default" by 0.30 to 0.60 points of CRPS at days 1 to 10 (371 removed the lags from XGBoost and
  lost +0.550). P2. The gain shrinks with lead: larger at days 1 to 2 than at days 3 to 10. P3.
  The placebos D2 and D4 gain less than D8 (371 found +0.109 and +0.283 for XGBoost, so some
  gain there is expected, not harm). P4. The signed-MW six move by less than 0.05 points. P5. The
  month-by-month D8 gain is positive in August and December 2025 and near zero in February 2026,
  as in 371.
decision_rule: >-
  Primary demand research default 1 extended to 336 hours (experiment 377's stack) is "default";
  "default plus origin lags" adds the terms. The target is D8 (own-residual arm as D8's row; the
  donor-spread arm is the labelled second arm). D2 and D4 are the pre-registered placebos: a
  placebo fails when it moves in the same direction as D8 beyond its own 95% interval, as the
  protocol defines it, and 371's XGBoost result already shows that D2 and D4 may gain, so a failed
  placebo here is a finding about generality and not a defect. Tier: "a new fitted step", minimum
  gain 0.02 points. The protocol's six adoption conditions decide the adoption outcome, 99%
  intervals with 20,000 resamples as the headline and 95% intervals beside them, per-Primary table,
  worst Primary, and the two leave-one-out contrasts. Nothing is adopted without the maintainer.
benchmark: >-
  Experiment 377's stack as stored, BG2, climatology and the manual heuristic of experiment 87 on
  `fold_a_daily_gap_v1`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS, MAE of the median, and twCRPS in points of effective capacity from `studies.eval_protocol`;
  pinball loss and coverage at the 0.01, 0.05, 0.95, and 0.99 quantiles; headline band days 1 to
  10, with days 1 to 2 and days 3 to 10 beside it.
code: ml-experiments/00382-origin-lags-as-gam-terms-at-d8-and-the-stack-primaries/
verdict: refuted
production_status: not-for-production
adopted_in: none
best_arm: >-
  None adopted. At D8, default plus origin lags (spline) scores 9.672 against default's 9.227
  (+0.445 points, 99% interval +0.090 to +0.983); BG2 scores 7.615.
headline: >-
  Adding the three origin lags to the stack's stage 2 as smooth terms did not close the D8 gap. It
  made D8 worse (+0.445), mostly in the first three months of the scored year, when D8's fits rested
  on 21 to 29 daily origins and the spline was close to unpenalised. A post hoc version with a
  penalty that bites is neutral at D8 (+0.080, -0.012 to +0.154). At 12 of the other 13 counted
  Primaries the pre-registered arm moves CRPS by under 0.05 points; D22 is worse by 0.096.
take_home_message: >-
  Putting recent power into the GAM stack as three smooth stage 2 terms does not recover what
  monthly XGBoost gets from its origin lags: D8 scores 0.445 points worse, not 0.3 to 0.6 better,
  and the harm is concentrated in August to October 2025, when D8 had little history to fit on.
  The signed-MW six improve by 0.02 points and D2 and D4 by 0.02, and a penalised post hoc version
  gives no gain at D8 either (+0.080), so the one additive stage 2 form tried here does not reproduce
  XGBoost's use of the lags. The experiment does not say whether the Kalman filter and recipe R
  already use the same information. The lower-tail coverage at D8 is unchanged (q05 0.058 against
  0.059). The interaction between the lags and stage 1's inputs, which XGBoost can learn and a
  separate additive term cannot, is one explanation not tested here.
lessons:
  - >-
    The harm at D8 is fit variance: the pre-registered penalty weight of 10 was almost zero next to
    a Gram matrix that grows with the row count, and the August 2025 fits used 21 to 29 independent
    daily origins per lead band. The monthly difference falls from +1.89 points in August 2025 to
    +0.03 to +0.15 from January 2026, and a penalty of 1e5 (post hoc) gives +0.080 (-0.012 to
    +0.154) for the year.
  - >-
    A linear version of the three terms is nearly neutral at D8 (+0.089, 99% interval +0.004 to
    +0.162) and gains 0.022 (0.001 to 0.043) at the signed-MW six, and the heavily penalised spline
    gives +0.080 at D8 and -0.027 (-0.056 to -0.004) at the signed-MW six.
advice:
  - >-
    Do not add the origin lags to the stack as stage 2 smooth terms for D8, because the spline
    scores 0.445 points worse (days 1 to 10, 99% interval +0.090 to +0.983).
  - >-
    Test the lags as inputs to stage 1 (an origin-dependent design) or as an interaction with the
    weather terms before concluding that recent power cannot help the GAM at D8, because the stage 2
    additive form is the only design tried here.
  - >-
    Count independent daily origins, not rows, when sizing a lag term's fit, and scale its penalty
    to the row count, because the unscaled penalty let 24 coefficients fit 21 to 29 origins
    (audit; D8 +1.89 in 2025-08).
date: 2026-10-28
---

## Do three recent-power lags, added to the GAM stack as smooth terms, close the D8 gap?

Giving the forecast three recent readings of the substation's power did not help at D8, the
substation where the competing XGBoost method leads, and made D8's forecast worse. We asked whether
a forecast of electricity demand improves when it is told how much power a substation drew in the
hours just before the forecast was made. The extra readings raised D8's forecast error by 0.445
points of the substation's effective capacity (error 9.672 against 9.227). At 12 of the other 13
counted substations the error moved by under 0.05 points; at D22 it rose by 0.096.

**The three readings are the newest one available at forecast time, the reading one day earlier at
the target half-hour, and the reading one week earlier.** [Experiment
371](../00371-d8-monthly-xgboost-input-group-ablation/result.md) found that monthly XGBoost leads
the GAM stack at D8 mainly through these readings. We gave the GAM stack of [experiment
377](../00377-demand-v1-first-run-feasibility/result.md) (Primary demand research default 1,
extended to 336 hours) the same readings. The stack fits in two stages: the first predicts from
weather and calendar for each target time, so it cannot use readings taken before the forecast was
made, and the second corrects the first stage's mean forecast. We added the readings to the second
stage as smooth terms, each fitted every month on earlier months only. Each reading enters as the
reading minus the stack's own forecast for that half-hour.

**Scores are the quantile score (CRPS, the continuous ranked probability score estimated from 11
quantiles) in points of effective capacity; lower is better, and a contrast is the arm with the
readings minus the stack without them.** The "spline arm" is the pre-registered design, the "linear
arm" is a sensitivity check, and the "penalised spline arm" was added after the audit. BG2 is the
best-guess forecast of experiment 87. D8 and the other labels D2 to D22 are Primary substations. The
signed-MW six are the six Primaries that report signed megawatts. The full design is in the plan
below.

![Figure 1: change in CRPS per Primary at days 1 to 10 (95% intervals)](per_primary.svg)

![Figure 2: D8, change in CRPS by month of forecast origin](d8_by_month.svg)

**Results (days 1 to 10, spline minus default, 99% intervals, 20,000 resamples of whole months).**

- D8 (own-residual arm): +0.445 (+0.090 to +0.983); default 9.227, spline 9.672, BG2 7.615.
- Days 1 to 2 at D8: +0.108 (-0.064 to +0.391). Days 3 to 10: +0.531 (+0.115 to +1.184). Days 8 to
  10: +0.695 (+0.162 to +1.489). The harm grows with lead, the opposite of the prediction that the
  gain would be larger at days 1 to 2.
- Signed-MW six: -0.019 (-0.052 to +0.008); days 1 to 2: -0.043 (-0.061 to -0.028).
- MVA-only eight with D8: +0.055 (+0.012 to +0.125); without D8: +0.004 (-0.022 to +0.027).
- Penalised spline arm (post hoc; smoothness penalty weight 100,000 in place of 10, added after the
  audit found the pre-registered weight almost zero; not adoptable): D8 +0.080 (-0.012 to +0.154);
  signed-MW six -0.027 (-0.056 to -0.004); MVA-only eight with D8 +0.004 (-0.018 to +0.023); D2 and
  D4 -0.026 (-0.054 to -0.003); short-history group +0.016 (-0.019 to +0.063).
- Placebos D2 and D4 pooled: -0.023 (-0.066 to +0.007). The pooled change is slightly towards a
  gain, the opposite direction to D8, so neither placebo fails.
- Per Primary (95%, pre-registered spline): D8 +0.445 (+0.127 to +0.838), D22 +0.096 (+0.037 to
  +0.159), D15 +0.283 (+0.020 to +0.654, report only), D14 -0.048 (-0.074 to -0.019), D4 -0.048
  (-0.095 to -0.010). The spline wins at 10 of 14 counted Primaries. The worst Primary is D8.
- Linear arm: D8 +0.089 (+0.004 to +0.162); signed-MW six -0.022 (-0.043 to -0.001).
- D8 by month (days 1 to 10): +1.89 (2025-08), +1.46, +0.50, +0.20, +0.32, +0.03, +0.10, +0.07,
  +0.15, +0.03, +0.04 (2026-06). Leaving out August 2025 moves the pooled D8 difference from +0.445
  to +0.293.
- Donor-spread arm at D8: default 11.588, spline 12.112 (+0.524).

**Tails.** At D8 the coverage of the 0.05 quantile is 0.0589 (default) and 0.0583 (spline), of the
0.95 quantile 0.934 and 0.933, and of the 0.99 quantile 0.970 and 0.968, so the lags neither fix nor
break the lower-tail undercoverage that experiment 377 found. The printed report holds the pinball
loss at the four tail quantiles and the twCRPS per group and per Primary for all six arms. The
twCRPS contrast at D8 is +0.050 (+0.011 to +0.118) (the spline is worse); at the signed-MW six it is
-0.004 (-0.015 to +0.002).

**Nulled rows.** A row is nulled when any of the three lags is missing (a gap in the power record).
Such rows are under 2% of every Primary: 19 to 3,226 rows per Primary against 55,178 to 172,783
un-nulled rows. Their coverage of the 5% to 95% interval is erratic because the samples are tiny
(0.42 to 1.00), and interval width on nulled rows is not systematically wider, since the treatment
adds no widening rule. A widening rule was not tried, because nulled rows are under 1% of every
Primary except D15 (1.1%) and D5 (1.9%).

## Checks

- The zero-shift gate (no shift reproduces experiment 377's stored stack) was run at D2 only.
- Median minus mean forecast per site is 0.0 points for default and spline at every Primary, so the
  spread does not shift the forecast. The mean-forecast absolute error at D8 rises from 15.93 to
  16.47 points, in the same direction as the quantile score.
- Triggers that fired: 2 (the result contradicts its prediction P1, and the only tested explanation
  is the monthly pattern above, which is post hoc) and 5 (the default arm, nested inside the spline
  arm, beats it at D8). Triggers 1, 3, 4, and 6 did not fire (the placebos did not fail; the
  quantile and median-error contrasts agree in sign; leaving one month out leaves D8 positive at
  +0.293 to +0.489). Trigger 7 does not apply. An independent audit by a Sonnet agent found no
  lookahead (it corrupted readings after the newest usable one and the anomalies did not change) and
  no code bug, and reproduced D8's stored forecast exactly. It found that the pre-registered penalty
  was almost zero and that D8's early fits rested on 21 to 29 independent daily origins per lead
  band; both are now stated here, and the post hoc penalised arm above was added in response. Not
  acted on: the missing intercept in the fit (a free constant made D8 worse in the audit's test),
  and the whole-table cleaning pass that 377 shares.

## Caveats and what to try next

- The terms are stage 2 additive terms, not stage 1 GAM terms and not interactions with the weather
  inputs. The refutation covers that form only.
- D8 has under 12 months of history for the whole scored year. Its first scored month rests on one
  month of history, and its fits use 21 to 29 independent daily origins per lead band in August 2025
  (rows per band overstate this, because the newest-lag term takes one value per origin). This
  departs from the 12-month rule in the plan, as experiment 377's D8 own-residual arm does. The
  linear and penalised versions stay slightly positive at D8 through spring 2026, so more history
  alone may not turn the result.
- Rows before 2024-07 carry no shift in the spread pool of the early fold months, which slightly
  widens the treated spread there.
- Next: the lags as origin-dependent inputs to stage 1, or an interaction of the day-lag anomaly
  with the stack's weather terms, tested at the Primaries with a year of history first.

Reuses experiment 377's folder (`e00377s2_common`, `e00377s2_corrections`, `e00377s3_stack`,
`e00377_adapter`, `e00377_score`) unchanged, and `studies.eval_protocol`. Schedule: monthly stage 2
refits, weekly stage 1 fits as stored by 377; the contrast depends on the schedule only through the
stage 2 fits. Lag source: the cleaned power the GAM trains on, available at forecast time up to 3.5
hours before the origin.

## Plan, registered before any score was read

**Where the terms enter.** The stack's stage 1 GAM predicts from the valid time's weather and
calendar and does not depend on the forecast origin, so a lag measured from the origin cannot be a
column of its design matrix without re-fitting stage 1 on one row per origin and valid time. The
stage 2 correction already depends on the origin (it contains recipe R, a profile of recent
residuals), so the terms enter there: one additive shift of the stage 2 mean, then the same
fold-by-fold spread. This is a stage 2 smooth term, not a stage 1 GAM term, and the write-up says so
in its opening.

**The three lags and their anomaly form.** For a run with origin `o` and a row with valid time `v`,
the newest usable reading is the half-hour ending at `o - 3.5 h` (371's telemetry delay of 3 hours
plus the half-hour a reading spans), the day lag is the latest earlier reading with `v`'s half-hour
of day, and the week lag is the latest earlier reading with `v`'s weekday and half-hour, all no
later than the newest usable reading (the definition of `origin_lags` in experiment 87). Each lag
enters as an **anomaly**: the reading minus the stack's own corrected mean for that half-hour, taken
from the run with the latest origin not after that half-hour (always before `o`, so it is known at
`o`). The anomaly removes the daily and weekly shape that a raw lag would carry, which a lag-only
additive term cannot remove.

**No lookahead, and no leaky lags.** Every lag time is at least 3.5 hours before `o`, so no lag is
at or after the origin at any lead; the nullification of leaky lags that the feature pipeline does
for lags shorter than the lead cannot trigger, and the code asserts that every lag index is at most
the newest usable index. A missing reading gives no contribution (the term is zero), so absent data
degrades to "default".

**Term form.** For each lag and each of the six protocol lead bands, a penalised cubic B-spline (8
basis functions, second-difference penalty, weight fixed at 10 before any score is read) of the
anomaly, clipped to the 1st to 99th percentile of the fit rows; the shift is the sum of the three
terms, fitted by penalised least squares on the residual of the stack's mean, on rows whose origin
and valid time both precede the fold month's first instant (the 12-month rule), outside the
hand-filtered events. The spread for each fold month is fitted on earlier months' residuals of the
treated mean, each earlier month using its own out-of-fold shift. Fits start at cut 2024-07-01 (rows
before it get no shift). A band with under 2,000 fit rows gets no shift and is listed. Sensitivity
arm, pre-registered: the three terms linear (penalty weight infinite).

**Arms.** `default` (the 377 stack as stored, unchanged), `default_plus_origin_lags`, and
`default_plus_origin_lags_linear` (sensitivity only; at most one pre-registered arm can be adopted,
and it is the spline arm). BG2, climatology and the manual heuristic are the references.

**Rows and unit.** The 16 Primaries of experiment 377; D15 and D18 are report only, D8 follows 377's
two-arm rule. Compute: 3 threads.

**Tails and nulled inputs.** The write-up reports pinball loss and coverage at the 0.01, 0.05, 0.95,
and 0.99 quantiles per group and per Primary, beside CRPS and twCRPS, and names the tail result in
the take-home. Rows whose lags are missing form the nulled rows; the write-up gives interval width
and coverage on nulled and un-nulled rows, and tries a simple widening rule if coverage falls short.
