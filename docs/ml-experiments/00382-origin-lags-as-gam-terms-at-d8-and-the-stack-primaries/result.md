---
id: "00382"
slug: origin-lags-as-gam-terms-at-d8-and-the-stack-primaries
title: >-
  Do three recent-power lags, added to the GAM stack as smooth terms, close the gap to monthly
  XGBoost at substation D8?
theme: lagged-power-and-error-correction
also_relevant_to:
  - production-reference-and-best-guess
status: planned
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
---

# Design (registered before any score is read)

**Where the terms enter.** The stack's stage 1 GAM predicts from the valid time's weather and
calendar and does not depend on the forecast origin, so a lag measured from the origin cannot be a
column of its design matrix without re-fitting stage 1 on one row per origin and valid time. The
stage 2 correction already depends on the origin (it contains recipe R, a profile of recent
residuals), so the terms enter there: one additive shift of the stage 2 mean, then the same
fold-by-fold spread. This is a stage 2 smooth term, not a stage 1 GAM term, and the write-up says so
in its opening.

**The three lags and their anomaly form.** For a run with origin `o` and a row with valid time `v`,
the newest usable reading is the half-hour ending at `o - 3.5 h` (371's telemetry delay of 3 hours
plus the half-hour a reading spans), the day lag is the latest earlier reading with `v`'s half-hour of
day, and the week lag is the latest earlier reading with `v`'s weekday and half-hour, all no later
than the newest usable reading (the definition of `origin_lags` in experiment 87). Each lag enters
as an **anomaly**: the reading minus the stack's own corrected mean for that half-hour, taken from
the run with the latest origin not after that half-hour (always before `o`, so it is known at `o`).
The anomaly removes the daily and weekly shape that a raw lag would carry, which a lag-only additive
term cannot remove.

**No lookahead, and no leaky lags.** Every lag time is at least 3.5 hours before `o`, so no lag is
at or after the origin at any lead; the nullification of leaky lags that the feature pipeline does
for lags shorter than the lead cannot trigger, and the code asserts that every lag index is at most
the newest usable index. A missing reading gives no contribution (the term is zero), so absent data
degrades to "default".

**Term form.** For each lag and each of the six protocol lead bands, a penalised cubic B-spline
(8 basis functions, second-difference penalty, weight fixed at 10 before any score is read) of the
anomaly, clipped to the 1st to 99th percentile of the fit rows; the shift is the sum of the three
terms, fitted by penalised least squares on the residual of the stack's mean, on rows whose origin
and valid time both precede the fold month's first instant (the 12-month rule), outside the
hand-filtered events. The spread for each fold month is fitted on earlier months' residuals of the
treated mean, each earlier month using its own out-of-fold shift. Fits start at cut 2024-07-01 (rows
before it get no shift). A band with under 2,000 fit rows gets no shift and is listed.
Sensitivity arm, pre-registered: the three terms linear (penalty weight infinite).

**Arms.** `default` (the 377 stack as stored, unchanged), `default_plus_origin_lags`, and
`default_plus_origin_lags_linear` (sensitivity only; at most one pre-registered arm can be adopted,
and it is the spline arm). BG2, climatology and the manual heuristic are the references.

**Rows and unit.** The 16 Primaries of experiment 377; D15 and D18 are report only, D8 follows
377's two-arm rule. Compute: 3 threads.

**Tails and nulled inputs.** The write-up reports pinball loss and coverage at the 0.01, 0.05, 0.95,
and 0.99 quantiles per group and per Primary, beside CRPS and twCRPS, and names the tail result in
the take-home. Rows whose lags are missing form the nulled rows; the write-up gives interval width
and coverage on nulled and un-nulled rows, and tries a simple widening rule if coverage falls short.
