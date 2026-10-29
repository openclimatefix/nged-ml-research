---
id: "00389"
slug: xgboost-tail-calibration-and-ensemble-spread-columns
title: >-
  Can per-Primary tail calibration, or ensemble spread columns, fix the tails of the simple weekly
  XGBoost demand model at all 16 Primaries?
theme: calibration-and-uncertainty
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: change
protocol: demand-v1
size: >-
  medium. Stored: forecasts and small result files in this experiment's private folder only;
  experiments 87 and 379 are read and never written. Designs: more than one defensible (global
  factor, per-Primary factor, per-Primary quantile offset, extra input columns), so all are
  registered as arms. Callers: the scripts copy experiment 87's fitter and experiment 379's adapter
  and scorer, whose callers all sit in those folders. Serving path and degradation rule: not touched
  (this repository holds neither).
hypothesis: >-
  The weekly XGBoost member's tails (the 1%, 5%, 95% and 99% quantiles) are miscalibrated in a way
  that differs by Primary and by lead, so a correction set per Primary and per lead band from the
  member's own trailing-year misses, or a pair of input columns that tell the model how much the 51
  ensemble weather members disagree, brings the tail coverage closer to nominal than the single
  global widening factor of 1.15 does, without a loss of CRPS.
prediction: >-
  Written before any score is read. P1. At the signed-MW six, the baseline's q95 coverage is about
  0.92 and its q99 coverage about 0.97 (experiment 379). T2 and T3 move q95 to within 0.02 of 0.95
  and q99 to within 0.01 of 0.99 pooled over those six, and move CRPS by less than 0.02 points.
  P2. At the MVA-only eight, q99 coverage is 0.995 in the baseline (too wide); T2 and T3 bring it
  to within 0.005 of 0.99 and lower pinball loss at q99. P3. T1 (factor 1.0) gives q95 coverage
  about 0.03 below the baseline's at both groups, because experiment 18 found 1.15 lifts coverage
  by about 4 points. P4. S1 changes the CRPS by less than 0.03 points and the tail coverage by less
  than 0.01, because experiment 169 found the ensemble spread added nothing to cold-night
  calibration and experiment 371 found the ensemble mean added 0.007. P5. Per-Primary, per-band
  factors are noisy at the Primaries with under 12 months of history (D8, D15, D18) and do not beat
  the donor spread there.
decision_rule: >-
  Contrasts, all registered before any score is read, each arm minus the baseline arm (the stored
  weekly member with factor 1.15), once per group (signed-MW six; MVA-only eight with and without D8;
  D15 and D18 report-only), days 1 to 10 headline, lead bands beside it, by the paired
  month-resampled bootstrap of `studies.eval_protocol` (99% interval, 20,000 resamples pooled; 95%
  interval, 2,000 resamples per Primary). C1: T1 minus baseline. C2: T2 minus baseline. C3: T3
  minus baseline. C4: S1 minus baseline. C5: S2 minus the better of T2 and T3, then S2 minus S1.
  Success in tail terms, pre-registered: an arm succeeds if its absolute coverage error
  |coverage - nominal| at q95 and q99 (signed-MW six pooled) and at q99 (MVA-only eight pooled)
  is smaller than the baseline's at all three cells (signed-MW six q95 and q99, MVA-only eight q99;
  AND, not OR), with the 99% pooled interval of the coverage-error difference (month resampling)
  excluding zero in that direction, and its CRPS (days 1 to 10) is no worse than the baseline's by
  more than a fixed 0.02 points in each group. The lower tails (q01, q05) are scored and reported
  but are not part of success. No threshold by porting cost for T1, which
  removes a constant. T2 and T3 add a trailing-coverage table refreshed weekly, a pipeline step the
  size of the existing spread factor, so each needs a coverage-error reduction of at least 30% at
  the target quantiles. S1 adds input columns built from the ensemble archive at serving time,
  a new data dependency, so it must meet the same tail rule and also show a CRPS gain whose 99%
  interval excludes zero. P1 to P5 are expectations, not tests: the decision rule alone decides
  success. C5 is evaluated only if S2 runs.
benchmark: >-
  The baseline arm (stored weekly member of experiment 87 with spread factor 1.15, the `only_W`
  arm of experiment 379), BG2 (stored), default 3 of the GAM stack (experiment 378), `manual_heuristic`,
  and `climatology`, all on the same rows.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, mean absolute error of the median,
  coverage and pinball loss at q01, q05, q95, q99, and twCRPS, in points of effective capacity, from
  `studies.eval_protocol`; leads 0.5 to 336 hours; the headline band is days 1 to 10.
code: ml-experiments/00389-xgboost-tail-calibration-and-ensemble-spread-columns/
---

**Plan only: no score has been read.** This page is the pre-registration of experiment 389.

We ask whether the tails of the simple weekly-refitted XGBoost demand model can be improved with two
small changes at the 16 Primary substations: a correction set separately for each Primary and each
lead band from the model's own recent misses, and two extra input columns that carry how much the 51
ensemble weather members disagree. The weekly XGBoost member of
[BG2](../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md) is the baseline. Its
upper tail is too narrow at the six Primaries whose meters record the sign of the flow, and its q99
is too wide at the eight whose meters do not
([experiment 379](../00379-bg2-components-all-16-primaries-to-14-days/result.md)).

## Facts this plan relies on

- The baseline trains `reg:quantileerror` at 11 levels on the control member, applies the one model
  to all 51 members, pools them as an equal-weight mixture, and widens about the median by 1.15
  ([experiment 87](../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md); recipe in
  `e00087_common.py` lines 101 and 167 to 174).
- Experiment 379 reports baseline q01/q05/q95/q99 coverage of 0.010/0.041/0.923/0.973 at the
  signed-MW six and 0.005/0.043/0.954/0.995 at the MVA-only eight.
- [Experiment 97](../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md) tried eight
  recalibrations of BG2 pooled by class over 9 months and none beat the interval; none was per
  Primary or per lead band on the XGBoost member alone.
- [Experiment 169](../00169-calibration-combined-fixes-and-ensemble-spread/result.md) found the
  ensemble spread added nothing to cold-night coverage of the GAM-based forecast. This plan departs
  from that advice only in testing the spread as an XGBoost input, which no experiment has done.
- Experiment 193 found a two-number solar summary (mean, standard deviation) kept 98% of the gain of
  all 51 members in the GAM stack.

## Arms

All arms use the weekly refit regime of experiment 87 (the `n1h` configuration, half-life 0.5
years, 500 rounds, eta 0.05, depth 6), `colsample_bytree` and `colsample_bynode` fixed at 1.0
(unchanged from the recipe, so no arm is given extra columns to sample from), `subsample` 0.8, the
fitting seed of experiment 87, the offset twins at the five offset Primaries (S1 follows the same
base-margin path as the baseline there), and the same rows.

- **Baseline.** The stored raw weekly member with factor 1.15 (experiment 379's `only_W`). Gate:
  the S1 fitting code, run with its spread columns switched off, refits the baseline at D5, D8,
  and D12 for the first four refits and must reproduce the stored raw quantiles to a mean absolute
  gap under 0.001 points of capacity and a CRPS gap under 0.005 points. The tolerance of experiment
  383's gate could not be found (its page is not on `main`), so this plan sets its own; the first
  run reports the gap.
- **T1, no widening.** The stored raw member with factor 1.0. No fit.
- **T2, per-Primary, per-lead-band factors.** For each Primary and each of the four lead bands of
  experiment 364 (0 to 24 h, 24 to 72 h, 72 to 168 h, 168 to 240 h), one upper and one lower factor
  multiplying the distance of each quantile from the median. Before each weekly refit, the factor
  is the smallest value in a grid (0.8 to 1.6 in steps of 0.05) at which the trailing year of the
  raw member's forecasts reaches nominal coverage. Only forecast rows whose valid time is at most
  3.5 hours before the refit enter (the observation is then known, as in `known_rows` of
  `e00087_fit_demand.py`); a forecast origin's 14-day horizon would otherwise reach past the refit.
  The scored runs start on 2025-07-01, so the window starts empty and grows. A Primary-band needs
  at least 60 days of such rows (about 60 origins); until then, and wherever the window is shorter,
  the factor is the global 1.15 of the baseline. The window is the trailing year of rows older
  than the refit, using the stored runs of 2025-07-01 onward only (earlier runs of the member are
  not stored). Coverage is checked at at q95 and q99 (upper) and q05 and q01
  (lower). The grid and the 14-day gap are fixed here. D8, D15, and D18 have under 12 months of
  history: D8 uses the donor spread (the mean of the factors of the MVA-only Primaries other than
  D8), and D15 and D18 are report-only with the same donor. The arm needs no refit, because
  recalibrating does not change the fitted model.
- **T3, per-Primary, per-lead-band quantile offsets.**
  The same trailing window, minimum of 60 days, bands, and
  fallback (no shift). Each of
  q01, q05, q95, q99 is shifted by the empirical quantile of the raw member's trailing misses
  (observed minus the member's quantile), in units of capacity, then the 11 quantiles are sorted to
  stay monotone. The same donor rule at D8, D15, and D18. No refit.
- **S1, ensemble spread columns.** The weekly member refitted with four extra columns: the mean and
  the standard deviation across the 51 members of `temperature_2m` and of `shortwave`, at the
  series' location and valid time. Training rows are the control member's rows, as in the baseline,
  with the spread columns taken from the same run's 51 members. Prediction rows carry the same
  spread columns on each of the 51 member rows (the column is a property of the run, not of the
  member), so each member's row differs only in its own weather. The 51-member archive for the
  training runs (2024-04-01 onward) comes from the stored ensemble stores of experiments 111, 152,
  374, and 377, which hold 51-member temperature and shortwave for 821 origins from 2024-04-01 to 2026-06-30:
  D5, D9, D10, D11, D14, D16, and D18 in experiment 111, D2, D3, D4, D8, D12, D19, D21, and D22 in
  experiments 152 and 374, and D15 in `00377*/work/d15/ens_store/`. The store origins are stamped
  at the run's initialisation time plus the 9-hour publication delay, so the join key is
  `power_fcst_init_time`, not `nwp_init_time`; the first job asserts the keys match the control rows
  and that the stores cover all 16 Primaries. Training rows older than 2024-04-01 have no spread
  columns; S1 drops them, and the baseline gate and a second control arm (the baseline refit on the
  same shortened history, columns off, `S0`) show how much of any S1 change comes from the history
  rather than the columns. A train/serve shift is accepted and reported: the control member is one
  of the 51, so its deviation from the ensemble mean is small in training and larger for the other
  members at prediction. A window of 0 (the value at the valid time itself) is used,
  not experiment 169's 24-half-hour window, because the control member's own columns are also
  pointwise. Cost to port: a new data dependency at serving time (the 51-member forecast at the
  same grid points, already read to run the 51 members).
- **Second baseline, the weekly bundle (B).** The release candidate of
  [experiment 381](../00381-simple-weekly-xgboost-demand-against-gam-stack/result.md): the plain weekly
  member plus a daily solar summary column, a sunny flag, and a 1-year half-life, read from 381's
  stored weekly forecasts (`arm_preds/weekly/bundle_solar+sunny+hl1`, read only). The first job
  checks whether those stored quantiles carry the 1.15 factor, by reproducing 381's printed bundle
  scores (D5 3.827, D12 6.716, D8 7.615) with and without it, and builds B-T1, B-T2, and B-T3 from
  the raw quantiles exactly as T1, T2, and T3 are built for the plain member. The success rule and
  contrasts apply to B-T1 to B-T3 against B with factor 1.15. The bundle already carries a solar
  summary column, so some of the gain from ensemble spread columns may be taken already; the
  write-up says so. S1 is fitted on the plain member first. **S1b**, the bundle refitted with the
  four spread columns, runs only if S1 meets the success rule, to avoid a second full refit.
- **S0, history control.** The baseline refit on S1's shortened history with the spread columns off.
- **S2, best of T2 and T3 on S1.** Run only if S1 and the better of T2 and T3 each succeed alone,
  with the calibration table fitted on S1's own trailing forecasts. Otherwise reported as not run.
- Not included: a second model, a stacked stage, or a blend.

## Scoring

All 16 Primaries to 14 days under [demand-v1](../evaluation-protocol-demand.md), through the scorer
of experiment 379, adding for every arm and every Primary: CRPS, coverage and pinball loss at q01,
q05, q95, q99, and twCRPS. Reported by lead band beside days 1 to 10, for the worst Primary, with
leave-one-out per group, on cold nights and sunny half-hours, and with D2 and D4 as placebos for any
change aimed at D8 or at one group. References: BG2, default 3 of the GAM stack, `manual_heuristic`,
`climatology`. The first fit and the gate print before any other arm is read.

## Compute

T1, T2, and T3 are post-hoc on stored forecasts and need no fit. S1 refits 16 series over 52
weekly refits (the weekly regime of experiment 87; that experiment's full `n1h` run, which also did
the monthly regime, BSP, and GSP, took 53 GPU hours). Slot: 3 threads, `MemoryMax=16G`, one GPU
fit at a time, checked with `uptime` and `nvidia-smi` in the first minutes. Folders of
experiments 379 and 381 are read only.
