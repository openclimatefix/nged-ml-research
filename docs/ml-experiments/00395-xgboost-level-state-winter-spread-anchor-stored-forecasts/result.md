---
id: "00395"
slug: xgboost-level-state-winter-spread-anchor-stored-forecasts
theme: production-reference-and-best-guess
also_relevant_to:
  - lagged-power-and-error-correction
  - calibration-and-uncertainty
  - weather-products-and-blending
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v1
size: >-
  medium. Stored: re-scored tables and figures in this experiment's private folder only; earlier
  experiments' folders are read and never written. Designs: more than one defensible (four level
  filters, each a named arm). Callers: the scripts import experiments 377, 381, 390 and
  `studies.eval_protocol`. Production serving path and degradation rule: not touched. Nothing is
  refitted, so the monthly-refit rule does not apply.
hypothesis: >-
  A level state in the style of the GAM stack's Kalman filter, run on the weekly XGBoost bundle's
  own residuals, closes the bundle's gap to the GAM stack at D11 and D14, and a one-state filter
  with three numbers per Primary does so almost as well as the stack's two-state filter.
prediction: >-
  Written before any score is read. P1 (A). K1, the stack's filter on the bundle's residuals,
  closes at least half of the bundle-minus-stack CRPS gap at days 1 to 10 at both D11 and D14. P2
  (A). K2, the one-state filter with three numbers per Primary, closes at least half of the gap at
  both Primaries, and K3 (pooled) is within 0.03 points of K2 on the signed-MW six. P3 (A). K4
  recovers the stack's filtered score within 0.03 points at the signed-MW six. P4 (A). The 0 to 24
  hour band worsens by under 0.10 points at the signed-MW six, because the level filter has no fast
  transient. P5 (B). The winter and top-quartile slices carry nearly all of the Economy 7 and
  spread-column differences, and the pooled differences stay inside refit noise. P6 (C). The
  coverage gain of S1M over the monthly member rises with temperature spread and with regime
  uncertainty. P7 (D). A 14-day trailing level anchor reduces D15's signed median bias by at least
  half, and does not harm the other 14 Primaries beyond their interval.
decision_rule: >-
  Descriptive re-scoring of stored forecasts under `demand-v1`: nothing is refitted, so no
  multiple-comparison correction applies and no claim of adoption is made. Part A, porting threshold
  (the cost is one scalar state per Primary and three numbers, updated as readings arrive): K2 is
  worth porting if it closes at least half of the gap at both D11 and D14, with no loss beyond its
  95% month-resampled interval at D2, D4 (placebos), the signed-MW six, or the other Primaries
  taken together. Parts B, C and D are descriptive; a slice with under 5 events or under 3 months is
  descriptive only.
benchmark: >-
  The GAM stack of experiment 377 (stack without S below 168 h), the weekly bundle of experiment
  381, BG2, `manual_heuristic`, and `climatology`, on `fold_a_daily_gap_v1`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles, median absolute error,
  pinball loss and coverage at q01, q05, q95, and q99, in points of effective capacity, from
  `studies.eval_protocol`; leads 0.5 to 336 hours; headline band days 1 to 10.
code: ml-experiments/00395-xgboost-level-state-winter-spread-anchor-stored-forecasts/
---

# Plan: a level state, a winter re-score, spread bins and a short-history anchor, all on stored forecasts

**This experiment fits no XGBoost model.** It re-scores forecasts that earlier experiments stored,
in four parts. Part A adds a Kalman-filter-style estimate of the recent level of demand to the
weekly XGBoost bundle. Part B scores the stored Economy 7 and ensemble-spread arms on winter, the
highest-demand quarter and cold nights. Part C asks whether ensemble spread helps calibration when
weather uncertainty is high. Part D tests a trailing level anchor at the two Primaries with under a
year of history, D15 and D18.

"The weekly bundle" is [experiment
381](../00381-simple-weekly-xgboost-demand-against-gam-stack/result.md)'s XGBoost model (solar
summary, sunny flag, 1-year half-life). "The GAM stack" is [experiment
377](../00377-demand-v1-first-run-feasibility/result.md)'s stack without S below 168 h. The filter
follows [experiment 136](../00136-dp-bg-stage1-assembly-on-the-ensemble-mean/result.md)'s two-state filter,
which in turn follows the variational noise filter of [de Vilmarest and Wintenberger
(2022)](https://arxiv.org/abs/2104.10777). [Experiment
390](../00390-xgboost-d11-d14-gap-and-load-shift-event-lags/result.md) found that a rolling median
offset does not copy the filter.

## Part A: arms and threshold

All arms re-score stored forecasts. Every arm is scored raw and at the 1.15 factor about the median,
because the stored weekly bundle is raw and the stack carries its own spread.

- **K0, reference.** The stored weekly bundle, raw and at 1.15.
- **K1, the stack's own filter.** Experiment 377's two-state filter (level plus AR(1) transient, one
  filter per hour band) with the pooled setting that `e00377_run.noise_hyper` returns, run on the
  bundle's residuals (actual minus median, in points of effective capacity). The state after the
  newest reading at least 3.5 hours before the origin is added to every quantile, as
  `dp_assimilation.correction_at` does for the stack.
- **K2, a one-state local-level filter per Primary.** A random-walk level observed with noise, run
  on all half-hours as one stream, and a decay of the predicted level towards zero over the gap
  between the newest reading and the target. Three numbers per Primary: the ratio of process to
  observation noise (reported as the steady-state gain), and the per-half-hour decay `phi`. The
  gain and `phi` are fitted on the bundle's residuals before each month by a grid search on the
  median absolute error of the corrected median (expanding window, July 2025 uses the fixed prior
  gain 0.05, `phi` 0.998).
- **K3, K2 pooled.** One gain and one `phi` for all 16 Primaries, fitted the same way.
- **K4, the stack without its filter, plus K2.** Experiment 380's `drop_k` arm (six Primaries) with
  K2 applied. The check is whether K2 recovers the stack's filtered score.

Production cost: K2 is a scalar per Primary updated as each reading arrives, plus three numbers; K1
is four hour-band filters of two states and four numbers.

## Part B: winter re-score (W0)

Score experiment 381's `monthly/e7`, experiment 389's S1M, and the other stored monthly arms against
`monthly/control` on winter (December to February, London calendar months), the top quarter of each
Primary's own actuals, and cold nights, with the share of the D11 and D14 gap to the GAM stack that
each closes. No fit.

## Part C: spread bins

[Experiment 394](../00394-xgboost-ensemble-mean-and-std-columns-separately/result.md) covers the
spread terciles and the sunny summer half-hours, so this part adds only the top decile of
temperature spread, the regime-uncertainty bins (the share of the 51 ensemble members below 5 °C at
the week's coldest day), and the interval width against absolute error, for the monthly member and
S1M. The scored year has three June to August months, so summer intervals are thin and a day-block
interval is given beside the month interval.

## Part D: short-history level anchor

The causal offset equal to the median of the bundle's own residuals over the previous 7, 14 and 28
days (readings at least 3.5 hours before the origin), shrunk towards zero by the days of history
available, applied to the stored bundle at D15 and D18 (raw and at 1.15) and, as a no-harm check,
at the other 14 Primaries.

## Scoring

All 16 Primaries to 14 days by lead band (0 to 24, 24 to 72, 72 to 168, 168 to 240 hours); the
days 1 to 10 headline; the D11 and D14 gap closed, with the winter and top-quartile splits of
experiment 390; per Primary and per group; coverage at q01, q05, q95 and q99; cold-night q95
coverage; curtailment cost; and load-shift event rows (rows with an event in the lags, rows in an
event, the 7 days after). D2 and D4 are placebos. Intervals resample whole months.
