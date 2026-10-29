---
id: "00393"
slug: gam-mva-fold-data-analysis
theme: production-reference-and-best-guess
also_relevant_to:
  - hierarchy-and-pooling
status: planned
kind: other
protocol: demand-v2
size: >-
  simple. Stored: small result tables and figures in this experiment's private folder only; no
  forecast is written and no model is fitted. Designs: the contrasts are fixed by the maintainer's
  hypothesis and registered below before any score is read; the one choice that admits another
  design, how to fold a quantile forecast, is fixed below. Callers: the script imports experiment
  377's adapter and scorer and nothing imports it. Production serving path and degradation rule:
  not touched (this repository has neither).
hypothesis: >-
  The GAM stack trails BG2 at the eight MVA-only Primaries (D2, D3, D4, D8, D12, D19, D21, D22)
  because apparent power (MVA) is a folded, non-linear function of the additive effects that a GAM
  learns for signed active power (MW), so the stack loses relative skill whenever demand is near
  zero. Stop-rule count for the idea "the MVA fold explains the GAM's gap": 0 of 3.
prediction: >-
  Written before any score is read. P1. Across the 14 countable Primaries, the stack-minus-BG2 gap
  (CRPS, days 1 to 10) grows with the share of half-hours below 10% of the Primary's peak
  (Spearman correlation positive, bootstrap interval over Primaries excludes zero). P2. At the six
  signed-MW Primaries the stack loses relative skill when forecast and observation are both
  folded: the stack-minus-BG2 gap is larger on the folded target than on the signed target, and
  larger at Primaries with more near-zero half-hours. If neither P1 nor P2 holds, the data do not
  support the hypothesis.
decision_rule: >-
  Descriptive, no adoption. The hypothesis counts as supported only if P1 and P2 both hold; as not
  supported if both fail; otherwise as partly supported. P1 is read from the Spearman correlation
  over the 14 Primaries with a bootstrap interval that resamples Primaries (95%). P2 is read from
  the paired month-resampled bootstrap of `studies.eval_protocol` (99% interval, 20,000 resamples)
  for the pooled signed-MW group, and the per-Primary table. The threshold is zero, because the
  experiment changes nothing in production. Contrast 3 and Contrast 4 are descriptive.
benchmark: >-
  BG2, `manual_heuristic`, and `climatology` as stored by experiment 87, and the GAM stack as stored
  by experiment 377, all on `fold_a_daily_gap_v1`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles (the quantile score) and mean
  absolute error of the median, in points of effective capacity, from `studies.eval_protocol`.
code: ml-experiments/00393-gam-mva-fold-data-analysis/
---

# Does the GAM trail at the MVA-only Primaries because MVA is a fold of signed MW?

**Planned.** This page is the plan, committed before any score is read. It is replaced by the
write-up when the analysis has run.

## Plan

- **Reused forecasts (nothing is refitted).** The GAM stack of experiment 377 (`work/*/arms/a0/dp_preds_s2/stack_d_*.parquet`,
  D8 as its own-residual arm), and BG2, `manual_heuristic`, and `climatology` of experiment 87 (`predictions/p_s6_bg2`, `ref_manual_heuristic`, `ref_climatology`), at the 14 countable Primaries and D15 and D18
  (report only). Scored by `studies.eval_protocol.metrics` through experiment 377's adapter, on the
  shared rows of the four arms.
- **Contrast 1 (fold the signed series).** At D5, D9, D10, D11, D14, and D16, take each forecast's
  11 quantiles and the observation in MW, replace each by its absolute value, and re-sort the
  quantiles. This approximates the quantiles of |X|: it is exact when the whole forecast sits on
  one side of zero and understates the mass near zero otherwise, which is the stack's behaviour
  we want to test. Score signed and folded targets identically and report stack minus BG2 and stack
  minus `manual_heuristic` with intervals. No reactive-power series is stored for these six, so true
  MVA is not built.
- **Contrast 2 (near-zero share).** For every Primary, the share of half-hours with |actual| below
  10% of the Primary's peak (the maximum |actual| before 2025-07-01), and the stack's gap to BG2
  and to `manual_heuristic` inside and outside those half-hours.
- **Contrast 3 (forecast-error shape).** The stack's residual by level of the observation and of
  the forecast. A sign estimate for the MVA-only Primaries exists only in the folded-latent
  forecasts of experiment 387 (stored read-only); if the stored latent mean cannot be aligned with
  the stack rows the contrast reports residual by flow level only, and the page says so.
- **Contrast 4 (other explanations).** The gap by lead band (days 1 to 3 against 8 to 10), the
  per-Primary measures of experiment 384 (including embedded generation where stored), and the
  gaps at the bulk and grid supply points of experiment 386.
- **Scoring.** CRPS and MAE of the median, days 1 to 10 headline and days 11 to 14 descriptive,
  per Primary and per group; the `demand-v2` floor (CRPS no greater than `manual_heuristic`'s and
  no more than 1.10 times BG2's) at each Primary; coverage at q05, q95, and q99.
- **Compute.** One core, no GPU, `MemoryMax=16G`, no new fits or fetches. Retraining cadence: not
  applicable, because the stored stack forecasts are reused (monthly refits, per experiment 377).
- **Earlier advice followed.** Experiment 384's advice that the metering group is not a proxy for
  regularity, and its within-group relation of the gap to the step rate, are read as the
  alternative explanation in Contrast 4.
