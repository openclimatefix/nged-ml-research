---
id: "00558"
slug: lags-in-pretraining-and-24-hour-cap-lag-light-demand-network
title: >-
  What does a 24-hour cap on the newest lag cost the lag-light demand network, and do lags in
  ERA5/CAMS pretraining recover it?
theme: lagged-power-and-error-correction
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: new models, forecasts, and scores in a private folder. More than one defensible
  design (where the cap applies, how dropout is drawn, whether pretraining sees lags). Spans code
  whose callers need a search (experiments 397, 418, 425, 510, 511, and 544). Serving path and
  degradation rule: not touched (this repository holds none). Cost: about 2 GPU-h.
hypothesis: >-
  A production recipe that uses no reading newer than 24 hours costs the lag-light network accuracy
  at short lead only, and putting the capped lags into ERA5/CAMS pretraining recovers part of the
  loss at 24-120 h.
prediction: >-
  Written before any fit. P1: the cap costs R-DW at least 0.05 CRPS at 0-24 h at the eight and at
  most 0.03 at 48-120 h; P1 fails if the cap costs over 0.05 at 48-120 h. P2: R-DW-24-PT beats
  R-DW-24 by at least 0.03 at 24-120 h at the eight; P2 fails if the gain is within 0.02. P3:
  A-pool-24 stays within 0.04 of R-DW-24-PT at 24-120 h. P4: the injected-shift leak of R-DW-24-PT
  is at most 6.0 points of capacity (L-1d 4.49).
decision_rule: >-
  Contrasts are paired by row. Intervals are 5-block month bootstraps (near the minimum to maximum
  of the folds), with per-fold signs and the minimum detectable effect beside each. Counted
  contrasts for the Bonferroni correction: P1 (2 bands), P2 (1 band), P3 (1 band), so 4 tests. The
  threshold for porting is zero for the cap itself (it is a production constraint, not an option)
  and 0.03 CRPS for pretraining lags, because it adds a pretraining step. Wording is "interval
  excludes zero" or "not shown".
---

# Plan: lags in pretraining and a 24-hour cap (interim, unaudited)

**Take-home message (plan only).** No number exists yet. The experiment measures what production's
"no reading newer than 24 hours" rule costs the lag-light network, and whether lags in the
ERA5/CAMS pretraining stage win some of it back.

## Step 0 answers (read from code)

- **A-pool pretraining includes lags.** `e00418_prep.pretrain_frame` fills the nine lag columns
  with `common.origin_lags(power=..., target=..., origin=...)` on the Primary's own ERA5/CAMS-era
  power history, with the same 3.5 h data-latency cut as fine-tuning rows. `e00425_prep.fold_pretrain`
  reads those stored columns and standardises them with the fold's standardiser. The lag-light
  networks of experiments 539 and 544 instead load lag-free pretrained weights (37 inputs) and
  start the six lag inputs at zero first-layer weights.
- **Lag value, flag, and age for pretraining rows.** Value is the Primary's own power over its
  effective capacity at the lag time. Flag is 1 when that value is not finite. Age is hours from
  the lag time to the target, divided by 336. Pretraining rows come from pseudo-runs whose origins
  step by `STRIDE_DAYS`, so lag times follow the same formula as for ENS rows.
- **Capped lag definition (new).** Day lag: the latest same-half-hour value at least 24 h before
  the origin (24 to 48 h old). Week lag: the latest same-weekday, same-half-hour value at least
  24 h before the origin. The newest lag is dropped. Flag and age as in 539; age is hours to the
  target over 336.

## Arms

1. R-DW (stored, experiment 539 scores).
2. R-DW-24: capped lags with block dropout, lags added at fine-tuning only.
3. R-DW-24-PT: as 2, with the capped lags also in ERA5/CAMS pretraining (new pretraining, 5 folds
   × 2 seeds).
4. A-pool-24: A-pool recipe with capped lags and block dropout, 2 seeds, with and without its row
   drops.
5. A-pool and L-1d (stored references).

Block dropout: per origin day, 30% of training days mask all lags (flag 1, value 0, age at maximum);
in 15% of those, only the day lag. Report the share of rows with each lag missing.

## Checks and reporting

- Row-by-row assertions: every lag time is at least 24 h before the origin (arms 2 to 4), every
  weather run is at or before the origin; one example row printed per lag per lead band.
- Standing rule: weather comes from the run issued at or before the origin.
- CRPS by seven lead bands plus 0-6 h and 6-12 h; the eight and the six; D15 and D18 report-only;
  rows with no labelled event; cold nights; Christmas week; injected-shift leak; lags removed and
  aged by 1, 2, and 3 extra days (inference only). Metered generators anonymised.
- Review: one fresh Sonnet code review before any number is used; no XGBoost; July 2026 unused.

## Open questions for the manager

- Experiment 539's fit script is not on `main` (experiment 544's `e00544_fit.py` is a copy with
  R-DROP added). The plan builds on 544's harness unless told otherwise.
- Cost: lag-aware pretraining for 10 fold-seed pairs may exceed 2 GPU-h; the first pretrain is
  timed before the rest start, and the manager is asked before going over.
