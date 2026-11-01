---
id: "00483"
slug: uncertainty-rerun-on-the-first-release-demand-candidates
title: >-
  Does a widening or table P make the tails of the first-release demand candidates honest?
theme: calibration-and-uncertainty
also_relevant_to: []
status: planned
kind: other
date: 2026-11-01
size: >-
  medium. Stored: tables and figures under the experiment's data folder only; no model is fitted.
  Serving path: not touched. Degradation rule: not touched. Designs: one, because the stages and
  the pass rule come from the accepted brief. Callers: this folder's scripts, which import
  experiments 435 and 470. One Sonnet plan review, one Sonnet code review before the first run, one
  Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  Experiment 435 ended refuted for table P on clean rows. The prior is that a widening, not table
  P, repairs the tails of A-pool (experiment 425's arm A, five seeds averaged) and of A-gam-in (the
  eight MVA-only Primaries). A widening from 1.0 to 1.15 brings clean-row coverage into the target
  ranges on held-out months, and table P adds nothing beyond it.
prediction: >-
  Written before any score is read. P1: at the eight, A-pool clean-row q95 coverage is below 0.93
  raw and in 0.93 to 0.97 at widening 1.10 or 1.15. P2: table P (leave-one-fold-out) does not beat
  the best widening on q95 coverage at the eight. P3: cold winter nights stay below target after
  every recalibration. P4: K2-425A's q99 pinball loss (+0.021 against A-pool in experiment 470) is
  repaired by a pool-specific widening, not by table P. P5: D15 and D18 are under-covered at q95
  and stay so.
decision_rule: >-
  A recalibration is adopted if, on held-out folds, it brings clean-row q95 coverage to 0.93 to 0.97
  and q99 to 0.98 to 0.995, keeps q05 in 0.03 to 0.07 and q01 in 0.005 to 0.02, and costs at most
  0.03 CRPS points and 5% q99 pinball loss against the 1.15 widening. Judged at the eight (A-pool
  and A-gam-in) and at the signed-MW six (A-pool). D15, D18, cold nights, and Christmas week are
  reported, never judged. Where cold winter nights stay below target, the gap is reported and no
  fix is added.
benchmark: >-
  The stored forecasts raw and with the 1.15 widening, as in experiment 435.
row_set_id: >-
  The `gam14` row set of experiment 470.
metric: >-
  Coverage and pinball loss at q01, q05, q95, and q99, and CRPS in points of effective capacity,
  per metering group, lead band, and condition.
---

# Experiment 483: does a widening or table P make the tails of the first-release demand candidates honest?

**Planned; written before any score is read.** The page will lead with the take-home message when
the experiment finishes.

## Question

How well calibrated are the bands of A-pool and of A-gam-in (the eight only), and which
recalibration makes the tails honest: table P, a widening, or a per-condition factor? Experiment
470 left two candidates and the maintainer chooses between them, so this experiment supplies the
calibration evidence. K2-425A (the eight only) is included for the q99 repair question.

## What is reused and what is new

- **Reused, not rebuilt:** `e00435_table_p.py`, `e00435_cold_nights.py`, `e00435_nra_actuals.py`,
  `e00470_tails.py`, and `e00470_score.py`. Experiment 470's `tails_summary.csv` already reports
  raw, widened, and table P coverage per group.
- **New:** table P fitted leave-one-fold-out; tails by month; the worst 20 half-hours by q99
  pinball loss with context charts; a cold-night class and a D15/D18 class; the K2-425A q99 test;
  the minimum detectable effect and Bonferroni count; a curtailment-cost row at the signed-MW six.

## Design

- **Rows.** The `gam14` set of experiment 470. "Rows with no labelled event" is `cls == "clean"` in
  `rows_*.parquet`. Flagged exact-zero half-hours are dropped, and the experiment 399 start cut is
  applied to D15 and D18. July 2026 is not used.
- **Actuals.** The stored forecasts carry neither actuals nor labels; both are rejoined on
  `primary, origin, target` as in experiments 435 and 470, with NRA-corrected actuals as in
  `e00435_nra_actuals.py`.
- **Order.** The free parts run first: tails by month, worst-20 charts, the widening sweep (1.0,
  1.05, 1.10, 1.15, for A-pool and A-gam-in only), and the K2-425A q99 test. Table P with the
  cold-night and D15/D18 classes runs last, because the sweep alone may decide the question.
- **Table P training.** The five folds (2025-10, 2025-12, 2026-01, 2026-03, 2026-06) are not
  contiguous, so each scored fold is fitted on the other four, leave-one-fold-out. Folds after the
  scored fold are therefore used for fitting; this is stated in the write-up and is a limit on how
  strictly "fitted on earlier months" holds. 2025-10 is scored this way too.
- **Scoring.** Coverage and pinball loss at q01, q05, q95, and q99 per group, lead band
  (including 12 to 36 hours), and condition (no labelled event, real-event rows against NRA
  actuals and metered actuals, cold winter nights, Christmas week, D15/D18, nulled-input rows
  where stored). Month-block bootstrap intervals. The Bonferroni count and the q99 minimum
  detectable effect are stated as numbers. A curtailment-cost row at the signed-MW six uses
  `metrics.avoidable_curtailment_cost` at f = 0.9, 0.75, and 0.5, for raw, 1.15, and table P.

## Standing practices

- **Cadence and models:** no model is fitted. The stored forecasts are monthly-refit.
- **Compute:** CPU only, 4 cores, `OMP_NUM_THREADS=1`, a spawn pool over (candidate, month) cells,
  `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`, about 1 to 1.5 hours.
- **Caveats:** every score is selection-biased, because the same five folds choose and score the
  candidates. NGED's event labels are incomplete, so rows with no labelled event may hold
  unlabelled events. The six and the eight are reported separately with no pooled constant.
- **Reviews:** Sonnet plan review, Sonnet code review before the first run, Sonnet results review,
  two Sonnet write-up reviews; results are recorded in the PR body.
- **Advice followed:** the plan follows experiment 435's advice to keep the 1.15 widening as the
  default until a recalibration passes its rule.
