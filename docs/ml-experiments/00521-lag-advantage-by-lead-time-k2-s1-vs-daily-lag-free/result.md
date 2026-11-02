---
id: "00521"
slug: lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free
title: >-
  At which lead times does the lagged-and-lag-free network K2-S1 beat the daily-refit lag-free
  network, and is the gap a lag effect, a refit-age effect, or a mixture-width effect?
theme: lagged-power-and-error-correction
also_relevant_to:
  - retraining-and-training-history
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: tables of scores, a figure folder, and the page only, under this experiment's
  private data folder and `docs/`; no forecast and no model weights are written. Serving path: not
  touched (this repository holds none). Degradation rule: not touched. Designs: one defensible
  (re-cut stored per-row scores by lead), with one open choice, how to score the monthly lag-free
  references, fixed below. Callers: the script reads experiment 511's stored row scores and
  experiment 454's stored forecasts, and imports experiments 454 and 470 for the scorer. One Sonnet
  code review before the first number is used, one Sonnet results review, and two Sonnet write-up
  reviews. CPU only, at most 2 cores, no fits.
hypothesis: >-
  Experiment 511 left the daily-refit lag-free network L-1d 0.091 points of CRPS behind K2-S1 at the
  eight MVA-only Primaries and 0.071 at the six signed-MW Primaries. The gap is not a lag advantage
  and so not concentrated at the shortest leads. L-1d beats K2-S1 below about 2 days, and the whole
  pooled gap sits at days 5 to 10 (stored finding, interim and unaudited, from a first look at
  experiment 511's row scores). The open question is whether the long-lead gap comes from the
  four-head mixture width of K2-S1 or from the monthly refit age of K2-S1's lag-free heads.
prediction: >-
  Written before the script runs, from the first look. P1: L-1d minus K2-S1 is negative (L-1d
  better) in every band below 48 h at both groups and positive at 72 to 240 h at both groups, with
  the 120 to 240 h band carrying more than 70% of the eight's pooled gap. P2: the monthly lag-free
  references (experiment 454's LL, LL-ens3, LL-ens5) score worse than L-1d at days 5 to 10 by at
  least as much as K2-S1 does if refit age drives the gap, and score between L-1d and K2-S1 if
  mixture width drives it. P3: K2-S1's q95 and q99 coverage at days 5 to 10 is closer to nominal
  than L-1d's. P4: D4 carries more than 40% of the eight's pooled gap. All four can fail.
decision_rule: >-
  The aim is a design decision, not a score. If the gap sits below 12 h (P1 fails), a short-horizon
  correction on a lag-free anchor is supported and gets its own brief. If the gap sits at days 5 to
  10 (P1 holds), a short-horizon correction is not supported, and the page says which of refit age
  and mixture width the references favour, or that they cannot tell. A band counts as a difference
  only when its Bonferroni interval (28 counted CRPS tests, 56 with q99) excludes zero under the
  origin-day blocks that experiment 511 used; the whole-month blocks (5, so near the
  minimum-to-maximum range of the five folds) are reported as the conservative check.
benchmark: >-
  L-1d (experiment 511, daily refit, two-seed quantile average) against K2-S1 (experiment 454's
  monthly refit schedule, three cells, each an equal-weight mixture of two S1 heads and two lag-free
  heads; the headline uses cell c0 alone, because that is what experiment 511's stored K2-S1 score
  is, and the mean of the three cells is the sensitivity). References by lead: L-Ww (weekly), and
  experiment 454's monthly LL, LL-ens3, and LL-ens5.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity by lead band (0 to 6 h, 6 to 12 h, 12 to 24 h, 24 to 48 h,
  48 to 72 h, 72 to 120 h, 120 to 240 h), q01, q05, q95, and q99 coverage and pinball loss by band.
---

*Planned; the finished write-up replaces this page. Every number the finished page quotes from the
first look is interim and unaudited until the script's output has been reviewed.*

## Plan

- **Rows.** Experiment 511's five folds (2025-10, 2025-12, 2026-01, 2026-03, 2026-06; July 2026 is
  not used), 1,145,256 scored rows from 153 origins. The eight are the MVA-only Primaries, the six
  are the signed-MW Primaries, and D15 and D18 are report-only. Experiment 511's scoring applied the
  1.15 widening; the references are widened the same way.
- **Lead profile.** Paired L-1d minus K2-S1 CRPS by lead band, per group, with a month-block
  bootstrap (5 blocks), per-fold signs, per-Primary shares, and the minimum detectable effect beside
  every contrast. Bonferroni count 28 (7 bands, 2 groups, 2 contrasts); 56 with q99.
- **Contrasts.** (1) L-1d minus K2-S1. (2) L-1d minus the monthly lag-free reference nearest in
  head count (LL-ens5 for K2-S1's four heads; LL and LL-ens3 as report-only references).
- **Calibration by lead.** Coverage and pinball loss at q01, q05, q95, and q99 for K2-S1 and L-1d
  in every band, because mixture width is a calibration effect.
- **Slices.** Rows with no labelled event (`tgt_real` false), cold winter nights, Christmas week,
  D15 and D18, and 454's `lag_real` and `win_real` flags joined from experiment 454's K2-S1 files.
  The caveat for every page: NGED's logs may miss events, so rows with no labelled event can hold
  unlogged events.
- **Not built.** A size-matched daily lag-free ensemble: L-1d has two seeds, so no four-head daily
  control exists without new fits.
- **Caveats stated on the page.** The L-1d versus K2-S1 gap mixes refit age (daily against
  monthly) with ensemble size (two against four lag-free heads) and with the averaging method
  (experiment 454 mixes the heads' distributions, experiment 511 averages the two seeds' quantiles).
  Only 153 origins, so rows are strongly correlated within a day. Experiment 454 does not say that
  K2-S1's gain is mostly ensembling; it says that mixing a lagged and a lag-free forecast beats
  averaging lagged networks, and it has no result by lead.
- **Standing practices.** Cleaned data (inherited from experiment 511's rows), D15 and D18 cut at
  2025-03-01 (inherited), no fits, nothing saved beyond score tables. Past-weather caveat: not
  applicable, no forecast is made.
- **Reviews.** One fresh Sonnet code review before any number is used, one Sonnet results review,
  and two Sonnet write-up reviews. No Opus.
