---
id: "00481"
slug: seed-averaging-diminishing-returns-for-demand-and-wind
title: >-
  How many fit seeds does the first-release neural forecaster need before averaging more seeds stops
  helping, for Primary demand and for wind farms?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
  - lagged-power-and-error-correction
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: per-seed quantile tables and fitted networks in this experiment's private folder
  only (new seeds of experiments 425, 429, and 454 are written here, never into their folders).
  Designs: the number of seeds to serve is a decision, but the curve over K is the same test
  whichever K is chosen. Callers: the fit script wraps experiment 425's `e00425_fit` and
  experiment 429's `e00429_run` and 454's `e00454_fit`. Production serving path: not touched (this
  repository holds none). Degradation rule: not touched.
hypothesis: >-
  Averaging the quantile forecasts of K networks that differ only in fit seed lowers the score with
  diminishing returns. For Primary demand (experiment 425's arm A) and for the wind farms
  (experiment 429's `member_plain`), the gain from K = 5 to K = 10 is inside the minimum
  detectable effect, and for the lagged S1 network of experiment 454 the gain from K = 3 to K = 6
  is.
prediction: >-
  Written before any new fit and before any score is read. P1. Arm A at the signed-MW six: K = 2
  gains 0.05 to 0.12 points of effective capacity over K = 1, K = 5 over K = 3 gains less than
  0.03, and K = 10 over K = 5 gains less than 0.02. P2. The best tail widening falls with K, from
  about 1.15 at K = 1 towards 1.05 to 1.10 at K = 10. P3. Wind `member_plain` gains less than 0.10
  points from K = 5 to K = 10, so five networks suffice there. P4. Averaging S1 seeds beats the
  fixed lagged-plus-lagless mixture B-mix at equal network counts (experiment 460 found a 3-seed S1
  ensemble 0.21 better than B-mix at 12 to 36 hours). Each prediction could fail: a steeper curve
  past K = 5 would mean the first release should serve more networks.
decision_rule: >-
  Score per K is the mean over 100 random draws of K seeds from the pool (all draws if fewer than
  100 exist), so the curve has a spread. Marginal gain is the score at K minus the score at the
  next K, with a month-block bootstrap 95% interval at a Bonferroni level over the stated number
  of marginal contrasts (the scorer prints the count: arm A 4, S1 3, wind 4, so 11 contrasts and
  alpha 0.0045). The minimum detectable effect is printed first: the standard deviation over folds
  of the difference between the mean of seeds 0 to 2 and the mean of seeds 3 and 4 of the same
  arm, divided by the square root of two (experiment 425's definition). The K at which the
  marginal gain is inside the minimum detectable effect and stays inside for every larger K is
  marked. Recommendation per family: the smallest K whose score is within the minimum detectable
  effect of the largest K (10 for arm A and wind, 6 for S1) and whose q99 pinball loss is within 5%
  of the largest K's. If K = 1 is within the minimum detectable effect of K = 10 for wind, one
  network suffices there. Compare averaging with a mixture at 2 and 4 networks served using
  experiment 454's K2-S1, K3-S1, and B-mix as stored. The smallest gain worth porting is 0.03
  points at the six, because each extra seed adds one network per serving pass and one fit per
  monthly refit.
benchmark: >-
  K = 1 of the same arm (the stored seed 0), K = 10 (or 6) as the ceiling, and for demand
  experiment 454's B-mix, K2-S1, and K3-S1 at their network counts.
row_set_id: >-
  Demand: the 425 loop (12 monthly refits, pretrained per fold, fold_a_daily_gap_v1 rows) read at
  five months, 2025-10, 2025-12, 2026-01, 2026-03, and 2026-06 (all in the stored set), the same
  five as experiments 449 and 454. S1 uses experiment 404's rows for those months. Wind: experiment
  429's fold_a_tuesdays_v1 kept rows.
metric: >-
  CRPS from the 11 quantiles in points of effective capacity for demand (signed-MW six and MVA-only
  eight reported separately, never pooled; the six also on rows with no labelled event, cold
  nights, and the 12 to 36 hour band); quantile score in points of capacity for wind; q01, q05, q95,
  q99 coverage and pinball loss; avoidable-curtailment cost at the signed-MW six (f = 0.9, with
  0.75 and 0.5 as a sweep); best widening for each K from 1.0 to 1.15.
---

# Seed averaging: how many networks does the first release need?

The solar half of the original brief is dropped because experiment 430 already stores six seeds
for `a2_member` and `g0_onepass`; its curve over K is re-scored in a closing section only if time
allows, with no new fits.

## Plan

- **Reuse.** Arm A seeds 0 to 4 for the five months (experiment 425 `work/seed_preds/A`); wind
  `member_plain` seeds 0 to 4 (experiment 429 `parts/`); S1 seeds 0 to 4 (experiment 454).
- **New fits (70).** Arm A seeds 5 to 9 on five months: 25 pretrains and 25 fine-tunes (pretraining
  is per seed and month). Wind `member_plain` seeds 5 to 9: 5 fits. S1 seeds 5 to 7 on five months:
  15 fits, so the S1 pool reaches K = 6 to 8. Seeds 3 and later of S1 share training worlds with
  seed `seed % 3` (experiment 454's caveat), so S1 curves past K = 3 are lower-bound on diversity
  and the page says so.
- **Cadence.** Monthly refits; five of the twelve months. No weekly refits.
- **Standing practices followed.** Cleaned data (flagged exact-zero half-hours dropped, counts
  reported per fold), the D15 and D18 cut before 2025-03-01, saved networks in the experiment's
  data folder, stored predictions reused, tails and curtailment cost reported, example forecasts
  plotted, draft pull request. Load-shift event plots are not needed because the experiment does
  not score events. Nearest-farm pairings are named by letter only.
- **Compute.** One timed arm A fit (month 2025-10, seed 5): 43 s pretrain plus 61 s fine-tune and
  forecast. Estimated total about 1.2 GPU-hours, one process under 8 GB, one feeder core.
- **Where new seeds are written.** This experiment's own data folder. The timing fit above wrote
  `pretrained/A/2025-10_s5.pt`, `seed_preds/A/2025-10_s5.parquet` and the matching `models/A/`
  files into experiment 425's folder; they are copied here and the wrapper refuses to overwrite
  any existing file.

## Results

Not yet run.
