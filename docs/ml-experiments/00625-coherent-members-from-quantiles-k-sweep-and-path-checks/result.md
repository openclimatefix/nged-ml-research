---
id: "00625"
slug: coherent-members-from-quantiles-k-sweep-and-path-checks
title: >-
  How few member paths can NGED's power-flow studies use, and does a Schaake shuffle of residual
  paths give the 16 Primaries any coherence gain?
theme: calibration-and-uncertainty
also_relevant_to:
  - hierarchy-and-pooling
  - generator-forecasting
status: planned
kind: other
protocol: demand-v2
date: 2026-11-07
size: >-
  medium. Stored: tables and figures in this experiment's private folder only; no forecast is
  fitted. Serving path: not touched. Degradation rule: not touched. Designs: more than one
  defensible (which residual paths feed the demand shuffle; how the K paths are chosen), so the plan
  registers each. Callers: this folder's scripts only, which import experiments 434 and 436 and
  `studies.*`. One Sonnet plan review, one Sonnet code review before the first run, one Sonnet
  results review, and two Sonnet write-up reviews.
hypothesis: >-
  Experiments 434 and 436 already scored independent draws (IND), ensemble copula coupling (ECC), a
  Schaake shuffle (SS), and member-wise quantiles (MW) at 51 paths. A Schaake shuffle gave the wind
  farms and solar farms variogram scores 28% to 43% below IND but gave the 16 Primaries nothing.
  This experiment tests three new claims. (1) SS paths cut to K = 10 lose under 2% of the variogram
  score of K = 51. (2) A Schaake shuffle that orders by anomaly paths (observed power minus the
  mean of the same clock time over the 28 preceding days) gives the Primaries a variogram score at
  least 5% below IND, because observed-power paths mix the seasonal level with the weather
  anomaly. (3) SS paths put neighbouring solar farms in opposite sky states less often than IND
  does and about as often as the observations do.
prediction: >-
  Written before any score is read. P1. For SS on the farms, K = 10 is within 2% of K = 51 on the
  variogram score; K = 5 is not. P2. For the Primaries, anomaly-path SS (SS-R) beats IND on the
  Primaries' spatial variogram score by 5% or more. We expect this to fail by a small margin,
  because experiment 436 measured a realised correlation of 0.29 between the Primaries' errors,
  against 0.93 under ECC. P3. The CRPS of K paths is within 1% of the direct 51-value CRPS at K =
  20 and above, and exceeds it by more than 1% at K = 5. P4. On days when the marginal spread of
  the solar farms is in the top quarter, the share of opposite-sky-state pairs is: IND about 8% of
  half-hours, ECC under 2%, SS between the observed share and IND's share. P5. The lag-1
  autocorrelation of step-to-step changes along MW paths is further from the observed value than
  SS's, because MW draws one latent level per member.
decision_rule: >-
  Scores are variogram scores (order 0.5, as in experiment 436) over the 44 Tuesday runs of
  2025-09 to 2026-06, grouped into calendar months, days 1 to 10 unless stated; the energy score is
  the fair form (pair term divided by M(M-1)) so that different K compare, and is secondary. A
  contrast is the relative difference of the monthly mean scores, paired by month, with a
  month-block bootstrap (20,000 resamples), 95% interval, and no multiplicity correction (descriptive
  where it is not named below). Margins: 5% for coherence, 2% for K, 1% for CRPS. `supported`: SS
  or ECC or SS-R beats IND on the variogram score by 5% or more with an interval wholly below zero
  in the pooled joint block, beats MW (farm families) on the same score, and the CRPS gap of the
  K = 51 paths is under 1%. `partly supported`: the same holds in one family (farms or Primaries)
  but not in the joint block, which is what experiment 436 found. Otherwise `not supported`. K
  recommendation, per family: the smallest K in {5, 10, 20, 51} of the best coupling whose
  variogram score is within 2% of K = 51 (point estimate), with its interval reported; the
  recommendation is `K = 10` only if the interval's upper end is also below 4%. The opposite-sky
  check and the autocorrelation check are descriptive (no verdict). No rescue tuning after a failed
  bar. The stop rules are: no stored paths on disk (experiment 436's `work/paths`), or a K = 51
  CRPS gap over 3% for the Primaries.
benchmark: >-
  IND (experiment 436's J0) for coherence; the direct 51-value CRPS for the marginal check; the
  observed share and observed autocorrelation for the path checks.
row_set_id: joint_tuesdays_v1
metric: >-
  Variogram score (order 0.5) and fair energy score of the member paths in experiment 436's four
  blocks; CRPS of the K values against the direct 51-value CRPS; coverage of the member spread at
  50%, 80%, and 90%; share of opposite-sky-state farm pairs; autocorrelation of step-to-step
  changes.
---

**Plan.** Experiments [434](../00434-demand-ensemble-copula-coupling-and-schaake-reordering-of-stored-forecasts/result.md)
and [436](../00436-joint-scenarios-across-demand-wind-solar-and-leads/result.md) already scored four
of this brief's arms at 51 paths. This experiment reuses their scenario tables and adds what they
did not score.

## Reused and new

- **Reused, not recomputed.** The 51 values per node and half-hour (`work/runs`), and the 51-path
  scenario tables of IND (J0), ECC (J2), SS (J3), the copula-corrected ranks (J5), and MW (J6;
  wind and solar only, demand follows J2) from experiment 436's `work/paths`. The marginals are
  those of experiments 425 (demand, widened 1.15), 429 (wind), and 430 (solar).
- **New.** Scores with the fair energy score at K = 5, 10, 20, 51; SS-R (anomaly-path shuffle);
  CRPS from K values; member-spread coverage; opposite-sky-state shares; step-change
  autocorrelation. Experiment 436's 51-path scores are recomputed under the fair energy form, so
  the page does not mix the two forms.

## Design

- **K paths.** For K below 51 the K values are the 51 values interpolated at levels j/(K+1). The
  ordering of the best coupling is cut to K by taking the K Schaake dates nearest in day of year
  (the first K of the pool of 436's J3) and ranking those K paths. ECC cut to K takes the first K
  members' ranks among K. Cutting after the fact never changes the marginal's shape: only the
  number of levels changes.
- **SS-R.** The anomaly of a past date is its observed path minus the mean of the same clock time
  over the 28 days before it, per node. The pool and the 16-day buffer are 436's. It is applied to
  all nodes; the report uses it for the Primaries, and for the farms beside J3. The weather source
  is none (observed power only), so every input is available at forecast time.
- **Why nothing helped across the Primaries.** Reported from the stored `diag_dependence` table:
  the realised and the constructed correlations of the normal scores between Primaries, and the
  share of the Primaries' PIT variance that two common components explain, from the other months
  (eigenvalues of the correlation matrix). No new fit.
- **Opposite sky state.** Pairs are the five solar-farm pairs with the highest correlation of
  daylight observed output over 2025-03 to 2026-06 (a descriptive grouping that stands in for
  distance). For a pair and a daylight half-hour (10:00 to 14:00 UTC, lead days 1 to 10), a path is
  opposite if one farm is above its own q80 of the 51 values and the other below its own q20. The
  share is taken over days whose farm-mean spread (q90 minus q10 of the 51 values) is in the top
  quarter of all run-days, and the observed share uses the actuals on the same half-hours.
- **Autocorrelation.** Lag-1 autocorrelation of the half-hourly change along each path (and along the
  observed path), over days 1 to 3, averaged over paths and runs per family.
- **Standing practices.** Cleaned data as in 436 (flagged half-hours dropped); the D15 and D18 cut
  before 2025-03-01; dev months only, no half-hour on or after 2026-07-01; generators anonymised in
  every chart and table; example fans reuse 436's figure 4 style (a winter and a summer day-ahead
  and 14-day example). Not applicable: no fit, so no refit cadence, model saving, GPU, load-shift
  events, or curtailment cost. The 1% and 3% stop rules above are the brief's.

## Compute and reviews

- CPU only, no GPU, `OMP_NUM_THREADS=1`, `systemd-run --user --scope -p MemoryMax=8G -p
  OOMPolicy=continue`, spawn-based workers. About 2 to 4 hours (estimate from 436's scoring times).
- Reviews: Sonnet plan review; Sonnet code review before the first run; Sonnet results review;
  two Sonnet write-up reviews.
