---
id: "00436"
slug: joint-scenarios-across-demand-wind-solar-and-leads
title: >-
  Which construction of 51 scenarios, joint over the 16 Primaries, three wind farms, six solar farms,
  and leads, best matches the observed joint behaviour, and does tying scenario k to ensemble member
  k earn its complexity?
theme: calibration-and-uncertainty
also_relevant_to:
  - hierarchy-and-pooling
  - generator-forecasting
status: planned
kind: other
protocol: demand-v2
date: 2026-10-31
size: >-
  complex. Stored: tables and figures in this experiment's private folder; one scenario table per arm
  (key: run, node, valid time, scenario) for reuse. Designs: more than one defensible (eight
  constructions are registered, and the choice among them is the question). Callers: this folder's
  scripts only, which import the folders of experiments 429, 430, and 434 and `studies.*`. No fit in
  tiers 0 and 1; tier 2 (12 demand fits) needs the maintainer's go-ahead and is not planned here.
  One Sonnet plan review, one Sonnet code review before the first run (a second if the first finds
  many problems), one Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  The stored marginal forecasts of demand (experiment 425 arm A), wind (experiment 429), and solar
  (experiment 430) can be turned into 51 scenarios that are one coherent realisation across the 16
  Primaries, the three wind farms, the six solar farms, and the leads, without changing any
  marginal. A single historical ordering (a Schaake shuffle, J3) or an ensemble ordering corrected
  by a fitted error copula (J5) matches the observed joint behaviour better than the ensemble's own
  member ranks (J2) do. Tying scenario k to ensemble member k (J4) costs no coherence for demand,
  and helps for wind and solar.
prediction: >-
  Written before any score is read. P1. For demand, J3 or J5 beats J2 on the within-family spatial
  and the joint block, and J4 ties J3 within the margin. P2. For wind and solar, J5 or J4 beats J2,
  and J6 (member-wise quantiles, latent level from J5's copula) only ties J5. P3. In the cross-family
  block in daylight, J2 or J4 beats J3, because shortwave radiation drives both net demand (through
  embedded solar) and solar output. P4. Every construction other than J1 leaves each marginal CRPS
  unchanged (a check, not a contrast), and J6 meets the marginal coverage tolerances. P5. With about
  43 joint runs the downstream proxies have wide intervals and no construction beats J0 on the
  simultaneous-exceedance Brier score at the 99% level.
decision_rule: >-
  Scores are in points of capacity (effective capacity for demand), lower is better. Rows are the
  Tuesday runs, 2025-09 to 2026-06, that all three families scored (the count is printed before any
  score). The 12 monthly folds are collapsed to calendar months; the bootstrap resamples months
  (month-block, 20,000 draws), with Bonferroni correction over the four registered contrasts (98.75%
  interval). Coherence scores are the energy score and the variogram score (order 0.5) divided by the
  block size, as in experiment 434, in four blocks (within-family spatial, cross-family, temporal,
  joint) at 12 to 36 hours and at days 1 to 10. The gating block is the joint block at days 1 to 10.
  The coherence margin for a block is a quarter of the relative gap between the comonotone (J1) and
  independent (J0) variogram scores in that block, bounded to 2% at the lowest and 5% at the highest;
  the margin is computed from the data before the contrasts are read. A construction counts as
  coherent if its joint variogram interval lies wholly below minus the margin against J0 and its
  multivariate rank histograms are no worse than J3's (the average-rank histogram's distance from
  flat, with J3's distance as the bound). Contrasts: C1 J4 against J3, C2 J5 against J2, C3 J6
  against the best of J2 to J5 on the joint block, C4 the best construction against J0 on the
  simultaneous-exceedance Brier score. Choosing a design: the simplest construction (order J0, J1,
  J2, J3, J4, J5, J6) within the margin of the best on the joint variogram score that is also no
  worse than the best on the simultaneous-exceedance Brier score. J6 advances only if its marginal
  meets the q95 and q99 coverage tolerances (within 0.03 of 0.95, and within 0.015 of 0.99) and its
  CRPS is within 0.03 points (demand), 0.10 points (wind), and 1% (solar) of the stored CRPS. An
  interval that straddles minus the margin is "no difference at the margin", reported with the
  direction of the point estimate. Tier 2 starts only if J4 or J6 beats J3 for demand by the margin,
  and only with the maintainer's go-ahead.
benchmark: >-
  J0, independent draws from the same 51 values per node and lead; J1 (comonotone) as the dependence
  upper bound; the stored quantile CRPS of each family for the marginal check.
row_set_id: joint_tuesdays_v1
metric: >-
  Marginal CRPS of the 51 values; energy score and variogram score (order 0.5) in four blocks;
  multivariate average-rank and band-depth histograms; the root-mean-square difference between the
  scenarios' and the observations' Spearman correlation matrices of probability-integral-transform
  values; spread-skill ratio; q95 and q99 coverage; Brier scores of simultaneous exceedance, group
  net flow below its export proxy, and ramps; group-sum CRPS; tier 1 curtailment volume.
production_status: not-for-production
adopted_in: none
---

**Plan only: no score has been read.** The question is how to turn the calibrated marginal forecasts
of the Primaries, wind farms, and solar farms into 51 scenarios that a power-flow study can use, each
scenario being one coherent realisation across substations, across demand, wind, and solar, and
across time. We take the stored 51 values per node and lead, re-order them in eight ways, and score
how well the re-ordered scenarios match the observed joint behaviour. We fit nothing in tiers 0 and
1, so the marginals do not change.

## Design

- **Why now.** Experiment 434 (pull request 91) found that no ordering beat independent draws for
  demand alone. Experiments [429](../00429-unified-forecaster-wind-ablations/result.md) and 430
  (pull request 87) found that a Schaake shuffle beat ensemble copula coupling (ECC) for the farms
  alone. No experiment has scored a
  scenario joint across families. The design note (the Opus note of 2026-10-31, in the private
  `briefs/` folder) gives the reasons and eight questions for NGED; the maintainer approved this
  experiment on 2026-10-31.
- **Working assumption, to be confirmed with NGED.** NGED needs each scenario to be one coherent
  realisation across demand, wind, and solar, across substations, and across time, to feed power-flow
  models. Requirement (d), that scenario k uses ensemble member k's weather, is untested.
- **Marginals.** All constructions share one set of 51 values per node and lead: the values at levels
  k/52 (k = 1 to 51), by linear interpolation of the stored 11 quantiles (as in experiment 434), no
  extrapolation. Demand: experiment 425 arm A, seeds averaged in quantile space, widened by 1.15
  (the recalibration table of experiment 435, pull request 92,
  has not reported, so the widening is the one 434 used). Wind: experiment 429's `one_pass_ms`
  quantiles, five seeds averaged. Solar: experiment 430's `g0_onepass`, seeds averaged.
- **Row set.** The Tuesday runs of 2025-09 to 2026-06 present in all three families. The three
  families used the same ensemble run per date: demand and wind publish at 09:00 UTC from the 00:00
  UTC ensemble run, and solar's `init_time` is that 00:00 UTC run. The builder checks the lead-0
  valid times match, aligns all families by valid time on the 702 half-hours after 09:00 UTC, and
  prints the run count before any score. Experiments 428, 431, and 432 hold no usable member input
  or forecasts and are not used.
- **Nodes.** 16 Primaries, wind farms W1 to W3, and solar farms A to F (anonymised in every chart
  and table).

### Arms

| Arm | Construction | Stands for |
|---|---|---|
| J0 | independent draw per node and lead | floor |
| J1 | comonotone: scenario k is level k everywhere | all nodes at one rank |
| J2 | ECC on a shared member index: temperature (shortwave in daylight, as 434's ECC-SW) for demand, 100 m wind for wind, shortwave for solar | ranks from the weather members |
| J3 | one Schaake template: one historical date orders every node, family, and lead | historical dependence, no member tie |
| J4 | member-anchored Schaake: scenario k takes the date whose observed weather over the farms' and Primaries' cells best matches member k | historical dependence, weak member tie |
| J5 | J2's ranks re-mixed with a Gaussian copula of PIT residuals fitted on the other months, mixing weight per family and lead band | ECC corrected by an error copula |
| J6 | member-wise quantiles where stored (wind `memberwise`, solar `a2_memberwise`), demand by J2; latent level per scenario drawn from J5's copula | member-wise with a residual copula |
| J6-median | J6 with each member's median path | member paths, no residual |

### Tier 0 and tier 1 fits

- **Tier 0 (no fit): J0 to J4 and J6-median.** J1 and J0 are exact. J3 follows experiment 434's
  `schaake_pool` extended to all nodes: the 51 candidate dates nearest in day of year, from
  2025-03-01 and at least 16 days before the run; a date is dropped if any node lacks more than 5%
  of its half-hours; the pool for the farms starts when each farm has data. J4 matches member k to
  the date by ERA5 temperature, 100 m wind, and shortwave over the nodes' cells (the weather source
  and its availability at forecast time are stated in the write-up; ERA5 at the matched date is
  past weather, so it is available at forecast time for dates at least five days old, and the pool
  respects that gap).
- **Tier 1 (cheap fits): J5 and J6.** The copula correlation is structured, not free: an exponential
  decay in lead time times a family-block matrix, because one year of data estimates a free 25-node
  by 28-lead matrix poorly. It is fitted on probability-integral-transform residuals of the other
  months (leave one month out). The mixing weight is chosen from {0, 0.25, 0.5, 0.75, 1} by the
  joint variogram score on the training months, per family and per lead band (12 to 36 hours, days 1
  to 10).
- **Tier 2 is not planned.** If tier 1 shows that member identity matters for demand (J4 or J6 beats
  J3 by the margin), the manager asks the maintainer for the go-ahead for 12 demand fits.

### Metrics

- **Marginal check.** CRPS per node from the 51 values, within 0.03 points of the stored quantile
  CRPS. The calibration tolerance for J6: q95 coverage within 0.03 of 0.95, q99 within 0.015 of 0.99
  at days 1 to 10, with the accepted clean-row bands (q95 0.93 to 0.97, q99 0.98 to 0.995) beside
  them.
- **Coherence scores, four blocks.** Within-family spatial (same half-hour), cross-family (the same
  half-hour across families), temporal (each node across leads), joint (every node at 6-hourly
  leads). Each is reported at 12 to 36 hours and days 1 to 10, with daylight and cold-night slices.
- **Multivariate rank histograms.** Average-rank and band-depth histograms of the 52 values (51
  scenarios and the observation) per block
  ([Thorarinsdottir et al. (2016)](https://doi.org/10.1080/10618600.2014.977447)).
- **Dependence check.** Spearman correlation matrices of the scenarios' PITs across nodes and across
  lead lags of 0.5, 3, 12, and 24 hours against the same matrices from the realised PITs: the
  root-mean-square difference per block, and the demand-to-solar and demand-to-wind entries.
- **Spread-skill ratio** per family and lead day, as the double-counting test; for J6, the variance
  between members and within members.
- **Downstream proxies.** Groups come from the NGED hierarchy file (two bulk supply points, Boston
  and Skegness, each with the Primaries and farms beneath it); the limit proxies are stated before
  scoring. Simultaneous exceedance (Brier score of at least two Primaries above their own static q95
  proxy limit in the same half-hour, and of group net flow below its export proxy); group-sum CRPS and
  q95 and q99 coverage; ramps (Brier score of a ramp larger than 20% of capacity within 3 hours, per
  node); tier 1 curtailment volume of the [cost-savings
  metric](https://openclimatefix.github.io/nged-substation-forecast/roadmap/cost-savings-metrics/)
  per scenario at each group, against the realised volume.

### Standing practices

- **Followed.** Cleaned data everywhere (flagged `substation_zero` half-hours dropped, counts per
  block reported); the D15 and D18 cut before 2025-03-01; tails reported at q01, q05, q95, q99;
  example forecasts (a day-ahead and a 14-day scenario fan, a winter and a summer run, anonymised
  generators); each review before its step, per the code-review-first order; draft pull request.
- **Not applicable.** No fit in tiers 0 and 1, so no refit cadence, model saving, or GPU use; no
  load-shift events are scored, so the event plots do not apply.
- **Earlier advice followed.** Experiment 429: Schaake gives more coherent farm paths than ECC;
  report the upper tail. Experiment 434: report the spatial and temporal blocks beside the joint
  block; score the 51 values with the energy form of the CRPS, and do not compare with the 11-level
  CRPS of earlier experiments (the check compares the 51-value CRPS with the stored 11-quantile CRPS
  recomputed in the same form).

## Compute and reviews

- **Compute.** CPU only, one core (`OMP_NUM_THREADS=1`), `nice -n 19`, memory capped with
  `systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue`. About 4 to 8 hours of scoring
  (estimate). No GPU.
- **Reuse.** The scorer and pool of experiment 434, the quantile tables of experiments 425, 429,
  and 430, and the member weather arrays of experiments 429 and 430 are read, not refitted; no fit is
  saved by repeating one.
- **Reviews.** Sonnet plan review; Sonnet code review before the first run; Sonnet results review
  (re-derives the headline numbers); two Sonnet write-up reviews.

## Details registered after the plan review

- **Leakage rules.** The copula correlation and the mixing weight are both re-selected inside each
  held-out-month fit, from the other months only (nested). The Schaake pools (J3, J4) hold only dates
  strictly earlier than the run date, at least 16 days earlier so that every path ends before the run
  (J3) and at least 5 days earlier for ERA5 (J4). J4 matches member k's forecast weather to the past
  observed ERA5; it never reads observed weather at the run date. A pool starts on 2025-03-01 and is
  thin for the earliest runs (about 6 months, seasonally mismatched); if fewer than 51 dates qualify,
  the nearest dates repeat cyclically, as in experiment 434, and the pool size per run is reported.
- **Bootstrap.** The resampling unit is the calendar month, with each month's score the mean over its
  runs; the number of runs per month is printed. With about 10 months the 98.75% interval is close to
  the range of the resampled means, so "straddles the margin" is the likely outcome and the page says
  so. A run-level block bootstrap (blocks of 2 weeks) is reported beside it as a sensitivity check.
- **Margin.** Relative gap = (J1 score minus J0 score) divided by J0's score, per block at each lead
  window (not per slice); margin = a quarter of it, bounded to 2% to 5%. The gap is computed from the
  data before any contrast is read. J0 is the reference of every coherence contrast.
- **Decision rule details.** A construction must first pass the coherence gate. "Within the margin of
  the best" means the interval of the difference to the best includes no value beyond the margin.
  "No worse on the exceedance Brier score" means the point estimate is not higher than the best's by
  more than 5% of J0's score and the interval of the difference includes zero. "Rank histograms no
  worse than J3's" means the average-rank histogram's distance from flat is below J3's in the
  bootstrap's median. J6-median is a diagnostic and is not ranked.
- **Expected sizes.** Demand: the J1-to-J0 gap on the joint block about 11% (experiment 434), so a
  margin about 2.8%. Farms: Schaake against ECC 9% to 13% lower on the variogram score (experiments
  429, 430), and J5 closing at least a third of that gap to J3. Spread-skill ratio of the
  calibrated scenarios near 1; J6's between-member share of variance below 50%. The dependence-check
  difference ranks J1 worst, then J2, and J3 or J5 best. P5's direction: the point estimate favours J3
  or J5 over J0 on simultaneous exceedance.
- **Per-family reporting.** C1 to C3 are also reported for demand, wind, and solar separately, as
  descriptive results without Bonferroni correction; the verdicts use the pooled contrasts.
  Expected outcome of tier 2: dropped (P1 predicts J4 ties J3 for demand); the gating block is the
  joint block at days 1 to 10.
- **Alignment.** 702 half-hours is lead 0.5 to 351 hours, the 14.6-day horizon of experiment 434.
  If a family's run for a date used a different ensemble run, the builder aligns by valid time and flags
  the date; the count is reported.
- **Slices and proxy file.** Daylight is the local mid-day band and a cold night is the shared slice
  table's flag, as in experiment 434. The group limit proxies are written to
  `proxies.json` in the experiment's folder and committed before the first scoring run. Curtailment
  volume is compared with the realised volume by absolute error per group and run.

## Results

To be written after the run.
