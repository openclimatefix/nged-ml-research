---
id: "00627"
slug: scenario-subset-selection-from-51-coherent-paths
title: >-
  Which 3, 5, or 10 of the 51 coherent member paths should NGED's power-flow studies use, with no
  label attached to any path?
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
  defensible (summary choice, clustering distance, selection window), so the plan registers each.
  Callers: this folder's scripts only, which import experiments 434, 436, and 625 and `studies.*`.
  One Sonnet plan review, one Sonnet code review before the first run, one Sonnet results review,
  and two Sonnet write-up reviews.
hypothesis: >-
  NGED will run a handful of power-flow studies by hand, so they need K of the 51 coherent member
  paths (K = 3, 5, 10) that cover the distribution of the 51, without labelling any path. A path
  that is low wind on day 1 need not be low wind on day 2, so a label such as "the low-wind
  scenario" does not carry over. We test whether picking K paths by stratifying on one summary of
  the run (S) or by clustering the 51 joint paths (C) covers the 51-path distribution better than a
  simple random sample (R).
prediction: >-
  Written before any score is read. P1. At K = 5 the random sample R has a relative gap above 10%
  at the 5th or 95th percentile of the 51-path distribution on at least one summary, and holds the
  tails in about 1 - (1 - 0.1)^5 = 41% of draws at each tail. P2. A stratified sample S matches the
  distribution of the summary it is ranked on (gap under 10%) but not the other two summaries, so
  the mean over the three summaries fails the 10% bar. P3. The cluster medoids C are central paths
  and rarely include a tail path, so C has lower tail coverage than R. Therefore the registered
  verdict is `not supported` at K = 5, and K = 10 narrows the gap without closing it. P4. The share
  of members that keep the same stratum from one weekly run to the next is within 3 points of the
  chance level 1 / K for the Schaake-shuffled members, and about the same for ensemble-copula
  members.
decision_rule: >-
  Fixed before any score is read. Rows: the Tuesday runs of 2025-09 to 2026-06 stored by experiment
  436 (44 expected; the count is printed first), grouped into calendar months for a month-block
  bootstrap (20,000 resamples). Paths: experiment 436's Schaake-shuffled scenarios (J3), 51 per run
  over 25 nodes (16 Primaries, 3 wind farms, 6 solar farms) and 702 half-hours, in points of
  capacity; the ensemble-copula scenarios (J2) are a sensitivity. Three summaries per path and lead
  band: the mean over the wind farms, the mean over the solar farms, and the peak over the band of
  the net demand (mean over the Primaries minus the wind-farm mean minus the solar-farm mean; an
  unweighted proxy, because MW weights are not available). Lead bands: 12 to 36 hours, days 1 to
  3, days 4 to 10, and days 1 to 10 (the headline). The selection is made once per run on the days 1
  to 10 window; a selection made inside each band is a secondary arm. Arms: R (1,000 random draws
  of K paths per run), S-wind, S-solar, and S-net (rank the 51 paths by that summary, cut into K
  equal-count strata, draw one path from each; 1,000 draws per run), C (k-medoids with k-medoids++
  starting points and seed 627, on the 51 joint paths with each node and half-hour standardised
  across the 51 paths and every 4th half-hour of the window, each medoid weighted by its cluster
  size), and All (the 51). Coverage: per summary and band, the Wasserstein-1 distance of the
  weighted K-path distribution to the 51-path distribution, and the gap at the 5th, 50th, and 95th
  percentiles (lower inverse CDF), each divided by the spread (95th minus 5th percentile) of the
  51-path distribution of the same run, then averaged over summaries and runs. Tail coverage: the
  share of draws (for C, of runs) whose K paths include a path at or beyond the 10th percentile
  (lower tail) or the 90th percentile (upper tail) of the 51 on a summary, averaged over summaries
  and both tails. Joint score: energy score and variogram score (order 0.5) of the K weighted paths
  against the observed vector over the 25 nodes at a coarse step (about 25 half-hours per band),
  reported as the relative difference to All. Stability: the share of members that fall in the same
  one of K strata on consecutive weekly runs (S-net at K = 5), against the chance level 1 / K.
  The verdict is judged at K = 5, days 1 to 10, J3. A label-free arm passes the coverage condition
  if its mean relative gap is at most 0.10 at each of the three percentiles (point estimate), and
  passes the tail condition if its tail coverage exceeds R's, with a month-block 98.75% interval (a
  Bonferroni correction over the four label-free arms S-wind, S-solar, S-net, and C) that lies wholly
  above zero. Verdict: `supported` if at least one label-free arm passes both conditions; `partly
  supported` if at least one passes the tail condition but none passes the coverage condition;
  `not supported` otherwise. K = 3 and K = 10, the other bands, the other summaries, the joint
  scores, the stability check, and J2 are descriptive and carry no verdict. No rescue tuning after
  a failed bar. Stop rule: experiment 436's `work/paths/J3` is missing for any of the stored runs.
benchmark: >-
  R (a simple random sample of K paths) for the verdict; All (the 51 paths) as the reference for
  every score.
row_set_id: joint_tuesdays_v1
metric: >-
  Relative gap at the 5th, 50th, and 95th percentiles and Wasserstein-1 distance of the K-path
  distribution to the 51-path distribution; tail coverage at the 10th and 90th percentiles; energy
  and variogram score relative to All; stratum stability across weekly runs.
code: ml-experiments/00627-scenario-subset-selection-from-51-coherent-paths/
---
Planned. The write-up replaces this page when the scoring has run.

## Plan

- **Reuse.** Experiment 436's Schaake-shuffled (J3) and ensemble-copula (J2) scenario files, and
  its run files for the observed vector, as in [experiment
  625](../00625-coherent-members-from-quantiles-k-sweep-and-path-checks/result.md). Nothing is
  written into either folder. No forecast is fitted.
- **Standing practices.** Dev months only (no valid time on or after 2026-07-01); cleaned data and
  the D15 and D18 cut before 2025-03-01 as in experiment 436; generators anonymised; no refit,
  model saving, GPU, load-shift event, or curtailment-cost work applies.
- **Compute.** CPU only, `OMP_NUM_THREADS=1`, `systemd-run --user --scope -p MemoryMax=8G -p
  OOMPolicy=continue`, spawn-based workers; a few minutes per run.
- **Related work.** Experiment 626 widens the tails of the 51 member values and is not used here.
