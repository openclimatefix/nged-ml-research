---
id: "00454"
slug: lag-trust-mixture-and-ensemble-constructs-for-lagged-and-lagless-heads
title: >-
  Which combination of lagged and lagless forecasts scores best, and is any combination better than
  a fixed 50/50 mixture of one of each?
theme: lagged-power-and-error-correction
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: one table of 11-quantile forecasts per construct, month, and cell in this
  experiment's private folder, plus 46 saved neural networks in experiment 404's model folder. More
  than one design would defensibly test the hypothesis (which heads, how many, which weights), so
  the plan fixes them in advance. The scripts import experiment 404's modules and experiment 435's
  tail helpers. Production serving path: not touched (this repository holds none). Degradation
  rule: not touched.
hypothesis: >-
  Part of the gain of B-mix (a fixed 50/50 mixture of a lagged network and a lagless network) over
  one lagged network comes from averaging several networks, and part from mixing two different
  kinds of network. Replacing the lagged network by S1 (the lagged network that also sees how far
  each of its three lags lies from the lagless forecast) makes the mixture better still.
prediction: >-
  Written before any score is read. P1. A two-seed ensemble of A0-inj networks recovers 0.3 to 0.4
  of B-mix's clean-row gain over one A0-inj network (experiment 445 found 0.29 for two seeds and
  0.39 for three), and does not reach B-mix on clean rows. P2. S1-mix beats B-mix on clean rows at
  the signed-MW six by 0.02 to 0.10 points of capacity, which is less than the 0.03 adoption bar
  more often than not. P3. A mixture of K lagged and K lagless networks improves on B-mix by less
  with each added K, and K = 5 is within 0.05 of K = 3. P4. Constructs with more heads need a
  smaller tail widening than 1.15.
decision_rule: >-
  Score = mass-weighted pinball loss (CRPS-like) in points of capacity on experiment 404's rows,
  seeds averaged within a month, then a mean over the five scored months (2025-10, 2025-12,
  2026-01, 2026-03, 2026-06), each contrast taken against B-mix on identical rows, with a
  month-block bootstrap 95% interval (Bonferroni across the stated number of contrasts; the
  interval is wide because only five months exist, so the sign count over months is shown beside
  it). The minimum detectable effect is read from the seed-to-seed spread of B-mix before any
  contrast is read. Adopt the smallest construct whose clean-row score at the signed-MW six is
  within 0.03 of the best construct, no worse than B-mix on real harm rows and on rows inside real
  events (NRA-corrected actuals), and within 0.02 on the q99 pinball loss. S1-mix replaces B-mix
  only if it is at least 0.03 better on clean rows at the six, not worse on harm rows or event
  rows, and within 0.02 on q99 pinball loss. If two-seed ensembles of A0-inj score within 0.03 of
  B-mix, the gain is ensembling and the lagless head adds serving cost for no gain. The smallest
  gain worth porting is 0.03 points of capacity, because a mixture of two networks doubles the
  networks run per forecast. "Clean" always means rows with no labelled event, because NGED's
  event labels are incomplete; a second row set of rows the label-free step scan finds step-free is
  added once that scan has reported, and the verdict is stated on both sets. The MVA-only eight are
  reported separately and never pooled with the six. Interim; no claim is made at the 99% level.
benchmark: >-
  B-mix, experiment 404's fixed 50/50 mixture of the A0-inj network and the frozen cross-fitted
  lagless table, recomputed for fit seeds 0 to 2 and checked against the stored seed 0 forecasts.
  Reference rows: A0-inj, S1, and the lagless head alone. The curtailment reference arms (BG2, S1,
  manual_heuristic) are named in the curtailment section.
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of capacity from 11 quantiles, by row class and
  group, as in experiment 404's report; tail coverage and pinball loss at q01, q05, q95, and q99.
---

## Overview

Planned; the write-up follows the results.

## Plan details

- **Harness.** Experiment 404's: its rows, its five months, six epochs of Adam on stride-2 rows,
  and the frozen cross-fitted lagless tables. Experiment 439's seeds 3 to 5 are not reused: for
  2026-06 on the real world, 404 holds 259,166 rows and 439 holds 192,056, the keys differ, and 439
  trains with early stopping (setting g09).
- **Fits.** A0-inj and S1 for fit seeds 1 to 4 in every month (seed 0 is stored; 2026-06 also holds
  seeds 1 and 2), using experiment 404's fit code. Seeds 3 and 4 reuse the training world of seed
  `seed % 3`, because 404 built training worlds for seeds 100 to 102 only. Each fit takes about 20
  seconds on the GPU.
- **Constructs, all scored on the real world only.** B-mix; S1-mix; two-seed ensembles of A0-inj,
  S1, and the lagless head; K lagged by K lagless mixtures (K = 2, 3, 5) for A0-inj and for S1;
  same-kind ensembles of K = 3 and 5 heads as the baseline for each. Gaussian components are mixed
  with equal weights and the quantiles are solved exactly. The injected worlds are not rescored.
- **Cells.** Constructs with fewer heads than 5 are scored in three seed cells (seeds, or seed
  pairs, 0 to 2) and averaged; K = 3 and K = 5 are single cells.
- **Row classes.** Experiment 404's, with the reconciled harm-row definition from experiment 455.
- **Tails.** q01, q05, q95, q99 coverage and pinball loss, raw, with the widening sweep 1.0, 1.05,
  1.10, 1.15, and after a table P fitted on earlier months' no-labelled-event rows per construct
  (so 2025-10 has no table P).
- **Standing practices followed.** Monthly refits, cleaned data, the D15 and D18 cut (all inherited
  from experiment 404's rows), saved networks, stored forecasts reused as references, a code
  review before the first scored number.
