---
id: "00522"
slug: primary-embedding-data-look-lagless-demand-network
title: >-
  Does the lag-free demand network's 8-wide Primary embedding carry more than a random identity tag?
theme: hierarchy-and-pooling
also_relevant_to:
  - demand-inputs
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: tables, figures, and one small parquet of per-Primary CRPS under this
  experiment's private data folder; every earlier experiment is read and never written. Serving
  path: not touched (this repository holds none). Degradation rule: not touched. Designs: more than
  one defensible (how the initial embedding is regenerated, which distance and which permutation
  test, which descriptors), so the plan fixes each before any statistic is read. Callers: this
  folder's scripts import experiments 397, 425, 428, 448, and 510 and the `studies` code. One Sonnet
  plan review, one Sonnet code review before the first number is quoted, one Sonnet results review,
  and two Sonnet write-up reviews.
hypothesis: >-
  The lag-free network (the L0PT network of experiment 448: pretrained on ERA5 and CAMS rows, no
  lagged-power input) identifies each Primary through an 8-wide embedding vector that starts as a
  random N(0, 1) draw. The vectors mostly stay at their random start, and the part they learn is
  small and not arranged the way a Primary's physical behaviour is. The network still uses the
  vectors, because a random fixed label is enough to separate 16 Primaries.
prediction: >-
  Written before any statistic is read. P1: the per-dimension standard deviation of the stored
  vectors is 1.0 to 1.2, and the learned movement (stored vector minus regenerated start) is 40% to
  60% of the start's norm after pretraining and under 5% in daily fine-tuning. P2: the distance
  matrix of the movement, averaged over 5 seeds, has a Mantel correlation with at most 2 of the 10
  descriptor-distance matrices that survives the Bonferroni count of 20. P3: swapping the vectors
  of two Primaries raises the CRPS of every scored Primary by more than 0.5 points of capacity (the
  removal of the embedding costs 0.65 to 0.85 points). P4: scoring every Primary with its own
  regenerated start vector (the "start" arm) costs under 0.3 points against the stored vectors, so
  the learned movement is nearly unused. P5: the lagged network (arm A of experiment 425) loses less
  than the lag-free network under the swap, by at least a third.
decision_rule: >-
  "Thin" if no Mantel test passes the Bonferroni count of 20 and the start arm scores within the
  minimum detectable effect of the stored vectors; then a follow-up brief for the best
  training-window-only option of Part 4 is warranted. "Healthy" if at least one Mantel test passes
  the count and the start arm scores worse than the stored vectors by more than the minimum
  detectable effect. Anything else is "mixed", and the page says which half failed. The page makes
  no claim of benefit either way: it is a diagnostic. The minimum detectable effect for one fold and
  one seed is taken as 1.5 to 2 times the two-seed, five-fold value of experiment 511: 0.164 at the
  eight and 0.123 at the six.
benchmark: >-
  Within the experiment: the stored vectors (identity) against the swap, the mean vector, the start
  vector, and fresh N(0, 1) vectors. The lagged network of experiment 425 (arm A) under the same
  tests.
row_set_id: experiment_522_test_2026_01_all_leads
metric: >-
  Rank correlation and Mantel test of 16 x 16 distance matrices; CRPS in points of effective
  capacity per Primary on the 2026-01 test month (all leads up to 240 hours, rows with a cleaned
  actual).
---

PLANNED. The write-up replaces this page when the numbers exist. Interim numbers in this plan come
from the manager's queue-order check and are unaudited.

## Question

Without a lagged-power input, the lag-free network learns which Primary it is forecasting for only
through the 8-wide Primary embedding. The weather columns come from each Primary's own H3 cell, and
the calendar and solar-geometry columns may also differ by site (checked in the code review). The
queue-order check found that the stored vectors are mostly their random N(0, 1) start. This
experiment measures how much the vectors learn, whether the learned part is arranged by physical
behaviour, and whether the network uses it.

## Parts

All parts use stored weights, CPU only, two cores, and no fitting.

- **Part 1: displacement from the start.** Regenerate each start with `torch.manual_seed(seed)`
  followed by the construction of `PlainMdn` (verified before the plan: the regenerated start
  correlates 0.88 to 0.91 with the stored vector of the same seed and about 0 with other seeds).
  Weights: 5 seeds x 4 folds of L0PT (experiment 448: 2025-10, 2026-01, 2026-03, 2026-06) and the
  L-1d seed 0 network of experiment 511 at 2026-01. Report the per-dimension standard deviation, the
  singular values and the number of dimensions for 90% of the variance, the displacement from the
  start as a share of the start's norm per fold and seed, and the movement per Primary (the only
  learned part). Alignment (Spearman correlation of the 120-pair distance matrices) only between
  same-seed pairs after removing the start, and between the movements of different seeds as a
  descriptive null. D15 and D18 are included and report-only.
- **Part 2: does the movement follow physical behaviour?** Ten descriptors per Primary, cut at
  each fold's pretraining end: median and 95th percentile of demand and load factor (points of
  effective capacity), the weekday daily profile in winter and in summer (48 values each), the
  weekend-to-weekday ratio in winter and in summer, and the slopes of demand on ERA5 temperature in
  winter and in summer. The shortwave sensitivity at midday, and the reverse-flow share (the six)
  and the midday-dip share (the eight, experiments 506 and 508), are reported beside the tests and
  stay out of the counted family, because each covers a subset of 6 or 8 Primaries. Assets and the
  grid supply point group are annotation only. A Mantel test (10,000 permutations of the 16 labels)
  on 120 pairs, not a Procrustes fit. Counted tests: 10 descriptors x 2 views (the stored vectors,
  and the movement) = 20, on fold 2026-01 with the seed-averaged distance matrix (the other three
  folds are descriptive), Bonferroni level 0.05 / 20. A hierarchical clustering of the movement is
  listed next to the descriptors. The 120 pairs come from 16 points, so the pair count overstates the
  independent evidence.
- **Part 3: does the network use the vectors?** Fold 2026-01, seed 0, with the pretrained L0PT
  weights and their stored standardiser. Each Primary keeps its own target statistics. Arms:
  identity (checked against experiment 448's stored forecasts), 20 random derangements of the
  vectors, the mean vector, the start vector of each Primary, and 20 draws of fresh N(0, 1)
  vectors. The same arms on arm A of experiment 425. Output: CRPS change per Primary in points of
  capacity. The swap shows that the vectors are used, not that they are informative.
- **Part 4: other ways to tell the network which Primary it is.** A design list only, with a
  prediction that could fail and a cost for each option: initialise the embedding from per-Primary
  descriptors, a higher learning rate or no weight decay on the embedding, static descriptor inputs
  recomputed at each refit, a wider embedding, a per-Primary output scale and bias head, and
  feature-wise modulation of the hidden layers. It cites experiments 406, 421, 425, and 448. None
  is run.
- **Consistency with experiment 511.** The page says whether 511's short-lead result (the lag-free
  network beats K2-S1 below about 2 days and the gap sits at days 5 to 10) is consistent with a thin
  embedding.

## Standing practices

- Cleaned data throughout (`studies.power.scan_power` or experiment 87's cleaned arrays); the
  D15 and D18 start cut at 2025-03-01 applies to every descriptor, and the count of dropped
  half-hours is reported.
- No fits, so no refit cadence and no seeds beyond those stored. Intervals are bootstrap intervals
  over the 5 stored seeds and 4 folds, which is very few; the page says so.
- Tail calibration, curtailment cost, and load-shift plots do not apply to a diagnostic of
  stored weights. The page says that.
- July 2026 is unused. Load-shift, never "switching", in every title.
- Metered generators do not appear. Substations are named.
