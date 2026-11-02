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
  The Primary embedding of the lag-free neural network does more than label the 16 Primaries. The
  embedding is the 16 x 8 table of vectors that tells the network which Primary (primary
  substation) a row belongs to. The network is the L0PT network of experiment 448: it reads no
  lagged power and is pretrained on ERA5 and CAMS estimates of past weather. Each vector starts as
  a random N(0, 1) draw. The hypothesis is that the movement from that start is small, is not
  arranged by a Primary's physical behaviour, and is not needed by the network, because a random
  fixed label is enough to separate 16 Primaries.
prediction: >-
  Written before any statistic is read. P1: the displacement (the norm of the stored vectors minus
  the regenerated start, divided by the norm of the start, over all 16 Primaries) is 0.40 to 0.60
  after pretraining, and under 0.05 between the pretrained and the daily fine-tuned weights. This
  differs from the queue-order check's 15% to 17%, which compared two folds of one seed with each
  other and not with the start. P2: none of the 11 counted Mantel tests (10 descriptors and the
  metering type) and the metering probe passes the level 0.05 / 12 = 0.0042. P3: under the 20 random
  derangements, the mean CRPS (continuous ranked probability score) of the 14 Primaries that are not
  report-only rises by more than 0.5 percentage points of effective capacity; removing the embedding
  altogether raised it by 0.65 to 0.85 in experiments 406, 425, and 448. P4: scoring each Primary
  with its own regenerated start vector (the "start" arm) raises CRPS by less than the minimum
  detectable effect, so the movement is nearly unused. P5: the lagged neural network (arm A of
  experiment 425) loses less than the lag-free neural network under the derangements, by at least a
  third.
decision_rule: >-
  The minimum detectable effect is fixed at 2 times the two-seed, five-fold value of experiment 511:
  0.328 percentage points at the 8 Primaries that report apparent power only (MVA) and 0.246 at the
  6 that report signed megawatts. Applying it to one fold and one seed, and to per-Primary changes,
  is an assumption. "Thin": no Mantel test and no probe passes 0.0042 and the 95% interval of the
  start-arm-minus-stored CRPS difference lies inside the minimum detectable effect; a follow-up
  brief for the best option of Part 4 that uses only the training window is then warranted.
  "Healthy": at least one Mantel test or the probe passes 0.0042 and the start arm is worse than the
  stored vectors by more than the minimum detectable effect. Anything else is "mixed", and the page
  says which half failed. A null Mantel result is weak evidence, because 16 points have little
  power; the planted-signal control below reports how much. The page makes no claim of benefit.
benchmark: >-
  Within the experiment: the stored vectors (identity) against the derangements, the mean vector,
  the start vector, and fresh N(0, 1) vectors. The lagged neural network of experiment 425 (arm A)
  under the same tests.
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

All parts read stored weights on CPU with two cores and fit nothing. "Primary" means a primary
substation. D15 and D18 (the two short-history Primaries) are among the 16 permuted labels and
every table row, and are left out only of the headline means.

- **Part 1: displacement from the start.** The start is regenerated with `torch.manual_seed(seed)`
  followed by the construction of the network. The check made before this plan found that the
  regenerated start correlates 0.88 to 0.91 with the stored vectors of the same seed and about 0
  with other seeds. The weights are 5 seeds x 4 folds of L0PT (2025-10, 2026-01, 2026-03, and
  2026-06), the same cells of arm A of experiment 425 (checked to share the start, because both
  build the embedding first), the fine-tuned L0PT weights of experiment 448, and one fine-tuned
  L-1d network of experiment 511 (daily refit, 2026-01, day 10, seed 0). The page reports the
  per-dimension standard deviation, the singular values, the number of dimensions that carry 90%
  of the variance, and any near-constant dimension. It reports the displacement per fold and seed
  and the movement per Primary. It also reports the drift between the pretrained and the
  fine-tuned weights. Spearman correlations of the 120-pair distance matrices come only from
  same-seed pairs after the start is removed, with the cross-seed correlations as a descriptive
  null. Each correlation is reported with its number of matrix pairs.
- **Part 2: does the movement follow physical behaviour?** Ten descriptors per Primary are cut at
  each fold's pretraining end: median and 95th percentile of demand, load factor, the weekday
  profile in winter and in summer (48 values each), the weekend-to-weekday ratio in winter and in
  summer, and the slope of demand on ERA5 temperature in winter and in summer. Coverage of ERA5 at
  the 16 H3 cells is verified first. The midday shortwave sensitivity, the reverse-flow share (the
  6 signed-MW Primaries), and the midday-dip share (the 8 MVA-only Primaries, experiments 506 and
  508) are reported beside the tests and stay out of the counted family, because each covers 6 or 8
  Primaries. Assets and the grid supply point group are annotation. The test is a two-sided Mantel
  test with Spearman correlation and 10,000 permutations of the 16 labels, with no Procrustes fit.
  The distance is plain Euclidean distance between the movement vectors, averaged over the 5 seeds
  of fold 2026-01, and the descriptor distance is Euclidean on standardised descriptors. The
  counted family is these 10 descriptor tests, the Mantel test of the binary metering type (signed
  MW or apparent power in MVA; distance 0 within a group and 1 between), and a leave-one-Primary-out
  linear probe of the metering type from the 8-wide vector (a logistic regression with L2 penalty,
  fitted on 15 Primaries and predicting the 16th, once per seed, with majority vote across the 5
  seeds and an exact binomial interval at the Bonferroni level). That is 12 tests at level
  0.05 / 12; the brief's count of 14 includes descriptors this plan reports beside the tests. D15 and
  D18 count in the group of their own metering unit, so there are two groups and not three. The
  sign of each Mantel correlation is reported per fold. The metering-flag arm that the probe
  motivates is a later experiment and is not part of this one; the other three folds, the stored-vector
  view, and a cosine-distance version are descriptive. The correlation of each descriptor with the
  movement norm is reported, because a movement norm that tracks demand level would drive a pass.
  A planted-signal control builds a movement that equals the descriptor-derived signal plus noise
  at the observed movement norm, and reports the detection rate. The page lists a hierarchical
  clustering of the movement next to the descriptors, says which expected pairs cluster and which do
  not, and states that 120 pairs from 16 points overstate the independent evidence.
- **Part 3: does the network use the vectors?** The weights are the pretrained L0PT weights of fold
  2026-01, seed 0, with the stored input standardiser, so that only the embedding changes. Each
  Primary keeps its own target statistics. The arms are the identity (checked against experiment
  448's stored forecasts), 20 random derangements of the 16 vectors, the mean vector, the start
  vector of each Primary, and 20 draws of fresh N(0, 1) vectors (one vector per Primary, drawn
  independently of the start seed). Arm A of experiment 425 gets the same arms except the start arm,
  unless the regenerated start is verified for it. The output is the per-Primary CRPS change in
  percentage points of effective capacity, averaged over the 20 draws. The rows are the 2026-01 test
  rows (all leads up to 240 hours, rows with a cleaned actual). Intervals come from the 20 draws
  and from a bootstrap over origin days; the part covers one fold and one seed. The derangements
  show that the vectors are used, not that they are informative.
- **Part 4: other ways to tell the network which Primary it is.** The page lists, without running
  them, these options, each with a prediction that could fail, a cost estimate, and whether it uses
  only the training window: initialise the embedding from per-Primary descriptors, a higher learning
  rate or no weight decay on the embedding, static descriptor inputs recomputed at each refit, a
  wider embedding, a per-Primary output scale and bias head, and feature-wise modulation of the
  hidden layers. The list cites experiments 406, 421, 425, and 448.
- **Consistency with experiment 511.** The page says whether 511's short-lead result (the lag-free
  neural network beats K2-S1 below about 2 days, and the gap to K2-S1 sits at days 5 to 10) is
  consistent with a thin embedding.

## Figures

The page charts the 16 x 16 pairwise-distance matrix of the movement, and the two leading
dimensions of the movement with Primary names, both with the OCF palette and no mark in the
background colour `#fffbf5`. It adds a chart of the displacement per Primary and one of the Part 3
CRPS changes.

## Standing practices

- Cleaned data throughout, with the flagged half-hours dropped; the D15 and D18 start cut at
  2025-03-01 applies to every descriptor (so their descriptor windows are shorter), and the count of
  dropped half-hours is reported. The page says "rows with no labelled event" where it means the
  rows without a labelled load-shift event.
- No fits, so no refit cadence and no seeds beyond those stored. Bootstrap intervals over the 5
  stored seeds and 4 folds rest on very few cells, and the page says so. The 2025-12 weights of
  experiment 510 are not used.
- Tail calibration, curtailment cost, and load-shift plots do not apply to a diagnostic of stored
  weights, and the page says that.
- July 2026 is unused. Titles say "load-shift", never "switching". No metered generator appears.
- The manager is messaged when the plan PR opens, when the code review is done, when results exist,
  and when the PR merges.
