---
id: '00443'
slug: solar-network-hyperparameter-and-architecture-sweep
title: >-
  Does any change to the size, depth, farm embedding, learning rate, weight decay, training length,
  mixture components, dropout, or recency weighting of the pooled solar network beat the plain
  network by 0.5%?
theme: metered-solar-farms
also_relevant_to:
- generator-forecasting
- production-reference-and-best-guess
status: planned
date: 2026-10-31
kind: change
size: >-
  medium. Stored: fitted-network quantile tables in this experiment's private folder only. Serving
  path: not touched (this repository holds none). Degradation rule: not touched. Designs: one
  skeleton, one factor changed at a time, so only the choice of factor levels is open, and it is
  fixed in this plan. Callers: this folder's scripts, which import experiment 310's and experiment
  430's modules and read experiment 430's stored forecasts. Asset graph: not touched. Sonnet writes
  and reviews; no Opus.
hypothesis: >-
  The plain pooled mixture-density network for the six metered solar farms (experiment 430's
  `g0_onepass`: two hidden layers of 64, three Gaussians, no farm embedding, learning rate 2e-3,
  weight decay 1e-4, 20 epochs, no dropout, equal row weights) sits near a flat optimum, so no
  single change of its hidden width, depth, farm-embedding width, learning rate, weight decay,
  number of epochs, number of mixture components, dropout, or row-recency weight lowers the quantile
  score by the margin.
prediction: >-
  Written before any fit and before any score is read. Scores are quantile scores in points of
  capacity (twice the mean pinball loss over 11 levels; lower is better), farm-equal means over
  experiment 430's scored year, all day bands. Most levels change the score by less than 0.5%.
  Because 25 levels are tested, one or two may pass by chance at the 99% level, and the
  confirmation on fresh seeds then removes them. The levels I think most likely to help are a wider
  network (128) and a small farm embedding (4 or 8), each by 0.3% to 0.8%; the levels most likely
  to hurt are one hidden layer, one mixture component, and 10 epochs. The prediction that could
  fail: at least one level beats the skeleton by 0.5% on three seeds and again on three fresh seeds.
  If none does, the experiment stops after the one-factor stage.
decision_rule: >-
  Contrast = percentage change of the farm-equal mean quantile score against the skeleton (stored
  `g0_onepass`, same seeds 0 to 2, same rows), month-block resampling of the 12 scored months, 10,000
  draws, 99% interval. The margin is the larger of 0.5% and the seed minimum detectable effect; the
  minimum detectable effect is 0.40% at three seeds and 0.28% at six (from the six stored
  `g0_onepass` seeds, 1.96 x sqrt(2) x sd / sqrt(n)), so the margin is 0.5%. A level passes when its
  point estimate is at least 0.5% better, its 99% interval excludes zero on the favourable side,
  and its 0.01 and 0.99 pinball losses are each no more than 0.02 points worse than the skeleton's.
  A factor moved if any of its levels passes. If no factor moved, the experiment stops, says so, and
  runs no random search. Otherwise the random search samples 45 configurations from the levels of
  the moved factors (plus the skeleton level of each), two seeds each; the best three by mean score
  are refitted on seeds 3 to 5 against the skeleton (`g0_onepass` seeds 3 to 5, stored), and a
  configuration is reported as beating the skeleton only if it passes the rule above on seeds 3 to 5
  alone. No arm is adopted by this experiment.
benchmark: >-
  Experiment 430's `g0_onepass` (the skeleton; seeds 0 to 5 stored) as the reference of every
  contrast; experiment 310's blend (Metered solar farms research default 1) as the absolute
  reference.
row_set_id: >-
  Experiment 430's rows: farms A to F, scored year 2025-09 to 2026-08, day bands d1, d2, d3, d4_5,
  d6_7, d8_10, quarterly refits, rows that every compared arm holds with finite quantiles.
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), pinball loss
  and coverage at 0.01, 0.05, 0.95, and 0.99, median minus mean forecast, seed spread
code: ml-experiments/00443-solar-network-hyperparameter-and-architecture-sweep/
---

**This plan tests whether any change to the plain solar network's settings is worth making.**
Experiment 430 found that nothing added to the plain pooled network beats it by 1%, so that network
is the skeleton. Here we leave its design alone and change its settings one at a time. The six
metered solar farms are labelled A to F. All numbers on this page are interim and unaudited until the
experiment finishes.

## Design

**The skeleton is experiment 430's `g0_onepass`.** It is a mixture-density network: it reads the
standardised inputs (calendar, sun position, ensemble mean and spread of shortwave radiation, and
the other columns of experiment 345's list), has two hidden layers of 64, and outputs a mixture of
three Gaussians for the farm's output as a fraction of its ceiling. It is trained with AdamW at
learning rate 2e-3, weight decay 1e-4, batch 2048, for 20 epochs with the best epoch chosen on the
last 61 days of the training rows. One network is pooled over the six farms and the six day bands and
refitted quarterly, four refits from 2025-09. A job is one (arm, seed, refit); the stored skeleton
supplies seeds 0 to 5.

**Stage 1 changes one factor at a time (25 arms, 3 seeds, 4 refits: 300 jobs).**

| Factor | Skeleton level | Other levels |
|---|---|---|
| Hidden width | 64 | 32, 128, 256 |
| Depth (hidden layers) | 2 | 1, 3, 4 |
| Farm embedding width | none | 4, 8, 16, and per-farm networks (six networks, each trained on one farm only) |
| Learning rate | 2e-3 | 5e-4, 1e-3, 4e-3 |
| Weight decay | 1e-4 | 0, 1e-3, 1e-2 |
| Epochs | 20 | 10, 40 |
| Mixture components | 3 | 1, 2, 5 |
| Dropout (after each hidden layer) | 0 | 0.1, 0.2 |
| Row-recency weight (half-life of the loss weight, by the row's age at the refit) | none | 0.5 year, 1 year |

The brief also named "the loss weights that the design note lists". The note lists one loss weight
for the demand networks, the row-recency half-life (0.5 year; 1 year and 0.25 year were no better),
so that is the loss-weight factor. It lists no tail or pinball weight, so none is tested.

**Stage 2 (only if a factor moved) is a random search.** 45 configurations are drawn at random from
the levels of the moved factors (each other factor stays at its skeleton level), two seeds each (360
jobs). The best three by mean score are refitted on seeds 3 to 5 and judged against the skeleton's
seeds 3 to 5.

**Compute.** Stage 1 is 300 jobs and stage 2 is about 400 jobs, at about 14 s per job on the GPU for
the skeleton (wider and deeper networks cost more). One GPU process, under 8 GB, `MemoryMax=8G`,
`OMP_NUM_THREADS=1`, one feeder core. Estimated GPU time: about 1.5 hours for stage 1 and about 2
hours for stage 2.

**Standing practices.** Monthly refitting does not apply to a farm network, which follows
experiment 345's quarterly refits so that the rows match. Cleaning and the D15/D18 start cut apply
to demand only. The farm rows are experiment 310's, with its idle-band handling. The fitted networks
are not saved: a job's output is its quantile table, and the sweep's 700 networks are small enough
to refit in minutes. Stored predictions reused: experiment 430's `g0_onepass` seeds 0 to 5, so the
skeleton is not refitted; the gate (G0) refits one job to prove the new code reproduces it.

**Gate G0.** The new code with every factor at its skeleton level must reproduce the stored
`g0_onepass` forecasts for seed 0 and all four refits to within 1e-4 points of capacity. If it
misses, the run stops and reports.

## Checks the write-up will carry

- **Tails:** coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99 for the skeleton and each
  passing level.
- **Median minus mean forecast, per farm,** and the error of the mean forecast beside each
  quantile-score contrast.
- **Audit triggers:** the seven triggers of the `ml-experiment` skill, run on the results.
- **Example forecasts:** a day-ahead forecast and a 10-day forecast of farm A, anonymised.
