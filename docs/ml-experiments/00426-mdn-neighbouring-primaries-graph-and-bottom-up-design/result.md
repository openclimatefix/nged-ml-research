---
id: "00426"
slug: mdn-neighbouring-primaries-graph-and-bottom-up-design
title: >-
  Is a graph network over the Primaries, a bulk supply point forecast built from the Primaries
  beneath it, or neighbour-based outlier spotting worth a fit, compared with simple neighbour
  statistics? (design note)
theme: hierarchy-and-pooling
also_relevant_to:
  - lagged-power-and-error-correction
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: one small table of residual correlations and one of ridge scores per stage in this
  experiment's private folder; and, if stage 1 runs, network weights and per-arm forecast tables
  there. Experiments 397, 403, 415, and 87 are read and never written. More than one design would
  test the question (how neighbours enter the network, which graph, which lags), so this plan fixes
  one before any score is read. Callers: the scripts import the folders of experiments 87, 397, and
  415. Production serving path: not touched (this repository holds none). Degradation rule: not
  touched.
hypothesis: >-
  Information from the Primaries connected to a Primary improves the pooled plain mixture density
  network's forecast at days 1 to 2, the only leads at which recent power helps the signed-MW six,
  by more than simple neighbour statistics do. Residuals of the stored forecast of interconnected
  Primaries correlate more strongly than residuals of Primaries joined at random or by distance
  alone, and a network that passes messages along the interconnection graph beats both its own
  rewired-graph control and the simple-statistics control.
prediction: >-
  Written before any score is read. The prior is low: experiment 415 (the Primaries beneath a node
  as inputs: N2 minus N2rw), experiment 391 (neighbour surprise columns in XGBoost), experiment
  402 (neighbour features in a lag gate), and experiment 117 (a graph prior on pooled parameters)
  each found nothing or a loss. P1. At lead 12 to 36 hours the mean residual correlation of the 25
  interconnected pairs among the 16 Primaries exceeds that of degree-matched random graphs by less
  than 0.10 and that of degree-matched nearest-by-distance graphs by less than 0.05 (so stage 1
  does not run). P2. A ridge regression of a Primary's day-ahead residual on its own lagged residual
  and the lagged residuals of its true neighbours lowers the held-out squared error by under 2%
  against the own-lag-only ridge, and by under 1% against the rewired control. P3. A bulk supply
  point's forecast built from the stored median forecasts of the 7 or 9 modelled Primaries beneath
  it (one fitted scale and offset) scores a mean absolute error at least 20% above experiment 415's
  N1, because the modelled Primaries carry about one third of a bulk supply point's mean demand
  (27% at D13, 35% at D17) with a correlation near 0.7. P4. If stage 1 runs, G1 minus A0 at the
  signed-MW six is between -0.05 and +0.05 CRPS points (days 1 to 10), G1 minus G1-rw is between
  -0.03 and +0.03, and G1 minus S1 is between -0.03 and +0.03.
decision_rule: >-
  See "Stop rule and decision rule", registered before any score is read.
benchmark: >-
  Experiment 397's stored forecasts of the pooled plain network (`plain_pooled`, 3 seeds, the five
  folds), which are arm A0 here, and experiment 415's N1 and the stored monthly XGBoost control at
  the four nodes. Hand-built neighbour statistics (arm S1) and the rewired graphs are the
  controls.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles, in points of effective
  capacity, from `studies.eval_protocol`; leads 0.5 to 336 hours; headline band days 1 to 10 and
  days 1 to 2; q01, q05, q95, and q99 coverage and pinball loss beside it.
code: ml-experiments/00426-mdn-neighbouring-primaries-graph-and-bottom-up-design/
date: 2026-10-31
---

**Take-home message to be written when the experiment finishes.** This page is the design note that
the weekend programme asks for. It decides which of the maintainer's three neighbour ideas deserves
a fit, and the answer is at most one, after a stage that fits nothing.

## The three ideas, and what the earlier experiments already say

We name three ideas from the weekend programme and judge each against what experiments 117, 195,
391, 392, 402, 404, and 415 already found. A Primary is one of the 16 primary substations of
`demand-v2`. A bulk supply point (BSP) connects the 132 kV to the 33 kV electricity network. A grid
supply point (GSP) connects the transmission network to the 132 kV network. The four nodes are D13 and D17 (bulk supply points) and
D1 and D6 (grid supply points) of [experiment
415](../00415-pooled-mixture-density-network-at-bulk-and-grid-supply-points/result.md).

- **Idea 1, a graph neural network over the Primaries.** NGED's interconnection file joins 25
  pairs among the 16 modelled Primaries. The graph is small and sparse: 15 Primaries have between 1
  and 5 neighbours, and 1 has none. [Campagne et al.
  (2026)](https://doi.org/10.48550/arXiv.2610.35074) separate the gain from sharing parameters
  from the gain from graph edges, and [experiment
  403](../00403-pooled-versus-per-primary-mixture-density-network/result.md) already measured the
  sharing gain. Three earlier results suggest the edges add little. [Experiment
  117](../00117-differentiable-physics-h2-graph-prior-on-pooled-convex-parameters/result.md) found
  that tying the parameters of interconnected substations worsened the score by 0.020 points.
  [Experiment 391](../00391-xgboost-lag-surprise-second-stage/result.md) found that neighbour
  surprise columns raised the XGBoost error on rows whose lags sit inside an event (+0.314,
  interval +0.062 to +0.749). [Experiment
  402](../00402-lag-gate-learned-weight-injected-load-shifts-part-1/result.md) found that the
  window means, the median gap, and the neighbour features add little at the signed-MW six and
  nothing at the MVA-only eight. The edge control of experiment 404 (B1-rw and B1-nn) is a
  batch-2 arm that runs only if batch 1 passes its stop rule, which its plan expects not to
  happen.
- **Idea 2, a bulk supply point forecast from the Primaries beneath it.** A preliminary look
  at the cleaned panel from 2025-03-01 (the script will be committed with stage 0) shows that the
  7 modelled Primaries beneath D13 and the 9 beneath D17 carry 27% and 35% of the bulk supply
  point's mean demand, with a half-hourly correlation of 0.71 and 0.69 with its demand. At the
  grid supply points D1 and D6 the correlation is -0.21 and +0.15. Experiment 415's N2 arm, which
  fed the beneath-Primaries' lags to the network, scored the same as N1 at the bulk supply points
  (+0.01) and worse at the grid supply points (+0.32), and a random set of the same size scored the
  same as the true set. The unmodelled children, embedded generation, and storage dominate the
  node, so a sum of the modelled Primaries' forecasts cannot be the forecast.
- **Idea 3, spotting outliers and load-shift events by comparing neighbours.** Three
  detectors already compare a substation with its neighbours: the refit-offset neighbour-count
  detector (RO-NC), [experiment
  195](../00195-switching-neural-plus-detector-features/result.md)'s learned score (22 to 24 of 35
  real transfers found, against 18 or 19 for RO-NC), and the neighbour veto of the "deep and long"
  detector in experiment 392. 30% of the 112 labelled events name no recipient Primary (experiment
  402), so a neighbour comparison cannot see them. A separate fit would need injected events as
  training data, which is experiment 404's generator and training scheme. This note therefore does
  not plan a separate fit for idea 3. Stage 1 scores the harm rows and the real-event rows of
  experiment 404 as slices, and 404's B1-nn and B1-rw arms remain the direct test.

## What would make a fit worthwhile

**A neighbour can add information only through its recent residual, so the first stage measures
that residual and fits nothing.** A Primary's weather forecast already reaches the pooled network
through the ensemble mean and spread, so a neighbour's weather adds nothing. A neighbour's recent
power can add something the network cannot see in its own inputs: the weather error that has already
happened nearby. A neighbour close in distance carries the same weather error whether or not an
interconnection joins the two, so the graph must beat a distance-matched graph, not only a random
one. Stage 0 measures both.

## Stage 0: residual structure from stored forecasts (no fits)

- **Inputs.** Experiment 397's stored `plain_pooled` forecasts (`work/seed_preds/summary/plain_pooled`,
  five folds, 3 seeds, 16 Primaries) and experiment 87's cleaned panel for the actuals. The residual
  is the actual minus the q50 forecast, divided by the Primary's effective capacity. D15 and D18 rows
  before 2025-03-01 are dropped. Flagged `substation_zero` half-hours are dropped, and the count is
  reported per fold.
- **0a, correlation by graph.** For each pair of Primaries, the correlation of the residuals at the
  same valid time, at lead 12 to 36 hours and at lead 120 to 240 hours. Three graphs are compared at
  equal degree: the 25 interconnected pairs, 1,000 degree-matched random graphs, and 1,000
  degree-matched nearest-by-distance graphs (distance between the Primaries' coordinates in
  `TimeSeriesMetadata`). The report gives the mean correlation of each graph, and the interconnected
  graph's excess over each control, with an interval from the random draws and from resampling the
  five test months.
- **0b, ceiling of neighbour lags.** For each Primary, a ridge regression of the day-ahead
  residual at the target on the Primary's own residual at the lag time (origin-lag and same-time-
  yesterday) and the mean, minimum, and maximum of its neighbours' residuals at the same lag time.
  The regression is fitted on the other four test months and scored on the held-out month. It is a
  diagnostic ceiling of what neighbour lags carry, not a forecast. Arms: own lags only, true
  neighbours, rewired neighbours, distance-matched neighbours. The score is the held-out squared
  error. Reported at the signed-MW six and the MVA-only eight separately.
- **0c, bottom-up bulk supply point.** The median of each node as a fitted scale and offset times
  the sum of the stored median forecasts of the modelled Primaries beneath it, scored on mean
  absolute error against experiment 415's N1 and the stored monthly XGBoost control at the same
  half-hours. Only the median is scored, because summing quantiles does not give the quantile of a
  sum. The table also reports the share of each node's mean demand that its modelled children
  carry.
- **Cost.** 0 fits. One CPU process on one core, under 8 GB of memory, about 1.5 hours including
  the 2,000 random draws.

## Stage 1 (runs only if stage 0 advances it): does a graph network beat simple statistics?

**Arms.** All arms are the pooled plain mixture density network of experiment 397 (width 64, two
layers, 3 components, dropout 0.1), the ensemble mean and spread as weather inputs, one fit per
monthly refit on the months before the scored month, 3 seeds, the five folds of experiment 397.

- **A0:** the pooled plain network with the three origin lags and no neighbour input. Read from
  experiment 397's stored `plain_pooled` forecasts, after checking that the rows, folds, and
  recipe match (state any gap in the write-up). Not refitted.
- **S1:** A0 plus hand-built neighbour statistics: the mean, minimum, and maximum, across the
  neighbours, of each of the three origin lags (capacity-normalised), and the share of neighbours
  with a reading. This follows experiment 415's N2 and experiment 402's gate features, which are
  the simple neighbour-statistic features that exist.
- **S1-rw:** S1 with a degree-matched random set of Primaries as neighbours.
- **G1:** one round of message passing over the interconnection graph. Each Primary's three origin
  lags are encoded by a shared layer, the neighbours' encodings are averaged along the fixed graph
  (no learned adjacency, because 25 edges cannot identify one), and the average is concatenated
  to the Primary's own encoding before the mixture head. A Primary with no neighbour receives a zero
  message and a missing flag. The ensemble weather of a neighbour is not passed, to keep the test
  on neighbours' power.
- **G1-rw:** G1 on a degree-matched random graph.

The contrast S1 minus A0 tests simple neighbour statistics. G1 minus S1 tests whether learned
message passing beats the simple statistics. G1 minus G1-rw tests whether neighbour identity
matters, and S1 minus S1-rw does the same for the statistics (the two readings of the edge test in
Campagne et al.).

**Slices.** Days 1 to 10 and days 1 to 2 (the leads at which lags matter), the signed-MW six and the
MVA-only eight separately (D15 and D18 report-only), harm rows and real-event rows of experiment
404's definitions with the event count per metering group, rows outside events, the sunny
low-demand half-hours, and the tails (q01, q05, q95, q99 coverage and pinball loss, raw and at the
1.15 widening). The curtailment-cost table uses the studies code, if its inputs exist for the
scored months, and otherwise the write-up says so.

**Fits.** 4 arms (S1, S1-rw, G1, G1-rw) times 5 folds times 3 seeds is 60 fits. No cross-fit table
is needed, because the arms use lags and not surprises. The timed cost is unknown: the estimate is
about 100 seconds for each S fit and about 250 seconds for each G fit (a graph sample holds all 16
Primaries at one origin and lead), so about 0.8 plus 2.1 hours, near 3 hours of GPU time and 4 hours
of wall time with data preparation. The first task after the slot is granted is to time one fit of
each kind and report it. Memory: one process on the GPU under 8 GB, one feeder core, host memory
under 16 GB (`systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`), after
`free -g` shows more than 12 GB available; if the available column falls to 12 GB the job stops.

## Stop rule and decision rule

We register these rules before any score is read.

- **Stage 0 advances to stage 1** if 0a finds the interconnected graph's mean residual correlation
  at lead 12 to 36 hours exceeds the degree-matched random graphs by at least 0.10 **and** the
  degree-matched nearest-by-distance graphs by at least 0.05, in at least 3 of the 5 test months;
  **or** if 0b finds the true neighbours lower the held-out squared error by at least 2% against
  the own-lag-only ridge and by at least 1% against both the rewired and the distance-matched
  controls, at the signed-MW six.
- **Otherwise the experiment stops after stage 0.** The write-up says that no neighbour idea is
  worth a fit, and which of the three stage-0 measurements decided it.
- **Idea 2 is not fitted either way.** Stage 0c settles it: if the bottom-up median beats experiment
  415's N1 on mean absolute error at either bulk supply point, the write-up says so and proposes a
  follow-up; otherwise it records the failure.
- **Stage 1 decision.** G1 counts as worth carrying forward only if all three hold at the
  signed-MW six on days 1 to 10: G1 minus A0 is at most -0.05 CRPS points with a 95% month-block
  interval below zero; G1 minus G1-rw is at most -0.03; and G1 minus S1 is at most -0.03. The
  smallest gain worth porting is 0.05 points, because a graph network adds a stage-one model and a
  stored graph file that production has not got. The minimum detectable effect is 2 times the
  standard deviation, across the 15 fold-and-seed pairs, of each contrast, and a gain below it is
  reported as inconclusive. If S1 minus A0 passes the same threshold but G1 does not, the write-up
  recommends the simple statistics only.
- **Checks.** The median forecast minus the mean forecast is not computed (the networks store
  quantiles only), the mean-forecast error is reported beside each quantile-score contrast, and the
  audit triggers of the skill are run on the results.

## Standing practices

- **Cadence and data.** Monthly refits. Cleaned data everywhere, with the count of dropped
  half-hours per fold. D15 and D18 rows before 2025-03-01 are dropped, in training, lags,
  inference, and scoring.
- **Stored forecasts.** Experiment 397's `plain_pooled` forecasts (A0), experiment 415's N1 forecasts
  and experiment 386's stored monthly XGBoost control (stage 0c) are reused and never refitted.
  Stage 1 saves each fitted network (`.pt`) in this experiment's private data folder, one per arm,
  fold, and seed.
- **Advice followed.** Experiment 415's advice not to add the beneath-Primaries' lags at nodes (C3,
  C4) is why stage 0c is a no-fit measurement. Experiment 391 and experiment 402 are why the simple
  statistics are an arm and not an assumption.
- **Not run in this PR.** No code and no fit. A fresh Sonnet agent reviews this plan first. The
  code is written afterwards and reviewed by a second fresh Sonnet agent before stage 0 runs, and
  the manager agrees the design before any fit slot is requested.
- **Anonymisation.** Demand series only. No metered generator's series appears.

## Problems caused by the move

None so far.
