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
status: done
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
  (27% at D13, 35% at D17) with a correlation near 0.7. The variant that adds the node's own three
  origin lags scores within 10% above N1's error but does not beat it. P4. If stage 1 runs, G1 minus
  A0 at the
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
verdict: not supported
best_arm: >-
  No neighbour arm beat the own-lags-only control. Stage 0 stopped the experiment, so no network was
  fitted.
headline: >-
  At the signed-MW six the interconnected graph's mean lagged residual correlation is between -0.08
  and +0.04 in all 5 test months, against -0.02 to +0.06 for random graphs with the same degrees
  (0 of 5 months pass the gate). Adding true neighbours' lag residuals lowers the held-out squared
  error of the day-ahead residual by 0.01% (95% interval -3.05 to +2.61) on all rows and raises it
  by 3.06% (1.91% to 4.40%) on rows outside labelled events.
take_home_message: >-
  In these tests, the recent forecast errors of a Primary's interconnected neighbours predicted
  none of its day-ahead forecast error, and adding them to a simple regression made it slightly
  worse on ordinary half-hours. A graph network over the Primaries therefore has no lagged-error
  signal to pass along, and none was fitted. A scaled sum of the forecasts of the Primaries beneath
  a bulk supply point scored 34% to 60% higher error than the network that sees the node's own
  readings, at two bulk supply points. Whether neighbours help inside load-shift events was not
  measured here, and experiment 404 tests it.
lessons:
  - >-
    The lagged residuals of a Primary's neighbours carry no information about its day-ahead
    residual in these tests (stage 0a), so a graph network has no lagged-error signal to pass.
  - >-
    The neighbour regression gains nothing on all rows and raises the error by 3.1% outside
    labelled events, so no signal was found on ordinary rows. Whether neighbours help inside events
    was not measured (stage 0b).
  - >-
    The modelled Primaries beneath a bulk supply point sum to 128% (D13) and 171% (D17) of its mean
    demand, because the node's net flow includes embedded generation, so a sum of their forecasts is
    the wrong size and only correlates 0.7 with the node (stage 0c).
advice:
  - >-
    Do not fit a graph network that passes lagged forecast errors between the 16 Primaries: the
    interconnected graph's lagged residual correlation is no larger than a random graph's (stage
    0a, figure 1).
  - >-
    Do not build a bulk supply point forecast from a scaled sum of the forecasts of the Primaries
    beneath it; fit the node on its own readings, as in experiment 415 (stage 0c, figure 3).
  - >-
    Test neighbour information inside load-shift events through experiment 404's B1-nn and B1-rw
    arms, because plain lag residuals showed no signal on ordinary rows (stage 0b).
production_status: not-for-production
adopted_in: none
date: 2026-10-31
---

**Take-home message: in these tests, the recent forecast errors of a Primary's interconnected
neighbours predicted none of its day-ahead forecast error.** A graph network over the Primaries
therefore has no lagged-error signal to pass along, and none was fitted. A scaled sum of the
forecasts of the Primaries beneath a bulk supply point also scored worse than a network that sees
the node's own readings.

We asked whether three ideas for using neighbouring substations are worth a neural-network fit. The
ideas are a graph network over the 16 Primaries, a forecast for a bulk supply point (BSP) built from
the Primaries beneath it, and spotting outliers and load-shift events by comparing neighbours. A
Primary is a primary substation of the distribution network, and a neighbour is a Primary that
NGED's interconnection file joins to it (25 pairs among the 16 Primaries). The first stage fitted
no network. It took the stored day-ahead forecasts of the pooled neural network of [experiment
397](../00397-plain-and-attention-mixture-density-network-for-primary-demand/result.md) and asked
whether a neighbour's recent forecast error predicts a Primary's own error. It did not. The second
stage would have fitted a graph network at 75 fits, and it did not run, so the verdict is `not
supported`. Experiment 404 (plan in draft PR 58) has the arms that test neighbour inputs inside
load-shift events (idea 3).

![Figure 1: lagged residual correlation by graph](figure_1_lagged_correlation.svg)

![Figure 2: squared-error change from neighbour lags](figure_2_neighbour_lag_ridge.svg)

![Figure 3: bottom-up bulk supply point forecast](figure_3_bottom_up_node.svg)

## Results

CRPS is not used here, because stage 0 fits no network. The residual is the actual minus the median
forecast, divided by the Primary's effective capacity. The signed-MW six are the Primaries D5, D9,
D10, D11, D14, and D16, whose meters record the sign of the flow. The MVA-only eight (D2, D3, D4,
D8, D12, D19, D21, and D22) record only its magnitude, and D15 and D18 are report-only. D13 and D17
are bulk supply points, and D1 and D6 are grid supply points. Stage 0a and 0b use the five test
months of experiment 397 (2025-10, 2025-12, 2026-01, 2026-03, and 2026-06), and 0c uses the four
months for which the node network of [experiment
415](../00415-pooled-mixture-density-network-at-bulk-and-grid-supply-points/result.md) stored
forecasts (2025-12 is absent there). Intervals resample the test months (2,000 draws), so they are
coarse. "Reduction" below is the percentage fall in held-out squared error, so a negative reduction
is a rise in error.

- **Gate 0a (lagged correlation, signed-MW six, lead 12 to 36 hours).** The gate required the
  interconnected graph's mean correlation to exceed random graphs' by 0.10 and shortest-edge random
  graphs' by 0.05, in 3 of 5 months. The graph's mean correlation, over the newest reading before
  the origin and the same time yesterday, is 0.001, -0.002, -0.078, 0.036, and -0.001 in the five
  months. Random graphs with the same degrees give -0.016, 0.008, -0.016, 0.055, and -0.001. The
  excess over random graphs is +0.017, -0.010, -0.062, -0.018, and 0.000, so 0 of 5 months pass.
  At leads 120 to 240 hours the excess lies between -0.031 and +0.011. At the MVA-only eight the
  graph's means (0.008 to 0.045) are within 0.06 of random graphs' (not gated).
- **Gate 0b (ridge on neighbour lags, signed-MW six).** The gate required a reduction of at least
  2% against own lags only and at least 1% against each control, on all rows and outside labelled
  events. On all rows the reduction is 0.01% (-3.05 to +2.61) against own lags only, 1.28% (-0.68
  to +3.67) against random neighbours, and 1.46% (-0.75 to +3.79) against shortest-edge random
  neighbours. Outside labelled events the three reductions are -3.06% (-4.40 to -1.91), -0.87%
  (-1.75 to +0.41), and -0.69% (-1.61 to +0.65), so the neighbour features raise the error on
  ordinary rows. The gate failed. At the MVA-only eight the all-rows reductions are -0.33% (-1.15 to
  +0.57) against own lags, +0.38% (+0.08 to +0.71) against random neighbours, and +0.43% (+0.12 to
  +0.77) against shortest-edge neighbours; the last two are positive but below the 1% gate. Outside
  events the eight's three reductions are -0.39%, -0.09%, and -0.06%. D15 and D18 are within 0.3%
  of zero.
- **0c (bottom-up bulk supply point, median only).** The modelled Primaries beneath D13 and D17 sum
  to 128% and 171% of the node's mean demand (the node's net flow includes embedded generation,
  which this experiment did not test), with a half-hourly correlation of 0.73 and 0.71. At D6 the
  sum is 20% of the node's mean and the correlation is 0.18, and at D1 the sum is 11 times the
  node's mean and the correlation is -0.23. The 27% and 35% in the registered prediction came from
  a preliminary look that scaled the cleaned readings by capacity twice, and its rationale was
  wrong; the 0.7 correlation was right. The mean absolute error of the median (points of node
  capacity, average of 4 months) is 8.32 (D13), 12.45 (D17), 27.03 (D1), and 25.43 (D6) for the
  network of experiment 415; 9.02, 13.30, 27.80, and 25.56 for the monthly XGBoost control; 13.35,
  18.33, 38.47, and 31.54 for the bottom-up median; and 11.39, 16.64, 35.08, and 29.42 for the
  bottom-up median plus the node's own three lags. The bottom-up medians are worse than the network
  in every month at D13, D17, and D1, and in 3 of 4 months at D6. They beat the XGBoost control in
  2 of the 16 node-months (D13 and D17 in 2026-01).
- **Idea 3.** The label table names one Primary in 31 of its 112 rows, so a neighbour comparison
  cannot see the other 81 events directly. This experiment did not measure whether neighbours help
  inside load-shift events. On ordinary rows the neighbour regression found no gain.
- **Predictions.** P1 held: the interconnected graph's excess over random graphs was under 0.10
  (and under 0.05 against shortest-edge graphs) at lead 12 to 36 hours. P2's first clause held: the
  reduction against own lags (0.01%) was under 2%. P2's second clause failed on all rows (1.28%
  and 1.46% against the two controls, above 1%) and held outside events (a loss). P3 held in its
  first clause (the first bottom-up variant was 60% and 47% above N1 at D13 and D17) and failed in
  its second: the variant with the node's own lags was 37% and 34% above N1, not within 10%. P4
  was not tested.

## Checks

- **Median minus mean forecast.** Not computed, because the stored forecasts hold quantiles only.
- **Mean-forecast error.** Not applicable: stage 0 scores the median residual and its squared error.
- **Triggers fired.** One: the second clause of P3 failed. A likely reason is that the variant was
  fitted on the children's actual sums and applied to their forecast sums, which makes the child
  term look more reliable than it is. The reason was not tested. No other trigger fired, and no audit
  was run.
- **Cleaned data.** Forecast rows with no cleaned actual, per fold (2025-10, 2025-12, 2026-01,
  2026-03, 2026-06): 1,520, 719, 0, 248, and 1,349 of about 348,000, of which 1,520, 629, 0, 28,
  and 264 are D15 and D18 rows before 2025-03-01 or other missing readings. Target rows (lead 12 to
  36 hours) lacking one of the two own lag residuals, because the lag falls in a month with no stored
  forecast: 1,337, 1,314, 496, 1,267, and 1,304 of about 23,700.

## Setup and caveats

- **Design.** Lagged residuals come from the stored forecast with the shortest lead (under 24 hours)
  for the lag time. Controls are 1,000 degree-preserving random graphs (swap chain, 10 discarded
  draws, 100 swap attempts between draws) and 1,000 shortest-edge random graphs, chosen from 20,000
  draws as the closest in mean edge length. The shortest-edge graphs still have edges up to 1.77
  times as long as the interconnected graph's (random graphs: 2.05 times), so the distance control
  is only partial. Gate 0b used 30 evenly spaced draws of each control and a ridge penalty chosen by
  leave-one-month-out inside the four training months. The ridge is fitted on four test months and
  scored on the fifth, so it may use later months to predict an earlier one. Event rows are
  excluded when the target or a lag time falls inside a labelled event that names the Primary.
- **0c fitting.** The scale, offset, and ridge of the bottom-up variants are fitted on actual
  readings before the scored month, because no earlier child forecasts exist.
- **Code.** `ml-experiments/00426-*/e00426_stage0.py` and `e00426_figures.py`. Tables are in this
  experiment's private data folder, `stage0/`. A fresh Sonnet agent reviewed the code before the run
  and found three blockers (a unit mismatch, an integer-type mismatch in a join, and an unmixed
  graph sampler), all fixed before the run.
- **Not covered.** No network was fitted, so there are no CRPS, tail, curtailment, or example
  forecast results. Only lagged residuals of one median forecast, a linear ridge, five test months,
  and the 16 Primaries were tested, so neighbour inputs of other kinds and nonlinear models remain
  untested.

## What to try next

- Read experiment 404's B1 minus B1-nn and B1 minus B1-rw if its batch 2 runs: they test neighbour
  information inside injected load-shift events, the one place the signal appeared.
- A regional weather-error input (not a graph) is the other way a neighbour's recent error could
  help, and stage 0a gives it no support at the day-ahead lead.

## Problems caused by the move

None.
