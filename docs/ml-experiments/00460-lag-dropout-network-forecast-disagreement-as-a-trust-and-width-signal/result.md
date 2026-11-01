---
id: "00460"
slug: lag-dropout-network-forecast-disagreement-as-a-trust-and-width-signal
title: >-
  When forecasts made from each power lag alone disagree, are the lags wrong, and can the
  disagreement set the width of the bands?
theme: lagged-power-and-error-correction
also_relevant_to:
  - calibration-and-uncertainty
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: 15 saved neural networks, five-pass forecast tables, and arm forecast tables in
  this experiment's private folder; every earlier experiment is read and never written. More than
  one design would defensibly test the hypothesis (how the spread is defined, how it maps to a
  band factor or a mixture weight), so the plan fixes each choice before any score is read.
  Production serving path: not touched (this repository holds none). Degradation rule: not
  touched. Callers: the scripts import the folders of experiments 404 and 435 and the `studies`
  evaluation code.
hypothesis: >-
  When one neural network, trained with random dropout of its three power lags, gives a different
  forecast median with all lags, with each lag alone, and with no lags, the rows where these
  forecasts disagree are the rows where the lags are wrong (rows whose target lies outside every
  labelled load-shift event but whose lags lie inside one). The amount of disagreement can therefore
  widen the bands, and can tilt a mixture of the all-lags forecast and the no-lags forecast toward
  the no-lags forecast, better than a fixed 50/50 mixture of a lagged and a lagless network
  (B-mix).
prediction: >-
  Written before any score is read. P1. The area under the curve (AUC) of the mean spread for
  separating real harm rows from rows with no labelled event is between 0.52 and 0.62 at the
  signed-MW six, so below the 0.65 bar, because experiment 445 found the lagless head about as good
  as the lagged head where lags are corrupted and experiment 401 found real events move the
  forecast by 0.03 of the labelled step or less. P2. On injected harm rows, where the corruption is
  large and exact, the same AUC is above 0.75 (the known-answer control). P3. The lag-dropout
  network (arm b) loses 0.00 to 0.05 points of capacity against the stored S1 network (arm a) on
  rows with no labelled event at the six. P4. Arm c (widened by spread) beats arm f (S1 with a
  constant 1.15 widening) on q99 coverage inside real events by under 0.03. P5. Arm d (mixed by
  spread) trails B-mix on rows with no labelled event at the six by 0.03 or more, because B-mix's
  gain is mainly the diversity of two separately trained networks (experiment 445, claim E).
decision_rule: >-
  Score = mass-weighted pinball loss (CRPS-like) in points of capacity on experiment 404's rows,
  seeds averaged within a month, then a mean over scored months, each contrast taken on identical
  rows. Rows with no labelled event are called that, never "clean", because NGED's event labels are
  incomplete; the rows the label-free step scan finds step-free are added as a second set once that
  scan has reported. The signed-MW six and the MVA-only eight are scored separately and never
  pooled; D15 and D18 are report-only. Pre-registered. (1) The spread signal is useful if its AUC
  for separating real harm rows from rows with no labelled event is above 0.65 with a month-block
  bootstrap 95% interval whose lower bound is above 0.6, at the six (the MVA-only eight is
  reported, with the same bar, and never pooled with the six). The signal is the mean spread (the
  mean over the four other passes of the absolute difference in forecast median from the all-lags
  pass, divided by the all-lags pass's q95 minus q05); the maximum over the ten pairs of passes,
  divided by the same width, is a secondary signal and does not decide. (2) Arm c or arm d is a
  candidate if, against B-mix, it is no worse than +0.03 points on rows with no labelled event, its
  q99 coverage inside real events is 0.98 or above, and its q99 pinball loss on rows with no
  labelled event is no more than +0.02 worse, each at the six. (3) Stop and close the idea if the
  lower bound of the AUC interval is not above 0.5, or if the lag-dropout network (arm b) loses more
  than 0.03 points against S1 (arm a) on rows with no labelled event at the six. Arms c and d are
  fitted on earlier months only and never see event labels (their fit rows are all real rows of
  earlier months), so 2025-10 has no arm c or arm d and their contrasts use four months; every
  other arm is reported on both the four months and all five. The smallest gain worth porting is
  0.03 points, because the serving cost is five passes of one network against B-mix's two networks
  per forecast. Intervals: harm rows and rows inside events carry whole-event bootstrap intervals;
  rows with no labelled event carry month-block bootstrap intervals with the sign count over
  months beside them. The Bonferroni count is 42 (7 contrasts, 2 groups, 3 row classes). The
  minimum detectable effect is read from the seed-to-seed spread of B-mix before any contrast is
  read. Leave-one-month-out and drop-largest-event intervals are given; leave-one-Primary-out is
  infeasible (21 events at the six). A candidate (rule 2) is only considered if rule 1 holds, and must also beat controls f, g, and c0 (contrasts C4 to C6 below, interval excluding zero). Rule 2 sets a non-inferiority bar; a gain of 0.03 or more against B-mix on rows with no labelled event is what would justify porting. The q99 coverage in rule 2 is on NRA-corrected actuals over all real events; metered actuals and the drop-largest-event version are secondary. The AUC is computed per seed on the real rows of all scored months pooled, averaged over the three seeds; its interval resamples months, and an event-cluster interval is reported beside it because five blocks give a coarse interval. The contrasts are C1 b minus a, C2 c minus e, C3 d minus e, C4 c minus f, C5 d minus g, C6 c minus c0, C7 g minus e. AUC tests and the later step-scan row set are not counted in the Bonferroni count. Interim; no claim is made at the 99% level.
benchmark: >-
  B-mix, experiment 404's fixed 50/50 mixture of the A0-inj network and the frozen cross-fitted
  lagless table, recomputed for fit seeds 0 to 2 and checked against the stored seed 0 forecasts.
  Reference rows: the stored S1 network (arm a), S1 with a constant 1.15 widening (arm f), and a
  three-seed S1 ensemble (arm g).
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of capacity from 11 quantiles, by row class and
  group, as in experiment 404's report; tail coverage and pinball loss at q01, q05, q95, and q99;
  AUC of the spread signal.
---

## Overview

Planned; the write-up follows the results.

## Plan details

- **Harness.** Experiment 404's: its rows, its five months (2025-10, 2025-12, 2026-01, 2026-03,
  2026-06), six epochs of Adam on stride-2 rows, training on the injected worlds of seeds 100 to
  102 (one per fit seed 0 to 2), the frozen cross-fitted lagless tables, and the same fit seeds as
  experiments 404 and 454. Rows are equal to experiment 454's. A fit takes about 20 seconds on the
  GPU, so 15 fits (3 seeds times 5 months) cost minutes of GPU; the first fit is timed and
  reported before the rest.
- **Network.** S1 (the plain network with three origin lags and the surprise at each lag) trained
  with random lag dropout: for each training row each of the three lags is masked independently
  with probability 0.3. A masked lag has its value, age, and surprise set to 0 and its missing flag
  set to 1; the missing flag is the mask flag, so a lag that is really missing and a lag that is
  masked look the same to the network. Experiment 404's `PlainLagMdn` and `fit_arm` are reused,
  but its `drop_lags` cannot handle the 12 columns of S1, so a subclass in this folder has new
  masking code (`mask_lags`). Training and the five passes call the same function, so the
  single-lag and no-lags passes are not off-distribution.
- **Passes.** The saved network is scored five times per row: all lags, each single lag alone
  (the other two masked), and no lags. One network, five passes per serving call.
- **Arms.** (a) the stored S1 network; (b) the lag-dropout network with all lags; (c) arm b with
  its tail quantiles widened by a factor that is a monotone function of the mean spread; (d) arm b
  mixed with its no-lags pass, with a mixture weight that is a monotone function of the mean
  spread; (e) B-mix; (f) arm a with a constant 1.15 widening; (g) the three-seed S1 ensemble, an
  equal-weight density mixture of the S1 networks of fit seeds 0 to 2. Control c0: arm b after a
  table P (experiment 435's tail recalibration) with no spread, so a gain from recalibration alone
  is not read as a spread signal.
- **How arms c and d are fitted.** On the real rows of earlier months only, per seed, per group,
  with leads 0 to 240 hours. Arm c: four spread bins whose edges are the quartiles of the fitting
  rows' spread per group; within each bin and lead band, table P's per-tail factors (the
  quantile of the standardised miss, as in experiment 435); the factors are then made
  non-decreasing in the bin with a running maximum. Arm d: the same four bins per group; within
  each bin, the weight on the no-lags pass from the grid 0, 0.1, ..., 1 that minimises the fitting
  rows' mean CRPS (a subsample of up to 60,000 rows per group per earlier month), made
  non-decreasing in the bin with a running maximum. Nothing in the fit uses an event label. The fitting rows of each earlier month are that month's
  own out-of-sample forecasts (made by the network trained before that month), so the fit is not
  on rows the scoring network trained on; the earlier months cover other seasons, which limits
  the fit for the summer month. Arm d is a density mixture (log weights shifted by log of the
  weight, as in B-mix), and its fitting CRPS is computed on the mixture's quantiles. Control c0
  uses the same procedure and rows as arm c with one bin, not experiment 435's table.
- **Confounds of the spread.** Lag age grows with lead, and the no-lags pass is seen in only 2.7%
  of training rows. The AUC is therefore also reported within each lead band (mean of the band
  AUCs) and beside a lead-only score (the lead in hours as the signal), and the 12 to 36 hour
  band's CRPS is reported separately.
- **Row classes.** Experiment 404's `report.load` classes with the window flags of experiment 428:
  a harm row has its target outside every padded real event (and outside the 96 half-hours after
  one) and an observed half-hour of its three day-windows inside a padded event; rows inside real
  events have the target inside a padded event or in the 96 half-hours after one; rows with no
  labelled event have neither. The injected scoring worlds (seeds 200 to 202) are used only for the
  AUC known-answer control (P2).
- **Actuals.** Inside real events the headline actual is experiment 435's NRA-corrected value
  (normal running arrangement), the metered value secondary, each with and without the largest
  event of each month; months with a single event are shown apart.
- **Tails and cost.** q01, q05, q95, q99 coverage and pinball loss, raw and after table P, per
  group and lead band, and the 12 to 36 hour band separately. The avoidable-curtailment cost is
  not computed: it needs forecasts for each grid supply node and its generators, and this
  harness forecasts the 16 Primaries only (the same reason experiment 445 gives).
- **Absolute scores** are printed beside every contrast. Serving cost: five passes of one network
  per forecast against B-mix's two networks.
- **Standing practices followed.** Monthly refits (not a cadence question, so no weekly gate),
  cleaned data with the flagged `substation_zero` half-hours dropped by experiment 404's rows, the
  D15 and D18 cut at 2025-03-01 (report-only), every fitted network saved under this experiment's
  data folder, stored forecasts reused as references, a fresh Sonnet code review before the first
  scored number, the five-month screen of experiment 404 rather than four folds (so rows equal
  experiments 404 and 454), tails reported beside CRPS, draft pull request.
- **Unlogged-event caveat.** NGED says load-shift events may be missing from its logs. Rows with no
  labelled event may therefore hold events no label covers, the harm table is optimistic, and the
  real-event rows undercount.
- **Earlier advice followed.** Experiment 445: score against both metered and NRA-corrected
  actuals. Experiment 404: report single-event months apart, and weight by event. Departure: the
  curtailment cost is not computed (above).
