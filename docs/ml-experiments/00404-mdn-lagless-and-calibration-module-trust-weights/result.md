---
id: "00404"
slug: mdn-lagless-and-calibration-module-trust-weights
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted networks, forecasts, and trust-weight logs in this experiment's private
  folder only; every earlier experiment is read and never written. More than one design would test
  the hypothesis (how per-lag trust weights combine into one mixture weight, the source of the
  surprise inputs, the injection), so each choice is fixed in this plan before any score is read.
  Production serving path: not touched (this repository holds none). Degradation rule: not touched.
  Callers: the scripts copy code from experiment 397 and import the folders of experiments 87, 48,
  383, 395, and 400 and the `studies` evaluation code.
hypothesis: >-
  One mixture density network trained end to end on injected load-conserving events, with a
  lagless module, a calibration module, and a separate trust head that reads only the trust inputs
  and outputs a continuous weight for each lagged power input, forecasts better than both the
  constant-trust control (B0) and the plain network trained on the same injected data (A0-inj) on
  harm rows (rows whose target lies outside every load-shift event but whose lags lie inside one),
  and loses no more than 0.05 points of CRPS (continuous ranked probability score, in points of
  effective capacity) on rows outside events.
prediction: >-
  Written before any score is read. P0. The batch 0 counterfactual gain at the signed-MW six is
  between
  0.05 and 0.3 CRPS points. P1. B1 minus A0-inj on harm rows at the signed-MW six
  Primaries is between -0.05 and -0.2 points. Experiment 402's `ceilings3.txt` (interim, unaudited)
  found -0.179 (interval -0.470 to +0.016)
  for a flag oracle against a fitted constant mixture, so the gain of a learned weight with no
  labels is capped near that oracle. P2. B1 minus B0 on harm rows at the signed-MW six is
  between -0.05 and -0.3 points: the trust inputs add some gain over a constant weight, less than
  the architecture adds (experiment 400 found a fixed 50/50 mixture as good as a logistic gate).
  P3. The loss on rows outside events, B1 minus A0-inj, is under 0.05 points at the signed-MW six.
  P4. On held-out real events at the signed-MW six, the mean combined trust (the geometric mean of
  the 3 trust weights) of B1 outside events exceeds its mean inside events by at least 0.1 (the
  inert rule does not fire), and the mean outside events stays above 0.7. P5. B1 beats B2 on harm
  rows at the
  signed-MW six (the maintainer's expectation); if B1's gain over A0-inj is positive, B2's gain is
  under half of it. P6. At the MVA-only eight the trust head is inert (the same difference is under
  0.1) and the 95% interval of B1 minus A0-inj on harm rows includes zero, because experiment 402
  step 1 (interim) found a flag oracle scores 7.297 against 7.298 for the forecast with lags there.
  P7. A0-inj minus A0 on harm rows is below -0.1 points (training on injected events alone helps)
  and on rows outside events is under +0.05. P8. B5 (no `lag_origin`) changes CRPS outside events
  by under 0.03 points in either direction, because the experiment 395 Part E sweep found the
  newest lag moves the median by only 1.3% of a step. Batch 2. P9. B3 (attention over the lag
  tokens) beats B2 on harm rows by under 0.1 points. P10. B4a, B4b, and B4c (N = 1, 3, and 5 weekly
  lags) each change CRPS outside events by under 0.05 points in either direction and the harm rows
  by under 0.1 in either direction. P11. B6 (learned windows) differs from B1 by under 0.1 points
  on harm rows; a gain above 0.1 would mean the hand-built statistics lose information. P12. B7
  (7-day demand window) lowers harm-row CRPS against B1 by under 0.05 points, because the window
  contains the suspect readings. P13. Replacing the product combining rule by the mean of the 3
  weights (sensitivity arm B1m) changes harm-row CRPS by under 0.1 points. P14. B8 minus B1 on harm
  rows at the signed-MW six is at most -0.05 points (the probe improves the harm rows). P15.
  B1-rw loses less than half of B1-nn's loss against B1 on harm rows (neighbours then only average
  out noise). P16. B1 minus B3 on harm rows is between -0.05 and +0.05 points (the trust head and
  attention over lags do about equally well).
decision_rule: >-
  Contrasts, registered before any score is read and read per metering group (signed-MW six;
  MVA-only eight; D15 and D18 report-only) on CRPS at days 1 to 10, each on three row sets: harm
  rows, rows outside events, and all rows. Batch 1: C1 B1 minus A0-inj (the architecture); C2
  A0-inj minus A0 (the training data); C3 B1 minus B0 (the trust inputs); C4 B1 minus A0; C5 B1
  minus B2; C6 B5 minus B1. Batch 2: C7 B3 minus B2 (attention over lags); C8a, C8b, and C8c B4a,
  B4b, and B4c minus B1; C9 B6 minus B1; C10 B7 minus B1; C11 B1m minus B1; C12 B8 minus B1; C13 B1
  minus B3; C14 B1 minus B1-rw; C15 B1 minus B1-nn; C16 B9 minus B1. The
  screen has five test months (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), so every number is
  labelled interim. Harm rows carry whole-event bootstrap intervals (95%, resampling events); rows
  outside events carry month-block bootstrap intervals with no 99% claim; the Bonferroni k is 32
  (16 contrasts, 2 groups) for the reported contrasts, and none for the stop rules. A contrast
  counts as a gain when its interval excludes zero and its sign agrees in at least 2 of 3 seeds.
  Batch 0 rule: re-score the stored forecasts of experiment 402 on harm rows with the lags inside
  each real event replaced by clean counterfactual values (experiment 401's harness); if the gain
  is under 0.1 CRPS points at the signed-MW six, skip batches 1 and 2, write up, and release the
  slot. Batch 1 stop rule, computed on harm rows at the signed-MW six with an event-cluster
  bootstrap (resampling whole real events with their harm rows, 1,000 resamples) and a 95%
  interval, without Bonferroni correction: run batch 2 only if, for both B1 minus A0-inj and B1
  minus B0, the point estimate is at or below -0.1 CRPS points and the upper bound of the interval
  is below zero, and if the point estimate of B1 minus A0-inj on rows outside events is at or
  below +0.05 points. Otherwise stop, write up, and release the slot. The threshold of -0.1 is
  about half of the oracle gain in experiment 402's `ceilings3.txt` (the flag oracle minus the
  fitted constant mixture is -0.179, interval -0.470 to +0.016, pooled; -0.197, interval -0.587 to
  +0.084, for the signed-MW group), whose interval includes zero. Batch 1 is therefore expected
  to stop, which is a cheap and useful answer. Inert rule: define the combined trust as
  the geometric mean of the 3 per-lag trust weights (for B3, the attention mass on the
  lagless-forecast token). A half-hour is inside an event if any of its 3 lags lies inside a real
  corrected event (unpadded). An arm is inert if its mean combined trust outside events minus its
  mean inside events is under 0.1, on held-out real events at the signed-MW six. The rule applies to
  B1, B3, B4a to B4c, B5, B6, B7, and B8. A fired rule stops that arm only, and the worker reports
  it. B0 has no input-dependent weight, so
  the rule does not apply to it. The MVA-only eight is reported beside the six and neither rule
  applies to it. The smallest gain worth porting is 0.1 points on harm rows, because the trust head
  and injected-event generation add a network head and a generator to the training loop; a pooled
  harm-row gain under 0.1 is `small gain`.
benchmark: >-
  Arm A0 (the pooled plain mixture density network with the three origin lags, arm `plain_pooled`
  of experiment 397 stage 2, trained on the clean data; its stored predictions if the rows, seeds,
  and folds match, otherwise refitted here), and the stored forecasts, never refitted, in the
  private data folder `data/ml-experiments/`: monthly XGBoost arm S0M
  (`00389-xgboost-tail-calibration-and-ensemble-spread-columns/work/arm_preds`), the weekly bundle
  (`00381-simple-weekly-xgboost-demand-against-gam-stack/work`), BG2 (experiment 87), the GAM stack
  (`00377-demand-v1-first-run-feasibility/work`), `manual_heuristic`, and `climatology`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS from the 11 quantiles (as the protocol computes it) with the exact mixture CRPS beside it;
  coverage and pinball loss at q01, q05, q95, and q99, raw and at the 1.15 widening; 5-95% width;
  the lag-encroachment sweep of experiment 395 Part E; and the trust weights.
code: ml-experiments/00404-mdn-lagless-and-calibration-module-trust-weights/
---

**Take-home message (to be written when the result is known).**

This page is the plan. It is committed before any network is fitted, so the commit order proves the
predictions came first. It follows the revised experiment 2 of the weekend programme brief
(revised after Opus and Sonnet design reviews; the maintainer has not yet approved the revision).

## Question

**We test whether one neural network can learn how much to trust each lagged power reading, with no
event labels.** A load-shift event moves demand between neighbouring Primaries for hours or weeks.
The power the network sees at the same time yesterday or last week can then be abnormal, and a
forecast that copies that power inherits the abnormality. NGED cannot supply event labels for about
1,300 substations, and every binary event detector tried so far has scored poorly. The maintainer's
design replaces the detector with a continuous trust weight per lagged input, learned end to end.

- **Lagless module.** A mixture density network (3 Gaussians) that sees weather, solar geometry,
  calendar, holidays, and a Primary embedding, and no power lag.
- **Calibration module.** A second network that outputs a lagged mixture (3 Gaussians) from the
  lagless forecast, the three lags, the Primary embedding, and an indicator of signed-MW or
  MVA-only metering. It does not see the trust inputs.
- **Trust head.** A separate small network that reads only the trust inputs below, the lead time,
  the Primary embedding, and the metering indicator, and outputs one weight in [0, 1] per lag.
  Feeding the trust inputs only to this head stops the calibration module from falling back on the
  lagless forecast inside the lagged head, which would leave the trust weight unidentified and let
  it collapse to a constant, as the logistic gate of experiment 400 did (about 0.6).
- **The three lags.** `lag_origin` (the newest reading before the forecast origin),
  `lag_same_time_latest_day` (the reading at the target's time of day on the latest day before the
  origin), and `lag_same_time_latest_week` (the same on the latest week before the origin), as in
  the weekly XGBoost member. Each carries its age in hours and an is-missing flag.
- **Output.** A mixture of the lagless and lagged distributions (6 Gaussians), never an average of
  quantiles. The weight on the lagged mixture is the product of the 3 per-lag trust weights, so one
  untrusted lag lowers it, and the lagless distribution takes the weight 1 minus that product.
  Arm B1m replaces the product by the mean of the 3 weights.
- **Training scheme (a departure from fully end to end).** Cross-fitting makes the lagless forecast
  a precomputed table. The lagless module trains on its own mixture negative log-likelihood, and
  the rest of the network trains on the final mixture negative log-likelihood of the clean series,
  on injected events, with the lagless forecast held fixed. No separate stage on event labels is
  used. The fully joint version, in which the lagless module trains jointly with a stop-gradient
  and no cross-fit, is arm B9, run last and labelled in-sample.
- **Series.** Each Primary series is one series per substation.

## Review against the latest evidence

**The plan was reviewed against the pages merged since the brief was written** (experiments 387,
398, 399, 401, and 403, and the standing practices of the `ml-experiment` skill, read from `main` on
2026-10-30). Four changes follow; none alters the scope.

- **Fixed-mixture control.** Experiment 401's advice is to use a fixed mixture with a weight of 0.3
  to 0.5 on the forecast with lags as the control for any learned blend at the signed-MW Primaries.
  The plan adds a free arm F0: a fixed mixture, with weight 0.4 on the lagged forecast of A0-inj
  and 0.6 on L0, computed from stored forecasts. F0 is reported at the signed-MW six only, and not
  at the MVA-only eight or at D15 and D18.
- **How the lag response is measured.** The lag response is the forecast with lags minus the
  forecast without lags on the same rows, and never 1 minus the residual over the labelled step.
- **Standing practices followed.** Monthly refits; cleaned data, with the number of dropped
  `substation_zero` half-hours reported per fold; the D15 and D18 cut at 2025-03-01; every fitted
  network saved in the experiment's data folder; stored predictions reused as references; tails and
  the avoidable-curtailment cost reported beside CRPS; example forecast plots and event plots;
  draft pull request. The plan departs from the practice of four screen folds: it uses the five
  screen folds of experiment 397, so that its scores sit on the same rows as experiments 397 and
  403, and labels every number interim.
- **Evidence that tempers the expected gain.** Experiment 401 found real events move the forecast
  by 0.03 of the step or less, and that one event at D16 carries most of a fixed mixture's gain at
  the six. Experiment 402's `ceilings3.txt` found the flag oracle's gain has an interval that
  includes zero. The batch 0 and batch 1 stop rules already encode this.

## Design choices fixed by this plan

**The smoother and the surprise source are fixed here because the brief leaves them open.** Each is
a stated choice; a different choice is a later experiment.

- **Light causal smoother.** After the trust head, a causal moving average of 3 half-hours is
  applied over the lag's valid times, so that weights do not jump between adjacent half-hours.
  Events start suddenly, so the smoother is short. The write-up reports how many half-hours the
  smoothed trust weight takes to cross 0.5 after an injected onset, beside the unsmoothed weight.
- **Trust weights start at 0.5** with no regulariser. The write-up reports whether they stay near
  0.5.
- **Stop-gradient.** The surprises and the gap between the lagged and lagless medians pass through
  a stop-gradient, so the lagless module cannot shape its own surprises through the trust head.
- **Cross-fitted lagless forecast (leakage rule).** For each monthly refit, the months before the
  scored month in the training window are split into K = 6 contiguous blocks. For each block, a
  lagless module is fitted on the other 5 blocks and forecasts the held-out block. The 6
  cross-fitted forecasts form the table of out-of-sample lagless forecasts for the training rows.
  One further lagless module is fitted on all months before the scored month, for the scored month
  and for test-time inference. A lag whose valid time falls before the scored month takes its
  surprise from the cross-fit table and not from the further module, because the further module was
  fitted on those months. The table is shared across all arms and seeds, and is fitted once
  per refit and seed. The calibration module and the trust head both take the table's lagless
  mixture as input in training, and the further module's mixture at test time. A stop-gradient
  on the lagless forecast stops the lagless module shaping its own surprises. Cross-fitting never
  uses the scored month or later. The other blocks include later months than the held-out block,
  which a live forecast would not have, so the training surprises are somewhat larger than a fully
  fitted lagless forecast would give; the write-up lists this mismatch as a caveat. If the timed
  fits show K = 6 is too expensive, the plan settles on a smaller K and tells the manager.
- **Forecast origin and lead of the lagless forecast.** At a lag's valid time, the lagless
  forecast uses the ECMWF ENS (European Centre for Medium-Range Weather Forecasts ensemble) mean
  and spread of the freshest run available at the forecast origin (the latest run published by the
  valid time, leads 9 to 33 hours). That run exists at forecast time. ERA5 is not used. The training
  table applies the same
  freshest-run rule at every origin.
- **One pass on the ensemble summary.** The network sees the ENS mean and spread of each weather
  variable once. The 51-member scheme runs only as a check on the best arm.

## Trust inputs

- **Per lag.** The single-step surprise (the lag minus the lagless median at the lag's valid time)
  in capacity-normalised MW, kept signed, and the same value in units of the lagless band width.
  Both are given, because a clipped score loses magnitude.
- **Per lag, window statistics** of the surprise over windows centred on the lag (half-widths 1.5,
  3, 6, 12, and 24 hours, and 84 hours for the week lag only): mean, median, minimum, maximum,
  standard deviation, signed sum, count above the band, and a 12-hour triangular tent mean. Every
  window is truncated at the forecast origin and at the telemetry cut-off, and a truncated tent is
  renormalised to sum to 1.
- **Neighbours.** The same quantities for neighbouring Primaries from `neighbours_nged.json` (in
  the folder of experiment 383), with the signed logic learned and never hand-coded. Bulk supply
  points are added if their stored series exist; if not, the write-up says they were left out.
  At scale the graph must come from the data (for example the correlation of forecast residuals);
  the write-up states what that needs.
- **Lag age and is-missing flag** for each lag, and the gap between the lagged and lagless medians.
- **Recomputation after injection.** Every one of these inputs is recomputed from the injected
  series, including the neighbour, bulk-supply-point, B6, and B7 windows.

## Arms

All arms use the pooled plain mixture density network of experiment 397 (arm `plain_pooled`) as
their base: one network for all 16 Primaries, a learned Primary embedding, a 3-component Gaussian
mixture head. The maintainer's update of 2026-10-30 moved the base from the uniform-token arm of
experiment 397, called U0 here, to the plain network. The interim scores of experiment 397
(unaudited) were
3.264 at the signed-MW six and 3.860 at the MVA-only eight for the plain network, against 3.320 and
3.775 for U0, and the plain network's lead rests mainly on one fold, so this
plan infers no ranking between them. Both modules get the Primary embedding. The group gate over
input groups stays closed in every arm. Every arm trains on the injected data below, except A0.

**Batch 0 (a cheap check before batch 1).** Re-score the stored forecasts of experiment 402 on harm
rows with the lags inside each real event replaced by clean counterfactual values, using experiment
401's harness. This gives the most that any lag-trust method can gain, because experiment 401 found
real events move the forecast by only 0.02 to 0.03 of the step, yet the forecast with lags scores
5.119 against 4.292 without lags on harm rows, so most of that gap is the two forecasts differing as
whole forecasts. If the counterfactual gain is under 0.1 CRPS points at the signed-MW six, the plan
skips batches 1 and 2.

**Deviation from the batch 0 rule.** Experiment 411 ran batch 0 and was inconclusive. The gain at
the signed-MW six was -0.836 CRPS points (95% interval -2.578 to +1.226) on 9 events. One D5 event
flips the sign: without it the gain is +0.460 (95% interval -0.102 to +1.631). At the eight MVA-only
Primaries the gain was +0.024 (95% interval -0.574 to +1.529). The harness scored only 15 of the
184 corrected events. Experiments 411 and 414 reached only 9 signed-MW-six events. The 4 test months
and the sign, step, and bias filters capped that count, not gaps: 129 of the 184 events lie outside
the test months. The ceiling is therefore a thin figure from XGBoost models alone. Batch 1 runs
regardless, with no 12-refit extension and no looser event filters on this experiment's account.
The point estimate is under 0.1, so the rule as written would skip batches 1
and 2, but the interval spans zero. The neural Discuss session therefore ruled that batches 1 and 2
run. The batch 1 stop rule stays as written.

**Batch 1:**

- **L0 (free).** The lagless module's own forecast, scored on all rows and on harm rows. Experiment
  402's weights suggest the lags help the signed-MW six only at days 1 to 2, and L0 shows that
  directly.
- **A0.** The pooled plain network with the three lags, trained on the plain clean data (the
  reference for the architecture alone).
- **A0-inj.** A0 trained on the same injected data as B1. B1 minus A0-inj isolates the
  architecture, and A0-inj minus A0 isolates the training data.
- **B0.** Constant trust: B1's architecture with one learned constant trust weight per metering
  group, lead band, and lag that does not depend on the trust inputs (the control experiment 400
  recommended). B1 minus
  B0 isolates the trust inputs.
- **B1.** The architecture above, with the three lags.
- **B2.** A0-inj with token-level lag dropout during training (each lag token zeroed with
  probability 0.2 and flagged missing) and no lagless module.
- **B5.** B1 without `lag_origin`.
- **B-mix (reference, free).** A fixed 50/50 mixture of A0-inj's lagged forecast and the lagless
  module's forecast (the neural Discuss session's request, 2026-10-31). It is scored on the same
  harm rows and event slices as every arm. Experiment 402 step 4 (interim, unaudited) found that a
  simple mixture captured most of a lag-surprise gate's gain on injected shifts at the six
  signed-MW Primaries (mean CRPS 4.307 for the mixture, 4.250 for the gate, 4.576 with lags, 4.439
  without). Any trust head or attention over lags in this experiment must beat B-mix to count.

**Batch 2** (runs straight away if the batch 1 stop rule does not fire):

- **Carry-forward rule (maintainer, 2026-10-30).** B3 attends across the lag set that wins among
  B4a, B4b, B4c, and B1's three lags, together with the XGBoost lag experiment 388 when its page
  is available. The order of work therefore runs B4a, B4b, and B4c before B3. If B3 has already run
  with the 3 lags, it is rerun with the winning set. The adopted set is named in the arm label (for
  example B3-lags5), and the write-up says which earlier components were adopted and which were
  not, with the reason.
- **B3.** B1's lagless module and cross-fitted table, with an attention gate in place of the trust
  head and the product rule. The gate attends over 4 tokens: the 3 lag tokens (each the lag value
  with its age, missing flag, and trust inputs) and a lagless-forecast token. Its query is built
  from the lead time, the Primary embedding, and the metering indicator. The output is the gated sum
  of tokens fed to a 3-component mixture head. B3 minus B2 tests attention over lags, and B1 minus
  B3 compares the trust head with attention.
- **B6.** B1 with a 1D convolutional neural network over the surprise series in a window of 24
  hours either side of each lag (97 half-hourly steps), in place of the hand-built window
  statistics. Each window is zero-padded and truncated at the forecast origin. The network has two
  convolution layers (kernel 5, 16 channels) with global pooling.
- **B4a, B4b, and B4c.** B1 with more lags: the same weekday and time from each of the last
  N = 1, 3, and 5 weeks, plus the median across the N weeks and the median minus the latest week.
  The three values of N are three separate arms.
- **B8a and B8b (low priority; run after B1 reads clean).** B1 plus an auxiliary probe head that
  reads the
  trust head's hidden layer and predicts, for each lagged timestep, whether the injected generator
  put it inside an event or a spike. The loss is binary cross-entropy at a weight of 0.1 of the
  mixture loss in B8a and 0.3 in B8b. The probe trains only on the injected labels, which
  are known truth and need no NGED label, and it is never used at forecast time. Real-event labels
  come from the 112-row corrected table, so the write-up reports the event count per metering
  group beside every area figure, and says plainly if the probe works on injected events and fails
  on real ones, which would mean the injection is unrealistic.
- **B1m.** B1 with the mean of the 3 trust weights as the mixture weight.
- **B1-rw and B1-nn (neighbour controls).** B1-rw replaces each Primary's neighbour inputs with
  those of a degree-matched randomly chosen set of other Primaries (the graph rewired, with the
  same number of neighbours). B1-nn removes the neighbour inputs. B1 minus B1-rw tests whether
  neighbour identity matters in the trust head, and B1 minus B1-nn tests whether the neighbour
  inputs matter at all. The cross-fitted tables are reused.
- **B9 (last).** The fully joint version of B1: the lagless module trains jointly with a
  stop-gradient and no cross-fit. It is labelled in-sample.
- **B7 (lowest priority; runs last).** B1 plus a 1D convolutional neural network over the latest 7
  days of actual demand up to the forecast origin, with the same trust inputs.

The maintainer's expectation (not a result): B1 beats B2.

## Training data and events

**Training uses injected, load-conserving events with clean targets.** A drop at a donor Primary
reappears at the connected recipients, with the depth as a share of the donor's trailing 30-day
mean (truncated at the forecast origin). Durations and depths follow the corrected label table
`sw_block_detector_label_table_corrected.csv` in the folder of experiment 395 (use the corrected
signs). In 30% of events no recipient is observed (a guess, varied as a sensitivity). Transient
spikes that are not load shifts are injected too. The targets stay clean, so a target is never
inside an injected event. The loss is the mixture negative log-likelihood, which is a proper
scoring rule like CRPS and is what experiment 397 used.

**Injection rate.** 20% of training rows have at least one lag inside an injected event or spike,
against about 2.6% in real data. The rate is a stated default, and the worker states and justifies
any change. The injection is fixed per refit and seed, saved to disk, and not resampled each epoch.
The generator tunes its coverage on the real arrays until the realised share of rows reaches 20%.

**Which experiment the injection follows.** Experiment 395 Parts E and F disagree on the share of a
step that reaches the forecast (36.8% with all three lags affected in the synthetic sweep of Part
E, against a median share of 0.94 for real events in Part F; experiment 401 shows the 0.94 is not a
lag effect). The injection writes the full event
depth into the lag values, as a real event does, and does not set a share of the forecast. The
share that reaches the forecast is then an output, measured by the lag-encroachment sweep.

**Realism check before training: match surprise statistics, not a lag response.** In experiment
401's counterfactual (replace the lags inside each real event and re-predict), the real lag
response of the forecast is about 0.02 to 0.03 of the event step, against 0.32 to 0.34 for
synthetic events of the same kind (interim and unaudited). That response belongs to the fitted
forecaster and not to the injection, so the plan does not tune the injection to a response. It
matches the injected surprise statistics (the window statistics of the surprise series, the event
depth as a share of the series, and the duration) to those of real events, and redesigns the
injection if they differ by more than a factor of 2, as the lag-gate brief requires. For each
fold, the injected depths and durations are drawn, and all tuning is done, only from real events
outside that fold's scored month, so that the events that score an arm never tune its training
data. The write-up reports the fitted network's counterfactual lag response on injected and on real
events as a diagnostic. Every arm is judged on real events as well as injected ones. The plan
follows experiment 395 Part F (the full depth in the lag values) and not Part E.

**Evidence on constant-step injection.** Experiment 402's redone step 2 (interim, unaudited) gave
the median share of a step passed on by the R1 boosters as 0.133, 0.150, and 0.305 for real events
with 1, 2, and 3 lags inside, against 0.146, 0.158, and 0.328 for a synthetic constant step (ratios
1.05 to 1.10; 17, 14, and 6 events). The factor-of-2 rule does not fire, so a constant-step
injection is realistic for those boosters, and experiment 402 repeats the check on its R0 lagged
boosters. The two experiments measure different quantities: experiment 401 takes the forecast with
lags minus the forecast without lags, and experiment 402 takes the share of the step passed on. The
plan uses both as diagnostics on the fitted networks and does not tune the injection to either.

**Harm-row results are a lower bound.** Because some events are unlogged, harm-row results are
reported as a lower bound on harm. Our series are one per substation, not per transformer, so a
transformer-level event that conserves load across a substation's transformers does not appear.

**Generator.** This experiment uses experiment 402's generator (`e00402_inject.py`, draft PR 56),
copied unchanged into this experiment's folder from commit `bfc32045` of branch
`lag-gate-injected-load-shifts`. The copy is unreviewed until this experiment's Sonnet code review
covers it or experiment 402's own review does, and it is re-copied if experiment 402's file changes
after that review. Experiment 402 pushed it on 2026-10-30 after
self-testing it, and it had not been code-reviewed then. It differs from the generator first
written here in four ways: the donor change keeps the donor's shape (the donor is multiplied by
`1 - f`), spikes are 4 to 20 times the stage-1 residual median absolute deviation, MVA-only
Primaries receive events only where the reading stays above 20% of capacity, and base periods can
exclude windows with a large stage-1 surprise. The signature takes the MVA-only set and each
Primary's residual deviation in MW. The write-up says which version of the generator each fit used.
The depths and durations come from the table's real events, so the test distribution informs the
generator's distributions even though the real events never enter training; the write-up states
this.

**Caveat on unlabelled events (carried into every write-up).** NGED says load-shift events may be
missing from its logs. The clean-lag rows and the clean training periods can therefore contain
events that no label covers. The harm table's clean-lag baseline is then optimistic, and the
real-event slices undercount the rows whose lags lie inside an event. Injected events stay the main
training signal because they need no logs.

**Real events are a held-out test only.** The label table's events never enter training. The
write-up reports the real-event count per fold and excludes folds without label coverage from the
harm rows.

## Leakage rules

- Cut every training row whose valid time is on or after the first day of the scored month.
- Truncate every window of every lag at the telemetry cut-off and at the forecast origin.
- Take surprises only from the cross-fitted lagless forecast described above, recomputed against
  the injected lag values after injection (surprises from the clean series would carry no event
  signal), never from the scored month or later.
- Inject the donor and its recipients together, so that the neighbour inputs never leak the clean
  value.
- Drop flagged exact-zero half-hours (`substation_zero`) from training, lags, inference, and scoring
  actuals.
- For D15 and D18, drop every row before 2025-03-01 from training rows, lag inputs, inference
  inputs, and scoring actuals (maintainer rule, 2026-10-30). This experiment does not test the cut.
  Experiment 399 may later supply better per-Primary dates, and the plan then follows them.

## Scoring

- **Harm rows (defined once).** A harm row is a half-hour of the real held-out data whose target
  is outside every real corrected event (unpadded, with the 7 days after each event excluded, as
  experiment 400 did) and at least one of whose 3 origin lags lies inside a real corrected event.
  The B4 arms' extra lags do not redefine the harm rows. A contrast on harm rows is the difference
  of the two arms' mean CRPS on those same rows, and scoring uses real data only.
- **Three row sets, scored separately beside the headline** (the neural Discuss session's request,
  2026-10-31): harm rows, clean rows (no lag and not the target inside an event), and rows whose
  target is inside a labelled real event. Experiment 402's slice tables (interim, unaudited) found
  that an injected-trained gate protects harm rows without losing on clean rows, but is worse than
  the fixed 50/50 mixture where the target is inside a real labelled event (signed-MW six: 10.75
  and 11.42 against 10.50 and 11.25, in points of effective capacity), because it cuts the lag weight
  to about 0.4. All experiment 402 figures are interim until its page merges. Every
  arm, B-mix included, is scored on all three sets, with whole-event bootstrap intervals.
- **The harm table** of the lag-gate brief: rows whose target is outside every event but whose lags
  lie inside one, minus the rows with clean lags, by number of lags inside the event. Each arm minus
  A0 on the harm rows, with whole-event bootstrap intervals.
- **The lag-encroachment sweep** (experiment 395 Part E harness): the median's share of a synthetic
  step by the number of lags inside the event, the band width ratio, and where the probability mass
  sits.
- **Probe (B8).** The precision-recall area of the probe on held-out real events from the corrected
  label table, per metering group.
- **Loss outside events** and the overall five-fold CRPS, per Primary.
- **Tails.** Coverage and pinball loss at q01, q05, q95, and q99, raw and at the 1.15 widening,
  beside every headline score.
- **Example forecast plots (maintainer rule, 2026-10-30).** Example time series charts of
  probabilistic forecasts against the actuals: one day-ahead forecast, and one full 14-day horizon
  of a single forecast, with bands that widen with the horizon. Both are shown for one signed-MW
  Primary and one MVA-only Primary (4 charts), and one further chart shows a real load-shift event.
  Each shows the actuals, the median, and the 90% band for B1 and the reference arms. Substations
  are not covered by the anonymisation rule, and no metered generator's series appears here.
- **Event time-series plots (maintainer rule, 2026-10-30).** At least 6 plots, covering both
  metering groups (at least 3 from the signed-MW six and 3 from the MVA-only eight), each marking
  when the event starts and ends and showing the actual power, the median forecast, the 90% band
  (q05 to q95), and the trust weight of each of the 3 lags (for B3, the per-lag attention weights).
  Each plot compares the reference arms (A0, A0-inj, B0) with B1 on the same event. The set
  includes real events from the corrected label table, injected events, and at least one transient
  spike, and each plot is labelled real or injected. The set also includes the failure cases: the
  plots with the largest harm-row loss, not only favourable examples. Both groups of events use
  anonymised labels only where a metered generator's series appears, and none does here.
- **Checks.** The median minus the mean forecast per Primary, the error of the mean beside every
  quantile-score contrast, and which of the seven audit triggers of the `ml-experiment` skill fired.
- **Saved outputs, per arm, fold, and seed:** per-row per-lag trust weights, the combined trust, the
  event class of each row (inside, touching, or clean, unpadded), per-row CRPS, B3's attention
  weights, the B8 probe scores, and the forecast quantiles q01 to q99.
- **Folds, seeds, cadence.** The five screen folds of experiment 397, monthly refits, 3 seeds. More
  seeds only for a contrast within the seed noise.

## Compute

**The concurrency limit is 1 GPU process under 8 GB of GPU memory, 1 feeder core, and
`MemoryMax=8G`,** a stricter slot than the brief's 16 GB, because of the machine's load. The
manager asked for a request for more resources if a batch would take over 12 hours.

**Fit counts.** Each fit is one arm on one refit with one seed. The brief reports 50 to 200 seconds
per fit and about 600 fits in all; the plan's own count follows, and the first task after the slot
is granted is to time one fit of each kind and report K and the timed duration to the manager.

- **Cross-fitted lagless networks.** 6 block fits plus 1 final fit, so 7 per refit and seed, and 105
  over the 5 refits and 3 seeds. They are precomputed once per refit and seed and shared by every
  arm.
- **Injected windows.** The injected series and every window statistic, neighbour input, and B6 or
  B7 window are precomputed once per refit, with a fixed injection seed, and shared by every arm and
  seed.
- **Batch 1.** 6 arms times 5 refits times 3 seeds is 90 fits (A0 is read from experiment 397 if
  its rows match, which would remove 15).
- **Batch 2.** B3, B6, B4a, B4b, B4c, B8a, B8b, B1-rw, B1-nn, B1m, B7, and B9 are 12 arms, 180
  fits.
- **Memory per fit.** A pooled refit has about 7 million rows. With the trust inputs (about 150
  columns in float32) the host matrix is about 4 GB, which fits the 8 GB cap with the batch
  streamed to the GPU in blocks of 16,384 rows. B6 and B7 add window tensors that are built in
  blocks from the precomputed series, not stored in full. These figures are estimates; the first
  task after the slot is granted is to time and measure one fit of each kind.

Order of work once granted: batch 0 (the counterfactual re-score of the stored forecasts, which
takes no fit); the cross-fitted lagless networks; the realism check; A0 (reused if
experiment 397's rows match); A0-inj; B0; B1; B2; B5; the stop-rule check; then batch 2 if the
stop rule passes, in the order B4a, B4b, B4c, B3, B6, B8a, B8b, B1-rw, B1-nn, B1m, then B7 and B9.
Every arm
is in scope over the weekend, gated only by the pre-registered stop rules.

## Problems caused by the move

(The move of the research code from the production repository to this one.) None so far.
Experiment 397's modules live on an unmerged branch, so this experiment copies the files it needs
into its own folder.
