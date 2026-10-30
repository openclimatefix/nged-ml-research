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
  Written before any score is read. P1. B1 minus A0-inj on harm rows at the signed-MW six
  Primaries is between -0.1 and -0.4 points. The interim XGBoost series (experiment 402,
  unaudited) found -0.40 for a flag oracle against a fitted constant mixture, and a learned weight
  with no labels should land below that oracle. P2. B1 minus B0 on harm rows at the signed-MW six is
  between -0.05 and -0.3 points: the trust inputs add some gain over a constant weight, less than
  the architecture adds (experiment 400 found a fixed 50/50 mixture as good as a logistic gate).
  P3. The loss on rows outside events, B1 minus A0-inj, is under 0.05 points at the signed-MW six.
  P4. On held-out real events at the signed-MW six, the mean combined weight (the product of the 3
  trust weights) of B1 inside events is at least 0.2 below its mean outside events (the inert rule
  does not fire), and the mean outside events stays above 0.7. P5. B1 beats B2 on harm rows at the
  signed-MW six (the maintainer's expectation); if B1's gain over A0-inj is positive, B2's gain is
  under half of it. P6. At the MVA-only eight the trust head is inert (the same difference is under
  0.2) and the 95% interval of B1 minus A0-inj on harm rows includes zero, because experiment 402
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
  rows at the signed-MW six is at most -0.05 points (the probe improves the harm rows).
decision_rule: >-
  Contrasts, registered before any score is read and read per metering group (signed-MW six;
  MVA-only eight; D15 and D18 report-only) on CRPS at days 1 to 10, each on three row sets: harm
  rows, rows outside events, and all rows. Batch 1: C1 B1 minus A0-inj (the architecture); C2
  A0-inj minus A0 (the training data); C3 B1 minus B0 (the trust inputs); C4 B1 minus A0; C5 B1
  minus B2; C6 B5 minus B1. Batch 2: C7 B3 minus B2 (attention over lags); C8a, C8b, and C8c B4a,
  B4b, and B4c minus B1; C9 B6 minus B1; C10 B7 minus B1; C11 B1m minus B1; C12 B8 minus B1. The
  screen has five test months (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), so every number is
  labelled interim. Harm rows carry whole-event bootstrap intervals (95%, resampling events); rows
  outside events carry month-block bootstrap intervals with no 99% claim; the Bonferroni k is 28
  (14 contrasts, 2 groups) for the reported contrasts, and none for the stop rules. A contrast
  counts as a gain when its interval excludes zero and its sign agrees in at least 2 of 3 seeds.
  Batch 1 stop rule, computed on harm rows at the signed-MW six with the whole-event bootstrap and a
  95% interval: stop batch 2 if the upper bound of the interval of B1 minus A0-inj, or of B1 minus
  B0, is above -0.1 CRPS points, or if the point estimate of B1 minus A0-inj on rows outside events
  is above +0.05 points. The rule is stricter than the lenient rule of the load-shift series because
  it gates the weekend's compute; the plan would relax a number only after telling the manager.
  Inert rule: on held-out real events from the corrected label table at the signed-MW six, define
  the combined weight as the product of the 3 per-lag weights; an arm among B1, B3, B6, and B8 is
  inert if its mean combined weight inside events minus its mean outside events is under 0.2. A
  fired rule stops that arm only, and the worker reports it. B0 has no input-dependent weight, so
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
- **Training.** The whole network trains end to end on the mixture negative log-likelihood of the
  clean series, on injected events. No separate stage and no event label is used.

## Design choices fixed by this plan

**The smoother and the surprise source are fixed here because the brief leaves them open.** Each is
a stated choice; a different choice is a later experiment.

- **No output smoother.** The window statistics of the trust inputs already average over 1.5 to 24
  hours around each lag, and events start suddenly, so a further smoother only adds delay. The
  write-up reports how many half-hours the trust weight takes to cross 0.5 after an injected onset.
- **Trust weights start at 0.5** with no regulariser. The write-up reports whether they stay near
  0.5.
- **Stop-gradient.** The surprises and the gap between the lagged and lagless medians pass through
  a stop-gradient, so the lagless module cannot shape its own surprises through the trust head.
- **Cross-fitted lagless forecast (leakage rule).** For each monthly refit, the training surprises
  come from a cross-fitted lagless forecast that uses only months before the scored month, blocked
  by month: the lagless forecast for each month of the training window comes from a lagless network
  fitted on the months of the window before that month, and never on the scored month or later. The
  first 6 months of the 24-month window have too little earlier history and serve only to fit
  lagless networks, not as training rows of the trust arms. During training, both the calibration
  module and the trust head take this cross-fitted lagless forecast as input. At test time both
  take the lagless module's own forecast for the scored month, which is out of sample by
  construction. The training surprises are somewhat larger than a fully fitted lagless forecast
  would give, and the write-up lists this mismatch as a caveat.
- **Forecast origin and lead of the lagless forecast.** At a lag's valid time, the lagless
  forecast uses the ECMWF ENS (European Centre for Medium-Range Weather Forecasts ensemble) mean
  and spread of the freshest run available at the forecast origin (leads 0 to 24 hours of that
  run). That run exists at forecast time. ERA5 is not used. The training table applies the same
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
mixture head. The maintainer's update of 2026-10-30 moved the base from the uniform-token arm A0
of experiment 397 to the plain network. The interim scores of experiment 397 (unaudited) were
3.264 at the signed-MW six and 3.860 at the MVA-only eight for the plain network, against 3.320 and
3.775 for the uniform-token arm, and the plain network's lead rests mainly on one fold, so this
plan infers no ranking between them. Both modules get the Primary embedding. The group gate over
input groups stays closed in every arm. Every arm trains on the injected data below, except A0.

**Batch 1:**

- **A0.** The pooled plain network with the three lags, trained on the plain clean data (the
  reference for the architecture alone).
- **A0-inj.** A0 trained on the same injected data as B1. B1 minus A0-inj isolates the
  architecture, and A0-inj minus A0 isolates the training data.
- **B0.** Constant trust: B1's architecture with a learned scalar trust weight per lead band and
  lag that does not depend on the trust inputs (the control experiment 400 recommended). B1 minus
  B0 isolates the trust inputs.
- **B1.** The architecture above, with the three lags.
- **B2.** A0-inj with token-level lag dropout during training (rate 0.3, chosen so that most rows
  keep at least one lag) and no lagless module.
- **B5.** B1 without `lag_origin`.

**Batch 2** (runs straight away if the batch 1 stop rule does not fire):

- **B3.** The pooled plain network with a softmax attention gate over the 3 lag tokens, with the
  trust inputs as the gate's query, and without the lagless module and the trust head. B3 minus B2
  tests attention over lags, and B1 minus B2 tests the trust head.
- **B6.** B1 with a 1D convolutional neural network over the surprise series in a window of 24
  hours either side of each lag (half-hourly steps), in place of the hand-built window statistics.
  Each window is truncated at the forecast origin.
- **B4a, B4b, and B4c.** B1 with more lags: the same weekday and time from each of the last
  N = 1, 3, and 5 weeks, plus the median across the N weeks and the median minus the latest week.
  The three values of N are three separate arms.
- **B8 (low priority; runs after B1 reads clean).** B1 plus an auxiliary probe head that reads the
  trust head's hidden layer and predicts, for each lagged timestep, whether the injected generator
  put it inside an event or a spike. The loss is binary cross-entropy at a weight of 0.1 of the
  mixture loss, and 0.3 if 0.1 moves nothing. The probe trains only on the injected labels, which
  are known truth and need no NGED label, and it is never used at forecast time. Real-event labels
  come from the 112-row corrected table, so the write-up reports the event count per metering
  group beside every area figure, and says plainly if the probe works on injected events and fails
  on real ones, which would mean the injection is unrealistic.
- **B1m.** B1 with the mean of the 3 trust weights as the mixture weight.
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

**Injection rate.** 25% of training rows have at least one lag inside an injected event, against
about 2.6% in real data. The rate is a stated choice and is varied in the sensitivity run if the
batch 1 results are borderline.

**Which experiment the injection follows.** Experiment 395 Parts E and F disagree on the share of a
step that reaches the forecast (36.8% with all three lags affected in the synthetic sweep of Part
E, against a median share of 0.94 for real events in Part F). The injection writes the full event
depth into the lag values, as a real event does, and does not set a share of the forecast. The
share that reaches the forecast is then an output, measured by the lag-encroachment sweep.

**Realism check before training.** The window statistics of injected surprises are compared with
those of real events from the label table. The injection is redesigned if they differ by more than
a factor of 2, as the lag-gate brief requires.

**Generator.** The manager confirmed on 2026-10-30 that experiment 402 (draft PR 56) has no
injected-event generator yet; its branch holds only the ceilings script and the counterfactual
script. This experiment therefore writes `e00404_inject.py`, and the write-up says so. If
experiment 402 publishes a generator before fitting starts, this experiment switches to it. The
depths and durations come from the table's real events, so the test distribution informs the
generator's distributions even though the real events never enter training; the write-up states
this.

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
- **Trust-weight plots** through a real event, a synthetic event, and a transient spike; for B3,
  the per-lag attention weights through the same cases.
- **Checks.** The median minus the mean forecast per Primary, the error of the mean beside every
  quantile-score contrast, and which of the seven audit triggers of the `ml-experiment` skill fired.
- **Folds, seeds, cadence.** The five screen folds of experiment 397, monthly refits, 3 seeds. More
  seeds only for a contrast within the seed noise.

## Compute

**The concurrency limit is 1 GPU process under 8 GB of GPU memory, 1 feeder core, and
`MemoryMax=8G`,** a stricter slot than the brief's 16 GB, because of the machine's load. The
manager asked for a request for more resources if a batch would take over 12 hours.

**Fit counts.** Each fit is one arm on one refit with one seed.

- **Cross-fitted lagless networks.** One per month of the training window after the first 6, so
  18 per refit, and 90 over the 5 refits (one seed). They are precomputed once per refit and shared
  by every arm and seed.
- **Injected windows.** The injected series and every window statistic, neighbour input, and B6 or
  B7 window are precomputed once per refit, with a fixed injection seed, and shared by every arm and
  seed.
- **Batch 1.** 6 arms times 5 refits times 3 seeds is 90 fits (A0 is read from experiment 397 if
  its rows match, which would remove 15).
- **Batch 2.** B3, B6, B4a, B4b, B4c, B8, B1m, and B7 are 8 arms, 120 fits.
- **Memory per fit.** A pooled refit has about 7 million rows. With the trust inputs (about 150
  columns in float32) the host matrix is about 4 GB, which fits the 8 GB cap with the batch
  streamed to the GPU in blocks of 16,384 rows. B6 and B7 add window tensors that are built in
  blocks from the precomputed series, not stored in full. These figures are estimates; the first
  task after the slot is granted is to time and measure one fit of each kind.

Order of work once granted: the cross-fitted lagless networks; the realism check; A0 (reused if
experiment 397's rows match); A0-inj; B0; B1; B2; B5; the stop-rule check; then batch 2 straight
away if the stop rule does not fire, in the order B3, B6, B4a, B4b, B4c, B8, B1m, then B7. Every arm
is in scope over the weekend, gated only by the pre-registered stop rules.

## Problems caused by the move

(The move of the research code from the production repository to this one.) None so far.
Experiment 397's modules live on an unmerged branch, so this experiment copies the files it needs
into its own folder.
