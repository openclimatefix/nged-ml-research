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
  the hypothesis (how per-lag trust weights combine into one mixture weight, the smoother, the
  source of the surprise inputs, the injection), so each choice is fixed in this plan before any
  score is read. Production serving path: not touched (this repository holds none). Degradation
  rule: not touched. Callers: the scripts copy code from experiment 397 and import the folders of
  experiments 87, 48, 395, and 400 and the `studies` evaluation code.
hypothesis: >-
  One mixture density network, trained end to end on injected load-conserving events, that
  contains a lagless module and a calibration module outputting a continuous trust weight for each
  lagged power input, forecasts better than the pooled uniform network with the three origin lags
  (arm P0) on rows whose target lies outside every load-shift event but whose lags lie inside one
  (harm rows), and
  loses no more than 0.05 points of CRPS (continuous ranked probability score, in points of
  effective capacity) on rows outside events.
prediction: >-
  Written before any score is read. P1. Arm B1 lowers CRPS on harm rows at the signed-MW six
  Primaries by 0.1 to 0.4 points against P0. The interim XGBoost series (experiment 402,
  unaudited) found 0.40 for a flag oracle against a fitted constant mixture, and a learned weight
  with no labels should land below that oracle. P2. The CRPS loss on rows outside events is under
  0.05 points at each of the two groups, signed-MW six and MVA-only eight. P3. B1 beats B2 on harm
  rows at the signed-MW six (the maintainer's expectation); if B1's gain is positive, B2's gain is
  under half of it. P4. At the MVA-only eight, the 95% interval of B1 minus P0 on harm rows includes
  zero, because the XGBoost flag oracle was worse than the forecast with lags there (+0.19). P5. B3
  (the attention gate of experiment 397 arm A2 on the lag tokens) differs from B1 by under 0.05
  points on harm rows. P6. B4 (more weekly lags) changes CRPS outside events by under 0.05 points in
  either direction and the harm rows by under 0.1 in either direction. P7. B5 (no `lag_origin`)
  changes CRPS outside events by under 0.03 points in either direction, because the experiment 395
  Part E sweep found the newest lag moves the median by only 1.3% of a step. P8. In B1 on synthetic
  events at the signed-MW six, the mean trust weight of a lag inside the event is below 0.3 from
  the fourth half-hour after the onset on, and the mean trust weight of lags outside every event
  stays above 0.7. P9. The nested-input ablation of B1 (single-step surprise only; plus window
  statistics; plus neighbour inputs) improves harm-row CRPS at each addition by at least 0.05
  points; an addition below that is reported as not earning its place. P10. Replacing the mean
  combining rule by the noisy-or rule (sensitivity arm B1n) changes harm-row CRPS by under 0.1
  points. P11. B6 (learned windows) differs from B1 by under 0.1 points on harm rows; a gain
  above 0.1 would mean the hand-built statistics lose information. P12. B7 (7-day demand window)
  does not lower harm-row CRPS against B1 by more than 0.05 points, because the window contains
  the suspect readings.
decision_rule: >-
  Contrasts, each registered before any score is read and read per group (signed-MW six; MVA-only
  eight; D15 and D18 report-only) on CRPS at days 1 to 10. C1 B1 minus P0; C2 B2 minus P0; C3 B1
  minus B2; C4 B3 minus B1; C5 B4 minus B1; C6 B5 minus B1; C7 B1n minus B1; C8 B6 minus B1; C9 B7
  minus B1 (if run). Each is read on three
  row sets: the harm rows (target outside every event, at least one lag inside one; for the
  contrast the harm-row difference of the two arms is taken, and the clean-lag rows are reported
  beside it as a reference level), the rows outside events, and all rows. The screen has five test
  months (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), so every number is labelled interim. Harm
  rows carry whole-event bootstrap intervals (95%, resampling events); rows outside events carry
  month-block bootstrap intervals with Bonferroni k = 18 (9 contrasts, 2 groups) and no 99% claim.
  A contrast counts as a gain when its interval excludes zero and the sign agrees in at least 2 of
  3 seeds. Stop rule (from the brief): stop after B1 and B2 if neither beats P0 on the harm rows
  without a loss above 0.05 points on rows outside events; otherwise run B3, B6, B4, B5, and (if a
  slot is spare) B7.
  The lenient rule of the load-shift series applies: stop a step only if it is very clear the path
  is not worth pursuing, and tell the manager first. The smallest gain worth porting is 0.1
  points on the harm rows, because the calibration module adds a second network head and
  injected-event generation to the training loop; a pooled harm-row gain under 0.1 is `small gain`.
benchmark: >-
  Arm P0 (the pooled plain mixture density network with the three origin lags, arm `plain_pooled`
  of experiment 397 stage 2; its stored predictions if the rows, seeds, and folds match, otherwise
  refitted here), and the stored
  forecasts, never refitted, in the private data folder `data/ml-experiments/`: monthly XGBoost
  arm S0M (`00389-xgboost-tail-calibration-and-ensemble-spread-columns/work/arm_preds`), the weekly
  bundle (`00381-simple-weekly-xgboost-demand-against-gam-stack/work`), BG2 (experiment 87), the GAM
  stack (`00377-demand-v1-first-run-feasibility/work`), `manual_heuristic`, and `climatology`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS from the 11 quantiles (as the protocol computes it) with the exact mixture CRPS beside it;
  coverage and pinball loss at q01, q05, q95, and q99, raw and at the 1.15 widening; 5-95% width;
  the lag-encroachment sweep of experiment 395 Part E; and the trust weights.
code: ml-experiments/00404-mdn-lagless-and-calibration-module-trust-weights/
---

**Take-home message (to be written when the result is known).**

This page is the plan. It is committed before any network is fitted, so the commit order proves the
predictions came first.

## Question

**We test whether one neural network can learn how much to trust each lagged power reading, with no
event labels.** A load-shift event moves demand between neighbouring Primaries for hours or weeks.
The power the network sees at the same time yesterday or last week can then be abnormal, and a
forecast that copies that power inherits the abnormality. NGED cannot supply event labels for about
1,300 substations, and every binary event detector tried so far has scored poorly. The maintainer's
design replaces the detector with a continuous trust weight per lagged input, learned end to end.

- **Lagless module.** A mixture density network (three Gaussians) that sees weather, solar
  geometry, calendar, holidays, and a Primary embedding, and no power lag.
- **Calibration module.** A second network that sees the lagless module's forecast, the lagged power
  inputs, and the trust inputs below. It outputs a lagged mixture and one trust weight in [0, 1]
  per lagged input.
- **Output.** A mixture of the lagless and lagged distributions (six Gaussians), weighted by the
  trust weights. It is a mixture of distributions, never an average of quantiles.
- **Training.** The whole network trains end to end on the mixture negative log-likelihood of the
  clean series, on injected load-conserving events. No separate stage and no event label is used.

## Design choices fixed by this plan

**The combining rule, the smoother, and the surprise source are fixed here because the brief leaves
them open.** Each is a stated choice; a different choice is a later experiment.

- **Combining rule.** Each lag token is mixed with a learned null token by its trust weight,
  `t_i * token_i + (1 - t_i) * null`, so a weight of 0 removes the lag's information and does not
  mean zero demand. The weight on the lagged mixture is the mean of the trust weights,
  `w = mean_i t_i`. The final distribution is `w * lagged + (1 - w) * lagless`. Arm B1n (a
  sensitivity arm, contrast C7) replaces the mean by a noisy-or, `w = 1 - prod_i (1 - t_i)`, which
  saturates near 1 as lags are added and so would confound B4. Trust weights start at 0.5 by
  initialisation with no regulariser; the write-up reports whether they stay near 0.5, which is
  the failure experiment 400 found.
- **Trust weight form.** `t_i = sigmoid(g_i(x))`, where `g_i` is a small network on the trust
  inputs,
  the lead features, the Primary embedding, and a signed-MW or MVA-only indicator (experiment 402's
  interim finding is that one pooled weight does not transfer between metering units, so the two
  groups are scored separately).
- **Smoother.** The trust weight is not smoothed at the output. The window statistics of the
  trust inputs already average over 1.5 to 24 hours around each lag, and events start suddenly,
  so a further smoother only adds delay. The write-up reports the trust weight's response at
  event onsets, and the half-hours it takes to cross 0.5 after an injected onset.
- **Surprise source (leakage rule).** The surprise of a lag is taken from an out-of-sample
  lagless forecast, not from the module being trained, because that module has seen the lag's
  target. The plan builds a cross-fitted lagless table: one lagless network per block of 12 weeks of
  the training window, fitted on the rest of the window, and reads the median and the 5-95% band
  width at each lag's valid time from it. The fits exclude the scored month and the 7 days
  before, and the weeks of every labelled real event, so the real events stay out of training. The
  other blocks include later weeks, which a live forecast would not have. The training surprises
  are therefore somewhat cleaner than the surprises at test time, which come from a lagless
  network fitted on the whole training window; the write-up lists this mismatch as a caveat.
  The calibration module reads the lagless module's own forecast (end to end) for the gap input and
  the cross-fitted table for the surprise inputs, and the write-up states which is which.
- **Weather at lag times.** The cross-fitted lagless forecast at a past lag time uses the ECMWF ENS
  mean and spread of the freshest run available at the forecast origin (leads 0 to 24 hours of
  that run), which exists at forecast time. ERA5 is not used. The training table applies the same
  freshest-run rule at every origin.
- **One pass on the ensemble summary.** The network sees the ENS mean and spread of each weather
  variable once, as in experiment 397's `summary` scheme. The 51-member scheme runs only as a check
  on the best arm.

## Trust inputs

- **Per lag.** The single-step surprise (the lag minus the cross-fitted lagless median at the lag's
  valid time) in capacity-normalised MW, kept signed, and the same value in units of the lagless
  band width. Both are given, because a clipped score loses magnitude.
- **Per lag, window statistics** of the surprise over windows centred on the lag (half-widths 1.5,
  3,
  6, 12, and 24 hours, and 84 hours for the week lag only): mean, median, minimum, maximum,
  standard deviation, signed sum, count above the band, and a 12-hour triangular tent mean. The
  origin lag's windows are truncated at the telemetry cut-off.
- **Neighbours.** The same quantities for neighbouring Primaries from `neighbours_nged.json` (in
  the folder of experiment 383), with the signed logic learned and never hand-coded. The neighbour
  inputs are recomputed after injection, with the neighbours' injected series. Bulk supply points
  are added if their stored series exist; if not, the write-up says they were left out. At scale the
  graph must come from the data
  (for example the correlation of forecast residuals); the write-up states what that needs.
- **Lag age and is-missing flag** for each lag, and the gap between the lagged and lagless medians.
- **Experiment 402 finding to test.** The interim finding that the window statistics and the
  neighbour inputs, not the single-step surprise, must improve CRPS gets a nested-input
  ablation of B1 (single-step surprise only; plus window statistics; plus neighbours), reported
  beside the arms, using the best seed count from the screen.

## Arms

All arms use the pooled plain mixture density network of experiment 397 (arm `plain_pooled`) as
their base: one network for all 16 Primaries, a learned Primary embedding, a 3-component Gaussian
mixture head. The maintainer's update of 2026-10-30 moved the base from the uniform-token arm A0 to
the plain network, which scored 3.264 at the signed-MW six and 3.860 at the MVA-only eight against
3.320 and 3.775 for A0 (interim and unaudited, from experiment 397). The lagless module and the
calibration module each get the Primary embedding. The input scheme is the one-pass ensemble mean
and spread. The group gate over input groups stays closed in every arm except B3's lag gate.

- **P0.** The pooled plain network with the three origin lags (the reference).
- **B1.** The architecture above, with the three origin lags.
- **B2.** P0 with token-level lag dropout during training and no separate lagless module.
- **B3.** B1 with attention over which lags to trust: a softmax gate (as in experiment 397 arm A2)
  over the lag tokens, so the gate can learn to ignore suspicious lags.
- **B4.** B1 with more lags: the same weekday and time from each of the last N = 1, 3, and 5 weeks,
  plus the median across the weeks and the median minus the latest week.
- **B5.** B1 without `lag_origin`, the newest reading before the forecast origin.
- **B6.** B1 with a 1D convolutional neural network over the surprise series in a window of 24
  hours either side of each lag (half-hourly steps), in place of the hand-built window statistics.
  The contrast is learned windows against hand-built statistics.
- **B7 (low priority; runs only if a slot is spare).** B1 plus a 1D convolutional neural network
  over the latest 7 days of actual demand up to the forecast origin, with the same trust inputs.
  The network must learn how far to trust that window when the last week holds suspect readings.

The maintainer's expectation (not a result): B1 beats B2.

## Training data and events

**Training uses injected, load-conserving events.** A drop at a donor Primary reappears at the
connected recipients, with the depth as a share of the donor's trailing 30-day mean. Durations and
depths follow the corrected label table `sw_block_detector_label_table_corrected.csv` in the folder
of experiment 395 (use the corrected signs).
In 30% of events no recipient is observed (a guess, varied as a sensitivity). Transient spikes that
are not load shifts are injected too. The training target is the clean series, so a target is never
inside an injected event.
The loss is the mixture negative log-likelihood, which is a proper scoring rule like CRPS and is
what experiment 397 used. It is not a classification score.

**Generator.** Experiment 402 (draft PR 56) is expected to supply the injected-event generator. The
manager confirmed on 2026-10-30 that experiment 402 has none. Its branch holds only the ceilings
script and the counterfactual script. This experiment therefore writes `e00404_inject.py`, and the
write-up says so. If experiment 402 publishes a generator before fitting starts, this experiment
switches to it. The fallback specification is the paragraph
above, with depths and durations drawn by resampling the corrected table's 112 usable rows.

**Real events are a held-out test only.** The label table's events never enter training. The four
scored months hold only 26 real events, so the write-up reports the counts.

## Leakage rules

- Cut every training row whose valid time is on or after the first day of the scored month.
- Truncate every lag window at the telemetry cut-off.
- Take surprises only from the cross-fitted out-of-sample lagless forecast, recomputed against the
  injected lag values after injection (surprises from the clean series would carry no event signal).
- Truncate the trailing 30-day mean that scales injected depths at the forecast origin, and
  inject the donor and its recipients together, so that the neighbour inputs never leak the clean
  value. A truncated triangular tent is renormalised to sum to 1.
- Drop flagged exact-zero half-hours (`substation_zero`) from training, lags, inference, and scoring
  actuals.

## Scoring

- **The harm table** of the lag-gate brief: rows whose target is outside every event but whose lags
  touch one, minus the rows with clean lags, by number of lags inside the event.
- **The lag-encroachment sweep** (experiment 395 Part E harness): the median's share of a synthetic
  step by the number of lags inside the event, the band width ratio, and where the probability mass
  sits.
- **Cost outside events** and the overall five-fold CRPS, per Primary.
- **Tails.** Coverage and pinball loss at q01, q05, q95, and q99, raw and at the 1.15 widening,
  beside every headline score.
- **Trust-weight plots** through a real event, a synthetic event, and a transient spike.
- **Checks.** The median minus the mean forecast per Primary, the error of the mean beside every
  quantile-score contrast, and which audit triggers fired.
- **Folds, seeds, cadence.** The five screen folds of experiment 397 (test months 2025-10, 2025-12,
  2026-01, 2026-03, 2026-06), monthly refits, 3 seeds. More seeds only for a contrast within the
  seed noise.

## Compute

**The slot the manager offers is 1 GPU process under 8 GB of GPU memory, 1 feeder core, and
`MemoryMax=8G`, from about 16:30 UTC on 2026-10-30 if memory then exceeds 15 GB (a stricter slot
than
the brief's 16 GB, because of the machine's load).** No fit starts before the manager grants the
slot. The workstation's load
average was 36.7 and 51 GB of 61 GB memory were in use when this plan was written. Order of work
once granted: the cross-fitted lagless table; P0 (reused if experiment 397's rows match); B1 and B2
on five folds and three seeds; the stop-rule check; then B3, B6, B4, B5, and B7. Each fit is timed
first.

## Problems caused by the move

(The move of the research code from the production repository to this one.) None so far. Experiment
397's modules live on an unmerged branch, so this experiment copies the
files it needs into its own folder.
