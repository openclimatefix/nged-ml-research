---
id: "00518"
slug: history-encoder-first-fit-stage-0-time-and-accuracy
title: >-
  Can a small neural network learn to find and size a shifted block in 14 days of power readings, and
  how long does one fit take?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
date: 2026-11-02
size: >-
  complex. Stored: one saved neural network (about 87,000 parameters), its training log, and score
  tables, all under the experiment's private data folder. Serving path: not touched (this
  repository holds none). Degradation rule: not touched. Designs: more than one design would
  defensibly test the hypothesis (how a step is read out of the network, the positive-unlabelled
  prior, the input channels), so the plan fixes each. Callers: the scripts import experiment 507's
  generator and loaders, and the callers of that chain cannot be named without searching. Reviews:
  one Sonnet plan review and one Sonnet code review before any fit, then a Sonnet results review
  and two Sonnet write-up reviews. The brief asks for Sonnet only.
hypothesis: >-
  A history encoder of about 90,000 parameters, trained on injected worlds of experiment 507's
  generator (families A and B) on fold 2026-01, finds injected steps and sizes them well enough to
  remove most of the error that a step puts into the three lag readings, without flagging many
  steps on real windows.
prediction: >-
  Written before any fit. Scores are on the 372 test windows of fold 2026-01 (origins in January
  2026, no labelled event within 48 half-hours of the history) in injected worlds 200 to 202. P1.
  The validation step loss keeps falling past epoch 10 (best epoch 8 or later). P2. The repaired
  lag error on rows where a step touches a lag is under half of the unrepaired error, and the mean
  absolute estimate on rows where no step touches a lag is under 0.1 of the mean injected step. P3. The
  encoder finds at least 70% as many steps within 6 hours of their onset as a perfect predictor
  would (the ceiling printed beside it). P4. The planted-step precision at a 2% planting rate is at
  least 0.3. The thresholds in P2 to P4 are guesses, not values from an earlier result. Any of the
  four can fail.
decision_rule: >-
  Fixed in the section "Decision rule", below, before any score is read. The plan's kill rule (c)
  has two clauses. The second clause (validation step loss stops falling before epoch 10) is read
  exactly. The first clause needs the oracle's noise curve, which this experiment does not run, so
  it is read through the proxy P2 to P4, chosen before the fit. The proxy is not the noise-curve
  region itself, and the neural Discuss session decides what follows.
benchmark: >-
  A perfect predictor (the injected step itself, read through the same candidate rule) and a zero
  predictor (no repair), both printed beside the encoder on the same windows and worlds. No
  forecast is scored, so there is no forecast benchmark.
row_set_id: injected_windows_fold_2026_01_worlds_200_to_202  # nothing is forecast, so no row set is stored
metric: >-
  Share of injected steps found within 2, 6, and 24 hours of their onset; median relative size error
  of the found steps; mean absolute error of the three lag readings in spreads, repaired and
  unrepaired; false-flag rate on real windows; planted-step precision at a 2% rate; area under the
  ROC curve of the genuine-change head.
code: >-
  ml-experiments/00518-history-encoder-first-fit-stage-0-time-and-accuracy/
---

**This page is the plan; the write-up replaces it when the experiment finishes.** The plan follows
the brief `briefs/history_encoder_first_test_brief.md` with both sets of edits of 2026-11-02 (step
T1 only), sections 2.2 to 2.8 of `briefs/history_encoder_debate_3_opus_reply.md`, and the merged
[experiment 507](../00507-history-encoder-first-test-oracle-repair-and-total-variation/result.md).
Rung R1 and later rungs are not planned and never started, and no oracle noise curve is run.

## Question

Experiment 507 showed that a perfect repair of the three lag readings improves the lag correction,
and that one non-learned repair did not. This experiment asks whether a small neural network can be
trained to find and size the shifts, and measures how long one such fit takes, so that the Discuss
session can decide whether the learned encoder is worth a further rung.

## Two limits to read first

- **The total-variation test of experiment 507 is weak.** Its tuning hit the edge of the grid
  (weight 16 and tau 5, the largest of each, which is the weakest repair), so the 507 result cannot
  say whether any non-learned repair could work. A learned encoder is therefore not compared with a
  well-tuned non-learned repair here.
- **The lab setting flatters detection.** The lagless median comes from ERA5 and CAMS weather that
  already happened (experiment 507's table, one seed), so the reading minus the median is smaller
  than at serving. A single-seed fit is informative only when its result is far from the thresholds.

## Design

### Data and split (fold 2026-01, refit 2026-01-01)

- **Inputs.** Experiment 507's injected-world generator, families A (temporary) and B (genuine) and
  family D (hard negatives, drawn three times as often). Family C and the reading-class head are
  dropped, as are the missing-mask, flagged-zero, and age channels. The two input channels are the
  surprise (shifted reading minus the lagless median, over the Primary's residual spread, clipped
  at 30 spreads and divided by 10) and the lagless median standardised within the window. Readings
  or medians that are missing enter as zero in both channels.
- **Windows.** One window per Primary and day: the 672 half-hours (14 days) ending 7 half-hours
  before the 09:00 UTC origin. The median must be finite at 90% of the half-hours. A window is
  dropped if its history, padded by 48 half-hours, touches a labelled load-shift event. D15 and D18
  need a history that starts after 2025-03-01.
- **Split.** Test windows: origins in January 2026. Embargo: no training or validation window ends
  within 24 days before the refit. Validation block: the 30 days before the embargo. Training
  windows end 14 days before the validation block begins.
- **Worlds.** Training uses a fresh generator seed (1000 plus the epoch number) for each epoch.
  Validation uses worlds 150 to 152, scoring uses worlds 200 to 202, and none of these is a world
  that experiment 507 trained or tuned on (100 to 102, 110).
- **Targets.** Signed MW at the six Primaries, apparent power at the eight (the shift is injected
  as the magnitude of the sum, and half-hours where the lagless median is below twice the injected
  step are masked from the step loss; the share masked is printed). D15 and D18 are scored as a
  third, report-only group.

### The encoder

- **Architecture.** A dilated convolution stem of width 32, a strided convolution to 168
  two-hour tokens of width 64, two transformer layers of width 64 with 4 heads and a learned
  relative-position bias, and a head that joins each token to the stem features of its four
  half-hours. The encoder has 86,747 parameters and writes, per half-hour, the
  probability that a step is in force and the step size in spreads, and, from the last token, the
  probability that the step in force at the newest reading is genuine.
- **Losses (weights fixed before the fit).** `L_u` (weight 0.5): a non-negative
  positive-unlabelled risk on the step probability, plus a Huber loss on the size at injected
  half-hours only; `L_g` (0.2): cross-entropy of the genuine-change head on windows with a step in
  force at the newest reading whose onset lies inside the window; `L_tv` (1e-3): L1 on first
  differences of the estimated step. The positive-unlabelled prior is 0.03, a guess, and is not
  swept here.
- **Training.** Adam at 2e-3, batch 128, epochs of 16,384 windows drawn with the family D weights,
  at most 60 epochs, early stopping 8 epochs after the best validation step loss (a loss counts as better only if it is 0.5% below the best) and never before
  epoch 10. One seed. A feeder thread builds the next epoch's world on one core while the device
  trains.
- **Cap.** The fit aborts, and the worker messages the manager and the neural Discuss session, if
  the time spent passes 2 hours or the time per epoch (after the first epoch) extrapolated to 60 epochs does.

### What is read

On the 372 test windows (124 at the six, 204 at the eight, 44 at D15 and D18), in worlds 200 to 202,
with intervals from resampling whole windows (the three worlds share each window):

- **Found within 2, 6, and 24 hours.** The share of injected events with a candidate onset within
  that time of their onset, with the ceiling (the same score for a perfect predictor) beside it.
  Candidates are defined in `e00518_score.py`, on the estimated step (the probability of a step
  times its size, in spreads); a step smaller than 0.5 spreads is never a candidate. Half-hours
  masked at the eight are excluded from the step loss only, and are scored.
  A step is missed if no candidate lies within 24 hours.
- **Size error.** The median relative difference between the estimated and injected step over the
  event's half-hours, among found events.
- **Repaired lag error.** The mean absolute error of the estimated step at the three lag positions,
  in spreads, against the mean absolute injected step, on rows where a step touches a lag, and the
  mean absolute estimate on rows where none does (what the repair adds).
- **Planted-step precision at 2%.** `0.02 r / (0.02 r + 0.98 f)`, where `r` is the share found
  within 2 hours and `f` is the share of real windows (no injection) with a candidate. The real
  windows hold events that NGED did not log, so `f` is an upper bound on the false-step rate. The
  ceiling of `r` (the perfect predictor) is printed beside P4, and P4 is flagged "ceiling-limited"
  if that ceiling is under 0.5. The 372 windows overlap within a
  Primary, so intervals resample whole Primaries, not windows.
- **Genuine head.** The area under the ROC curve on windows that carry a label. The generator makes
  75% of steps in force at the newest reading genuine (experiment 507 measured 79%).
  The genuine head, the median size error, and the 24-hour found share carry no threshold and are
  diagnostics.
- **Limits of the test windows.** After the labelled-event exclusion the test month holds 4
  Primaries at the six (D5 and D16 are excluded entirely) and 7 at the eight. 78% of the test
  half-hours take their median from the table's post-refit module, while training and validation
  take theirs from cross-fitted modules, so the test surprise is quieter. The false-flag rate is
  therefore also printed on the validation block's real windows, which share the training
  provenance. A candidate cannot lie within 4 half-hours of either end of the window.
- **P4 depends on the training mix.** 19% of the valid training half-hours are positives, against
  at most a few per cent in real data, so the step probability is calibrated to a busier world
  than the real one. Precision, recall, and the false-flag rate are also printed at candidate
  thresholds of 0.5, 0.75, 1.0, and 1.5 spreads, as a sensitivity (the P4 threshold stays at
  0.5, fixed now). The positive-unlabelled clamp stops penalising false positives below about
  `prior` times the positive loss, and low-median half-hours that no injection touched stay in
  the unlabelled set.
- **Validation step loss by epoch**, and the time per epoch of the feeder and of the device.

## Decision rule

Fixed before any fit. In the order of the brief's kill rule (c):

1. **The encoder's validation step loss stops falling before epoch 10.** The best epoch of the
   validation step loss (fixed worlds 150 to 152, no smoothing) is under 8: the neural encoder
   stops. A best epoch of 8 to 12 is reported as inconclusive, because one seed and a validation
   block of 401 windows cannot separate epoch 9 from epoch 10.
2. **The first clause of (c) asks whether the errors fall in the region of the oracle's noise curve
   that keeps under half of the oracle gain.** The noise curve is not run here. The proxy chosen
   now is: the encoder passes if P2, P3, and P4 all hold. Failing any one is this plan's reading of
   the first clause, so the neural encoder stops unless the neural Discuss session overrules. The
   thresholds are not mapped to the noise curve's points: P3 (70% found within 6 hours) resembles
   its joint point (30% of steps missed, 6-hour onset error) and P2 (error under half) resembles a
   50% size error, but only roughly. A pass shows that the learned encoder reaches an accuracy
   worth costing a noise curve and a retrained correction. A pass does not show headroom in forecast
   score. The smallest result worth acting on is a pass of all three, because the encoder adds a
   network and a 672-half-hour history store to serving.
3. **A single-seed null is read only if far from the thresholds.** A threshold is met if the point
   estimate meets it. If the 95% interval spans the threshold, the result is reported as
   "inconclusive". The three thresholds are three planned tests, and the intervals are not adjusted
   for multiplicity.
4. **If the fit aborts at the cap,** P1 is not read, the partial scores are reported, and the page
   has status `failed`.

## Cost

One encoder fit and one scoring pass, on one GPU process under 8 GB (`MemoryMax=8G`,
`OMP_NUM_THREADS=1`) and one feeder core. No other fit runs: no noise curve, no retrained
correction, no total-variation retune. The cap is 2 GPU-hours, and about 2.3 of the original 3
GPU-hours remain. On the shared workstation, with other jobs running, one core takes 0.20 ms to
build a window and 1.06 s for one training step of 128 windows, so one epoch on a single core
would take about 135 s. These CPU figures are an upper bound and were not measured on the GPU,
which stays untouched until the manager releases it.

## Standing practices

- **Followed:** cleaned data in training and scoring; the D15 and D18 start cut; model saved to the
  private data folder; reuse of experiment 507's generator, loaders, and lab lagless table; the
  unlogged-event caveat; PR opened as a draft.
- **Not applicable:** monthly refit and the forecast-score practices (no forecast is scored), the
  climatology row, plots of forecasts.
- **Advice from earlier results followed.** Experiment 507's advice to score a repair on rows whose
  lag positions a shift touches (split into temporary and genuine) is followed for the lag error.
  Its advice to run the noise curve before building the encoder is **not** followed: the manager
  withheld the noise curve for this experiment.

## Implementation details (deleted when this ships)

- Scripts: `e00518_model.py` (encoder and losses), `e00518_data.py` (windows, split, worlds),
  `e00518_score.py` (candidates and metrics), `e00518_fit.py` (commands `selftest`, `counts`,
  `timing`, `fit`, `score`), and `run.sh`, which puts every earlier experiment folder on
  `PYTHONPATH` and runs under `MemoryMax=8G`.
- The scripts import experiment 507's `e00507_common`, `e00507_generator`, and `e00507_lagless`
  (and, through them, experiments 87, 397, 402, 404, 418, 428, 435, and 449). Nothing from 507 is
  edited.
- Outputs go to `data/ml-experiments/00518-history-encoder-first-fit-stage-0-time-and-accuracy/`:
  `fit_log.csv`, `fit_summary.json`, `models/h_stage0_2026-01_seed0.pt`, `t1_score.json`.
