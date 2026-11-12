---
id: "00696"
slug: gas-engine-state-tracker-part-3-engine-state-in-the-pre-training-mixer
title: >-
  Does giving the pre-training mixer of the demand neural network the inferred state of D8's gas
  engine, and then fine-tuning with a fresh mixer that has no engine input, lower the error of the
  forecast of D8's metered apparent power?
theme: gross-demand-simulation
also_relevant_to:
  - production-reference-and-best-guess
  - differentiable-physics-embedded-generation
status: planned
kind: other
protocol: demand-v2
size: >-
  Anything else (a new training-only input and a new model class, no new data). Stored: 10 seeds of
  a neural network per arm and fold (state files), one prediction set per arm, the state columns
  per fold, and the encoder-probe fits, all private under
  data/ml-experiments/00696-gas-engine-state-tracker-part-3-engine-state-in-the-pre-training-mixer/.
  More
  than one design is defensible (which state, in which form, into which stage, with or without
  dropout), so the choices are fixed below. Callers: this folder's scripts, which import experiment
  695's tracker states, engine injection, chain scripts, and scorer, experiment 677's step-5 chain,
  experiment 591's model class, experiment 574's fit module, and studies.eval_protocol.
hypothesis: >-
  At D8, giving the pre-training mixer of experiment 677's step 9a neural network the tracked
  gas-engine state at the valid time (the expected engine output as a share of capacity and the
  probability that one or more units run), so that the weather encoder and the date-time encoder
  need not explain the engine's steps, and then training the stage 2 and stage 3 networks with a
  fresh mixer that has no engine input, lowers the CRPS of the forecast of D8's metered apparent
  power against the same neural network pre-trained without the state.
prediction: >-
  Written before any fit, from experiment 695's stored scores and experiment 570's result. (P1) C1,
  arm S1 minus arm A0 at D8 on all scored rows, lies between -0.08 and +0.05 points of capacity,
  with a central guess of -0.02, so the design below does not resolve C1 from zero and the most
  likely verdict is `not supported`. The reasoning: knowing the state at the valid time lowers the
  error by 1.4 to 2.4 points when the state is added to the output, but the state here reaches only
  the weights of two encoders that 16 Primaries share; experiment 695's cleaned arm, a stronger
  intervention on the same encoders, gained 0.11 points on the rows where the engine is tracked as
  off and lost 0.55 where it runs; and experiment 570's pre-training-only price encoder made the
  fine-tuned forecast worse. (P2) S1 scores 0 to 0.10 points better than A0 on rows tracked as off
  and 0 to 0.15 points worse on rows tracked as running. (P3) The placebo arm SP scores within 0.05
  points of A0. (P4) At the host, the arm with the true state, WST, scores between 0.10 points
  better and 0.03 points worse than W0, and the arm with the tracked state, WS1, scores within 0.05
  points of WST. (P5) In the encoder probe M1, a fresh mixer on WST's frozen encoders predicts the
  host's reading without the injected engine with a lower mean absolute error than the same mixer
  on W0's frozen encoders, by 0.02 to 0.3 points of capacity. (P6) No Primary other than D8 moves.
decision_rule: >-
  Fixed before any fit. Metric of record: CRPS of the forecast of metered apparent power at D8, in
  points of effective capacity (capacity fixed at 10.379 MVA for every arm), leads 24 to 240 hours
  from the 12:00 UTC contract origin, origins 2025-11-01 to 2026-06-30, valid times before
  2026-07-01. Control: arm A0, experiment 695's refitted step 9a (seeds 0 to 2 reused, seeds 3 to 9
  fitted here by the same chain). Smallest gain worth porting: 0.16 points at D8. The change adds a
  fitted tracker with per-era parameters to the training pipeline of one Primary and nothing to the
  serving path. demand-v2's targeted path judges a change aimed at one Primary at that Primary,
  with a minimum of 0.02 points for a new fitted step. This plan asks for 8 times that figure, so
  that the change also moves the headline of the MVA-only eight by 0.02. The 0.16 threshold is
  therefore stricter than demand-v2 requires, and it follows the precedent of experiment 695, which
  set its threshold at D8 from the same headline. Four planned contrasts are read at 98.75%
  (Bonferroni for four). C1: S1 minus A0, D8. C2: S1 minus SP, D8. C3: WST minus W0, host. C4: WS1
  minus W0, host. Each contrast has three decision intervals: a paired t interval on the 8 monthly
  differences of the seed means, a bootstrap over whole origin months of the seed-averaged losses
  (40,000 resamples, 250 draws per tail), and a paired t interval on the 10 per-seed mean
  differences. The library interval, which draws one seed per resample, is printed beside them. A
  contrast "passes" when all three decision intervals lie below zero. M1 "passes" when the 95%
  paired t interval of the probe's mean absolute error, WST minus W0, over the 8 folds' seed means
  lies below zero. Verdict, first branch that applies. (1) `invalid` if a leakage test fails:
  weather after the origin, a state or a pre-training row that used a reading after its cut, or a
  stage 2 or stage 3 forward pass that reads a state column. (2) `failed` if the reproduction gate
  G0, the identity control, the liveness check, or an alignment test cannot be met after
  investigation. (3) `not supported`, with audit trigger 5 fired, if C1 passes and C2 does not,
  because a gain that the placebo matches is not information about the engine. (4) `inconclusive`
  if C1 and C2 pass and M1 fails, because the gain then has no measured mechanism. (5) `supported`
  if C1 passes with an estimate of -0.16 or beyond, C2 passes, M1 passes, the C3 intervals do not
  all lie above zero, the normalised MAE of the median at D8 does not rise beyond its 95% interval,
  and no other Primary moves for the worse. (6) `partly supported` if C1 passes at -0.16 or beyond,
  C2 and M1 pass, and another condition of branch 5 fails. (7) `small gain` if C1 passes with an
  estimate short of -0.16 and C2 and M1 pass. (8) `refuted` if all three C1 intervals lie above
  zero. (9) `not supported` in every other case, the default branch. A Primary other than D8
  "moves" by experiment 695's rule: the 95% interval of its paired difference excludes zero and the
  absolute estimate exceeds its stored sister-arm difference from experiment 677.
benchmark: >-
  Arm A0: experiment 695's refitted raw arm (experiment 677's step-5 network at monthly refits,
  step 9a), read from data/ml-experiments/00695-.../runs/A0 for seeds 0 to 2 and extended here to
  seeds 3 to 9. At the host, arm W0 of experiment 695, extended the same way. Reference rows on the
  same half-hours: experiment 695's A1 and its bounds, the manual heuristic, and
  ClimatologyForecaster on the raw reading, repeated once per seed. Production does not run BG2.
row_set_id: >-
  backtest_year_2025_11_to_2026_10_daily_12utc, restricted to origins 2025-11-01 to 2026-06-30 and
  valid times before 2026-07-01 (the rows of backtest_year_2025_11_to_2026_06_daily_12utc), all 16
  Primaries scored, with D8 as the decision series
metric: >-
  CRPS (twice the mean pinball loss over 11 quantiles) and normalised MAE of the median, in points
  of effective capacity, against metered apparent power; twCRPS; coverage and pinball loss at q01,
  q05, q95, and q99; by lead band, by engine months and fade months, by tracked engine state, and on
  cold winter nights; all imported from studies.eval_protocol through experiment 695's scorer. The
  avoidable-curtailment cost is not defined at an MVA-only Primary and is not reported.
code: ml-experiments/00696-gas-engine-state-tracker-part-3-engine-state-in-the-pre-training-mixer/
---

**This file is a plan in the shape of a planned `result.md`, and nothing in it has run.** The
maintainer asked for the experiment by name. D8 is a Primary substation with a gas engine behind
its meter, and "Gas A" is the anonymised label of the one engine record we hold. The plan never
shows an engine series with dates. Links are written for the page's place under
`docs/ml-experiments/00696-gas-engine-state-tracker-part-3-engine-state-in-the-pre-training-mixer/`.

## What earlier experiments tried

**No experiment has given a forecasting model the engine state as a training-only input, but
experiment 570 ran the same design with electricity price in place of the engine state, and the
forecast came out worse.**
[Experiment 570](../00570-price-encoder-part-1-known-answer-test-and-pretrained-stack/result.md)
put a price encoder beside the weather encoder and the date-time encoder in pre-training, and
discarded the price encoder before fine-tuning, expecting the two kept encoders to come out
cleaner. Its verdict is `not supported`. After fine-tuning, CRPS was 0.29 points worse than the
network pre-trained without price at the six Primaries metered in MW (interval +0.05 to +0.56), and
0.09 worse at the eight metered in MVA (interval -0.02 to +0.18). An encoder fed noise gave 61% of
the harm at the six, and an encoder fed a constant gave 43%, so most of the harm came from the
extra part and not from the price. Experiment 570 is the closest precedent we found, and it is the
reason this plan has a placebo arm, a zero-state control, and a small predicted effect.

**Experiment 695 cleaned the target and added a calendar average back, and fed no state to the
network.**
[Experiment 695](../00695-gas-engine-state-tracker-part-2-cleaned-target-and-climatology-add-back/result.md)
(PR 382) removed the tracked engine output from D8's training target and lagged inputs in
all three training stages, and added a calendar average of the engine back to the output. Its
bounds added the tracked state at the valid time to the output. Its write-up names origin-time
state features as an option for a part 3, and that option is still not run.

**Three earlier experiments fed an engine state or a detected step to a model, and each used the
state at forecast time.**

- **Experiment 294** added one term to a generalised additive model with a Kalman filter: a break
  indicator and the detector's probability that the engine runs. With the state at the target
  half-hour the score fell by 24%, but the detector builds that state from the reading being
  forecast. With the state of the last settled day and a coefficient fitted on the contemporaneous
  state, the score was 10% worse. Trained on the day-old state, the term gained 3% (post hoc).
- **Experiment 361** removed or masked the steps that a plain residual-median rule found in D8's
  training target of the GAM stack. The score moved by -0.007 points (99% interval -0.097 to
  +0.053).
- **Experiment 363** gave the GAM stack a penalised level for each stretch between detected steps,
  fitted jointly with the weather and calendar terms. The score rose by 0.291 points, because the
  latest stretch's level is a noisy guess for the following weeks.

**The maintainer's idea differs from those three in one respect: the state never reaches a
forecast.** Experiment 294's finding is that a contemporaneous state is a tracker artefact when a
forecast reads it. Here the state is a covariate of the pre-training loss only. The method is
learning using privileged information
([Vapnik and Vashist (2009)](https://doi.org/10.1016/j.neunet.2009.06.042);
[Lopez-Paz et al. (2016)](https://arxiv.org/abs/1511.03643)).

## Where the mixer lives in step 9a

**The network already has two mixers, and stage 2 already starts a fresh mixer.** The class is
`e00591_model.AuxLagMdn`, built by `e00591_arm.new_model` and driven by `e00574_fit.py` under
`e00677_k1.py`.

- **Stage 1 (pre-training, `pretrain_wt`)** trains every part on ERA5 and CAMS rows up to 3.5 hours
  before the refit: the Primary embedding, the weather encoder, the date-time encoder, the lag
  encoder, the offset knots, and `mixer_pre`. `mixer_pre` reads the two encoders' outputs, the lag
  encoder's output, and the embedding.
- **Stage 2 (`step2`)** loads the pre-trained weights (`load_pretrained_fresh`,
  `e00572_model.load_lag_pre_into_ft`), gives `mixer_ft` a fresh seed-12345 initialisation
  (`fresh_mixer`), freezes the weather encoder and the date-time encoder, and trains the rest on
  ECMWF ensemble rows with uniform weights. `mixer_ft` also reads 7 spread columns and 2 lead
  columns. `mixer_pre` is loaded and never called again.
- **Stage 3 (`fine_tune`)** loads the stage 2 file strictly, keeps the two encoders frozen, and
  trains the rest with recency weights, a 7-day embargo, and injected lag-only shifts.

**So the idea needs a change to `mixer_pre` alone.** The weather encoder and the date-time encoder
reach every forecast with their pre-training weights. The embedding, the lag encoder, and the
offset knots start stage 2 from their pre-training weights and stay trainable.

## Arms

| Arm | Series | What differs from A0 | Seeds | Role |
|---|---|---|---|---|
| A0 | D8 | Nothing (experiment 695's raw control) | 0 to 2 reused, 3 to 9 new | Control |
| S1 | D8 | `mixer_pre` reads D8's tracked state at the valid time | 0 to 9 | C1, C2 |
| SP | D8 | `mixer_pre` reads a day-block permutation of that state | 0 to 9 | C2 |
| SZ | D8 | `mixer_pre` reads a state of all zeros | 0 to 2, fold 2025-11 | Identity control |
| W0 | Host D4 | Raw reading with Gas A injected (experiment 695) | 0 to 2 reused, 3 to 9 new | Control |
| WST | Host D4 | `mixer_pre` reads the true injected state | 0 to 9 | C3, M1 |
| WS1 | Host D4 | `mixer_pre` reads the tracked state of the injected reading | 0 to 9 | C4 |
| S1d | D8 | S1 with the state hidden on half of the origin days | 0 to 2 | Descriptive |

**The state input is three columns, appended to the pre-training rows only.** The columns are the
expected engine output as a share of capacity (the sum over unit counts j of j times the era's
unit size times P(k = j), over the capacity), the probability that one or more units run, and a
"hidden" flag. The 15 Primaries without a tracker read zeros.

**The state enters at the valid time.** A valid-time state is legitimate here because the
pre-training target is the metered reading and the state is a covariate of that loss. A state at
the lag times would be a different experiment, because the forecast could also read it.

**Arm SP tests whether a gain is information about the engine or an effect of the extra input.** SP
permutes whole days of S1's state columns within calendar month and within weekday or weekend class
(seed 0), using only days before the fold's cut. SP keeps the state's seasonal profile, its
calendar average, and its within-day shape, and loses which days the engine ran. Experiment 570's
noise-fed encoder is the reason SP sits in the verdict table.

**Arm S1d is descriptive and decides nothing.** S1d hides the state on a random half of origin days
(zeros, with the flag set), so the encoders must still carry the engine's calendar pattern on
those rows. The fresh mixer drops the input in every arm, so the dropout is not needed for
inference.

**A stage-2 variant is left out.** Stage 2 freezes the two encoders, so a state in the stage 2
mixer acts on the lag encoder, the embedding, and the offset knots. That is a second hypothesis,
and it needs a fresh mixer at stage 3 and a control of its own.

## Data

- **States:** experiment 695's stored per-fold files, `states/A1/<month>.npz` for D8 and
  `states/W1/<month>.npz` for the host (the smoothed marginals P(k = 0 to 3) on the pipeline grid),
  with the unit sizes from the fold's era 2 fit and the cached era 1 fit. No tracker is refitted.
  The true host state comes from `e00695_engine.drive_series` (seed 0).
- **Power:** the cleaned power stack (`e00510_common.clean_stack`), unchanged for the D8 arms. The
  host arms read experiment 695's `stacks/W0` row (D4 with Gas A injected) in all three stages.
- **Weather:** ERA5 and CAMS in pre-training; the ECMWF ensemble 00Z run of the origin day in
  stages 2 and 3 and at inference, treated as available at 09:00 UTC.

## Training

- **Everything outside `mixer_pre`'s input is step 9a unchanged:** widths, loss, optimiser, epochs,
  early stopping on the held-out weeks, lag block dropout, the 7-day embargo, the injected lag-only
  shifts, and monthly refits for the 8 test months 2025-11 to 2026-06.
- **The wrapper replaces `e00591_model.AuxLagMdn` with a subclass, and copies nothing.**
  `e00591_arm.py` assigns its own `new_model` to `e00574_fit.new_model` when `e00677_k1.py` imports
  it, so a patch of `e00574_fit.new_model` made before that import is overwritten and S1 would
  train as A0. `e00591_arm.new_model` looks `AuxLagMdn` up at call time, so the new `eNNNNN_k1.py`
  replaces the class on the `e00591_model` module and then runs `e00677_k1.py`, as `e00695_k1.py`
  does for the power stack.
- **The subclass widens `mixer_pre` by 3 inputs at all three stages.** `load_lag_pre_into_ft` loads
  `mixer_pre` with `strict=False`, which does not forgive a shape mismatch, and stage 3 loads the
  stage 2 file strictly. The subclass's `forward` strips the 3 state columns first, because
  `AuxLagMdn.forward` treats 2 trailing columns as auxiliary labels.
- **The wider first layer is built at the original width and then padded with zero columns.** A
  layer built 3 inputs wider would draw different weights for the existing columns and shift the
  random-number stream of every part built after it. The subclass builds `mixer_pre` exactly as
  `AuxLagMdn` does, under the same seed, and then concatenates 3 zero columns to the first layer's
  weight, as `AuxLagMdn.load_state_dict` already does for the detector input.
- **The state columns are looked up from the rows' own indices.** `pretrain_wt` deletes its rows
  object before the seed loop, so a pass-through wrapper of `block_of`, which receives the rows'
  Primary and target index, keeps the two arrays.
- **The pre-training tensor is widened once per fold.** The wrapper of `train` allocates one tensor
  3 columns wider on the first seed, writes the state columns once, and copies the other columns
  into it in place for each later seed. Measure the first job's peak GPU memory before running 10
  jobs together, and lower the worker count if two tensors of about 7.2 million rows do not fit.
- **Ten seeds, in groups.** Each (arm, fold, seed group) is one job with its own run folder under
  `E00695_ROOT`, so jobs of one arm never share a log or a checks file. `e00695_chain.sh` and
  `e00695_arm.sh` run the chain.

## Scoring and contrasts

- **Scorer:** a new `eNNNNN_score.py` imports `read_fits`, `row_scores`, `contrast`,
  `other_primaries`, `slice_scores`, `host_slices`, `assert_origin_time_weather`, and
  `climatology_reference` from `e00695_score.py`. The add-back is not used, so every arm's
  forecast is the network's own 11 quantiles.
- **Family:** C1 to C4 at 98.75%, as the decision rule states. Descriptive at 95%: SP minus A0,
  WS1 minus WST, S1d minus A0, C1 by engine months and fade months, C1 by tracked state, the MAE
  of the median, and S1 minus A0 at each of the other 15 Primaries.
- **Resolvable gain, best case.** In experiment 695 the paired difference of single seeds within a
  month had a standard deviation of 0.249 points. If seeds are independent between folds, 10 seeds
  give a standard error of the 8-month mean of about 0.028 points and a half-width at 98.75% of
  about 0.09, which resolves a gain of 0.16.
- **Resolvable gain, worst case.** The standard deviation of experiment 695's 8 monthly seed-mean
  differences was 0.138, and the November fold (+0.427) carries it: without November it is 0.060.
  That spread is therefore partly a difference between months, which more seeds do not shrink. If
  the 0.138 does not shrink, the half-width is 0.163 and a gain of 0.16 is not resolvable. The
  fade-month seeds of experiment 695 (+0.055, +0.007, +0.146) also spread more than the monthly
  figures allow, which suggests a seed effect that persists across folds. The interval over the 10
  per-seed means guards against that case. Neither case resolves the central guess of 0.02.

## Controls and checks

- **G0, reproduction:** refit A0 seed 0 of fold 2025-11 in the new run folder and match experiment
  695's stored fit to 1e-4 of capacity.
- **Identity, arm SZ:** forecasts match A0's to 1e-4 of capacity for fold 2025-11, seeds 0 to 2.
  With zero inputs the padded weights receive no gradient and no weight decay, so they stay at
  zero; the tolerance allows for a matrix product that rounds differently at the wider shape.
- **Liveness:** after pre-training, the first-layer `mixer_pre` weights on the state columns are
  non-zero for S1 and WST, and exactly zero for SZ. A zero in S1 means the patch did not take.
- **Alignment:** for 10,000 sampled pre-training rows of D8 per fold, the state columns equal the
  fold's state file at the row's target index; the same rows of the other Primaries read zeros;
  and every row read has a target index more than 7 half-hours before the refit, because the state
  file switches to full readings from the refit on.
- **M1, encoder probe at the host (known answer).** The probe follows the linear probe A6-2 of
  experiment 570. For each fold and seed, freeze the pre-trained weather encoder, date-time
  encoder, and embedding of W0 and of WST. Fit a fresh mixer with no state input and no lag input
  (2 hidden layers of 64) on D4's reading without the engine, on the pre-training rows outside the
  held-out weeks, stopping early on half of the held-out weeks. Score the mean absolute error, in
  points of capacity, on the other half. M1 passes when the 95% paired t interval of WST minus W0
  over the 8 folds' seed means lies below zero. M1 tests the premise that the encoders describe
  demand better when the pre-training mixer absorbs the engine. Branch 4 says what a failure means.
- **No engine input at forecast time:** assert that the stage 2 and stage 3 tensors have A0's
  column layout.
- **Leakage:** rerun experiment 695's cut tests (`check_cut_at_train_end`, `check_real_stack_cut`)
  on the state columns: perturbing readings from 3.5 hours before the refit onward changes no
  state column and no pre-training row. Assert origin-time weather for every origin of every arm
  with `assert_origin_time_weather` (242 origins per arm).
- **Checks list:** the median minus the mean forecast per site, the MAE contrast beside every CRPS
  contrast, pre-training epochs and validation loss per arm, and the audit triggers.

## What could leak, and what is only a risk

- **The state at the valid time is built from the target reading, by design.** The smoothed state
  at time t reads the reading at t and up to 20 hours after. `mixer_pre` can therefore read part
  of its own target. The weights that carry forward saw only readings before the cut, so no scored
  reading reaches a forecast, and the pre-training validation loss is not comparable between arms.
  Experiment 570's price arms stopped pre-training after 22 to 26 epochs against 40.5 for the
  control, so an earlier stop is a second route by which S1 could differ from A0.
- **A false engine flag is correlated with the target.** Where the tracker reads a real demand dip
  as the engine, the state excuses the encoders from explaining that dip. WS1 against WST measures
  this at the host; at D8 it cannot be measured.
- **The fresh mixer may lack the engine's calendar pattern.** The frozen date-time encoder no
  longer needs to encode when the engine tends to run, yet the forecast target still contains the
  engine. This is the mechanism by which S1 could score worse on running rows (P2).
- **The tracker is refitted per fold.** Era 2's parameters and unit size change by fold, so the
  state columns differ between folds for the same half-hour. Each fold reads only its own file.
- **ERA5 arrives about 5 days late.** Pre-training rows up to 3.5 hours before the refit use ERA5,
  as in A0. Both arms share this property of step 9a, so it does not bias a contrast.

## Standing practices and earlier advice

- **Followed:** monthly refits; cleaned data with `substation_zero` half-hours dropped and the count
  reported per fold; D15 and D18 cut before 2025-03-01; every fitted network saved; stored
  predictions reused as controls (695's A0 and W0); tails and cold winter nights reported; a
  day-ahead and a 14-day example forecast plotted; the draft pull request; the climatology row; no
  scored valid time on or after 2026-07-01.
- **Followed from earlier advice:** compare an added input with a control of the same size (570);
  keep the encoders frozen in fine-tuning and keep the spread columns out of the mixer (574); keep
  the synthetic shifts in the fine-tune only, and do not freeze a first mixer stage (603); never
  score a valid-time state as a forecast (294, 695); read a seed-averaged month bootstrap beside
  the library interval (695); use enough seeds for a contrast under 0.1 points (569, 586); test
  against an injected real record (694).
- **Departed from:** "train an engine-state coefficient on the state a forecast will see" (294)
  holds only in spirit, because no forecast sees a state here. The four-month screen is replaced by
  scoring at every Primary, as the stop rule for blind alleys requires. The four load-shift
  scenarios are not reported, as in 695, because the injected shifts are an unchanged part of every
  arm.
- **Other evidence for a small effect:** in experiment 564, arm M2-b, which added extra series to
  pre-training only, left those series near or worse than a climatology.
- **Stop rule for blind alleys:** six or more experiments on D8's gas engine ended `not supported`
  (264, 265, 294, 361, 363, and 695). The maintainer asked for this experiment by name, which is
  the recorded approval, and all 16 Primaries are scored.

## Compute budget and stop rules

- **Budget:** about 432 fold-and-seed chains for the family's arms, 24 for S1d, and 3 for SZ, plus
  160 small probe fits. Experiment 695 ran about 171 chains in 1.5 hours of wall-clock time on the
  one A6000 (about 0.25 hours per arm with 8 to 10 jobs together). The estimate is 4 to 5 hours,
  with a cap of 10. No money is spent and no data is downloaded.
- **Stop before the main run** if G0, the identity control, the liveness check, or an alignment
  test fails, or if a leakage test gives a change other than 0.
- **Stop at the cap** between jobs, never inside a fit, and report the arms that finished.

## Audit triggers expected

**Trigger 5 fires if SP matches or beats S1 while C1 passes, and trigger 7 applies if a later
design depends on C1's sign.** A gain at D8 with no gain at the host with the true state (C1
passes and C3 does not) contradicts the known-answer arm, and gets an audit before it merges.
