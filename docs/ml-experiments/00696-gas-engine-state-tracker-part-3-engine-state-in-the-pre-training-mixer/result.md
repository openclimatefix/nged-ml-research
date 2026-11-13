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
status: done
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
  all lie above zero, the normalised MAE of the median at D8 does not rise (the code reads "rises"
  as the lower end of the 95% month-bootstrap interval of the S1 minus A0 MAE contrast lying
  above zero),
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
verdict: not supported
best_arm: >-
  None is better than the control. The arm with the state, S1, scores 7.299 points of capacity at D8
  against 7.290 for the control A0, and the placebo SP scores 7.301.
headline: >-
  Pre-registered branch 9, `not supported`. Giving the pre-training mixer the engine state left the
  forecast of D8 unchanged: C1 is +0.009 points of capacity, and its decision intervals end at
  -0.074 at the lowest, so any net gain above about 0.07 points is ruled out, against a threshold of
  0.16 and a predicted change of -0.08 to +0.05. At the host, the true state lowered the error by
  0.059 on half-hours when the engine is off (95% interval -0.079 to -0.039) and raised it by 0.225
  on half-hours when the engine runs (+0.095 to +0.356). Limits: D8 and one host Primary; leads 24
  to 240 hours; origins 2025-11 to 2026-06; 8 monthly refits, 10 seeds.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  Letting the neural network see the gas engine's state while it learns from weather and the
  calendar, and then dropping that knowledge before the forecast, did not improve the forecast of
  D8. At another substation with an engine added, where the engine's true state is known, the
  change helped by about 0.06 points when the engine is off and hurt by about 0.23 points when it
  runs, so the two cancelled. The weather and calendar parts of the neural network seem to lose
  the engine's daily pattern and the fresh final stage cannot recover it. The result rules out
  "extra signal during pre-training only" as a free gain here. A next experiment could combine the
  pre-training state with a way of putting the engine back after pre-training, and test the
  combination at the host first.
lessons: >-
  The positive control was a cancellation and not an absence of mechanism: the state does change what
  the encoders learn (the weights on the state columns are large, the encoders differ from the
  control's by 40% to 48%, and the pre-training loss falls), yet the net forecast change is zero at
  both series. The control arm's first three seeds, fitted in experiment 695, scored 0.083 points
  better than its other seven seeds fitted here, which moved C1 from +0.067 on seeds 0 to 2 to -0.016
  on seeds 3 to 9; the pre-registered 10-seed interval is the decision interval, and the gap is
  probably chance among three seeds.
advice:
  - >-
      Do not expect an engine state used only in pre-training to improve a forecast: at D8 the change
      is +0.009 points (C1, interval -0.074 to +0.091), and an input that is dropped before
      fine-tuning gave a worse forecast in experiment 570 too.
  - >-
      Test a pre-training state only together with a way of putting the engine back after
      pre-training, such as experiment 695's add-back or an origin-time state in the fine-tune mixer,
      and test it at the host with the true state first (the host split, figure 2).
  - >-
      Split every host contrast by whether the injected engine runs, because a net change near zero
      can hide a gain when the engine is off and an equal loss when it runs (WST minus W0: -0.059
      and +0.225).
  - >-
      Fit extra seeds of a control in the same batch as the arms it is compared with, or compare on
      the shared seeds only, because the control's three stored seeds scored 0.083 points better
      than seven new seeds of the same arm (the seed-group check).
  - >-
      Read a seed-averaged month bootstrap and a t interval over per-seed means beside the library
      interval, because the library interval draws one seed per resample and was several times wider
      than the decision intervals (C1 to C4).
code: ml-experiments/00696-gas-engine-state-tracker-part-3-engine-state-in-the-pre-training-mixer/
---

**Letting the neural network see the gas engine's state while it learns from weather and the
calendar, and then dropping that knowledge before the forecast, did not improve the forecast.** At
D8 the change in error is +0.009 points of capacity, and the lower ends of the intervals rule out
a net gain above about 0.07 points. At a second substation, where the engine's true state is known
because we added it, the true state lowered the error by 0.059 points on half-hours when the
engine is off and raised it by 0.225 on half-hours when the engine runs, so the two cancelled. The
verdict is `not supported`.

## The question and how we tested it

**This experiment asks whether a neural network learns better weather and calendar patterns if it is
told, during its first training stage only, whether D8's gas engine is running.** D8 is one of the
16 Primary substations (the substations that supply the local distribution network) that this
research forecasts. A gas engine, a set of generating units that switch on and off, sits behind D8's
meter, so D8's recorded power includes the engine's output. The forecast is the 11 quantiles of the
power for each half-hour 24 to 240 hours ahead. A "point of capacity" is one percentage point of
D8's effective capacity of 10.379 MVA (megavolt-amperes), and CRPS (the continuous ranked
probability score, smaller is better) is twice the mean pinball loss over the 11 quantiles.

**The neural network trains in three stages, and the engine state enters only the first.** In
stage 1, the pre-training stage, a weather encoder and a date-time encoder (the parts that turn
weather and the calendar into numbers) learn together with the pre-training mixer, which combines
their outputs. In stages 2 and 3, the encoders are frozen and a fresh fine-tune mixer, which has
no engine input, learns from ensemble weather forecasts. The idea is learning with privileged
information ([Vapnik and Vashist (2009)](https://doi.org/10.1016/j.neunet.2009.06.042); [Lopez-Paz
et al. (2016)](https://arxiv.org/abs/1511.03643)): the state helps training and is never available
at forecast time. The state is three columns from the tracker of [experiment
694](../00694-gas-engine-state-tracker-part-1/result.md) as stored by [experiment
695](../00695-gas-engine-state-tracker-part-2-cleaned-target-and-climatology-add-back/result.md):
the expected engine output as a share of capacity, the probability that one or more units run, and
a flag for hidden rows. The closest earlier test, [experiment
570](../00570-price-encoder-part-1-known-answer-test-and-pretrained-stack/result.md), used
electricity price in place of the engine state and made the forecast worse.

**The arms differ only in the state the pre-training mixer reads.** Each arm is refitted at the
start of each of 8 months (2025-11 to 2026-06, the refit months) with 10 seeds (starting points for
the training's random numbers).

| Arm | Series | What the pre-training mixer reads |
|---|---|---|
| A0 | D8 | No state (the control, from experiment 695) |
| S1 | D8 | The tracked state |
| SP | D8 | The same state with whole days shuffled within era, calendar month, and weekday class (placebo) |
| S1d | D8 | The tracked state, hidden on half of the origin days (3 seeds, descriptive) |
| W0 | Host D4 | No state; D4's reading with the real engine record added ("injection", from experiment 695) |
| WST | Host D4 | The true injected state |
| WS1 | Host D4 | The tracked state of the injected reading |

## Verdict and predictions

**The verdict is `not supported`, branch 9 of the pre-registered table, the default branch.** No
leakage test failed, the reproduction and identity gates passed, and C1 does not pass: its three
decision intervals contain zero. The table of the six pre-registered predictions follows.

| Prediction | Pre-registered | Observed | Holds? |
|---|---|---|---|
| P1 | C1 between -0.08 and +0.05 | +0.009 | Yes |
| P2 | S1 0 to 0.10 better when the engine is tracked off, 0 to 0.15 worse when it runs | -0.016 off, +0.058 running | Yes |
| P3 | SP within 0.05 of A0 | +0.011 | Yes |
| P4 | WST minus W0 between -0.10 and +0.03; WS1 within 0.05 of WST | -0.004; +0.008 | Yes |
| P5 | The encoder probe M1 lower by 0.02 to 0.3 | -0.049, but the 95% interval [-0.099, +0.002] misses the gate | Estimate in range; gate failed |
| P6 | No other Primary moves | None moves by the rule (below) | Yes |

## Results

**No planned contrast passes.** Each is read at 98.75% (Bonferroni for four), and a contrast passes
only when all three decision intervals lie below zero. The estimate is the change in CRPS in points
of capacity, so negative is better. The three intervals are a t interval over the 8 monthly
differences of the seed means, a bootstrap over whole origin months of the seed-averaged losses, and
a t interval over the 10 per-seed means.

| Contrast | Estimate | t over months | Month bootstrap | t over seeds |
|---|---|---|---|---|
| C1: S1 minus A0, D8 | +0.009 | -0.074 to +0.091 | -0.055 to +0.064 | -0.074 to +0.092 |
| C2: S1 minus SP, D8 | -0.002 | -0.092 to +0.086 | -0.062 to +0.064 | -0.059 to +0.055 |
| C3: WST minus W0, host | -0.004 | -0.047 to +0.038 | -0.032 to +0.029 | -0.023 to +0.015 |
| C4: WS1 minus W0, host | +0.004 | -0.021 to +0.029 | -0.012 to +0.022 | -0.030 to +0.038 |

![The four planned contrasts with 98.75% month-bootstrap intervals](figure_1_contrasts.svg)

**The intervals rule out a net gain above about 0.07 points at D8 and about 0.05 at the host, so the
0.16 points needed to justify porting is out of reach.** The lower ends of the three C1 intervals
are -0.054 to -0.074, and of the C3 intervals -0.023 to -0.047. The standard deviation of one seed's
monthly difference is 0.233 points, and the smallest effect these 10 seeds could resolve is about
0.10 points at D8 and 0.05 at the host, so the design had the power the plan assumed. The library
interval, which draws one seed per resample, is several times wider (C1: -0.353 to +0.309).

**At the host the state helps when the engine is off and hurts when it runs, and the two cancel.**
Mean CRPS is 4.487 for W0, 4.483 for WST, and 4.492 for WS1. Splitting WST minus W0 by whether the
injected engine runs at the valid time (95% t intervals over 10 seeds) gives -0.059 (-0.079 to
-0.039) when the engine is off and +0.225 (+0.095 to +0.356) when it runs. The normalised MAE of the
median shows the same split, -0.119 and +0.312. WS1 shows the pattern at about half the size.

![The change in CRPS split by whether the engine runs](figure_2_engine_off_and_running.svg)

**The split fits the risk that the plan named, and the encoder probe agrees in sign.** The frozen
date-time encoder no longer needs to carry when the engine tends to run, and the fresh fine-tune
mixer cannot recover that pattern from inputs that no longer hold it. We did not measure the
mechanism directly. The probe M1 freezes each arm's pre-trained encoders and fits a fresh mixer with
no state to the host's reading without the engine. Its held-out error is 6.456 points for WST
against 6.504 for W0, a change of -0.049 (95% t interval over the 8 folds -0.099 to +0.002), so
M1 missed its gate. A t interval over the 10 seed means, -0.090 to -0.007, would pass, but that
reading is post hoc.

**The state does reach the encoders, so the null result is not a failed patch.** After
pre-training, S1's weights on the two state columns average 1.10 and 0.71 in size, against 0.10 on
the original inputs, and SP's average 0.44. In one fold and seed the weather and date-time
encoders of S1 differ from the control's by 40% to 48% in relative norm, against 22% to 28% for
SP. The stored final neural networks have encoders identical to the pre-trained file, and the
fine-tune mixer has the same shape in S1 as in A0, so no state reaches a forecast. Pre-training
validation loss falls with the state (S1 0.193 against A0 0.207, WST 0.188 against W0 0.203), and
mean pre-training epochs are 56.8 for A0 and S1, so early stopping is not a second route.

**Post hoc, the control's first three seeds score better than its other seven, and S1d compares
better with S1 and SP than with A0.** A0 seeds 0 to 2, fitted in experiment 695, average 7.232
points, and A0 seeds 3 to 9, fitted here, average 7.315. Between the same seed groups S1 moves by
-0.001, SP by -0.001, and the host's W0 by +0.002. On seeds 3 to 9 only, C1 is -0.016 (95% t
interval over months -0.092 to +0.065), against +0.067 on seeds 0 to 2. The gap is probably chance
among three seeds: arms G0 and SZ reproduce experiment 695's A0 to 2e-5, the pre-training rows and
epochs match, and the other 15 Primaries show an A0 gap no larger than S1's. The decision interval
stays the pre-registered 10-seed interval. On seeds 0 to 2, S1d scores 7.318 against 7.298 for S1
and 7.300 for SP, so S1d minus S1 is +0.020 (-0.173 to +0.214) and S1d minus SP is +0.018 (-0.016
to +0.051), against +0.087 for S1d minus A0. The state is hidden by origin day but indexed by valid
time, so one target half-hour is shown with the state on some rows and without it on others.

![CRPS at D8 of each seed, by arm](figure_3_seed_groups.svg)

**No other Primary moves by the pre-registered rule, which uses the wide library interval.** The
rule (from experiment 695) needs the 95% interval of the paired difference to exclude zero and the
estimate to exceed its sister-arm difference. With a seed-averaged month bootstrap, two of 15
Primaries exclude zero: D10 at +0.053 (+0.007 to +0.109), which would count as worse, and D11 at
-0.031 (-0.068 to -0.001). One in 15 at 95% is the rate chance gives, and the page reports both
without using either against the verdict.

**Next steps are options, and none is committed.** The pre-training state could be tried together
with a way of putting the engine back after pre-training, either experiment 695's calendar add-back
or an origin-time engine state in the fine-tune mixer, and tested first at the host, where the true
state is known.

## Checks

- **Median minus mean forecast (mean of the 11 quantiles minus the median).** At D8 it is +1.716
  for A0, +1.629 for S1, +1.744 for SP, and +1.435 for S1d. At the host it is -4.727 for W0,
  -5.009 for WST, and -4.952 for WS1. The check was not run at the other 15 Primaries.
- **Error of the median beside each CRPS contrast.** C1's MAE contrast is +0.019 (95% interval
  -0.165 to +0.197), the sign of C1. At the host the MAE contrast is -0.036 (-0.061 to -0.011 over
  seeds), the sign of C3.
- **Gates.** SZ, G0, and G0W match experiment 695's fits to 1.9e-5, 1.9e-5, and 5.6e-6 of capacity
  (tolerance 1e-4). Liveness holds for every arm, and exactly 0 for SZ. Alignment gaps are 0.0 in
  all 8 folds. Every real-series leakage test gives a change of exactly 0, and every arm has 242
  origins with origin-time weather.
- **Audit triggers.** Trigger 1 fired (M1 missed its gate and C3 shows no net gain, which the host
  split explains). Trigger 6 fired in the letter only: the C1 and C3 intervals are about 4 times
  narrower than experiment 695's, which compared different contrasts with 3 seeds. Trigger 7
  applies only to a later design that reuses the host split. Triggers 2, 3, and 5 did not fire.
  Trigger 4 was not run, because each contrast sits at one series.
- **Audit findings** (an Opus audit, `printed/audit_*.txt`; none rejected). Every headline number
  reproduced, so branch 9 stands. The audit's seed-group, S1d, host-split, and other-Primaries
  readings are on this page and labelled post hoc. The patch, the frozen encoders, and the absence
  of leakage were confirmed.
- **Not done.** The climatology row, the example-forecast plots, the load-shift scenarios, the
  count of dropped half-hours per fold, and the check of the median minus the mean at the other
  Primaries.

## Method

### Architecture

- **The neural network is experiment 677's step-5 chain at monthly refits** (step 9a of
  [experiment
  677](../00677-building-up-the-candidate-demand-network-one-component-at-a-time/result.md), class
  `AuxLagMdn` of [experiment
  591](../00591-shared-lag-encoder-auxiliary-shift-depth-loss/result.md)), pooled over 16
  Primaries, with a three-component Gaussian mixture output from which 11 quantiles are read and
  sorted.
- **The inputs are weather, date-time, and lagged power.** The weather encoder reads 7 weather
  means, 7 spread columns, 2 lead columns, and 14 thermal-lag and solar-gain columns; the
  date-time encoder reads 21 columns; the lag encoder reads power at 24 hours and 1 week before
  the origin; every power input is at least 24 hours older than its origin.
- **The widths are an 8-wide Primary embedding, encoders from 64 to 32, a mixer of 64, and dropout
  0.1.** Stage 2 and stage 3 freeze the weather and date-time encoders and train a fresh fine-tune
  mixer, as in [experiment
  574](../00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/result.md).
- **The state arms widen the first layer of the pre-training mixer by 3 inputs.** The layer is
  built at the original width under the original random-number stream and then padded with 3 zero
  columns, so a zero state reproduces A0. The state columns come from the stored per-fold tracker
  states of experiment 695 at the valid time of each pre-training row, and other Primaries read
  zeros. The widened layer is loaded at all three stages and is never called after pre-training.
- **The placebo SP shuffles whole days of S1's columns within era (to 2025-03-23, then later),
  calendar month, and weekday class,** using only days at or before the fold's cut, with seed 0.
- **S1d hides the state on a random half of origin days, with the hidden flag set. WST reads the
  true injected output and whether any unit runs at the host, and WS1 reads the tracked state of
  the injected reading.** The injected engine is experiment 695's resampled 44-day record.

### Training regime

- **Training data are ERA5 and CAMS rows to 3.5 hours before the refit in stage 1, and the ECMWF
  ensemble 00Z run of the origin day in stages 2 and 3 and at inference.** Power comes from the
  cleaned table with `substation_zero` half-hours dropped, and D15 and D18 are cut before
  2025-03-01.
- **Everything outside the pre-training mixer's input is step 9a unchanged:** the loss, the
  optimiser, early stopping on held-out weeks, the 7-day embargo, the injected lag-only shifts, and
  monthly refits for 2025-11 to 2026-06.
- **There are 10 seeds per arm and fold, with seeds 0 to 2 of A0 and W0 reused from experiment
  695's stored fits and seeds 3 to 9 fitted here by the same chain.** The fits used about 28 GPU
  job-hours in 4.7 hours of wall-clock time on one 48 GB card, with 6 to 7 jobs at once, a peak of
  33 GB of GPU memory, and about 5.5 GB of host memory per job.
- **The state never reaches a forecast.** The state at a pre-training row reads readings only before
  the refit minus 3.5 hours, and a perturbation of every later reading changes no state value.

### Measurement

- **The metric of record is CRPS at D8 in points of effective capacity (fixed at 10.379 MVA),** with
  leads 24 to 240 hours from the 12:00 UTC contract origin, origins 2025-11-01 to 2026-06-30, valid
  times before 2026-07-01, and all 16 Primaries scored. The normalised MAE of the median, pinball
  loss, and coverage at q01, q05, q95, and q99 are printed beside it.
- **The host's truth is the injected reading.** The 102,310 D8 rows per seed and the host's rows
  come from the scorer of experiment 695, and the audit re-derives each number from the raw fits.
- **Intervals are the three decision intervals above, a library interval for the other-Primaries
  rule, and 40,000 resamples (250 draws per tail at 98.75%).** The smallest gain worth porting is
  0.16 points at D8, which is 8 times demand-v2's minimum of 0.02 for a change aimed at one Primary.
- **The verdict follows the pre-registered table of nine branches, evaluated in order.** A
  contrast passes when all three decision intervals lie below zero.

## Files

- `ml-experiments/00696-*/e00696_common.py`, `e00696_state.py`, `e00696_model.py`,
  `e00696_probe.py`, `e00696_hooks.py`, and `e00696_k1.py`: the arm table, the state builders, the
  widened neural network, the encoder probe, the hooks that attach the state to the pre-training
  rows, and the wrapper that runs experiment 677's step-5 chain.
- `e00696_chain.sh`, `e00696_gpu.sh`, `e00696_queue.sh`, and `e00696_wait.sh`: one fold of one
  arm, one process under a memory cap, the job queue, and the wait loop.
- `e00696_checks.py`: the synthetic and real-series checks and the identity and reproduction
  gates.
- `e00696_score.py` and `e00696_verdict.py`: the scorer (contrasts, the probe, slices, the weather
  assertion) and the verdict function, with `e00696_selftest_*.py` covering every branch.
- `e00696_audit_*.py`: the audit's loss, contrast, slice, weight, epoch, and host code.
  `e00696_figures.py` draws the three figures from the audit's losses, so its bootstrap bounds
  differ from the scorer's by up to 0.001.
- `printed/`: the printed reports of the checks, the scorer, the audit, and the job lists.
- Data under `data/ml-experiments/00696-*/` (read and written): `runs/<arm>/` (fits, models,
  logs), `states/<arm>/<month>.npz` (the state columns), `checks.json`, and `audit/` (per-row
  losses). Experiment 695's `states/`, `stacks/`, and the stored fits of A0 and W0 seeds 0 to 2
  are read.
