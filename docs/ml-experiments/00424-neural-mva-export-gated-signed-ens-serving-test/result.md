---
id: "00424"
slug: neural-mva-export-gated-signed-ens-serving-test
title: >-
  Does the export-gated neural network still forecast signed power better than the plain reading
  when it sees ensemble weather forecasts instead of recorded weather?
theme: production-reference-and-best-guess
also_relevant_to:
  - solar-disaggregation
  - weather-products-and-blending
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted networks (state files) and per-row forecasts of the hidden pair and of the
  MVA-only eight, in this experiment's private folder. Designs: more than one is defensible (retrain
  on the ensemble inputs, apply the laboratory network to them, or pretrain on recorded weather and
  fine-tune on the ensemble; this plan fixes the first and runs the second as a diagnostic).
  Callers: the code imports experiments 87, 397, 405, 412, 417, and 420. Serving path and
  degradation rule: not touched (this repository holds neither).
hypothesis: >-
  Experiment 420's export-gated network (arm A) still forecasts signed power on reverse-flow
  half-hours much better than the same network with its gate shut (P0) when both are trained and
  scored on ECMWF ensemble (ENS) mean and spread weather, which is the weather a live service would
  hold, instead of the recorded CAMS irradiance and ERA5 temperature of the laboratory test. The gain
  shrinks, because the ENS forecast of cloud is less accurate than recorded cloud, but most of the
  laboratory gain comes from letting the forecast go below zero, which does not depend on weather
  accuracy.
prediction: >-
  Written before any score is read. S1: A beats P0 on reverse-flow rows by at least the minimum
  detectable effect with the interval excluding zero, pooled and at each of the six hidden
  Primaries, and the pooled gain is at least 10 points of effective capacity (about half the
  laboratory gain of 19.2). This fails if the pooled gain falls below 10 or if any Primary's interval
  includes zero. S2: no harm at the no-injection Primaries (the experiment 420 tolerance of +0.05
  pooled and +0.20 single). S3: no reading harm at the MVA-only eight (+0.05 pooled). The gate
  decisions are descriptive: with ENS inputs the Primary-level gate may open at fewer of the MVA-only
  eight than the four it opened at in experiment 420.
decision_rule: >-
  S1, S2, and S3 all pass for A: label the signed output at the eight "inferred, ensemble-tested,
  gated" and take the serving question to the next step (a live-service port is a separate decision
  for the maintainer). S1 passes and S2 or S3 fails: the label stays "inferred, laboratory, gated"
  and the page says which bar failed. S1 fails: the laboratory result does not survive ensemble
  weather, the signed output at the eight is not shown to NGED, and the unsigned forecast stays the
  published one. No porting threshold applies, because nothing here ports to production. Stage-1 stop
  rule: if A fails S1 on rotation r1 (pooled over D5 and D10, and at each), stop and tell the
  manager; otherwise tell the manager and run stage 2.
benchmark: >-
  P0 on ENS inputs (experiment 420's network with every gate forced shut, the plain neural network
  read as signed). Experiment 420's A, ref, and P0 on CAMS and ERA5 inputs are reported only, paired
  row by row.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Signed CRPS (twice the mean pinball loss over the 11 quantile levels) in points of effective
  capacity on rows with reverse flow, days 1 to 10, pooled over the six hidden Primaries; reading
  CRPS at the MVA-only eight; descriptive sign, tail, lead-band, and gate metrics.
date: 2026-10-31
code: ml-experiments/00424-neural-mva-export-gated-signed-ens-serving-test/
---
**This experiment asks whether the signed forecast of the export-gated neural network survives the
step from recorded weather to forecast weather.** Eight Primary substations report only the size of
the power that flows, not its direction (signed power is positive for demand and negative for
reverse flow, where solar generation exports power up the electricity network). [Experiment
420](../00420-neural-mva-export-gated-signed-final/result.md) hid the direction at six Primaries
where it is known, added a real solar farm's recorded output to some of them, and found that a
neural network with a gate that decides once per Primary whether it exports forecasts the signed
power far better than the same network with the gate shut (7.1 against 26.3 points of capacity on
reverse-flow half-hours). The weather in that test was recorded CAMS irradiance and ERA5
temperature, which a live service does not have at forecast time. This experiment repeats the test
with the ECMWF ensemble (ENS) mean and spread of the same weather variables, trains the network on
them, and scores it on the same rows. The result is not yet known.

## Evidence the plan follows

- **Experiment 420 (merged, pull request 77) found a laboratory gain of 19.2 points (16.5 to 21.5)**
  of the gated network over the gate-shut network, and said most of it comes from allowing negative
  values, because the ungated reference scored the same as A on reverse-flow rows. The gate matters
  at the no-injection Primary D5. This plan keeps those caveats.
- **Experiment 418 (merged) found that pretraining on recorded weather and then fitting on the ENS
  archive helps** (about 0.06 points at the six signed-MW Primaries and 0.16 at the eight, over 12
  test months), and that an ENS-only fit degrades badly when the archive covers only 6 months. The
  ENS archive used here covers 2024-04 to 2026-06, so each refit has 15 to 24 months, as in
  experiment 420. A pretrain-then-fine-tune arm would add a design question, so this plan leaves it
  as a named next step.
- **Experiments 419 and 423 (merged) found no input change that beats seed noise** for the pooled
  neural network. This plan adds ENS spread columns as inputs, so seed noise is measured here (a
  second seed of A) before any contrast is read.
- **The data-processing page says ENS irradiance needs calibration against CAMS** (shifts of up to
  20% in irradiance and 2 K in temperature) **and is a 6-hour mean after lead 144 hours.** Both
  facts are why the preferred arm is retrained on ENS inputs, so the network learns the bias, and
  why an applied-only network is reported as a diagnostic with no bar.
- **Standing practices applied:** monthly refits; cleaned data with the dropped count stored per
  fit (experiment 417's fold drops the flagged half-hours); D15 and D18 cut before 2025-03-01;
  every network saved as a state file; tails and examples reported; draft pull request. Departures
  are listed below. Earlier advice followed: score a signed forecast against the ungated network as
  well as the gate-shut network (reference rows below); label the signed output by what was tested.
  Advice departed from: none.

## Design

**The preferred arm is retrained on ENS inputs, and the weather it sees at training time and at
inference time is the same ENS mean and spread.** Training on realised weather and serving ENS
weather would put the network in a distribution it never saw, and the ENS forecast carries known
biases. A network applied without retraining is reported as a diagnostic (arm A-applied).

| Arm | What it is | Role |
|---|---|---|
| P0 | Experiment 420's P0 (gate forced shut, plain reading read as signed), trained on ENS inputs | Benchmark |
| A | Experiment 420's arm A (Primary-level export gate), trained on ENS inputs | Candidate |
| A-applied | Experiment 420's saved arm A networks (trained on CAMS and ERA5, seed 0), applied to ENS inputs with no refit | Diagnostic, reported only, zero fits |
| lab A, lab ref, lab P0 | Experiment 420's stored forecasts on CAMS and ERA5 inputs | Reported only |

**What changes in the inputs.** Experiment 417's fold is reused unchanged except for three series:

- **PV term and Primary gate.** The scaled CAMS unit shape `u` and clear-sky index `k` become the
  ENS-mean unit shape (from the ENS mean shortwave irradiance, divided by the same `u99`) and the ENS
  clearness index, taken from experiment 405's `extras` arrays row by row.
- **Temperature.** ERA5 temperature becomes the ENS mean temperature.
- **Spread.** Three standardised input columns are added to the demand module: the temperature
  spread (standard deviation across the 51 members), the unit-shape spread, and the mean temperature
  itself. The columns are computed once per run, so the network is passed once per row, with no
  member loop.

Everything else is as in experiment 420: injected farm and reactive floor Q = 2 points at power
factor 1, rotations r1 (D5, D10 hidden), r2 (D9, D14), and r3 (D11, D16), the seven test months
2025-07, 2025-08, 2025-09, 2026-03, 2026-04, 2026-05, and 2026-06, 730-day training windows, monthly
refits, the same early-stopping weeks, and scoring on leads up to 240 hours (days 1 to 10). Rows are
the same rows as experiment 420's, so experiment 420's stored forecasts pair with these row by row.

**Settings.** Level 2.0 (main) and the no-injection placebo. Level 1.0 is not run, because the
question is whether the main result survives ensemble weather.

**Weather availability at forecast time.** ENS runs are usable 9 hours after their start time, and
each row's ENS values come from the run available at its forecast origin, so no lag-time weather is
recorded weather. The lags in the demand module are power lags, as in experiment 420.

## Scoring and pass bars

**The primary contrast is signed CRPS in points of effective capacity, days 1 to 10, on the
hidden-sign rows with reverse flow, A against P0, both on ENS inputs, pooled over the six hidden
Primaries.** Intervals come from a paired month-block bootstrap (resample whole Primary-months,
20,000 resamples), because each half-hour appears in about 15 runs.

**The minimum detectable effect is written down before any arm-against-arm score is read.** It is
2.8 times the month-block bootstrap standard deviation of the null difference between A's two seeds
(ENS inputs) on reverse-flow rows. Stage 1 fits A's second seed at level 2.0 on rotation r1 so that
the figure exists first; the full-run figure uses all three rotations. Each figure is committed to
this page before the matching contrast is read. A figure above 0.5 points marks the contrast as
underpowered.

- **S1 (gain over P0).** A's reverse-row CRPS is below P0's by at least the minimum detectable effect,
  with the interval excluding zero, pooled and at each hidden Primary at level 2.0. On rotation r1,
  stage 1 pools D5 and D10 and reads each of the two. The prediction adds that the pooled gain is at
  least 10 points.
- **S2 (no harm where there is no export).** In the placebo setting, signed CRPS on all rows of D5,
  D9, D11, D14, and D16 is within +0.05 points of P0 pooled, with no single Primary above +0.20.
  D10 is left out because it has real reverse flow.
- **S3 (the MVA-only eight).** Reading CRPS at the eight, from the same fits, is within +0.05 points
  of P0 pooled.

**Reported with no bar:** the gain by lead band (days 1 to 2, 3 to 5, 6 to 10; headline days 1 to
10); experiment 420's laboratory A, ref, and P0 beside the ENS arms, and the paired difference of A
on ENS inputs against A on laboratory inputs; A-applied against P0 and against A; q01 and q99
coverage and pinball loss on reverse-flow rows; sign balanced accuracy, recall, ROC area, and the
false-reverse rate at `P(X < 0) > 0.5` on both denominators of experiment 420; each Primary-level
gate decision with its Spearman statistic, with the decisions at the eight set beside experiment
420's (open at D8 and D12, shut at D2, D3, D4, D19, D21, and D22); the dropped half-hours per fit.
Experiment 420's caveat is kept in the write-up: most of the gain over P0 is being allowed to go
below zero, and the gate matters at the D5 placebo.

**Observation made before scoring.** Building the stage-1 training fold for the 2025-07 refit (no
fit, no score) shows the Primary-level gate shut at D8 and D12 on ENS inputs, where experiment 420's
gate opened at both on CAMS inputs. The gate decision is made on training rows only, so this is an
input-side fact and not a score. It means the gate decisions at the eight are a result to report,
not a prediction.

## Fits, compute, and stop rule

| Batch | Fits |
|---|---|
| Stage 1: P0 and A at level 2.0 and the placebo, seed 0, rotation r1, 7 months (28), plus A seed 1 at level 2.0 (7) | 35 |
| Full run: all three rotations, P0 and A at both settings (84) plus A seed 1 at level 2.0 (21) | 105 |
| A-applied: forecast only, 3 rotations × 7 months × 2 settings | 0 fits, 42 forecasts |

**Stage 1 runs rotation r1 only.** If A fails S1 on r1, stop. Otherwise tell the manager and run the
remaining rotations (70 fits). Experiment 420 timed a fit at 26 to 83 seconds including the fold
build, so stage 1 takes about 30 to 60 minutes and the full run about 1.5 to 3 hours, on one GPU
process using about 1.4 GB of GPU memory, one feeder core, `systemd-run --user --scope -p
MemoryMax=8G -p OOMPolicy=continue`, with `OMP_NUM_THREADS=1`. The fit script reads `MemAvailable`
before every fold and every fit and stops if it falls below 12 GiB (the `available` column of
`free -g`). A fit is skipped when its JSON file exists, so a stopped run resumes. Experiment 420's
fits are not repeated: its stored forecasts are the laboratory reference rows. Outputs go to
`work/fits/<arm>/`, state files to `work/models/<arm>/`, in this experiment's private folder.

## Code

All code is in `ml-experiments/00424-neural-mva-export-gated-signed-ens-serving-test/`.

- `e00424_inputs.py` builds experiment 417's fold with ENS inputs (`build_fold`) and the ENS model
  inputs.
- `e00424_fit.py` defines the jobs, fits P0 and A, forecasts the hidden pair and the eight, and runs
  A-applied from experiment 420's state files. It imports experiment 420's arms, gates, training
  loop, and forecast helpers.
- `e00424_score.py` scores S1 to S3, the minimum detectable effect, the lead bands, and the
  descriptive tables, importing experiment 420's scoring functions.

## Departures

- **Level 1.0 and the shuffled-sky arm are not run.** Experiment 420 already showed the shuffled sky
  is far worse than A on recorded weather, and the serving question needs the main setting only.
- **Avoidable-curtailment cost is not computed,** for the reason experiment 420 gives: the injected
  series have no node and generator tables. The q01 coverage and pinball loss on reverse-flow rows
  stand in.
- **Seed asymmetry.** A has two seeds and P0 one, so A's mean over two seeds meets P0's single seed,
  which lowers A's noise without biasing the contrast.

## Risks

- **The injected farm is real recorded output and the weather inputs are ENS, so the test is still
  a laboratory test of the sign question,** although the weather side is now what a live service
  holds.
- **A wrong Primary-level gate decision makes a whole Primary's forecast wrong.** The gate decisions
  are stored for every fit and reported.
- **ENS irradiance after lead 144 hours is a 6-hour mean,** so the unit shape is blurred at days 6
  to 10. The lead-band table shows whether the gain falls there.
- **Unlogged load-shift events** may sit in the readings, as NGED says its logs are incomplete.
