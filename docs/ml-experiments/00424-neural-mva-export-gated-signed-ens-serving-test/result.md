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
status: done
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
  Written before any score is read. Scores are signed continuous ranked probability scores (CRPS)
  in points of effective capacity; A is the export-gated network and P0 the same network with its
  gate shut, both on ENS inputs. S1: A beats P0 on reverse-flow rows by at least the minimum
  detectable effect, with the interval excluding zero, pooled and at each of the six hidden
  Primaries, and the pooled gain is at least 10 points over the three rotations (about half the
  laboratory gain of 19.2) and at least 4 points on rotation r1 (about half the laboratory gain of
  8.29 at D5 and D10). This fails if a pooled gain falls below its floor or if any Primary's
  interval includes zero. S2: no harm at the no-injection Primaries (experiment 420's tolerance of
  +0.05 points pooled and +0.20 at a single Primary, read on the point estimate). S3: no reading
  harm at the MVA-only eight (+0.05 points pooled, point estimate). The Primary-level gate is shut
  Gate decisions are descriptive: on the shortest-lead rows the gate opens at D5, D8, D10, and D12
  in the one training fold built before scoring, as in experiment 420.
decision_rule: >-
  S1, S2, and S3 all pass for A, with S1's pooled floor met: label the signed output "inferred,
  ensemble-tested, gated" at those of the eight Primaries whose gate opened in the fits, and
  "inferred, laboratory, gated" at the rest (if the gate is shut at all eight, no label is
  upgraded, and the page says so). A live-service port is a separate decision for the maintainer.
  S1 passes its interval and minimum-detectable-effect conditions but misses the pooled floor: the
  verdict is small gain and the label stays "inferred, laboratory, gated". S1 passes and S2 or S3
  fails: the label stays "inferred, laboratory, gated" and the page says which bar failed. S1 fails
  (an interval includes zero at any hidden Primary): the laboratory result does not survive ensemble
  weather, the signed output at the eight is not shown to NGED, and the unsigned forecast stays the
  published one. No porting threshold applies, because nothing here ports to production. Stage-1
  stop rule: if A fails S1 on rotation r1 (pooled over D5 and D10 with a floor of 4 points, and at
  each of the two with the interval excluding zero), stop and tell the manager; otherwise tell the
  manager and run stage 2. Stage 1 reads S2 at D5 only, so S2 stays untested at the other four
  placebo Primaries until the full run.
benchmark: >-
  P0 on ENS inputs (experiment 420's network with every gate forced shut, the plain neural network
  read as signed). Experiment 420's A, ref, and P0 on CAMS and ERA5 inputs are reported only, paired
  row by row.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Signed CRPS (twice the mean pinball loss over the 11 quantile levels) in points of effective
  capacity on rows with reverse flow, days 1 to 10, pooled over the six hidden Primaries; reading
  CRPS at the MVA-only eight; descriptive sign, tail, lead-band, and gate metrics.
verdict: supported
best_arm: >-
  A trained on ENS inputs: reverse-flow signed CRPS 14.26 points against 29.97 for P0 on ENS inputs,
  a gain of 15.72 (95% interval 13.85 to 17.42), against 19.2 in the laboratory test. The
  laboratory-trained network applied unchanged to ENS inputs scored 8.76 (unplanned diagnostic).
headline: >-
  With ensemble weather in place of recorded weather, the export-gated network (A) still cut the
  signed CRPS on reverse-flow half-hours from 29.97 to 14.26 points of capacity against the gate-shut
  network (P0), at every one of the six hidden Primaries, with no harm at the no-injection Primaries
  (+0.009 pooled, +0.051 at D9) or at the MVA-only eight (-0.005). The gain is 3.5 points smaller than
  in the laboratory test, and the network trained on recorded weather and applied to ensemble weather
  scored 5.50 better (4.35 to 6.61) than the network retrained on ensemble weather.
lessons:
  - >-
    The export gate and the permission to go below zero survive ensemble weather: the gain over the
    gate-shut network is 15.7 points (13.8 to 17.4), at least 5.7 points at every hidden Primary,
    and roughly equal in all three lead bands (16.2, 15.9, and 15.3 points).
  - >-
    Ensemble inputs cost accuracy: A scores 7.20 points worse (5.85 to 8.54) than the same network on
    recorded weather, and its sign calls are much weaker (balanced accuracy 0.60 against 0.85, seed 0).
  - >-
    Unplanned: the laboratory-trained network applied to ensemble inputs (8.76) beat the network
    retrained on them (14.26) by 5.50 points (4.35 to 6.61). The retrained network learned a smaller
    and less stable solar term (learned capacity 39 to 106 points of capacity by fit at D9, against 98
    to 107 in the laboratory), and its two seeds differ by 2.09 points on average. The cause is not
    tested.
  - >-
    The Primary-level gate opened at D8 and D12 in all 21 fits and stayed shut at the other six
    MVA-only Primaries, as in the laboratory test, once the gate statistic uses each valid time's
    shortest-lead ensemble row.
take_home_message: >-
  With forecast weather instead of recorded weather, the neural network with the export gate still
  forecasts signed power on reverse-flow half-hours much better than the same network with the gate
  shut (a gain of 15.7 points of capacity, 13.8 to 17.4, against 19.2 in the laboratory test), and it
  does no harm where nothing is exported. The gate opened at the same two Primaries as before, D8 and
  D12. An unplanned finding is that the network trained on recorded weather and applied unchanged to
  forecast weather scored 5.5 points better (4.3 to 6.6) than the network retrained on forecast
  weather, so the next step is a training recipe that keeps the solar term stable, not a longer fit.
advice:
  - >-
    Label the signed output at D8 and D12 "inferred, ensemble-tested, gated" and keep "inferred,
    laboratory, gated" for the other six MVA-only Primaries, whose gate stayed shut (S1 to S3, gate
    table).
  - >-
    Compute a Primary-level gate on each valid time's shortest-lead ensemble row, because keeping the
    first row per valid time keeps the longest-lead row and shut the gate at D8 and D12 (design).
  - >-
    Before retraining a solar-term network on ensemble inputs, score the network trained on recorded
    weather applied to ensemble inputs beside it, because it scored 5.5 points better here (A-applied
    against A).
  - >-
    Fit two seeds before reading a contrast of a retrained solar-term network on ensemble inputs:
    the null difference between seeds is 2.09 points and the minimum detectable effect 1.73 (minimum
    detectable effect, full run).
production_status: candidate
adopted_in: none
date: 2026-10-31
code: ml-experiments/00424-neural-mva-export-gated-signed-ens-serving-test/
---
**Take-home: the export-gated network still forecasts signed power on reverse-flow half-hours much
better than the gate-shut network when it sees ensemble weather forecasts, and does no harm
elsewhere,
but ensemble weather costs accuracy.** The gain over the gate-shut network is 15.7 points of
effective capacity (95% interval 13.8 to 17.4), against 19.2 in the laboratory test. The gate opened
at D8 and D12, as before. **An unplanned finding:** the network trained on recorded weather and
applied unchanged to ensemble weather scored 5.5 points better (4.3 to 6.6) than the network
retrained
on ensemble weather. The registered bars did not test this comparison, and no prediction covered it.

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
them, and scores it on the same rows. The result: the gate still helps a great deal, though less
than with recorded weather.

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
  ENS-mean unit shape (from the ENS mean shortwave irradiance, divided by the same `u99`) and the
ENS
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

**Weather availability at forecast time.** ENS runs are usable 9 hours after their start time (the
`NWP_PUBLICATION_DELAY_HOURS` constant), and each row's ENS values come from the 00Z-run summary
that
experiment 397 attaches to the row's forecast origin, so no lag-time weather is recorded weather.
Before 2025-07-01 experiment 397 builds the ENS mean and spread from the archive (only the control
member is stored); from 2025-07-01 it uses the stored 51-member frames. Both arms see this source
change, and the month-by-month table shows whether it matters. Rows with missing ENS values stay in
both arms with the missing inputs set to zero after standardisation, so the rows match experiment
420's; the count of such rows per fit is stored. The lags in the demand module are power lags, as in
experiment 420.

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

- **S1 (gain over P0).** A's reverse-row CRPS is below P0's by at least the minimum detectable
  effect,
  with the interval excluding zero, pooled and at each hidden Primary at level 2.0. On rotation r1,
  stage 1 pools D5 and D10 and reads each of the two. S1 also needs a pooled floor, 10 points over
  the three rotations and 4 points on rotation r1, because experiment 420's minimum detectable
  effect (0.27 to 0.40 points) is far below every laboratory gain and cannot fail alone. The floors
  are about half the laboratory gains. A gain above the minimum detectable effect but below its
floor
  is a small gain, not a pass. The minimum detectable effect governs only the contrasts it can
  resolve (A against A-applied, and A on ENS against A on laboratory inputs); a figure above 0.5
  points marks those contrasts, and S1 if it is that large, as underpowered.
- **S2 (no harm where there is no export).** Read on the point estimate, with the interval end
  reported. In the placebo setting, signed CRPS on all rows of D5,
  D9, D11, D14, and D16 is within +0.05 points of P0 pooled, with no single Primary above +0.20.
  D10 is left out because it has real reverse flow.
- **S3 (the MVA-only eight).** Read on the point estimate. At a Primary whose gate is shut, A's
  reading forecast equals P0's by construction, so S3 can fail only through the Primaries whose gate
  opens. Reading CRPS at the eight, from the same fits, is within +0.05 points
  of P0 pooled.

**Reported with no bar:** the gain by lead band (days 1 to 2, days 3 to 6 up to a lead of 144 hours,
and days 6 to 10 above
144 hours, where the ENS stamps change from 3-hourly to 6-hourly; headline days 1 to 10); experiment
420's laboratory A, ref, and P0 beside the ENS arms, and the paired difference of A
on ENS inputs against A on laboratory inputs; A-applied against P0 and against A; q01 and q99
coverage and pinball loss on reverse-flow rows; sign balanced accuracy, recall, ROC area, and the
false-reverse rate at `P(X < 0) > 0.5` on both denominators of experiment 420; each Primary-level
gate decision with its Spearman statistic, with the decisions at the eight set beside experiment
420's (open at D8 and D12, shut at D2, D3, D4, D19, D21, and D22); the dropped half-hours per fit.
Experiment 420's caveat is kept in the write-up: most of the gain over P0 is being allowed to go
below zero, and the gate matters at the D5 placebo.

**Observation made before scoring, and a correction to the gate input.** Building the stage-1
training
fold for the 2025-07 refit (no fit, no score) first gave a gate shut at D8 and D12. The cause was
experiment 417's gate code, which keeps the first row of each valid time. With CAMS every row of a
valid time has the same weather, but each ENS row has its own run, and the first row is the
longest-lead one (mean lead 333 hours at D5), whose irradiance is a blurred 6-hour mean. The plan
therefore computes the Primary-level gate on each valid time's shortest-lead row
(`e00424_inputs.primary_gates`), which is a design choice made before any score. On those rows the
gate opens at D5, D8, D10, and D12 in that fold, as in experiment 420. The ENS clearness index is
the
ENS mean shortwave over `1000 * cos(zenith)`, not a CAMS clear-sky index, so the gate's `k >= 0.7`
selects a different, sparser set of rows; the number of rows behind each gate decision is stored.

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
  which lowers A's noise without biasing the contrast. The minimum detectable effect comes from
  single seeds, so it is conservative for the two-seed mean.
- **No climatology row.** The target is a signed series with an injected farm, for which no
  month-by-half-hour climatology forecaster exists in this repository; experiment 420 had none.
- **No multiplicity correction.** S1 to S3 are conjunctive pass bars, each read at 95%; the decision
  rule needs all three, so the family-wise error of a joint pass is below the level of any one.
- **Figures.** The write-up carries a day-ahead example and a 10-day example at D5 with a farm
  injected, from experiment 420's figure script adapted to these files, and q01, q05, q95, and q99
  coverage and pinball loss on reverse-flow rows.

## Risks

- **The injected farm is real recorded output and the weather inputs are ENS, so the test is still
  a laboratory test of the sign question,** although the weather side is now what a live service
  holds.
- **A wrong Primary-level gate decision makes a whole Primary's forecast wrong.** The gate decisions
  are stored for every fit and reported.
- **ENS irradiance after lead 144 hours is a 6-hour mean,** so the unit shape is blurred at days 6
  to 10. Experiment 397's summaries do not treat the two stamp spacings differently, and this plan
  does not change that. The lead-band table, split at 144 hours, shows whether the gain falls there.
- **A single scale constant `u99`, fixed on one Primary's daylight rows over the whole record,**
  also
  covers the test months. It is a scalar shared by both arms and is accepted as negligible.
- **Calibration of ENS shifts by season, lead band, and hour band** (experiment 111) may exceed what
  one retrained network with a lead input learns. The lead-band and month tables show it.
- **Unlogged load-shift events** may sit in the readings, as NGED says its logs are incomplete.

## Minimum detectable effect, stage 1 (written before any arm-against-arm score)

**On rotation r1 the minimum detectable effect is 0.907 points of effective capacity.** The null
difference between A's two seeds (ENS inputs) on reverse-flow rows at level 2.0 has a mean of -0.024
(95% interval -0.725 to 0.549; 14 Primary-month blocks, 22,443 rows) and a bootstrap standard
deviation of 0.324; 2.8 times 0.324 is 0.907. The figure is above the 0.5-point limit, so contrasts
of that size or smaller (A against A-applied, A on ENS against A on laboratory inputs) are reported
as underpowered. S1 is not weakened by this, because its floors (4 and 10 points) sit far above the
figure. The figure is in `work/mde_stage1.json`, which the script refuses to overwrite.

## Minimum detectable effect, full run (written before the full scoring)

**Over all three rotations the minimum detectable effect is 1.735 points of effective capacity.**
All 105 fits and 42 forecast-only runs finished. The null difference between A's two seeds on
reverse-flow rows at level 2.0 has a mean of 2.09 (95% interval 0.88 to 3.29; 42 Primary-month
blocks, 79,399 rows) and a bootstrap standard deviation of 0.620; 2.8 times 0.620 is 1.735. The two
seeds of A therefore differ by a mean of 2.09 points on these rows, so seed noise is large next to
the 10-point S1 floor, and contrasts under about 2 points are underpowered. The figure is in
`work/mde_full.json`.

## Results

![Gain over the gate-shut network by hidden Primary](gain_by_primary.svg)

**S1, S2, and S3 pass for A, and the pooled floor is met.** All 105 fits and 42 forecast-only runs
finished. Values are points of effective capacity, level 2.0 unless stated, with 95% Primary-month
block intervals; A is the mean of two seeds and the other arms use seed 0. The minimum detectable
effect, written before scoring, was 1.735. All scores are in `work/scores_full.json`.

| Bar | Result | Passes? |
|---|---|---|
| S1, gain of A over P0 on reverse-flow rows, pooled | 15.72 (13.85 to 17.42), floor 10 | Yes |
| S1 by Primary | D5 5.75, D9 20.67, D10 10.01, D11 16.67, D14 17.84, D16 19.80; every interval excludes zero and clears 1.735 | Yes |
| S2, harm at the no-injection Primaries, pooled | +0.009 (-0.003 to 0.023); D9 +0.051, others -0.004 to +0.003 | Yes (limits +0.05 pooled, +0.20 single) |
| S3, reading harm at the MVA-only eight, pooled | -0.005 (-0.036 to 0.020) | Yes (limit +0.05) |

- **Absolute scores on reverse-flow rows:** P0 29.97, A 14.26, and for reference experiment 420's
  recorded-weather arms P0 26.26, A 7.05, ref (no gate) 6.99. The laboratory-trained network applied
to
  ensemble inputs scored 8.76.
- **Lead bands (gain of A over P0):** days 1 to 2 16.18 (14.24 to 17.90), days 3 to 6 15.86 (13.93
  to
  17.57), days 6 to 10 15.35 (13.49 to 17.00). The gain does not fall after the ensemble's
  irradiance becomes a 6-hour mean at 144 hours.
- **Ensemble against recorded weather for A:** worse by 7.20 (5.85 to 8.54).
- **Rotation r1 (stage 1) before the full run:** gain 7.79 (6.39 to 9.12) against 8.29 in the
  laboratory test at D5 and D10, floor 4, minimum detectable effect 0.907; stage 1 passed.

![Mean CRPS on reverse-flow rows by arm](crps_by_arm.svg)

**Unplanned: the laboratory-trained network applied to ensemble inputs beat the retrained network.**
A-applied scored 5.50 points better than A (4.35 to 6.61; 8.76 against 14.26, A as the mean of two
seeds), and 21.22 better than
P0 (18.40 to 23.77). The plan called A-applied a diagnostic. No bar or prediction covered it, and
the
result contradicts the plan's reason for retraining (a network that has seen ensemble inputs should
cope with their bias). A likely explanation, supported by the stored fits but not tested: the
learned capacity of the solar term (the multiple of the unit shape subtracted from demand) is lower
and unstable when trained on ensemble inputs, for example 39 to 106 points of capacity by fit at D9
against 98 to 107 with recorded weather, which fits a noisy regressor shrinking the fitted
coefficient. The two seeds of A differ by 2.09 points (0.88 to 3.29) on average, so the seed noise
is
large next to every other contrast except S1.

**The sign is called much less well with ensemble inputs, and the lower tail is too narrow.** The
numbers are descriptive, level 2.0, seed 0, on the half-hours with the CAMS unit shape at least 0.3
and clear-sky index at least 0.7 (115,040 rows). The slice is defined by the recorded CAMS values,
as
in experiment 420, for every arm.

| Measure | A (ENS) | A-applied | A (recorded, experiment 420) |
|---|---|---|---|
| Balanced accuracy at `P(X < 0) > 0.5` | 0.600 | 0.781 | 0.852 |
| Recall of reverse flow | 0.235 | 0.676 | 0.859 |
| False-reverse rate, every non-reverse slice row | 3.6% | 11.5% | 15.5% |
| Area under the receiver operating characteristic curve | 0.876 | 0.891 | 0.930 |
| q01 coverage on reverse-flow rows (nominal 0.01) | 0.091 | 0.118 | 0.062 |
| q05 coverage (nominal 0.05) | 0.308 | 0.254 | not recorded here |
| q95 coverage (nominal 0.95) | 1.000 | 0.987 | not recorded here |
| q99 coverage (nominal 0.99) | 1.000 | 0.998 | 0.998 |

- **Tail pinball losses (A, ENS):** q01 0.82, q05 3.37, q95 3.11, q99 0.74; P0: 19.20, 20.20, 3.40,
  0.79. The signed lower tail is too narrow, as in experiment 420.
- **Primary-level gates:** open at D8 and D12 in all 21 fits and shut at D2, D3, D4, D19, D21, and
  D22 in all 21, as in experiment 420. All six hidden Primaries' gates opened in every fit
  (7 fits each).
- **Cleaned data:** each refit drops 57,531 to 94,724 training rows (summed over the 16 Primaries)
  that have no usable power reading, which includes the half-hours the `substation_zero` rule
  flags; the counts per fit are in each fit's JSON file. No training row has a missing ensemble
  unit shape (0 in every fold).

![A day-ahead and a 10-day forecast at D5 with a farm injected](example_forecast.svg)

## Checks

- **Median forecast minus the mean implied by the quantiles, on reverse-flow rows, pooled**
  (trapezoid
  integration of the 11 quantiles, a rough check): P0 -1.87 points (negative at every Primary), A
  +0.04 (-1.31 at D5 to +0.95 at D11), A-applied -0.10, recorded-weather A -0.14. The spread does
  not shift A's forecast.
- **Error of the median forecast beside the CRPS contrast** (mean absolute error of q50 on
  reverse-flow rows): P0 39.83, A 26.99, A-applied 15.05, recorded-weather A 12.24. It points the
  same way as every CRPS contrast. Audit trigger 3 did not fire.
- **Leave-one-Primary-out for A-applied against A:** A-applied stays better when any one of the six
  hidden Primaries is left out (4.55 when D9 is left out, up to 6.31 when D5 is). D9 carries the
  most weight. Trigger 4 did not fire.
- **Which triggers fired.** Trigger 2 fired in part: the A-applied result contradicts the plan's
  reason for retraining, and the explanation (lower, unstable learned capacity) is descriptive, not
  tested. No audit was run, because the S1 to S3 verdicts do not depend on it and the page states
  the cause as untested. Trigger 5 does not apply (A-applied is not nested in A). Trigger 7 is
  met: a live-service port would depend on the 15.7-point headline, which is a single replication of
  the laboratory result on the same injected series. Controls: S2 at the placebo Primaries held.
- **Mixed seed counts.** A is the mean of two seeds in S1, S3, and the A-applied contrast; P0 and
  A-applied use seed 0. The seed-0-only A-applied contrast is 6.54 (5.09 to 8.00) and the
seed-1-only
  contrast 4.45, so the A-applied contrast depends on which A seed is read; the sign never changes.
- **S2 pooled excludes D10** (real reverse flow); D10's A-minus-P0 harm is -0.043.
- **Caveats.** The reverse-flow rows come from six Primaries in one rotation each, so rotation and
  Primary are confounded and there are 42 blocks. The injected farm is real recorded output on top
  of real demand, and the test months are 2025-07 to 2025-09 and 2026-03 to 2026-06, so no winter
  month is scored. The scored lead range is 0 to 240 hours (days 1 to 10). The 2025-07 refit trains
  on 15 months, not 24. Unlogged load-shift events may sit in the readings.
- **Not done.** Avoidable-curtailment cost (see Departures), a climatology row, a second example
  forecast in winter, and a pretrain-then-fine-tune arm (experiment 418's recipe).

## Problems caused by the move

- **The docstrings of experiments 412, 417, and 420 tell the reader to source an `env.sh`, which
  does
  not exist in this repository.** I assembled `PYTHONPATH` and `DATA_PATH_INTERNAL` by hand (the
  script header of `e00424_fit.py` lists them). Command that showed it: `ls env.sh`.
- **`pgrep -f` patterns in a wait loop match the loop's own shell** unless one letter is in
  brackets;
  not a move problem, noted for the next worker.

## Review log

**Plan review (fresh Sonnet agent, before any fit).** 12 findings, all triaged. Accepted: a numeric
S1 floor in the bars and an outcome for a gain between the minimum detectable effect and the floor;
the label covers only the Primaries whose gate opened; S2 and S3 read on point estimates and stage 1
tests S2 at D5 only; missing-ensemble rows are kept and counted; the 9-hour delay, the archive
switch
on 2025-07-01, the `u99` constant, and the 6-hour-mean limit are stated; lead bands split at 144
hours; departures for climatology and multiplicity; prose fixes. Not changed: the draft-PR timing,
because the brief asks for a draft plan PR.

**Code review (fresh Sonnet agent, before any fit).** No leakage and no crash on first use. One
important finding, accepted and fixed: experiment 417's gate keeps each valid time's longest-lead
row,
which with ensemble inputs gave a gate shut at D8 and D12 (the gate now uses the shortest-lead row).
Minor, accepted: S1 code stricter than the plan; the ensemble-minus-laboratory interval was dropped;
the ensemble clearness index is not a clear-sky index (stated). Not changed: the applied-only fold
is rebuilt per job (efficiency only).

**Results review (fresh Sonnet agent, from the stored forecasts).** Every headline number was
reproduced (reverse-row CRPS of all six arms, the pooled and per-Primary gains with intervals, S2
and
S3, and identical key sets across arms). One discrepancy, accepted: the A-applied contrast used A's
seed 0 (6.54) while every other figure used A's two-seed mean (5.50); the page and the scoring
script
now use the two-seed mean. The checks above come from this review.

**Write-up review and prose review:** pending.
