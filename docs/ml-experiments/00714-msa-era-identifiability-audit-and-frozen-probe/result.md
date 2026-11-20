---
id: "00714"
slug: msa-era-identifiability-audit-and-frozen-probe
title: >-
  Can the host substation's last 14 days tell a permanent level change from a temporary load move?
theme: lagged-power-and-error-correction
also_relevant_to:
  - switching-labels-in-the-forecast
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v2
size: >-
  Not simple. (1) It stores a feature table, the fitted logistic coefficients, and scored
  predictions per event, and no neural network; (2) more than one design defensibly tests the
  hypothesis (the reference level for the size of a change, the features, and the second generator
  of events); (3) it reads experiment 709's injection and storage code and experiment 660's
  generator, whose callers cannot be listed without searching the repository; (4) it touches no
  serving path or degradation rule; (5) it changes no contract. It gets a Sonnet review of the plan,
  a fresh Sonnet review of the code before the first run, a Sonnet review of the write-up against the
  printed report, and a Sonnet prose review. It runs on CPU only.
hypothesis: >-
  From the host Primary's own last 14 days of cleaned power, before the forecast origin, a
  hand-built classifier can tell an injected permanent level change from an injected temporary load
  move with an area under the receiver operating characteristic curve (AUROC) of at least 0.70 at 7
  and at 14 days after the onset, on a second generator of events that the classifier never saw.
  Rows of older days, and rows of the electrical neighbours, add to that AUROC.
prediction: >-
  Written before any data file is read. Host rows only, second generator: AUROC 0.60 to 0.70 at 7
  days after the onset, 0.60 to 0.72 at 14 days, and 0.65 to 0.78 at 28 days. On experiment 709's
  bank the same classifier scores 0.05 to 0.15 higher than on the second generator, because the
  bank's permanent changes are flat steps and its temporary moves follow the donor's daily profile.
  Adding six older rows raises the AUROC at 14 days by 0.02 to 0.08. Adding neighbour rows raises it
  to 0.85 or more on events whose mirror is metered (full or partial), changes it by under 0.02 on
  events with a hidden mirror, and returns to the host-only value when the neighbours are rewired.
  The shuffled-label placebo scores 0.50 within 0.03. This plan expects the host-only AUROC to fail
  the 0.70 threshold on the second generator at 7 days.
decision_rule: >-
  Six planned contrasts, so each interval is 99.17% (1 - 0.05/6), from a bootstrap that resamples
  whole passes (a pass is one injection run over the panel of 16 Primaries) within fold months and
  then resamples fold months. C1 is the host-only AUROC at 7 days after the onset on the second
  generator, C2 the same at 14 days, and C3 the same at 28 days. C4 is the gain in AUROC at 14 days
  from adding six older rows to the host rows, on the second generator. C5 is the gain at 14 days
  from adding neighbour rows to the host rows, on events whose mirror is metered. C6 is the host-only
  AUROC at 14 days on experiment 709's bank minus the same on the second generator. Host rows are
  enough to separate the classes when C1 and C2 both have a lower interval end of at least 0.70.
  Host rows are not enough when the upper end of C1 or the upper end of C2 is below 0.70, which
  tells us that no loss and no input stem can let a single-Primary matrix pass experiment 713's
  secondary readings R2 and R3 together. An injector artefact is the finding when C6's interval
  lies above +0.10. The neighbour rows pass when C5's lower end is at or above +0.05 and the AUROC
  on mirrored events is at least 0.85. The shuffled-label placebo must score 0.50 within 0.03, and
  the rewired-neighbour placebo within 0.03 of the host-only value, or the run is inconclusive
  pending an audit. The experiment ports nothing to production, so the usual threshold for the
  smallest gain worth porting does not apply.
benchmark: >-
  The shuffled-label placebo (AUROC 0.50), the rewired-neighbour placebo, and the shuffled placebo
  of experiment 660's generator. Experiment 660's classifier, which scored AUROC 0.55 when it tried
  to tell generated events from real windows, is context.
row_set_id: experiment_709_bank_64_passes_and_second_generator_128_passes
metric: >-
  AUROC for the label "permanent" against "temporary" at 7, 14, and 28 days after the onset, by
  input set, by event kind, by fold month, and by mirror state (full, partial, hidden); the
  Bayes-style ceiling of the same features (a depth-2 boosted tree, reported as a sensitivity only);
  the fitted coefficient of each feature; the calibration of the predicted probability; the number
  of events dropped for missing data.
---

**We ask whether the 14 most recent days of a substation's own power show if a level change will
last.** The substation is a Primary, one of the 16 Primary substations ("Primaries") of one
electricity network operator. A permanent change, such as a large generator or solar farm
connecting, starts a new era in the Primary's demand. A temporary load move transfers demand
between Primaries for days or weeks and then reverts. A forecast should follow the first and
should not carry the second forward. In
[experiment 709](../00709-era-signal-in-the-candidate-part-1/result.md), the learned era signals
that lowered the error most after an injected permanent change also raised the error most after an
injected temporary move, by 3.9 points of capacity for the arm `all`. The planned
[experiment 713](../00713-msa-axial-attention-demand-network-part-1/result.md) feeds the last 14
days of one Primary to an attention network (an MSA network, named after the multiple sequence
alignment that [Jumper et al. (2021)](https://doi.org/10.1038/s41586-021-03819-2) used in
AlphaFold 2) and reads whether the network separates the two classes in its secondary readings R2
(the gain after injected permanent changes) and R3 (the harm after injected temporary moves). This
page is the plan for a test of whether such a network could possibly separate them. No result on
the full bank or on the full second generator has been computed for this experiment. A smoke run of
the code on 16 bank passes and 16 second-generator passes checked that the code runs. Its AUROC
values are not reported and do not change the predictions below.

**The test builds a small classifier by hand and fits no neural network.** For each injected event
in the 64 passes that experiment 709 stored, and in a fresh set of passes from a second generator
of events, the classifier computes a few summary numbers from data before the forecast origin. It
predicts whether the event is a permanent change or a temporary move. The summary numbers are the
size of the change, the ratio of the change at night to the change at midday, the correlation of the
change with shortwave irradiance across days, and the share of the change that the sum of the
electrical neighbours mirrors. Three nested input sets show where the information is: the host's 14
rows, then those rows plus six older rows, then those rows plus the neighbours' rows. A
logistic-regression classifier cannot do worse than the best a network could do on the same
summaries by more than the nonlinearity it lacks, and a depth-2 boosted tree bounds that gap.

## Why the test comes first

**If the host's last 14 days cannot tell the classes apart, no loss and no input stem will make
experiment 713's network do so.** The memo behind this plan
(`briefs/msa_architecture/era_and_extensions.md`, section A, "The cheapest decisive test", in the
private data folder) argues that an auxiliary loss cannot create information that the input lacks:
experiments 590 and 594 found that the shape of a synthetic shift in training did not matter, and
noise on the same lag rows gave the same coverage. A host-only AUROC under about 0.70 at 7 to 14
days after the onset would close the question of a single-Primary matrix and send the era line to
neighbour rows or to labels, which [experiment
712](../00712-clean-era-labels-part-1/result.md) builds. A host-only AUROC at or above 0.70 would
justify a loss or a stem, and the coefficients would say which signature carries the information.
The test also gates follow-up F4 (a neighbour row group), which the follow-up queue starts only if
the host-only AUROC is at or above 0.70 and the neighbour rows add signal.

**The test can fail in a second direction, which concerns the injector and not the Primary.** The
permanent kinds of experiment 709 are flat level steps of −10%, −25%, and +15% of capacity and a
solar-shaped reduction. The temporary moves follow the donor's daily profile in part (experiment
660). A classifier could separate them by "flat means permanent", which no real event supports. A
high host-only AUROC on the bank and a low one on the second generator is the signature of that
artefact (contrast C6). Experiment 699 saw a gain on one generator's bank shrink from 0.035 to
0.002 on a second generator, which is why the second generator is part of the main design and not a
robustness check.

## Take-home logic and what each outcome says

**Four outcomes are possible, and each decides a follow-up.** The thresholds are fixed here, before
any data file is read.

- **Host rows alone score below 0.70 at 7 and 14 days on the second generator (C1 and C2).** The
  host's last 14 days mostly lack the information that decides whether a change lasts. No auxiliary
  loss, token, or stem on a single-Primary matrix can pass R2 and R3 together, and experiment 713's
  verdict on R2 and R3 is expected to show the trade-off of experiment 709 again. The era line then
  needs neighbour rows (follow-up F4, if C5 passes) or labels (experiment 712). Follow-up F3 (sparse
  older history rows) is judged on C4.
- **Host rows score at or above 0.70 on the bank and on the second generator.** The classes can be
  told apart from the host's own rows. An auxiliary loss on the injected class is then worth testing
  in a later experiment, and the fitted coefficients name the signature to give the network
  (change shape, elapsed days, or sunshine dependence).
- **Host rows score high on the bank and low on the second generator (C6 above +0.10).** The bank
  separates the classes by an artefact of the injector. No loss may train on the bank's class
  labels, and experiment 712's and experiment 713's injected readings need the second generator
  before their numbers are trusted.
- **Neighbour rows lift the AUROC to 0.85 or more on events with a metered mirror and by under 0.02
  on events with a hidden mirror (C5).** The mirror at an electrical neighbour is the working
  discriminator, as the memo expects, and only 2 real transfers are verified in the telemetry
  (experiments 580 and 660). The result then supports follow-up F4 and says it would work only
  where the neighbour is metered. A gain of under 0.05 from neighbour rows stops F4.

**An AUROC measured on injected events bounds what injected events can teach, and says nothing
about the rate of real events.** Real permanent changes of 1 MW or more are rare, and the base rate
favours a temporary move (the memo cites experiment 712's plan). The page therefore reports the
AUROC and not an accuracy, and states the base rate it would need to turn an AUROC into a decision.

## Design

**The classifier is built in three nested steps, each adding rows that sit before the forecast
origin.** The label is "permanent" for an injected era and "temporary" for an injected load move,
including the moves of 60 to 120 days that have not reverted by the origin (these look permanent
and are the hard cases; a second set of readings drops them).

1. **Host rows.** The 14 most recent cleaned days of the host Primary's power, as shares of
   capacity, at 48 half-hours a day. The features are the median change against a reference level,
   the ratio of the change at 00:00 to 06:00 against the change at 11:00 to 14:00, the correlation
   across days of the daily change with the daily mean shortwave irradiance (the weather inputs are
   the ones the Candidate sees at forecast origin, never actual weather after the origin), and the
   number of days with the change at the same sign. The reference level is the Candidate's
   lag-free forecast of experiment 667 where the bank stored it. Where it is not stored for the
   bank's passes, the fallback is a weather-only linear fit trained on months before the fold. The
   choice is settled in the plan review and recorded under Method before the first run.
2. **Older rows.** Add six rows of single same-weekday days at ages 21, 28, 42, 63, 91, and 364 days
   before the origin, when the injected panel stores them. Rows older than the onset are real,
   uninjected power, so they give the classifier a clean level to measure the change against. The
   feature set adds the change measured against the median of the older rows and the spread of the
   older rows.
3. **Neighbour rows.** Add the cleaned power of the electrical neighbours of the host, from the
   reconciled interconnection table of experiment 395. The feature is the share of the host's change
   that the sum of the neighbours mirrors, with its sign. The rewired placebo replaces each real
   neighbour set with the neighbours of another Primary that is not electrically linked to the host.

**The classifier is a logistic regression with an L2 penalty, fitted separately for each origin,
with whole passes held out.** Origins are 7, 14, and 28 days after the onset, so days since the
onset is fixed within a fit and is not a feature (this gives the classifier the true onset, which an
online detector would not know, so the result is an upper bound for the information in the rows).
The penalty strength comes from an inner grouped split by pass. Folds hold out whole passes and
whole fold months, so no event and no pass seen in fitting is scored. The standardising mean and
standard deviation come from the training passes only.

**The second generator is a fresh draw that differs from experiment 709's in kinds and sizes.**
It reuses [experiment 660's
generator](../00660-shared-generator-of-injected-load-transfer-events/result.md) for the temporary
moves, with new random seeds, a new host and neighbour draw, and a different duration draw. It adds
three permanent kinds that experiment 709 lacks: a step that follows the host's daily profile in
proportion, a daylight-only step, and a quantised step in 2 or 3 levels (the signatures in the
memo's section A table, from experiments 694 and 196). Step sizes come from a grid of −15%, −5%,
+8%, and +30% of capacity, none of which experiment 709 used. The generator writes 32 passes for
each of the four fold months, which is 128 passes, 256 permanent changes, and 256 temporary moves.
The code review checks that the second generator shares no function with experiment 709's
injection beyond the base panel.

**The leakage statement.** Every feature is computed from power and weather at valid times strictly
before the forecast origin. The injector's class, kind, onset, duration, and mirror parameters
never enter a feature, and the class enters only as the label. No pass appears on both sides of a
split. The scaler and the penalty are chosen on training passes only. The neighbour feature uses
neighbour power before the origin, so a mirrored move shows the mirror in the data the classifier
sees, which is the same information a live forecast has. The code review checks the origin cut on a
shuffled copy of the post-origin power, where the AUROC must not move.

**Placebos and controls.**

- **Shuffled labels.** Permute the class labels within each pass, keeping the pair structure, and
  refit. The AUROC must be 0.50 within 0.03, or the splits are wrong. The shuffle breaks any leak
  through the features, so this placebo checks the cross-validation bookkeeping only. The test
  that scrambles the data after the cut checks the features for leakage.
- **No-event placebo (the reading of experiment 660's `shuffled_placebo`).** Experiment 660's
  placebo events carry no real transfer. Scored at the original onsets, the windows hold no
  event, so the classifier is fitted on genuine events and scores the real, uninjected panel at the
  same hosts and onsets. The AUROC must be 0.50 within 0.05; a larger gap means that the draw of
  hosts differs between classes. Re-timing the events with `shuffled_placebo` is not run, because
  the re-timed windows still hold the injected shape and could not show that a real shape is
  needed.
- **Rewired neighbours.** Described above. The AUROC with rewired neighbours returns to the
  host-only value within 0.03.
- **Known-answer.** Feed the classifier the true onset-minus-baseline change without noise on the
  bank. The AUROC must be above 0.95, or the features cannot carry the class at all.

**Sensitivity readings, all labelled post hoc unless named here.** Drop the 60 to 120 day moves
(pre-registered). Fit a depth-2 boosted tree on the same features (pre-registered, a ceiling for
the logistic regression). Report AUROC by event kind and by mirror state (pre-registered). Report
the order of the fit's features by absolute standardised coefficient.

**The frozen probe is a second stage that runs only if experiment 713 stored the vectors.** Addition
3 of the memo's section D has experiment 713 save the 64 numbers per target token of `M*`, `P`, and
`K` for the injected bank. If those files exist when this experiment starts, a linear probe on
them predicts the class, the kind, and the days since the onset, with whole passes held out, and the
probe on arm `P`'s vectors is the placebo. If the files do not exist, the page says the probe was
not run, and the verdict rests on the classifier alone. The probe has no decision rule of its own
and is reported as context for experiment 713.

## Standing practices

- **Cleaned data.** All rows drop the half-hours that the `substation_zero` rule flags. The page
  prints the number dropped per fold month.
- **D15 and D18 cut.** Every scored month falls after 2025-03-01, so the cut drops no row. The page
  says so.
- **Stored data reused.** The experiment reads experiment 709's stored bank
  (`worlds/bank_<month>_p<n>.pkl`) and writes nothing under the bank. It does not rerun any 709
  script.
- **Refit monthly, saved models, the leaderboard.** None applies, because no forecast model is fitted
  or scored. The page stores the logistic coefficients as JSON.
- **Load-shift scenarios.** No forecast is scored, so the four scenarios do not apply. The page
  shows at least 6 example time series of the host's 14 rows, from 14 days before to 7 days after
  an event, across both classes and both mirror states, with the injected change marked. No
  generator series appears. The unlogged-event caveat is stated: the logs may miss real events, and
  the experiment scores injected events only.
- **Earlier advice followed.** Experiment 709's advice to score an era signal on injected eras and
  injected temporary moves together, and experiment 699's to test on a second generator, are both
  followed. The advice to store the prevailing output concerns forecasts and does not apply.

## Secondary readings: shape of the change, step frequency, and the mirror

**The maintainer's working definition decides which cues are primary.** A load shift moves a
roughly constant number of megawatts from one substation to another, often with a mirror at a
neighbour. Every other level shift, such as a new gas generator or a large solar farm, changes the
daily shape of the load and its response to sunlight and price. The shape-flatness of the change
(how constant the megawatts are across the half-hours of the day) and the neighbour mirror are
therefore the two cues that the classifier is built around. The host feature set (the pre-registered
set of the Design section) holds the shape-flatness as one of its six features, and the neighbour
feature set holds the mirror.

**The pre-registered contrasts C1 to C6 and the thresholds stay as written above.** The readings in
this section are secondary: they add detail and carry no decision rule.

- **Each cue alone and both together.** For each permanent kind in turn, the AUROC for telling that
  kind from the temporary moves, using the shape-flatness alone, the mirror alone, and both, at
  7, 14, and 28 days after the onset. The classifier of each cue set is fitted on the events of a
  world with whole fold months held out. Three groups of temporary moves serve as the negatives:
  all, those with a metered mirror (full or partial), and those with a hidden mirror.
- **Kinds in the second generator.** A flat step (the case where shape cannot help), a step that
  follows the host's daily profile, a daylight-only step, and a quantised step. Experiment 709's
  bank adds its own four kinds (steps of -10%, -25%, +15%, and a solar-shaped reduction).
- **A gas-generator kind.** A fifth permanent kind in a separate world of 32 passes (8 per fold
  month, 64 permanent changes and 64 temporary moves): constant-megawatt steps that switch on and
  off at random times within and across days, with a running share that changes from week to week.
  It uses the pulse rows of experiment 660's generator, with steps of 4% to 12% of capacity (the
  range of the real gas-engine episodes of a Primary that experiment 194 describes). The
  generator's own time series is never plotted. These passes are not part of the 128 passes that
  carry C1 to C6.
- **The step-frequency cue.** The number of up and down steps per day in the post-onset rows, and
  the fraction of half-hours at the high level, from a two-threshold trigger on the running median
  of the change. The cue is a secondary feature set and is not part of the host set.

**Pre-registered prediction for the gas-generator kind (written before any run on the full worlds,
after the smoke primary table was read, and without looking at the per-cue readings of the smoke
run).**
The shape-flatness cannot separate an intermittent step from a sustained temporary move, because
both are constant-megawatt steps, so its AUROC for that pair is predicted to lie between 0.40 and
0.60. The step-frequency cue alone is predicted to score 0.70 or more against sustained temporary
moves (temporary moves still in force at the data cut of the origin), because an intermittent step
switches several times per day and a sustained move switches rarely. The mirror cue is predicted to
score 0.50 within 0.10 against temporary moves with a hidden mirror, and to score above 0.50 only
because the temporary moves with a metered mirror carry a mirror. A flat permanent step is the case
where shape cannot help, so the shape-flatness AUROC for a flat step against temporary moves is
predicted to lie between 0.40 and 0.60. A permanent kind that follows the host's daily profile or
daylight is predicted to score above 0.65 on the shape-flatness alone.

**The 82 labelled real events are a descriptive check only.** The load-transfer logs hold 82
labelled events, 22 of them in the scored year. The page may print the cue values of those events
where the telemetry shows them, to compare against the injected events. No classifier trains or
scores on them, because experiment 580 confirmed 1 of the 23 labelled events of at least 7 days
in the telemetry.

## Assumptions and risks

**A load shift is assumed to look like a roughly constant decrease in megawatts at the donor
substation and a matching increase at the receiver.** A load shift switches a circuit, and a circuit
can carry any behaviour. In the extreme, a switched circuit could look exactly like a generator or a
solar farm connected to a substation for a while, or moved from one substation to another. The
maintainer has never seen that in the data. This experiment assumes that a load shift moves a
substantially large number of connections, so that in aggregate it is a roughly constant change in
megawatts. A load shift of the other kind is not modelled, and the classifier would misread it as a
permanent change.

**The classifier is given the true onset.** An online detector would not know the onset, so an AUROC
from this experiment is an upper bound on what the rows can tell.

**The weather-only reference level is noisy.** Its error, a few points of capacity at a Primary, is
the floor on the size of change that the host rows can resolve, and the older rows exist to lower
that floor.

**ERA5 stands in for the weather analysis that a live forecast reads at the origin.** The
reanalysis arrives about 5 days late in live use, so the weather of the last 5 days before the
origin is read more accurately here than a live forecast could read it. The shortwave correlation is
the only feature that reads weather directly.

## Compute

**The experiment runs on the CPU with at most 2 cores and no GPU, in 1 to 2 hours.** Unpickling
the bank's passes dominates (the 64 files hold about 5 GB), and the logistic fits take seconds.
The code limits BLAS to one thread and reads one pass at a time. The workstation rules of the
manager apply: a check-in before the run, and the `systemd-run` wrapper.

## Method

**Architecture (no neural network).**

- Each event at each origin (7, 14, or 28 days after the onset day, at 09:00 UTC) becomes one row of
  summary numbers computed from the 14 days of power that end 6 hours before the origin (the
  Candidate's data cut), as shares of capacity.
- **Reference level for the size of a change: a weather-only linear fit, fitted per Primary and per
  fold month.** Experiment 667's stored forecasts cannot serve as the reference. They start on
  2025-11-01, so the 14 days before the onset of the 2025-11 fold are not stored. Their
  normal-running-arrangement output reads the Candidate's lags, so it is not lag-free. The fit
  regresses the Primary's power on 96 calendar dummies (half-hour of the day, weekday or weekend),
  daily mean ERA5 temperature and its square, ERA5 shortwave radiation and its square, and the annual
  cycle, ridge penalty 0.001, over the 730 days that end 28 days before the start of the fold month.
  It reads real, uninjected power only.
- **Host features (six).** The median change against the reference level; the shape-flatness of the
  change (the absolute mean of the daily profile over its root mean square, 1 for a constant
  number of megawatts); the night-to-midday contrast; the correlation across days of the daily
  change with daily mean shortwave; the share of days with the change at the sign of the median; and
  the change of the last 3 days minus the median change.
- **Older-row features (four).** Six single same-weekday days at ages 21, 28, 42, 63, 91, and 364
  days before the origin, left out when the day ends after the onset: the change against their
  median, their spread, the ratio of the two, and the shape-flatness of the change against their
  median. Experiment 709's injected panels hold the 364-day rows, because each stored panel covers
  the whole grid of 119,760 half-hours from 2019-09-01 to 2026-06-30.
- **Neighbour features (two).** Each site (the host and each neighbour) is measured against its own
  mean change at each half-hour of the day over the 14 days before the onset, which removes the
  reference level's error that neighbouring Primaries share. `mirror_share` is the negated median
  post-onset step of the neighbours' summed change in MW over the host's median step in MW.
  `mirror_corr` is the correlation of the two series before the onset minus the correlation over
  the whole span from 14 days before the onset to the data cut. The neighbours of an event are the
  host's electrical neighbours, without the sites of other events of the same pass.
- **Step features (two, secondary).** Steps per day and the fraction of half-hours at the high level.

**Training regime.**

- **Classifier.** A logistic regression with an L2 penalty (numpy and scipy, because scikit-learn is
  not a declared dependency), fitted separately for each origin. The penalty comes from an inner
  split of whole passes into 4 groups, over 0.01, 0.1, 1, 10, and 100. The standardising statistics
  and the fill value for a missing feature come from the training passes.
- **Boosted tree (sensitivity).** XGBoost, which is a declared dependency, with depth 2, 80 trees, and
  learning rate 0.1, on the host and host-plus-neighbour features.
- **Worlds.** Experiment 709's 64 bank passes (read only; the unpickler allows four globals, and
  `e00709_signal.Signal` is replaced by a stub, because importing it needs the chain
  `e00709_classify`, `e00687_features`, `e00598_signals` and further modules) and 128 passes of the
  second generator, 32 for each fold month, with seeds 714,100,000 and up. The second generator's
  passes have onsets at a random half-hour on the first or second day of the month. No event of a
  pass sits in the neighbour set of another event of the pass.
- **Which Primary is scored in a temporary move.** In experiment 709's bank the scored Primary of a
  temporary move is always the donor, so its change is always a decrease, and 25% of its permanent
  changes are increases. A classifier that read only the sign of the change would therefore score
  above 0.5 on the bank. In the second generator, half of the temporary moves are scored at the
  receiving Primary (a gain), so the sign mix of the temporary moves matches the sign mix of the
  permanent changes (half increases). This design change follows from a code review and not from
  any AUROC. The page prints a sign-only AUROC beside C1 to C3 and C6, and a secondary C6 on the
  events with a decrease only.
- **Splits.** Leave one fold month out, so no pass and no fold month seen in fitting is scored.
- **Compute.** CPU only, 2 cores. `e00714_threads.py` sets the BLAS and OpenMP thread counts to 1
  before numpy loads, and XGBoost runs with one thread.

**Measurement.**

- **Metric.** The AUROC for the label "permanent" against "temporary", pooled over the out-of-fold
  predictions of all four fold months.
- **Intervals.** A bootstrap that resamples whole passes within fold months and then fold months,
  10,000 draws, at 99.17% for the six planned contrasts (C1 to C6). The predictions are not refitted
  per draw. Secondary readings use 95% intervals, labelled secondary. With 4 fold months resampled
  with replacement, 1.6% of draws repeat one fold month four times, which is more than the 0.42% of
  each tail, so the lower ends of C1 and C2 sit close to the worst single fold month's AUROC.
  "Host rows are enough" is therefore a statement about every fold month.
- **Readings beside the six contrasts (secondary).** The AUROC of the sign of the change alone; C6
  on the events with a decrease only; C1 to C3 and C6 on the permanent changes and the temporary
  moves still in force at the data cut (a move that has ended before the cut shows only the
  reference level's error); the AUROC of each raw cue in the direction the plan states.
- **C5 and its companions** are computed, point estimate and interval, on the one subset that each
  names (the mirrored subset, or the hidden subset).
- **Mirror subsets.** The mirrored subset holds every permanent change and the temporary moves with a
  full or partial mirror. The hidden subset holds every permanent change and the temporary moves
  with a hidden mirror.
- **Placebos.** Class labels shuffled within each pass (20 permutations); neighbours rewired to those
  of an unlinked Primary; the no-event placebo, which scores the classifier on the real panel at the
  same hosts and onsets; and the noiseless known-answer, which replaces the reference level by the
  real power so that the change is the injected change exactly.
- **Audit checks.** A test scrambles the power, the reference level, and the weather at and after the
  data cut and requires every feature to stay identical.

## Files

- `ml-experiments/00714-msa-era-identifiability-audit-and-frozen-probe/e00714_common.py`: constants,
  the bank loader, the weather-only reference level.
- `.../e00714_generator2.py`: the second generator (worlds `g2` and `gas`), built on experiment 660's
  generator.
- `.../e00714_threads.py`: sets the BLAS and OpenMP thread counts to 1 before numpy loads.
- `.../e00714_features.py`: the host, step, older-row, and neighbour features.
- `.../e00714_build.py`: generates the second generator's passes and builds the feature tables.
- `.../e00714_model.py`: the logistic regression, the boosted tree, the cross-validation, and the
  bootstrap.
- `.../e00714_analyse.py`: the six contrasts, the placebos, the secondary readings, and the outcome
  table.
- `.../e00714_figures.py`: the three figures.
- `.../test_e00714.py`: unit and known-answer tests on synthetic data.
- `data/ml-experiments/00709-era-signal-in-the-candidate-part-1/worlds/bank_<month>_p<pass>.pkl`
  (read only, private): experiment 709's 64 injected passes, 16 for each of the months 2025-11,
  2026-01, 2026-04, and 2026-06. See the [experiment 709
  write-up](../00709-era-signal-in-the-candidate-part-1/result.md).
- `data/ml-experiments/00714-msa-era-identifiability-audit-and-frozen-probe/` (private): will hold
  the second generator's passes, the feature table, the fitted coefficients, and the scored events.
- `data/ml-experiments/briefs/msa_architecture/era_and_extensions.md` and `followup_queue.md`
  (private): the memo and the queue that define this experiment as follow-up F1.
