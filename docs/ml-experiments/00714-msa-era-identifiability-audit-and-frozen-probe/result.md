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
page is the plan for a test of whether such a network could possibly separate them. No file has
been read and no feature has been computed for this experiment.

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
  refit. The AUROC must be 0.50 within 0.03, or the pipeline leaks.
- **Experiment 660's `shuffled_placebo`.** Score the same classifier on the generator's placebo
  events, which carry no real transfer, to confirm that a high AUROC needs a real shape.
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

## Compute

**The experiment runs on the CPU with at most 2 cores and no GPU, in 1 to 2 hours.** Unpickling
the bank's passes dominates (the 64 files hold about 5 GB), and the logistic fits take seconds.
The code limits BLAS to one thread and reads one pass at a time. The workstation rules of the
manager apply: a check-in before the run, and the `systemd-run` wrapper.

## Method

*To be completed with the finished write-up.* The Method section will list, as bullets, the
architecture (there is no neural network; the three input sets and the features), the training
regime (the classifier, the penalty, the splits, the second generator's parameters, and the seeds),
and the measurement (the AUROC, the bootstrap, the six contrasts, the placebos, and the audit
checks).

## Files

- `ml-experiments/00714-msa-era-identifiability-audit-and-frozen-probe/`: the scripts, to be
  written after the plan review. They will build the feature table from the bank, generate the
  second generator's passes, fit and score the classifiers, and draw the figures.
- `data/ml-experiments/00709-era-signal-in-the-candidate-part-1/worlds/bank_<month>_p<pass>.pkl`
  (read only, private): experiment 709's 64 injected passes, 16 for each of the months 2025-11,
  2026-01, 2026-04, and 2026-06. See the [experiment 709
  write-up](../00709-era-signal-in-the-candidate-part-1/result.md).
- `data/ml-experiments/00714-msa-era-identifiability-audit-and-frozen-probe/` (private): will hold
  the second generator's passes, the feature table, the fitted coefficients, and the scored events.
- `data/ml-experiments/briefs/msa_architecture/era_and_extensions.md` and `followup_queue.md`
  (private): the memo and the queue that define this experiment as follow-up F1.
