---
id: "00712"
slug: clean-era-labels-part-1
title: >-
  Do labels for permanent changes in demand, made without hindsight, help the demand-forecasting
  neural network?
theme: switching-labels-in-the-forecast
also_relevant_to: [lsc-detectors, retraining-and-training-history]
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores per-row forecasts, network weights for each arm, fold, and seed, and a
  store of language-model calls; (2) more than one design defensibly tests the hypothesis (the
  detector, the check days, the prompt, and the register rule); (3) it reuses the Candidate's
  training and injection code from experiment 709 and the prevailing-head code from experiment 667,
  and the callers of that code cannot be listed without searching the repository; (4) it touches no
  serving path or degradation
  rule; (5) it changes no contract. Full set of reviews: plan, code, results, write-up, prose.
hypothesis: >-
  Era labels built by a detector and a language model from data before the forecast origin, and fed
  to the Candidate neural network with the delay a live forecast would impose (a label made on day d
  reaches the network at the 09:00 UTC forecast of day d + 1), lower the error of the
  normal-running-arrangement (NRA) forecast in the weeks after an injected era by at least 1.5
  points of capacity. The error is the continuous ranked probability score (CRPS). The labels do not
  raise the error by more than 0.1 points of capacity after an injected switching event. An era is a
  permanent change in a substation's demand, and an injected era or switching event is a synthetic
  one that experiment 709 added to real histories, so that its date is known.
prediction: >-
  Written before any label or fit. Arms are defined under "The arms" on this page: `none` is the
  Candidate with no era signal, `det` labels every detection `era`, `llm` is the language-model
  labeller, `ecr` labels from the Embedded Capacity Register alone, `llm-ecr` gives the language
  model the register rows too, and `ceiling` reads the true class. All figures are CRPS differences
  from `none`, in points of capacity, on the 128 injected eras and 128 injected switching events in
  the 64 synthetic passes that experiment 709 stored. `det`, eras, 7 to 28 days after the onset:
  -3.2 to -4.5. `ceiling`: within 0.3 of `det`. `llm`, eras: -2.0 to -3.5. `llm`, switching events
  at the donor Primary: -0.1 to +0.1. `det`, switching events at the donor: +3 to +5. `llm-ecr`
  minus `llm`, eras with a register row, 0 to 7 days: -0.5 to -1.5. `ecr`, eras with a register row,
  7 to 28 days: -2 to -4. `ecr`, switching events with a decoy register row, donor: +1 or more.
  `llm` NRA output minus A0-prev, eras, 7 to 28 days: -0.5 to +1.5; we do not predict the sign.
  Every new arm on the real rows of the four test months: within ±0.05. At least 70% of detected
  injected eras are labelled `era` within 7 days of the alarm, and at most 5% of detected injected
  switching events are ever labelled `era`.
decision_rule: >-
  Twelve planned contrasts, so each interval is 99.58% (1 - 0.05/12), from a bootstrap that
  resamples whole injected passes and whole fold months. K1 is `llm` minus `none` on injected eras,
  7 to 28 days after the onset. K2 is the donor's error after injected switching events, and K4 is
  the error on all real rows of the four test months. K3 is `llm` minus `soft` on injected eras.
  Supported: K1 is at or below -1.5 points with an interval that excludes zero, K2 and K4 pass, and
  K3's upper end is at or below +0.5. Small gain: K1's interval excludes zero in the gain direction
  but K1 is above -1.5, and K2 and K4 pass. Partly supported: K1, K2, and K4 pass and K3 fails. Not
  supported: K1's interval contains zero. Not supported for adoption: K2 fails (the donor's error
  after injected switching events rises by more than 0.1 points, or the upper interval end exceeds
  +0.2) or K4 fails (all real rows: point estimate above +0.03, or upper end above the stored
  placebo's), whatever K1 shows. Inconclusive pending an audit: the `none` reproduction gate fails,
  or the stored placebo arm differs from `none` by more than 0.06 points on the injected passes.
benchmark: >-
  `none` (the Candidate with no era signal) and the stored 709 arms `oracle`, `placebo`, `all`,
  `soft`, `rule`, `stat`, and `hmm`; A0-prev (the prevailing output of `none`); climatology; the
  manual heuristic.
row_set_id: experiment_677_monthly_refit_rows_valid_2025_11_to_2026_06
metric: >-
  CRPS in points of capacity on injected eras, injected switching events (donor and recipient), and
  real rows; label-level confusion counts with Wilson intervals; mean absolute error (MAE) of the
  median, normalised by capacity;
  q01, q05, q95, and q99 pinball loss and coverage; the four load-shift scenarios;
  avoidable-curtailment cost; detection and classification delay.
---

**We ask whether a forecast can use labels that say "this change is permanent" when those labels are
made only from data available at the time.** In
[experiment 709](../00709-era-signal-in-the-candidate-part-1/result.md), an earlier experiment in
this series, the demand-forecasting neural network (called the Candidate
below) forecast demand at the 16 Primary substations ("Primaries") of one electricity network
operator. Experiment 709 gave the Candidate the true dates of permanent changes, called eras, such
as a solar farm connecting to a Primary. Because real eras are rare, experiment 709 also added
synthetic eras and synthetic temporary moves of load between Primaries (switching events) to real
histories. These synthetic events are called injected, and their true dates are known. With the true
dates, the error in the weeks after an injected era fell by 7.4 points of capacity (the oracle arm of
experiment 709). The five versions of the era signal that experiment 709 built from data available
at the time (`rule`, `soft`, `hmm`, `stat`, and `all`) lowered that error by less, from 0.3 to 2.9
points. The versions that helped most also raised the error after an injected switching event (`all`
by 3.9 points), because they treated the move as permanent.

**The experiment adds a language model between the detectors and the Candidate.** A detector flags
each sudden shift in a substation's load as a flagged shift. A language model reads the load around
the flagged shift, as shares of the substation's capacity, and labels the shift a permanent change,
a temporary load move, or noise. The Candidate then trains and forecasts with those labels, each
delayed until the first forecast time after the label could have been made. This page is the plan.
No label has been made and no neural network has been trained for this experiment. The figures
taken from experiment 709 and from a small trial of the language-model calls (the cost trial,
described under Cost, billing, and staging) are provisional, because experiment 667 stored only two
random seeds.

**The plan runs in two stages, and stage 2 runs only if stage 1 passes its gate.** Stage 1 labels
the real record of the 16 Primaries, without the injected bank, then fits the arms that need no
language-model call. Stage 2 labels the injected bank and fits the remaining arms. The injected
bank is the 64 synthetic passes stored by experiment 709, where a pass is one injection run over
the panel of 16 Primaries with two eras and two switching events. No labelling run starts until this
plan's pull request is merged. The ML experiment manager, the session that allocates GPU slots to
experiments, then names the go, and only after the maintainer has said so.

## What is tested

**The hypothesis is a gain of at least 1.5 points of capacity with no rise in error after a
switching event.** Labels made without hindsight lower the error of the normal-running-arrangement
(NRA) forecast by at least 1.5 points of capacity in the 7 to 28 days after an injected era, and do
not raise the error after an injected switching event. The error is the continuous ranked
probability score (CRPS), in points of the Primary's effective capacity. The threshold of 1.5 points
is 0.5 points for each of three components the pipeline would add to production: a daily detector
job, a language-model labeller with an audit trail, and a register ingest. The 0.5 points is a
stated convention. Dated changes are rare: the register dates about 0.04 connections of 1 MW or more
per Primary-year. A gain of 1.5 points in the 28 days after an era is therefore about 0.005 points
averaged over all rows. The maintainer decides whether the gain in those weeks justifies the
pipeline.

**Five new arms are compared with eight arms stored by experiment 709.** All use the stage 3 recipe
of experiment 709 on four test months (2025-11, 2026-01, 2026-04, and 2026-06) with two seeds. The
Embedded Capacity Register (ECR) is the distribution network operator's list of connected and
pending generators.

| Arm | Class of each flagged shift comes from | Calls |
|---|---|---|
| `det` | none: every flagged shift is `era` from its alarm day | 0 |
| `llm` | the language model, reading the load only | yes |
| `ecr` | none: the first alarm in a register watch window is `era` | 0 |
| `llm-ecr` | the language model, which also sees nearby ECR rows | yes |
| `ceiling` | the true class, revealed on the alarm day; it trains on hindsight and is never a candidate for adoption | 0 |
| `none`, `oracle`, `placebo`, `all`, `soft`, `rule`, `stat`, `hmm` (hidden Markov model) | experiment 709's | stored |

K6b below is not an arm. It re-scores `ecr` on the injected bank with each register row visible 30
days before the connection date, which needs bank inference jobs and no new fit.

**K1 is the gain, K2 and K4 are the vetoes, and the other nine contrasts explain the result.**
Twelve contrasts are planned, so each two-sided interval is 99.58%.

| Contrast | Arms | Rows | Reads as |
|---|---|---|---|
| K1 | `llm` minus `none` | injected eras, 7 to 28 days | the gain; threshold -1.5 |
| K2, K2-ecr, K2-llm-ecr | `llm`, `ecr`, `llm-ecr` minus `none` | injected switching events, donor Primary | veto: above +0.1, or upper end above +0.2 |
| K3 | `llm` minus `soft` | injected eras, 7 to 28 days | `llm` loses at most 0.5 points of capacity relative to `soft` (upper end) |
| K4 | `llm` minus `none` | all real rows of the four test months | veto: at most +0.03, and an upper end no higher than the stored placebo's |
| K5 | `llm-ecr` minus `llm` | eras with a simulated register row | what the register adds |
| K6, K6b | `ecr` minus `none`; `ecr` with the row visible 30 days early minus `none` | the same eras | the effect of the register alone; K6b is simulated |
| K7 | `llm` minus `det` | switching events, donor | the harm the language model removes |
| K8, K9 | `llm` NRA output minus A0-prev | eras; switching events at the donor | whether the labelled NRA forecast beats the prevailing forecast |

**A0-prev is the Candidate's prevailing output.** It is the NRA forecast mixed with a shifted copy
of itself by a survival probability, reused from the heads of experiment 667. Computing A0-prev
needs no language-model call and no GPU, and the manager may defer it. If A0-prev is deferred, a
`supported` verdict means only that `llm` beats the neural network with no era signal.

## Gates before any score is read

**Stage 1 ends at a gate on the detectors, before any bank call.** The gate reads `det` minus
`none` on injected eras at 7 to 28 days, with `ceiling` beside it. The two detectors are a
change-point detector and a year-on-year detector, both from experiment 709. If `det` is above -1.5
points, the two detectors alarm too late or too rarely for `llm` to pass K1, because `llm` can only
relabel detections the detectors have already made. The experiment then stops as `not supported`
after the stage 1 labelling. Stage 2 starts with a pilot of two passes per fold month (about 60
calls). The pilot stops the run if the language model labels three or more of the detected
switching events in the pilot as `era` with confidence 0.8 or more, or labels fewer than 40% of the
detected eras in the pilot as `era` within 7 days of the alarm.

**Two reproduction gates stop the run before any score is read.** The new script must reproduce
experiment 709's `none` forecast to within 1e-5 on the median for one fold and seed. A fixed probe
of 10 calls asks the language model to name the year and month from real prompts. The prompt is
pinned (fixed as the prompt used for every later call) only if at most 3 of the 10 years are right.

## The label pipeline

**Each label is built from data up to the end of the day before the forecast origin.** A forecast
origin is the time at which a forecast is issued, here 09:00 coordinated universal time (UTC). A
candidate is one flagged shift. The language model sees the substation's load around the candidate,
as shares of capacity and as offsets from the check day, with no name, date, weekday flag, or
megawatt value. Checks run on the alarm day, 7 days later, and 21 days later, and on the day a
closure becomes known. A label from check day d reaches the neural network at the origin of day
d + 1. A perturbation test changes every reading from 00:00 of day D onward. The test then asserts
that every prompt table for a check before day D has an unchanged 256-bit Secure Hash Algorithm
(SHA-256) fingerprint. Labels are made once and replayed from the call store, because the language
model's sampling cannot be fixed.

**Example: the 2025 solar connection at Primary D8 (register date 2025-03-24, 8.91 MW).** The
stored `stat` arm of experiment 709 entered the `era` state 32 days after the register date, and
then withdrew the class when the level fell back. The columns are the day the state was written,
the state, the era identifier, the estimated onset, and the first forecast origin that reads the
state:

```text
entry day    stat state  era_id  onset_est    first origin reading it
2025-04-25   era         4       2024-12-21   2025-04-26 09:00 UTC
2025-04-26   era         6       2025-03-29   2025-04-27 09:00 UTC
2025-06-08   switch      4       2025-03-29   2025-06-09 09:00 UTC  (withdrawn=1)
2025-06-09   era         4       2024-12-21   2025-06-10 09:00 UTC
```

In the new pipeline the language model's answer, not the detector's alarm, ends an era: the state
changes only if a closure check answers `switch` or `noise` at confidence 0.8 or more.

## Cost, billing, and staging

**Labelling bills to the maintainer's shared plan quota, so the dollar figures are prices, not
charges.** The cost trial's init event showed `apiKeySource = none`, which means the calls ran on
the subscription login. Every US$ figure below is the price the command-line tool reports, which is
the API-equivalent price. The rate-limit events do carry an account-wide seven-day utilisation
(`unifiedWindows.seven_day.utilization` = 0.32, constant across all 22 calls, shared by every
session), so the trial's own share is below the resolution of that field and cannot be attributed.
The labelling script counts calls and tokens, and the plan treats the quota as the limit.

**The cost trial on a single Primary measured cost and nothing else.** It sent 22 calls (one probe
and 21 check calls on 5 candidates) with 167,708 input tokens and 10,071 output tokens in 198.9 s,
at a reported US$0.743. The 21 check calls are 20 scheduled checks (4 per candidate) and 1 retry
after a reply failed the schema. That gives 4.2 check calls per candidate with no early stop, or 4.0
without the retry. The mean price is US$0.0353 per check call and US$0.0338 per call over all 22
including the probe. The median is 7,800 input tokens per check call. The labels from the trial
were never read and are not results.

**The call counts below cover different arms, and the totals close.** Stage 1 covers the real
record (about 92 candidates): arm `llm` takes about 245 calls (the design's early-stop estimate) to
386 calls (92 candidates at the trial's 4.2 calls each). Arm `llm-ecr` adds about 45 calls, the
billing probe, year probe, and prompt development add 21, and a repeat sample for agreement adds 30.
Stage 2 covers the injected bank (182 candidates), which uses 2 checks per candidate (the alarm day
and 7 days later), not the trial's 4.2. Arm `llm` therefore takes at most 364 calls, and arm
`llm-ecr` adds about 104 calls for about 52 candidates. Prices use US$0.0353 per call and 7,957
input tokens per call.

| Stage | Block | Calls | Input tokens | Reported price |
|---|---|---|---|---|
| 1 | Real record, arm `llm`, 16 Primaries | about 245 to 386 | 2.0 to 3.1 million | about US$9 to 14 |
| 1 | Same stage, plus `llm-ecr` (45), probes and development (21), and the repeat sample (30) | about 340 to 480 | 2.7 to 3.8 million | about US$12 to 17 |
| 2 | Injected bank, arm `llm` only (182 candidates x 2 checks) | at most 364 | about 2.9 million | at most about US$13 |
| 2 | Same stage, plus `llm-ecr` (about 104) | at most about 470 | about 3.7 million | at most about US$17 |

**Staging and limits.** At most two calls run in parallel. The run stops on any rate-limit event
and reports to the manager. The labelling script prints running totals of calls, tokens, and
reported price every 25 calls. `BudgetGuard` takes explicit limits on calls, tokens, and reported
price for each stage, and the script never calls its `with_round` method, which caps any round at
US$15.

**Scaling to 2,500 substations is an argument, not work to run.** At 0.8 candidates per
substation-year and about 4.2 calls each, 2,500 substations with the same 7.2 years of record as
the 16 Primaries would need about 480 million input tokens once to label the history and about 67
million a year after that.

## Compute

**The fits run on the manager's GPU slot, and none starts until the manager names a share.** A fit
takes about 3.5 minutes (50 s to build the fold's rows and 158 s to train) and peaks at 1.45 GiB of
graphics processing unit (GPU) memory and 6.3 GiB of host memory. A bank inference job takes about
2.5 minutes. The main set is 71 fits: 1 for the reproduction gate, 40 for the screen (5 new arms,
4 test months, 2 seeds), and 30 for the 2025 connection at D8 (5 new arms, 3 months, 2 seeds). The
fits alone take about 4 job-hours. Adding 70 forecast jobs at about 1 minute and 64 bank inference
jobs (40 for the screen and 24 for the register sensitivities) gives about 8 GPU job-hours, or
about 4 hours at two jobs. The A0-prev runs use at most 4 CPU cores for about 4 to 7 job-hours. The
experiment follows the standing practices of experiment 709's recipe: the flagged zero half-hours
dropped and counted, rows of Primaries D15 and D18 before 2025-03-01 dropped, monthly refits,
stored predictions reused, and fitted networks saved.

## Method

- **Architecture:** the Candidate neural network of experiment 709 with its stage 1 and 2 weights
  from experiment 667; the era signal enters through 709's four states (`unknown`, `era`, `switch`,
  and `none`), where `era` turns on the era vector and anchor and `switch` subtracts its depth from
  the lags.
- **Candidates and labels:** the candidates are the detections of either of experiment 709's two
  detectors. They number about 92 on the real record and 182 on the injected bank. The language
  model is `claude-sonnet-5-5` with all tools off and no fallback to another model. Its output is
  one JSON object with `label`, `confidence`, and an onset estimate.
- **Training regime:** 709's stage 3 recipe with early stopping, 2 seeds, monthly refits, and the
  training epoch that early stopping kept, logged for every new fit.
- **Measurement:** the 4 test months of experiment 677's row set, the 64 cached passes of the
  injected bank, and phase E, which scores the months around the 2025 connection at D8 (2025-04,
  2025-05, and 2025-06). Every number is interim because experiment 667 stores seeds 0 and 1 only.
- **Register:** the ECR is one snapshot with no history, so visibility uses an assumed publication
  lag of 14 days (0 and 60 days as sensitivities), and register rows on the bank are simulated:
  half the eras get a true row and a quarter of the switching events get a decoy.

## Files

- `ml-experiments/00712-clean-era-labels-part-1/e00712_trial.py`: the cost trial. The script sends
  the probe and the 20 scheduled checks for the Primary, plus any retry, and writes the call log.
  The script produces no label that is used. The `--summary` option reads the utilisation from
  `unifiedWindows.<type>.utilization`, where the payloads hold it, as well as from
  `rate_limit_info.utilization`, and prints that the figure is account-wide.
- `data/ml-experiments/00712-clean-era-labels-part-1/`: the experiment's private folder (the call
  store, label tables, forecasts, and fitted networks). The folder is never committed.
- `data/ml-experiments/briefs/clean_era_labels/design.md`: the design behind this plan.
- [Experiment 709](../00709-era-signal-in-the-candidate-part-1/result.md): the stored signals,
  forecasts, and injected bank that the stored arms read.
