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
The only labelling run so far is an interim pilot on one Primary (see "D8 pilot of the labelling (interim)"
below), and no neural network has been trained for this experiment. The figures
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
the application programming interface (API) equivalent price. The rate-limit events do carry an
account-wide seven-day utilisation (`unifiedWindows.seven_day.utilization` = 0.32, constant across
all 22 calls, shared by every session), so the trial's own share is below the resolution of that
field and cannot be attributed. The labelling script counts calls and tokens, and the plan treats
the quota as the limit.

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

## D8 pilot of the labelling (interim)

**A pilot on the 19 candidates of Primary D8 ran the labelling script before the 16-Primary run.**
Every figure in this section is interim. The pilot is not the write-up and sets no verdict. D8 is a
substation, so its Embedded Capacity Register (ECR) dates are public facts. The pilot used the
prompt as pinned after the year probe. No cell below holds more than 3 candidates, so we give no
intervals.

**The year probe passed.** The year probe sent the language model 10 prompts, built from 5
candidates of Primary D10, and asked it for the calendar year of each. The 10 replies are not
independent, because several prompts come from the same candidate. The language model named the
exact year for 2 of the 10 prompts, and a year within 1 year of the true year for 3 of the 10
(including the 2 exact years). The pin rule under "Gates before any score is read" allows at most 3
exact years of 10, so the prompt passes. The year probe made 11 calls: the 10 prompts and 1 billing
probe, which shows which key source bills the calls. The calls used 61,140 input tokens and 8,697
output tokens, at a reported US$0.33 (price as reported by the command-line tool, the
application-programming-interface (API) equivalent).

**The labelling made 87 calls on D8, of which 65 were distinct checks.** A check is one request to
label one candidate on one check day. The 19 candidates took 65 checks. A strict reply parser
rejected the first reply to 20 of the checks, and the script retried each of those once, so the 65
checks took 85 label calls. The other 2 calls were billing probes. The first run stopped at its
80-call cap, and a second run resumed it. Together the 87 calls used 551,253 input tokens and 44,666
output tokens, at a reported US$2.179 (API-equivalent). The summed call time was 830 s. The elapsed
time was about 440 s, with 2 calls in parallel. The init event showed `apiKeySource = none`, so the
calls ran on the subscription login. No call was limited by a rate-limit event. The account-wide
seven-day utilisation was 0.36 before and after the run, so the pilot's own share of the quota
cannot be attributed.

**The strict reply parser left 10 of 65 checks without a valid reply, and a looser parser accepts
all 65 on the first reply.** Of the 85 label replies, 55 were valid under the strict parser and 30
were invalid. Of the 65 checks, 45 had a valid first reply, 10 more had a valid retry, and 10 never
had a valid reply. Of the 30 invalid replies, 16 lacked the `register_row` key (the field that
names the ECR row the language model cites, if any) and 14 had a `reason` (the language model's
free-text justification) longer than 60 words. The looser parser makes `register_row` optional and
cuts a longer `reason` to 60 words. The prompt is unchanged. We re-parsed the stored first replies
offline with the `--reparse` option, with no new call, and all 65 checks are valid on the first
reply. The 20 retries would not be needed. The looser parser is the parser to use for any later run,
and the prompt pin stays.

Attempt 2, which changed the reply format only, is in the next section.

**Under the rule that a label counts only at confidence 0.8 or more, the labeller recovered none of
D8's known eras or switching events.** The classes recorded for D8 come from its dated ECR
connections and from switching events verified in the load. In the table, a state is one of the four
states of the era signal (`unknown`, `era`, `switch`, and `none`), and `unknown` means no label has
reached confidence 0.8. Under both parsers, all three known eras and both known switching events
stayed `unknown`.

| Known class | Candidates | Recovered | Notes |
|---|---|---|---|
| Era (dated ECR connection) | 3 | 0 | all three stayed `unknown` |
| Verified switching event | 2 | 0 | neither candidate was labelled `switch` |
| Unresolved | 1 | not applicable | the recorded classes do not settle this candidate |
| No known class | 13 | not applicable | not scored |

The 3, 2, 1, and 13 candidates sum to the 19 candidates. One of the 13 candidates with no known
class was labelled `era` at the 21-day check, at confidence exactly 0.8, and only under the looser
parser. Under the strict parser, no candidate reached state `era`.

**The detectors alarmed late on two of the three known eras.** The alarms came 73 days and 33 days
after the ECR dates, and 10 days and 21 days after the dates visible in the telemetry. No candidate
with a known class reached state `era`, so the delay from alarm to a first `era` label cannot be
measured.

**The labeller is below the plan's pilot rule for this setting, so continuing to the 16 Primaries is
the maintainer's choice.** The plan states no gate for moving from D8 to the other 15 Primaries. The
nearest rule is the stage 2 pilot rule, which stops the run when fewer than 40% of detected eras
are labelled `era` within 7 days of the alarm. That rule is written for the injected bank, so it
does not bind this pilot, but 0 of 3 known eras is far below 40%. A change to the prompt would need
a new pin and a new year probe. The change must not be tuned on D8's known classes, because tuning
on them would leak them into the prompt.

**Scaling D8's counts to the 16 Primaries gives a rough cost, on two bases.** D8 holds 19 of the 92
candidates, and no other Primary holds more than 13. Counting every call the pilot made, including
the 20 retries, the 16 Primaries would need about 412 calls (85 / 19 x 92), 2.7 million input
tokens, a reported US$10.6 (API-equivalent), and about 35 minutes at 2 parallel calls. Counting
only first replies under the looser parser, the 16 Primaries would need about 315 calls (65 / 19 x
92). Tokens, price, and time on that basis are not measured, and scale to roughly 20 to 25% below
the first basis. Both bases exceed the 245 to 386 calls for arm `llm` under "Cost, billing, and
staging" because D8 needed 4.5 calls per candidate on the first basis and 3.4 on the second, against
the cost trial's 4.2. Both figures are interim.

## D8 attempt 2 of the labelling (interim)

**The maintainer approved one more attempt on D8 alone.** The labeller again recovered none of D8's
known eras or switching events. Every figure in this section is interim. The attempt is not the
write-up and sets no verdict. No run was made on the 16 Primaries, and no label was injected into the
bank. The attempt 1 numbers in the section above are unchanged, and attempt 1 is closed to spending.
In this section, a known class counts as recovered when a check labels the candidate with that class
at confidence 0.8 or more, as in the pilot section. A parser is the code that reads a reply and
checks its format. The strict parser is attempt 1's original parser, and the looser parser is the
parser described in the pilot section, which makes `register_row` optional and cuts a longer `reason`
to 60 words.

**Attempt 2 changed the reply format only.** The worker who designed the change says it was decided
without reading D8's label content or its agreement with the known classes. The page holds no
evidence for that claim, and the null result of attempt 1 on D8 was known when the change was
designed. The attempt 2 prompt tells the language model to reply with raw JSON and nothing else. The
`llm` arm has no `register_row` key. The `reason` (the language model's free-text justification) is
limited to 40 words. When a reply fails to parse, the script retries the check once and counts the
retry as a call. The prompt has a new hash and a new pin file. The hash is a fingerprint of the prompt
text, the schema text, and the prompt-builder code, and the pin file records it, so a changed prompt
cannot reuse an earlier approval. A go token is a single-use approval string that the manager writes
before each paid step, and the script now refuses a token it has used. The scoring rule is the
pre-registered rule, unchanged.

**A code review of the changed scripts found 3 blockers and 4 should-fix items, and the fixes are
recorded in `review_triage.md`.** The review was a fresh review by a larger language model (Opus),
and the findings were verified before they were applied. A blocker is a finding that must be fixed
before any call, and a should-fix item is a finding that is worth fixing but does not stop a run. The
blockers were that the attempt 2 run could overwrite attempt 1's signal files, that a go token was
not truly single-use, and that nothing stopped the run from continuing past D8 once the reported
price reached about US$4. The should-fix items included two findings: the probe scorer could record
the wrong prompt hash, and the script did not count reply formats (for example, how many replies were
raw JSON) without reading the labels.

**The year probe passed again.** The year probe tests whether the language model can recover the
calendar year from the prompt. A pass means it cannot do so beyond chance: the pin rule allows at most
3 exact years of 10. The probe used the same 10 prompts as attempt 1's year probe, rebuilt with the
attempt 2 prompt builder. The language model named the exact year
for 2 of the 10 prompts, and a year within 1 year for 4 of the 10 (including the 2 exact years). The
10 answers are not independent, because they come from 5 candidates. The probe made 11 calls (10
prompts and 1 billing probe) and used 61,384 input tokens and 10,432 output tokens, at a reported
US$0.350 (the price the command-line tool reports, which is the price the same calls would cost
through the pay-per-token application-programming-interface, API).

**The D8 run made 68 calls: 65 checks, 1 billing probe, and 2 retries after unterminated JSON.** The
19 candidates took the same 65 checks as attempt 1. The calls used 438,551 input tokens (a mean of
6,449 per call) and 32,732 output tokens, at a reported US$2.033 (API-equivalent). The summed call
time was 584 s, and the elapsed time was 301 s with 2 calls in parallel. The init event showed
`apiKeySource = none`, so the calls ran on the subscription login. No call was limited by a
rate-limit event. The account-wide seven-day utilisation was 0.39 before and after the run, so the
run's own share cannot be attributed.

**Under the looser parser, all 65 checks reached a valid reply, and the format change did not
measurably improve on that.** Of the 65 checks, 63 were valid at the first reply, 2 were valid after
the retry, and none lacked a valid reply. The looser parser alone had already accepted all 65 of
attempt 1's first replies. Attempt 1 and attempt 2 are separate sets of calls, not the same replies.
Attempt 1's strict parser accepted 55 of its 65 checks: 45 at the first reply, 10 only after a retry,
and 10 never. Read by the structure of the 67 label replies of attempt 2 (65 first replies and 2
retries), the 40-word limit was ignored in 30 replies, and 1 reply still carried `register_row`. The
looser parser cuts a longer `reason` to 60 words, and the attempt 2 prompt asks for 40.

**Under the 0.8 rule, the labeller again recovered none of D8's known eras or switching events.** The
table states each check as `unknown` at the alarm, 7 days after it (+7 d), 21 days after it (+21 d),
and at closure (the last check of the candidate). The +21 d check exists for only 1 of the 2
switching-event candidates. The cells for the known eras, the verified switching events, and the
unresolved candidate each hold at most 3 candidates, so we give no intervals.

| Known class | Candidates | Recovered | Notes |
|---|---|---|---|
| Era (dated ECR connection) | 3 | 0 | `unknown` at every check |
| Verified switching event | 2 | 0 | `unknown` at every check |
| Unresolved | 1 | not applicable | the recorded classes do not settle this candidate |
| No known class | 13 | not applicable | not scored |

The 3, 2, 1, and 13 candidates sum to the 19 candidates. Every check of every candidate was `unknown`
or `none`, so no candidate of any class reached state `era`, and the delay from alarm to a first
`era` label cannot be measured. The detectors' own delays are the same as in attempt 1: the register
date to the alarm was 73 days and 33 days, and the telemetry date to the alarm was 10 days and 21
days.

**After two attempts, the labeller at either prompt and the 0.8 rule recovers none of D8's known eras
or switching events.** The reply-format change did not change the null. Whether to continue, to stop,
or to try a different design is the maintainer's choice. A different prompt would need a new pin and
a new year probe, and must not be tuned on D8's known classes.

**Scaling attempt 2's D8 counts to the 16 Primaries gives about 329 calls, US$9.8, 2.1 million input
tokens, and about 24 minutes at 2 parallel calls.** D8 holds 19 of the 92 candidates, and the
estimate scales attempt 2's run by 92 / 19 (68 / 19 x 92 is about 329 calls). The estimate is
interim, and no 16-Primary run was made.

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
  the billing probe and the 20 scheduled checks for the one Primary tried, plus any retry, and
  writes the call log. The script produces no label that is used. The `--summary` option reads the
  seven-day utilisation of the shared quota from `unifiedWindows.<type>.utilization` and from
  `rate_limit_info.utilization`. The option also prints that the figure is account-wide.
- `ml-experiments/00712-clean-era-labels-part-1/e00712_prompt.py`: lists the candidates and the
  checks and builds the prompt. The prompt carries no calendar key (a field that reveals the date;
  see `calendar_keys.md`), scales capacity using readings up to the check day only, and includes the
  register table for arm `llm-ecr`. The module makes no call.
- `ml-experiments/00712-clean-era-labels-part-1/e00712_asof.py`: builds the check schedule and, for
  every arm, a table of the label known on each day (the as-of table), in the format of experiment
  709's signals. The module makes no call.
- `ml-experiments/00712-clean-era-labels-part-1/e00712_label.py`: the stage 1 labelling script.
  The script runs as a dry run unless told otherwise. A call needs the manager's go token, a billing
  probe that confirms which credential pays for the calls, the prompt pin, and explicit limits on
  calls, tokens, and reported price. The script also builds the arms that need no call. The
  `--reparse` option re-parses the stored replies with the looser parser (see the D8 pilot section)
  and makes no call. The `--agreement` option compares the call log with the known classes of the
  candidates and makes no call. The script ran the labelling of attempt 1 and of attempt 2.
- `ml-experiments/00712-clean-era-labels-part-1/e00712_prompt_v2.py`: the attempt 2 prompt builder.
  The module replaces the reply-format instructions of `e00712_prompt.py` with an instruction to
  reply with raw JSON and nothing else, no `register_row` key for arm `llm`, and a 40-word `reason`.
  The module reuses the rest of `e00712_prompt.py` and makes no call. The attempt 2 outputs sit beside
  the attempt 1 outputs in the private folder:
    - `stage1_v2/`: the call log and stored replies of the attempt 2 D8 run.
    - `signals_v2/`: the label signals the run built.
    - `prompt_pin_v2.json`: the hash of the approved attempt 2 prompt.
    - `probe/prompts_v2/` and `probe/probe_result_v2.json`: the year probe's prompts and result.
    - `stage1_go_token_v2.txt`: the go token, which a run renames to `<name>.used.<timestamp>` when
      the run claims it.
- `ml-experiments/00712-clean-era-labels-part-1/e00712_probe.py`: the year-guessing probe, whose
  result decides whether the prompt is pinned. A probe reply is never used as a label.
- `ml-experiments/00712-clean-era-labels-part-1/e00712_score.py`: the scorer for the planned
  contrasts, vetoes, gates, and checks. It has been self-tested only, and no score has been read.
- `ml-experiments/00712-clean-era-labels-part-1/test_e00712_label.py` and `test_e00712_probe.py`:
  tests of the labelling script and the probe. No test makes a call, because each test replaces
  the command-line tool that makes the calls with a stand-in.
- `ml-experiments/00712-clean-era-labels-part-1/calendar_keys.md`: the calendar keys the pinned
  prompt still carries and why the plan accepts them.
- `ml-experiments/00712-clean-era-labels-part-1/review_triage.md`: what each review of the plan
  found and what the plan did with each finding.
- `data/ml-experiments/00712-clean-era-labels-part-1/`: the experiment's private folder (the call
  store, label tables, forecasts, and fitted networks). The folder is never committed.
- `data/ml-experiments/briefs/clean_era_labels/design.md`: the private design that this plan
  summarises. The file is not committed, so a reader of the published docs cannot open it; the
  committed `review_triage.md` records what the reviews of the design changed.
- [Experiment 709](../00709-era-signal-in-the-candidate-part-1/result.md): the stored signals,
  forecasts, and injected bank that the stored arms read.
- Experiment 712 reads the era table of experiment 709 and takes the Primary under which the
  register lists the gas engine from the `source` text of era 1. The era table's format is
  unchanged, although experiment 709 now reads that Primary from the private file
  `data/ml-experiments/private/engine_constants.json`, so a rerun of experiment 709 does not change
  the output of experiment 712.
