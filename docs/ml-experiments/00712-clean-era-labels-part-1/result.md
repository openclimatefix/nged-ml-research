---
id: "00712"
slug: clean-era-labels-part-1
title: >-
  Do era labels made without hindsight help the demand network after a permanent change?
theme: switching-labels-in-the-forecast
also_relevant_to: [lsc-detectors, retraining-and-training-history]
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores per-row forecasts, network weights for each arm, fold, and seed, and a
  store of language-model calls; (2) more than one design defensibly tests the hypothesis (the
  detector, the check days, the prompt, and the register rule); (3) it spans the Candidate's
  training and injection code from experiment 709 and the prevailing-head code from experiment 667,
  whose callers cannot be named without searching; (4) it touches no serving path or degradation
  rule; (5) it changes no contract. Full set of reviews: plan, code, results, write-up, prose.
hypothesis: >-
  Era labels built by a detector and a language model from data before the forecast origin, and fed
  to the Candidate with the delay an origin would have had, lower the error of the
  normal-running-arrangement (NRA) forecast in the weeks after an injected era by at least 1.5
  points of capacity, without raising the error after an injected switching event.
prediction: >-
  Written before any label or fit. All figures are CRPS differences from `none`, in points of
  capacity, on the 128 injected eras and 128 injected switching events of experiment 709's bank.
  `det`, eras, 7 to 28 days after the onset: -3.2 to -4.5. `ceiling`: within 0.3 of `det`. `llm`,
  eras: -2.0 to -3.5. `llm`, switching events at the donor Primary: -0.1 to +0.1. `det`, switching
  events at the donor: +3 to +5. `llm-ecr` minus `llm`, eras with a register row, 0 to 7 days: -0.5
  to -1.5. `ecr`, eras with a register row, 7 to 28 days: -2 to -4. `ecr`, switching events with a
  decoy register row, donor: +1 or more. `llm` NRA output minus A0-prev, eras, 7 to 28 days: -0.5
  to +1.5, sign not predicted. Every new arm on the real rows of the four test months: within
  +-0.05. At least 0.7 of detected injected eras are classed `era` within 7 days of the alarm, and
  at most 0.05 of detected injected switching events are ever classed `era`.
decision_rule: >-
  Twelve planned contrasts, so each interval is 99.58% (1 - 0.05/12), from a bootstrap that
  resamples whole bank passes and whole fold months. Supported: K1 (`llm` minus `none`, injected
  eras, 7 to 28 days) is at or below -1.5 points with an interval that excludes zero, vetoes K2 and
  K4 hold, and K3's upper end (`llm` minus `soft`) is at or below +0.5. Small gain: K1's interval
  excludes zero in the gain direction but K1 is above -1.5, and K2 and K4 hold. Partly supported:
  K1, K2, and K4 pass and K3 fails. Not supported: K1's interval contains zero. Not supported for
  adoption: K2 (the donor's error after injected switching events rises by more than 0.1 points, or
  the upper interval end exceeds +0.2) or K4 (all real rows: point estimate above +0.03, or upper
  end above the stored placebo's) fails, whatever K1 shows. Inconclusive pending an audit: the
  `none` reproduction gate fails, or the stored placebo leaves 0.06 points of `none` on the bank.
  The threshold of 1.5 points is 0.5 points for each of three components that the pipeline adds to
  production (a daily detector job, a language-model labeller with a stored-call audit trail, and a
  register ingest); the 0.5 is a stated convention, not a measurement.
benchmark: >-
  `none` (the Candidate with no era signal) and the stored 709 arms `oracle`, `placebo`, `all`,
  `soft`, `rule`, `stat`, and `hmm`; A0-prev (the prevailing output of `none`); climatology; the
  manual heuristic.
row_set_id: experiment_677_monthly_refit_rows_valid_2025_11_to_2026_06
metric: >-
  CRPS in points of capacity on injected eras, injected switching events (donor and recipient), and
  real rows; label-level confusion counts with Wilson intervals; normalised MAE of the median;
  q01, q05, q95, and q99 pinball loss and coverage; the four load-shift scenarios;
  avoidable-curtailment cost; detection and classification delay.
---
**We ask whether a forecast can use labels that say "this shift is permanent" when those labels
are made only from data available at the time.** Experiment 709 gave the demand-forecasting neural
network (the Candidate) the true dates of permanent changes, called eras, such as a solar farm
connecting. The network then forecast much better in the weeks after an era. The versions of that
signal that a real system could build also did harm after a temporary move of load between
substations (a switching event), because they treated the move as permanent. Here a detector
proposes candidate shifts, a language model reads numbers about each candidate and classes it as a
permanent change, a temporary load move, or noise, and the network trains and forecasts with the
delayed labels. This page is the plan. No label or fit exists yet, and every number below that
comes from experiment 709 or from the cost trial is interim.

**The plan runs in two stages, and the second stage runs only if the first passes its gate.** Stage
1 labels the real record of the 16 Primary substations, without the bank of injected events, then
fits the arms that need no language-model call. Stage 2 labels the injected bank and fits the
remaining arms. No labelling run starts until this plan's pull request is merged and the manager
names the go, after the maintainer says so.

## What is tested

**The hypothesis and the threshold.** Labels made without hindsight lower the error of the
normal-running-arrangement (NRA) forecast by at least 1.5 points of capacity in the 7 to 28 days
after an injected era, and do not raise the error after an injected switching event. The error is
the continuous ranked probability score (CRPS), in points of the Primary's effective capacity. The
threshold of 1.5 points is 0.5 points for each of three components the pipeline would add to
production: a daily detector job, a language-model labeller with an audit trail, and a register
ingest. The 0.5 points is a stated convention. Because dated changes are rare, 1.5 points in 28
days is about 0.005 points on all rows, so the maintainer decides whether those weeks justify the
pipeline.

**The arms.** Five arms are new and eight are read from experiment 709's store. All use 709's stage
3 recipe and 4 test months with 2 seeds.

| Arm | Class of each candidate comes from | Calls |
|---|---|---|
| `det` | none: every candidate is `era` from its alarm day | 0 |
| `llm` | the language model, reading numbers only | yes |
| `ecr` | none: the first alarm in a register watch window is `era` | 0 |
| `llm-ecr` | the language model, which also sees nearby Embedded Capacity Register (ECR) rows | yes |
| `ceiling` | the true class, revealed on the alarm day; it trains on hindsight and is never a candidate for adoption | 0 |
| `none`, `oracle`, `placebo`, `all`, `soft`, `rule`, `stat`, `hmm` | experiment 709's | stored |

**The contrasts.** Twelve contrasts are planned, so each two-sided interval is 99.58%. K1 is the
gain and K2 and K4 are vetoes.

| Contrast | Arms | Rows | Reads as |
|---|---|---|---|
| K1 | `llm` minus `none` | injected eras, 7 to 28 days | the gain; threshold -1.5 |
| K2, K2-ecr, K2-llm-ecr | `llm`, `ecr`, `llm-ecr` minus `none` | injected switching events, donor Primary | veto: above +0.1, or upper end above +0.2 |
| K3 | `llm` minus `soft` | injected eras, 7 to 28 days | gives up at most +0.5 (upper end) |
| K4 | `llm` minus `none` | all real rows of the four test months | veto: at most +0.03, and an upper end no higher than the stored placebo's |
| K5 | `llm-ecr` minus `llm` | eras with a simulated register row | what the register adds |
| K6, K6b | `ecr` minus `none`; advance-`ecr` minus `none` | the same eras | what the register gives alone; K6b is simulated |
| K7 | `llm` minus `det` | switching events, donor | the harm the language model removes |
| K8, K9 | `llm` NRA output minus A0-prev | eras; switching events at the donor | whether the labelled NRA forecast beats the prevailing forecast |

**A0-prev is the reference on which experiment 709 stopped.** It is the Candidate's prevailing
output (the NRA forecast mixed with a shifted copy by a survival probability), reused from
experiment 667's heads. The block needs no call and no GPU, and may be deferred by the manager. If
it is deferred, a `supported` verdict means only "beats the network with no era signal".

## Gates and vetoes

**Stage 1 ends at a gate on the detectors, before any bank call.** The gate reads `det` minus
`none` on injected eras at 7 to 28 days, with `ceiling` beside it. If `det` is above -1.5 points,
the two detectors alarm too late or too rarely for any classifier to pass K1, and the experiment
stops as `not supported` after the stage 1 labelling. Stage 2 starts with a pilot of 2 passes per
fold month (about 60 calls) and stops if the model classes 3 or more detected pilot switching
events `era` at confidence 0.8 or more, or classes fewer than 40% of detected pilot eras `era`
within 7 days of the alarm.

**Two reproduction gates stop the run before any score is read.** The new script must reproduce
experiment 709's `none` forecast to within 1e-5 on the median for one fold and seed. A fixed probe
of 10 calls asks the model to name the year and month from real prompts, and the prompt is pinned
only if at most 3 years are right.

## The label pipeline

**Each label is built from data up to the end of the day before the origin.** A candidate is one
detected shift. The model sees numbers about one unnamed substation, as shares of capacity and as
offsets from the check day, with no name, date, weekday flag, or megawatt value. Checks run on the
alarm day, 7 days later, and 21 days later, and on the day a closure becomes known. A label from
check day d reaches the network at the origin of day d + 1, because the network forecasts at 09:00
UTC. A perturbation test changes every reading from 00:00 of day D and asserts that the SHA-256 of
every prompt table for a check before day D is unchanged. Labels are made once and replayed from
the store, because the model's sampling cannot be fixed.

**Example: the 2025 solar connection at Primary D8 (register date 2025-03-24, 8.91 MW).** The
stored `stat` arm of experiment 709 raised an alarm 33 days after the register date, and then
withdrew the class when the level fell back:

```text
entry day    stat state  era_id  onset_est    first origin reading it
2025-04-25   era         4       2024-12-21   2025-04-26 09:00 UTC
2025-04-26   era         6       2025-03-29   2025-04-27 09:00 UTC
2025-06-08   switch      4       2025-03-29   2025-06-09 09:00 UTC  (withdrawn=1)
2025-06-09   era         4       2024-12-21   2025-06-10 09:00 UTC
```

In the new pipeline the language model, not the detector, decides whether an era ends: the state
changes only if a closure check answers `switch` or `noise` at confidence 0.8 or more.

## Cost, billing, and staging

**Labelling bills to the maintainer's shared plan quota, so the dollar figures are prices, not
charges.** The cost trial's init event showed `apiKeySource = none`, which means the calls ran on
the subscription login. Every US$ figure below is the price the command-line tool reports, which is
the API-equivalent price. The share of the plan quota that the calls use cannot be observed from
the command line, so the plan counts calls and tokens and treats the quota as the limit.

**The cost trial on a single Primary measured cost and nothing else.** It sent 22 calls (one probe
and 21 checks on 5 candidates) with 167,708 input tokens and 10,071 output tokens in 198.9 s, at a
reported US$0.743. That is 4.2 calls per candidate with no early stop, a mean of US$0.0353 per
call, and a median of 7,800 input tokens. The labels from the trial were never read and are not
results.

| Stage | Block | Calls | Input tokens | Reported price |
|---|---|---|---|---|
| 1 | Real record, 16 Primaries, no bank (probes, development, and `llm-ecr` rows included) | about 245 to 386 | 2 to 3 million | about US$9 to 14 |
| 2 | Injected bank, arms `llm` and `llm-ecr` | at most 364 | about 2.9 million | at most about US$13 |

**Staging and limits.** Stage 2 runs only if stage 1 passes its gate. At most 2 calls run in
parallel, and the run stops on any rate-limit event and reports to the manager. The labelling
script prints running totals of calls, tokens, and reported price every 25 calls. `BudgetGuard`
takes explicit call, token, and reported-price limits for each stage and never `with_round`.

**Scaling to 2,500 substations is an argument, not work to run.** At 0.8 candidates per
substation-year and about 4.2 calls each, 2,500 substations would need about 480 million input
tokens once to label the history and about 67 million a year after that. This experiment does not
validate those figures.

## Compute

**The fits run on the manager's GPU slot, and none starts until the manager names a share.** A
fit takes about 3.5 minutes and peaks at 1.45 GiB of GPU memory and 6.3 GiB of host memory. The
main block is 71 fits, about 8 GPU job-hours, or about 4 hours at 2 jobs. The A0-prev block runs
on at most 4 CPU cores for about 4 to 7 job-hours. The experiment follows the standing practices
for cleaned data, monthly refits, the D15 and D18 cut, and saved networks.

## Method

- **Architecture:** the Candidate network of experiment 709 with its stage 1 and 2 weights from
  experiment 667; the era signal enters through 709's four states (`unknown`, `era`, `switch`, and
  `none`), where `era` turns on the era vector and anchor and `switch` subtracts its depth from the
  lags.
- **Candidates and labels:** the union of 709's change-point and year-on-year detectors, about 92
  candidates on the real record and 182 on the bank; the model is `claude-sonnet-5-5` with all
  tools off and no fallback; the output is one JSON object with `label`, `confidence`, and an
  onset estimate.
- **Training regime:** 709's stage 3 recipe with early stopping, 2 seeds, monthly refits, and the
  kept epoch logged for every new fit.
- **Measurement:** the 4 test months of experiment 677's row set, 709's 64 cached bank worlds,
  and phase E at D8 on the months around the 2025 connection; every number is interim because
  experiment 667 stores seeds 0 and 1 only.
- **Register:** the ECR is one snapshot with no history, so visibility uses an assumed publication
  lag of 14 days (0 and 60 days as sensitivities), and register rows on the bank are simulated:
  half the eras get a true row and a quarter of the switching events get a decoy.

## Files

- `ml-experiments/00712-clean-era-labels-part-1/e00712_trial.py`: the cost trial. It sends the
  probe and the 21 checks for the Primary and writes the call log. It produces no label that is
  used.
- `data/ml-experiments/00712-clean-era-labels-part-1/`: the experiment's private folder (the call
  store, label tables, forecasts, and fitted networks). It is never committed.
- `data/ml-experiments/briefs/clean_era_labels/design.md`: the design behind this plan.
- [Experiment 709](../00709-era-signal-in-the-candidate-part-1/result.md): the stored signals,
  forecasts, and bank that the stored arms read.
