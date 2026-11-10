---
id: "00692"
slug: event-table-reviewer-calibration-gate
title: >-
  Can a Claude reviewer looking at a figure of each candidate load transfer cut the event table's
  false transfers?
theme: lsc-detectors
also_relevant_to: []
status: planned
date: 2026-11-10
kind: other
size: >-
  medium. (1) Stored: private figures, answer files, and answer key under
  data/ml-experiments/00692-event-table-reviewer-calibration-gate/ only. (2) Serving path: not
  touched. (3) Degradation rule: not touched. (4) Designs: more than one is defensible for the
  candidate set, the calibration mix, and the figure, so this plan fixes the cheapest of each before
  any answer is read. (5) Callers: the scripts import the folders of experiments 653, 660, 670, and
  680 by path; nothing imports this folder. CPU only, 2 cores, plus Sonnet reviewer calls.
hypothesis: >-
  A Sonnet reviewer shown one figure per candidate transfer of the event table (E1-J-sym) agrees
  with the injected truth well enough to remove most false transfers.
prediction: >-
  Written before any answer was read. On the calibration set the reviewer's precision for calling
  "transfer" is 0.80 or more at a recall of 0.60 or more (the spec's probability is 50%), its
  direction is right for 90% of the true transfers it calls, and it calls the two verified real
  transfers (D5 to D16 in 2025-12, and D3 and D22 in 2024-10) a transfer. On the natural mix of
  candidates the false-transfer share falls from about 0.6 before the gate to 0.45 or less after it.
  A separate reviewer asked "injected or real?" is right on 60% or fewer of 60 mixed figures.
decision_rule: >-
  Gate, fixed before any answer was read: precision 0.80 or more at recall 0.60 or more, direction
  right for 90% of the called true transfers, both verified real transfers called a transfer, and
  realness accuracy of 60% or less. If the reviewer fails the gate, drop the manual review and
  keep the automatic detector as it is (the spec's stop rule); a pass is reported with the interval
  and is not adopted into any pipeline here. The porting threshold is zero, because no production
  code changes. Data stop before 2026-07-01, and the final-test window is never read.
benchmark: >-
  The false-transfer share of E1-J-sym on the 40 scored worlds of experiment 680 (1,230 decoded
  transfer ends, 283 correct, so 0.77 of the ends).
row_set_id: >-
  experiment 680's 40 scored default worlds (seeds 900 to 939), events from 2024-01-01 to
  2026-06-30; not comparable with the leaderboard
metric: >-
  Reviewer precision and recall for "transfer" on the calibration set with Wilson intervals;
  direction accuracy; the false-transfer share before and after the gate, reweighted to the
  natural mix of candidates; tokens per review; a list of what the reviewer got wrong.
code: ml-experiments/00692-event-table-reviewer-calibration-gate/
production_status: not-for-production
adopted_in: none
---

**Plan only. The reviewer has not run.** This is check 4 of the private spec
(`data/ml-experiments/briefs/event_detector_next_spec.md`) for the next steps of the hindsight
load-shift detector. The detector's main fault is false
transfers: about three in four decoded transfer ends match no injected end. We ask whether a
Claude reviewer that looks at one figure per candidate can tell a real transfer from a false one,
by first measuring its agreement with the injected truth on a calibration set.

## Plan (choices the spec left open, all the cheapest)

- **Candidates.** The pairs of a donor end and a recipient end that E1-J-sym decodes in
  experiment 680's 40 scored default worlds. A candidate is a *true* transfer when its donor end
  matches an injected donor end (same Primary, same sign, start within 24 half-hours, overlap of
  0.5 or more) and a decoded partner is a partner of the injected transfer. The spec's scorer over
  candidate pairs (part A) does not exist, so the *score* used to find the "highest-scoring false
  pairs" is the detector's own confidence for the donor end.
- **Calibration set of 100 figures.** 30 true candidates drawn at random; 20 false candidates with
  the highest confidence; 30 false candidates drawn at random (these give the natural false-call
  rate); 20 placebo candidates decoded on a rewired graph (false by construction). Plus the two
  verified real transfers, as 2 more figures. The spec asks for 400 figures with routine-step and
  sunny-run negatives; we use 100 because the reviewer's precision needs only an interval of about
  ±0.10, and routine steps and sunny runs are left out.
- **Figure.** One image per candidate, identical for injected and real cases: no NGED labels, no
  truth marks, no dates (days relative to the onset), neutral Primary labels. Panels: observed power
  and expected power for the donor and the recipient; each residual in noise units with the decoded
  steps; the median residual of the other Primaries; the residual of the other neighbours. Three
  time ranges: an overview from 7 days before the onset to 7 days after the end (capped at 60
  days), and half-hourly zooms of ±48 hours at the onset and at the end. The spec's ERA5 and CAMS
  panels are left out.
- **Reviewer.** Sonnet only, never Opus, with one fixed brief (the nine rubric items R1 to R9 of
  the spec scored 0 to 2, a verdict among transfer with direction, solo shift, generator or
  routine, weather or common mode, PV, and unclear, a confidence from 1 to 5, JSON output). It runs
  in fresh sessions of about 10 figures each to bound the context, with the order shuffled; the
  brief is identical, so this is one reviewer. It never starts a sub-agent.
- **Realness check.** A separate Sonnet session sees 30 injected and 30 real candidate figures
  (E1-J-sym's real transfers before 2026-07-01) and answers "injected or real?". The spec asks for
  100; 60 gives an interval of about ±0.13.
- **Natural mix.** The false-transfer share after the gate is reweighted from the calibration set's
  rates to the 40 worlds' natural mix of true and false candidates, using the random-false stratum
  for the false-call rate, and is reported with a Wilson interval.
- **Model tier.** No Opus comparison (the brief forbids Opus), so the spec's step of using Sonnet
  only if it is within 0.05 of Opus is not run.
- **Cost.** CPU: 40 worlds at about 3 seconds, rendering a few minutes. Tokens: about 1,000 for each
  image plus about 3,000 for the brief and reply, so about 160 reviews cost under 1 million tokens;
  stop and tell the manager if the first batch's cost projects above 2 million.

## Predictions that could fail

| Measure | Prediction |
|---|---|
| Precision for "transfer" on the calibration set | 0.80 or more at recall 0.60 or more |
| Direction of called true transfers | right for 90% or more |
| D5 to D16 and D3/D22 | both called a transfer |
| False-transfer share, natural mix | 0.45 or less after the gate, from about 0.6 before |
| Realness accuracy | 60% or less |

**Stop rule.** If the reviewer fails the gate, the manual review is dropped and the detector's
automatic tiers stay.

## Standing practices

- **Followed:** cleaned power, the D15 and D18 cut before 2025-03-01 (inherited from experiment
  653's residual), draft pull request, code review before the run, and prose review.
- **Not applicable:** no forecast is scored, so there is no refit, no tail calibration, and no
  curtailment cost. The unlogged-event caveat applies: NGED's event labels are an incomplete dump.
