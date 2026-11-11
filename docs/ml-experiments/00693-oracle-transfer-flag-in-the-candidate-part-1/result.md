---
id: "00693"
slug: oracle-transfer-flag-in-the-candidate-part-1
title: >-
  Would a perfect, causal flag for load transfers improve the Candidate demand forecast at the 16
  Primaries?
theme: switching-labels-in-the-forecast
also_relevant_to:
  - lsc-detectors
  - lagged-power-and-error-correction
status: planned
kind: other
size: >-
  Anything else. (1) It stores fitted networks and prediction files per arm, seed, and fold; (2)
  more than one design would defensibly test the hypothesis (flag in the lags only, in the
  prevailing head only, or both), so the arms are fixed in the plan; (3) it imports code from
  experiments 613, 647, 650, 660, 667, 677, and 681; (4) it touches no serving path or degradation
  rule; (5) it changes no contract.
hypothesis: >-
  A perfect telemetry detector, which knows that a reading lay inside a load transfer (and when the
  transfer began, once a reading after the onset exists), lowers the CRPS of the Candidate demand
  network on transfer-affected rows by at least 0.3 points of capacity against a placebo flag,
  without raising CRPS on event-free rows by more than 0.05.
prediction: >-
  Written before any score. No-retrain-arm (NRA) output: the gain over the placebo is under 0.3
  points, because the Candidate already ignores most of a shifted lag. Prevailing output: the gain
  is 0.3 to 1.0 points inside a transfer, because the R-haz head otherwise infers the onset from
  residuals and is held at p = 0 until its heuristic age passes 0.5 days. Real-event slices hold
  about two verified episodes before July 2026 and are descriptive.
decision_rule: >-
  Smallest gain worth porting: 0.3 points of CRPS on affected rows, because a detector adds a
  pipeline step, a model to maintain, and a false-alarm failure mode (the best real detector of
  experiment 208 recovered 53% of the perfect-knowledge gain). All bars apply to the point estimate,
  and the Bonferroni interval (five tests: C1, C2, C3, C4-NRA, C4-prevailing) must exclude zero in
  the stated direction. Supported - C1, C2, or C3 gains at least 0.3 points; both C4 point estimates
  rise by no more than 0.05; and C5 (placebo equivalence, A2 minus A0, 90% interval inside plus or
  minus 0.05) passes. Small gain - a C1 to C3 gain whose interval excludes zero but whose point
  estimate is under 0.3, with C4 and C5 as above. Not supported - no C1 to C3 interval excludes zero
  in the predicted direction. Veto - C4 above 0.05, or C5 failing, makes the verdict inconclusive
  pending an audit. A null is read as "no detector is worth building" only if the ceilings are
  computed (A0-clean on bank L, oracle B-haz on bank F) and the oracle age is shown to lift the age
  floor (share of P1 rows with p > 0, per arm). Stage 0 gate - if CRPS(A0) minus CRPS(A0-clean) on
  bank L's affected rows is under 0.3 points, the NRA retraining is not run.
benchmark: >-
  Experiment 677's quarterly step-8 Candidate (arm A0, near 7.99 MAE and 4.57 CRPS), experiment 667's
  monthly file (7.53 MAE) for scale, ClimatologyForecaster, and NGED's manual heuristic.
row_set_id: oracle693_c1_bankL_nov25_jun26
metric: >-
  CRPS (twice the mean pinball loss over 11 quantiles) in points of capacity, normalised MAE of the
  median, q05-to-q95 coverage, q01, q05, q95, and q99 coverage and pinball loss, cold winter nights,
  avoidable-curtailment cost, copied share of an injected lag-only shift, reaction time.
---

**Status: planned. The design is fixed in the brief at
`data/ml-experiments/briefs/oracle_switching_flag_design.md` (design v2, reviewed by an adversarial
Opus reviewer and revised). No code has run.**

We ask whether telling the demand forecast neural network exactly which of its past power readings
fell inside a load transfer would lower its error at 16 primary substations. We give one copy of the
network the true flags and another copy flags at random times with the same lengths, then compare
their errors on injected transfers and on real rows. The answer bounds what a real switching
detector could add to the forecast.

## Plan

- **Arms.** A0 (default: [experiment 677](../00677-building-up-the-candidate-demand-network-one-component-at-a-time/result.md)'s
  step-8 Candidate network and the adopted lag-input prevailing head of [experiment
  681](../00681-extra-lagged-power-inputs-for-the-prevailing-forecast/result.md)), A1 (default plus
  oracle flags), A2 (placebo flags with matched span lengths, overlaps, and per-Primary counts).
  Stage 0 adds the inference-only ceilings A0-mask, A0-clean, and oracle B-haz (the no-fit hazard
  baseline of
  [experiment 667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md)).
- **Folds.** Quarterly refits at the maintainer's request (cut-offs 2025-11-01, 2026-02-01,
  2026-05-01), scored on valid times 2025-11-01 to 2026-06-30 only. Seeds 0 to 2.
- **Information sets.** The Candidate lag flags use 09:00 UTC origins, and every lag lies at least
  24 hours before the origin. The prevailing head uses the 12:00 UTC contract origin, and its freshest
  reading is 6 hours before the network origin. `in_progress` is the transfer state at that freshest
  reading, never at the origin: it is 1 only if the onset is at or before it and no end is at or
  before it. The oracle age is the origin minus the earliest onset in progress, and is 0 when
  `in_progress` is 0. The flags cover the row's own Primary for the prevailing head, and the own
  Primary and its electrical neighbours for the Candidate. The placebo flags obey the same rules. Both
  rules are asserted per row in code.
- **Banks (pre-registered).** Deciding banks L and F use the experiment 660 generator with
  `hard_negatives=False`, class probabilities edge_transfer 0.55, edge_transfer_unseen_donor 0.10,
  edge_transfer_unseen_recipient 0.15, and permanent_shift 0.20, `events_per_primary_year=3`, and 48
  passes, injected into the cleaned panel of 15 Primaries (the 16th has no noise unit) from 60 days
  before the fold's first scored month to the end of its scored months. A pass's seed is 693000
  plus 100 times (calendar year times 12 plus fold month) plus the pass number, so all arms and
  network seeds see the same injected worlds. A smoke pass at 6 events per Primary-year left 99.7% of
  phase P1 rows with a neighbour flag, so the rate was lowered before any quoted run. If a scored
  month still holds under about 200 events, the pass count is raised in a commit before the first fit. Check banks: experiment 581's generator with seeds disjoint from
  training (bank L), and experiment 650's held-out passes (bank F).
- **Contrasts** (A1 minus A2 unless stated; negative is better). C1: no-retrain-arm (NRA) output,
  bank L, phases P1 plus P2, November 2025 to June 2026. C2: prevailing output, bank F, phase P1,
  February to June 2026. C3: prevailing output, bank F, phase P2, February to June 2026. C4-NRA and
  C4-prevailing: A1 minus A0 on real rows with an event-free origin and target (phase P4), verified
  transfers deciding. C5 (outside the family): A2 minus A0 on banks L and F and on real P4 rows.
  Phases: P1 = a lag (for the prevailing head, the origin) is inside a transfer and the target is
  inside it; P2 = same, but the target is after the end; P3 = lags clean, target after the end; P4 =
  event-free origin and target. Each contrast has its own row set: `oracle693_c1_bankL_nov25_jun26`,
  `oracle693_c2_c3_bankF_feb_jun26`, and `oracle693_c4_real_p4`. Experiment 677's file has another
  `row_set_id`, so A0 is re-scored on these rows (valid times to 2026-06-30) and checked against 677's
  stored scores, rather than read as 677's file.
- **Bars apply to the unweighted deciding-bank contrast.** The reweighting to the verified transfers'
  depth and duration and the implied all-rows annual effect are reported beside it. Each contrast is
  also reported per seed.
- **Stage 0 gate.** Seed-mean CRPS(A0) minus CRPS(A0-clean) on bank L, phases P1 plus P2, all three
  folds pooled. If under 0.3 points, the NRA retraining is not run, C1 and C4-NRA are not computed,
  the Bonferroni family is C2, C3, and C4-prevailing (three tests), and "supported" needs C2 or C3
  at 0.3, C4-prevailing within 0.05, and C5.
- **Screen advance rule.** Run stage 0, then fold 2 for A0, A1, and A2 (all numbers interim).
  Continue to the other folds only if A1's copied share on bank L is at most half of A0's, the share
  of P1 rows with p above 0 is higher in A1 than in A0, and scoring A1 with its flags permuted
  changes its predictions. Otherwise stop and debug the wiring.
- **Reading a null.** A `not supported` or `small gain` verdict supports "switching labels belong in
  evaluation, not in the forecast" only if A0-clean (bank L) and oracle B-haz (bank F) have been
  computed and the share of P1 rows with p above 0 is higher in A1 than in A0. Otherwise the verdict
  stays as scored and the consequence is withheld. A `supported` verdict on C2 or C3 alone means a
  detector is worth testing as an input to the prevailing head, with a degradation sweep next.
- **R-haz fit boundary.** Every R-haz training row has its target valid time before the fold's
  cut-off, and its labels (end, verification) known before the cut-off. Spans of verified real
  transfers are truncated at the cut-off. Both are asserted per fold.
- **Standing practices followed:** cleaned data (count of `substation_zero` half-hours printed per
  fold), D15 and D18 cut before 2025-03-01, every network saved, stored predictions reused,
  origin-time weather only (asserted per row), no new calendar inputs (the Candidate's own date-time
  encoder is unchanged), and normalised MAE of the median in every table. The four load-shift
  scenarios (all rows; outside events and at least 7 days after; inside events; the 7 days after) are
  reported, with a day-ahead forecast, a 14-day forecast, and at least 6 load-shift time series across
  both metering groups. **Departures:** quarterly instead of monthly refits (the maintainer's
  request; experiment 677's advice favours monthly, and any gain here applies to quarterly refits),
  and a one-fold screen instead of four seasonal months.
- **Earlier advice and background.** `docs/ml-experiments/background/index.md`, `data-processing.md`,
  and the `advice` lists of experiments 667, 677, 681, 687, and the switching-detector experiments were
  read. The plan follows experiment 677's advice to judge on copied share and event-free error
  together, and treats a seed contrast under 0.03 points as unresolved without five seeds
  ([experiment 651](../00651-graph-layer-over-electrical-neighbours-in-the-demand-network-part-1/result.md));
  it departs from the advice to refit monthly.
- **Caveats.** NGED's logs miss transfers, so event-free real rows may hold transfers no label covers
  and the C4 harm figure is optimistic. C4 compares A1 with A0 on rows the oracle never flags, so it
  measures the extra input columns and refit noise. C4's real rows are four or five month blocks, so
  its interval is coarse.

## Method

To be written with the results.

## Files

To be written with the results.
