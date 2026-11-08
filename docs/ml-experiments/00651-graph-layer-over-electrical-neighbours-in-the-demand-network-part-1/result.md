---
id: "00651"
slug: graph-layer-over-electrical-neighbours-in-the-demand-network-part-1
title: >-
  Does a graph layer over the electrical neighbours let the class 2 candidate keep its low follow of a
  load shift while removing its extra error on ordinary days at the six MW Primaries?
theme: lagged-power-and-error-correction
also_relevant_to:
  - hierarchy-and-pooling
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. Answers to the five triggers: (1) it stores fitted network weights and forecasts for 6 arms
  in the private data folder only; (2) more than one design could test the question, so the plan
  follows the maintainer-approved design (Opus, 2026-11-08) section by section; (3) it spans code whose
  callers cannot be named without searching (experiments 581, 592, 606, 610, 612, 613); (4) it touches
  no serving path and no degradation rule; (5) it changes no contract. Ceiling 5 GPU-hours (process
  hours); stop and ask before crossing.
hypothesis: >-
  A graph layer between the existing encoders (weather, date-time, lag) and the mixer of the class 2
  candidate, on the electrical neighbour graph (29 edges among 16 Primaries), learns when to listen to
  a neighbour. It keeps the candidate's follow of an injected load shift and removes the candidate's
  extra event-free error at the six MW Primaries, and the electrical graph does this better than a
  degree-preserving rewired graph.
prediction: >-
  Written before any score. P1: GAT-E's follow on bank M is within 0.01 of the candidate's (about
  0.040). P2: GAT-E's event-free CRPS at the six MW Primaries is within the minimum detectable effect
  (0.044 points of capacity) of G0, that is, the candidate's extra +0.059 is removed. P3: GAT-R follows at
  least 0.01 more than GAT-E on banks M and S. P4: GCN-E and GAT-E differ by less than the minimum
  detectable effect on every counted measure. P5: GAT-F is within the minimum detectable effect of G0 on
  event-free CRPS. P6: FLOW-E's follow on bank X is no worse than the candidate's by more than 0.01. The
  author expects P1, P4, and P5 to hold and P2 and P3 to fail.
decision_rule: >-
  The gate to a confirmation experiment (G2) is checked by script. G2 starts only if some arm X in
  {GCN-E, GAT-E, FLOW-E} meets all three: (i) follow on bank M at most the candidate's + 0.01, and on
  bank S at most the candidate's; (ii) event-free CRPS at the six MW Primaries below the candidate's by at
  least 0.044, or follow on bank S below the candidate's by at least 0.03; (iii) X beats GAT-R on the
  measure that passed (ii) by at least half the minimum detectable effect. If no arm passes, G1 is
  written up as the answer and neither G2 nor G3 runs. The experiment adopts nothing.
benchmark: >-
  The class 2 candidate (`P7_D50_e3`) of experiments 612 and 613, stored seeds 0 and 1, and G0 (own gap
  signals only) refitted at the same seeds, folds, and band width.
row_set_id: experiment_470_intersection_v1
metric: >-
  Follow and q05-q95 coverage of the normal running arrangement on three injected banks (day 4 to the
  end, 0 to 7 days after); event-free CRPS at 24-120 h and 120-240 h by metering group; the tail table
  (q01, q05, q95, q99 coverage and pinball loss).
---

# Plan: a graph layer over the electrical neighbours (G1)

Status `planned`: no fit has run. The development months are 2025-10 and 2025-12 only for G1. The
final-test window (2026-07-01 onwards) is never read. The target is the normal running arrangement
(NRA), so no arm may follow a transfer.

## Design

- **Source.** The maintainer-approved design (Opus, 2026-11-08) and its docs research. This plan
  follows it section by section.
- **Edges.** `neighbours_nged.json` of experiment 383: 29 undirected edges among the 16 Primaries,
  built from NGED's interconnection file. No edge comes from a switching log, and the D2-D12 pair is
  not added.
- **Network.** The class 2 network of experiments 612 and 613. One graph layer sits between the
  encoders and the mixer: `h_i + U * aggregate_j(message_ij)`, with `U` initialised to zero, so every
  graph arm starts as the own-signal network. Messages are hidden on 50% of training rows (D50).
  Pretraining is unchanged. The recipe is E3, D5, and the injected bank of experiment 581.
- **Cadence.** Monthly folds (2025-10, 2025-12), refits on days 0 and 14, seeds 0 and 1, as the
  design says. The cleaning and the D15 and D18 cut before 2025-03-01 are those of experiment 613.

## Arms

| Arm | Neighbour block |
|---|---|
| G0 | None: own gap signals only (control) |
| C | The stored class 2 candidate, seeds 0 and 1 of experiment 612 (reused, not refitted) |
| GCN-E | Degree-normalised mean of neighbours' lag-encoder states, one linear layer |
| GAT-E | Attention over neighbours' lag-encoder states and gap signals, 2 heads, sigmoid gate on the sum |
| GAT-R | GAT-E on a degree-preserving rewired graph (one rewiring per seed): the placebo |
| GAT-F | GAT-E over all 15 other Primaries, no topology |
| FLOW-E | Antisymmetric per-edge flow in MW, summed per node over capacity, with an auxiliary loss on the known synthetic flow |

## Scoring

- Experiment 613's scorer: follow and coverage on three injected banks. Bank M mirrors each shift at
  one neighbour (experiment 581's bank). Bank S splits the shift 60/40 over two neighbours. Bank X
  has no mirror. S and X are inference-only additions.
- Event-free CRPS at 24-120 h and 120-240 h by group, with the tail table.
- Minimum detectable effects from experiment 610: 0.024 (eight MVA Primaries) and 0.044 (six MW
  Primaries) in event-free CRPS, and about 0.03 in follow.
- Stored forecasts of C are reused. Before reuse, one refitted C cutoff is checked to reproduce the
  stored C within seed noise.

## Checks before any fit

- **Leak guard.** A unit test replaces every neighbour reading later than the origin minus 24 h, and
  every neighbour target, with noise. The forecasts must be bit-identical on CPU. No normalisation
  layer pools over the node dimension.
- **Injection guard.** A mirrored injection changes the neighbour's lags only, never its target.
- **Code review.** A fresh Sonnet sub-agent reviews the code before the first fit.

## GPU budget and stop rules

- Ceiling 5 GPU-hours for G1; one GPU process at a time, started only when no other experiment's fit
  is running. One GAT-E cutoff is timed first, and GCN-E is dropped before the ceiling is crossed.
- Stop and tell the manager if the ceiling would be exceeded or free disk falls under 5 GB.
- If no arm passes the gate, this page is the answer and G2 and G3 do not run.

## Caveats

- NGED says load-shift events may be missing from its logs, so clean rows can hold unlabelled events.
  The harm table is optimistic and real-event slices undercount.
- Eight of the 14 scored Primaries report MVA, which has no direction, so FLOW-E's conservation is
  approximate at those nodes.
