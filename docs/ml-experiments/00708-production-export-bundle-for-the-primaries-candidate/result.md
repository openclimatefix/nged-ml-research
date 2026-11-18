---
id: "00708"
slug: production-export-bundle-for-the-primaries-candidate
title: >-
  Can production rebuild the Primaries Candidate's forecasts, with its K1 network and table T, from
  a plain directory of weights, standardisers, a config, and golden fixtures, without importing
  research code?
theme: production-reference-and-best-guess
also_relevant_to: []
status: planned
kind: other
size: >-
  Complex. (1) It stores one refit of the Candidate (arm B of experiment 703, refit 2026-10) and one
  export bundle; (2) the extension of experiment 705's bundle format to three linked networks (K1,
  table T, and the Candidate) has several defensible layouts; (3) it imports the chain of
  experiments 574, 591, 598, 613, 667, 703, 704, and 705, whose callers cannot be named without
  searching; (4) it touches no serving path and no degradation rule; (5) it changes no contract.
  Reviews: code, write-up, and prose.
hypothesis: >-
  Experiment 705's bundle format, with a `components/k1/` and a `components/candidate/` folder and
  matching fixtures, is enough for a script that imports only torch, numpy, and polars to reproduce
  the Candidate's stored quantile forecasts of the Primaries, including table T, which the
  Candidate reads from K1's lags-masked forecasts.
prediction: >-
  At five fixture origins in October 2026 and both seeds, the standalone reference check reproduces
  K1's lags-masked medians and the table T entries the Candidate reads to 1e-5 of capacity, the
  Candidate's input matrix to 1e-5, and each seed's quantiles and the seed-mean quantiles to 2e-5 of
  capacity; and the exporter's own rerun of the prediction pass equals the chain's stored per-seed
  forecasts exactly.
decision_rule: >-
  Supported if every tolerance in the prediction holds at every fixture origin and seed; otherwise
  the bundle is not fit to hand to production.
benchmark: >-
  The per-seed forecasts that experiment 704's `chain667.sh` writes for arm `drop`, month 2026-10,
  and the input matrices and tables T that its prediction pass builds.
row_set_id: none
metric: >-
  Largest absolute difference between the reference check's values and the stored values, per
  stage (K1 lags-masked medians, table T entries, Candidate input matrix, raw network outputs,
  per-seed quantiles, seed-mean quantiles, contract megawatts over capacity).
---

**Status: planned.** This experiment is a tool for the port rather than a test of a change: it
exports the Candidate, the Primaries demand neural network of [experiment
667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md) (arm
`P7+D50`), in the form that [experiment
705](../00705-production-export-bundle-for-the-primaries-step-5-network/result.md) built for the
step-5 network. The hypothesis above is the acceptance test the code is written against.

## Plan

**The arm is arm B of [experiment
703](../00703-cams-free-inference-for-the-primaries-candidate/result.md), refitted at 2026-10-01.**
Arm B sets the two previous-day CAMS columns to zero in every stage, and experiment 703 chose it.
The 2026-10 refit is the latest that the research inputs support, because experiment 607's cached
inputs end at 2026-10-31. The refit runs experiment 704's `chain667.sh drop 2026-10` once, both
seeds, which writes new folders only (`arms/drop/chains/2026-10/` and `arms/drop/P7_D50_2026-10/`
under experiment 704's data folder).

**The bundle extends format 1 of experiment 705 without changing the step-5 layout.**

- `components/k1/`: K1's step-2 weights per seed, its standardisers, and its column order. K1 is
  the step-5-like network that the Candidate is grafted onto, and its lags-masked forecast fills
  table T.
- `components/candidate/`: the Candidate's weights per seed, its extra signal columns in order, the
  neighbour list of each Primary, and the rule that builds table T.
- `fixtures/k1/` and `fixtures/candidate/`: per origin, the inputs, K1's lags-masked quantiles, the
  table T entries the Candidate reads, the Candidate's input matrix raw and standardised, the raw
  outputs, and the quantiles per seed, seed-averaged, and at the 12:00 UTC contract origin in
  megawatts.
- The bundle states what production must keep between daily runs, such as the store of K1's
  earlier lags-masked forecasts that table T reads.

**The fixture origins are those of experiment 705's production bundle:** 2026-10-01, 2026-10-03,
2026-10-24, 2026-10-25 (the clock change), and 2026-10-31.

**The code follows experiment 705:** a GPU capture of the prediction pass in a scratch root, a CPU
exporter that asserts the rerun equals the stored forecasts, and a `reference_check.py` run with
`python -I` in a clean interpreter. Every command reads cached inputs from experiment folders, and
none reads the shared NWP table.

**Compute:** one chain of about 15 minutes on the RTX A6000 (one GPU process, 8 GB host-memory
cap), then about 1 hour on 1 to 4 CPU cores, inside the manager's slot C.
