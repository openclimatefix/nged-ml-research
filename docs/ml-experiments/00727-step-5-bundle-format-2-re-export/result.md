---
id: "00727"
slug: step-5-bundle-format-2-re-export
title: >-
  Can the step-5 demand network's export bundle be re-exported as bundle format 2, with numpy
  weights and NGED's 13 delivery quantile levels, without changing any forecast?
theme: production-reference-and-best-guess
also_relevant_to: []
status: planned
kind: other
size: >-
  Simple. (1) It stores one re-exported bundle per source bundle; (2) one design is defensible,
  because the brief fixes the three changes; (3) it imports the export and check code of
  experiment 705 only; (4) it touches no serving path and no degradation rule; (5) it changes no
  contract. Reviews: code and prose.
hypothesis: >-
  The step-5 network's mixture-density head gives any quantile level by bisection of its
  cumulative distribution function, so the format 1 bundle can be re-exported with numpy weights
  and NGED's 13 delivery levels, and the quantiles at the 7 levels that format 1 also holds do not
  change.
prediction: >-
  At every fixture origin and seed, each of the 7 shared quantile levels differs from the format 1
  fixture by at most 2e-5 of capacity, every numpy array equals its torch tensor exactly, and the
  format 2 reference check passes at the format 1 tolerances.
decision_rule: >-
  Supported if every tolerance in the prediction holds. Stop and report to the manager if any
  shared level differs by more than its tolerance.
benchmark: >-
  The format 1 bundles of experiment 705 (`bundles/replay/2026-04` and `bundles/drop/2026-10`).
row_set_id: none
metric: >-
  Largest absolute difference, as a fraction of capacity, between the format 2 fixtures and the
  format 1 fixtures at the 7 shared levels, and the largest error of the format 2 reference check
  per stage.
---

**Status: planned.** This experiment changes only an export, and trains nothing.

## The question

**Production pull request 1707 needs the step-5 network's bundle without torch, and with NGED's 13
delivery levels.** The experiment re-exports experiment 705's format 1 bundle as format 2. Weights
become numpy arrays in `.npz` files, `manifest.json` gains a `model_release` field (0), and the
quantile levels in `config.json` become 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80,
0.90, 0.95, 0.98, and 0.99. The network's head is a three-component Gaussian mixture, so the new
levels need no retraining.
