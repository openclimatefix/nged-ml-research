---
id: "00727"
slug: step-5-bundle-format-2-re-export
title: >-
  Can the step-5 demand network's export bundle be re-exported as bundle format 2, with numpy
  weights and NGED's 13 delivery quantile levels, without changing any forecast?
theme: production-reference-and-best-guess
also_relevant_to: []
status: finished
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
verdict: supported
best_arm: >-
  Format version 2; largest difference to format 1 at the 7 shared levels 1.1e-5 of capacity
  (replay bundle) against a tolerance of 2e-5.
headline: >-
  Both step-5 bundles are re-exported as format 2 with numpy weights and NGED's 13 quantile
  levels, and the 7 levels that format 1 also held move by at most 1.1e-5 of capacity.
lessons: >-
  The step-5 neural network's mixture head gives any quantile level by bisection, so changing the
  levels needed no retraining. The format 2 reference check cannot test the regenerated quantiles,
  because it runs the same quantile code that generated them.
take_home_message: >-
  Production can now read the step-5 neural network, the demand forecaster for the 16 primary
  substations, without the PyTorch library, and gets the 13 quantile levels that National Grid
  Electricity Distribution (NGED) asked for. The forecasts did not change: at the 7 levels that the
  old bundle also held, the new values differ by at most 1.1e-5 of a substation's capacity.
advice:
  - >-
    Change a mixture-density network's output levels in the bundle's config and regenerate the
    quantile fixtures, without retraining, and compare the levels the old and new bundles share
    (Checks).
  - >-
    Do not read a pass of the bundle's own reference check as evidence for regenerated quantile
    fixtures, because the same code wrote them; rest the evidence on the comparison with the
    previous bundle (Checks).
production_status: candidate
adopted_in: none
---

**Status: finished.** The experiment re-exports the two bundles that experiment 705 wrote, and
checks that no forecast changes. The hypothesis in the front matter was the acceptance test.

## The question

**Production needs the step-5 neural network's bundle without the PyTorch library (`torch`), and
with the 13 quantile levels that National Grid Electricity Distribution (NGED) asked for.** A bundle
is the directory that holds everything production needs to run the neural network: weights,
standardisers (the means and standard deviations that scale each input column), configuration, and
test fixtures (stored inputs and outputs that production's code must reproduce). The step-5 neural
network forecasts the demand of the 16 primary substations (Primaries) of the Flexpectation study.
The consumer is [PR #1707 (Plan: serve the step-5 demand network for the 16 Primaries in the live
service)](https://github.com/openclimatefix/nged-substation-forecast/pull/1707) in the production
repository.

[Experiment 705](../00705-production-export-bundle-for-the-primaries-step-5-network/result.md)
wrote two bundles in format 1. This experiment re-writes them as format 2, with three changes:

- Each seed's weights are also stored as numpy arrays.
- The manifest gains `model_release`, set to 0 because neither bundle is a release of the
  forecasting model.
- The quantile levels become 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95,
  0.98, and 0.99.

No neural network is retrained, and nothing under `src/` or any production package changes.

**Yes: the forecasts do not change beyond 1.1e-5 of capacity, and the new levels need no
retraining, because the neural network outputs a mixture of three Gaussian distributions.** The
read-out bisects the mixture's cumulative distribution function at each level, so a level is a
number in the configuration. Format 1 held 11 levels. Seven of them (0.01, 0.05, 0.10, 0.50, 0.90,
0.95, and 0.99) are also among the 13 new levels.

**The 7 shared levels agree with format 1 to within 1.1e-5 of capacity, against the 2e-5
tolerance of experiment 705's reference check.** The levels 0.25, 0.40, 0.60, and 0.75 leave the
bundle, and the levels 0.02, 0.20, 0.35, 0.65, 0.80, and 0.98 enter it.

**Two re-exported bundles are ready for production PR #1707, at these paths, and the PR is not
blocked on anything else in this repository:**

- `data/ml-experiments/00727-step-5-bundle-format-2-re-export/bundles/replay/2026-04` (experiment
  677's arm `k1_5w_sp1` refitted at 2026-04-01, 102 MiB; the arm that the request named).
- `data/ml-experiments/00727-step-5-bundle-format-2-re-export/bundles/drop/2026-10` (experiment
  704's arm B refitted at 2026-10-01, 97 MiB; the bundle that experiment 705 called the production
  bundle).

## The bundle

**Each bundle is one directory, and `README.md` inside it describes the format for a reader who has
not seen experiment 705.** The directory holds these files:

- `manifest.json`: format version "2", `model_release` 0, the arm, the refit time, the training
  cut-off, the research commit, library versions, the fixture origins, the export checks, and the
  SHA-256 hash of every other file.
- `config.json`: every constant that inference reads, including the 13 quantile levels.
- `primaries.json`: the 16 Primaries with their H3 hexagonal-grid cell, coordinates, and capacity.
- `components/step5/`: the 60 input columns, the standardisers, the offset-knot dates, the
  expected-weather rows, and each seed's weights as `weights/seed<N>.npz` (numpy arrays keyed by
  the `state_dict` names) and `weights/seed<N>.pt` (the same tensors, for research code).
- `fixtures/step5/origin=<date>/`: for each of five fixture origins, the raw and interpolated
  ensemble weather, every input column, the raw network outputs, and the quantiles at the 13
  levels per seed and as the seed mean, at the 09:00 native origin and the 12:00 contract origin.
- `reference_check.py`: a standalone forward pass and check that needs only `torch`, `numpy`, and
  `polars`.

## Checks

| Quantity | Replay 2026-04 | Drop 2026-10 | Tolerance |
|---|---|---|---|
| Raw network outputs against format 1 (absolute) | 3.3e-6 | 2.6e-6 | 1e-5 |
| Per-seed quantiles at the 7 shared levels (fraction of capacity) | 1.1e-5 | 2.3e-6 | 2e-5 |
| Seed-mean quantiles at the 7 shared levels | 3.7e-6 | 7.2e-7 | 2e-5 |
| Contract-origin quantiles in megawatts, divided by capacity, per seed and seed mean, at the 7 shared levels | 1.1e-5 | 2.3e-6 | 2e-5 |
| Quantiles with the day lag and the week lag missing, at the 7 shared levels | 2.1e-6 | 3.0e-6 | 2e-5 |
| `.npz` against `.pt` (names, shapes, dtypes, bytes) | 96 of 96 arrays equal | 96 of 96 arrays equal | exact |
| Format 2 reference check, all stages | passed in 9 s | passed in 8 s | 1e-5 and 2e-5 |

- **The comparison with format 1 is the evidence for the new quantiles.** The format 2 reference
  check reports 0 for every quantile stage, because the same read-out wrote the regenerated
  fixtures. The check still tests the input matrix (exact) and the raw outputs (3.3e-6 and 2.6e-6).
  The numbers in the first five rows are stored in each bundle's `manifest.json`, under
  `export_checks.format_2_reexport`.
- **The largest difference, 1.1e-5, is at the 2026-04-30 origin of the replay bundle.** Format 1's
  stored quantiles came from a graphics-processing-unit (GPU) pass, and the raw outputs of the
  central-processing-unit (CPU) pass differ by 3.3e-6 from them, probably through float32 rounding.
  Experiment 705's own check of this bundle found a per-seed quantile difference of 1.1e-5 at the
  same origin.
- **The exporter's comparison covers more than the reference check does.** The exporter compares
  all five origins, the per-seed contract columns, and the variant with missing lags at every
  origin. The reference check tests the variant with missing lags at one origin only, and not the
  per-seed contract columns.
- **No check exceeded its tolerance.**

## Method

**Architecture.** The architecture is unchanged from experiment 705 and
[experiment 677](../00677-building-up-the-candidate-demand-network-one-component-at-a-time/result.md)
step 5.

- The neural network reads 60 input columns: ensemble weather summaries and spreads, calendar
  columns, the lead time, a day lag and a week lag of recent power, an offset-knot position, and 14
  columns of the B1 stage (the weather-and-calendar baseline that experiment 677 builds first).
- The neural network's outputs are 9 numbers: three mixture weights (logits), three means, and
  three log standard deviations, which are clamped to -3 to 3. The means include an offset
  interpolated between monthly knots.
- Each quantile is found by bisecting the mixture's cumulative distribution function over -40 to 40
  standardised units, in 40 steps, in float64 arithmetic.
- The forecast uses only information from before the origin, as in experiment 705.

**Training regime.** No training. Each bundle holds the stored weights of three neural networks, each
trained from one random seed (0, 1, or 2), from experiment 677 (replay) and experiment 704 arm B (drop).

**Export and measurement.**

- The exporter copies the format 1 bundle, writes each seed's `.npz` from its `.pt`, and sets the
  13 levels in `config.json`. It then re-runs each seed's forward pass on the CPU, from the stored
  standardised input matrix, for all five origins and for the variant with the day lag and the week
  lag missing.
- The exporter averages the three seeds in float32 and sorts, as format 1 did, and multiplies by
  `capacity_mw` in float64 before rounding to float32 for the contract rows.
- The exporter compares the 7 shared levels with the format 1 fixtures and stops with an error if
  any difference exceeds the format 1 tolerance (1e-5 for raw outputs, 2e-5 of capacity for
  quantiles).
- The exporter builds each bundle in a `.partial` folder and renames the folder on success. It files
  the format 1 export checks under `format_1_export`, asserts that each level name (`q01` to `q99`)
  matches its level value, and asserts the order of the contract rows.

## Files

- `ml-experiments/00727-step-5-bundle-format-2-re-export/e00727_reexport.py`: builds a format 2
  bundle from a format 1 bundle and runs the comparison.
- `ml-experiments/00727-step-5-bundle-format-2-re-export/e00727_verify.py`: checks that every
  `.npz` array equals its `.pt` tensor bit for bit, that the file hashes in `manifest.json` match,
  and that the format 2 fields are set.
- `ml-experiments/00727-step-5-bundle-format-2-re-export/reference_check.py`: the format 2 copy of
  experiment 705's reference check (13 level names, weights from `.npz`, the same tolerances),
  copied into each bundle.
- `ml-experiments/00727-step-5-bundle-format-2-re-export/bundle_readme.md`: the bundle's `README.md`.
- `data/ml-experiments/00705-production-export-bundle-for-the-primaries-step-5-network/bundles/`:
  the format 1 bundles that the exporter read (`replay/2026-04` and `drop/2026-10`), from
  [experiment 705](../00705-production-export-bundle-for-the-primaries-step-5-network/result.md).
- `data/ml-experiments/00727-step-5-bundle-format-2-re-export/bundles/`: the two format 2 bundles
  written by this experiment (`replay/2026-04` and `drop/2026-10`).
